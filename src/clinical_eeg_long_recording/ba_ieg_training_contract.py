"""Fail-closed training contracts for the BA-IEG event encoder.

This module is deliberately downstream of canonical EEG/view materialization
and upstream of any trainable BA-IEG model.  It does not open EDF files, run a
detector, read annotations/spreadsheets, or choose events from labels.  Its
job is to make the ragged, multi-scale event representation and the two
currently defensible supervision routes explicit:

* signal-derived deterministic measurement regression; and
* patient-level DeepSOZ positive-set MIL over complete detector-frozen bags.

Clinical terms, private doctor labels, EDF annotations and spreadsheet fields
are not accepted by any public constructor.  Detector-provider tensors are
also excluded: the detector may navigate to an event, but Findings tokens must
come from evidence-eligible Findings/spatial views.

The contract stores physical time intervals rather than token ordinals.  Each
analysis unit is represented by a signed linear combination of the frozen
standard-19 physical electrode basis, allowing referential, bipolar and
Laplacian views to coexist without pretending that a bipolar lead is one of
its endpoint electrodes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Final, Iterable, Mapping, Sequence

import torch

from src.soz.geometry import CHANNEL_INDEX, STANDARD_19

from .ba_ieg_numerical_kernel import (
    BA_IEG_BASE_MEASUREMENT_NAMES,
    BA_IEG_BASE_NUMERICAL_KERNEL_ID,
    BAIEGBaseNumericalPolicy,
    measure_ba_ieg_base_numerical_features,
)


BA_IEG_EVENT_TOKEN_SCHEMA_VERSION: Final[str] = "ba_ieg_event_tokens_v2"
BA_IEG_P0_VIEW_PROFILE_LEGACY_8: Final[str] = "onset_offline_8_v1"
BA_IEG_P0_VIEW_PROFILE_NATIVE_12: Final[str] = (
    "onset_offline_native_12_v1"
)
BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION: Final[str] = (
    "ba_ieg_target_free_ragged_event_materialization_v4"
)
BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_NATIVE_12: Final[str] = (
    "ba_ieg_target_free_ragged_event_materialization_v5"
)
BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_A0_NATIVE_12: Final[str] = (
    "ba_ieg_a0_oracle_navigation_ragged_event_materialization_v6"
)
BA_IEG_P0_IMPLEMENTATION_ID: Final[str] = (
    "ba_ieg_p0_deterministic_multiclock_physical_time_measurements_v2"
)
BA_IEG_P0_IMPLEMENTATION_ID_NATIVE_12: Final[str] = (
    "ba_ieg_p0_deterministic_multiclock_physical_time_measurements_native_12_v1"
)
BA_IEG_P0_IMPLEMENTATION_ID_A0_NATIVE_12: Final[str] = (
    "ba_ieg_p0_a0_initial_watchdog_deterministic_native_12_v1"
)
BA_IEG_P0_NAVIGATION_ARM_A0: Final[str] = "A0_conditional_on_oracle_navigation"
BA_IEG_P0_A0_EVALUATION_SEMANTICS: Final[str] = (
    "conditional_on_seizure_interval_upper_bound"
)
BA_IEG_PATIENT_BAG_SCHEMA_VERSION: Final[str] = (
    "ba_ieg_complete_patient_positive_set_bag_v1"
)
BA_IEG_PHASE_STATES: Final[tuple[str, ...]] = ("S0", "S1", "S2", "S3")
BA_IEG_TOKEN_SCALES: Final[tuple[str, ...]] = ("fine", "coarse", "context")
BA_IEG_EFFECTIVE_TEMPORAL_ROLES: Final[tuple[str, ...]] = (
    "morphology_native",
    "onset_causal",
    "context_offline",
)
BA_IEG_DEPENDENCY_POLICIES: Final[tuple[str, ...]] = (
    "instantaneous",
    "past_and_present_only",
    "bidirectional_or_unknown",
)
BA_IEG_EVIDENCE_FAMILIES: Final[tuple[str, ...]] = (
    "amplitude",
    "morphology",
    "spectral",
    "spatial_field",
    "high_frequency",
)
BA_IEG_ALLOWED_VIEW_ROLES: Final[frozenset[str]] = frozenset(
    {
        "findings_native",
        "findings_clinical",
        "findings_native_morphology",
        "onset_causal",
        "context_offline",
        "spatial_reference",
    }
)
BA_IEG_REFERENCE_FAMILIES: Final[tuple[str, ...]] = (
    "referential",
    "bipolar",
    "common_average",
    "laplacian",
    "other_versioned_linear",
)
BA_IEG_ALLOWED_REFERENCE_FAMILIES: Final[frozenset[str]] = frozenset(
    BA_IEG_REFERENCE_FAMILIES
)
BA_IEG_ALLOWED_ENCODER_LINEAGES: Final[frozenset[str]] = frozenset(
    {
        "ba_ieg_native",
        "tfm_repository_local_style",
        "labram_official_pinned",
        "cbramod_repository_inspired",
        "deterministic_projection",
    }
)
BA_IEG_ALLOWED_SPLITS: Final[frozenset[str]] = frozenset(
    {"source_train", "source_dev", "source_eval", "private_inference"}
)
BA_IEG_C18: Final[tuple[str, ...]] = tuple(
    electrode for electrode in STANDARD_19 if electrode != "PZ"
)

# These are measurement targets, not clinical terms.  Fold-local transforms
# (for example log/robust scaling) belong in the training policy and must not
# mutate the raw replayable targets stored here.
BA_IEG_DETERMINISTIC_TARGETS: Final[tuple[str, ...]] = (
    "rms_uv",
    "peak_to_peak_uv",
    "line_length_uv_per_sample",
    "dominant_frequency_hz",
    "spectral_concentration",
    "spectral_entropy",
    "rhythmicity_index",
    "delta_power_ratio",
    "theta_power_ratio",
    "alpha_power_ratio",
    "beta_power_ratio",
    "low_gamma_power_ratio",
    "robust_multifeature_change_score",
)

# P0 model inputs are deliberately deterministic measurements, not the dense
# supervision sidecar above and not a frozen embedding from an unverified
# checkpoint.  The last feature is computed only from preceding tiles of the
# same physical unit/scale after robust within-event standardisation.
BA_IEG_P0_TOKEN_FEATURES: Final[tuple[str, ...]] = (
    "rms_uv",
    "peak_to_peak_uv",
    "line_length_uv_per_sample",
    "zero_crossing_rate_hz",
    "slope_reversal_rate_hz",
    "excess_kurtosis",
    "dominant_frequency_hz",
    "spectral_concentration",
    "spectral_entropy",
    "rhythmicity_index",
    "delta_power_ratio",
    "theta_power_ratio",
    "alpha_power_ratio",
    "beta_power_ratio",
    "low_gamma_power_ratio",
    "robust_previous_tile_change_score",
)
BA_IEG_P0_SHARED_BASE_FEATURE_INDICES: Final[tuple[int, ...]] = tuple(
    BA_IEG_P0_TOKEN_FEATURES.index(name)
    for name in BA_IEG_BASE_MEASUREMENT_NAMES
)
BA_IEG_P0_FEATURE_FAMILIES: Final[tuple[str, ...]] = (
    "amplitude",
    "amplitude",
    "amplitude",
    "morphology",
    "morphology",
    "morphology",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "change_composite",
)
_DETERMINISTIC_TARGET_FAMILIES: Final[tuple[str, ...]] = (
    "amplitude",
    "amplitude",
    "amplitude",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "spectral",
    "change_composite",
)

_SCALE_DURATION_BOUNDS: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 2.0),
    (2.0, 8.0),
    (8.0, 32.0),
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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


def _finite_pair(value: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item interval")
    start, stop = float(value[0]), float(value[1])
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError(f"{name} must be a finite positive-duration interval")
    return start, stop


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _frozen_tensor(
    value: torch.Tensor,
    *,
    name: str,
    ndim: int,
    dtype: torch.dtype | None = None,
    floating: bool | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional torch.Tensor")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if floating is True and not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if floating is False and value.is_floating_point():
        raise TypeError(f"{name} must not be floating point")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached from autograd")
    result = value.detach().clone().contiguous()
    if result.is_floating_point() and not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class EEGOnlyFeatureScope:
    """Feature-lineage firewall shared by every BA-IEG event input."""

    eeg_samples_used: bool = True
    detector_navigation_used: bool = True
    detector_provider_tensor_used_as_findings: bool = False
    edf_annotations_used: bool = False
    spreadsheet_used: bool = False
    private_doctor_labels_used: bool = False
    public_targets_used_as_model_input: bool = False
    clinical_text_used: bool = False
    video_used: bool = False
    sleep_or_activation_labels_used: bool = False
    ecg_emg_eog_used: bool = False

    def __post_init__(self) -> None:
        expected = {
            "eeg_samples_used": True,
            "detector_provider_tensor_used_as_findings": False,
            "edf_annotations_used": False,
            "spreadsheet_used": False,
            "private_doctor_labels_used": False,
            "public_targets_used_as_model_input": False,
            "clinical_text_used": False,
            "video_used": False,
            "sleep_or_activation_labels_used": False,
            "ecg_emg_eog_used": False,
        }
        for name, required in expected.items():
            if getattr(self, name) is not required:
                raise ValueError(f"BA-IEG feature scope violates firewall field {name}")
        if type(self.detector_navigation_used) is not bool:
            raise TypeError("detector_navigation_used must be boolean")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, bool]:
        return {
            name: bool(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class BAIEGDeterministicTargets:
    """Ragged signal-derived measurement targets for one event.

    Rows are aligned to physical unit/time/view coordinates, not to a padded
    batch position.  Missing measurements are represented only by
    ``value_mask=False``; the corresponding stored value must be zero.
    """

    values: torch.Tensor
    value_mask: torch.Tensor
    row_time_bounds_seconds: torch.Tensor
    row_unit_index: torch.Tensor
    row_view_index: torch.Tensor
    policy_sha256: str
    source_binding_sha256: str
    target_names: tuple[str, ...] = BA_IEG_DETERMINISTIC_TARGETS
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        values = _frozen_tensor(self.values, name="deterministic values", ndim=2, floating=True)
        masks = _frozen_tensor(
            self.value_mask,
            name="deterministic value_mask",
            ndim=2,
            dtype=torch.bool,
        )
        times = _frozen_tensor(
            self.row_time_bounds_seconds,
            name="deterministic row_time_bounds_seconds",
            ndim=2,
            floating=True,
        )
        units = _frozen_tensor(
            self.row_unit_index,
            name="deterministic row_unit_index",
            ndim=1,
            dtype=torch.long,
        )
        views = _frozen_tensor(
            self.row_view_index,
            name="deterministic row_view_index",
            ndim=1,
            dtype=torch.long,
        )
        rows = int(values.shape[0])
        if rows < 1 or tuple(values.shape) != (rows, len(BA_IEG_DETERMINISTIC_TARGETS)):
            raise ValueError("deterministic values must have shape [M,13] with M >= 1")
        if tuple(masks.shape) != tuple(values.shape):
            raise ValueError("deterministic value_mask must align with values")
        if tuple(times.shape) != (rows, 2) or tuple(units.shape) != (rows,) or tuple(views.shape) != (rows,):
            raise ValueError("deterministic coordinate rows do not align")
        if self.target_names != BA_IEG_DETERMINISTIC_TARGETS:
            raise ValueError("deterministic target vocabulary drifted")
        if torch.any(times[:, 1] <= times[:, 0]):
            raise ValueError("deterministic target intervals must have positive duration")
        if torch.any(units < 0) or torch.any(views < 0):
            raise ValueError("deterministic target indices cannot be negative")
        if not masks.any(dim=1).all():
            raise ValueError("every deterministic target row needs at least one measurement")
        if torch.any(values.masked_select(~masks) != 0):
            raise ValueError("masked deterministic target values must be zero")
        order = sorted(
            range(rows),
            key=lambda index: (
                int(views[index]),
                int(units[index]),
                float(times[index, 0]),
                float(times[index, 1]),
            ),
        )
        if order != list(range(rows)):
            raise ValueError("deterministic target rows must use canonical view/unit/time order")
        _sha256(self.policy_sha256, "deterministic policy_sha256")
        _sha256(self.source_binding_sha256, "deterministic source_binding_sha256")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value_mask", masks)
        object.__setattr__(self, "row_time_bounds_seconds", times)
        object.__setattr__(self, "row_unit_index", units)
        object.__setattr__(self, "row_view_index", views)
        object.__setattr__(self, "receipt_sha256", self._compute_sha256())

    def _compute_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": "ba_ieg_deterministic_measurement_targets_v1",
                "target_names": list(self.target_names),
                "policy_sha256": self.policy_sha256,
                "source_binding_sha256": self.source_binding_sha256,
                "tensor_sha256": {
                    "values": _tensor_sha256(self.values),
                    "value_mask": _tensor_sha256(self.value_mask),
                    "row_time_bounds_seconds": _tensor_sha256(self.row_time_bounds_seconds),
                    "row_unit_index": _tensor_sha256(self.row_unit_index),
                    "row_view_index": _tensor_sha256(self.row_view_index),
                },
            }
        )

    def verify_integrity(self) -> None:
        if self.receipt_sha256 != self._compute_sha256():
            raise ValueError("deterministic target tensor content changed after registration")


@dataclass(frozen=True)
class BAIEGEventTokens:
    """One variable-length event represented as physical-time tokens."""

    event_id: str
    recording_id: str
    patient_uid: str
    model_split: str
    analysis_interval_seconds: tuple[float, float]
    navigation_anchor_seconds: float
    canonical_receipt_sha256: str
    adaptive_window_receipt_sha256: str
    encoder_implementation_id: str
    encoder_lineage: str
    encoder_receipt_sha256: str
    physical_electrode_ids: tuple[str, ...]
    physical_xyz: torch.Tensor
    physical_xyz_mask: torch.Tensor
    physical_evidence_mask: torch.Tensor
    view_ids: tuple[str, ...]
    view_roles: tuple[str, ...]
    view_effective_temporal_roles: tuple[str, ...]
    view_dependency_policies: tuple[str, ...]
    view_future_sample_access: torch.Tensor
    view_onset_evidence_authorized: torch.Tensor
    view_temporal_evidence_sha256s: tuple[str, ...]
    view_receipt_sha256s: tuple[str, ...]
    view_transform_sha256s: tuple[str, ...]
    reference_families: tuple[str, ...]
    unit_ids: tuple[str, ...]
    unit_source_ids: tuple[str, ...]
    unit_types: tuple[str, ...]
    unit_view_index: torch.Tensor
    unit_reference_matrix: torch.Tensor
    unit_evidence_mask: torch.Tensor
    unit_family_mask: torch.Tensor
    token_values: torch.Tensor
    token_feature_mask: torch.Tensor
    token_time_bounds_seconds: torch.Tensor
    token_unit_index: torch.Tensor
    token_view_index: torch.Tensor
    token_scale_index: torch.Tensor
    token_signal_mask: torch.Tensor
    token_family_mask: torch.Tensor
    phase_posterior: torch.Tensor
    token_future_sample_access: torch.Tensor = field(init=False)
    token_onset_evidence_mask: torch.Tensor = field(init=False)
    token_positive_onset_mask: torch.Tensor = field(init=False)
    token_phase_context_mask: torch.Tensor = field(init=False)
    feature_scope: EEGOnlyFeatureScope = field(default_factory=EEGOnlyFeatureScope)
    deterministic_targets: BAIEGDeterministicTargets | None = None
    input_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "recording_id",
            "patient_uid",
            "encoder_implementation_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.model_split not in BA_IEG_ALLOWED_SPLITS:
            raise ValueError("BA-IEG event model_split is unsupported")
        interval = _finite_pair(self.analysis_interval_seconds, "analysis_interval_seconds")
        anchor = float(self.navigation_anchor_seconds)
        if not math.isfinite(anchor) or anchor < interval[0] or anchor > interval[1]:
            raise ValueError("navigation anchor must lie inside the physical event interval")
        for name in (
            "canonical_receipt_sha256",
            "adaptive_window_receipt_sha256",
            "encoder_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.encoder_lineage not in BA_IEG_ALLOWED_ENCODER_LINEAGES:
            raise ValueError("BA-IEG encoder lineage is unsupported or overclaims provenance")
        if tuple(self.physical_electrode_ids) != tuple(STANDARD_19):
            raise ValueError("BA-IEG v1 uses the frozen canonical standard-19 physical basis")

        xyz = _frozen_tensor(self.physical_xyz, name="physical_xyz", ndim=2, floating=True)
        xyz_mask = _frozen_tensor(
            self.physical_xyz_mask,
            name="physical_xyz_mask",
            ndim=1,
            dtype=torch.bool,
        )
        physical_evidence = _frozen_tensor(
            self.physical_evidence_mask,
            name="physical_evidence_mask",
            ndim=1,
            dtype=torch.bool,
        )
        if tuple(xyz.shape) != (len(STANDARD_19), 3) or tuple(xyz_mask.shape) != (
            len(STANDARD_19),
        ) or tuple(physical_evidence.shape) != (len(STANDARD_19),):
            raise ValueError(
                "physical coordinates/evidence must have shape [19,3], [19] and [19]"
            )
        if torch.any(xyz[~xyz_mask] != 0):
            raise ValueError("unavailable physical coordinates must be zero")
        if not physical_evidence.any():
            raise ValueError("event has no observed physical EEG electrode")

        views_count = len(self.view_ids)
        view_fields = (
            self.view_roles,
            self.view_effective_temporal_roles,
            self.view_dependency_policies,
            self.view_temporal_evidence_sha256s,
            self.view_receipt_sha256s,
            self.view_transform_sha256s,
            self.reference_families,
        )
        if views_count < 1 or any(len(field_value) != views_count for field_value in view_fields):
            raise ValueError("view metadata arrays must be aligned and non-empty")
        if len(set(self.view_ids)) != views_count:
            raise ValueError("BA-IEG view IDs must be unique")
        view_future = _frozen_tensor(
            self.view_future_sample_access,
            name="view_future_sample_access",
            ndim=1,
            dtype=torch.bool,
        )
        view_onset = _frozen_tensor(
            self.view_onset_evidence_authorized,
            name="view_onset_evidence_authorized",
            ndim=1,
            dtype=torch.bool,
        )
        if tuple(view_future.shape) != (views_count,) or tuple(view_onset.shape) != (
            views_count,
        ):
            raise ValueError("view temporal-permission tensors must align with views")
        for index, view_id in enumerate(self.view_ids):
            _identifier(view_id, f"view_ids[{index}]")
            if self.view_roles[index] not in BA_IEG_ALLOWED_VIEW_ROLES:
                raise ValueError("detector/boundary/display views cannot supply Findings tokens")
            effective_role = self.view_effective_temporal_roles[index]
            dependency = self.view_dependency_policies[index]
            if effective_role not in BA_IEG_EFFECTIVE_TEMPORAL_ROLES:
                raise ValueError("view effective temporal role is unsupported")
            if dependency not in BA_IEG_DEPENDENCY_POLICIES:
                raise ValueError("view dependency policy is unsupported")
            _sha256(
                self.view_temporal_evidence_sha256s[index],
                f"view_temporal_evidence_sha256s[{index}]",
            )
            _sha256(self.view_receipt_sha256s[index], f"view_receipt_sha256s[{index}]")
            _sha256(self.view_transform_sha256s[index], f"view_transform_sha256s[{index}]")
            if self.reference_families[index] not in BA_IEG_ALLOWED_REFERENCE_FAMILIES:
                raise ValueError("reference family is unsupported")
            task_role = self.view_roles[index]
            expected_direct_role = {
                "findings_native": "morphology_native",
                "findings_native_morphology": "morphology_native",
                "findings_clinical": "context_offline",
                "context_offline": "context_offline",
                "onset_causal": "onset_causal",
            }.get(task_role)
            if (
                expected_direct_role is not None
                and effective_role != expected_direct_role
            ):
                raise ValueError(
                    "direct task role and effective temporal role disagree"
                )
            if effective_role == "onset_causal":
                if dependency != "past_and_present_only" or bool(view_future[index]):
                    raise ValueError(
                        "onset-causal lane must be future-free and past-only"
                    )
            elif effective_role == "context_offline":
                if (
                    dependency != "bidirectional_or_unknown"
                    or not bool(view_future[index])
                    or bool(view_onset[index])
                ):
                    raise ValueError(
                        "offline-context view must declare future access and cannot authorize onset"
                    )
            elif (
                dependency != "instantaneous"
                or bool(view_future[index])
                or bool(view_onset[index])
            ):
                raise ValueError(
                    "native-morphology view must remain instantaneous and onset-ineligible"
                )

        units_count = len(self.unit_ids)
        if (
            units_count < 1
            or len(self.unit_source_ids) != units_count
            or len(self.unit_types) != units_count
            or len(set(self.unit_ids)) != units_count
        ):
            raise ValueError("unit metadata must be aligned, unique and non-empty")
        if any(unit_type not in {"electrode", "lead", "virtual"} for unit_type in self.unit_types):
            raise ValueError("unit type is unsupported")
        for index, unit_id in enumerate(self.unit_ids):
            _identifier(unit_id, f"unit_ids[{index}]")
            _identifier(self.unit_source_ids[index], f"unit_source_ids[{index}]")
        unit_views = _frozen_tensor(
            self.unit_view_index, name="unit_view_index", ndim=1, dtype=torch.long
        )
        reference = _frozen_tensor(
            self.unit_reference_matrix,
            name="unit_reference_matrix",
            ndim=2,
            floating=True,
        )
        unit_evidence = _frozen_tensor(
            self.unit_evidence_mask,
            name="unit_evidence_mask",
            ndim=1,
            dtype=torch.bool,
        )
        unit_families = _frozen_tensor(
            self.unit_family_mask,
            name="unit_family_mask",
            ndim=2,
            dtype=torch.bool,
        )
        if tuple(unit_views.shape) != (units_count,) or tuple(reference.shape) != (
            units_count,
            len(STANDARD_19),
        ) or tuple(unit_evidence.shape) != (units_count,) or tuple(unit_families.shape) != (
            units_count,
            len(BA_IEG_EVIDENCE_FAMILIES),
        ):
            raise ValueError("unit tensors do not match the declared unit/physical bases")
        if torch.any(unit_views < 0) or torch.any(unit_views >= views_count):
            raise ValueError("unit_view_index refers to an absent view")
        if torch.any(reference.abs().sum(dim=1)[unit_evidence] <= 0):
            raise ValueError("evidence-eligible units need a non-zero physical reference row")
        if torch.any(reference[unit_evidence][:, ~physical_evidence] != 0):
            raise ValueError(
                "evidence-eligible reference rows cannot use unobserved physical electrodes"
            )
        if torch.any(unit_families[~unit_evidence]):
            raise ValueError("evidence-ineligible units cannot enable a Findings family")

        tokens = _frozen_tensor(self.token_values, name="token_values", ndim=2, floating=True)
        feature_mask = _frozen_tensor(
            self.token_feature_mask,
            name="token_feature_mask",
            ndim=2,
            dtype=torch.bool,
        )
        times = _frozen_tensor(
            self.token_time_bounds_seconds,
            name="token_time_bounds_seconds",
            ndim=2,
            floating=True,
        )
        token_units = _frozen_tensor(
            self.token_unit_index, name="token_unit_index", ndim=1, dtype=torch.long
        )
        token_views = _frozen_tensor(
            self.token_view_index, name="token_view_index", ndim=1, dtype=torch.long
        )
        scales = _frozen_tensor(
            self.token_scale_index, name="token_scale_index", ndim=1, dtype=torch.long
        )
        signal_mask = _frozen_tensor(
            self.token_signal_mask, name="token_signal_mask", ndim=1, dtype=torch.bool
        )
        token_families = _frozen_tensor(
            self.token_family_mask,
            name="token_family_mask",
            ndim=2,
            dtype=torch.bool,
        )
        phase = _frozen_tensor(
            self.phase_posterior, name="phase_posterior", ndim=2, floating=True
        )
        token_count = int(tokens.shape[0])
        if token_count < 1 or tokens.shape[1] < 1:
            raise ValueError("BA-IEG event requires at least one non-empty token")
        if tuple(feature_mask.shape) != tuple(tokens.shape):
            raise ValueError("token_feature_mask must align with token_values")
        expected_vector = (token_count,)
        if tuple(times.shape) != (token_count, 2) or any(
            tuple(value.shape) != expected_vector
            for value in (token_units, token_views, scales, signal_mask)
        ) or tuple(token_families.shape) != (
            token_count,
            len(BA_IEG_EVIDENCE_FAMILIES),
        ) or tuple(phase.shape) != (token_count, len(BA_IEG_PHASE_STATES)):
            raise ValueError("token coordinate/mask tensors do not align")
        if torch.any(token_units < 0) or torch.any(token_units >= units_count):
            raise ValueError("token_unit_index refers to an absent unit")
        if torch.any(token_views < 0) or torch.any(token_views >= views_count):
            raise ValueError("token_view_index refers to an absent view")
        if not torch.equal(token_views, unit_views[token_units]):
            raise ValueError("each token must inherit the view of its analysis unit")
        if torch.any(scales < 0) or torch.any(scales >= len(BA_IEG_TOKEN_SCALES)):
            raise ValueError("token_scale_index is invalid")
        if torch.any(times[:, 1] <= times[:, 0]) or torch.any(times[:, 0] < interval[0] - 1e-6) or torch.any(
            times[:, 1] > interval[1] + 1e-6
        ):
            raise ValueError("token physical intervals lie outside the adaptive event")
        duration = times[:, 1] - times[:, 0]
        for scale_index, (lower, upper) in enumerate(_SCALE_DURATION_BOUNDS):
            selected = scales == scale_index
            if not selected.any():
                continue
            if scale_index == 0:
                invalid = (duration[selected] <= lower) | (duration[selected] > upper + 1e-6)
            else:
                invalid = (duration[selected] <= lower + 1e-6) | (duration[selected] > upper + 1e-6)
            if invalid.any():
                raise ValueError(
                    f"{BA_IEG_TOKEN_SCALES[scale_index]} token support violates its physical-duration contract"
                )
        if not signal_mask.any():
            raise ValueError("event has no signal-eligible token")
        if not torch.equal(signal_mask, feature_mask.any(dim=1)):
            raise ValueError(
                "token signal opportunity must equal its per-feature opportunity"
            )
        if torch.any(tokens[~feature_mask] != 0):
            raise ValueError(
                "masked tokens: feature-masked token values must be zero"
            )
        if torch.any(tokens[~signal_mask] != 0) or torch.any(token_families[~signal_mask]):
            raise ValueError("masked tokens must carry neither embeddings nor family eligibility")
        inherited_families = unit_families[token_units]
        if torch.any(token_families & ~inherited_families):
            raise ValueError("token family eligibility cannot exceed its source unit")
        if tokens.shape[1] == len(BA_IEG_P0_FEATURE_FAMILIES):
            for feature_index, family in enumerate(
                BA_IEG_P0_FEATURE_FAMILIES
            ):
                selected = feature_mask[:, feature_index]
                if not selected.any():
                    continue
                if family == "change_composite":
                    required = (
                        BA_IEG_EVIDENCE_FAMILIES.index("amplitude"),
                        BA_IEG_EVIDENCE_FAMILIES.index("spectral"),
                    )
                    if not token_families[selected][:, required].all():
                        raise ValueError(
                            "P0 change feature exceeds amplitude/spectral eligibility"
                        )
                elif not token_families[selected][
                    :, BA_IEG_EVIDENCE_FAMILIES.index(family)
                ].all():
                    raise ValueError(
                        f"P0 {family} feature exceeds family eligibility"
                    )
        if torch.any(phase[~signal_mask] != 0):
            raise ValueError("masked tokens cannot carry phase posterior")
        active_phase = phase[signal_mask]
        if torch.any(active_phase < 0) or torch.any(active_phase > 1) or not torch.allclose(
            active_phase.sum(dim=1),
            torch.ones(active_phase.shape[0], dtype=active_phase.dtype),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("active token S0/S1/S2/S3 posterior must be a probability simplex")
        token_future = view_future[token_views]
        token_onset = (
            signal_mask
            & unit_evidence[token_units]
            & view_onset[token_views]
            & ~token_future
        )
        token_positive_onset = (
            token_onset
            & token_families[
                :, BA_IEG_EVIDENCE_FAMILIES.index("spatial_field")
            ]
        )
        temporal_role_indices = torch.tensor(
            [
                BA_IEG_EFFECTIVE_TEMPORAL_ROLES.index(role)
                for role in self.view_effective_temporal_roles
            ],
            dtype=torch.long,
        )
        context_role_index = BA_IEG_EFFECTIVE_TEMPORAL_ROLES.index(
            "context_offline"
        )
        token_phase_context = (
            signal_mask
            & (temporal_role_indices[token_views] == context_role_index)
        )
        non_context_phase = signal_mask & ~token_phase_context
        if non_context_phase.any():
            neutral = torch.full(
                (int(non_context_phase.sum()), len(BA_IEG_PHASE_STATES)),
                1.0 / len(BA_IEG_PHASE_STATES),
                dtype=phase.dtype,
            )
            if not torch.allclose(
                phase[non_context_phase], neutral, atol=1e-6, rtol=1e-6
            ):
                raise ValueError(
                    "future-derived phase posterior cannot enter causal/native tokens"
                )
        order = sorted(
            range(token_count),
            key=lambda index: (
                int(token_views[index]),
                int(token_units[index]),
                int(scales[index]),
                float(times[index, 0]),
                float(times[index, 1]),
            ),
        )
        if order != list(range(token_count)):
            raise ValueError("tokens must use canonical view/unit/scale/time order")

        if not isinstance(self.feature_scope, EEGOnlyFeatureScope):
            raise TypeError("feature_scope must be EEGOnlyFeatureScope")
        targets = self.deterministic_targets
        if targets is not None:
            if not isinstance(targets, BAIEGDeterministicTargets):
                raise TypeError("deterministic_targets has the wrong contract")
            targets.verify_integrity()
            if int(targets.row_unit_index.max()) >= units_count or int(targets.row_view_index.max()) >= views_count:
                raise ValueError("deterministic targets refer to an absent unit/view")
            if not torch.equal(
                targets.row_view_index,
                unit_views[targets.row_unit_index],
            ):
                raise ValueError("deterministic target view disagrees with its source unit")
            target_times = targets.row_time_bounds_seconds
            if torch.any(target_times[:, 0] < interval[0] - 1e-6) or torch.any(
                target_times[:, 1] > interval[1] + 1e-6
            ):
                raise ValueError("deterministic target interval lies outside the event")
            for target_index, family in enumerate(_DETERMINISTIC_TARGET_FAMILIES):
                selected_rows = targets.value_mask[:, target_index]
                if not selected_rows.any():
                    continue
                eligible_rows = unit_families[targets.row_unit_index[selected_rows]]
                if family == "change_composite":
                    required = [
                        BA_IEG_EVIDENCE_FAMILIES.index("amplitude"),
                        BA_IEG_EVIDENCE_FAMILIES.index("spectral"),
                        BA_IEG_EVIDENCE_FAMILIES.index("spatial_field"),
                    ]
                    if not eligible_rows[:, required].all():
                        raise ValueError("change-score targets require amplitude/spectral/spatial eligibility")
                else:
                    family_index = BA_IEG_EVIDENCE_FAMILIES.index(family)
                    if not eligible_rows[:, family_index].all():
                        raise ValueError(f"{family} deterministic target exceeds view eligibility")

        object.__setattr__(self, "analysis_interval_seconds", interval)
        object.__setattr__(self, "physical_xyz", xyz)
        object.__setattr__(self, "physical_xyz_mask", xyz_mask)
        object.__setattr__(self, "physical_evidence_mask", physical_evidence)
        object.__setattr__(self, "view_future_sample_access", view_future)
        object.__setattr__(self, "view_onset_evidence_authorized", view_onset)
        object.__setattr__(self, "unit_view_index", unit_views)
        object.__setattr__(self, "unit_reference_matrix", reference)
        object.__setattr__(self, "unit_evidence_mask", unit_evidence)
        object.__setattr__(self, "unit_family_mask", unit_families)
        object.__setattr__(self, "token_values", tokens)
        object.__setattr__(self, "token_feature_mask", feature_mask)
        object.__setattr__(self, "token_time_bounds_seconds", times)
        object.__setattr__(self, "token_unit_index", token_units)
        object.__setattr__(self, "token_view_index", token_views)
        object.__setattr__(self, "token_scale_index", scales)
        object.__setattr__(self, "token_signal_mask", signal_mask)
        object.__setattr__(self, "token_family_mask", token_families)
        object.__setattr__(self, "phase_posterior", phase)
        object.__setattr__(self, "token_future_sample_access", token_future)
        object.__setattr__(self, "token_onset_evidence_mask", token_onset)
        object.__setattr__(self, "token_positive_onset_mask", token_positive_onset)
        object.__setattr__(self, "token_phase_context_mask", token_phase_context)
        object.__setattr__(self, "input_receipt_sha256", self._compute_input_sha256())

    @property
    def feature_dim(self) -> int:
        return int(self.token_values.shape[1])

    @property
    def reference_family_count(self) -> int:
        eligible_views = {
            int(self.unit_view_index[index])
            for index in torch.nonzero(
                self.unit_evidence_mask
                & self.unit_family_mask[
                    :, BA_IEG_EVIDENCE_FAMILIES.index("spatial_field")
                ]
                & self.view_onset_evidence_authorized[self.unit_view_index],
                as_tuple=False,
            )
            .flatten()
            .tolist()
        }
        return len({self.reference_families[index] for index in eligible_views})

    def require_multireference_spatial_field(self, *, minimum_families: int = 2) -> None:
        if minimum_families < 2:
            raise ValueError("multi-reference qualification requires at least two families")
        if self.reference_family_count < minimum_families:
            raise ValueError("event lacks independent reference families for spatial-field training")

    def _compute_input_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": BA_IEG_EVENT_TOKEN_SCHEMA_VERSION,
                "identity": {
                    "event_id": self.event_id,
                    "recording_id": self.recording_id,
                    "patient_uid": self.patient_uid,
                    "model_split": self.model_split,
                },
                "time": {
                    "analysis_interval_seconds": list(self.analysis_interval_seconds),
                    "navigation_anchor_seconds": self.navigation_anchor_seconds,
                },
                "lineage": {
                    "canonical_receipt_sha256": self.canonical_receipt_sha256,
                    "adaptive_window_receipt_sha256": self.adaptive_window_receipt_sha256,
                    "encoder_implementation_id": self.encoder_implementation_id,
                    "encoder_lineage": self.encoder_lineage,
                    "encoder_receipt_sha256": self.encoder_receipt_sha256,
                    "feature_scope_sha256": self.feature_scope.sha256,
                },
                "physical_electrode_ids": list(self.physical_electrode_ids),
                "views": [
                    {
                        "view_id": self.view_ids[index],
                        "role": self.view_roles[index],
                        "effective_temporal_role": self.view_effective_temporal_roles[
                            index
                        ],
                        "dependency_policy": self.view_dependency_policies[index],
                        "future_sample_access": bool(
                            self.view_future_sample_access[index]
                        ),
                        "onset_evidence_authorized": bool(
                            self.view_onset_evidence_authorized[index]
                        ),
                        "temporal_evidence_sha256": self.view_temporal_evidence_sha256s[
                            index
                        ],
                        "receipt_sha256": self.view_receipt_sha256s[index],
                        "transform_sha256": self.view_transform_sha256s[index],
                        "reference_family": self.reference_families[index],
                    }
                    for index in range(len(self.view_ids))
                ],
                "units": {
                    "unit_ids": list(self.unit_ids),
                    "unit_source_ids": list(self.unit_source_ids),
                    "unit_types": list(self.unit_types),
                },
                # Deterministic supervision is intentionally absent.  A label
                # or target mutation must not alter the model input receipt.
                "tensor_sha256": {
                    name: _tensor_sha256(getattr(self, name))
                    for name in (
                        "physical_xyz",
                        "physical_xyz_mask",
                        "physical_evidence_mask",
                        "view_future_sample_access",
                        "view_onset_evidence_authorized",
                        "unit_view_index",
                        "unit_reference_matrix",
                        "unit_evidence_mask",
                        "unit_family_mask",
                        "token_values",
                        "token_feature_mask",
                        "token_time_bounds_seconds",
                        "token_unit_index",
                        "token_view_index",
                        "token_scale_index",
                        "token_signal_mask",
                        "token_family_mask",
                        "phase_posterior",
                        "token_future_sample_access",
                        "token_onset_evidence_mask",
                        "token_positive_onset_mask",
                        "token_phase_context_mask",
                    )
                },
            }
        )

    def verify_integrity(self) -> None:
        if self.input_receipt_sha256 != self._compute_input_sha256():
            raise ValueError("BA-IEG event input changed after registration")
        if self.deterministic_targets is not None:
            self.deterministic_targets.verify_integrity()


@dataclass(frozen=True)
class BAIEGCollatedEventBatch:
    """Padded event tensors with model input and supervision kept separate."""

    event_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    patient_uids: tuple[str, ...]
    model_split: str
    input_event_receipt_sha256s: tuple[str, ...]
    token_values: torch.Tensor
    token_feature_mask: torch.Tensor
    token_row_mask: torch.Tensor
    token_signal_mask: torch.Tensor
    token_time_bounds_seconds: torch.Tensor
    token_unit_index: torch.Tensor
    token_view_index: torch.Tensor
    token_scale_index: torch.Tensor
    token_family_mask: torch.Tensor
    phase_posterior: torch.Tensor
    token_future_sample_access: torch.Tensor
    token_onset_evidence_mask: torch.Tensor
    token_positive_onset_mask: torch.Tensor
    token_phase_context_mask: torch.Tensor
    view_row_mask: torch.Tensor
    view_temporal_role_index: torch.Tensor
    view_dependency_policy_index: torch.Tensor
    view_reference_family_index: torch.Tensor
    view_future_sample_access: torch.Tensor
    view_onset_evidence_authorized: torch.Tensor
    view_temporal_evidence_sha256s: tuple[tuple[str | None, ...], ...]
    unit_row_mask: torch.Tensor
    unit_view_index: torch.Tensor
    unit_reference_matrix: torch.Tensor
    unit_evidence_mask: torch.Tensor
    unit_family_mask: torch.Tensor
    physical_xyz: torch.Tensor
    physical_xyz_mask: torch.Tensor
    physical_evidence_mask: torch.Tensor
    deterministic_values: torch.Tensor
    deterministic_value_mask: torch.Tensor
    deterministic_row_mask: torch.Tensor
    deterministic_time_bounds_seconds: torch.Tensor
    deterministic_unit_index: torch.Tensor
    deterministic_view_index: torch.Tensor
    deterministic_receipt_sha256s: tuple[str | None, ...]
    input_batch_sha256: str

    def model_inputs(self) -> dict[str, torch.Tensor]:
        """Return permission-aware shared inputs, never offline phase hints.

        A joint encoder may use these tensors only with the explicit temporal
        masks.  Positive onset heads should prefer :meth:`onset_causal_inputs`;
        the future-derived phase posterior is exposed exclusively by
        :meth:`offline_context_inputs`.
        """

        return {
            name: getattr(self, name)
            for name in (
                "token_values",
                "token_feature_mask",
                "token_row_mask",
                "token_signal_mask",
                "token_time_bounds_seconds",
                "token_unit_index",
                "token_view_index",
                "token_scale_index",
                "token_family_mask",
                "token_future_sample_access",
                "token_onset_evidence_mask",
                "token_positive_onset_mask",
                "token_phase_context_mask",
                "view_row_mask",
                "view_temporal_role_index",
                "view_dependency_policy_index",
                "view_reference_family_index",
                "view_future_sample_access",
                "view_onset_evidence_authorized",
                "unit_row_mask",
                "unit_view_index",
                "unit_reference_matrix",
                "unit_evidence_mask",
                "unit_family_mask",
                "physical_xyz",
                "physical_xyz_mask",
                "physical_evidence_mask",
            )
        }

    def _branch_inputs(self, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(
            self.token_row_mask.shape
        ):
            raise ValueError("branch token mask must align with token rows")
        active = mask & self.token_row_mask & self.token_signal_mask
        expanded = active.unsqueeze(-1)
        result = self.model_inputs()
        result.update(
            {
                "token_values": torch.where(
                    expanded, self.token_values, torch.zeros_like(self.token_values)
                ),
                "token_feature_mask": self.token_feature_mask & expanded,
                "token_row_mask": active,
                "token_signal_mask": active,
                "token_family_mask": self.token_family_mask & expanded,
            }
        )
        return result

    def onset_causal_inputs(self) -> dict[str, torch.Tensor]:
        """Return future-free tokens only; no offline phase posterior exists."""

        return self._branch_inputs(self.token_onset_evidence_mask)

    def positive_onset_inputs(self) -> dict[str, torch.Tensor]:
        """Return only spatial-field-qualified causal tokens for SOZ scoring."""

        return self._branch_inputs(self.token_positive_onset_mask)

    def offline_context_inputs(self) -> dict[str, torch.Tensor]:
        """Return future-dependent context tokens and their phase posterior."""

        result = self._branch_inputs(self.token_phase_context_mask)
        phase_mask = result["token_row_mask"].unsqueeze(-1)
        result["phase_posterior"] = torch.where(
            phase_mask,
            self.phase_posterior,
            torch.zeros_like(self.phase_posterior),
        )
        return result

    def deterministic_supervision(self) -> dict[str, torch.Tensor]:
        return {
            "values": self.deterministic_values,
            "value_mask": self.deterministic_value_mask,
            "row_mask": self.deterministic_row_mask,
            "time_bounds_seconds": self.deterministic_time_bounds_seconds,
            "unit_index": self.deterministic_unit_index,
            "view_index": self.deterministic_view_index,
        }


def collate_ba_ieg_events(events: Sequence[BAIEGEventTokens]) -> BAIEGCollatedEventBatch:
    if not events:
        raise ValueError("cannot collate an empty BA-IEG event sequence")
    if not all(isinstance(event, BAIEGEventTokens) for event in events):
        raise TypeError("collate_ba_ieg_events requires BAIEGEventTokens")
    for event in events:
        event.verify_integrity()
    if len({event.event_id for event in events}) != len(events):
        raise ValueError("duplicate event IDs are forbidden")
    feature_dims = {event.feature_dim for event in events}
    if len(feature_dims) != 1:
        raise ValueError("one event batch must share an encoder feature dimension")
    encoder_receipts = {
        (event.encoder_lineage, event.encoder_implementation_id, event.encoder_receipt_sha256)
        for event in events
    }
    if len(encoder_receipts) != 1:
        raise ValueError("one event batch cannot mix encoder implementations/checkpoints")
    model_splits = {event.model_split for event in events}
    if len(model_splits) != 1:
        raise ValueError("one event batch cannot mix model splits")
    model_split = next(iter(model_splits))

    batch_size = len(events)
    feature_dim = next(iter(feature_dims))
    max_tokens = max(int(event.token_values.shape[0]) for event in events)
    max_units = max(len(event.unit_ids) for event in events)
    max_views = max(len(event.view_ids) for event in events)
    max_targets = max(
        int(event.deterministic_targets.values.shape[0])
        if event.deterministic_targets is not None
        else 0
        for event in events
    )

    def zeros(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    token_values = zeros(batch_size, max_tokens, feature_dim, dtype=torch.float32)
    token_features = zeros(
        batch_size, max_tokens, feature_dim, dtype=torch.bool
    )
    token_rows = zeros(batch_size, max_tokens, dtype=torch.bool)
    token_signal = zeros(batch_size, max_tokens, dtype=torch.bool)
    token_times = zeros(batch_size, max_tokens, 2, dtype=torch.float32)
    token_units = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
    token_views = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
    token_scales = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
    token_families = zeros(
        batch_size, max_tokens, len(BA_IEG_EVIDENCE_FAMILIES), dtype=torch.bool
    )
    phase = zeros(batch_size, max_tokens, len(BA_IEG_PHASE_STATES), dtype=torch.float32)
    token_future = zeros(batch_size, max_tokens, dtype=torch.bool)
    token_onset = zeros(batch_size, max_tokens, dtype=torch.bool)
    token_positive_onset = zeros(batch_size, max_tokens, dtype=torch.bool)
    token_phase_context = zeros(batch_size, max_tokens, dtype=torch.bool)
    view_rows = zeros(batch_size, max_views, dtype=torch.bool)
    view_temporal_roles = torch.full(
        (batch_size, max_views), -1, dtype=torch.long
    )
    view_dependencies = torch.full(
        (batch_size, max_views), -1, dtype=torch.long
    )
    view_reference_families = torch.full(
        (batch_size, max_views), -1, dtype=torch.long
    )
    view_future = zeros(batch_size, max_views, dtype=torch.bool)
    view_onset = zeros(batch_size, max_views, dtype=torch.bool)
    unit_rows = zeros(batch_size, max_units, dtype=torch.bool)
    unit_views = torch.full((batch_size, max_units), -1, dtype=torch.long)
    reference = zeros(batch_size, max_units, len(STANDARD_19), dtype=torch.float32)
    unit_evidence = zeros(batch_size, max_units, dtype=torch.bool)
    unit_families = zeros(
        batch_size, max_units, len(BA_IEG_EVIDENCE_FAMILIES), dtype=torch.bool
    )
    xyz = zeros(batch_size, len(STANDARD_19), 3, dtype=torch.float32)
    xyz_mask = zeros(batch_size, len(STANDARD_19), dtype=torch.bool)
    physical_evidence = zeros(batch_size, len(STANDARD_19), dtype=torch.bool)
    target_values = zeros(
        batch_size, max_targets, len(BA_IEG_DETERMINISTIC_TARGETS), dtype=torch.float32
    )
    target_masks = zeros(
        batch_size, max_targets, len(BA_IEG_DETERMINISTIC_TARGETS), dtype=torch.bool
    )
    target_rows = zeros(batch_size, max_targets, dtype=torch.bool)
    target_times = zeros(batch_size, max_targets, 2, dtype=torch.float32)
    target_units = torch.full((batch_size, max_targets), -1, dtype=torch.long)
    target_views = torch.full((batch_size, max_targets), -1, dtype=torch.long)

    target_receipts: list[str | None] = []
    for batch_index, event in enumerate(events):
        n_tokens = int(event.token_values.shape[0])
        n_units = len(event.unit_ids)
        n_views = len(event.view_ids)
        token_values[batch_index, :n_tokens] = event.token_values.to(torch.float32)
        token_features[batch_index, :n_tokens] = event.token_feature_mask
        token_rows[batch_index, :n_tokens] = True
        token_signal[batch_index, :n_tokens] = event.token_signal_mask
        token_times[batch_index, :n_tokens] = event.token_time_bounds_seconds.to(torch.float32)
        token_units[batch_index, :n_tokens] = event.token_unit_index
        token_views[batch_index, :n_tokens] = event.token_view_index
        token_scales[batch_index, :n_tokens] = event.token_scale_index
        token_families[batch_index, :n_tokens] = event.token_family_mask
        phase[batch_index, :n_tokens] = event.phase_posterior.to(torch.float32)
        token_future[batch_index, :n_tokens] = event.token_future_sample_access
        token_onset[batch_index, :n_tokens] = event.token_onset_evidence_mask
        token_positive_onset[batch_index, :n_tokens] = (
            event.token_positive_onset_mask
        )
        token_phase_context[batch_index, :n_tokens] = event.token_phase_context_mask
        view_rows[batch_index, :n_views] = True
        view_temporal_roles[batch_index, :n_views] = torch.tensor(
            [
                BA_IEG_EFFECTIVE_TEMPORAL_ROLES.index(role)
                for role in event.view_effective_temporal_roles
            ],
            dtype=torch.long,
        )
        view_dependencies[batch_index, :n_views] = torch.tensor(
            [
                BA_IEG_DEPENDENCY_POLICIES.index(policy)
                for policy in event.view_dependency_policies
            ],
            dtype=torch.long,
        )
        view_reference_families[batch_index, :n_views] = torch.tensor(
            [
                BA_IEG_REFERENCE_FAMILIES.index(reference_family)
                for reference_family in event.reference_families
            ],
            dtype=torch.long,
        )
        view_future[batch_index, :n_views] = event.view_future_sample_access
        view_onset[batch_index, :n_views] = event.view_onset_evidence_authorized
        unit_rows[batch_index, :n_units] = True
        unit_views[batch_index, :n_units] = event.unit_view_index
        reference[batch_index, :n_units] = event.unit_reference_matrix.to(torch.float32)
        unit_evidence[batch_index, :n_units] = event.unit_evidence_mask
        unit_families[batch_index, :n_units] = event.unit_family_mask
        xyz[batch_index] = event.physical_xyz.to(torch.float32)
        xyz_mask[batch_index] = event.physical_xyz_mask
        physical_evidence[batch_index] = event.physical_evidence_mask
        targets = event.deterministic_targets
        if targets is None:
            target_receipts.append(None)
        else:
            n_targets = int(targets.values.shape[0])
            target_values[batch_index, :n_targets] = targets.values.to(torch.float32)
            target_masks[batch_index, :n_targets] = targets.value_mask
            target_rows[batch_index, :n_targets] = True
            target_times[batch_index, :n_targets] = targets.row_time_bounds_seconds.to(torch.float32)
            target_units[batch_index, :n_targets] = targets.row_unit_index
            target_views[batch_index, :n_targets] = targets.row_view_index
            target_receipts.append(targets.receipt_sha256)

    input_hash = _canonical_sha256(
        {
            "schema": "ba_ieg_collated_event_model_inputs_v3",
            "model_split": model_split,
            "event_input_receipts": [event.input_receipt_sha256 for event in events],
            "view_temporal_evidence_sha256s": [
                list(event.view_temporal_evidence_sha256s) for event in events
            ],
            "tensor_sha256": {
                name: _tensor_sha256(value)
                for name, value in {
                    "token_values": token_values,
                    "token_feature_mask": token_features,
                    "token_row_mask": token_rows,
                    "token_signal_mask": token_signal,
                    "token_time_bounds_seconds": token_times,
                    "token_unit_index": token_units,
                    "token_view_index": token_views,
                    "token_scale_index": token_scales,
                    "token_family_mask": token_families,
                    "phase_posterior": phase,
                    "token_future_sample_access": token_future,
                    "token_onset_evidence_mask": token_onset,
                    "token_positive_onset_mask": token_positive_onset,
                    "token_phase_context_mask": token_phase_context,
                    "view_row_mask": view_rows,
                    "view_temporal_role_index": view_temporal_roles,
                    "view_dependency_policy_index": view_dependencies,
                    "view_reference_family_index": view_reference_families,
                    "view_future_sample_access": view_future,
                    "view_onset_evidence_authorized": view_onset,
                    "unit_row_mask": unit_rows,
                    "unit_view_index": unit_views,
                    "unit_reference_matrix": reference,
                    "unit_evidence_mask": unit_evidence,
                    "unit_family_mask": unit_families,
                    "physical_xyz": xyz,
                    "physical_xyz_mask": xyz_mask,
                    "physical_evidence_mask": physical_evidence,
                }.items()
            },
        }
    )
    if max_views < 1:  # pragma: no cover - guarded by BAIEGEventTokens
        raise RuntimeError("collation lost all views")
    return BAIEGCollatedEventBatch(
        event_ids=tuple(event.event_id for event in events),
        recording_ids=tuple(event.recording_id for event in events),
        patient_uids=tuple(event.patient_uid for event in events),
        model_split=model_split,
        input_event_receipt_sha256s=tuple(event.input_receipt_sha256 for event in events),
        token_values=token_values,
        token_feature_mask=token_features,
        token_row_mask=token_rows,
        token_signal_mask=token_signal,
        token_time_bounds_seconds=token_times,
        token_unit_index=token_units,
        token_view_index=token_views,
        token_scale_index=token_scales,
        token_family_mask=token_families,
        phase_posterior=phase,
        token_future_sample_access=token_future,
        token_onset_evidence_mask=token_onset,
        token_positive_onset_mask=token_positive_onset,
        token_phase_context_mask=token_phase_context,
        view_row_mask=view_rows,
        view_temporal_role_index=view_temporal_roles,
        view_dependency_policy_index=view_dependencies,
        view_reference_family_index=view_reference_families,
        view_future_sample_access=view_future,
        view_onset_evidence_authorized=view_onset,
        view_temporal_evidence_sha256s=tuple(
            tuple(event.view_temporal_evidence_sha256s)
            + (None,) * (max_views - len(event.view_ids))
            for event in events
        ),
        unit_row_mask=unit_rows,
        unit_view_index=unit_views,
        unit_reference_matrix=reference,
        unit_evidence_mask=unit_evidence,
        unit_family_mask=unit_families,
        physical_xyz=xyz,
        physical_xyz_mask=xyz_mask,
        physical_evidence_mask=physical_evidence,
        deterministic_values=target_values,
        deterministic_value_mask=target_masks,
        deterministic_row_mask=target_rows,
        deterministic_time_bounds_seconds=target_times,
        deterministic_unit_index=target_units,
        deterministic_view_index=target_views,
        deterministic_receipt_sha256s=tuple(target_receipts),
        input_batch_sha256=input_hash,
    )


@dataclass(frozen=True)
class BAIEGPatientEventBagManifest:
    """Target-independent complete event roster for one patient."""

    patient_uid: str
    model_split: str
    event_ids: tuple[str, ...]
    roster_source: str = "frozen_detector_event_roster_eeg_only_v1"
    target_conditioned_selection: bool = False
    private_labels_used_for_selection: bool = False
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "patient_uid", _identifier(self.patient_uid, "patient_uid"))
        if self.model_split not in {"source_train", "source_dev", "source_eval"}:
            raise ValueError("positive-set bags are restricted to public source splits")
        if self.roster_source != "frozen_detector_event_roster_eeg_only_v1":
            raise ValueError("patient event roster must be frozen independently of targets")
        if self.target_conditioned_selection is not False or self.private_labels_used_for_selection is not False:
            raise ValueError("target/private-label-conditioned event selection is forbidden")
        event_ids = tuple(_identifier(item, "event_id") for item in self.event_ids)
        if not event_ids or len(set(event_ids)) != len(event_ids):
            raise ValueError("patient event roster must be unique and non-empty")
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(
            self,
            "receipt_sha256",
            _canonical_sha256(
                {
                    "schema": "ba_ieg_target_independent_event_roster_v1",
                    "patient_uid": self.patient_uid,
                    "model_split": self.model_split,
                    "event_ids": list(self.event_ids),
                    "roster_source": self.roster_source,
                    "target_conditioned_selection": False,
                    "private_labels_used_for_selection": False,
                }
            ),
        )


@dataclass(frozen=True)
class BAIEGDeepSOZPositiveSet:
    """Known-positive scalp electrodes; all other candidates remain unknown."""

    patient_uid: str
    model_split: str
    positive_electrode_ids: tuple[str, ...]
    source_reference_sha256: str
    candidate_electrode_ids: tuple[str, ...] = BA_IEG_C18
    label_source: str = "deepsoz_public_patient_positive_set_v1"
    semantics: str = "scalp_visible_positive_set_not_cortical_soz"
    private_source: bool = False
    training_only_not_model_input: bool = True
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "patient_uid", _identifier(self.patient_uid, "patient_uid"))
        if self.model_split not in {"source_train", "source_dev", "source_eval"}:
            raise ValueError("DeepSOZ reference split is unsupported")
        if tuple(self.candidate_electrode_ids) != BA_IEG_C18:
            raise ValueError("BA-IEG DeepSOZ v1 uses the frozen C18 candidate space")
        positives = tuple(self.positive_electrode_ids)
        if not positives or len(set(positives)) != len(positives) or not set(positives).issubset(BA_IEG_C18):
            raise ValueError("DeepSOZ positive set must be a non-empty C18 subset")
        if tuple(sorted(positives, key=BA_IEG_C18.index)) != positives:
            raise ValueError("DeepSOZ positives must follow canonical C18 order")
        if self.label_source != "deepsoz_public_patient_positive_set_v1" or self.semantics != (
            "scalp_visible_positive_set_not_cortical_soz"
        ):
            raise ValueError("DeepSOZ label semantics drifted")
        if self.private_source is not False or self.training_only_not_model_input is not True:
            raise ValueError("private labels/model-input targets are forbidden")
        _sha256(self.source_reference_sha256, "source_reference_sha256")
        object.__setattr__(
            self,
            "receipt_sha256",
            _canonical_sha256(
                {
                    "schema": "ba_ieg_deepsoz_positive_set_v1",
                    "patient_uid": self.patient_uid,
                    "model_split": self.model_split,
                    "positive_electrode_ids": list(positives),
                    "candidate_electrode_ids": list(self.candidate_electrode_ids),
                    "source_reference_sha256": self.source_reference_sha256,
                    "label_source": self.label_source,
                    "semantics": self.semantics,
                    "private_source": False,
                    "training_only_not_model_input": True,
                }
            ),
        )

    @property
    def positive_mask(self) -> torch.Tensor:
        return torch.tensor(
            [electrode in set(self.positive_electrode_ids) for electrode in STANDARD_19],
            dtype=torch.bool,
        )

    @property
    def candidate_mask(self) -> torch.Tensor:
        return torch.tensor(
            [electrode in set(self.candidate_electrode_ids) for electrode in STANDARD_19],
            dtype=torch.bool,
        )


def deepsoz_positive_set_from_reference(reference: object) -> BAIEGDeepSOZPositiveSet:
    """Project the existing DeepSOZ registry row to positive-only semantics.

    Explicit dataset zeros and missing fields are both discarded here.  They
    are not serialized as per-channel negative targets.  The C18 candidate
    mask supplies the listwise denominator, while only explicit observed ones
    enter the numerator.
    """

    required = ("patient_id", "model_split", "values", "mask", "target_states")
    if any(not hasattr(reference, name) for name in required):
        raise TypeError("reference is not a DeepSOZ patient reference")
    values = getattr(reference, "values")
    masks = getattr(reference, "mask")
    if not isinstance(values, torch.Tensor) or not isinstance(masks, torch.Tensor) or tuple(values.shape) != (
        len(STANDARD_19),
    ) or tuple(masks.shape) != (len(STANDARD_19),):
        raise ValueError("DeepSOZ reference tensors must use standard-19")
    positives = tuple(
        electrode
        for index, electrode in enumerate(STANDARD_19)
        if electrode != "PZ" and bool(masks[index]) and float(values[index]) == 1.0
    )
    source_hash = _canonical_sha256(
        {
            "patient_id": str(getattr(reference, "patient_id")),
            "model_split": str(getattr(reference, "model_split")),
            "values_sha256": _tensor_sha256(values),
            "mask_sha256": _tensor_sha256(masks),
            "target_states": list(getattr(reference, "target_states")),
            "source_record_count": int(getattr(reference, "source_record_count", 0)),
        }
    )
    return BAIEGDeepSOZPositiveSet(
        patient_uid=str(getattr(reference, "patient_id")),
        model_split=str(getattr(reference, "model_split")),
        positive_electrode_ids=positives,
        source_reference_sha256=source_hash,
    )


@dataclass(frozen=True)
class BAIEGPatientBag:
    manifest: BAIEGPatientEventBagManifest
    events: tuple[BAIEGEventTokens, ...]
    positive_set: BAIEGDeepSOZPositiveSet

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, BAIEGPatientEventBagManifest) or not isinstance(
            self.positive_set, BAIEGDeepSOZPositiveSet
        ):
            raise TypeError("BA-IEG patient bag has invalid manifest/reference types")
        if not self.events or not all(isinstance(event, BAIEGEventTokens) for event in self.events):
            raise TypeError("BA-IEG patient bag requires event-token inputs")
        ids = tuple(event.event_id for event in self.events)
        if ids != self.manifest.event_ids:
            raise ValueError("patient bag must contain the frozen complete event roster in order")
        if any(event.patient_uid != self.manifest.patient_uid for event in self.events):
            raise ValueError("patient bag crosses patient identities")
        if any(event.model_split != self.manifest.model_split for event in self.events):
            raise ValueError("patient bag crosses model splits")
        if self.positive_set.patient_uid != self.manifest.patient_uid or self.positive_set.model_split != (
            self.manifest.model_split
        ):
            raise ValueError("positive-set reference does not match the frozen patient bag")
        observation_opportunity = torch.stack(
            [event.physical_evidence_mask for event in self.events]
        ).any(dim=0)
        if torch.any(self.positive_set.positive_mask & ~observation_opportunity):
            raise ValueError(
                "DeepSOZ positive set contains an electrode with no EEG observation opportunity"
            )


class BAIEGPatientBagDataset(Sequence[BAIEGPatientBag]):
    """One index per patient with a complete, target-independent event bag."""

    _PURPOSE_SPLIT = {
        "train": "source_train",
        "calibrate": "source_dev",
        "evaluate": "source_eval",
    }

    def __init__(
        self,
        events: Sequence[BAIEGEventTokens],
        manifests: Sequence[BAIEGPatientEventBagManifest],
        references: Sequence[BAIEGDeepSOZPositiveSet],
        *,
        purpose: str,
    ) -> None:
        if purpose not in self._PURPOSE_SPLIT:
            raise ValueError("purpose must be train, calibrate or evaluate")
        expected_split = self._PURPOSE_SPLIT[purpose]
        by_event = {event.event_id: event for event in events}
        by_manifest = {item.patient_uid: item for item in manifests}
        by_reference = {item.patient_uid: item for item in references}
        if len(by_event) != len(events) or len(by_manifest) != len(manifests) or len(by_reference) != len(references):
            raise ValueError("dataset inputs contain duplicate event/patient identities")
        if not by_manifest or set(by_manifest) != set(by_reference):
            raise ValueError("manifest/reference patient rosters must match and be non-empty")
        if any(item.model_split != expected_split for item in manifests) or any(
            item.model_split != expected_split for item in references
        ) or any(event.model_split != expected_split for event in events):
            raise ValueError("dataset purpose is inconsistent with its patient-disjoint split")
        expected_event_ids = {
            event_id for manifest in manifests for event_id in manifest.event_ids
        }
        if set(by_event) != expected_event_ids:
            raise ValueError(
                "dataset requires every and only event in the frozen rosters; "
                f"missing={sorted(expected_event_ids - set(by_event))[:5]}, "
                f"extra={sorted(set(by_event) - expected_event_ids)[:5]}"
            )
        patient_uids = tuple(sorted(by_manifest))
        bags = tuple(
            BAIEGPatientBag(
                manifest=by_manifest[patient_uid],
                events=tuple(by_event[event_id] for event_id in by_manifest[patient_uid].event_ids),
                positive_set=by_reference[patient_uid],
            )
            for patient_uid in patient_uids
        )
        self._purpose = purpose
        self._model_split = expected_split
        self._patient_uids = patient_uids
        self._bags = bags

    @property
    def patient_uids(self) -> tuple[str, ...]:
        return self._patient_uids

    @property
    def model_split(self) -> str:
        return self._model_split

    def __len__(self) -> int:
        return len(self._bags)

    def __getitem__(self, index: int) -> BAIEGPatientBag:
        return self._bags[index]


@dataclass(frozen=True)
class BAIEGPatientBagBatch:
    event_batch: BAIEGCollatedEventBatch
    patient_uids: tuple[str, ...]
    event_patient_index: torch.Tensor
    expected_event_counts: torch.Tensor
    positive_mask: torch.Tensor
    candidate_mask: torch.Tensor
    roster_receipt_sha256s: tuple[str, ...]
    target_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        patients = len(self.patient_uids)
        events = len(self.event_batch.event_ids)
        if tuple(self.event_patient_index.shape) != (events,) or self.event_patient_index.dtype != torch.long:
            raise ValueError("event_patient_index must be long [E]")
        if tuple(self.expected_event_counts.shape) != (patients,) or self.expected_event_counts.dtype != torch.long:
            raise ValueError("expected_event_counts must be long [P]")
        if tuple(self.positive_mask.shape) != (patients, len(STANDARD_19)) or self.positive_mask.dtype != torch.bool:
            raise ValueError("positive_mask must be bool [P,19]")
        if tuple(self.candidate_mask.shape) != tuple(self.positive_mask.shape) or self.candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be bool [P,19]")
        if torch.any(self.positive_mask & ~self.candidate_mask):
            raise ValueError("known positives must lie inside the listwise candidate set")
        if not self.positive_mask.any(dim=1).all():
            raise ValueError("each patient needs at least one known positive")
        observed_counts = torch.bincount(self.event_patient_index, minlength=patients)
        if not torch.equal(observed_counts, self.expected_event_counts):
            raise ValueError("collated patient batch lost or duplicated a frozen event")

    def model_inputs(self) -> dict[str, torch.Tensor]:
        return self.event_batch.model_inputs()

    def positive_set_supervision(self) -> dict[str, torch.Tensor]:
        # No negative/zero target tensor is exposed.
        return {
            "positive_mask": self.positive_mask,
            "candidate_mask": self.candidate_mask,
            "event_patient_index": self.event_patient_index,
            "expected_event_counts": self.expected_event_counts,
        }


def collate_ba_ieg_patient_bags(
    bags: Sequence[BAIEGPatientBag],
) -> BAIEGPatientBagBatch:
    if not bags or not all(isinstance(bag, BAIEGPatientBag) for bag in bags):
        raise TypeError("collation requires a non-empty BAIEGPatientBag sequence")
    patient_uids = tuple(bag.manifest.patient_uid for bag in bags)
    if len(set(patient_uids)) != len(patient_uids):
        raise ValueError("a patient may appear only once in one optimizer batch")
    splits = {bag.manifest.model_split for bag in bags}
    if len(splits) != 1:
        raise ValueError("one patient batch cannot mix data splits")
    events = tuple(event for bag in bags for event in bag.events)
    event_batch = collate_ba_ieg_events(events)
    event_patient_index = torch.tensor(
        [patient_index for patient_index, bag in enumerate(bags) for _ in bag.events],
        dtype=torch.long,
    )
    return BAIEGPatientBagBatch(
        event_batch=event_batch,
        patient_uids=patient_uids,
        event_patient_index=event_patient_index,
        expected_event_counts=torch.tensor([len(bag.events) for bag in bags], dtype=torch.long),
        positive_mask=torch.stack([bag.positive_set.positive_mask for bag in bags]),
        candidate_mask=torch.stack(
            [
                bag.positive_set.candidate_mask
                & torch.stack(
                    [event.physical_evidence_mask for event in bag.events]
                ).any(dim=0)
                for bag in bags
            ]
        ),
        roster_receipt_sha256s=tuple(bag.manifest.receipt_sha256 for bag in bags),
        target_receipt_sha256s=tuple(bag.positive_set.receipt_sha256 for bag in bags),
    )


def positive_set_mass_loss(
    patient_logits: torch.Tensor,
    batch: BAIEGPatientBagBatch,
) -> torch.Tensor:
    """Listwise partial-label loss without creating per-channel negatives."""

    if not isinstance(batch, BAIEGPatientBagBatch):
        raise TypeError("positive_set_mass_loss requires BAIEGPatientBagBatch")
    if not isinstance(patient_logits, torch.Tensor) or tuple(patient_logits.shape) != (
        len(batch.patient_uids),
        len(STANDARD_19),
    ) or not patient_logits.is_floating_point() or not torch.isfinite(patient_logits).all():
        raise ValueError("patient_logits must be finite floating point [P,19]")
    if patient_logits.device != batch.positive_mask.device:
        raise ValueError("patient logits and supervision masks must share a device")
    rows = []
    for patient_index in range(len(batch.patient_uids)):
        candidates = batch.candidate_mask[patient_index]
        positives = batch.positive_mask[patient_index]
        rows.append(
            torch.logsumexp(patient_logits[patient_index, candidates], dim=0)
            - torch.logsumexp(patient_logits[patient_index, positives], dim=0)
        )
    return torch.stack(rows).mean()


def validate_patient_disjoint_event_partitions(
    partitions: Mapping[str, Sequence[BAIEGEventTokens]],
) -> None:
    """Reject patient, recording, event or canonical-signal overlap across splits.

    ``patient_uid`` must already be a cross-corpus stable identity.  This
    validator cannot infer that two differently formatted TUH identifiers are
    the same person; the upstream identity ledger remains a required gate.
    """

    if not partitions:
        raise ValueError("patient-disjoint validation requires partitions")
    seen_patients: dict[str, str] = {}
    seen_recordings: dict[str, str] = {}
    seen_events: dict[str, str] = {}
    seen_canonical: dict[str, str] = {}
    for split, events in partitions.items():
        if split not in {"source_train", "source_dev", "source_eval"}:
            raise ValueError("partition key must be a public source split")
        for event in events:
            if not isinstance(event, BAIEGEventTokens):
                raise TypeError("partitions must contain BAIEGEventTokens")
            event.verify_integrity()
            if event.model_split != split:
                raise ValueError("event model_split disagrees with its partition")
            for value, registry, label in (
                (event.patient_uid, seen_patients, "patient"),
                (event.recording_id, seen_recordings, "recording"),
                (event.event_id, seen_events, "event"),
                (event.canonical_receipt_sha256, seen_canonical, "canonical signal"),
            ):
                previous = registry.setdefault(value, split)
                if previous != split:
                    raise ValueError(f"{label} crosses patient-disjoint splits")


@dataclass(frozen=True)
class BAIEGP0TokenizationPolicy:
    """Frozen P0 physical-time tiling and deterministic measurement policy.

    Durations and steps are expressed on one recording-relative physical-time
    grid.  Each already-materialized task view maps those nominal intervals
    inward on its own sampling clock.  An event is never stretched, squeezed,
    resampled or silently padded to obtain a fixed token count.  Coarse/context
    tiles must be complete.  Only trailing fine tiles may be shorter than the
    nominal duration, and they remain subject to a per-view minimum sample
    count.
    """

    fine_duration_seconds: float = 1.0
    fine_step_seconds: float = 1.0
    coarse_duration_seconds: float = 4.0
    coarse_step_seconds: float = 4.0
    context_duration_seconds: float = 16.0
    context_step_seconds: float = 16.0
    minimum_fine_samples: int = 4
    spectral_low_hz: float = 0.5
    spectral_high_hz: float = 45.0
    rhythmic_half_bandwidth_hz: float = 0.5

    def __post_init__(self) -> None:
        durations = (
            float(self.fine_duration_seconds),
            float(self.coarse_duration_seconds),
            float(self.context_duration_seconds),
        )
        steps = (
            float(self.fine_step_seconds),
            float(self.coarse_step_seconds),
            float(self.context_step_seconds),
        )
        if any(not math.isfinite(value) or value <= 0 for value in durations + steps):
            raise ValueError("P0 token durations and steps must be finite and positive")
        for index, duration in enumerate(durations):
            lower, upper = _SCALE_DURATION_BOUNDS[index]
            if index == 0:
                valid = lower < duration <= upper
            else:
                valid = lower < duration <= upper
            if not valid:
                raise ValueError("P0 token duration violates the BA-IEG scale contract")
        if any(step > duration for step, duration in zip(steps, durations)):
            raise ValueError("P0 token steps cannot leave physical-time gaps")
        if type(self.minimum_fine_samples) is not int or self.minimum_fine_samples < 2:
            raise ValueError("minimum_fine_samples must be an integer >= 2")
        spectral = (
            float(self.spectral_low_hz),
            float(self.spectral_high_hz),
            float(self.rhythmic_half_bandwidth_hz),
        )
        if (
            any(not math.isfinite(value) or value <= 0 for value in spectral)
            or spectral[0] >= spectral[1]
        ):
            raise ValueError("P0 spectral measurement policy is invalid")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "ba_ieg_p0_tokenization_policy_v1",
            "scale_duration_seconds": {
                "fine": float(self.fine_duration_seconds),
                "coarse": float(self.coarse_duration_seconds),
                "context": float(self.context_duration_seconds),
            },
            "scale_step_seconds": {
                "fine": float(self.fine_step_seconds),
                "coarse": float(self.coarse_step_seconds),
                "context": float(self.context_step_seconds),
            },
            "minimum_fine_samples": int(self.minimum_fine_samples),
            "spectral_interval_hz": [
                float(self.spectral_low_hz),
                float(self.spectral_high_hz),
            ],
            "rhythmic_half_bandwidth_hz": float(
                self.rhythmic_half_bandwidth_hz
            ),
            "fft_window": "periodic_hann_v1",
            "coarse_or_context_partial_tiles_used": False,
            "event_local_resampling_used": False,
            "event_time_warp_used": False,
            "silent_padding_used": False,
        }


_P0_SCOPE_RECEIPT: Final[dict[str, bool]] = {
    "eeg_samples_used": True,
    "detector_anchor_used_for_navigation_only": True,
    "detector_provider_tensor_used_as_findings": False,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "private_doctor_labels_used": False,
    "public_targets_used_as_model_input": False,
    "clinical_text_used": False,
    "video_used": False,
    "sleep_or_activation_labels_used": False,
    "ecg_emg_eog_used": False,
    "event_local_resampling_used": False,
    "event_time_warp_used": False,
    "silent_padding_used": False,
}

_P0_SCOPE_RECEIPT_A0: Final[dict[str, object]] = {
    "eeg_samples_used": True,
    "detector_anchor_used_for_navigation_only": False,
    "public_tusz_seizure_interval_used_for_navigation_only": True,
    "oracle_navigation_used": True,
    "detector_output_used": False,
    "detector_frozen_claim_authorized": False,
    "detector_provider_tensor_used_as_findings": False,
    "oracle_interval_used_as_phase_feature": False,
    "oracle_interval_available_to_model_forward": False,
    "initial_support_only": True,
    "fixed_watchdog_is_final_analysis_window": False,
    "final_support_requires_iterative_rule_adaptive_acquisition": True,
    "iterative_rule_adaptive_acquisition_status": "not_materialized",
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "private_doctor_labels_used": False,
    "public_targets_used_as_model_input": False,
    "clinical_text_used": False,
    "video_used": False,
    "sleep_or_activation_labels_used": False,
    "ecg_emg_eog_used": False,
    "event_local_resampling_used": False,
    "event_time_warp_used": False,
    "silent_padding_used": False,
}

_P0_FAILURE_CODES: Final[frozenset[str]] = frozenset(
    {
        "invalid_canonical_bundle",
        "invalid_adaptive_search_receipt",
        "invalid_adaptive_window_receipt",
        "invalid_a0_navigation_window",
        "a0_navigation_argument_mismatch",
        "a0_canonical_identity_mismatch",
        "adaptive_receipt_binding_mismatch",
        "canonical_adaptive_identity_mismatch",
        "recording_clock_mismatch",
        "event_interval_unavailable",
        "view_clock_or_reference_mismatch",
        "no_evidence_eligible_tokens",
        "tokenization_failed",
    }
)


def validate_ba_ieg_p0_materialization_receipt(
    payload: object,
) -> dict[str, Any]:
    """Validate a content-addressed P0 success or failure receipt."""

    if type(payload) is not dict:
        raise TypeError("BA-IEG P0 materialization receipt must be an object")
    required = {
        "schema_version",
        "receipt_id",
        "status",
        "failure_code",
        "failure_stage",
        "event_identity",
        "lineage",
        "timing",
        "censoring",
        "eligibility",
        "views",
        "masks",
        "tokens",
        "policy",
        "scope_receipt",
        "receipt_sha256",
    }
    schema_version = payload.get("schema_version")
    if schema_version == BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION:
        expected_view_profile = BA_IEG_P0_VIEW_PROFILE_LEGACY_8
        a0_navigation = False
    elif schema_version == BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_NATIVE_12:
        required.add("view_profile")
        expected_view_profile = BA_IEG_P0_VIEW_PROFILE_NATIVE_12
        a0_navigation = False
    elif (
        schema_version
        == BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_A0_NATIVE_12
    ):
        required.update(
            {
                "view_profile",
                "navigation_arm",
                "evaluation_semantics",
                "support_role",
            }
        )
        expected_view_profile = BA_IEG_P0_VIEW_PROFILE_NATIVE_12
        a0_navigation = True
    else:
        raise ValueError("BA-IEG P0 materialization schema drifted")
    if set(payload) != required:
        raise ValueError("BA-IEG P0 receipt has missing or unknown fields")
    data = deepcopy(payload)
    if (
        schema_version
        in {
            BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_NATIVE_12,
            BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_A0_NATIVE_12,
        }
        and data["view_profile"] != expected_view_profile
    ):
        raise ValueError("BA-IEG P0 native-12 view profile drifted")
    if a0_navigation and (
        data["navigation_arm"] != BA_IEG_P0_NAVIGATION_ARM_A0
        or data["evaluation_semantics"]
        != BA_IEG_P0_A0_EVALUATION_SEMANTICS
        or data["support_role"] != "initial_bootstrap_watchdog_only"
    ):
        raise ValueError("BA-IEG P0 A0 oracle-navigation authority drifted")
    if data["status"] not in {"materialized", "failed"}:
        raise ValueError("BA-IEG P0 materialization status is invalid")
    identity = data["event_identity"]
    if type(identity) is not dict or set(identity) != {
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
    }:
        raise ValueError("BA-IEG P0 event identity is invalid")
    for key in ("event_id", "recording_id", "patient_uid"):
        _identifier(identity[key], f"event_identity.{key}")
    if identity["model_split"] not in BA_IEG_ALLOWED_SPLITS:
        raise ValueError("BA-IEG P0 model split is invalid")
    if a0_navigation and identity["model_split"] != "source_train":
        raise ValueError("BA-IEG P0 A0 materialization is source-train-only")
    expected_scope = _P0_SCOPE_RECEIPT_A0 if a0_navigation else _P0_SCOPE_RECEIPT
    if data["scope_receipt"] != expected_scope:
        raise ValueError("BA-IEG P0 materialization violates the EEG-only scope")
    if (
        type(data["policy"]) is not dict
        or data["policy"].get("event_time_warp_used") is not False
        or data["policy"].get("silent_padding_used") is not False
    ):
        raise ValueError("BA-IEG P0 policy reverted to warped/padded event semantics")
    failure = data["failure_code"]
    if data["status"] == "materialized":
        if failure is not None or data["failure_stage"] is not None:
            raise ValueError("successful BA-IEG P0 receipt cannot carry a failure")
        if (
            not isinstance(data["tokens"], dict)
            or int(data["tokens"].get("token_count", 0)) < 1
        ):
            raise ValueError("successful BA-IEG P0 receipt needs tokens")
        onset_qualification = data["tokens"].get("clinical_onset_input_qualification")
        if type(onset_qualification) is not dict or set(onset_qualification) != {
            "status",
            "onset_localization_input_authorized",
            "research_channel_ranking_input_authorized",
            "reason_codes",
        }:
            raise ValueError(
                "successful BA-IEG P0 receipt needs typed onset-input qualification"
            )
        onset_authorized = onset_qualification["onset_localization_input_authorized"]
        ranking_authorized = onset_qualification[
            "research_channel_ranking_input_authorized"
        ]
        reasons = onset_qualification["reason_codes"]
        if (
            type(onset_authorized) is not bool
            or type(ranking_authorized) is not bool
            or onset_qualification["status"] not in {"evaluable", "not_evaluable"}
            or not isinstance(reasons, list)
            or any(not isinstance(item, str) or not item for item in reasons)
            or len(reasons) != len(set(reasons))
            or reasons != sorted(reasons)
            or ranking_authorized
            and not onset_authorized
            or (onset_qualification["status"] == "evaluable") is not onset_authorized
            or bool(reasons) is onset_authorized
        ):
            raise ValueError("P0 onset-input qualification is inconsistent")
        masks = data["masks"]
        if (
            not isinstance(masks, dict)
            or (
                onset_authorized
                is not (int(masks.get("onset_evidence_eligible_token_count", 0)) > 0)
            )
            or (
                ranking_authorized
                is not (int(masks.get("positive_onset_spatial_token_count", 0)) > 0)
            )
        ):
            raise ValueError(
                "P0 onset-input qualification disagrees with permission masks"
            )
        _sha256(
            data["tokens"].get("input_receipt_sha256"),
            "P0 input_receipt_sha256",
        )
        expected_view_count = (
            8
            if expected_view_profile == BA_IEG_P0_VIEW_PROFILE_LEGACY_8
            else 12
        )
        if len(data["views"]) != expected_view_count:
            raise ValueError("successful BA-IEG P0 receipt has the wrong view count")
        if expected_view_profile == BA_IEG_P0_VIEW_PROFILE_NATIVE_12 and (
            data["tokens"].get("native_view_used_as_model_input") is not True
            or data["tokens"].get("native_view_used_as_dense_supervision")
            is not False
            or data["tokens"].get("dense_supervision_view_count") != 8
        ):
            raise ValueError(
                "native-12 P0 receipt does not disclose its supervision boundary"
            )
        if a0_navigation and (
            data["tokens"].get("oracle_interval_used_as_phase_feature")
            is not False
            or data["tokens"].get("oracle_interval_available_to_model_forward")
            is not False
            or data["tokens"].get("initial_support_only") is not True
            or data["tokens"].get("final_rule_adaptive_support_materialized")
            is not False
        ):
            raise ValueError("A0 P0 receipt leaked oracle/final-window semantics")
    else:
        if failure not in _P0_FAILURE_CODES or not isinstance(
            data["failure_stage"], str
        ):
            raise ValueError("failed BA-IEG P0 receipt needs a controlled reason")
        if data["tokens"] is not None:
            raise ValueError("failed BA-IEG P0 receipt cannot claim event tokens")
    if not isinstance(data["views"], list):
        raise TypeError("BA-IEG P0 view bindings must be an array")
    _sha256(data["receipt_sha256"], "P0 receipt_sha256")
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("BA-IEG P0 receipt hash does not bind its content")
    id_source = deepcopy(data)
    id_source["receipt_id"] = "CONTENT-ADDRESS-PENDING"
    id_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "BAIEG-P0-" + _canonical_sha256(id_source)[:20]
    if data["receipt_id"] != expected_id:
        raise ValueError("BA-IEG P0 receipt ID does not bind its content")
    return data


@dataclass(frozen=True)
class BAIEGP0MaterializationResult:
    """An auditable P0 event tokenization result, including fail-closed cases."""

    event_tokens: BAIEGEventTokens | None
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        receipt = validate_ba_ieg_p0_materialization_receipt(self.receipt)
        if receipt["status"] == "materialized":
            if not isinstance(self.event_tokens, BAIEGEventTokens):
                raise ValueError("materialized P0 result requires BAIEGEventTokens")
            self.event_tokens.verify_integrity()
            if (
                receipt["tokens"]["input_receipt_sha256"]
                != self.event_tokens.input_receipt_sha256
            ):
                raise ValueError("P0 receipt does not bind the event-token input")
            if len(receipt["views"]) != len(self.event_tokens.view_ids):
                raise ValueError("P0 receipt lost a temporal Findings view")
            for index, view in enumerate(receipt["views"]):
                expected = {
                    "view_id": self.event_tokens.view_ids[index],
                    "task_role": self.event_tokens.view_roles[index],
                    "effective_temporal_role": (
                        self.event_tokens.view_effective_temporal_roles[index]
                    ),
                    "dependency_policy": (
                        self.event_tokens.view_dependency_policies[index]
                    ),
                    "future_sample_access": bool(
                        self.event_tokens.view_future_sample_access[index]
                    ),
                    "onset_evidence_authorized": bool(
                        self.event_tokens.view_onset_evidence_authorized[index]
                    ),
                    "temporal_evidence_sha256": (
                        self.event_tokens.view_temporal_evidence_sha256s[index]
                    ),
                    "receipt_sha256": self.event_tokens.view_receipt_sha256s[
                        index
                    ],
                    "transform_spec_sha256": (
                        self.event_tokens.view_transform_sha256s[index]
                    ),
                    "reference_family": self.event_tokens.reference_families[
                        index
                    ],
                }
                for field, value in expected.items():
                    if view.get(field) != value:
                        raise ValueError(
                            f"P0 receipt view {index} {field} drifted from event tokens"
                        )
            targets = self.event_tokens.deterministic_targets
            attached = receipt["tokens"].get(
                "deterministic_targets_attached"
            )
            if targets is None:
                if attached is not False:
                    raise ValueError(
                        "P0 receipt claims deterministic targets that are absent"
                    )
            elif (
                attached is not True
                or receipt["tokens"].get(
                    "deterministic_target_receipt_sha256"
                )
                != targets.receipt_sha256
                or receipt["tokens"].get(
                    "dense_measurement_source_binding_sha256"
                )
                != targets.source_binding_sha256
                or receipt["lineage"].get(
                    "dense_measurement_source_binding_sha256"
                )
                != targets.source_binding_sha256
            ):
                raise ValueError(
                    "P0 receipt does not bind its dense deterministic targets"
                )
            if receipt["schema_version"] == (
                BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_A0_NATIVE_12
            ):
                if (
                    self.event_tokens.feature_scope.detector_navigation_used
                    is not False
                    or self.event_tokens.adaptive_window_receipt_sha256
                    != receipt["lineage"].get(
                        "a0_navigation_window_receipt_sha256"
                    )
                ):
                    raise ValueError(
                        "A0 P0 event input acquired detector navigation semantics"
                    )
                active = self.event_tokens.token_signal_mask
                neutral = torch.full(
                    (int(active.sum()), len(BA_IEG_PHASE_STATES)),
                    1.0 / len(BA_IEG_PHASE_STATES),
                    dtype=self.event_tokens.phase_posterior.dtype,
                )
                if (
                    not torch.equal(
                        self.event_tokens.phase_posterior[active], neutral
                    )
                    or torch.any(
                        self.event_tokens.phase_posterior[~active] != 0
                    )
                ):
                    raise ValueError(
                        "A0 oracle interval leaked into phase-posterior model input"
                    )
        elif self.event_tokens is not None:
            raise ValueError("failed P0 result cannot carry event tokens")
        object.__setattr__(self, "receipt", receipt)


def _finalize_p0_receipt(body: dict[str, Any]) -> dict[str, Any]:
    body = deepcopy(body)
    body["receipt_id"] = "CONTENT-ADDRESS-PENDING"
    body["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    body["receipt_id"] = "BAIEG-P0-" + _canonical_sha256(body)[:20]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_ba_ieg_p0_materialization_receipt(body)


def _empty_p0_lineage() -> dict[str, object | None]:
    return {
        "canonical_signal_id": None,
        "canonical_receipt_sha256": None,
        "canonical_materialization_receipt_sha256": None,
        "adaptive_search_receipt_id": None,
        "adaptive_search_receipt_sha256": None,
        "adaptive_preprocessing_receipt_sha256": None,
        "canonical_adaptive_binding_sha256": None,
        "adaptive_window_receipt_id": None,
        "adaptive_window_receipt_sha256": None,
        "dense_measurement_sidecar_receipt_sha256": None,
        "dense_measurement_source_binding_sha256": None,
        "source_binding_sha256": None,
    }


def _empty_p0_lineage_a0() -> dict[str, object | None]:
    return {
        "canonical_signal_id": None,
        "canonical_receipt_sha256": None,
        "canonical_materialization_receipt_sha256": None,
        "a0_candidate_roster_receipt_sha256": None,
        "a0_oracle_navigation_receipt_sha256": None,
        "a0_canonical_identity_binding_receipt_sha256": None,
        "a0_navigation_window_id": None,
        "a0_navigation_window_receipt_sha256": None,
        "dense_measurement_sidecar_receipt_sha256": None,
        "dense_measurement_source_binding_sha256": None,
        "source_binding_sha256": None,
    }


def _p0_failure_result(
    *,
    event_id: str,
    recording_id: str,
    patient_uid: str,
    model_split: str,
    policy: BAIEGP0TokenizationPolicy,
    view_profile: str = BA_IEG_P0_VIEW_PROFILE_LEGACY_8,
    a0_navigation: bool = False,
    code: str,
    stage: str,
    lineage: Mapping[str, object | None] | None = None,
    timing: Mapping[str, object] | None = None,
    censoring: Mapping[str, object] | None = None,
    eligibility: Mapping[str, object] | None = None,
    views: Sequence[Mapping[str, object]] = (),
) -> BAIEGP0MaterializationResult:
    if code not in _P0_FAILURE_CODES:
        raise ValueError("unknown controlled P0 failure code")
    if a0_navigation:
        if view_profile != BA_IEG_P0_VIEW_PROFILE_NATIVE_12:
            raise ValueError("A0 P0 failures require the native-12 profile")
        schema_version = BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_A0_NATIVE_12
    elif view_profile == BA_IEG_P0_VIEW_PROFILE_LEGACY_8:
        schema_version = BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION
    elif view_profile == BA_IEG_P0_VIEW_PROFILE_NATIVE_12:
        schema_version = BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_NATIVE_12
    else:
        raise ValueError("unknown BA-IEG P0 view profile")
    body = {
        "schema_version": schema_version,
        "receipt_id": "CONTENT-ADDRESS-PENDING",
        "status": "failed",
        "failure_code": code,
        "failure_stage": stage,
        "event_identity": {
            "event_id": event_id,
            "recording_id": recording_id,
            "patient_uid": patient_uid,
            "model_split": model_split,
        },
        "lineage": deepcopy(
            dict(
                lineage
                or (
                    _empty_p0_lineage_a0()
                    if a0_navigation
                    else _empty_p0_lineage()
                )
            )
        ),
        "timing": deepcopy(dict(timing)) if timing is not None else None,
        "censoring": deepcopy(dict(censoring)) if censoring is not None else None,
        "eligibility": deepcopy(dict(eligibility)) if eligibility is not None else None,
        "views": [deepcopy(dict(item)) for item in views],
        "masks": None,
        "tokens": None,
        "policy": policy.to_dict(),
        "scope_receipt": deepcopy(
            _P0_SCOPE_RECEIPT_A0 if a0_navigation else _P0_SCOPE_RECEIPT
        ),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    if view_profile == BA_IEG_P0_VIEW_PROFILE_NATIVE_12:
        body["view_profile"] = view_profile
    if a0_navigation:
        body.update(
            {
                "navigation_arm": BA_IEG_P0_NAVIGATION_ARM_A0,
                "evaluation_semantics": BA_IEG_P0_A0_EVALUATION_SEMANTICS,
                "support_role": "initial_bootstrap_watchdog_only",
            }
        )
    return BAIEGP0MaterializationResult(
        event_tokens=None,
        receipt=_finalize_p0_receipt(body),
    )


def _p0_view_clock(view: Mapping[str, Any]) -> tuple[int, int]:
    clock = view["transform_spec"]["output_clock"]
    return int(clock["sampling_rate_numerator"]), int(
        clock["sampling_rate_denominator"]
    )


def _p0_temporal_binding(
    view: Mapping[str, Any],
    *,
    expected_effective_temporal_role: str,
) -> dict[str, object]:
    """Bind one validated canonical view to its explicit P0 permission lane.

    ``onset_evidence_authorized`` is a *clinical admission* decision, not a
    reliable discriminator between causal and offline preprocessing.  In
    particular, an otherwise valid future-free onset carrier remains in the
    onset-causal lane when the fail-closed clinical gate withholds permission
    to use it as positive onset evidence.  The caller obtains the expected
    lane from the already validated ``task_reference_views`` bundle key; this
    function then checks that the view's replayed temporal contract is
    compatible with that lane.  It never promotes the authorization boolean.
    """

    temporal = view["temporal_evidence"]
    task_role = str(view["task_role"])
    dependency = str(temporal["dependency_policy"])
    future = bool(temporal["future_sample_access"])
    onset = bool(temporal["onset_evidence_authorized"])

    if expected_effective_temporal_role == "onset_causal":
        if task_role not in {"onset_causal", "spatial_reference"}:
            raise ValueError("onset lane contains an incompatible task role")
        if future or dependency != "past_and_present_only":
            raise ValueError("onset lane is not future-free causal evidence")
        effective_role = "onset_causal"
    elif expected_effective_temporal_role == "context_offline":
        if task_role not in {
            "context_offline",
            "findings_clinical",
            "spatial_reference",
        }:
            raise ValueError("offline lane contains an incompatible task role")
        if not future or dependency != "bidirectional_or_unknown" or onset:
            raise ValueError("offline lane temporal permission is inconsistent")
        effective_role = "context_offline"
    elif expected_effective_temporal_role == "morphology_native":
        if task_role not in {
            "findings_native",
            "findings_native_morphology",
            "spatial_reference",
        }:
            raise ValueError("native lane contains an incompatible task role")
        if dependency != "instantaneous" or future or onset:
            raise ValueError("native morphology lane temporal permission is unsafe")
        effective_role = "morphology_native"
    else:
        raise ValueError("expected P0 temporal role is unsupported")
    return {
        "effective_temporal_role": effective_role,
        "dependency_policy": dependency,
        "future_sample_access": future,
        "onset_evidence_authorized": onset,
        "temporal_evidence_sha256": _canonical_sha256(temporal),
    }


def _p0_internal_unit_id(view_id: str, source_unit_id: str) -> str:
    return f"{view_id}::{source_unit_id}"


def _p0_scale_physical_tiles(
    *,
    interval: tuple[float, float],
    policy: BAIEGP0TokenizationPolicy,
) -> tuple[tuple[int, float, float, bool, bool], ...]:
    """Define scale tiles once in recording-relative seconds.

    The final two booleans record contact with the nominal adaptive left/right
    boundary.  No sampling clock participates in this function.
    """

    durations = (
        policy.fine_duration_seconds,
        policy.coarse_duration_seconds,
        policy.context_duration_seconds,
    )
    steps = (
        policy.fine_step_seconds,
        policy.coarse_step_seconds,
        policy.context_step_seconds,
    )
    start, stop = interval
    rows: list[tuple[int, float, float, bool, bool]] = []
    for scale_index, (duration_seconds, step_seconds) in enumerate(
        zip(durations, steps)
    ):
        tile_index = 0
        while True:
            cursor = start + tile_index * float(step_seconds)
            if cursor >= stop - 1e-10:
                break
            tile_stop = cursor + float(duration_seconds)
            if tile_stop > stop + 1e-10:
                if scale_index != 0:
                    break
                tile_stop = stop
            rows.append(
                (
                    scale_index,
                    float(cursor),
                    float(tile_stop),
                    abs(cursor - start) <= 1e-10,
                    abs(tile_stop - stop) <= 1e-10,
                )
            )
            tile_index += 1
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return tuple(rows)


def _p0_map_physical_interval_inward(
    receipt: Mapping[str, Any],
    interval: tuple[float, float],
    *,
    minimum_samples: int = 2,
) -> tuple[int, int, tuple[float, float]] | None:
    """Map one physical interval inward without time warp or padding."""

    clock = _p0_view_clock(receipt)
    start_position = interval[0] * clock[0] / clock[1]
    stop_position = interval[1] * clock[0] / clock[1]
    global_start = int(math.ceil(start_position - 1e-10))
    global_stop = int(math.floor(stop_position + 1e-10))
    if global_stop - global_start < minimum_samples:
        return None
    selected_start, selected_stop = (
        int(item)
        for item in receipt["coordinates"][
            "selected_global_output_sample_interval"
        ]
    )
    if global_start < selected_start or global_stop > selected_stop:
        raise ValueError("physical interval lies outside a P0 Findings view")
    valid_start = int(
        receipt["tensor_layout"]["valid_data_tensor_sample_interval"][0]
    )
    tensor_start = valid_start + global_start - selected_start
    tensor_stop = valid_start + global_stop - selected_start
    support = (
        global_start * clock[1] / clock[0],
        global_stop * clock[1] / clock[0],
    )
    if (
        support[0] < interval[0] - 1e-8
        or support[1] > interval[1] + 1e-8
    ):
        raise ValueError("P0 view mapping escaped its nominal physical tile")
    return tensor_start, tensor_stop, support


def _p0_intersects(left: tuple[int, int], right: Sequence[int]) -> bool:
    return int(right[0]) < left[1] and int(right[1]) > left[0]


def _p0_measurements(
    signal_volts: torch.Tensor,
    *,
    sampling_rate_hz: float,
    base_policy: BAIEGBaseNumericalPolicy,
    effective_bandwidth_hz: Sequence[float],
    amplitude_eligible: bool,
    morphology_eligible: bool,
    spectral_eligible: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return P0 inputs plus an explicit per-feature opportunity mask.

    The twelve features shared with deterministic supervision are delegated to
    :func:`measure_ba_ieg_base_numerical_features`.  Only the three P0-only
    morphology proxies and the later previous-tile contrast are local here.
    """

    values = signal_volts.detach().cpu().to(torch.float64).flatten()
    if values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("P0 measurement tile must contain finite physical EEG samples")
    shared = measure_ba_ieg_base_numerical_features(
        values.numpy(),
        sampling_rate_hz=float(sampling_rate_hz),
        effective_bandwidth_hz=effective_bandwidth_hz,
        policy=base_policy,
        amplitude_reason_codes=(
            () if amplitude_eligible else ("amplitude_family_ineligible",)
        ),
        spectral_reason_codes=(
            () if spectral_eligible else ("spectral_family_ineligible",)
        ),
    )
    result = torch.zeros(len(BA_IEG_P0_TOKEN_FEATURES), dtype=torch.float32)
    feature_mask = torch.zeros(
        len(BA_IEG_P0_TOKEN_FEATURES), dtype=torch.bool
    )
    for shared_index, p0_index in enumerate(
        BA_IEG_P0_SHARED_BASE_FEATURE_INDICES
    ):
        result[p0_index] = float(shared.values[shared_index])
        feature_mask[p0_index] = bool(shared.value_mask[shared_index])

    if morphology_eligible and values.numel() >= 4:
        centered = values - torch.median(values)
        centered_std = torch.sqrt(torch.mean(centered.square()))
        if float(centered_std) <= torch.finfo(torch.float64).eps:
            zero_crossing = slope_reversal = kurtosis = torch.zeros(
                (), dtype=torch.float64
            )
        else:
            signs = torch.sign(centered)
            zero_crossing = (
                signs[1:] * signs[:-1] < 0
            ).to(torch.float64).sum()
            zero_crossing *= sampling_rate_hz / max(1, values.numel() - 1)
            slope = torch.diff(values)
            slope_reversal = (
                slope[1:] * slope[:-1] < 0
            ).to(torch.float64).sum()
            slope_reversal *= sampling_rate_hz / max(1, slope.numel() - 1)
            standardized = centered / centered_std
            kurtosis = torch.mean(standardized.pow(4)) - 3.0
        morphology_values = (zero_crossing, slope_reversal, kurtosis)
        for name, value in zip(
            (
                "zero_crossing_rate_hz",
                "slope_reversal_rate_hz",
                "excess_kurtosis",
            ),
            morphology_values,
        ):
            index = BA_IEG_P0_TOKEN_FEATURES.index(name)
            result[index] = value.to(torch.float32)
            feature_mask[index] = True
    if torch.any(result[~feature_mask] != 0) or not torch.isfinite(result).all():
        raise ValueError("P0 deterministic measurements are non-finite")
    return result, feature_mask


def _p0_phase_posterior(
    interval: tuple[float, float],
    *,
    onset_seconds: float | None,
    peak_seconds: float | None,
    termination_seconds: float | None,
    left_censored: bool,
) -> torch.Tensor:
    if left_censored or onset_seconds is None:
        return torch.full((len(BA_IEG_PHASE_STATES),), 0.25, dtype=torch.float32)
    start, stop = interval
    peak = onset_seconds if peak_seconds is None else max(onset_seconds, peak_seconds)
    termination = stop if termination_seconds is None else max(peak, termination_seconds)
    boundaries = (
        (-math.inf, onset_seconds),
        (onset_seconds, peak),
        (peak, termination),
        (termination, math.inf),
    )
    weights = []
    for lower, upper in boundaries:
        overlap = max(0.0, min(stop, upper) - max(start, lower))
        weights.append(overlap)
    tensor = torch.tensor(weights, dtype=torch.float32)
    if float(tensor.sum()) <= 0:
        midpoint = 0.5 * (start + stop)
        index = (
            0
            if midpoint < onset_seconds
            else 1
            if midpoint < peak
            else 2
            if midpoint < termination
            else 3
        )
        tensor[index] = 1.0
        return tensor
    return tensor / tensor.sum()


def _p0_add_robust_change_feature(
    values: torch.Tensor,
    *,
    feature_mask: torch.Tensor,
    signal_mask: torch.Tensor,
    family_mask: torch.Tensor,
    unit_index: torch.Tensor,
    view_index: torch.Tensor,
    scale_index: torch.Tensor,
    view_future_sample_access: torch.Tensor,
) -> None:
    amplitude = BA_IEG_EVIDENCE_FAMILIES.index("amplitude")
    spectral = BA_IEG_EVIDENCE_FAMILIES.index("spectral")
    groups = sorted(
        {
            (int(view_index[row]), int(unit_index[row]), int(scale_index[row]))
            for row in torch.nonzero(signal_mask, as_tuple=False).flatten().tolist()
        }
    )
    for view, unit, scale in groups:
        rows = torch.nonzero(
            signal_mask
            & (view_index == view)
            & (unit_index == unit)
            & (scale_index == scale)
            & family_mask[:, amplitude]
            & family_mask[:, spectral],
            as_tuple=False,
        ).flatten()
        if rows.numel():
            rows = rows[
                feature_mask[rows][
                    :, BA_IEG_P0_SHARED_BASE_FEATURE_INDICES
                ].all(dim=1)
            ]
        if rows.numel() < 2:
            continue
        base = values[rows][
            :, BA_IEG_P0_SHARED_BASE_FEATURE_INDICES
        ].to(torch.float64)
        if bool(view_future_sample_access[view]):
            median = torch.median(base, dim=0).values
            mad = torch.median(torch.abs(base - median), dim=0).values
            scale_value = torch.where(
                mad > 1e-9,
                1.4826 * mad,
                torch.maximum(torch.abs(median), torch.ones_like(median)),
            )
            standardized = (base - median) / scale_value
            change = torch.mean(torch.abs(torch.diff(standardized, dim=0)), dim=1)
            values[rows[1:], -1] = torch.clamp(change, max=50.0).to(
                torch.float32
            )
            feature_mask[rows[1:], -1] = True
            continue

        # Onset-authorized views must remain prefix invariant.  A global
        # event median/MAD would let later spread change an earlier token even
        # though the underlying FIR is causal.  Each change score therefore
        # uses only the current prefix, including the current tile but never a
        # later one.
        for position in range(1, int(rows.numel())):
            history = base[: position + 1]
            median = torch.median(history, dim=0).values
            mad = torch.median(torch.abs(history - median), dim=0).values
            scale_value = torch.where(
                mad > 1e-9,
                1.4826 * mad,
                torch.maximum(torch.abs(median), torch.ones_like(median)),
            )
            change = torch.mean(
                torch.abs(base[position] - base[position - 1]) / scale_value
            )
            values[rows[position], -1] = torch.clamp(change, max=50.0).to(
                torch.float32
            )
            feature_mask[rows[position], -1] = True


def materialize_ba_ieg_p0_event_tokens(
    canonical_bundle: object,
    adaptive_search_receipt: object,
    adaptive_window_receipt: object,
    *,
    event_id: str,
    recording_id: str,
    patient_uid: str,
    model_split: str,
    view_profile: str = BA_IEG_P0_VIEW_PROFILE_LEGACY_8,
    policy: BAIEGP0TokenizationPolicy = BAIEGP0TokenizationPolicy(),
    physical_xyz: torch.Tensor | None = None,
    physical_xyz_mask: torch.Tensor | None = None,
    a0_navigation_window_receipt: object | None = None,
) -> BAIEGP0MaterializationResult:
    """Materialize target-free ragged tokens from canonical EEG-only views.

    The detector-derived anchor is read only from the validated adaptive
    search receipt and remains a navigation coordinate.  Token values come
    solely from validated canonical views.  The default legacy profile uses
    onset-causal and offline-context referential/TCP/CAR/Laplacian views.  The
    explicit native-12 profile appends the four unfiltered native-morphology
    references as model inputs without extending the legacy dense-supervision
    ledger.  Temporal permissions remain separate through event and batch
    tensors.  The API has no annotation, spreadsheet, clinical-text, label,
    target, or event-selection argument.

    Expected production failures return a content-addressed failure receipt
    instead of deleting the event.  Unsafe identity or policy arguments still
    raise because no trustworthy receipt can be attached to them.
    """

    event_id = _identifier(event_id, "event_id")
    recording_id = _identifier(recording_id, "recording_id")
    patient_uid = _identifier(patient_uid, "patient_uid")
    if model_split not in BA_IEG_ALLOWED_SPLITS:
        raise ValueError("BA-IEG P0 model_split is unsupported")
    if view_profile not in {
        BA_IEG_P0_VIEW_PROFILE_LEGACY_8,
        BA_IEG_P0_VIEW_PROFILE_NATIVE_12,
    }:
        raise ValueError("BA-IEG P0 view_profile is unsupported")
    native_12_enabled = view_profile == BA_IEG_P0_VIEW_PROFILE_NATIVE_12
    a0_navigation = a0_navigation_window_receipt is not None
    if a0_navigation and not native_12_enabled:
        raise ValueError("A0 P0 materialization requires explicit native-12")
    if a0_navigation and model_split != "source_train":
        raise ValueError("A0 P0 materialization is source-train-only")
    materialization_schema_version = (
        BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_A0_NATIVE_12
        if a0_navigation
        else BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_NATIVE_12
        if native_12_enabled
        else BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION
    )
    encoder_implementation_id = (
        BA_IEG_P0_IMPLEMENTATION_ID_A0_NATIVE_12
        if a0_navigation
        else BA_IEG_P0_IMPLEMENTATION_ID_NATIVE_12
        if native_12_enabled
        else BA_IEG_P0_IMPLEMENTATION_ID
    )
    if not isinstance(policy, BAIEGP0TokenizationPolicy):
        raise TypeError("policy must be BAIEGP0TokenizationPolicy")

    lineage = _empty_p0_lineage_a0() if a0_navigation else _empty_p0_lineage()
    if a0_navigation and (
        adaptive_search_receipt is not None or adaptive_window_receipt is not None
    ):
        return _p0_failure_result(
            event_id=event_id,
            recording_id=recording_id,
            patient_uid=patient_uid,
            model_split=model_split,
            view_profile=view_profile,
            a0_navigation=True,
            policy=policy,
            code="a0_navigation_argument_mismatch",
            stage="navigation_authority_selection",
            lineage=lineage,
        )
    try:
        from .canonical_edf_materialization import (
            CanonicalEDFViewBundle,
            validate_canonical_edf_materialization,
        )

        if not isinstance(canonical_bundle, CanonicalEDFViewBundle):
            raise TypeError("canonical_bundle must be CanonicalEDFViewBundle")
        materialization = validate_canonical_edf_materialization(canonical_bundle)
        canonical = canonical_bundle.canonical_record.canonical_receipt
        lineage.update(
            {
                "canonical_signal_id": canonical["canonical_signal_id"],
                "canonical_receipt_sha256": canonical["receipt_sha256"],
                "canonical_materialization_receipt_sha256": materialization[
                    "receipt_sha256"
                ],
            }
        )
    except (TypeError, ValueError, RuntimeError):
        return _p0_failure_result(
            event_id=event_id,
            recording_id=recording_id,
            patient_uid=patient_uid,
            model_split=model_split,
            view_profile=view_profile,
            a0_navigation=a0_navigation,
            policy=policy,
            code="invalid_canonical_bundle",
            stage="canonical_validation",
            lineage=lineage,
        )

    if a0_navigation:
        try:
            from .ba_ieg_a0_navigation_window_v1 import (
                validate_ba_ieg_a0_navigation_window_v1,
            )

            a0_window = validate_ba_ieg_a0_navigation_window_v1(
                a0_navigation_window_receipt
            )
        except (TypeError, ValueError, RuntimeError):
            return _p0_failure_result(
                event_id=event_id,
                recording_id=recording_id,
                patient_uid=patient_uid,
                model_split=model_split,
                view_profile=view_profile,
                a0_navigation=True,
                policy=policy,
                code="invalid_a0_navigation_window",
                stage="a0_navigation_window_validation",
                lineage=lineage,
            )
        a0_identity = a0_window["event_identity"]
        a0_canonical = a0_window["canonical_signal_binding"]
        a0_timing = a0_window["timing"]
        a0_mismatch = any(
            (
                a0_identity["event_id"] != event_id,
                a0_identity["recording_id"] != recording_id,
                a0_identity["patient_uid"] != patient_uid,
                a0_identity["model_split"] != model_split,
                a0_canonical["canonical_signal_id"]
                != canonical["canonical_signal_id"],
                a0_canonical["recording_id"] != canonical["recording_id"],
                a0_canonical["source_signal_sha256"]
                != canonical["source_signal_sha256"],
                a0_canonical["canonical_receipt_sha256"]
                != canonical["receipt_sha256"],
                a0_canonical["canonical_materialization_receipt_sha256"]
                != materialization["receipt_sha256"],
                a0_canonical["source_header_receipt_sha256"]
                != canonical_bundle.canonical_record.source_header_receipt[
                    "receipt_sha256"
                ],
                tuple(a0_canonical["observed_channel_ids"])
                != tuple(canonical_bundle.canonical_record.observed_channel_ids),
                abs(
                    float(a0_canonical["recording_duration_seconds"])
                    - float(canonical["recording_duration_seconds"])
                )
                > 1e-6,
            )
        )
        if a0_mismatch:
            return _p0_failure_result(
                event_id=event_id,
                recording_id=recording_id,
                patient_uid=patient_uid,
                model_split=model_split,
                view_profile=view_profile,
                a0_navigation=True,
                policy=policy,
                code="a0_canonical_identity_mismatch",
                stage="a0_canonical_identity_binding",
                lineage=lineage,
            )
        analysis_support = list(
            a0_timing["analysis_interval_recording_seconds"]
        )
        search = {
            "recording_duration_seconds": a0_timing[
                "recording_duration_seconds"
            ],
            "envelope_interval_recording_seconds": analysis_support,
            "coarse_anchor_recording_seconds": a0_timing[
                "navigation_anchor_seconds"
            ],
            "critical_transition": None,
            "stage_evidence": {},
        }
        window = {
            "analysis_interval_recording_seconds": analysis_support,
            "baseline_context_recording_seconds": a0_timing[
                "baseline_context_recording_seconds"
            ],
            "censoring": {
                "left": bool(a0_timing["analysis_support_clipped_left"]),
                "right": bool(a0_timing["analysis_support_clipped_right"]),
            },
            "eligibility": {
                "signal_findings": True,
                "onset_localization": False,
                "reason_codes": ["a0_initial_watchdog_not_final_adaptive_support"],
            },
        }
        search_sha256 = a0_window["a0_oracle_navigation_receipt_sha256"]
        window_sha256 = a0_window["receipt_sha256"]
        lineage.update(
            {
                "a0_candidate_roster_receipt_sha256": a0_window[
                    "a0_candidate_roster_receipt_sha256"
                ],
                "a0_oracle_navigation_receipt_sha256": a0_window[
                    "a0_oracle_navigation_receipt_sha256"
                ],
                "a0_canonical_identity_binding_receipt_sha256": a0_window[
                    "canonical_identity_binding_receipt_sha256"
                ],
                "a0_navigation_window_id": a0_window["window_id"],
                "a0_navigation_window_receipt_sha256": window_sha256,
            }
        )
    else:
        try:
            from .adaptive_search import validate_adaptive_search_receipt

            search = validate_adaptive_search_receipt(adaptive_search_receipt)
            search_sha256 = _canonical_sha256(search)
            canonical_adaptive_binding = search["canonical_signal_binding"]
            lineage.update(
                {
                    "adaptive_search_receipt_id": search["search_receipt_id"],
                    "adaptive_search_receipt_sha256": search_sha256,
                    "adaptive_preprocessing_receipt_sha256": search[
                        "preprocessing_receipt_sha256"
                    ],
                    "canonical_adaptive_binding_sha256": (
                        canonical_adaptive_binding["binding_sha256"]
                        if canonical_adaptive_binding is not None
                        else None
                    ),
                }
            )
        except (TypeError, ValueError, RuntimeError):
            return _p0_failure_result(
                event_id=event_id,
                recording_id=recording_id,
                patient_uid=patient_uid,
                model_split=model_split,
                view_profile=view_profile,
                policy=policy,
                code="invalid_adaptive_search_receipt",
                stage="adaptive_search_validation",
                lineage=lineage,
            )

        if canonical_adaptive_binding is None or any(
            (
                canonical_adaptive_binding["canonical_signal_id"]
                != canonical["canonical_signal_id"],
                canonical_adaptive_binding["canonical_recording_id"]
                != canonical["recording_id"],
                canonical_adaptive_binding["canonical_source_signal_sha256"]
                != canonical["source_signal_sha256"],
                canonical_adaptive_binding["canonical_receipt_sha256"]
                != canonical["receipt_sha256"],
                canonical_adaptive_binding[
                    "canonical_source_header_receipt_sha256"
                ]
                != canonical_bundle.canonical_record.source_header_receipt[
                    "receipt_sha256"
                ],
                abs(
                    float(canonical_adaptive_binding["recording_duration_seconds"])
                    - float(canonical["recording_duration_seconds"])
                )
                > 1e-6,
                tuple(canonical_adaptive_binding["observed_channel_ids"])
                != tuple(canonical_bundle.canonical_record.observed_channel_ids),
            )
        ):
            return _p0_failure_result(
                event_id=event_id,
                recording_id=recording_id,
                patient_uid=patient_uid,
                model_split=model_split,
                view_profile=view_profile,
                policy=policy,
                code="canonical_adaptive_identity_mismatch",
                stage="canonical_adaptive_identity_binding",
                lineage=lineage,
            )

        try:
            from .adaptive_event_window import validate_adaptive_event_analysis_window

            window = validate_adaptive_event_analysis_window(adaptive_window_receipt)
            window_sha256 = _canonical_sha256(window)
            lineage.update(
                {
                    "adaptive_window_receipt_id": window["window_receipt_id"],
                    "adaptive_window_receipt_sha256": window_sha256,
                }
            )
        except (TypeError, ValueError, RuntimeError):
            return _p0_failure_result(
                event_id=event_id,
                recording_id=recording_id,
                patient_uid=patient_uid,
                model_split=model_split,
                view_profile=view_profile,
                policy=policy,
                code="invalid_adaptive_window_receipt",
                stage="adaptive_window_validation",
                lineage=lineage,
            )

        if (
            window["source_search_receipt_id"] != search["search_receipt_id"]
            or window["source_search_receipt_sha256"] != search_sha256
        ):
            return _p0_failure_result(
                event_id=event_id,
                recording_id=recording_id,
                patient_uid=patient_uid,
                model_split=model_split,
                view_profile=view_profile,
                policy=policy,
                code="adaptive_receipt_binding_mismatch",
                stage="adaptive_search_window_binding",
                lineage=lineage,
                censoring=window["censoring"],
                eligibility=window["eligibility"],
            )

    duration = float(canonical["recording_duration_seconds"])
    search_duration = search["recording_duration_seconds"]
    envelope_start, envelope_stop = map(
        float, search["envelope_interval_recording_seconds"]
    )
    anchor = float(search["coarse_anchor_recording_seconds"])
    support_interval_field = (
        "initial_watchdog_support_interval_seconds"
        if a0_navigation
        else "adaptive_envelope_interval_seconds"
    )
    if (
        recording_id != canonical["recording_id"]
        or search_duration is None
        or abs(float(search_duration) - duration) > 1e-6
        or envelope_start < -1e-6
        or envelope_stop > duration + 1e-6
    ):
        return _p0_failure_result(
            event_id=event_id,
            recording_id=recording_id,
            patient_uid=patient_uid,
            model_split=model_split,
            view_profile=view_profile,
            a0_navigation=a0_navigation,
            policy=policy,
            code="recording_clock_mismatch",
            stage=(
                "canonical_a0_watchdog_clock_binding"
                if a0_navigation
                else "canonical_adaptive_clock_binding"
            ),
            lineage=lineage,
            timing={
                "canonical_recording_duration_seconds": duration,
                (
                    "initial_watchdog_recording_duration_seconds"
                    if a0_navigation
                    else "adaptive_recording_duration_seconds"
                ): search_duration,
                support_interval_field: [envelope_start, envelope_stop],
                "navigation_anchor_seconds": anchor,
            },
            censoring=window["censoring"],
            eligibility=window["eligibility"],
        )

    requested_raw = window["analysis_interval_recording_seconds"]
    if (
        requested_raw is None
        or window["eligibility"]["signal_findings"] is not True
    ):
        return _p0_failure_result(
            event_id=event_id,
            recording_id=recording_id,
            patient_uid=patient_uid,
            model_split=model_split,
            view_profile=view_profile,
            a0_navigation=a0_navigation,
            policy=policy,
            code="event_interval_unavailable",
            stage=(
                "a0_initial_watchdog_eligibility"
                if a0_navigation
                else "adaptive_event_eligibility"
            ),
            lineage=lineage,
            timing={
                "requested_analysis_interval_seconds": None,
                "navigation_anchor_seconds": anchor,
                support_interval_field: [envelope_start, envelope_stop],
            },
            censoring=window["censoring"],
            eligibility=window["eligibility"],
        )

    requested = (float(requested_raw[0]), float(requested_raw[1]))
    if (
        requested[0] < envelope_start - 1e-6
        or requested[1] > envelope_stop + 1e-6
        or requested[0] < -1e-6
        or requested[1] > duration + 1e-6
        or not requested[0] <= anchor <= requested[1]
    ):
        return _p0_failure_result(
            event_id=event_id,
            recording_id=recording_id,
            patient_uid=patient_uid,
            model_split=model_split,
            view_profile=view_profile,
            a0_navigation=a0_navigation,
            policy=policy,
            code="recording_clock_mismatch",
            stage=(
                "a0_initial_watchdog_clock_binding"
                if a0_navigation
                else "adaptive_event_clock_binding"
            ),
            lineage=lineage,
            timing={
                "requested_analysis_interval_seconds": list(requested),
                "navigation_anchor_seconds": anchor,
                support_interval_field: [envelope_start, envelope_stop],
            },
            censoring=window["censoring"],
            eligibility=window["eligibility"],
        )

    try:
        reference_order = (
            "referential",
            "tcp_bipolar",
            "car",
            "laplacian",
        )
        reference_family_by_kind = {
            "referential": "referential",
            "tcp_bipolar": "bipolar",
            "car": "common_average",
            "laplacian": "laplacian",
        }
        view_lane_specs = (
            ("onset_causal", "onset_causal"),
            ("context_offline", "context_offline"),
        )
        if native_12_enabled:
            view_lane_specs = (
                *view_lane_specs,
                ("findings_native_morphology", "morphology_native"),
            )
        view_objects = tuple(
            canonical_bundle.task_reference_views[task_view_role][reference_kind]
            for task_view_role, _effective_role in view_lane_specs
            for reference_kind in reference_order
        )
        view_receipts = tuple(item.receipt for item in view_objects)
        expected_task_roles = tuple(
            task_view_role if reference_kind == "referential" else "spatial_reference"
            for task_view_role, _effective_role in view_lane_specs
            for reference_kind in reference_order
        )
        if tuple(item["task_role"] for item in view_receipts) != expected_task_roles:
            raise ValueError("canonical bundle lacks the selected P0 Findings views")
        expected_effective_temporal_roles = tuple(
            effective_role
            for _task_view_role, effective_role in view_lane_specs
            for _reference_kind in reference_order
        )
        view_temporal_bindings = tuple(
            _p0_temporal_binding(
                item,
                expected_effective_temporal_role=expected_role,
            )
            for item, expected_role in zip(
                view_receipts,
                expected_effective_temporal_roles,
            )
        )
        if (
            tuple(item["effective_temporal_role"] for item in view_temporal_bindings)
            != expected_effective_temporal_roles
        ):
            raise ValueError("P0 temporal-role inheritance drifted")
        clocks = tuple(_p0_view_clock(item) for item in view_receipts)
        physical_tiles = _p0_scale_physical_tiles(
            interval=requested,
            policy=policy,
        )
        if not physical_tiles:
            raise ValueError("P0 policy produced no physical-time tile")
        view_analysis_supports: list[tuple[float, float]] = []
        view_tile_mappings: list[
            tuple[
                tuple[
                    int,
                    int,
                    int,
                    tuple[float, float],
                    bool,
                    bool,
                    tuple[float, float],
                ],
                ...,
            ]
        ] = []
        for receipt in view_receipts:
            input_ids = tuple(receipt["transform_spec"]["input_unit_ids"])
            if input_ids != tuple(STANDARD_19):
                raise ValueError("P0 reference matrix is not on standard-19")
            analysis_mapping = _p0_map_physical_interval_inward(
                receipt,
                requested,
                minimum_samples=2,
            )
            if analysis_mapping is None:
                raise ValueError("adaptive event is too short on a supplied view clock")
            view_analysis_supports.append(analysis_mapping[2])
            mapped_tiles: list[
                tuple[
                    int,
                    int,
                    int,
                    tuple[float, float],
                    bool,
                    bool,
                    tuple[float, float],
                ]
            ] = []
            for (
                scale_index,
                nominal_start,
                nominal_stop,
                left_contact,
                right_contact,
            ) in physical_tiles:
                mapping = _p0_map_physical_interval_inward(
                    receipt,
                    (nominal_start, nominal_stop),
                    minimum_samples=(
                        policy.minimum_fine_samples
                        if scale_index == 0
                        else 2
                    ),
                )
                # A very short trailing fine tile may be representable on a
                # faster view but not a slower one.  Omit it for that view;
                # never manufacture samples or time-warp it into existence.
                if mapping is None:
                    continue
                tensor_start, tensor_stop, support = mapping
                mapped_tiles.append(
                    (
                        scale_index,
                        tensor_start,
                        tensor_stop,
                        support,
                        left_contact,
                        right_contact,
                        (nominal_start, nominal_stop),
                    )
                )
            if not mapped_tiles:
                raise ValueError("P0 policy produced no tile on a supplied view clock")
            view_tile_mappings.append(tuple(mapped_tiles))
    except (TypeError, ValueError, RuntimeError, KeyError, IndexError):
        return _p0_failure_result(
            event_id=event_id,
            recording_id=recording_id,
            patient_uid=patient_uid,
            model_split=model_split,
            view_profile=view_profile,
            a0_navigation=a0_navigation,
            policy=policy,
            code="view_clock_or_reference_mismatch",
            stage="canonical_view_binding",
            lineage=lineage,
            timing={
                "requested_analysis_interval_seconds": list(requested),
                "navigation_anchor_seconds": anchor,
                support_interval_field: [envelope_start, envelope_stop],
            },
            censoring=window["censoring"],
            eligibility=window["eligibility"],
        )

    view_bindings: list[dict[str, object]] = []
    reference_kinds_for_views = reference_order * len(view_lane_specs)
    for receipt, reference_kind, temporal_binding, view_clock, view_support in zip(
        view_receipts,
        reference_kinds_for_views,
        view_temporal_bindings,
        clocks,
        view_analysis_supports,
    ):
        view_bindings.append(
            {
                "view_id": receipt["view_id"],
                "task_role": receipt["task_role"],
                **temporal_binding,
                "receipt_sha256": receipt["receipt_sha256"],
                "transform_spec_sha256": receipt["transform_spec"][
                    "transform_spec_sha256"
                ],
                "processed_view_sha256": receipt["processed_view_sha256"],
                "mask_sha256": receipt["masks"]["mask_sha256"],
                "reference_family": reference_family_by_kind[reference_kind],
                "unit_count": len(receipt["output_units"]),
                "output_clock": {
                    "sampling_rate_numerator": view_clock[0],
                    "sampling_rate_denominator": view_clock[1],
                    "global_origin_recording_seconds": 0.0,
                },
                "mapped_analysis_support_seconds": list(view_support),
            }
        )
    source_binding_body = {
        "canonical_receipt_sha256": lineage["canonical_receipt_sha256"],
        "canonical_materialization_receipt_sha256": lineage[
            "canonical_materialization_receipt_sha256"
        ],
        "view_bindings": view_bindings,
        "recording_duration_seconds": duration,
    }
    if a0_navigation:
        source_binding_body.update(
            {
                "navigation_arm": BA_IEG_P0_NAVIGATION_ARM_A0,
                "a0_candidate_roster_receipt_sha256": lineage[
                    "a0_candidate_roster_receipt_sha256"
                ],
                "a0_oracle_navigation_receipt_sha256": lineage[
                    "a0_oracle_navigation_receipt_sha256"
                ],
                "a0_canonical_identity_binding_receipt_sha256": lineage[
                    "a0_canonical_identity_binding_receipt_sha256"
                ],
                "a0_navigation_window_receipt_sha256": window_sha256,
                "support_role": "initial_bootstrap_watchdog_only",
            }
        )
    else:
        source_binding_body.update(
            {
                "adaptive_search_receipt_sha256": search_sha256,
                "adaptive_window_receipt_sha256": window_sha256,
            }
        )
    lineage["source_binding_sha256"] = _canonical_sha256(source_binding_body)

    try:
        physical_evidence = torch.tensor(
            [bool(item["observed"]) for item in canonical["channels"]],
            dtype=torch.bool,
        )
        if physical_xyz is None and physical_xyz_mask is None:
            xyz = torch.zeros((len(STANDARD_19), 3), dtype=torch.float32)
            xyz_mask = torch.zeros(len(STANDARD_19), dtype=torch.bool)
        elif physical_xyz is None or physical_xyz_mask is None:
            raise ValueError("physical_xyz and physical_xyz_mask must be supplied together")
        else:
            xyz = physical_xyz
            xyz_mask = physical_xyz_mask

        unit_ids: list[str] = []
        unit_source_ids: list[str] = []
        unit_types: list[str] = []
        unit_view_indices: list[int] = []
        reference_rows: list[list[float]] = []
        unit_evidence_rows: list[bool] = []
        unit_family_rows: list[list[bool]] = []
        unit_receipt_rows: list[dict[str, Any]] = []
        for view_index, receipt in enumerate(view_receipts):
            matrix = receipt["transform_spec"]["reference"]["matrix"]
            for row_index, output in enumerate(receipt["output_units"]):
                source_unit_id = str(output["unit_id"])
                unit_id = _p0_internal_unit_id(
                    str(receipt["view_id"]), source_unit_id
                )
                if unit_id in unit_ids:
                    raise ValueError("P0 analysis unit IDs collide across views")
                family_by_name = {
                    str(item["family"]): bool(item["eligible"])
                    for item in output["evidence_eligibility"]
                }
                family_flags = [
                    bool(family_by_name[name])
                    for name in BA_IEG_EVIDENCE_FAMILIES
                ]
                unit_ids.append(unit_id)
                unit_source_ids.append(source_unit_id)
                unit_types.append(str(output["unit_type"]))
                unit_view_indices.append(view_index)
                reference_rows.append([float(value) for value in matrix[row_index]])
                unit_evidence_rows.append(
                    bool(output["evidence_eligible"]) and any(family_flags)
                )
                unit_family_rows.append(family_flags)
                unit_receipt_rows.append(output)

        unit_view_tensor = torch.tensor(unit_view_indices, dtype=torch.long)
        reference_tensor = torch.tensor(reference_rows, dtype=torch.float32)
        unit_evidence_tensor = torch.tensor(unit_evidence_rows, dtype=torch.bool)
        unit_family_tensor = torch.tensor(unit_family_rows, dtype=torch.bool)
        view_future_tensor = torch.tensor(
            [
                bool(item["future_sample_access"])
                for item in view_temporal_bindings
            ],
            dtype=torch.bool,
        )
        view_onset_tensor = torch.tensor(
            [
                bool(item["onset_evidence_authorized"])
                for item in view_temporal_bindings
            ],
            dtype=torch.bool,
        )
        # The dense deterministic sidecar is the single persisted source of
        # BA-IEG numerical supervision.  P0 token measurements remain model
        # inputs; the replayable 13-target ledger is attached separately and
        # carries per-value masks/reason codes rather than manufacturing zeros
        # for missing or QC-failed channels.
        from .ba_ieg_dense_measurement_sidecar import (
            BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
            BAIEGDenseMeasurementPolicy,
            BAIEGDenseMeasurementViewInput,
            materialize_ba_ieg_dense_measurement_sidecar,
        )

        dense_policy = BAIEGDenseMeasurementPolicy(
            window_seconds=float(policy.fine_duration_seconds),
            step_seconds=float(policy.fine_step_seconds),
            # P0 tiles are anchored to the nominal physical start of this
            # event.  Use that same origin for the dense sidecar so
            # every complete fine token has an exactly matching
            # (view, unit, start, stop) supervision row.
            global_grid_origin_seconds=float(requested[0]),
            analysis_low_hz=float(policy.spectral_low_hz),
            analysis_high_hz=float(policy.spectral_high_hz),
        )
        baseline_raw = window["baseline_context_recording_seconds"]
        dense_background: list[tuple[float, float]] = []
        if baseline_raw is not None:
            baseline_start = max(requested[0], float(baseline_raw[0]))
            baseline_stop = min(requested[1], float(baseline_raw[1]))
            if baseline_stop - baseline_start >= dense_policy.window_seconds:
                dense_background.append((baseline_start, baseline_stop))
        dense_view_inputs: list[BAIEGDenseMeasurementViewInput] = []
        unit_offset = 0
        dense_supervision_view_indices = range(2 * len(reference_order))
        for view_index in dense_supervision_view_indices:
            view_object = view_objects[view_index]
            view_receipt = view_receipts[view_index]
            view_unit_count = len(view_receipt["output_units"])
            dense_view_inputs.append(
                BAIEGDenseMeasurementViewInput(
                    view_index=view_index,
                    unit_indices=tuple(
                        range(unit_offset, unit_offset + view_unit_count)
                    ),
                    view_receipt=view_receipt,
                    tensor=view_object.tensor,
                )
            )
            unit_offset += view_unit_count
        dense_sidecar = materialize_ba_ieg_dense_measurement_sidecar(
            canonical_receipt=canonical,
            views=tuple(dense_view_inputs),
            analysis_interval_seconds=requested,
            background_intervals_seconds=dense_background,
            policy=dense_policy,
            trusted_parent_views={
                str(view_receipts[0]["view_id"]): view_receipts[0],
                str(view_receipts[len(reference_order)]["view_id"]): (
                    view_receipts[len(reference_order)]
                ),
            },
        )
        if int(dense_sidecar.targets.row_view_index.max()) >= len(
            dense_supervision_view_indices
        ):
            raise ValueError("native morphology leaked into dense supervision")
        lineage.update(
            {
                "dense_measurement_sidecar_receipt_sha256": (
                    dense_sidecar.receipt_sha256
                ),
                "dense_measurement_source_binding_sha256": (
                    dense_sidecar.source_binding_sha256
                ),
            }
        )

        critical = search["critical_transition"]
        onset_seconds: float | None = None
        termination_seconds: float | None = None
        if critical is not None and critical[
            "start_offset_seconds_relative_to_anchor"
        ] is not None:
            onset_seconds = anchor + float(
                critical["start_offset_seconds_relative_to_anchor"]
            )
        if critical is not None and critical[
            "stop_offset_seconds_relative_to_anchor"
        ] is not None:
            termination_seconds = anchor + float(
                critical["stop_offset_seconds_relative_to_anchor"]
            )
        onset_stage = search["stage_evidence"].get("onset")
        peak_seconds: float | None = None
        if isinstance(onset_stage, Mapping) and onset_stage.get(
            "peak_offset_seconds_relative_to_anchor"
        ) is not None:
            peak_seconds = anchor + float(
                onset_stage["peak_offset_seconds_relative_to_anchor"]
            )

        token_values_rows: list[torch.Tensor] = []
        token_feature_masks: list[torch.Tensor] = []
        token_times: list[tuple[float, float]] = []
        token_units: list[int] = []
        token_views: list[int] = []
        token_scales: list[int] = []
        token_signal: list[bool] = []
        token_families: list[list[bool]] = []
        token_phase: list[torch.Tensor] = []
        left_contacts: list[bool] = []
        right_contacts: list[bool] = []
        amplitude_index = BA_IEG_EVIDENCE_FAMILIES.index("amplitude")
        morphology_index = BA_IEG_EVIDENCE_FAMILIES.index("morphology")
        spectral_index = BA_IEG_EVIDENCE_FAMILIES.index("spectral")

        global_unit_index = 0
        for view_index, (view_object, receipt) in enumerate(
            zip(view_objects, view_receipts)
        ):
            tensor = view_object.tensor.detach().cpu().to(torch.float32)
            sampling_rate_hz = clocks[view_index][0] / clocks[view_index][1]
            edge_intervals = tuple(
                tuple(int(value) for value in item)
                for item in receipt["masks"]["edge_invalid_intervals"]
            )
            quality_by_unit: dict[str, list[dict[str, Any]]] = {}
            for quality in receipt["masks"]["quality_invalid_intervals"]:
                quality_by_unit.setdefault(str(quality["unit_id"]), []).append(
                    quality
                )
            for local_unit_index, output in enumerate(receipt["output_units"]):
                unit_id = str(output["unit_id"])
                base_families = unit_family_rows[global_unit_index]
                for (
                    scale_index,
                    tensor_start,
                    tensor_stop,
                    interval_seconds,
                    left_contact,
                    right_contact,
                    _nominal_interval,
                ) in view_tile_mappings[view_index]:
                    local_interval = (tensor_start, tensor_stop)
                    families = list(base_families)
                    if any(
                        _p0_intersects(local_interval, interval)
                        for interval in edge_intervals
                    ):
                        families = [False] * len(BA_IEG_EVIDENCE_FAMILIES)
                    for quality in quality_by_unit.get(unit_id, []):
                        if not _p0_intersects(
                            local_interval, quality["tensor_sample_interval"]
                        ):
                            continue
                        for disabled in quality["disabled_evidence_families"]:
                            if disabled in BA_IEG_EVIDENCE_FAMILIES:
                                families[BA_IEG_EVIDENCE_FAMILIES.index(disabled)] = False
                    signal_available = (
                        unit_evidence_rows[global_unit_index]
                        and any(families)
                        and tensor_stop - tensor_start >= 2
                    )
                    if signal_available:
                        measurements, measurement_mask = _p0_measurements(
                            tensor[local_unit_index, tensor_start:tensor_stop],
                            sampling_rate_hz=sampling_rate_hz,
                            base_policy=dense_policy.base_numerical_policy,
                            effective_bandwidth_hz=output[
                                "effective_bandwidth_hz"
                            ],
                            amplitude_eligible=bool(
                                families[amplitude_index]
                            ),
                            morphology_eligible=bool(
                                families[morphology_index]
                            ),
                            spectral_eligible=bool(
                                families[spectral_index]
                            ),
                        )
                    else:
                        measurements = torch.zeros(
                            len(BA_IEG_P0_TOKEN_FEATURES), dtype=torch.float32
                        )
                        measurement_mask = torch.zeros(
                            len(BA_IEG_P0_TOKEN_FEATURES), dtype=torch.bool
                        )
                    active = bool(measurement_mask.any())
                    if not active:
                        families = [False] * len(BA_IEG_EVIDENCE_FAMILIES)
                    token_values_rows.append(measurements)
                    token_feature_masks.append(measurement_mask)
                    token_times.append(interval_seconds)
                    token_units.append(global_unit_index)
                    token_views.append(view_index)
                    token_scales.append(scale_index)
                    token_signal.append(active)
                    token_families.append(families)
                    token_phase.append(
                        (
                            torch.full(
                                (len(BA_IEG_PHASE_STATES),),
                                1.0 / len(BA_IEG_PHASE_STATES),
                                dtype=torch.float32,
                            )
                            if view_temporal_bindings[view_index][
                                "effective_temporal_role"
                            ]
                            != "context_offline"
                            else _p0_phase_posterior(
                                interval_seconds,
                                onset_seconds=onset_seconds,
                                peak_seconds=peak_seconds,
                                termination_seconds=termination_seconds,
                                left_censored=bool(window["censoring"]["left"]),
                            )
                        )
                        if active
                        else torch.zeros(len(BA_IEG_PHASE_STATES))
                    )
                    left_contacts.append(left_contact)
                    right_contacts.append(right_contact)
                global_unit_index += 1

        values_tensor = torch.stack(token_values_rows).contiguous()
        feature_mask_tensor = torch.stack(token_feature_masks).contiguous()
        times_tensor = torch.tensor(token_times, dtype=torch.float32)
        token_unit_tensor = torch.tensor(token_units, dtype=torch.long)
        token_view_tensor = torch.tensor(token_views, dtype=torch.long)
        token_scale_tensor = torch.tensor(token_scales, dtype=torch.long)
        signal_tensor = torch.tensor(token_signal, dtype=torch.bool)
        family_tensor = torch.tensor(token_families, dtype=torch.bool)
        phase_tensor = torch.stack(token_phase).to(torch.float32).contiguous()
        left_contact_tensor = torch.tensor(left_contacts, dtype=torch.bool)
        right_contact_tensor = torch.tensor(right_contacts, dtype=torch.bool)
        _p0_add_robust_change_feature(
            values_tensor,
            feature_mask=feature_mask_tensor,
            signal_mask=signal_tensor,
            family_mask=family_tensor,
            unit_index=token_unit_tensor,
            view_index=token_view_tensor,
            scale_index=token_scale_tensor,
            view_future_sample_access=view_future_tensor,
        )
        signal_tensor = feature_mask_tensor.any(dim=1)
        if torch.any(values_tensor[~feature_mask_tensor] != 0):
            raise ValueError("P0 feature-masked values must remain zero")
        if not signal_tensor.any():
            raise LookupError("no evidence-eligible P0 token")

        encoder_receipt = {
            "implementation_id": encoder_implementation_id,
            "encoder_lineage": "deterministic_projection",
            "policy_sha256": policy.sha256,
            "feature_names": list(BA_IEG_P0_TOKEN_FEATURES),
            "clinical_terms_emitted": False,
            "trained_checkpoint_used": False,
        }
        if native_12_enabled:
            encoder_receipt["view_profile"] = view_profile
        if a0_navigation:
            encoder_receipt.update(
                {
                    "navigation_arm": BA_IEG_P0_NAVIGATION_ARM_A0,
                    "evaluation_semantics": BA_IEG_P0_A0_EVALUATION_SEMANTICS,
                    "support_role": "initial_bootstrap_watchdog_only",
                    "oracle_interval_used_as_phase_feature": False,
                }
            )
        encoder_receipt_sha256 = _canonical_sha256(encoder_receipt)
        event = BAIEGEventTokens(
            event_id=event_id,
            recording_id=recording_id,
            patient_uid=patient_uid,
            model_split=model_split,
            analysis_interval_seconds=requested,
            navigation_anchor_seconds=anchor,
            canonical_receipt_sha256=str(lineage["canonical_receipt_sha256"]),
            adaptive_window_receipt_sha256=window_sha256,
            encoder_implementation_id=encoder_implementation_id,
            encoder_lineage="deterministic_projection",
            encoder_receipt_sha256=encoder_receipt_sha256,
            physical_electrode_ids=STANDARD_19,
            physical_xyz=xyz,
            physical_xyz_mask=xyz_mask,
            physical_evidence_mask=physical_evidence,
            view_ids=tuple(str(item["view_id"]) for item in view_receipts),
            view_roles=tuple(str(item["task_role"]) for item in view_receipts),
            view_effective_temporal_roles=tuple(
                str(item["effective_temporal_role"])
                for item in view_temporal_bindings
            ),
            view_dependency_policies=tuple(
                str(item["dependency_policy"])
                for item in view_temporal_bindings
            ),
            view_future_sample_access=view_future_tensor,
            view_onset_evidence_authorized=view_onset_tensor,
            view_temporal_evidence_sha256s=tuple(
                str(item["temporal_evidence_sha256"])
                for item in view_temporal_bindings
            ),
            view_receipt_sha256s=tuple(
                str(item["receipt_sha256"]) for item in view_receipts
            ),
            view_transform_sha256s=tuple(
                str(item["transform_spec"]["transform_spec_sha256"])
                for item in view_receipts
            ),
            reference_families=tuple(
                reference_family_by_kind[item]
                for item in reference_kinds_for_views
            ),
            unit_ids=tuple(unit_ids),
            unit_source_ids=tuple(unit_source_ids),
            unit_types=tuple(unit_types),
            unit_view_index=unit_view_tensor,
            unit_reference_matrix=reference_tensor,
            unit_evidence_mask=unit_evidence_tensor,
            unit_family_mask=unit_family_tensor,
            token_values=values_tensor,
            token_feature_mask=feature_mask_tensor,
            token_time_bounds_seconds=times_tensor,
            token_unit_index=token_unit_tensor,
            token_view_index=token_view_tensor,
            token_scale_index=token_scale_tensor,
            token_signal_mask=signal_tensor,
            token_family_mask=family_tensor,
            phase_posterior=phase_tensor,
            feature_scope=EEGOnlyFeatureScope(
                detector_navigation_used=not a0_navigation
            ),
            deterministic_targets=dense_sidecar.targets,
        )
    except LookupError:
        return _p0_failure_result(
            event_id=event_id,
            recording_id=recording_id,
            patient_uid=patient_uid,
            model_split=model_split,
            view_profile=view_profile,
            a0_navigation=a0_navigation,
            policy=policy,
            code="no_evidence_eligible_tokens",
            stage="token_evidence_masking",
            lineage=lineage,
            timing={
                "requested_analysis_interval_seconds": list(requested),
                "acquired_analysis_interval_seconds": list(requested),
                "navigation_anchor_seconds": anchor,
                "view_clock": None,
                "view_clocks": [
                    {
                        "view_id": str(receipt["view_id"]),
                        "sampling_rate_numerator": view_clock[0],
                        "sampling_rate_denominator": view_clock[1],
                        "mapped_analysis_support_seconds": list(view_support),
                    }
                    for receipt, view_clock, view_support in zip(
                        view_receipts, clocks, view_analysis_supports
                    )
                ],
                "per_view_inward_sample_mapping": True,
            },
            censoring=window["censoring"],
            eligibility=window["eligibility"],
            views=view_bindings,
        )
    except (TypeError, ValueError, RuntimeError, KeyError, IndexError):
        return _p0_failure_result(
            event_id=event_id,
            recording_id=recording_id,
            patient_uid=patient_uid,
            model_split=model_split,
            view_profile=view_profile,
            a0_navigation=a0_navigation,
            policy=policy,
            code="tokenization_failed",
            stage="deterministic_token_materialization",
            lineage=lineage,
            timing={
                "requested_analysis_interval_seconds": list(requested),
                "acquired_analysis_interval_seconds": list(requested),
                "navigation_anchor_seconds": anchor,
                "view_clock": None,
                "view_clocks": [
                    {
                        "view_id": str(receipt["view_id"]),
                        "sampling_rate_numerator": view_clock[0],
                        "sampling_rate_denominator": view_clock[1],
                        "mapped_analysis_support_seconds": list(view_support),
                    }
                    for receipt, view_clock, view_support in zip(
                        view_receipts, clocks, view_analysis_supports
                    )
                ],
                "per_view_inward_sample_mapping": True,
            },
            censoring=window["censoring"],
            eligibility=window["eligibility"],
            views=view_bindings,
        )

    refined_interval = (
        None
        if onset_seconds is None
        else [onset_seconds, termination_seconds]
    )
    scale_counts = {
        name: int((event.token_scale_index == index).sum())
        for index, name in enumerate(BA_IEG_TOKEN_SCALES)
    }
    timing = {
        "requested_analysis_interval_seconds": list(requested),
        "acquired_analysis_interval_seconds": list(requested),
        "subsample_boundary_trim_seconds": [0.0, 0.0],
        support_interval_field: [envelope_start, envelope_stop],
        "navigation_anchor_seconds": anchor,
        "navigation_anchor_is_confirmed_onset": False,
        "refined_transition_interval_seconds": refined_interval,
        "refined_transition_is_confirmed_seizure": False,
        "phase_peak_seconds": peak_seconds,
        "view_clock": None,
        "view_clocks": [
            {
                "view_id": str(receipt["view_id"]),
                "sampling_rate_numerator": view_clock[0],
                "sampling_rate_denominator": view_clock[1],
                "global_origin_recording_seconds": 0.0,
                "mapped_analysis_support_seconds": list(view_support),
                "subsample_boundary_trim_seconds": [
                    view_support[0] - requested[0],
                    requested[1] - view_support[1],
                ],
            }
            for receipt, view_clock, view_support in zip(
                view_receipts, clocks, view_analysis_supports
            )
        ],
        "physical_tiling_clock": "recording_relative_seconds",
        "per_view_inward_sample_mapping": True,
        "event_local_resampling_used": False,
        "event_time_warp_used": False,
        "silent_padding_used": False,
    }
    if a0_navigation:
        timing.update(
            {
                "navigation_anchor_source": (
                    "public_tusz_seizure_interval_start_for_navigation_only"
                ),
                "initial_support_only": True,
                "fixed_watchdog_is_final_analysis_window": False,
                "final_support_requires_iterative_rule_adaptive_acquisition": True,
                "iterative_rule_adaptive_acquisition_status": "not_materialized",
                "oracle_interval_used_as_phase_feature": False,
            }
        )
    masks = {
        "physical_evidence_mask_sha256": _tensor_sha256(
            event.physical_evidence_mask
        ),
        "unit_evidence_mask_sha256": _tensor_sha256(event.unit_evidence_mask),
        "unit_family_mask_sha256": _tensor_sha256(event.unit_family_mask),
        "token_signal_mask_sha256": _tensor_sha256(event.token_signal_mask),
        "token_feature_mask_sha256": _tensor_sha256(
            event.token_feature_mask
        ),
        "token_family_mask_sha256": _tensor_sha256(event.token_family_mask),
        "phase_posterior_sha256": _tensor_sha256(event.phase_posterior),
        "view_future_sample_access_sha256": _tensor_sha256(
            event.view_future_sample_access
        ),
        "view_onset_evidence_authorized_sha256": _tensor_sha256(
            event.view_onset_evidence_authorized
        ),
        "token_future_sample_access_sha256": _tensor_sha256(
            event.token_future_sample_access
        ),
        "token_onset_evidence_mask_sha256": _tensor_sha256(
            event.token_onset_evidence_mask
        ),
        "token_positive_onset_mask_sha256": _tensor_sha256(
            event.token_positive_onset_mask
        ),
        "token_phase_context_mask_sha256": _tensor_sha256(
            event.token_phase_context_mask
        ),
        "left_boundary_contact_mask_sha256": _tensor_sha256(
            left_contact_tensor
        ),
        "right_boundary_contact_mask_sha256": _tensor_sha256(
            right_contact_tensor
        ),
        "signal_eligible_token_count": int(event.token_signal_mask.sum()),
        "onset_evidence_eligible_token_count": int(
            event.token_onset_evidence_mask.sum()
        ),
        "positive_onset_spatial_token_count": int(
            event.token_positive_onset_mask.sum()
        ),
        "offline_context_token_count": int(
            event.token_phase_context_mask.sum()
        ),
        "masked_token_count": int((~event.token_signal_mask).sum()),
        "left_boundary_contact_token_count": int(left_contact_tensor.sum()),
        "right_boundary_contact_token_count": int(right_contact_tensor.sum()),
        "left_censored": bool(window["censoring"]["left"]),
        "right_censored": bool(window["censoring"]["right"]),
    }
    token_receipt = {
        "token_count": int(event.token_values.shape[0]),
        "feature_dimension": event.feature_dim,
        "feature_names": list(BA_IEG_P0_TOKEN_FEATURES),
        "base_numerical_kernel_id": BA_IEG_BASE_NUMERICAL_KERNEL_ID,
        "feature_opportunity_count": {
            name: int(event.token_feature_mask[:, index].sum())
            for index, name in enumerate(BA_IEG_P0_TOKEN_FEATURES)
        },
        "scale_names": list(BA_IEG_TOKEN_SCALES),
        "token_count_by_scale": scale_counts,
        "token_values_sha256": _tensor_sha256(event.token_values),
        "token_time_bounds_sha256": _tensor_sha256(event.token_time_bounds_seconds),
        "input_receipt_sha256": event.input_receipt_sha256,
        "deterministic_targets_attached": True,
        "deterministic_target_source": (
            BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION
        ),
        "dense_measurement_sidecar_receipt_sha256": (dense_sidecar.receipt_sha256),
        "deterministic_target_receipt_sha256": (dense_sidecar.targets.receipt_sha256),
        "dense_measurement_source_binding_sha256": (
            dense_sidecar.source_binding_sha256
        ),
        "trained_checkpoint_used": False,
        "clinical_terms_emitted": False,
    }
    if native_12_enabled:
        token_receipt.update(
            {
                "native_view_used_as_model_input": True,
                "native_view_used_as_dense_supervision": False,
                "dense_supervision_view_count": 2 * len(reference_order),
            }
        )
    if a0_navigation:
        token_receipt.update(
            {
                "oracle_interval_used_as_phase_feature": False,
                "oracle_interval_available_to_model_forward": False,
                "initial_support_only": True,
                "final_rule_adaptive_support_materialized": False,
            }
        )
    onset_localization_input_authorized = bool(event.token_onset_evidence_mask.any())
    research_channel_ranking_input_authorized = bool(
        event.token_positive_onset_mask.any()
    )
    onset_input_reason_codes: set[str] = set()
    if not onset_localization_input_authorized:
        for receipt in view_receipts[: len(reference_order)]:
            onset_input_reason_codes.update(
                str(item)
                for item in receipt["temporal_evidence"]["authorization_reason_codes"]
            )
        if not onset_input_reason_codes:
            onset_input_reason_codes.add("no_onset_evidence_eligible_token")
    token_receipt["clinical_onset_input_qualification"] = {
        "status": (
            "evaluable" if onset_localization_input_authorized else "not_evaluable"
        ),
        "onset_localization_input_authorized": (onset_localization_input_authorized),
        "research_channel_ranking_input_authorized": (
            research_channel_ranking_input_authorized
        ),
        "reason_codes": sorted(onset_input_reason_codes),
    }
    body = {
        "schema_version": materialization_schema_version,
        "receipt_id": "CONTENT-ADDRESS-PENDING",
        "status": "materialized",
        "failure_code": None,
        "failure_stage": None,
        "event_identity": {
            "event_id": event_id,
            "recording_id": recording_id,
            "patient_uid": patient_uid,
            "model_split": model_split,
        },
        "lineage": deepcopy(lineage),
        "timing": timing,
        "censoring": deepcopy(window["censoring"]),
        "eligibility": deepcopy(window["eligibility"]),
        "views": view_bindings,
        "masks": masks,
        "tokens": token_receipt,
        "policy": policy.to_dict(),
        "scope_receipt": deepcopy(
            _P0_SCOPE_RECEIPT_A0 if a0_navigation else _P0_SCOPE_RECEIPT
        ),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    if native_12_enabled:
        body["view_profile"] = view_profile
    if a0_navigation:
        body.update(
            {
                "navigation_arm": BA_IEG_P0_NAVIGATION_ARM_A0,
                "evaluation_semantics": BA_IEG_P0_A0_EVALUATION_SEMANTICS,
                "support_role": "initial_bootstrap_watchdog_only",
            }
        )
    return BAIEGP0MaterializationResult(
        event_tokens=event,
        receipt=_finalize_p0_receipt(body),
    )


__all__ = [
    "BA_IEG_ALLOWED_ENCODER_LINEAGES",
    "BA_IEG_ALLOWED_REFERENCE_FAMILIES",
    "BA_IEG_ALLOWED_SPLITS",
    "BA_IEG_ALLOWED_VIEW_ROLES",
    "BA_IEG_C18",
    "BA_IEG_DETERMINISTIC_TARGETS",
    "BA_IEG_DEPENDENCY_POLICIES",
    "BA_IEG_EFFECTIVE_TEMPORAL_ROLES",
    "BA_IEG_EVENT_TOKEN_SCHEMA_VERSION",
    "BA_IEG_EVIDENCE_FAMILIES",
    "BA_IEG_P0_IMPLEMENTATION_ID",
    "BA_IEG_P0_IMPLEMENTATION_ID_A0_NATIVE_12",
    "BA_IEG_P0_IMPLEMENTATION_ID_NATIVE_12",
    "BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION",
    "BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_A0_NATIVE_12",
    "BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_NATIVE_12",
    "BA_IEG_P0_A0_EVALUATION_SEMANTICS",
    "BA_IEG_P0_NAVIGATION_ARM_A0",
    "BA_IEG_P0_TOKEN_FEATURES",
    "BA_IEG_P0_VIEW_PROFILE_LEGACY_8",
    "BA_IEG_P0_VIEW_PROFILE_NATIVE_12",
    "BA_IEG_PATIENT_BAG_SCHEMA_VERSION",
    "BA_IEG_PHASE_STATES",
    "BA_IEG_REFERENCE_FAMILIES",
    "BA_IEG_TOKEN_SCALES",
    "BAIEGCollatedEventBatch",
    "BAIEGDeepSOZPositiveSet",
    "BAIEGDeterministicTargets",
    "BAIEGEventTokens",
    "BAIEGP0MaterializationResult",
    "BAIEGP0TokenizationPolicy",
    "BAIEGPatientBag",
    "BAIEGPatientBagBatch",
    "BAIEGPatientBagDataset",
    "BAIEGPatientEventBagManifest",
    "EEGOnlyFeatureScope",
    "collate_ba_ieg_events",
    "collate_ba_ieg_patient_bags",
    "deepsoz_positive_set_from_reference",
    "materialize_ba_ieg_p0_event_tokens",
    "positive_set_mass_loss",
    "validate_ba_ieg_p0_materialization_receipt",
    "validate_patient_disjoint_event_partitions",
]
