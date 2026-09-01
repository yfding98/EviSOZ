"""Replayable dense deterministic supervision for BA-IEG event windows.

The sidecar is a target producer, not a clinical EEG interpreter.  It accepts
only an immutable canonical EEG receipt and already-materialized
Findings/spatial views whose tensors match their content hashes.  It emits the
13 numerical targets frozen by :mod:`ba_ieg_training_contract` on a physical
recording-time grid.

Every requested ``(view, unit, time)`` row is retained in the provenance
ledger.  A target that cannot be measured because of missing signal, a QC
mask, or insufficient bandwidth is represented by ``value_mask=False`` and a
typed reason code.  A row for which all 13 targets are unavailable remains in
the ledger but is omitted from ``BAIEGDeterministicTargets`` because that
training contract intentionally admits only rows with at least one supervised
value.  Masked numerical values are always zero and never act as negatives.

The producer deliberately does not open EDF files, call an annotation API,
read spreadsheets or doctor labels, select events, label spikes/IEDs, or infer
SOZ.  Its robust change score is a numerical contrast against explicit
EEG-derived background intervals; it is not ACNS evolution or seizure onset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch

from .ba_ieg_training_contract import (
    BA_IEG_ALLOWED_VIEW_ROLES,
    BA_IEG_DETERMINISTIC_TARGETS,
    BAIEGDeterministicTargets,
)
from .ba_ieg_numerical_kernel import (
    BA_IEG_BASE_BAND_TARGETS,
    BA_IEG_BASE_MEASUREMENT_NAMES,
    BAIEGBaseNumericalPolicy,
    measure_ba_ieg_base_numerical_features,
)
from .canonical_signal_views import (
    recording_seconds_to_view_tensor_index,
    validate_canonical_signal_receipt,
    validate_signal_view_receipt,
    view_tensor_index_to_recording_seconds,
)
from .deterministic_event_findings import deterministic_view_tensor_sha256


BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION: Final[str] = (
    "ba_ieg_dense_measurement_sidecar_v2"
)
BA_IEG_DENSE_MEASUREMENT_METHOD_ID: Final[str] = (
    "BA-IEG-DENSE-DETERMINISTIC-MEASUREMENTS-V2"
)

_TOL = 1e-8
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_TARGET_INDEX = {
    name: index for index, name in enumerate(BA_IEG_DETERMINISTIC_TARGETS)
}
_BAND_TARGETS = BA_IEG_BASE_BAND_TARGETS
_AMPLITUDE_TARGETS = tuple(range(0, 3))
_SPECTRAL_TARGETS = tuple(range(3, 12))
_CHANGE_TARGET_INDEX = _TARGET_INDEX["robust_multifeature_change_score"]

if tuple(BA_IEG_DETERMINISTIC_TARGETS[:-1]) != BA_IEG_BASE_MEASUREMENT_NAMES:
    raise RuntimeError(
        "dense target vocabulary drifted from the shared numerical kernel"
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_CHARACTERS for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return text


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item interval")
    start = _finite(value[0], f"{name}[0]", minimum=0.0)
    stop = _finite(value[1], f"{name}[1]", minimum=0.0)
    if stop <= start + _TOL:
        raise ValueError(f"{name} must have positive duration")
    return start, stop


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _contained(
    interval: tuple[float, float], carriers: Sequence[tuple[float, float]]
) -> bool:
    return any(
        interval[0] >= carrier[0] - _TOL
        and interval[1] <= carrier[1] + _TOL
        for carrier in carriers
    )


def _sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(str(item) for item in values)))


@dataclass(frozen=True)
class BAIEGDenseMeasurementPolicy:
    """Frozen numerical policy; its values are research defaults, not norms."""

    window_seconds: float = 1.0
    step_seconds: float = 0.5
    global_grid_origin_seconds: float = 0.0
    analysis_low_hz: float = 0.5
    analysis_high_hz: float = 45.0
    minimum_spectral_bins: int = 3
    minimum_baseline_windows: int = 4
    spectral_power_floor_uv2: float = 1e-12
    log_amplitude_robust_scale_floor: float = 0.15
    band_ratio_robust_scale_floor: float = 0.04
    entropy_robust_scale_floor: float = 0.08
    centering: str = "median"
    taper: str = "hann"

    def __post_init__(self) -> None:
        window = _finite(self.window_seconds, "window_seconds", minimum=_TOL)
        step = _finite(self.step_seconds, "step_seconds", minimum=_TOL)
        origin = _finite(
            self.global_grid_origin_seconds,
            "global_grid_origin_seconds",
            minimum=0.0,
        )
        low = _finite(self.analysis_low_hz, "analysis_low_hz", minimum=0.0)
        high = _finite(self.analysis_high_hz, "analysis_high_hz", minimum=_TOL)
        if high <= low + _TOL:
            raise ValueError("analysis frequency band is empty")
        if (
            isinstance(self.minimum_spectral_bins, bool)
            or not isinstance(self.minimum_spectral_bins, int)
            or self.minimum_spectral_bins < 3
        ):
            raise ValueError("minimum_spectral_bins must be an integer >= 3")
        if (
            isinstance(self.minimum_baseline_windows, bool)
            or not isinstance(self.minimum_baseline_windows, int)
            or self.minimum_baseline_windows < 2
        ):
            raise ValueError("minimum_baseline_windows must be an integer >= 2")
        for name in (
            "spectral_power_floor_uv2",
            "log_amplitude_robust_scale_floor",
            "band_ratio_robust_scale_floor",
            "entropy_robust_scale_floor",
        ):
            if _finite(getattr(self, name), name, minimum=0.0) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.centering != "median" or self.taper != "hann":
            raise ValueError("v1 freezes median centering and a Hann taper")
        object.__setattr__(self, "window_seconds", window)
        object.__setattr__(self, "step_seconds", step)
        object.__setattr__(self, "global_grid_origin_seconds", origin)
        object.__setattr__(self, "analysis_low_hz", low)
        object.__setattr__(self, "analysis_high_hz", high)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["band_targets_hz"] = [
            {"target": name, "low": low, "high": high}
            for name, low, high in _BAND_TARGETS
        ]
        result["method_id"] = BA_IEG_DENSE_MEASUREMENT_METHOD_ID
        result["base_numerical_kernel"] = self.base_numerical_policy.to_dict()
        return result

    @property
    def base_numerical_policy(self) -> BAIEGBaseNumericalPolicy:
        return BAIEGBaseNumericalPolicy(
            analysis_low_hz=self.analysis_low_hz,
            analysis_high_hz=self.analysis_high_hz,
            minimum_spectral_bins=self.minimum_spectral_bins,
            spectral_power_floor_uv2=self.spectral_power_floor_uv2,
        )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_BA_IEG_DENSE_MEASUREMENT_POLICY = BAIEGDenseMeasurementPolicy()


@dataclass(frozen=True)
class BAIEGDenseMeasurementViewInput:
    """Host-supplied tensor plus its mapping into a future event contract."""

    view_index: int
    unit_indices: tuple[int, ...]
    view_receipt: object
    tensor: torch.Tensor


@dataclass(frozen=True)
class BAIEGDenseMeasurementViewBinding:
    view_index: int
    view_id: str
    task_role: str
    view_receipt_id: str
    view_receipt_sha256: str
    transform_spec_sha256: str
    processed_view_sha256: str
    quality_mask_sha256: str
    reference_type: str
    reference_matrix_sha256: str
    output_sampling_rate_hz: float
    output_unit_ids: tuple[str, ...]
    unit_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.view_index, bool) or not isinstance(self.view_index, int) or self.view_index < 0:
            raise ValueError("view_index must be a non-negative integer")
        for name in (
            "view_id",
            "task_role",
            "view_receipt_id",
            "reference_type",
        ):
            _identifier(getattr(self, name), name)
        for name in (
            "view_receipt_sha256",
            "transform_spec_sha256",
            "processed_view_sha256",
            "quality_mask_sha256",
            "reference_matrix_sha256",
        ):
            _sha256(getattr(self, name), name)
        _finite(
            self.output_sampling_rate_hz,
            "output_sampling_rate_hz",
            minimum=_TOL,
        )
        if not self.output_unit_ids or len(self.output_unit_ids) != len(self.unit_indices):
            raise ValueError("view output unit IDs and event unit indices must align")
        if len(set(self.output_unit_ids)) != len(self.output_unit_ids):
            raise ValueError("view output unit IDs must be unique")
        if len(set(self.unit_indices)) != len(self.unit_indices) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.unit_indices
        ):
            raise ValueError("event unit indices must be unique non-negative integers")


@dataclass(frozen=True)
class BAIEGDenseMeasurementRowBinding:
    """Immutable provenance for one requested physical measurement row."""

    requested_row_index: int
    training_row_index: int | None
    view_index: int
    unit_index: int
    view_id: str
    unit_id: str
    unit_type: str
    requested_recording_interval_seconds: tuple[float, float]
    recording_interval_seconds: tuple[float, float]
    tensor_sample_interval: tuple[int, int]
    reference_type: str
    reference_row_sha256: str
    canonical_source_channel_ids: tuple[str, ...]
    effective_bandwidth_hz: tuple[float, float]
    quality_mask_sha256: str
    overlapping_quality_reason_codes: tuple[str, ...]
    target_value_mask: tuple[bool, ...]
    target_reason_codes: tuple[tuple[str, ...], ...]
    policy_sha256: str
    source_binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("requested_row_index", "view_index", "unit_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.training_row_index is not None and (
            isinstance(self.training_row_index, bool)
            or not isinstance(self.training_row_index, int)
            or self.training_row_index < 0
        ):
            raise ValueError("training_row_index must be null or non-negative")
        for name in ("view_id", "unit_id", "unit_type", "reference_type"):
            _identifier(getattr(self, name), name)
        requested_interval = _interval(
            self.requested_recording_interval_seconds,
            "requested_recording_interval_seconds",
        )
        interval = _interval(self.recording_interval_seconds, "recording_interval_seconds")
        if (
            interval[0] < requested_interval[0] - _TOL
            or interval[1] > requested_interval[1] + _TOL
        ):
            raise ValueError(
                "mapped recording support must remain inside the requested physical window"
            )
        tensor_start, tensor_stop = self.tensor_sample_interval
        if (
            isinstance(tensor_start, bool)
            or isinstance(tensor_stop, bool)
            or not isinstance(tensor_start, int)
            or not isinstance(tensor_stop, int)
            or tensor_start < 0
            or tensor_stop <= tensor_start
        ):
            raise ValueError("tensor_sample_interval must be a positive integer interval")
        bandwidth = _interval(self.effective_bandwidth_hz, "effective_bandwidth_hz")
        _sha256(self.reference_row_sha256, "reference_row_sha256")
        _sha256(self.quality_mask_sha256, "quality_mask_sha256")
        _sha256(self.policy_sha256, "policy_sha256")
        if len(self.target_value_mask) != len(BA_IEG_DETERMINISTIC_TARGETS):
            raise ValueError("target_value_mask must align with the 13-target vocabulary")
        if len(self.target_reason_codes) != len(BA_IEG_DETERMINISTIC_TARGETS):
            raise ValueError("target_reason_codes must align with the target vocabulary")
        normalized_reasons: list[tuple[str, ...]] = []
        for index, raw in enumerate(self.target_reason_codes):
            reasons = _sorted_unique(raw)
            if bool(self.target_value_mask[index]) == bool(reasons):
                raise ValueError(
                    "available targets need no reason; masked targets need a reason"
                )
            normalized_reasons.append(reasons)
        any_target = any(bool(value) for value in self.target_value_mask)
        if (self.training_row_index is not None) is not any_target:
            raise ValueError(
                "training_row_index must be present exactly when a row has supervision"
            )
        object.__setattr__(
            self,
            "requested_recording_interval_seconds",
            requested_interval,
        )
        object.__setattr__(self, "recording_interval_seconds", interval)
        object.__setattr__(self, "effective_bandwidth_hz", bandwidth)
        object.__setattr__(
            self,
            "canonical_source_channel_ids",
            _sorted_unique(self.canonical_source_channel_ids),
        )
        object.__setattr__(
            self,
            "overlapping_quality_reason_codes",
            _sorted_unique(self.overlapping_quality_reason_codes),
        )
        object.__setattr__(self, "target_reason_codes", tuple(normalized_reasons))
        object.__setattr__(self, "source_binding_sha256", self._compute_sha256())

    def _compute_sha256(self) -> str:
        material = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "source_binding_sha256"
        }
        return _canonical_sha256(material)

    def verify_integrity(self) -> None:
        if self.source_binding_sha256 != self._compute_sha256():
            raise ValueError("dense measurement row binding changed after registration")


@dataclass(frozen=True)
class BAIEGDenseMeasurementSidecar:
    """Dense row ledger plus the compact BA-IEG training target object."""

    canonical_signal_id: str
    canonical_receipt_sha256: str
    source_signal_sha256: str
    recording_id: str
    analysis_interval_seconds: tuple[float, float]
    background_intervals_seconds: tuple[tuple[float, float], ...]
    policy: BAIEGDenseMeasurementPolicy
    view_bindings: tuple[BAIEGDenseMeasurementViewBinding, ...]
    row_bindings: tuple[BAIEGDenseMeasurementRowBinding, ...]
    targets: BAIEGDeterministicTargets
    source_binding_sha256: str
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("canonical_signal_id", "recording_id"):
            _identifier(getattr(self, name), name)
        _sha256(self.canonical_receipt_sha256, "canonical_receipt_sha256")
        _sha256(self.source_signal_sha256, "source_signal_sha256")
        _sha256(self.source_binding_sha256, "source_binding_sha256")
        analysis = _interval(self.analysis_interval_seconds, "analysis_interval_seconds")
        if not isinstance(self.policy, BAIEGDenseMeasurementPolicy):
            raise TypeError("policy must be BAIEGDenseMeasurementPolicy")
        if not self.view_bindings or not self.row_bindings:
            raise ValueError("dense measurement sidecar requires views and requested rows")
        if not isinstance(self.targets, BAIEGDeterministicTargets):
            raise TypeError("targets must satisfy BAIEGDeterministicTargets")
        object.__setattr__(self, "analysis_interval_seconds", analysis)
        self._verify_content()
        object.__setattr__(self, "receipt_sha256", self._compute_receipt_sha256())

    @property
    def excluded_all_masked_row_count(self) -> int:
        return sum(row.training_row_index is None for row in self.row_bindings)

    def _expected_source_binding_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
                "method_id": BA_IEG_DENSE_MEASUREMENT_METHOD_ID,
                "canonical_signal_id": self.canonical_signal_id,
                "canonical_receipt_sha256": self.canonical_receipt_sha256,
                "source_signal_sha256": self.source_signal_sha256,
                "recording_id": self.recording_id,
                "analysis_interval_seconds": list(self.analysis_interval_seconds),
                "background_intervals_seconds": [
                    list(item) for item in self.background_intervals_seconds
                ],
                "policy_sha256": self.policy.sha256,
                "view_bindings": [asdict(item) for item in self.view_bindings],
                "row_source_binding_sha256s": [
                    row.source_binding_sha256 for row in self.row_bindings
                ],
            }
        )

    def _verify_content(self) -> None:
        self.targets.verify_integrity()
        if self.targets.policy_sha256 != self.policy.sha256:
            raise ValueError("target policy hash does not match the sidecar policy")
        if self.targets.source_binding_sha256 != self.source_binding_sha256:
            raise ValueError("target source binding does not match the sidecar")
        if self.source_binding_sha256 != self._expected_source_binding_sha256():
            raise ValueError("dense measurement aggregate source binding drifted")
        view_indices = [item.view_index for item in self.view_bindings]
        if view_indices != sorted(view_indices) or len(view_indices) != len(set(view_indices)):
            raise ValueError("view bindings must use unique canonical view-index order")
        if len({index for item in self.view_bindings for index in item.unit_indices}) != sum(
            len(item.unit_indices) for item in self.view_bindings
        ):
            raise ValueError("event unit indices cannot be shared across views")
        included: list[BAIEGDenseMeasurementRowBinding] = []
        for expected_index, row in enumerate(self.row_bindings):
            row.verify_integrity()
            if row.requested_row_index != expected_index:
                raise ValueError("requested rows must have contiguous canonical indices")
            if row.recording_interval_seconds[0] < self.analysis_interval_seconds[0] - _TOL or row.recording_interval_seconds[1] > self.analysis_interval_seconds[1] + _TOL:
                raise ValueError("measurement row lies outside the analysis interval")
            if row.training_row_index is not None:
                included.append(row)
        if not included:
            raise ValueError("no deterministic measurement survived eligibility masks")
        for expected_index, row in enumerate(included):
            if row.training_row_index != expected_index:
                raise ValueError("training rows must have contiguous indices")
            if int(self.targets.row_view_index[expected_index]) != row.view_index:
                raise ValueError("target/view binding mismatch")
            if int(self.targets.row_unit_index[expected_index]) != row.unit_index:
                raise ValueError("target/unit binding mismatch")
            if not torch.allclose(
                self.targets.row_time_bounds_seconds[expected_index].to(torch.float64),
                torch.tensor(row.recording_interval_seconds, dtype=torch.float64),
                atol=1e-7,
                rtol=0.0,
            ):
                raise ValueError("target/time binding mismatch")
            if tuple(bool(item) for item in self.targets.value_mask[expected_index]) != row.target_value_mask:
                raise ValueError("target mask differs from its row binding")
        if len(included) != int(self.targets.values.shape[0]):
            raise ValueError("row ledger and target tensor row counts disagree")

    def _compute_receipt_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
                "method_id": BA_IEG_DENSE_MEASUREMENT_METHOD_ID,
                "source_binding_sha256": self.source_binding_sha256,
                "target_receipt_sha256": self.targets.receipt_sha256,
                "requested_row_count": len(self.row_bindings),
                "training_row_count": int(self.targets.values.shape[0]),
                "excluded_all_masked_row_count": self.excluded_all_masked_row_count,
            }
        )

    def verify_integrity(self) -> None:
        self._verify_content()
        if self.receipt_sha256 != self._compute_receipt_sha256():
            raise ValueError("dense measurement sidecar changed after registration")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable, integrity-checked materialization."""

        self.verify_integrity()
        return {
            "schema_version": BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
            "method_id": BA_IEG_DENSE_MEASUREMENT_METHOD_ID,
            "canonical_signal_id": self.canonical_signal_id,
            "canonical_receipt_sha256": self.canonical_receipt_sha256,
            "source_signal_sha256": self.source_signal_sha256,
            "recording_id": self.recording_id,
            "analysis_interval_seconds": list(self.analysis_interval_seconds),
            "background_intervals_seconds": [
                list(item) for item in self.background_intervals_seconds
            ],
            "policy": self.policy.to_dict(),
            "policy_sha256": self.policy.sha256,
            "target_names": list(BA_IEG_DETERMINISTIC_TARGETS),
            "view_bindings": [asdict(item) for item in self.view_bindings],
            "row_bindings": [asdict(item) for item in self.row_bindings],
            "targets": {
                "values": self.targets.values.tolist(),
                "value_mask": self.targets.value_mask.tolist(),
                "row_time_bounds_seconds": self.targets.row_time_bounds_seconds.tolist(),
                "row_unit_index": self.targets.row_unit_index.tolist(),
                "row_view_index": self.targets.row_view_index.tolist(),
                "receipt_sha256": self.targets.receipt_sha256,
            },
            "source_binding_sha256": self.source_binding_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass
class _PreparedView:
    binding: BAIEGDenseMeasurementViewBinding
    receipt: dict[str, Any]
    tensor: np.ndarray
    sampling_rate_hz: float


@dataclass
class _RowWork:
    requested_row_index: int
    view: _PreparedView
    local_unit_index: int
    unit_index: int
    requested_recording_interval: tuple[float, float]
    recording_interval: tuple[float, float]
    tensor_interval: tuple[int, int]
    values: np.ndarray
    masks: np.ndarray
    reasons: list[list[str]]
    overlapping_quality_reasons: tuple[str, ...]


def _prepare_views(
    canonical: Mapping[str, Any],
    inputs: Sequence[BAIEGDenseMeasurementViewInput],
    *,
    analysis_interval: tuple[float, float],
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None,
) -> list[_PreparedView]:
    if not inputs:
        raise ValueError("at least one Findings/spatial view is required")
    prepared: list[_PreparedView] = []
    seen_view_indices: set[int] = set()
    seen_view_ids: set[str] = set()
    seen_unit_indices: set[int] = set()
    for input_index, item in enumerate(inputs):
        if not isinstance(item, BAIEGDenseMeasurementViewInput):
            raise TypeError(
                f"views[{input_index}] must be BAIEGDenseMeasurementViewInput"
            )
        if isinstance(item.view_index, bool) or not isinstance(item.view_index, int) or item.view_index < 0:
            raise ValueError("view_index must be a non-negative integer")
        if item.view_index in seen_view_indices:
            raise ValueError("view_index values must be unique")
        seen_view_indices.add(item.view_index)
        receipt = validate_signal_view_receipt(
            item.view_receipt,
            canonical,
            trusted_parent_views=trusted_parent_views,
        )
        if receipt["task_role"] not in BA_IEG_ALLOWED_VIEW_ROLES:
            raise ValueError("detector/boundary/display views cannot supply BA-IEG targets")
        view_id = str(receipt["view_id"])
        if view_id in seen_view_ids:
            raise ValueError("sidecar view IDs must be unique")
        seen_view_ids.add(view_id)
        selected_start, selected_stop = (
            float(value) for value in receipt["coordinates"]["selected_recording_seconds"]
        )
        if analysis_interval[0] < selected_start - _TOL or analysis_interval[1] > selected_stop + _TOL:
            raise ValueError("analysis interval lies outside a supplied signal view")
        output_units = receipt["output_units"]
        unit_ids = tuple(str(row["unit_id"]) for row in output_units)
        unit_indices = tuple(item.unit_indices)
        if len(unit_indices) != len(unit_ids):
            raise ValueError("unit_indices must align with the view output order")
        if len(set(unit_indices)) != len(unit_indices) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in unit_indices
        ):
            raise ValueError("unit_indices must be unique non-negative integers")
        if seen_unit_indices.intersection(unit_indices):
            raise ValueError("one event unit index cannot belong to multiple views")
        seen_unit_indices.update(unit_indices)
        tensor = item.tensor.detach().cpu().to(torch.float32).contiguous()
        expected_shape = (
            len(unit_ids),
            int(receipt["tensor_layout"]["tensor_sample_count"]),
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"view tensor shape {tuple(tensor.shape)} != receipt {expected_shape}"
            )
        actual_hash = deterministic_view_tensor_sha256(tensor, unit_ids=unit_ids)
        if actual_hash != receipt["processed_view_sha256"]:
            raise ValueError("processed view tensor hash does not match its receipt")
        clock = receipt["transform_spec"]["output_clock"]
        sampling_rate = float(clock["sampling_rate_numerator"]) / float(
            clock["sampling_rate_denominator"]
        )
        transform = receipt["transform_spec"]
        binding = BAIEGDenseMeasurementViewBinding(
            view_index=item.view_index,
            view_id=view_id,
            task_role=str(receipt["task_role"]),
            view_receipt_id=str(receipt["view_receipt_id"]),
            view_receipt_sha256=str(receipt["receipt_sha256"]),
            transform_spec_sha256=str(transform["transform_spec_sha256"]),
            processed_view_sha256=str(receipt["processed_view_sha256"]),
            quality_mask_sha256=str(receipt["masks"]["mask_sha256"]),
            reference_type=str(transform["reference"]["reference_type"]),
            reference_matrix_sha256=str(transform["reference"]["matrix_sha256"]),
            output_sampling_rate_hz=sampling_rate,
            output_unit_ids=unit_ids,
            unit_indices=unit_indices,
        )
        prepared.append(
            _PreparedView(
                binding=binding,
                receipt=receipt,
                tensor=tensor.numpy().astype(np.float64, copy=False),
                sampling_rate_hz=sampling_rate,
            )
        )
    prepared.sort(key=lambda item: item.binding.view_index)
    return prepared


def _normalise_background_intervals(
    values: Sequence[Sequence[float]],
    *,
    analysis_interval: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    result = tuple(_interval(item, f"background_intervals_seconds[{index}]") for index, item in enumerate(values))
    if result != tuple(sorted(result)):
        raise ValueError("background intervals must be sorted")
    for index, current in enumerate(result):
        if current[0] < analysis_interval[0] - _TOL or current[1] > analysis_interval[1] + _TOL:
            raise ValueError("background interval lies outside the analysis interval")
        if index and current[0] < result[index - 1][1] - _TOL:
            raise ValueError("background intervals must not overlap")
    return result


def _physical_windows(
    analysis_interval: tuple[float, float],
    policy: BAIEGDenseMeasurementPolicy,
) -> tuple[tuple[float, float], ...]:
    start, stop = analysis_interval
    origin = policy.global_grid_origin_seconds
    first_index = math.ceil((start - origin) / policy.step_seconds - _TOL)
    windows: list[tuple[float, float]] = []
    index = max(0, first_index)
    while True:
        window_start = origin + index * policy.step_seconds
        window_stop = window_start + policy.window_seconds
        if window_stop > stop + _TOL:
            break
        if window_start >= start - _TOL:
            windows.append((float(window_start), float(window_stop)))
        index += 1
    if not windows:
        raise ValueError("analysis interval contains no complete policy-aligned window")
    return tuple(windows)


def _family_row(unit: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    return next(
        row for row in unit["evidence_eligibility"] if row["family"] == family
    )


def _family_reasons(
    view: _PreparedView,
    *,
    local_unit_index: int,
    tensor_interval: tuple[int, int],
    family: str,
) -> tuple[str, ...]:
    unit = view.receipt["output_units"][local_unit_index]
    eligibility = _family_row(unit, family)
    reasons = [str(item) for item in eligibility["reason_codes"]]
    for start, stop in view.receipt["masks"]["padding_intervals"]:
        if _overlaps(tensor_interval, (int(start), int(stop))):
            reasons.append("view_padding_overlap")
    for start, stop in view.receipt["masks"]["edge_invalid_intervals"]:
        if _overlaps(tensor_interval, (int(start), int(stop))):
            reasons.append("view_filter_edge_overlap")
    for row in view.receipt["masks"]["quality_invalid_intervals"]:
        if str(row["unit_id"]) != str(unit["unit_id"]):
            continue
        interval = tuple(int(value) for value in row["tensor_sample_interval"])
        if not _overlaps(tensor_interval, interval) or family not in row["disabled_evidence_families"]:
            continue
        reasons.append(f"quality_severity:{row['severity']}")
        reasons.extend(str(item) for item in row["reason_codes"])
    return _sorted_unique(reasons)


def _overlapping_quality_reasons(
    view: _PreparedView,
    *,
    local_unit_index: int,
    tensor_interval: tuple[int, int],
) -> tuple[str, ...]:
    unit_id = str(view.receipt["output_units"][local_unit_index]["unit_id"])
    reasons: list[str] = []
    for start, stop in view.receipt["masks"]["padding_intervals"]:
        if _overlaps(tensor_interval, (int(start), int(stop))):
            reasons.append("view_padding_overlap")
    for start, stop in view.receipt["masks"]["edge_invalid_intervals"]:
        if _overlaps(tensor_interval, (int(start), int(stop))):
            reasons.append("view_filter_edge_overlap")
    for row in view.receipt["masks"]["quality_invalid_intervals"]:
        if str(row["unit_id"]) != unit_id:
            continue
        interval = tuple(int(value) for value in row["tensor_sample_interval"])
        if _overlaps(tensor_interval, interval):
            reasons.append(f"quality_severity:{row['severity']}")
            reasons.extend(str(item) for item in row["reason_codes"])
    return _sorted_unique(reasons)


def _reference_row_sha256(view: _PreparedView, local_unit_index: int) -> str:
    transform = view.receipt["transform_spec"]
    reference = transform["reference"]
    return _canonical_sha256(
        {
            "reference_matrix_sha256": reference["matrix_sha256"],
            "reference_type": reference["reference_type"],
            "input_unit_ids": reference["input_unit_ids"],
            "output_unit_id": reference["output_unit_ids"][local_unit_index],
            "coefficients": reference["matrix"][local_unit_index],
        }
    )


def _map_window_to_tensor(
    view: _PreparedView, window: tuple[float, float]
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Map one nominal physical window inward on a view-specific clock.

    The nominal grid is shared in recording-relative seconds.  Each view may
    legitimately have a different sample rate, so the left edge is rounded
    up and the right edge down.  The returned physical interval is the actual
    sample support and is never replaced by the nominal interval.
    """

    start = recording_seconds_to_view_tensor_index(
        view.receipt,
        recording_seconds=window[0],
        rounding="ceil",
    )
    stop = recording_seconds_to_view_tensor_index(
        view.receipt,
        recording_seconds=window[1],
        rounding="floor",
    )
    if stop - start < 2:
        raise ValueError("physical measurement window contains fewer than two samples")
    replayed_start = view_tensor_index_to_recording_seconds(
        view.receipt, tensor_sample_index=start
    )
    replayed_stop = view_tensor_index_to_recording_seconds(
        view.receipt, tensor_sample_index=stop
    )
    tolerance = 1.0 / view.sampling_rate_hz + _TOL
    if (
        replayed_start < window[0] - _TOL
        or replayed_stop > window[1] + _TOL
        or replayed_start - window[0] > tolerance
        or window[1] - replayed_stop > tolerance
    ):
        raise ValueError("measurement window cannot be mapped inward on the view clock")
    return (int(start), int(stop)), (
        float(replayed_start),
        float(replayed_stop),
    )


def _append_reason(row: _RowWork, target_index: int, reason: str) -> None:
    if reason not in row.reasons[target_index]:
        row.reasons[target_index].append(reason)
    row.masks[target_index] = False
    row.values[target_index] = 0.0


def _initial_row(
    *,
    requested_row_index: int,
    view: _PreparedView,
    local_unit_index: int,
    unit_index: int,
    requested_window: tuple[float, float],
    recording_interval: tuple[float, float],
    tensor_interval: tuple[int, int],
    policy: BAIEGDenseMeasurementPolicy,
) -> _RowWork:
    values = np.zeros(len(BA_IEG_DETERMINISTIC_TARGETS), dtype=np.float64)
    masks = np.zeros(len(BA_IEG_DETERMINISTIC_TARGETS), dtype=bool)
    reasons: list[list[str]] = [[] for _ in BA_IEG_DETERMINISTIC_TARGETS]
    amplitude_reasons = list(
        _family_reasons(
            view,
            local_unit_index=local_unit_index,
            tensor_interval=tensor_interval,
            family="amplitude",
        )
    )
    spectral_reasons = list(
        _family_reasons(
            view,
            local_unit_index=local_unit_index,
            tensor_interval=tensor_interval,
            family="spectral",
        )
    )
    spatial_reasons = list(
        _family_reasons(
            view,
            local_unit_index=local_unit_index,
            tensor_interval=tensor_interval,
            family="spatial_field",
        )
    )
    for target_index in _AMPLITUDE_TARGETS:
        reasons[target_index].extend(amplitude_reasons)
        masks[target_index] = not reasons[target_index]
    for target_index in _SPECTRAL_TARGETS:
        reasons[target_index].extend(spectral_reasons)
        masks[target_index] = not reasons[target_index]
    reasons[_CHANGE_TARGET_INDEX].extend(
        _sorted_unique(amplitude_reasons + spectral_reasons + spatial_reasons)
    )
    return _RowWork(
        requested_row_index=requested_row_index,
        view=view,
        local_unit_index=local_unit_index,
        unit_index=unit_index,
        requested_recording_interval=requested_window,
        recording_interval=recording_interval,
        tensor_interval=tensor_interval,
        values=values,
        masks=masks,
        reasons=reasons,
        overlapping_quality_reasons=_overlapping_quality_reasons(
            view,
            local_unit_index=local_unit_index,
            tensor_interval=tensor_interval,
        ),
    )


def _measure_row(row: _RowWork, policy: BAIEGDenseMeasurementPolicy) -> None:
    start, stop = row.tensor_interval
    segment_volts = row.view.tensor[row.local_unit_index, start:stop]
    unit = row.view.receipt["output_units"][row.local_unit_index]
    result = measure_ba_ieg_base_numerical_features(
        segment_volts,
        sampling_rate_hz=row.view.sampling_rate_hz,
        effective_bandwidth_hz=unit["effective_bandwidth_hz"],
        policy=policy.base_numerical_policy,
        amplitude_reason_codes=row.reasons[_TARGET_INDEX["rms_uv"]],
        spectral_reason_codes=row.reasons[
            _TARGET_INDEX["dominant_frequency_hz"]
        ],
    )
    for target_index in range(len(BA_IEG_BASE_MEASUREMENT_NAMES)):
        row.values[target_index] = result.values[target_index]
        row.masks[target_index] = result.value_mask[target_index]
        row.reasons[target_index] = list(result.reason_codes[target_index])


def _robust_z(
    value: float,
    baseline_values: np.ndarray,
    *,
    floor: float,
) -> float | None:
    finite = baseline_values[np.isfinite(baseline_values)]
    if finite.size < 2 or not math.isfinite(value):
        return None
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = max(1.4826 * mad, floor)
    return abs(float(value) - center) / scale


def _add_change_scores(
    rows: Sequence[_RowWork],
    *,
    background_intervals: tuple[tuple[float, float], ...],
    policy: BAIEGDenseMeasurementPolicy,
) -> None:
    groups: dict[tuple[int, int], list[_RowWork]] = {}
    for row in rows:
        groups.setdefault(
            (row.view.binding.view_index, row.unit_index), []
        ).append(row)
    for group_rows in groups.values():
        if not background_intervals:
            for row in group_rows:
                if not row.reasons[_CHANGE_TARGET_INDEX]:
                    _append_reason(
                        row,
                        _CHANGE_TARGET_INDEX,
                        "background_reference_absent",
                    )
            continue
        baseline_rows = [
            row
            for row in group_rows
            if _contained(row.recording_interval, background_intervals)
        ]
        if len(baseline_rows) < policy.minimum_baseline_windows:
            for row in group_rows:
                if not row.reasons[_CHANGE_TARGET_INDEX]:
                    _append_reason(
                        row,
                        _CHANGE_TARGET_INDEX,
                        "insufficient_baseline_windows",
                    )
            continue

        component_specs: tuple[tuple[int, str, float], ...] = (
            (
                _TARGET_INDEX["rms_uv"],
                "log",
                policy.log_amplitude_robust_scale_floor,
            ),
            (
                _TARGET_INDEX["line_length_uv_per_sample"],
                "log",
                policy.log_amplitude_robust_scale_floor,
            ),
            (
                _TARGET_INDEX["spectral_entropy"],
                "identity",
                policy.entropy_robust_scale_floor,
            ),
        )
        baseline_by_target: dict[int, np.ndarray] = {}
        for target_index, transform, _ in component_specs:
            values = [
                row.values[target_index]
                for row in baseline_rows
                if row.masks[target_index]
            ]
            array = np.asarray(values, dtype=np.float64)
            if transform == "log":
                array = np.log(array + 1e-6)
            baseline_by_target[target_index] = array
        band_baselines: dict[int, np.ndarray] = {}
        for target_name, _, _ in _BAND_TARGETS:
            target_index = _TARGET_INDEX[target_name]
            band_baselines[target_index] = np.asarray(
                [
                    row.values[target_index]
                    for row in baseline_rows
                    if row.masks[target_index]
                ],
                dtype=np.float64,
            )

        for row in group_rows:
            if row.reasons[_CHANGE_TARGET_INDEX]:
                row.masks[_CHANGE_TARGET_INDEX] = False
                continue
            components: list[float] = []
            for target_index, transform, floor in component_specs:
                if not row.masks[target_index]:
                    continue
                value = float(row.values[target_index])
                if transform == "log":
                    value = math.log(value + 1e-6)
                score = _robust_z(
                    value,
                    baseline_by_target[target_index],
                    floor=floor,
                )
                if score is not None and baseline_by_target[target_index].size >= policy.minimum_baseline_windows:
                    components.append(score)
            band_scores: list[float] = []
            for target_name, _, _ in _BAND_TARGETS:
                target_index = _TARGET_INDEX[target_name]
                baseline = band_baselines[target_index]
                if not row.masks[target_index] or baseline.size < policy.minimum_baseline_windows:
                    continue
                score = _robust_z(
                    float(row.values[target_index]),
                    baseline,
                    floor=policy.band_ratio_robust_scale_floor,
                )
                if score is not None:
                    band_scores.append(score)
            if band_scores:
                components.append(max(band_scores))
            if len(components) < 2:
                _append_reason(
                    row,
                    _CHANGE_TARGET_INDEX,
                    "insufficient_change_components",
                )
                continue
            row.values[_CHANGE_TARGET_INDEX] = float(
                np.mean(sorted(components)[-2:])
            )
            row.masks[_CHANGE_TARGET_INDEX] = True


def _finalize_rows(
    rows: Sequence[_RowWork],
    *,
    policy: BAIEGDenseMeasurementPolicy,
) -> tuple[
    tuple[BAIEGDenseMeasurementRowBinding, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    bindings: list[BAIEGDenseMeasurementRowBinding] = []
    included_values: list[np.ndarray] = []
    included_masks: list[np.ndarray] = []
    included_times: list[tuple[float, float]] = []
    included_units: list[int] = []
    included_views: list[int] = []
    for requested_index, row in enumerate(rows):
        if requested_index != row.requested_row_index:
            raise ValueError("internal requested-row order drifted")
        if not np.isfinite(row.values[row.masks]).all():
            raise ValueError("deterministic measurement produced a non-finite value")
        row.values[~row.masks] = 0.0
        for target_index in range(len(BA_IEG_DETERMINISTIC_TARGETS)):
            row.reasons[target_index] = list(
                _sorted_unique(row.reasons[target_index])
            )
            if row.masks[target_index] and row.reasons[target_index]:
                raise ValueError("available deterministic target carries a reason code")
            if not row.masks[target_index] and not row.reasons[target_index]:
                row.reasons[target_index].append("measurement_unavailable")
        training_row_index = len(included_values) if row.masks.any() else None
        unit = row.view.receipt["output_units"][row.local_unit_index]
        binding = BAIEGDenseMeasurementRowBinding(
            requested_row_index=requested_index,
            training_row_index=training_row_index,
            view_index=row.view.binding.view_index,
            unit_index=row.unit_index,
            view_id=row.view.binding.view_id,
            unit_id=str(unit["unit_id"]),
            unit_type=str(unit["unit_type"]),
            requested_recording_interval_seconds=(
                row.requested_recording_interval
            ),
            recording_interval_seconds=row.recording_interval,
            tensor_sample_interval=row.tensor_interval,
            reference_type=row.view.binding.reference_type,
            reference_row_sha256=_reference_row_sha256(
                row.view, row.local_unit_index
            ),
            canonical_source_channel_ids=tuple(
                str(item) for item in unit["canonical_source_channel_ids"]
            ),
            effective_bandwidth_hz=tuple(
                float(item) for item in unit["effective_bandwidth_hz"]
            ),
            quality_mask_sha256=row.view.binding.quality_mask_sha256,
            overlapping_quality_reason_codes=row.overlapping_quality_reasons,
            target_value_mask=tuple(bool(item) for item in row.masks),
            target_reason_codes=tuple(
                tuple(items) for items in row.reasons
            ),
            policy_sha256=policy.sha256,
        )
        bindings.append(binding)
        if training_row_index is not None:
            included_values.append(row.values.copy())
            included_masks.append(row.masks.copy())
            included_times.append(row.recording_interval)
            included_units.append(row.unit_index)
            included_views.append(row.view.binding.view_index)
    if not included_values:
        raise ValueError("all requested deterministic measurement rows are unavailable")
    return (
        tuple(bindings),
        torch.from_numpy(np.stack(included_values)).to(torch.float32),
        torch.from_numpy(np.stack(included_masks)).to(torch.bool),
        torch.tensor(included_times, dtype=torch.float64),
        torch.tensor(included_units, dtype=torch.long),
        torch.tensor(included_views, dtype=torch.long),
    )


def materialize_ba_ieg_dense_measurement_sidecar(
    *,
    canonical_receipt: object,
    views: Sequence[BAIEGDenseMeasurementViewInput],
    analysis_interval_seconds: Sequence[float],
    background_intervals_seconds: Sequence[Sequence[float]] = (),
    policy: BAIEGDenseMeasurementPolicy = DEFAULT_BA_IEG_DENSE_MEASUREMENT_POLICY,
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
) -> BAIEGDenseMeasurementSidecar:
    """Materialize replayable BA-IEG numerical targets on a physical grid."""

    if not isinstance(policy, BAIEGDenseMeasurementPolicy):
        raise TypeError("policy must be BAIEGDenseMeasurementPolicy")
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    analysis_interval = _interval(
        analysis_interval_seconds, "analysis_interval_seconds"
    )
    if analysis_interval[1] > float(canonical["recording_duration_seconds"]) + _TOL:
        raise ValueError("analysis interval exceeds the canonical recording")
    background_intervals = _normalise_background_intervals(
        background_intervals_seconds,
        analysis_interval=analysis_interval,
    )
    prepared = _prepare_views(
        canonical,
        views,
        analysis_interval=analysis_interval,
        trusted_parent_views=trusted_parent_views,
    )
    windows = _physical_windows(analysis_interval, policy)
    rows: list[_RowWork] = []
    requested_row_index = 0
    for view in prepared:
        local_units = sorted(
            range(len(view.binding.unit_indices)),
            key=lambda index: view.binding.unit_indices[index],
        )
        mapped_windows = {
            window: _map_window_to_tensor(view, window) for window in windows
        }
        for local_unit_index in local_units:
            unit_index = int(view.binding.unit_indices[local_unit_index])
            for window in windows:
                tensor_interval, recording_interval = mapped_windows[window]
                row = _initial_row(
                    requested_row_index=requested_row_index,
                    view=view,
                    local_unit_index=local_unit_index,
                    unit_index=unit_index,
                    requested_window=window,
                    recording_interval=recording_interval,
                    tensor_interval=tensor_interval,
                    policy=policy,
                )
                _measure_row(row, policy)
                rows.append(row)
                requested_row_index += 1
    _add_change_scores(
        rows,
        background_intervals=background_intervals,
        policy=policy,
    )
    (
        row_bindings,
        values,
        masks,
        times,
        unit_indices,
        view_indices,
    ) = _finalize_rows(rows, policy=policy)
    view_bindings = tuple(view.binding for view in prepared)
    aggregate_source_binding = _canonical_sha256(
        {
            "schema_version": BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
            "method_id": BA_IEG_DENSE_MEASUREMENT_METHOD_ID,
            "canonical_signal_id": canonical["canonical_signal_id"],
            "canonical_receipt_sha256": canonical["receipt_sha256"],
            "source_signal_sha256": canonical["source_signal_sha256"],
            "recording_id": canonical["recording_id"],
            "analysis_interval_seconds": list(analysis_interval),
            "background_intervals_seconds": [
                list(item) for item in background_intervals
            ],
            "policy_sha256": policy.sha256,
            "view_bindings": [asdict(item) for item in view_bindings],
            "row_source_binding_sha256s": [
                row.source_binding_sha256 for row in row_bindings
            ],
        }
    )
    targets = BAIEGDeterministicTargets(
        values=values,
        value_mask=masks,
        row_time_bounds_seconds=times,
        row_unit_index=unit_indices,
        row_view_index=view_indices,
        policy_sha256=policy.sha256,
        source_binding_sha256=aggregate_source_binding,
    )
    return BAIEGDenseMeasurementSidecar(
        canonical_signal_id=str(canonical["canonical_signal_id"]),
        canonical_receipt_sha256=str(canonical["receipt_sha256"]),
        source_signal_sha256=str(canonical["source_signal_sha256"]),
        recording_id=str(canonical["recording_id"]),
        analysis_interval_seconds=analysis_interval,
        background_intervals_seconds=background_intervals,
        policy=policy,
        view_bindings=view_bindings,
        row_bindings=row_bindings,
        targets=targets,
        source_binding_sha256=aggregate_source_binding,
    )
