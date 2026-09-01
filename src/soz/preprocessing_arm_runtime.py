"""Runtime signal implementations for the frozen preprocessing-parity arms.

This module is deliberately target-free.  It reads direct physical EDF
channels, keeps the raw reference geometry intact until the requested arm is
applied, and returns model-ready volt-valued intervals.  Numerical parity
evaluation lives above this layer; SOZ labels are never accepted here.

The deployable arms operate on the frozen standard-19 carrier.  ``O-REF`` is
the separate official 23-channel/full-record sanity geometry and therefore has
its own loader and interval function instead of being silently squeezed into
the 19-channel deployment contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Sequence

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt
import torch
import torch.nn as nn

from .data.edf import CausalEDFConfig, causal_bandpass_resample
from .geometry import STANDARD_19, normalize_electrode_name
from .models.labram import (
    AUDITED_ENCODER_TENSOR_COUNT,
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LABRAM_POSITION_ID_BY_NAME,
    _extract_student_encoder_state,
    _load_checkpoint_snapshot,
    _load_modeling_module,
    _stable_file_snapshot,
)
from .preprocessing_parity import (
    DEPLOYABLE_PREPROCESSING_ARM_IDS,
    FROZEN_PREPROCESSING_ARM_SPEC_BY_ID,
    OFFICIAL_REF23_CHANNELS,
)


_UNIT_TO_VOLTS = {"v": 1.0, "mv": 1e-3, "uv": 1e-6}
_OUTPUT_SFREQ_HZ = 200.0
CAUSAL_REFERENCE_SENSITIVITY_ARM_ID = "C-REF19"
CAUSAL_REFERENCE_PAIR_SCHEMA = "soz_causal_reference_pair_v1"
CAUSAL_REFERENCE_PAIR_ROLE = (
    "target_free_reference_robustness_and_abstention_only_not_arm_selection"
)
_OFFICIAL_ONLY_LABRAM_POSITION_ID_BY_NAME = {
    "A1": 95,
    "A2": 96,
    "T1": 113,
    "T2": 114,
}
OFFICIAL_REF23_LABRAM_POSITION_IDS = tuple(
    LABRAM_POSITION_ID_BY_NAME.get(
        name, _OFFICIAL_ONLY_LABRAM_POSITION_ID_BY_NAME.get(name, -1)
    )
    for name in OFFICIAL_REF23_CHANNELS
)
if any(position_id < 1 for position_id in OFFICIAL_REF23_LABRAM_POSITION_IDS):
    raise RuntimeError("Official REF23 geometry contains an unknown LaBraM position")


def _half_up(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("Sample coordinate must be finite")
    return int(
        Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def _unit_scale(unit: object) -> float:
    normalized = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")
    try:
        return _UNIT_TO_VOLTS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported EDF physical unit: {unit!r}") from exc


def _raw_position_name(raw_name: object) -> str:
    text = str(raw_name).strip().upper().replace("_", "-")
    for prefix in ("EEG ", "EEG-"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    for suffix in ("-REF", "-LE", "-AR", "-AVG", "-AV", "-CAR"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip("- ")


def _default_reader_factory(path: str) -> object:
    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover - deployment dependency gate
        raise RuntimeError("pyedflib is required for preprocessing parity") from exc
    return pyedflib.EdfReader(path)


@dataclass(frozen=True)
class PhysicalEDFRecord:
    """One direct physical EDF payload in a frozen channel order."""

    path: Path
    channel_names: tuple[str, ...]
    channel_keys: tuple[str, ...]
    units: tuple[str, ...]
    source_sfreq_hz: float
    values_volts: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("Physical EDF path must be absolute")
        n_channels = len(self.channel_keys)
        if n_channels not in {19, 23}:
            raise ValueError("Physical EDF parity payload must contain 19 or 23 channels")
        if len(self.channel_names) != n_channels or len(self.units) != n_channels:
            raise ValueError("Physical EDF per-channel metadata is misaligned")
        values = np.asarray(self.values_volts)
        if values.ndim != 2 or values.shape[0] != n_channels or values.shape[1] < 2:
            raise ValueError("Physical EDF payload has an invalid shape")
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError("Physical EDF payload must be finite floating point")
        if not math.isfinite(self.source_sfreq_hz) or self.source_sfreq_hz <= 0:
            raise ValueError("Physical EDF sampling rate must be positive")

    @property
    def duration_sec(self) -> float:
        return self.values_volts.shape[1] / self.source_sfreq_hz


@dataclass(frozen=True)
class PreparedArmInterval:
    """One target-free interval produced by a frozen arm."""

    arm_id: str
    data_volts: np.ndarray
    output_sfreq_hz: float
    start_sec: float
    stop_sec: float
    receipt: dict[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.data_volts)
        expected_samples = _half_up((self.stop_sec - self.start_sec) * self.output_sfreq_hz)
        if values.ndim != 2 or values.shape[1] != expected_samples:
            raise ValueError("Prepared parity interval has the wrong time geometry")
        expected_channels = 23 if self.arm_id == "O-REF" else 19
        if values.shape[0] != expected_channels:
            raise ValueError("Prepared parity interval has the wrong channel geometry")
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Prepared parity interval must be finite float32")


@dataclass(frozen=True)
class PreparedCausalReferencePair:
    """Same-event causal REF19/CAR19 pair for target-free robustness checks.

    ``C-REF19`` is deliberately not added to the frozen five-arm selection.
    Both members are derived from one forward filter/resample call and one
    logical crop, so disagreement isolates rereferencing rather than phase,
    bandwidth, state-reset, or sample-alignment changes.
    """

    ref19: PreparedArmInterval
    car19: PreparedArmInterval
    receipt: dict[str, object]

    def __post_init__(self) -> None:
        if self.ref19.arm_id != CAUSAL_REFERENCE_SENSITIVITY_ARM_ID:
            raise ValueError("Reference pair ref19 member has the wrong arm ID")
        if self.car19.arm_id != "C-CAR19":
            raise ValueError("Reference pair car19 member has the wrong arm ID")
        shared_geometry = (
            self.ref19.output_sfreq_hz == self.car19.output_sfreq_hz
            and self.ref19.start_sec == self.car19.start_sec
            and self.ref19.stop_sec == self.car19.stop_sec
            and self.ref19.data_volts.shape == self.car19.data_volts.shape
        )
        if not shared_geometry:
            raise ValueError("Causal reference-pair members have mismatched geometry")
        expected_car = _apply_car(self.ref19.data_volts).astype(np.float32)
        if not np.allclose(
            self.car19.data_volts,
            expected_car,
            rtol=5e-6,
            atol=5e-12,
        ):
            raise ValueError("C-CAR19 is not the common-average view of C-REF19")
        if self.receipt.get("schema_version") != CAUSAL_REFERENCE_PAIR_SCHEMA:
            raise ValueError("Causal reference-pair receipt has the wrong schema")
        if self.receipt.get("role") != CAUSAL_REFERENCE_PAIR_ROLE:
            raise ValueError("Causal reference-pair receipt has the wrong role")
        if self.receipt.get("shared_filter_resample_crop") is not True:
            raise ValueError("Causal reference pair must share filter/resample/crop")


class OfficialReference23LaBraMEncoder(nn.Module):
    """Audited LaBraM encoder for the non-deployable official 23-lead sanity.

    The production encoder intentionally accepts only the standard-19 carrier.
    This separate class keeps the official TUEV geometry from weakening that
    contract while still loading exactly the same pinned encoder tensors.
    """

    token_dim = 200
    samples_per_token = 200
    seconds_per_call = 5

    def __init__(
        self,
        *,
        modeling_path: str | Path,
        checkpoint_path: str | Path,
    ) -> None:
        super().__init__()
        modeling = _stable_file_snapshot(
            modeling_path, label="Official LaBraM modeling source"
        )
        if modeling.sha256 != AUDITED_LABRAM_MODELING_SHA256:
            raise ValueError("Official LaBraM modeling source SHA-256 mismatch")
        module = _load_modeling_module(
            str(modeling.path), modeling.sha256, modeling.content
        )
        self.backbone = module.labram_base_patch200_200(
            EEG_size=1600,
            in_chans=1,
            out_chans=8,
            num_classes=0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_values=0.1,
            use_mean_pooling=False,
        )
        payload, checkpoint = _load_checkpoint_snapshot(
            checkpoint_path,
            expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        )
        state = _extract_student_encoder_state(payload)
        if len(state) != AUDITED_ENCODER_TENSOR_COUNT:
            raise ValueError("Unexpected official LaBraM encoder tensor count")
        self.backbone.load_state_dict(state, strict=True)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer(
            "input_chans",
            torch.tensor(
                (0, *OFFICIAL_REF23_LABRAM_POSITION_IDS), dtype=torch.long
            ),
            persistent=True,
        )
        self.checkpoint_sha256 = checkpoint.sha256
        self.modeling_sha256 = modeling.sha256

    def train(self, mode: bool = True) -> "OfficialReference23LaBraMEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        expected = (23, self.seconds_per_call, self.samples_per_token)
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected:
            raise ValueError(f"Official O-REF input must have shape [B,{expected}]")
        if not patches.is_floating_point() or not torch.isfinite(patches).all():
            raise ValueError("Official O-REF input must be finite floating point")
        self.backbone.eval()
        with torch.no_grad():
            flat = self.backbone.forward_features(
                patches * 1e4,
                input_chans=self.input_chans,
                return_patch_tokens=True,
            )
        expected_flat = (patches.shape[0], 23 * 5, self.token_dim)
        if tuple(flat.shape) != expected_flat:
            raise ValueError("Official O-REF encoder returned the wrong token shape")
        return flat.reshape(patches.shape[0], 23, 5, self.token_dim).detach()


def read_physical_edf(
    path: str | Path,
    *,
    geometry: str,
    reader_factory: Callable[[str], object] | None = None,
) -> PhysicalEDFRecord:
    """Read one unambiguous direct physical montage without rereferencing."""

    source = Path(path).resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Parity EDF must be a canonical regular file")
    if geometry == "standard19":
        requested = STANDARD_19
        keyer = normalize_electrode_name
    elif geometry == "official_ref23":
        requested = OFFICIAL_REF23_CHANNELS
        keyer = _raw_position_name
    else:
        raise ValueError("Unknown parity channel geometry")
    factory = _default_reader_factory if reader_factory is None else reader_factory
    reader = factory(str(source))
    try:
        labels = tuple(str(value).strip() for value in reader.getSignalLabels())
        candidates: dict[str, list[int]] = {channel: [] for channel in requested}
        for index, label in enumerate(labels):
            key = keyer(label)
            if key in candidates:
                candidates[key].append(index)
        missing = tuple(channel for channel, rows in candidates.items() if not rows)
        duplicates = {
            channel: tuple(labels[index] for index in rows)
            for channel, rows in candidates.items()
            if len(rows) > 1
        }
        if missing or duplicates:
            raise ValueError(
                "EDF lacks an unambiguous direct parity montage; "
                f"missing={missing}, duplicates={duplicates}"
            )
        indices = tuple(candidates[channel][0] for channel in requested)
        sfreqs = tuple(float(reader.getSampleFrequency(index)) for index in indices)
        if any(not math.isfinite(value) or value <= 0 for value in sfreqs):
            raise ValueError("EDF contains an invalid sampling rate")
        if any(abs(value - sfreqs[0]) > 1e-9 for value in sfreqs):
            raise ValueError("Selected physical channels have mixed sampling rates")
        sample_counts_raw = reader.getNSamples()
        sample_counts = tuple(int(sample_counts_raw[index]) for index in indices)
        if any(value <= 0 for value in sample_counts) or len(set(sample_counts)) != 1:
            raise ValueError("Selected physical channels have mixed sample counts")
        units = tuple(str(reader.getPhysicalDimension(index)).strip() for index in indices)
        scales = np.asarray([_unit_scale(value) for value in units], dtype=np.float64)
        arrays = tuple(
            np.asarray(reader.readSignal(index), dtype=np.float64) for index in indices
        )
        if any(array.shape != (sample_counts[0],) for array in arrays):
            raise ValueError("EDF reader returned an invalid physical payload")
        values = np.stack(arrays) * scales[:, None]
        return PhysicalEDFRecord(
            path=source,
            channel_names=tuple(labels[index] for index in indices),
            channel_keys=tuple(requested),
            units=units,
            source_sfreq_hz=sfreqs[0],
            values_volts=np.ascontiguousarray(values, dtype=np.float64),
        )
    finally:
        if hasattr(reader, "close"):
            reader.close()


def _apply_car(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    return data - data.mean(axis=0, keepdims=True)


def prepare_full_record_arm(
    record: PhysicalEDFRecord,
    *,
    arm_id: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Preprocess a complete record for one full-record arm."""

    if arm_id == "O-REF":
        if record.channel_keys != OFFICIAL_REF23_CHANNELS:
            raise ValueError("O-REF requires the frozen official 23-channel order")
    elif arm_id in {"O-CAR19", "Z-REF19", "Z-CAR19"}:
        if record.channel_keys != STANDARD_19:
            raise ValueError(f"{arm_id} requires the frozen standard-19 order")
    else:
        raise ValueError("Full-record preprocessing does not implement this arm")
    spec = FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[arm_id]
    if spec.lowpass_hz >= 0.5 * record.source_sfreq_hz:
        raise ValueError("Arm lowpass is invalid for this EDF sampling rate")
    # MNE imports numba helpers whose default in-package cache is not writable
    # in a read-only deployment environment.  A task-scoped cache preserves
    # normal JIT behaviour without mutating package files.
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "neurosoz-numba-cache"),
    )
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - formal dependency gate
        raise RuntimeError("MNE is required for zero-phase parity arms") from exc
    if arm_id in {"O-REF", "O-CAR19"}:
        filtered = mne.filter.filter_data(
            record.values_volts,
            sfreq=record.source_sfreq_hz,
            l_freq=spec.highpass_hz,
            h_freq=spec.lowpass_hz,
            method="fir",
            phase="zero-double",
            fir_design="firwin",
            copy=True,
            verbose=False,
        )
        filtered = mne.filter.notch_filter(
            filtered,
            Fs=record.source_sfreq_hz,
            freqs=np.asarray([spec.notch_hz], dtype=float),
            method="fir",
            phase="zero-double",
            fir_design="firwin",
            copy=True,
            verbose=False,
        )
        implementation = "mne_fir_zero_double_notch_then_fft_resample_v1"
    else:
        sos = butter(
            4,
            [spec.highpass_hz, spec.lowpass_hz],
            btype="bandpass",
            fs=record.source_sfreq_hz,
            output="sos",
        )
        filtered = sosfiltfilt(sos, record.values_volts, axis=-1)
        implementation = "scipy_sosfiltfilt_order4_then_fft_resample_v1"
    output_count = _half_up(record.duration_sec * _OUTPUT_SFREQ_HZ)
    processed = resample(filtered, output_count, axis=-1)
    if spec.reference == "car19":
        processed = _apply_car(processed)
    elif spec.reference != "physical_ref_no_rereference":
        raise ValueError("Unknown frozen parity reference")
    processed = np.ascontiguousarray(processed, dtype=np.float64)
    if not np.isfinite(processed).all():
        raise ValueError("Full-record parity preprocessing produced non-finite values")
    return processed, {
        "arm_id": arm_id,
        "arm_spec_receipt_sha256": spec.receipt_sha256,
        "implementation": implementation,
        "state_scope": spec.state_scope,
        "source_sfreq_hz": record.source_sfreq_hz,
        "output_sfreq_hz": _OUTPUT_SFREQ_HZ,
        "source_sample_count": record.values_volts.shape[1],
        "output_sample_count": output_count,
        "reference": spec.reference,
    }


def _causal_pre_reference_interval(
    record: PhysicalEDFRecord,
    *,
    start_sec: float,
    stop_sec: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if record.channel_keys != STANDARD_19:
        raise ValueError("C-CAR19 requires the frozen standard-19 order")
    if start_sec < 30.0:
        raise ValueError("C-CAR19 interval lacks 30 seconds of real warmup")
    if stop_sec <= start_sec or stop_sec > record.duration_sec + 1e-9:
        raise ValueError("C-CAR19 interval lies outside the EDF")
    sfreq = record.source_sfreq_hz
    state_reset = int(math.floor((start_sec - 30.0) * sfreq + 1e-12))
    read_stop = min(
        record.values_volts.shape[1],
        int(math.ceil(stop_sec * sfreq - 1e-12)) + 1,
    )
    source_segment = record.values_volts[:, state_reset:read_stop]
    config = CausalEDFConfig(
        output_sfreq_hz=_OUTPUT_SFREQ_HZ,
        highpass_hz=0.5,
        lowpass_hz=45.0,
        butterworth_order=4,
        warmup_sec=30.0,
        apply_car19=True,
    )
    processed, up, down, n_taps, latency_sec = causal_bandpass_resample(
        source_segment,
        source_sfreq_hz=sfreq,
        config=config,
    )
    segment_start_sec = state_reset / sfreq
    latency_samples = latency_sec * _OUTPUT_SFREQ_HZ
    crop_start = _half_up(
        (start_sec - segment_start_sec) * _OUTPUT_SFREQ_HZ + latency_samples
    )
    n_output = _half_up((stop_sec - start_sec) * _OUTPUT_SFREQ_HZ)
    crop_stop = crop_start + n_output
    if crop_start < 0 or crop_stop > processed.shape[1]:
        raise ValueError("C-CAR19 delayed crop lies outside the finite segment")
    ref_values = np.ascontiguousarray(
        processed[:, crop_start:crop_stop], dtype=np.float32
    )
    car_values = np.ascontiguousarray(
        _apply_car(processed)[:, crop_start:crop_stop], dtype=np.float32
    )
    if (
        ref_values.shape != (19, n_output)
        or car_values.shape != (19, n_output)
        or not np.isfinite(ref_values).all()
        or not np.isfinite(car_values).all()
    ):
        raise ValueError("C-CAR19 returned an invalid interval")
    spec = FROZEN_PREPROCESSING_ARM_SPEC_BY_ID["C-CAR19"]
    return ref_values, car_values, {
        "arm_spec_receipt_sha256": spec.receipt_sha256,
        "implementation": "scipy_sosfilt_upfirdn_finite_segment_v1",
        "state_scope": spec.state_scope,
        "state_reset_source_sample": state_reset,
        "read_stop_source_sample": read_stop,
        "real_warmup_sec": start_sec - segment_start_sec,
        "source_sfreq_hz": sfreq,
        "output_sfreq_hz": _OUTPUT_SFREQ_HZ,
        "resample_up": up,
        "resample_down": down,
        "resample_fir_taps": n_taps,
        "resampling_delay_sec": latency_sec,
        "delay_compensated_crop_offset": crop_start,
    }


def _causal_interval(
    record: PhysicalEDFRecord,
    *,
    start_sec: float,
    stop_sec: float,
) -> tuple[np.ndarray, dict[str, object]]:
    _, values, shared = _causal_pre_reference_interval(
        record,
        start_sec=start_sec,
        stop_sec=stop_sec,
    )
    return values, {
        **shared,
        "arm_id": "C-CAR19",
        "reference": "car19",
    }


def _explicit_reference_suffix(raw_name: object) -> str | None:
    text = str(raw_name).strip().upper().replace("_", "-")
    for suffix in ("REF", "LE", "AR", "AVG", "AV", "CAR"):
        if text.endswith(f"-{suffix}"):
            return suffix
    return None


def prepare_causal_reference_pair(
    record: PhysicalEDFRecord,
    *,
    start_sec: float,
    stop_sec: float,
) -> PreparedCausalReferencePair:
    """Create paired causal REF19/CAR19 views without touching any target.

    The sensitivity member is valid only for a uniformly explicit ``-REF``
    source montage, matching the DeepSOZ official input convention.  It is not
    a selectable preprocessing arm and must not be used to refit or choose the
    SOZ localizer on the already-consumed public development cohort.
    """

    references = tuple(_explicit_reference_suffix(name) for name in record.channel_names)
    if references != ("REF",) * len(record.channel_names):
        raise ValueError(
            "C-REF19 sensitivity requires uniformly explicit source -REF channels"
        )
    start = float(start_sec)
    stop = float(stop_sec)
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError("Causal reference-pair interval is invalid")
    ref_values, car_values, shared = _causal_pre_reference_interval(
        record,
        start_sec=start,
        stop_sec=stop,
    )
    ref_receipt = {
        **shared,
        "arm_id": CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
        "paired_primary_arm_id": "C-CAR19",
        "paired_primary_arm_spec_receipt_sha256": shared[
            "arm_spec_receipt_sha256"
        ],
        "reference": "source_uniform_REF_no_rereference",
        "role": CAUSAL_REFERENCE_PAIR_ROLE,
    }
    # C-REF19 is a sensitivity derivative, not a frozen arm.  Do not expose a
    # synthetic arm-spec digest that could be mistaken for selection authority.
    ref_receipt.pop("arm_spec_receipt_sha256")
    car_receipt = {
        **shared,
        "arm_id": "C-CAR19",
        "reference": "car19",
    }
    ref_interval = PreparedArmInterval(
        arm_id=CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
        data_volts=ref_values,
        output_sfreq_hz=_OUTPUT_SFREQ_HZ,
        start_sec=start,
        stop_sec=stop,
        receipt=ref_receipt,
    )
    car_interval = PreparedArmInterval(
        arm_id="C-CAR19",
        data_volts=car_values,
        output_sfreq_hz=_OUTPUT_SFREQ_HZ,
        start_sec=start,
        stop_sec=stop,
        receipt=car_receipt,
    )
    pair_receipt = {
        "schema_version": CAUSAL_REFERENCE_PAIR_SCHEMA,
        "role": CAUSAL_REFERENCE_PAIR_ROLE,
        "primary_arm_id": "C-CAR19",
        "sensitivity_arm_id": CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
        "source_reference": "REF",
        "shared_filter_resample_crop": True,
        "target_values_loaded": False,
        "private_data_loaded": False,
    }
    return PreparedCausalReferencePair(
        ref19=ref_interval,
        car19=car_interval,
        receipt=pair_receipt,
    )


def prepare_arm_interval(
    record: PhysicalEDFRecord,
    *,
    arm_id: str,
    start_sec: float,
    stop_sec: float,
    full_record: tuple[np.ndarray, dict[str, object]] | None = None,
) -> PreparedArmInterval:
    """Return one exact interval from a frozen parity arm."""

    start = float(start_sec)
    stop = float(stop_sec)
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError("Parity interval is invalid")
    if arm_id == "C-CAR19":
        if full_record is not None:
            raise ValueError("C-CAR19 cannot reuse a zero-state full-record result")
        data, receipt = _causal_interval(record, start_sec=start, stop_sec=stop)
    else:
        if arm_id not in {*DEPLOYABLE_PREPROCESSING_ARM_IDS, "O-REF"}:
            raise ValueError("Unknown frozen parity arm")
        if full_record is None:
            full_record = prepare_full_record_arm(record, arm_id=arm_id)
        values, record_receipt = full_record
        begin = _half_up(start * _OUTPUT_SFREQ_HZ)
        n_samples = _half_up((stop - start) * _OUTPUT_SFREQ_HZ)
        end = begin + n_samples
        if begin < 0 or end > values.shape[1]:
            raise ValueError("Parity interval lies outside the full-record arm")
        data = np.ascontiguousarray(values[:, begin:end], dtype=np.float32)
        receipt = {
            **record_receipt,
            "logical_start_sample": begin,
            "logical_stop_sample": end,
        }
    return PreparedArmInterval(
        arm_id=arm_id,
        data_volts=data,
        output_sfreq_hz=_OUTPUT_SFREQ_HZ,
        start_sec=start,
        stop_sec=stop,
        receipt=receipt,
    )


__all__ = [
    "CAUSAL_REFERENCE_PAIR_ROLE",
    "CAUSAL_REFERENCE_PAIR_SCHEMA",
    "CAUSAL_REFERENCE_SENSITIVITY_ARM_ID",
    "OfficialReference23LaBraMEncoder",
    "PhysicalEDFRecord",
    "PreparedArmInterval",
    "PreparedCausalReferencePair",
    "prepare_arm_interval",
    "prepare_causal_reference_pair",
    "prepare_full_record_arm",
    "read_physical_edf",
]
