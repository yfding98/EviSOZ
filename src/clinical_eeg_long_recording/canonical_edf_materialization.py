"""Signal-only EDF materialization into canonical clinical EEG views.

The public loader in this module intentionally has a very small EDF API
surface.  It reads only per-signal labels, physical dimensions, sampling
clocks, sample counts, acquisition-prefilter strings, and physical samples.
It never reads the EDF patient/recording header or EDF+ annotations, and it
has no argument through which a spreadsheet, physician label, clinical text,
or identity field can enter inference.

One immutable physical root is materialized before task-specific views:

``EDF physical samples -> CanonicalEEGRecord``
``CanonicalEEGRecord -> findings_native_morphology (native physical samples)``
``CanonicalEEGRecord -> onset_causal (one-sided FIR with delay receipt)``
``CanonicalEEGRecord -> context_offline (zero-phase context only)``
``each task view -> referential / TCP-20 / CAR / Laplacian views``

``detector_native`` is intentionally not fabricated here: each promoted
detector must publish its own provider-versioned child transform from the same
canonical receipt.  The signal-view contract exposes that role but prevents
its tensor/posterior from becoming clinical Findings evidence.

Missing standard-19 electrodes are represented as *unobserved* in the
canonical receipt.  Finite zero placeholders exist only in derived tensors,
where they are explicitly marked ``observed=False, imputed=True`` and are
therefore ineligible for every evidence family.  They never become part of
the canonical mother-signal tensor or source-signal hash.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.signal import butter, firwin, freqz, lfilter, resample_poly, sosfiltfilt
import torch

from src.soz.geometry import STANDARD_19, TCP_20_EDGES

from .canonical_signal_views import (
    CANONICAL_EDF_ONSET_TRANSFORM_NAME,
    EVIDENCE_FAMILIES,
    ONSET_FIR_CLINICAL_ADMISSION_AUTHORIZATION_SOFTWARE_KEY,
    ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE,
    ONSET_FIR_RESPONSE_AUTHORIZATION_SOFTWARE_KEY,
    ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE,
    build_canonical_signal_receipt,
    build_signal_view_receipt,
    build_transform_spec,
    validate_canonical_signal_receipt,
    validate_signal_view_receipt,
)
from .deterministic_event_findings import deterministic_view_tensor_sha256
from .montage_reference_observability import (
    build_montage_reference_observability_receipt,
    classify_signal_labels,
    require_reference_materialization_authorized,
    validate_montage_reference_observability_receipt,
)


CANONICAL_EDF_MATERIALIZATION_SCHEMA_VERSION = (
    "clinical_eeg_canonical_edf_materialization_v4"
)
CANONICAL_EDF_SOURCE_HEADER_SCHEMA_VERSION = "clinical_eeg_edf_signal_header_v3"
CANONICAL_EDF_PRODUCER_ID = "eeg_only_standard19_canonical_task_views_v4"
CANONICAL_SOURCE_TENSOR_HASH_DOMAIN = "canonical-observed-eeg-volts-float32-le-v1"
ONSET_FIR_RESPONSE_QUALIFICATION_SCHEMA_VERSION = (
    "clinical_eeg_onset_fir_response_qualification_v1"
)
ONSET_FIR_DESIGN_SELECTION_SCHEMA_VERSION = "clinical_eeg_onset_fir_design_selection_v1"
ONSET_FIR_CLINICAL_ADMISSION_QUALIFICATION_SCHEMA_VERSION = (
    "clinical_eeg_onset_fir_clinical_admission_qualification_v1"
)
ONSET_FIR_AUTO_SELECTION_POLICY_ID = "hamming_response_qualified_half_support_cycles_v1"
# For a Type-I linear-phase FIR, ``half_support_cycles / highpass_hz`` is
# both the group delay and half of the causal impulse span in seconds.  The
# frozen initial value gives a 1.5 s group delay for the default 0.5 Hz
# high-pass.  It clears the existing -20 dB DC gate with margin at
# 200/250/256/500/512 Hz.  Later candidates are a fail-closed escape hatch for
# another admissible clock/band; the response receipt, rather than this
# heuristic, remains the target-response claim authority.  The separate
# clinical-admission receipt remains fail-closed until every required gate is
# qualified.
_ONSET_FIR_AUTO_INITIAL_HALF_SUPPORT_CYCLES = 0.75
_ONSET_FIR_AUTO_STEP_HALF_SUPPORT_CYCLES = 0.125
_ONSET_FIR_AUTO_MAXIMUM_HALF_SUPPORT_CYCLES = 1.5

EEG_ONLY_SCOPE_RECEIPT: dict[str, object] = {
    "eeg_samples_used": True,
    "edf_signal_header_used": True,
    "edf_patient_or_recording_header_api_called": False,
    "edf_annotation_api_called": False,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "identity_fields_used": False,
    "video_used": False,
    "sleep_or_activation_labels_used": False,
}

_UNIT_TO_CANONICAL = {
    "v": ("V", 1.0),
    "mv": ("mV", 1e-3),
    "uv": ("uV", 1e-6),
}
_REFERENCE_SUFFIXES = ("REF", "LE", "AR", "AVG", "AV", "CAR")
_LAPLACIAN_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "FP1": ("F7", "F3", "FZ"),
    "FP2": ("FZ", "F4", "F8"),
    "F7": ("FP1", "F3", "T7"),
    "F3": ("FP1", "F7", "FZ", "C3"),
    "FZ": ("FP1", "FP2", "F3", "F4", "CZ"),
    "F4": ("FP2", "FZ", "F8", "C4"),
    "F8": ("FP2", "F4", "T8"),
    "T7": ("F7", "C3", "P7"),
    "C3": ("F3", "T7", "CZ", "P3"),
    "CZ": ("FZ", "C3", "C4", "PZ"),
    "C4": ("F4", "CZ", "T8", "P4"),
    "T8": ("F8", "C4", "P8"),
    "P7": ("T7", "P3", "O1"),
    "P3": ("C3", "P7", "PZ", "O1"),
    "PZ": ("CZ", "P3", "P4", "O1", "O2"),
    "P4": ("C4", "PZ", "P8", "O2"),
    "P8": ("T8", "P4", "O2"),
    "O1": ("P7", "P3", "PZ"),
    "O2": ("PZ", "P4", "P8"),
}
_HP_PATTERN = re.compile(
    r"(?:^|[\s;,])(?:HP|HIGH\s*PASS)\s*[:=]?\s*(DC|[-+]?[0-9]*\.?[0-9]+)",
    flags=re.IGNORECASE,
)
_LP_PATTERN = re.compile(
    r"(?:^|[\s;,])(?:LP|LOW\s*PASS)\s*[:=]?\s*([-+]?[0-9]*\.?[0-9]+)",
    flags=re.IGNORECASE,
)
_STANDARD_DATE_OR_TIME = re.compile(rb"^[0-9]{2}[.:][0-9]{2}[.:][0-9]{2}$")
_NATIVE_READER_POLICY = "pyedflib_physical_signal_methods_v1"
_COMPACT_READER_POLICY = "compact_edf_signal_only_parser_v1"
_STANDARD_SIGNAL_ONLY_READER_POLICY = "standard_edf_signal_only_parser_v1"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_edf(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ValueError("canonical EDF source must not be a symbolic link")
    resolved = source.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix.lower() != ".edf":
        raise ValueError("canonical source must be a regular EDF file")
    return resolved


def _ascii_field(value: bytes, *, context: str, integer: bool = False) -> float | int:
    try:
        text = value.decode("ascii").strip()
        return int(text) if integer else float(text)
    except (UnicodeDecodeError, ValueError, OverflowError) as exc:
        raise ValueError(f"signal-only EDF {context} is invalid") from exc


def _split_signal_fields(
    header: bytes,
    *,
    count: int,
    width: int,
    offset: int,
) -> tuple[list[str], int]:
    stop = offset + count * width
    if stop > len(header):
        raise ValueError("signal-only EDF signal header is truncated")
    values = [
        header[offset + index * width : offset + (index + 1) * width]
        .decode("latin-1")
        .strip()
        for index in range(count)
    ]
    return values, stop


class _SignalOnlyEDFReader:
    """Read standard or legacy-compact EDF signals without TAL parsing.

    The standard fallback is used only when pyedflib rejects a malformed date
    field.  It skips patient/recording/date bytes, validates the ordinary
    256-byte fixed-header framing, and reads signal headers plus interleaved
    int16 samples.  The compact layout supports eleven legacy private files
    with an actual 176-byte fixed header but an otherwise ordinary EDF signal
    layout.  Neither path decodes annotation/TAL payloads.  EDF+D is rejected
    because concatenating discontinuous data records would fabricate a clock.
    """

    def __init__(self, path: str, *, layout: str) -> None:
        self._path = Path(path)
        if layout == "standard":
            context = "standard"
            fixed_bytes = 256
            date_slice = slice(168, 176)
            time_slice = slice(176, 184)
            header_bytes_slice = slice(184, 192)
            reserved_slice = slice(192, 236)
            record_count_slice = slice(236, 244)
            duration_slice = slice(244, 252)
            signal_count_slice = slice(252, 256)
            require_standard_date_marker = False
            self.canonical_reader_policy = _STANDARD_SIGNAL_ONLY_READER_POLICY
        elif layout == "compact":
            context = "compact"
            fixed_bytes = 176
            date_slice = slice(88, 96)
            time_slice = slice(96, 104)
            header_bytes_slice = slice(104, 112)
            reserved_slice = slice(112, 156)
            record_count_slice = slice(156, 164)
            duration_slice = slice(164, 172)
            signal_count_slice = slice(172, 176)
            require_standard_date_marker = True
            self.canonical_reader_policy = _COMPACT_READER_POLICY
        else:
            raise ValueError("signal-only EDF layout is unsupported")
        self._context = context
        with self._path.open("rb") as stream:
            fixed = stream.read(fixed_bytes)
            if len(fixed) != fixed_bytes or fixed[:1] != b"0":
                raise ValueError(f"{context} EDF fixed header is invalid")
            if require_standard_date_marker and (
                not _STANDARD_DATE_OR_TIME.fullmatch(fixed[date_slice])
                or not _STANDARD_DATE_OR_TIME.fullmatch(fixed[time_slice])
            ):
                raise ValueError(f"{context} EDF date/time marker is absent")
            reserved = fixed[reserved_slice].decode("latin-1").strip().upper()
            if reserved.startswith("EDF+D"):
                raise ValueError(
                    "discontinuous EDF+D is unsupported by the signal-only clock "
                    "contract; no annotation/TAL fallback was attempted"
                )
            declared_header_bytes = int(
                _ascii_field(
                    fixed[header_bytes_slice],
                    context="header byte count",
                    integer=True,
                )
            )
            data_record_duration = float(
                _ascii_field(fixed[duration_slice], context="data-record duration")
            )
            signal_count = int(
                _ascii_field(
                    fixed[signal_count_slice], context="signal count", integer=True
                )
            )
            if signal_count < 1 or signal_count > 1024 or data_record_duration <= 0:
                raise ValueError(f"{context} EDF clock or signal count is invalid")
            expected_standard_header = 256 + signal_count * 256
            if declared_header_bytes != expected_standard_header:
                raise ValueError(f"{context} EDF declared header size is unsupported")
            signal_header = stream.read(signal_count * 256)
            if len(signal_header) != signal_count * 256:
                raise ValueError(f"{context} EDF signal header is truncated")

        offset = 0
        labels, offset = _split_signal_fields(
            signal_header, count=signal_count, width=16, offset=offset
        )
        _, offset = _split_signal_fields(
            signal_header, count=signal_count, width=80, offset=offset
        )
        dimensions, offset = _split_signal_fields(
            signal_header, count=signal_count, width=8, offset=offset
        )
        physical_min_text, offset = _split_signal_fields(
            signal_header, count=signal_count, width=8, offset=offset
        )
        physical_max_text, offset = _split_signal_fields(
            signal_header, count=signal_count, width=8, offset=offset
        )
        digital_min_text, offset = _split_signal_fields(
            signal_header, count=signal_count, width=8, offset=offset
        )
        digital_max_text, offset = _split_signal_fields(
            signal_header, count=signal_count, width=8, offset=offset
        )
        prefilters, offset = _split_signal_fields(
            signal_header, count=signal_count, width=80, offset=offset
        )
        samples_text, offset = _split_signal_fields(
            signal_header, count=signal_count, width=8, offset=offset
        )
        _, offset = _split_signal_fields(
            signal_header, count=signal_count, width=32, offset=offset
        )
        if offset != len(signal_header):
            raise ValueError(f"{context} EDF signal header framing drifted")

        try:
            physical_min = np.asarray(physical_min_text, dtype=np.float64)
            physical_max = np.asarray(physical_max_text, dtype=np.float64)
            digital_min = np.asarray(digital_min_text, dtype=np.float64)
            digital_max = np.asarray(digital_max_text, dtype=np.float64)
            samples_per_record = np.asarray(samples_text, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{context} EDF signal calibration is invalid") from exc
        if (
            not np.isfinite(physical_min).all()
            or not np.isfinite(physical_max).all()
            or np.any(samples_per_record <= 0)
        ):
            raise ValueError(f"{context} EDF signal calibration is invalid")
        data_start = fixed_bytes + signal_count * 256
        record_bytes = int(2 * int(np.sum(samples_per_record)))
        data_bytes = self._path.stat().st_size - data_start
        if data_bytes <= 0 or data_bytes % record_bytes:
            raise ValueError(f"{context} EDF data-record framing is invalid")
        record_count = data_bytes // record_bytes
        declared_records = int(
            _ascii_field(
                fixed[record_count_slice], context="data-record count", integer=True
            )
        )
        if declared_records not in {-1, record_count}:
            raise ValueError(
                f"{context} EDF data-record count disagrees with file size"
            )

        self._labels = tuple(labels)
        self._dimensions = tuple(dimensions)
        self._prefilters = tuple(prefilters)
        self._physical_min = physical_min
        self._physical_max = physical_max
        self._digital_min = digital_min
        self._digital_max = digital_max
        self._samples_per_record = samples_per_record
        self._duration = data_record_duration
        self._record_count = int(record_count)
        self._data_start = int(data_start)
        self._record_bytes = record_bytes
        self._signal_offsets = (
            np.concatenate(
                [np.asarray([0], dtype=np.int64), np.cumsum(samples_per_record[:-1])]
            )
            * 2
        )
        self._closed = False

    def getSignalLabels(self) -> list[str]:
        return list(self._labels)

    def getSampleFrequency(self, index: int) -> float:
        return float(self._samples_per_record[index]) / self._duration

    def getPhysicalDimension(self, index: int) -> str:
        return self._dimensions[index]

    def getPrefilter(self, index: int) -> str:
        return self._prefilters[index]

    def getPhysicalMinimum(self, index: int) -> float:
        return float(self._physical_min[index])

    def getPhysicalMaximum(self, index: int) -> float:
        return float(self._physical_max[index])

    def getDigitalMinimum(self, index: int) -> float:
        return float(self._digital_min[index])

    def getDigitalMaximum(self, index: int) -> float:
        return float(self._digital_max[index])

    def getNSamples(self) -> np.ndarray:
        return self._samples_per_record * self._record_count

    def readSignal(self, index: int, start: int, count: int) -> np.ndarray:
        if self._closed:
            raise ValueError(f"{self._context} EDF reader is closed")
        samples_per_record = int(self._samples_per_record[index])
        total = samples_per_record * self._record_count
        if start < 0 or count < 0 or start + count > total:
            raise ValueError(
                f"{self._context} EDF signal interval lies outside the recording"
            )
        if count == 0:
            return np.empty((0,), dtype=np.float64)
        if (
            self._physical_max[index] == self._physical_min[index]
            or self._digital_max[index] == self._digital_min[index]
        ):
            raise ValueError(
                f"{self._context} EDF requested signal calibration is invalid"
            )
        first_record = start // samples_per_record
        last_record = (start + count - 1) // samples_per_record
        chunks: list[np.ndarray] = []
        with self._path.open("rb") as stream:
            for record_index in range(first_record, last_record + 1):
                stream.seek(
                    self._data_start
                    + record_index * self._record_bytes
                    + int(self._signal_offsets[index])
                )
                raw = stream.read(samples_per_record * 2)
                if len(raw) != samples_per_record * 2:
                    raise ValueError(f"{self._context} EDF signal payload is truncated")
                chunks.append(np.frombuffer(raw, dtype="<i2").astype(np.float64))
        digital = np.concatenate(chunks)
        left = start - first_record * samples_per_record
        digital = digital[left : left + count]
        physical = (digital - self._digital_min[index]) * (
            self._physical_max[index] - self._physical_min[index]
        ) / (self._digital_max[index] - self._digital_min[index]) + self._physical_min[
            index
        ]
        return physical.astype(np.float64, copy=False)

    def close(self) -> None:
        self._closed = True


class _StandardEDFSignalOnlyReader(_SignalOnlyEDFReader):
    def __init__(self, path: str) -> None:
        super().__init__(path, layout="standard")


class _CompactEDFSignalOnlyReader(_SignalOnlyEDFReader):
    def __init__(self, path: str) -> None:
        super().__init__(path, layout="compact")


def _reader_factory(path: str) -> object:
    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover - deployment dependency gate
        raise RuntimeError("pyedflib is required for canonical EDF loading") from exc
    try:
        reader = pyedflib.EdfReader(path)
    except OSError as exc:
        message = str(exc)
        if "startdate is incorrect" in message:
            try:
                return _StandardEDFSignalOnlyReader(path)
            except ValueError as standard_error:
                if "discontinuous EDF+D" in str(standard_error):
                    raise standard_error from None
                try:
                    return _CompactEDFSignalOnlyReader(path)
                except ValueError as compact_error:
                    if "discontinuous EDF+D" in str(compact_error):
                        raise compact_error from None
                    raise ValueError(
                        "EDF with an invalid start date is not a supported "
                        "standard or legacy-compact signal-only container"
                    ) from None
        if "file is discontinuous" in message:
            raise ValueError(
                "discontinuous EDF+D is unsupported by the signal-only clock "
                "contract; no annotation/TAL fallback was attempted"
            ) from None
        raise ValueError(
            "EDF container is unreadable by the signal-only reader"
        ) from None
    try:
        setattr(reader, "canonical_reader_policy", _NATIVE_READER_POLICY)
    except (AttributeError, TypeError):
        pass
    return reader


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _rational_rate(value: object, *, context: str) -> tuple[int, int]:
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} is not a valid sampling rate") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{context} must be finite and positive")
    fraction = Fraction(str(rate)).limit_denominator(1_000_000)
    if abs(float(fraction) - rate) > 1e-10:
        raise ValueError(f"{context} cannot be represented on a reproducible clock")
    return fraction.numerator, fraction.denominator


def _rate_hz(rate: tuple[int, int]) -> float:
    return float(rate[0]) / float(rate[1])


def _unit_and_scale(value: object) -> tuple[str, float, str]:
    raw = str(value).strip()
    normalized = raw.lower().replace("µ", "u").replace("μ", "u")
    if normalized not in _UNIT_TO_CANONICAL:
        raise ValueError(f"unsupported EDF physical unit: {value!r}")
    canonical, scale = _UNIT_TO_CANONICAL[normalized]
    return canonical, scale, raw


def _reference_from_signal_label(label: str) -> str:
    text = str(label).strip().upper().replace("_", "-")
    for suffix in _REFERENCE_SUFFIXES:
        if text.endswith(f"-{suffix}"):
            return suffix
    return "EDF_HEADER_UNSPECIFIED"


def _acquisition_bandwidth(
    prefilter: object, nyquist_hz: float
) -> tuple[float | None, float | None]:
    text = str(prefilter or "").strip()
    highpass: float | None = None
    lowpass: float | None = None
    match = _HP_PATTERN.search(text)
    if match:
        highpass = 0.0 if match.group(1).upper() == "DC" else float(match.group(1))
    match = _LP_PATTERN.search(text)
    if match:
        lowpass = float(match.group(1))
    if highpass is not None and (
        not math.isfinite(highpass) or highpass < 0 or highpass >= nyquist_hz
    ):
        highpass = None
    if lowpass is not None and (not math.isfinite(lowpass) or lowpass <= 0):
        lowpass = None
    if lowpass is not None:
        lowpass = min(float(lowpass), float(nyquist_hz))
    if highpass is not None and lowpass is not None and highpass >= lowpass:
        return None, None
    return highpass, lowpass


def _edf_signal_calibration(
    reader: object,
    *,
    signal_index: int,
    scale_to_volts: float,
) -> dict[str, float | str | None]:
    """Read signal calibration rails without consulting any EEG sample.

    EDF physical/digital extrema are acquisition calibration fields, not
    extrema estimated from the full recording.  When a constrained test or
    legacy reader does not expose all four methods, clipping is *not*
    asserted from record-wide min/max; a local plateau candidate is used by
    :func:`_quality_primitives` and is labelled as generic signal quality.
    """

    method_names = (
        "getPhysicalMinimum",
        "getPhysicalMaximum",
        "getDigitalMinimum",
        "getDigitalMaximum",
    )
    if not all(callable(getattr(reader, name, None)) for name in method_names):
        return {
            "physical_minimum_raw": None,
            "physical_maximum_raw": None,
            "digital_minimum": None,
            "digital_maximum": None,
            "adc_rail_minimum_volts": None,
            "adc_rail_maximum_volts": None,
            "adc_lsb_volts": None,
            "clipping_qc_source": "header_rails_unavailable_local_plateau_only_v1",
        }
    try:
        physical_minimum = float(reader.getPhysicalMinimum(signal_index))
        physical_maximum = float(reader.getPhysicalMaximum(signal_index))
        digital_minimum = float(reader.getDigitalMinimum(signal_index))
        digital_maximum = float(reader.getDigitalMaximum(signal_index))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("EDF signal calibration rails are invalid") from exc
    values = (
        physical_minimum,
        physical_maximum,
        digital_minimum,
        digital_maximum,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or physical_maximum <= physical_minimum
        or digital_maximum <= digital_minimum
    ):
        raise ValueError("EDF signal calibration rails are empty or non-finite")
    rail_minimum_volts = physical_minimum * float(scale_to_volts)
    rail_maximum_volts = physical_maximum * float(scale_to_volts)
    lsb_volts = (
        (physical_maximum - physical_minimum)
        * float(scale_to_volts)
        / (digital_maximum - digital_minimum)
    )
    if not math.isfinite(lsb_volts) or lsb_volts <= 0:
        raise ValueError("EDF signal calibration LSB is invalid")
    return {
        "physical_minimum_raw": physical_minimum,
        "physical_maximum_raw": physical_maximum,
        "digital_minimum": digital_minimum,
        "digital_maximum": digital_maximum,
        "adc_rail_minimum_volts": rail_minimum_volts,
        "adc_rail_maximum_volts": rail_maximum_volts,
        "adc_lsb_volts": lsb_volts,
        "clipping_qc_source": "edf_signal_header_calibration_rails_v1",
    }


def _true_runs(mask: np.ndarray, *, minimum_length: int) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("QC run mask must be one-dimensional")
    if minimum_length < 1:
        raise ValueError("QC minimum run length must be positive")
    padded = np.concatenate([np.asarray([False]), values, np.asarray([False])]).astype(
        np.int8
    )
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if int(stop) - int(start) >= minimum_length
    ]


def _quality_primitives(
    signal_volts: torch.Tensor,
    channel_ids: Sequence[str],
    *,
    sample_rate: tuple[int, int],
    flatline_run_seconds: float,
    clipping_run_seconds: float,
    tolerance_volts: float,
    clipping_calibration_by_channel: Mapping[str, Mapping[str, float | str | None]],
) -> list[dict[str, object]]:
    """Build sample-local QC without ever estimating clipping rails globally."""

    values = signal_volts.detach().cpu().to(torch.float32).numpy()
    sfreq = _rate_hz(sample_rate)
    flat_samples = max(2, int(math.ceil(flatline_run_seconds * sfreq)))
    clipping_samples = max(2, int(math.ceil(clipping_run_seconds * sfreq)))
    rows: list[dict[str, object]] = []
    for channel_id, signal in zip(channel_ids, values):
        flat_differences = np.abs(np.diff(signal.astype(np.float64))) <= tolerance_volts
        # A run of N-1 flat differences binds N samples.
        for start_diff, stop_diff in _true_runs(
            flat_differences,
            minimum_length=flat_samples - 1,
        ):
            start_sample = start_diff
            stop_sample = stop_diff + 1
            rows.append(
                {
                    "quality_id": (
                        f"QC-FLAT-{channel_id}-{start_sample:012d}-{stop_sample:012d}"
                    ),
                    "channel_ids": [channel_id],
                    "start_recording_seconds": start_sample / sfreq,
                    "stop_recording_seconds": stop_sample / sfreq,
                    "kind": "flat",
                    "severity": "unusable",
                    "disabled_evidence_families": list(EVIDENCE_FAMILIES),
                }
            )

        calibration = clipping_calibration_by_channel[channel_id]
        if calibration["clipping_qc_source"] == (
            "edf_signal_header_calibration_rails_v1"
        ):
            rail_tolerance = max(
                float(tolerance_volts),
                0.51 * float(calibration["adc_lsb_volts"]),
            )
            rail_minimum = float(calibration["adc_rail_minimum_volts"])
            rail_maximum = float(calibration["adc_rail_maximum_volts"])
            at_rail = np.isclose(
                signal,
                rail_minimum,
                rtol=0.0,
                atol=rail_tolerance,
            ) | np.isclose(
                signal,
                rail_maximum,
                rtol=0.0,
                atol=rail_tolerance,
            )
            for start_sample, stop_sample in _true_runs(
                at_rail,
                minimum_length=clipping_samples,
            ):
                rows.append(
                    {
                        "quality_id": (
                            "QC-CLIP-EDFRAIL-"
                            f"{channel_id}-{start_sample:012d}-{stop_sample:012d}"
                        ),
                        "channel_ids": [channel_id],
                        "start_recording_seconds": start_sample / sfreq,
                        "stop_recording_seconds": stop_sample / sfreq,
                        "kind": "clipping",
                        "severity": "unusable",
                        "disabled_evidence_families": list(EVIDENCE_FAMILIES),
                    }
                )
        else:
            # Without header rails, a repeated-value segment is unsafe but it
            # is not entitled to the clinical/digital-saturation label
            # ``clipping``.  Longer runs are already represented as ``flat``.
            plateau_differences = (
                np.abs(np.diff(signal.astype(np.float64))) <= tolerance_volts
            )
            for start_diff, stop_diff in _true_runs(
                plateau_differences,
                minimum_length=clipping_samples - 1,
            ):
                start_sample = start_diff
                stop_sample = stop_diff + 1
                if stop_sample - start_sample >= flat_samples:
                    continue
                rows.append(
                    {
                        "quality_id": (
                            "QC-PLATEAU-NORAIL-"
                            f"{channel_id}-{start_sample:012d}-{stop_sample:012d}"
                        ),
                        "channel_ids": [channel_id],
                        "start_recording_seconds": start_sample / sfreq,
                        "stop_recording_seconds": stop_sample / sfreq,
                        "kind": "other_signal_quality",
                        "severity": "unusable",
                        "disabled_evidence_families": list(EVIDENCE_FAMILIES),
                    }
                )
    rows.sort(
        key=lambda row: (
            float(row["start_recording_seconds"]),
            float(row["stop_recording_seconds"]),
            str(row["channel_ids"][0]),
            str(row["kind"]),
        )
    )
    return rows


def _source_tensor_sha256(
    tensor: torch.Tensor,
    *,
    channel_ids: Sequence[str],
) -> str:
    values = tensor.detach().cpu().to(torch.float32).contiguous()
    if values.ndim != 2 or values.shape[0] != len(channel_ids):
        raise ValueError("canonical source tensor shape and channel order disagree")
    if values.shape[1] < 1 or not torch.isfinite(values).all():
        raise ValueError("canonical source tensor must be finite and non-empty")
    header = {
        "domain": CANONICAL_SOURCE_TENSOR_HASH_DOMAIN,
        "dtype": "float32-le",
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "channel_ids": [str(item) for item in channel_ids],
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(values.numpy().astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _source_signal_sha256(
    *,
    source_header_core: Mapping[str, object],
    source_tensor_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "domain": "physical-edf-signal-plus-signal-header-v1",
            "source_header_core": source_header_core,
            "source_tensor_sha256": source_tensor_sha256,
        }
    )


@dataclass(frozen=True)
class CanonicalEDFConfig:
    """Versioned clinical-view preprocessing and signal-only QC policy."""

    output_sampling_rate_hz: float | None = None
    findings_highpass_hz: float = 0.5
    findings_lowpass_hz: float = 45.0
    onset_highpass_hz: float = 0.5
    onset_lowpass_hz: float = 45.0
    # ``None`` selects a response-qualified, sampling-rate-aware Type-I FIR.
    # Supplying an odd integer preserves the historical explicit fixed-tap
    # path, including fail-closed 101-tap receipts in archived materializations.
    onset_fir_numtaps: int | None = None
    onset_fir_numtaps_policy: str = ONSET_FIR_AUTO_SELECTION_POLICY_ID
    butterworth_order: int = 4
    edge_guard_seconds: float = 2.0
    cache_tile_seconds: float = 30.0
    flatline_run_seconds: float = 2.0
    clipping_run_seconds: float = 0.5
    qc_tolerance_volts: float = 1e-12
    minimum_observed_standard19: int = 1

    def __post_init__(self) -> None:
        numeric = (
            self.findings_highpass_hz,
            self.findings_lowpass_hz,
            self.onset_highpass_hz,
            self.onset_lowpass_hz,
            self.edge_guard_seconds,
            self.cache_tile_seconds,
            self.flatline_run_seconds,
            self.clipping_run_seconds,
        )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0 for value in numeric
        ):
            raise ValueError("canonical EDF frequencies and durations must be positive")
        if self.findings_highpass_hz >= self.findings_lowpass_hz:
            raise ValueError("findings highpass must be below lowpass")
        if self.onset_highpass_hz >= self.onset_lowpass_hz:
            raise ValueError("onset highpass must be below lowpass")
        if self.onset_fir_numtaps is not None and (
            isinstance(self.onset_fir_numtaps, bool)
            or not isinstance(self.onset_fir_numtaps, int)
            or self.onset_fir_numtaps < 3
            or self.onset_fir_numtaps % 2 != 1
        ):
            raise ValueError("onset_fir_numtaps must be null or an odd integer >= 3")
        if self.onset_fir_numtaps_policy != ONSET_FIR_AUTO_SELECTION_POLICY_ID:
            raise ValueError("onset_fir_numtaps_policy is unsupported")
        if self.output_sampling_rate_hz is not None and (
            not math.isfinite(float(self.output_sampling_rate_hz))
            or float(self.output_sampling_rate_hz) <= 0
        ):
            raise ValueError("output_sampling_rate_hz must be positive when supplied")
        if self.butterworth_order < 1:
            raise ValueError("butterworth_order must be positive")
        if self.qc_tolerance_volts < 0:
            raise ValueError("qc_tolerance_volts must be non-negative")
        if not 1 <= self.minimum_observed_standard19 <= len(STANDARD_19):
            raise ValueError("minimum_observed_standard19 must lie in [1,19]")


@lru_cache(maxsize=128)
def _cached_onset_causal_fir_response_json(
    *,
    sampling_rate_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
    numtaps: int,
    response_grid_points: int = 262_145,
) -> str:
    """Measure, qualify and content-bind the actual causal FIR response.

    ``firwin`` cutoff arguments are design parameters, not proof of a usable
    passband.  In particular, a fixed 101-tap 0.5-Hz high-pass can be nearly
    transparent at DC when the sampling clock increases.  This receipt keeps
    the design targets separate from a measured -3 dB support and denies the
    nominal target-band claim unless frozen cutoff/DC criteria pass.
    """

    rate = _rate_hz(
        _rational_rate(sampling_rate_hz, context="onset FIR qualification rate")
    )
    highpass = float(highpass_hz)
    lowpass = float(lowpass_hz)
    if (
        not math.isfinite(highpass)
        or not math.isfinite(lowpass)
        or highpass <= 0
        or lowpass <= highpass
        or lowpass >= 0.5 * rate
    ):
        raise ValueError("onset FIR qualification design band is invalid")
    if isinstance(numtaps, bool) or not isinstance(numtaps, int):
        raise TypeError("onset FIR qualification numtaps must be an integer")
    if numtaps < 3 or numtaps % 2 != 1:
        raise ValueError("onset FIR qualification numtaps must be odd and >=3")
    if (
        isinstance(response_grid_points, bool)
        or not isinstance(response_grid_points, int)
        or response_grid_points < 16_385
    ):
        raise ValueError("onset FIR response grid is too small")

    coefficients = firwin(
        numtaps,
        [highpass, lowpass],
        pass_zero=False,
        window="hamming",
        scale=True,
        fs=rate,
    )
    frequencies, response = freqz(
        coefficients,
        worN=response_grid_points,
        whole=False,
        include_nyquist=True,
        fs=rate,
    )
    magnitude = np.abs(response)
    peak = float(np.max(magnitude))
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("onset FIR response is empty")

    def _relative_gain_db(frequency_hz: float) -> float:
        sample_index = np.arange(numtaps, dtype=np.float64)
        kernel = np.exp(-2j * np.pi * float(frequency_hz) * sample_index / rate)
        amplitude = float(abs(np.sum(coefficients * kernel)))
        return float(20.0 * math.log10(max(amplitude / peak, 1e-15)))

    minus_3db_amplitude = peak / math.sqrt(2.0)
    above = magnitude >= minus_3db_amplitude
    runs = _true_runs(above, minimum_length=1)
    if not runs:
        raise ValueError("onset FIR has no measured -3 dB support")
    band_center = 0.5 * (highpass + lowpass)
    center_index = int(np.argmin(np.abs(frequencies - band_center)))
    containing = [run for run in runs if run[0] <= center_index < run[1]]
    selected_run = (
        containing[0]
        if containing
        else max(runs, key=lambda run: int(run[1]) - int(run[0]))
    )
    effective_lower = float(frequencies[selected_run[0]])
    effective_upper = float(frequencies[selected_run[1] - 1])
    dc_gain_db = _relative_gain_db(0.0)
    highpass_gain_db = _relative_gain_db(highpass)
    lowpass_gain_db = _relative_gain_db(lowpass)
    cutoff_gain_bounds_db = (-9.0, -3.0)
    dc_attenuation_max_db = -20.0
    qualified = bool(
        dc_gain_db <= dc_attenuation_max_db
        and cutoff_gain_bounds_db[0] <= highpass_gain_db <= cutoff_gain_bounds_db[1]
        and cutoff_gain_bounds_db[0] <= lowpass_gain_db <= cutoff_gain_bounds_db[1]
    )
    group_delay_samples = (numtaps - 1) / 2.0
    body: dict[str, Any] = {
        "schema_version": ONSET_FIR_RESPONSE_QUALIFICATION_SCHEMA_VERSION,
        "design": {
            "family": "linear_phase_fir",
            "window": "hamming",
            "scale": True,
            "sampling_rate_hz": rate,
            "numtaps": numtaps,
            "target_highpass_hz": highpass,
            "target_lowpass_hz": lowpass,
        },
        "measurement": {
            "method": "scipy_freqz_dense_grid_plus_exact_frequency_dft_v1",
            "response_grid_points": response_grid_points,
            "response_grid_resolution_hz": float(frequencies[1] - frequencies[0]),
            "relative_dc_gain_db": dc_gain_db,
            "relative_target_highpass_gain_db": highpass_gain_db,
            "relative_target_lowpass_gain_db": lowpass_gain_db,
            "measured_minus_3db_bandwidth_hz": [
                effective_lower,
                effective_upper,
            ],
            "group_delay_samples": group_delay_samples,
            "group_delay_seconds": group_delay_samples / rate,
        },
        "qualification_policy": {
            "maximum_relative_dc_gain_db": dc_attenuation_max_db,
            "target_cutoff_relative_gain_bounds_db": list(cutoff_gain_bounds_db),
            "nominal_design_cutoffs_are_not_effective_bandwidth": True,
        },
        "target_response_qualified": qualified,
        "target_band_claim_authorized": qualified,
        "reportable_bandwidth_policy": (
            "measured_minus_3db_only_nominal_target_denied_when_unqualified_v1"
        ),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def qualify_onset_causal_fir_response(
    *,
    sampling_rate_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
    numtaps: int,
    response_grid_points: int = 262_145,
) -> dict[str, Any]:
    """Return an independent copy of the cached FIR qualification receipt."""

    return json.loads(
        _cached_onset_causal_fir_response_json(
            sampling_rate_hz=sampling_rate_hz,
            highpass_hz=highpass_hz,
            lowpass_hz=lowpass_hz,
            numtaps=numtaps,
            response_grid_points=response_grid_points,
        )
    )


def _onset_fir_coefficient_receipt(
    *,
    sampling_rate_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
    numtaps: int,
) -> dict[str, Any]:
    """Bind the actual float64 firwin coefficients and non-zero support."""

    coefficients = firwin(
        int(numtaps),
        [float(highpass_hz), float(lowpass_hz)],
        pass_zero=False,
        window="hamming",
        scale=True,
        fs=float(sampling_rate_hz),
    ).astype("<f8", copy=False)
    nonzero = np.flatnonzero(coefficients != 0.0)
    if nonzero.size < 1 or not np.isfinite(coefficients).all():
        raise ValueError("onset FIR coefficients have no finite non-zero support")
    first = int(nonzero[0])
    last = int(nonzero[-1])
    header = {
        "domain": "clinical-eeg-onset-fir-coefficients-float64-le-v1",
        "sampling_rate_hz": float(sampling_rate_hz),
        "highpass_hz": float(highpass_hz),
        "lowpass_hz": float(lowpass_hz),
        "numtaps": int(numtaps),
        "dtype": "float64-le",
        "shape": [int(numtaps)],
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(coefficients.tobytes(order="C"))
    return {
        "coefficient_sha256": digest.hexdigest(),
        "nonzero_coefficient_index_interval": [first, last + 1],
        "raw_nonzero_support_offset_samples": [-last, -first],
        "raw_nonzero_support_span_samples": last - first,
        "maximum_nonzero_lag_samples": last,
    }


def _type_i_numtaps_for_half_support_cycles(
    *,
    sampling_rate_hz: float,
    highpass_hz: float,
    half_support_cycles: float,
) -> int:
    """Return odd taps whose even order meets the requested physical span."""

    target_order = int(
        math.ceil(
            2.0
            * float(half_support_cycles)
            * float(sampling_rate_hz)
            / float(highpass_hz)
        )
    )
    if target_order % 2:
        target_order += 1
    return target_order + 1


def select_onset_causal_fir_design(
    *,
    sampling_rate_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
    requested_numtaps: int | None,
    selection_policy_id: str = ONSET_FIR_AUTO_SELECTION_POLICY_ID,
) -> dict[str, Any]:
    """Select and content-bind the causal FIR without trusting a heuristic.

    An explicit odd tap count preserves the historical path and may remain
    unqualified.  The automatic path evaluates a frozen sequence of physical
    half-supports and accepts only the first candidate whose independently
    replayable response receipt passes.  No candidate is silently shortened
    to fit a recording; an insufficiently long recording fails later instead
    of weakening the frequency-response contract.
    """

    rate = _rate_hz(
        _rational_rate(sampling_rate_hz, context="onset FIR selection rate")
    )
    highpass = float(highpass_hz)
    lowpass = float(lowpass_hz)
    if selection_policy_id != ONSET_FIR_AUTO_SELECTION_POLICY_ID:
        raise ValueError("onset FIR selection policy is unsupported")
    if (
        not math.isfinite(highpass)
        or not math.isfinite(lowpass)
        or highpass <= 0
        or lowpass <= highpass
        or lowpass >= 0.5 * rate
    ):
        raise ValueError("onset FIR selection band is invalid")
    if requested_numtaps is not None and (
        isinstance(requested_numtaps, bool)
        or not isinstance(requested_numtaps, int)
        or requested_numtaps < 3
        or requested_numtaps % 2 != 1
    ):
        raise ValueError("requested onset FIR taps must be null or odd and >=3")

    if requested_numtaps is None:
        candidate_cycles: list[float] = []
        value = _ONSET_FIR_AUTO_INITIAL_HALF_SUPPORT_CYCLES
        while value <= _ONSET_FIR_AUTO_MAXIMUM_HALF_SUPPORT_CYCLES + 1e-12:
            candidate_cycles.append(float(value))
            value += _ONSET_FIR_AUTO_STEP_HALF_SUPPORT_CYCLES
        candidate_numtaps = []
        for cycles in candidate_cycles:
            numtaps = _type_i_numtaps_for_half_support_cycles(
                sampling_rate_hz=rate,
                highpass_hz=highpass,
                half_support_cycles=cycles,
            )
            if numtaps not in candidate_numtaps:
                candidate_numtaps.append(numtaps)
        selection_mode = "automatic_response_qualified"
    else:
        candidate_cycles = []
        candidate_numtaps = [int(requested_numtaps)]
        selection_mode = "explicit_fixed_numtaps_legacy_compatible"

    evaluated: list[dict[str, Any]] = []
    selected_numtaps: int | None = None
    selected_qualification: dict[str, Any] | None = None
    for numtaps in candidate_numtaps:
        qualification = qualify_onset_causal_fir_response(
            sampling_rate_hz=rate,
            highpass_hz=highpass,
            lowpass_hz=lowpass,
            numtaps=numtaps,
        )
        evaluated.append(
            {
                "numtaps": numtaps,
                "target_response_qualified": bool(
                    qualification["target_response_qualified"]
                ),
                "qualification_receipt_sha256": qualification["receipt_sha256"],
            }
        )
        if requested_numtaps is not None or qualification["target_response_qualified"]:
            selected_numtaps = numtaps
            selected_qualification = qualification
            break
    if selected_numtaps is None or selected_qualification is None:
        raise ValueError(
            "automatic onset FIR policy found no response-qualified candidate"
        )

    order = selected_numtaps - 1
    group_delay_samples = order / 2.0
    coefficient_receipt = _onset_fir_coefficient_receipt(
        sampling_rate_hz=rate,
        highpass_hz=highpass,
        lowpass_hz=lowpass,
        numtaps=selected_numtaps,
    )
    if coefficient_receipt["maximum_nonzero_lag_samples"] != order:
        raise ValueError(
            "selected onset FIR has a zero endpoint and unsupported shortened support"
        )
    body: dict[str, Any] = {
        "schema_version": ONSET_FIR_DESIGN_SELECTION_SCHEMA_VERSION,
        "selection_policy_id": selection_policy_id,
        "selection_mode": selection_mode,
        "requested_numtaps": requested_numtaps,
        "target": {
            "sampling_rate_hz": rate,
            "highpass_hz": highpass,
            "lowpass_hz": lowpass,
        },
        "automatic_policy": {
            "initial_half_support_cycles_at_highpass": (
                _ONSET_FIR_AUTO_INITIAL_HALF_SUPPORT_CYCLES
            ),
            "step_half_support_cycles_at_highpass": (
                _ONSET_FIR_AUTO_STEP_HALF_SUPPORT_CYCLES
            ),
            "maximum_half_support_cycles_at_highpass": (
                _ONSET_FIR_AUTO_MAXIMUM_HALF_SUPPORT_CYCLES
            ),
            "candidate_type": "odd_tap_type_i_linear_phase_hamming",
            "first_replayably_qualified_candidate_selected": True,
        },
        "evaluated_candidates": evaluated,
        "selected_numtaps": selected_numtaps,
        "selected_filter_order": order,
        "impulse_support": {
            "coefficient_count": selected_numtaps,
            **coefficient_receipt,
            "raw_support_offset_samples": [-order, 0],
            "raw_support_span_samples": order,
            "raw_support_span_seconds": order / rate,
            "warm_up_output_samples": order,
            "group_delay_samples": group_delay_samples,
            "group_delay_seconds": group_delay_samples / rate,
            "timestamp_advance_samples": 0,
        },
        "selected_qualification_receipt_sha256": selected_qualification[
            "receipt_sha256"
        ],
        "selected_target_response_qualified": bool(
            selected_qualification["target_response_qualified"]
        ),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def validate_onset_causal_fir_design_selection(
    payload: object,
) -> dict[str, Any]:
    """Replay an automatic or explicit FIR selection receipt."""

    if type(payload) is not dict:
        raise TypeError("onset FIR design selection must be an object")
    data = deepcopy(payload)
    try:
        target = data["target"]
        if type(target) is not dict:
            raise TypeError
        expected = select_onset_causal_fir_design(
            sampling_rate_hz=float(target["sampling_rate_hz"]),
            highpass_hz=float(target["highpass_hz"]),
            lowpass_hz=float(target["lowpass_hz"]),
            requested_numtaps=data["requested_numtaps"],
            selection_policy_id=str(data["selection_policy_id"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("onset FIR design selection is malformed") from exc
    if data != expected:
        raise ValueError("onset FIR design selection does not replay")
    return expected


def validate_onset_causal_fir_response_qualification(
    payload: object,
) -> dict[str, Any]:
    """Recompute a FIR response receipt instead of trusting its boolean.

    The receipt is self-contained, but its SHA-256 alone is not an admission
    authority: a caller able to edit the payload could also recompute that
    hash.  Re-running the frozen dense-grid/exact-DFT qualification prevents
    a false receipt from being promoted by merely changing
    ``target_band_claim_authorized`` to true.
    """

    if type(payload) is not dict:
        raise TypeError("onset FIR response qualification must be an object")
    data = deepcopy(payload)
    try:
        design = data["design"]
        measurement = data["measurement"]
        if type(design) is not dict or type(measurement) is not dict:
            raise TypeError
        expected = qualify_onset_causal_fir_response(
            sampling_rate_hz=float(design["sampling_rate_hz"]),
            highpass_hz=float(design["target_highpass_hz"]),
            lowpass_hz=float(design["target_lowpass_hz"]),
            numtaps=int(design["numtaps"]),
            response_grid_points=int(measurement["response_grid_points"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("onset FIR response qualification is malformed") from exc
    if data != expected:
        raise ValueError("onset FIR response qualification does not replay")
    return expected


def qualify_onset_causal_fir_clinical_admission(
    *,
    design_selection: object,
    response_qualification: object,
) -> dict[str, Any]:
    """Build the fail-closed clinical admission gate above response QC.

    The current response receipt qualifies only DC attenuation and gain at
    the two requested cutoffs.  Passing that narrow gate is useful for
    reporting the achieved response, but it is not sufficient authority for
    clinical onset timing or SOZ evidence.  The missing full-response,
    line-noise, and acquisition-bandwidth gates remain explicit and false;
    no documentation-only disclaimer can be bypassed downstream.
    """

    selection = validate_onset_causal_fir_design_selection(design_selection)
    response = validate_onset_causal_fir_response_qualification(response_qualification)
    if selection["selected_qualification_receipt_sha256"] != response["receipt_sha256"]:
        raise ValueError("clinical admission inputs bind different FIR responses")
    if int(selection["selected_numtaps"]) != int(response["design"]["numtaps"]):
        raise ValueError("clinical admission inputs bind different FIR orders")
    if bool(selection["selected_target_response_qualified"]) is not bool(
        response["target_response_qualified"]
    ):
        raise ValueError("clinical admission inputs disagree on response status")

    support = selection["impulse_support"]
    order = int(selection["selected_filter_order"])
    finite_causal_support_qualified = bool(
        support["raw_support_offset_samples"] == [-order, 0]
        and int(support["raw_support_span_samples"]) == order
        and int(support["warm_up_output_samples"]) == order
        and int(support["timestamp_advance_samples"]) == 0
    )
    gate_rows: list[dict[str, Any]] = [
        {
            "gate_id": "narrow_dc_and_target_cutoff_response_v1",
            "required_for_clinical_admission": True,
            "qualified": bool(response["target_response_qualified"]),
            "status": (
                "qualified" if response["target_response_qualified"] else "failed"
            ),
            "reason_code_when_unqualified": "target_response_gate_unqualified",
        },
        {
            "gate_id": "finite_causal_impulse_support_v1",
            "required_for_clinical_admission": True,
            "qualified": finite_causal_support_qualified,
            "status": "qualified" if finite_causal_support_qualified else "failed",
            "reason_code_when_unqualified": "causal_support_gate_unqualified",
        },
        {
            "gate_id": "full_passband_ripple_v1",
            "required_for_clinical_admission": True,
            "qualified": False,
            "status": "not_implemented_fail_closed",
            "reason_code_when_unqualified": "passband_ripple_gate_not_frozen",
        },
        {
            "gate_id": "full_stopband_attenuation_v1",
            "required_for_clinical_admission": True,
            "qualified": False,
            "status": "not_implemented_fail_closed",
            "reason_code_when_unqualified": "stopband_attenuation_gate_not_frozen",
        },
        {
            "gate_id": "line_noise_rejection_50_or_60_hz_v1",
            "required_for_clinical_admission": True,
            "qualified": False,
            "status": "not_implemented_fail_closed",
            "reason_code_when_unqualified": "line_noise_rejection_gate_not_qualified",
        },
        {
            "gate_id": "acquisition_bandwidth_compatibility_v1",
            "required_for_clinical_admission": True,
            "qualified": False,
            "status": "not_implemented_fail_closed",
            "reason_code_when_unqualified": (
                "acquisition_bandwidth_compatibility_gate_not_qualified"
            ),
        },
    ]
    all_required = all(
        bool(row["qualified"])
        for row in gate_rows
        if row["required_for_clinical_admission"]
    )
    reason_codes = [
        str(row["reason_code_when_unqualified"])
        for row in gate_rows
        if row["required_for_clinical_admission"] and not row["qualified"]
    ]
    body: dict[str, Any] = {
        "schema_version": (ONSET_FIR_CLINICAL_ADMISSION_QUALIFICATION_SCHEMA_VERSION),
        "authorization_scope": "clinical_onset_timing_and_soz_evidence_v1",
        "narrow_response_gate_is_not_clinical_admission": True,
        "input_receipts": {
            "fir_design_selection": selection,
            "fir_response_qualification": response,
        },
        "admission_gates": gate_rows,
        "all_required_gates_qualified": all_required,
        "clinical_onset_support_authorized": all_required,
        "authorization_reason_codes": reason_codes,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def validate_onset_causal_fir_clinical_admission_qualification(
    payload: object,
) -> dict[str, Any]:
    """Replay the layered clinical gate instead of trusting its boolean."""

    if type(payload) is not dict:
        raise TypeError("onset FIR clinical admission qualification must be an object")
    data = deepcopy(payload)
    try:
        inputs = data["input_receipts"]
        if type(inputs) is not dict:
            raise TypeError
        expected = qualify_onset_causal_fir_clinical_admission(
            design_selection=inputs["fir_design_selection"],
            response_qualification=inputs["fir_response_qualification"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "onset FIR clinical admission qualification is malformed"
        ) from exc
    if data != expected:
        raise ValueError("onset FIR clinical admission qualification does not replay")
    return expected


@dataclass(frozen=True)
class CanonicalEEGRecord:
    """Observed physical channels and their immutable canonical receipt.

    ``observed_signal_volts`` contains only channels in
    ``observed_channel_ids``; missing standard-19 channels never appear as
    samples in this mother tensor.
    """

    observed_signal_volts: torch.Tensor
    observed_channel_ids: tuple[str, ...]
    source_header_receipt: dict[str, Any]
    canonical_receipt: dict[str, Any]
    montage_reference_observability_receipt: dict[str, Any]


@dataclass(frozen=True)
class CanonicalEDFPhysicalSourceIdentity:
    """Canonical physical tensor/header identity before QC/view production.

    This narrow carrier exists for complete-corpus signal-equivalence audits.
    It is not a Findings input and intentionally has no canonical QC receipt:
    callers needing QC, evidence permissions, or task views must use
    :class:`CanonicalEEGRecord` / :func:`load_canonical_edf_record` instead.
    The tensor and source-header receipt are produced by the exact same code
    path as the full record, then the reader closes before expensive QC.
    """

    observed_signal_volts: torch.Tensor
    observed_channel_ids: tuple[str, ...]
    source_header_receipt: dict[str, Any]


@dataclass(frozen=True)
class MaterializedCanonicalEEGView:
    tensor: torch.Tensor
    receipt: dict[str, Any]


@dataclass(frozen=True)
class CanonicalEDFViewBundle:
    canonical_record: CanonicalEEGRecord
    findings_native_morphology: MaterializedCanonicalEEGView
    onset_causal: MaterializedCanonicalEEGView
    context_offline: MaterializedCanonicalEEGView
    task_reference_views: dict[str, dict[str, MaterializedCanonicalEEGView]]
    # Compatibility aliases.  They bind exactly to context_offline and its
    # TCP bipolar child; they are not a fourth preprocessing branch.
    findings_clinical: MaterializedCanonicalEEGView
    spatial_bipolar: MaterializedCanonicalEEGView
    materialization_receipt: dict[str, Any]


def _validate_source_header_receipt(payload: object) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("EDF signal-header receipt must be an object")
    required = {
        "schema_version",
        "reader_policy",
        "source_signal_sha256",
        "source_tensor_sha256",
        "observed_channel_ids",
        "unobserved_channel_ids",
        "channel_signal_headers",
        "scope_receipt",
        "receipt_sha256",
    }
    if set(payload) != required:
        raise ValueError("EDF signal-header receipt has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != CANONICAL_EDF_SOURCE_HEADER_SCHEMA_VERSION:
        raise ValueError("unsupported EDF signal-header receipt schema")
    if data["reader_policy"] not in {
        _NATIVE_READER_POLICY,
        _COMPACT_READER_POLICY,
        _STANDARD_SIGNAL_ONLY_READER_POLICY,
    }:
        raise ValueError("unsupported EDF signal-only reader policy")
    for field in ("source_signal_sha256", "source_tensor_sha256", "receipt_sha256"):
        if not _is_sha256(data[field]):
            raise ValueError(f"EDF signal-header {field} must be SHA-256")
    observed = data["observed_channel_ids"]
    unobserved = data["unobserved_channel_ids"]
    if (
        not isinstance(observed, list)
        or not isinstance(unobserved, list)
        or observed != [item for item in STANDARD_19 if item in set(observed)]
        or unobserved != [item for item in STANDARD_19 if item in set(unobserved)]
        or set(observed).intersection(unobserved)
        or set(observed).union(unobserved) != set(STANDARD_19)
    ):
        raise ValueError("EDF signal-header observed/unobserved partition is invalid")
    headers = data["channel_signal_headers"]
    header_required = {
        "channel_id",
        "raw_label",
        "raw_physical_dimension",
        "canonical_physical_unit",
        "scale_to_volts",
        "sampling_rate_numerator",
        "sampling_rate_denominator",
        "sample_count",
        "raw_prefilter",
        "acquisition_highpass_hz",
        "acquisition_lowpass_hz",
        "reference_label",
        "physical_minimum_raw",
        "physical_maximum_raw",
        "digital_minimum",
        "digital_maximum",
        "adc_rail_minimum_volts",
        "adc_rail_maximum_volts",
        "adc_lsb_volts",
        "clipping_qc_source",
    }
    if not isinstance(headers, list) or len(headers) != len(observed):
        raise ValueError("EDF signal-header rows do not match observed channels")
    if [row.get("channel_id") for row in headers] != observed:
        raise ValueError("EDF signal-header channel order drifted")
    for row in headers:
        if type(row) is not dict or set(row) != header_required:
            raise ValueError("EDF signal-header row has missing or unknown fields")
        if row["canonical_physical_unit"] not in {"V", "mV", "uV"}:
            raise ValueError("EDF signal-header physical unit is unsupported")
        _rational_rate(
            int(row["sampling_rate_numerator"]) / int(row["sampling_rate_denominator"]),
            context="EDF signal-header sampling rate",
        )
        if not isinstance(row["sample_count"], int) or row["sample_count"] <= 0:
            raise ValueError("EDF signal-header sample count is invalid")
        calibration_fields = (
            "physical_minimum_raw",
            "physical_maximum_raw",
            "digital_minimum",
            "digital_maximum",
            "adc_rail_minimum_volts",
            "adc_rail_maximum_volts",
            "adc_lsb_volts",
        )
        if row["clipping_qc_source"] == "edf_signal_header_calibration_rails_v1":
            if any(
                isinstance(row[field], bool)
                or not isinstance(row[field], (int, float))
                or not math.isfinite(float(row[field]))
                for field in calibration_fields
            ):
                raise ValueError("EDF signal-header calibration rails are invalid")
            physical_minimum = float(row["physical_minimum_raw"])
            physical_maximum = float(row["physical_maximum_raw"])
            digital_minimum = float(row["digital_minimum"])
            digital_maximum = float(row["digital_maximum"])
            scale = float(row["scale_to_volts"])
            if (
                physical_maximum <= physical_minimum
                or digital_maximum <= digital_minimum
                or not math.isclose(
                    float(row["adc_rail_minimum_volts"]),
                    physical_minimum * scale,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    float(row["adc_rail_maximum_volts"]),
                    physical_maximum * scale,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    float(row["adc_lsb_volts"]),
                    (physical_maximum - physical_minimum)
                    * scale
                    / (digital_maximum - digital_minimum),
                    rel_tol=1e-12,
                    abs_tol=1e-18,
                )
                or float(row["adc_lsb_volts"]) <= 0
            ):
                raise ValueError("EDF signal-header calibration receipt drifted")
        elif row["clipping_qc_source"] == (
            "header_rails_unavailable_local_plateau_only_v1"
        ):
            if any(row[field] is not None for field in calibration_fields):
                raise ValueError("unavailable EDF rails must not carry numeric extrema")
        else:
            raise ValueError("EDF signal-header clipping-QC source is unsupported")
    if data["scope_receipt"] != EEG_ONLY_SCOPE_RECEIPT:
        raise ValueError("EDF signal-header violates the EEG-only firewall")
    core = {
        "schema_version": data["schema_version"],
        "observed_channel_ids": observed,
        "unobserved_channel_ids": unobserved,
        "channel_signal_headers": headers,
        "scope_receipt": data["scope_receipt"],
    }
    expected_source = _source_signal_sha256(
        source_header_core=core,
        source_tensor_sha256=data["source_tensor_sha256"],
    )
    if data["source_signal_sha256"] != expected_source:
        raise ValueError(
            "EDF source-signal hash does not bind signal header and tensor"
        )
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("EDF signal-header receipt hash does not bind content")
    return data


def validate_canonical_edf_source_header_receipt(
    payload: object,
) -> dict[str, Any]:
    """Validate the signal-only EDF source/header identity receipt.

    This public wrapper deliberately exposes only the already frozen,
    EEG-signal-header validator.  It does not open an EDF and cannot read a
    patient/recording header, TAL annotation, spreadsheet, or clinical text.
    Downstream detector lineage authorities use it to replay the observed
    physical roster, unit conversion, common clock and source-tensor binding
    instead of trusting a bare channel list or hash-shaped string.
    """

    return _validate_source_header_receipt(payload)


def canonical_source_tensor_sha256(
    tensor: torch.Tensor,
    *,
    channel_ids: Sequence[str],
) -> str:
    """Content-address one canonical observed EEG tensor in volts.

    The result is byte-identical to the hash stored by the canonical EDF
    loader.  Making this narrow primitive public lets consumers verify the
    actual carrier payload against its typed source receipt; the digest alone
    is never treated as source authority.
    """

    return _source_tensor_sha256(tensor, channel_ids=channel_ids)


def _read_canonical_record(
    edf_path: str | Path,
    *,
    config: CanonicalEDFConfig,
    reader_factory: Callable[[str], object] | None,
    source_identity_only: bool = False,
) -> CanonicalEEGRecord | CanonicalEDFPhysicalSourceIdentity:
    path = _source_edf(edf_path)
    factory = _reader_factory if reader_factory is None else reader_factory
    reader = factory(str(path))
    try:
        labels = tuple(str(value).strip() for value in reader.getSignalLabels())
        montage_preflight = classify_signal_labels(labels)
        candidates: dict[str, list[int]] = {name: [] for name in STANDARD_19}
        reference_token_by_signal_index: dict[int, str] = {}
        for observation in montage_preflight["signal_label_observations"]:
            if observation["signal_role"] not in {
                "direct_standard_electrode",
                "direct_standard_electrode_unknown_reference",
            }:
                continue
            channel = str(observation["positive_electrode"])
            signal_index = int(observation["signal_index"])
            candidates[channel].append(signal_index)
            if observation["reference_token"] is not None:
                reference_token_by_signal_index[signal_index] = str(
                    observation["reference_token"]
                )
        duplicates = {
            name: tuple(labels[index] for index in indices)
            for name, indices in candidates.items()
            if len(indices) > 1
        }
        if duplicates:
            raise ValueError(
                "canonical EDF has ambiguous direct standard-19 channels: "
                f"{sorted(duplicates)}"
            )
        observed_ids = tuple(name for name in STANDARD_19 if candidates[name])
        if len(observed_ids) < config.minimum_observed_standard19:
            raise ValueError(
                "canonical EDF has insufficient directly observed standard-19 coverage; "
                f"acquisition montage={montage_preflight['montage_class']}"
            )
        unobserved_ids = tuple(name for name in STANDARD_19 if not candidates[name])
        indices = tuple(candidates[name][0] for name in observed_ids)

        rates = tuple(
            _rational_rate(
                reader.getSampleFrequency(index),
                context=f"EDF signal {name} sampling rate",
            )
            for name, index in zip(observed_ids, indices)
        )
        if len(set(rates)) != 1:
            raise ValueError(
                "canonical findings views currently require one shared physical sampling clock"
            )
        source_rate = rates[0]
        source_sfreq = _rate_hz(source_rate)
        raw_counts = reader.getNSamples()
        counts = tuple(int(raw_counts[index]) for index in indices)
        if any(count <= 0 for count in counts) or len(set(counts)) != 1:
            raise ValueError(
                "canonical findings views require equal positive sample counts"
            )
        sample_count = counts[0]
        duration = sample_count / source_sfreq

        header_rows: list[dict[str, object]] = []
        clipping_calibration_by_channel: dict[str, dict[str, float | str | None]] = {}
        payload_rows: list[np.ndarray] = []
        for channel_id, index in zip(observed_ids, indices):
            raw_label = labels[index]
            unit, scale, raw_unit = _unit_and_scale(reader.getPhysicalDimension(index))
            raw_prefilter = (
                str(reader.getPrefilter(index)).strip()
                if hasattr(reader, "getPrefilter")
                else ""
            )
            highpass, lowpass = _acquisition_bandwidth(
                raw_prefilter,
                0.5 * source_sfreq,
            )
            calibration = _edf_signal_calibration(
                reader,
                signal_index=index,
                scale_to_volts=float(scale),
            )
            clipping_calibration_by_channel[channel_id] = calibration
            raw_values = np.asarray(
                reader.readSignal(index, 0, sample_count), dtype=np.float64
            )
            if raw_values.shape != (sample_count,) or not np.isfinite(raw_values).all():
                raise ValueError("EDF reader returned invalid physical EEG samples")
            payload_rows.append(
                (raw_values * float(scale)).astype(np.float32, copy=False)
            )
            header_rows.append(
                {
                    "channel_id": channel_id,
                    "raw_label": raw_label,
                    "raw_physical_dimension": raw_unit,
                    "canonical_physical_unit": unit,
                    "scale_to_volts": float(scale),
                    "sampling_rate_numerator": source_rate[0],
                    "sampling_rate_denominator": source_rate[1],
                    "sample_count": sample_count,
                    "raw_prefilter": raw_prefilter,
                    "acquisition_highpass_hz": highpass,
                    "acquisition_lowpass_hz": lowpass,
                    "reference_label": reference_token_by_signal_index.get(
                        index, _reference_from_signal_label(raw_label)
                    ),
                    **calibration,
                }
            )

        observed_tensor = torch.from_numpy(
            np.stack(payload_rows).astype(np.float32, copy=False)
        ).contiguous()
        tensor_hash = _source_tensor_sha256(
            observed_tensor,
            channel_ids=observed_ids,
        )
        source_hash_core: dict[str, object] = {
            "schema_version": CANONICAL_EDF_SOURCE_HEADER_SCHEMA_VERSION,
            "observed_channel_ids": list(observed_ids),
            "unobserved_channel_ids": list(unobserved_ids),
            "channel_signal_headers": header_rows,
            "scope_receipt": deepcopy(EEG_ONLY_SCOPE_RECEIPT),
        }
        source_signal_hash = _source_signal_sha256(
            source_header_core=source_hash_core,
            source_tensor_sha256=tensor_hash,
        )
        source_header_receipt = {
            **source_hash_core,
            "reader_policy": str(
                getattr(reader, "canonical_reader_policy", _NATIVE_READER_POLICY)
            ),
            "source_signal_sha256": source_signal_hash,
            "source_tensor_sha256": tensor_hash,
            "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        }
        source_header_receipt["receipt_sha256"] = _canonical_sha256(
            source_header_receipt
        )
        source_header_receipt = _validate_source_header_receipt(source_header_receipt)
        if source_identity_only:
            return CanonicalEDFPhysicalSourceIdentity(
                observed_signal_volts=observed_tensor,
                observed_channel_ids=observed_ids,
                source_header_receipt=source_header_receipt,
            )
        montage_reference_receipt = build_montage_reference_observability_receipt(
            signal_labels=labels,
            source_signal_sha256=source_signal_hash,
        )

        quality = _quality_primitives(
            observed_tensor,
            observed_ids,
            sample_rate=source_rate,
            flatline_run_seconds=config.flatline_run_seconds,
            clipping_run_seconds=config.clipping_run_seconds,
            tolerance_volts=config.qc_tolerance_volts,
            clipping_calibration_by_channel=clipping_calibration_by_channel,
        )
        header_by_id = {str(row["channel_id"]): row for row in header_rows}
        channels: list[dict[str, object]] = []
        for channel_id in STANDARD_19:
            if channel_id in header_by_id:
                row = header_by_id[channel_id]
                channels.append(
                    {
                        "channel_id": channel_id,
                        "raw_label": row["raw_label"],
                        "canonical_name": channel_id,
                        "source_physical_unit": row["canonical_physical_unit"],
                        "scale_to_volts": row["scale_to_volts"],
                        "sample_rate_numerator": source_rate[0],
                        "sample_rate_denominator": source_rate[1],
                        "sample_count": sample_count,
                        "observed": True,
                        "imputed": False,
                        "acquisition_highpass_hz": row["acquisition_highpass_hz"],
                        "acquisition_lowpass_hz": row["acquisition_lowpass_hz"],
                        "reference_label": row["reference_label"],
                    }
                )
            else:
                channels.append(
                    {
                        "channel_id": channel_id,
                        "raw_label": f"UNOBSERVED_STANDARD19_{channel_id}",
                        "canonical_name": channel_id,
                        "source_physical_unit": "V",
                        "scale_to_volts": 1.0,
                        "sample_rate_numerator": source_rate[0],
                        "sample_rate_denominator": source_rate[1],
                        "sample_count": 0,
                        "observed": False,
                        "imputed": False,
                        "acquisition_highpass_hz": None,
                        "acquisition_lowpass_hz": None,
                        "reference_label": "UNOBSERVED",
                    }
                )
        canonical_receipt = build_canonical_signal_receipt(
            recording_id=f"EEGREC-{source_signal_hash[:24]}",
            source_signal_sha256=source_signal_hash,
            recording_duration_seconds=duration,
            channels=channels,
            quality_primitives=quality,
        )
        return CanonicalEEGRecord(
            observed_signal_volts=observed_tensor,
            observed_channel_ids=observed_ids,
            source_header_receipt=source_header_receipt,
            canonical_receipt=canonical_receipt,
            montage_reference_observability_receipt=montage_reference_receipt,
        )
    finally:
        if hasattr(reader, "close"):
            reader.close()


def _expanded_standard19(record: CanonicalEEGRecord) -> torch.Tensor:
    source = record.observed_signal_volts.detach().cpu().to(torch.float32)
    samples = int(source.shape[1])
    expanded = torch.zeros((len(STANDARD_19), samples), dtype=torch.float32)
    source_index = {
        channel_id: index
        for index, channel_id in enumerate(record.observed_channel_ids)
    }
    for target_index, channel_id in enumerate(STANDARD_19):
        if channel_id in source_index:
            expanded[target_index] = source[source_index[channel_id]]
    return expanded.contiguous()


def _resampling_ratio(
    source_rate: tuple[int, int],
    output_rate: tuple[int, int],
) -> tuple[int, int]:
    ratio = Fraction(output_rate[0], output_rate[1]) / Fraction(
        source_rate[0], source_rate[1]
    )
    return ratio.numerator, ratio.denominator


def _filter_and_resample_findings(
    expanded_volts: torch.Tensor,
    *,
    source_rate: tuple[int, int],
    output_rate: tuple[int, int],
    highpass_hz: float,
    lowpass_hz: float,
    order: int,
) -> torch.Tensor:
    source_sfreq = _rate_hz(source_rate)
    values = expanded_volts.detach().cpu().to(torch.float64).numpy()
    sos = butter(
        order,
        [highpass_hz / (0.5 * source_sfreq), lowpass_hz / (0.5 * source_sfreq)],
        btype="bandpass",
        output="sos",
    )
    filtered = sosfiltfilt(sos, values, axis=1)
    up, down = _resampling_ratio(source_rate, output_rate)
    if (up, down) != (1, 1):
        filtered = resample_poly(filtered, up, down, axis=1)
    result = torch.from_numpy(filtered.astype(np.float32, copy=False)).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError("clinical findings preprocessing produced non-finite values")
    return result


def _quality_tile_sha256(
    *,
    canonical: Mapping[str, Any],
    view_id: str,
    tile_interval: tuple[int, int],
    output_rate: tuple[int, int],
    output_sources: Mapping[str, Sequence[str]],
    observed_by_unit: Mapping[str, bool],
    edge_left_invalid_samples: int,
    edge_right_invalid_samples: int,
    total_samples: int,
) -> str:
    sfreq = _rate_hz(output_rate)
    start_seconds = tile_interval[0] / sfreq
    stop_seconds = tile_interval[1] / sfreq
    relevant_quality = []
    for primitive in canonical["quality_primitives"]:
        if (
            float(primitive["start_recording_seconds"]) < stop_seconds
            and float(primitive["stop_recording_seconds"]) > start_seconds
        ):
            relevant_quality.append(primitive)
    descriptor = {
        "domain": "canonical-view-quality-tile-v1",
        "view_id": view_id,
        "global_output_sample_interval": list(tile_interval),
        "output_sources": {
            key: list(output_sources[key]) for key in sorted(output_sources)
        },
        "observed_by_unit": {
            key: bool(observed_by_unit[key]) for key in sorted(observed_by_unit)
        },
        "canonical_quality_primitives": relevant_quality,
        "left_global_edge_overlap": tile_interval[0] < edge_left_invalid_samples,
        "right_global_edge_overlap": (
            tile_interval[1] > total_samples - edge_right_invalid_samples
            if edge_right_invalid_samples
            else False
        ),
    }
    return _canonical_sha256(descriptor)


def _cache_tiles(
    tensor: torch.Tensor,
    *,
    unit_ids: Sequence[str],
    tile_size: int,
    canonical: Mapping[str, Any],
    view_id: str,
    output_rate: tuple[int, int],
    output_sources: Mapping[str, Sequence[str]],
    observed_by_unit: Mapping[str, bool],
    edge_left_invalid_samples: int,
    edge_right_invalid_samples: int,
) -> list[dict[str, object]]:
    total = int(tensor.shape[1])
    rows: list[dict[str, object]] = []
    for tile_index, start in enumerate(range(0, total, tile_size)):
        stop = min(start + tile_size, total)
        rows.append(
            {
                "tile_index": tile_index,
                "global_output_sample_interval": [start, stop],
                "signal_sha256": deterministic_view_tensor_sha256(
                    tensor[:, start:stop], unit_ids=unit_ids
                ),
                "quality_mask_sha256": _quality_tile_sha256(
                    canonical=canonical,
                    view_id=view_id,
                    tile_interval=(start, stop),
                    output_rate=output_rate,
                    output_sources=output_sources,
                    observed_by_unit=observed_by_unit,
                    edge_left_invalid_samples=edge_left_invalid_samples,
                    edge_right_invalid_samples=edge_right_invalid_samples,
                    total_samples=total,
                ),
            }
        )
    return rows


def _canonical_source_rate(record: CanonicalEEGRecord) -> tuple[int, int]:
    first_channel = record.canonical_receipt["channels"][0]
    return (
        int(first_channel["sample_rate_numerator"]),
        int(first_channel["sample_rate_denominator"]),
    )


def _standard19_identity_matrix() -> list[list[float]]:
    return [
        [1.0 if row == column else 0.0 for column in range(len(STANDARD_19))]
        for row in range(len(STANDARD_19))
    ]


def _direct_standard19_view(
    record: CanonicalEEGRecord,
    *,
    tensor: torch.Tensor,
    transform: Mapping[str, Any],
    task_role: str,
    view_id: str,
    config: CanonicalEDFConfig,
) -> MaterializedCanonicalEEGView:
    canonical = record.canonical_receipt
    output_clock = transform["output_clock"]
    output_rate = (
        int(output_clock["sampling_rate_numerator"]),
        int(output_clock["sampling_rate_denominator"]),
    )
    output_sfreq = _rate_hz(output_rate)
    expected_position = (
        float(canonical["recording_duration_seconds"]) * output_rate[0] / output_rate[1]
    )
    expected_samples = int(round(expected_position))
    if abs(expected_position - expected_samples) > 1e-8:
        raise ValueError("task view duration is not aligned to its output clock")
    tensor = tensor.detach().cpu().to(torch.float32).contiguous()
    if tensor.shape != (len(STANDARD_19), expected_samples):
        raise ValueError("task view tensor shape drifted from standard-19/global clock")
    if not torch.isfinite(tensor).all():
        raise ValueError("task view contains non-finite values")

    observed = set(record.observed_channel_ids)
    definitions = [
        {
            "unit_id": channel_id,
            "unit_type": "electrode",
            "physical_unit": "V",
            "observed": channel_id in observed,
            "imputed": channel_id not in observed,
        }
        for channel_id in STANDARD_19
    ]
    output_sources = {channel: [channel] for channel in STANDARD_19}
    observed_by_unit = {channel: channel in observed for channel in STANDARD_19}
    edge_left_samples = int(transform["edge_handling"]["left_invalid_samples"])
    edge_right_samples = int(transform["edge_handling"]["right_invalid_samples"])
    tile_size = max(1, int(round(config.cache_tile_seconds * output_sfreq)))
    processed_hash = deterministic_view_tensor_sha256(tensor, unit_ids=STANDARD_19)
    tiles = _cache_tiles(
        tensor,
        unit_ids=STANDARD_19,
        tile_size=tile_size,
        canonical=canonical,
        view_id=view_id,
        output_rate=output_rate,
        output_sources=output_sources,
        observed_by_unit=observed_by_unit,
        edge_left_invalid_samples=edge_left_samples,
        edge_right_invalid_samples=edge_right_samples,
    )
    receipt = build_signal_view_receipt(
        canonical,
        view_id=view_id,
        task_role=task_role,
        transform_spec=transform,
        output_unit_definitions=definitions,
        selected_global_output_sample_interval=(0, expected_samples),
        processed_view_sha256=processed_hash,
        cache_tile_size_samples=tile_size,
        cache_tiles=tiles,
    )
    return MaterializedCanonicalEEGView(tensor=tensor, receipt=receipt)


def _materialize_native_morphology_view(
    record: CanonicalEEGRecord,
    *,
    config: CanonicalEDFConfig,
) -> MaterializedCanonicalEEGView:
    canonical = record.canonical_receipt
    source_rate = _canonical_source_rate(record)
    identity = _standard19_identity_matrix()
    transform = build_transform_spec(
        transform_name="edf_to_findings_native_morphology_referential_v1",
        input_unit_ids=STANDARD_19,
        output_unit_ids=STANDARD_19,
        source_sampling_rate=source_rate,
        output_sampling_rate=source_rate,
        resample_up=1,
        resample_down=1,
        resampler_implementation="none",
        anti_alias_filter="none",
        anti_alias_lowpass_hz=None,
        filter_family="none",
        filter_order=None,
        highpass_hz=None,
        lowpass_hz=None,
        phase_policy="none",
        normalization_method="physical_unit_scale_only",
        normalization_source="channel_metadata",
        clipping_applied=False,
        clipping_policy="none",
        clipping_source="none",
        reference_type="source_header_reference_preserved_per_channel_v1",
        reference_matrix=identity,
        edge_policy="none",
        edge_left_invalid_samples=0,
        edge_right_invalid_samples=0,
        software_versions={
            "producer": CANONICAL_EDF_PRODUCER_ID,
            "torch": torch.__version__,
        },
    )
    return _direct_standard19_view(
        record,
        tensor=_expanded_standard19(record),
        transform=transform,
        task_role="findings_native_morphology",
        view_id=f"FINDINGS-NATIVE-{canonical['source_signal_sha256'][:20]}",
        config=config,
    )


def _require_onset_causal_tensor_replay(
    record: CanonicalEEGRecord,
    actual_tensor: torch.Tensor,
    *,
    sampling_rate_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
    numtaps: int,
) -> None:
    """Recompute the one-sided FIR from the mother signal and require identity.

    Content-addressing an attacker-supplied processed tensor is insufficient:
    hashes can be recomputed after shifting the tensor toward the past.  This
    replay makes the no-timestamp-advance claim depend on the actual sample
    transform, one channel at a time to avoid a second full float64 recording.
    """

    coefficients = firwin(
        int(numtaps),
        [float(highpass_hz), float(lowpass_hz)],
        pass_zero=False,
        window="hamming",
        scale=True,
        fs=float(sampling_rate_hz),
    )
    expanded = _expanded_standard19(record)
    actual = actual_tensor.detach().cpu().to(torch.float32).contiguous()
    if actual.shape != expanded.shape:
        raise ValueError("causal onset tensor shape does not replay from mother signal")
    for channel_index, channel_id in enumerate(STANDARD_19):
        source = expanded[channel_index].detach().cpu().to(torch.float64).numpy()
        replayed = lfilter(
            coefficients,
            np.asarray([1.0], dtype=np.float64),
            source,
        ).astype(np.float32, copy=False)
        if not np.array_equal(
            replayed,
            actual[channel_index].numpy(),
            equal_nan=False,
        ):
            raise ValueError(
                "causal onset tensor does not replay from canonical mother signal "
                f"for {channel_id}"
            )


def _materialize_onset_causal_view(
    record: CanonicalEEGRecord,
    *,
    config: CanonicalEDFConfig,
) -> MaterializedCanonicalEEGView:
    canonical = record.canonical_receipt
    source_rate = _canonical_source_rate(record)
    source_sfreq = _rate_hz(source_rate)
    lowpass = min(float(config.onset_lowpass_hz), 0.45 * source_sfreq)
    highpass = float(config.onset_highpass_hz)
    if lowpass <= highpass:
        raise ValueError("causal onset bandwidth is empty on this EDF clock")
    design_selection = select_onset_causal_fir_design(
        sampling_rate_hz=source_sfreq,
        highpass_hz=highpass,
        lowpass_hz=lowpass,
        requested_numtaps=config.onset_fir_numtaps,
        selection_policy_id=config.onset_fir_numtaps_policy,
    )
    numtaps = int(design_selection["selected_numtaps"])
    order = numtaps - 1
    expanded = _expanded_standard19(record)
    if int(expanded.shape[1]) <= order:
        raise ValueError("recording is too short for the configured causal FIR warm-up")
    coefficients = firwin(
        numtaps,
        [highpass, lowpass],
        pass_zero=False,
        window="hamming",
        scale=True,
        fs=source_sfreq,
    )
    response_qualification = qualify_onset_causal_fir_response(
        sampling_rate_hz=source_sfreq,
        highpass_hz=highpass,
        lowpass_hz=lowpass,
        numtaps=numtaps,
    )
    if (
        response_qualification["receipt_sha256"]
        != design_selection["selected_qualification_receipt_sha256"]
    ):
        raise ValueError("onset FIR selection and response qualification disagree")
    clinical_admission_qualification = qualify_onset_causal_fir_clinical_admission(
        design_selection=design_selection,
        response_qualification=response_qualification,
    )
    measured_bandwidth = response_qualification["measurement"][
        "measured_minus_3db_bandwidth_hz"
    ]
    # lfilter is one-sided.  Unlike filtfilt/resample_poly it cannot consume a
    # sample after the output timestamp.  The global prefix of length `order`
    # is masked as filter warm-up, and the constant order/2 delay is receipted.
    filtered = lfilter(
        coefficients,
        np.asarray([1.0], dtype=np.float64),
        expanded.detach().cpu().to(torch.float64).numpy(),
        axis=1,
    )
    tensor = torch.from_numpy(filtered.astype(np.float32, copy=False)).contiguous()
    identity = _standard19_identity_matrix()
    transform = build_transform_spec(
        transform_name=CANONICAL_EDF_ONSET_TRANSFORM_NAME,
        input_unit_ids=STANDARD_19,
        output_unit_ids=STANDARD_19,
        source_sampling_rate=source_rate,
        output_sampling_rate=source_rate,
        resample_up=1,
        resample_down=1,
        resampler_implementation="none",
        anti_alias_filter="none",
        anti_alias_lowpass_hz=None,
        filter_family="fir",
        filter_order=order,
        # These fields feed the downstream *effective* bandwidth ledger.
        # Design cutoffs remain separately frozen in the qualification
        # receipt and must not be silently presented as achieved response.
        highpass_hz=float(measured_bandwidth[0]),
        lowpass_hz=float(measured_bandwidth[1]),
        phase_policy="causal_with_group_delay_receipt",
        normalization_method="physical_unit_scale_only",
        normalization_source="channel_metadata",
        clipping_applied=False,
        clipping_policy="none",
        clipping_source="none",
        reference_type="source_header_reference_preserved_per_channel_v1",
        reference_matrix=identity,
        edge_policy="global_recording_edges",
        edge_left_invalid_samples=order,
        edge_right_invalid_samples=0,
        software_versions={
            "producer": CANONICAL_EDF_PRODUCER_ID,
            "fir_design": "scipy.signal.firwin_hamming_scale_true_v1",
            "fir_design_target_band_hz": f"{highpass:.17g},{lowpass:.17g}",
            "fir_design_selection_policy": design_selection["selection_policy_id"],
            "fir_design_selection_mode": design_selection["selection_mode"],
            "fir_design_selection_sha256": design_selection["receipt_sha256"],
            "fir_requested_numtaps": (
                "auto"
                if design_selection["requested_numtaps"] is None
                else str(design_selection["requested_numtaps"])
            ),
            "fir_selected_numtaps": str(numtaps),
            "fir_coefficients_sha256": design_selection["impulse_support"][
                "coefficient_sha256"
            ],
            "fir_impulse_support_offset_samples": f"{-order},0",
            "fir_nonzero_support_offset_samples": (
                f"{design_selection['impulse_support']['raw_nonzero_support_offset_samples'][0]},"
                f"{design_selection['impulse_support']['raw_nonzero_support_offset_samples'][1]}"
            ),
            "fir_impulse_support_span_samples": str(order),
            "fir_impulse_support_span_seconds": f"{order / source_sfreq:.17g}",
            ONSET_FIR_RESPONSE_AUTHORIZATION_SOFTWARE_KEY: (
                "true"
                if response_qualification["target_band_claim_authorized"]
                else "false"
            ),
            ONSET_FIR_CLINICAL_ADMISSION_AUTHORIZATION_SOFTWARE_KEY: (
                "true"
                if clinical_admission_qualification["clinical_onset_support_authorized"]
                else "false"
            ),
            "fir_response_qualification_sha256": response_qualification[
                "receipt_sha256"
            ],
            "fir_clinical_admission_qualification_sha256": (
                clinical_admission_qualification["receipt_sha256"]
            ),
            "causal_apply": "scipy.signal.lfilter_zero_initial_state_v1",
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "torch": torch.__version__,
        },
    )
    return _direct_standard19_view(
        record,
        tensor=tensor,
        transform=transform,
        task_role="onset_causal",
        view_id=f"ONSET-CAUSAL-{canonical['source_signal_sha256'][:20]}",
        config=config,
    )


def _materialize_context_offline_view(
    record: CanonicalEEGRecord,
    *,
    config: CanonicalEDFConfig,
) -> MaterializedCanonicalEEGView:
    canonical = record.canonical_receipt
    source_rate = _canonical_source_rate(record)
    output_rate = (
        source_rate
        if config.output_sampling_rate_hz is None
        else _rational_rate(
            config.output_sampling_rate_hz,
            context="clinical findings output sampling rate",
        )
    )
    source_sfreq = _rate_hz(source_rate)
    output_sfreq = _rate_hz(output_rate)
    actual_lowpass = min(
        float(config.findings_lowpass_hz),
        0.45 * source_sfreq,
        0.45 * output_sfreq,
    )
    if actual_lowpass <= config.findings_highpass_hz:
        raise ValueError("clinical findings bandwidth is empty on this EDF clock")
    expanded = _expanded_standard19(record)
    tensor = _filter_and_resample_findings(
        expanded,
        source_rate=source_rate,
        output_rate=output_rate,
        highpass_hz=float(config.findings_highpass_hz),
        lowpass_hz=actual_lowpass,
        order=int(config.butterworth_order),
    )
    duration = float(canonical["recording_duration_seconds"])
    expected_samples_position = duration * output_rate[0] / output_rate[1]
    expected_samples = int(round(expected_samples_position))
    if abs(expected_samples_position - expected_samples) > 1e-8:
        raise ValueError(
            "recording duration is not aligned to the requested findings output clock"
        )
    if tensor.shape[1] != expected_samples:
        raise ValueError("clinical findings resampler drifted from the global clock")

    up, down = _resampling_ratio(source_rate, output_rate)
    identity = _standard19_identity_matrix()
    edge_samples = min(
        int(math.ceil(config.edge_guard_seconds * output_sfreq)),
        max(0, (expected_samples - 1) // 2),
    )
    transform = build_transform_spec(
        transform_name="edf_to_context_offline_referential_v1",
        input_unit_ids=STANDARD_19,
        output_unit_ids=STANDARD_19,
        source_sampling_rate=source_rate,
        output_sampling_rate=output_rate,
        resample_up=up,
        resample_down=down,
        resampler_implementation=(
            "none" if (up, down) == (1, 1) else "scipy.signal.resample_poly_v1"
        ),
        anti_alias_filter=(
            "none" if (up, down) == (1, 1) else "scipy_resample_poly_kaiser5_default_v1"
        ),
        anti_alias_lowpass_hz=(
            None
            if (up, down) == (1, 1)
            else min(0.5 * source_sfreq, 0.5 * output_sfreq)
        ),
        filter_family="butterworth",
        filter_order=int(config.butterworth_order),
        highpass_hz=float(config.findings_highpass_hz),
        lowpass_hz=actual_lowpass,
        phase_policy="offline_zero_phase",
        normalization_method="physical_unit_scale_only",
        normalization_source="channel_metadata",
        clipping_applied=False,
        clipping_policy="none",
        clipping_source="none",
        reference_type="source_header_reference_preserved_per_channel_v1",
        reference_matrix=identity,
        edge_policy="global_recording_edges" if edge_samples else "none",
        edge_left_invalid_samples=edge_samples,
        edge_right_invalid_samples=edge_samples,
        software_versions={
            "producer": CANONICAL_EDF_PRODUCER_ID,
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "torch": torch.__version__,
            "pyedflib": _package_version("pyedflib"),
        },
    )
    observed = set(record.observed_channel_ids)
    definitions = [
        {
            "unit_id": channel_id,
            "unit_type": "electrode",
            "physical_unit": "V",
            "observed": channel_id in observed,
            "imputed": channel_id not in observed,
        }
        for channel_id in STANDARD_19
    ]
    view_id = f"CONTEXT-OFFLINE-{canonical['source_signal_sha256'][:20]}"
    output_sources = {channel: [channel] for channel in STANDARD_19}
    observed_by_unit = {channel: channel in observed for channel in STANDARD_19}
    tile_size = max(1, int(round(config.cache_tile_seconds * output_sfreq)))
    processed_hash = deterministic_view_tensor_sha256(
        tensor,
        unit_ids=STANDARD_19,
    )
    tiles = _cache_tiles(
        tensor,
        unit_ids=STANDARD_19,
        tile_size=tile_size,
        canonical=canonical,
        view_id=view_id,
        output_rate=output_rate,
        output_sources=output_sources,
        observed_by_unit=observed_by_unit,
        edge_left_invalid_samples=edge_samples,
        edge_right_invalid_samples=edge_samples,
    )
    receipt = build_signal_view_receipt(
        canonical,
        view_id=view_id,
        task_role="context_offline",
        transform_spec=transform,
        output_unit_definitions=definitions,
        selected_global_output_sample_interval=(0, expected_samples),
        processed_view_sha256=processed_hash,
        cache_tile_size_samples=tile_size,
        cache_tiles=tiles,
    )
    return MaterializedCanonicalEEGView(tensor=tensor, receipt=receipt)


def _reference_recipe(
    reference_kind: str,
) -> tuple[tuple[str, ...], list[list[float]], dict[str, list[str]], str, str,]:
    channel_index = {name: index for index, name in enumerate(STANDARD_19)}
    if reference_kind == "tcp_bipolar":
        output_ids = tuple(f"{left}-{right}" for left, right in TCP_20_EDGES)
        sources = {
            output_id: [left, right]
            for output_id, (left, right) in zip(output_ids, TCP_20_EDGES)
        }
        matrix: list[list[float]] = []
        for left, right in TCP_20_EDGES:
            row = [0.0] * len(STANDARD_19)
            row[channel_index[left]] = 1.0
            row[channel_index[right]] = -1.0
            matrix.append(row)
        return (
            output_ids,
            matrix,
            sources,
            "lead",
            "longitudinal_bipolar_tcp20_frozen_carriers_v1",
        )
    if reference_kind == "car":
        output_ids = tuple(f"{channel}-CAR" for channel in STANDARD_19)
        carriers = list(STANDARD_19)
        scale = 1.0 / len(STANDARD_19)
        matrix = []
        for target in STANDARD_19:
            row = [-scale] * len(STANDARD_19)
            row[channel_index[target]] += 1.0
            matrix.append(row)
        return (
            output_ids,
            matrix,
            {output_id: carriers.copy() for output_id in output_ids},
            "virtual",
            "common_average_standard19_frozen_all_carriers_v1",
        )
    if reference_kind == "laplacian":
        output_ids = tuple(f"{channel}-LAP" for channel in STANDARD_19)
        matrix = []
        sources: dict[str, list[str]] = {}
        for target, output_id in zip(STANDARD_19, output_ids):
            neighbours = _LAPLACIAN_NEIGHBORS[target]
            row = [0.0] * len(STANDARD_19)
            row[channel_index[target]] = 1.0
            weight = -1.0 / len(neighbours)
            for neighbour in neighbours:
                row[channel_index[neighbour]] = weight
            matrix.append(row)
            sources[output_id] = [target, *neighbours]
        return (
            output_ids,
            matrix,
            sources,
            "virtual",
            "surface_laplacian_standard19_frozen_neighbour_graph_v1",
        )
    raise ValueError(f"unsupported reference kind: {reference_kind}")


def _materialize_reference_view(
    record: CanonicalEEGRecord,
    parent: MaterializedCanonicalEEGView,
    *,
    reference_kind: str,
    config: CanonicalEDFConfig,
) -> MaterializedCanonicalEEGView:
    canonical = record.canonical_receipt
    parent_transform = parent.receipt["transform_spec"]
    output_clock = parent_transform["output_clock"]
    output_rate = (
        int(output_clock["sampling_rate_numerator"]),
        int(output_clock["sampling_rate_denominator"]),
    )
    output_sfreq = _rate_hz(output_rate)
    output_ids, matrix, output_sources, unit_type, reference_type = _reference_recipe(
        reference_kind
    )
    reference_matrix = torch.tensor(matrix, dtype=torch.float32)
    tensor = torch.matmul(
        reference_matrix, parent.tensor.to(torch.float32)
    ).contiguous()
    edge_left = int(parent_transform["edge_handling"]["left_invalid_samples"])
    edge_right = int(parent_transform["edge_handling"]["right_invalid_samples"])
    transform = build_transform_spec(
        transform_name=f"task_referential_to_{reference_kind}_v1",
        input_unit_ids=STANDARD_19,
        output_unit_ids=output_ids,
        source_sampling_rate=output_rate,
        output_sampling_rate=output_rate,
        resample_up=1,
        resample_down=1,
        resampler_implementation="none",
        anti_alias_filter="none",
        anti_alias_lowpass_hz=None,
        filter_family="none",
        filter_order=None,
        highpass_hz=None,
        lowpass_hz=None,
        phase_policy="none",
        normalization_method="none",
        normalization_source="none",
        clipping_applied=False,
        clipping_policy="none",
        clipping_source="none",
        reference_type=reference_type,
        reference_matrix=matrix,
        edge_policy="global_recording_edges" if edge_left or edge_right else "none",
        edge_left_invalid_samples=edge_left,
        edge_right_invalid_samples=edge_right,
        software_versions={
            "producer": CANONICAL_EDF_PRODUCER_ID,
            "torch": torch.__version__,
        },
    )
    montage = require_reference_materialization_authorized(
        record.montage_reference_observability_receipt,
        reference_kinds=(reference_kind,),
    )
    qualification = montage["derived_reference_contracts"][reference_kind]
    support_by_unit = {
        str(row["unit_id"]): row for row in qualification["output_support"]
    }
    qualified_matrix = qualification["reference_matrix_observability"]
    if (
        qualified_matrix["row_unit_ids"] != list(output_ids)
        or qualified_matrix["column_unit_ids"] != list(STANDARD_19)
        or qualified_matrix["matrix"] != matrix
    ):
        raise ValueError("derived reference recipe drifted from montage qualification")
    definitions: list[dict[str, object]] = []
    observed_by_unit: dict[str, bool] = {}
    for output_id in output_ids:
        support = support_by_unit[output_id]
        if support["quality_dependency_channel_ids"] != output_sources[output_id]:
            raise ValueError("derived reference carrier support drifted")
        evidence_carrier = bool(support["evidence_eligible"])
        observed_by_unit[output_id] = evidence_carrier
        definitions.append(
            {
                "unit_id": output_id,
                "unit_type": unit_type,
                "physical_unit": "V",
                "observed": evidence_carrier,
                "imputed": not evidence_carrier,
            }
        )
    parent_role = str(parent.receipt["task_role"]).upper().replace("_", "-")
    reference_slug = reference_kind.upper().replace("_", "-")
    view_id = f"{parent_role}-{reference_slug}-{canonical['source_signal_sha256'][:20]}"
    tile_size = max(1, int(round(config.cache_tile_seconds * output_sfreq)))
    processed_hash = deterministic_view_tensor_sha256(tensor, unit_ids=output_ids)
    tiles = _cache_tiles(
        tensor,
        unit_ids=output_ids,
        tile_size=tile_size,
        canonical=canonical,
        view_id=view_id,
        output_rate=output_rate,
        output_sources=output_sources,
        observed_by_unit=observed_by_unit,
        edge_left_invalid_samples=edge_left,
        edge_right_invalid_samples=edge_right,
    )
    receipt = build_signal_view_receipt(
        canonical,
        view_id=view_id,
        task_role="spatial_reference",
        transform_spec=transform,
        output_unit_definitions=definitions,
        selected_global_output_sample_interval=(0, int(tensor.shape[1])),
        processed_view_sha256=processed_hash,
        cache_tile_size_samples=tile_size,
        cache_tiles=tiles,
        parent_views=[parent.receipt],
    )
    return MaterializedCanonicalEEGView(tensor=tensor, receipt=receipt)


def _materialized_view_binding(
    view: MaterializedCanonicalEEGView,
) -> dict[str, object]:
    return {
        "view_id": view.receipt["view_id"],
        "view_receipt_id": view.receipt["view_receipt_id"],
        "receipt_sha256": view.receipt["receipt_sha256"],
        "transform_spec_sha256": view.receipt["transform_spec"][
            "transform_spec_sha256"
        ],
        "tensor_sha256": view.receipt["processed_view_sha256"],
        "mask_sha256": view.receipt["masks"]["mask_sha256"],
        "temporal_evidence": deepcopy(view.receipt["temporal_evidence"]),
    }


def _quality_control_qualification(record: CanonicalEEGRecord) -> dict[str, Any]:
    source_by_channel = {
        str(row["channel_id"]): str(row["clipping_qc_source"])
        for row in record.source_header_receipt["channel_signal_headers"]
    }
    body: dict[str, Any] = {
        "schema_version": "clinical_eeg_canonical_qc_qualification_v1",
        "clipping_policy": (
            "edf_header_calibration_rail_or_local_unlabelled_plateau_v1"
        ),
        "full_record_sample_extrema_used": False,
        "clipping_qc_source_by_observed_channel": source_by_channel,
        "onset_causal_support_projection": (
            "exact_raw_interval_right_dilation_by_fir_order_v1"
        ),
        "context_offline_support_projection": (
            "full_selected_fail_closed_for_nonfinite_or_unreceipted_support_v1"
        ),
        "reference_child_policy": "inherit_parent_quality_support_without_shrinkage_v1",
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _offline_edge_support_qualification(
    context_view: MaterializedCanonicalEEGView,
    *,
    configured_guard_seconds: float,
) -> dict[str, Any]:
    transform = context_view.receipt["transform_spec"]
    body: dict[str, Any] = {
        "schema_version": "clinical_eeg_offline_edge_support_qualification_v1",
        "view_id": context_view.receipt["view_id"],
        "view_receipt_sha256": context_view.receipt["receipt_sha256"],
        "transform_spec_sha256": transform["transform_spec_sha256"],
        "filter_family": transform["filter"]["family"],
        "phase_policy": transform["filter"]["phase_policy"],
        "configured_edge_guard_seconds": float(configured_guard_seconds),
        "exact_finite_filter_support_receipted": False,
        "resampler_kernel_support_receipted": (
            transform["resampler"]["implementation"] == "none"
        ),
        "edge_support_qualified": False,
        "onset_evidence_authorized": False,
        "edge_sensitive_morphology_claim_authorized": False,
        "reason_codes": [
            "butterworth_iir_has_no_exact_finite_impulse_support",
            "fixed_edge_guard_not_response_qualified",
            *(
                []
                if transform["resampler"]["implementation"] == "none"
                else ["resample_poly_kernel_support_not_explicitly_receipted"]
            ),
        ],
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _build_materialization_receipt(
    record: CanonicalEEGRecord,
    task_reference_views: Mapping[str, Mapping[str, MaterializedCanonicalEEGView]],
    *,
    config: CanonicalEDFConfig,
) -> dict[str, Any]:
    montage = validate_montage_reference_observability_receipt(
        record.montage_reference_observability_receipt
    )
    source_rate = _canonical_source_rate(record)
    source_sfreq = _rate_hz(source_rate)
    onset_lowpass = min(float(config.onset_lowpass_hz), 0.45 * source_sfreq)
    onset_design_selection = select_onset_causal_fir_design(
        sampling_rate_hz=source_sfreq,
        highpass_hz=float(config.onset_highpass_hz),
        lowpass_hz=onset_lowpass,
        requested_numtaps=config.onset_fir_numtaps,
        selection_policy_id=config.onset_fir_numtaps_policy,
    )
    onset_qualification = qualify_onset_causal_fir_response(
        sampling_rate_hz=source_sfreq,
        highpass_hz=float(config.onset_highpass_hz),
        lowpass_hz=onset_lowpass,
        numtaps=int(onset_design_selection["selected_numtaps"]),
    )
    onset_clinical_admission = qualify_onset_causal_fir_clinical_admission(
        design_selection=onset_design_selection,
        response_qualification=onset_qualification,
    )
    onset_transform = task_reference_views["onset_causal"]["referential"].receipt[
        "transform_spec"
    ]
    if (
        onset_transform["software_versions"].get("fir_response_qualification_sha256")
        != onset_qualification["receipt_sha256"]
    ):
        raise ValueError("onset FIR view and response qualification disagree")
    if (
        onset_transform["software_versions"].get("fir_design_selection_sha256")
        != onset_design_selection["receipt_sha256"]
    ):
        raise ValueError("onset FIR view and design selection disagree")
    if (
        onset_transform["software_versions"].get(
            "fir_clinical_admission_qualification_sha256"
        )
        != onset_clinical_admission["receipt_sha256"]
    ):
        raise ValueError("onset FIR view and clinical admission disagree")
    quality_qualification = _quality_control_qualification(record)
    offline_edge_qualification = _offline_edge_support_qualification(
        task_reference_views["context_offline"]["referential"],
        configured_guard_seconds=float(config.edge_guard_seconds),
    )
    config_payload = {
        "output_sampling_rate_hz": config.output_sampling_rate_hz,
        "findings_highpass_hz": config.findings_highpass_hz,
        "findings_lowpass_hz": config.findings_lowpass_hz,
        "onset_highpass_hz": config.onset_highpass_hz,
        "onset_lowpass_hz": config.onset_lowpass_hz,
        "onset_fir_numtaps": config.onset_fir_numtaps,
        "onset_fir_numtaps_policy": config.onset_fir_numtaps_policy,
        "butterworth_order": config.butterworth_order,
        "edge_guard_seconds": config.edge_guard_seconds,
        "cache_tile_seconds": config.cache_tile_seconds,
        "flatline_run_seconds": config.flatline_run_seconds,
        "clipping_run_seconds": config.clipping_run_seconds,
        "qc_tolerance_volts": config.qc_tolerance_volts,
        "minimum_observed_standard19": config.minimum_observed_standard19,
    }
    body: dict[str, Any] = {
        "schema_version": CANONICAL_EDF_MATERIALIZATION_SCHEMA_VERSION,
        "producer_id": CANONICAL_EDF_PRODUCER_ID,
        "source_signal_sha256": record.canonical_receipt["source_signal_sha256"],
        "source_header_receipt_sha256": record.source_header_receipt["receipt_sha256"],
        "source_tensor_sha256": record.source_header_receipt["source_tensor_sha256"],
        "canonical_signal_id": record.canonical_receipt["canonical_signal_id"],
        "canonical_receipt_sha256": record.canonical_receipt["receipt_sha256"],
        "observed_channel_ids": list(record.observed_channel_ids),
        "unobserved_channel_ids": [
            item for item in STANDARD_19 if item not in set(record.observed_channel_ids)
        ],
        "quality_primitive_count": len(record.canonical_receipt["quality_primitives"]),
        "quality_control_qualification": quality_qualification,
        "onset_fir_design_selection": onset_design_selection,
        "onset_fir_response_qualification": onset_qualification,
        "onset_fir_clinical_admission_qualification": onset_clinical_admission,
        "offline_edge_support_qualification": offline_edge_qualification,
        "montage_reference_observability_receipt_sha256": montage["receipt_sha256"],
        "acquisition_montage_class": montage["montage_class"],
        "derived_reference_matrix_sha256_by_kind": {
            kind: montage["derived_reference_contracts"][kind][
                "reference_matrix_observability"
            ]["matrix_sha256"]
            for kind in ("tcp_bipolar", "car", "laplacian")
        },
        "acquisition_reference_by_channel": {
            str(channel["channel_id"]): str(channel["reference_label"])
            for channel in record.canonical_receipt["channels"]
        },
        "reference_interpretation_policy": {
            "referential": "acquisition_reference_lineage_only_v1",
            "tcp_bipolar": "scalp_field_montage_consistency_not_source_localization_v1",
            "car": "montage_robustness_only_not_source_localization_v1",
            "laplacian": "montage_robustness_only_not_source_localization_v1",
        },
        "task_reference_views": {
            task_role: {
                reference_kind: _materialized_view_binding(view)
                for reference_kind, view in reference_views.items()
            }
            for task_role, reference_views in task_reference_views.items()
        },
        # Frozen aliases for v1 consumers.  They bind to context_offline and
        # never imply that the offline carrier is onset-authorized.
        "findings_view": _materialized_view_binding(
            task_reference_views["context_offline"]["referential"]
        ),
        "spatial_view": _materialized_view_binding(
            task_reference_views["context_offline"]["tcp_bipolar"]
        ),
        "preprocessing_config": config_payload,
        "scope_receipt": deepcopy(EEG_ONLY_SCOPE_RECEIPT),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def validate_canonical_edf_materialization(
    bundle: CanonicalEDFViewBundle,
) -> dict[str, Any]:
    """Validate all task/reference tensors and their EEG-only provenance."""

    if not isinstance(bundle, CanonicalEDFViewBundle):
        raise TypeError("bundle must be a CanonicalEDFViewBundle")
    record = bundle.canonical_record
    canonical = validate_canonical_signal_receipt(record.canonical_receipt)
    source_header = _validate_source_header_receipt(record.source_header_receipt)
    if source_header["source_signal_sha256"] != canonical["source_signal_sha256"]:
        raise ValueError("canonical receipt and EDF signal-header hash disagree")
    montage = validate_montage_reference_observability_receipt(
        record.montage_reference_observability_receipt
    )
    if montage["source_signal_sha256"] != canonical["source_signal_sha256"]:
        raise ValueError("montage/reference receipt belongs to another source signal")
    if tuple(montage["direct_electrode_ids"]) != record.observed_channel_ids:
        raise ValueError("montage/reference direct-electrode ledger drifted")
    require_reference_materialization_authorized(montage)
    if tuple(source_header["observed_channel_ids"]) != record.observed_channel_ids:
        raise ValueError("canonical observed channel order drifted")
    actual_source_tensor_hash = _source_tensor_sha256(
        record.observed_signal_volts,
        channel_ids=record.observed_channel_ids,
    )
    if actual_source_tensor_hash != source_header["source_tensor_sha256"]:
        raise ValueError("canonical observed tensor hash drifted")

    task_roles = (
        "findings_native_morphology",
        "onset_causal",
        "context_offline",
    )
    reference_kinds = ("referential", "tcp_bipolar", "car", "laplacian")
    task_bases = {
        "findings_native_morphology": bundle.findings_native_morphology,
        "onset_causal": bundle.onset_causal,
        "context_offline": bundle.context_offline,
    }
    if (
        type(bundle.task_reference_views) is not dict
        or tuple(bundle.task_reference_views) != task_roles
    ):
        raise ValueError("canonical task-view roles are absent or out of frozen order")

    validated_views: dict[str, dict[str, dict[str, Any]]] = {}
    for task_role in task_roles:
        family = bundle.task_reference_views[task_role]
        if type(family) is not dict or tuple(family) != reference_kinds:
            raise ValueError(
                "canonical reference views are absent or out of frozen order"
            )
        if (
            family["referential"].receipt["receipt_sha256"]
            != task_bases[task_role].receipt["receipt_sha256"]
        ):
            raise ValueError(f"{task_role} referential alias drifted")
        parent_view = family["referential"]
        parent = validate_signal_view_receipt(parent_view.receipt, canonical)
        if parent["task_role"] != task_role:
            raise ValueError(f"{task_role} task receipt carries the wrong role")
        parent_ids = [str(row["unit_id"]) for row in parent["output_units"]]
        if (
            deterministic_view_tensor_sha256(parent_view.tensor, unit_ids=parent_ids)
            != parent["processed_view_sha256"]
        ):
            raise ValueError(f"{task_role} referential tensor hash drifted")
        validated_views[task_role] = {"referential": parent}

        for reference_kind in reference_kinds[1:]:
            child_view = family[reference_kind]
            child = validate_signal_view_receipt(
                child_view.receipt,
                canonical,
                trusted_parent_views={parent["view_id"]: parent},
            )
            output_ids, matrix, sources, _, reference_type = _reference_recipe(
                reference_kind
            )
            if child["transform_spec"]["reference"]["matrix"] != matrix:
                raise ValueError(f"{reference_kind} reference matrix drifted")
            if child["transform_spec"]["reference"]["reference_type"] != reference_type:
                raise ValueError(f"{reference_kind} reference policy drifted")
            if tuple(row["unit_id"] for row in child["output_units"]) != output_ids:
                raise ValueError(f"{reference_kind} output order drifted")
            child_ids = [str(row["unit_id"]) for row in child["output_units"]]
            if (
                deterministic_view_tensor_sha256(child_view.tensor, unit_ids=child_ids)
                != child["processed_view_sha256"]
            ):
                raise ValueError(f"{task_role}/{reference_kind} tensor hash drifted")
            expected_tensor = torch.matmul(
                torch.tensor(matrix, dtype=torch.float32),
                parent_view.tensor.detach().cpu().to(torch.float32),
            ).contiguous()
            if not torch.equal(
                expected_tensor,
                child_view.tensor.detach().cpu().to(torch.float32).contiguous(),
            ):
                raise ValueError(
                    f"{task_role}/{reference_kind} tensor does not equal its reference transform"
                )

            qualification = montage["derived_reference_contracts"][reference_kind]
            support_by_unit = {
                str(row["unit_id"]): row for row in qualification["output_support"]
            }
            qualified_matrix = qualification["reference_matrix_observability"]
            if (
                qualified_matrix["row_unit_ids"] != list(output_ids)
                or qualified_matrix["column_unit_ids"] != list(STANDARD_19)
                or qualified_matrix["matrix"] != matrix
            ):
                raise ValueError(
                    f"{task_role}/{reference_kind} montage matrix qualification drifted"
                )
            for unit in child["output_units"]:
                support = support_by_unit[str(unit["unit_id"])]
                carriers = sources[str(unit["unit_id"])]
                if support["quality_dependency_channel_ids"] != carriers:
                    raise ValueError(
                        f"{task_role}/{reference_kind} carrier support drifted"
                    )
                expected_observed = bool(support["evidence_eligible"])
                if bool(unit["observed"]) is not expected_observed:
                    raise ValueError(
                        f"{task_role}/{reference_kind} carrier eligibility drifted"
                    )
                if not expected_observed and bool(unit["evidence_eligible"]):
                    raise ValueError(
                        "missing reference carrier became evidence eligible"
                    )
            validated_views[task_role][reference_kind] = child

    expanded = _expanded_standard19(record)
    if not torch.equal(
        expanded,
        bundle.findings_native_morphology.tensor.detach().cpu().to(torch.float32),
    ):
        raise ValueError("native morphology is not the canonical mother signal")
    if bundle.onset_causal.receipt["temporal_evidence"]["future_sample_access"]:
        raise ValueError("causal onset view declares future-sample access")
    if bundle.context_offline.receipt["temporal_evidence"]["onset_evidence_authorized"]:
        raise ValueError("offline context illegally authorizes onset")
    if (
        bundle.findings_clinical.receipt["receipt_sha256"]
        != bundle.context_offline.receipt["receipt_sha256"]
        or bundle.spatial_bipolar.receipt["receipt_sha256"]
        != bundle.task_reference_views["context_offline"]["tcp_bipolar"].receipt[
            "receipt_sha256"
        ]
    ):
        raise ValueError("legacy canonical-view aliases drifted")
    if not torch.equal(
        bundle.findings_clinical.tensor.detach().cpu().to(torch.float32).contiguous(),
        bundle.context_offline.tensor.detach().cpu().to(torch.float32).contiguous(),
    ):
        raise ValueError("findings tensor hash drifted from context_offline alias")
    if not torch.equal(
        bundle.spatial_bipolar.tensor.detach().cpu().to(torch.float32).contiguous(),
        bundle.task_reference_views["context_offline"]["tcp_bipolar"]
        .tensor.detach()
        .cpu()
        .to(torch.float32)
        .contiguous(),
    ):
        raise ValueError("spatial tensor drifted from context_offline TCP alias")

    receipt = deepcopy(bundle.materialization_receipt)
    required = {
        "schema_version",
        "producer_id",
        "source_signal_sha256",
        "source_header_receipt_sha256",
        "source_tensor_sha256",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "observed_channel_ids",
        "unobserved_channel_ids",
        "quality_primitive_count",
        "quality_control_qualification",
        "onset_fir_response_qualification",
        "offline_edge_support_qualification",
        "montage_reference_observability_receipt_sha256",
        "acquisition_montage_class",
        "derived_reference_matrix_sha256_by_kind",
        "acquisition_reference_by_channel",
        "reference_interpretation_policy",
        "task_reference_views",
        "findings_view",
        "spatial_view",
        "preprocessing_config",
        "scope_receipt",
        "receipt_sha256",
    }
    current_fir_extensions = {
        "onset_fir_design_selection",
        "onset_fir_clinical_admission_qualification",
    }
    if type(receipt) is not dict or frozenset(receipt) not in {
        frozenset(required),
        frozenset(required | current_fir_extensions),
    }:
        raise ValueError(
            "canonical EDF materialization receipt has missing or unknown fields"
        )
    has_design_selection = "onset_fir_design_selection" in receipt
    has_clinical_admission = "onset_fir_clinical_admission_qualification" in receipt
    if has_design_selection is not has_clinical_admission:
        raise ValueError(
            "canonical EDF FIR selection and clinical-admission extensions disagree"
        )
    has_current_fir_extensions = has_design_selection
    if receipt["schema_version"] != CANONICAL_EDF_MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported canonical EDF materialization schema")
    if receipt["producer_id"] != CANONICAL_EDF_PRODUCER_ID:
        raise ValueError("canonical EDF producer ID drifted")
    expected_bindings = {
        "source_signal_sha256": canonical["source_signal_sha256"],
        "source_header_receipt_sha256": source_header["receipt_sha256"],
        "source_tensor_sha256": source_header["source_tensor_sha256"],
        "canonical_signal_id": canonical["canonical_signal_id"],
        "canonical_receipt_sha256": canonical["receipt_sha256"],
        "observed_channel_ids": list(record.observed_channel_ids),
        "unobserved_channel_ids": [
            item for item in STANDARD_19 if item not in set(record.observed_channel_ids)
        ],
        "quality_primitive_count": len(canonical["quality_primitives"]),
        "montage_reference_observability_receipt_sha256": montage["receipt_sha256"],
        "acquisition_montage_class": montage["montage_class"],
        "derived_reference_matrix_sha256_by_kind": {
            kind: montage["derived_reference_contracts"][kind][
                "reference_matrix_observability"
            ]["matrix_sha256"]
            for kind in reference_kinds[1:]
        },
        "acquisition_reference_by_channel": {
            str(channel["channel_id"]): str(channel["reference_label"])
            for channel in canonical["channels"]
        },
        "reference_interpretation_policy": {
            "referential": "acquisition_reference_lineage_only_v1",
            "tcp_bipolar": "scalp_field_montage_consistency_not_source_localization_v1",
            "car": "montage_robustness_only_not_source_localization_v1",
            "laplacian": "montage_robustness_only_not_source_localization_v1",
        },
    }
    for field, expected in expected_bindings.items():
        if receipt[field] != expected:
            raise ValueError(f"canonical EDF materialization {field} drifted")
    expected_quality_qualification = _quality_control_qualification(record)
    if receipt["quality_control_qualification"] != expected_quality_qualification:
        raise ValueError("canonical EDF quality-control qualification drifted")
    expected_task_bindings = {
        task_role: {
            reference_kind: _materialized_view_binding(
                bundle.task_reference_views[task_role][reference_kind]
            )
            for reference_kind in reference_kinds
        }
        for task_role in task_roles
    }
    expected_view_bindings = {
        "task_reference_views": expected_task_bindings,
        "findings_view": expected_task_bindings["context_offline"]["referential"],
        "spatial_view": expected_task_bindings["context_offline"]["tcp_bipolar"],
    }
    for field, expected in expected_view_bindings.items():
        if receipt[field] != expected:
            raise ValueError(f"canonical EDF materialization {field} drifted")
    if receipt["scope_receipt"] != EEG_ONLY_SCOPE_RECEIPT:
        raise ValueError("canonical EDF materialization violates EEG-only scope")
    if not isinstance(receipt["preprocessing_config"], dict):
        raise ValueError("canonical EDF preprocessing config is absent")
    raw_preprocessing_config = receipt["preprocessing_config"]
    if has_current_fir_extensions != (
        "onset_fir_numtaps_policy" in raw_preprocessing_config
    ):
        raise ValueError(
            "canonical EDF FIR selection extension/config version disagree"
        )
    try:
        receipt_config = CanonicalEDFConfig(**raw_preprocessing_config)
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical EDF preprocessing config is invalid") from exc
    onset_order = int(bundle.onset_causal.receipt["transform_spec"]["filter"]["order"])
    source_rate = _canonical_source_rate(record)
    source_sfreq = _rate_hz(source_rate)
    onset_design_lowpass = min(
        float(receipt_config.onset_lowpass_hz), 0.45 * source_sfreq
    )
    expected_onset_selection = select_onset_causal_fir_design(
        sampling_rate_hz=source_sfreq,
        highpass_hz=float(receipt_config.onset_highpass_hz),
        lowpass_hz=onset_design_lowpass,
        requested_numtaps=receipt_config.onset_fir_numtaps,
        selection_policy_id=receipt_config.onset_fir_numtaps_policy,
    )
    selected_numtaps = int(expected_onset_selection["selected_numtaps"])
    if onset_order != selected_numtaps - 1:
        raise ValueError(
            "causal onset filter order is not bound to preprocessing config"
        )
    if has_current_fir_extensions:
        validated_selection = validate_onset_causal_fir_design_selection(
            receipt["onset_fir_design_selection"]
        )
        if validated_selection != expected_onset_selection:
            raise ValueError("canonical EDF onset FIR design selection drifted")
    elif receipt_config.onset_fir_numtaps is None:
        raise ValueError("legacy FIR receipt cannot omit an explicit tap count")
    expected_onset_qualification = qualify_onset_causal_fir_response(
        sampling_rate_hz=source_sfreq,
        highpass_hz=float(receipt_config.onset_highpass_hz),
        lowpass_hz=onset_design_lowpass,
        numtaps=selected_numtaps,
    )
    if receipt["onset_fir_response_qualification"] != expected_onset_qualification:
        raise ValueError("canonical EDF onset FIR response qualification drifted")
    onset_transform = bundle.onset_causal.receipt["transform_spec"]
    measured_bandwidth = expected_onset_qualification["measurement"][
        "measured_minus_3db_bandwidth_hz"
    ]
    expected_onset_transform = build_transform_spec(
        transform_name=CANONICAL_EDF_ONSET_TRANSFORM_NAME,
        input_unit_ids=STANDARD_19,
        output_unit_ids=STANDARD_19,
        source_sampling_rate=source_rate,
        output_sampling_rate=source_rate,
        resample_up=1,
        resample_down=1,
        resampler_implementation="none",
        anti_alias_filter="none",
        anti_alias_lowpass_hz=None,
        filter_family="fir",
        filter_order=onset_order,
        highpass_hz=float(measured_bandwidth[0]),
        lowpass_hz=float(measured_bandwidth[1]),
        phase_policy="causal_with_group_delay_receipt",
        normalization_method="physical_unit_scale_only",
        normalization_source="channel_metadata",
        clipping_applied=False,
        clipping_policy="none",
        clipping_source="none",
        reference_type="source_header_reference_preserved_per_channel_v1",
        reference_matrix=_standard19_identity_matrix(),
        edge_policy="global_recording_edges",
        edge_left_invalid_samples=onset_order,
        edge_right_invalid_samples=0,
        software_versions=onset_transform["software_versions"],
    )
    if onset_transform != expected_onset_transform:
        raise ValueError(
            "causal onset transform does not replay the frozen standard-19 recipe"
        )
    _require_onset_causal_tensor_replay(
        record,
        bundle.onset_causal.tensor,
        sampling_rate_hz=source_sfreq,
        highpass_hz=float(receipt_config.onset_highpass_hz),
        lowpass_hz=onset_design_lowpass,
        numtaps=selected_numtaps,
    )
    response_band_authorized = bool(
        expected_onset_qualification["target_band_claim_authorized"]
    )
    if has_current_fir_extensions:
        expected_clinical_admission = qualify_onset_causal_fir_clinical_admission(
            design_selection=expected_onset_selection,
            response_qualification=expected_onset_qualification,
        )
        validated_clinical_admission = (
            validate_onset_causal_fir_clinical_admission_qualification(
                receipt["onset_fir_clinical_admission_qualification"]
            )
        )
        if validated_clinical_admission != expected_clinical_admission:
            raise ValueError(
                "canonical EDF onset FIR clinical admission qualification drifted"
            )
        expected_onset_authorized = bool(
            expected_clinical_admission["clinical_onset_support_authorized"]
        )
    else:
        # The only supported pre-extension audit path is an explicitly
        # configured FIR that was already response-unqualified (notably the
        # archived 101-tap path).  A legacy response-qualified artifact would
        # otherwise regain positive clinical permission without the new gate.
        if response_band_authorized:
            raise ValueError(
                "legacy response-qualified FIR lacks clinical admission qualification"
            )
        expected_clinical_admission = None
        expected_onset_authorized = False
    onset_temporal = bundle.onset_causal.receipt["temporal_evidence"]
    if (
        bool(onset_temporal["onset_evidence_authorized"])
        is not expected_onset_authorized
    ):
        raise ValueError(
            "causal onset evidence authorization disagrees with clinical admission"
        )
    if expected_onset_authorized:
        expected_authorization_reasons = []
    elif not response_band_authorized:
        expected_authorization_reasons = [ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE]
    else:
        expected_authorization_reasons = [
            ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE
        ]
    if onset_temporal["authorization_reason_codes"] != expected_authorization_reasons:
        raise ValueError(
            "causal onset evidence denial reason disagrees with FIR qualification"
        )
    expected_software_authorization = "true" if response_band_authorized else "false"
    if (
        onset_transform["software_versions"].get(
            ONSET_FIR_RESPONSE_AUTHORIZATION_SOFTWARE_KEY
        )
        != expected_software_authorization
    ):
        raise ValueError(
            "causal onset transform lost its FIR response authorization binding"
        )
    if (
        onset_transform["software_versions"].get("fir_response_qualification_sha256")
        != expected_onset_qualification["receipt_sha256"]
    ):
        raise ValueError("causal onset transform lost its FIR qualification binding")
    clinical_marker = onset_transform["software_versions"].get(
        ONSET_FIR_CLINICAL_ADMISSION_AUTHORIZATION_SOFTWARE_KEY
    )
    clinical_receipt_hash = onset_transform["software_versions"].get(
        "fir_clinical_admission_qualification_sha256"
    )
    if has_current_fir_extensions:
        expected_clinical_marker = "true" if expected_onset_authorized else "false"
        if clinical_marker != expected_clinical_marker:
            raise ValueError(
                "causal onset transform lost its clinical admission authorization binding"
            )
        assert expected_clinical_admission is not None
        if clinical_receipt_hash != expected_clinical_admission["receipt_sha256"]:
            raise ValueError(
                "causal onset transform lost its clinical admission qualification binding"
            )
    elif clinical_marker is not None or clinical_receipt_hash is not None:
        raise ValueError("legacy FIR transform carries a partial clinical extension")
    expected_selection_software: dict[str, str] = {}
    if has_current_fir_extensions:
        expected_selection_software = {
            "fir_design_selection_policy": expected_onset_selection[
                "selection_policy_id"
            ],
            "fir_design_selection_mode": expected_onset_selection["selection_mode"],
            "fir_design_selection_sha256": expected_onset_selection["receipt_sha256"],
            "fir_requested_numtaps": (
                "auto"
                if expected_onset_selection["requested_numtaps"] is None
                else str(expected_onset_selection["requested_numtaps"])
            ),
            "fir_selected_numtaps": str(selected_numtaps),
            "fir_coefficients_sha256": expected_onset_selection["impulse_support"][
                "coefficient_sha256"
            ],
            "fir_impulse_support_offset_samples": f"{-onset_order},0",
            "fir_nonzero_support_offset_samples": (
                f"{expected_onset_selection['impulse_support']['raw_nonzero_support_offset_samples'][0]},"
                f"{expected_onset_selection['impulse_support']['raw_nonzero_support_offset_samples'][1]}"
            ),
            "fir_impulse_support_span_samples": str(onset_order),
            "fir_impulse_support_span_seconds": f"{onset_order / source_sfreq:.17g}",
        }
        for key, expected in expected_selection_software.items():
            if onset_transform["software_versions"].get(key) != expected:
                raise ValueError(
                    "causal onset transform lost its FIR selection/support binding"
                )
    expected_software_versions = {
        "producer": CANONICAL_EDF_PRODUCER_ID,
        "fir_design": "scipy.signal.firwin_hamming_scale_true_v1",
        "fir_design_target_band_hz": (
            f"{float(receipt_config.onset_highpass_hz):.17g},"
            f"{onset_design_lowpass:.17g}"
        ),
        ONSET_FIR_RESPONSE_AUTHORIZATION_SOFTWARE_KEY: (
            expected_software_authorization
        ),
        "fir_response_qualification_sha256": expected_onset_qualification[
            "receipt_sha256"
        ],
        "causal_apply": "scipy.signal.lfilter_zero_initial_state_v1",
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "torch": torch.__version__,
    }
    if has_current_fir_extensions:
        assert expected_clinical_admission is not None
        expected_software_versions.update(expected_selection_software)
        expected_software_versions.update(
            {
                ONSET_FIR_CLINICAL_ADMISSION_AUTHORIZATION_SOFTWARE_KEY: (
                    "true" if expected_onset_authorized else "false"
                ),
                "fir_clinical_admission_qualification_sha256": (
                    expected_clinical_admission["receipt_sha256"]
                ),
            }
        )
    if onset_transform["software_versions"] != expected_software_versions:
        raise ValueError("causal onset transform software semantics do not replay")
    onset_temporal_support = bundle.onset_causal.receipt["temporal_evidence"]
    if (
        int(onset_temporal_support["warm_up_samples"]) != onset_order
        or float(onset_temporal_support["group_delay_samples"]) != onset_order / 2.0
        or int(onset_temporal_support["latest_raw_support_offset_samples"]) != 0
        or bundle.onset_causal.receipt["masks"]["edge_invalid_intervals"]
        != [[0, onset_order]]
    ):
        raise ValueError("causal onset temporal/QC support drifted from FIR order")
    if float(onset_transform["filter"]["highpass_hz"]) != float(
        measured_bandwidth[0]
    ) or float(onset_transform["filter"]["lowpass_hz"]) != float(measured_bandwidth[1]):
        raise ValueError(
            "causal onset transform reports nominal rather than measured bandwidth"
        )
    expected_offline_edge_qualification = _offline_edge_support_qualification(
        bundle.context_offline,
        configured_guard_seconds=float(receipt_config.edge_guard_seconds),
    )
    if (
        receipt["offline_edge_support_qualification"]
        != expected_offline_edge_qualification
    ):
        raise ValueError("canonical EDF offline edge-support qualification drifted")
    if receipt["offline_edge_support_qualification"][
        "edge_sensitive_morphology_claim_authorized"
    ]:
        raise ValueError("unqualified fixed offline guard authorized morphology")
    if not _is_sha256(receipt["receipt_sha256"]):
        raise ValueError("canonical EDF materialization receipt hash is invalid")
    digest_source = deepcopy(receipt)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if receipt["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("canonical EDF materialization hash does not bind content")
    return receipt


def load_canonical_edf_views(
    edf_path: str | Path,
    *,
    config: CanonicalEDFConfig = CanonicalEDFConfig(),
    reader_factory: Callable[[str], object] | None = None,
) -> CanonicalEDFViewBundle:
    """Load raw EDF EEG into one mother tensor and three task-view families.

    No file path or EDF identity/clinical header value is persisted.  The
    content identity is derived only from selected physical EEG signal headers
    and samples, so changing EDF+ annotations or patient-header text cannot
    change the canonical signal or view hashes.
    """

    record = load_canonical_edf_record(
        edf_path,
        config=config,
        reader_factory=reader_factory,
    )
    # A direct electrode tensor is not enough to authorize a second field.
    # Mixed, already-bipolar, and unknown acquisition references fail here
    # before any CAR/Laplacian/TCP tensor is constructed.
    require_reference_materialization_authorized(
        record.montage_reference_observability_receipt
    )
    native = _materialize_native_morphology_view(record, config=config)
    onset = _materialize_onset_causal_view(record, config=config)
    context = _materialize_context_offline_view(record, config=config)
    task_bases = {
        "findings_native_morphology": native,
        "onset_causal": onset,
        "context_offline": context,
    }
    task_reference_views: dict[str, dict[str, MaterializedCanonicalEEGView]] = {}
    for task_role, parent in task_bases.items():
        task_reference_views[task_role] = {
            "referential": parent,
            "tcp_bipolar": _materialize_reference_view(
                record,
                parent,
                reference_kind="tcp_bipolar",
                config=config,
            ),
            "car": _materialize_reference_view(
                record,
                parent,
                reference_kind="car",
                config=config,
            ),
            "laplacian": _materialize_reference_view(
                record,
                parent,
                reference_kind="laplacian",
                config=config,
            ),
        }
    receipt = _build_materialization_receipt(
        record,
        task_reference_views,
        config=config,
    )
    bundle = CanonicalEDFViewBundle(
        canonical_record=record,
        findings_native_morphology=native,
        onset_causal=onset,
        context_offline=context,
        task_reference_views=task_reference_views,
        findings_clinical=context,
        spatial_bipolar=task_reference_views["context_offline"]["tcp_bipolar"],
        materialization_receipt=receipt,
    )
    validate_canonical_edf_materialization(bundle)
    return bundle


def load_canonical_edf_record(
    edf_path: str | Path,
    *,
    config: CanonicalEDFConfig = CanonicalEDFConfig(),
    reader_factory: Callable[[str], object] | None = None,
) -> CanonicalEEGRecord:
    """Load only the immutable physical EEG root, without task views.

    Detector providers need the same canonical signal identity and physical
    channel ledger as Findings, but must retain their checkpoint-native
    transforms.  This entry point therefore stops before morphology, causal,
    offline-context, or montage views are materialized.  It has the same
    signal-only EDF API and input firewall as :func:`load_canonical_edf_views`.
    """

    record = _read_canonical_record(
        edf_path,
        config=config,
        reader_factory=reader_factory,
    )
    if not isinstance(record, CanonicalEEGRecord):  # defensive type narrowing
        raise AssertionError("full canonical EDF loader returned a source-only carrier")
    return record


def load_canonical_edf_physical_source_identity(
    edf_path: str | Path,
    *,
    config: CanonicalEDFConfig = CanonicalEDFConfig(),
    reader_factory: Callable[[str], object] | None = None,
) -> CanonicalEDFPhysicalSourceIdentity:
    """Load the canonical tensor/header identity without deriving QC or views.

    The function is restricted to corpus identity/duplicate audits.  It does
    not authorize Findings, onset timing, quality assertions, or model input.
    Its source tensor and header hashes are byte-identical to those returned
    by :func:`load_canonical_edf_record` for the same immutable EDF.
    """

    identity = _read_canonical_record(
        edf_path,
        config=config,
        reader_factory=reader_factory,
        source_identity_only=True,
    )
    if not isinstance(identity, CanonicalEDFPhysicalSourceIdentity):
        raise AssertionError("source-only canonical EDF loader returned a full record")
    return identity


__all__ = [
    "CANONICAL_EDF_MATERIALIZATION_SCHEMA_VERSION",
    "CANONICAL_EDF_PRODUCER_ID",
    "CANONICAL_EDF_SOURCE_HEADER_SCHEMA_VERSION",
    "CANONICAL_SOURCE_TENSOR_HASH_DOMAIN",
    "EEG_ONLY_SCOPE_RECEIPT",
    "ONSET_FIR_AUTO_SELECTION_POLICY_ID",
    "ONSET_FIR_DESIGN_SELECTION_SCHEMA_VERSION",
    "ONSET_FIR_RESPONSE_QUALIFICATION_SCHEMA_VERSION",
    "CanonicalEDFConfig",
    "CanonicalEDFPhysicalSourceIdentity",
    "CanonicalEDFViewBundle",
    "CanonicalEEGRecord",
    "MaterializedCanonicalEEGView",
    "canonical_source_tensor_sha256",
    "load_canonical_edf_record",
    "load_canonical_edf_physical_source_identity",
    "load_canonical_edf_views",
    "qualify_onset_causal_fir_response",
    "select_onset_causal_fir_design",
    "validate_canonical_edf_materialization",
    "validate_canonical_edf_source_header_receipt",
    "validate_onset_causal_fir_design_selection",
    "validate_onset_causal_fir_response_qualification",
]
