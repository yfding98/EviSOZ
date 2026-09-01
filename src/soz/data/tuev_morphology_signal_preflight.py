"""First-party, replayable TUEV morphology header and signal preflight.

This module is the deterministic producer for the closed external-metadata
schema consumed by :mod:`src.soz.data.tuev_morphology`.  It reads the actual
EDF payload and the complete EDF/REC/LAB/HTK parent groups.  In particular,
no caller-provided channel, unit, sample-count, or QC summary is accepted.

The formal path is intentionally Linux/``pyedflib`` specific.  Each EDF is
opened with ``O_NOFOLLOW`` and held by descriptor while ``pyedflib`` reads the
same inode through ``/proc/self/fd``.  File identity and bytes are checked
before and after the header/signal pass, and the complete TUEV source roster is
rediscovered after all records have been inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np

from ..geometry import LEGACY_TO_CANONICAL, STANDARD_19, normalize_electrode_name
from .edf import CausalEDFConfig
from .tuev import parse_tuev_rec
from .tuev_morphology import (
    MORPHOLOGY_ALIGNMENT_TOLERANCE_SEC,
    MORPHOLOGY_CONTEXT_SAMPLES,
    MORPHOLOGY_DURATION_TOLERANCE_SEC,
    MORPHOLOGY_OUTPUT_SFREQ_HZ,
    MORPHOLOGY_WARMUP_SAMPLES,
    TUEV_MORPHOLOGY_EXTERNAL_METADATA_SCHEMA,
    TUEVExactSignalDuplicateLedger,
    TUEVMorphologyRecordMetadata,
    TUEVMorphologySourceRecord,
    build_tuev_exact_signal_duplicate_ledger,
    discover_tuev_morphology_sources,
)


TUEV_MORPHOLOGY_PRODUCER_SCHEMA = (
    "tuev_morphology_first_party_signal_metadata_producer_v2"
)
TUEV_MORPHOLOGY_PREPROCESSING_POLICY_SCHEMA = (
    "tuev_morphology_causal_preprocessing_preflight_policy_v1"
)
TUEV_MORPHOLOGY_STANDARD19_MAPPING_POLICY_SCHEMA = (
    "tuev_morphology_direct_physical_standard19_mapping_policy_v1"
)
TUEV_MORPHOLOGY_SIGNAL_QC_SCHEMA = "tuev_morphology_raw_signal_qc_v2"
TUEV_MORPHOLOGY_TARGET_BOUNDS_SCHEMA = "tuev_morphology_target_bounds_v1"

_REFERENCE_PATTERN = re.compile(r"-(REF|LE|AR|AVG|AV|CAR)$", re.IGNORECASE)
_UNIT_TO_VOLTS = {"v": 1.0, "mv": 1e-3, "uv": 1e-6}
_GAP_WORDS = ("boundary", "discont", "gap")
_QC_CHUNK_SAMPLES = 65_536
_MAX_EDF_BYTES = 32 * 1024 * 1024 * 1024
_EXPECTED_DERIVATIVE_INDICES = tuple(range(22))
_SOURCE_FILES = (
    "scripts/build_tuev_morphology_preflight.py",
    "scripts/produce_tuev_morphology_external_metadata.py",
    "src/soz/data/edf.py",
    "src/soz/data/tuev.py",
    "src/soz/data/tuev_morphology.py",
    "src/soz/data/tuev_morphology_signal_preflight.py",
    "src/soz/geometry.py",
    "src/soz/signal.py",
)


class TUEVMorphologySignalPreflightError(ValueError):
    """Fail-closed error raised for an inadmissible real TUEV source."""

    def __init__(self, code: str, message: str) -> None:
        normalized = str(code).strip()
        if not normalized or not isinstance(message, str) or not message.strip():
            raise ValueError("Preflight errors require a code and message")
        self.code = normalized
        self.detail = message.strip()
        super().__init__(f"{normalized}: {self.detail}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"Producer source changed while hashing: {path}")
    return digest.hexdigest()


def _producer_root() -> Path:
    root = Path(__file__).resolve(strict=True).parents[3]
    if root.name == "" or not root.is_dir():  # pragma: no cover - invariant
        raise RuntimeError("Cannot resolve the producer repository root")
    return root


def tuev_morphology_producer_source_roster() -> tuple[tuple[str, str], ...]:
    """Return the closed implementation-source roster used by the producer."""

    root = _producer_root()
    rows: list[tuple[str, str]] = []
    for relative in _SOURCE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            raise RuntimeError(f"Producer source is absent or non-canonical: {relative}")
        rows.append((relative, _file_sha256(path)))
    return tuple(rows)


def tuev_morphology_producer_source_sha256() -> str:
    return _canonical_sha256(
        {
            "schema_version": TUEV_MORPHOLOGY_PRODUCER_SCHEMA,
            "source_files": tuev_morphology_producer_source_roster(),
        }
    )


def _dependency_version(distribution: str) -> str:
    try:
        value = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required producer dependency is absent: {distribution}") from exc
    if not value or value != value.strip():
        raise RuntimeError(f"Invalid dependency version for {distribution}")
    return value


def tuev_morphology_preprocessing_policy() -> dict[str, object]:
    """Return the exact raw-QC, bounds, and causal preprocessing policy."""

    config = CausalEDFConfig()
    return {
        "schema_version": TUEV_MORPHOLOGY_PREPROCESSING_POLICY_SCHEMA,
        "reader": "pyedflib.EdfReader_via_held_O_NOFOLLOW_fd",
        "pyedflib_version": _dependency_version("pyEDFlib"),
        "numpy_version": _dependency_version("numpy"),
        "output_sfreq_hz": MORPHOLOGY_OUTPUT_SFREQ_HZ,
        "output_count_rule": "ceil(source_samples*200/source_sfreq)",
        "highpass_hz": config.highpass_hz,
        "lowpass_hz": config.lowpass_hz,
        "butterworth_order": config.butterworth_order,
        "causal_filter": True,
        "warmup_samples": MORPHOLOGY_WARMUP_SAMPLES,
        "context_samples": MORPHOLOGY_CONTEXT_SAMPLES,
        "duration_tolerance_sec": MORPHOLOGY_DURATION_TOLERANCE_SEC,
        "alignment_tolerance_sec": MORPHOLOGY_ALIGNMENT_TOLERANCE_SEC,
        "target_slot": 0,
        "flatline_run_sec": config.flatline_run_sec,
        "adc_extreme_run_sec": config.clipping_run_sec,
        "qc_tolerance_volts": config.qc_tolerance_volts,
        "qc_scope": "complete_selected_physical_record_before_filter_or_CAR",
        "gap_annotation_words": _GAP_WORDS,
        "signal_chunk_samples": _QC_CHUNK_SAMPLES,
        "apply_car19_after_raw_qc": config.apply_car19,
        "reference_policy": config.reference_policy,
    }


def tuev_morphology_preprocessing_policy_sha256() -> str:
    return _canonical_sha256(tuev_morphology_preprocessing_policy())


def tuev_morphology_standard19_mapping_policy() -> dict[str, object]:
    """Return the direct-physical channel and unit mapping policy."""

    return {
        "schema_version": TUEV_MORPHOLOGY_STANDARD19_MAPPING_POLICY_SCHEMA,
        "semantic_order": STANDARD_19,
        "identity_aliases": tuple(sorted(LEGACY_TO_CANONICAL.items())),
        "mapping_cardinality": "exactly_one_raw_signal_per_semantic_channel",
        "source_reference": "REF",
        "accepted_voltage_units": ("V", "mV", "uV", "micro-sign-V"),
        "bipolar_inversion": "forbidden",
        "missing_or_duplicate_policy": "fail_closed",
        "extra_nonstandard_signals": "ignored_after_full_header_binding",
    }


def tuev_morphology_standard19_mapping_policy_sha256() -> str:
    return _canonical_sha256(tuev_morphology_standard19_mapping_policy())


def first_party_tuev_morphology_bindings() -> dict[str, str]:
    return {
        "producer_source_sha256": tuev_morphology_producer_source_sha256(),
        "preprocessing_policy_sha256": (
            tuev_morphology_preprocessing_policy_sha256()
        ),
        "standard19_mapping_policy_sha256": (
            tuev_morphology_standard19_mapping_policy_sha256()
        ),
    }


def require_first_party_tuev_morphology_bindings(
    *,
    producer_source_sha256: str,
    preprocessing_policy_sha256: str,
    standard19_mapping_policy_sha256: str,
) -> None:
    expected = first_party_tuev_morphology_bindings()
    observed = {
        "producer_source_sha256": producer_source_sha256,
        "preprocessing_policy_sha256": preprocessing_policy_sha256,
        "standard19_mapping_policy_sha256": standard19_mapping_policy_sha256,
    }
    if observed != expected:
        raise ValueError(
            "Formal TUEV morphology preflight requires the current first-party "
            f"producer/policy/mapping bindings; expected={expected}"
        )


def _decode_header_text(value: object, *, field: str) -> str:
    if isinstance(value, bytes):
        try:
            text = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise TUEVMorphologySignalPreflightError(
                "invalid_header", f"{field} is not ASCII"
            ) from exc
    else:
        text = str(value)
    text = text.strip()
    if not text or any(ord(character) < 32 for character in text):
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", f"{field} is empty or contains control characters"
        )
    return text


def _unit_scale(unit: str) -> tuple[str, float]:
    normalized = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    try:
        return normalized, _UNIT_TO_VOLTS[normalized]
    except KeyError as exc:
        raise TUEVMorphologySignalPreflightError(
            "invalid_unit", f"unsupported or non-voltage EDF unit {unit!r}"
        ) from exc


def _reference_suffix(raw_name: str) -> str | None:
    match = _REFERENCE_PATTERN.search(raw_name.strip().upper().replace("_", "-"))
    return None if match is None else match.group(1).upper()


def _finite_number(value: object, *, field: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", f"{field} is not numeric"
        ) from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", f"{field} is not finite and valid"
        )
    return result


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", f"{field} is not an integer"
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", f"{field} is not an integer"
        ) from exc
    try:
        exact = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", f"{field} is not numeric"
        ) from exc
    if result <= 0 or not math.isfinite(exact) or exact != result:
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", f"{field} is not a positive integer"
        )
    return result


@dataclass(frozen=True)
class _HeaderSnapshot:
    labels: tuple[str, ...]
    all_sample_counts: tuple[int, ...]
    selected_indices: tuple[int, ...]
    selected_names: tuple[str, ...]
    source_units: tuple[str, ...]
    canonical_units: tuple[str, ...]
    unit_scales: tuple[float, ...]
    source_sfreq_hz: float
    source_sample_count: int
    physical_minimum: tuple[float, ...]
    physical_maximum: tuple[float, ...]
    digital_minimum: tuple[int, ...]
    digital_maximum: tuple[int, ...]

    @property
    def mapping_payload(self) -> dict[str, object]:
        rows = []
        for semantic, index, raw_name, source_unit, canonical_unit in zip(
            STANDARD_19,
            self.selected_indices,
            self.selected_names,
            self.source_units,
            self.canonical_units,
        ):
            rows.append(
                {
                    "semantic_channel": semantic,
                    "raw_signal_index": index,
                    "raw_signal_name": raw_name,
                    "source_reference": "REF",
                    "source_unit": source_unit,
                    "canonical_unit": canonical_unit,
                }
            )
        return {
            "schema_version": TUEV_MORPHOLOGY_STANDARD19_MAPPING_POLICY_SCHEMA,
            "all_raw_signal_labels": self.labels,
            "selected_channels": rows,
            "source_sfreq_hz": self.source_sfreq_hz,
            "source_sample_count": self.source_sample_count,
        }


def _read_header(reader: object) -> _HeaderSnapshot:
    try:
        labels = tuple(
            _decode_header_text(value, field="signal label")
            for value in reader.getSignalLabels()
        )
        raw_counts = tuple(reader.getNSamples())
    except TUEVMorphologySignalPreflightError:
        raise
    except Exception as exc:
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", "EDF signal labels/sample counts cannot be read"
        ) from exc
    if not labels or len(labels) != len(raw_counts):
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", "EDF label and sample-count rosters disagree"
        )
    declared_count = getattr(reader, "signals_in_file", len(labels))
    if int(declared_count) != len(labels):
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", "EDF signals_in_file disagrees with its label roster"
        )
    all_counts = tuple(
        _positive_integer(value, field=f"sample_count[{index}]")
        for index, value in enumerate(raw_counts)
    )
    candidates: dict[str, list[int]] = {channel: [] for channel in STANDARD_19}
    for index, label in enumerate(labels):
        canonical = normalize_electrode_name(label)
        if canonical in candidates:
            candidates[canonical].append(index)
    missing = tuple(channel for channel, indices in candidates.items() if not indices)
    duplicates = {
        channel: tuple(labels[index] for index in indices)
        for channel, indices in candidates.items()
        if len(indices) > 1
    }
    if missing or duplicates:
        raise TUEVMorphologySignalPreflightError(
            "ambiguous_standard19",
            f"direct physical standard-19 is incomplete/ambiguous; "
            f"missing={missing}, duplicates={duplicates}",
        )
    indices = tuple(candidates[channel][0] for channel in STANDARD_19)
    selected_names = tuple(labels[index] for index in indices)
    bad_reference = {
        semantic: raw
        for semantic, raw in zip(STANDARD_19, selected_names)
        if _reference_suffix(raw) != "REF"
    }
    if bad_reference:
        raise TUEVMorphologySignalPreflightError(
            "nonphysical_reference",
            f"selected channels must be direct physical -REF signals: {bad_reference}",
        )
    try:
        sampling_rates = tuple(
            _finite_number(
                reader.getSampleFrequency(index),
                field=f"sample_frequency[{index}]",
                positive=True,
            )
            for index in indices
        )
        source_units = tuple(
            _decode_header_text(
                reader.getPhysicalDimension(index), field=f"physical_unit[{index}]"
            )
            for index in indices
        )
        unit_rows = tuple(_unit_scale(unit) for unit in source_units)
        physical_minimum = tuple(
            _finite_number(
                reader.getPhysicalMinimum(index), field=f"physical_minimum[{index}]"
            )
            for index in indices
        )
        physical_maximum = tuple(
            _finite_number(
                reader.getPhysicalMaximum(index), field=f"physical_maximum[{index}]"
            )
            for index in indices
        )
        digital_minimum = tuple(
            int(reader.getDigitalMinimum(index)) for index in indices
        )
        digital_maximum = tuple(
            int(reader.getDigitalMaximum(index)) for index in indices
        )
    except TUEVMorphologySignalPreflightError:
        raise
    except Exception as exc:
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", "selected EDF signal metadata cannot be read"
        ) from exc
    source_sfreq = sampling_rates[0]
    if any(abs(value - source_sfreq) > 1e-9 for value in sampling_rates):
        raise TUEVMorphologySignalPreflightError(
            "mixed_sfreq", "selected standard-19 signals use mixed sampling rates"
        )
    selected_counts = tuple(all_counts[index] for index in indices)
    if len(set(selected_counts)) != 1:
        raise TUEVMorphologySignalPreflightError(
            "mixed_sample_count", "selected standard-19 signals have unequal lengths"
        )
    if source_sfreq <= 2.0 * CausalEDFConfig().lowpass_hz:
        raise TUEVMorphologySignalPreflightError(
            "invalid_sfreq", "source Nyquist does not support the frozen 45-Hz lowpass"
        )
    if any(low >= high for low, high in zip(physical_minimum, physical_maximum)):
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", "EDF physical minimum/maximum are not ordered"
        )
    if any(low >= high for low, high in zip(digital_minimum, digital_maximum)):
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", "EDF digital minimum/maximum are not ordered"
        )
    return _HeaderSnapshot(
        labels=labels,
        all_sample_counts=all_counts,
        selected_indices=indices,
        selected_names=selected_names,
        source_units=source_units,
        canonical_units=tuple(row[0] for row in unit_rows),
        unit_scales=tuple(row[1] for row in unit_rows),
        source_sfreq_hz=source_sfreq,
        source_sample_count=selected_counts[0],
        physical_minimum=physical_minimum,
        physical_maximum=physical_maximum,
        digital_minimum=digital_minimum,
        digital_maximum=digital_maximum,
    )


def _annotation_snapshot(reader: object) -> tuple[tuple[float, float, str], ...]:
    if not hasattr(reader, "readAnnotations"):
        return ()
    try:
        onsets, durations, descriptions = reader.readAnnotations()
    except NotImplementedError:
        return ()
    except Exception as exc:
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", "EDF annotations cannot be read"
        ) from exc
    if not (len(onsets) == len(durations) == len(descriptions)):
        raise TUEVMorphologySignalPreflightError(
            "invalid_header", "EDF annotation columns have unequal lengths"
        )
    rows: list[tuple[float, float, str]] = []
    for index, (onset, duration, description) in enumerate(
        zip(onsets, durations, descriptions)
    ):
        start = _finite_number(onset, field=f"annotation_onset[{index}]")
        span = _finite_number(duration, field=f"annotation_duration[{index}]")
        if start < 0 or span < 0:
            raise TUEVMorphologySignalPreflightError(
                "invalid_header", "EDF annotation onset/duration cannot be negative"
            )
        text = _decode_header_text(description, field=f"annotation_text[{index}]")
        rows.append((start, span, text))
    return tuple(rows)


def _longest_true_run_with_prefix(
    flags: np.ndarray, previous_suffix: int
) -> tuple[int, int]:
    values = np.asarray(flags, dtype=bool).reshape(-1)
    if values.size == 0:
        return previous_suffix, previous_suffix
    false_indices = np.flatnonzero(~values)
    if false_indices.size == 0:
        total = previous_suffix + int(values.size)
        return total, total
    prefix = int(false_indices[0])
    suffix = int(values.size - 1 - false_indices[-1])
    if false_indices.size > 1:
        internal = int(np.max(np.diff(false_indices) - 1))
    else:
        internal = 0
    return max(previous_suffix + prefix, internal, suffix), suffix


def _read_signal_chunk(
    reader: object, channel_index: int, start: int, count: int
) -> np.ndarray:
    try:
        payload = reader.readSignal(channel_index, start, count, digital=True)
    except TypeError:
        # Test doubles may expose only the historical positional signature.
        payload = reader.readSignal(channel_index, start, count)
    except Exception as exc:
        raise TUEVMorphologySignalPreflightError(
            "signal_read", "EDF digital signal payload cannot be read"
        ) from exc
    try:
        values = np.asarray(payload, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TUEVMorphologySignalPreflightError(
            "signal_shape", "EDF reader returned a nonnumeric signal payload"
        ) from exc
    if values.ndim != 1 or values.size != count:
        raise TUEVMorphologySignalPreflightError(
            "signal_shape", "EDF reader returned an incomplete signal chunk"
        )
    return values


def _scan_signal_qc(
    reader: object, header: _HeaderSnapshot
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    config = CausalEDFConfig()
    flatline_limit = int(
        Decimal(str(config.flatline_run_sec * header.source_sfreq_hz)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    clipping_limit = int(
        Decimal(str(config.clipping_run_sec * header.source_sfreq_hz)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    if flatline_limit < 2 or clipping_limit < 1:
        raise RuntimeError("Frozen QC duration thresholds produced invalid samples")
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for semantic, raw_index, raw_name, source_unit, unit_scale, pmin, pmax, dmin, dmax in zip(
        STANDARD_19,
        header.selected_indices,
        header.selected_names,
        header.source_units,
        header.unit_scales,
        header.physical_minimum,
        header.physical_maximum,
        header.digital_minimum,
        header.digital_maximum,
    ):
        previous_value: float | None = None
        flat_edge_suffix = 0
        max_flat_edges = 0
        clipping_suffix = 0
        max_clipping_run = 0
        minimum_volts = math.inf
        maximum_volts = -math.inf
        offset = 0
        digital_span = dmax - dmin
        physical_span = pmax - pmin
        while offset < header.source_sample_count:
            count = min(_QC_CHUNK_SAMPLES, header.source_sample_count - offset)
            digital = _read_signal_chunk(reader, raw_index, offset, count)
            if not np.isfinite(digital).all():
                raise TUEVMorphologySignalPreflightError(
                    "nonfinite_signal", f"{semantic} contains NaN or Inf"
                )
            if np.any(digital < dmin) or np.any(digital > dmax):
                raise TUEVMorphologySignalPreflightError(
                    "digital_range", f"{semantic} exceeds declared ADC limits"
                )
            if not np.all(np.equal(digital, np.rint(digital))):
                raise TUEVMorphologySignalPreflightError(
                    "digital_range", f"{semantic} digital payload is not integral"
                )
            physical_source_unit = (
                (digital - float(dmin)) * physical_span / float(digital_span) + pmin
            )
            volts = physical_source_unit * unit_scale
            if not np.isfinite(volts).all():
                raise TUEVMorphologySignalPreflightError(
                    "nonfinite_signal", f"{semantic} conversion produced NaN or Inf"
                )
            minimum_volts = min(minimum_volts, float(np.min(volts)))
            maximum_volts = max(maximum_volts, float(np.max(volts)))

            if previous_value is None:
                flat_edges = np.abs(np.diff(volts)) <= config.qc_tolerance_volts
            else:
                first = np.asarray(
                    [abs(float(volts[0]) - previous_value) <= config.qc_tolerance_volts],
                    dtype=bool,
                )
                flat_edges = np.concatenate(
                    (first, np.abs(np.diff(volts)) <= config.qc_tolerance_volts)
                )
            longest, flat_edge_suffix = _longest_true_run_with_prefix(
                flat_edges, flat_edge_suffix
            )
            max_flat_edges = max(max_flat_edges, longest)
            clipped = (digital == dmin) | (digital == dmax)
            longest_clip, clipping_suffix = _longest_true_run_with_prefix(
                clipped, clipping_suffix
            )
            max_clipping_run = max(max_clipping_run, longest_clip)
            previous_value = float(volts[-1])
            offset += count
        max_flat_samples = max_flat_edges + 1
        flatline_pass = max_flat_samples < flatline_limit
        saturation_pass = max_clipping_run < clipping_limit
        if not flatline_pass:
            failures.append(
                {
                    "code": "persistent_flatline",
                    "semantic_channel": semantic,
                    "observed_run_samples": max_flat_samples,
                    "limit_samples": flatline_limit,
                }
            )
        if not saturation_pass:
            failures.append(
                {
                    "code": "adc_saturation",
                    "semantic_channel": semantic,
                    "observed_run_samples": max_clipping_run,
                    "limit_samples": clipping_limit,
                }
            )
        results.append(
            {
                "semantic_channel": semantic,
                "raw_signal_index": raw_index,
                "raw_signal_name": raw_name,
                "source_unit": source_unit,
                "sample_count": header.source_sample_count,
                "finite_pass": True,
                "max_flatline_run_samples": max_flat_samples,
                "flatline_pass": flatline_pass,
                "max_adc_extreme_run_samples": max_clipping_run,
                "adc_saturation_pass": saturation_pass,
                "minimum_volts": minimum_volts,
                "maximum_volts": maximum_volts,
                "peak_to_peak_volts": maximum_volts - minimum_volts,
            }
        )
    return tuple(results), tuple(failures)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _fd_sha256(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(fd, min(1024 * 1024, size - offset), offset)
        if not block:
            raise RuntimeError("EDF descriptor ended before its stat size")
        digest.update(block)
        offset += len(block)
    if os.pread(fd, 1, size):
        raise RuntimeError("EDF descriptor grew while it was hashed")
    return digest.hexdigest()


def _default_reader_factory(path: str) -> object:
    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("pyedflib is required for TUEV signal preflight") from exc
    return pyedflib.EdfReader(path)


def _validate_complete_parent_rosters(
    sources: Sequence[TUEVMorphologySourceRecord],
) -> None:
    by_group: dict[str, list[TUEVMorphologySourceRecord]] = {}
    for source in sources:
        by_group.setdefault(source.group_id, []).append(source)
    for group_id, records in by_group.items():
        observed_receipts = {record.parent_group_files for record in records}
        if len(observed_receipts) != 1:
            raise RuntimeError(f"Parent group {group_id} has contradictory rosters")
        observed = {path for path, _ in records[0].parent_group_files}
        expected: set[str] = set()
        for record in records:
            edf = Path(record.relative_edf_path)
            stem = edf.stem
            parent = edf.parent
            expected.add((parent / f"{stem}.edf").as_posix())
            expected.add((parent / f"{stem}.rec").as_posix())
            for index in _EXPECTED_DERIVATIVE_INDICES:
                expected.add((parent / f"{stem}_ch{index:03d}.lab").as_posix())
                expected.add((parent / f"{stem}_ch{index:03d}.htk").as_posix())
        if observed != expected:
            missing = tuple(sorted(expected - observed))
            unknown = tuple(sorted(observed - expected))
            raise TUEVMorphologySignalPreflightError(
                "partial_parent_roster",
                f"{group_id} must contain exactly EDF+REC+22 LAB+22 HTK per record; "
                f"missing={missing[:5]}, unknown={unknown[:5]}",
            )


def _stable_annotation(source: TUEVMorphologySourceRecord):
    before = source.rec_path.stat()
    annotation = parse_tuev_rec(source.rec_path)
    after = source.rec_path.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"REC changed while parsed: {source.rec_path}")
    if annotation.receipt.rec_sha256 != source.rec_sha256:
        raise RuntimeError(f"REC bytes differ from parent roster: {source.rec_path}")
    return annotation


def _half_up_sample(seconds: float, sfreq_hz: float) -> int:
    value = Decimal(str(float(seconds))) * Decimal(str(float(sfreq_hz)))
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _target_bounds_receipt(
    source: TUEVMorphologySourceRecord,
    *,
    source_sfreq_hz: float,
    source_sample_count: int,
    output_sample_count: int,
) -> dict[str, object]:
    annotation = _stable_annotation(source)
    if not annotation.intervals:
        raise TUEVMorphologySignalPreflightError(
            "empty_rec", f"REC contains no target rows: {source.relative_rec_path}"
        )
    record_duration = source_sample_count / source_sfreq_hz
    rows: list[tuple[object, ...]] = []
    warmup_count = context_count = joint_count = duration_count = 0
    for interval in annotation.intervals:
        if interval.stop_sec > record_duration + 0.5 / source_sfreq_hz + 1e-12:
            raise TUEVMorphologySignalPreflightError(
                "rec_out_of_record",
                f"REC line {interval.source_line} exceeds the physical EDF duration",
            )
        start_sample = _half_up_sample(
            interval.start_sec, MORPHOLOGY_OUTPUT_SFREQ_HZ
        )
        alignment_error = (
            start_sample / MORPHOLOGY_OUTPUT_SFREQ_HZ - interval.start_sec
        )
        alignment_pass = (
            abs(alignment_error) <= MORPHOLOGY_ALIGNMENT_TOLERANCE_SEC + 1e-12
        )
        duration_pass = (
            abs((interval.stop_sec - interval.start_sec) - 1.0)
            <= MORPHOLOGY_DURATION_TOLERANCE_SEC
        )
        warmup_pass = start_sample >= MORPHOLOGY_WARMUP_SAMPLES
        context_pass = (
            start_sample + MORPHOLOGY_CONTEXT_SAMPLES <= output_sample_count
        )
        duration_count += int(duration_pass)
        warmup_count += int(warmup_pass)
        context_count += int(context_pass)
        joint_count += int(warmup_pass and context_pass)
        rows.append(
            (
                interval.source_line,
                interval.official_channel_index,
                interval.label_code,
                start_sample,
                duration_pass,
                alignment_pass,
                warmup_pass,
                context_pass,
            )
        )
    return {
        "schema_version": TUEV_MORPHOLOGY_TARGET_BOUNDS_SCHEMA,
        "relative_rec_path": source.relative_rec_path,
        "rec_sha256": source.rec_sha256,
        "source_record_duration_sec": record_duration,
        "row_count": len(rows),
        "duration_tolerance_pass_count": duration_count,
        "warmup_30s_pass_count": warmup_count,
        "complete_4s_context_pass_count": context_count,
        "joint_warmup_context_pass_count": joint_count,
        "target_bounds_roster_sha256": _canonical_sha256(rows),
    }


def _inspect_record(
    source: TUEVMorphologySourceRecord,
    *,
    reader_factory: Callable[[str], object],
) -> tuple[TUEVMorphologyRecordMetadata, dict[str, int]]:
    path = source.edf_path
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError(f"EDF must be a canonical non-symlink path: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Formal TUEV preflight requires O_NOFOLLOW support")
    flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    reader: object | None = None
    opened: os.stat_result | None = None
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not 1 <= opened.st_size <= _MAX_EDF_BYTES:
            raise ValueError(f"EDF is empty, oversized, or not regular: {path}")
        lexical_before = path.stat()
        if _stat_identity(opened) != _stat_identity(lexical_before):
            raise RuntimeError(f"EDF path changed before inspection: {path}")
        if _fd_sha256(fd, opened.st_size) != source.edf_sha256:
            raise RuntimeError(f"EDF bytes differ from the parent roster: {path}")
        descriptor_path = Path(f"/proc/self/fd/{fd}")
        if not descriptor_path.exists():
            raise RuntimeError("Formal TUEV preflight requires /proc/self/fd")
        reader = reader_factory(str(descriptor_path))
        header_before = _read_header(reader)
        annotations_before = _annotation_snapshot(reader)
        qc_rows, qc_failures_tuple = _scan_signal_qc(reader, header_before)
        qc_failures = list(qc_failures_tuple)
        try:
            header_after = _read_header(reader)
            annotations_after = _annotation_snapshot(reader)
        except (TUEVMorphologySignalPreflightError, RuntimeError) as exc:
            raise RuntimeError(
                f"EDF header drifted during signal inspection: {path}"
            ) from exc
        if header_after != header_before or annotations_after != annotations_before:
            raise RuntimeError(f"EDF header drifted during signal inspection: {path}")
        gap_annotations = tuple(
            row
            for row in annotations_before
            if any(word in row[2].lower() for word in _GAP_WORDS)
        )
        if gap_annotations:
            qc_failures.append(
                {
                    "code": "gap_annotation",
                    "annotation_count": len(gap_annotations),
                    "annotation_roster_sha256": _canonical_sha256(
                        gap_annotations
                    ),
                }
            )
    finally:
        try:
            if reader is not None and hasattr(reader, "close"):
                reader.close()
        finally:
            try:
                if opened is not None:
                    closed_check = os.fstat(fd)
                    lexical_after = path.stat()
                    if _stat_identity(opened) != _stat_identity(
                        closed_check
                    ) or _stat_identity(opened) != _stat_identity(lexical_after):
                        raise RuntimeError(
                            f"EDF changed during header/signal inspection: {path}"
                        )
                    final_sha = _fd_sha256(fd, opened.st_size)
                    if final_sha != source.edf_sha256:
                        raise RuntimeError(
                            f"EDF bytes changed during signal inspection: {path}"
                        )
            finally:
                os.close(fd)

    output_samples = int(
        math.ceil(
            header_before.source_sample_count
            * MORPHOLOGY_OUTPUT_SFREQ_HZ
            / header_before.source_sfreq_hz
        )
    )
    target_bounds = _target_bounds_receipt(
        source,
        source_sfreq_hz=header_before.source_sfreq_hz,
        source_sample_count=header_before.source_sample_count,
        output_sample_count=output_samples,
    )
    mapping_sha = _canonical_sha256(header_before.mapping_payload)
    preprocessing_receipt = {
        "schema_version": TUEV_MORPHOLOGY_PREPROCESSING_POLICY_SCHEMA,
        "relative_edf_path": source.relative_edf_path,
        "edf_sha256": source.edf_sha256,
        "preprocessing_policy_sha256": (
            tuev_morphology_preprocessing_policy_sha256()
        ),
        "source_sfreq_hz": header_before.source_sfreq_hz,
        "source_sample_count": header_before.source_sample_count,
        "source_record_duration_sec": (
            header_before.source_sample_count / header_before.source_sfreq_hz
        ),
        "output_sfreq_hz": MORPHOLOGY_OUTPUT_SFREQ_HZ,
        "output_sample_count": output_samples,
        "target_bounds_receipt_sha256": _canonical_sha256(target_bounds),
    }
    signal_receipt = {
        "schema_version": TUEV_MORPHOLOGY_SIGNAL_QC_SCHEMA,
        "relative_edf_path": source.relative_edf_path,
        "edf_sha256": source.edf_sha256,
        "mapping_receipt_sha256": mapping_sha,
        "qc_scope": "complete_selected_physical_record_before_filter_or_CAR",
        "gap_annotation_count": len(gap_annotations),
        "channel_qc": qc_rows,
        "qc_failures": qc_failures,
        "signal_qc_passed": not qc_failures,
    }
    metadata = TUEVMorphologyRecordMetadata(
        relative_edf_path=source.relative_edf_path,
        edf_sha256=source.edf_sha256,
        source_sfreq_hz=header_before.source_sfreq_hz,
        source_sample_count=header_before.source_sample_count,
        output_sample_count=output_samples,
        direct_standard19=True,
        standard19_mapping_sha256=mapping_sha,
        preprocessing_receipt_sha256=_canonical_sha256(preprocessing_receipt),
        signal_qc_passed=not qc_failures,
        signal_qc_receipt_sha256=_canonical_sha256(signal_receipt),
    )
    counts = {
        "rec_row_count": int(target_bounds["row_count"]),
        "duration_tolerance_pass_count": int(
            target_bounds["duration_tolerance_pass_count"]
        ),
        "warmup_30s_pass_count": int(target_bounds["warmup_30s_pass_count"]),
        "complete_4s_context_pass_count": int(
            target_bounds["complete_4s_context_pass_count"]
        ),
        "joint_warmup_context_pass_count": int(
            target_bounds["joint_warmup_context_pass_count"]
        ),
    }
    return metadata, counts


@dataclass(frozen=True)
class TUEVMorphologyMetadataProduction:
    """Deterministic output and aggregate audit counts from one full replay."""

    payload: dict[str, object]
    external_metadata_sha256: str
    record_count: int
    rec_row_count: int
    duration_tolerance_pass_count: int
    warmup_30s_pass_count: int
    complete_4s_context_pass_count: int
    joint_warmup_context_pass_count: int
    signal_qc_failed_record_count: int
    signal_qc_failed_parent_group_count: int
    signal_qc_failed_rec_row_upper_bound: int
    signal_qc_failed_joint_target_upper_bound: int
    duplicate_ledger_sha256: str
    exact_duplicate_class_count: int
    conflicting_duplicate_class_count: int
    quarantined_record_count: int
    content_component_count: int
    cross_split_component_count: int
    output_path: Path | None

    def __post_init__(self) -> None:
        if self.external_metadata_sha256 != hashlib.sha256(
            _canonical_json(self.payload)
        ).hexdigest():
            raise ValueError("Production metadata SHA is not reproducible")
        if self.record_count < 1:
            raise ValueError("Production must contain at least one EDF record")
        ledger = self.payload.get("duplicate_ledger")
        if not isinstance(ledger, dict) or self.payload.get(
            "duplicate_ledger_sha256"
        ) != self.duplicate_ledger_sha256:
            raise ValueError("Production payload lacks its duplicate ledger binding")
        if self.duplicate_ledger_sha256 != _canonical_sha256(ledger):
            raise ValueError("Production duplicate-ledger SHA is not reproducible")
        for field_name in (
            "signal_qc_failed_record_count",
            "signal_qc_failed_parent_group_count",
            "signal_qc_failed_rec_row_upper_bound",
            "signal_qc_failed_joint_target_upper_bound",
            "exact_duplicate_class_count",
            "conflicting_duplicate_class_count",
            "quarantined_record_count",
            "content_component_count",
            "cross_split_component_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.output_path is not None and not self.output_path.is_absolute():
            raise ValueError("Production output path must be absolute")

    @property
    def summary(self) -> dict[str, object]:
        return {
            "schema_version": TUEV_MORPHOLOGY_PRODUCER_SCHEMA,
            "dry_audit": self.output_path is None,
            "output_path": None if self.output_path is None else str(self.output_path),
            "external_metadata_sha256": self.external_metadata_sha256,
            **first_party_tuev_morphology_bindings(),
            "record_count": self.record_count,
            "rec_row_count": self.rec_row_count,
            "duration_tolerance_pass_count": self.duration_tolerance_pass_count,
            "warmup_30s_pass_count": self.warmup_30s_pass_count,
            "complete_4s_context_pass_count": (
                self.complete_4s_context_pass_count
            ),
            "joint_warmup_context_pass_count": (
                self.joint_warmup_context_pass_count
            ),
            "signal_qc_failed_record_count": self.signal_qc_failed_record_count,
            "signal_qc_failed_parent_group_count": (
                self.signal_qc_failed_parent_group_count
            ),
            "signal_qc_failed_rec_row_upper_bound": (
                self.signal_qc_failed_rec_row_upper_bound
            ),
            "signal_qc_failed_joint_target_upper_bound": (
                self.signal_qc_failed_joint_target_upper_bound
            ),
            "duplicate_ledger_sha256": self.duplicate_ledger_sha256,
            "exact_duplicate_class_count": self.exact_duplicate_class_count,
            "conflicting_duplicate_class_count": (
                self.conflicting_duplicate_class_count
            ),
            "quarantined_record_count": self.quarantined_record_count,
            "content_component_count": self.content_component_count,
            "cross_split_component_count": self.cross_split_component_count,
        }


def _duplicate_audit_counts(
    ledger: TUEVExactSignalDuplicateLedger,
) -> dict[str, int | str]:
    return {
        "duplicate_ledger_sha256": ledger.ledger_sha256,
        "exact_duplicate_class_count": len(ledger.duplicate_classes),
        "conflicting_duplicate_class_count": sum(
            item.annotation_status == "conflicting_rec_bytes"
            for item in ledger.duplicate_classes
        ),
        "quarantined_record_count": sum(
            item.quarantined for item in ledger.record_decisions
        ),
        "content_component_count": len(ledger.group_components),
        "cross_split_component_count": sum(
            item.crosses_official_split for item in ledger.group_components
        ),
    }


def _produce(
    edf_root: str | Path,
    *,
    reader_factory: Callable[[str], object],
) -> TUEVMorphologyMetadataProduction:
    bindings = first_party_tuev_morphology_bindings()
    sources_before = discover_tuev_morphology_sources(edf_root)
    duplicate_ledger = build_tuev_exact_signal_duplicate_ledger(sources_before)
    _validate_complete_parent_rosters(sources_before)
    records: list[TUEVMorphologyRecordMetadata] = []
    totals = {
        "rec_row_count": 0,
        "duration_tolerance_pass_count": 0,
        "warmup_30s_pass_count": 0,
        "complete_4s_context_pass_count": 0,
        "joint_warmup_context_pass_count": 0,
    }
    signal_qc_failed_record_ids: set[str] = set()
    signal_qc_failed_group_ids: set[str] = set()
    signal_qc_failed_rec_row_upper_bound = 0
    signal_qc_failed_joint_target_upper_bound = 0
    for source in sources_before:
        try:
            metadata, counts = _inspect_record(
                source,
                reader_factory=reader_factory,
            )
        except TUEVMorphologySignalPreflightError as exc:
            raise TUEVMorphologySignalPreflightError(
                exc.code,
                f"{source.record_id} ({source.relative_edf_path}): {exc.detail}",
            ) from exc
        records.append(metadata)
        for field in totals:
            totals[field] += counts[field]
        if not metadata.signal_qc_passed:
            signal_qc_failed_record_ids.add(source.record_id)
            signal_qc_failed_group_ids.add(source.group_id)
            signal_qc_failed_rec_row_upper_bound += counts["rec_row_count"]
            signal_qc_failed_joint_target_upper_bound += counts[
                "joint_warmup_context_pass_count"
            ]
    sources_after = discover_tuev_morphology_sources(edf_root)
    _validate_complete_parent_rosters(sources_after)
    if sources_after != sources_before:
        raise RuntimeError("TUEV EDF/REC/LAB/HTK roster changed during preflight")
    if build_tuev_exact_signal_duplicate_ledger(sources_after) != duplicate_ledger:
        raise RuntimeError(
            "TUEV exact-EDF-byte duplicate ledger changed during preflight"
        )
    if first_party_tuev_morphology_bindings() != bindings:
        raise RuntimeError("Producer source or runtime policy changed during preflight")
    ordered = tuple(sorted(records, key=lambda record: record.relative_edf_path))
    if tuple(record.relative_edf_path for record in ordered) != tuple(
        source.relative_edf_path for source in sources_before
    ):
        raise RuntimeError("Produced metadata does not cover the complete source roster")
    payload: dict[str, object] = {
        "schema_version": TUEV_MORPHOLOGY_EXTERNAL_METADATA_SCHEMA,
        **bindings,
        "duplicate_ledger": duplicate_ledger.canonical_payload,
        "duplicate_ledger_sha256": duplicate_ledger.ledger_sha256,
        "records": [record.canonical_payload for record in ordered],
    }
    raw = _canonical_json(payload)
    return TUEVMorphologyMetadataProduction(
        payload=payload,
        external_metadata_sha256=hashlib.sha256(raw).hexdigest(),
        record_count=len(ordered),
        signal_qc_failed_record_count=len(signal_qc_failed_record_ids),
        signal_qc_failed_parent_group_count=len(signal_qc_failed_group_ids),
        signal_qc_failed_rec_row_upper_bound=(
            signal_qc_failed_rec_row_upper_bound
        ),
        signal_qc_failed_joint_target_upper_bound=(
            signal_qc_failed_joint_target_upper_bound
        ),
        output_path=None,
        **_duplicate_audit_counts(duplicate_ledger),
        **totals,
    )


def _atomic_publish(path: str | Path, raw: bytes) -> Path:
    target = Path(path).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"TUEV morphology metadata already exists: {target}")
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise ValueError("Metadata output parent must be an existing canonical directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"TUEV morphology metadata already exists: {target}")
        os.replace(temporary, target)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def produce_tuev_morphology_external_metadata(
    edf_root: str | Path,
    *,
    output_path: str | Path | None,
) -> TUEVMorphologyMetadataProduction:
    """Audit real TUEV sources and optionally atomically publish canonical JSON.

    ``output_path=None`` is a complete dry audit: all EDF signals are still
    read and checked, but no artifact is written.  The formal function has no
    injectable reader or caller-supplied metadata backend.
    """

    production = _produce(edf_root, reader_factory=_default_reader_factory)
    if output_path is None:
        return production
    raw = _canonical_json(production.payload)
    published = _atomic_publish(output_path, raw)
    reread = published.read_bytes()
    if reread != raw:
        raise RuntimeError("Published morphology metadata bytes failed replay")
    return TUEVMorphologyMetadataProduction(
        payload=production.payload,
        external_metadata_sha256=production.external_metadata_sha256,
        record_count=production.record_count,
        rec_row_count=production.rec_row_count,
        duration_tolerance_pass_count=production.duration_tolerance_pass_count,
        warmup_30s_pass_count=production.warmup_30s_pass_count,
        complete_4s_context_pass_count=(
            production.complete_4s_context_pass_count
        ),
        joint_warmup_context_pass_count=(
            production.joint_warmup_context_pass_count
        ),
        signal_qc_failed_record_count=(
            production.signal_qc_failed_record_count
        ),
        signal_qc_failed_parent_group_count=(
            production.signal_qc_failed_parent_group_count
        ),
        signal_qc_failed_rec_row_upper_bound=(
            production.signal_qc_failed_rec_row_upper_bound
        ),
        signal_qc_failed_joint_target_upper_bound=(
            production.signal_qc_failed_joint_target_upper_bound
        ),
        duplicate_ledger_sha256=production.duplicate_ledger_sha256,
        exact_duplicate_class_count=production.exact_duplicate_class_count,
        conflicting_duplicate_class_count=(
            production.conflicting_duplicate_class_count
        ),
        quarantined_record_count=production.quarantined_record_count,
        content_component_count=production.content_component_count,
        cross_split_component_count=production.cross_split_component_count,
        output_path=published,
    )


def replay_tuev_morphology_first_party_metadata(
    edf_root: str | Path,
    expected_payload: Mapping[str, object],
) -> TUEVMorphologyMetadataProduction:
    """Independently reproduce and compare a claimed first-party JSON file."""

    if not isinstance(expected_payload, Mapping):
        raise TypeError("Expected TUEV morphology metadata must be a mapping")
    claimed = dict(expected_payload)
    require_first_party_tuev_morphology_bindings(
        producer_source_sha256=str(claimed.get("producer_source_sha256", "")),
        preprocessing_policy_sha256=str(
            claimed.get("preprocessing_policy_sha256", "")
        ),
        standard19_mapping_policy_sha256=str(
            claimed.get("standard19_mapping_policy_sha256", "")
        ),
    )
    replayed = _produce(edf_root, reader_factory=_default_reader_factory)
    if replayed.payload != claimed:
        raise ValueError(
            "First-party TUEV morphology metadata differs from independent "
            "EDF/REC/LAB/HTK replay"
        )
    return replayed


__all__ = [
    "TUEVMorphologyMetadataProduction",
    "TUEVMorphologySignalPreflightError",
    "TUEV_MORPHOLOGY_PREPROCESSING_POLICY_SCHEMA",
    "TUEV_MORPHOLOGY_PRODUCER_SCHEMA",
    "TUEV_MORPHOLOGY_SIGNAL_QC_SCHEMA",
    "TUEV_MORPHOLOGY_STANDARD19_MAPPING_POLICY_SCHEMA",
    "TUEV_MORPHOLOGY_TARGET_BOUNDS_SCHEMA",
    "first_party_tuev_morphology_bindings",
    "produce_tuev_morphology_external_metadata",
    "replay_tuev_morphology_first_party_metadata",
    "require_first_party_tuev_morphology_bindings",
    "tuev_morphology_preprocessing_policy",
    "tuev_morphology_preprocessing_policy_sha256",
    "tuev_morphology_producer_source_roster",
    "tuev_morphology_producer_source_sha256",
    "tuev_morphology_standard19_mapping_policy",
    "tuev_morphology_standard19_mapping_policy_sha256",
]
