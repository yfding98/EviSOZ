"""First-party, receipt-bearing TUEV morphology token production.

The formal API has no tensor, label, crop-roster, or fit/held roster input.
It accepts only opaque replay-verified capabilities, a holding manifest whose
contents are rebuilt from the live EDF/REC tree, and the two audited LaBraM
files.  The producer reads physical standard-19 EDF signals, applies the
preprocessing arm selected by the formal five-arm gate, crops in absolute
200-Hz coordinates, and publishes a label-free master corpus atomically.

The fixed training carrier remains ``[19,4,200]`` for compatibility with the
morphology head.  Only LaBraM output slot zero is retained; slots 1--3 are
explicitly zeroed and are never supervised.  This preserves the four-second
contextual receptive field without silently fitting time positions that are
not used by deployment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt
import torch
import torch.nn as nn

from .data.edf import CausalEDFConfig, causal_bandpass_resample
from .data.tuev_morphology import (
    HOLDING_COUNT_SEMANTICS,
    MORPHOLOGY_CONTEXT_SAMPLES,
    MORPHOLOGY_OUTPUT_SFREQ_HZ,
    TUEVMorphologyManifest,
    TUEVMorphologyRecordReceipt,
    TUEVMorphologySourceRecord,
    VerifiedTUEVMorphologyCohortAuthorization,
    VerifiedTUEVMorphologyPreflight,
    build_tuev_morphology_manifest,
    discover_tuev_morphology_sources,
    replay_tuev_morphology_source_bindings,
)
from .data.tuev_morphology_signal_preflight import (
    _default_reader_factory,
    _fd_sha256,
    _read_header,
    _stat_identity,
)
from .geometry import N_STANDARD_CHANNELS, STANDARD_19
from .models.labram import (
    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
    OfficialLaBraMEncoder,
    LaBraMFeatureReceipt,
    LaBraMRecordPositionBinding,
    bind_labram_record_positions,
    require_feature_receipt_position_binding,
)
from .morphology_features import (
    MORPHOLOGY_CONTEXT_SECONDS,
    MORPHOLOGY_READ_SLOT,
    MORPHOLOGY_SAMPLES_PER_SECOND,
    MORPHOLOGY_TOKEN_DIM,
)
from .morphology_token_io import (
    MORPHOLOGY_TRAINING_CORPUS_PURPOSE,
    MORPHOLOGY_TRAINING_CORPUS_SCHEMA,
    MORPHOLOGY_TRAINING_TOKEN_SHAPE,
    MorphologyTrainingTokenBinding,
    VerifiedMorphologyTrainingTokenCorpus,
    _crop_directory_name,
    _load_morphology_training_token_corpus_structural,
    load_morphology_training_group_tokens,
    morphology_foundation_receipt_sha256,
    save_morphology_training_group_tokens,
)
from .preprocessing_parity import (
    AuthorizedPreprocessingSelection,
    FROZEN_PREPROCESSING_ARM_SPEC_BY_ID,
    VerifiedPreprocessingSelectionCapability,
    preprocessing_foundation_policy_receipt_sha256,
)


TUEV_MORPHOLOGY_FORMAL_TOKEN_SCHEMA = "soz_morphology_token_corpus_index_v4"
TUEV_MORPHOLOGY_PRODUCER_RECEIPT_SCHEMA = (
    "soz_tuev_morphology_first_party_producer_receipt_v1"
)
TUEV_MORPHOLOGY_PRODUCER_KIND = "tuev_morphology"
TUEV_MORPHOLOGY_PRODUCER_SERIALIZATION = (
    "canonical_json_plus_safetensors_no_pickle_atomic_directory_v1"
)
TUEV_MORPHOLOGY_TOKEN_CORPUS_DIRECTORY = "corpus"
TUEV_MORPHOLOGY_PRODUCER_RECEIPT_FILE = "producer_receipt.json"
TUEV_MORPHOLOGY_PRODUCER_MANIFEST_FILE = "manifest.json"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_BYTES = 128 * 1024 * 1024
_MAX_EDF_BYTES = 32 * 1024 * 1024 * 1024
_FORMAL_CONTAINER_FIELDS = frozenset(
    {
        "schema_version",
        "serialization",
        "producer_receipt_file",
        "producer_receipt_sha256",
        "producer_receipt_size_bytes",
        "token_corpus_directory",
        "token_index_sha256",
        "holding_manifest_sha256",
        "crop_count",
        "record_count",
    }
)
_PRODUCER_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "token_schema_version",
        "producer_kind",
        "serialization",
        "producer_source_sha256",
        "holding_manifest_sha256",
        "preflight_bundle_manifest_sha256",
        "preflight_receipt_sha256",
        "external_metadata_sha256",
        "source_roster_sha256",
        "cohort_authorization_sha256",
        "preprocessing_authorization",
        "preprocessing_authorization_receipt_sha256",
        "selected_arm_spec",
        "selected_arm_spec_receipt_sha256",
        "foundation_feature_receipt",
        "foundation_feature_receipt_sha256",
        "token_index_sha256",
        "crop_count",
        "record_count",
        "crop_roster_sha256",
        "waveform_roster_sha256",
        "tensor_roster_sha256",
        "semantic_channels",
        "output_sfreq_hz",
        "context_samples",
        "context_seconds",
        "retained_read_slot",
        "nonretained_slots_zeroed",
        "target_payload_absent",
        "records",
    }
)
_RECORD_RECEIPT_FIELDS = frozenset(
    {
        "record_id",
        "relative_edf_path",
        "edf_sha256",
        "rec_sha256",
        "source_sfreq_hz",
        "source_sample_count",
        "logical_output_sample_count",
        "raw_channel_names",
        "raw_units",
        "position_binding",
        "standard19_mapping_sha256",
        "selected_raw_volts_sha256",
        "record_preprocessing_receipt",
        "record_preprocessing_receipt_sha256",
        "crops",
    }
)
_CROP_RECEIPT_FIELDS = frozenset(
    {
        "crop_id",
        "start_sample",
        "stop_sample",
        "waveform_sha256",
        "preprocessing_receipt",
        "preprocessing_receipt_sha256",
        "relative_bundle_path",
        "bundle_manifest_sha256",
        "tensor_sha256",
    }
)
_SOURCE_FILES = (
    "src/soz/tuev_morphology_producer.py",
    "src/soz/morphology_token_io.py",
    "src/soz/preprocessing_parity.py",
    "src/soz/data/edf.py",
    "src/soz/data/tuev_morphology.py",
    "src/soz/data/tuev_morphology_signal_preflight.py",
    "src/soz/models/labram.py",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON value is forbidden: {value}")


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
    if not isinstance(value, dict) or _canonical_json(value) != raw:
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _require_fields(
    value: Mapping[str, object], expected: frozenset[str], *, label: str
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} violates its closed schema; missing={missing}, unknown={unknown}"
        )


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def _payload_sha256(array: np.ndarray, *, name: str) -> str:
    values = np.ascontiguousarray(array)
    header = _canonical_json(
        {"name": name, "shape": list(values.shape), "dtype": str(values.dtype)}
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(4, "little"))
    digest.update(header)
    raw = values.tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


def _typed_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _half_up(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("Sample coordinate must be finite")
    return int(
        Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
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


def _producer_root() -> Path:
    root = Path(__file__).resolve(strict=True).parents[2]
    if not root.is_dir():
        raise RuntimeError("Cannot resolve morphology producer source root")
    return root


def tuev_morphology_token_producer_source_sha256() -> str:
    root = _producer_root()
    roster: list[tuple[str, str]] = []
    for relative in _SOURCE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            raise RuntimeError(f"Morphology producer source is not canonical: {relative}")
        roster.append((relative, _file_sha256(path)))
    return _typed_sha256(
        {
            "schema_version": "soz_tuev_morphology_producer_source_v1",
            "files": roster,
        }
    )


@dataclass(frozen=True)
class _ReadPhysicalRecord:
    raw_volts: np.ndarray
    raw_channel_names: tuple[str, ...]
    raw_units: tuple[str, ...]
    source_sfreq_hz: float
    source_sample_count: int
    position_binding: LaBraMRecordPositionBinding
    mapping_sha256: str
    raw_volts_sha256: str


@dataclass(frozen=True)
class _PreparedCrop:
    waveform: np.ndarray
    preprocessing_receipt: dict[str, object]
    record_preprocessing_receipt: dict[str, object]


def _validate_edf_root(edf_root: str | Path) -> Path:
    root = Path(edf_root).absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("TUEV EDF root must be one canonical regular directory")
    return root


def _read_physical_record(
    source: TUEVMorphologySourceRecord,
    record: TUEVMorphologyRecordReceipt,
) -> _ReadPhysicalRecord:
    path = source.edf_path
    if record.relative_edf_path != source.relative_edf_path:
        raise ValueError("Manifest record was swapped across the live TUEV source")
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("TUEV EDF path must remain canonical and non-symlinked")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Formal morphology production requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    reader: object | None = None
    opened: os.stat_result | None = None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not 1 <= opened.st_size <= _MAX_EDF_BYTES:
            raise ValueError("TUEV EDF is empty, oversized, or not a regular file")
        if _stat_identity(opened) != _stat_identity(path.stat()):
            raise RuntimeError("TUEV EDF path changed before token production")
        if _fd_sha256(descriptor, opened.st_size) != record.edf_sha256:
            raise ValueError("TUEV EDF bytes differ from the holding manifest")
        descriptor_path = Path(f"/proc/self/fd/{descriptor}")
        if not descriptor_path.exists():
            raise RuntimeError("Formal morphology production requires /proc/self/fd")
        reader = _default_reader_factory(str(descriptor_path))
        header_before = _read_header(reader)
        arrays = tuple(
            np.asarray(reader.readSignal(index), dtype=np.float64)
            for index in header_before.selected_indices
        )
        if any(array.shape != (header_before.source_sample_count,) for array in arrays):
            raise ValueError("EDF reader returned an invalid standard-19 payload shape")
        raw = np.stack(arrays)
        if raw.shape != (N_STANDARD_CHANNELS, header_before.source_sample_count):
            raise RuntimeError("Selected standard-19 EDF payload has the wrong shape")
        if not np.isfinite(raw).all():
            raise ValueError("Selected standard-19 EDF payload contains non-finite values")
        header_after = _read_header(reader)
        if header_after != header_before:
            raise RuntimeError("EDF header changed during morphology signal read")
        raw_volts = raw * np.asarray(header_before.unit_scales, dtype=np.float64)[:, None]
        binding = bind_labram_record_positions(
            header_before.selected_names,
            semantic_channels=STANDARD_19,
        )
        mapping_sha = _typed_sha256(header_before.mapping_payload)
    finally:
        try:
            if reader is not None and hasattr(reader, "close"):
                reader.close()
        finally:
            try:
                if opened is not None:
                    after = os.fstat(descriptor)
                    if (
                        _stat_identity(opened) != _stat_identity(after)
                        or _stat_identity(opened) != _stat_identity(path.stat())
                        or _fd_sha256(descriptor, opened.st_size) != record.edf_sha256
                    ):
                        raise RuntimeError("TUEV EDF changed during token production")
            finally:
                os.close(descriptor)

    metadata = record.metadata
    observed = (
        source.edf_sha256,
        header_before.source_sfreq_hz,
        header_before.source_sample_count,
        int(
            math.ceil(
                header_before.source_sample_count
                * MORPHOLOGY_OUTPUT_SFREQ_HZ
                / header_before.source_sfreq_hz
            )
        ),
        mapping_sha,
    )
    expected = (
        metadata.edf_sha256,
        metadata.source_sfreq_hz,
        metadata.source_sample_count,
        metadata.output_sample_count,
        metadata.standard19_mapping_sha256,
    )
    if observed != expected or not metadata.direct_standard19 or not metadata.signal_qc_passed:
        raise ValueError("Live EDF/header/QC binding differs from strict TUEV preflight")
    raw_float32 = np.ascontiguousarray(raw_volts, dtype=np.float32)
    return _ReadPhysicalRecord(
        raw_volts=np.ascontiguousarray(raw_volts, dtype=np.float64),
        raw_channel_names=header_before.selected_names,
        raw_units=header_before.source_units,
        source_sfreq_hz=header_before.source_sfreq_hz,
        source_sample_count=header_before.source_sample_count,
        position_binding=binding,
        mapping_sha256=mapping_sha,
        raw_volts_sha256=_payload_sha256(
            raw_float32, name="selected_standard19_raw_volts"
        ),
    )


def _rational_rate(source_sfreq_hz: float) -> tuple[int, int]:
    ratio = Fraction(str(MORPHOLOGY_OUTPUT_SFREQ_HZ)) / Fraction(
        str(float(source_sfreq_hz))
    )
    ratio = ratio.limit_denominator(10_000)
    reconstructed = source_sfreq_hz * ratio.numerator / ratio.denominator
    if abs(reconstructed - MORPHOLOGY_OUTPUT_SFREQ_HZ) > 1e-9:
        raise ValueError("EDF sampling-rate ratio is not reproducible")
    return ratio.numerator, ratio.denominator


def _apply_car19(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != N_STANDARD_CHANNELS:
        raise ValueError("CAR19 expects a [19,T] signal")
    return result - result.mean(axis=0, keepdims=True)


def _prepare_full_record(
    record: _ReadPhysicalRecord,
    *,
    arm_id: str,
    logical_output_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    spec = FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[arm_id]
    sfreq = record.source_sfreq_hz
    if not spec.state_scope.startswith("full_record"):
        raise ValueError("Selected arm is not a full-record preprocessing arm")
    if spec.lowpass_hz >= 0.5 * sfreq:
        raise ValueError(
            f"{arm_id} lowpass {spec.lowpass_hz:g} Hz is invalid at {sfreq:g} Hz"
        )
    if arm_id == "O-CAR19":
        try:
            import mne
        except ImportError as exc:  # pragma: no cover - formal dependency gate
            raise RuntimeError("O-CAR19 requires MNE for its frozen spectral family") from exc
        filtered = mne.filter.filter_data(
            record.raw_volts,
            sfreq=sfreq,
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
            Fs=sfreq,
            freqs=np.asarray([spec.notch_hz], dtype=float),
            method="fir",
            phase="zero-double",
            fir_design="firwin",
            copy=True,
            verbose=False,
        )
        processed = resample(filtered, logical_output_count, axis=-1)
        implementation = "mne_fir_zero_double_notch_then_scipy_fft_resample_v1"
    elif arm_id in {"Z-REF19", "Z-CAR19"}:
        sos = butter(
            4,
            [spec.highpass_hz, spec.lowpass_hz],
            btype="bandpass",
            fs=sfreq,
            output="sos",
        )
        filtered = sosfiltfilt(sos, record.raw_volts, axis=-1)
        processed = resample(filtered, logical_output_count, axis=-1)
        implementation = "scipy_sosfiltfilt_order4_then_fft_resample_v1"
    else:
        raise ValueError(f"{arm_id} is not a supported full-record deployment arm")
    if spec.reference == "car19":
        processed = _apply_car19(processed)
    elif spec.reference != "physical_ref_no_rereference":
        raise ValueError("Frozen preprocessing reference is unsupported")
    processed = np.ascontiguousarray(processed, dtype=np.float64)
    if processed.shape != (N_STANDARD_CHANNELS, logical_output_count):
        raise RuntimeError("Full-record preprocessing returned the wrong sample count")
    if not np.isfinite(processed).all():
        raise ValueError("Full-record preprocessing produced non-finite values")
    receipt = {
        "arm_id": arm_id,
        "arm_spec_receipt_sha256": spec.receipt_sha256,
        "implementation": implementation,
        "state_scope": spec.state_scope,
        "state_reset_source_sample": 0,
        "read_stop_source_sample": record.source_sample_count,
        "source_sfreq_hz": sfreq,
        "output_sfreq_hz": MORPHOLOGY_OUTPUT_SFREQ_HZ,
        "logical_output_sample_count": logical_output_count,
        "processed_array_sample_count": processed.shape[1],
        "resampling_delay_output_samples": 0.0,
        "reference": spec.reference,
    }
    return processed, receipt


def _prepare_causal_crop(
    record: _ReadPhysicalRecord,
    *,
    start_sample: int,
    stop_sample: int,
) -> _PreparedCrop:
    arm_id = "C-CAR19"
    spec = FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[arm_id]
    if stop_sample - start_sample != MORPHOLOGY_CONTEXT_SAMPLES:
        raise ValueError("Morphology crop must contain exactly four seconds")
    start_sec = start_sample / MORPHOLOGY_OUTPUT_SFREQ_HZ
    stop_sec = stop_sample / MORPHOLOGY_OUTPUT_SFREQ_HZ
    sfreq = record.source_sfreq_hz
    state_reset = int(math.floor((start_sec - 30.0) * sfreq + 1e-12))
    if state_reset < 0:
        raise ValueError("C-CAR19 crop lacks 30 seconds of real warmup")
    read_stop = min(
        record.source_sample_count,
        int(math.ceil(stop_sec * sfreq - 1e-12)) + 1,
    )
    if read_stop <= state_reset:
        raise ValueError("C-CAR19 crop has an invalid finite source segment")
    source_segment = record.raw_volts[:, state_reset:read_stop]
    config = CausalEDFConfig(
        output_sfreq_hz=MORPHOLOGY_OUTPUT_SFREQ_HZ,
        highpass_hz=spec.highpass_hz,
        lowpass_hz=spec.lowpass_hz,
        butterworth_order=4,
        warmup_sec=30.0,
        apply_car19=True,
    )
    processed, up, down, n_taps, latency_sec = causal_bandpass_resample(
        source_segment,
        source_sfreq_hz=sfreq,
        config=config,
    )
    processed = _apply_car19(processed)
    segment_start_sec = state_reset / sfreq
    latency_output_samples = latency_sec * MORPHOLOGY_OUTPUT_SFREQ_HZ
    crop_offset = _half_up(
        (start_sec - segment_start_sec) * MORPHOLOGY_OUTPUT_SFREQ_HZ
        + latency_output_samples
    )
    crop_stop = crop_offset + MORPHOLOGY_CONTEXT_SAMPLES
    if crop_offset < 0 or crop_stop > processed.shape[1]:
        raise ValueError("C-CAR19 delayed crop falls outside its real finite segment")
    waveform = np.ascontiguousarray(
        processed[:, crop_offset:crop_stop], dtype=np.float32
    )
    if waveform.shape != (N_STANDARD_CHANNELS, MORPHOLOGY_CONTEXT_SAMPLES):
        raise RuntimeError("C-CAR19 produced an invalid morphology crop")
    if not np.isfinite(waveform).all():
        raise ValueError("C-CAR19 morphology crop contains non-finite values")
    warmup_sec = start_sec - segment_start_sec
    if warmup_sec + 1e-12 < 30.0:
        raise RuntimeError("C-CAR19 state reset violated the 30-second warmup")
    preprocessing = {
        "arm_id": arm_id,
        "arm_spec_receipt_sha256": spec.receipt_sha256,
        "implementation": "scipy_sosfilt_upfirdn_finite_segment_v1",
        "state_scope": spec.state_scope,
        "state_reset_source_sample": state_reset,
        "read_stop_source_sample": read_stop,
        "source_segment_sample_count": read_stop - state_reset,
        "real_warmup_sec": warmup_sec,
        "source_sfreq_hz": sfreq,
        "output_sfreq_hz": MORPHOLOGY_OUTPUT_SFREQ_HZ,
        "resample_up": up,
        "resample_down": down,
        "resample_fir_taps": n_taps,
        "resampling_delay_sec": latency_sec,
        "resampling_delay_output_samples": latency_output_samples,
        "delay_compensated_crop_offset": crop_offset,
        "logical_start_sample": start_sample,
        "logical_stop_sample": stop_sample,
        "reference": spec.reference,
    }
    record_receipt = {
        "arm_id": arm_id,
        "arm_spec_receipt_sha256": spec.receipt_sha256,
        "implementation": "per_crop_finite_segment_zero_state",
        "state_scope": spec.state_scope,
        "source_sfreq_hz": sfreq,
        "output_sfreq_hz": MORPHOLOGY_OUTPUT_SFREQ_HZ,
        "reference": spec.reference,
    }
    return _PreparedCrop(
        waveform=waveform,
        preprocessing_receipt=preprocessing,
        record_preprocessing_receipt=record_receipt,
    )


def _prepare_crop(
    record: _ReadPhysicalRecord,
    *,
    arm_id: str,
    start_sample: int,
    stop_sample: int,
    logical_output_count: int,
    full_record: tuple[np.ndarray, dict[str, object]] | None,
) -> _PreparedCrop:
    if arm_id == "C-CAR19":
        if full_record is not None:
            raise RuntimeError("C-CAR19 cannot reuse a full-record zero-state result")
        return _prepare_causal_crop(
            record,
            start_sample=start_sample,
            stop_sample=stop_sample,
        )
    if full_record is None:
        raise RuntimeError("A full-record arm lacks its preprocessed record")
    values, record_receipt = full_record
    if stop_sample - start_sample != MORPHOLOGY_CONTEXT_SAMPLES:
        raise ValueError("Morphology crop must contain exactly four seconds")
    if start_sample < 0 or stop_sample > logical_output_count:
        raise ValueError("Morphology crop escapes absolute 200-Hz record coordinates")
    waveform = np.ascontiguousarray(values[:, start_sample:stop_sample], dtype=np.float32)
    if waveform.shape != (N_STANDARD_CHANNELS, MORPHOLOGY_CONTEXT_SAMPLES):
        raise RuntimeError("Full-record arm produced an invalid morphology crop")
    preprocessing = {
        **record_receipt,
        "logical_start_sample": start_sample,
        "logical_stop_sample": stop_sample,
        "delay_compensated_crop_offset": start_sample,
    }
    return _PreparedCrop(
        waveform=waveform,
        preprocessing_receipt=preprocessing,
        record_preprocessing_receipt=record_receipt,
    )


def _authorize_selection(
    capability: VerifiedPreprocessingSelectionCapability,
) -> AuthorizedPreprocessingSelection:
    if not isinstance(capability, VerifiedPreprocessingSelectionCapability):
        raise TypeError(
            "preprocessing_selection must come from the strict five-arm loader"
        )
    return capability.authorize_producer(
        arm_id=capability.selected_arm_id,
        expected_arm_result_receipt_sha256=(
            capability.selected_arm_result_receipt_sha256
        ),
        producer_kind=TUEV_MORPHOLOGY_PRODUCER_KIND,
        token_schema_version=TUEV_MORPHOLOGY_FORMAL_TOKEN_SCHEMA,
    )


def _assert_preflight_bundle_unchanged(
    preflight: VerifiedTUEVMorphologyPreflight,
) -> None:
    if not isinstance(preflight, VerifiedTUEVMorphologyPreflight):
        raise TypeError("preflight must come from the strict TUEV preflight loader")
    source = preflight.path
    if source.is_symlink() or not source.is_dir() or source.resolve(strict=True) != source:
        raise ValueError("Strict TUEV preflight bundle is no longer canonical")
    if {item.name for item in source.iterdir()} != {"manifest.json", "preflight.json"}:
        raise ValueError("Strict TUEV preflight bundle roster changed")
    manifest_path = source / "manifest.json"
    receipt_path = source / "preflight.json"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, receipt_path)):
        raise ValueError("Strict TUEV preflight members changed type")
    if _file_sha256(manifest_path) != preflight.bundle_manifest_sha256:
        raise ValueError("Strict TUEV preflight manifest changed after load")
    if _file_sha256(receipt_path) != preflight.preflight_receipt_sha256:
        raise ValueError("Strict TUEV preflight receipt changed after load")


def _replay_holding_inputs(
    *,
    edf_root: Path,
    holding_manifest: TUEVMorphologyManifest,
    preflight: VerifiedTUEVMorphologyPreflight,
    cohort_authorization: VerifiedTUEVMorphologyCohortAuthorization,
) -> tuple[TUEVMorphologySourceRecord, ...]:
    if not isinstance(holding_manifest, TUEVMorphologyManifest):
        raise TypeError("holding_manifest must be a TUEVMorphologyManifest")
    if holding_manifest.count_semantics != HOLDING_COUNT_SEMANTICS:
        raise ValueError("Only the authorization-bound holding manifest may own tokens")
    if not isinstance(
        cohort_authorization, VerifiedTUEVMorphologyCohortAuthorization
    ):
        raise TypeError("cohort_authorization must be replay-derived")
    _assert_preflight_bundle_unchanged(preflight)
    replay_tuev_morphology_source_bindings(holding_manifest, edf_root)
    sources = discover_tuev_morphology_sources(edf_root)
    rebuilt = build_tuev_morphology_manifest(
        sources,
        preflight,
        cohort_authorization,
        preprocessing_policy_sha256=holding_manifest.preprocessing_policy_sha256,
        holding_reference_target_count=(
            holding_manifest.holding_reference_target_count
        ),
    )
    if rebuilt != holding_manifest:
        raise ValueError("Holding manifest crops were not reproduced from live EDF/REC")
    if holding_manifest.cohort_authorization_sha256 != cohort_authorization.receipt_sha256:
        raise ValueError("Holding manifest belongs to another cohort authorization")
    record_by_group = {
        record.parent_group_id: record for record in holding_manifest.records
    }
    official_eval_groups = {
        record.parent_group_id
        for record in holding_manifest.records
        if record.official_split == "eval"
    }
    if official_eval_groups & set(cohort_authorization.fit_group_ids):
        raise ValueError("Official TUEV evaluation groups cannot enter fitting")
    if set(record_by_group) != set(cohort_authorization.eligible_group_ids) | set(
        cohort_authorization.excluded_group_ids
    ):
        raise ValueError("Cohort authorization does not cover every TUEV parent group")
    return sources


def _foundation_payload(receipt: LaBraMFeatureReceipt) -> dict[str, object]:
    if not isinstance(receipt, LaBraMFeatureReceipt):
        raise TypeError("LaBraM encoder must expose a typed feature receipt")
    payload = receipt.to_dict()
    payload["semantic_channels"] = list(receipt.semantic_channels)
    payload["position_names"] = list(receipt.position_names)
    payload["position_ids"] = list(receipt.position_ids)
    # Reuse the morphology cache validator, which checks audited hashes,
    # position IDs, four-second calls, token geometry, and volts x 1e4.
    morphology_foundation_receipt_sha256(receipt)
    return payload


def _prepare_encoder(
    encoder: nn.Module,
    *,
    authorization: AuthorizedPreprocessingSelection,
    device: torch.device,
) -> tuple[nn.Module, LaBraMFeatureReceipt, str]:
    if not isinstance(encoder, nn.Module):
        raise TypeError("encoder must be a torch.nn.Module")
    receipt = getattr(encoder, "receipt", None)
    if not isinstance(receipt, LaBraMFeatureReceipt):
        raise TypeError("encoder lacks a typed audited LaBraM feature receipt")
    foundation_sha = morphology_foundation_receipt_sha256(receipt)
    selection_policy_sha = preprocessing_foundation_policy_receipt_sha256(
        checkpoint_sha256=receipt.checkpoint_sha256,
        modeling_sha256=receipt.modeling_sha256,
        position_binding_policy=LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
        record_specific_position_ids=True,
        token_dim=receipt.token_dim,
        input_scale_from_volts=receipt.input_scale_from_volts,
    )
    if selection_policy_sha != authorization.foundation_feature_receipt_sha256:
        raise ValueError(
            "Selected preprocessing protocol belongs to another foundation "
            "compatibility policy"
        )
    if (
        int(getattr(encoder, "seconds_per_call", -1)) != MORPHOLOGY_CONTEXT_SECONDS
        or int(getattr(encoder, "samples_per_token", -1))
        != MORPHOLOGY_SAMPLES_PER_SECOND
        or int(getattr(encoder, "token_dim", -1)) != MORPHOLOGY_TOKEN_DIM
    ):
        raise ValueError("LaBraM encoder geometry differs from morphology production")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Formal morphology production requires a frozen foundation")
    encoder = encoder.to(device)
    encoder.eval()
    return encoder, receipt, foundation_sha


def _encode_slot_zero(
    encoder: nn.Module,
    waveforms: Sequence[np.ndarray],
    *,
    device: torch.device,
    microbatch_size: int,
) -> tuple[torch.Tensor, ...]:
    if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int):
        raise TypeError("microbatch_size must be an integer")
    if not 1 <= microbatch_size <= 256:
        raise ValueError("microbatch_size must lie in [1,256]")
    outputs: list[torch.Tensor] = []
    for start in range(0, len(waveforms), microbatch_size):
        batch_arrays = waveforms[start : start + microbatch_size]
        array = np.stack(batch_arrays).reshape(
            len(batch_arrays),
            N_STANDARD_CHANNELS,
            MORPHOLOGY_CONTEXT_SECONDS,
            MORPHOLOGY_SAMPLES_PER_SECOND,
        )
        batch = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(
            device
        )
        with torch.inference_mode():
            encoded = encoder(batch)
        expected = (
            len(batch_arrays),
            N_STANDARD_CHANNELS,
            MORPHOLOGY_CONTEXT_SECONDS,
            MORPHOLOGY_TOKEN_DIM,
        )
        if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != expected:
            raise ValueError(
                f"LaBraM returned {getattr(encoded, 'shape', None)}, expected {expected}"
            )
        if encoded.dtype != torch.float32:
            encoded = encoded.to(torch.float32)
        if not torch.isfinite(encoded).all().item():
            raise ValueError("LaBraM returned non-finite morphology tokens")
        # Slot zero is contextualized by the full four-second input.  Other
        # output slots are not part of the frozen training/deployment policy.
        retained = torch.zeros_like(encoded)
        retained[:, :, MORPHOLOGY_READ_SLOT, :] = encoded[
            :, :, MORPHOLOGY_READ_SLOT, :
        ]
        outputs.extend(item.detach().cpu().contiguous() for item in retained)
    if len(outputs) != len(waveforms):
        raise RuntimeError("Morphology encoder omitted a source crop")
    return tuple(outputs)


def _validate_output_parent(path: str | Path) -> Path:
    target = Path(path).absolute()
    if target.name in {"", ".", ".."}:
        raise ValueError("Morphology corpus output requires a concrete directory")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Morphology master corpus already exists: {target}")
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise ValueError("Morphology corpus parent must be an existing canonical directory")
    return target


def _write_token_index(
    corpus_directory: Path,
    *,
    holding_manifest: TUEVMorphologyManifest,
    foundation_receipt: LaBraMFeatureReceipt,
    entries: Sequence[Mapping[str, object]],
) -> tuple[str, str, str]:
    ordered = tuple(sorted((dict(entry) for entry in entries), key=lambda row: str(row["crop_id"])))
    crop_ids = tuple(str(entry["crop_id"]) for entry in ordered)
    crop_roster_sha = _typed_sha256(list(crop_ids))
    tensor_roster_sha = _typed_sha256(
        [[entry["crop_id"], entry["tensor_sha256"]] for entry in ordered]
    )
    index = {
        "schema_version": MORPHOLOGY_TRAINING_CORPUS_SCHEMA,
        "purpose": MORPHOLOGY_TRAINING_CORPUS_PURPOSE,
        "serialization": "canonical_json_and_safe_tensors_no_pickle",
        "source_morphology_manifest_sha256": holding_manifest.manifest_sha256,
        "foundation_feature_receipt_sha256": (
            morphology_foundation_receipt_sha256(foundation_receipt)
        ),
        "crop_count": len(ordered),
        "crop_roster_sha256": crop_roster_sha,
        "tensor_roster_sha256": tensor_roster_sha,
        "entries": list(ordered),
    }
    raw = _canonical_json(index)
    if not 1 <= len(raw) <= _MAX_JSON_BYTES:
        raise ValueError("Morphology token index has an invalid size")
    path = corpus_directory / "index.json"
    path.write_bytes(raw)
    _fsync_file(path)
    return hashlib.sha256(raw).hexdigest(), crop_roster_sha, tensor_roster_sha


@dataclass(frozen=True)
class TUEVMorphologyMasterCorpusArtifact:
    path: Path
    bundle_manifest_sha256: str
    producer_receipt_sha256: str
    token_index_sha256: str
    holding_manifest_sha256: str
    crop_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("Morphology producer artifact path must be absolute")
        for field in (
            "bundle_manifest_sha256",
            "producer_receipt_sha256",
            "token_index_sha256",
            "holding_manifest_sha256",
        ):
            _sha(getattr(self, field), field=field)
        if isinstance(self.crop_count, bool) or not isinstance(self.crop_count, int):
            raise TypeError("crop_count must be an integer")
        if self.crop_count < 1:
            raise ValueError("Morphology producer artifact cannot be empty")


_VERIFIED_FORMAL_CORPUS_ISSUER = object()


@dataclass(frozen=True, init=False)
class VerifiedTUEVMorphologyMasterCorpus:
    """Opaque formal producer verification plus its training-token carrier."""

    path: Path
    bundle_manifest_sha256: str
    producer_receipt_sha256: str
    producer_source_sha256: str
    selected_arm_id: str
    token_corpus: VerifiedMorphologyTrainingTokenCorpus
    _holding_manifest: TUEVMorphologyManifest
    _preprocessing_authorization: AuthorizedPreprocessingSelection

    def __init__(
        self,
        *,
        _issuer: object,
        path: Path,
        bundle_manifest_sha256: str,
        producer_receipt_sha256: str,
        producer_source_sha256: str,
        selected_arm_id: str,
        token_corpus: VerifiedMorphologyTrainingTokenCorpus,
        holding_manifest: TUEVMorphologyManifest,
        preprocessing_authorization: AuthorizedPreprocessingSelection,
    ) -> None:
        if _issuer is not _VERIFIED_FORMAL_CORPUS_ISSUER:
            raise TypeError(
                "VerifiedTUEVMorphologyMasterCorpus can only be issued by the strict loader"
            )
        if not isinstance(token_corpus, VerifiedMorphologyTrainingTokenCorpus):
            raise TypeError("token_corpus must come from the structural strict loader")
        values = {
            "path": path,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "producer_receipt_sha256": producer_receipt_sha256,
            "producer_source_sha256": producer_source_sha256,
            "selected_arm_id": selected_arm_id,
            "token_corpus": token_corpus,
            "_holding_manifest": holding_manifest,
            "_preprocessing_authorization": preprocessing_authorization,
        }
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("Verified morphology producer path must be absolute")
        for field in (
            "bundle_manifest_sha256",
            "producer_receipt_sha256",
            "producer_source_sha256",
        ):
            _sha(values[field], field=field)
        preprocessing_authorization.require_selected_arm(selected_arm_id)
        for field, value in values.items():
            object.__setattr__(self, field, value)

    @property
    def crop_count(self) -> int:
        return self.token_corpus.crop_count

    @property
    def index_sha256(self) -> str:
        return self.token_corpus.index_sha256

    @property
    def bindings(self) -> tuple[MorphologyTrainingTokenBinding, ...]:
        return self.token_corpus.bindings

    def assert_unchanged(self) -> None:
        self._preprocessing_authorization.assert_unchanged()
        _, receipt = _load_formal_container(
            self.path,
            expected_bundle_manifest_sha256=self.bundle_manifest_sha256,
            expected_producer_receipt_sha256=self.producer_receipt_sha256,
            expected_token_index_sha256=self.token_corpus.index_sha256,
        )
        if receipt["producer_source_sha256"] != self.producer_source_sha256:
            raise ValueError("Morphology producer source binding changed")
        replay = _load_morphology_training_token_corpus_structural(
            self.path / TUEV_MORPHOLOGY_TOKEN_CORPUS_DIRECTORY,
            self._holding_manifest,
            expected_index_sha256=self.token_corpus.index_sha256,
        )
        if (
            replay.crop_roster_sha256 != self.token_corpus.crop_roster_sha256
            or replay.tensor_roster_sha256 != self.token_corpus.tensor_roster_sha256
        ):
            raise ValueError("Morphology token corpus changed after strict load")


def _materialize_first_party_core(
    output_directory: str | Path,
    *,
    edf_root: str | Path,
    holding_manifest: TUEVMorphologyManifest,
    preflight: VerifiedTUEVMorphologyPreflight,
    cohort_authorization: VerifiedTUEVMorphologyCohortAuthorization,
    preprocessing_selection: VerifiedPreprocessingSelectionCapability,
    encoder: nn.Module,
    device: str | torch.device,
    microbatch_size: int,
) -> TUEVMorphologyMasterCorpusArtifact:
    target = _validate_output_parent(output_directory)
    root = _validate_edf_root(edf_root)
    authorization = _authorize_selection(preprocessing_selection)
    authorization.assert_unchanged()
    selected_arm_id = authorization.selected_arm_id
    selected_spec = FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[selected_arm_id]
    authorization.require_selected_arm(selected_arm_id)
    if selected_spec.receipt_sha256 != authorization.receipt.selected_arm_spec_receipt_sha256:
        raise ValueError("Selected preprocessing arm spec changed after authorization")
    sources = _replay_holding_inputs(
        edf_root=root,
        holding_manifest=holding_manifest,
        preflight=preflight,
        cohort_authorization=cohort_authorization,
    )
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for morphology production but is unavailable")
    encoder, foundation_receipt, foundation_sha = _prepare_encoder(
        encoder,
        authorization=authorization,
        device=torch_device,
    )
    producer_source_sha = tuev_morphology_token_producer_source_sha256()
    holding_manifest_sha = holding_manifest.manifest_sha256

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        corpus_directory = staging / TUEV_MORPHOLOGY_TOKEN_CORPUS_DIRECTORY
        groups_directory = corpus_directory / "groups"
        groups_directory.mkdir(parents=True)
        record_by_id = {record.record_id: record for record in holding_manifest.records}
        source_by_id = {source.record_id: source for source in sources}
        groups_by_record: dict[str, list[object]] = {}
        for interval_group in holding_manifest.interval_groups:
            groups_by_record.setdefault(interval_group.record_id, []).append(interval_group)
        if not groups_by_record:
            raise ValueError("Cannot materialize an empty morphology holding corpus")
        token_entries: list[dict[str, object]] = []
        producer_records: list[dict[str, object]] = []
        waveform_roster: list[list[str]] = []

        for record_id in sorted(groups_by_record):
            authorization.assert_unchanged()
            record_receipt = record_by_id[record_id]
            source = source_by_id.get(record_id)
            if source is None:
                raise ValueError("Holding manifest record disappeared from the TUEV tree")
            physical = _read_physical_record(source, record_receipt)
            require_feature_receipt_position_binding(
                foundation_receipt,
                physical.position_binding,
            )
            logical_count = record_receipt.metadata.output_sample_count
            full_record = (
                None
                if selected_arm_id == "C-CAR19"
                else _prepare_full_record(
                    physical,
                    arm_id=selected_arm_id,
                    logical_output_count=logical_count,
                )
            )
            interval_groups = tuple(
                sorted(groups_by_record[record_id], key=lambda group: group.crop_id)
            )
            prepared = tuple(
                _prepare_crop(
                    physical,
                    arm_id=selected_arm_id,
                    start_sample=group.start_sample,
                    stop_sample=group.stop_sample,
                    logical_output_count=logical_count,
                    full_record=full_record,
                )
                for group in interval_groups
            )
            record_preprocessing = prepared[0].record_preprocessing_receipt
            if any(
                item.record_preprocessing_receipt != record_preprocessing
                for item in prepared[1:]
            ):
                raise RuntimeError("One EDF record mixed preprocessing state policies")
            tokens = _encode_slot_zero(
                encoder,
                [item.waveform for item in prepared],
                device=torch_device,
                microbatch_size=microbatch_size,
            )
            crop_rows: list[dict[str, object]] = []
            for group, item, token in zip(interval_groups, prepared, tokens):
                authorization.require_selected_arm(selected_arm_id)
                waveform_sha = _payload_sha256(
                    item.waveform,
                    name="standard19_morphology_crop_volts",
                )
                preprocessing_sha = _typed_sha256(item.preprocessing_receipt)
                child_name = _crop_directory_name(group.crop_id)
                relative_bundle_path = f"groups/{child_name}"
                artifact = save_morphology_training_group_tokens(
                    groups_directory / child_name,
                    token,
                    interval_group=group,
                    record=record_receipt,
                    source_morphology_manifest_sha256=holding_manifest_sha,
                    foundation_feature_receipt=foundation_receipt,
                )
                entry = {
                    "crop_id": group.crop_id,
                    "relative_bundle_path": relative_bundle_path,
                    "bundle_manifest_sha256": artifact.manifest_sha256,
                    "tensor_sha256": artifact.tensor_sha256,
                }
                token_entries.append(entry)
                crop_row = {
                    "crop_id": group.crop_id,
                    "start_sample": group.start_sample,
                    "stop_sample": group.stop_sample,
                    "waveform_sha256": waveform_sha,
                    "preprocessing_receipt": item.preprocessing_receipt,
                    "preprocessing_receipt_sha256": preprocessing_sha,
                    **{
                        "relative_bundle_path": relative_bundle_path,
                        "bundle_manifest_sha256": artifact.manifest_sha256,
                        "tensor_sha256": artifact.tensor_sha256,
                    },
                }
                _require_fields(crop_row, _CROP_RECEIPT_FIELDS, label="crop receipt")
                crop_rows.append(crop_row)
                waveform_roster.append([group.crop_id, waveform_sha])
            record_row = {
                "record_id": record_id,
                "relative_edf_path": record_receipt.relative_edf_path,
                "edf_sha256": record_receipt.edf_sha256,
                "rec_sha256": record_receipt.rec_sha256,
                "source_sfreq_hz": physical.source_sfreq_hz,
                "source_sample_count": physical.source_sample_count,
                "logical_output_sample_count": logical_count,
                "raw_channel_names": list(physical.raw_channel_names),
                "raw_units": list(physical.raw_units),
                "position_binding": physical.position_binding.to_dict(),
                "standard19_mapping_sha256": physical.mapping_sha256,
                "selected_raw_volts_sha256": physical.raw_volts_sha256,
                "record_preprocessing_receipt": record_preprocessing,
                "record_preprocessing_receipt_sha256": _typed_sha256(
                    record_preprocessing
                ),
                "crops": crop_rows,
            }
            _require_fields(
                record_row, _RECORD_RECEIPT_FIELDS, label="record producer receipt"
            )
            producer_records.append(record_row)

        token_entries.sort(key=lambda row: str(row["crop_id"]))
        expected_crop_ids = tuple(
            group.crop_id for group in holding_manifest.interval_groups
        )
        observed_crop_ids = tuple(str(row["crop_id"]) for row in token_entries)
        if observed_crop_ids != expected_crop_ids:
            raise RuntimeError("First-party producer omitted or reordered a morphology crop")
        token_index_sha, crop_roster_sha, tensor_roster_sha = _write_token_index(
            corpus_directory,
            holding_manifest=holding_manifest,
            foundation_receipt=foundation_receipt,
            entries=token_entries,
        )
        _fsync_directory(groups_directory)
        _fsync_directory(corpus_directory)
        waveform_roster.sort(key=lambda row: row[0])
        authorization.assert_unchanged()
        preprocessing_authorization_payload = asdict(authorization.receipt)
        selected_spec_payload = asdict(selected_spec)
        foundation_payload = _foundation_payload(foundation_receipt)
        producer_receipt = {
            "schema_version": TUEV_MORPHOLOGY_PRODUCER_RECEIPT_SCHEMA,
            "token_schema_version": TUEV_MORPHOLOGY_FORMAL_TOKEN_SCHEMA,
            "producer_kind": TUEV_MORPHOLOGY_PRODUCER_KIND,
            "serialization": TUEV_MORPHOLOGY_PRODUCER_SERIALIZATION,
            "producer_source_sha256": producer_source_sha,
            "holding_manifest_sha256": holding_manifest_sha,
            "preflight_bundle_manifest_sha256": preflight.bundle_manifest_sha256,
            "preflight_receipt_sha256": preflight.preflight_receipt_sha256,
            "external_metadata_sha256": preflight.external_metadata_sha256,
            "source_roster_sha256": preflight.source_roster_sha256,
            "cohort_authorization_sha256": cohort_authorization.receipt_sha256,
            "preprocessing_authorization": preprocessing_authorization_payload,
            "preprocessing_authorization_receipt_sha256": (
                authorization.receipt.receipt_sha256
            ),
            "selected_arm_spec": selected_spec_payload,
            "selected_arm_spec_receipt_sha256": selected_spec.receipt_sha256,
            "foundation_feature_receipt": foundation_payload,
            "foundation_feature_receipt_sha256": foundation_sha,
            "token_index_sha256": token_index_sha,
            "crop_count": len(token_entries),
            "record_count": len(producer_records),
            "crop_roster_sha256": crop_roster_sha,
            "waveform_roster_sha256": _typed_sha256(waveform_roster),
            "tensor_roster_sha256": tensor_roster_sha,
            "semantic_channels": list(STANDARD_19),
            "output_sfreq_hz": MORPHOLOGY_OUTPUT_SFREQ_HZ,
            "context_samples": MORPHOLOGY_CONTEXT_SAMPLES,
            "context_seconds": MORPHOLOGY_CONTEXT_SECONDS,
            "retained_read_slot": MORPHOLOGY_READ_SLOT,
            "nonretained_slots_zeroed": True,
            "target_payload_absent": True,
            "records": producer_records,
        }
        _require_fields(
            producer_receipt,
            _PRODUCER_RECEIPT_FIELDS,
            label="morphology producer receipt",
        )
        receipt_raw = _canonical_json(producer_receipt)
        if not 1 <= len(receipt_raw) <= _MAX_JSON_BYTES:
            raise ValueError("Morphology producer receipt has an invalid size")
        receipt_path = staging / TUEV_MORPHOLOGY_PRODUCER_RECEIPT_FILE
        receipt_path.write_bytes(receipt_raw)
        _fsync_file(receipt_path)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        container = {
            "schema_version": TUEV_MORPHOLOGY_FORMAL_TOKEN_SCHEMA,
            "serialization": TUEV_MORPHOLOGY_PRODUCER_SERIALIZATION,
            "producer_receipt_file": TUEV_MORPHOLOGY_PRODUCER_RECEIPT_FILE,
            "producer_receipt_sha256": receipt_sha,
            "producer_receipt_size_bytes": len(receipt_raw),
            "token_corpus_directory": TUEV_MORPHOLOGY_TOKEN_CORPUS_DIRECTORY,
            "token_index_sha256": token_index_sha,
            "holding_manifest_sha256": holding_manifest_sha,
            "crop_count": len(token_entries),
            "record_count": len(producer_records),
        }
        _require_fields(container, _FORMAL_CONTAINER_FIELDS, label="producer manifest")
        container_raw = _canonical_json(container)
        manifest_path = staging / TUEV_MORPHOLOGY_PRODUCER_MANIFEST_FILE
        manifest_path.write_bytes(container_raw)
        _fsync_file(manifest_path)
        _fsync_directory(staging)
        authorization.assert_unchanged()
        if tuev_morphology_token_producer_source_sha256() != producer_source_sha:
            raise RuntimeError("Morphology producer source changed during publication")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Morphology master corpus already exists: {target}")
        os.replace(staging, target)
        published = True
        _fsync_directory(target.parent)
        return TUEVMorphologyMasterCorpusArtifact(
            path=target,
            bundle_manifest_sha256=hashlib.sha256(container_raw).hexdigest(),
            producer_receipt_sha256=receipt_sha,
            token_index_sha256=token_index_sha,
            holding_manifest_sha256=holding_manifest_sha,
            crop_count=len(token_entries),
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def materialize_tuev_morphology_master_corpus(
    output_directory: str | Path,
    *,
    edf_root: str | Path,
    holding_manifest: TUEVMorphologyManifest,
    preflight: VerifiedTUEVMorphologyPreflight,
    cohort_authorization: VerifiedTUEVMorphologyCohortAuthorization,
    preprocessing_selection: VerifiedPreprocessingSelectionCapability,
    labram_modeling_path: str | Path,
    labram_checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    microbatch_size: int = 16,
) -> TUEVMorphologyMasterCorpusArtifact:
    """Materialize the formal corpus directly from real EDF and audited LaBraM.

    No tensor, target label, interval roster, or fit/held/excluded roster is a
    caller parameter.  The holding crop roster is rebuilt from live REC bytes,
    and the preprocessing arm is obtained only from the opaque five-arm
    selection capability.
    """

    encoder = OfficialLaBraMEncoder(
        modeling_path=labram_modeling_path,
        checkpoint_path=labram_checkpoint_path,
        tile_seconds=MORPHOLOGY_CONTEXT_SECONDS,
    )
    return _materialize_first_party_core(
        output_directory,
        edf_root=edf_root,
        holding_manifest=holding_manifest,
        preflight=preflight,
        cohort_authorization=cohort_authorization,
        preprocessing_selection=preprocessing_selection,
        encoder=encoder,
        device=device,
        microbatch_size=microbatch_size,
    )


def _load_formal_container(
    path: str | Path,
    *,
    expected_bundle_manifest_sha256: str,
    expected_producer_receipt_sha256: str,
    expected_token_index_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_dir() or source.resolve(strict=True) != source:
        raise ValueError("Morphology producer bundle must be a canonical directory")
    expected_entries = {
        TUEV_MORPHOLOGY_PRODUCER_MANIFEST_FILE,
        TUEV_MORPHOLOGY_PRODUCER_RECEIPT_FILE,
        TUEV_MORPHOLOGY_TOKEN_CORPUS_DIRECTORY,
    }
    if {item.name for item in source.iterdir()} != expected_entries:
        raise ValueError("Morphology producer bundle contains missing or unknown entries")
    manifest_path = source / TUEV_MORPHOLOGY_PRODUCER_MANIFEST_FILE
    receipt_path = source / TUEV_MORPHOLOGY_PRODUCER_RECEIPT_FILE
    corpus_path = source / TUEV_MORPHOLOGY_TOKEN_CORPUS_DIRECTORY
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or corpus_path.is_symlink()
        or not corpus_path.is_dir()
    ):
        raise ValueError("Morphology producer bundle members changed type")
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha != _sha(
        expected_bundle_manifest_sha256,
        field="expected_bundle_manifest_sha256",
    ):
        raise ValueError("Morphology producer manifest SHA-256 mismatch")
    manifest = _parse_canonical_json(manifest_raw, label="producer manifest")
    _require_fields(manifest, _FORMAL_CONTAINER_FIELDS, label="producer manifest")
    if (
        manifest["schema_version"] != TUEV_MORPHOLOGY_FORMAL_TOKEN_SCHEMA
        or manifest["serialization"] != TUEV_MORPHOLOGY_PRODUCER_SERIALIZATION
        or manifest["producer_receipt_file"]
        != TUEV_MORPHOLOGY_PRODUCER_RECEIPT_FILE
        or manifest["token_corpus_directory"]
        != TUEV_MORPHOLOGY_TOKEN_CORPUS_DIRECTORY
    ):
        raise ValueError("Morphology producer schema/serialization boundary changed")
    receipt_raw = receipt_path.read_bytes()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    expected_receipt_sha = _sha(
        expected_producer_receipt_sha256,
        field="expected_producer_receipt_sha256",
    )
    if receipt_sha != expected_receipt_sha or receipt_sha != _sha(
        manifest["producer_receipt_sha256"],
        field="producer_receipt_sha256",
    ):
        raise ValueError("Morphology producer receipt SHA-256 mismatch")
    receipt_size = manifest["producer_receipt_size_bytes"]
    if (
        isinstance(receipt_size, bool)
        or not isinstance(receipt_size, int)
        or receipt_size != len(receipt_raw)
    ):
        raise ValueError("Morphology producer receipt size mismatch")
    receipt = _parse_canonical_json(receipt_raw, label="producer receipt")
    _require_fields(receipt, _PRODUCER_RECEIPT_FIELDS, label="producer receipt")
    if (
        receipt["schema_version"] != TUEV_MORPHOLOGY_PRODUCER_RECEIPT_SCHEMA
        or receipt["token_schema_version"] != TUEV_MORPHOLOGY_FORMAL_TOKEN_SCHEMA
        or receipt["producer_kind"] != TUEV_MORPHOLOGY_PRODUCER_KIND
        or receipt["serialization"] != TUEV_MORPHOLOGY_PRODUCER_SERIALIZATION
        or receipt["semantic_channels"] != list(STANDARD_19)
        or receipt["output_sfreq_hz"] != MORPHOLOGY_OUTPUT_SFREQ_HZ
        or receipt["context_samples"] != MORPHOLOGY_CONTEXT_SAMPLES
        or receipt["context_seconds"] != MORPHOLOGY_CONTEXT_SECONDS
        or receipt["retained_read_slot"] != MORPHOLOGY_READ_SLOT
        or receipt["nonretained_slots_zeroed"] is not True
        or receipt["target_payload_absent"] is not True
    ):
        raise ValueError("Morphology producer receipt changed its frozen policy")
    token_index_sha = _sha(
        expected_token_index_sha256,
        field="expected_token_index_sha256",
    )
    index_path = corpus_path / "index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("Morphology token corpus lacks a regular index")
    if (
        _file_sha256(index_path) != token_index_sha
        or manifest["token_index_sha256"] != token_index_sha
        or receipt["token_index_sha256"] != token_index_sha
    ):
        raise ValueError("Morphology token index SHA-256 mismatch")
    linked = (
        manifest["holding_manifest_sha256"],
        manifest["crop_count"],
        manifest["record_count"],
    )
    receipt_linked = (
        receipt["holding_manifest_sha256"],
        receipt["crop_count"],
        receipt["record_count"],
    )
    if linked != receipt_linked:
        raise ValueError("Morphology producer manifest and receipt disagree")
    records = receipt["records"]
    if not isinstance(records, list) or len(records) != receipt["record_count"]:
        raise ValueError("Morphology producer record count is invalid")
    crop_count = 0
    record_ids: list[str] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"records[{record_index}] must be an object")
        _require_fields(
            record,
            _RECORD_RECEIPT_FIELDS,
            label=f"records[{record_index}]",
        )
        record_ids.append(str(record["record_id"]))
        crops = record["crops"]
        if not isinstance(crops, list) or not crops:
            raise ValueError("Every producer record must contain at least one crop")
        crop_ids: list[str] = []
        for crop_index, crop in enumerate(crops):
            if not isinstance(crop, dict):
                raise TypeError("Producer crop receipt must be an object")
            _require_fields(
                crop,
                _CROP_RECEIPT_FIELDS,
                label=f"records[{record_index}].crops[{crop_index}]",
            )
            crop_ids.append(str(crop["crop_id"]))
            crop_count += 1
        if crop_ids != sorted(set(crop_ids)):
            raise ValueError("Producer crop receipts must be unique and sorted")
    if record_ids != sorted(set(record_ids)) or crop_count != receipt["crop_count"]:
        raise ValueError("Morphology producer record/crop roster is not canonical")
    return manifest, receipt


def _record_receipt_by_id(
    receipt: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    rows = receipt["records"]
    assert isinstance(rows, list)
    return {str(row["record_id"]): row for row in rows if isinstance(row, dict)}


def _replay_signal_and_crop_receipts(
    *,
    artifact_path: Path,
    edf_root: Path,
    holding_manifest: TUEVMorphologyManifest,
    sources: Sequence[TUEVMorphologySourceRecord],
    producer_receipt: Mapping[str, object],
    foundation_receipt: LaBraMFeatureReceipt,
    selected_arm_id: str,
) -> None:
    source_by_id = {source.record_id: source for source in sources}
    manifest_record_by_id = {
        record.record_id: record for record in holding_manifest.records
    }
    groups_by_record: dict[str, list[object]] = {}
    for group in holding_manifest.interval_groups:
        groups_by_record.setdefault(group.record_id, []).append(group)
    receipt_by_id = _record_receipt_by_id(producer_receipt)
    if set(receipt_by_id) != set(groups_by_record):
        raise ValueError("Producer receipt does not cover the active morphology records")
    waveform_roster: list[list[str]] = []
    tensor_roster: list[list[str]] = []
    for record_id in sorted(groups_by_record):
        record_receipt = manifest_record_by_id[record_id]
        physical = _read_physical_record(source_by_id[record_id], record_receipt)
        require_feature_receipt_position_binding(
            foundation_receipt,
            physical.position_binding,
        )
        stored_record = receipt_by_id[record_id]
        observed_record_identity = (
            stored_record["relative_edf_path"],
            stored_record["edf_sha256"],
            stored_record["rec_sha256"],
            stored_record["source_sfreq_hz"],
            stored_record["source_sample_count"],
            stored_record["logical_output_sample_count"],
            stored_record["raw_channel_names"],
            stored_record["raw_units"],
            stored_record["position_binding"],
            stored_record["standard19_mapping_sha256"],
            stored_record["selected_raw_volts_sha256"],
        )
        expected_record_identity = (
            record_receipt.relative_edf_path,
            record_receipt.edf_sha256,
            record_receipt.rec_sha256,
            physical.source_sfreq_hz,
            physical.source_sample_count,
            record_receipt.metadata.output_sample_count,
            list(physical.raw_channel_names),
            list(physical.raw_units),
            json.loads(
                _canonical_json(physical.position_binding.to_dict()).decode(
                    "utf-8"
                )
            ),
            physical.mapping_sha256,
            physical.raw_volts_sha256,
        )
        if observed_record_identity != expected_record_identity:
            raise ValueError("Producer raw EDF/header/position receipt failed replay")
        logical_count = record_receipt.metadata.output_sample_count
        full_record = (
            None
            if selected_arm_id == "C-CAR19"
            else _prepare_full_record(
                physical,
                arm_id=selected_arm_id,
                logical_output_count=logical_count,
            )
        )
        groups = tuple(sorted(groups_by_record[record_id], key=lambda group: group.crop_id))
        stored_crops = stored_record["crops"]
        assert isinstance(stored_crops, list)
        if [row["crop_id"] for row in stored_crops] != [group.crop_id for group in groups]:
            raise ValueError("Producer crop roster differs from the holding manifest")
        replayed_record_preprocessing: dict[str, object] | None = None
        for group, stored_crop in zip(groups, stored_crops):
            assert isinstance(stored_crop, dict)
            prepared = _prepare_crop(
                physical,
                arm_id=selected_arm_id,
                start_sample=group.start_sample,
                stop_sample=group.stop_sample,
                logical_output_count=logical_count,
                full_record=full_record,
            )
            waveform_sha = _payload_sha256(
                prepared.waveform,
                name="standard19_morphology_crop_volts",
            )
            preprocessing_sha = _typed_sha256(prepared.preprocessing_receipt)
            expected_relative = f"groups/{_crop_directory_name(group.crop_id)}"
            observed_crop = (
                stored_crop["crop_id"],
                stored_crop["start_sample"],
                stored_crop["stop_sample"],
                stored_crop["waveform_sha256"],
                stored_crop["preprocessing_receipt"],
                stored_crop["preprocessing_receipt_sha256"],
                stored_crop["relative_bundle_path"],
            )
            expected_crop = (
                group.crop_id,
                group.start_sample,
                group.stop_sample,
                waveform_sha,
                prepared.preprocessing_receipt,
                preprocessing_sha,
                expected_relative,
            )
            if observed_crop != expected_crop:
                raise ValueError("Producer absolute crop/preprocessing receipt failed replay")
            bundle_path = (
                artifact_path
                / TUEV_MORPHOLOGY_TOKEN_CORPUS_DIRECTORY
                / PurePosixPath(expected_relative)
            )
            loaded = load_morphology_training_group_tokens(
                bundle_path,
                expected_manifest_sha256=_sha(
                    stored_crop["bundle_manifest_sha256"],
                    field="bundle_manifest_sha256",
                ),
            )
            if loaded.tensor_sha256 != stored_crop["tensor_sha256"]:
                raise ValueError("Producer crop tensor receipt failed replay")
            if torch.count_nonzero(loaded.tokens[:, 1:, :]).item() != 0:
                raise ValueError("Formal morphology corpus retained a non-slot-zero token")
            replayed_record_preprocessing = prepared.record_preprocessing_receipt
            waveform_roster.append([group.crop_id, waveform_sha])
            tensor_roster.append([group.crop_id, loaded.tensor_sha256])
        assert replayed_record_preprocessing is not None
        if (
            stored_record["record_preprocessing_receipt"]
            != replayed_record_preprocessing
            or stored_record["record_preprocessing_receipt_sha256"]
            != _typed_sha256(replayed_record_preprocessing)
        ):
            raise ValueError("Producer record preprocessing receipt failed replay")
    waveform_roster.sort(key=lambda row: row[0])
    tensor_roster.sort(key=lambda row: row[0])
    if producer_receipt["waveform_roster_sha256"] != _typed_sha256(waveform_roster):
        raise ValueError("Producer waveform-roster SHA failed replay")
    if producer_receipt["tensor_roster_sha256"] != _typed_sha256(tensor_roster):
        raise ValueError("Producer tensor-roster SHA failed replay")


def _load_first_party_core(
    path: str | Path,
    *,
    edf_root: str | Path,
    holding_manifest: TUEVMorphologyManifest,
    preflight: VerifiedTUEVMorphologyPreflight,
    cohort_authorization: VerifiedTUEVMorphologyCohortAuthorization,
    preprocessing_selection: VerifiedPreprocessingSelectionCapability,
    encoder: nn.Module,
    expected_bundle_manifest_sha256: str,
    expected_producer_receipt_sha256: str,
    expected_token_index_sha256: str,
    replay_live_signal: bool = True,
) -> VerifiedTUEVMorphologyMasterCorpus:
    """Load a first-party corpus and optionally replay the live raw signal.

    Corpus publication performs the expensive EDF/REC, preprocessing and crop
    replay.  A later training run needs one numerical/structural token replay,
    but does not need to recompute the already published raw-signal audit.
    ``replay_live_signal=True`` remains the default for independent artifact
    auditing; the training entry point explicitly selects the faster path.
    """

    if not isinstance(replay_live_signal, bool):
        raise TypeError("replay_live_signal must be bool")

    source_path = Path(path).absolute()
    _, producer_receipt = _load_formal_container(
        source_path,
        expected_bundle_manifest_sha256=expected_bundle_manifest_sha256,
        expected_producer_receipt_sha256=expected_producer_receipt_sha256,
        expected_token_index_sha256=expected_token_index_sha256,
    )
    root = _validate_edf_root(edf_root) if replay_live_signal else None
    authorization = _authorize_selection(preprocessing_selection)
    authorization.assert_unchanged()
    selected_arm_id = authorization.selected_arm_id
    authorization.require_selected_arm(selected_arm_id)
    expected_preprocessing = asdict(authorization.receipt)
    expected_selected_arm_spec = json.loads(
        _canonical_json(
            asdict(FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[selected_arm_id])
        ).decode("utf-8")
    )
    if (
        producer_receipt["preprocessing_authorization"]
        != expected_preprocessing
        or producer_receipt["preprocessing_authorization_receipt_sha256"]
        != authorization.receipt.receipt_sha256
        or producer_receipt["selected_arm_spec"]
        != expected_selected_arm_spec
        or producer_receipt["selected_arm_spec_receipt_sha256"]
        != FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[selected_arm_id].receipt_sha256
    ):
        raise ValueError("Producer preprocessing selection receipt failed replay")
    sources = (
        _replay_holding_inputs(
            edf_root=root,
            holding_manifest=holding_manifest,
            preflight=preflight,
            cohort_authorization=cohort_authorization,
        )
        if replay_live_signal
        else None
    )
    expected_bindings = (
        producer_receipt["holding_manifest_sha256"],
        producer_receipt["preflight_bundle_manifest_sha256"],
        producer_receipt["preflight_receipt_sha256"],
        producer_receipt["external_metadata_sha256"],
        producer_receipt["source_roster_sha256"],
        producer_receipt["cohort_authorization_sha256"],
    )
    observed_bindings = (
        holding_manifest.manifest_sha256,
        preflight.bundle_manifest_sha256,
        preflight.preflight_receipt_sha256,
        preflight.external_metadata_sha256,
        preflight.source_roster_sha256,
        cohort_authorization.receipt_sha256,
    )
    if expected_bindings != observed_bindings:
        raise ValueError("Producer source/preflight/cohort binding failed replay")
    # The source digest records which implementation created the immutable
    # corpus; it is historical provenance, not a runtime compatibility gate.
    # Compatibility is established below by replaying the current signal,
    # crop, preprocessing, encoder and tensor contracts against the published
    # receipts.  Requiring the current Python file to remain byte-identical
    # would reject scientifically equivalent bug fixes before any numerical
    # validation can run.
    producer_source_sha = _sha(
        producer_receipt["producer_source_sha256"],
        field="producer_receipt.producer_source_sha256",
    )
    encoder, foundation_receipt, foundation_sha = _prepare_encoder(
        encoder,
        authorization=authorization,
        device=torch.device("cpu"),
    )
    del encoder
    if (
        producer_receipt["foundation_feature_receipt"]
        != _foundation_payload(foundation_receipt)
        or producer_receipt["foundation_feature_receipt_sha256"]
        != foundation_sha
    ):
        raise ValueError("Producer audited LaBraM receipt failed replay")
    corpus = _load_morphology_training_token_corpus_structural(
        source_path / TUEV_MORPHOLOGY_TOKEN_CORPUS_DIRECTORY,
        holding_manifest,
        expected_index_sha256=expected_token_index_sha256,
    )
    if (
        corpus.crop_count != producer_receipt["crop_count"]
        or corpus.crop_roster_sha256 != producer_receipt["crop_roster_sha256"]
        or corpus.tensor_roster_sha256 != producer_receipt["tensor_roster_sha256"]
        or corpus.foundation_feature_receipt_sha256 != foundation_sha
    ):
        raise ValueError("Producer token corpus lineage failed replay")
    if replay_live_signal:
        assert root is not None and sources is not None
        _replay_signal_and_crop_receipts(
            artifact_path=source_path,
            edf_root=root,
            holding_manifest=holding_manifest,
            sources=sources,
            producer_receipt=producer_receipt,
            foundation_receipt=foundation_receipt,
            selected_arm_id=selected_arm_id,
        )
    authorization.assert_unchanged()
    return VerifiedTUEVMorphologyMasterCorpus(
        _issuer=_VERIFIED_FORMAL_CORPUS_ISSUER,
        path=source_path,
        bundle_manifest_sha256=_sha(
            expected_bundle_manifest_sha256,
            field="expected_bundle_manifest_sha256",
        ),
        producer_receipt_sha256=_sha(
            expected_producer_receipt_sha256,
            field="expected_producer_receipt_sha256",
        ),
        producer_source_sha256=producer_source_sha,
        selected_arm_id=selected_arm_id,
        token_corpus=corpus,
        holding_manifest=holding_manifest,
        preprocessing_authorization=authorization,
    )


def load_tuev_morphology_master_corpus(
    path: str | Path,
    *,
    edf_root: str | Path,
    holding_manifest: TUEVMorphologyManifest,
    preflight: VerifiedTUEVMorphologyPreflight,
    cohort_authorization: VerifiedTUEVMorphologyCohortAuthorization,
    preprocessing_selection: VerifiedPreprocessingSelectionCapability,
    labram_modeling_path: str | Path,
    labram_checkpoint_path: str | Path,
    expected_bundle_manifest_sha256: str,
    expected_producer_receipt_sha256: str,
    expected_token_index_sha256: str,
    replay_live_signal: bool = True,
) -> VerifiedTUEVMorphologyMasterCorpus:
    """Strictly replay source, crop, preprocessing, checkpoint, and corpus bytes."""

    encoder = OfficialLaBraMEncoder(
        modeling_path=labram_modeling_path,
        checkpoint_path=labram_checkpoint_path,
        tile_seconds=MORPHOLOGY_CONTEXT_SECONDS,
    )
    return _load_first_party_core(
        path,
        edf_root=edf_root,
        holding_manifest=holding_manifest,
        preflight=preflight,
        cohort_authorization=cohort_authorization,
        preprocessing_selection=preprocessing_selection,
        encoder=encoder,
        expected_bundle_manifest_sha256=expected_bundle_manifest_sha256,
        expected_producer_receipt_sha256=expected_producer_receipt_sha256,
        expected_token_index_sha256=expected_token_index_sha256,
        replay_live_signal=replay_live_signal,
    )


__all__ = [
    "TUEV_MORPHOLOGY_FORMAL_TOKEN_SCHEMA",
    "TUEV_MORPHOLOGY_PRODUCER_KIND",
    "TUEV_MORPHOLOGY_PRODUCER_RECEIPT_SCHEMA",
    "TUEVMorphologyMasterCorpusArtifact",
    "VerifiedTUEVMorphologyMasterCorpus",
    "load_tuev_morphology_master_corpus",
    "materialize_tuev_morphology_master_corpus",
    "tuev_morphology_token_producer_source_sha256",
]
