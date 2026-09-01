"""Upstream-literal 18-lead SeizureTransformer primitives for TUSZ.

This module is intentionally separate from
``seizuretransformer_cleanroom_registry_v1``.  That registry freezes a project
robust transform (polyphase resampling, zero-phase 0.5--100 Hz filtering and
median/MAD scaling).  The public SeizureTransformer seizure code instead does
the following:

1. load the 19 10--20 referential electrodes, replacing absent electrodes by
   zeros (``epilepsy2bids==0.0.7`` behaviour);
2. construct the exact 18-entry ``Eeg.BIPOLAR_DBANANA`` montage;
3. fit population mean/std on every complete bipolar recording;
4. use SciPy Fourier resampling to 256 Hz;
5. make 60 s tiles (15 s hop in training, 60 s hop at evaluation);
6. reset causal 0.5--120 Hz, 1 Hz-notch and 60 Hz-notch filter state in every
   tile; and
7. decode ``posterior > 0.8`` by five-sample opening, five-sample closing and
   removal of runs shorter than two seconds.

The functions below replay those numerical semantics without opening an EDF,
annotation, source-dev/source-eval file, spreadsheet, or model checkpoint.
Missing-electrode zero filling is retained and explicitly receipted; it is not
silently converted into a technical failure.  A record can still fail if the
resulting bipolar lead has zero/non-finite population variance, because the
upstream z-score would then create non-finite values.

The public repository's default materializer combines Siena and TUSZ before
training.  ``TUSZ_ONLY_PROFILE_ID`` therefore names a deliberate exposure
change, not a reproduction of the paper checkpoint.  Likewise, a project
clean-room run must replace upstream official-dev checkpoint selection with a
patient-disjoint source-train inner validation phase.  Both contracts are
reported by :func:`native18_training_contract` instead of being conflated.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import scipy
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import butter, iirnotch, lfilter, resample
import torch
from torch import Tensor, nn


SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_seizuretransformer_tusz_native18_upstream_literal_v1"
)
PROFILE_ID: Final[str] = "seizuretransformer_upstream_literal_native18_v1"
TUSZ_ONLY_PROFILE_ID: Final[str] = (
    "seizuretransformer_tusz_only_upstream_native18_cleanroom_v1"
)
EXTERNAL_NATIVE19_PROFILE_ID: Final[str] = (
    "seizuretransformer_external_artifact_upstream_native19_diagnostic_v1"
)
EXTERNAL_CHECKPOINT_SHA256: Final[str] = (
    "2cdc841001a0fbcdf1dfcbb02b3a26fa7af14002e01ebf9815fa09c82be06f61"
)
UPSTREAM_COMMIT: Final[str] = "cf83f5906a8aea88b60b56e4f962c5d6657c28f7"
UPSTREAM_HANDLE_DATA_SHA256: Final[str] = (
    "cbc088d9c5ba9b78b1457c461d1788419ae13c3b82c06442291e34cebcf6f2f0"
)
UPSTREAM_RESULT_SHA256: Final[str] = (
    "33b4b626cf8f23d6f127354431e89b9e08a71e6fa9ddd01d719eac59c86fff98"
)
UPSTREAM_POST_PROCESS_SHA256: Final[str] = (
    "e7eb3939d13c169efbbbbe8ff7a31f7597b77c3633ebcb6917aa89342ad01ebe"
)
UPSTREAM_TRAIN_SHA256: Final[str] = (
    "3b655f0b81dc9324f2a041173fe720cbd4556f4930c5e8e1a748ec28781ed101"
)
UPSTREAM_DATASET_SHA256: Final[str] = (
    "0ab8d19853470250e262187bfba2725731f1b5bd5daa9c907d30678af3c7c4fe"
)

TARGET_FS_HZ: Final[int] = 256
TILE_SECONDS: Final[int] = 60
TILE_SAMPLES: Final[int] = TARGET_FS_HZ * TILE_SECONDS
TRAIN_HOP_SAMPLES: Final[int] = TILE_SAMPLES // 4
TEST_HOP_SAMPLES: Final[int] = TILE_SAMPLES
RELEASED_THRESHOLD: Final[float] = 0.8
MORPHOLOGY_KERNEL_SAMPLES: Final[int] = 5
MINIMUM_EVENT_SECONDS: Final[float] = 2.0
MINIMUM_EVENT_SAMPLES: Final[int] = int(MINIMUM_EVENT_SECONDS * TARGET_FS_HZ)

# These are the exact epilepsy2bids 0.0.7 values used by get_data_18().  The
# historical names are part of the upstream axis contract; canonical aliases
# are exposed separately for callers whose physical carrier uses T7/T8/P7/P8.
UPSTREAM_REFERENTIAL_19: Final[tuple[str, ...]] = (
    "FP1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "FZ",
    "CZ",
    "PZ",
    "FP2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T4",
    "T6",
)
UPSTREAM_BIPOLAR_DBANANA_18: Final[tuple[str, ...]] = (
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FZ-CZ",
    "CZ-PZ",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
)
CANONICAL_REFERENTIAL_19: Final[tuple[str, ...]] = (
    "FP1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T7",
    "P7",
    "FZ",
    "CZ",
    "PZ",
    "FP2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T8",
    "P8",
)
CANONICAL_BIPOLAR_18: Final[tuple[str, ...]] = tuple(
    name.replace("T3", "T7")
    .replace("T5", "P7")
    .replace("T4", "T8")
    .replace("T6", "P8")
    for name in UPSTREAM_BIPOLAR_DBANANA_18
)

_ALIAS_TO_UPSTREAM: Final[dict[str, str]] = {
    **{name: name for name in UPSTREAM_REFERENTIAL_19},
    "T7": "T3",
    "P7": "T5",
    "T8": "T4",
    "P8": "T6",
}
_CONTENT_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"


class Native18NumericalFailure(ValueError):
    """The literal upstream transform would produce non-finite samples."""


@dataclass(frozen=True)
class Native18ReferentialCarrier:
    """Exact zero-filled upstream 19-axis referential carrier."""

    signal: np.ndarray
    _receipt_json: str

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class Native18Record:
    """Whole-record z-scored and Fourier-resampled 18-axis model carrier."""

    signal: np.ndarray
    _receipt_json: str

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class ExternalNative19Record:
    """Whole-record upstream referential carrier for the external 19-axis artifact."""

    signal: np.ndarray
    _receipt_json: str

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class Native18InferenceTile:
    tile_index: int
    start_sample: int
    observed_sample_count: int
    right_padding_sample_count: int


@dataclass(frozen=True)
class Native18DecodedEvents:
    binary_mask: np.ndarray
    event_sample_spans: tuple[tuple[int, int], ...]
    _receipt_json: str

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class Native18InferenceResult:
    posterior: np.ndarray
    decoded: Native18DecodedEvents
    _receipt_json: str

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _CONTENT_PENDING
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _float_payload_receipt(value: np.ndarray, *, semantic: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype="<f8")
    return {
        "semantic": semantic,
        "dtype": "float64_little_endian",
        "shape": [int(item) for item in array.shape],
        "payload_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "minimum": float(np.min(array)) if array.size else None,
        "maximum": float(np.max(array)) if array.size else None,
    }


def _float32_payload_receipt(value: np.ndarray, *, semantic: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype="<f4")
    return {
        "semantic": semantic,
        "dtype": "float32_little_endian",
        "shape": [int(item) for item in array.shape],
        "payload_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "minimum": float(np.min(array)) if array.size else None,
        "maximum": float(np.max(array)) if array.size else None,
    }


def _normalize_electrode_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("electrode names must be non-empty strings")
    name = value.strip().upper()
    if name not in _ALIAS_TO_UPSTREAM:
        raise ValueError(
            "native18 accepts only explicit 10-20 names and T3/T4/T5/T6 "
            "or T7/T8/P7/P8 aliases"
        )
    return _ALIAS_TO_UPSTREAM[name]


def materialize_upstream_literal_referential19(
    observed_referential: np.ndarray,
    observed_electrodes: Sequence[str],
) -> Native18ReferentialCarrier:
    """Replay ``Eeg.loadEdf`` missing-electrode zero filling from an array.

    The caller supplies already physical, same-clock referential EEG.  Axis
    recognition is deliberately stricter than epilepsy2bids' regular
    expression lookup so an ambiguous raw EDF label cannot silently choose the
    first match.  A canonical EDF reader must resolve those labels first.
    """

    values = np.asarray(observed_referential)
    if values.ndim != 2 or values.shape[0] != len(observed_electrodes):
        raise ValueError("observed referential EEG must be [electrodes,samples]")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("observed referential EEG may not be empty")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("observed referential EEG must be finite floating point")

    normalized = tuple(_normalize_electrode_name(name) for name in observed_electrodes)
    if len(normalized) != len(set(normalized)):
        raise ValueError("observed electrodes alias to a duplicate 10-20 axis")
    index_by_name = {name: index for index, name in enumerate(normalized)}
    unexpected = sorted(set(index_by_name).difference(UPSTREAM_REFERENTIAL_19))
    if unexpected:
        raise ValueError(f"unexpected referential electrodes: {unexpected}")

    output = np.zeros((len(UPSTREAM_REFERENTIAL_19), values.shape[1]), dtype="<f8")
    missing: list[str] = []
    for output_index, name in enumerate(UPSTREAM_REFERENTIAL_19):
        source_index = index_by_name.get(name)
        if source_index is None:
            missing.append(name)
        else:
            output[output_index] = np.asarray(values[source_index], dtype=np.float64)
    output.setflags(write=False)

    affected_leads = [
        lead
        for lead in UPSTREAM_BIPOLAR_DBANANA_18
        if any(electrode in set(missing) for electrode in lead.split("-"))
    ]
    both_missing_leads = [
        lead
        for lead in UPSTREAM_BIPOLAR_DBANANA_18
        if set(lead.split("-")).issubset(missing)
    ]
    receipt = _content_address(
        {
            "schema_version": "native18_upstream_literal_referential19_v1",
            "profile_id": PROFILE_ID,
            "input_electrodes_as_supplied": list(observed_electrodes),
            "input_electrodes_upstream_aliases": list(normalized),
            "output_electrodes": list(UPSTREAM_REFERENTIAL_19),
            "missing_electrodes_zero_filled": missing,
            "missing_electrode_count": len(missing),
            "bipolar_leads_affected_by_zero_fill": affected_leads,
            "bipolar_leads_with_both_electrodes_missing": both_missing_leads,
            "upstream_literal_missing_electrode_policy": "replace_by_zeros",
            "physically_strict_missing_electrode_rejection_used": False,
            "output_payload": _float_payload_receipt(
                output, semantic="upstream_literal_zero_filled_referential19"
            ),
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return Native18ReferentialCarrier(
        signal=output,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
    )


def materialize_physically_strict_referential19(
    observed_referential: np.ndarray,
    observed_electrodes: Sequence[str],
) -> Native18ReferentialCarrier:
    """Sensitivity analysis that rejects missing axes before transformation."""

    result = materialize_upstream_literal_referential19(
        observed_referential, observed_electrodes
    )
    if result.receipt["missing_electrode_count"]:
        raise ValueError(
            "physically-strict native18 sensitivity profile requires all 19 electrodes"
        )
    receipt = result.receipt
    receipt["physically_strict_missing_electrode_rejection_used"] = True
    receipt["receipt_sha256"] = _CONTENT_PENDING
    receipt = _content_address(receipt)
    return Native18ReferentialCarrier(
        signal=result.signal,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
    )


def _derive_upstream_bipolar18(referential19: np.ndarray) -> np.ndarray:
    if referential19.shape[0] != len(UPSTREAM_REFERENTIAL_19):
        raise ValueError("upstream referential carrier must have exactly 19 axes")
    index = {name: axis for axis, name in enumerate(UPSTREAM_REFERENTIAL_19)}
    bipolar = np.stack(
        [
            referential19[index[left]] - referential19[index[right]]
            for left, right in (
                tuple(name.split("-", 1)) for name in UPSTREAM_BIPOLAR_DBANANA_18
            )
        ],
        axis=0,
    )
    return np.ascontiguousarray(bipolar, dtype="<f8")


def transform_upstream_native18_record(
    carrier: Native18ReferentialCarrier,
    *,
    source_sampling_rate_hz: float,
) -> Native18Record:
    """Montage, whole-record population z-score, then SciPy FFT resample."""

    if not isinstance(carrier, Native18ReferentialCarrier):
        raise TypeError("native18 transform requires a receipted referential carrier")
    source_fs = float(source_sampling_rate_hz)
    if not math.isfinite(source_fs) or source_fs <= 0:
        raise ValueError("source sampling rate must be finite and positive")
    referential = np.asarray(carrier.signal, dtype=np.float64)
    if referential.ndim != 2 or not np.isfinite(referential).all():
        raise ValueError("native18 referential carrier is invalid")

    bipolar = _derive_upstream_bipolar18(referential)
    center = np.mean(bipolar, axis=1, dtype=np.float64)
    scale = np.std(bipolar, axis=1, dtype=np.float64, ddof=0)
    invalid = np.flatnonzero(~np.isfinite(scale) | (scale <= 0.0))
    if invalid.size:
        names = [UPSTREAM_BIPOLAR_DBANANA_18[int(index)] for index in invalid]
        raise Native18NumericalFailure(
            "upstream whole-record z-score has zero/non-finite scale for "
            + ",".join(names)
        )
    normalized = (bipolar - center[:, None]) / scale[:, None]
    if not np.isfinite(normalized).all():
        raise Native18NumericalFailure("upstream whole-record z-score became non-finite")

    if source_fs == float(TARGET_FS_HZ):
        target = normalized
        resample_applied = False
    else:
        # This deliberately mirrors upstream ``int(n / fs * float(256))`` and
        # scipy.signal.resample rather than using exact-rational polyphase DSP.
        target_sample_count = int(
            normalized.shape[1] / source_fs * float(TARGET_FS_HZ)
        )
        if target_sample_count < 1:
            raise Native18NumericalFailure("FFT resampling would create an empty record")
        target = resample(normalized, target_sample_count, axis=1)
        resample_applied = True
    target = np.ascontiguousarray(target, dtype="<f8")
    if not np.isfinite(target).all():
        raise Native18NumericalFailure("upstream FFT-resampled record is non-finite")
    target.setflags(write=False)

    receipt = _content_address(
        {
            "schema_version": SCHEMA_VERSION,
            "profile_id": PROFILE_ID,
            "referential_carrier_receipt_sha256": carrier.receipt[
                "receipt_sha256"
            ],
            "ordered_operations": [
                "upstream_literal_missing_referential_zero_fill",
                "exact_Eeg_BIPOLAR_DBANANA_matrix_first_minus_second",
                "whole_record_per_bipolar_population_mean_std_zscore",
                "whole_record_scipy_signal_FFT_resample_to_256Hz_if_needed",
            ],
            "upstream_bipolar_order": list(UPSTREAM_BIPOLAR_DBANANA_18),
            "canonical_alias_bipolar_order": list(CANONICAL_BIPOLAR_18),
            "source_sampling_rate_hz": source_fs,
            "source_sample_count": int(referential.shape[1]),
            "target_sampling_rate_hz": TARGET_FS_HZ,
            "target_sample_count": int(target.shape[1]),
            "normalization_center": "whole_record_float64_mean",
            "normalization_scale": "whole_record_float64_population_std_ddof0",
            "normalization_epsilon_or_clip": None,
            "FFT_resample_applied": resample_applied,
            "resample_target_length_formula": "int(N/source_fs*256.0)",
            "tile_filter_applied_to_whole_record": False,
            "output_payload": _float_payload_receipt(
                target, semantic="native18_whole_record_zscore_FFT_resampled"
            ),
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return Native18Record(
        signal=target,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
    )


def transform_external_upstream_native19_record(
    carrier: Native18ReferentialCarrier,
    *,
    source_sampling_rate_hz: float,
) -> ExternalNative19Record:
    """Replay public ``get_data()`` for the provenance-unknown 19-axis artifact.

    Unlike ``get_data_18()``, upstream ``get_data()`` asserts that every one of
    its 19 referential axes exists.  A zero-filled missing-electrode carrier is
    therefore rejected here.  This diagnostic is deliberately not a substitute
    for the paper-native 18-lead path.
    """

    if not isinstance(carrier, Native18ReferentialCarrier):
        raise TypeError("external native19 transform requires a receipted carrier")
    if carrier.receipt["missing_electrode_count"]:
        raise ValueError("upstream get_data native19 requires all 19 electrodes")
    source_fs = float(source_sampling_rate_hz)
    if not math.isfinite(source_fs) or source_fs <= 0:
        raise ValueError("source sampling rate must be finite and positive")
    source = np.asarray(carrier.signal, dtype=np.float64)
    center = np.mean(source, axis=1, dtype=np.float64)
    scale = np.std(source, axis=1, dtype=np.float64, ddof=0)
    invalid = np.flatnonzero(~np.isfinite(scale) | (scale <= 0.0))
    if invalid.size:
        names = [UPSTREAM_REFERENTIAL_19[int(index)] for index in invalid]
        raise Native18NumericalFailure(
            "external native19 whole-record z-score has invalid scale for "
            + ",".join(names)
        )
    normalized = (source - center[:, None]) / scale[:, None]
    if source_fs == float(TARGET_FS_HZ):
        target = normalized
        resample_applied = False
    else:
        target_sample_count = int(source.shape[1] / source_fs * float(TARGET_FS_HZ))
        if target_sample_count < 1:
            raise Native18NumericalFailure("native19 FFT resampling creates empty support")
        target = resample(normalized, target_sample_count, axis=1)
        resample_applied = True
    target = np.ascontiguousarray(target, dtype="<f8")
    if target.shape[0] != 19 or not np.isfinite(target).all():
        raise Native18NumericalFailure("external native19 carrier is malformed")
    target.setflags(write=False)
    receipt = _content_address(
        {
            "schema_version": "external_seizuretransformer_native19_transform_v1",
            "profile_id": EXTERNAL_NATIVE19_PROFILE_ID,
            "artifact_sha256_expected": EXTERNAL_CHECKPOINT_SHA256,
            "artifact_uploader_is_upstream_author_verified": False,
            "artifact_original_checkpoint_hash_verified": False,
            "artifact_conversion_log_verified": False,
            "referential_carrier_receipt_sha256": carrier.receipt["receipt_sha256"],
            "ordered_operations": [
                "exact_upstream_get_data_referential19_order",
                "whole_record_per_referential_population_mean_std_zscore",
                "whole_record_scipy_signal_FFT_resample_to_256Hz_if_needed",
            ],
            "referential_order": list(UPSTREAM_REFERENTIAL_19),
            "source_sampling_rate_hz": source_fs,
            "source_sample_count": int(source.shape[1]),
            "target_sampling_rate_hz": TARGET_FS_HZ,
            "target_sample_count": int(target.shape[1]),
            "FFT_resample_applied": resample_applied,
            "normalization_epsilon_or_clip": None,
            "output_payload": _float_payload_receipt(
                target, semantic="external_artifact_native19_referential_carrier"
            ),
            "paper_native18_reproduction": False,
            "diagnostic_only": True,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return ExternalNative19Record(
        signal=target,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
    )


def upstream_training_tile_starts(sample_count: int) -> tuple[int, ...]:
    """Exact ``SeizureDataset.window_idx`` starts, including its end omission."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count < 0:
        raise ValueError("sample_count may not be negative")
    whole_second_samples = int(sample_count / TARGET_FS_HZ) * TARGET_FS_HZ
    exclusive_stop = whole_second_samples - TILE_SAMPLES
    if exclusive_stop <= 0:
        return ()
    return tuple(range(0, exclusive_stop, TRAIN_HOP_SAMPLES))


def plan_upstream_inference_tiles(sample_count: int) -> tuple[Native18InferenceTile, ...]:
    """Exact non-overlap testing starts with one zero-padded final tile."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count < TILE_SAMPLES:
        tile_count = 1
    else:
        tile_count = 1 + math.ceil((sample_count - TILE_SAMPLES) / TILE_SAMPLES)
    result: list[Native18InferenceTile] = []
    for tile_index in range(tile_count):
        start = tile_index * TILE_SAMPLES
        observed = min(TILE_SAMPLES, sample_count - start)
        if observed <= 0:
            raise AssertionError("upstream inference tile plan crossed record end")
        result.append(
            Native18InferenceTile(
                tile_index=tile_index,
                start_sample=start,
                observed_sample_count=observed,
                right_padding_sample_count=TILE_SAMPLES - observed,
            )
        )
    if sum(tile.observed_sample_count for tile in result) != sample_count:
        raise AssertionError("upstream inference plan does not cover every sample once")
    return tuple(result)


def upstream_tile_filter_coefficients() -> dict[str, np.ndarray]:
    """Return the exact coefficients constructed by the vendored data helpers."""

    nyquist = 0.5 * TARGET_FS_HZ
    band_b, band_a = butter(
        3, [0.5 / nyquist, 120.0 / nyquist], btype="band"
    )
    notch_1_b, notch_1_a = iirnotch(1.0, Q=30, fs=TARGET_FS_HZ)
    notch_60_b, notch_60_a = iirnotch(60.0, Q=30, fs=TARGET_FS_HZ)
    return {
        "bandpass_b": np.asarray(band_b, dtype=np.float64),
        "bandpass_a": np.asarray(band_a, dtype=np.float64),
        "notch_1_b": np.asarray(notch_1_b, dtype=np.float64),
        "notch_1_a": np.asarray(notch_1_a, dtype=np.float64),
        "notch_60_b": np.asarray(notch_60_b, dtype=np.float64),
        "notch_60_a": np.asarray(notch_60_a, dtype=np.float64),
    }


def _filter_upstream_tile(tile: np.ndarray, *, channel_count: int) -> np.ndarray:
    values = np.asarray(tile)
    if values.shape != (channel_count, TILE_SAMPLES):
        raise ValueError(
            f"upstream model tile must have shape [{channel_count},15360]"
        )
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("upstream model tile must be finite floating point")
    coefficients = upstream_tile_filter_coefficients()
    filtered = lfilter(
        coefficients["bandpass_b"], coefficients["bandpass_a"], values, axis=-1
    )
    filtered = lfilter(
        coefficients["notch_1_b"], coefficients["notch_1_a"], filtered, axis=-1
    )
    filtered = lfilter(
        coefficients["notch_60_b"],
        coefficients["notch_60_a"],
        filtered,
        axis=-1,
    )
    result = np.ascontiguousarray(filtered, dtype="<f4")
    if not np.isfinite(result).all():
        raise Native18NumericalFailure("tile-local upstream filters became non-finite")
    return result


def filter_upstream_native18_tile(tile: np.ndarray) -> np.ndarray:
    """Apply the three causal filters with fresh zero state for one 18-axis tile."""

    return _filter_upstream_tile(tile, channel_count=18)


def filter_external_upstream_native19_tile(tile: np.ndarray) -> np.ndarray:
    """The identical tile-local filters for the external 19-axis artifact."""

    return _filter_upstream_tile(tile, channel_count=19)


def materialize_upstream_training_tile(
    record: Native18Record,
    target: np.ndarray,
    *,
    start_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and filter one fully observed 75%-overlap training tile."""

    if not isinstance(record, Native18Record):
        raise TypeError("training tile requires a Native18Record")
    if start_sample not in set(upstream_training_tile_starts(record.signal.shape[1])):
        raise ValueError("start_sample is not an exact upstream training window")
    labels = np.asarray(target)
    if labels.shape != (record.signal.shape[1],):
        raise ValueError("dense target must align with the transformed record")
    if not np.isfinite(labels).all() or np.any((labels != 0) & (labels != 1)):
        raise ValueError("dense target must be binary and finite")
    stop = start_sample + TILE_SAMPLES
    model_input = filter_upstream_native18_tile(record.signal[:, start_sample:stop])
    model_target = np.ascontiguousarray(labels[start_sample:stop], dtype="<f4")
    return model_input, model_target


def upstream_training_window_category(target_tile: np.ndarray) -> str:
    """Replay upstream partial/full/background window stratification."""

    target = np.asarray(target_tile)
    if target.shape != (TILE_SAMPLES,) or np.any((target != 0) & (target != 1)):
        raise ValueError("training target tile must be a binary 15360-vector")
    positive = int(np.sum(target))
    if positive == 0:
        return "no_seizure"
    if positive == TILE_SAMPLES:
        return "full_seizure"
    return "partial_seizure"


def select_upstream_training_window_indices(
    categories: Sequence[str],
    *,
    seed: int,
    alpha: float = 0.7,
    beta: float = 2.0,
) -> tuple[int, ...]:
    """Deterministic clean-room replay of upstream category truncation.

    Upstream calls ``sklearn.utils.shuffle`` without ``random_state``.  The
    clean-room lane therefore binds an explicit seed while retaining its
    category quotas and final shuffle.  This is a disclosed determinism repair.
    """

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(alpha) or not math.isfinite(beta) or alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be finite and nonnegative")
    allowed = {"partial_seizure", "full_seizure", "no_seizure"}
    if any(category not in allowed for category in categories):
        raise ValueError("unknown upstream training window category")
    partial = [index for index, value in enumerate(categories) if value == "partial_seizure"]
    full = [index for index, value in enumerate(categories) if value == "full_seizure"]
    negative = [index for index, value in enumerate(categories) if value == "no_seizure"]
    generator = np.random.default_rng(seed)
    generator.shuffle(partial)
    generator.shuffle(full)
    generator.shuffle(negative)
    full = full[: int(len(partial) * alpha)]
    negative = negative[: int(len(partial) * beta)]
    selected = partial + full + negative
    generator.shuffle(selected)
    return tuple(int(index) for index in selected)


def build_upstream_native18_model(*, seed: int, device: str = "cpu") -> nn.Module:
    """Randomly initialize the exact vendored 18-channel architecture."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    from third_party.SeizureTransformer.time_step_level.model import (
        SeizureTransformer,
    )

    devices = [torch_device.index or 0] if torch_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed(seed)
        model = SeizureTransformer(
            in_channels=18,
            in_samples=TILE_SAMPLES,
            dim_feedforward=2048,
            num_layers=8,
            num_heads=4,
            drop_rate=0.1,
        ).to(torch_device)
    return model


def build_upstream_native18_optimizer(model: nn.Module) -> torch.optim.RAdam:
    """Build the public training script's RAdam optimizer."""

    if not isinstance(model, nn.Module) or not list(model.parameters()):
        raise TypeError("optimizer requires a non-empty Torch model")
    return torch.optim.RAdam(model.parameters(), lr=1e-4, weight_decay=2e-5)


def train_upstream_native18_batches(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: str,
) -> dict[str, Any]:
    """Run one executable upstream-style BCE epoch over caller-authorized tiles.

    Split/reference authority remains outside this numerical function.  The
    caller must supply only source-train selection-fit or final-refit batches;
    source-dev/source-eval access is neither accepted nor performed here.
    """

    torch_device = torch.device(device)
    model.train()
    batch_losses: list[float] = []
    sample_count = 0
    for model_input, dense_target in batches:
        if model_input.ndim != 3 or model_input.shape[1:] != (18, TILE_SAMPLES):
            raise ValueError("training input batch must be [batch,18,15360]")
        if dense_target.shape != (model_input.shape[0], TILE_SAMPLES):
            raise ValueError("training dense target batch geometry differs")
        values = model_input.to(device=torch_device, dtype=torch.float32)
        target = dense_target.to(device=torch_device, dtype=torch.float32)
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(target).all()):
            raise ValueError("training batch contains non-finite values")
        if bool(torch.any((target != 0) & (target != 1))):
            raise ValueError("training target must be binary")
        optimizer.zero_grad()
        probability = model(values)
        if probability.shape != target.shape:
            raise ValueError("native18 model output geometry differs from target")
        loss = torch.nn.functional.binary_cross_entropy(probability, target)
        if not bool(torch.isfinite(loss)):
            raise Native18NumericalFailure("native18 BCE became non-finite")
        loss.backward()
        optimizer.step()
        batch_losses.append(float(loss.detach().cpu()))
        sample_count += int(target.numel())
    if not batch_losses:
        raise ValueError("training epoch received no batches")
    return _content_address(
        {
            "schema_version": "native18_upstream_BCE_epoch_numeric_receipt_v1",
            "profile_id": TUSZ_ONLY_PROFILE_ID,
            "batch_count": len(batch_losses),
            "dense_target_sample_count": sample_count,
            "mean_batch_BCE": float(np.mean(batch_losses)),
            "minimum_batch_BCE": float(np.min(batch_losses)),
            "maximum_batch_BCE": float(np.max(batch_losses)),
            "optimizer": "torch.optim.RAdam",
            "learning_rate": 1e-4,
            "weight_decay": 2e-5,
            "gradient_clipping": None,
            "source_dev_or_source_eval_opened_by_numeric_executor": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def decode_upstream_native18_posterior(
    posterior: np.ndarray,
    *,
    threshold: float = RELEASED_THRESHOLD,
) -> Native18DecodedEvents:
    """Exact released threshold/open/close/min-duration sample decoder."""

    values = np.asarray(posterior)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("posterior must be a non-empty vector")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("posterior must be finite floating point")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("posterior must lie in [0,1]")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0,1]")
    binary = values > threshold
    structure = np.ones(MORPHOLOGY_KERNEL_SAMPLES, dtype=bool)
    binary = binary_opening(binary, structure=structure)
    binary = binary_closing(binary, structure=structure)

    output = binary.astype(np.uint8, copy=True)
    padded = np.pad(output, (1, 1), mode="constant")
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    spans = [(int(start), int(stop)) for start, stop in changes.reshape(-1, 2)]
    for start, stop in spans:
        if stop - start < MINIMUM_EVENT_SAMPLES:
            output[start:stop] = 0
    padded = np.pad(output, (1, 1), mode="constant")
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    retained = tuple(
        (int(start), int(stop)) for start, stop in changes.reshape(-1, 2)
    )
    output.setflags(write=False)
    receipt = _content_address(
        {
            "schema_version": "native18_upstream_released_decoder_v1",
            "profile_id": PROFILE_ID,
            "comparison": "posterior_strictly_greater_than_threshold",
            "threshold": float(threshold),
            "operation_order": [
                "binary_opening",
                "binary_closing",
                "remove_runs_strictly_shorter_than_minimum",
            ],
            "morphology_kernel_samples": MORPHOLOGY_KERNEL_SAMPLES,
            "minimum_event_seconds": MINIMUM_EVENT_SECONDS,
            "minimum_event_samples": MINIMUM_EVENT_SAMPLES,
            "sampling_rate_hz": TARGET_FS_HZ,
            "event_sample_spans_half_open": [list(span) for span in retained],
            "binary_payload_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return Native18DecodedEvents(
        binary_mask=output,
        event_sample_spans=retained,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
    )


def infer_upstream_native18_full_record(
    model: nn.Module,
    record: Native18Record,
    *,
    device: str = "cpu",
    batch_size: int = 1,
    threshold: float = RELEASED_THRESHOLD,
) -> Native18InferenceResult:
    """Run non-overlap upstream evaluation tiles, trim padding, then decode."""

    if not isinstance(model, nn.Module):
        raise TypeError("inference requires a Torch model")
    if not isinstance(record, Native18Record):
        raise TypeError("inference requires a Native18Record")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    torch_device = torch.device(device)
    tiles = plan_upstream_inference_tiles(record.signal.shape[1])
    pieces: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_start in range(0, len(tiles), batch_size):
                batch_tiles = tiles[batch_start : batch_start + batch_size]
                model_inputs: list[np.ndarray] = []
                for tile in batch_tiles:
                    observed = record.signal[
                        :, tile.start_sample : tile.start_sample + tile.observed_sample_count
                    ]
                    if tile.right_padding_sample_count:
                        observed = np.pad(
                            observed,
                            ((0, 0), (0, tile.right_padding_sample_count)),
                            mode="constant",
                        )
                    model_inputs.append(filter_upstream_native18_tile(observed))
                tensor = torch.from_numpy(np.stack(model_inputs)).to(
                    device=torch_device, dtype=torch.float32
                )
                prediction = model(tensor)
                if not isinstance(prediction, Tensor):
                    raise TypeError("native18 model must return a Torch tensor")
                if prediction.shape != (len(batch_tiles), TILE_SAMPLES):
                    raise ValueError("native18 model output must be [batch,15360]")
                array = prediction.detach().float().cpu().numpy()
                if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
                    raise Native18NumericalFailure(
                        "native18 model posterior is non-finite or outside [0,1]"
                    )
                for row, tile in zip(array, batch_tiles):
                    pieces.append(row[: tile.observed_sample_count])
    finally:
        model.train(was_training)
    posterior = np.ascontiguousarray(np.concatenate(pieces), dtype="<f4")
    if posterior.shape != (record.signal.shape[1],):
        raise AssertionError("native18 inference did not cover the complete record")
    posterior.setflags(write=False)
    decoded = decode_upstream_native18_posterior(posterior, threshold=threshold)
    receipt = _content_address(
        {
            "schema_version": "native18_upstream_full_record_inference_v1",
            "profile_id": PROFILE_ID,
            "record_transform_receipt_sha256": record.receipt["receipt_sha256"],
            "sampling_rate_hz": TARGET_FS_HZ,
            "sample_count": int(posterior.size),
            "tile_seconds": TILE_SECONDS,
            "tile_hop_seconds": TILE_SECONDS,
            "tile_overlap_fraction": 0.0,
            "tile_count": len(tiles),
            "tile_ledger": [
                {
                    "tile_index": tile.tile_index,
                    "start_sample": tile.start_sample,
                    "observed_sample_count": tile.observed_sample_count,
                    "right_padding_sample_count": tile.right_padding_sample_count,
                    "tile_filter_state_reset": True,
                }
                for tile in tiles
            ],
            "posterior_payload": _float32_payload_receipt(
                posterior, semantic="native18_dense_sample_posterior"
            ),
            "decoder_receipt_sha256": decoded.receipt["receipt_sha256"],
            "annotation_spreadsheet_doctor_text_opened": False,
            "source_dev_or_source_eval_opened_by_inference_primitive": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return Native18InferenceResult(
        posterior=posterior,
        decoded=decoded,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
    )


def load_external_native19_model(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load only the pinned research-only 19-axis safetensors artifact.

    Loading establishes tensor/architecture compatibility, not author
    provenance, checkpoint equivalence, training exposure, or clinical use.
    """

    source = Path(checkpoint_path).resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("external native19 checkpoint must be a regular file")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    checkpoint_sha256 = digest.hexdigest()
    if checkpoint_sha256 != EXTERNAL_CHECKPOINT_SHA256:
        raise ValueError("external native19 checkpoint bytes drifted")
    from safetensors.torch import load_file
    from third_party.SeizureTransformer.time_step_level.model import (
        SeizureTransformer,
    )

    torch_device = torch.device(device)
    state = load_file(str(source), device="cpu")
    model = SeizureTransformer(
        in_channels=19,
        in_samples=TILE_SAMPLES,
        dim_feedforward=2048,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
    )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("external native19 state_dict is not architecture-exact")
    model.to(torch_device).eval()
    receipt = _content_address(
        {
            "schema_version": "external_seizuretransformer_native19_model_load_v1",
            "profile_id": EXTERNAL_NATIVE19_PROFILE_ID,
            "checkpoint_sha256": checkpoint_sha256,
            "tensor_count": len(state),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "in_channels": 19,
            "in_samples": TILE_SAMPLES,
            "device": str(torch_device),
            "architecture_exact_load": True,
            "uploader_is_upstream_author_verified": False,
            "original_checkpoint_hash_verified": False,
            "conversion_log_verified": False,
            "training_exposure_documented": False,
            "research_only_noncommercial_no_clinical_use": True,
            "paper_native18_reproduction": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return model, receipt


def infer_external_upstream_native19_full_record(
    model: nn.Module,
    record: ExternalNative19Record,
    *,
    device: str = "cpu",
    batch_size: int = 1,
    threshold: float = RELEASED_THRESHOLD,
) -> Native18InferenceResult:
    """Diagnostic full-record native19 inference for the external artifact."""

    if not isinstance(record, ExternalNative19Record):
        raise TypeError("external native19 inference requires its receipted carrier")
    if not isinstance(model, nn.Module):
        raise TypeError("external native19 inference requires a Torch model")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    torch_device = torch.device(device)
    tiles = plan_upstream_inference_tiles(record.signal.shape[1])
    pieces: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_start in range(0, len(tiles), batch_size):
                rows = tiles[batch_start : batch_start + batch_size]
                inputs: list[np.ndarray] = []
                for tile in rows:
                    observed = record.signal[
                        :, tile.start_sample : tile.start_sample + tile.observed_sample_count
                    ]
                    if tile.right_padding_sample_count:
                        observed = np.pad(
                            observed,
                            ((0, 0), (0, tile.right_padding_sample_count)),
                            mode="constant",
                        )
                    inputs.append(filter_external_upstream_native19_tile(observed))
                prediction = model(
                    torch.from_numpy(np.stack(inputs)).to(
                        device=torch_device, dtype=torch.float32
                    )
                )
                if not isinstance(prediction, Tensor) or prediction.shape != (
                    len(rows),
                    TILE_SAMPLES,
                ):
                    raise ValueError("external native19 output must be [batch,15360]")
                array = prediction.detach().float().cpu().numpy()
                if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
                    raise Native18NumericalFailure("external native19 posterior is malformed")
                pieces.extend(
                    row[: tile.observed_sample_count]
                    for row, tile in zip(array, rows)
                )
    finally:
        model.train(was_training)
    posterior = np.ascontiguousarray(np.concatenate(pieces), dtype="<f4")
    if posterior.shape != (record.signal.shape[1],):
        raise AssertionError("external native19 inference left incomplete coverage")
    posterior.setflags(write=False)
    decoded = decode_upstream_native18_posterior(posterior, threshold=threshold)
    receipt = _content_address(
        {
            "schema_version": "external_seizuretransformer_native19_inference_v1",
            "profile_id": EXTERNAL_NATIVE19_PROFILE_ID,
            "record_transform_receipt_sha256": record.receipt["receipt_sha256"],
            "expected_checkpoint_sha256": EXTERNAL_CHECKPOINT_SHA256,
            "sample_count": int(posterior.size),
            "tile_count": len(tiles),
            "tile_hop_samples": TEST_HOP_SAMPLES,
            "tail_zero_padded_before_filter": True,
            "tile_filter_state_reset": True,
            "posterior_payload": _float32_payload_receipt(
                posterior, semantic="external_native19_dense_sample_posterior"
            ),
            "decoder_receipt_sha256": decoded.receipt["receipt_sha256"],
            "paper_native18_reproduction": False,
            "external_artifact_provenance_unknown": True,
            "diagnostic_question": (
                "whether_17axis_first_convolution_projection_is_a_major_transfer_loss_source"
            ),
            "accuracy_or_equivalence_claim": False,
            "source_dev_or_source_eval_opened_by_inference_primitive": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return Native18InferenceResult(
        posterior=posterior,
        decoded=decoded,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
    )


def native18_training_contract() -> dict[str, Any]:
    """Freeze upstream and TUSZ-only clean-room training identities separately."""

    return _content_address(
        {
            "schema_version": "native18_upstream_and_tusz_only_training_contract_v1",
            "upstream_public_code_profile": {
                "profile_id": "seizuretransformer_public_code_siena_plus_tusz_v1",
                "training_exposure": ["Siena_train_materializer", "TUSZ_v2.0.3_train"],
                "window_seconds": 60,
                "training_hop_seconds": 15,
                "category_sampling": {
                    "partial_seizure": "all",
                    "full_seizure_max_ratio_to_partial": 0.7,
                    "no_seizure_max_ratio_to_partial": 2.0,
                },
                "batch_size": 86,
                "epochs": 100,
                "loss": "torch.nn.functional.binary_cross_entropy_unweighted",
                "optimizer": "torch.optim.RAdam",
                "learning_rate": 1e-4,
                "weight_decay": 2e-5,
                "checkpoint_selection": (
                    "after_every_epoch_official_TUSZ_dev_event_F1_strictly_greater"
                ),
                "checkpoint_selection_threshold": 0.8,
                "checkpoint_selection_decoder": (
                    "5sample_open_5sample_close_remove_shorter_than_2s"
                ),
                "official_dev_used_during_training": True,
                "paper_checkpoint_exact_exposure_or_bytes_available": False,
            },
            "tusz_only_cleanroom_profile": {
                "profile_id": TUSZ_ONLY_PROFILE_ID,
                "training_exposure": ["TUSZ_v2.0.3_source_train_only"],
                "siena_used": False,
                "same_native18_preprocessing_and_model_hyperparameters": True,
                "same_batch_size": 86,
                "same_epoch_ceiling": 100,
                "same_loss_optimizer_learning_rate_weight_decay": True,
                "selection_phase": (
                    "patient_disjoint_source_train_inner_validation_only"
                ),
                "official_source_dev_role": (
                    "post_checkpoint_policy_and_operating_point_calibration_only"
                ),
                "official_source_eval_role": "untouched_one_shot_only",
                "may_be_named_paper_checkpoint_reproduction": False,
                "may_be_named_upstream_architecture_native_preprocessing_reproduction": True,
            },
            "known_upstream_code_disclosures": [
                "get_dataset_default_beta_is_1_0_but_train_sd_default_beta_is_2_0",
                "training_window_arange_excludes_exact_terminal_full_window",
                "sklearn_shuffle_has_no_explicit_random_state",
                "missing_referential_electrodes_are_replaced_by_zeros",
                "tile_local_causal_IIR_state_is_reset_for_every_train_and_test_tile",
                "test_tail_is_zero_padded_before_tile_local_filtering",
            ],
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def audit_native18_runtime(project_root: str | Path) -> dict[str, Any]:
    """Hash-check vendored sources and epilepsy2bids axis constants."""

    root = Path(project_root).resolve()
    paths = {
        "handle_data": (
            "third_party/SeizureTransformer/time_step_level/service/handle_data.py",
            UPSTREAM_HANDLE_DATA_SHA256,
        ),
        "result": (
            "third_party/SeizureTransformer/time_step_level/service/result.py",
            UPSTREAM_RESULT_SHA256,
        ),
        "post_process": (
            "third_party/SeizureTransformer/time_step_level/service/post_process.py",
            UPSTREAM_POST_PROCESS_SHA256,
        ),
        "train_sd": (
            "third_party/SeizureTransformer/time_step_level/train_sd.py",
            UPSTREAM_TRAIN_SHA256,
        ),
        "get_dataset": (
            "third_party/SeizureTransformer/time_step_level/get_dataset.py",
            UPSTREAM_DATASET_SHA256,
        ),
    }
    source_rows: list[dict[str, Any]] = []
    for semantic, (relative, expected) in paths.items():
        path = root / relative
        observed = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        source_rows.append(
            {
                "semantic": semantic,
                "relative_path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matches": observed == expected,
            }
        )
    from epilepsy2bids.eeg import Eeg

    observed_referential = tuple(name.upper() for name in Eeg.ELECTRODES_10_20)
    observed_bipolar = tuple(name.upper() for name in Eeg.BIPOLAR_DBANANA)
    coefficient_rows = {
        name: hashlib.sha256(
            np.ascontiguousarray(value, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        for name, value in upstream_tile_filter_coefficients().items()
    }
    receipt = _content_address(
        {
            "schema_version": "native18_upstream_runtime_audit_v1",
            "profile_id": PROFILE_ID,
            "upstream_repository_commit": UPSTREAM_COMMIT,
            "source_rows": source_rows,
            "source_hashes_all_match": all(row["matches"] for row in source_rows),
            "epilepsy2bids_version": importlib.metadata.version("epilepsy2bids"),
            "epilepsy2bids_referential19": list(observed_referential),
            "epilepsy2bids_bipolar18": list(observed_bipolar),
            "referential_order_matches": observed_referential == UPSTREAM_REFERENTIAL_19,
            "bipolar_order_matches": observed_bipolar == UPSTREAM_BIPOLAR_DBANANA_18,
            "historical_alias_binding": {
                "T3": "T7",
                "T4": "T8",
                "T5": "P7",
                "T6": "P8",
            },
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "torch_version": torch.__version__,
            "upstream_requirements_versions": {
                "numpy": "1.26.4",
                "scipy": "1.15.1",
                "torch": "2.0.1",
                "epilepsy2bids": "0.0.7 editable source",
            },
            "filter_coefficient_payload_sha256": coefficient_rows,
            "runtime_exact_version_match": (
                np.__version__ == "1.26.4"
                and scipy.__version__ == "1.15.1"
                and torch.__version__.split("+")[0] == "2.0.1"
                and importlib.metadata.version("epilepsy2bids") == "0.0.7"
            ),
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return receipt


__all__ = [
    "CANONICAL_BIPOLAR_18",
    "CANONICAL_REFERENTIAL_19",
    "EXTERNAL_CHECKPOINT_SHA256",
    "EXTERNAL_NATIVE19_PROFILE_ID",
    "ExternalNative19Record",
    "MINIMUM_EVENT_SAMPLES",
    "MORPHOLOGY_KERNEL_SAMPLES",
    "Native18DecodedEvents",
    "Native18InferenceResult",
    "Native18InferenceTile",
    "Native18NumericalFailure",
    "Native18Record",
    "Native18ReferentialCarrier",
    "PROFILE_ID",
    "RELEASED_THRESHOLD",
    "TARGET_FS_HZ",
    "TEST_HOP_SAMPLES",
    "TILE_SAMPLES",
    "TRAIN_HOP_SAMPLES",
    "TUSZ_ONLY_PROFILE_ID",
    "UPSTREAM_BIPOLAR_DBANANA_18",
    "UPSTREAM_REFERENTIAL_19",
    "audit_native18_runtime",
    "build_upstream_native18_model",
    "build_upstream_native18_optimizer",
    "decode_upstream_native18_posterior",
    "filter_external_upstream_native19_tile",
    "filter_upstream_native18_tile",
    "infer_external_upstream_native19_full_record",
    "infer_upstream_native18_full_record",
    "load_external_native19_model",
    "materialize_physically_strict_referential19",
    "materialize_upstream_literal_referential19",
    "materialize_upstream_training_tile",
    "native18_training_contract",
    "plan_upstream_inference_tiles",
    "select_upstream_training_window_indices",
    "train_upstream_native18_batches",
    "transform_external_upstream_native19_record",
    "transform_upstream_native18_record",
    "upstream_tile_filter_coefficients",
    "upstream_training_tile_starts",
    "upstream_training_window_category",
]
