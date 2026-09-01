"""Safe research adapter for the released EventNet 2024 checkpoint.

The upstream SzCORE-compatible release contains a tensor-only ``state_dict``
at commit ``d13866820f436b1d767ef7f27a5419a7735efa5b``.  Its runner, however,
does not provide a replayable full-record receipt: it omits the first second,
can leave the final tail unmodelled, shifts the returned mask to time zero,
and its nominal 120-second tiling is shape-safe only on a 256 Hz grid.  This
module keeps the released network, channel order, 0.44 center threshold,
Gaussian smoothing and center/duration decoder, while making those hidden
choices explicit and covering every target sample.

Scientific boundary
-------------------
This is a *research-only boundary-lane reproduction*.  The exact patient and
record exposure of the released checkpoint is not documented by the release,
and the local 256 Hz resampling/edge-padding compatibility policy has not yet
been calibrated on a patient-disjoint source-development roster.  Predictions
are detector navigation candidates, not confirmed seizures, clinical onsets,
Findings, or SOZ evidence.

Security and data boundary
--------------------------
The checkpoint is accepted only at one pinned SHA-256, statically inspected,
copied into an immutable byte snapshot and loaded with
``torch.load(..., weights_only=True)``.  The adapter consumes only canonical
physical EEG samples, channel/time metadata and transform receipts.  It has no
annotation, spreadsheet, physician-label, clinical-text or identity input.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample_poly
import torch
from torch import nn

from src.soz.geometry import STANDARD_19

from .canonical_detector_input_binding import (
    build_canonical_detector_input_binding,
    validate_canonical_detector_input_binding,
)
from .canonical_edf_materialization import CanonicalEEGRecord
from .canonical_signal_views import validate_canonical_signal_receipt
from .detector_provider_contract import (
    audit_checkpoint_container,
    materialize_full_record_provider_result,
    validate_full_record_provider_result,
)


EVENTNET_PROVIDER_ID = "eventnet_event_boundary_shadow_v1"
EVENTNET_UPSTREAM_COMMIT = "d13866820f436b1d767ef7f27a5419a7735efa5b"
EVENTNET_UPSTREAM_CHECKPOINT_BLOB = "812ccc089cd1e1d0a5be52cd7315396a93f726e5"
EVENTNET_CHECKPOINT_SHA256 = (
    "f1c1d9409a13ba9e12916036206813ebedf164fea7221083c2bac214e476de34"
)
EVENTNET_CHECKPOINT_SIZE_BYTES = 6_660_832
EVENTNET_WEIGHT_MANIFEST_SHA256 = (
    "c8c46e2f8db65d94c23a16f68eff49f7deec8e2cf852738a1358a5b7ffc69dee"
)
EVENTNET_CHECKPOINT_SCHEMA_VERSION = "eventnet_verified_checkpoint_v1"
EVENTNET_INPUT_SCHEMA_VERSION = "eventnet_canonical_physical_carrier_v1"
EVENTNET_PREPROCESSING_SCHEMA_VERSION = "eventnet_full_record_preprocessing_v1"
EVENTNET_DECODER_SCHEMA_VERSION = "eventnet_direct_event_decoder_v1"
EVENTNET_PREDICTION_SCHEMA_VERSION = "eventnet_full_record_prediction_v1"
EVENTNET_ADAPTER_METHOD_ID = (
    "eventnet_release_d138668_full_record_256hz_tile120_safe_edges_v1"
)

EVENTNET_SAMPLING_RATE_HZ = 256
EVENTNET_TARGET_TILE_SECONDS = 120
EVENTNET_TARGET_TILE_SAMPLES = EVENTNET_SAMPLING_RATE_HZ * EVENTNET_TARGET_TILE_SECONDS
EVENTNET_CONTEXT_SAMPLES_PER_SIDE = 128
EVENTNET_MODEL_INPUT_SAMPLES = (
    EVENTNET_TARGET_TILE_SAMPLES + 2 * EVENTNET_CONTEXT_SAMPLES_PER_SIDE
)
EVENTNET_CENTER_THRESHOLD = 0.44
EVENTNET_CENTER_SMOOTHING_SIGMA_SAMPLES = 100
EVENTNET_MINIMUM_PEAK_DISTANCE_SECONDS = 60
EVENTNET_MAXIMUM_DURATION_SECONDS = 300

# Exact order used by epilepsy2bids.Eeg.ELECTRODES_10_20 in the released
# runner.  Legacy T3/T4/T5/T6 names are represented canonically as T7/T8/P7/P8.
EVENTNET_CHANNEL_ORDER = (
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

_EEG_ONLY_SCOPE = {
    "eeg_samples_used": True,
    "edf_signal_header_used": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "identity_fields_used": False,
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _tensor_payload_receipt(value: np.ndarray, *, semantic: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype="<f4")
    return {
        "semantic": semantic,
        "dtype": "float32_little_endian",
        "shape": list(array.shape),
        "payload_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "minimum": float(array.min(initial=np.inf)),
        "maximum": float(array.max(initial=-np.inf)),
    }


def eventnet_adapter_code_sha256() -> str:
    """Return the exact adapter source hash bound by research execution."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def eventnet_research_provider_definition() -> dict[str, Any]:
    """Return the executable, deliberately unqualified research definition."""

    return {
        "provider_id": EVENTNET_PROVIDER_ID,
        "model_family": "EventNet_direct_event_center_duration_detector",
        "research_role": "shadow_continuous_comparator",
        "implementation_status": "runnable_research",
        "qualification_status": "benchmark_pending",
        "weights_manifest_sha256": EVENTNET_WEIGHT_MANIFEST_SHA256,
        "adapter_code_sha256": eventnet_adapter_code_sha256(),
        "checkpoint_loader_policy": ("hash_allowlist_then_torch_weights_only_true"),
        "training_corpus": (
            "released_EventNet_2024_checkpoint_exact_patient_exposure_unverified"
        ),
        "posterior_calibration_status": (
            "released_center_threshold_0_44_not_locally_source_dev_calibrated"
        ),
        "continuous_operating_point_status": (
            "same_protocol_source_dev_benchmark_pending"
        ),
        "eeg_signal_only": True,
        "edf_annotations_allowed": False,
        "excel_or_clinical_labels_allowed": False,
        "claimed_sota": False,
    }


class EventNetUNet(nn.Module):
    """State-key-compatible copy of the MIT-licensed released architecture."""

    def __init__(self, input_channels: int = 19) -> None:
        super().__init__()
        channels = 16
        stride = 4
        padding = "same"
        kernel_size = 9
        bias = False

        self.kernel_size = kernel_size
        self.up = nn.Upsample(scale_factor=stride)
        self.sigm = nn.Sigmoid()
        self.enc0 = nn.Sequential(
            nn.Conv1d(
                input_channels,
                channels,
                kernel_size,
                stride=1,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
        )
        self.down = nn.MaxPool1d(kernel_size=stride)
        self.enc1 = nn.Sequential(
            nn.Conv1d(
                channels,
                2 * channels,
                kernel_size,
                bias=bias,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(2 * channels),
            nn.ELU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(
                2 * channels,
                4 * channels,
                kernel_size,
                bias=bias,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(4 * channels),
            nn.ELU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(
                12 * channels,
                4 * channels,
                15,
                bias=bias,
                padding=padding,
            ),
            nn.BatchNorm1d(4 * channels),
            nn.ELU(),
            nn.Conv1d(
                4 * channels,
                4 * channels,
                kernel_size,
                bias=bias,
                padding=padding,
            ),
            nn.BatchNorm1d(4 * channels),
            nn.ELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(
                4 * channels,
                8 * channels,
                kernel_size,
                bias=bias,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(8 * channels),
            nn.ELU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv1d(
                24 * channels,
                8 * channels,
                15,
                bias=bias,
                padding=padding,
            ),
            nn.BatchNorm1d(8 * channels),
            nn.ELU(),
            nn.Conv1d(
                8 * channels,
                8 * channels,
                15,
                bias=bias,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(8 * channels),
            nn.ELU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(
                8 * channels,
                16 * channels,
                kernel_size,
                bias=bias,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(16 * channels),
            nn.ELU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(
                6 * channels,
                2 * channels,
                15,
                bias=bias,
                padding=padding,
            ),
            nn.BatchNorm1d(2 * channels),
            nn.ELU(inplace=True),
        )
        self.dec0 = nn.Sequential(
            nn.Conv1d(
                3 * channels,
                channels,
                15,
                bias=bias,
                padding=padding,
            ),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
        )
        self.center_logit = nn.Sequential(
            nn.Conv1d(channels, 1, 21, stride=1, padding=padding),
            nn.MaxPool1d(kernel_size=21, stride=1),
        )
        self.duration_logit = nn.Sequential(
            nn.Conv1d(channels, 1, 21, stride=1, padding=padding),
            nn.MaxPool1d(kernel_size=21, stride=1),
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lvl0 = self.enc0(value)
        lvl1 = self.enc1(self.down(lvl0))
        lvl2 = self.enc2(self.down(lvl1))
        lvl3 = self.enc3(self.down(lvl2))
        lvl4 = self.enc4(self.down(lvl3))

        out3 = self.dec3(torch.cat((self.up(lvl4), lvl3), dim=1))
        out2 = self.dec2(torch.cat((self.up(out3), lvl2), dim=1))
        out1 = self.dec1(torch.cat((self.up(out2), lvl1), dim=1))
        out0 = self.dec0(torch.cat((self.up(out1), lvl0), dim=1))

        crop = 256 - 21 + 1
        center = self.sigm(self.center_logit(out0))[:, :, crop // 2 : -crop // 2]
        duration = self.sigm(self.duration_logit(out0))[:, :, crop // 2 : -crop // 2]
        return center, duration


def _validate_checkpoint_receipt(payload: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "receipt_id",
        "upstream_commit",
        "upstream_checkpoint_blob",
        "artifact_id",
        "size_bytes",
        "sha256",
        "weight_manifest_sha256",
        "static_audit",
        "loader_policy",
        "state_tensor_count",
        "parameter_count",
        "exact_state_keys_and_shapes_verified",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("EventNet checkpoint receipt fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != EVENTNET_CHECKPOINT_SCHEMA_VERSION
        or data["upstream_commit"] != EVENTNET_UPSTREAM_COMMIT
        or data["upstream_checkpoint_blob"] != EVENTNET_UPSTREAM_CHECKPOINT_BLOB
        or data["artifact_id"] != "model.pth"
        or data["size_bytes"] != EVENTNET_CHECKPOINT_SIZE_BYTES
        or data["sha256"] != EVENTNET_CHECKPOINT_SHA256
        or data["weight_manifest_sha256"] != EVENTNET_WEIGHT_MANIFEST_SHA256
        or data["loader_policy"]
        != "exact_hash_then_immutable_snapshot_torch_weights_only_true"
        or data["exact_state_keys_and_shapes_verified"] is not True
    ):
        raise ValueError("EventNet checkpoint identity or loader policy drifted")
    if (
        not isinstance(data["state_tensor_count"], int)
        or data["state_tensor_count"] <= 0
    ):
        raise ValueError("EventNet checkpoint tensor count is invalid")
    if not isinstance(data["parameter_count"], int) or data["parameter_count"] <= 0:
        raise ValueError("EventNet parameter count is invalid")
    audit = data["static_audit"]
    if (
        type(audit) is not dict
        or audit.get("sha256") != EVENTNET_CHECKPOINT_SHA256
        or audit.get("direct_python_pickle_load_allowed") is not False
        or audit.get("static_audit_executes_pickle") is not False
    ):
        raise ValueError("EventNet static checkpoint audit is invalid")
    digest = deepcopy(data)
    digest["receipt_id"] = "EVENTNET-CHECKPOINT-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "EVNCKPT-" + _canonical_sha256(digest)[:24]
    if data["receipt_id"] != expected_id:
        raise ValueError("EventNet checkpoint receipt ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet checkpoint receipt hash is invalid")
    return data


def load_verified_eventnet_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[EventNetUNet, dict[str, Any]]:
    """Hash-gate and safely load the exact released tensor state."""

    path = Path(checkpoint_path)
    if path.is_symlink():
        raise ValueError("EventNet checkpoint must not be a symbolic link")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("EventNet checkpoint must be a regular file")
    blob = resolved.read_bytes()
    if len(blob) != EVENTNET_CHECKPOINT_SIZE_BYTES:
        raise ValueError("EventNet checkpoint size does not match the allowlist")
    if hashlib.sha256(blob).hexdigest() != EVENTNET_CHECKPOINT_SHA256:
        raise ValueError("EventNet checkpoint SHA-256 does not match the allowlist")
    audit = audit_checkpoint_container(resolved)
    if audit["sha256"] != EVENTNET_CHECKPOINT_SHA256:
        raise ValueError("EventNet checkpoint changed during static audit")

    try:
        state = torch.load(
            io.BytesIO(blob),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:  # pragma: no cover - exact error is torch-version specific
        raise ValueError("EventNet tensor-only checkpoint load failed") from exc
    if not isinstance(state, Mapping) or not state:
        raise ValueError("EventNet checkpoint is not a non-empty tensor state_dict")
    if any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("EventNet checkpoint contains a non-tensor state entry")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("EventNet checkpoint contains non-finite tensor values")

    model = EventNetUNet(input_channels=len(EVENTNET_CHANNEL_ORDER))
    expected = model.state_dict()
    if set(state) != set(expected) or any(
        tuple(state[name].shape) != tuple(expected[name].shape) for name in expected
    ):
        raise ValueError("EventNet checkpoint state keys or shapes drifted")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    body: dict[str, Any] = {
        "schema_version": EVENTNET_CHECKPOINT_SCHEMA_VERSION,
        "receipt_id": "EVENTNET-CHECKPOINT-PENDING",
        "upstream_commit": EVENTNET_UPSTREAM_COMMIT,
        "upstream_checkpoint_blob": EVENTNET_UPSTREAM_CHECKPOINT_BLOB,
        "artifact_id": "model.pth",
        "size_bytes": len(blob),
        "sha256": EVENTNET_CHECKPOINT_SHA256,
        "weight_manifest_sha256": EVENTNET_WEIGHT_MANIFEST_SHA256,
        "static_audit": audit,
        "loader_policy": ("exact_hash_then_immutable_snapshot_torch_weights_only_true"),
        "state_tensor_count": len(state),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "exact_state_keys_and_shapes_verified": True,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_id"] = "EVNCKPT-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return model, _validate_checkpoint_receipt(body)


def _canonical_sampling_rate(record: CanonicalEEGRecord) -> float:
    rows = record.source_header_receipt["channel_signal_headers"]
    rates = {
        (int(row["sampling_rate_numerator"]), int(row["sampling_rate_denominator"]))
        for row in rows
    }
    if len(rates) != 1:
        raise ValueError("EventNet requires one canonical physical sampling grid")
    numerator, denominator = next(iter(rates))
    return numerator / denominator


def _materialize_physical_carrier(
    record: CanonicalEEGRecord,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any]]:
    canonical = validate_canonical_signal_receipt(record.canonical_receipt)
    montage = record.montage_reference_observability_receipt
    if montage.get("montage_class") not in {
        "common_compatible_referential",
        "mixed",
    }:
        raise ValueError("EventNet released checkpoint requires referential electrodes")
    sampling_rate = _canonical_sampling_rate(record)
    sample_count = int(record.observed_signal_volts.shape[1])
    if sample_count < 1 or not math.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("EventNet canonical carrier has an invalid clock")
    observed = tuple(record.observed_channel_ids)
    observed_index = {name: index for index, name in enumerate(observed)}
    carrier_uv = np.zeros((len(STANDARD_19), sample_count), dtype=np.float64)
    source = record.observed_signal_volts.detach().cpu().to(torch.float64).numpy()
    for index, channel in enumerate(STANDARD_19):
        if channel in observed_index:
            carrier_uv[index] = source[observed_index[channel]] * 1_000_000.0
    imputed = tuple(channel for channel in STANDARD_19 if channel not in observed_index)
    input_receipt: dict[str, Any] = {
        "schema_version": EVENTNET_INPUT_SCHEMA_VERSION,
        "receipt_id": "EVENTNET-INPUT-PENDING",
        "provider_id": EVENTNET_PROVIDER_ID,
        "canonical_signal_id": canonical["canonical_signal_id"],
        "canonical_source_signal_sha256": canonical["source_signal_sha256"],
        "channel_order_before_provider_reorder": list(STANDARD_19),
        "provider_channel_order": list(EVENTNET_CHANNEL_ORDER),
        "observed_channel_ids": list(observed),
        "imputed_channel_ids": list(imputed),
        "sampling_rate_hz": sampling_rate,
        "sample_count": sample_count,
        "physical_unit": "uV",
        "conversion_from_canonical_volts": 1_000_000.0,
        "missing_channel_policy": "exact_zero_before_provider_transform",
        "imputed_channels_clinical_evidence_eligible": False,
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
    }
    input_receipt["receipt_id"] = "EVNINPUT-" + _canonical_sha256(input_receipt)[:24]
    binding = build_canonical_detector_input_binding(
        canonical_record=record,
        provider_id=EVENTNET_PROVIDER_ID,
        detector_input=carrier_uv,
        detector_channel_ids=STANDARD_19,
        detector_sampling_rate_hz=sampling_rate,
        detector_physical_unit="uV",
        observed_channel_ids=observed,
        imputed_channel_ids=imputed,
        provider_input_receipt_id=input_receipt["receipt_id"],
        provider_input_receipt_sha256=_canonical_sha256(input_receipt),
    )
    return carrier_uv, sampling_rate, input_receipt, binding


def _provider_preprocess(
    carrier_standard19_uv: np.ndarray,
    *,
    source_sampling_rate_hz: float,
    recording_duration_seconds: float,
    input_receipt: Mapping[str, Any],
    canonical_binding: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    source_rate = Fraction(str(source_sampling_rate_hz)).limit_denominator(1_000_000)
    target_rate = Fraction(EVENTNET_SAMPLING_RATE_HZ, 1)
    ratio = target_rate / source_rate
    up, down = ratio.numerator, ratio.denominator
    if (up, down) == (1, 1):
        resampled = np.asarray(carrier_standard19_uv, dtype=np.float64).copy()
        resampler = "identity_exact_256Hz"
        support_policy = "tile_local_observed_support"
    else:
        resampled = resample_poly(
            np.asarray(carrier_standard19_uv, dtype=np.float64),
            up,
            down,
            axis=1,
        )
        resampler = "scipy_signal_resample_poly_default_kaiser5_zero_pad_v1"
        # The released code has no replayable polyphase support ledger.  Until
        # the exact finite support is qualified, use a conservative whole-record
        # support over-approximation rather than understate future access.
        support_policy = "whole_record_conservative_until_resampler_support_audit"
    expected_samples = int(round(recording_duration_seconds * target_rate))
    length_adjustment = expected_samples - int(resampled.shape[1])
    if abs(length_adjustment) > 1:
        raise ValueError("EventNet resampler drifted from the canonical clock")
    if length_adjustment < 0:
        resampled = resampled[:, :expected_samples]
    elif length_adjustment > 0:
        resampled = np.pad(resampled, ((0, 0), (0, length_adjustment)))
    if expected_samples < 1 or not np.isfinite(resampled).all():
        raise ValueError("EventNet preprocessed carrier is empty or non-finite")
    standard_index = {channel: index for index, channel in enumerate(STANDARD_19)}
    provider = np.stack(
        [resampled[standard_index[channel]] for channel in EVENTNET_CHANNEL_ORDER]
    ).astype(np.float32, copy=False)
    binding = validate_canonical_detector_input_binding(dict(canonical_binding))
    receipt: dict[str, Any] = {
        "schema_version": EVENTNET_PREPROCESSING_SCHEMA_VERSION,
        "method_id": EVENTNET_ADAPTER_METHOD_ID,
        "input_receipt_id": input_receipt["receipt_id"],
        "input_receipt_sha256": _canonical_sha256(input_receipt),
        "canonical_binding_id": binding["binding_id"],
        "canonical_binding_receipt_sha256": binding["receipt_sha256"],
        "source_sampling_rate_hz": source_sampling_rate_hz,
        "target_sampling_rate_hz": EVENTNET_SAMPLING_RATE_HZ,
        "resample_up": up,
        "resample_down": down,
        "resampler": resampler,
        "resampling_temporal_support_policy": support_policy,
        "provider_channel_order": list(EVENTNET_CHANNEL_ORDER),
        "physical_unit": "uV",
        "filtering": "none_released_runner",
        "normalization": "none_released_runner",
        "clipping": "none_released_runner",
        "target_tile_seconds": EVENTNET_TARGET_TILE_SECONDS,
        "target_tile_samples": EVENTNET_TARGET_TILE_SAMPLES,
        "context_samples_each_side": EVENTNET_CONTEXT_SAMPLES_PER_SIDE,
        "edge_extension": "explicit_zero_not_observed_eeg",
        "source_sample_count": int(carrier_standard19_uv.shape[1]),
        "provider_sample_count": int(provider.shape[1]),
        "resampler_length_adjustment_samples": length_adjustment,
        "provider_tensor": _tensor_payload_receipt(
            provider, semantic="eventnet_provider_input_uV"
        ),
        "navigation_only": True,
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return np.ascontiguousarray(provider), receipt


def _build_tile_schedule(
    provider_eeg: np.ndarray,
    *,
    recording_duration_seconds: float,
    whole_record_support: bool,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if provider_eeg.ndim != 2 or provider_eeg.shape[0] != len(EVENTNET_CHANNEL_ORDER):
        raise ValueError("EventNet provider tensor must have shape [19,samples]")
    sample_count = int(provider_eeg.shape[1])
    inputs: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    for tile_index, target_start in enumerate(
        range(0, sample_count, EVENTNET_TARGET_TILE_SAMPLES)
    ):
        actual = min(EVENTNET_TARGET_TILE_SAMPLES, sample_count - target_start)
        wanted_start = target_start - EVENTNET_CONTEXT_SAMPLES_PER_SIDE
        wanted_stop = (
            target_start
            + EVENTNET_TARGET_TILE_SAMPLES
            + EVENTNET_CONTEXT_SAMPLES_PER_SIDE
        )
        observed_start = max(0, wanted_start)
        observed_stop = min(sample_count, wanted_stop)
        destination_start = observed_start - wanted_start
        destination_stop = destination_start + observed_stop - observed_start
        model_input = np.zeros(
            (len(EVENTNET_CHANNEL_ORDER), EVENTNET_MODEL_INPUT_SAMPLES),
            dtype=np.float32,
        )
        model_input[:, destination_start:destination_stop] = provider_eeg[
            :, observed_start:observed_stop
        ]
        left_padding = max(0, -wanted_start)
        right_padding = max(0, wanted_stop - sample_count)
        target_stop = target_start + actual
        if whole_record_support:
            support_start_seconds = 0.0
            support_stop_seconds = recording_duration_seconds
        else:
            support_start_seconds = observed_start / EVENTNET_SAMPLING_RATE_HZ
            support_stop_seconds = min(
                recording_duration_seconds,
                observed_stop / EVENTNET_SAMPLING_RATE_HZ,
            )
        target_stop_seconds = min(
            recording_duration_seconds,
            target_stop / EVENTNET_SAMPLING_RATE_HZ,
        )
        body: dict[str, Any] = {
            "tile_id": f"EVNTILE-{tile_index:06d}",
            "target_start_sample": target_start,
            "target_stop_sample_exclusive": target_stop,
            "target_start_offset_seconds": (target_start / EVENTNET_SAMPLING_RATE_HZ),
            "target_stop_offset_seconds": target_stop_seconds,
            "actual_target_samples": actual,
            "model_target_capacity_samples": EVENTNET_TARGET_TILE_SAMPLES,
            "observed_provider_start_sample": observed_start,
            "observed_provider_stop_sample_exclusive": observed_stop,
            "observed_support_start_offset_seconds": support_start_seconds,
            "observed_support_stop_offset_seconds": support_stop_seconds,
            "decision_available_offset_seconds": support_stop_seconds,
            "future_lookahead_seconds_at_target_stop": max(
                0.0, support_stop_seconds - target_stop_seconds
            ),
            "left_padding_samples": left_padding,
            "right_padding_samples": right_padding,
            "padding_is_observed_eeg": False,
            "model_input_tensor": _tensor_payload_receipt(
                model_input, semantic="eventnet_model_tile_input_uV"
            ),
        }
        inputs.append(model_input)
        receipts.append(body)
    if not inputs:
        raise ValueError("EventNet full-record schedule contains no tile")
    return inputs, receipts


def _predict_tiles(
    model: EventNetUNet,
    model_inputs: Sequence[np.ndarray],
    tile_receipts: Sequence[Mapping[str, Any]],
    *,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    centers: list[np.ndarray] = []
    durations: list[np.ndarray] = []
    output_receipts: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for model_input, raw_receipt in zip(model_inputs, tile_receipts):
            value = torch.from_numpy(np.ascontiguousarray(model_input))[None].to(device)
            output_center, output_duration = model(value)
            if tuple(output_center.shape) != (1, 1, EVENTNET_TARGET_TILE_SAMPLES):
                raise ValueError("EventNet center output shape drifted")
            if tuple(output_duration.shape) != (1, 1, EVENTNET_TARGET_TILE_SAMPLES):
                raise ValueError("EventNet duration output shape drifted")
            actual = int(raw_receipt["actual_target_samples"])
            center = (
                output_center[0, 0, :actual]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
                .copy()
            )
            duration = (
                output_duration[0, 0, :actual]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
                .copy()
            )
            if (
                not np.isfinite(center).all()
                or not np.isfinite(duration).all()
                or np.any(center < 0)
                or np.any(center > 1)
                or np.any(duration < 0)
                or np.any(duration > 1)
            ):
                raise ValueError("EventNet output is not a finite probability pair")
            receipt = deepcopy(dict(raw_receipt))
            receipt["center_output"] = _tensor_payload_receipt(
                center, semantic="event_center_probability"
            )
            receipt["duration_output"] = _tensor_payload_receipt(
                duration, semantic="event_duration_fraction_of_300_seconds"
            )
            centers.append(center)
            durations.append(duration)
            output_receipts.append(receipt)
    return (
        np.ascontiguousarray(np.concatenate(centers).astype(np.float32)),
        np.ascontiguousarray(np.concatenate(durations).astype(np.float32)),
        output_receipts,
    )


def _decode_direct_events(
    center_probability: np.ndarray,
    duration_fraction: np.ndarray,
    tile_receipts: Sequence[Mapping[str, Any]],
    *,
    recording_duration_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        center_probability.shape != duration_fraction.shape
        or center_probability.ndim != 1
    ):
        raise ValueError("EventNet decoder requires paired one-dimensional outputs")
    smoothed = np.empty_like(center_probability, dtype=np.float32)
    raw_events: list[dict[str, Any]] = []
    minimum_distance = (
        EVENTNET_MINIMUM_PEAK_DISTANCE_SECONDS * EVENTNET_SAMPLING_RATE_HZ
    )
    for tile in tile_receipts:
        start = int(tile["target_start_sample"])
        stop = int(tile["target_stop_sample_exclusive"])
        tile_center = center_probability[start:stop]
        tile_smoothed = gaussian_filter1d(
            tile_center,
            EVENTNET_CENTER_SMOOTHING_SIGMA_SAMPLES,
        ).astype(np.float32, copy=False)
        smoothed[start:stop] = tile_smoothed
        peaks, properties = find_peaks(
            tile_smoothed,
            height=EVENTNET_CENTER_THRESHOLD,
            distance=minimum_distance,
        )
        for local_peak, peak_probability in zip(peaks, properties["peak_heights"]):
            global_peak = start + int(local_peak)
            duration_samples = float(duration_fraction[global_peak]) * (
                EVENTNET_MAXIMUM_DURATION_SECONDS * EVENTNET_SAMPLING_RATE_HZ
            )
            event_start = max(0.0, global_peak - 0.5 * duration_samples)
            event_stop = min(
                float(center_probability.shape[0]),
                global_peak + 0.5 * duration_samples,
            )
            if event_stop <= event_start:
                continue
            raw_events.append(
                {
                    "raw_event_id": f"EVNRAW-{len(raw_events):06d}",
                    "tile_id": tile["tile_id"],
                    "center_offset_seconds": (global_peak / EVENTNET_SAMPLING_RATE_HZ),
                    "start_offset_seconds": (event_start / EVENTNET_SAMPLING_RATE_HZ),
                    "stop_offset_seconds": min(
                        recording_duration_seconds,
                        event_stop / EVENTNET_SAMPLING_RATE_HZ,
                    ),
                    "smoothed_center_probability": float(peak_probability),
                    "duration_fraction": float(duration_fraction[global_peak]),
                    "predicted_duration_seconds_before_record_clipping": (
                        duration_samples / EVENTNET_SAMPLING_RATE_HZ
                    ),
                }
            )

    merged: list[dict[str, Any]] = []
    for event in sorted(
        raw_events,
        key=lambda row: (row["start_offset_seconds"], row["stop_offset_seconds"]),
    ):
        if (
            not merged
            or event["start_offset_seconds"] > merged[-1]["stop_offset_seconds"]
        ):
            merged.append(
                {
                    "event_id": f"EVNALARM-{len(merged):06d}",
                    "start_offset_seconds": event["start_offset_seconds"],
                    "stop_offset_seconds": event["stop_offset_seconds"],
                    "maximum_center_probability": event["smoothed_center_probability"],
                    "contributing_raw_event_ids": [event["raw_event_id"]],
                }
            )
        else:
            merged[-1]["stop_offset_seconds"] = max(
                merged[-1]["stop_offset_seconds"],
                event["stop_offset_seconds"],
            )
            merged[-1]["maximum_center_probability"] = max(
                merged[-1]["maximum_center_probability"],
                event["smoothed_center_probability"],
            )
            merged[-1]["contributing_raw_event_ids"].append(event["raw_event_id"])

    policy = {
        "decoder_method": "released_eventnet_center_duration_mask_equivalent_v1",
        "center_smoothing": "scipy_ndimage_gaussian_filter1d",
        "center_smoothing_sigma_samples": EVENTNET_CENTER_SMOOTHING_SIGMA_SAMPLES,
        "center_threshold": EVENTNET_CENTER_THRESHOLD,
        "minimum_peak_distance_seconds_within_tile": (
            EVENTNET_MINIMUM_PEAK_DISTANCE_SECONDS
        ),
        "maximum_duration_seconds": EVENTNET_MAXIMUM_DURATION_SECONDS,
        "cross_tile_policy": "union_overlapping_predicted_intervals",
        "threshold_status": "released_value_not_local_source_dev_operating_point",
        "decoded_start_is_clipped_center_minus_half_duration_not_clinical_onset": True,
    }
    receipt: dict[str, Any] = {
        "schema_version": EVENTNET_DECODER_SCHEMA_VERSION,
        "decoding_receipt_id": "EVENTNET-DECODER-PENDING",
        "decoder_policy": policy,
        "decoder_policy_sha256": _canonical_sha256(policy),
        "raw_event_count": len(raw_events),
        "event_proposal_count": len(merged),
        "raw_events": raw_events,
        "event_proposals": merged,
        "smoothed_center_tensor": _tensor_payload_receipt(
            smoothed, semantic="event_center_probability_gaussian_smoothed"
        ),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    receipt["decoding_receipt_id"] = "EVNDEC-" + _canonical_sha256(receipt)[:24]
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return smoothed, receipt


def _generic_posterior_rows(
    smoothed_center: np.ndarray,
    tile_receipts: Sequence[Mapping[str, Any]],
    *,
    recording_duration_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start_sample in range(0, smoothed_center.shape[0], EVENTNET_SAMPLING_RATE_HZ):
        stop_sample = min(
            smoothed_center.shape[0], start_sample + EVENTNET_SAMPLING_RATE_HZ
        )
        tile_index = min(
            start_sample // EVENTNET_TARGET_TILE_SAMPLES,
            len(tile_receipts) - 1,
        )
        tile = tile_receipts[tile_index]
        target_start = start_sample / EVENTNET_SAMPLING_RATE_HZ
        target_stop = min(
            recording_duration_seconds,
            stop_sample / EVENTNET_SAMPLING_RATE_HZ,
        )
        support_start = float(tile["observed_support_start_offset_seconds"])
        support_stop = float(tile["observed_support_stop_offset_seconds"])
        if target_stop <= target_start:
            continue
        rows.append(
            {
                "window_id": f"EVNPOST-{len(rows):06d}",
                "target_start_offset_seconds": target_start,
                "target_stop_offset_seconds": target_stop,
                "observed_support_start_offset_seconds": support_start,
                "observed_support_stop_offset_seconds": support_stop,
                "decision_available_offset_seconds": support_stop,
                "future_lookahead_seconds": max(0.0, support_stop - target_stop),
                "right_padding_seconds": (
                    float(tile["right_padding_samples"]) / EVENTNET_SAMPLING_RATE_HZ
                ),
                # The generic provider contract has one scalar probability.
                # For EventNet it is explicitly a center-candidate aggregate,
                # not an occupancy posterior; the prediction receipt binds
                # this semantic and the direct-event tensors remain primary.
                "seizure_probability": float(
                    np.max(smoothed_center[start_sample:stop_sample])
                ),
                "signal_usable": True,
                "row_status": "modeled",
            }
        )
    return rows


@dataclass(frozen=True)
class EventNetFullRecordPrediction:
    center_probability: np.ndarray
    duration_fraction: np.ndarray
    smoothed_center_probability: np.ndarray
    receipt: dict[str, Any]


def validate_eventnet_prediction_receipt(payload: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "prediction_id",
        "provider_id",
        "adapter_method_id",
        "recording_id",
        "provider_execution_receipt",
        "checkpoint_receipt",
        "input_receipt",
        "canonical_detector_input_binding",
        "preprocessing_receipt",
        "tile_receipts",
        "output_tensors",
        "decoder_receipt",
        "generic_full_record_result",
        "runtime_receipt",
        "scientific_permissions",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("EventNet prediction receipt fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != EVENTNET_PREDICTION_SCHEMA_VERSION
        or data["provider_id"] != EVENTNET_PROVIDER_ID
        or data["adapter_method_id"] != EVENTNET_ADAPTER_METHOD_ID
        or not isinstance(data["recording_id"], str)
        or not data["recording_id"]
    ):
        raise ValueError("EventNet prediction identity drifted")
    checkpoint = _validate_checkpoint_receipt(data["checkpoint_receipt"])
    binding = validate_canonical_detector_input_binding(
        data["canonical_detector_input_binding"]
    )
    if binding["provider_id"] != EVENTNET_PROVIDER_ID:
        raise ValueError("EventNet canonical binding provider drifted")
    input_receipt = data["input_receipt"]
    if (
        type(input_receipt) is not dict
        or input_receipt.get("schema_version") != EVENTNET_INPUT_SCHEMA_VERSION
        or input_receipt.get("scope_receipt") != _EEG_ONLY_SCOPE
        or binding["provider_input_receipt_id"] != input_receipt.get("receipt_id")
        or binding["provider_input_receipt_sha256"] != _canonical_sha256(input_receipt)
    ):
        raise ValueError("EventNet input receipt and canonical binding disagree")
    preprocessing = data["preprocessing_receipt"]
    if (
        type(preprocessing) is not dict
        or preprocessing.get("schema_version") != EVENTNET_PREPROCESSING_SCHEMA_VERSION
        or preprocessing.get("scope_receipt") != _EEG_ONLY_SCOPE
        or preprocessing.get("canonical_binding_id") != binding["binding_id"]
    ):
        raise ValueError("EventNet preprocessing receipt is invalid")
    preprocessing_digest = deepcopy(preprocessing)
    preprocessing_digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if preprocessing.get("receipt_sha256") != _canonical_sha256(preprocessing_digest):
        raise ValueError("EventNet preprocessing receipt is not content-bound")
    if not isinstance(data["tile_receipts"], list) or not data["tile_receipts"]:
        raise ValueError("EventNet prediction has no tile receipts")
    tensors = data["output_tensors"]
    if type(tensors) is not dict or set(tensors) != {
        "center_probability",
        "duration_fraction",
        "smoothed_center_probability",
    }:
        raise ValueError("EventNet output tensor inventory is invalid")
    if any(
        type(row) is not dict or not _is_sha256(row.get("payload_sha256"))
        for row in tensors.values()
    ):
        raise ValueError("EventNet output tensor hash is invalid")
    decoder = data["decoder_receipt"]
    if (
        type(decoder) is not dict
        or decoder.get("schema_version") != EVENTNET_DECODER_SCHEMA_VERSION
        or not _is_sha256(decoder.get("decoder_policy_sha256"))
        or not _is_sha256(decoder.get("receipt_sha256"))
    ):
        raise ValueError("EventNet decoder receipt is invalid")
    decoder_digest = deepcopy(decoder)
    decoder_digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if decoder["receipt_sha256"] != _canonical_sha256(decoder_digest):
        raise ValueError("EventNet decoder receipt is not content-bound")
    full = validate_full_record_provider_result(data["generic_full_record_result"])
    if (
        full["provider_id"] != EVENTNET_PROVIDER_ID
        or full["recording_id"] != data["recording_id"]
        or full["decoder_outcome"]["event_proposal_count"]
        != decoder["event_proposal_count"]
    ):
        raise ValueError("EventNet generic result and direct decoder disagree")
    execution = data["provider_execution_receipt"]
    if (
        type(execution) is not dict
        or execution.get("provider_id") != EVENTNET_PROVIDER_ID
        or execution.get("checkpoint_sha256") != EVENTNET_WEIGHT_MANIFEST_SHA256
        or execution.get("annotations_used_for_current_recording") is not False
        or execution.get("labels_used_for_current_recording") is not False
        or full["provider_execution_receipt_id"]
        != execution.get("execution_receipt_id")
    ):
        raise ValueError("EventNet execution receipt is invalid")
    if checkpoint["weight_manifest_sha256"] != execution["checkpoint_sha256"]:
        raise ValueError("EventNet execution and checkpoint manifest disagree")
    permissions = data["scientific_permissions"]
    if permissions != {
        "navigation_candidate_only": True,
        "clinical_seizure_or_onset_claim_authorized": False,
        "findings_or_soz_evidence_authorized": False,
        "source_dev_calibration_complete": False,
        "production_or_private_route_authorized": False,
    }:
        raise ValueError("EventNet prediction permissions were widened")
    if data["scope_receipt"] != _EEG_ONLY_SCOPE:
        raise ValueError("EventNet prediction violates the EEG-only scope")
    runtime = data["runtime_receipt"]
    if type(runtime) is not dict or any(
        _finite(value, f"runtime {name}") < 0
        for name, value in runtime.items()
        if name.endswith("_seconds")
    ):
        raise ValueError("EventNet runtime receipt is invalid")
    digest = deepcopy(data)
    digest["prediction_id"] = "EVENTNET-PREDICTION-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "EVNPRED-" + _canonical_sha256(digest)[:24]
    if data["prediction_id"] != expected_id:
        raise ValueError("EventNet prediction ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet prediction receipt hash is invalid")
    return data


def _predict_eventnet_full_record_with_loaded_model(
    *,
    canonical_record: CanonicalEEGRecord,
    recording_id: str,
    model: EventNetUNet,
    checkpoint_receipt: Mapping[str, Any],
    provider_execution_receipt: Mapping[str, Any],
    device: str | torch.device = "cpu",
    checkpoint_seconds_in_record_wall: float = 0.0,
    service_state: str = "warm_same_process",
    session_receipt_sha256: str | None = None,
) -> EventNetFullRecordPrediction:
    """Run one record with an already verified, already loaded model."""

    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError("EventNet recording_id must be a non-empty string")
    if not isinstance(model, EventNetUNet):
        raise TypeError("EventNet loaded-model prediction requires EventNetUNet")
    checkpoint_receipt = _validate_checkpoint_receipt(checkpoint_receipt)
    checkpoint_seconds = _finite(
        checkpoint_seconds_in_record_wall,
        "checkpoint seconds included in record wall",
    )
    if checkpoint_seconds < 0.0:
        raise ValueError("checkpoint seconds included in record wall cannot be negative")
    if service_state not in {
        "cold_single_record",
        "cold_process_start",
        "warm_same_process",
    }:
        raise ValueError("EventNet service state is unsupported")
    if session_receipt_sha256 is not None and not _is_sha256(
        session_receipt_sha256
    ):
        raise ValueError("EventNet session receipt must be a SHA-256")
    started = time.perf_counter()

    stage = time.perf_counter()
    carrier, source_rate, input_receipt, binding = _materialize_physical_carrier(
        canonical_record
    )
    binding_seconds = time.perf_counter() - stage
    duration = float(canonical_record.canonical_receipt["recording_duration_seconds"])

    stage = time.perf_counter()
    provider_eeg, preprocessing = _provider_preprocess(
        carrier,
        source_sampling_rate_hz=source_rate,
        recording_duration_seconds=duration,
        input_receipt=input_receipt,
        canonical_binding=binding,
    )
    whole_record_support = (
        preprocessing["resampling_temporal_support_policy"]
        == "whole_record_conservative_until_resampler_support_audit"
    )
    model_inputs, tile_receipts = _build_tile_schedule(
        provider_eeg,
        recording_duration_seconds=duration,
        whole_record_support=whole_record_support,
    )
    preprocessing_seconds = time.perf_counter() - stage

    stage = time.perf_counter()
    center, duration_fraction, tiles = _predict_tiles(
        model,
        model_inputs,
        tile_receipts,
        device=device,
    )
    inference_seconds = time.perf_counter() - stage
    if center.shape[0] != provider_eeg.shape[1]:
        raise ValueError("EventNet prediction does not cover the provider clock")

    stage = time.perf_counter()
    smoothed, decoder = _decode_direct_events(
        center,
        duration_fraction,
        tiles,
        recording_duration_seconds=duration,
    )
    posterior_rows = _generic_posterior_rows(
        smoothed,
        tiles,
        recording_duration_seconds=duration,
    )
    canonical = validate_canonical_signal_receipt(canonical_record.canonical_receipt)
    execution = deepcopy(dict(provider_execution_receipt))
    generic = materialize_full_record_provider_result(
        provider_id=EVENTNET_PROVIDER_ID,
        provider_execution_receipt_id=execution["execution_receipt_id"],
        recording_id=recording_id,
        source_signal_sha256=canonical["source_signal_sha256"],
        recording_duration_seconds=duration,
        posterior_timeline=posterior_rows,
        decoding_receipt_id=decoder["decoding_receipt_id"],
        decoder_policy_sha256=decoder["decoder_policy_sha256"],
        event_proposal_count=decoder["event_proposal_count"],
        complete_recording_scan_attempted=True,
    )
    decoding_seconds = time.perf_counter() - stage
    processing_seconds = time.perf_counter() - started
    total_seconds = checkpoint_seconds + processing_seconds
    runtime = {
        "checkpoint_static_audit_and_safe_load_seconds": checkpoint_seconds,
        "canonical_carrier_binding_seconds": binding_seconds,
        "provider_preprocessing_and_tiling_seconds": preprocessing_seconds,
        "model_inference_seconds": inference_seconds,
        "direct_event_decoding_seconds": decoding_seconds,
        "full_adapter_wall_seconds": total_seconds,
        "recording_duration_seconds": duration,
        "real_time_factor": total_seconds / duration,
        "device": str(device),
        "tile_count": len(tiles),
        "service_state": service_state,
        "model_session_receipt_sha256": session_receipt_sha256,
        "checkpoint_load_included_in_full_adapter_wall": checkpoint_seconds > 0.0,
    }
    output_tensors = {
        "center_probability": _tensor_payload_receipt(
            center, semantic="event_center_probability"
        ),
        "duration_fraction": _tensor_payload_receipt(
            duration_fraction, semantic="event_duration_fraction_of_300_seconds"
        ),
        "smoothed_center_probability": _tensor_payload_receipt(
            smoothed, semantic="event_center_probability_gaussian_smoothed"
        ),
    }
    body: dict[str, Any] = {
        "schema_version": EVENTNET_PREDICTION_SCHEMA_VERSION,
        "prediction_id": "EVENTNET-PREDICTION-PENDING",
        "provider_id": EVENTNET_PROVIDER_ID,
        "adapter_method_id": EVENTNET_ADAPTER_METHOD_ID,
        "recording_id": recording_id,
        "provider_execution_receipt": execution,
        "checkpoint_receipt": checkpoint_receipt,
        "input_receipt": input_receipt,
        "canonical_detector_input_binding": binding,
        "preprocessing_receipt": preprocessing,
        "tile_receipts": tiles,
        "output_tensors": output_tensors,
        "decoder_receipt": decoder,
        "generic_full_record_result": generic,
        "runtime_receipt": runtime,
        "scientific_permissions": {
            "navigation_candidate_only": True,
            "clinical_seizure_or_onset_claim_authorized": False,
            "findings_or_soz_evidence_authorized": False,
            "source_dev_calibration_complete": False,
            "production_or_private_route_authorized": False,
        },
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["prediction_id"] = "EVNPRED-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    validated = validate_eventnet_prediction_receipt(body)
    return EventNetFullRecordPrediction(
        center_probability=center,
        duration_fraction=duration_fraction,
        smoothed_center_probability=smoothed,
        receipt=validated,
    )


class EventNetFullRecordSession:
    """One verified model load reused across a continuous-record batch.

    Session initialization is kept outside per-record warm end-to-end timing.
    The first Stage-P record may still be labelled ``cold_process_start`` to
    preserve kernel/cache warm-up as an explicit denominator rather than
    silently dropping it.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        provider_execution_receipt: Mapping[str, Any],
        device: str | torch.device = "cpu",
    ) -> None:
        started = time.perf_counter()
        model, checkpoint = load_verified_eventnet_model(
            checkpoint_path, device=device
        )
        initialization_seconds = time.perf_counter() - started
        execution = deepcopy(dict(provider_execution_receipt))
        if (
            execution.get("provider_id") != EVENTNET_PROVIDER_ID
            or execution.get("checkpoint_sha256")
            != EVENTNET_WEIGHT_MANIFEST_SHA256
            or execution.get("annotations_used_for_current_recording") is not False
            or execution.get("labels_used_for_current_recording") is not False
        ):
            raise ValueError("EventNet session execution receipt is invalid")
        body: dict[str, Any] = {
            "schema_version": "eventnet_verified_loaded_model_session_v1",
            "session_id": "EVENTNET-SESSION-PENDING",
            "provider_id": EVENTNET_PROVIDER_ID,
            "provider_execution_receipt_id": execution["execution_receipt_id"],
            "checkpoint_receipt_sha256": _canonical_sha256(checkpoint),
            "adapter_code_sha256": eventnet_adapter_code_sha256(),
            "device": str(device),
            "initialization_seconds": initialization_seconds,
            "initialization_excluded_from_warm_record_wall": True,
            "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
            "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        }
        body["session_id"] = "EVNSESSION-" + _canonical_sha256(body)[:24]
        body["receipt_sha256"] = _canonical_sha256(body)
        self.model = model
        self.checkpoint_receipt = checkpoint
        self.provider_execution_receipt = execution
        self.device = device
        self.session_receipt = body

    def predict(
        self,
        *,
        canonical_record: CanonicalEEGRecord,
        recording_id: str,
        service_state: str = "warm_same_process",
    ) -> EventNetFullRecordPrediction:
        return _predict_eventnet_full_record_with_loaded_model(
            canonical_record=canonical_record,
            recording_id=recording_id,
            model=self.model,
            checkpoint_receipt=self.checkpoint_receipt,
            provider_execution_receipt=self.provider_execution_receipt,
            device=self.device,
            checkpoint_seconds_in_record_wall=0.0,
            service_state=service_state,
            session_receipt_sha256=self.session_receipt["receipt_sha256"],
        )


def predict_eventnet_full_record(
    *,
    canonical_record: CanonicalEEGRecord,
    recording_id: str,
    checkpoint_path: str | Path,
    provider_execution_receipt: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> EventNetFullRecordPrediction:
    """Cold one-record compatibility route.

    Complete benchmarks should prefer :class:`EventNetFullRecordSession` so
    the warm runtime denominator is measured with one model load per process.
    """

    started = time.perf_counter()
    model, checkpoint = load_verified_eventnet_model(
        checkpoint_path, device=device
    )
    checkpoint_seconds = time.perf_counter() - started
    return _predict_eventnet_full_record_with_loaded_model(
        canonical_record=canonical_record,
        recording_id=recording_id,
        model=model,
        checkpoint_receipt=checkpoint,
        provider_execution_receipt=provider_execution_receipt,
        device=device,
        checkpoint_seconds_in_record_wall=checkpoint_seconds,
        service_state="cold_single_record",
        session_receipt_sha256=None,
    )


__all__ = [
    "EVENTNET_ADAPTER_METHOD_ID",
    "EVENTNET_CHANNEL_ORDER",
    "EVENTNET_CHECKPOINT_SHA256",
    "EVENTNET_CHECKPOINT_SIZE_BYTES",
    "EVENTNET_PROVIDER_ID",
    "EVENTNET_SAMPLING_RATE_HZ",
    "EVENTNET_UPSTREAM_COMMIT",
    "EVENTNET_WEIGHT_MANIFEST_SHA256",
    "EventNetFullRecordPrediction",
    "EventNetFullRecordSession",
    "EventNetUNet",
    "eventnet_adapter_code_sha256",
    "eventnet_research_provider_definition",
    "load_verified_eventnet_model",
    "predict_eventnet_full_record",
    "validate_eventnet_prediction_receipt",
]
