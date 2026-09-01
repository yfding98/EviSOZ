"""Research-only DeepSOZ temporal adapter for dense one-second posteriors.

This is a clean-room adapter for the published DeepSOZ tensor schema.  It does
not read EDF annotations, seizure times, spreadsheets, clinical text or SOZ
labels.  Public TUSZ evaluation is allowed only with an external receipt that
the selected published fold held out the current patient.  A synthetic smoke
mode exists solely for adapter verification.

Security boundary
-----------------
The checkpoint is hash-verified, copied to an immutable in-memory snapshot and
loaded only with ``torch.load(..., weights_only=True)``.  There is no fallback
to ``weights_only=False``.  The upstream Python model class is not imported.

Scientific boundary
-------------------
The published branch predicts one-second seizure logits within 600-second
contexts; it was not published as a frozen 30--60 minute continuous detector.
This adapter chunks a complete recording into 600-second contexts with 60
seconds of overlap and fuses overlap posteriors with deterministic edge ramps.
That transfer policy must be calibrated and evaluated at patient level before
any deployment use.  Returned events are posterior samples, not confirmed
seizures or onset times.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal
import torch
from torch import nn

from .detector_provider_contract import audit_checkpoint_container


DEEPSOZ_TEMPORAL_ADAPTER_SCHEMA_VERSION = "deepsoz_dense_temporal_posterior_v1"
DEEPSOZ_OOF_ENSEMBLE_SCHEMA_VERSION = "deepsoz_oof_dense_temporal_posterior_v1"
DEEPSOZ_FOLD_ASSIGNMENT_SCHEMA_VERSION = "deepsoz_official_fold_assignment_v1"
DEEPSOZ_TEMPORAL_ADAPTER_ID = "deepsoz_temporal_chunk600_overlap60_v1"
DEEPSOZ_TARGET_SAMPLING_RATE_HZ = 200.0
DEEPSOZ_CHUNK_SECONDS = 600
DEEPSOZ_OVERLAP_SECONDS = 60
DEEPSOZ_STRIDE_SECONDS = DEEPSOZ_CHUNK_SECONDS - DEEPSOZ_OVERLAP_SECONDS

STANDARD_19 = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
)

PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256 = {
    0: "a2e8e41bb2a63ea545eb989ae9cf77838b477b013dc8135b48d167eb76ccb4d0",
    1: "9a0fa6d5d70c602ddc2a6588fb8742f2bd791dcfb653b5b08f42f6897bfc6017",
    2: "caba3b0c78b7fb04e50c34928c319eb4b26554418640e52867873cab7c796213",
    3: "07b75f394d7b02b592cdc07d8bafa62e9900db8ab5ff422acb2e2d31b0544cfa",
    4: "fbcd9500a7825d5e0111b3f905ac81166d145719a6ee0b30af4cbb9af53f1148",
    5: "f2847e1a7681575bf856dd410da502a544bdce34981d9b438dc72888cb258fcf",
    6: "fb2d048c81c5a94e9a48ed5cc81cc8417e128e609123ea22274f6514d5880acf",
    7: "6e909e74cf51b9b3edc869cbe01bc1c6f1a39610eb08e38ef1ebda8638f3966f",
    8: "66626d654d983daeb306803212b860c1c84ad8d2ac4f3353eeeb0148cdc85080",
    9: "209995c04350f357761238e33511bed3599cf824972ae748e7ef8fdba737478f",
    10: "9cbf25b5722ab7a2c309423246ef0ce507b68aa0b78f274a9784749f916e1042",
    11: "592a6beb4b06a25b819d0c0715a3a6556ce67ba1b5860e9e8ec9a4bed6095761",
    12: "13451e92a8ad8d14e48f62ab9769df8e7132ef17c0b5747d10a5cb9243e70540",
    13: "9a2df5182284fe69ddf37efa92d54cd06ccd9fe29e4d1c9550756ff63623bb4c",
    14: "9411969c8f40f0829b60ca0cca38e203e01c2f1b6d309e8cbbcc5748249f711b",
}
PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256 = (
    "7aaef158669ff8bb5e05e05b8aeefa4a81b7391bd6119b5d9b66f41a3fc918f7"
)
PUBLISHED_DEEPSOZ_TEST_FOLD_NPY_SHA256 = {
    0: "25d7624cd68b52f974a412a4fe9191cba7a0085194881568908493b02e980564",
    1: "d39a8b94921586efe3c022648fbc75b95c78ade9ea9fadf482aeec5369918665",
    2: "85fea4d59ccf1a1b06b7377aa85b15a77b204c5a40400a5039b75c09b8872979",
    3: "ebefeabc1f6090700a25c9cae5b5d4ef1b4ac5454041e789cfa9ef0a09cd5f2f",
    4: "307aad7a043f30f8b490dd1c30bc4fa06ec678ad7c9edb2cfa371b74791798b5",
    5: "1c7015b291f354ecbace6aab85a0de896fc798f67bcaa2459f34f386ba68af14",
    6: "a42b6aa1cc92f1987d53b5e69054727a1ad35a8aeab269b885ff15bae7f2dafd",
    7: "fd8d1f0b2a578a29f79a07de14bcc4bf21bbabcb9afb28f0fc5f470e7f824f8c",
    8: "5171ade47e83b35de34d337c78e1d35b12c5b5322b32ee5f55b99e0ceabbe320",
    9: "e9214c506cce19f349555bcb2ebf164cf7a8468d4eae4e8a80700204c7bc9251",
    10: "19d88e522ea171c210a237f2612481f63cf645a97e2c006dd65dbad87eeb8bfd",
    11: "533034ac20d1e72ce053a0a58b2e34592d0d4aea3898c94cc1d1b7a7353ef6e2",
    12: "029358ef4d004100441ee34214d39410221353e3f882d6d2841df57ab7e95556",
    13: "8d3e00b4e255c899e0164feea613efb11351d2aa39df3aeb38ff884ded2eda91",
    14: "061495603fc60f91c864d7efabc407624d2373597a752e5e6cf27898eb0a40b3",
}

_INFERENCE_MODES = ("synthetic_smoke_test", "tusz_patient_oof")
_EXPECTED_STATE_SHAPES = {
    "detector.pos_encoder.weight": (20, 200),
    "detector.tx_encoder.self_attn.in_proj_weight": (600, 200),
    "detector.tx_encoder.self_attn.in_proj_bias": (600,),
    "detector.tx_encoder.self_attn.out_proj.weight": (200, 200),
    "detector.tx_encoder.self_attn.out_proj.bias": (200,),
    "detector.tx_encoder.linear1.weight": (256, 200),
    "detector.tx_encoder.linear1.bias": (256,),
    "detector.tx_encoder.linear2.weight": (200, 256),
    "detector.tx_encoder.linear2.bias": (200,),
    "detector.tx_encoder.norm1.weight": (200,),
    "detector.tx_encoder.norm1.bias": (200,),
    "detector.tx_encoder.norm2.weight": (200,),
    "detector.tx_encoder.norm2.bias": (200,),
    "detector.multi_lstm.weight_ih_l0": (400, 200),
    "detector.multi_lstm.weight_hh_l0": (400, 100),
    "detector.multi_lstm.bias_ih_l0": (400,),
    "detector.multi_lstm.bias_hh_l0": (400,),
    "detector.multi_lstm.weight_ih_l0_reverse": (400, 200),
    "detector.multi_lstm.weight_hh_l0_reverse": (400, 100),
    "detector.multi_lstm.bias_ih_l0_reverse": (400,),
    "detector.multi_lstm.bias_hh_l0_reverse": (400,),
    "detector.multi_linear.weight": (2, 200),
    "detector.multi_linear.bias": (2,),
    "hc_linear.weight": (1, 200),
    "hc_linear.bias": (1,),
}


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


def _normalize_patient_id(value: object) -> str:
    text = str(value).strip()
    if not text or not text.isdigit():
        raise ValueError("DeepSOZ patient ID must be numeric")
    return str(int(text))


def load_published_deepsoz_oof_fold_assignment(
    fold_directory: str | Path,
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    """Load the 15 exact official test arrays with ``allow_pickle=False``."""

    directory = Path(fold_directory)
    membership: dict[str, list[int]] = {}
    files: list[dict[str, Any]] = []
    for fold in range(15):
        path = directory / f"deepsoz_official_pts_test_fold{fold}.npy"
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"official DeepSOZ test-fold array missing: {path}")
        payload = path.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        expected = PUBLISHED_DEEPSOZ_TEST_FOLD_NPY_SHA256[fold]
        if sha256 != expected:
            raise ValueError(
                f"DeepSOZ fold {fold} test-array SHA-256 mismatch: "
                f"expected {expected}, got {sha256}"
            )
        values = np.load(io.BytesIO(payload), allow_pickle=False)
        if values.ndim != 1 or values.size != 24 or values.dtype.kind not in "iu":
            raise ValueError(f"DeepSOZ fold {fold} test-array schema drifted")
        patient_ids = [_normalize_patient_id(value) for value in values.tolist()]
        if len(set(patient_ids)) != 24:
            raise ValueError(f"DeepSOZ fold {fold} contains duplicate patients")
        for patient_id in patient_ids:
            membership.setdefault(patient_id, []).append(fold)
        files.append(
            {
                "fold_index": fold,
                "filename": path.name,
                "sha256": sha256,
                "patient_count": 24,
            }
        )
    normalized = {
        patient_id: tuple(folds) for patient_id, folds in sorted(membership.items())
    }
    distribution = {
        str(count): sum(len(folds) == count for folds in normalized.values())
        for count in sorted({len(folds) for folds in normalized.values()})
    }
    if len(normalized) != 124 or distribution != {"1": 1, "2": 10, "3": 113}:
        raise ValueError("DeepSOZ official held-out patient coverage drifted")
    assignments = [[patient_id, list(folds)] for patient_id, folds in normalized.items()]
    body: dict[str, Any] = {
        "schema_version": DEEPSOZ_FOLD_ASSIGNMENT_SCHEMA_VERSION,
        "receipt_id": "DEEPSOZ-FOLD-ASSIGNMENT-PENDING",
        "file_count": len(files),
        "files": files,
        "unique_patient_count": len(normalized),
        "total_held_out_memberships": sum(len(folds) for folds in normalized.values()),
        "held_out_repeat_count_distribution": distribution,
        "patient_fold_assignments": assignments,
        "patient_assignment_sha256": _canonical_sha256(assignments),
        "npy_allow_pickle": False,
        "all_124_patients_have_at_least_one_held_out_fold": True,
    }
    body["receipt_id"] = "DSZFOLD-" + _canonical_sha256(body)[:24]
    return normalized, body


class _PublishedDeepSOZTemporal(nn.Module):
    """Architecture matching the 25 published DeepSOZ state tensors."""

    def __init__(self, dropout: float = 0.15) -> None:
        super().__init__()
        self.detector = nn.Module()
        self.detector.pos_encoder = nn.Embedding(20, 200)
        self.detector.tx_encoder = nn.TransformerEncoderLayer(
            d_model=200,
            nhead=8,
            dim_feedforward=256,
            dropout=float(dropout),
            batch_first=True,
        )
        self.detector.multi_lstm = nn.LSTM(
            input_size=200,
            hidden_size=100,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.detector.multi_linear = nn.Linear(200, 2)
        self.hc_linear = nn.Linear(200, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[0:2] != (1, 1) or x.shape[-2:] != (19, 200):
            raise ValueError(f"DeepSOZ input must be [1,1,T,19,200], got {tuple(x.shape)}")
        _, _, seconds, _, _ = x.shape
        positions = self.detector.pos_encoder(
            torch.arange(19, device=x.device)
        ).view(1, 19, 200)
        channel = x.reshape(seconds, 19, 200) + positions
        global_token = self.detector.pos_encoder(
            torch.full(
                (seconds, 1), 19, device=x.device, dtype=torch.long
            )
        )
        encoded = self.detector.tx_encoder(torch.cat([channel, global_token], dim=1))
        global_encoded = encoded[:, 19].reshape(1, seconds, 200)
        temporal, _ = self.detector.multi_lstm(global_encoded)
        logits = self.detector.multi_linear(temporal).reshape(seconds, 2)
        return logits


def _validate_state_dict(state: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError("DeepSOZ checkpoint is not a tensor state mapping")
    if set(state) != set(_EXPECTED_STATE_SHAPES):
        missing = sorted(set(_EXPECTED_STATE_SHAPES).difference(state))
        extra = sorted(set(state).difference(_EXPECTED_STATE_SHAPES))
        raise ValueError(
            f"DeepSOZ checkpoint state keys drifted; missing={missing}, extra={extra}"
        )
    for name, expected_shape in _EXPECTED_STATE_SHAPES.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"DeepSOZ checkpoint entry {name!r} is not a tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"DeepSOZ checkpoint shape drift for {name}: "
                f"expected {expected_shape}, got {tuple(value.shape)}"
            )
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"DeepSOZ checkpoint tensor {name!r} is invalid")
    return state


def _snapshot_weights_only_state(
    path: Path, *, expected_sha256: str
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    audit = audit_checkpoint_container(
        path,
        expected_sha256=expected_sha256,
        artifact_id=path.name,
    )
    if audit["container_format"] != "pytorch_zip_pickle":
        raise ValueError("DeepSOZ checkpoint is not the audited PyTorch ZIP format")
    snapshot = path.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != expected_sha256:
        raise RuntimeError("DeepSOZ checkpoint changed after static audit")
    try:
        state = torch.load(
            io.BytesIO(snapshot),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise RuntimeError(
            "DeepSOZ weights-only load failed; weights_only=False is forbidden"
        ) from error
    return _validate_state_dict(state), audit


def _signal_tensor_sha256(
    raw: np.ndarray, *, sampling_rate_hz: float, channel_names: Sequence[str]
) -> str:
    metadata = json.dumps(
        {
            "shape": list(raw.shape),
            "dtype": "little_endian_float64",
            "sampling_rate_hz": sampling_rate_hz,
            "channel_names": list(channel_names),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    canonical = np.ascontiguousarray(raw, dtype="<f8")
    digest = hashlib.sha256(metadata)
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _preprocess_standard19(
    raw_standard19_uv: np.ndarray,
    *,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(raw_standard19_uv, np.ndarray):
        raise TypeError("raw_standard19_uv must be a NumPy array")
    if raw_standard19_uv.ndim != 2 or raw_standard19_uv.shape[0] != 19:
        raise ValueError("raw_standard19_uv must have shape [19,n_samples]")
    if not np.issubdtype(raw_standard19_uv.dtype, np.number):
        raise TypeError("raw_standard19_uv must be numeric")
    raw = np.asarray(raw_standard19_uv, dtype=np.float64)
    if raw.shape[1] < 2 or not np.isfinite(raw).all():
        raise ValueError("raw_standard19_uv must contain finite EEG samples")
    if isinstance(sampling_rate_hz, bool) or not isinstance(
        sampling_rate_hz, (int, float)
    ):
        raise TypeError("sampling_rate_hz must be numeric")
    source_rate = float(sampling_rate_hz)
    if not math.isfinite(source_rate) or source_rate <= 0:
        raise ValueError("sampling_rate_hz must be positive and finite")
    target_samples = int(raw.shape[1] * DEEPSOZ_TARGET_SAMPLING_RATE_HZ / source_rate)
    if target_samples < 200:
        raise ValueError("recording is shorter than one DeepSOZ output second")

    low_b, low_a = signal.butter(4, 30.0 / 100.0)
    high_b, high_a = signal.butter(4, 1.6 / 100.0, btype="highpass")
    output: list[np.ndarray] = []
    constant_channel_count = 0
    for channel in raw:
        resampled = signal.resample(channel, target_samples)
        filtered = signal.filtfilt(low_b, low_a, resampled, method="gust")
        filtered = signal.filtfilt(high_b, high_a, filtered, method="gust")
        center = float(filtered.mean())
        spread = float(filtered.std())
        if not math.isfinite(spread) or spread <= 0:
            constant_channel_count += 1
            output.append(np.zeros(target_samples, dtype=np.float64))
        else:
            output.append(
                np.clip(filtered, center - 2.0 * spread, center + 2.0 * spread)
            )
    processed = np.stack(output)
    full_seconds = processed.shape[1] // 200
    processed = processed[:, : full_seconds * 200]
    global_center = float(processed.mean())
    global_scale = float(processed.std())
    if not math.isfinite(global_scale) or global_scale <= 0:
        raise ValueError("DeepSOZ record normalization has zero/nonfinite scale")
    normalized = np.ascontiguousarray(
        (processed - global_center) / global_scale, dtype=np.float64
    )
    receipt = {
        "source_sampling_rate_hz": source_rate,
        "target_sampling_rate_hz": DEEPSOZ_TARGET_SAMPLING_RATE_HZ,
        "source_sample_count": int(raw.shape[1]),
        "resampled_sample_count_before_full_second_trim": int(target_samples),
        "modeled_full_second_count": int(full_seconds),
        "resampling": "scipy_signal_fft_resample_whole_record",
        "low_pass_filter": "butterworth_order4_30Hz_then_filtfilt_gust",
        "high_pass_filter": "butterworth_order4_1_6Hz_then_filtfilt_gust",
        "per_channel_clipping": "mean_plus_or_minus_2_standard_deviations",
        "record_normalization": "whole_processed_record_global_mean_standard_deviation",
        "constant_channel_count": constant_channel_count,
        "missing_channel_imputation": False,
        "silent_time_padding": False,
    }
    return normalized, receipt


def _chunk_plan(full_seconds: int) -> list[tuple[int, int]]:
    if full_seconds < 1:
        raise ValueError("DeepSOZ chunk plan needs at least one second")
    chunks: list[tuple[int, int]] = []
    start = 0
    while True:
        stop = min(full_seconds, start + DEEPSOZ_CHUNK_SECONDS)
        chunks.append((start, stop))
        if stop == full_seconds:
            break
        start += DEEPSOZ_STRIDE_SECONDS
    return chunks


def _edge_fusion_weights(
    *, start: int, stop: int, full_seconds: int
) -> np.ndarray:
    length = stop - start
    weights = np.ones(length, dtype=np.float64)
    ramp_length = min(DEEPSOZ_OVERLAP_SECONDS, length)
    if start > 0:
        weights[:ramp_length] *= np.arange(1, ramp_length + 1) / ramp_length
    if stop < full_seconds:
        weights[-ramp_length:] *= np.arange(ramp_length, 0, -1) / ramp_length
    return weights


class DeepSOZTemporalResearchAdapter:
    """Hash-bound, weights-only research adapter for one published fold."""

    provider_id = "deepsoz_temporal_oof_candidate_v1"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        expected_checkpoint_sha256: str,
        weights_manifest_sha256: str,
        fold_index: int,
        inference_mode: str,
        fold_assignment_receipt_sha256: str | None = None,
        device: str = "cpu",
    ) -> None:
        if not _is_sha256(expected_checkpoint_sha256):
            raise ValueError("expected checkpoint SHA-256 is invalid")
        if not _is_sha256(weights_manifest_sha256):
            raise ValueError("weight manifest SHA-256 is invalid")
        if isinstance(fold_index, bool) or not isinstance(fold_index, int) or not 0 <= fold_index < 15:
            raise ValueError("DeepSOZ fold_index must be between 0 and 14")
        if inference_mode not in _INFERENCE_MODES:
            raise ValueError("DeepSOZ inference mode is invalid")
        if inference_mode == "tusz_patient_oof" and not _is_sha256(
            fold_assignment_receipt_sha256
        ):
            raise ValueError("TUSZ OOF mode requires a fold-assignment receipt SHA-256")
        if inference_mode == "synthetic_smoke_test" and fold_assignment_receipt_sha256 is not None:
            raise ValueError("synthetic smoke mode must not claim a patient fold receipt")
        self.checkpoint_path = Path(checkpoint_path)
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self.weights_manifest_sha256 = weights_manifest_sha256
        self.fold_index = fold_index
        self.inference_mode = inference_mode
        self.fold_assignment_receipt_sha256 = fold_assignment_receipt_sha256
        self.device = torch.device(device)
        state, audit = _snapshot_weights_only_state(
            self.checkpoint_path, expected_sha256=expected_checkpoint_sha256
        )
        model = _PublishedDeepSOZTemporal(dropout=0.15).double()
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        self.model = model
        self.checkpoint_audit = audit

    def materialize_dense_posterior(
        self,
        *,
        recording_id: str,
        standardized_eeg: object,
        sampling_rate_hz: float,
        channel_names: Sequence[str],
    ) -> dict[str, Any]:
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise TypeError("recording_id must be a non-empty string")
        names = tuple(str(value).strip().upper() for value in channel_names)
        if names != STANDARD_19:
            raise ValueError("DeepSOZ adapter requires the exact STANDARD_19 channel order")
        if not isinstance(standardized_eeg, np.ndarray):
            raise TypeError("standardized_eeg must be a NumPy array in microvolts")
        raw = np.asarray(standardized_eeg, dtype=np.float64)
        source_tensor_sha256 = _signal_tensor_sha256(
            raw, sampling_rate_hz=float(sampling_rate_hz), channel_names=names
        )
        original_duration = raw.shape[1] / float(sampling_rate_hz)
        processed, preprocessing = _preprocess_standard19(
            raw, sampling_rate_hz=sampling_rate_hz
        )
        full_seconds = int(preprocessing["modeled_full_second_count"])
        windows = processed.reshape(19, full_seconds, 200).transpose(1, 0, 2)
        probability_sum = np.zeros(full_seconds, dtype=np.float64)
        weight_sum = np.zeros(full_seconds, dtype=np.float64)
        chunks: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, (start, stop) in enumerate(_chunk_plan(full_seconds), start=1):
                x = torch.from_numpy(windows[start:stop]).unsqueeze(0).unsqueeze(0)
                logits = self.model(x.to(device=self.device, dtype=torch.float64))
                probability = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                if probability.shape != (stop - start,) or not np.isfinite(probability).all():
                    raise RuntimeError("DeepSOZ produced an invalid posterior chunk")
                weights = _edge_fusion_weights(
                    start=start, stop=stop, full_seconds=full_seconds
                )
                probability_sum[start:stop] += probability * weights
                weight_sum[start:stop] += weights
                chunks.append(
                    {
                        "chunk_id": f"DEEPSOZ-CHUNK-{index:04d}",
                        "start_offset_seconds": float(start),
                        "stop_offset_seconds": float(stop),
                        "modeled_seconds": stop - start,
                        "left_overlap_seconds": (
                            0 if start == 0 else min(DEEPSOZ_OVERLAP_SECONDS, stop - start)
                        ),
                        "right_overlap_seconds": (
                            0
                            if stop == full_seconds
                            else min(DEEPSOZ_OVERLAP_SECONDS, stop - start)
                        ),
                    }
                )
        if np.any(weight_sum <= 0):
            raise RuntimeError("DeepSOZ overlap fusion left a coverage hole")
        fused = probability_sum / weight_sum
        timeline: list[dict[str, Any]] = [
            {
                "window_id": f"DEEPSOZ-SEC-{second:07d}",
                "start_offset_seconds": float(second),
                "stop_offset_seconds": float(second + 1),
                "seizure_probability": float(fused[second]),
                "signal_usable": True,
            }
            for second in range(full_seconds)
        ]
        if original_duration - full_seconds > 1e-9:
            timeline.append(
                {
                    "window_id": "DEEPSOZ-PARTIAL-TAIL",
                    "start_offset_seconds": float(full_seconds),
                    "stop_offset_seconds": float(original_duration),
                    "seizure_probability": 0.0,
                    "signal_usable": False,
                }
            )

        body: dict[str, Any] = {
            "schema_version": DEEPSOZ_TEMPORAL_ADAPTER_SCHEMA_VERSION,
            "posterior_artifact_id": "DEEPSOZ-POSTERIOR-PENDING",
            "adapter_id": DEEPSOZ_TEMPORAL_ADAPTER_ID,
            "adapter_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provider_id": self.provider_id,
            "recording_id": recording_id,
            "inference_mode": self.inference_mode,
            "fold_index": self.fold_index,
            "fold_assignment_receipt_sha256": self.fold_assignment_receipt_sha256,
            "checkpoint_sha256": self.expected_checkpoint_sha256,
            "weights_manifest_sha256": self.weights_manifest_sha256,
            "source_signal_tensor_sha256": source_tensor_sha256,
            "recording_duration_seconds": float(original_duration),
            "preprocessing_receipt": preprocessing,
            "chunking_receipt": {
                "chunk_seconds": DEEPSOZ_CHUNK_SECONDS,
                "overlap_seconds": DEEPSOZ_OVERLAP_SECONDS,
                "stride_seconds": DEEPSOZ_STRIDE_SECONDS,
                "chunk_count": len(chunks),
                "chunks": chunks,
                "overlap_fusion": "linear_edge_ramp_weighted_probability_mean",
                "bidirectional_chunk_context": True,
                "chunk_boundary_probabilities_are_calibrated": False,
            },
            "posterior_timeline": timeline,
            "scope_receipt": {
                "eeg_signal_only": True,
                "edf_annotations_used": False,
                "excel_used": False,
                "clinical_context_used": False,
                "seizure_or_soz_labels_used_for_inference": False,
                "posterior_is_confirmed_seizure_or_onset": False,
                "research_only": True,
                "sota_claim_authorized": False,
            },
        }
        body["posterior_artifact_id"] = "DSZPOST-" + _canonical_sha256(body)[:24]
        return body

    def predict_dense_posterior(
        self,
        *,
        recording_id: str,
        standardized_eeg: object,
        sampling_rate_hz: float,
        channel_names: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        artifact = self.materialize_dense_posterior(
            recording_id=recording_id,
            standardized_eeg=standardized_eeg,
            sampling_rate_hz=sampling_rate_hz,
            channel_names=channel_names,
        )
        return deepcopy(artifact["posterior_timeline"])


def aggregate_deepsoz_oof_fold_posteriors(
    *,
    patient_id: str,
    expected_fold_indices: Sequence[int],
    fold_assignment_receipt: Mapping[str, Any],
    fold_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Average all published held-out-repeat posteriors for one TUSZ patient."""

    normalized_patient = _normalize_patient_id(patient_id)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in expected_fold_indices):
        raise TypeError("expected DeepSOZ OOF fold indices must be integers")
    folds = tuple(int(value) for value in expected_fold_indices)
    if not folds or len(folds) != len(set(folds)) or any(not 0 <= value < 15 for value in folds):
        raise ValueError("expected DeepSOZ OOF fold indices are invalid")
    assignment = deepcopy(dict(fold_assignment_receipt))
    if assignment.get("schema_version") != DEEPSOZ_FOLD_ASSIGNMENT_SCHEMA_VERSION:
        raise ValueError("DeepSOZ fold-assignment receipt schema drifted")
    digest = deepcopy(assignment)
    receipt_id = digest.get("receipt_id")
    digest["receipt_id"] = "DEEPSOZ-FOLD-ASSIGNMENT-PENDING"
    if receipt_id != "DSZFOLD-" + _canonical_sha256(digest)[:24]:
        raise ValueError("DeepSOZ fold-assignment receipt content binding failed")
    assignment_sha256 = _canonical_sha256(assignment)
    raw_assignments = assignment.get("patient_fold_assignments")
    if not isinstance(raw_assignments, list):
        raise ValueError("DeepSOZ fold-assignment receipt lacks patient bindings")
    assignment_lookup: dict[str, tuple[int, ...]] = {}
    for value in raw_assignments:
        if not isinstance(value, list) or len(value) != 2 or not isinstance(value[1], list):
            raise ValueError("DeepSOZ patient fold binding is malformed")
        binding_patient = _normalize_patient_id(value[0])
        binding_folds = tuple(int(fold) for fold in value[1])
        if binding_patient in assignment_lookup:
            raise ValueError("DeepSOZ patient fold binding is duplicated")
        assignment_lookup[binding_patient] = binding_folds
    if assignment_lookup.get(normalized_patient) != tuple(folds):
        raise ValueError("requested folds do not match the patient's official binding")
    if not isinstance(fold_artifacts, Sequence) or not fold_artifacts:
        raise TypeError("DeepSOZ OOF fold artifacts must be non-empty")
    artifacts = [deepcopy(dict(value)) for value in fold_artifacts]
    observed_folds = tuple(sorted(int(value.get("fold_index", -1)) for value in artifacts))
    if tuple(sorted(folds)) != observed_folds:
        raise ValueError("DeepSOZ OOF fold artifacts do not match held-out assignment")
    first = artifacts[0]
    timeline_count = len(first.get("posterior_timeline", []))
    if timeline_count < 1:
        raise ValueError("DeepSOZ OOF posterior timeline is empty")
    comparison_fields = (
        "recording_id",
        "adapter_code_sha256",
        "source_signal_tensor_sha256",
        "recording_duration_seconds",
        "preprocessing_receipt",
    )
    for artifact in artifacts:
        if artifact.get("schema_version") != DEEPSOZ_TEMPORAL_ADAPTER_SCHEMA_VERSION:
            raise ValueError("DeepSOZ fold posterior schema drifted")
        content = deepcopy(artifact)
        content["posterior_artifact_id"] = "DEEPSOZ-POSTERIOR-PENDING"
        if artifact.get("posterior_artifact_id") != (
            "DSZPOST-" + _canonical_sha256(content)[:24]
        ):
            raise ValueError("DeepSOZ fold posterior content binding failed")
        if artifact.get("inference_mode") != "tusz_patient_oof":
            raise ValueError("DeepSOZ OOF aggregate received a non-OOF artifact")
        if artifact.get("fold_assignment_receipt_sha256") != assignment_sha256:
            raise ValueError("DeepSOZ fold posterior used another assignment receipt")
        if artifact.get("weights_manifest_sha256") != (
            PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256
        ):
            raise ValueError("DeepSOZ fold posterior weight manifest drifted")
        if any(artifact.get(field) != first.get(field) for field in comparison_fields):
            raise ValueError("DeepSOZ fold posteriors do not bind the same signal")
        if len(artifact.get("posterior_timeline", [])) != timeline_count:
            raise ValueError("DeepSOZ fold posterior timeline lengths differ")

    timeline: list[dict[str, Any]] = []
    for index in range(timeline_count):
        rows = [artifact["posterior_timeline"][index] for artifact in artifacts]
        reference = rows[0]
        interval_fields = (
            "start_offset_seconds",
            "stop_offset_seconds",
            "signal_usable",
        )
        if any(
            any(row.get(field) != reference.get(field) for field in interval_fields)
            for row in rows[1:]
        ):
            raise ValueError("DeepSOZ fold posterior time grids differ")
        usable = reference["signal_usable"] is True
        probabilities = [float(row["seizure_probability"]) for row in rows]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("DeepSOZ fold posterior probability is invalid")
        timeline.append(
            {
                "window_id": f"DEEPSOZ-OOF-SEC-{index:07d}",
                "start_offset_seconds": float(reference["start_offset_seconds"]),
                "stop_offset_seconds": float(reference["stop_offset_seconds"]),
                "seizure_probability": (
                    float(sum(probabilities) / len(probabilities)) if usable else 0.0
                ),
                "signal_usable": usable,
            }
        )

    body: dict[str, Any] = {
        "schema_version": DEEPSOZ_OOF_ENSEMBLE_SCHEMA_VERSION,
        "posterior_artifact_id": "DEEPSOZ-OOF-POSTERIOR-PENDING",
        "provider_id": "deepsoz_temporal_oof_candidate_v1",
        "recording_id": first["recording_id"],
        "deepsoz_patient_id": normalized_patient,
        "held_out_fold_indices": list(sorted(folds)),
        "held_out_repeat_count": len(folds),
        "fold_assignment_receipt_sha256": assignment_sha256,
        "patient_fold_binding_sha256": _canonical_sha256(
            [normalized_patient, list(sorted(folds)), assignment["receipt_id"]]
        ),
        "weights_manifest_sha256": PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256,
        "adapter_code_sha256": first["adapter_code_sha256"],
        "fold_posterior_artifact_ids": [
            artifact["posterior_artifact_id"]
            for artifact in sorted(artifacts, key=lambda value: value["fold_index"])
        ],
        "source_signal_tensor_sha256": first["source_signal_tensor_sha256"],
        "recording_duration_seconds": first["recording_duration_seconds"],
        "preprocessing_receipt": first["preprocessing_receipt"],
        "fold_fusion": "arithmetic_mean_of_all_published_patient_held_out_repeat_probabilities",
        "posterior_timeline": timeline,
        "scope_receipt": {
            "eeg_signal_only": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "clinical_context_used": False,
            "reference_seizure_times_used_for_inference": False,
            "fold_assignment_uses_patient_split_metadata_only": True,
            "research_only": True,
            "posterior_is_confirmed_seizure_or_onset": False,
            "sota_claim_authorized": False,
        },
    }
    body["posterior_artifact_id"] = "DSZOOF-" + _canonical_sha256(body)[:24]
    return body


__all__ = [
    "DEEPSOZ_CHUNK_SECONDS",
    "DEEPSOZ_FOLD_ASSIGNMENT_SCHEMA_VERSION",
    "DEEPSOZ_OOF_ENSEMBLE_SCHEMA_VERSION",
    "DEEPSOZ_OVERLAP_SECONDS",
    "DEEPSOZ_STRIDE_SECONDS",
    "DEEPSOZ_TARGET_SAMPLING_RATE_HZ",
    "DEEPSOZ_TEMPORAL_ADAPTER_ID",
    "DEEPSOZ_TEMPORAL_ADAPTER_SCHEMA_VERSION",
    "DeepSOZTemporalResearchAdapter",
    "PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256",
    "PUBLISHED_DEEPSOZ_TEST_FOLD_NPY_SHA256",
    "PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256",
    "STANDARD_19",
    "aggregate_deepsoz_oof_fold_posteriors",
    "load_published_deepsoz_oof_fold_assignment",
]
