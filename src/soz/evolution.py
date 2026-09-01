"""Transparent primary temporal-evolution descriptors and train-fold scaler.

No learned head is used on the primary path.  Six named descriptors are
computed directly from complete standard-19, 200 Hz, volt-valued EEG in
fifteen independent four-second tiles.  The robust scaler is fitted only from
complete, explicitly declared source-train patient bags and gives every
patient equal weight regardless of their number of events.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Final, Sequence

import torch

from .geometry import (
    CHANNEL_INDEX,
    EVOLUTION_FEATURES,
    N_NODE_FEATURES,
    N_STANDARD_CHANNELS,
    N_TIME_TILES,
    STANDARD_19,
)


EVOLUTION_DESCRIPTOR_SCHEMA: Final[str] = "direct_temporal_evolution_v2"
PATIENT_ROBUST_SCALER_SCHEMA: Final[str] = "patient_balanced_robust_scaler_v2"
EVOLUTION_SAMPLE_RATE_HZ: Final[float] = 200.0
EVOLUTION_INPUT_SAMPLES: Final[int] = 12_000
EVOLUTION_TILE_SECONDS: Final[float] = 4.0
EVOLUTION_TILE_SAMPLES: Final[int] = 800
EVOLUTION_N_TILES: Final[int] = 15
EVOLUTION_UV_PER_VOLT: Final[float] = 1_000_000.0
EVOLUTION_LOG_FLOOR_UV: Final[float] = 1e-12
EVOLUTION_POWER_FLOOR: Final[float] = 1e-24
WELCH_SEGMENT_SAMPLES: Final[int] = 200
WELCH_STEP_SAMPLES: Final[int] = 100
WELCH_SEGMENTS_PER_TILE: Final[int] = 7
DEFAULT_SCALER_CLIP: Final[float] = 10.0
SCALER_IQR_FLOOR: Final[float] = 1e-6

# Frozen undirected 10--20 scalp-neighbor graph.  FZ and PZ are explicit
# physical nodes even though they do not occur in the common TCP20 edge set.
STANDARD19_NEIGHBOR_EDGES: Final[tuple[tuple[str, str], ...]] = (
    ("FP1", "FP2"),
    ("FP1", "F7"),
    ("FP1", "F3"),
    ("FP2", "F4"),
    ("FP2", "F8"),
    ("F7", "F3"),
    ("F7", "T7"),
    ("F3", "FZ"),
    ("F3", "C3"),
    ("FZ", "F4"),
    ("FZ", "CZ"),
    ("F4", "F8"),
    ("F4", "C4"),
    ("F8", "T8"),
    ("T7", "C3"),
    ("T7", "P7"),
    ("C3", "CZ"),
    ("C3", "P3"),
    ("CZ", "C4"),
    ("CZ", "PZ"),
    ("C4", "T8"),
    ("C4", "P4"),
    ("T8", "P8"),
    ("P7", "P3"),
    ("P7", "O1"),
    ("P3", "PZ"),
    ("P3", "O1"),
    ("PZ", "P4"),
    ("PZ", "O1"),
    ("PZ", "O2"),
    ("P4", "P8"),
    ("P4", "O2"),
    ("P8", "O2"),
    ("O1", "O2"),
)
STANDARD19_NEIGHBORS: Final[tuple[tuple[str, ...], ...]]

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


COMPLETE19_DESCRIPTOR_MASK_SHA256: Final[str] = _tensor_sha256(
    torch.ones((N_STANDARD_CHANNELS, N_TIME_TILES), dtype=torch.bool)
)


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return text


def _build_neighbors() -> tuple[tuple[str, ...], ...]:
    neighbors: dict[str, set[str]] = {channel: set() for channel in STANDARD_19}
    seen: set[frozenset[str]] = set()
    for left, right in STANDARD19_NEIGHBOR_EDGES:
        if left not in neighbors or right not in neighbors or left == right:
            raise RuntimeError("Invalid standard-19 neighbor edge")
        undirected = frozenset((left, right))
        if undirected in seen:
            raise RuntimeError("Duplicate standard-19 neighbor edge")
        seen.add(undirected)
        neighbors[left].add(right)
        neighbors[right].add(left)
    if any(not neighbors[channel] for channel in STANDARD_19):
        raise RuntimeError("Every standard-19 channel must have a fixed neighbor")
    return tuple(
        tuple(sorted(neighbors[channel], key=CHANNEL_INDEX.__getitem__))
        for channel in STANDARD_19
    )


STANDARD19_NEIGHBORS = _build_neighbors()

_FEATURE_DEFINITION_PAYLOAD: Final[dict[str, object]] = {
    "schema": EVOLUTION_DESCRIPTOR_SCHEMA,
    "features": EVOLUTION_FEATURES,
    "input": {
        "channels": STANDARD_19,
        "unit": "volts",
        "sample_rate_hz": EVOLUTION_SAMPLE_RATE_HZ,
        "samples": EVOLUTION_INPUT_SAMPLES,
        "tiles": EVOLUTION_N_TILES,
        "tile_seconds": EVOLUTION_TILE_SECONDS,
        "eligibility": "complete_physical_standard19_event_or_reject",
    },
    "computation": {
        "device": "cpu_only",
        "dtype": "torch.float64",
        "output_dtype": "torch.float64",
        "accelerator_forbidden": True,
    },
    "mask": {
        "policy": "constant_all_true_complete19_no_tile_qc",
        "event_shape": [N_STANDARD_CHANNELS, N_TIME_TILES],
        "event_mask_sha256": COMPLETE19_DESCRIPTOR_MASK_SHA256,
    },
    "amplitude": {
        "log": "natural_log",
        "rms_unit": "microvolts",
        "line_length": "mean_absolute_first_difference_microvolts_per_sample",
        "floor_uv": EVOLUTION_LOG_FLOOR_UV,
    },
    "spectrum": {
        "tile_detrend": "subtract_mean",
        "window": "periodic_hann_800",
        "centroid_band_hz": [1.0, 45.0],
        "entropy_band_hz": [1.0, 45.0],
        "entropy_normalization": "log_number_of_frequency_bins",
        "rhythmicity_numerator": (
            "maximum_2Hz_frequency-bin-aligned_sliding_band_power"
        ),
        "rhythmicity_denominator_band_hz": [1.0, 30.0],
        "power_floor": EVOLUTION_POWER_FLOOR,
    },
    "coherence": {
        "segment_seconds": 1.0,
        "overlap_fraction": 0.5,
        "segments_per_tile": WELCH_SEGMENTS_PER_TILE,
        "segment_detrend": "subtract_mean",
        "window": "periodic_hann_200",
        "band_hz": [4.0, 30.0],
        "frequency_reduction": "unweighted_mean",
        "neighbor_reduction": "unweighted_mean_over_available_fixed_neighbors",
        "neighbor_edges": STANDARD19_NEIGHBOR_EDGES,
    },
}
EVOLUTION_FEATURE_SCHEMA_SHA256: Final[str] = _sha256_json(
    _FEATURE_DEFINITION_PAYLOAD
)


@dataclass(frozen=True)
class EvolutionDescriptorReceipt:
    """Frozen definitions for the directly computed descriptor tensor."""

    feature_names: tuple[str, ...] = EVOLUTION_FEATURES
    feature_schema_sha256: str = EVOLUTION_FEATURE_SCHEMA_SHA256
    standard_channels: tuple[str, ...] = STANDARD_19
    neighbor_edges: tuple[tuple[str, str], ...] = STANDARD19_NEIGHBOR_EDGES
    input_unit: str = "volts"
    sample_rate_hz: float = EVOLUTION_SAMPLE_RATE_HZ
    input_samples: int = EVOLUTION_INPUT_SAMPLES
    tile_seconds: float = EVOLUTION_TILE_SECONDS
    tile_samples: int = EVOLUTION_TILE_SAMPLES
    n_tiles: int = EVOLUTION_N_TILES
    output_unit_policy: str = "named_mixed_units_before_train_fold_scaling"
    log_policy: str = "natural_log_with_1e-12_microvolt_floor"
    spectral_policy: str = "periodic_hann_4s_demeaned_rfft"
    rhythmicity_policy: str = "max_sliding_2Hz_power_over_1_30Hz_power"
    coherence_policy: str = "welch_1s_hann_50pct_overlap_mean_4_30Hz"
    compute_device_policy: str = "cpu_only_no_accelerator"
    compute_dtype: str = "torch.float64"
    output_dtype: str = "torch.float64"
    mask_policy: str = "constant_all_true_complete19_no_tile_qc"
    event_mask_sha256: str = COMPLETE19_DESCRIPTOR_MASK_SHA256
    schema_version: str = EVOLUTION_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        if self.feature_names != EVOLUTION_FEATURES:
            raise ValueError("Evolution feature order must match geometry.EVOLUTION_FEATURES")
        if self.feature_schema_sha256 != EVOLUTION_FEATURE_SCHEMA_SHA256:
            raise ValueError("Evolution feature schema hash mismatch")
        if self.standard_channels != STANDARD_19:
            raise ValueError("Evolution receipt must use the frozen standard-19 order")
        if self.neighbor_edges != STANDARD19_NEIGHBOR_EDGES:
            raise ValueError("Evolution receipt must use the frozen neighbor graph")
        if self.input_unit != "volts" or self.sample_rate_hz != 200.0:
            raise ValueError("Evolution input contract is volts at 200 Hz")
        if (
            self.input_samples != EVOLUTION_INPUT_SAMPLES
            or self.tile_samples != EVOLUTION_TILE_SAMPLES
            or self.n_tiles != EVOLUTION_N_TILES
            or self.tile_seconds != EVOLUTION_TILE_SECONDS
        ):
            raise ValueError("Evolution receipt has an invalid fixed time grid")
        if self.schema_version != EVOLUTION_DESCRIPTOR_SCHEMA:
            raise ValueError("Unsupported evolution descriptor schema")
        if (
            self.compute_device_policy != "cpu_only_no_accelerator"
            or self.compute_dtype != "torch.float64"
            or self.output_dtype != "torch.float64"
        ):
            raise ValueError("Evolution descriptors require frozen CPU float64")
        if self.mask_policy != "constant_all_true_complete19_no_tile_qc":
            raise ValueError("Evolution descriptor mask policy cannot be changed")
        if self.event_mask_sha256 != COMPLETE19_DESCRIPTOR_MASK_SHA256:
            raise ValueError("Evolution complete19 mask SHA mismatch")


@dataclass(frozen=True)
class TemporalEvolutionDescriptors:
    """Detached direct descriptors and the fixed all-true compatibility mask."""

    descriptors: torch.Tensor
    mask: torch.Tensor
    receipt: EvolutionDescriptorReceipt

    def __post_init__(self) -> None:
        if self.descriptors.ndim != 4:
            raise ValueError("Evolution descriptors must have rank four")
        batch_size = self.descriptors.shape[0]
        if tuple(self.descriptors.shape) != (
            batch_size,
            N_STANDARD_CHANNELS,
            N_TIME_TILES,
            N_NODE_FEATURES,
        ):
            raise ValueError("Evolution descriptors must have shape [B,19,15,6]")
        if tuple(self.mask.shape) != (
            batch_size,
            N_STANDARD_CHANNELS,
            N_TIME_TILES,
        ):
            raise ValueError("Evolution mask must have shape [B,19,15]")
        if self.descriptors.dtype != torch.float64 or self.mask.dtype != torch.bool:
            raise TypeError("Evolution descriptors must be CPU float64 and mask bool")
        if self.descriptors.device.type != "cpu" or self.mask.device.type != "cpu":
            raise TypeError("Evolution descriptors and mask must remain on CPU")
        if self.descriptors.requires_grad or self.mask.requires_grad:
            raise ValueError("Evolution descriptors and mask must be detached")
        if not torch.isfinite(self.descriptors).all():
            raise ValueError("Evolution descriptors must be finite")
        if torch.any(self.descriptors[~self.mask] != 0):
            raise ValueError("Masked evolution descriptors must use finite zero fill")
        if not self.mask.all().item():
            raise ValueError("Complete19 descriptor mask must be constant all-true")
        for event_mask in self.mask:
            if _tensor_sha256(event_mask) != COMPLETE19_DESCRIPTOR_MASK_SHA256:
                raise ValueError("Complete19 descriptor mask SHA mismatch")
        if self.receipt.feature_names != EVOLUTION_FEATURES:
            raise ValueError("Evolution descriptor feature order disagrees with receipt")


def _validate_eeg(eeg_volts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(eeg_volts, torch.Tensor):
        raise TypeError("eeg_volts must be a torch.Tensor")
    if eeg_volts.ndim != 3 or tuple(eeg_volts.shape[1:]) != (
        N_STANDARD_CHANNELS,
        EVOLUTION_INPUT_SAMPLES,
    ):
        raise ValueError("Evolution input must have shape [B,19,12000]")
    if eeg_volts.shape[0] < 1:
        raise ValueError("Evolution input batch cannot be empty")
    if not eeg_volts.is_floating_point():
        raise TypeError("Evolution input must be a floating-point tensor in volts")
    if eeg_volts.device.type != "cpu":
        raise ValueError("Primary evolution computation is CPU-only")
    if not torch.isfinite(eeg_volts).all():
        raise ValueError("Evolution input must be finite")

    expected_mask_shape = (
        eeg_volts.shape[0],
        N_STANDARD_CHANNELS,
        N_TIME_TILES,
    )
    valid = torch.ones(expected_mask_shape, dtype=torch.bool, device="cpu")
    return eeg_volts.detach(), valid


def _band_mask(frequencies: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return (frequencies >= low - TIME_FREQUENCY_EPS) & (
        frequencies <= high + TIME_FREQUENCY_EPS
    )


TIME_FREQUENCY_EPS: Final[float] = 1e-9


def _local_descriptors(tiles_uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return five local descriptors plus the Welch Fourier coefficients."""

    rms = torch.sqrt(torch.mean(tiles_uv.square(), dim=-1))
    log_rms = torch.log(rms.clamp_min(EVOLUTION_LOG_FLOOR_UV))
    mean_line_length = torch.mean(torch.abs(torch.diff(tiles_uv, dim=-1)), dim=-1)
    log_line_length = torch.log(
        mean_line_length.clamp_min(EVOLUTION_LOG_FLOOR_UV)
    )

    demeaned = tiles_uv - tiles_uv.mean(dim=-1, keepdim=True)
    tile_window = torch.hann_window(
        EVOLUTION_TILE_SAMPLES,
        periodic=True,
        dtype=tiles_uv.dtype,
        device=tiles_uv.device,
    )
    spectrum = torch.fft.rfft(demeaned * tile_window, dim=-1)
    power = spectrum.abs().square()
    frequencies = torch.fft.rfftfreq(
        EVOLUTION_TILE_SAMPLES,
        d=1.0 / EVOLUTION_SAMPLE_RATE_HZ,
        device=tiles_uv.device,
    ).to(dtype=tiles_uv.dtype)

    band_1_45 = _band_mask(frequencies, 1.0, 45.0)
    selected_1_45 = power[..., band_1_45]
    frequencies_1_45 = frequencies[band_1_45]
    total_1_45 = selected_1_45.sum(dim=-1)
    spectral_centroid = (
        selected_1_45 * frequencies_1_45
    ).sum(dim=-1) / total_1_45.clamp_min(EVOLUTION_POWER_FLOOR)
    probabilities = selected_1_45 / total_1_45.clamp_min(EVOLUTION_POWER_FLOOR)[
        ..., None
    ]
    entropy_terms = torch.where(
        probabilities > 0,
        probabilities * torch.log(probabilities.clamp_min(EVOLUTION_POWER_FLOOR)),
        torch.zeros_like(probabilities),
    )
    normalized_entropy = -entropy_terms.sum(dim=-1) / math.log(
        selected_1_45.shape[-1]
    )

    band_1_30 = _band_mask(frequencies, 1.0, 30.0)
    selected_1_30 = power[..., band_1_30]
    frequency_resolution = EVOLUTION_SAMPLE_RATE_HZ / EVOLUTION_TILE_SAMPLES
    subband_bins = int(round(2.0 / frequency_resolution))
    if subband_bins < 1 or selected_1_30.shape[-1] < subband_bins:
        raise RuntimeError("Invalid frozen rhythmicity frequency grid")
    sliding_power = selected_1_30.unfold(-1, subband_bins, 1).sum(dim=-1)
    rhythmicity = sliding_power.max(dim=-1).values / selected_1_30.sum(
        dim=-1
    ).clamp_min(EVOLUTION_POWER_FLOOR)

    local = torch.stack(
        (
            log_rms,
            log_line_length,
            spectral_centroid,
            normalized_entropy,
            rhythmicity,
        ),
        dim=-1,
    )

    segments = tiles_uv.unfold(
        -1, WELCH_SEGMENT_SAMPLES, WELCH_STEP_SAMPLES
    )
    if segments.shape[-2] != WELCH_SEGMENTS_PER_TILE:
        raise RuntimeError("Welch segmentation disagrees with the frozen 4 s grid")
    segments = segments - segments.mean(dim=-1, keepdim=True)
    welch_window = torch.hann_window(
        WELCH_SEGMENT_SAMPLES,
        periodic=True,
        dtype=tiles_uv.dtype,
        device=tiles_uv.device,
    )
    welch_fft = torch.fft.rfft(segments * welch_window, dim=-1)
    welch_frequencies = torch.fft.rfftfreq(
        WELCH_SEGMENT_SAMPLES,
        d=1.0 / EVOLUTION_SAMPLE_RATE_HZ,
        device=tiles_uv.device,
    )
    coherence_band = _band_mask(welch_frequencies, 4.0, 30.0)
    return local, welch_fft[..., coherence_band]


def _mean_neighbor_coherence(
    welch_fft: torch.Tensor, tile_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = welch_fft.shape[0]
    coherence_sum = torch.zeros(
        (batch_size, N_STANDARD_CHANNELS, N_TIME_TILES),
        dtype=welch_fft.real.dtype,
        device=welch_fft.device,
    )
    neighbor_count = torch.zeros_like(coherence_sum)
    for left_name, right_name in STANDARD19_NEIGHBOR_EDGES:
        left = CHANNEL_INDEX[left_name]
        right = CHANNEL_INDEX[right_name]
        left_fft = welch_fft[:, left]
        right_fft = welch_fft[:, right]
        cross = (left_fft * right_fft.conj()).mean(dim=-2)
        left_power = left_fft.abs().square().mean(dim=-2)
        right_power = right_fft.abs().square().mean(dim=-2)
        coherence_by_frequency = cross.abs().square() / (
            left_power * right_power
        ).clamp_min(EVOLUTION_POWER_FLOOR)
        coherence = coherence_by_frequency.mean(dim=-1).clamp(0.0, 1.0)
        pair_valid = tile_mask[:, left] & tile_mask[:, right]
        pair_weight = pair_valid.to(dtype=coherence.dtype)
        coherence_sum[:, left] += coherence * pair_weight
        coherence_sum[:, right] += coherence * pair_weight
        neighbor_count[:, left] += pair_weight
        neighbor_count[:, right] += pair_weight
    available = tile_mask & (neighbor_count > 0)
    mean_coherence = coherence_sum / neighbor_count.clamp_min(1.0)
    return mean_coherence, available


def compute_temporal_evolution_descriptors(
    eeg_volts: torch.Tensor,
) -> TemporalEvolutionDescriptors:
    """Compute the frozen six-feature primary evolution representation.

    Parameters
    ----------
    eeg_volts:
        Complete physical standard-19 EEG with exact shape ``[B,19,12000]``
        and a fixed sampling rate of 200 Hz.  Values are interpreted as volts.
    The upstream causal EDF gate accepts only complete physical standard-19
    events. There is no tile-level QC/missingness API on the primary path;
    every returned event mask is the frozen all-true ``[19,15]`` tensor.
    """

    eeg, valid = _validate_eeg(eeg_volts)
    with torch.no_grad():
        work = eeg.to(dtype=torch.float64) * EVOLUTION_UV_PER_VOLT
        tiles_uv = work.reshape(
            eeg.shape[0],
            N_STANDARD_CHANNELS,
            N_TIME_TILES,
            EVOLUTION_TILE_SAMPLES,
        )
        local, welch_fft = _local_descriptors(tiles_uv)
        coherence, output_mask = _mean_neighbor_coherence(welch_fft, valid)
        descriptors = torch.cat((local, coherence[..., None]), dim=-1)
        descriptors = descriptors.to(dtype=torch.float64, device="cpu")
        descriptors = torch.where(
            output_mask[..., None], descriptors, torch.zeros_like(descriptors)
        )
        if not torch.isfinite(descriptors).all():
            raise ValueError(
                "Evolution descriptor computation produced non-finite values; "
                "check source amplitude/unit QC"
            )
        if not output_mask.all().item():
            raise RuntimeError("Complete19 descriptor mask must remain all-true")
    return TemporalEvolutionDescriptors(
        descriptors=descriptors.detach(),
        mask=output_mask.detach(),
        receipt=EvolutionDescriptorReceipt(),
    )


def _normalize_patient_id(value: object) -> str:
    patient_id = str(value).strip()
    if not patient_id or patient_id.lower() in {"nan", "none", "null"}:
        raise ValueError("Patient ID is missing")
    return patient_id


def patient_roster_sha256(patient_ids: Sequence[object]) -> str:
    normalized = tuple(sorted(_normalize_patient_id(value) for value in patient_ids))
    if len(set(normalized)) != len(normalized):
        raise ValueError("Patient roster contains duplicate IDs")
    return _sha256_json(normalized)


@dataclass(frozen=True)
class PatientBalancedScalerReceipt:
    """Lineage and statistics for a source-train-only robust scaler."""

    feature_names: tuple[str, ...]
    feature_schema_sha256: str
    patient_roster_sha256: str
    split_manifest_sha256: str
    fit_split_sha256: str
    fit_split: str
    patient_count: int
    patient_feature_medians_sha256: str
    center: tuple[float, ...]
    iqr: tuple[float, ...]
    scale: tuple[float, ...]
    clip: float
    statistic_scope: str = "global_feature_wise_non_channel_wise"
    patient_balance_policy: str = (
        "median_within_patient_then_median_iqr_across_patients"
    )
    zero_iqr_policy: str = "unit_scale_when_iqr_below_1e-6"
    schema_version: str = PATIENT_ROBUST_SCALER_SCHEMA

    def __post_init__(self) -> None:
        if self.feature_names != EVOLUTION_FEATURES:
            raise ValueError("Scaler feature order must match EVOLUTION_FEATURES")
        if self.feature_schema_sha256 != EVOLUTION_FEATURE_SCHEMA_SHA256:
            raise ValueError("Scaler feature schema hash mismatch")
        for field in (
            "patient_roster_sha256",
            "split_manifest_sha256",
            "fit_split_sha256",
            "patient_feature_medians_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.fit_split != "source_train":
            raise ValueError("Robust scaler receipt must be fitted on source_train")
        expected_split_hash = _sha256_json(
            {
                "fit_split": self.fit_split,
                "split_manifest_sha256": self.split_manifest_sha256,
            }
        )
        if self.fit_split_sha256 != expected_split_hash:
            raise ValueError("Scaler fit split hash mismatch")
        if self.patient_count < 1:
            raise ValueError("Scaler receipt requires at least one patient")
        for name, values in (
            ("center", self.center),
            ("iqr", self.iqr),
            ("scale", self.scale),
        ):
            if len(values) != N_NODE_FEATURES or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError(f"Scaler {name} must contain six finite values")
        if any(value < 0 for value in self.iqr) or any(
            value <= 0 for value in self.scale
        ):
            raise ValueError("Scaler IQR must be non-negative and scale positive")
        if not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("Scaler clip must be finite and positive")
        if self.statistic_scope != "global_feature_wise_non_channel_wise":
            raise ValueError("Channel-wise evolution scaling is forbidden")
        if self.schema_version != PATIENT_ROBUST_SCALER_SCHEMA:
            raise ValueError("Unsupported robust scaler schema")


@dataclass(frozen=True)
class PatientBalancedRobustScaler:
    """Six global robust statistics fitted with equal patient weight."""

    center: torch.Tensor
    iqr: torch.Tensor
    scale: torch.Tensor
    clip: float
    receipt: PatientBalancedScalerReceipt

    def __post_init__(self) -> None:
        for name, value in (
            ("center", self.center),
            ("iqr", self.iqr),
            ("scale", self.scale),
        ):
            if tuple(value.shape) != (N_NODE_FEATURES,):
                raise ValueError(f"Scaler {name} must have shape [6]")
            if value.dtype != torch.float64 or value.requires_grad:
                raise TypeError(f"Scaler {name} must be detached CPU float64")
            if value.device.type != "cpu":
                raise TypeError(f"Scaler {name} must remain on CPU")
            if not torch.isfinite(value).all():
                raise ValueError(f"Scaler {name} must be finite")
        if torch.any(self.iqr < 0) or torch.any(self.scale <= 0):
            raise ValueError("Scaler requires non-negative IQR and positive scale")
        if self.clip != self.receipt.clip:
            raise ValueError("Scaler clip disagrees with receipt")
        for tensor, recorded, name in (
            (self.center, self.receipt.center, "center"),
            (self.iqr, self.receipt.iqr, "iqr"),
            (self.scale, self.receipt.scale, "scale"),
        ):
            expected = torch.tensor(recorded, dtype=tensor.dtype, device=tensor.device)
            if not torch.equal(tensor, expected):
                raise ValueError(f"Scaler {name} disagrees with receipt")

    def transform(
        self, descriptors: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply global feature-wise scaling to complete19 descriptors."""

        if not isinstance(descriptors, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("descriptors and mask must be torch tensors")
        if descriptors.ndim < 2 or descriptors.shape[-1] != N_NODE_FEATURES:
            raise ValueError("Descriptors must end in the six frozen features")
        if tuple(mask.shape) != tuple(descriptors.shape[:-1]):
            raise ValueError("Scaler mask must match descriptor leading dimensions")
        if descriptors.dtype != torch.float64 or mask.dtype != torch.bool:
            raise TypeError("Descriptors must be CPU float64 and mask must be bool")
        if descriptors.device.type != "cpu" or mask.device.type != "cpu":
            raise TypeError("Primary evolution scaling is CPU-only")
        if not torch.isfinite(descriptors).all():
            raise ValueError("Descriptors must be finite before scaling")
        if not mask.all().item():
            raise ValueError("Complete19 evolution scaling forbids tile masks")
        center = self.center.to(device=descriptors.device, dtype=descriptors.dtype)
        scale = self.scale.to(device=descriptors.device, dtype=descriptors.dtype)
        with torch.no_grad():
            transformed = ((descriptors.detach() - center) / scale).clamp(
                -self.clip, self.clip
            )
            transformed = torch.where(
                mask.detach()[..., None], transformed, torch.zeros_like(transformed)
            )
        if not torch.isfinite(transformed).all():
            raise RuntimeError("Robust scaling produced non-finite values")
        return transformed.detach()


def _validate_patient_tensor(
    descriptors: torch.Tensor,
    mask: torch.Tensor,
    *,
    patient_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(descriptors, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError(f"Patient {patient_id}: descriptors and mask must be tensors")
    if descriptors.ndim not in {3, 4} or tuple(descriptors.shape[-3:]) != (
        N_STANDARD_CHANNELS,
        N_TIME_TILES,
        N_NODE_FEATURES,
    ):
        raise ValueError(
            f"Patient {patient_id}: descriptors must have shape [19,15,6] "
            "or [E,19,15,6]"
        )
    if tuple(mask.shape) != tuple(descriptors.shape[:-1]):
        raise ValueError(f"Patient {patient_id}: mask shape does not match descriptors")
    if descriptors.dtype != torch.float64 or mask.dtype != torch.bool:
        raise TypeError(
            f"Patient {patient_id}: descriptors must be CPU float64 and mask bool"
        )
    if descriptors.device.type != "cpu" or mask.device.type != "cpu":
        raise TypeError(f"Patient {patient_id}: scaler fit is CPU-only")
    if descriptors.requires_grad or mask.requires_grad:
        raise ValueError(f"Patient {patient_id}: scaler fit inputs must be detached")
    if not torch.isfinite(descriptors).all():
        raise ValueError(f"Patient {patient_id}: descriptors must be finite")
    if not mask.any():
        raise ValueError(f"Patient {patient_id}: no valid descriptor observations")
    if not mask.all().item():
        raise ValueError(
            f"Patient {patient_id}: complete19 scaler fit forbids tile masks"
        )
    return descriptors.detach().to(dtype=torch.float64, device="cpu"), mask.detach().cpu()


def fit_patient_balanced_robust_scaler(
    patient_descriptors: Sequence[torch.Tensor],
    patient_masks: Sequence[torch.Tensor],
    patient_ids: Sequence[object],
    *,
    expected_patient_ids: Sequence[object],
    fit_split: str,
    split_manifest_sha256: str,
    clip: float = DEFAULT_SCALER_CLIP,
) -> PatientBalancedRobustScaler:
    """Fit six global statistics from one unique entry per train patient.

    Each entry may contain one event ``[19,15,6]`` or a complete event bag
    ``[E,19,15,6]``.  Patient IDs must be unique, and the observed roster must
    exactly match ``expected_patient_ids``; this prevents silent omission,
    duplication, or addition of dev/eval patients.
    """

    if fit_split != "source_train":
        raise ValueError("Evolution scaler may be fitted only on source_train")
    split_hash = _require_sha256(
        split_manifest_sha256, field="split_manifest_sha256"
    )
    if not math.isfinite(float(clip)) or float(clip) <= 0:
        raise ValueError("clip must be finite and positive")
    if not (
        len(patient_descriptors) == len(patient_masks) == len(patient_ids)
    ):
        raise ValueError("Scaler patient descriptors, masks, and IDs must align")
    if not patient_ids:
        raise ValueError("Scaler fit patient roster cannot be empty")

    normalized_ids = tuple(_normalize_patient_id(value) for value in patient_ids)
    expected_ids = tuple(
        sorted(_normalize_patient_id(value) for value in expected_patient_ids)
    )
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Scaler fit contains duplicate patient entries")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected scaler roster contains duplicate patient IDs")
    observed_set = set(normalized_ids)
    expected_set = set(expected_ids)
    if observed_set != expected_set:
        raise ValueError(
            "Scaler fit roster mismatch; "
            f"missing={sorted(expected_set - observed_set)}, "
            f"extra={sorted(observed_set - expected_set)}"
        )

    medians_by_patient: list[torch.Tensor] = []
    for patient_id, descriptors, mask in zip(
        normalized_ids, patient_descriptors, patient_masks
    ):
        values, valid = _validate_patient_tensor(
            descriptors, mask, patient_id=patient_id
        )
        feature_medians = torch.stack(
            [
                torch.quantile(values[..., feature][valid], 0.5)
                for feature in range(N_NODE_FEATURES)
            ]
        )
        if not torch.isfinite(feature_medians).all():
            raise ValueError(f"Patient {patient_id}: non-finite feature median")
        medians_by_patient.append(feature_medians)

    patient_matrix = torch.stack(medians_by_patient, dim=0)
    center = torch.quantile(patient_matrix, 0.5, dim=0)
    q25 = torch.quantile(patient_matrix, 0.25, dim=0)
    q75 = torch.quantile(patient_matrix, 0.75, dim=0)
    iqr = (q75 - q25).clamp_min(0.0)
    scale = torch.where(
        iqr >= SCALER_IQR_FLOOR, iqr, torch.ones_like(iqr)
    )
    center64 = center.to(dtype=torch.float64, device="cpu").detach()
    iqr64 = iqr.to(dtype=torch.float64, device="cpu").detach()
    scale64 = scale.to(dtype=torch.float64, device="cpu").detach()

    roster_hash = patient_roster_sha256(expected_ids)
    fit_split_hash = _sha256_json(
        {"fit_split": fit_split, "split_manifest_sha256": split_hash}
    )
    patient_median_payload = tuple(
        (patient_id, tuple(float(value) for value in medians.tolist()))
        for patient_id, medians in sorted(
            zip(normalized_ids, medians_by_patient), key=lambda item: item[0]
        )
    )
    receipt = PatientBalancedScalerReceipt(
        feature_names=EVOLUTION_FEATURES,
        feature_schema_sha256=EVOLUTION_FEATURE_SCHEMA_SHA256,
        patient_roster_sha256=roster_hash,
        split_manifest_sha256=split_hash,
        fit_split_sha256=fit_split_hash,
        fit_split=fit_split,
        patient_count=len(normalized_ids),
        patient_feature_medians_sha256=_sha256_json(patient_median_payload),
        center=tuple(float(value) for value in center64.tolist()),
        iqr=tuple(float(value) for value in iqr64.tolist()),
        scale=tuple(float(value) for value in scale64.tolist()),
        clip=float(clip),
    )
    return PatientBalancedRobustScaler(
        center=center64,
        iqr=iqr64,
        scale=scale64,
        clip=float(clip),
        receipt=receipt,
    )


if EVOLUTION_N_TILES != N_TIME_TILES:
    raise RuntimeError("Evolution time grid disagrees with canonical geometry")
if EVOLUTION_N_TILES * EVOLUTION_TILE_SAMPLES != EVOLUTION_INPUT_SAMPLES:
    raise RuntimeError("Evolution tile grid does not cover exactly 60 seconds")
if len(EVOLUTION_FEATURES) != N_NODE_FEATURES:
    raise RuntimeError("Evolution feature dimension disagrees with geometry")
if not STANDARD19_NEIGHBORS[CHANNEL_INDEX["FZ"]]:
    raise RuntimeError("FZ must occur in the frozen coherence graph")
if not STANDARD19_NEIGHBORS[CHANNEL_INDEX["PZ"]]:
    raise RuntimeError("PZ must occur in the frozen coherence graph")


__all__ = [
    "COMPLETE19_DESCRIPTOR_MASK_SHA256",
    "DEFAULT_SCALER_CLIP",
    "EVOLUTION_DESCRIPTOR_SCHEMA",
    "EVOLUTION_FEATURE_SCHEMA_SHA256",
    "EvolutionDescriptorReceipt",
    "PatientBalancedRobustScaler",
    "PatientBalancedScalerReceipt",
    "STANDARD19_NEIGHBOR_EDGES",
    "STANDARD19_NEIGHBORS",
    "TemporalEvolutionDescriptors",
    "compute_temporal_evolution_descriptors",
    "fit_patient_balanced_robust_scaler",
    "patient_roster_sha256",
]
