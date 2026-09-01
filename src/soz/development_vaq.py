"""Target-free development V and A/Q evidence production.

This module materializes two deliberately narrow evidence families from the
already verified DeepSOZ--TUSZ *signal* timeline:

* ``V`` is the frozen direct temporal-evolution representation: six named
  observable descriptors on standard-19 EEG in fifteen ordered four-second
  tiles.  Temporal differences are audit descriptors only; no tensor is named
  or supervised as seizure origin, propagation, or SOZ.
* ``A/Q`` is a deterministic gross-signal-quality sentinel.  It can only
  produce a multiplier in ``[0, 1]`` or recommend abstention.  Its diagnostic
  values are reporting-only and are explicitly forbidden as localizing model
  inputs.

The producer is hard limited to ``source_dev``.  It reads neither a DeepSOZ
electrode vector nor a TUSZ channel target/mask, and it rejects source-eval and
private data by construction.  Every accepted EDF window is replayed and
matched to the externally pinned signal-preflight receipt before evidence is
published.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from . import evolution as _evolution_module
from . import geometry as _geometry_module
from .data.edf import CausalEDFConfig, load_standard19_edf_event
from .evolution import (
    EVOLUTION_FEATURE_SCHEMA_SHA256,
    EVOLUTION_FEATURES,
    EVOLUTION_INPUT_SAMPLES,
    EVOLUTION_N_TILES,
    EVOLUTION_SAMPLE_RATE_HZ,
    EVOLUTION_TILE_SAMPLES,
    compute_temporal_evolution_descriptors,
)
from .evolution_io import (
    ComputedEvolutionScalerArtifact,
    load_externally_pinned_computed_evolution_scaler_artifact,
)
from .geometry import N_STANDARD_CHANNELS, STANDARD_19
from .ictal_native_eval import load_bound_deepsoz_signal_preflight_artifact
from .temporal_masks import OffsetAwarePhaseMasks, build_offset_aware_phase_masks


DEVELOPMENT_VAQ_SCHEMA = "soz_target_free_development_vaq_v1"
DEVELOPMENT_VAQ_MANIFEST_SCHEMA = "soz_target_free_development_vaq_manifest_v1"
DEVELOPMENT_VAQ_EVENT_SCHEMA = "soz_target_free_development_vaq_event_v1"
DEVELOPMENT_VAQ_PURPOSE = "source_dev_model_selection_and_diagnostics_only"
DEVELOPMENT_VAQ_SPLIT = "source_dev"
DEVELOPMENT_VAQ_SERIALIZATION = "canonical_json_plus_safetensors_no_pickle"
DEVELOPMENT_VAQ_TENSOR_FILENAME = "vaq.safetensors"
DEVELOPMENT_VAQ_EVENTS_FILENAME = "events.json"
DEVELOPMENT_VAQ_MANIFEST_FILENAME = "manifest.json"

QUALITY_POLICY_SCHEMA = "target_free_conservative_gross_signal_quality_v1"
QUALITY_DIAGNOSTIC_NAMES = (
    "peak_abs_uv",
    "peak_to_peak_uv",
    "rms_uv",
    "max_abs_step_uv_per_sample",
    "near_flat_step_fraction",
    "relative_power_30_45_hz",
)
QUALITY_BURDEN_COMPONENT_NAMES = (
    "gross_amplitude_burden",
    "gross_step_burden",
    "near_flat_burden",
)

# These are conservative engineering sentinels, not learned clinical cutoffs.
# Their only effect is attenuation/abstention; they never add localization
# evidence.  Ramps avoid a brittle discontinuity while retaining explicit
# target-free bounds.
QUALITY_AMPLITUDE_RAMP_UV = (500.0, 2_000.0)
QUALITY_STEP_RAMP_UV = (200.0, 1_000.0)
QUALITY_FLAT_STEP_TOLERANCE_UV = 0.02
QUALITY_FLAT_FRACTION_RAMP = (0.98, 0.999)
QUALITY_RELIABLE_THRESHOLD = 0.5
QUALITY_MIN_RELIABLE_CHANNEL_FRACTION = 0.8

_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_EVENTS_BYTES = 64 * 1024 * 1024
_MAX_TENSOR_BYTES = 512 * 1024 * 1024
_SHA256_HEX = frozenset("0123456789abcdef")

_PHASE_TENSOR_NAMES = (
    "ictal_phase_mask",
    "pre_anchor_context_mask",
    "pre_previous_seizure_overlap_mask",
    "pre_unknown_context_mask",
    "within_trusted_ictal_mask",
    "transition_mask",
    "postictal_mask",
    "unknown_offset_mask",
)
_TENSOR_NAMES = (
    "evolution_raw",
    "evolution_scaled",
    "evolution_ordered_delta",
    "evolution_mask",
    "evolution_delta_mask",
    "quality_diagnostics",
    "quality_burden_components",
    "artifact_burden",
    "reliability",
    "reliable_mask",
    "tile_abstain",
    "event_abstain",
    "tile_start_sec",
    *_PHASE_TENSOR_NAMES,
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("V+A/Q artifact contains non-canonical JSON data") from exc
    return encoded.encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_sha256(module: object, *, field: str) -> str:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"{field} has no auditable source path")
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must be a regular non-symlinked file")
    return _file_sha256(path)


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in _SHA256_HEX for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    metadata = _canonical_json_bytes(
        {"dtype": str(value.dtype), "shape": list(value.shape)}
    )
    raw = value.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _signal_window_sha256(tensor: torch.Tensor) -> str:
    """Match the verified DeepSOZ signal-preflight window digest exactly."""

    return _tensor_sha256(tensor)


def _ramp(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    if not 0 <= lower < upper:
        raise ValueError("Quality ramp bounds are invalid")
    return ((value - lower) / (upper - lower)).clamp(0.0, 1.0)


@dataclass(frozen=True)
class TargetFreeQualityEvidence:
    """Detached target-free quality diagnostics and one-way reliability gate."""

    diagnostics: torch.Tensor
    burden_components: torch.Tensor
    artifact_burden: torch.Tensor
    reliability: torch.Tensor
    reliable_mask: torch.Tensor
    tile_abstain: torch.Tensor

    def __post_init__(self) -> None:
        if self.diagnostics.ndim != 4:
            raise ValueError("quality diagnostics must have shape [B,19,15,6]")
        batch = self.diagnostics.shape[0]
        tile_shape = (batch, N_STANDARD_CHANNELS, EVOLUTION_N_TILES)
        if tuple(self.diagnostics.shape) != (*tile_shape, len(QUALITY_DIAGNOSTIC_NAMES)):
            raise ValueError("quality diagnostics shape drifted")
        if tuple(self.burden_components.shape) != (
            *tile_shape,
            len(QUALITY_BURDEN_COMPONENT_NAMES),
        ):
            raise ValueError("quality burden-component shape drifted")
        for name, value in (
            ("artifact_burden", self.artifact_burden),
            ("reliability", self.reliability),
        ):
            if tuple(value.shape) != tile_shape:
                raise ValueError(f"{name} shape drifted")
            if value.dtype != torch.float64 or value.device.type != "cpu":
                raise TypeError(f"{name} must be detached CPU float64")
            if value.requires_grad or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite and detached")
            if torch.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"{name} must lie in [0,1]")
        for name, value in (
            ("diagnostics", self.diagnostics),
            ("burden_components", self.burden_components),
        ):
            if value.dtype != torch.float64 or value.device.type != "cpu":
                raise TypeError(f"{name} must be detached CPU float64")
            if value.requires_grad or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite and detached")
        if torch.any((self.burden_components < 0) | (self.burden_components > 1)):
            raise ValueError("quality burdens must lie in [0,1]")
        if not torch.equal(self.artifact_burden, self.burden_components.amax(dim=-1)):
            raise ValueError("artifact burden must be the maximum component burden")
        if not torch.equal(self.reliability, 1.0 - self.artifact_burden):
            raise ValueError("quality reliability must equal one minus burden")
        if self.reliable_mask.dtype != torch.bool or tuple(
            self.reliable_mask.shape
        ) != tile_shape:
            raise TypeError("reliable_mask must be bool [B,19,15]")
        if not torch.equal(
            self.reliable_mask,
            self.reliability >= QUALITY_RELIABLE_THRESHOLD,
        ):
            raise ValueError("reliable_mask disagrees with the frozen threshold")
        if self.tile_abstain.dtype != torch.bool or tuple(self.tile_abstain.shape) != (
            batch,
            EVOLUTION_N_TILES,
        ):
            raise TypeError("tile_abstain must be bool [B,15]")
        coverage = self.reliable_mask.to(torch.float64).mean(dim=1)
        expected_abstain = coverage < QUALITY_MIN_RELIABLE_CHANNEL_FRACTION
        if not torch.equal(self.tile_abstain, expected_abstain):
            raise ValueError("tile abstention disagrees with quality coverage")


def compute_target_free_quality_evidence(
    eeg_volts: torch.Tensor,
) -> TargetFreeQualityEvidence:
    """Compute conservative signal-quality evidence without labels or fitting.

    The 30--45 Hz ratio is retained only as a diagnostic.  It is intentionally
    excluded from ``burden_components`` because true ictal fast activity and
    muscle artifact cannot be safely separated here without supervision.
    """

    if not isinstance(eeg_volts, torch.Tensor):
        raise TypeError("eeg_volts must be a torch.Tensor")
    if eeg_volts.ndim != 3 or tuple(eeg_volts.shape[1:]) != (
        N_STANDARD_CHANNELS,
        EVOLUTION_INPUT_SAMPLES,
    ):
        raise ValueError("A/Q input must have shape [B,19,12000]")
    if eeg_volts.shape[0] < 1 or not eeg_volts.is_floating_point():
        raise ValueError("A/Q input must be a non-empty floating-point batch")
    if eeg_volts.device.type != "cpu":
        raise ValueError("A/Q computation is CPU-only")
    if not torch.isfinite(eeg_volts).all():
        raise ValueError("A/Q input must be finite")

    with torch.no_grad():
        tiles = (
            eeg_volts.detach().to(torch.float64)
            .mul(1_000_000.0)
            .reshape(
                eeg_volts.shape[0],
                N_STANDARD_CHANNELS,
                EVOLUTION_N_TILES,
                EVOLUTION_TILE_SAMPLES,
            )
        )
        peak_abs = tiles.abs().amax(dim=-1)
        peak_to_peak = tiles.amax(dim=-1) - tiles.amin(dim=-1)
        rms = torch.sqrt(tiles.square().mean(dim=-1))
        steps = torch.diff(tiles, dim=-1).abs()
        max_step = steps.amax(dim=-1)
        flat_fraction = (steps <= QUALITY_FLAT_STEP_TOLERANCE_UV).to(
            torch.float64
        ).mean(dim=-1)

        demeaned = tiles - tiles.mean(dim=-1, keepdim=True)
        window = torch.hann_window(
            EVOLUTION_TILE_SAMPLES, periodic=True, dtype=torch.float64
        )
        power = torch.fft.rfft(demeaned * window, dim=-1).abs().square()
        frequencies = torch.fft.rfftfreq(
            EVOLUTION_TILE_SAMPLES,
            d=1.0 / EVOLUTION_SAMPLE_RATE_HZ,
        ).to(torch.float64)
        band_1_45 = (frequencies >= 1.0) & (frequencies <= 45.0)
        band_30_45 = (frequencies >= 30.0) & (frequencies <= 45.0)
        relative_high = power[..., band_30_45].sum(dim=-1) / power[
            ..., band_1_45
        ].sum(dim=-1).clamp_min(1e-24)

        diagnostics = torch.stack(
            (
                peak_abs,
                peak_to_peak,
                rms,
                max_step,
                flat_fraction,
                relative_high,
            ),
            dim=-1,
        )
        burdens = torch.stack(
            (
                _ramp(peak_abs, *QUALITY_AMPLITUDE_RAMP_UV),
                _ramp(max_step, *QUALITY_STEP_RAMP_UV),
                _ramp(flat_fraction, *QUALITY_FLAT_FRACTION_RAMP),
            ),
            dim=-1,
        )
        artifact_burden = burdens.amax(dim=-1)
        reliability = 1.0 - artifact_burden
        reliable_mask = reliability >= QUALITY_RELIABLE_THRESHOLD
        tile_abstain = (
            reliable_mask.to(torch.float64).mean(dim=1)
            < QUALITY_MIN_RELIABLE_CHANNEL_FRACTION
        )
    return TargetFreeQualityEvidence(
        diagnostics=diagnostics.detach().cpu(),
        burden_components=burdens.detach().cpu(),
        artifact_burden=artifact_burden.detach().cpu(),
        reliability=reliability.detach().cpu(),
        reliable_mask=reliable_mask.detach().cpu(),
        tile_abstain=tile_abstain.detach().cpu(),
    )


def attenuate_nonnegative_support(
    support: torch.Tensor, reliability: torch.Tensor
) -> torch.Tensor:
    """Apply A/Q as a one-way gate to non-negative localizing support only."""

    if not isinstance(support, torch.Tensor) or not isinstance(
        reliability, torch.Tensor
    ):
        raise TypeError("support and reliability must be tensors")
    if support.shape != reliability.shape:
        raise ValueError("support and reliability must have identical shape")
    if not torch.isfinite(support).all() or torch.any(support < 0):
        raise ValueError("A/Q may gate only finite non-negative support")
    if not torch.isfinite(reliability).all() or torch.any(
        (reliability < 0) | (reliability > 1)
    ):
        raise ValueError("reliability must lie in [0,1]")
    gated = support * reliability
    if torch.any(gated > support):
        raise RuntimeError("A/Q reliability increased localizing support")
    return gated


def _safe_source_file(root: Path, relative_value: object) -> tuple[str, Path]:
    if not isinstance(relative_value, str) or not relative_value or "\\" in relative_value:
        raise ValueError("relative_edf_path must be a canonical POSIX path")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("relative_edf_path is not canonical")
    root_absolute = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(root_absolute.joinpath(*relative.parts)))
    try:
        candidate.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError("relative_edf_path escapes TUSZ root") from exc
    for component in (candidate, *candidate.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("TUSZ path cannot contain symlink components")
        if component == root_absolute:
            break
    if not candidate.is_file():
        raise FileNotFoundError(f"Required source-dev EDF is unavailable: {candidate}")
    return relative.as_posix(), candidate


@dataclass(frozen=True)
class _SignalTimelineEvent:
    event_id: str
    patient_id: str
    event_record_sha256: str
    relative_edf_path: str
    edf_sha256: str
    global_t0_sec: float
    global_stop_sec: float
    seizure_duration_sec: float
    previous_seizure_gap_sec: float | None
    timeline_sha256: str
    preprocess_config_sha256: str
    edf_receipt_sha256: str
    signal_receipt_sha256: str
    processed_window_sha256: str


def _signal_only_source_dev_timeline(
    signal_receipt: Mapping[str, object],
    *,
    signal_artifact_sha256: str,
    signal_receipt_sha256: str,
) -> tuple[tuple[_SignalTimelineEvent, ...], OffsetAwarePhaseMasks]:
    """Derive record-local timing solely from the verified signal receipt."""

    accepted = tuple(signal_receipt["events"])
    excluded = tuple(signal_receipt["exclusions"])
    all_rows = (*accepted, *excluded)
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in all_rows:
        groups.setdefault(str(row["deepsoz_source_record_sha256"]), []).append(row)

    previous_stop: dict[str, float | None] = {}
    timeline_sha: dict[str, str] = {}
    for source_record_sha, group in groups.items():
        ordered = tuple(sorted(group, key=lambda row: int(row["global_event_index"])))
        indices = tuple(int(row["global_event_index"]) for row in ordered)
        if indices != tuple(range(len(ordered))):
            raise ValueError("Signal receipt does not contain a complete record timeline")
        starts = tuple(float(row["global_t0_sec"]) for row in ordered)
        if any(right < left - 1e-6 for left, right in zip(starts, starts[1:])):
            raise ValueError("Signal receipt record timeline is not chronological")
        payload = {
            "schema_version": "target_free_record_local_timeline_v1",
            "signal_preflight_artifact_sha256": signal_artifact_sha256,
            "signal_preflight_receipt_sha256": signal_receipt_sha256,
            "deepsoz_source_record_sha256": source_record_sha,
            "events": [
                {
                    "event_id": str(row["event_id"]),
                    "event_record_sha256": str(row["event_record_sha256"]),
                    "global_event_index": int(row["global_event_index"]),
                    "global_t0_sec": float(row["global_t0_sec"]),
                    "global_stop_sec": float(row["global_stop_sec"]),
                }
                for row in ordered
            ],
        }
        group_sha = _canonical_sha256(payload)
        running_max: float | None = None
        for row in ordered:
            event_id = str(row["event_id"])
            previous_stop[event_id] = running_max
            timeline_sha[event_id] = group_sha
            stop = float(row["global_stop_sec"])
            running_max = stop if running_max is None else max(running_max, stop)

    selected: list[_SignalTimelineEvent] = []
    for row in sorted(accepted, key=lambda value: str(value["event_id"])):
        if str(row["model_split"]) != DEVELOPMENT_VAQ_SPLIT:
            continue
        event_id = str(row["event_id"])
        start = float(row["global_t0_sec"])
        stop = float(row["global_stop_sec"])
        if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
            raise ValueError("Source-dev signal event has invalid seizure timing")
        prior = previous_stop[event_id]
        selected.append(
            _SignalTimelineEvent(
                event_id=event_id,
                patient_id=str(row["patient_id"]),
                event_record_sha256=_require_sha256(
                    row["event_record_sha256"], field="event_record_sha256"
                ),
                relative_edf_path=str(row["relative_edf_path"]),
                edf_sha256=_require_sha256(row["edf_sha256"], field="edf_sha256"),
                global_t0_sec=start,
                global_stop_sec=stop,
                seizure_duration_sec=stop - start,
                previous_seizure_gap_sec=(
                    None if prior is None else max(0.0, start - prior)
                ),
                timeline_sha256=timeline_sha[event_id],
                preprocess_config_sha256=_require_sha256(
                    row["preprocess_config_sha256"],
                    field="preprocess_config_sha256",
                ),
                edf_receipt_sha256=_require_sha256(
                    row["edf_receipt_sha256"], field="edf_receipt_sha256"
                ),
                signal_receipt_sha256=_require_sha256(
                    row["signal_receipt_sha256"], field="signal_receipt_sha256"
                ),
                processed_window_sha256=_require_sha256(
                    row["processed_window_sha256"],
                    field="processed_window_sha256",
                ),
            )
        )
    if not selected:
        raise ValueError("Verified signal receipt contains no source-dev events")
    if len({event.event_id for event in selected}) != len(selected):
        raise ValueError("Source-dev timeline contains duplicate event IDs")

    phase_masks = build_offset_aware_phase_masks(
        [event.seizure_duration_sec for event in selected],
        offset_trustworthy=[True] * len(selected),
        previous_seizure_gap_sec=[
            event.previous_seizure_gap_sec for event in selected
        ],
        previous_timeline_trustworthy=[True] * len(selected),
    )
    return tuple(selected), phase_masks


def _validate_scaler_lineage(
    scaler_artifact: ComputedEvolutionScalerArtifact,
    signal_receipt: Mapping[str, object],
) -> None:
    if scaler_artifact.receipt.oof_fold is not None:
        raise ValueError("source-dev V must use the final source-train scaler")
    if scaler_artifact.scaler.receipt.fit_split != "source_train":
        raise ValueError("V scaler was not fitted on source_train")
    if (
        scaler_artifact.scaler.receipt.split_manifest_sha256
        != signal_receipt["split_manifest_sha256"]
    ):
        raise ValueError("Signal preflight and V scaler use different split manifests")
    evolution_sha = _module_sha256(_evolution_module, field="evolution.py")
    geometry_sha = _module_sha256(_geometry_module, field="geometry.py")
    if evolution_sha != scaler_artifact.computation_receipt.evolution_source_sha256:
        raise ValueError("Current V implementation differs from the verified scaler")
    if geometry_sha != scaler_artifact.computation_receipt.geometry_source_sha256:
        raise ValueError("Current standard-19 geometry differs from the verified scaler")


def preflight_development_vaq_inputs(
    *,
    signal_preflight_bundle: str | Path,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
    evolution_scaler_bundle: str | Path,
    expected_evolution_scaler_artifact_sha256: str,
    tusz_root: str | Path,
) -> dict[str, object]:
    """Validate the target-free source-dev boundary without writing outputs."""

    signal = load_bound_deepsoz_signal_preflight_artifact(
        signal_preflight_bundle,
        expected_artifact_sha256=expected_signal_artifact_sha256,
        expected_receipt_sha256=expected_signal_receipt_sha256,
    )
    scaler = load_externally_pinned_computed_evolution_scaler_artifact(
        evolution_scaler_bundle,
        oof_fold=None,
        expected_artifact_sha256=expected_evolution_scaler_artifact_sha256,
    )
    _validate_scaler_lineage(scaler, signal.receipt)
    timeline, _ = _signal_only_source_dev_timeline(
        signal.receipt,
        signal_artifact_sha256=signal.artifact_sha256,
        signal_receipt_sha256=signal.receipt_sha256,
    )
    root = Path(tusz_root)
    for event in timeline:
        _safe_source_file(root, event.relative_edf_path)
    patients = tuple(sorted({event.patient_id for event in timeline}))
    return {
        "schema_version": DEVELOPMENT_VAQ_SCHEMA,
        "status": "ready",
        "purpose": DEVELOPMENT_VAQ_PURPOSE,
        "model_split": DEVELOPMENT_VAQ_SPLIT,
        "event_count": len(timeline),
        "patient_count": len(patients),
        "event_roster_sha256": _canonical_sha256(
            tuple(event.event_id for event in timeline)
        ),
        "patient_roster_sha256": _canonical_sha256(patients),
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "source_eval_allowed": False,
        "private_allowed": False,
        "training_authorized": False,
    }


def _phase_tensors(phase_masks: OffsetAwarePhaseMasks) -> dict[str, torch.Tensor]:
    return {
        name: getattr(phase_masks, name).detach().cpu().contiguous()
        for name in _PHASE_TENSOR_NAMES
    }


def _tensor_specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, object]:
    if set(tensors) != set(_TENSOR_NAMES):
        raise ValueError("V+A/Q tensors do not match the closed schema")
    return {
        name: {
            "shape": list(tensors[name].shape),
            "dtype": str(tensors[name].dtype).removeprefix("torch."),
        }
        for name in _TENSOR_NAMES
    }


def _validate_tensor_payload(tensors: Mapping[str, torch.Tensor]) -> None:
    if set(tensors) != set(_TENSOR_NAMES):
        raise ValueError("V+A/Q tensor payload violates its closed schema")
    event_count = tensors["evolution_raw"].shape[0]
    expected_v = (event_count, N_STANDARD_CHANNELS, EVOLUTION_N_TILES, 6)
    expected_tile = (event_count, N_STANDARD_CHANNELS, EVOLUTION_N_TILES)
    float64_names = {
        "evolution_raw",
        "evolution_scaled",
        "evolution_ordered_delta",
        "quality_diagnostics",
        "quality_burden_components",
        "artifact_burden",
        "reliability",
        "tile_start_sec",
    }
    for name, value in tensors.items():
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError(f"{name} must be detached CPU evidence")
        expected_dtype = torch.float64 if name in float64_names else torch.bool
        if value.dtype != expected_dtype:
            raise TypeError(f"{name} must use {expected_dtype}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    for name in ("evolution_raw", "evolution_scaled", "evolution_ordered_delta"):
        if tuple(tensors[name].shape) != expected_v:
            raise ValueError(f"{name} must have shape [E,19,15,6]")
    if tuple(tensors["quality_diagnostics"].shape) != (
        *expected_tile,
        len(QUALITY_DIAGNOSTIC_NAMES),
    ):
        raise ValueError("quality diagnostics shape drifted")
    if tuple(tensors["quality_burden_components"].shape) != (
        *expected_tile,
        len(QUALITY_BURDEN_COMPONENT_NAMES),
    ):
        raise ValueError("quality burden shape drifted")
    for name in (
        "evolution_mask",
        "evolution_delta_mask",
        "artifact_burden",
        "reliability",
        "reliable_mask",
    ):
        if tuple(tensors[name].shape) != expected_tile:
            raise ValueError(f"{name} must have shape [E,19,15]")
    for name in (*_PHASE_TENSOR_NAMES, "tile_abstain"):
        if tuple(tensors[name].shape) != (event_count, EVOLUTION_N_TILES):
            raise ValueError(f"{name} must have shape [E,15]")
    if tuple(tensors["event_abstain"].shape) != (event_count,):
        raise ValueError("event_abstain must have shape [E]")
    expected_grid = torch.arange(-12.0, 48.0, 4.0, dtype=torch.float64)
    if not torch.equal(tensors["tile_start_sec"], expected_grid):
        raise ValueError("V tile grid must be the frozen [-12,+48) four-second grid")
    if not tensors["evolution_mask"].all():
        raise ValueError("Complete standard-19 V mask must be all true")
    expected_delta_mask = tensors["evolution_mask"].clone()
    expected_delta_mask[:, :, 0] = False
    if not torch.equal(tensors["evolution_delta_mask"], expected_delta_mask):
        raise ValueError("V ordered-delta mask must exclude the first tile")
    if torch.any(tensors["evolution_ordered_delta"][:, :, 0] != 0):
        raise ValueError("First V ordered delta must use finite zero fill")
    if not torch.equal(
        tensors["evolution_ordered_delta"][:, :, 1:],
        tensors["evolution_scaled"][:, :, 1:]
        - tensors["evolution_scaled"][:, :, :-1],
    ):
        raise ValueError("V ordered deltas disagree with adjacent observed tiles")
    if torch.any((tensors["artifact_burden"] < 0) | (tensors["artifact_burden"] > 1)):
        raise ValueError("artifact burden must lie in [0,1]")
    if not torch.equal(tensors["reliability"], 1.0 - tensors["artifact_burden"]):
        raise ValueError("A/Q reliability must be exactly one minus burden")
    if not torch.equal(
        tensors["reliable_mask"],
        tensors["reliability"] >= QUALITY_RELIABLE_THRESHOLD,
    ):
        raise ValueError("A/Q reliable mask drifted")


def _atomic_publish(
    output_directory: Path,
    tensors: Mapping[str, torch.Tensor],
    events_payload: Mapping[str, object],
    manifest_core: Mapping[str, object],
) -> tuple[Path, str]:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("safetensors is required for V+A/Q publication") from exc

    output = Path(os.path.abspath(output_directory))
    if output.name in {"", ".", ".."}:
        raise ValueError("V+A/Q output requires a concrete directory")
    for component in (output.parent, *output.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("V+A/Q output path cannot contain symlink components")
    if os.path.lexists(output):
        raise FileExistsError(f"V+A/Q output already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError("V+A/Q output parent does not exist")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        tensor_path = temporary / DEVELOPMENT_VAQ_TENSOR_FILENAME
        save_file(
            {name: value.contiguous() for name, value in tensors.items()},
            str(tensor_path),
        )
        events_path = temporary / DEVELOPMENT_VAQ_EVENTS_FILENAME
        events_bytes = _canonical_json_bytes(events_payload)
        events_path.write_bytes(events_bytes)
        tensor_size = tensor_path.stat().st_size
        if not 1 <= tensor_size <= _MAX_TENSOR_BYTES:
            raise ValueError("V+A/Q tensor artifact has an invalid size")
        if not 1 <= len(events_bytes) <= _MAX_EVENTS_BYTES:
            raise ValueError("V+A/Q event artifact has an invalid size")
        manifest = {
            **manifest_core,
            "files": {
                DEVELOPMENT_VAQ_TENSOR_FILENAME: {
                    "sha256": _file_sha256(tensor_path),
                    "size_bytes": tensor_size,
                },
                DEVELOPMENT_VAQ_EVENTS_FILENAME: {
                    "sha256": _bytes_sha256(events_bytes),
                    "size_bytes": len(events_bytes),
                },
            },
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        if not 1 <= len(manifest_bytes) <= _MAX_MANIFEST_BYTES:
            raise ValueError("V+A/Q manifest has an invalid size")
        manifest_path = temporary / DEVELOPMENT_VAQ_MANIFEST_FILENAME
        manifest_path.write_bytes(manifest_bytes)
        for path in (tensor_path, events_path, manifest_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(temporary, output)
        published = True
        return output, _bytes_sha256(manifest_bytes)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def materialize_development_vaq_evidence(
    *,
    signal_preflight_bundle: str | Path,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
    evolution_scaler_bundle: str | Path,
    expected_evolution_scaler_artifact_sha256: str,
    tusz_root: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Replay and publish the complete source-dev V+A/Q event roster."""

    signal = load_bound_deepsoz_signal_preflight_artifact(
        signal_preflight_bundle,
        expected_artifact_sha256=expected_signal_artifact_sha256,
        expected_receipt_sha256=expected_signal_receipt_sha256,
    )
    scaler_artifact = load_externally_pinned_computed_evolution_scaler_artifact(
        evolution_scaler_bundle,
        oof_fold=None,
        expected_artifact_sha256=expected_evolution_scaler_artifact_sha256,
    )
    _validate_scaler_lineage(scaler_artifact, signal.receipt)
    timeline, phase_masks = _signal_only_source_dev_timeline(
        signal.receipt,
        signal_artifact_sha256=signal.artifact_sha256,
        signal_receipt_sha256=signal.receipt_sha256,
    )
    config = CausalEDFConfig(**dict(signal.receipt["preprocess_config"]))
    root = Path(tusz_root)

    raw_rows: list[torch.Tensor] = []
    scaled_rows: list[torch.Tensor] = []
    quality_rows: list[TargetFreeQualityEvidence] = []
    event_rows: list[dict[str, object]] = []
    for index, event in enumerate(timeline):
        relative_path, source = _safe_source_file(root, event.relative_edf_path)
        loaded = load_standard19_edf_event(
            source,
            event.global_t0_sec,
            config=config,
        )
        replay_checks = {
            "edf_sha256": loaded.edf_receipt.edf_sha256 == event.edf_sha256,
            "processed_window_sha256": (
                _signal_window_sha256(loaded.window.data)
                == event.processed_window_sha256
            ),
            "edf_receipt_sha256": (
                _canonical_sha256(asdict(loaded.edf_receipt))
                == event.edf_receipt_sha256
            ),
            "signal_receipt_sha256": (
                _canonical_sha256(asdict(loaded.signal_receipt))
                == event.signal_receipt_sha256
            ),
            "preprocess_config_sha256": (
                event.preprocess_config_sha256
                == signal.receipt["preprocess_config_sha256"]
            ),
        }
        failed = tuple(name for name, passed in replay_checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Source-dev signal replay failed for {event.event_id}: {failed}"
            )
        eeg = loaded.window.data.detach().cpu().unsqueeze(0)
        evolution = compute_temporal_evolution_descriptors(eeg)
        scaled = scaler_artifact.scaler.transform(
            evolution.descriptors, evolution.mask
        )
        quality = compute_target_free_quality_evidence(eeg)
        raw_rows.append(evolution.descriptors[0])
        scaled_rows.append(scaled[0])
        quality_rows.append(quality)
        phase_valid = phase_masks.ictal_phase_mask[index]
        event_abstain = bool((quality.tile_abstain[0] & phase_valid).any().item())
        event_rows.append(
            {
                "schema_version": DEVELOPMENT_VAQ_EVENT_SCHEMA,
                "event_id": event.event_id,
                "patient_id": event.patient_id,
                "model_split": DEVELOPMENT_VAQ_SPLIT,
                "relative_edf_path": relative_path,
                "event_record_sha256": event.event_record_sha256,
                "edf_sha256": event.edf_sha256,
                "processed_window_sha256": event.processed_window_sha256,
                "timeline_sha256": event.timeline_sha256,
                "global_t0_sec": event.global_t0_sec,
                "global_stop_sec": event.global_stop_sec,
                "seizure_duration_sec": event.seizure_duration_sec,
                "previous_seizure_gap_sec": event.previous_seizure_gap_sec,
                "evolution_raw_sha256": _tensor_sha256(evolution.descriptors[0]),
                "evolution_scaled_sha256": _tensor_sha256(scaled[0]),
                "quality_diagnostics_sha256": _tensor_sha256(
                    quality.diagnostics[0]
                ),
                "artifact_burden_sha256": _tensor_sha256(
                    quality.artifact_burden[0]
                ),
                "reliability_sha256": _tensor_sha256(quality.reliability[0]),
                "abstention_recommended": event_abstain,
            }
        )

    evolution_raw = torch.stack(raw_rows, dim=0)
    evolution_scaled = torch.stack(scaled_rows, dim=0)
    evolution_delta = torch.zeros_like(evolution_scaled)
    evolution_delta[:, :, 1:] = (
        evolution_scaled[:, :, 1:] - evolution_scaled[:, :, :-1]
    )
    evolution_mask = torch.ones(
        (len(timeline), N_STANDARD_CHANNELS, EVOLUTION_N_TILES),
        dtype=torch.bool,
    )
    evolution_delta_mask = evolution_mask.clone()
    evolution_delta_mask[:, :, 0] = False
    diagnostics = torch.cat([quality.diagnostics for quality in quality_rows], dim=0)
    burden_components = torch.cat(
        [quality.burden_components for quality in quality_rows], dim=0
    )
    burden = torch.cat([quality.artifact_burden for quality in quality_rows], dim=0)
    reliability = torch.cat([quality.reliability for quality in quality_rows], dim=0)
    reliable_mask = torch.cat(
        [quality.reliable_mask for quality in quality_rows], dim=0
    )
    tile_abstain = torch.cat(
        [quality.tile_abstain for quality in quality_rows], dim=0
    )
    phase_tensors = _phase_tensors(phase_masks)
    event_abstain = (tile_abstain & phase_tensors["ictal_phase_mask"]).any(dim=1)
    tensors = {
        "evolution_raw": evolution_raw,
        "evolution_scaled": evolution_scaled,
        "evolution_ordered_delta": evolution_delta,
        "evolution_mask": evolution_mask,
        "evolution_delta_mask": evolution_delta_mask,
        "quality_diagnostics": diagnostics,
        "quality_burden_components": burden_components,
        "artifact_burden": burden,
        "reliability": reliability,
        "reliable_mask": reliable_mask,
        "tile_abstain": tile_abstain,
        "event_abstain": event_abstain,
        "tile_start_sec": torch.arange(-12.0, 48.0, 4.0, dtype=torch.float64),
        **phase_tensors,
    }
    _validate_tensor_payload(tensors)

    patients = tuple(sorted({event.patient_id for event in timeline}))
    event_ids = tuple(event.event_id for event in timeline)
    quality_policy = {
        "schema_version": QUALITY_POLICY_SCHEMA,
        "diagnostic_names": list(QUALITY_DIAGNOSTIC_NAMES),
        "burden_component_names": list(QUALITY_BURDEN_COMPONENT_NAMES),
        "amplitude_ramp_uv": list(QUALITY_AMPLITUDE_RAMP_UV),
        "step_ramp_uv_per_sample": list(QUALITY_STEP_RAMP_UV),
        "flat_step_tolerance_uv": QUALITY_FLAT_STEP_TOLERANCE_UV,
        "flat_fraction_ramp": list(QUALITY_FLAT_FRACTION_RAMP),
        "reliable_threshold": QUALITY_RELIABLE_THRESHOLD,
        "minimum_reliable_channel_fraction": (
            QUALITY_MIN_RELIABLE_CHANNEL_FRACTION
        ),
        "high_frequency_ratio_policy": "reporting_only_never_a_burden",
        "reasoner_access_policy": (
            "reliability_may_only_multiply_nonnegative_support_or_trigger_abstention;"
            "diagnostics_and_burdens_are_forbidden_as_localizing_features"
        ),
        "threshold_status": (
            "fixed_conservative_engineering_sentinels_not_clinically_validated"
        ),
    }
    events_payload = {
        "schema_version": "soz_target_free_development_vaq_event_roster_v1",
        "event_count": len(event_rows),
        "events": event_rows,
    }
    producer_sha = _module_sha256(
        __import__(__name__, fromlist=["unused"]), field="development_vaq.py"
    )
    manifest_core = {
        "schema_version": DEVELOPMENT_VAQ_MANIFEST_SCHEMA,
        "serialization": DEVELOPMENT_VAQ_SERIALIZATION,
        "purpose": DEVELOPMENT_VAQ_PURPOSE,
        "model_split": DEVELOPMENT_VAQ_SPLIT,
        "development_only": True,
        "training_authorized": False,
        "source_eval_allowed": False,
        "private_allowed": False,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "event_count": len(timeline),
        "patient_count": len(patients),
        "event_roster_sha256": _canonical_sha256(event_ids),
        "patient_roster_sha256": _canonical_sha256(patients),
        "signal_preflight_artifact_sha256": signal.artifact_sha256,
        "signal_preflight_receipt_sha256": signal.receipt_sha256,
        "split_manifest_sha256": signal.receipt["split_manifest_sha256"],
        "evolution_scaler_artifact_sha256": scaler_artifact.artifact_sha256,
        "evolution_scaler_receipt_sha256": (
            scaler_artifact.receipt.receipt_sha256
        ),
        "evolution_feature_names": list(EVOLUTION_FEATURES),
        "evolution_feature_schema_sha256": EVOLUTION_FEATURE_SCHEMA_SHA256,
        "evolution_semantics": (
            "observable_descriptors_and_adjacent_ordered_differences_only;"
            "not_origin_not_propagation_not_soz"
        ),
        "quality_policy": quality_policy,
        "quality_policy_sha256": _canonical_sha256(quality_policy),
        "producer_source_sha256": producer_sha,
        "evolution_source_sha256": _module_sha256(
            _evolution_module, field="evolution.py"
        ),
        "geometry_source_sha256": _module_sha256(
            _geometry_module, field="geometry.py"
        ),
        "tensor_specs": _tensor_specs(tensors),
    }
    path, manifest_sha = _atomic_publish(
        Path(output_directory), tensors, events_payload, manifest_core
    )
    return {
        "path": str(path),
        "manifest_sha256": manifest_sha,
        "event_count": len(timeline),
        "patient_count": len(patients),
        "abstention_recommended_count": int(event_abstain.sum().item()),
        "model_split": DEVELOPMENT_VAQ_SPLIT,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
    }


def load_development_vaq_evidence(
    bundle_directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[Mapping[str, object], Mapping[str, object], dict[str, torch.Tensor]]:
    """Strictly load a published development V+A/Q bundle."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("safetensors is required for V+A/Q loading") from exc
    bundle = Path(os.path.abspath(bundle_directory))
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("V+A/Q bundle must be a regular directory")
    expected_files = {
        DEVELOPMENT_VAQ_MANIFEST_FILENAME,
        DEVELOPMENT_VAQ_EVENTS_FILENAME,
        DEVELOPMENT_VAQ_TENSOR_FILENAME,
    }
    actual_files = {path.name for path in bundle.iterdir()}
    if actual_files != expected_files:
        raise ValueError("V+A/Q bundle violates its closed file schema")
    manifest_raw = (bundle / DEVELOPMENT_VAQ_MANIFEST_FILENAME).read_bytes()
    if _bytes_sha256(manifest_raw) != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("V+A/Q manifest SHA mismatch")
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("V+A/Q manifest is not strict JSON") from exc
    if _canonical_json_bytes(manifest) != manifest_raw:
        raise ValueError("V+A/Q manifest is not canonical JSON")
    if manifest.get("schema_version") != DEVELOPMENT_VAQ_MANIFEST_SCHEMA:
        raise ValueError("Unsupported V+A/Q manifest schema")
    immutable_boundary = {
        "purpose": DEVELOPMENT_VAQ_PURPOSE,
        "model_split": DEVELOPMENT_VAQ_SPLIT,
        "development_only": True,
        "training_authorized": False,
        "source_eval_allowed": False,
        "private_allowed": False,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
    }
    for field, expected in immutable_boundary.items():
        if manifest.get(field) != expected:
            raise ValueError(f"V+A/Q boundary field changed: {field}")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        DEVELOPMENT_VAQ_EVENTS_FILENAME,
        DEVELOPMENT_VAQ_TENSOR_FILENAME,
    }:
        raise ValueError("V+A/Q file receipt schema changed")
    for filename in files:
        path = bundle / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("V+A/Q payload must be regular and non-symlinked")
        if _file_sha256(path) != files[filename]["sha256"]:
            raise ValueError(f"V+A/Q payload SHA mismatch: {filename}")

    events_raw = (bundle / DEVELOPMENT_VAQ_EVENTS_FILENAME).read_bytes()
    try:
        events_payload = json.loads(events_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("V+A/Q events payload is not strict JSON") from exc
    if _canonical_json_bytes(events_payload) != events_raw:
        raise ValueError("V+A/Q events payload is not canonical JSON")
    if events_payload.get("event_count") != manifest.get("event_count"):
        raise ValueError("V+A/Q event count disagrees with manifest")
    event_rows = events_payload.get("events")
    if not isinstance(event_rows, list) or any(
        not isinstance(row, dict) for row in event_rows
    ):
        raise ValueError("V+A/Q event roster is invalid")
    if any(row.get("model_split") != DEVELOPMENT_VAQ_SPLIT for row in event_rows):
        raise ValueError("V+A/Q event roster escaped source-dev")

    tensors = load_file(str(bundle / DEVELOPMENT_VAQ_TENSOR_FILENAME), device="cpu")
    _validate_tensor_payload(tensors)
    if _tensor_specs(tensors) != manifest.get("tensor_specs"):
        raise ValueError("V+A/Q tensor specs disagree with manifest")
    if tensors["evolution_raw"].shape[0] != manifest.get("event_count"):
        raise ValueError("V+A/Q tensor event count disagrees with manifest")
    return manifest, events_payload, tensors


__all__ = [
    "DEVELOPMENT_VAQ_MANIFEST_SCHEMA",
    "DEVELOPMENT_VAQ_PURPOSE",
    "DEVELOPMENT_VAQ_SCHEMA",
    "DEVELOPMENT_VAQ_SPLIT",
    "QUALITY_BURDEN_COMPONENT_NAMES",
    "QUALITY_DIAGNOSTIC_NAMES",
    "QUALITY_POLICY_SCHEMA",
    "TargetFreeQualityEvidence",
    "attenuate_nonnegative_support",
    "compute_target_free_quality_evidence",
    "load_development_vaq_evidence",
    "materialize_development_vaq_evidence",
    "preflight_development_vaq_inputs",
]
