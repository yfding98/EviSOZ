"""Target-free V and A/Q production for the locked source-eval roster.

Only deterministic temporal-evolution descriptors and conservative signal
quality evidence are produced.  The module accepts no SOZ target artifact and
no TUSZ channel target/mask.  Its outputs are inference-only inputs for an
already frozen SOZ pipeline; training, model selection, and threshold tuning
are forbidden by the artifact schema.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Mapping

import torch

from . import development_vaq as _quality_module
from . import evolution as _evolution_module
from . import geometry as _geometry_module
from . import locked_source_eval_roster as _roster_module
from .data.edf import CausalEDFConfig, load_standard19_edf_event
from .development_vaq import (
    QUALITY_AMPLITUDE_RAMP_UV,
    QUALITY_BURDEN_COMPONENT_NAMES,
    QUALITY_DIAGNOSTIC_NAMES,
    QUALITY_FLAT_FRACTION_RAMP,
    QUALITY_FLAT_STEP_TOLERANCE_UV,
    QUALITY_MIN_RELIABLE_CHANNEL_FRACTION,
    QUALITY_POLICY_SCHEMA,
    QUALITY_RELIABLE_THRESHOLD,
    QUALITY_STEP_RAMP_UV,
    TargetFreeQualityEvidence,
    compute_target_free_quality_evidence,
)
from .evolution import (
    EVOLUTION_FEATURE_SCHEMA_SHA256,
    EVOLUTION_FEATURES,
    EVOLUTION_N_TILES,
    compute_temporal_evolution_descriptors,
)
from .evolution_io import (
    ComputedEvolutionScalerArtifact,
    load_externally_pinned_computed_evolution_scaler_artifact,
)
from .geometry import N_STANDARD_CHANNELS
from .ictal_native_eval import load_bound_deepsoz_signal_preflight_artifact
from .locked_source_eval_roster import (
    EXPECTED_SOURCE_EVAL_EVENT_COUNT,
    EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
    LOCKED_SOURCE_EVAL_MODEL_SPLIT,
    LockedSourceEvalEvent,
    VerifiedLockedSourceEvalRoster,
    derive_locked_source_eval_roster_receipt,
    load_locked_source_eval_roster,
)
from .temporal_masks import OffsetAwarePhaseMasks, build_offset_aware_phase_masks


LOCKED_SOURCE_EVAL_VAQ_SCHEMA = "soz_locked_target_free_source_eval_vaq_v1"
LOCKED_SOURCE_EVAL_VAQ_MANIFEST_SCHEMA = (
    "soz_locked_target_free_source_eval_vaq_manifest_v1"
)
LOCKED_SOURCE_EVAL_VAQ_EVENT_SCHEMA = (
    "soz_locked_target_free_source_eval_vaq_event_v1"
)
LOCKED_SOURCE_EVAL_VAQ_EVENT_ROSTER_SCHEMA = (
    "soz_locked_target_free_source_eval_vaq_event_roster_v1"
)
LOCKED_SOURCE_EVAL_VAQ_PURPOSE = (
    "frozen_v9_source_eval_target_free_vaq_inference_only"
)
LOCKED_SOURCE_EVAL_VAQ_SERIALIZATION = (
    "canonical_json_plus_safetensors_no_pickle"
)
LOCKED_SOURCE_EVAL_VAQ_TENSOR_FILENAME = "vaq.safetensors"
LOCKED_SOURCE_EVAL_VAQ_EVENTS_FILENAME = "events.json"
LOCKED_SOURCE_EVAL_VAQ_MANIFEST_FILENAME = "manifest.json"

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
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "ordinal",
        "event_id",
        "patient_id",
        "model_split",
        "relative_edf_path",
        "signal_event_record_sha256",
        "record_timeline_sha256",
        "global_t0_sec",
        "global_stop_sec",
        "seizure_duration_sec",
        "previous_seizure_gap_sec",
        "edf_sha256",
        "processed_window_sha256",
        "evolution_raw_sha256",
        "evolution_scaled_sha256",
        "quality_diagnostics_sha256",
        "artifact_burden_sha256",
        "reliability_sha256",
        "abstention_recommended",
    }
)
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_EVENTS_BYTES = 64 * 1024 * 1024
_MAX_TENSOR_BYTES = 1024 * 1024 * 1024
_SHA256_HEX = frozenset("0123456789abcdef")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Locked source-eval V+A/Q is not canonical JSON data") from exc


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
        raise ValueError(f"{field} must be a regular source file")
    return _file_sha256(path)


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


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


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return result


def _safe_source_file(root: Path, relative_value: object) -> tuple[str, Path]:
    if not isinstance(relative_value, str) or not relative_value or "\\" in relative_value:
        raise ValueError("relative_edf_path must be a canonical POSIX path")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("relative_edf_path is not canonical")
    root_absolute = _absolute_no_symlink(root, field="TUSZ root")
    if not root_absolute.is_dir():
        raise FileNotFoundError("TUSZ root does not exist")
    candidate = _absolute_no_symlink(
        root_absolute.joinpath(*relative.parts), field="source-eval EDF"
    )
    try:
        candidate.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError("relative_edf_path escapes TUSZ root") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"Required source-eval EDF is unavailable: {candidate}")
    return relative.as_posix(), candidate


def _validate_scaler_lineage(
    scaler: ComputedEvolutionScalerArtifact,
    roster: VerifiedLockedSourceEvalRoster,
) -> None:
    if scaler.receipt.oof_fold is not None:
        raise ValueError("Locked source-eval V must use the final source-train scaler")
    if scaler.scaler.receipt.fit_split != "source_train":
        raise ValueError("Evolution scaler was not fitted on source_train")
    if scaler.scaler.receipt.split_manifest_sha256 != roster.receipt[
        "split_manifest_sha256"
    ]:
        raise ValueError("Locked roster and evolution scaler use different splits")
    if _module_sha256(
        _evolution_module, field="evolution.py"
    ) != scaler.computation_receipt.evolution_source_sha256:
        raise ValueError("Current evolution implementation differs from the scaler")
    if _module_sha256(
        _geometry_module, field="geometry.py"
    ) != scaler.computation_receipt.geometry_source_sha256:
        raise ValueError("Current standard-19 geometry differs from the scaler")


def _phase_masks(roster: VerifiedLockedSourceEvalRoster) -> OffsetAwarePhaseMasks:
    return build_offset_aware_phase_masks(
        [event.seizure_duration_sec for event in roster.events],
        offset_trustworthy=[True] * len(roster.events),
        previous_seizure_gap_sec=[
            event.previous_seizure_gap_sec for event in roster.events
        ],
        previous_timeline_trustworthy=[True] * len(roster.events),
    )


def _phase_tensors(phase_masks: OffsetAwarePhaseMasks) -> dict[str, torch.Tensor]:
    return {
        name: getattr(phase_masks, name).detach().cpu().contiguous()
        for name in _PHASE_TENSOR_NAMES
    }


def _tensor_specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, object]:
    if set(tensors) != set(_TENSOR_NAMES):
        raise ValueError("Locked V+A/Q tensors violate the closed schema")
    return {
        name: {
            "shape": list(tensors[name].shape),
            "dtype": str(tensors[name].dtype).removeprefix("torch."),
            "tensor_sha256": _tensor_sha256(tensors[name]),
        }
        for name in _TENSOR_NAMES
    }


def _validate_tensor_payload(tensors: Mapping[str, torch.Tensor]) -> None:
    # Reuse the already tested V+A/Q numerical invariants, then add the locked
    # eval cardinality constraint.  The manifest schema remains independent.
    _quality_module._validate_tensor_payload(tensors)
    if tensors["evolution_raw"].shape[0] != EXPECTED_SOURCE_EVAL_EVENT_COUNT:
        raise ValueError("Locked source-eval V+A/Q must contain exactly 185 events")


def _load_inputs(
    *,
    roster_bundle: str | Path,
    expected_roster_artifact_sha256: str,
    signal_preflight_bundle: str | Path,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
    evolution_scaler_bundle: str | Path,
    expected_evolution_scaler_artifact_sha256: str,
    tusz_root: str | Path,
):
    roster = load_locked_source_eval_roster(
        roster_bundle,
        expected_artifact_sha256=expected_roster_artifact_sha256,
        expected_signal_artifact_sha256=expected_signal_artifact_sha256,
        expected_signal_receipt_sha256=expected_signal_receipt_sha256,
    )
    signal = load_bound_deepsoz_signal_preflight_artifact(
        signal_preflight_bundle,
        expected_artifact_sha256=expected_signal_artifact_sha256,
        expected_receipt_sha256=expected_signal_receipt_sha256,
    )
    replayed = derive_locked_source_eval_roster_receipt(
        signal.receipt,
        signal_artifact_sha256=signal.artifact_sha256,
        signal_receipt_sha256=signal.receipt_sha256,
    )
    if _canonical_json_bytes(replayed) != _canonical_json_bytes(roster.receipt):
        raise ValueError("Locked source-eval roster does not replay from signal input")
    scaler = load_externally_pinned_computed_evolution_scaler_artifact(
        evolution_scaler_bundle,
        oof_fold=None,
        expected_artifact_sha256=expected_evolution_scaler_artifact_sha256,
    )
    _validate_scaler_lineage(scaler, roster)
    root = _absolute_no_symlink(tusz_root, field="TUSZ root")
    if not root.is_dir():
        raise FileNotFoundError("TUSZ root does not exist")
    for event in roster.events:
        _safe_source_file(root, event.relative_edf_path)
    return roster, signal, scaler, root


def preflight_locked_source_eval_vaq_inputs(
    *,
    roster_bundle: str | Path,
    expected_roster_artifact_sha256: str,
    signal_preflight_bundle: str | Path,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
    evolution_scaler_bundle: str | Path,
    expected_evolution_scaler_artifact_sha256: str,
    tusz_root: str | Path,
) -> dict[str, object]:
    """Validate all target-free source-eval V+A/Q inputs without EDF reads."""

    roster, _, scaler, _ = _load_inputs(
        roster_bundle=roster_bundle,
        expected_roster_artifact_sha256=expected_roster_artifact_sha256,
        signal_preflight_bundle=signal_preflight_bundle,
        expected_signal_artifact_sha256=expected_signal_artifact_sha256,
        expected_signal_receipt_sha256=expected_signal_receipt_sha256,
        evolution_scaler_bundle=evolution_scaler_bundle,
        expected_evolution_scaler_artifact_sha256=(
            expected_evolution_scaler_artifact_sha256
        ),
        tusz_root=tusz_root,
    )
    return {
        "schema_version": LOCKED_SOURCE_EVAL_VAQ_SCHEMA,
        "status": "ready_target_free_source_eval_vaq",
        "purpose": LOCKED_SOURCE_EVAL_VAQ_PURPOSE,
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "event_count": len(roster.events),
        "patient_count": len(roster.patient_ids),
        "event_order_sha256": roster.receipt["event_order_sha256"],
        "patient_roster_sha256": roster.receipt["patient_roster_sha256"],
        "roster_artifact_sha256": roster.artifact_sha256,
        "evolution_scaler_artifact_sha256": scaler.artifact_sha256,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "target_values_loaded": False,
        "target_paths_accepted": False,
        "training_authorized": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
        "edf_payloads_read": False,
        "model_forward_count": 0,
    }


def _event_output_row(
    event: LockedSourceEvalEvent,
    *,
    relative_path: str,
    raw: torch.Tensor,
    scaled: torch.Tensor,
    quality: TargetFreeQualityEvidence,
    abstain: bool,
) -> dict[str, object]:
    return {
        "schema_version": LOCKED_SOURCE_EVAL_VAQ_EVENT_SCHEMA,
        "ordinal": event.ordinal,
        "event_id": event.event_id,
        "patient_id": event.patient_id,
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "relative_edf_path": relative_path,
        "signal_event_record_sha256": event.signal_event_record_sha256,
        "record_timeline_sha256": event.record_timeline_sha256,
        "global_t0_sec": event.global_t0_sec,
        "global_stop_sec": event.global_stop_sec,
        "seizure_duration_sec": event.seizure_duration_sec,
        "previous_seizure_gap_sec": event.previous_seizure_gap_sec,
        "edf_sha256": event.edf_sha256,
        "processed_window_sha256": event.processed_window_sha256,
        "evolution_raw_sha256": _tensor_sha256(raw),
        "evolution_scaled_sha256": _tensor_sha256(scaled),
        "quality_diagnostics_sha256": _tensor_sha256(quality.diagnostics[0]),
        "artifact_burden_sha256": _tensor_sha256(quality.artifact_burden[0]),
        "reliability_sha256": _tensor_sha256(quality.reliability[0]),
        "abstention_recommended": bool(abstain),
    }


def _quality_policy() -> dict[str, object]:
    return {
        "schema_version": QUALITY_POLICY_SCHEMA,
        "diagnostic_names": list(QUALITY_DIAGNOSTIC_NAMES),
        "burden_component_names": list(QUALITY_BURDEN_COMPONENT_NAMES),
        "amplitude_ramp_uv": list(QUALITY_AMPLITUDE_RAMP_UV),
        "step_ramp_uv_per_sample": list(QUALITY_STEP_RAMP_UV),
        "flat_step_tolerance_uv": QUALITY_FLAT_STEP_TOLERANCE_UV,
        "flat_fraction_ramp": list(QUALITY_FLAT_FRACTION_RAMP),
        "reliable_threshold": QUALITY_RELIABLE_THRESHOLD,
        "minimum_reliable_channel_fraction": QUALITY_MIN_RELIABLE_CHANNEL_FRACTION,
        "high_frequency_ratio_policy": "reporting_only_never_a_burden",
        "reasoner_access_policy": (
            "reliability_may_only_multiply_nonnegative_support_or_trigger_abstention;"
            "diagnostics_and_burdens_are_forbidden_as_localizing_features"
        ),
        "threshold_status": (
            "fixed_conservative_engineering_sentinels_not_clinically_validated"
        ),
    }


def _atomic_publish(
    output_directory: str | Path,
    *,
    tensors: Mapping[str, torch.Tensor],
    events_payload: Mapping[str, object],
    manifest_core: Mapping[str, object],
) -> tuple[Path, str]:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for V+A/Q publication") from exc
    _validate_tensor_payload(tensors)
    output = _absolute_no_symlink(output_directory, field="locked V+A/Q output")
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise FileExistsError(f"Locked V+A/Q output exists or is invalid: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError("Locked V+A/Q output parent does not exist")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        tensor_path = temporary / LOCKED_SOURCE_EVAL_VAQ_TENSOR_FILENAME
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
            str(tensor_path),
        )
        events_path = temporary / LOCKED_SOURCE_EVAL_VAQ_EVENTS_FILENAME
        events_raw = _canonical_json_bytes(events_payload)
        events_path.write_bytes(events_raw)
        tensor_size = tensor_path.stat().st_size
        if not 1 <= tensor_size <= _MAX_TENSOR_BYTES:
            raise ValueError("Locked V+A/Q tensor file has an invalid size")
        if not 1 <= len(events_raw) <= _MAX_EVENTS_BYTES:
            raise ValueError("Locked V+A/Q event file has an invalid size")
        manifest = {
            **manifest_core,
            "files": {
                LOCKED_SOURCE_EVAL_VAQ_TENSOR_FILENAME: {
                    "sha256": _file_sha256(tensor_path),
                    "size_bytes": tensor_size,
                },
                LOCKED_SOURCE_EVAL_VAQ_EVENTS_FILENAME: {
                    "sha256": _bytes_sha256(events_raw),
                    "size_bytes": len(events_raw),
                },
            },
        }
        manifest_raw = _canonical_json_bytes(manifest)
        if not 1 <= len(manifest_raw) <= _MAX_MANIFEST_BYTES:
            raise ValueError("Locked V+A/Q manifest has an invalid size")
        manifest_path = temporary / LOCKED_SOURCE_EVAL_VAQ_MANIFEST_FILENAME
        manifest_path.write_bytes(manifest_raw)
        for path in (tensor_path, events_path, manifest_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if os.path.lexists(output):
            raise FileExistsError(output)
        os.rename(temporary, output)
        published = True
        descriptor = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return output, _bytes_sha256(manifest_raw)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def materialize_locked_source_eval_vaq(
    *,
    roster_bundle: str | Path,
    expected_roster_artifact_sha256: str,
    signal_preflight_bundle: str | Path,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
    evolution_scaler_bundle: str | Path,
    expected_evolution_scaler_artifact_sha256: str,
    tusz_root: str | Path,
    output_directory: str | Path,
    progress_every: int = 20,
) -> dict[str, object]:
    """Replay all 185 EDF windows and publish locked target-free V+A/Q."""

    if isinstance(progress_every, bool) or not isinstance(progress_every, int) or progress_every < 1:
        raise ValueError("progress_every must be a positive integer")
    roster, signal, scaler, root = _load_inputs(
        roster_bundle=roster_bundle,
        expected_roster_artifact_sha256=expected_roster_artifact_sha256,
        signal_preflight_bundle=signal_preflight_bundle,
        expected_signal_artifact_sha256=expected_signal_artifact_sha256,
        expected_signal_receipt_sha256=expected_signal_receipt_sha256,
        evolution_scaler_bundle=evolution_scaler_bundle,
        expected_evolution_scaler_artifact_sha256=(
            expected_evolution_scaler_artifact_sha256
        ),
        tusz_root=tusz_root,
    )
    config = CausalEDFConfig(**dict(roster.receipt["preprocess_config"]))
    phase_masks = _phase_masks(roster)
    raw_rows: list[torch.Tensor] = []
    scaled_rows: list[torch.Tensor] = []
    quality_rows: list[TargetFreeQualityEvidence] = []
    event_rows: list[dict[str, object]] = []
    started = time.monotonic()
    for index, event in enumerate(roster.events):
        relative_path, source = _safe_source_file(root, event.relative_edf_path)
        loaded = load_standard19_edf_event(source, event.global_t0_sec, config=config)
        replay_checks = {
            "edf_sha256": loaded.edf_receipt.edf_sha256 == event.edf_sha256,
            "processed_window_sha256": (
                _tensor_sha256(loaded.window.data) == event.processed_window_sha256
            ),
            "processed_window_shape": tuple(loaded.window.data.shape)
            == event.processed_window_shape,
            "processed_window_dtype": str(loaded.window.data.dtype)
            == event.processed_window_dtype,
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
                == roster.receipt["preprocess_config_sha256"]
            ),
        }
        failed = tuple(name for name, passed in replay_checks.items() if not passed)
        if failed:
            raise ValueError(f"Locked source-eval replay failed for {event.event_id}: {failed}")
        eeg = loaded.window.data.detach().cpu().unsqueeze(0)
        evolution = compute_temporal_evolution_descriptors(eeg)
        scaled = scaler.scaler.transform(evolution.descriptors, evolution.mask)
        quality = compute_target_free_quality_evidence(eeg)
        raw_rows.append(evolution.descriptors[0])
        scaled_rows.append(scaled[0])
        quality_rows.append(quality)
        phase_valid = phase_masks.ictal_phase_mask[index]
        abstain = bool((quality.tile_abstain[0] & phase_valid).any().item())
        event_rows.append(
            _event_output_row(
                event,
                relative_path=relative_path,
                raw=evolution.descriptors[0],
                scaled=scaled[0],
                quality=quality,
                abstain=abstain,
            )
        )
        if (index + 1) % progress_every == 0 or index + 1 == len(roster.events):
            print(
                json.dumps(
                    {
                        "stage": "locked_source_eval_vaq_materialization",
                        "completed": index + 1,
                        "total": len(roster.events),
                        "elapsed_sec": time.monotonic() - started,
                        "target_values_loaded": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    evolution_raw = torch.stack(raw_rows)
    evolution_scaled = torch.stack(scaled_rows)
    evolution_delta = torch.zeros_like(evolution_scaled)
    evolution_delta[:, :, 1:] = (
        evolution_scaled[:, :, 1:] - evolution_scaled[:, :, :-1]
    )
    evolution_mask = torch.ones(
        (EXPECTED_SOURCE_EVAL_EVENT_COUNT, N_STANDARD_CHANNELS, EVOLUTION_N_TILES),
        dtype=torch.bool,
    )
    evolution_delta_mask = evolution_mask.clone()
    evolution_delta_mask[:, :, 0] = False
    diagnostics = torch.cat([value.diagnostics for value in quality_rows], dim=0)
    burden_components = torch.cat(
        [value.burden_components for value in quality_rows], dim=0
    )
    burden = torch.cat([value.artifact_burden for value in quality_rows], dim=0)
    reliability = torch.cat([value.reliability for value in quality_rows], dim=0)
    reliable_mask = torch.cat([value.reliable_mask for value in quality_rows], dim=0)
    tile_abstain = torch.cat([value.tile_abstain for value in quality_rows], dim=0)
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
    events_payload = {
        "schema_version": LOCKED_SOURCE_EVAL_VAQ_EVENT_ROSTER_SCHEMA,
        "event_count": len(event_rows),
        "patient_count": len(roster.patient_ids),
        "event_order_sha256": roster.receipt["event_order_sha256"],
        "patient_roster_sha256": roster.receipt["patient_roster_sha256"],
        "events": event_rows,
    }
    policy = _quality_policy()
    manifest_core = {
        "schema_version": LOCKED_SOURCE_EVAL_VAQ_MANIFEST_SCHEMA,
        "serialization": LOCKED_SOURCE_EVAL_VAQ_SERIALIZATION,
        "purpose": LOCKED_SOURCE_EVAL_VAQ_PURPOSE,
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "locked_evaluation": True,
        "training_authorized": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "target_values_loaded": False,
        "target_paths_accepted": False,
        "model_forward_count": 0,
        "event_count": len(roster.events),
        "patient_count": len(roster.patient_ids),
        "event_order_sha256": roster.receipt["event_order_sha256"],
        "patient_roster_sha256": roster.receipt["patient_roster_sha256"],
        "roster_artifact_sha256": roster.artifact_sha256,
        "roster_receipt_sha256": roster.receipt_sha256,
        "signal_preflight_artifact_sha256": signal.artifact_sha256,
        "signal_preflight_receipt_sha256": signal.receipt_sha256,
        "split_manifest_sha256": roster.receipt["split_manifest_sha256"],
        "evolution_scaler_artifact_sha256": scaler.artifact_sha256,
        "evolution_scaler_receipt_sha256": scaler.receipt.receipt_sha256,
        "evolution_feature_names": list(EVOLUTION_FEATURES),
        "evolution_feature_schema_sha256": EVOLUTION_FEATURE_SCHEMA_SHA256,
        "evolution_semantics": (
            "observable_descriptors_and_adjacent_ordered_differences_only;"
            "not_origin_not_propagation_not_soz"
        ),
        "quality_policy": policy,
        "quality_policy_sha256": _canonical_sha256(policy),
        "producer_source_sha256": _module_sha256(
            __import__(__name__, fromlist=["unused"]), field="locked_source_eval_vaq.py"
        ),
        "roster_source_sha256": _module_sha256(
            _roster_module, field="locked_source_eval_roster.py"
        ),
        "quality_source_sha256": _module_sha256(
            _quality_module, field="development_vaq.py"
        ),
        "evolution_source_sha256": _module_sha256(
            _evolution_module, field="evolution.py"
        ),
        "geometry_source_sha256": _module_sha256(
            _geometry_module, field="geometry.py"
        ),
        "tensor_specs": _tensor_specs(tensors),
    }
    path, manifest_sha = _atomic_publish(
        output_directory,
        tensors=tensors,
        events_payload=events_payload,
        manifest_core=manifest_core,
    )
    return {
        "status": "published_target_free_source_eval_vaq",
        "path": str(path),
        "manifest_sha256": manifest_sha,
        "event_count": len(roster.events),
        "patient_count": len(roster.patient_ids),
        "abstention_recommended_count": int(event_abstain.sum().item()),
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "target_values_loaded": False,
        "model_forward_count": 0,
    }


def _strict_json_file(path: Path, *, field: str, maximum: int) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= maximum:
        raise ValueError(f"{field} must be a bounded regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict JSON") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise ValueError(f"{field} is not canonical JSON")
    return value, raw


def load_locked_source_eval_vaq(
    bundle_directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[Mapping[str, object], Mapping[str, object], dict[str, torch.Tensor]]:
    """Strictly load a published locked source-eval V+A/Q artifact."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for V+A/Q loading") from exc
    bundle = _absolute_no_symlink(bundle_directory, field="locked V+A/Q bundle")
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("Locked V+A/Q bundle must be a regular directory")
    expected_files = {
        LOCKED_SOURCE_EVAL_VAQ_MANIFEST_FILENAME,
        LOCKED_SOURCE_EVAL_VAQ_EVENTS_FILENAME,
        LOCKED_SOURCE_EVAL_VAQ_TENSOR_FILENAME,
    }
    if {path.name for path in bundle.iterdir()} != expected_files:
        raise ValueError("Locked V+A/Q bundle violates its closed file schema")
    manifest, manifest_raw = _strict_json_file(
        bundle / LOCKED_SOURCE_EVAL_VAQ_MANIFEST_FILENAME,
        field="locked V+A/Q manifest",
        maximum=_MAX_MANIFEST_BYTES,
    )
    if _bytes_sha256(manifest_raw) != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Locked V+A/Q manifest SHA mismatch")
    boundary = {
        "schema_version": LOCKED_SOURCE_EVAL_VAQ_MANIFEST_SCHEMA,
        "serialization": LOCKED_SOURCE_EVAL_VAQ_SERIALIZATION,
        "purpose": LOCKED_SOURCE_EVAL_VAQ_PURPOSE,
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "locked_evaluation": True,
        "training_authorized": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "target_values_loaded": False,
        "target_paths_accepted": False,
        "model_forward_count": 0,
        "event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
        "patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
    }
    changed = tuple(field for field, expected in boundary.items() if manifest.get(field) != expected)
    if changed:
        raise ValueError(f"Locked V+A/Q boundary changed: {changed}")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        LOCKED_SOURCE_EVAL_VAQ_EVENTS_FILENAME,
        LOCKED_SOURCE_EVAL_VAQ_TENSOR_FILENAME,
    }:
        raise ValueError("Locked V+A/Q file receipt schema changed")
    for filename, receipt_value in files.items():
        if not isinstance(receipt_value, dict):
            raise TypeError("Locked V+A/Q file receipt must be an object")
        path = bundle / filename
        if _file_sha256(path) != receipt_value.get("sha256"):
            raise ValueError(f"Locked V+A/Q payload SHA mismatch: {filename}")
    events_payload, events_raw = _strict_json_file(
        bundle / LOCKED_SOURCE_EVAL_VAQ_EVENTS_FILENAME,
        field="locked V+A/Q events",
        maximum=_MAX_EVENTS_BYTES,
    )
    if _bytes_sha256(events_raw) != files[
        LOCKED_SOURCE_EVAL_VAQ_EVENTS_FILENAME
    ]["sha256"]:
        raise ValueError("Locked V+A/Q event receipt SHA mismatch")
    if (
        events_payload.get("schema_version")
        != LOCKED_SOURCE_EVAL_VAQ_EVENT_ROSTER_SCHEMA
        or events_payload.get("event_count") != EXPECTED_SOURCE_EVAL_EVENT_COUNT
        or events_payload.get("patient_count") != EXPECTED_SOURCE_EVAL_PATIENT_COUNT
        or events_payload.get("event_order_sha256") != manifest.get("event_order_sha256")
        or events_payload.get("patient_roster_sha256")
        != manifest.get("patient_roster_sha256")
    ):
        raise ValueError("Locked V+A/Q event roster boundary changed")
    rows = events_payload.get("events")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SOURCE_EVAL_EVENT_COUNT:
        raise ValueError("Locked V+A/Q event rows are incomplete")
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(_EVENT_FIELDS):
            raise ValueError("Locked V+A/Q event row violates its closed schema")
        if (
            row.get("schema_version") != LOCKED_SOURCE_EVAL_VAQ_EVENT_SCHEMA
            or row.get("ordinal") != ordinal
            or row.get("model_split") != LOCKED_SOURCE_EVAL_MODEL_SPLIT
        ):
            raise ValueError("Locked V+A/Q event row escaped its frozen boundary")
    tensors = load_file(
        str(bundle / LOCKED_SOURCE_EVAL_VAQ_TENSOR_FILENAME), device="cpu"
    )
    _validate_tensor_payload(tensors)
    if _tensor_specs(tensors) != manifest.get("tensor_specs"):
        raise ValueError("Locked V+A/Q tensor specs disagree with the manifest")
    return manifest, events_payload, tensors


__all__ = [
    "LOCKED_SOURCE_EVAL_VAQ_EVENT_SCHEMA",
    "LOCKED_SOURCE_EVAL_VAQ_MANIFEST_SCHEMA",
    "LOCKED_SOURCE_EVAL_VAQ_PURPOSE",
    "LOCKED_SOURCE_EVAL_VAQ_SCHEMA",
    "load_locked_source_eval_vaq",
    "materialize_locked_source_eval_vaq",
    "preflight_locked_source_eval_vaq_inputs",
]
