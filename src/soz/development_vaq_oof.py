"""Target-free patient-OOF V+A/Q evidence for source-train development.

Every source-train patient is routed through
``TargetFreeOOFProtocolView.fold_for_target`` and the matching fold-specific
evolution scaler.  The final scaler is structurally unavailable on this path.
The bundle may support development diagnostics or a candidate reasoner only;
it is not authority for formal promotion or evaluation claims.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from . import evolution as _evolution_module
from . import geometry as _geometry_module
from .data.edf import CausalEDFConfig, load_standard19_edf_event
from .data.deepsoz_signal_preflight import VerifiedDeepSOZSignalPreflightBundle
from .development_vaq import (
    DEVELOPMENT_VAQ_EVENT_SCHEMA,
    QUALITY_BURDEN_COMPONENT_NAMES,
    QUALITY_DIAGNOSTIC_NAMES,
    QUALITY_FLAT_FRACTION_RAMP,
    QUALITY_FLAT_STEP_TOLERANCE_UV,
    QUALITY_AMPLITUDE_RAMP_UV,
    QUALITY_MIN_RELIABLE_CHANNEL_FRACTION,
    QUALITY_POLICY_SCHEMA,
    QUALITY_RELIABLE_THRESHOLD,
    QUALITY_STEP_RAMP_UV,
    _PHASE_TENSOR_NAMES,
    _atomic_publish,
    _canonical_sha256,
    _module_sha256,
    _phase_tensors,
    _safe_source_file,
    _signal_window_sha256,
    _tensor_sha256,
    _tensor_specs,
    _validate_tensor_payload,
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
from .ictal_recovery_evidence import (
    TargetFreeOOFProtocolView,
    load_target_free_ictal_oof_protocol,
)
from .temporal_masks import OffsetAwarePhaseMasks, build_offset_aware_phase_masks


SOURCE_TRAIN_OOF_VAQ_SCHEMA = "soz_target_free_source_train_oof_vaq_v1"
SOURCE_TRAIN_OOF_VAQ_MANIFEST_SCHEMA = (
    "soz_target_free_source_train_oof_vaq_manifest_v1"
)
SOURCE_TRAIN_OOF_VAQ_EVENT_SCHEMA = (
    "soz_target_free_source_train_oof_vaq_event_v1"
)
SOURCE_TRAIN_OOF_VAQ_PURPOSE = (
    "source_train_patient_oof_development_diagnostics_and_candidate_reasoner_only"
)
SOURCE_TRAIN_SPLIT = "source_train"

_EXPECTED_FOLDS = tuple(range(5))
_SHA256_CHARS = frozenset("0123456789abcdef")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in _SHA256_CHARS for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(raw: bytes, *, field: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"{field} contains non-finite constant {value}")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class _SourceTrainTimelineEvent:
    __slots__ = (
        "event_id",
        "patient_id",
        "oof_fold",
        "event_record_sha256",
        "relative_edf_path",
        "edf_sha256",
        "global_t0_sec",
        "global_stop_sec",
        "seizure_duration_sec",
        "previous_seizure_gap_sec",
        "timeline_sha256",
        "preprocess_config_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
    )

    def __init__(self, **values: object) -> None:
        for name in self.__slots__:
            setattr(self, name, values[name])


def _validate_target_free_bindings(
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    *,
    expected_target_v2_artifact_sha256: str,
    expected_target_v2_receipt_sha256: str,
    expected_target_v2_policy_sha256: str,
) -> None:
    protocol.assert_unchanged()
    expected = {
        "verified_target_v2_artifact_sha256": _require_sha256(
            expected_target_v2_artifact_sha256,
            field="expected_target_v2_artifact_sha256",
        ),
        "verified_target_v2_receipt_sha256": _require_sha256(
            expected_target_v2_receipt_sha256,
            field="expected_target_v2_receipt_sha256",
        ),
        "verified_target_v2_policy_sha256": _require_sha256(
            expected_target_v2_policy_sha256,
            field="expected_target_v2_policy_sha256",
        ),
        "split_manifest_sha256": protocol.receipt.split_manifest_sha256,
    }
    for field, value in expected.items():
        if signal.receipt.get(field) != value:
            raise ValueError(f"Signal/OOF target-free binding changed: {field}")


def _load_fold_scalers(
    fold_scaler_specs: Mapping[int, tuple[str | Path, str]],
) -> Mapping[int, ComputedEvolutionScalerArtifact]:
    if set(fold_scaler_specs) != set(_EXPECTED_FOLDS):
        raise ValueError("Source-train OOF requires fold_0..fold_4 scalers exactly once")
    loaded = {
        fold: load_externally_pinned_computed_evolution_scaler_artifact(
            fold_scaler_specs[fold][0],
            oof_fold=fold,
            expected_artifact_sha256=fold_scaler_specs[fold][1],
        )
        for fold in _EXPECTED_FOLDS
    }
    return MappingProxyType(loaded)


def _validate_fold_scaler_lineage(
    scalers: Mapping[int, ComputedEvolutionScalerArtifact],
    protocol: TargetFreeOOFProtocolView,
    signal: VerifiedDeepSOZSignalPreflightBundle,
) -> None:
    if set(scalers) != set(_EXPECTED_FOLDS):
        raise ValueError("Fold scaler set is incomplete")
    current_evolution_sha = _module_sha256(_evolution_module, field="evolution.py")
    current_geometry_sha = _module_sha256(_geometry_module, field="geometry.py")
    for fold in _EXPECTED_FOLDS:
        scaler = scalers[fold]
        if not isinstance(scaler, ComputedEvolutionScalerArtifact):
            raise TypeError("Every OOF scaler must come from the strict artifact loader")
        plan = protocol.fold_plan_receipts[fold]
        receipt = scaler.receipt
        checks = {
            "fold": receipt.oof_fold == fold,
            "split": receipt.split_manifest_sha256
            == protocol.receipt.split_manifest_sha256
            == signal.receipt["split_manifest_sha256"],
            "protocol": receipt.oof_protocol_receipt_sha256
            == protocol.receipt_sha256,
            "plan": receipt.oof_plan_receipt_sha256 == plan.receipt_sha256,
            "held_target": receipt.held_out_target_roster_sha256
            == plan.held_out_target_roster_sha256,
            "held_public": receipt.held_out_public_roster_sha256
            == plan.held_out_public_roster_sha256,
            "authorized_records": receipt.authorized_source_record_roster_sha256
            == plan.authorized_record_roster_sha256,
            "evolution_source": scaler.computation_receipt.evolution_source_sha256
            == current_evolution_sha,
            "geometry_source": scaler.computation_receipt.geometry_source_sha256
            == current_geometry_sha,
            "fit_split": scaler.scaler.receipt.fit_split == SOURCE_TRAIN_SPLIT,
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"Fold {fold} scaler lineage mismatch: {failed}")
        if set(receipt.fit_public_patient_keys) & set(
            plan.held_out_public_patient_keys
        ):
            raise ValueError(f"Fold {fold} scaler fit includes held-out patients")


def _source_train_timeline(
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
) -> tuple[tuple[_SourceTrainTimelineEvent, ...], OffsetAwarePhaseMasks]:
    receipt = signal.receipt
    expected_train = set(protocol.receipt.source_train_patient_ids)
    split_rows = {
        str(row[0]): tuple(str(value) for value in row[1])
        for row in receipt["eligible_split_patient_ids"]
    }
    eligible_train = set(split_rows.get(SOURCE_TRAIN_SPLIT, ()))
    if not eligible_train or not eligible_train <= expected_train:
        raise ValueError("Signal-preflight source-train roster differs from OOF protocol")

    # source-dev/eval/private rows are never consumed.  Accepted and excluded
    # source-train rows are both needed to reconstruct complete record-local
    # previous-event timing without converting an excluded prior event into
    # baseline context.
    accepted = tuple(
        row for row in receipt["events"] if row["model_split"] == SOURCE_TRAIN_SPLIT
    )
    excluded = tuple(
        row
        for row in receipt["exclusions"]
        if row["model_split"] == SOURCE_TRAIN_SPLIT
    )
    observed_patients = {str(row["patient_id"]) for row in accepted}
    if observed_patients != eligible_train:
        raise ValueError("Source-train accepted events do not cover the eligible roster")

    by_record: dict[str, list[Mapping[str, object]]] = {}
    for row in (*accepted, *excluded):
        patient_id = str(row["patient_id"])
        if patient_id not in expected_train:
            raise ValueError("Source-train signal identity is outside OOF protocol")
        by_record.setdefault(str(row["deepsoz_source_record_sha256"]), []).append(row)
    prior_stop: dict[str, float | None] = {}
    timeline_sha: dict[str, str] = {}
    for source_record, rows in by_record.items():
        ordered = tuple(sorted(rows, key=lambda row: int(row["global_event_index"])))
        indices = tuple(int(row["global_event_index"]) for row in ordered)
        if indices != tuple(range(len(ordered))):
            raise ValueError("Source-train signal timeline is incomplete")
        starts = tuple(float(row["global_t0_sec"]) for row in ordered)
        if any(right < left - 1e-6 for left, right in zip(starts, starts[1:])):
            raise ValueError("Source-train signal timeline is not chronological")
        group_payload = {
            "schema_version": "target_free_source_train_record_timeline_v1",
            "signal_preflight_artifact_sha256": signal.artifact_sha256,
            "signal_preflight_receipt_sha256": signal.receipt_sha256,
            "source_record_sha256": source_record,
            "events": [
                [
                    str(row["event_id"]),
                    str(row["event_record_sha256"]),
                    int(row["global_event_index"]),
                    float(row["global_t0_sec"]),
                    float(row["global_stop_sec"]),
                ]
                for row in ordered
            ],
        }
        group_sha = _canonical_sha256(group_payload)
        running: float | None = None
        for row in ordered:
            event_id = str(row["event_id"])
            prior_stop[event_id] = running
            timeline_sha[event_id] = group_sha
            stop = float(row["global_stop_sec"])
            running = stop if running is None else max(running, stop)

    events: list[_SourceTrainTimelineEvent] = []
    for row in sorted(
        accepted, key=lambda value: (str(value["patient_id"]), str(value["event_id"]))
    ):
        event_id = str(row["event_id"])
        patient_id = str(row["patient_id"])
        start = float(row["global_t0_sec"])
        stop = float(row["global_stop_sec"])
        if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
            raise ValueError("Source-train event timing is invalid")
        previous = prior_stop[event_id]
        events.append(
            _SourceTrainTimelineEvent(
                event_id=event_id,
                patient_id=patient_id,
                oof_fold=protocol.fold_for_target(patient_id),
                event_record_sha256=_require_sha256(
                    row["event_record_sha256"], field="event_record_sha256"
                ),
                relative_edf_path=str(row["relative_edf_path"]),
                edf_sha256=_require_sha256(row["edf_sha256"], field="edf_sha256"),
                global_t0_sec=start,
                global_stop_sec=stop,
                seizure_duration_sec=stop - start,
                previous_seizure_gap_sec=(
                    None if previous is None else max(0.0, start - previous)
                ),
                timeline_sha256=timeline_sha[event_id],
                preprocess_config_sha256=_require_sha256(
                    row["preprocess_config_sha256"], field="preprocess_config_sha256"
                ),
                edf_receipt_sha256=_require_sha256(
                    row["edf_receipt_sha256"], field="edf_receipt_sha256"
                ),
                signal_receipt_sha256=_require_sha256(
                    row["signal_receipt_sha256"], field="signal_receipt_sha256"
                ),
                processed_window_sha256=_require_sha256(
                    row["processed_window_sha256"], field="processed_window_sha256"
                ),
            )
        )
    if not events:
        raise ValueError("No source-train events are eligible for OOF V+A/Q")
    phases = build_offset_aware_phase_masks(
        [event.seizure_duration_sec for event in events],
        offset_trustworthy=[True] * len(events),
        previous_seizure_gap_sec=[event.previous_seizure_gap_sec for event in events],
        previous_timeline_trustworthy=[True] * len(events),
    )
    return tuple(events), phases


def preflight_source_train_oof_vaq_inputs(
    *,
    signal_preflight_bundle: str | Path,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
    oof_protocol_bundle: str | Path,
    expected_oof_protocol_artifact_sha256: str,
    expected_oof_protocol_receipt_sha256: str,
    expected_target_v2_artifact_sha256: str,
    expected_target_v2_receipt_sha256: str,
    expected_target_v2_policy_sha256: str,
    fold_scaler_specs: Mapping[int, tuple[str | Path, str]],
    tusz_root: str | Path,
) -> dict[str, object]:
    signal = load_bound_deepsoz_signal_preflight_artifact(
        signal_preflight_bundle,
        expected_artifact_sha256=expected_signal_artifact_sha256,
        expected_receipt_sha256=expected_signal_receipt_sha256,
    )
    protocol = load_target_free_ictal_oof_protocol(
        oof_protocol_bundle,
        expected_artifact_sha256=expected_oof_protocol_artifact_sha256,
        expected_protocol_receipt_sha256=expected_oof_protocol_receipt_sha256,
    )
    _validate_target_free_bindings(
        signal,
        protocol,
        expected_target_v2_artifact_sha256=expected_target_v2_artifact_sha256,
        expected_target_v2_receipt_sha256=expected_target_v2_receipt_sha256,
        expected_target_v2_policy_sha256=expected_target_v2_policy_sha256,
    )
    scalers = _load_fold_scalers(fold_scaler_specs)
    _validate_fold_scaler_lineage(scalers, protocol, signal)
    events, _ = _source_train_timeline(signal, protocol)
    root = Path(tusz_root)
    for event in events:
        _safe_source_file(root, event.relative_edf_path)
    patients = tuple(sorted({event.patient_id for event in events}))
    patient_folds = tuple((patient, protocol.fold_for_target(patient)) for patient in patients)
    fold_patient_counts = tuple(
        (fold, sum(value == fold for _, value in patient_folds)) for fold in _EXPECTED_FOLDS
    )
    fold_event_counts = tuple(
        (fold, sum(event.oof_fold == fold for event in events)) for fold in _EXPECTED_FOLDS
    )
    return {
        "schema_version": SOURCE_TRAIN_OOF_VAQ_SCHEMA,
        "status": "ready",
        "purpose": SOURCE_TRAIN_OOF_VAQ_PURPOSE,
        "model_split": SOURCE_TRAIN_SPLIT,
        "event_count": len(events),
        "patient_count": len(patients),
        "fold_patient_counts": fold_patient_counts,
        "fold_event_counts": fold_event_counts,
        "patient_fold_assignment_sha256": _canonical_sha256(patient_folds),
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "target_vectors_loaded": False,
        "source_dev_events_used": False,
        "source_eval_events_used": False,
        "private_events_used": False,
        "final_scaler_used": False,
        "candidate_reasoner_input_authorized": True,
        "formal_promotion_authorized": False,
    }


def materialize_source_train_oof_vaq_evidence(
    *,
    signal_preflight_bundle: str | Path,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
    oof_protocol_bundle: str | Path,
    expected_oof_protocol_artifact_sha256: str,
    expected_oof_protocol_receipt_sha256: str,
    expected_target_v2_artifact_sha256: str,
    expected_target_v2_receipt_sha256: str,
    expected_target_v2_policy_sha256: str,
    fold_scaler_specs: Mapping[int, tuple[str | Path, str]],
    tusz_root: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    signal = load_bound_deepsoz_signal_preflight_artifact(
        signal_preflight_bundle,
        expected_artifact_sha256=expected_signal_artifact_sha256,
        expected_receipt_sha256=expected_signal_receipt_sha256,
    )
    protocol = load_target_free_ictal_oof_protocol(
        oof_protocol_bundle,
        expected_artifact_sha256=expected_oof_protocol_artifact_sha256,
        expected_protocol_receipt_sha256=expected_oof_protocol_receipt_sha256,
    )
    _validate_target_free_bindings(
        signal,
        protocol,
        expected_target_v2_artifact_sha256=expected_target_v2_artifact_sha256,
        expected_target_v2_receipt_sha256=expected_target_v2_receipt_sha256,
        expected_target_v2_policy_sha256=expected_target_v2_policy_sha256,
    )
    scalers = _load_fold_scalers(fold_scaler_specs)
    _validate_fold_scaler_lineage(scalers, protocol, signal)
    events, phases = _source_train_timeline(signal, protocol)
    config = CausalEDFConfig(**dict(signal.receipt["preprocess_config"]))
    root = Path(tusz_root)

    raw_rows: list[torch.Tensor] = []
    scaled_rows: list[torch.Tensor] = []
    quality_rows = []
    event_rows: list[dict[str, object]] = []
    for index, event in enumerate(events):
        relative_path, source = _safe_source_file(root, event.relative_edf_path)
        loaded = load_standard19_edf_event(
            source, event.global_t0_sec, config=config
        )
        checks = {
            "edf": loaded.edf_receipt.edf_sha256 == event.edf_sha256,
            "window": _signal_window_sha256(loaded.window.data)
            == event.processed_window_sha256,
            "edf_receipt": _canonical_sha256(asdict(loaded.edf_receipt))
            == event.edf_receipt_sha256,
            "signal_receipt": _canonical_sha256(asdict(loaded.signal_receipt))
            == event.signal_receipt_sha256,
            "preprocess": event.preprocess_config_sha256
            == signal.receipt["preprocess_config_sha256"],
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"Source-train replay failed for {event.event_id}: {failed}")
        eeg = loaded.window.data.detach().cpu().unsqueeze(0)
        evolution = compute_temporal_evolution_descriptors(eeg)
        selected_scaler = scalers[event.oof_fold]
        scaled = selected_scaler.scaler.transform(
            evolution.descriptors, evolution.mask
        )
        quality = compute_target_free_quality_evidence(eeg)
        raw_rows.append(evolution.descriptors[0])
        scaled_rows.append(scaled[0])
        quality_rows.append(quality)
        abstain = bool((quality.tile_abstain[0] & phases.ictal_phase_mask[index]).any())
        event_rows.append(
            {
                "schema_version": SOURCE_TRAIN_OOF_VAQ_EVENT_SCHEMA,
                "event_id": event.event_id,
                "patient_id": event.patient_id,
                "model_split": SOURCE_TRAIN_SPLIT,
                "oof_fold": event.oof_fold,
                "relative_edf_path": relative_path,
                "event_record_sha256": event.event_record_sha256,
                "edf_sha256": event.edf_sha256,
                "processed_window_sha256": event.processed_window_sha256,
                "timeline_sha256": event.timeline_sha256,
                "global_t0_sec": event.global_t0_sec,
                "global_stop_sec": event.global_stop_sec,
                "seizure_duration_sec": event.seizure_duration_sec,
                "previous_seizure_gap_sec": event.previous_seizure_gap_sec,
                "evolution_scaler_artifact_sha256": selected_scaler.artifact_sha256,
                "evolution_scaler_receipt_sha256": selected_scaler.receipt.receipt_sha256,
                "evolution_raw_sha256": _tensor_sha256(evolution.descriptors[0]),
                "evolution_scaled_sha256": _tensor_sha256(scaled[0]),
                "quality_diagnostics_sha256": _tensor_sha256(quality.diagnostics[0]),
                "artifact_burden_sha256": _tensor_sha256(quality.artifact_burden[0]),
                "reliability_sha256": _tensor_sha256(quality.reliability[0]),
                "abstention_recommended": abstain,
            }
        )

    evolution_raw = torch.stack(raw_rows)
    evolution_scaled = torch.stack(scaled_rows)
    ordered_delta = torch.zeros_like(evolution_scaled)
    ordered_delta[:, :, 1:] = evolution_scaled[:, :, 1:] - evolution_scaled[:, :, :-1]
    evolution_mask = torch.ones(
        (len(events), N_STANDARD_CHANNELS, EVOLUTION_N_TILES), dtype=torch.bool
    )
    delta_mask = evolution_mask.clone()
    delta_mask[:, :, 0] = False
    phase_tensors = _phase_tensors(phases)
    tile_abstain = torch.cat([value.tile_abstain for value in quality_rows], dim=0)
    tensors = {
        "evolution_raw": evolution_raw,
        "evolution_scaled": evolution_scaled,
        "evolution_ordered_delta": ordered_delta,
        "evolution_mask": evolution_mask,
        "evolution_delta_mask": delta_mask,
        "quality_diagnostics": torch.cat(
            [value.diagnostics for value in quality_rows], dim=0
        ),
        "quality_burden_components": torch.cat(
            [value.burden_components for value in quality_rows], dim=0
        ),
        "artifact_burden": torch.cat(
            [value.artifact_burden for value in quality_rows], dim=0
        ),
        "reliability": torch.cat([value.reliability for value in quality_rows], dim=0),
        "reliable_mask": torch.cat(
            [value.reliable_mask for value in quality_rows], dim=0
        ),
        "tile_abstain": tile_abstain,
        "event_abstain": (tile_abstain & phase_tensors["ictal_phase_mask"]).any(dim=1),
        "tile_start_sec": torch.arange(-12.0, 48.0, 4.0, dtype=torch.float64),
        **phase_tensors,
    }
    _validate_tensor_payload(tensors)

    patients = tuple(sorted({event.patient_id for event in events}))
    patient_folds = tuple((patient, protocol.fold_for_target(patient)) for patient in patients)
    fold_lineage = [
        {
            "oof_fold": fold,
            "scaler_artifact_sha256": scalers[fold].artifact_sha256,
            "scaler_receipt_sha256": scalers[fold].receipt.receipt_sha256,
            "oof_plan_receipt_sha256": scalers[fold].receipt.oof_plan_receipt_sha256,
            "held_out_target_roster_sha256": scalers[fold].receipt.held_out_target_roster_sha256,
            "held_out_public_roster_sha256": scalers[fold].receipt.held_out_public_roster_sha256,
            "fit_public_patient_roster_sha256": scalers[fold].receipt.fit_patient_roster_sha256,
        }
        for fold in _EXPECTED_FOLDS
    ]
    quality_policy = {
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
    }
    events_payload = {
        "schema_version": "soz_target_free_source_train_oof_vaq_event_roster_v1",
        "event_count": len(event_rows),
        "events": event_rows,
    }
    manifest_core = {
        "schema_version": SOURCE_TRAIN_OOF_VAQ_MANIFEST_SCHEMA,
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "purpose": SOURCE_TRAIN_OOF_VAQ_PURPOSE,
        "model_split": SOURCE_TRAIN_SPLIT,
        "development_only": True,
        "candidate_reasoner_input_authorized": True,
        "formal_promotion_authorized": False,
        "evaluation_result_authorized": False,
        "source_dev_events_used": False,
        "source_eval_events_used": False,
        "private_events_used": False,
        "final_scaler_used": False,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "target_vectors_loaded": False,
        "event_count": len(events),
        "patient_count": len(patients),
        "event_roster_sha256": _canonical_sha256(tuple(event.event_id for event in events)),
        "patient_roster_sha256": _canonical_sha256(patients),
        "patient_fold_assignments": [list(value) for value in patient_folds],
        "patient_fold_assignment_sha256": _canonical_sha256(patient_folds),
        "fold_event_counts": [
            [fold, sum(event.oof_fold == fold for event in events)]
            for fold in _EXPECTED_FOLDS
        ],
        "fold_scaler_lineage": fold_lineage,
        "oof_protocol_artifact_sha256": protocol.artifact_sha256,
        "oof_protocol_receipt_sha256": protocol.receipt_sha256,
        "signal_preflight_artifact_sha256": signal.artifact_sha256,
        "signal_preflight_receipt_sha256": signal.receipt_sha256,
        "split_manifest_sha256": signal.receipt["split_manifest_sha256"],
        "evolution_feature_names": list(EVOLUTION_FEATURES),
        "evolution_feature_schema_sha256": EVOLUTION_FEATURE_SCHEMA_SHA256,
        "evolution_semantics": (
            "observable_descriptors_and_adjacent_ordered_differences_only;"
            "not_origin_not_propagation_not_soz"
        ),
        "quality_policy": quality_policy,
        "quality_policy_sha256": _canonical_sha256(quality_policy),
        "producer_source_sha256": _module_sha256(
            __import__(__name__, fromlist=["unused"]), field="development_vaq_oof.py"
        ),
        "quality_producer_source_sha256": _module_sha256(
            __import__("src.soz.development_vaq", fromlist=["unused"]),
            field="development_vaq.py",
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
        Path(output_directory), tensors, events_payload, manifest_core
    )
    return {
        "path": str(path),
        "manifest_sha256": manifest_sha,
        "event_count": len(events),
        "patient_count": len(patients),
        "abstention_recommended_count": int(tensors["event_abstain"].sum()),
        "final_scaler_used": False,
        "formal_promotion_authorized": False,
    }


_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "serialization",
        "purpose",
        "model_split",
        "development_only",
        "candidate_reasoner_input_authorized",
        "formal_promotion_authorized",
        "evaluation_result_authorized",
        "source_dev_events_used",
        "source_eval_events_used",
        "private_events_used",
        "final_scaler_used",
        "contains_soz_labels",
        "contains_tusz_channel_targets_or_masks",
        "target_vectors_loaded",
        "event_count",
        "patient_count",
        "event_roster_sha256",
        "patient_roster_sha256",
        "patient_fold_assignments",
        "patient_fold_assignment_sha256",
        "fold_event_counts",
        "fold_scaler_lineage",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "signal_preflight_artifact_sha256",
        "signal_preflight_receipt_sha256",
        "split_manifest_sha256",
        "evolution_feature_names",
        "evolution_feature_schema_sha256",
        "evolution_semantics",
        "quality_policy",
        "quality_policy_sha256",
        "producer_source_sha256",
        "quality_producer_source_sha256",
        "evolution_source_sha256",
        "geometry_source_sha256",
        "tensor_specs",
        "files",
    }
)


def load_source_train_oof_vaq_evidence(
    bundle_directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[Mapping[str, object], Mapping[str, object], dict[str, torch.Tensor]]:
    """Strict closed loader for target-free source-train patient-OOF V+A/Q."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    bundle = Path(os.path.abspath(bundle_directory))
    expected_names = {"manifest.json", "events.json", "vaq.safetensors"}
    if (
        not bundle.is_dir()
        or bundle.is_symlink()
        or {path.name for path in bundle.iterdir()} != expected_names
    ):
        raise ValueError("Source-train OOF V+A/Q bundle violates its closed schema")
    manifest_raw = (bundle / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Source-train OOF V+A/Q manifest SHA mismatch")
    manifest = _strict_json(manifest_raw, field="manifest")
    if _canonical_bytes(manifest) != manifest_raw:
        raise ValueError("Source-train OOF V+A/Q manifest is not canonical JSON")
    if not isinstance(manifest, dict) or set(manifest) != set(_MANIFEST_REQUIRED_FIELDS):
        raise ValueError("Source-train OOF V+A/Q manifest fields changed")
    boundary = {
        "schema_version": SOURCE_TRAIN_OOF_VAQ_MANIFEST_SCHEMA,
        "purpose": SOURCE_TRAIN_OOF_VAQ_PURPOSE,
        "model_split": SOURCE_TRAIN_SPLIT,
        "development_only": True,
        "candidate_reasoner_input_authorized": True,
        "formal_promotion_authorized": False,
        "evaluation_result_authorized": False,
        "source_dev_events_used": False,
        "source_eval_events_used": False,
        "private_events_used": False,
        "final_scaler_used": False,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "target_vectors_loaded": False,
    }
    for field, expected in boundary.items():
        if manifest[field] != expected:
            raise ValueError(f"Source-train OOF boundary changed: {field}")
    assignments = manifest["patient_fold_assignments"]
    if (
        not isinstance(assignments, list)
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or row[1] not in _EXPECTED_FOLDS
            for row in assignments
        )
        or len({row[0] for row in assignments}) != len(assignments)
    ):
        raise ValueError("Patient-fold assignments are invalid")
    assignment_tuple = tuple((row[0], int(row[1])) for row in assignments)
    if _canonical_sha256(assignment_tuple) != manifest["patient_fold_assignment_sha256"]:
        raise ValueError("Patient-fold assignment SHA mismatch")
    fold_lineage = manifest["fold_scaler_lineage"]
    if not isinstance(fold_lineage, list) or [row.get("oof_fold") for row in fold_lineage] != list(
        _EXPECTED_FOLDS
    ):
        raise ValueError("Fold scaler lineage must contain fold_0..fold_4 only")

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {"events.json", "vaq.safetensors"}:
        raise ValueError("Source-train OOF file receipt changed")
    for name, record in files.items():
        path = bundle / name
        if path.is_symlink() or not path.is_file() or _file_sha256(path) != record["sha256"]:
            raise ValueError(f"Source-train OOF payload SHA mismatch: {name}")
    events_raw = (bundle / "events.json").read_bytes()
    events_payload = _strict_json(events_raw, field="events")
    if _canonical_bytes(events_payload) != events_raw or not isinstance(events_payload, dict):
        raise ValueError("Source-train OOF events are not canonical JSON")
    if set(events_payload) != {"schema_version", "event_count", "events"}:
        raise ValueError("Source-train OOF event roster fields changed")
    rows = events_payload["events"]
    assignment_by_patient = dict(assignment_tuple)
    if (
        not isinstance(rows, list)
        or len(rows) != manifest["event_count"]
        or any(
            not isinstance(row, dict)
            or row.get("model_split") != SOURCE_TRAIN_SPLIT
            or row.get("oof_fold") != assignment_by_patient.get(row.get("patient_id"))
            for row in rows
        )
    ):
        raise ValueError("Source-train OOF event fold assignment changed")
    tensors = load_file(str(bundle / "vaq.safetensors"), device="cpu")
    _validate_tensor_payload(tensors)
    if _tensor_specs(tensors) != manifest["tensor_specs"]:
        raise ValueError("Source-train OOF tensor specs changed")
    if tensors["evolution_raw"].shape[0] != manifest["event_count"]:
        raise ValueError("Source-train OOF tensor event count changed")
    return manifest, events_payload, tensors


__all__ = [
    "SOURCE_TRAIN_OOF_VAQ_MANIFEST_SCHEMA",
    "SOURCE_TRAIN_OOF_VAQ_PURPOSE",
    "SOURCE_TRAIN_OOF_VAQ_SCHEMA",
    "load_source_train_oof_vaq_evidence",
    "materialize_source_train_oof_vaq_evidence",
    "preflight_source_train_oof_vaq_inputs",
]
