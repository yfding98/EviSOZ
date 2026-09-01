#!/usr/bin/env python3
"""Close one real public-TUSZ P0 -> segmental-training smoke path.

This driver is intentionally small and research-only.  It consumes frozen,
EEG-only DeepSOZ posterior artifacts, decodes detector proposals, materializes
adaptive EEG windows and P0 tokens, and writes a target-free candidate roster.
Only after that roster has been durably written does it open the public TUSZ
``.csv_bi`` interval sidecars and create event/boundary supervision.

The source-train row is used for exactly one optimizer smoke step.  A public
source-dev row is carried in the same patient-disjoint disk manifest and is
used only for a no-gradient calibration forward.  No channel annotations,
EDF annotations, spreadsheets, clinical text, private data, SOZ labels or
Qwen/report paths are accepted by this CLI.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.adaptive_event_window import (  # noqa: E402
    derive_adaptive_event_analysis_window,
)
from src.clinical_eeg_long_recording.adaptive_search_materialization import (  # noqa: E402
    materialize_adaptive_eeg_search,
)
from src.clinical_eeg_long_recording.ba_ieg_event_model_input_projection_v1 import (  # noqa: E402
    BAIEGEventModelInputProjectionV1,
    project_ba_ieg_event_model_input_v1,
)
from src.clinical_eeg_long_recording.ba_ieg_permission_split_segmental_disk_training_v1 import (  # noqa: E402
    BAIEGPermissionSplitSegmentalTrainerV1,
    BAIEGSegmentalDiskDatasetV1,
    ba_ieg_segmental_event_input_metadata_v1,
    ba_ieg_segmental_event_target_payload_v1,
    ba_ieg_segmental_event_tensor_arrays_v1,
    build_ba_ieg_segmental_disk_loader_v1,
    build_ba_ieg_segmental_disk_manifest_v1,
)
from src.clinical_eeg_long_recording.ba_ieg_permission_split_segmental_state_model_v1 import (  # noqa: E402
    BAIEGPermissionSplitSegmentalStateModel,
)
from src.clinical_eeg_long_recording.ba_ieg_training_contract import (  # noqa: E402
    BAIEGP0TokenizationPolicy,
    materialize_ba_ieg_p0_event_tokens,
)
from src.clinical_eeg_long_recording.ba_ieg_tusz_candidate_interval_target_materializer_v1 import (  # noqa: E402
    ValidatedBAIEGTUSZCandidateEnvelopeV1,
    build_ba_ieg_tusz_public_interval_reference_v1,
    freeze_ba_ieg_tusz_candidate_envelope_after_tokenization_v1,
    materialize_ba_ieg_tusz_candidate_interval_target_v1,
)
from src.clinical_eeg_long_recording.canonical_detector_input_binding import (  # noqa: E402
    validate_canonical_detector_input_binding,
)
from src.clinical_eeg_long_recording.canonical_edf_materialization import (  # noqa: E402
    CanonicalEDFConfig,
    load_canonical_edf_views,
    validate_canonical_edf_materialization,
)
from src.clinical_eeg_long_recording.continuous_detection import (  # noqa: E402
    decode_continuous_seizure_posterior,
)
from src.clinical_eeg_long_recording.deepsoz_posterior_batch_validation import (  # noqa: E402
    DEEPSOZ_BATCH_SCHEMA_VERSION,
    DEEPSOZ_DECISION_AVAILABILITY,
    DEEPSOZ_MATERIALIZED_OOF_SCHEMA_VERSION,
    DEEPSOZ_PARTIAL_TAIL_POLICY,
    DEEPSOZ_PROVIDER_ID,
    DEEPSOZ_TIME_SUPPORT_SCHEMA_VERSION,
    _ARTIFACT_FIELDS,
    _BATCH_FIELDS,
    _INDEX_FIELDS,
    _SCOPE,
    _file_sha256 as _validated_file_sha256,
    _safe_relative_posterior,
    _sha256 as _deepsoz_sha256,
    _validate_fold_assignment,
    _validate_runtime_receipt,
    _validate_time_support,
    _validate_timeline,
)
from src.clinical_eeg_long_recording.detection import (  # noqa: E402
    build_long_term_detection_manifest,
)
from src.clinical_eeg_long_recording.detector_provider_contract import (  # noqa: E402
    validate_provider_registry,
)


SCHEMA_VERSION = "ba_ieg_real_public_tusz_segmental_smoke_v1"
CANDIDATE_ROSTER_SCHEMA_VERSION = "ba_ieg_tusz_target_independent_candidate_roster_v1"
REFERENCE_PARSE_SCHEMA_VERSION = "tusz_csv_bi_event_boundary_parse_v1"
DEFAULT_DECODER_POLICY: dict[str, Any] = {
    "on_threshold": 0.4,
    "off_threshold": 0.2,
    "minimum_on_windows": 1,
    "minimum_off_windows": 2,
    "merge_gap_seconds": 2.0,
    "maximum_coverage_gap_seconds": 2.01,
    "minimum_event_seconds": 2.0,
    "force_minimum_candidate_count": False,
    "anchor_semantics": "first_persistent_on_threshold_navigation_coordinate",
}
_ALLOWED_SPLITS = {"source_train": "train", "source_dev": "dev"}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _atomic_json(path: Path, value: object) -> None:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
        os.chmod(target, 0o600)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".npz", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
        os.chmod(target, 0o600)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@dataclass(frozen=True)
class VerifiedPosteriorRecord:
    batch_root: Path
    batch: dict[str, Any]
    index: dict[str, Any]
    artifact: dict[str, Any]
    posterior_path: Path
    posterior_file_sha256: str
    provider_receipt: dict[str, Any]


def _load_json_bytes(path: Path, context: str) -> tuple[object, bytes]:
    source = _regular_file(path, context)
    payload = source.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error
    return value, payload


def load_verified_posterior_record_without_references(
    batch_root: Path,
    *,
    ordinal: int,
    expected_split: str,
    provider_registry_path: Path,
) -> VerifiedPosteriorRecord:
    """Validate one frozen posterior row without accepting a reference path."""

    if expected_split not in _ALLOWED_SPLITS:
        raise ValueError("posterior subset accepts source_train/source_dev only")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise ValueError("posterior ordinal must be positive")
    raw_root = batch_root
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("posterior batch root must be a regular directory")
    root = raw_root.resolve(strict=True)
    batch_raw, batch_bytes = _load_json_bytes(
        root / "batch_receipt.json", "posterior batch receipt"
    )
    if type(batch_raw) is not dict or set(batch_raw) != _BATCH_FIELDS:
        raise ValueError("posterior batch receipt schema drifted")
    batch = deepcopy(batch_raw)
    expected_false = (
        "detector_imputed_channels_clinical_evidence_eligible",
        "real_time_latency_metric_authorized",
        "silent_time_padding_used",
        "edf_annotations_used",
        "label_bearing_manifest_fields_retained_for_inference",
        "seizure_or_soz_labels_used_for_inference",
        "production_qualified",
        "sota_claim_authorized",
    )
    if (
        batch["schema_version"] != DEEPSOZ_BATCH_SCHEMA_VERSION
        or batch["provider_id"] != DEEPSOZ_PROVIDER_ID
        or batch["selected_split"] != expected_split
        or batch["all_selected_records_materialized"] is not True
        or batch["materialized_oof_schema_version"]
        != DEEPSOZ_MATERIALIZED_OOF_SCHEMA_VERSION
        or batch["posterior_time_support_schema_version"]
        != DEEPSOZ_TIME_SUPPORT_SCHEMA_VERSION
        or batch["posterior_only_operating_point_not_applied"] is not True
        or batch["all_posteriors_have_explicit_physical_time_support"] is not True
        or batch["all_posteriors_offline_future_dependent"] is not True
        or batch["decision_availability_semantics"] != DEEPSOZ_DECISION_AVAILABILITY
        or batch["partial_tail_policy"] != DEEPSOZ_PARTIAL_TAIL_POLICY
        or any(batch[field] is not False for field in expected_false)
    ):
        raise ValueError("posterior batch scope or completion gate drifted")
    batch_id_source = deepcopy(batch)
    batch_id_source["receipt_id"] = "DEEPSOZ-BATCH-PENDING"
    if batch["receipt_id"] != "DSZBATCH-" + _deepsoz_sha256(batch_id_source)[:24]:
        raise ValueError("posterior batch receipt ID drifted")

    index_path = _regular_file(root / "posterior_index.jsonl", "posterior index")
    index_bytes = index_path.read_bytes()
    if batch["index_sha256"] != hashlib.sha256(index_bytes).hexdigest():
        raise ValueError("posterior batch index hash drifted")
    lines = [line for line in index_bytes.splitlines() if line.strip()]
    if len(lines) != batch["recording_count"] or ordinal > len(lines):
        raise ValueError("posterior ordinal is outside the frozen batch inventory")
    try:
        index = json.loads(lines[ordinal - 1])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("posterior index row is invalid JSON") from error
    if type(index) is not dict or set(index) != _INDEX_FIELDS:
        raise ValueError("posterior index row schema drifted")
    if index["ordinal"] != ordinal or index["model_split"] != expected_split:
        raise ValueError("posterior index ordinal/split drifted")
    split_prefix = _ALLOWED_SPLITS[expected_split] + "/"
    recording_id = str(index["recording_id"])
    relative_record = PurePosixPath(recording_id)
    if (
        relative_record.is_absolute()
        or ".." in relative_record.parts
        or not recording_id.startswith(split_prefix)
        or not recording_id.endswith(".edf")
        or "eval/" in recording_id.lower()
    ):
        raise ValueError("posterior recording identity escaped the allowed split")

    fold_lookup, fold_receipt_sha256 = _validate_fold_assignment(
        batch["fold_assignment_receipt"]
    )
    patient_id = str(int(str(index["deepsoz_patient_id"])))
    folds = tuple(index["held_out_fold_indices"])
    if fold_lookup.get(patient_id) != folds:
        raise ValueError("posterior index folds disagree with OOF assignment")
    posterior_path = _safe_relative_posterior(root, index["posterior_relative_path"])
    artifact_bytes = posterior_path.read_bytes()
    posterior_hash = hashlib.sha256(artifact_bytes).hexdigest()
    if posterior_hash != index["posterior_file_sha256"]:
        raise ValueError("posterior artifact file hash drifted")
    try:
        artifact = json.loads(artifact_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("posterior artifact is invalid JSON") from error
    if type(artifact) is not dict or set(artifact) != _ARTIFACT_FIELDS:
        raise ValueError("posterior artifact schema drifted")
    if (
        artifact["materialization_schema_version"]
        != DEEPSOZ_MATERIALIZED_OOF_SCHEMA_VERSION
        or artifact["provider_id"] != DEEPSOZ_PROVIDER_ID
        or artifact["recording_id"] != recording_id
        or artifact["deepsoz_patient_id"] != patient_id
        or tuple(artifact["held_out_fold_indices"]) != folds
        or artifact["held_out_repeat_count"] != len(folds)
        or artifact["fold_assignment_receipt_sha256"] != fold_receipt_sha256
        or artifact["weights_manifest_sha256"] != batch["weights_manifest_sha256"]
        or artifact["adapter_code_sha256"] != batch["adapter_code_sha256"]
        or artifact["scope_receipt"] != _SCOPE
        or artifact["posterior_artifact_id"] != index["posterior_artifact_id"]
    ):
        raise ValueError("posterior artifact identity/provenance drifted")
    artifact_id_source = deepcopy(artifact)
    artifact_id_source["posterior_artifact_id"] = "DEEPSOZ-OOF-POSTERIOR-PENDING"
    if artifact["posterior_artifact_id"] != (
        "DSZOOF-" + _deepsoz_sha256(artifact_id_source)[:24]
    ):
        raise ValueError("posterior artifact content ID drifted")
    duration = float(artifact["recording_duration_seconds"])
    timeline = _validate_timeline(artifact["posterior_timeline"], duration=duration)
    if float(index["recording_duration_seconds"]) != duration or index[
        "timeline_window_count"
    ] != len(timeline):
        raise ValueError("posterior physical timeline disagrees with its index")
    binding = validate_canonical_detector_input_binding(
        artifact["canonical_detector_input_binding"]
    )
    if (
        binding["provider_id"] != DEEPSOZ_PROVIDER_ID
        or binding["binding_id"] != index["canonical_detector_input_binding_id"]
        or binding["receipt_sha256"]
        != index["canonical_detector_input_binding_receipt_sha256"]
        or binding["canonical_signal_id"] != index["canonical_signal_id"]
    ):
        raise ValueError("posterior canonical physical binding drifted")
    _validate_time_support(
        artifact["posterior_time_support_receipt"],
        recording_id=recording_id,
        duration=duration,
        timeline=timeline,
    )
    _validate_runtime_receipt(
        artifact["posterior_runtime_receipt"],
        recording_id=recording_id,
        expected_execution_modes={"new_oof_inference"},
    )
    _validate_runtime_receipt(
        index["current_run_runtime_receipt"],
        recording_id=recording_id,
        expected_execution_modes={"new_oof_inference", "resume_validation_only"},
    )

    registry_raw, _ = _load_json_bytes(
        provider_registry_path, "detector provider registry"
    )
    registry = validate_provider_registry(registry_raw)
    definitions = {
        row["execution_definition"]["provider_id"]: row["execution_definition"]
        for row in registry["providers"]
    }
    definition = definitions.get(DEEPSOZ_PROVIDER_ID)
    if (
        definition is None
        or definition["weights_manifest_sha256"] != artifact["weights_manifest_sha256"]
        or definition["adapter_code_sha256"] != artifact["adapter_code_sha256"]
    ):
        raise ValueError("posterior code/weights drifted from provider registry")
    provider_receipt = {
        "provider_id": DEEPSOZ_PROVIDER_ID,
        "model_family": definition["model_family"],
        "checkpoint_sha256": definition["weights_manifest_sha256"],
        "code_sha256": definition["adapter_code_sha256"],
        "training_corpus": definition["training_corpus"],
        "posterior_calibration_status": definition["posterior_calibration_status"],
        "deployment_qualification_status": "research_candidate_failed_gate",
        "annotations_used_for_current_recording": False,
        "labels_used_for_current_recording": False,
    }
    return VerifiedPosteriorRecord(
        batch_root=root,
        batch=batch,
        index=index,
        artifact=artifact,
        posterior_path=posterior_path,
        posterior_file_sha256=posterior_hash,
        provider_receipt=provider_receipt,
    )


def _resolve_tusz_edf(tusz_root: Path, recording_id: str) -> Path:
    root = tusz_root.resolve(strict=True)
    relative = PurePosixPath(recording_id)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("TUSZ EDF identity is unsafe")
    candidate = _regular_file(root.joinpath(*relative.parts), "public TUSZ EDF")
    if root not in candidate.parents:
        raise ValueError("TUSZ EDF escaped the pinned public root")
    return candidate


def _patient_uid(recording_id: str, model_split: str) -> str:
    parts = PurePosixPath(recording_id).parts
    if len(parts) < 2 or parts[0] != _ALLOWED_SPLITS[model_split]:
        raise ValueError("TUSZ recording lacks a stable public patient identity")
    return f"TUSZ-{model_split.upper()}-{parts[1].upper()}"


def _build_long_detection(
    verified: VerifiedPosteriorRecord,
    *,
    edf_sha256: str,
    patient_uid: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = verified.artifact
    analysis_recording_id = (
        "TUSZREC-"
        + hashlib.sha256(artifact["recording_id"].encode("utf-8")).hexdigest()[:24]
    )
    decoded = decode_continuous_seizure_posterior(
        recording_id=analysis_recording_id,
        source_signal_sha256=artifact["source_signal_tensor_sha256"],
        recording_duration_seconds=artifact["recording_duration_seconds"],
        provider_receipt=verified.provider_receipt,
        posterior_timeline=artifact["posterior_timeline"],
        policy=DEFAULT_DECODER_POLICY,
    )
    if not decoded["event_proposals"]:
        raise ValueError("selected target-free posterior row produced zero candidates")
    detector_receipt = {
        "detector_id": DEEPSOZ_PROVIDER_ID,
        "detector_role": "heuristic_preselector",
        "weights_sha256": artifact["weights_manifest_sha256"],
        "code_sha256": artifact["adapter_code_sha256"],
        "policy_sha256": _canonical_sha256(DEFAULT_DECODER_POLICY),
        "operating_point": {
            "operating_point_id": "DEEPSOZ-GRID-040-020-SMOKE",
            "threshold": DEFAULT_DECODER_POLICY["on_threshold"],
            "score_direction": "greater_or_equal",
            "selection_source": "engineering_heuristic_frozen",
            "frozen_before_recording": True,
        },
        "promotion_status": "not_evaluated_for_deployment",
        "promotion_receipt_sha256": None,
        "annotations_used": False,
        "labels_used": False,
    }
    raw = [
        {
            "start_offset_seconds": row["start_offset_seconds"],
            "stop_offset_seconds": row["stop_offset_seconds"],
            "anchor_offset_seconds": row["anchor_offset_seconds"],
            "score": row["peak_probability"],
            # DeepSOZ preprocessing and model context are offline/full-record.
            "decision_available_offset_seconds": artifact["recording_duration_seconds"],
            "support_window_ids": row["support_window_ids"],
        }
        for row in decoded["event_proposals"]
    ]
    manifest = build_long_term_detection_manifest(
        recording_id=analysis_recording_id,
        patient_pseudonym=patient_uid,
        source_signal_sha256=edf_sha256,
        recording_duration_seconds=artifact["recording_duration_seconds"],
        detector_receipt=detector_receipt,
        raw_alarm_observations=raw,
        merge_gap_seconds=DEFAULT_DECODER_POLICY["merge_gap_seconds"],
        max_selected_candidates=None,
    )
    return decoded, manifest


@dataclass
class TargetFreeEvent:
    model_split: str
    patient_uid: str
    edf_path: Path
    reference_relative_path: str
    verified: VerifiedPosteriorRecord
    detection_manifest: dict[str, Any]
    adaptive_artifact: dict[str, Any]
    adaptive_event: dict[str, Any]
    adaptive_window: dict[str, Any]
    detector_candidate: dict[str, Any]
    projection: BAIEGEventModelInputProjectionV1
    input_metadata_path: Path
    input_tensors_path: Path
    adaptive_acquisition_receipt_sha256: str
    candidate_envelope: ValidatedBAIEGTUSZCandidateEnvelopeV1 | None = None


def _candidate_event_id(
    recording_id: str, model_split: str, candidate_index: int
) -> str:
    digest = hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:16]
    return f"BAIEG-{model_split.upper()}-{digest}-E{candidate_index:04d}"


def materialize_target_free_record(
    verified: VerifiedPosteriorRecord,
    *,
    model_split: str,
    tusz_root: Path,
    output_root: Path,
    maximum_events: int,
) -> list[TargetFreeEvent]:
    """Materialize EEG inputs; this function has no reference-path argument."""

    recording_id = verified.artifact["recording_id"]
    edf_path = _resolve_tusz_edf(tusz_root, recording_id)
    edf_hash = _file_sha256(edf_path)
    patient_uid = _patient_uid(recording_id, model_split)
    decoded, detection = _build_long_detection(
        verified,
        edf_sha256=edf_hash,
        patient_uid=patient_uid,
    )
    record_key = hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:24]
    record_root = output_root / "target_free" / "records" / record_key
    detection_path = record_root / "detection_manifest.json"
    _atomic_json(detection_path, detection)
    selected = [
        row
        for row in detection["merge_candidates"]
        if row["decision"]
        in {"selected_for_event_analysis", "rejected_insufficient_fixed_window"}
    ]
    event_map = {
        row["candidate_id"]: _candidate_event_id(recording_id, model_split, index)
        for index, row in enumerate(selected, start=1)
    }
    adaptive_path = record_root / "adaptive_search.json"
    adaptive = materialize_adaptive_eeg_search(
        detection_manifest_path=detection_path,
        edf_path=edf_path,
        output_path=adaptive_path,
        event_id_by_candidate=event_map,
    )
    config = CanonicalEDFConfig(
        output_sampling_rate_hz=100.0,
        findings_highpass_hz=0.5,
        findings_lowpass_hz=40.0,
        butterworth_order=4,
    )
    bundle = load_canonical_edf_views(edf_path, config=config)
    validate_canonical_edf_materialization(bundle)
    posterior_binding = validate_canonical_detector_input_binding(
        verified.artifact["canonical_detector_input_binding"]
    )
    canonical = bundle.canonical_record.canonical_receipt
    if (
        canonical["canonical_signal_id"] != posterior_binding["canonical_signal_id"]
        or canonical["source_signal_sha256"]
        != posterior_binding["canonical_source_signal_sha256"]
    ):
        raise ValueError("P0 canonical EEG does not bind to the detector carrier")
    candidate_by_id = {
        row["candidate_id"]: row for row in detection["merge_candidates"]
    }
    results: list[TargetFreeEvent] = []
    for adaptive_event in adaptive["events"]:
        if len(results) >= maximum_events:
            break
        search = adaptive_event["adaptive_search_receipt"]
        if search is None:
            continue
        window = derive_adaptive_event_analysis_window(search)
        p0 = materialize_ba_ieg_p0_event_tokens(
            bundle,
            search,
            window,
            event_id=adaptive_event["eeg_event_id"],
            # P0 is rooted in the immutable physical EEG identity.  The
            # detector manifest deliberately uses a separate opaque navigation
            # identity, so passing that value here would fail the canonical
            # clock/record binding even for the same samples.
            recording_id=canonical["recording_id"],
            patient_uid=patient_uid,
            model_split=model_split,
            policy=BAIEGP0TokenizationPolicy(),
        )
        event_root = output_root / "events" / adaptive_event["eeg_event_id"]
        _atomic_json(event_root / "p0_receipt.json", p0.receipt)
        if p0.receipt["status"] != "materialized" or p0.event_tokens is None:
            continue
        projection = project_ba_ieg_event_model_input_v1(p0)
        model_input = projection.model_input_event
        preprocessing = adaptive_event["preprocessing_receipt"]
        acquisition_hash = preprocessing["receipt_sha256"]
        reasons = set(adaptive_event["plan"]["boundary_truncation_reasons"])
        left_code = (
            "recording_edge"
            if window["censoring"]["left"] and "recording_start" in reasons
            else "search_cap"
            if window["censoring"]["left"]
            else "none"
        )
        right_code = (
            "recording_edge"
            if window["censoring"]["right"] and "recording_stop" in reasons
            else "search_cap"
            if window["censoring"]["right"]
            else "none"
        )
        metadata = ba_ieg_segmental_event_input_metadata_v1(
            model_input,
            adaptive_acquisition_receipt_sha256=acquisition_hash,
            quality_gap_intervals_seconds=(),
            left_censor_reason_code=left_code,
            right_censor_reason_code=right_code,
        )
        arrays = ba_ieg_segmental_event_tensor_arrays_v1(model_input)
        if event_root.name != model_input.event_id:
            raise RuntimeError("P0 changed the frozen target-free event identity")
        metadata_path = event_root / "input_metadata.json"
        tensors_path = event_root / "input_tensors.npz"
        _atomic_json(metadata_path, metadata)
        _atomic_npz(tensors_path, arrays)
        _atomic_json(
            event_root / "projection_receipt.json",
            {
                "schema_version": projection.schema_version,
                "receipt_sha256": projection.receipt_sha256,
                "source_p0_materialization_receipt_sha256": (
                    projection.source_p0_materialization_receipt_sha256
                ),
                "model_input_event_receipt_sha256": model_input.input_receipt_sha256,
                "deterministic_target_sidecar_receipt_sha256": (
                    projection.deterministic_target_sidecar.receipt_sha256
                ),
                "deterministic_target_available_to_model_forward": False,
            },
        )
        reference_relative = str(PurePosixPath(recording_id).with_suffix(".csv_bi"))
        results.append(
            TargetFreeEvent(
                model_split=model_split,
                patient_uid=patient_uid,
                edf_path=edf_path,
                reference_relative_path=reference_relative,
                verified=verified,
                detection_manifest=detection,
                adaptive_artifact=adaptive,
                adaptive_event=adaptive_event,
                adaptive_window=window,
                detector_candidate=candidate_by_id[adaptive_event["candidate_id"]],
                projection=projection,
                input_metadata_path=metadata_path,
                input_tensors_path=tensors_path,
                adaptive_acquisition_receipt_sha256=acquisition_hash,
            )
        )
    if not results:
        raise ValueError("real EEG produced no materializable target-free P0 event")
    # The decoded object is deliberately retained only through its hash in the
    # target-free roster; it never reaches the public interval parser.
    if decoded["scope_receipt"]["labels_or_ground_truth_used_for_current_recording"]:
        raise RuntimeError("detector decode acquired a forbidden target")
    return results


def _target_free_roster(events: Sequence[TargetFreeEvent]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in events:
        event = item.projection.model_input_event
        plan = item.adaptive_event["plan"]
        search = item.adaptive_event["adaptive_search_receipt"]
        rows.append(
            {
                "event_id": event.event_id,
                "recording_id": event.recording_id,
                "patient_uid": event.patient_uid,
                "model_split": event.model_split,
                "posterior_artifact_id": item.verified.artifact[
                    "posterior_artifact_id"
                ],
                "posterior_file_sha256": item.verified.posterior_file_sha256,
                "detector_decoding_policy_sha256": _canonical_sha256(
                    DEFAULT_DECODER_POLICY
                ),
                "detection_manifest_sha256": _canonical_sha256(item.detection_manifest),
                "detector_candidate_id": item.detector_candidate["candidate_id"],
                "detector_candidate_receipt_sha256": _canonical_sha256(
                    item.detector_candidate
                ),
                "detector_candidate_support_interval_recording_seconds": [
                    item.detector_candidate["start_offset_seconds"],
                    item.detector_candidate["stop_offset_seconds"],
                ],
                "adaptive_materialization_sha256": item.adaptive_artifact[
                    "artifact_sha256"
                ],
                "adaptive_plan_sha256": _canonical_sha256(plan),
                "adaptive_acquisition_receipt_sha256": (
                    item.adaptive_acquisition_receipt_sha256
                ),
                "adaptive_search_receipt_sha256": _canonical_sha256(search),
                "adaptive_window_receipt_sha256": _canonical_sha256(
                    item.adaptive_window
                ),
                "p0_input_event_receipt_sha256": event.input_receipt_sha256,
                "model_input_projection_receipt_sha256": item.projection.receipt_sha256,
                "token_count": int(event.token_values.shape[0]),
            }
        )
    rows.sort(key=lambda row: (row["model_split"], row["patient_uid"], row["event_id"]))
    body: dict[str, Any] = {
        "schema_version": CANDIDATE_ROSTER_SCHEMA_VERSION,
        "receipt_id": "BAIEG-TUSZ-ROSTER-PENDING",
        "method_id": "post_p0_projection_pre_reference_global_roster_freeze_v1",
        "rows": rows,
        "row_roster_sha256": _canonical_sha256(rows),
        "scope_receipt": {
            "eeg_signal_and_acquisition_metadata_only": True,
            "candidate_window_tokenization_frozen": True,
            "public_reference_path_accepted_before_freeze": False,
            "public_reference_files_opened_before_freeze": 0,
            "edf_annotations_opened": 0,
            "spreadsheets_opened": 0,
            "private_or_doctor_labels_opened": 0,
            "clinical_text_opened": 0,
            "target_available_to_model_forward": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    id_source = deepcopy(body)
    id_source["receipt_id"] = "BAIEG-TUSZ-ROSTER-PENDING"
    body["receipt_id"] = "BAIEGTUSZROSTER-" + _canonical_sha256(id_source)[:24]
    hash_source = deepcopy(body)
    hash_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    body["receipt_sha256"] = _canonical_sha256(hash_source)
    return body


def _freeze_candidate_envelopes(
    events: Sequence[TargetFreeEvent], roster: Mapping[str, Any], output_root: Path
) -> None:
    roster_hash = str(roster["receipt_sha256"])
    for item in events:
        event = item.projection.model_input_event
        support = [
            float(item.detector_candidate["start_offset_seconds"]),
            float(item.detector_candidate["stop_offset_seconds"]),
        ]
        analysis = list(event.analysis_interval_seconds)
        if support[0] < analysis[0] - 1e-6 or support[1] > analysis[1] + 1e-6:
            # Preserve the complete candidate in the global roster.  The
            # capability binds the actually inspected detector support only.
            support = [max(support[0], analysis[0]), min(support[1], analysis[1])]
        if support[1] <= support[0]:
            raise ValueError("adaptive P0 interval does not inspect detector support")
        capability = freeze_ba_ieg_tusz_candidate_envelope_after_tokenization_v1(
            event,
            detector_candidate_id=item.detector_candidate["candidate_id"],
            detector_candidate_receipt_sha256=_canonical_sha256(
                item.detector_candidate
            ),
            adaptive_envelope_receipt_sha256=_canonical_sha256(
                item.adaptive_event["plan"]
            ),
            adaptive_acquisition_receipt_sha256=(
                item.adaptive_acquisition_receipt_sha256
            ),
            target_independent_candidate_roster_receipt_sha256=roster_hash,
            detector_candidate_support_interval_recording_seconds=support,
        )
        item.candidate_envelope = capability
        _atomic_json(
            output_root / "events" / event.event_id / "candidate_envelope.json",
            capability.payload(),
        )


def parse_public_tusz_csv_bi_after_freeze(
    path: Path,
    *,
    expected_duration_seconds: float,
    annotation_resolution_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse only global public TUSZ event intervals after candidate freeze."""

    source = _regular_file(path, "public TUSZ csv_bi reference")
    if source.suffix != ".csv_bi":
        raise ValueError("boundary supervision must come from a .csv_bi sidecar")
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("public TUSZ csv_bi is not UTF-8") from error
    duration_rows = [
        line for line in text.splitlines() if line.startswith("# duration = ")
    ]
    if len(duration_rows) != 1 or not duration_rows[0].endswith(" secs"):
        raise ValueError("public TUSZ csv_bi duration header is missing")
    declared = float(duration_rows[0][len("# duration = ") : -len(" secs")])
    if abs(declared - expected_duration_seconds) > 1e-6:
        raise ValueError("public TUSZ csv_bi duration disagrees with frozen EEG")
    data_lines = [
        line for line in text.splitlines() if line and not line.startswith("#")
    ]
    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
    expected_fields = ["channel", "start_time", "stop_time", "label", "confidence"]
    if reader.fieldnames != expected_fields:
        raise ValueError("public TUSZ csv_bi columns drifted")
    intervals: list[dict[str, Any]] = []
    previous_stop: float | None = None
    for index, row in enumerate(reader, start=1):
        if set(row) != set(expected_fields) or row["channel"] != "TERM":
            raise ValueError("csv_bi contains non-global or malformed rows")
        label = str(row["label"]).lower()
        if label not in {"seiz", "bckg"}:
            raise ValueError("csv_bi contains an unsupported event label")
        confidence = float(row["confidence"])
        start = float(row["start_time"])
        stop = float(row["stop_time"])
        if (
            not all(math.isfinite(value) for value in (confidence, start, stop))
            or not 0.0 <= confidence <= 1.0
            or start < 0.0
            or stop <= start
            or stop > expected_duration_seconds + 1e-6
        ):
            raise ValueError("csv_bi event interval is invalid")
        if label == "bckg":
            continue
        if previous_stop is not None and start < previous_stop:
            raise ValueError("csv_bi seizure intervals overlap or are unsorted")
        intervals.append(
            {
                "public_event_id": f"TUSZ-SEIZ-{len(intervals) + 1:04d}",
                "onset_recording_seconds": start,
                "offset_recording_seconds": stop,
            }
        )
        previous_stop = stop
    receipt: dict[str, Any] = {
        "schema_version": REFERENCE_PARSE_SCHEMA_VERSION,
        "source_reference_artifact_sha256": digest,
        "source_format": "tusz_csv_bi",
        "recording_duration_seconds": expected_duration_seconds,
        "annotation_timestamp_resolution_seconds": annotation_resolution_seconds,
        "reference_coverage_status": "complete_recording",
        "seizure_interval_count": len(intervals),
        "scope_receipt": {
            "public_tusz_event_boundary_sidecar_only": True,
            "channel_level_csv_opened": False,
            "edf_annotation_api_called": False,
            "soz_or_channel_target_read": False,
            "private_data_read": False,
            "spreadsheet_or_clinical_text_read": False,
            "reference_opened_after_global_candidate_roster_freeze": True,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    hash_source = deepcopy(receipt)
    receipt["receipt_sha256"] = _canonical_sha256(hash_source)
    return intervals, receipt


def _join_references_and_write_targets(
    events: Sequence[TargetFreeEvent],
    *,
    tusz_root: Path,
    output_root: Path,
    annotation_resolution_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = tusz_root.resolve(strict=True)
    for item in events:
        if item.candidate_envelope is None:
            raise RuntimeError(
                "public reference join attempted before candidate freeze"
            )
        reference_path = _regular_file(
            root.joinpath(*PurePosixPath(item.reference_relative_path).parts),
            "public TUSZ csv_bi reference",
        )
        if root not in reference_path.parents:
            raise ValueError("public reference escaped the pinned TUSZ root")
        intervals, parse_receipt = parse_public_tusz_csv_bi_after_freeze(
            reference_path,
            expected_duration_seconds=item.verified.artifact[
                "recording_duration_seconds"
            ],
            annotation_resolution_seconds=annotation_resolution_seconds,
        )
        reference = build_ba_ieg_tusz_public_interval_reference_v1(
            source_dataset_id="TUSZ",
            source_dataset_version="2.0.3",
            patient_uid=item.patient_uid,
            recording_id=item.projection.model_input_event.recording_id,
            model_split=item.model_split,
            recording_duration_seconds=item.verified.artifact[
                "recording_duration_seconds"
            ],
            source_reference_artifact_id=(
                "TUSZCSVBI-" + parse_receipt["source_reference_artifact_sha256"][:24]
            ),
            source_reference_artifact_sha256=parse_receipt[
                "source_reference_artifact_sha256"
            ],
            source_format="tusz_csv_bi",
            annotation_timestamp_resolution_seconds=annotation_resolution_seconds,
            reference_coverage_status="complete_recording",
            covered_recording_intervals_seconds=[
                [0.0, item.verified.artifact["recording_duration_seconds"]]
            ],
            seizure_intervals=intervals,
        )
        materialized = materialize_ba_ieg_tusz_candidate_interval_target_v1(
            item.candidate_envelope, reference
        )
        event = item.projection.model_input_event
        event_root = output_root / "events" / event.event_id
        reference_payload = reference.payload()
        target_payload = ba_ieg_segmental_event_target_payload_v1(materialized.target)
        reference_out = event_root / "public_interval_reference.json"
        target_materialization_out = event_root / "target_materialization.json"
        target_out = event_root / "target.json"
        _atomic_json(reference_out, reference_payload)
        _atomic_json(event_root / "reference_parse_receipt.json", parse_receipt)
        _atomic_json(target_materialization_out, materialized.receipt)
        _atomic_json(target_out, target_payload)
        rows.append(
            {
                "event_id": event.event_id,
                "recording_id": event.recording_id,
                "patient_uid": event.patient_uid,
                "model_split": event.model_split,
                "input_metadata_relative_path": str(
                    item.input_metadata_path.relative_to(output_root)
                ),
                "input_metadata_file_sha256": _file_sha256(item.input_metadata_path),
                "input_tensors_relative_path": str(
                    item.input_tensors_path.relative_to(output_root)
                ),
                "input_tensors_file_sha256": _file_sha256(item.input_tensors_path),
                "target_relative_path": str(target_out.relative_to(output_root)),
                "target_file_sha256": _file_sha256(target_out),
                "input_event_receipt_sha256": event.input_receipt_sha256,
                "target_receipt_sha256": materialized.target.receipt_sha256,
                "adaptive_acquisition_receipt_sha256": (
                    item.adaptive_acquisition_receipt_sha256
                ),
                "target_independent_candidate_roster_receipt_sha256": (
                    item.candidate_envelope.payload()[
                        "target_independent_candidate_roster_receipt_sha256"
                    ]
                ),
                "source_reference_receipt_sha256": reference_payload["receipt_sha256"],
                "token_count": int(event.token_values.shape[0]),
            }
        )
    return rows


def _tensor_parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _run_real_training_smoke(
    manifest_path: Path,
    *,
    hidden_dim: int,
) -> dict[str, Any]:
    train = BAIEGSegmentalDiskDatasetV1(manifest_path, purpose="optimize")
    dev = BAIEGSegmentalDiskDatasetV1(manifest_path, purpose="calibrate")
    maximum_tokens = max(
        max(counts)
        for counts in train.patient_event_token_counts + dev.patient_event_token_counts
    )
    train_loader = build_ba_ieg_segmental_disk_loader_v1(
        train,
        maximum_padded_tokens_per_batch=maximum_tokens,
        maximum_patients_per_batch=1,
        shuffle=False,
        num_workers=0,
    )
    dev_loader = build_ba_ieg_segmental_disk_loader_v1(
        dev,
        maximum_padded_tokens_per_batch=maximum_tokens,
        maximum_patients_per_batch=1,
        shuffle=False,
        num_workers=0,
    )
    train_batch = next(iter(train_loader))
    dev_batch = next(iter(dev_loader))
    torch.manual_seed(20260823)
    feature_dim = int(train_batch.event_batch.token_values.shape[-1])
    model = BAIEGPermissionSplitSegmentalStateModel(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        maximum_segments=6,
        maximum_paths=4,
        dropout=0.0,
        allow_heuristic_phase_posterior=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = BAIEGPermissionSplitSegmentalTrainerV1(
        model, optimizer, maximum_gradient_norm=1.0
    )
    before = _tensor_parameter_sha256(model)
    loss = trainer.optimize_batch(train_batch)
    after = _tensor_parameter_sha256(model)
    if before == after or not torch.isfinite(loss.total_loss):
        raise RuntimeError(
            "real source-train optimizer smoke did not update parameters"
        )
    calibration = trainer.calibration_forward(dev_batch)
    if calibration.output.exact_path_log_partition.requires_grad:
        raise RuntimeError("source-dev calibration retained gradients")
    return {
        "schema_version": "ba_ieg_real_segmental_optimizer_smoke_v1",
        "manifest_id": train.manifest_id,
        "manifest_file_sha256": train.manifest_file_sha256,
        "source_train_patient_uids": list(train.patient_uids),
        "source_dev_patient_uids": list(dev.patient_uids),
        "source_train_event_ids": list(train_batch.event_batch.event_ids),
        "source_dev_event_ids": list(dev_batch.event_batch.event_ids),
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "source_train_total_loss": float(loss.total_loss.detach().cpu()),
        "source_train_causal_onset_nll": float(loss.causal_onset_nll.detach().cpu()),
        "source_train_offline_constrained_path_nll": float(
            loss.offline_constrained_path_nll.detach().cpu()
        ),
        "source_train_lattice_target_projection_receipt_sha256s": list(
            loss.lattice_target_projection_receipt_sha256s
        ),
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "parameters_updated": True,
        "source_dev_no_gradient_calibration_forward": True,
        "target_available_to_model_forward": False,
        "source_train_gradient_updates_only": True,
        "raw_annotation_intervals_preserved_before_lattice_projection": True,
        "lattice_projection_stage": "after_target_free_forward_and_lattice_freeze",
        "target_authority_scope": "public_event_presence_and_weak_boundary_only",
        "morphology_rhythm_spatial_or_soz_gold_claimed": False,
        "production_or_private_route_authorized": False,
    }


def _parse_ordinals(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ordinals must be comma-separated integers"
        ) from error
    if (
        not result
        or any(item < 1 for item in result)
        or len(set(result)) != len(result)
    ):
        raise argparse.ArgumentTypeError("ordinals must be unique positive integers")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source-train-batch", type=Path, required=True)
    parser.add_argument("--source-dev-batch", type=Path, required=True)
    parser.add_argument("--source-train-ordinals", type=_parse_ordinals, default=(1,))
    parser.add_argument("--source-dev-ordinals", type=_parse_ordinals, default=(1,))
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-events-per-record", type=int, default=1)
    parser.add_argument("--annotation-resolution-seconds", type=float, default=0.0001)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument(
        "--skip-optimizer-step",
        action="store_true",
        help="Materialize and replay the disk route without an optimizer smoke step.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.maximum_events_per_record < 1
        or args.hidden_dim < 1
        or not math.isfinite(args.annotation_resolution_seconds)
        or args.annotation_resolution_seconds <= 0
    ):
        raise ValueError("smoke materialization numeric policy is invalid")
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o700)
    registry_path = ROOT / "configs" / "continuous_detector_providers_v1.json"

    verified_records: list[tuple[str, VerifiedPosteriorRecord]] = []
    for split, batch_root, ordinals in (
        ("source_train", args.source_train_batch, args.source_train_ordinals),
        ("source_dev", args.source_dev_batch, args.source_dev_ordinals),
    ):
        for ordinal in ordinals:
            verified_records.append(
                (
                    split,
                    load_verified_posterior_record_without_references(
                        batch_root,
                        ordinal=ordinal,
                        expected_split=split,
                        provider_registry_path=registry_path,
                    ),
                )
            )

    # Phase A: all candidate selection, adaptive windows, P0 tokens and model
    # inputs are frozen without constructing or opening any reference path.
    target_free_events: list[TargetFreeEvent] = []
    for split, verified in verified_records:
        target_free_events.extend(
            materialize_target_free_record(
                verified,
                model_split=split,
                tusz_root=args.tusz_root,
                output_root=output,
                maximum_events=args.maximum_events_per_record,
            )
        )
    roster = _target_free_roster(target_free_events)
    roster_path = output / "target_free_candidate_roster.json"
    _atomic_json(roster_path, roster)
    _freeze_candidate_envelopes(target_free_events, roster, output)
    freeze_receipt = {
        "schema_version": "ba_ieg_tusz_pre_reference_freeze_barrier_v1",
        "candidate_roster_receipt_sha256": roster["receipt_sha256"],
        "candidate_roster_file_sha256": _file_sha256(roster_path),
        "candidate_count": len(target_free_events),
        "all_candidate_capabilities_written": True,
        "public_reference_files_opened_before_this_receipt": 0,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    freeze_receipt["receipt_sha256"] = _canonical_sha256(freeze_receipt)
    _atomic_json(output / "pre_reference_freeze_receipt.json", freeze_receipt)

    # Phase B begins only after the durable global freeze barrier above.
    rows = _join_references_and_write_targets(
        target_free_events,
        tusz_root=args.tusz_root,
        output_root=output,
        annotation_resolution_seconds=args.annotation_resolution_seconds,
    )
    split_rows = sorted(
        [[item.patient_uid, item.model_split] for item in target_free_events]
    )
    provenance = {
        "source_dataset_id": "TUSZ",
        "source_dataset_version": "2.0.3",
        "source_corpus_manifest_sha256": verified_records[0][1].batch[
            "manifest_sha256"
        ],
        "source_patient_split_receipt_sha256": _canonical_sha256(split_rows),
        "input_materializer_code_sha256": _file_sha256(Path(__file__)),
        "target_materializer_code_sha256": _file_sha256(
            ROOT
            / "src"
            / "clinical_eeg_long_recording"
            / "ba_ieg_tusz_candidate_interval_target_materializer_v1.py"
        ),
        "target_materialization_policy_sha256": _canonical_sha256(
            {
                "source_format": "tusz_csv_bi",
                "annotation_resolution_seconds": args.annotation_resolution_seconds,
                "join_stage": "after_global_candidate_window_token_roster_freeze",
                "supervision_scope": "event_and_boundary_only",
            }
        ),
    }
    if any(
        verified.batch["manifest_sha256"] != provenance["source_corpus_manifest_sha256"]
        for _, verified in verified_records
    ):
        raise ValueError("train/dev posterior batches bind different source manifests")
    manifest = build_ba_ieg_segmental_disk_manifest_v1(rows=rows, provenance=provenance)
    manifest_path = output / "segmental_disk_manifest.json"
    _atomic_json(manifest_path, manifest)
    training = (
        {
            "optimizer_step_executed": False,
            "reason": "explicit_skip_optimizer_step",
        }
        if args.skip_optimizer_step
        else _run_real_training_smoke(manifest_path, hidden_dim=args.hidden_dim)
    )
    _atomic_json(output / "training_smoke_receipt.json", training)
    final: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "materialized_and_replayed",
        "output_root": str(output),
        "candidate_roster_receipt_sha256": roster["receipt_sha256"],
        "pre_reference_freeze_receipt_sha256": freeze_receipt["receipt_sha256"],
        "segmental_disk_manifest_id": manifest["manifest_id"],
        "segmental_disk_manifest_file_sha256": _file_sha256(manifest_path),
        "source_train_record_count": len(
            {
                item.verified.artifact["recording_id"]
                for item in target_free_events
                if item.model_split == "source_train"
            }
        ),
        "source_dev_record_count": len(
            {
                item.verified.artifact["recording_id"]
                for item in target_free_events
                if item.model_split == "source_dev"
            }
        ),
        "source_train_event_count": sum(
            item.model_split == "source_train" for item in target_free_events
        ),
        "source_dev_event_count": sum(
            item.model_split == "source_dev" for item in target_free_events
        ),
        "target_status_counts": {
            status: sum(
                json.loads(
                    (output / row["target_relative_path"]).read_text(encoding="utf-8")
                )["target"]["event_status"]
                == status
                for row in rows
            )
            for status in ("present", "absent", "not_evaluable")
        },
        "optimizer_step_executed": not args.skip_optimizer_step,
        "scope_receipt": {
            "public_tusz_only": True,
            "tusz_event_boundary_supervision_only": True,
            "tusz_channel_or_soz_targets_used": False,
            "candidate_window_tokenization_frozen_before_reference_open": True,
            "edf_annotations_used": False,
            "private_data_used": False,
            "spreadsheets_or_clinical_text_used": False,
            "qwen_or_report_route_used": False,
            "production_or_private_route_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    final["receipt_sha256"] = _canonical_sha256(final)
    _atomic_json(output / "smoke_receipt.json", final)
    print(_canonical_json(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
