"""Sealed, signal-only LaBraM k31 I-gate prediction materialization.

This module performs inference only.  It cannot accept a TUSZ target snapshot,
native target mask, DeepSOZ source, private dataset, gate outcome, optimizer,
calibrator, or evaluator.  Six already-trained v1.2 k31 heads are replayed in
the fixed order ``fold0..fold4, final`` on the exact 12-patient/212-event gate
token subset.  Logits are preserved per producer; no ensemble, threshold,
metric, promotion decision, or SOZ interpretation is computed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load as _load_safetensors_bytes
from safetensors.torch import save_file as _save_safetensors_file

from .concept_token_io import (
    CONCEPT_TOKEN_PURPOSE,
    CONCEPT_TOKEN_SHAPE,
    load_labram_concept_tokens,
)
from .formal_token_corpus import (
    FormalTokenSubsetEventBinding,
    VerifiedFormalTokenCorpusSubsetArtifact,
    formal_token_subset_roster_sha256,
)
from .geometry import N_TCP_EDGES
from .ictal_inference_primitives_v13 import patient_roster_sha256
from .ictal_k31_inference_projection_v13 import (
    LoadedK31InferenceProjectionProducerV13,
    LoadedK31InferenceProjectionV13,
)
from .models.labram import AUDITED_LABRAM_BASE_SHA256, AUDITED_LABRAM_MODELING_SHA256


GATE_PREDICTION_SCHEMA_V13 = "soz_labram_k31_ictal_gate_predictions_v13"
GATE_PREDICTION_PURPOSE = (
    "sealed_i_gate_candidate_logits_component_not_complete_stage_a_or_soz_evidence"
)
GATE_PREDICTION_SERIALIZATION = "canonical_json_plus_safetensors_no_pickle"
GATE_PREDICTION_MANIFEST_FILENAME = "manifest.json"
GATE_PREDICTION_TENSOR_FILENAME = "predictions.safetensors"
GATE_PREDICTION_TENSOR_NAME = "logits"
GATE_ACCESS_RECEIPT_SCHEMA = "soz_signal_only_i_gate_access_receipt_v1"
GATE_EXECUTION_RECEIPT_SCHEMA = "soz_signal_only_i_gate_execution_receipt_v1"

EXPECTED_PRODUCER_ORDER = (
    "fold0",
    "fold1",
    "fold2",
    "fold3",
    "fold4",
    "final",
)
EXPECTED_GATE_PATIENT_COUNT = 12
EXPECTED_GATE_EVENT_COUNT = 212
EXPECTED_LOGIT_TAIL = (N_TCP_EDGES, CONCEPT_TOKEN_SHAPE[1], 1)
EXPECTED_CANDIDATE = "labram_temporal_residual_k31"
EXPECTED_CONTEXT_SECONDS = 31
EXPECTED_CONTEXT_DIRECTION = "symmetric_retrospective_not_causal_onset"
EXPECTED_TARGET_SEMANTICS = "tusz_bipolar_edge_time_involvement_not_soz"
V13_EXECUTION_HOLD = True
V13_EXECUTION_HOLD_BLOCKERS = (
    "missing_independent_time_only_prevalence_and_mask_only_control_parameters",
    "missing_target_free_scale_probe",
    "missing_target_free_fold_identity_probe",
    "missing_closed_execution_authorization_artifact_and_hash",
    "filesystem_subset_confinement_not_proven",
    "atomic_no_replace_publication_not_proven",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PATIENT_RE = re.compile(r"[a-z0-9]{8}")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_TENSOR_BYTES = 128 * 1024 * 1024
_MAX_TOKEN_INDEX_BYTES = 64 * 1024 * 1024

_TOKEN_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "formal",
        "smoke_only",
        "serialization",
        "master_manifest",
        "training_manifest",
        "preprocessing_selection",
        "preprocess",
        "foundation",
        "event_count",
        "patient_count",
        "event_roster_sha256",
        "patient_roster_sha256",
        "patient_event_roster_sha256",
        "tensor_roster_sha256",
        "events",
    }
)
_TOKEN_MASTER_FIELDS = frozenset(
    {
        "bundle_manifest_sha256",
        "source_manifest_sha256",
        "cohort_receipt_sha256",
        "preflight_performed",
        "event_count",
        "patient_count",
    }
)
_TOKEN_TRAINING_FIELDS = frozenset(
    {
        *_TOKEN_MASTER_FIELDS,
        "role",
        "derived_from_master_source_manifest_sha256",
    }
)
_TOKEN_PREPROCESSING_FIELDS = frozenset(
    {
        "schema_version",
        "producer_kind",
        "selected_arm_id",
        "selected_arm_spec_receipt_sha256",
        "selected_arm_result_receipt_sha256",
        "selection_artifact_sha256",
        "selection_bundle_receipt_sha256",
        "protocol_receipt_sha256",
        "nested_dev_manifest_receipt_sha256",
        "raw_qc_intersection_receipt_sha256",
        "content_component_split_receipt_sha256",
        "source_patient_roster_sha256",
        "foundation_feature_receipt_sha256",
        "token_schema_version",
        "authorization_receipt_sha256",
    }
)
_TOKEN_FOUNDATION_FIELDS = frozenset(
    {
        "feature_receipt_sha256",
        "checkpoint_sha256",
        "audited_expected_checkpoint_sha256",
        "modeling_sha256",
        "audited_expected_modeling_sha256",
        "token_shape",
        "tile_seconds",
        "frozen",
        "materialization_device",
    }
)
_TOKEN_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "patient_id",
        "event_record_sha256",
        "preprocess_receipt_sha256",
        "bundle_path",
        "bundle_manifest_sha256",
        "tensor_sha256",
    }
)

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "serialization",
        "candidate",
        "context_seconds",
        "context_direction",
        "target_semantics",
        "logit_semantics",
        "development_only",
        "formal_promotion",
        "authorized_for_formal_evidence_or_reasoner",
        "candidate_logits_present",
        "independent_control_logits_present",
        "time_only_control_logits_present",
        "prevalence_control_logits_present",
        "mask_only_control_logits_present",
        "target_free_scale_probe_passed",
        "target_free_fold_identity_probe_passed",
        "complete_stage_a_seal",
        "execution_authorized",
        "execution_hold_blockers",
        "v5_split_sha256",
        "inference_projection_manifest_sha256",
        "producer_order",
        "producers",
        "gate_patient_ids",
        "gate_patient_roster_sha256",
        "gate_patient_count",
        "corpus",
        "event_count",
        "event_roster_sha256",
        "events",
        "tensor",
        "execution_receipt",
        "access_receipt",
    }
)
_PRODUCER_FIELDS = frozenset(
    {
        "selection",
        "oof_fold",
        "projection_bundle_manifest_sha256",
        "manifest_sha256",
        "legacy_recovery_manifest_sha256",
        "checkpoint_sha256",
        "head_state_sha256",
        "fit_patient_ids",
        "fit_patient_roster_sha256",
        "i_gate_patient_roster_sha256",
        "v5_split_sha256",
        "upstream_tusz_ictal_involvement_targets_loaded",
        "legacy_full_manifest_loaded_by_broker",
        "legacy_native_evaluation_roster_metadata_loaded_by_broker",
        "legacy_training_run_metrics_loaded_by_broker",
        "legacy_checkpoint_weights_loaded_by_broker",
        "legacy_training_process_loaded_full_tusz_target_snapshot_arrays",
        "legacy_training_process_snapshot_contained_gate_rows",
        "upstream_i_gate_used_for_fit_loss_gradient_or_native_metric",
        "upstream_i_gate_confirmation_metrics_computed",
    }
)
_CORPUS_FIELDS = frozenset(
    {
        "index_sha256",
        "master_bundle_manifest_sha256",
        "master_source_manifest_sha256",
        "training_bundle_manifest_sha256",
        "training_source_manifest_sha256",
        "preprocessing_selection_artifact_sha256",
        "preprocessing_selection_bundle_receipt_sha256",
        "preprocessing_protocol_receipt_sha256",
        "preprocessing_selected_arm_result_receipt_sha256",
        "preprocessing_selected_arm_id",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "foundation_modeling_sha256",
        "full_event_count",
        "full_patient_count",
        "selected_event_count",
        "selected_event_roster_sha256",
        "unselected_event_bundles_opened",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "patient_id",
        "event_id",
        "event_record_sha256",
        "preprocess_receipt_sha256",
        "input_bundle_manifest_sha256",
        "input_tensor_sha256",
    }
)
_TENSOR_FIELDS = frozenset(
    {
        "filename",
        "name",
        "dtype",
        "shape",
        "file_size_bytes",
        "file_sha256",
        "value_sha256",
    }
)
_EXECUTION_FIELDS = frozenset(
    {
        "schema_version",
        "torch_version",
        "device_type",
        "device_name",
        "cuda_runtime_version",
        "cudnn_version",
        "compute_capability",
        "inference_mode_used",
        "gradient_enabled_during_forward",
        "batch_size",
    }
)
_ACCESS_FIELDS = frozenset(
    {
        "schema_version",
        "master_signal_index_loaded",
        "minimal_k31_inference_projection_loaded",
        "legacy_recovery_bundle_loaded",
        "legacy_recovery_manifest_metadata_loaded",
        "legacy_native_evaluation_roster_metadata_loaded",
        "legacy_training_run_metrics_loaded",
        "gate_token_values_loaded",
        "non_gate_token_bundles_opened",
        "deepsoz_source_eval_source_loaded",
        "deepsoz_identity_source_loaded",
        "filesystem_subset_confinement_proven",
        "execution_authorization_loaded",
        "tusz_target_values_loaded",
        "tusz_target_masks_loaded",
        "deepsoz_target_source_loaded",
        "deepsoz_soz_values_loaded",
        "private_eeg_loaded",
        "private_targets_loaded",
        "gate_split_bytes_hashed_only",
        "gate_split_json_parsed",
        "gate_outcomes_opened",
        "training",
        "calibration",
        "model_selection",
        "evaluation",
        "prediction_forward_executed",
        "output_published",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Value cannot be represented as canonical JSON") from exc
    return text.encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_exact_fields(
    value: object, expected: frozenset[str], *, field: str
) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be a JSON object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{field} violates its closed schema; missing={missing}, unknown={unknown}"
        )
    return value


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"Non-finite JSON value is forbidden: {value}")


def _parse_canonical_json(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Prediction manifest is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise ValueError("Prediction manifest must be one canonical JSON object")
    return value


def _reject_symlink_components(path: Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot contain symlink components")
    return absolute


def _read_stable_regular_file(
    path: Path, *, field: str, maximum_bytes: int
) -> tuple[bytes, str]:
    source = _reject_symlink_components(path, field=field)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{field} must be a regular file: {source}")
    before = source.stat()
    if before.st_size < 1 or before.st_size > maximum_bytes:
        raise ValueError(f"{field} has an invalid size")
    raw = source.read_bytes()
    after = source.stat()
    fingerprint = lambda stat: (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
    if fingerprint(before) != fingerprint(after):
        raise RuntimeError(f"{field} changed while it was read")
    return raw, hashlib.sha256(raw).hexdigest()


def hash_gate_split_without_parsing(
    path: str | Path, *, expected_sha256: str
) -> str:
    """Hash-pin the frozen split without decoding any outcome-bearing JSON."""

    source = _reject_symlink_components(Path(path), field="v5 split")
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError("v5 split must be a regular file")
    before = source.stat()
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    after = source.stat()
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_fingerprint != after_fingerprint or size != before.st_size or size < 1:
        raise RuntimeError("v5 split changed while it was hash-pinned")
    actual = digest.hexdigest()
    if actual != _require_sha256(expected_sha256, field="expected_v5_split_sha256"):
        raise ValueError("v5 split SHA-256 mismatch")
    return actual


def inspect_master_gate_index_metadata_v13(
    corpus_directory: str | Path,
    *,
    expected_index_sha256: str,
    expected_master_bundle_manifest_sha256: str,
    expected_master_source_manifest_sha256: str,
    expected_preprocessing_selection_artifact_sha256: str,
    expected_preprocessing_protocol_receipt_sha256: str,
    gate_patient_ids: Sequence[str],
) -> dict[str, object]:
    """Validate only the frozen master index; never enumerate/open bundles."""

    patients = tuple(str(value).strip() for value in gate_patient_ids)
    if (
        len(patients) != EXPECTED_GATE_PATIENT_COUNT
        or patients != tuple(sorted(patients))
        or len(set(patients)) != len(patients)
        or any(not _PATIENT_RE.fullmatch(patient) for patient in patients)
    ):
        raise ValueError("Metadata preflight requires the frozen 12-patient gate")
    corpus = _reject_symlink_components(Path(corpus_directory), field="master corpus")
    if corpus.is_symlink() or not corpus.is_dir():
        raise ValueError("Master corpus must be a regular directory")
    raw, index_sha = _read_stable_regular_file(
        corpus / "index.json",
        field="master token index",
        maximum_bytes=_MAX_TOKEN_INDEX_BYTES,
    )
    if index_sha != _require_sha256(
        expected_index_sha256, field="expected_master_token_corpus_index_sha256"
    ):
        raise ValueError("Master token index SHA-256 mismatch")
    index = _require_exact_fields(
        _parse_canonical_json(raw), _TOKEN_INDEX_FIELDS, field="master token index"
    )
    if (
        index["schema_version"] != "soz_tusz_ictal_token_corpus_index_v4"
        or index["purpose"] != CONCEPT_TOKEN_PURPOSE
        or index["formal"] is not True
        or index["smoke_only"] is not False
        or index["serialization"]
        != "canonical_json_and_safe_event_bundles"
    ):
        raise ValueError("Master token index scientific boundary changed")

    master = _require_exact_fields(
        index["master_manifest"], _TOKEN_MASTER_FIELDS, field="master_manifest"
    )
    training = _require_exact_fields(
        index["training_manifest"],
        _TOKEN_TRAINING_FIELDS,
        field="training_manifest",
    )
    expected_master_bundle = _require_sha256(
        expected_master_bundle_manifest_sha256,
        field="expected_master_manifest_bundle_sha256",
    )
    expected_master_source = _require_sha256(
        expected_master_source_manifest_sha256,
        field="expected_master_manifest_source_sha256",
    )
    for block_name, block in (("master", master), ("training", training)):
        for field in (
            "bundle_manifest_sha256",
            "source_manifest_sha256",
            "cohort_receipt_sha256",
        ):
            _require_sha256(block[field], field=f"{block_name}.{field}")
        if block["preflight_performed"] is not True:
            raise ValueError(f"{block_name} manifest is not signal-preflighted")
    if (
        master["bundle_manifest_sha256"] != expected_master_bundle
        or master["source_manifest_sha256"] != expected_master_source
        or training["role"] != "master"
        or training["derived_from_master_source_manifest_sha256"] is not None
        or any(
            training[field] != master[field]
            for field in _TOKEN_MASTER_FIELDS
        )
    ):
        raise ValueError("Token index is not the hash-pinned exact master corpus")

    preprocessing = _require_exact_fields(
        index["preprocessing_selection"],
        _TOKEN_PREPROCESSING_FIELDS,
        field="preprocessing_selection",
    )
    for field in _TOKEN_PREPROCESSING_FIELDS - {
        "schema_version",
        "producer_kind",
        "selected_arm_id",
        "token_schema_version",
    }:
        _require_sha256(preprocessing[field], field=f"preprocessing.{field}")
    if (
        preprocessing["schema_version"]
        != "soz_preprocessing_producer_authorization_v1"
        or preprocessing["producer_kind"] != "tusz_ictal"
        or preprocessing["selected_arm_id"] != "C-CAR19"
        or preprocessing["token_schema_version"]
        != "soz_tusz_ictal_token_corpus_index_v4"
        or preprocessing["selection_artifact_sha256"]
        != _require_sha256(
            expected_preprocessing_selection_artifact_sha256,
            field="expected_preprocessing_selection_artifact_sha256",
        )
        or preprocessing["protocol_receipt_sha256"]
        != _require_sha256(
            expected_preprocessing_protocol_receipt_sha256,
            field="expected_preprocessing_protocol_receipt_sha256",
        )
    ):
        raise ValueError("Master index preprocessing selection changed")

    foundation = _require_exact_fields(
        index["foundation"], _TOKEN_FOUNDATION_FIELDS, field="foundation"
    )
    for field in (
        "feature_receipt_sha256",
        "checkpoint_sha256",
        "audited_expected_checkpoint_sha256",
        "modeling_sha256",
        "audited_expected_modeling_sha256",
    ):
        _require_sha256(foundation[field], field=f"foundation.{field}")
    if (
        foundation["checkpoint_sha256"] != AUDITED_LABRAM_BASE_SHA256
        or foundation["audited_expected_checkpoint_sha256"]
        != AUDITED_LABRAM_BASE_SHA256
        or foundation["modeling_sha256"] != AUDITED_LABRAM_MODELING_SHA256
        or foundation["audited_expected_modeling_sha256"]
        != AUDITED_LABRAM_MODELING_SHA256
        or foundation["token_shape"] != list(CONCEPT_TOKEN_SHAPE)
        or foundation["tile_seconds"] != 4
        or foundation["frozen"] is not True
        or foundation["materialization_device"] not in {"cpu", "cuda"}
    ):
        raise ValueError("Master index foundation identity changed")

    raw_events = index["events"]
    event_count = index["event_count"]
    patient_count = index["patient_count"]
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < EXPECTED_GATE_EVENT_COUNT
        or isinstance(patient_count, bool)
        or not isinstance(patient_count, int)
        or patient_count < EXPECTED_GATE_PATIENT_COUNT
        or not isinstance(raw_events, list)
        or len(raw_events) != event_count
        or master["event_count"] != event_count
        or master["patient_count"] != patient_count
    ):
        raise ValueError("Master index event/patient counts are invalid")
    events: list[dict[str, object]] = []
    for index_number, raw_event in enumerate(raw_events):
        event = _require_exact_fields(
            raw_event, _TOKEN_EVENT_FIELDS, field=f"events[{index_number}]"
        )
        patient = event["patient_id"]
        event_id = event["event_id"]
        if (
            not isinstance(patient, str)
            or not _PATIENT_RE.fullmatch(patient)
            or not isinstance(event_id, str)
            or not event_id.startswith(f"{patient}_")
            or event["bundle_path"] != f"events/{event_id}"
        ):
            raise ValueError("Master index contains a non-canonical event identity")
        for field in _TOKEN_EVENT_FIELDS - {"event_id", "patient_id", "bundle_path"}:
            _require_sha256(event[field], field=f"events[{index_number}].{field}")
        events.append(dict(event))
    if (
        tuple((event["patient_id"], event["event_id"]) for event in events)
        != tuple(sorted((event["patient_id"], event["event_id"]) for event in events))
        or len({event["event_id"] for event in events}) != event_count
    ):
        raise ValueError("Master index event order or uniqueness changed")
    all_patients = tuple(sorted({str(event["patient_id"]) for event in events}))
    patient_events = tuple(
        (
            patient,
            tuple(
                str(event["event_id"])
                for event in events
                if event["patient_id"] == patient
            ),
        )
        for patient in all_patients
    )
    event_roster = tuple(
        (
            event["event_id"],
            event["patient_id"],
            event["event_record_sha256"],
            event["preprocess_receipt_sha256"],
        )
        for event in events
    )
    receipts = {
        "event_roster_sha256": _canonical_sha256(event_roster),
        "patient_roster_sha256": _canonical_sha256(all_patients),
        "patient_event_roster_sha256": _canonical_sha256(patient_events),
        "tensor_roster_sha256": _canonical_sha256(
            tuple((event["event_id"], event["tensor_sha256"]) for event in events)
        ),
    }
    if len(all_patients) != patient_count or any(
        index[field] != expected for field, expected in receipts.items()
    ):
        raise ValueError("Master index roster receipts changed")

    selected = tuple(event for event in events if event["patient_id"] in set(patients))
    if (
        len(selected) != EXPECTED_GATE_EVENT_COUNT
        or {event["patient_id"] for event in selected} != set(patients)
    ):
        raise ValueError("Master index no longer contains the frozen 12/212 gate")
    selected_roster = tuple(
        (
            event["patient_id"],
            event["event_id"],
            event["event_record_sha256"],
            event["preprocess_receipt_sha256"],
            event["bundle_manifest_sha256"],
            event["tensor_sha256"],
        )
        for event in selected
    )
    return {
        "schema_version": "soz_labram_k31_i_gate_metadata_preflight_v13",
        "index_sha256": index_sha,
        "master_bundle_manifest_sha256": expected_master_bundle,
        "master_source_manifest_sha256": expected_master_source,
        "full_event_count": event_count,
        "full_patient_count": patient_count,
        "gate_patient_count": len(patients),
        "gate_event_count": len(selected),
        "gate_patient_roster_sha256": patient_roster_sha256(patients),
        "gate_event_roster_sha256": _canonical_sha256(selected_roster),
        "gate_token_values_loaded": False,
        "event_bundle_directories_enumerated": False,
        "prediction_forward_executed": False,
        "output_published": False,
        "execution_hold": V13_EXECUTION_HOLD,
        "execution_hold_blockers": list(V13_EXECUTION_HOLD_BLOCKERS),
    }


def _tensor_value_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {"dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape)}
    )
    return hashlib.sha256(header + b"\0" + value.numpy().tobytes(order="C")).hexdigest()


def _event_payload(event: FormalTokenSubsetEventBinding) -> dict[str, object]:
    return {
        "patient_id": event.patient_id,
        "event_id": event.event_id,
        "event_record_sha256": event.event_record_sha256,
        "preprocess_receipt_sha256": event.preprocess_receipt_sha256,
        "input_bundle_manifest_sha256": event.bundle_manifest_sha256,
        "input_tensor_sha256": event.tensor_sha256,
    }


def _event_roster_sha256(events: Sequence[Mapping[str, object]]) -> str:
    return _canonical_sha256(
        tuple(
            (
                event["patient_id"],
                event["event_id"],
                event["event_record_sha256"],
                event["preprocess_receipt_sha256"],
                event["input_bundle_manifest_sha256"],
                event["input_tensor_sha256"],
            )
            for event in events
        )
    )


def access_receipt_v13(
    *, prediction_forward_executed: bool, output_published: bool
) -> dict[str, object]:
    if output_published and not prediction_forward_executed:
        raise ValueError("An output cannot be published without prediction forward")
    return {
        "schema_version": GATE_ACCESS_RECEIPT_SCHEMA,
        "master_signal_index_loaded": True,
        "minimal_k31_inference_projection_loaded": True,
        "legacy_recovery_bundle_loaded": False,
        "legacy_recovery_manifest_metadata_loaded": False,
        "legacy_native_evaluation_roster_metadata_loaded": False,
        "legacy_training_run_metrics_loaded": False,
        "gate_token_values_loaded": True,
        "non_gate_token_bundles_opened": False,
        "deepsoz_source_eval_source_loaded": False,
        "deepsoz_identity_source_loaded": False,
        "filesystem_subset_confinement_proven": False,
        "execution_authorization_loaded": False,
        "tusz_target_values_loaded": False,
        "tusz_target_masks_loaded": False,
        "deepsoz_target_source_loaded": False,
        "deepsoz_soz_values_loaded": False,
        "private_eeg_loaded": False,
        "private_targets_loaded": False,
        "gate_split_bytes_hashed_only": True,
        "gate_split_json_parsed": False,
        "gate_outcomes_opened": False,
        "training": False,
        "calibration": False,
        "model_selection": False,
        "evaluation": False,
        "prediction_forward_executed": prediction_forward_executed,
        "output_published": output_published,
    }


@dataclass(frozen=True)
class GatePredictionProducerBindingV13:
    selection: str
    oof_fold: int | None
    projection_bundle_manifest_sha256: str
    manifest_sha256: str
    legacy_recovery_manifest_sha256: str
    checkpoint_sha256: str
    head_state_sha256: str
    fit_patient_ids: tuple[str, ...]
    fit_patient_roster_sha256: str
    i_gate_patient_roster_sha256: str
    v5_split_sha256: str

    def __post_init__(self) -> None:
        if self.selection not in EXPECTED_PRODUCER_ORDER:
            raise ValueError("Unknown producer selection")
        expected_fold = (
            None if self.selection == "final" else int(self.selection.removeprefix("fold"))
        )
        if self.oof_fold != expected_fold:
            raise ValueError("Producer selection and OOF fold disagree")
        for field in (
            "projection_bundle_manifest_sha256",
            "manifest_sha256",
            "legacy_recovery_manifest_sha256",
            "checkpoint_sha256",
            "head_state_sha256",
            "fit_patient_roster_sha256",
            "i_gate_patient_roster_sha256",
            "v5_split_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        roster = tuple(self.fit_patient_ids)
        if roster != tuple(sorted(roster)) or not roster or len(set(roster)) != len(roster):
            raise ValueError("Producer fit roster must be sorted and unique")
        if patient_roster_sha256(roster) != self.fit_patient_roster_sha256:
            raise ValueError("Producer fit roster receipt mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "selection": self.selection,
            "oof_fold": self.oof_fold,
            "projection_bundle_manifest_sha256": (
                self.projection_bundle_manifest_sha256
            ),
            "manifest_sha256": self.manifest_sha256,
            "legacy_recovery_manifest_sha256": (
                self.legacy_recovery_manifest_sha256
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
            "head_state_sha256": self.head_state_sha256,
            "fit_patient_ids": list(self.fit_patient_ids),
            "fit_patient_roster_sha256": self.fit_patient_roster_sha256,
            "i_gate_patient_roster_sha256": self.i_gate_patient_roster_sha256,
            "v5_split_sha256": self.v5_split_sha256,
            "upstream_tusz_ictal_involvement_targets_loaded": True,
            "legacy_full_manifest_loaded_by_broker": True,
            "legacy_native_evaluation_roster_metadata_loaded_by_broker": True,
            "legacy_training_run_metrics_loaded_by_broker": True,
            "legacy_checkpoint_weights_loaded_by_broker": True,
            "legacy_training_process_loaded_full_tusz_target_snapshot_arrays": True,
            "legacy_training_process_snapshot_contained_gate_rows": True,
            "upstream_i_gate_used_for_fit_loss_gradient_or_native_metric": False,
            "upstream_i_gate_confirmation_metrics_computed": False,
        }


@dataclass(frozen=True)
class PreparedGatePredictionMaterializationV13:
    v5_split_sha256: str
    gate_patient_ids: tuple[str, ...]
    gate_patient_roster_sha256: str
    token_corpus: VerifiedFormalTokenCorpusSubsetArtifact
    producer_bindings: tuple[GatePredictionProducerBindingV13, ...]
    inference_projection: LoadedK31InferenceProjectionV13

    def __post_init__(self) -> None:
        _require_sha256(self.v5_split_sha256, field="v5_split_sha256")
        if (
            len(self.gate_patient_ids) != EXPECTED_GATE_PATIENT_COUNT
            or self.gate_patient_ids != tuple(sorted(self.gate_patient_ids))
            or len(set(self.gate_patient_ids)) != len(self.gate_patient_ids)
            or any(not _PATIENT_RE.fullmatch(value) for value in self.gate_patient_ids)
        ):
            raise ValueError("Prepared gate roster is not the frozen 12-patient roster")
        if patient_roster_sha256(self.gate_patient_ids) != self.gate_patient_roster_sha256:
            raise ValueError("Prepared gate roster receipt mismatch")
        if not isinstance(self.token_corpus, VerifiedFormalTokenCorpusSubsetArtifact):
            raise TypeError("token_corpus must come from the selective strict loader")
        if self.token_corpus.unselected_event_bundles_opened is not False:
            raise ValueError("Prepared corpus opened non-gate bundles")
        if self.token_corpus.selected_patient_ids != self.gate_patient_ids:
            raise ValueError("Prepared corpus patient roster changed")
        if self.token_corpus.selected_event_count != EXPECTED_GATE_EVENT_COUNT:
            raise ValueError("Prepared corpus does not contain exactly 212 gate events")
        if tuple(item.selection for item in self.producer_bindings) != EXPECTED_PRODUCER_ORDER:
            raise ValueError("Producer binding order changed")
        if not isinstance(
            self.inference_projection, LoadedK31InferenceProjectionV13
        ):
            raise TypeError("All six producers must come from the strict projection")
        for binding, producer in zip(
            self.producer_bindings, self.inference_projection.producers, strict=True
        ):
            if (
                binding.manifest_sha256 != producer.projection_record_sha256
                or binding.projection_bundle_manifest_sha256
                != self.inference_projection.manifest_sha256
            ):
                raise ValueError("Producer binding and loaded projection drifted")

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.token_corpus.events)

    def preflight_payload(self) -> dict[str, object]:
        return {
            "schema_version": "soz_labram_k31_i_gate_prediction_preflight_v13",
            "v5_split_sha256": self.v5_split_sha256,
            "producer_order": list(EXPECTED_PRODUCER_ORDER),
            "producer_manifest_sha256s": [
                item.manifest_sha256 for item in self.producer_bindings
            ],
            "inference_projection_manifest_sha256": (
                self.inference_projection.manifest_sha256
            ),
            "gate_patient_count": len(self.gate_patient_ids),
            "gate_event_count": self.token_corpus.selected_event_count,
            "gate_patient_roster_sha256": self.gate_patient_roster_sha256,
            "gate_event_roster_sha256": self.token_corpus.selected_event_roster_sha256,
            "corpus_index_sha256": self.token_corpus.index_sha256,
            "access_receipt": access_receipt_v13(
                prediction_forward_executed=False, output_published=False
            ),
        }


@dataclass(frozen=True)
class LoadedGatePredictionArtifactV13:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    logits: torch.Tensor


def _producer_binding(
    projection: LoadedK31InferenceProjectionV13,
    producer: LoadedK31InferenceProjectionProducerV13,
) -> GatePredictionProducerBindingV13:
    return GatePredictionProducerBindingV13(
        selection=producer.selection,
        oof_fold=producer.oof_fold,
        projection_bundle_manifest_sha256=projection.manifest_sha256,
        manifest_sha256=producer.projection_record_sha256,
        legacy_recovery_manifest_sha256=(
            producer.legacy_recovery_manifest_sha256
        ),
        checkpoint_sha256=producer.checkpoint_sha256,
        head_state_sha256=producer.head_state_sha256,
        fit_patient_ids=producer.fit_patient_ids,
        fit_patient_roster_sha256=producer.fit_patient_roster_sha256,
        i_gate_patient_roster_sha256=producer.gate_patient_roster_sha256,
        v5_split_sha256=producer.v5_split_sha256,
    )


def prepare_gate_prediction_materialization_v13(
    *,
    inference_projection: LoadedK31InferenceProjectionV13,
    token_corpus: VerifiedFormalTokenCorpusSubsetArtifact,
    v5_split_sha256: str,
) -> PreparedGatePredictionMaterializationV13:
    """Close all identities and firewalls without executing a model forward."""

    split_sha = _require_sha256(v5_split_sha256, field="v5_split_sha256")
    if not isinstance(inference_projection, LoadedK31InferenceProjectionV13):
        raise TypeError("A strict minimal k31 inference projection is required")
    producers = inference_projection.producers
    if tuple(item.selection for item in producers) != EXPECTED_PRODUCER_ORDER:
        raise ValueError("Projection producer order must be fold0..fold4, final")
    first_gate = inference_projection.gate_patient_ids
    gate_sha = patient_roster_sha256(first_gate)
    if (
        inference_projection.gate_patient_roster_sha256 != gate_sha
        or inference_projection.v5_split_sha256 != split_sha
        or inference_projection.manifest.get("v13_execution_hold") is not True
    ):
        raise ValueError("Projection I-gate identity or HOLD changed")
    for producer in producers:
        manifest = producer.manifest
        if (
            producer.gate_patient_ids != first_gate
            or producer.gate_patient_roster_sha256 != gate_sha
            or producer.v5_split_sha256 != split_sha
        ):
            raise ValueError("Producer I-gate identity changed")
        if set(producer.fit_patient_ids) & set(first_gate):
            raise ValueError("Producer fit roster leaks an I-gate patient")
        historical = {
            "legacy_training_process_loaded_full_tusz_target_snapshot_arrays": True,
            "legacy_training_process_snapshot_contained_gate_rows": True,
            "legacy_gate_rows_used_for_fit_loss_gradient_or_native_metric": False,
            "legacy_gate_confirmation_metrics_computed": False,
        }
        if any(manifest[field] is not expected for field, expected in historical.items()):
            raise ValueError("Projection softened historical target exposure")
    manifest_hashes = tuple(item.projection_record_sha256 for item in producers)
    checkpoint_hashes = tuple(item.checkpoint_sha256 for item in producers)
    if len(set(manifest_hashes)) != 6 or len(set(checkpoint_hashes)) != 6:
        raise ValueError("The six fixed producers must be distinct")

    if not isinstance(token_corpus, VerifiedFormalTokenCorpusSubsetArtifact):
        raise TypeError("token_corpus must come from the selective strict loader")
    if (
        token_corpus.training_bundle_manifest_sha256
        != token_corpus.master_bundle_manifest_sha256
        or token_corpus.training_source_manifest_sha256
        != token_corpus.master_source_manifest_sha256
    ):
        raise ValueError("Gate inference requires the exact master token corpus")
    if token_corpus.unselected_event_bundles_opened is not False:
        raise ValueError("Non-gate token bundles were opened")
    if (
        token_corpus.selected_patient_ids != first_gate
        or token_corpus.selected_event_count != EXPECTED_GATE_EVENT_COUNT
        or len(first_gate) != EXPECTED_GATE_PATIENT_COUNT
    ):
        raise ValueError("Selective token corpus is not the frozen 12/212 I-gate")
    if formal_token_subset_roster_sha256(token_corpus.events) != (
        token_corpus.selected_event_roster_sha256
    ):
        raise ValueError("Gate event roster receipt drifted")

    bindings = tuple(
        _producer_binding(inference_projection, producer)
        for producer in producers
    )
    return PreparedGatePredictionMaterializationV13(
        v5_split_sha256=split_sha,
        gate_patient_ids=first_gate,
        gate_patient_roster_sha256=gate_sha,
        token_corpus=token_corpus,
        producer_bindings=bindings,
        inference_projection=inference_projection,
    )


def _execution_receipt(device: torch.device) -> dict[str, object]:
    device_name = "cpu"
    capability = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
    return {
        "schema_version": GATE_EXECUTION_RECEIPT_SCHEMA,
        "torch_version": str(torch.__version__),
        "device_type": device.type,
        "device_name": device_name,
        "cuda_runtime_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "compute_capability": capability,
        "inference_mode_used": True,
        "gradient_enabled_during_forward": False,
        "batch_size": 1,
    }


def _validate_execution_receipt(value: object) -> dict[str, object]:
    receipt = _require_exact_fields(value, _EXECUTION_FIELDS, field="execution_receipt")
    if receipt["schema_version"] != GATE_EXECUTION_RECEIPT_SCHEMA:
        raise ValueError("Unsupported gate execution receipt schema")
    for field in ("torch_version", "device_type", "device_name"):
        if not isinstance(receipt[field], str) or not receipt[field]:
            raise ValueError(f"execution_receipt.{field} must be non-empty")
    if receipt["device_type"] not in {"cpu", "cuda"}:
        raise ValueError("Execution device must be cpu or cuda")
    if receipt["cuda_runtime_version"] is not None and not isinstance(
        receipt["cuda_runtime_version"], str
    ):
        raise TypeError("CUDA runtime version must be string or null")
    if receipt["cudnn_version"] is not None and (
        isinstance(receipt["cudnn_version"], bool)
        or not isinstance(receipt["cudnn_version"], int)
        or receipt["cudnn_version"] < 1
    ):
        raise ValueError("cuDNN version must be positive or null")
    capability = receipt["compute_capability"]
    if capability is not None and (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability)
    ):
        raise ValueError("Compute capability is invalid")
    if (
        receipt["inference_mode_used"] is not True
        or receipt["gradient_enabled_during_forward"] is not False
        or receipt["batch_size"] != 1
    ):
        raise ValueError("Gate execution must use batch-one inference without gradients")
    return dict(receipt)


def _corpus_payload(
    corpus: VerifiedFormalTokenCorpusSubsetArtifact,
) -> dict[str, object]:
    return {
        "index_sha256": corpus.index_sha256,
        "master_bundle_manifest_sha256": corpus.master_bundle_manifest_sha256,
        "master_source_manifest_sha256": corpus.master_source_manifest_sha256,
        "training_bundle_manifest_sha256": corpus.training_bundle_manifest_sha256,
        "training_source_manifest_sha256": corpus.training_source_manifest_sha256,
        "preprocessing_selection_artifact_sha256": (
            corpus.preprocessing_selection_artifact_sha256
        ),
        "preprocessing_selection_bundle_receipt_sha256": (
            corpus.preprocessing_selection_bundle_receipt_sha256
        ),
        "preprocessing_protocol_receipt_sha256": (
            corpus.preprocessing_protocol_receipt_sha256
        ),
        "preprocessing_selected_arm_result_receipt_sha256": (
            corpus.preprocessing_selected_arm_result_receipt_sha256
        ),
        "preprocessing_selected_arm_id": corpus.preprocessing_selected_arm_id,
        "foundation_feature_receipt_sha256": corpus.foundation_feature_receipt_sha256,
        "foundation_checkpoint_sha256": corpus.foundation_checkpoint_sha256,
        "foundation_modeling_sha256": corpus.foundation_modeling_sha256,
        "full_event_count": corpus.full_event_count,
        "full_patient_count": corpus.full_patient_count,
        "selected_event_count": corpus.selected_event_count,
        "selected_event_roster_sha256": corpus.selected_event_roster_sha256,
        "unselected_event_bundles_opened": False,
    }


def _build_manifest(
    prepared: PreparedGatePredictionMaterializationV13,
    *,
    tensor_file_size: int,
    tensor_file_sha256: str,
    tensor_value_sha256: str,
    tensor_shape: Sequence[int],
    execution_receipt: Mapping[str, object],
) -> dict[str, object]:
    events = [_event_payload(event) for event in prepared.token_corpus.events]
    return {
        "schema_version": GATE_PREDICTION_SCHEMA_V13,
        "purpose": GATE_PREDICTION_PURPOSE,
        "serialization": GATE_PREDICTION_SERIALIZATION,
        "candidate": EXPECTED_CANDIDATE,
        "context_seconds": EXPECTED_CONTEXT_SECONDS,
        "context_direction": EXPECTED_CONTEXT_DIRECTION,
        "target_semantics": EXPECTED_TARGET_SEMANTICS,
        "logit_semantics": "raw_per_producer_bipolar_edge_second_logits_no_sigmoid",
        "development_only": True,
        "formal_promotion": False,
        "authorized_for_formal_evidence_or_reasoner": False,
        "candidate_logits_present": True,
        "independent_control_logits_present": False,
        "time_only_control_logits_present": False,
        "prevalence_control_logits_present": False,
        "mask_only_control_logits_present": False,
        "target_free_scale_probe_passed": False,
        "target_free_fold_identity_probe_passed": False,
        "complete_stage_a_seal": False,
        "execution_authorized": False,
        "execution_hold_blockers": list(V13_EXECUTION_HOLD_BLOCKERS),
        "v5_split_sha256": prepared.v5_split_sha256,
        "inference_projection_manifest_sha256": (
            prepared.inference_projection.manifest_sha256
        ),
        "producer_order": list(EXPECTED_PRODUCER_ORDER),
        "producers": [item.to_payload() for item in prepared.producer_bindings],
        "gate_patient_ids": list(prepared.gate_patient_ids),
        "gate_patient_roster_sha256": prepared.gate_patient_roster_sha256,
        "gate_patient_count": len(prepared.gate_patient_ids),
        "corpus": _corpus_payload(prepared.token_corpus),
        "event_count": len(events),
        "event_roster_sha256": _event_roster_sha256(events),
        "events": events,
        "tensor": {
            "filename": GATE_PREDICTION_TENSOR_FILENAME,
            "name": GATE_PREDICTION_TENSOR_NAME,
            "dtype": "float32",
            "shape": list(tensor_shape),
            "file_size_bytes": tensor_file_size,
            "file_sha256": tensor_file_sha256,
            "value_sha256": tensor_value_sha256,
        },
        "execution_receipt": dict(execution_receipt),
        "access_receipt": access_receipt_v13(
            prediction_forward_executed=True, output_published=True
        ),
    }


def _validate_manifest(value: dict[str, object]) -> dict[str, object]:
    manifest = _require_exact_fields(value, _MANIFEST_FIELDS, field="manifest")
    fixed = {
        "schema_version": GATE_PREDICTION_SCHEMA_V13,
        "purpose": GATE_PREDICTION_PURPOSE,
        "serialization": GATE_PREDICTION_SERIALIZATION,
        "candidate": EXPECTED_CANDIDATE,
        "context_seconds": EXPECTED_CONTEXT_SECONDS,
        "context_direction": EXPECTED_CONTEXT_DIRECTION,
        "target_semantics": EXPECTED_TARGET_SEMANTICS,
        "logit_semantics": "raw_per_producer_bipolar_edge_second_logits_no_sigmoid",
        "development_only": True,
        "formal_promotion": False,
        "authorized_for_formal_evidence_or_reasoner": False,
        "candidate_logits_present": True,
        "independent_control_logits_present": False,
        "time_only_control_logits_present": False,
        "prevalence_control_logits_present": False,
        "mask_only_control_logits_present": False,
        "target_free_scale_probe_passed": False,
        "target_free_fold_identity_probe_passed": False,
        "complete_stage_a_seal": False,
        "execution_authorized": False,
        "execution_hold_blockers": list(V13_EXECUTION_HOLD_BLOCKERS),
    }
    if any(manifest[field] != expected for field, expected in fixed.items()):
        raise ValueError("Prediction manifest changed a scientific boundary")
    split_sha = _require_sha256(manifest["v5_split_sha256"], field="v5_split_sha256")
    projection_sha = _require_sha256(
        manifest["inference_projection_manifest_sha256"],
        field="inference_projection_manifest_sha256",
    )

    raw_patients = manifest["gate_patient_ids"]
    if not isinstance(raw_patients, list) or any(not isinstance(item, str) for item in raw_patients):
        raise TypeError("gate_patient_ids must be a string list")
    patients = tuple(raw_patients)
    if (
        len(patients) != EXPECTED_GATE_PATIENT_COUNT
        or patients != tuple(sorted(patients))
        or len(set(patients)) != len(patients)
        or any(not _PATIENT_RE.fullmatch(patient) for patient in patients)
        or manifest["gate_patient_count"] != len(patients)
    ):
        raise ValueError("Prediction manifest gate roster is invalid")
    gate_sha = _require_sha256(
        manifest["gate_patient_roster_sha256"], field="gate_patient_roster_sha256"
    )
    if gate_sha != patient_roster_sha256(patients):
        raise ValueError("Prediction manifest gate roster receipt mismatch")

    if manifest["producer_order"] != list(EXPECTED_PRODUCER_ORDER):
        raise ValueError("Prediction producer order changed")
    raw_producers = manifest["producers"]
    if not isinstance(raw_producers, list) or len(raw_producers) != 6:
        raise ValueError("Prediction manifest must contain six producers")
    producer_manifests: list[str] = []
    producer_checkpoints: list[str] = []
    for index, raw_producer in enumerate(raw_producers):
        producer = _require_exact_fields(
            raw_producer, _PRODUCER_FIELDS, field=f"producers[{index}]"
        )
        selection = EXPECTED_PRODUCER_ORDER[index]
        expected_fold = None if selection == "final" else index
        if producer["selection"] != selection or producer["oof_fold"] != expected_fold:
            raise ValueError("Producer selection/fold order changed")
        for field in (
            "projection_bundle_manifest_sha256",
            "manifest_sha256",
            "legacy_recovery_manifest_sha256",
            "checkpoint_sha256",
            "head_state_sha256",
            "fit_patient_roster_sha256",
            "i_gate_patient_roster_sha256",
            "v5_split_sha256",
        ):
            _require_sha256(producer[field], field=f"producers[{index}].{field}")
        fit = producer["fit_patient_ids"]
        if not isinstance(fit, list) or any(not isinstance(item, str) for item in fit):
            raise TypeError("Producer fit roster must be a string list")
        fit_tuple = tuple(fit)
        if (
            not fit_tuple
            or fit_tuple != tuple(sorted(fit_tuple))
            or len(set(fit_tuple)) != len(fit_tuple)
            or set(fit_tuple) & set(patients)
            or patient_roster_sha256(fit_tuple)
            != producer["fit_patient_roster_sha256"]
        ):
            raise ValueError("Producer fit roster leaks or changed")
        if (
            producer["projection_bundle_manifest_sha256"] != projection_sha
            or producer["i_gate_patient_roster_sha256"] != gate_sha
            or producer["v5_split_sha256"] != split_sha
            or producer["upstream_tusz_ictal_involvement_targets_loaded"] is not True
            or producer["legacy_full_manifest_loaded_by_broker"] is not True
            or producer[
                "legacy_native_evaluation_roster_metadata_loaded_by_broker"
            ] is not True
            or producer["legacy_training_run_metrics_loaded_by_broker"] is not True
            or producer["legacy_checkpoint_weights_loaded_by_broker"] is not True
            or producer[
                "legacy_training_process_loaded_full_tusz_target_snapshot_arrays"
            ] is not True
            or producer[
                "legacy_training_process_snapshot_contained_gate_rows"
            ] is not True
            or producer[
                "upstream_i_gate_used_for_fit_loss_gradient_or_native_metric"
            ] is not False
            or producer["upstream_i_gate_confirmation_metrics_computed"] is not False
        ):
            raise ValueError("Producer access lineage changed")
        producer_manifests.append(str(producer["manifest_sha256"]))
        producer_checkpoints.append(str(producer["checkpoint_sha256"]))
    if len(set(producer_manifests)) != 6 or len(set(producer_checkpoints)) != 6:
        raise ValueError("Prediction manifest duplicated a producer")

    corpus = _require_exact_fields(manifest["corpus"], _CORPUS_FIELDS, field="corpus")
    for field in _CORPUS_FIELDS - {
        "preprocessing_selected_arm_id",
        "full_event_count",
        "full_patient_count",
        "selected_event_count",
        "unselected_event_bundles_opened",
    }:
        _require_sha256(corpus[field], field=f"corpus.{field}")
    if (
        corpus["preprocessing_selected_arm_id"] != "C-CAR19"
        or corpus["training_bundle_manifest_sha256"]
        != corpus["master_bundle_manifest_sha256"]
        or corpus["training_source_manifest_sha256"]
        != corpus["master_source_manifest_sha256"]
        or corpus["unselected_event_bundles_opened"] is not False
    ):
        raise ValueError("Prediction corpus is not the sealed C-CAR19 master subset")
    for field in ("full_event_count", "full_patient_count", "selected_event_count"):
        if isinstance(corpus[field], bool) or not isinstance(corpus[field], int) or corpus[field] < 1:
            raise ValueError(f"corpus.{field} must be positive")

    event_count = manifest["event_count"]
    raw_events = manifest["events"]
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count != EXPECTED_GATE_EVENT_COUNT
        or corpus["selected_event_count"] != event_count
        or not isinstance(raw_events, list)
        or len(raw_events) != event_count
    ):
        raise ValueError("Prediction event count is not the frozen 212-event gate")
    normalized_events: list[dict[str, object]] = []
    for index, raw_event in enumerate(raw_events):
        event = _require_exact_fields(raw_event, _EVENT_FIELDS, field=f"events[{index}]")
        patient = event["patient_id"]
        event_id = event["event_id"]
        if (
            not isinstance(patient, str)
            or patient not in patients
            or not isinstance(event_id, str)
            or not event_id.startswith(f"{patient}_")
        ):
            raise ValueError("Prediction event identity is invalid")
        for field in _EVENT_FIELDS - {"patient_id", "event_id"}:
            _require_sha256(event[field], field=f"events[{index}].{field}")
        normalized_events.append(dict(event))
    if (
        tuple((event["patient_id"], event["event_id"]) for event in normalized_events)
        != tuple(sorted((event["patient_id"], event["event_id"]) for event in normalized_events))
        or len({event["event_id"] for event in normalized_events}) != event_count
        or {event["patient_id"] for event in normalized_events} != set(patients)
    ):
        raise ValueError("Prediction events are not canonical, unique, and complete")
    event_sha = _require_sha256(manifest["event_roster_sha256"], field="event_roster_sha256")
    if event_sha != _event_roster_sha256(normalized_events):
        raise ValueError("Prediction event roster receipt mismatch")
    if corpus["selected_event_roster_sha256"] != event_sha:
        raise ValueError("Prediction events differ from the selective corpus receipt")
    tensor = _require_exact_fields(manifest["tensor"], _TENSOR_FIELDS, field="tensor")
    expected_shape = [6, event_count, *EXPECTED_LOGIT_TAIL]
    if (
        tensor["filename"] != GATE_PREDICTION_TENSOR_FILENAME
        or tensor["name"] != GATE_PREDICTION_TENSOR_NAME
        or tensor["dtype"] != "float32"
        or tensor["shape"] != expected_shape
        or isinstance(tensor["file_size_bytes"], bool)
        or not isinstance(tensor["file_size_bytes"], int)
        or not 1 <= tensor["file_size_bytes"] <= _MAX_TENSOR_BYTES
    ):
        raise ValueError("Prediction tensor specification changed")
    _require_sha256(tensor["file_sha256"], field="tensor.file_sha256")
    _require_sha256(tensor["value_sha256"], field="tensor.value_sha256")

    access = _require_exact_fields(
        manifest["access_receipt"], _ACCESS_FIELDS, field="access_receipt"
    )
    if access != access_receipt_v13(
        prediction_forward_executed=True, output_published=True
    ):
        raise ValueError("Prediction access firewall changed")
    manifest["execution_receipt"] = _validate_execution_receipt(
        manifest["execution_receipt"]
    )
    manifest["events"] = normalized_events
    return manifest


def _validate_logits(logits: torch.Tensor, *, event_count: int) -> torch.Tensor:
    expected_shape = (6, event_count, *EXPECTED_LOGIT_TAIL)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.dtype != torch.float32 or tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"logits must have shape {expected_shape} and dtype float32"
        )
    value = logits.detach().to(device="cpu").contiguous()
    if not torch.isfinite(value).all():
        raise ValueError("logits must be finite")
    return value


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_gate_prediction_artifact_v13_unsealed(
    output_directory: str | Path,
    *,
    prepared: PreparedGatePredictionMaterializationV13,
    logits: torch.Tensor,
    execution_receipt: Mapping[str, object],
) -> LoadedGatePredictionArtifactV13:
    """Atomically publish one closed prediction tensor without evaluation."""

    if not isinstance(prepared, PreparedGatePredictionMaterializationV13):
        raise TypeError("prepared must be a validated v13 materialization")
    value = _validate_logits(logits, event_count=prepared.token_corpus.selected_event_count)
    execution = _validate_execution_receipt(dict(execution_receipt))
    target = _reject_symlink_components(Path(output_directory), field="output directory")
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("Output requires a concrete path with an existing parent")
    if os.path.lexists(target):
        raise FileExistsError(f"Output already exists: {target}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    moved = False
    try:
        tensor_path = staging / GATE_PREDICTION_TENSOR_FILENAME
        _save_safetensors_file({GATE_PREDICTION_TENSOR_NAME: value}, str(tensor_path))
        tensor_raw, tensor_file_sha = _read_stable_regular_file(
            tensor_path,
            field="staged prediction tensor",
            maximum_bytes=_MAX_TENSOR_BYTES,
        )
        manifest = _build_manifest(
            prepared,
            tensor_file_size=len(tensor_raw),
            tensor_file_sha256=tensor_file_sha,
            tensor_value_sha256=_tensor_value_sha256(value),
            tensor_shape=value.shape,
            execution_receipt=execution,
        )
        raw = _canonical_json_bytes(manifest)
        manifest_path = staging / GATE_PREDICTION_MANIFEST_FILENAME
        with manifest_path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_file(tensor_path)
        _fsync_directory(staging)
        manifest_sha = hashlib.sha256(raw).hexdigest()
        load_gate_prediction_artifact_v13(
            staging,
            expected_manifest_sha256=manifest_sha,
            expected_producer_manifest_sha256s=tuple(
                binding.manifest_sha256 for binding in prepared.producer_bindings
            ),
        )
        os.rename(staging, target)
        moved = True
        _fsync_directory(target.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    if not moved:
        raise RuntimeError("Gate prediction artifact was not atomically published")
    return load_gate_prediction_artifact_v13(
        target,
        expected_manifest_sha256=manifest_sha,
        expected_producer_manifest_sha256s=tuple(
            binding.manifest_sha256 for binding in prepared.producer_bindings
        ),
    )


def load_gate_prediction_artifact_v13(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_producer_manifest_sha256s: Sequence[str] | None = None,
) -> LoadedGatePredictionArtifactV13:
    """Strictly load a v13 tensor and reject schema, lineage, or value drift."""

    source = _reject_symlink_components(Path(path), field="prediction artifact")
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Prediction artifact must be a regular directory")
    if {entry.name for entry in source.iterdir()} != {
        GATE_PREDICTION_MANIFEST_FILENAME,
        GATE_PREDICTION_TENSOR_FILENAME,
    }:
        raise ValueError("Prediction artifact has missing or unknown files")
    manifest_raw, manifest_sha = _read_stable_regular_file(
        source / GATE_PREDICTION_MANIFEST_FILENAME,
        field="prediction manifest",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Prediction manifest SHA-256 mismatch")
    manifest = _validate_manifest(_parse_canonical_json(manifest_raw))
    producer_hashes = tuple(
        str(producer["manifest_sha256"]) for producer in manifest["producers"]
    )
    if expected_producer_manifest_sha256s is not None:
        expected = tuple(
            _require_sha256(value, field="expected_producer_manifest_sha256")
            for value in expected_producer_manifest_sha256s
        )
        if expected != producer_hashes:
            raise ValueError("Prediction producer manifest lineage drifted")

    tensor_raw, tensor_file_sha = _read_stable_regular_file(
        source / GATE_PREDICTION_TENSOR_FILENAME,
        field="prediction tensor",
        maximum_bytes=_MAX_TENSOR_BYTES,
    )
    tensor_spec = manifest["tensor"]
    if (
        len(tensor_raw) != tensor_spec["file_size_bytes"]
        or tensor_file_sha != tensor_spec["file_sha256"]
    ):
        raise ValueError("Prediction tensor file drifted")
    try:
        tensors = _load_safetensors_bytes(tensor_raw)
    except Exception as exc:
        raise ValueError("Prediction tensor is not valid safetensors") from exc
    if set(tensors) != {GATE_PREDICTION_TENSOR_NAME}:
        raise ValueError("Prediction safetensors names changed")
    logits = _validate_logits(
        tensors[GATE_PREDICTION_TENSOR_NAME], event_count=int(manifest["event_count"])
    )
    if _tensor_value_sha256(logits) != tensor_spec["value_sha256"]:
        raise ValueError("Prediction tensor values drifted")
    return LoadedGatePredictionArtifactV13(
        path=source,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        logits=logits,
    )


def _validate_forward_token(
    event: FormalTokenSubsetEventBinding,
    corpus: VerifiedFormalTokenCorpusSubsetArtifact,
):
    token = load_labram_concept_tokens(
        event.bundle_path,
        expected_manifest_sha256=event.bundle_manifest_sha256,
    )
    checks = {
        "event_id": token.event_id == event.event_id,
        "event_record": token.event_record_sha256 == event.event_record_sha256,
        "preprocess": token.preprocess_receipt_sha256 == event.preprocess_receipt_sha256,
        "source_manifest": (
            token.source_concept_manifest_sha256 == corpus.training_source_manifest_sha256
        ),
        "foundation_receipt": (
            token.foundation_feature_receipt_sha256
            == corpus.foundation_feature_receipt_sha256
        ),
        "foundation_checkpoint": (
            token.foundation_checkpoint_sha256 == corpus.foundation_checkpoint_sha256
        ),
        "foundation_modeling": (
            token.foundation_feature_receipt.modeling_sha256
            == corpus.foundation_modeling_sha256
        ),
        "tensor": token.tensor_sha256 == event.tensor_sha256,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Forward token {event.event_id} drifted in fields {failed}")
    if tuple(token.tokens.shape) != CONCEPT_TOKEN_SHAPE or token.tokens.dtype != torch.float32:
        raise ValueError("Forward token shape or dtype changed")
    return token


def _materialize_gate_predictions_v13_unsealed(
    *,
    prepared: PreparedGatePredictionMaterializationV13,
    output_directory: str | Path,
    device: str | torch.device = "cuda",
) -> LoadedGatePredictionArtifactV13:
    """Execute the six fixed heads; do not aggregate, evaluate, or promote."""

    if not isinstance(prepared, PreparedGatePredictionMaterializationV13):
        raise TypeError("prepared must be a validated v13 materialization")
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"} or execution_device.index is not None:
        raise ValueError("device must be unindexed cpu or cuda")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    heads = tuple(
        producer.head.to(execution_device).eval()
        for producer in prepared.inference_projection.producers
    )
    logits = torch.empty(
        (len(heads), prepared.token_corpus.selected_event_count, *EXPECTED_LOGIT_TAIL),
        dtype=torch.float32,
        device="cpu",
    )
    try:
        with torch.inference_mode():
            for event_index, event in enumerate(prepared.token_corpus.events):
                token = _validate_forward_token(event, prepared.token_corpus)
                inputs = token.tokens.unsqueeze(0).to(execution_device)
                for producer_index, head in enumerate(heads):
                    prediction = head(inputs)
                    expected_shape = (1, *EXPECTED_LOGIT_TAIL)
                    if (
                        tuple(prediction.shape) != expected_shape
                        or prediction.dtype != torch.float32
                        or not torch.isfinite(prediction).all()
                    ):
                        raise ValueError(
                            "Producer forward changed the fixed [1,20,60,1] "
                            "finite-float32 contract"
                        )
                    logits[producer_index, event_index].copy_(prediction[0].cpu())
        return _save_gate_prediction_artifact_v13_unsealed(
            output_directory,
            prepared=prepared,
            logits=logits,
            execution_receipt=_execution_receipt(execution_device),
        )
    finally:
        for head in heads:
            head.to("cpu")


def save_gate_prediction_artifact_v13(
    output_directory: str | Path,
    *,
    prepared: PreparedGatePredictionMaterializationV13,
    logits: torch.Tensor,
    execution_receipt: Mapping[str, object],
) -> LoadedGatePredictionArtifactV13:
    """Fail closed until the complete Stage-A seal is independently authorized."""

    if V13_EXECUTION_HOLD:
        raise RuntimeError(
            "V13_EXECUTION_HOLD forbids publishing the candidate-only component; "
            "missing=" + ",".join(V13_EXECUTION_HOLD_BLOCKERS)
        )
    return _save_gate_prediction_artifact_v13_unsealed(
        output_directory,
        prepared=prepared,
        logits=logits,
        execution_receipt=execution_receipt,
    )


def materialize_gate_predictions_v13(
    *,
    prepared: PreparedGatePredictionMaterializationV13,
    output_directory: str | Path,
    device: str | torch.device = "cuda",
) -> LoadedGatePredictionArtifactV13:
    """Fail closed while v13 lacks controls, probes, and authorization."""

    if V13_EXECUTION_HOLD:
        raise RuntimeError(
            "V13_EXECUTION_HOLD forbids real token forward; missing="
            + ",".join(V13_EXECUTION_HOLD_BLOCKERS)
        )
    return _materialize_gate_predictions_v13_unsealed(
        prepared=prepared,
        output_directory=output_directory,
        device=device,
    )


def _materialize_gate_predictions_v13_for_synthetic_test(
    *,
    prepared: PreparedGatePredictionMaterializationV13,
    output_directory: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedGatePredictionArtifactV13:
    """Private test seam; never exposed by the CLI or ``__all__``."""

    return _materialize_gate_predictions_v13_unsealed(
        prepared=prepared,
        output_directory=output_directory,
        device=device,
    )


__all__ = (
    "EXPECTED_GATE_EVENT_COUNT",
    "EXPECTED_GATE_PATIENT_COUNT",
    "EXPECTED_PRODUCER_ORDER",
    "GATE_PREDICTION_SCHEMA_V13",
    "GatePredictionProducerBindingV13",
    "LoadedGatePredictionArtifactV13",
    "PreparedGatePredictionMaterializationV13",
    "access_receipt_v13",
    "hash_gate_split_without_parsing",
    "load_gate_prediction_artifact_v13",
    "materialize_gate_predictions_v13",
    "prepare_gate_prediction_materialization_v13",
    "save_gate_prediction_artifact_v13",
)
