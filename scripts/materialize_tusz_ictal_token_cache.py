#!/usr/bin/env python3
"""Materialize a formal, target-free TUSZ ictal LaBraM token corpus.

The command accepts only frozen, fully signal-preflighted TUSZ training
manifests.  It never reads TUSZ target sidecars or DeepSOZ labels.  Every raw
EDF event is replayed with the manifest's causal preprocessing policy, checked
against its frozen signal-preflight receipt, passed through the audited frozen
LaBraM-Base encoder, and serialized with ``concept_token_io``.

All per-event bundles and one canonical corpus index are built in a sibling
staging directory.  The formal output directory is published once, only after
the complete event/patient roster and every pinned token bundle have been
strictly reloaded.  Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Callable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.concept_token_io import (  # noqa: E402
    CONCEPT_TOKEN_PURPOSE,
    CONCEPT_TOKEN_SHAPE,
    LaBraMConceptTokenArtifact,
    labram_feature_receipt_sha256,
    load_labram_concept_tokens,
    save_labram_concept_tokens,
)
from src.soz.data.edf import (  # noqa: E402
    CausalEDFConfig,
    EDF_PREPROCESS_SCHEMA,
    load_standard19_edf_event,
)
from src.soz.data.tusz_training import (  # noqa: E402
    TUSZIctalEventRecord,
    TUSZIctalTrainingManifest,
    load_tusz_ictal_training_manifest,
    parse_tusz_official_train_path,
    tusz_signal_preflight_receipt_sha256,
)
from src.soz.models.foundation import TiledFoundationEncoder  # noqa: E402
from src.soz.formal_token_corpus import (  # noqa: E402
    FormalTokenEventBinding,
    FormalTokenSubsetEventBinding,
    VerifiedFormalTokenCorpusArtifact,
    VerifiedFormalTokenCorpusSubsetArtifact,
    _issue_verified_formal_token_corpus_artifact,
    _issue_verified_formal_token_corpus_subset_artifact,
    formal_token_subset_roster_sha256,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    AuthorizedPreprocessingSelection,
    FROZEN_PREPROCESSING_ARM_SPEC_BY_ID,
    PreprocessingProducerAuthorizationReceipt,
    VerifiedPreprocessingSelectionCapability,
    load_preprocessing_selection_capability,
)
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
    LaBraMFeatureReceipt,
    OfficialLaBraMEncoder,
    bind_labram_record_positions,
    require_feature_receipt_position_binding,
)


FORMAL_TOKEN_CORPUS_SCHEMA = "soz_tusz_ictal_token_corpus_index_v4"
LEGACY_CANDIDATE_TOKEN_CORPUS_SCHEMA = "soz_tusz_ictal_token_corpus_index_v3"
FORMAL_TOKEN_CORPUS_SERIALIZATION = "canonical_json_and_safe_event_bundles"
TUSZ_ICTAL_PREPROCESSING_PRODUCER_KIND = "tusz_ictal"
REQUIRED_PREPROCESSING_ARM_ID = "C-CAR19"
INDEX_FILENAME = "index.json"
EVENTS_DIRECTORY = "events"
MAX_INDEX_BYTES = 64 * 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EVENT_ID_RE = re.compile(
    r"[a-z0-9_]+__global_ictal_[0-9]{4}"
)
_PATIENT_ID_RE = re.compile(r"[a-z0-9]{8}")

_INDEX_FIELDS = frozenset(
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
_PREPROCESSING_SELECTION_FIELDS = frozenset(
    {
        *(field.name for field in fields(PreprocessingProducerAuthorizationReceipt)),
        "authorization_receipt_sha256",
    }
)
_MASTER_MANIFEST_FIELDS = frozenset(
    {
        "bundle_manifest_sha256",
        "source_manifest_sha256",
        "cohort_receipt_sha256",
        "preflight_performed",
        "event_count",
        "patient_count",
    }
)
_TRAINING_MANIFEST_FIELDS = frozenset(
    {
        "bundle_manifest_sha256",
        "source_manifest_sha256",
        "cohort_receipt_sha256",
        "preflight_performed",
        "event_count",
        "patient_count",
        "role",
        "derived_from_master_source_manifest_sha256",
    }
)
_PREPROCESS_FIELDS = frozenset(
    {
        "config",
        "config_sha256",
        "edf_preprocess_schema",
        "labram_position_binding_policy",
        "causal_iir_phase_state_receipt_required",
        "preflight_performed",
        "signal_preflight_receipt_roster_sha256",
    }
)
_FOUNDATION_FIELDS = frozenset(
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
_EVENT_FIELDS = frozenset(
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


FormalTokenCorpusArtifact = VerifiedFormalTokenCorpusArtifact


class CandidateOnlyTokenCorpusError(ValueError):
    """A structurally old corpus that cannot be auto-promoted or trained."""


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Token corpus index contains non-canonical JSON data") from exc
    return encoded.encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_arg(value: str) -> str:
    try:
        return _require_sha256(value, field="SHA-256 argument")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return value


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{field} violates the closed schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _parse_canonical_json(raw: bytes, *, field: str) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be a JSON object")
    if _canonical_json_bytes(payload) != raw:
        raise ValueError(f"{field} must use canonical JSON encoding")
    return payload


def _reject_symlink_components(path: Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot contain symlink components")
    return absolute


def _read_stable_regular_file(
    path: Path,
    *,
    field: str,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    source = _reject_symlink_components(path, field=field)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{field} must be a regular file: {source}")
    before = source.stat()
    if before.st_size < 1 or (
        max_bytes is not None and before.st_size > max_bytes
    ):
        raise ValueError(f"{field} has an invalid size")
    payload = source.read_bytes()
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
    if before_fingerprint != after_fingerprint:
        raise RuntimeError(f"{field} changed while it was read")
    return payload, hashlib.sha256(payload).hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_bound_manifest(
    bundle_directory: str | Path,
    *,
    expected_bundle_manifest_sha256: str,
    expected_source_manifest_sha256: str,
    field: str,
) -> tuple[TUSZIctalTrainingManifest, dict[str, object]]:
    bundle = _reject_symlink_components(Path(bundle_directory), field=field)
    _, actual_bundle_sha = _read_stable_regular_file(
        bundle / "manifest.json", field=f"{field} manifest.json"
    )
    _, actual_receipt_sha = _read_stable_regular_file(
        bundle / "receipt.json", field=f"{field} receipt.json"
    )
    expected_bundle = _require_sha256(
        expected_bundle_manifest_sha256,
        field=f"expected_{field}_bundle_manifest_sha256",
    )
    expected_source = _require_sha256(
        expected_source_manifest_sha256,
        field=f"expected_{field}_source_manifest_sha256",
    )
    if actual_bundle_sha != expected_bundle:
        raise ValueError(f"{field} bundle manifest SHA-256 mismatch")
    if actual_receipt_sha != expected_source:
        raise ValueError(f"{field} source manifest SHA-256 mismatch")
    manifest = load_tusz_ictal_training_manifest(
        bundle,
        expected_bundle_manifest_sha256=actual_bundle_sha,
        expected_source_manifest_sha256=actual_receipt_sha,
    )
    if manifest.manifest_sha256 != expected_source:
        raise RuntimeError(f"{field} reconstructed source SHA drifted")
    return manifest, {
        "bundle_manifest_sha256": actual_bundle_sha,
        "source_manifest_sha256": manifest.manifest_sha256,
        "cohort_receipt_sha256": manifest.cohort_receipt.receipt_sha256,
        "preflight_performed": manifest.preflight_performed,
        "event_count": len(manifest),
        "patient_count": len(manifest.patient_ids),
    }


def _require_fully_preflighted(
    manifest: TUSZIctalTrainingManifest, *, field: str
) -> None:
    if not manifest.preflight_performed:
        raise ValueError(f"{field} must have preflight_performed=True")
    missing = tuple(
        event.event_id
        for event in manifest
        if event.signal_preflight_receipt_sha256 is None
    )
    if missing:
        raise ValueError(
            f"{field} lacks signal preflight receipts for {missing[:5]}"
        )


def _validate_master_training_relation(
    master: TUSZIctalTrainingManifest,
    training: TUSZIctalTrainingManifest,
) -> str:
    _require_fully_preflighted(master, field="master manifest")
    _require_fully_preflighted(training, field="training manifest")
    if training.preprocess_config != master.preprocess_config:
        raise ValueError("Training manifest preprocessing differs from its master")

    if training.manifest_sha256 == master.manifest_sha256:
        if training != master or training.derived_from_manifest_sha256 is not None:
            raise ValueError("Master-role training manifest is not the exact master")
        return "master"

    if training.derived_from_manifest_sha256 != master.manifest_sha256:
        raise ValueError("Training manifest is not derived from the supplied master")
    if training.discovered_source_count != master.discovered_source_count:
        raise ValueError("Derived training manifest discovery count differs from master")
    if training.duplicate_edf_aliases != master.duplicate_edf_aliases:
        raise ValueError("Derived training manifest duplicate aliases differ from master")

    master_events = {event.event_id: event for event in master}
    for event in training:
        if master_events.get(event.event_id) != event:
            raise ValueError(
                f"Derived training event changed from master: {event.event_id}"
            )
    if not set(training.patient_ids) <= set(master.patient_ids):
        raise ValueError("Derived training patient roster is not a master subset")
    return "derived_fold"


def _authorize_preprocessing_selection(
    capability: VerifiedPreprocessingSelectionCapability,
    *,
    preprocess_config: CausalEDFConfig,
) -> AuthorizedPreprocessingSelection:
    """Issue and replay the producer-scoped C-CAR19 authorization."""

    if not isinstance(capability, VerifiedPreprocessingSelectionCapability):
        raise TypeError(
            "preprocessing_selection must come from the strict five-arm loader"
        )
    capability.require_selected_arm(REQUIRED_PREPROCESSING_ARM_ID)
    authorization = capability.authorize_producer(
        arm_id=REQUIRED_PREPROCESSING_ARM_ID,
        expected_arm_result_receipt_sha256=(
            capability.selected_arm_result_receipt_sha256
        ),
        producer_kind=TUSZ_ICTAL_PREPROCESSING_PRODUCER_KIND,
        token_schema_version=FORMAL_TOKEN_CORPUS_SCHEMA,
    )
    authorization.assert_unchanged()
    selected_spec = FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[
        REQUIRED_PREPROCESSING_ARM_ID
    ]
    if (
        authorization.receipt.selected_arm_spec_receipt_sha256
        != selected_spec.receipt_sha256
    ):
        raise ValueError("Selected preprocessing arm spec receipt changed")
    if preprocess_config != CausalEDFConfig():
        raise ValueError(
            "Formal TUSZ token preprocessing must exactly match the frozen "
            "C-CAR19 CausalEDFConfig"
        )
    return authorization


def _preprocessing_selection_payload(
    authorization: AuthorizedPreprocessingSelection,
) -> dict[str, object]:
    authorization.assert_unchanged()
    return {
        **asdict(authorization.receipt),
        "authorization_receipt_sha256": authorization.receipt.receipt_sha256,
    }


def _validate_preprocessing_selection_payload(
    value: object,
    *,
    authorization: AuthorizedPreprocessingSelection,
) -> dict[str, object]:
    payload = _json_object(value, field="preprocessing_selection")
    _require_exact_fields(
        payload,
        _PREPROCESSING_SELECTION_FIELDS,
        field="preprocessing_selection",
    )
    expected = _preprocessing_selection_payload(authorization)
    if payload != expected:
        raise ValueError(
            "Token corpus preprocessing selection differs from the live "
            "producer authorization"
        )
    return dict(payload)


def _device(value: str | torch.device) -> torch.device:
    result = torch.device(value)
    if result.type not in {"cpu", "cuda"} or result.index is not None:
        raise ValueError("Formal token materialization supports only cpu or cuda")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return result


def _validate_frozen_foundation(
    encoder: torch.nn.Module,
    *,
    expected_feature_receipt_sha256: str,
    expected_modeling_sha256: str,
) -> tuple[LaBraMFeatureReceipt, str]:
    receipt = getattr(encoder, "receipt", None)
    if not isinstance(receipt, LaBraMFeatureReceipt):
        raise TypeError("Foundation encoder lacks a LaBraMFeatureReceipt")
    if receipt.checkpoint_sha256 != AUDITED_LABRAM_BASE_SHA256:
        raise ValueError("Foundation encoder is not the audited LaBraM-Base checkpoint")
    expected_modeling_sha = _require_sha256(
        expected_modeling_sha256,
        field="expected_labram_modeling_sha256",
    )
    if expected_modeling_sha != AUDITED_LABRAM_MODELING_SHA256:
        raise ValueError("Expected modeling source is not the audited LaBraM source")
    if receipt.modeling_sha256 != expected_modeling_sha:
        raise ValueError("Foundation modeling source SHA-256 mismatch")
    actual_receipt_sha = labram_feature_receipt_sha256(receipt)
    expected_receipt_sha = _require_sha256(
        expected_feature_receipt_sha256,
        field="expected_foundation_feature_receipt_sha256",
    )
    if actual_receipt_sha != expected_receipt_sha:
        raise ValueError("Foundation feature receipt SHA-256 mismatch")
    trainable = tuple(
        name for name, parameter in encoder.named_parameters() if parameter.requires_grad
    )
    if trainable:
        raise ValueError(f"Foundation encoder contains trainable parameters: {trainable}")
    encoder.eval()
    return receipt, actual_receipt_sha


def _event_source(
    edf_root: str | Path,
    event: TUSZIctalEventRecord,
):
    source = parse_tusz_official_train_path(edf_root, event.relative_edf_path)
    checks = {
        "patient_id": source.patient_id == event.patient_id,
        "session_id": source.session_id == event.session_id,
        "montage": source.montage == event.montage,
        "record_id": source.record_id == event.record_id,
        "relative_edf_path": source.relative_edf_path == event.relative_edf_path,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Frozen event {event.event_id} changed source identity fields {failed}"
        )
    return source


def _load_target_free_signal(
    event: TUSZIctalEventRecord,
    *,
    manifest: TUSZIctalTrainingManifest,
    edf_root: str | Path,
    reader_factory: Callable[[str], object] | None,
    foundation_receipt: LaBraMFeatureReceipt,
) -> torch.Tensor:
    source = _event_source(edf_root, event)
    loaded = load_standard19_edf_event(
        source.edf_path,
        event.event_t0_sec,
        config=manifest.preprocess_config,
        reader_factory=reader_factory,
    )
    if loaded.edf_receipt.edf_sha256 != event.edf_sha256:
        raise ValueError(f"Frozen EDF changed after manifest: {event.event_id}")
    replay_sha = tusz_signal_preflight_receipt_sha256(loaded)
    if replay_sha != event.signal_preflight_receipt_sha256:
        raise ValueError(
            f"Signal preflight receipt changed for event {event.event_id}"
        )
    binding = bind_labram_record_positions(
        loaded.edf_receipt.raw_channel_names,
        semantic_channels=loaded.edf_receipt.semantic_channels,
    )
    require_feature_receipt_position_binding(foundation_receipt, binding)
    eeg = loaded.window.data.detach().to(dtype=torch.float32, device="cpu")
    if tuple(eeg.shape) != (19, 12_000) or not torch.isfinite(eeg).all().item():
        raise ValueError("Preflighted TUSZ event must be finite [19,12000]")
    return eeg


def _validate_generated_token(
    artifact: LaBraMConceptTokenArtifact,
    event: TUSZIctalEventRecord,
    *,
    training_manifest_sha256: str,
    foundation_feature_receipt_sha256: str,
) -> dict[str, object]:
    token = load_labram_concept_tokens(
        artifact.path,
        expected_manifest_sha256=artifact.manifest_sha256,
    )
    checks = {
        "event_id": token.event_id == event.event_id,
        "source_manifest": (
            token.source_concept_manifest_sha256 == training_manifest_sha256
        ),
        "event_record": token.event_record_sha256 == event.event_record_sha256,
        "preprocess": (
            token.preprocess_receipt_sha256
            == event.signal_preflight_receipt_sha256
        ),
        "foundation_receipt": (
            token.foundation_feature_receipt_sha256
            == foundation_feature_receipt_sha256
        ),
        "foundation_checkpoint": (
            token.foundation_checkpoint_sha256 == AUDITED_LABRAM_BASE_SHA256
        ),
        "tensor_sha": token.tensor_sha256 == artifact.tensor_sha256,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Generated token {event.event_id} failed lineage fields {failed}"
        )
    return {
        "event_id": event.event_id,
        "patient_id": event.patient_id,
        "event_record_sha256": event.event_record_sha256,
        "preprocess_receipt_sha256": event.signal_preflight_receipt_sha256,
        "bundle_path": f"{EVENTS_DIRECTORY}/{event.event_id}",
        "bundle_manifest_sha256": artifact.manifest_sha256,
        "tensor_sha256": artifact.tensor_sha256,
    }


def _roster_payloads(
    events: Sequence[Mapping[str, object]],
) -> tuple[tuple[object, ...], tuple[str, ...], tuple[object, ...]]:
    event_roster = tuple(
        (
            event["event_id"],
            event["patient_id"],
            event["event_record_sha256"],
            event["preprocess_receipt_sha256"],
        )
        for event in events
    )
    patient_ids = tuple(sorted({str(event["patient_id"]) for event in events}))
    patient_events = tuple(
        (
            patient_id,
            tuple(
                sorted(
                    str(event["event_id"])
                    for event in events
                    if event["patient_id"] == patient_id
                )
            ),
        )
        for patient_id in patient_ids
    )
    return event_roster, patient_ids, patient_events


def _build_index_payload(
    *,
    master_binding: Mapping[str, object],
    training_binding: Mapping[str, object],
    training_role: str,
    training_manifest: TUSZIctalTrainingManifest,
    preprocessing_authorization: AuthorizedPreprocessingSelection,
    foundation_receipt: LaBraMFeatureReceipt,
    foundation_receipt_sha256: str,
    materialization_device: torch.device,
    events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    event_roster, patient_ids, patient_events = _roster_payloads(events)
    preprocess_config = asdict(training_manifest.preprocess_config)
    preflight_roster = tuple(
        (
            event.event_id,
            event.signal_preflight_receipt_sha256,
        )
        for event in training_manifest
    )
    master_payload = dict(master_binding)
    training_payload = {
        **training_binding,
        "role": training_role,
        "derived_from_master_source_manifest_sha256": (
            training_manifest.derived_from_manifest_sha256
        ),
    }
    return {
        "schema_version": FORMAL_TOKEN_CORPUS_SCHEMA,
        "purpose": CONCEPT_TOKEN_PURPOSE,
        "formal": True,
        "smoke_only": False,
        "serialization": FORMAL_TOKEN_CORPUS_SERIALIZATION,
        "master_manifest": master_payload,
        "training_manifest": training_payload,
        "preprocessing_selection": _preprocessing_selection_payload(
            preprocessing_authorization
        ),
        "preprocess": {
            "config": preprocess_config,
            "config_sha256": _canonical_sha256(preprocess_config),
            "edf_preprocess_schema": EDF_PREPROCESS_SCHEMA,
            "labram_position_binding_policy": (
                LABRAM_RAW_HEADER_POSITION_BINDING_POLICY
            ),
            "causal_iir_phase_state_receipt_required": True,
            "preflight_performed": True,
            "signal_preflight_receipt_roster_sha256": _canonical_sha256(
                preflight_roster
            ),
        },
        "foundation": {
            "feature_receipt_sha256": foundation_receipt_sha256,
            "checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "audited_expected_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "modeling_sha256": foundation_receipt.modeling_sha256,
            "audited_expected_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "token_shape": list(CONCEPT_TOKEN_SHAPE),
            "tile_seconds": 4,
            "frozen": True,
            "materialization_device": str(materialization_device),
        },
        "event_count": len(events),
        "patient_count": len(patient_ids),
        "event_roster_sha256": _canonical_sha256(event_roster),
        "patient_roster_sha256": _canonical_sha256(patient_ids),
        "patient_event_roster_sha256": _canonical_sha256(patient_events),
        "tensor_roster_sha256": _canonical_sha256(
            tuple((event["event_id"], event["tensor_sha256"]) for event in events)
        ),
        "events": list(events),
    }


def _json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _validate_index_payload(
    payload: Mapping[str, object],
    *,
    preprocessing_authorization: AuthorizedPreprocessingSelection,
) -> dict[str, object]:
    schema = payload.get("schema_version")
    if schema == LEGACY_CANDIDATE_TOKEN_CORPUS_SCHEMA:
        raise CandidateOnlyTokenCorpusError(
            "Formal-v3 token corpora are candidate-only because they lack the "
            "causal-IIR phase/state and raw-header LaBraM position receipts; "
            "automatic promotion is forbidden"
        )
    if schema != FORMAL_TOKEN_CORPUS_SCHEMA:
        raise ValueError("Unsupported formal token corpus schema")
    _require_exact_fields(payload, _INDEX_FIELDS, field="token corpus index")
    if payload["purpose"] != CONCEPT_TOKEN_PURPOSE:
        raise ValueError("Token corpus purpose boundary is invalid")
    if payload["formal"] is not True or payload["smoke_only"] is not False:
        raise ValueError("Formal token corpus must be formal=true and smoke_only=false")
    if payload["serialization"] != FORMAL_TOKEN_CORPUS_SERIALIZATION:
        raise ValueError("Token corpus serialization policy is invalid")

    master = _json_object(payload["master_manifest"], field="master_manifest")
    training = _json_object(
        payload["training_manifest"], field="training_manifest"
    )
    preprocessing_selection = _validate_preprocessing_selection_payload(
        payload["preprocessing_selection"],
        authorization=preprocessing_authorization,
    )
    preprocess = _json_object(payload["preprocess"], field="preprocess")
    foundation = _json_object(payload["foundation"], field="foundation")
    _require_exact_fields(master, _MASTER_MANIFEST_FIELDS, field="master_manifest")
    _require_exact_fields(
        training, _TRAINING_MANIFEST_FIELDS, field="training_manifest"
    )
    _require_exact_fields(preprocess, _PREPROCESS_FIELDS, field="preprocess")
    _require_exact_fields(foundation, _FOUNDATION_FIELDS, field="foundation")

    for block_name, block in (("master_manifest", master), ("training_manifest", training)):
        for sha_field in (
            "bundle_manifest_sha256",
            "source_manifest_sha256",
            "cohort_receipt_sha256",
        ):
            _require_sha256(block[sha_field], field=f"{block_name}.{sha_field}")
        if block["preflight_performed"] is not True:
            raise ValueError(f"{block_name} is not signal-preflighted")
        _positive_int(block["event_count"], field=f"{block_name}.event_count")
        _positive_int(block["patient_count"], field=f"{block_name}.patient_count")

    role = training["role"]
    derived = training["derived_from_master_source_manifest_sha256"]
    if role == "master":
        if derived is not None:
            raise ValueError("Master-role training manifest cannot be derived")
        for field_name in (
            "bundle_manifest_sha256",
            "source_manifest_sha256",
            "cohort_receipt_sha256",
            "event_count",
            "patient_count",
        ):
            if training[field_name] != master[field_name]:
                raise ValueError("Master-role training binding differs from master")
    elif role == "derived_fold":
        if _require_sha256(
            derived,
            field="derived_from_master_source_manifest_sha256",
        ) != master["source_manifest_sha256"]:
            raise ValueError("Derived training manifest points to another master")
        if training["source_manifest_sha256"] == master["source_manifest_sha256"]:
            raise ValueError("Derived-fold source SHA cannot equal master source SHA")
    else:
        raise ValueError("training_manifest.role must be master or derived_fold")

    _require_exact_fields(
        _json_object(preprocess["config"], field="preprocess.config"),
        frozenset(field.name for field in fields(CausalEDFConfig)),
        field="preprocess.config",
    )
    config = CausalEDFConfig(**preprocess["config"])
    preprocessing_authorization.assert_unchanged()
    if config != CausalEDFConfig():
        raise ValueError(
            "Formal TUSZ token preprocessing must exactly match the frozen "
            "C-CAR19 CausalEDFConfig"
        )
    config_payload = asdict(config)
    config_sha = _require_sha256(
        preprocess["config_sha256"], field="preprocess.config_sha256"
    )
    if config_sha != _canonical_sha256(config_payload):
        raise ValueError("Preprocessing config SHA-256 mismatch")
    if preprocess["edf_preprocess_schema"] != EDF_PREPROCESS_SCHEMA:
        raise ValueError("Token corpus uses a legacy EDF preprocessing receipt")
    if (
        preprocess["labram_position_binding_policy"]
        != LABRAM_RAW_HEADER_POSITION_BINDING_POLICY
    ):
        raise ValueError("Token corpus LaBraM position-binding policy is invalid")
    if preprocess["causal_iir_phase_state_receipt_required"] is not True:
        raise ValueError("Token corpus does not require causal-IIR phase/state receipts")
    if preprocess["preflight_performed"] is not True:
        raise ValueError("Token corpus preprocessing must be preflighted")
    _require_sha256(
        preprocess["signal_preflight_receipt_roster_sha256"],
        field="signal_preflight_receipt_roster_sha256",
    )

    for field_name in (
        "feature_receipt_sha256",
        "checkpoint_sha256",
        "audited_expected_checkpoint_sha256",
        "modeling_sha256",
        "audited_expected_modeling_sha256",
    ):
        _require_sha256(foundation[field_name], field=f"foundation.{field_name}")
    if (
        foundation["checkpoint_sha256"] != AUDITED_LABRAM_BASE_SHA256
        or foundation["audited_expected_checkpoint_sha256"]
        != AUDITED_LABRAM_BASE_SHA256
    ):
        raise ValueError("Token corpus does not use the audited LaBraM-Base checkpoint")
    if (
        foundation["modeling_sha256"] != AUDITED_LABRAM_MODELING_SHA256
        or foundation["audited_expected_modeling_sha256"]
        != AUDITED_LABRAM_MODELING_SHA256
    ):
        raise ValueError("Token corpus does not use the audited LaBraM modeling source")
    if foundation["token_shape"] != list(CONCEPT_TOKEN_SHAPE):
        raise ValueError("Foundation token shape is not the fixed cache shape")
    if foundation["tile_seconds"] != 4 or foundation["frozen"] is not True:
        raise ValueError("Foundation must be frozen with four-second calls")
    device_text = foundation["materialization_device"]
    if device_text not in {"cpu", "cuda"}:
        raise ValueError("materialization_device must be cpu or cuda")

    event_count = _positive_int(payload["event_count"], field="event_count")
    patient_count = _positive_int(payload["patient_count"], field="patient_count")
    if event_count != training["event_count"] or patient_count != training["patient_count"]:
        raise ValueError("Corpus counts disagree with the training manifest binding")
    for field_name in (
        "event_roster_sha256",
        "patient_roster_sha256",
        "patient_event_roster_sha256",
    ):
        _require_sha256(payload[field_name], field=field_name)

    raw_events = payload["events"]
    if not isinstance(raw_events, list) or len(raw_events) != event_count:
        raise ValueError("events must contain the declared complete event roster")
    normalized_events: list[dict[str, object]] = []
    for index, value in enumerate(raw_events):
        event = _json_object(value, field=f"events[{index}]")
        _require_exact_fields(event, _EVENT_FIELDS, field=f"events[{index}]")
        event_id = event["event_id"]
        patient_id = event["patient_id"]
        if not isinstance(event_id, str) or not _EVENT_ID_RE.fullmatch(event_id):
            raise ValueError(f"events[{index}].event_id is not canonical")
        if not isinstance(patient_id, str) or not _PATIENT_ID_RE.fullmatch(patient_id):
            raise ValueError(f"events[{index}].patient_id is not canonical")
        if not event_id.startswith(f"{patient_id}_"):
            raise ValueError(f"events[{index}] patient and event identities disagree")
        expected_path = f"{EVENTS_DIRECTORY}/{event_id}"
        if event["bundle_path"] != expected_path:
            raise ValueError(f"events[{index}].bundle_path is not canonical")
        for field_name in (
            "event_record_sha256",
            "preprocess_receipt_sha256",
            "bundle_manifest_sha256",
            "tensor_sha256",
        ):
            _require_sha256(event[field_name], field=f"events[{index}].{field_name}")
        normalized_events.append(dict(event))

    event_ids = tuple(str(event["event_id"]) for event in normalized_events)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("Token corpus contains duplicate event IDs")
    canonical_event_order = tuple(
        sorted(
            normalized_events,
            key=lambda event: (str(event["patient_id"]), str(event["event_id"])),
        )
    )
    if tuple(normalized_events) != canonical_event_order:
        raise ValueError("Token corpus events are not in canonical order")
    bundle_paths = tuple(str(event["bundle_path"]) for event in normalized_events)
    if len(set(bundle_paths)) != len(bundle_paths):
        raise ValueError("Token corpus contains duplicate bundle paths")
    event_roster, patient_ids, patient_events = _roster_payloads(normalized_events)
    if len(patient_ids) != patient_count:
        raise ValueError("Token corpus patient count disagrees with event roster")
    expected_hashes = {
        "event_roster_sha256": _canonical_sha256(event_roster),
        "patient_roster_sha256": _canonical_sha256(patient_ids),
        "patient_event_roster_sha256": _canonical_sha256(patient_events),
        "tensor_roster_sha256": _canonical_sha256(
            tuple(
                (event["event_id"], event["tensor_sha256"])
                for event in normalized_events
            )
        ),
        "signal_preflight_receipt_roster_sha256": _canonical_sha256(
            tuple(
                (
                    event["event_id"],
                    event["preprocess_receipt_sha256"],
                )
                for event in normalized_events
            )
        ),
    }
    for field_name, expected_hash in expected_hashes.items():
        actual_hash = (
            preprocess[field_name]
            if field_name == "signal_preflight_receipt_roster_sha256"
            else payload[field_name]
        )
        if actual_hash != expected_hash:
            raise ValueError(f"{field_name} does not match the exact roster")

    normalized = dict(payload)
    normalized["master_manifest"] = dict(master)
    normalized["training_manifest"] = dict(training)
    normalized["preprocessing_selection"] = preprocessing_selection
    normalized["preprocess"] = {**preprocess, "config": config_payload}
    normalized["foundation"] = dict(foundation)
    normalized["events"] = normalized_events
    return normalized


def load_formal_token_corpus(
    corpus_directory: str | Path,
    *,
    expected_index_sha256: str,
    preprocessing_selection: VerifiedPreprocessingSelectionCapability,
) -> VerifiedFormalTokenCorpusArtifact:
    """Strictly reload a complete formal corpus and every pinned event bundle."""

    preprocessing_authorization = _authorize_preprocessing_selection(
        preprocessing_selection,
        preprocess_config=CausalEDFConfig(),
    )

    source = _reject_symlink_components(
        Path(corpus_directory), field="formal token corpus"
    )
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Formal token corpus must be a regular directory: {source}")
    actual_entries = {entry.name for entry in source.iterdir()}
    expected_entries = {INDEX_FILENAME, EVENTS_DIRECTORY}
    if actual_entries != expected_entries:
        raise ValueError(
            "Formal token corpus contains missing or unknown entries; "
            f"expected={sorted(expected_entries)}, actual={sorted(actual_entries)}"
        )
    events_directory = source / EVENTS_DIRECTORY
    if events_directory.is_symlink() or not events_directory.is_dir():
        raise ValueError("Formal token corpus events entry must be a regular directory")
    raw, actual_index_sha = _read_stable_regular_file(
        source / INDEX_FILENAME,
        field="formal token corpus index",
        max_bytes=MAX_INDEX_BYTES,
    )
    expected_sha = _require_sha256(
        expected_index_sha256, field="expected_index_sha256"
    )
    if actual_index_sha != expected_sha:
        raise ValueError("Formal token corpus index SHA-256 mismatch")
    index = _validate_index_payload(
        _parse_canonical_json(raw, field="formal token corpus index"),
        preprocessing_authorization=preprocessing_authorization,
    )

    expected_event_directories = {
        str(event["event_id"]) for event in index["events"]
    }
    actual_event_directories = {entry.name for entry in events_directory.iterdir()}
    if actual_event_directories != expected_event_directories:
        raise ValueError(
            "Formal token corpus event directories do not match the index; "
            f"missing={sorted(expected_event_directories-actual_event_directories)[:5]}, "
            f"extra={sorted(actual_event_directories-expected_event_directories)[:5]}"
        )

    training_sha = str(index["training_manifest"]["source_manifest_sha256"])
    foundation_sha = str(index["foundation"]["feature_receipt_sha256"])
    checkpoint_sha = str(index["foundation"]["checkpoint_sha256"])
    modeling_sha = str(index["foundation"]["modeling_sha256"])
    for event in index["events"]:
        bundle = source / str(event["bundle_path"])
        if bundle.is_symlink() or not bundle.is_dir():
            raise ValueError(f"Token bundle is not a regular directory: {bundle}")
        token = load_labram_concept_tokens(
            bundle,
            expected_manifest_sha256=str(event["bundle_manifest_sha256"]),
        )
        checks = {
            "event_id": token.event_id == event["event_id"],
            "source_manifest": token.source_concept_manifest_sha256 == training_sha,
            "event_record": token.event_record_sha256 == event["event_record_sha256"],
            "preprocess": (
                token.preprocess_receipt_sha256
                == event["preprocess_receipt_sha256"]
            ),
            "foundation_receipt": (
                token.foundation_feature_receipt_sha256 == foundation_sha
            ),
            "foundation_checkpoint": token.foundation_checkpoint_sha256 == checkpoint_sha,
            "foundation_modeling": (
                token.foundation_feature_receipt.modeling_sha256 == modeling_sha
            ),
            "tensor": token.tensor_sha256 == event["tensor_sha256"],
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Token bundle {event['event_id']} disagrees with index fields {failed}"
            )

    return _issue_verified_formal_token_corpus_artifact(
        path=source,
        index_sha256=actual_index_sha,
        master_bundle_manifest_sha256=str(
            index["master_manifest"]["bundle_manifest_sha256"]
        ),
        master_source_manifest_sha256=str(
            index["master_manifest"]["source_manifest_sha256"]
        ),
        training_bundle_manifest_sha256=str(
            index["training_manifest"]["bundle_manifest_sha256"]
        ),
        training_source_manifest_sha256=training_sha,
        preprocessing_selection_artifact_sha256=str(
            index["preprocessing_selection"]["selection_artifact_sha256"]
        ),
        preprocessing_selection_bundle_receipt_sha256=str(
            index["preprocessing_selection"][
                "selection_bundle_receipt_sha256"
            ]
        ),
        preprocessing_protocol_receipt_sha256=str(
            index["preprocessing_selection"]["protocol_receipt_sha256"]
        ),
        preprocessing_selected_arm_result_receipt_sha256=str(
            index["preprocessing_selection"][
                "selected_arm_result_receipt_sha256"
            ]
        ),
        preprocessing_selected_arm_id=str(
            index["preprocessing_selection"]["selected_arm_id"]
        ),
        event_roster_sha256=str(index["event_roster_sha256"]),
        patient_roster_sha256=str(index["patient_roster_sha256"]),
        patient_event_roster_sha256=str(index["patient_event_roster_sha256"]),
        tensor_roster_sha256=str(index["tensor_roster_sha256"]),
        event_count=int(index["event_count"]),
        patient_count=int(index["patient_count"]),
        events=tuple(
            FormalTokenEventBinding(
                event_id=str(event["event_id"]),
                bundle_path=source / str(event["bundle_path"]),
                bundle_manifest_sha256=str(event["bundle_manifest_sha256"]),
                tensor_sha256=str(event["tensor_sha256"]),
            )
            for event in index["events"]
        ),
    )


def load_formal_token_corpus_patient_subset(
    corpus_directory: str | Path,
    *,
    expected_index_sha256: str,
    preprocessing_selection: VerifiedPreprocessingSelectionCapability,
    patient_ids: Sequence[str],
    expected_selected_event_count: int,
    expected_master_bundle_manifest_sha256: str,
    expected_master_source_manifest_sha256: str,
) -> VerifiedFormalTokenCorpusSubsetArtifact:
    """Strictly validate a master index while opening only selected bundles.

    This loader exists for a sealed-gate inference boundary.  It deliberately
    does *not* call :func:`load_formal_token_corpus`, because that loader opens
    every event tensor.  The complete canonical index and directory roster are
    still validated, but only bundles owned by ``patient_ids`` are passed to
    ``load_labram_concept_tokens``.
    """

    if isinstance(patient_ids, (str, bytes)):
        raise TypeError("patient_ids must be a sequence")
    selected_patients = tuple(str(value).strip() for value in patient_ids)
    if (
        not selected_patients
        or selected_patients != tuple(sorted(selected_patients))
        or len(set(selected_patients)) != len(selected_patients)
        or any(not _PATIENT_ID_RE.fullmatch(value) for value in selected_patients)
    ):
        raise ValueError("patient_ids must be canonical, sorted, and unique")
    if (
        isinstance(expected_selected_event_count, bool)
        or not isinstance(expected_selected_event_count, int)
        or expected_selected_event_count < 1
    ):
        raise ValueError("expected_selected_event_count must be positive")

    preprocessing_authorization = _authorize_preprocessing_selection(
        preprocessing_selection,
        preprocess_config=CausalEDFConfig(),
    )
    source = _reject_symlink_components(
        Path(corpus_directory), field="formal token corpus subset"
    )
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Formal token corpus must be a regular directory: {source}")
    actual_entries = {entry.name for entry in source.iterdir()}
    if actual_entries != {INDEX_FILENAME, EVENTS_DIRECTORY}:
        raise ValueError("Formal token corpus contains missing or unknown entries")
    events_directory = source / EVENTS_DIRECTORY
    if events_directory.is_symlink() or not events_directory.is_dir():
        raise ValueError("Formal token corpus events entry must be a regular directory")

    raw, actual_index_sha = _read_stable_regular_file(
        source / INDEX_FILENAME,
        field="formal token corpus index",
        max_bytes=MAX_INDEX_BYTES,
    )
    if actual_index_sha != _require_sha256(
        expected_index_sha256, field="expected_index_sha256"
    ):
        raise ValueError("Formal token corpus index SHA-256 mismatch")
    index = _validate_index_payload(
        _parse_canonical_json(raw, field="formal token corpus index"),
        preprocessing_authorization=preprocessing_authorization,
    )

    master = index["master_manifest"]
    training = index["training_manifest"]
    expected_master_bundle = _require_sha256(
        expected_master_bundle_manifest_sha256,
        field="expected_master_bundle_manifest_sha256",
    )
    expected_master_source = _require_sha256(
        expected_master_source_manifest_sha256,
        field="expected_master_source_manifest_sha256",
    )
    if (
        training["role"] != "master"
        or training["bundle_manifest_sha256"] != master["bundle_manifest_sha256"]
        or training["source_manifest_sha256"] != master["source_manifest_sha256"]
    ):
        raise ValueError("Selective gate inference requires the exact master corpus")
    if (
        master["bundle_manifest_sha256"] != expected_master_bundle
        or master["source_manifest_sha256"] != expected_master_source
    ):
        raise ValueError("Master manifest identity changed")

    expected_event_directories = {
        str(event["event_id"]) for event in index["events"]
    }
    actual_event_directories = {entry.name for entry in events_directory.iterdir()}
    if actual_event_directories != expected_event_directories:
        raise ValueError("Formal token corpus event-directory roster changed")

    selected_set = set(selected_patients)
    selected_rows = tuple(
        event for event in index["events"] if event["patient_id"] in selected_set
    )
    if len(selected_rows) != expected_selected_event_count:
        raise ValueError(
            "Selected event count changed: "
            f"expected={expected_selected_event_count}, actual={len(selected_rows)}"
        )
    if {str(event["patient_id"]) for event in selected_rows} != selected_set:
        raise ValueError("One or more selected patients have no token event")

    training_sha = str(training["source_manifest_sha256"])
    foundation_sha = str(index["foundation"]["feature_receipt_sha256"])
    checkpoint_sha = str(index["foundation"]["checkpoint_sha256"])
    modeling_sha = str(index["foundation"]["modeling_sha256"])
    bindings: list[FormalTokenSubsetEventBinding] = []
    for event in selected_rows:
        bundle = source / str(event["bundle_path"])
        if bundle.is_symlink() or not bundle.is_dir():
            raise ValueError(f"Selected token bundle is not a regular directory: {bundle}")
        token = load_labram_concept_tokens(
            bundle,
            expected_manifest_sha256=str(event["bundle_manifest_sha256"]),
        )
        checks = {
            "event_id": token.event_id == event["event_id"],
            "source_manifest": token.source_concept_manifest_sha256 == training_sha,
            "event_record": token.event_record_sha256 == event["event_record_sha256"],
            "preprocess": (
                token.preprocess_receipt_sha256
                == event["preprocess_receipt_sha256"]
            ),
            "foundation_receipt": (
                token.foundation_feature_receipt_sha256 == foundation_sha
            ),
            "foundation_checkpoint": token.foundation_checkpoint_sha256
            == checkpoint_sha,
            "foundation_modeling": (
                token.foundation_feature_receipt.modeling_sha256 == modeling_sha
            ),
            "tensor": token.tensor_sha256 == event["tensor_sha256"],
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Selected token bundle {event['event_id']} disagrees with "
                f"index fields {failed}"
            )
        bindings.append(
            FormalTokenSubsetEventBinding(
                event_id=str(event["event_id"]),
                patient_id=str(event["patient_id"]),
                event_record_sha256=str(event["event_record_sha256"]),
                preprocess_receipt_sha256=str(event["preprocess_receipt_sha256"]),
                bundle_path=bundle,
                bundle_manifest_sha256=str(event["bundle_manifest_sha256"]),
                tensor_sha256=str(event["tensor_sha256"]),
            )
        )
    preprocessing_authorization.assert_unchanged()
    binding_tuple = tuple(bindings)
    return _issue_verified_formal_token_corpus_subset_artifact(
        path=source,
        index_sha256=actual_index_sha,
        master_bundle_manifest_sha256=str(master["bundle_manifest_sha256"]),
        master_source_manifest_sha256=str(master["source_manifest_sha256"]),
        training_bundle_manifest_sha256=str(training["bundle_manifest_sha256"]),
        training_source_manifest_sha256=training_sha,
        preprocessing_selection_artifact_sha256=str(
            index["preprocessing_selection"]["selection_artifact_sha256"]
        ),
        preprocessing_selection_bundle_receipt_sha256=str(
            index["preprocessing_selection"]["selection_bundle_receipt_sha256"]
        ),
        preprocessing_protocol_receipt_sha256=str(
            index["preprocessing_selection"]["protocol_receipt_sha256"]
        ),
        preprocessing_selected_arm_result_receipt_sha256=str(
            index["preprocessing_selection"]["selected_arm_result_receipt_sha256"]
        ),
        preprocessing_selected_arm_id=str(
            index["preprocessing_selection"]["selected_arm_id"]
        ),
        foundation_feature_receipt_sha256=foundation_sha,
        foundation_checkpoint_sha256=checkpoint_sha,
        foundation_modeling_sha256=modeling_sha,
        full_event_count=int(index["event_count"]),
        full_patient_count=int(index["patient_count"]),
        selected_patient_ids=selected_patients,
        selected_patient_roster_sha256=_canonical_sha256(selected_patients),
        selected_event_roster_sha256=formal_token_subset_roster_sha256(
            binding_tuple
        ),
        selected_event_count=len(binding_tuple),
        events=binding_tuple,
        unselected_event_bundles_opened=False,
    )


def load_formal_token_corpus_fit_subset(
    corpus_directory: str | Path,
    *,
    expected_index_sha256: str,
    preprocessing_selection: VerifiedPreprocessingSelectionCapability,
    patient_ids: Sequence[str],
    forbidden_patient_ids: Sequence[str],
    expected_selected_event_count: int,
    expected_training_bundle_manifest_sha256: str,
    expected_training_source_manifest_sha256: str,
) -> VerifiedFormalTokenCorpusSubsetArtifact:
    """Strict-load only an exact fit allow-list from one training corpus.

    Unlike :func:`load_formal_token_corpus`, this function never opens an
    unselected event bundle.  It validates the pinned canonical index, proves
    that its complete patient roster is exactly ``patient_ids`` plus the
    forbidden I-gate roster, and only then strict-loads the selected bundle
    manifests/tensors.  Directory enumeration is deliberately avoided so the
    loader does not traverse forbidden bundle directories merely to compare a
    physical directory roster.
    """

    def _patients(values: Sequence[str], *, field: str) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{field} must be a sequence")
        result = tuple(str(value).strip() for value in values)
        if (
            not result
            or result != tuple(sorted(result))
            or len(set(result)) != len(result)
            or any(not _PATIENT_ID_RE.fullmatch(value) for value in result)
        ):
            raise ValueError(f"{field} must be canonical, sorted, and unique")
        return result

    selected_patients = _patients(patient_ids, field="patient_ids")
    forbidden_patients = _patients(
        forbidden_patient_ids, field="forbidden_patient_ids"
    )
    if set(selected_patients) & set(forbidden_patients):
        raise ValueError("Selected token patients overlap the forbidden I-gate")
    if (
        isinstance(expected_selected_event_count, bool)
        or not isinstance(expected_selected_event_count, int)
        or expected_selected_event_count < 1
    ):
        raise ValueError("expected_selected_event_count must be positive")

    preprocessing_authorization = _authorize_preprocessing_selection(
        preprocessing_selection,
        preprocess_config=CausalEDFConfig(),
    )
    source = _reject_symlink_components(
        Path(corpus_directory), field="formal fit-only token corpus subset"
    )
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Formal token corpus must be a regular directory: {source}")
    # Only root metadata is enumerated.  The forbidden event directories are
    # not traversed or opened.
    if {entry.name for entry in source.iterdir()} != {INDEX_FILENAME, EVENTS_DIRECTORY}:
        raise ValueError("Formal token corpus contains missing or unknown root entries")
    events_directory = source / EVENTS_DIRECTORY
    if events_directory.is_symlink() or not events_directory.is_dir():
        raise ValueError("Formal token corpus events entry must be a regular directory")

    raw, actual_index_sha = _read_stable_regular_file(
        source / INDEX_FILENAME,
        field="formal fit-only token corpus index",
        max_bytes=MAX_INDEX_BYTES,
    )
    if actual_index_sha != _require_sha256(
        expected_index_sha256, field="expected_index_sha256"
    ):
        raise ValueError("Formal token corpus index SHA-256 mismatch")
    index = _validate_index_payload(
        _parse_canonical_json(raw, field="formal fit-only token corpus index"),
        preprocessing_authorization=preprocessing_authorization,
    )
    training = index["training_manifest"]
    expected_training_bundle = _require_sha256(
        expected_training_bundle_manifest_sha256,
        field="expected_training_bundle_manifest_sha256",
    )
    expected_training_source = _require_sha256(
        expected_training_source_manifest_sha256,
        field="expected_training_source_manifest_sha256",
    )
    if (
        training["bundle_manifest_sha256"] != expected_training_bundle
        or training["source_manifest_sha256"] != expected_training_source
    ):
        raise ValueError("Fit-only token corpus training-manifest identity changed")

    index_patients = tuple(
        sorted({str(event["patient_id"]) for event in index["events"]})
    )
    if set(index_patients) != set(selected_patients) | set(forbidden_patients):
        raise ValueError("Token index is not exactly fit plus forbidden I-gate")
    selected_set = set(selected_patients)
    forbidden_set = set(forbidden_patients)
    selected_rows = tuple(
        event for event in index["events"] if event["patient_id"] in selected_set
    )
    if len(selected_rows) != expected_selected_event_count:
        raise ValueError(
            "Selected event count changed: "
            f"expected={expected_selected_event_count}, actual={len(selected_rows)}"
        )
    if {str(event["patient_id"]) for event in selected_rows} != selected_set:
        raise ValueError("One or more selected patients have no token event")
    if any(str(event["patient_id"]) in forbidden_set for event in selected_rows):
        raise RuntimeError("Forbidden I-gate row entered the selected token allow-list")

    training_sha = str(training["source_manifest_sha256"])
    foundation_sha = str(index["foundation"]["feature_receipt_sha256"])
    checkpoint_sha = str(index["foundation"]["checkpoint_sha256"])
    modeling_sha = str(index["foundation"]["modeling_sha256"])
    bindings: list[FormalTokenSubsetEventBinding] = []
    for event in selected_rows:
        # This is the first operation that enters an event bundle directory;
        # the row has already passed the explicit patient allow-list.
        bundle = source / str(event["bundle_path"])
        if bundle.is_symlink() or not bundle.is_dir():
            raise ValueError(f"Selected token bundle is not regular: {bundle}")
        token = load_labram_concept_tokens(
            bundle,
            expected_manifest_sha256=str(event["bundle_manifest_sha256"]),
        )
        checks = {
            "event_id": token.event_id == event["event_id"],
            "source_manifest": token.source_concept_manifest_sha256 == training_sha,
            "event_record": token.event_record_sha256 == event["event_record_sha256"],
            "preprocess": (
                token.preprocess_receipt_sha256
                == event["preprocess_receipt_sha256"]
            ),
            "foundation_receipt": (
                token.foundation_feature_receipt_sha256 == foundation_sha
            ),
            "foundation_checkpoint": token.foundation_checkpoint_sha256
            == checkpoint_sha,
            "foundation_modeling": (
                token.foundation_feature_receipt.modeling_sha256 == modeling_sha
            ),
            "tensor": token.tensor_sha256 == event["tensor_sha256"],
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Selected token bundle {event['event_id']} disagrees with "
                f"index fields {failed}"
            )
        bindings.append(
            FormalTokenSubsetEventBinding(
                event_id=str(event["event_id"]),
                patient_id=str(event["patient_id"]),
                event_record_sha256=str(event["event_record_sha256"]),
                preprocess_receipt_sha256=str(event["preprocess_receipt_sha256"]),
                bundle_path=bundle,
                bundle_manifest_sha256=str(event["bundle_manifest_sha256"]),
                tensor_sha256=str(event["tensor_sha256"]),
            )
        )
    preprocessing_authorization.assert_unchanged()
    binding_tuple = tuple(bindings)
    master = index["master_manifest"]
    return _issue_verified_formal_token_corpus_subset_artifact(
        path=source,
        index_sha256=actual_index_sha,
        master_bundle_manifest_sha256=str(master["bundle_manifest_sha256"]),
        master_source_manifest_sha256=str(master["source_manifest_sha256"]),
        training_bundle_manifest_sha256=str(training["bundle_manifest_sha256"]),
        training_source_manifest_sha256=training_sha,
        preprocessing_selection_artifact_sha256=str(
            index["preprocessing_selection"]["selection_artifact_sha256"]
        ),
        preprocessing_selection_bundle_receipt_sha256=str(
            index["preprocessing_selection"]["selection_bundle_receipt_sha256"]
        ),
        preprocessing_protocol_receipt_sha256=str(
            index["preprocessing_selection"]["protocol_receipt_sha256"]
        ),
        preprocessing_selected_arm_result_receipt_sha256=str(
            index["preprocessing_selection"]["selected_arm_result_receipt_sha256"]
        ),
        preprocessing_selected_arm_id=str(
            index["preprocessing_selection"]["selected_arm_id"]
        ),
        foundation_feature_receipt_sha256=foundation_sha,
        foundation_checkpoint_sha256=checkpoint_sha,
        foundation_modeling_sha256=modeling_sha,
        full_event_count=int(index["event_count"]),
        full_patient_count=int(index["patient_count"]),
        selected_patient_ids=selected_patients,
        selected_patient_roster_sha256=_canonical_sha256(selected_patients),
        selected_event_roster_sha256=formal_token_subset_roster_sha256(
            binding_tuple
        ),
        selected_event_count=len(binding_tuple),
        events=binding_tuple,
        unselected_event_bundles_opened=False,
    )


def materialize_formal_tusz_ictal_token_corpus(
    *,
    master_manifest_bundle: str | Path,
    expected_master_bundle_manifest_sha256: str,
    expected_master_source_manifest_sha256: str,
    training_manifest_bundle: str | Path,
    expected_training_bundle_manifest_sha256: str,
    expected_training_source_manifest_sha256: str,
    preprocessing_selection: VerifiedPreprocessingSelectionCapability,
    edf_root: str | Path,
    labram_modeling_path: str | Path,
    labram_checkpoint_path: str | Path,
    expected_labram_modeling_sha256: str,
    expected_foundation_feature_receipt_sha256: str,
    output_directory: str | Path,
    device: str | torch.device = "cuda",
    reader_factory: Callable[[str], object] | None = None,
) -> VerifiedFormalTokenCorpusArtifact:
    """Materialize and atomically publish one complete formal token corpus."""

    target = _reject_symlink_components(
        Path(output_directory), field="formal token corpus output"
    )
    if target.name in {"", ".", ".."}:
        raise ValueError("Formal token corpus output requires a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(f"Formal token corpus output already exists: {target}")

    master, master_binding = _load_bound_manifest(
        master_manifest_bundle,
        expected_bundle_manifest_sha256=expected_master_bundle_manifest_sha256,
        expected_source_manifest_sha256=expected_master_source_manifest_sha256,
        field="master_manifest",
    )
    training, training_binding = _load_bound_manifest(
        training_manifest_bundle,
        expected_bundle_manifest_sha256=expected_training_bundle_manifest_sha256,
        expected_source_manifest_sha256=expected_training_source_manifest_sha256,
        field="training_manifest",
    )
    training_role = _validate_master_training_relation(master, training)
    preprocessing_authorization = _authorize_preprocessing_selection(
        preprocessing_selection,
        preprocess_config=training.preprocess_config,
    )
    execution_device = _device(device)

    # Formal publication intentionally has no encoder-factory injection point.
    # Synthetic tests may monkeypatch this module global, but callers cannot
    # publish ``formal=true`` through a custom factory argument.
    encoder = OfficialLaBraMEncoder(
        modeling_path=labram_modeling_path,
        checkpoint_path=labram_checkpoint_path,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=expected_labram_modeling_sha256,
        tile_seconds=4,
    )
    foundation_receipt, foundation_receipt_sha = _validate_frozen_foundation(
        encoder,
        expected_feature_receipt_sha256=(
            expected_foundation_feature_receipt_sha256
        ),
        expected_modeling_sha256=expected_labram_modeling_sha256,
    )
    encoder.to(execution_device)
    encoder.eval()
    tiled_encoder = TiledFoundationEncoder(encoder, n_calls=15).to(execution_device)
    tiled_encoder.eval()
    if tuple(
        name
        for name, parameter in tiled_encoder.named_parameters()
        if parameter.requires_grad
    ):
        raise ValueError("Tiled foundation encoder unexpectedly contains trainable state")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        event_root = temporary / EVENTS_DIRECTORY
        event_root.mkdir()
        event_rows: list[dict[str, object]] = []
        for event in training:
            eeg = _load_target_free_signal(
                event,
                manifest=training,
                edf_root=edf_root,
                reader_factory=reader_factory,
                foundation_receipt=foundation_receipt,
            )
            with torch.inference_mode():
                tokens = tiled_encoder(
                    eeg.unsqueeze(0).to(
                        device=execution_device, dtype=torch.float32
                    )
                )[0].detach().to(dtype=torch.float32, device="cpu")
            if tuple(tokens.shape) != CONCEPT_TOKEN_SHAPE:
                raise ValueError(
                    f"Foundation returned invalid token shape for {event.event_id}"
                )
            if any(parameter.requires_grad for parameter in encoder.parameters()):
                raise RuntimeError("Foundation encoder was unfrozen during inference")
            bundle = event_root / event.event_id
            artifact = save_labram_concept_tokens(
                bundle,
                tokens,
                event_id=event.event_id,
                source_concept_manifest_sha256=training.manifest_sha256,
                event_record_sha256=event.event_record_sha256,
                preprocess_receipt_sha256=(
                    event.signal_preflight_receipt_sha256
                ),
                foundation_feature_receipt=foundation_receipt,
            )
            event_rows.append(
                _validate_generated_token(
                    artifact,
                    event,
                    training_manifest_sha256=training.manifest_sha256,
                    foundation_feature_receipt_sha256=foundation_receipt_sha,
                )
            )
            del tokens, eeg

        if len(event_rows) != len(training):
            raise RuntimeError("Token materialization omitted a frozen event")
        if tuple(row["event_id"] for row in event_rows) != tuple(
            event.event_id for event in training
        ):
            raise RuntimeError("Token materialization reordered the frozen event roster")

        index = _validate_index_payload(
            _build_index_payload(
                master_binding=master_binding,
                training_binding=training_binding,
                training_role=training_role,
                training_manifest=training,
                preprocessing_authorization=preprocessing_authorization,
                foundation_receipt=foundation_receipt,
                foundation_receipt_sha256=foundation_receipt_sha,
                materialization_device=execution_device,
                events=event_rows,
            ),
            preprocessing_authorization=preprocessing_authorization,
        )
        index_bytes = _canonical_json_bytes(index)
        if len(index_bytes) < 1 or len(index_bytes) > MAX_INDEX_BYTES:
            raise ValueError("Formal token corpus index has an invalid size")
        index_path = temporary / INDEX_FILENAME
        index_path.write_bytes(index_bytes)
        _fsync_file(index_path)
        _fsync_directory(event_root)
        _fsync_directory(temporary)
        index_sha = hashlib.sha256(index_bytes).hexdigest()

        # Strictly reload the staged artifact before it receives its formal name.
        load_formal_token_corpus(
            temporary,
            expected_index_sha256=index_sha,
            preprocessing_selection=preprocessing_selection,
        )
        if os.path.lexists(target):
            raise FileExistsError(
                f"Formal token corpus output already exists: {target}"
            )
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return load_formal_token_corpus(
            target,
            expected_index_sha256=index_sha,
            preprocessing_selection=preprocessing_selection,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a complete target-free TUSZ ictal LaBraM token corpus"
        )
    )
    parser.add_argument("--master-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-master-bundle-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-master-source-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--training-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-training-bundle-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-training-source-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--preprocessing-selection-bundle",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-preprocessing-selection-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-preprocessing-protocol-receipt-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--labram-modeling-path", type=Path, required=True)
    parser.add_argument(
        "--expected-labram-modeling-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--labram-checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--expected-foundation-feature-receipt-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    preprocessing_selection = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        expected_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
    )
    artifact = materialize_formal_tusz_ictal_token_corpus(
        master_manifest_bundle=args.master_manifest_bundle,
        expected_master_bundle_manifest_sha256=(
            args.expected_master_bundle_manifest_sha256
        ),
        expected_master_source_manifest_sha256=(
            args.expected_master_source_manifest_sha256
        ),
        training_manifest_bundle=args.training_manifest_bundle,
        expected_training_bundle_manifest_sha256=(
            args.expected_training_bundle_manifest_sha256
        ),
        expected_training_source_manifest_sha256=(
            args.expected_training_source_manifest_sha256
        ),
        preprocessing_selection=preprocessing_selection,
        edf_root=args.edf_root,
        labram_modeling_path=args.labram_modeling_path,
        labram_checkpoint_path=args.labram_checkpoint_path,
        expected_labram_modeling_sha256=args.expected_labram_modeling_sha256,
        expected_foundation_feature_receipt_sha256=(
            args.expected_foundation_feature_receipt_sha256
        ),
        output_directory=args.output_directory,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "index_sha256": artifact.index_sha256,
                "master_source_manifest_sha256": (
                    artifact.master_source_manifest_sha256
                ),
                "training_source_manifest_sha256": (
                    artifact.training_source_manifest_sha256
                ),
                "preprocessing_selection_artifact_sha256": (
                    artifact.preprocessing_selection_artifact_sha256
                ),
                "preprocessing_selected_arm_id": (
                    artifact.preprocessing_selected_arm_id
                ),
                "event_count": artifact.event_count,
                "patient_count": artifact.patient_count,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
