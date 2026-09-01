"""Physical fit-only LaBraM token view for v13 control preparation.

The broker may inspect the pinned full corpus index, but it opens only the
selected fit bundle contents and publishes a new directory containing hard
links to those selected immutable files.  The trainer receives this view, not
the source corpus root or a module exposing the full-corpus loader.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from .concept_token_io import load_labram_concept_tokens
from .formal_token_corpus import (
    FormalTokenSubsetEventBinding,
    VerifiedFormalTokenCorpusSubsetArtifact,
    _issue_verified_formal_token_corpus_subset_artifact,
    formal_token_subset_roster_sha256,
)
from .ictal_fit_only_consumer_v13 import LoadedFitOnlyTargetArtifactV13
from .ictal_fit_primitives_v13 import (
    canonical_json_bytes as _canonical_json_bytes,
    file_sha256 as _file_sha256,
    patient_roster as _patient_roster,
    patient_roster_sha256,
    require_sha256 as _require_sha256,
    safe_new_output as _safe_new_output,
    selection as _selection,
)


FIT_TOKEN_VIEW_SCHEMA_V13 = "soz_ictal_fit_token_view_v13"
FIT_TOKEN_VIEW_RECEIPT_SCHEMA_V13 = "soz_ictal_fit_token_view_receipt_v13"
FIT_TOKEN_VIEW_MANIFEST = "manifest.json"
FIT_TOKEN_VIEW_RECEIPT = "receipt.json"
FIT_TOKEN_VIEW_EVENTS = "events"
_ROOT_FILES = frozenset(
    {FIT_TOKEN_VIEW_MANIFEST, FIT_TOKEN_VIEW_RECEIPT, FIT_TOKEN_VIEW_EVENTS}
)
_BUNDLE_FILES = frozenset({"manifest.json", "concept_tokens.safetensors"})
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "selection",
        "oof_fold",
        "matched_k31_manifest_sha256",
        "training_manifest_bundle_sha256",
        "training_manifest_sha256",
        "source_corpus_index_sha256",
        "preprocessing_selection_artifact_sha256",
        "preprocessing_selection_bundle_receipt_sha256",
        "preprocessing_protocol_receipt_sha256",
        "preprocessing_selected_arm_result_receipt_sha256",
        "preprocessing_selected_arm_id",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "foundation_modeling_sha256",
        "fit_patient_ids",
        "fit_patient_roster_sha256",
        "i_gate_patient_ids_excluded_unopened",
        "i_gate_patient_roster_sha256",
        "fit_event_count",
        "fit_event_rows",
        "fit_event_roster_sha256",
        "fit_tensor_roster_sha256",
        "serialization",
        "physical_view_contains_fit_bundles_only",
        "unselected_event_bundles_present",
        "source_full_corpus_index_metadata_loaded_by_broker",
        "source_full_corpus_root_accessible_to_broker",
        "source_unselected_bundle_contents_opened_by_broker",
        "trainer_source_full_corpus_root_reachable",
        "trainer_imports_full_corpus_loader",
        "deepsoz_identity_outcome_prediction_reachable",
        "private_signal_identity_outcome_reachable",
        "i_gate_signal_or_tokens_opened",
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], str]:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError(f"fit-token JSON must be regular: {source.name}")
    before = source.stat()
    if before.st_size < 1 or before.st_size > _MAX_JSON_BYTES:
        raise ValueError(f"fit-token JSON size is invalid: {source.name}")
    raw = source.read_bytes()
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"fit-token JSON changed while read: {source.name}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _require_sha256(expected_sha256, field=f"expected_{source.name}_sha256"):
        raise ValueError(f"fit-token JSON SHA mismatch: {source.name}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"fit-token JSON is invalid: {source.name}") from exc
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != raw:
        raise ValueError(f"fit-token JSON is not canonical: {source.name}")
    return payload, digest


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"fit-token view already exists: {target}")
        raise OSError(error, os.strerror(error), str(target))


def _event_rows(value: object) -> tuple[FormalTokenSubsetEventBinding, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("fit-token event rows must be non-empty")
    rows: list[FormalTokenSubsetEventBinding] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("fit-token event row schema changed")
        event_id, patient_id, record_sha, preprocess_sha, manifest_sha, tensor_sha, relative = row
        if relative != f"events/{event_id}":
            raise ValueError("fit-token event path is not canonical")
        rows.append(
            FormalTokenSubsetEventBinding(
                event_id=str(event_id),
                patient_id=str(patient_id),
                event_record_sha256=_require_sha256(record_sha, field="event_record_sha256"),
                preprocess_receipt_sha256=_require_sha256(preprocess_sha, field="preprocess_receipt_sha256"),
                bundle_path=Path("/") / str(relative),
                bundle_manifest_sha256=_require_sha256(manifest_sha, field="bundle_manifest_sha256"),
                tensor_sha256=_require_sha256(tensor_sha, field="tensor_sha256"),
            )
        )
    if tuple(sorted(rows, key=lambda item: (item.patient_id, item.event_id))) != tuple(rows):
        raise ValueError("fit-token event rows are not canonical")
    if len({row.event_id for row in rows}) != len(rows):
        raise ValueError("fit-token event rows contain duplicate IDs")
    return tuple(rows)


@dataclass(frozen=True)
class LoadedFitTokenViewV13:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    receipt_sha256: str
    corpus: VerifiedFormalTokenCorpusSubsetArtifact


def materialize_fit_token_view_v13(
    output_directory: str | Path,
    *,
    source: VerifiedFormalTokenCorpusSubsetArtifact,
    fit_targets: LoadedFitOnlyTargetArtifactV13,
) -> LoadedFitTokenViewV13:
    if not isinstance(source, VerifiedFormalTokenCorpusSubsetArtifact):
        raise TypeError("source must come from the selective corpus loader")
    if not isinstance(fit_targets, LoadedFitOnlyTargetArtifactV13):
        raise TypeError("fit_targets must come from the fit-only target loader")
    target_manifest = fit_targets.manifest
    fit = _patient_roster(target_manifest["fit_patient_ids"], field="fit_patient_ids", allow_empty=False)
    gate = _patient_roster(
        target_manifest["i_gate_patient_ids_excluded_unopened"],
        field="i_gate_patient_ids_excluded_unopened",
        allow_empty=False,
    )
    if (
        source.selected_patient_ids != fit
        or source.unselected_event_bundles_opened is not False
        or source.index_sha256 != target_manifest["training_corpus_index_sha256"]
        or source.training_bundle_manifest_sha256 != target_manifest["training_manifest_bundle_sha256"]
        or source.training_source_manifest_sha256 != target_manifest["training_manifest_sha256"]
    ):
        raise ValueError("Selective source differs from fit-only authority")
    target_rows = tuple((event.event_id, event.patient_id) for event in fit_targets.events)
    source_rows = tuple((event.event_id, event.patient_id) for event in source.events)
    if target_rows != source_rows:
        raise ValueError("Fit target/token event rosters differ")
    target = _safe_new_output(output_directory)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.fit-token-", dir=target.parent))
    published = False
    try:
        events_dir = staging / FIT_TOKEN_VIEW_EVENTS
        events_dir.mkdir()
        event_payload = []
        for binding in source.events:
            source_bundle = binding.bundle_path
            if {item.name for item in source_bundle.iterdir()} != _BUNDLE_FILES:
                raise ValueError("Selected source token bundle file roster changed")
            destination = events_dir / binding.event_id
            destination.mkdir()
            for filename in sorted(_BUNDLE_FILES):
                source_file = source_bundle / filename
                destination_file = destination / filename
                if source_file.is_symlink() or not source_file.is_file():
                    raise ValueError("Selected source token file must be regular")
                os.link(source_file, destination_file, follow_symlinks=False)
                _fsync_file(destination_file)
            _fsync_directory(destination)
            event_payload.append(
                [
                    binding.event_id,
                    binding.patient_id,
                    binding.event_record_sha256,
                    binding.preprocess_receipt_sha256,
                    binding.bundle_manifest_sha256,
                    binding.tensor_sha256,
                    f"events/{binding.event_id}",
                ]
            )
        _fsync_directory(events_dir)
        payload = {
            "schema_version": FIT_TOKEN_VIEW_SCHEMA_V13,
            "purpose": "v13_matched_control_fit_only_tokens",
            "selection": target_manifest["selection"],
            "oof_fold": target_manifest["oof_fold"],
            "matched_k31_manifest_sha256": target_manifest["matched_k31_manifest_sha256"],
            "training_manifest_bundle_sha256": source.training_bundle_manifest_sha256,
            "training_manifest_sha256": source.training_source_manifest_sha256,
            "source_corpus_index_sha256": source.index_sha256,
            "preprocessing_selection_artifact_sha256": source.preprocessing_selection_artifact_sha256,
            "preprocessing_selection_bundle_receipt_sha256": source.preprocessing_selection_bundle_receipt_sha256,
            "preprocessing_protocol_receipt_sha256": source.preprocessing_protocol_receipt_sha256,
            "preprocessing_selected_arm_result_receipt_sha256": source.preprocessing_selected_arm_result_receipt_sha256,
            "preprocessing_selected_arm_id": source.preprocessing_selected_arm_id,
            "foundation_feature_receipt_sha256": source.foundation_feature_receipt_sha256,
            "foundation_checkpoint_sha256": source.foundation_checkpoint_sha256,
            "foundation_modeling_sha256": source.foundation_modeling_sha256,
            "fit_patient_ids": list(fit),
            "fit_patient_roster_sha256": patient_roster_sha256(fit),
            "i_gate_patient_ids_excluded_unopened": list(gate),
            "i_gate_patient_roster_sha256": patient_roster_sha256(gate),
            "fit_event_count": len(event_payload),
            "fit_event_rows": event_payload,
            "fit_event_roster_sha256": formal_token_subset_roster_sha256(source.events),
            "fit_tensor_roster_sha256": source.selected_tensor_roster_sha256,
            "serialization": "canonical_json_and_hardlinked_selected_safe_bundles",
            "physical_view_contains_fit_bundles_only": True,
            "unselected_event_bundles_present": False,
            "source_full_corpus_index_metadata_loaded_by_broker": True,
            "source_full_corpus_root_accessible_to_broker": True,
            "source_unselected_bundle_contents_opened_by_broker": False,
            "trainer_source_full_corpus_root_reachable": False,
            "trainer_imports_full_corpus_loader": False,
            "deepsoz_identity_outcome_prediction_reachable": False,
            "private_signal_identity_outcome_reachable": False,
            "i_gate_signal_or_tokens_opened": False,
        }
        raw = _canonical_json_bytes(payload)
        if set(payload) != _MANIFEST_FIELDS:
            raise ValueError("fit-token view manifest violates its closed schema")
        (staging / FIT_TOKEN_VIEW_MANIFEST).write_bytes(raw)
        manifest_sha = hashlib.sha256(raw).hexdigest()
        receipt = {
            "schema_version": FIT_TOKEN_VIEW_RECEIPT_SCHEMA_V13,
            "artifact_sha256": manifest_sha,
            "artifact_size_bytes": len(raw),
        }
        receipt_raw = _canonical_json_bytes(receipt)
        (staging / FIT_TOKEN_VIEW_RECEIPT).write_bytes(receipt_raw)
        _fsync_file(staging / FIT_TOKEN_VIEW_MANIFEST)
        _fsync_file(staging / FIT_TOKEN_VIEW_RECEIPT)
        _fsync_directory(staging)
        _rename_noreplace(staging, target)
        _fsync_directory(target.parent)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return load_fit_token_view_v13(
        target,
        expected_manifest_sha256=_file_sha256(target / FIT_TOKEN_VIEW_MANIFEST),
        expected_receipt_sha256=_file_sha256(target / FIT_TOKEN_VIEW_RECEIPT),
    )


def load_fit_token_view_v13(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> LoadedFitTokenViewV13:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("fit-token view must be a regular absolute directory")
    if {item.name for item in source.iterdir()} != _ROOT_FILES:
        raise ValueError("fit-token view root roster changed")
    manifest, manifest_sha = _strict_json(
        source / FIT_TOKEN_VIEW_MANIFEST, expected_sha256=expected_manifest_sha256
    )
    receipt, receipt_sha = _strict_json(
        source / FIT_TOKEN_VIEW_RECEIPT, expected_sha256=expected_receipt_sha256
    )
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("fit-token view manifest violates its closed schema")
    if receipt != {
        "schema_version": FIT_TOKEN_VIEW_RECEIPT_SCHEMA_V13,
        "artifact_sha256": manifest_sha,
        "artifact_size_bytes": (source / FIT_TOKEN_VIEW_MANIFEST).stat().st_size,
    }:
        raise ValueError("fit-token view receipt changed")
    fixed = {
        "schema_version": FIT_TOKEN_VIEW_SCHEMA_V13,
        "purpose": "v13_matched_control_fit_only_tokens",
        "preprocessing_selected_arm_id": "C-CAR19",
        "serialization": "canonical_json_and_hardlinked_selected_safe_bundles",
        "physical_view_contains_fit_bundles_only": True,
        "unselected_event_bundles_present": False,
        "source_full_corpus_index_metadata_loaded_by_broker": True,
        "source_full_corpus_root_accessible_to_broker": True,
        "source_unselected_bundle_contents_opened_by_broker": False,
        "trainer_source_full_corpus_root_reachable": False,
        "trainer_imports_full_corpus_loader": False,
        "deepsoz_identity_outcome_prediction_reachable": False,
        "private_signal_identity_outcome_reachable": False,
        "i_gate_signal_or_tokens_opened": False,
    }
    if any(manifest.get(field) != value for field, value in fixed.items()):
        raise ValueError("fit-token view changed an access boundary")
    _, fold = _selection(manifest["selection"])
    if manifest["oof_fold"] != fold:
        raise ValueError("fit-token view selection/fold mismatch")
    fit = _patient_roster(manifest["fit_patient_ids"], field="fit_patient_ids", allow_empty=False)
    gate = _patient_roster(
        manifest["i_gate_patient_ids_excluded_unopened"],
        field="i_gate_patient_ids_excluded_unopened",
        allow_empty=False,
    )
    if (
        len(gate) != 12
        or set(fit) & set(gate)
        or patient_roster_sha256(fit) != manifest["fit_patient_roster_sha256"]
        or patient_roster_sha256(gate) != manifest["i_gate_patient_roster_sha256"]
    ):
        raise ValueError("fit-token view patient firewall failed")
    for field in (
        "matched_k31_manifest_sha256",
        "training_manifest_bundle_sha256",
        "training_manifest_sha256",
        "source_corpus_index_sha256",
        "preprocessing_selection_artifact_sha256",
        "preprocessing_selection_bundle_receipt_sha256",
        "preprocessing_protocol_receipt_sha256",
        "preprocessing_selected_arm_result_receipt_sha256",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "foundation_modeling_sha256",
        "fit_event_roster_sha256",
        "fit_tensor_roster_sha256",
    ):
        _require_sha256(manifest[field], field=field)
    template_rows = _event_rows(manifest["fit_event_rows"])
    if len(template_rows) != manifest["fit_event_count"]:
        raise ValueError("fit-token view event count changed")
    if {row.patient_id for row in template_rows} != set(fit):
        raise ValueError("fit-token view event/patient coverage changed")
    events_dir = source / FIT_TOKEN_VIEW_EVENTS
    if events_dir.is_symlink() or not events_dir.is_dir():
        raise ValueError("fit-token events directory must be regular")
    expected_dirs = {row.event_id for row in template_rows}
    if {item.name for item in events_dir.iterdir()} != expected_dirs:
        raise ValueError("fit-token physical event roster changed")
    bindings: list[FormalTokenSubsetEventBinding] = []
    for template in template_rows:
        bundle = events_dir / template.event_id
        if bundle.is_symlink() or not bundle.is_dir() or {item.name for item in bundle.iterdir()} != _BUNDLE_FILES:
            raise ValueError("fit-token physical bundle changed")
        token = load_labram_concept_tokens(
            bundle, expected_manifest_sha256=template.bundle_manifest_sha256
        )
        checks = (
            token.event_id == template.event_id,
            token.source_concept_manifest_sha256 == manifest["training_manifest_sha256"],
            token.event_record_sha256 == template.event_record_sha256,
            token.preprocess_receipt_sha256 == template.preprocess_receipt_sha256,
            token.foundation_feature_receipt_sha256 == manifest["foundation_feature_receipt_sha256"],
            token.foundation_checkpoint_sha256 == manifest["foundation_checkpoint_sha256"],
            token.foundation_feature_receipt.modeling_sha256 == manifest["foundation_modeling_sha256"],
            token.tensor_sha256 == template.tensor_sha256,
        )
        if not all(checks):
            raise ValueError("fit-token physical bundle lineage changed")
        bindings.append(
            FormalTokenSubsetEventBinding(
                event_id=template.event_id,
                patient_id=template.patient_id,
                event_record_sha256=template.event_record_sha256,
                preprocess_receipt_sha256=template.preprocess_receipt_sha256,
                bundle_path=bundle,
                bundle_manifest_sha256=template.bundle_manifest_sha256,
                tensor_sha256=template.tensor_sha256,
            )
        )
    binding_tuple = tuple(bindings)
    if formal_token_subset_roster_sha256(binding_tuple) != manifest["fit_event_roster_sha256"]:
        raise ValueError("fit-token event roster receipt mismatch")
    corpus = _issue_verified_formal_token_corpus_subset_artifact(
        path=source,
        index_sha256=str(manifest["source_corpus_index_sha256"]),
        master_bundle_manifest_sha256=str(manifest["training_manifest_bundle_sha256"]),
        master_source_manifest_sha256=str(manifest["training_manifest_sha256"]),
        training_bundle_manifest_sha256=str(manifest["training_manifest_bundle_sha256"]),
        training_source_manifest_sha256=str(manifest["training_manifest_sha256"]),
        preprocessing_selection_artifact_sha256=str(manifest["preprocessing_selection_artifact_sha256"]),
        preprocessing_selection_bundle_receipt_sha256=str(manifest["preprocessing_selection_bundle_receipt_sha256"]),
        preprocessing_protocol_receipt_sha256=str(manifest["preprocessing_protocol_receipt_sha256"]),
        preprocessing_selected_arm_result_receipt_sha256=str(manifest["preprocessing_selected_arm_result_receipt_sha256"]),
        preprocessing_selected_arm_id=str(manifest["preprocessing_selected_arm_id"]),
        foundation_feature_receipt_sha256=str(manifest["foundation_feature_receipt_sha256"]),
        foundation_checkpoint_sha256=str(manifest["foundation_checkpoint_sha256"]),
        foundation_modeling_sha256=str(manifest["foundation_modeling_sha256"]),
        full_event_count=len(binding_tuple),
        full_patient_count=len(fit),
        selected_patient_ids=fit,
        selected_patient_roster_sha256=str(manifest["fit_patient_roster_sha256"]),
        selected_event_roster_sha256=str(manifest["fit_event_roster_sha256"]),
        selected_event_count=len(binding_tuple),
        events=binding_tuple,
        unselected_event_bundles_opened=False,
    )
    if corpus.selected_tensor_roster_sha256 != manifest["fit_tensor_roster_sha256"]:
        raise ValueError("fit-token tensor roster receipt mismatch")
    return LoadedFitTokenViewV13(
        path=source,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        receipt_sha256=receipt_sha,
        corpus=corpus,
    )


__all__ = (
    "FIT_TOKEN_VIEW_MANIFEST",
    "FIT_TOKEN_VIEW_RECEIPT",
    "FIT_TOKEN_VIEW_SCHEMA_V13",
    "LoadedFitTokenViewV13",
    "load_fit_token_view_v13",
    "materialize_fit_token_view_v13",
)
