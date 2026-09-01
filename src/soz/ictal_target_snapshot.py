"""Hash-pinned target snapshots for repeated LaBraM recovery fits.

The formal-v4 native-prediction artifacts contain target tensors that were
already regenerated from TUSZ ``.csv/.csv_bi`` files under strict EDF and
annotation replay.  Re-reading every EDF once per post-hoc recovery head adds
hours of I/O but no new scientific information.  This module consumes only a
fully hash-pinned final formal-v4 artifact and exposes its source-native target
arrays as an immutable cache.  Prediction logits are deliberately ignored.

The snapshot contains TUSZ bipolar edge-time involvement targets.  It is not
SOZ, onset-channel, origin, or propagation supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

import numpy as np
import torch

from .cached_concept_training import IctalTokenBagDataset, IctalTokenPatientBag
from .concept_token_io import load_labram_concept_tokens
from .data.tusz_training import TUSZIctalTrainingManifest
from .formal_token_corpus import VerifiedFormalTokenCorpusArtifact
from .geometry import N_TCP_EDGES


ICTAL_TARGET_SNAPSHOT_SCHEMA = "soz_verified_formal_v4_ictal_target_snapshot_v1"
ICTAL_TARGET_SNAPSHOT_SEMANTICS = "tusz_bipolar_edge_time_involvement_not_soz"
_EXPECTED_FILES = {
    "manifest.json",
    "receipt.json",
    "full_native_logits.npy",
    "native_targets.npy",
    "native_target_mask.npy",
    "training_targets.npy",
    "training_target_mask.npy",
}
_MAX_JSON_BYTES = 16 * 1024 * 1024


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _read_regular(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Snapshot file must be regular: {path.name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Snapshot file must be regular: {path.name}")
        limit = None if maximum_bytes is None else maximum_bytes + 1
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(limit)
        if maximum_bytes is not None and len(raw) > maximum_bytes:
            raise ValueError(f"Snapshot file is too large: {path.name}")
        if metadata.st_size != len(raw):
            raise ValueError(f"Snapshot file changed while read: {path.name}")
        return raw
    finally:
        os.close(descriptor)


def _strict_json(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], str]:
    raw = _read_regular(path, maximum_bytes=_MAX_JSON_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _require_sha256(expected_sha256, field=f"expected_{path.name}_sha256"):
        raise ValueError(f"Snapshot {path.name} SHA mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Snapshot {path.name} is not valid JSON") from exc
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != raw:
        raise ValueError(f"Snapshot {path.name} is not canonical JSON")
    return payload, digest


def _tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = f"{name}|{tuple(value.shape)}|{value.dtype}".encode("ascii")
    digest.update(len(metadata).to_bytes(4, "little"))
    digest.update(metadata)
    raw = value.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


def _native_target_sha256(tensor: torch.Tensor) -> str:
    """Replay the target hash embedded in each TUSZ event manifest record."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_tensor(
    source: Path,
    *,
    name: str,
    record: object,
    expected_filename: str,
) -> torch.Tensor:
    if not isinstance(record, Mapping):
        raise TypeError(f"Snapshot tensor record must be a mapping: {name}")
    payload = dict(record)
    if payload.get("filename") != expected_filename:
        raise ValueError(f"Snapshot tensor filename changed: {name}")
    raw = _read_regular(source / expected_filename)
    if hashlib.sha256(raw).hexdigest() != _require_sha256(
        payload.get("file_sha256"), field=f"{name}.file_sha256"
    ):
        raise ValueError(f"Snapshot tensor file SHA mismatch: {name}")
    if payload.get("file_size_bytes") != len(raw):
        raise ValueError(f"Snapshot tensor file size mismatch: {name}")
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Snapshot tensor is not safe NumPy: {name}") from exc
    tensor = torch.from_numpy(np.ascontiguousarray(array)).contiguous()
    if list(tensor.shape) != payload.get("shape") or str(tensor.dtype) != payload.get(
        "dtype"
    ):
        raise ValueError(f"Snapshot tensor shape/dtype mismatch: {name}")
    if _tensor_sha256(name, tensor) != _require_sha256(
        payload.get("tensor_sha256"), field=f"{name}.tensor_sha256"
    ):
        raise ValueError(f"Snapshot tensor receipt mismatch: {name}")
    return tensor


def _event_rows(value: object, *, field: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    rows: list[tuple[str, str]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{field} rows must be [event_id, patient_id]")
        event_id, patient_id = (str(item).strip() for item in row)
        if not event_id or not patient_id:
            raise ValueError(f"{field} contains an empty identity")
        rows.append((event_id, patient_id))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"{field} contains duplicate event IDs")
    return tuple(rows)


@dataclass(frozen=True)
class VerifiedIctalTargetSnapshot:
    path: Path
    manifest_sha256: str
    receipt_sha256: str
    training_manifest_sha256: str
    training_corpus_index_sha256: str
    native_manifest_sha256: str
    native_corpus_index_sha256: str
    training_event_rows: tuple[tuple[str, str], ...]
    native_event_rows: tuple[tuple[str, str], ...]
    training_targets: torch.Tensor
    training_target_mask: torch.Tensor
    native_targets: torch.Tensor
    native_target_mask: torch.Tensor


def load_verified_ictal_target_snapshot(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedIctalTargetSnapshot:
    """Load only target tensors from one exact formal-v4 final artifact."""

    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("Target snapshot must be a regular absolute directory")
    if {item.name for item in source.iterdir()} != _EXPECTED_FILES:
        raise ValueError("Target snapshot bundle has missing or unknown files")
    manifest, manifest_sha = _strict_json(
        source / "manifest.json", expected_sha256=expected_manifest_sha256
    )
    receipt, receipt_sha = _strict_json(
        source / "receipt.json", expected_sha256=expected_receipt_sha256
    )
    if manifest.get("schema_version") != "soz_ictal_native_prediction_artifact_v1":
        raise ValueError("Target snapshot uses an unsupported source artifact schema")
    if receipt.get("schema_version") != "soz_ictal_native_prediction_bundle_receipt_v1":
        raise ValueError("Target snapshot uses an unsupported receipt schema")
    if receipt.get("artifact_sha256") != manifest_sha:
        raise ValueError("Target snapshot receipt does not bind its manifest")
    fixed = {
        "selection": "final",
        "target_semantics": ICTAL_TARGET_SNAPSHOT_SEMANTICS,
        "deepsoz_soz_labels_used": False,
        "private_labels_used": False,
        "missing_tusz_bins_imputed_as_negative": False,
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise ValueError("Target snapshot changed a frozen target semantic boundary")
    tensor_files = manifest.get("tensor_files")
    if not isinstance(tensor_files, Mapping):
        raise TypeError("Target snapshot tensor_files must be a mapping")
    training_rows = _event_rows(
        manifest.get("training_event_rows"), field="training_event_rows"
    )
    native_rows = _event_rows(manifest.get("native_event_rows"), field="native_event_rows")
    training_targets = _load_tensor(
        source,
        name="training_targets",
        record=tensor_files.get("training_targets"),
        expected_filename="training_targets.npy",
    ).to(torch.float32)
    training_mask = _load_tensor(
        source,
        name="training_target_mask",
        record=tensor_files.get("training_target_mask"),
        expected_filename="training_target_mask.npy",
    ).to(torch.bool)
    native_targets = _load_tensor(
        source,
        name="native_targets",
        record=tensor_files.get("native_targets"),
        expected_filename="native_targets.npy",
    ).to(torch.float32)
    native_mask = _load_tensor(
        source,
        name="native_target_mask",
        record=tensor_files.get("native_target_mask"),
        expected_filename="native_target_mask.npy",
    ).to(torch.bool)
    pairs = (
        ("training", training_rows, training_targets, training_mask),
        ("native", native_rows, native_targets, native_mask),
    )
    for name, rows, targets, mask in pairs:
        if tuple(targets.shape) != (len(rows), N_TCP_EDGES, 60):
            raise ValueError(f"Target snapshot {name} targets have the wrong shape")
        if mask.shape != targets.shape:
            raise ValueError(f"Target snapshot {name} mask has the wrong shape")
        if not torch.isfinite(targets).all():
            raise ValueError(f"Target snapshot {name} targets are non-finite")
        observed = targets[mask]
        if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
            raise ValueError(f"Target snapshot {name} observed targets are not binary")
    return VerifiedIctalTargetSnapshot(
        path=source,
        manifest_sha256=manifest_sha,
        receipt_sha256=receipt_sha,
        training_manifest_sha256=_require_sha256(
            manifest.get("training_manifest_sha256"),
            field="training_manifest_sha256",
        ),
        training_corpus_index_sha256=_require_sha256(
            manifest.get("training_corpus_index_sha256"),
            field="training_corpus_index_sha256",
        ),
        native_manifest_sha256=_require_sha256(
            manifest.get("native_evaluation_manifest_sha256"),
            field="native_evaluation_manifest_sha256",
        ),
        native_corpus_index_sha256=_require_sha256(
            manifest.get("native_evaluation_corpus_index_sha256"),
            field="native_evaluation_corpus_index_sha256",
        ),
        training_event_rows=training_rows,
        native_event_rows=native_rows,
        training_targets=training_targets.contiguous(),
        training_target_mask=training_mask.contiguous(),
        native_targets=native_targets.contiguous(),
        native_target_mask=native_mask.contiguous(),
    )


def _normalized_patients(
    values: Sequence[object], *, available: Sequence[str]
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("patient_ids must be a sequence")
    patients = tuple(str(value).strip() for value in values)
    if not patients or patients != tuple(sorted(patients)) or len(set(patients)) != len(
        patients
    ):
        raise ValueError("patient_ids must be non-empty, sorted, and unique")
    if not set(patients) <= set(available):
        raise ValueError("Target snapshot request includes an unknown patient")
    return patients


def build_tusz_ictal_token_bag_dataset_from_target_snapshot(
    manifest: TUSZIctalTrainingManifest,
    formal_corpus: VerifiedFormalTokenCorpusArtifact,
    snapshot: VerifiedIctalTargetSnapshot,
    *,
    patient_ids: Sequence[object],
) -> IctalTokenBagDataset:
    """Join verified tokens to the previously replayed master target cache.

    ``manifest`` may be a strict OOF-derived subset of the master manifest.
    Every requested event must exist in the hash-pinned master snapshot with
    the same patient identity.  No prediction logit is loaded or exposed.
    """

    if not isinstance(manifest, TUSZIctalTrainingManifest):
        raise TypeError("manifest must be TUSZIctalTrainingManifest")
    if not isinstance(formal_corpus, VerifiedFormalTokenCorpusArtifact):
        raise TypeError("formal_corpus must come from the strict corpus loader")
    if not isinstance(snapshot, VerifiedIctalTargetSnapshot):
        raise TypeError("snapshot must be a verified target snapshot")
    if formal_corpus.training_source_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("Token corpus does not bind the requested training manifest")
    if tuple(event.event_id for event in manifest) != tuple(
        binding.event_id for binding in formal_corpus.events
    ):
        raise ValueError("Token corpus and manifest event rosters differ")
    patients = _normalized_patients(patient_ids, available=manifest.patient_ids)
    snapshot_index = {
        event_id: (index, patient_id)
        for index, (event_id, patient_id) in enumerate(snapshot.training_event_rows)
    }
    selected_events = tuple(
        event
        for patient_id in patients
        for event in manifest.events_for_patient(patient_id)
    )
    if not selected_events:
        raise ValueError("Target snapshot dataset has no selected event")
    for event in selected_events:
        if snapshot_index.get(event.event_id, (None, None))[1] != event.patient_id:
            raise ValueError("Target snapshot event/patient identity mismatch")
        row_index = snapshot_index[event.event_id][0]
        target = snapshot.training_targets[row_index]
        mask = snapshot.training_target_mask[row_index]
        if (
            _native_target_sha256(target) != event.target_sha256
            or _native_target_sha256(mask) != event.target_mask_sha256
        ):
            raise ValueError(
                "Target snapshot tensor differs from the current event manifest"
            )
    binding_by_event = {binding.event_id: binding for binding in formal_corpus.events}
    first = selected_events[0]
    first_binding = binding_by_event[first.event_id]
    first_token = load_labram_concept_tokens(
        first_binding.bundle_path,
        expected_manifest_sha256=first_binding.bundle_manifest_sha256,
    )
    foundation_receipt = first_token.foundation_feature_receipt_sha256
    foundation_checkpoint = first_token.foundation_checkpoint_sha256

    def load_patient(patient_id: str) -> IctalTokenPatientBag:
        if patient_id not in set(patients):
            raise KeyError(f"Patient is outside the target-snapshot view: {patient_id}")
        events = manifest.events_for_patient(patient_id)
        token_events = []
        targets = []
        masks = []
        for event in events:
            binding = binding_by_event[event.event_id]
            token = load_labram_concept_tokens(
                binding.bundle_path,
                expected_manifest_sha256=binding.bundle_manifest_sha256,
            )
            checks = (
                token.event_id == event.event_id,
                token.source_concept_manifest_sha256 == manifest.manifest_sha256,
                token.event_record_sha256 == event.event_record_sha256,
                token.preprocess_receipt_sha256
                == event.signal_preflight_receipt_sha256,
                token.foundation_feature_receipt_sha256 == foundation_receipt,
                token.foundation_checkpoint_sha256 == foundation_checkpoint,
            )
            if not all(checks):
                raise ValueError("Target-snapshot token/event lineage mismatch")
            row_index, row_patient = snapshot_index[event.event_id]
            if row_patient != patient_id:
                raise ValueError("Target-snapshot patient changed during lazy replay")
            token_events.append(token)
            targets.append(snapshot.training_targets[row_index].clone())
            masks.append(snapshot.training_target_mask[row_index].clone())
        event_ids = tuple(event.event_id for event in events)
        return IctalTokenPatientBag(
            patient_id=patient_id,
            event_ids=event_ids,
            expected_event_ids=event_ids,
            training_manifest_sha256=manifest.manifest_sha256,
            expected_event_record_sha256s=tuple(
                event.event_record_sha256 for event in events
            ),
            token_events=tuple(token_events),
            targets=torch.stack(targets, dim=0).to(torch.float32),
            target_mask=torch.stack(masks, dim=0).to(torch.bool),
        )

    return IctalTokenBagDataset(
        patients,
        load_patient,
        training_manifest_sha256=manifest.manifest_sha256,
        token_source_manifest_sha256=manifest.manifest_sha256,
        foundation_feature_receipt_sha256=foundation_receipt,
        formal_token_corpus_verified=True,
        formal_token_corpus_index_sha256=formal_corpus.index_sha256,
        formal_token_corpus_training_bundle_manifest_sha256=(
            formal_corpus.training_bundle_manifest_sha256
        ),
        formal_token_corpus_event_roster_sha256=formal_corpus.event_roster_sha256,
        formal_token_corpus_patient_roster_sha256=formal_corpus.patient_roster_sha256,
        formal_token_corpus_tensor_roster_sha256=formal_corpus.tensor_roster_sha256,
    )


__all__ = (
    "ICTAL_TARGET_SNAPSHOT_SCHEMA",
    "VerifiedIctalTargetSnapshot",
    "build_tusz_ictal_token_bag_dataset_from_target_snapshot",
    "load_verified_ictal_target_snapshot",
)
