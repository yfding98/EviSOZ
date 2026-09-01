"""Strict lazy join from a frozen TUSZ manifest to concept-token caches.

The join never decodes an EDF signal.  TUSZ EDF paths are used only through
the canonical official-train path contract while native ``.csv`` and
``.csv_bi`` targets are independently replayed for each requested patient.
Cached LaBraM tokens remain concept-training inputs and are not converted to
SOZ evidence or endpoint labels.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from .cached_concept_training import IctalTokenBagDataset, IctalTokenPatientBag
from .concept_token_io import (
    LoadedLaBraMConceptTokens,
    load_labram_concept_tokens,
)
from .data.tusz import (
    TUSZIctalInvolvementTarget,
    load_tusz_ictal_involvement_target,
)
from .data.tusz_training import (
    TUSZIctalEventRecord,
    TUSZIctalTrainingManifest,
    TUSZOfficialTrainFile,
    parse_tusz_official_train_path,
)
from .formal_token_corpus import VerifiedFormalTokenCorpusArtifact


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(values.shape), "dtype": str(values.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_bundle_directory(value: str | Path, *, event_id: str) -> Path:
    candidate = Path(value)
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(
            f"Concept-token path for {event_id!r} is not lexically canonical"
        )
    lexical = candidate.absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError(
            f"Concept-token path for {event_id!r} must be a regular directory"
        )
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise ValueError(
            f"Concept-token path for {event_id!r} may not traverse symlinks"
        )
    return resolved


def _validate_source_identity(
    source: TUSZOfficialTrainFile,
    event: TUSZIctalEventRecord,
) -> None:
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
            f"Frozen TUSZ event path replay failed identity fields {failed}"
        )


def _validate_replayed_target(
    target: TUSZIctalInvolvementTarget,
    event: TUSZIctalEventRecord,
) -> None:
    """Independently bind native TUSZ annotations and tensors to one event."""

    receipt = target.receipt
    checks = {
        "event_index": receipt.selected_global_event_index == event.event_index,
        "global_event_count": (
            receipt.global_seizure_event_count == event.global_event_count
        ),
        "event_t0": target.event_t0_sec == event.event_t0_sec,
        "event_stop": target.event_stop_sec == event.event_stop_sec,
        "seizure_type": (
            receipt.selected_global_seizure_type == event.seizure_type
        ),
        "edf_sha256": receipt.source_sha256 == event.edf_sha256,
        "channel_annotation_sha256": (
            receipt.channel_annotation_sha256 == event.channel_annotation_sha256
        ),
        "global_annotation_sha256": (
            receipt.global_annotation_sha256 == event.global_annotation_sha256
        ),
        "annotation_pair_sha256": (
            receipt.annotation_pair_sha256 == event.annotation_pair_sha256
        ),
        "target_sha256": _tensor_sha256(target.targets) == event.target_sha256,
        "target_mask_sha256": (
            _tensor_sha256(target.source_target_mask)
            == event.target_mask_sha256
        ),
        "bin_states_sha256": (
            _canonical_sha256(target.bin_states) == event.bin_states_sha256
        ),
        "observed_label_count": (
            int(target.source_target_mask.sum().item())
            == event.observed_label_count
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Frozen TUSZ target {event.event_id} failed replay fields {failed}"
        )


def _validate_token_event(
    token: LoadedLaBraMConceptTokens,
    event: TUSZIctalEventRecord,
    *,
    training_manifest_sha256: str,
    foundation_feature_receipt_sha256: str | None = None,
    foundation_checkpoint_sha256: str | None = None,
) -> None:
    checks = {
        "event_id": token.event_id == event.event_id,
        "source_manifest_sha256": (
            token.source_concept_manifest_sha256 == training_manifest_sha256
        ),
        "event_record_sha256": (
            token.event_record_sha256 == event.event_record_sha256
        ),
        "preprocess_receipt_sha256": (
            event.signal_preflight_receipt_sha256 is not None
            and token.preprocess_receipt_sha256
            == event.signal_preflight_receipt_sha256
        ),
        "foundation_feature_receipt_sha256": (
            foundation_feature_receipt_sha256 is None
            or token.foundation_feature_receipt_sha256
            == foundation_feature_receipt_sha256
        ),
        "foundation_checkpoint_sha256": (
            foundation_checkpoint_sha256 is None
            or token.foundation_checkpoint_sha256 == foundation_checkpoint_sha256
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Concept-token event {event.event_id} failed frozen lineage fields {failed}"
        )


def _validate_mapping(
    manifest: TUSZIctalTrainingManifest,
    token_bundle_by_event_id: Mapping[str, str | Path],
) -> dict[str, Path]:
    if not isinstance(token_bundle_by_event_id, Mapping):
        raise TypeError("token_bundle_by_event_id must be an explicit mapping")
    if any(not isinstance(key, str) or not key for key in token_bundle_by_event_id):
        raise TypeError("Every concept-token mapping key must be a non-empty event ID")
    expected = tuple(event.event_id for event in manifest.events)
    actual = tuple(token_bundle_by_event_id)
    missing = tuple(sorted(set(expected) - set(actual)))
    extra = tuple(sorted(set(actual) - set(expected)))
    if missing or extra or len(actual) != len(expected):
        raise ValueError(
            "Concept-token mapping must exactly equal the frozen event roster; "
            f"missing={missing}, extra={extra}"
        )
    directories = {
        event_id: _canonical_bundle_directory(
            token_bundle_by_event_id[event_id], event_id=event_id
        )
        for event_id in expected
    }
    directory_roster = tuple(directories.values())
    if len(set(directory_roster)) != len(directory_roster):
        raise ValueError("Each frozen event must map to one unique token bundle")
    return directories


def _replay_target(
    event: TUSZIctalEventRecord,
    *,
    source: TUSZOfficialTrainFile,
) -> TUSZIctalInvolvementTarget:
    _validate_source_identity(source, event)
    target = load_tusz_ictal_involvement_target(
        source.channel_annotation_path,
        source.global_annotation_path,
        event_index=event.event_index,
        source_path=source.edf_path,
    )
    _validate_replayed_target(target, event)
    return target


def _build_tusz_ictal_token_bag_dataset_from_mapping(
    manifest: TUSZIctalTrainingManifest,
    edf_root: str | Path,
    token_bundle_by_event_id: Mapping[str, str | Path],
    *,
    formal_artifact: VerifiedFormalTokenCorpusArtifact | None,
) -> IctalTokenBagDataset:
    """Freeze an exact cache join and return lazy complete-patient token bags.

    Bundle contents are fully validated once to pin their manifest hashes and
    common foundation lineage.  Every native target is independently replayed
    exactly once and retained only as a small CPU target/mask snapshot; no
    token tensor is retained by the dataset. A later patient lookup revalidates
    only that patient's pinned bundles and clones the already-validated target.
    """

    if not isinstance(manifest, TUSZIctalTrainingManifest):
        raise TypeError("manifest must be a TUSZIctalTrainingManifest")
    if not manifest.preflight_performed or any(
        event.signal_preflight_receipt_sha256 is None for event in manifest.events
    ):
        raise ValueError(
            "Formal cached training requires a fully signal-preflighted TUSZ manifest"
        )
    directories = _validate_mapping(manifest, token_bundle_by_event_id)

    # Parsing canonical paths does not decode EDF payloads.  It establishes
    # one immutable root and verifies every event still resolves to its
    # official-train source and paired annotation sidecars.
    canonical_root: Path | None = None
    source_by_event_id: dict[str, TUSZOfficialTrainFile] = {}
    for event in manifest.events:
        source = parse_tusz_official_train_path(edf_root, event.relative_edf_path)
        _validate_source_identity(source, event)
        if canonical_root is None:
            canonical_root = source.edf_root
        elif source.edf_root != canonical_root:
            raise RuntimeError("Canonical TUSZ root drifted while freezing token join")
        source_by_event_id[event.event_id] = source
    if canonical_root is None:  # Manifest itself already forbids this.
        raise RuntimeError("Frozen TUSZ manifest unexpectedly contains no events")

    pinned_bundle_manifest_sha256s: dict[str, str] = {}
    foundation_feature_receipt_sha256: str | None = None
    foundation_checkpoint_sha256: str | None = None
    for event in manifest.events:
        expected_bundle_sha = None
        if formal_artifact is not None:
            expected_bundle_sha = next(
                binding.bundle_manifest_sha256
                for binding in formal_artifact.events
                if binding.event_id == event.event_id
            )
        token = load_labram_concept_tokens(
            directories[event.event_id],
            expected_manifest_sha256=expected_bundle_sha,
        )
        _validate_token_event(
            token,
            event,
            training_manifest_sha256=manifest.manifest_sha256,
            foundation_feature_receipt_sha256=foundation_feature_receipt_sha256,
            foundation_checkpoint_sha256=foundation_checkpoint_sha256,
        )
        if foundation_feature_receipt_sha256 is None:
            foundation_feature_receipt_sha256 = (
                token.foundation_feature_receipt_sha256
            )
            foundation_checkpoint_sha256 = token.foundation_checkpoint_sha256
        pinned_bundle_manifest_sha256s[event.event_id] = token.manifest_sha256
        del token

    if (
        foundation_feature_receipt_sha256 is None
        or foundation_checkpoint_sha256 is None
    ):
        raise RuntimeError("Concept-token join failed to establish foundation lineage")

    # This is the only target replay in the cache-backed lifecycle.  It may
    # stream the source EDF once for its frozen SHA receipt, but never decodes
    # EEG samples. Epoch iteration uses only these small native target tensors.
    pinned_targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for event in manifest.events:
        target = _replay_target(
            event,
            source=source_by_event_id[event.event_id],
        )
        pinned_targets[event.event_id] = (
            target.targets.detach().to(dtype=torch.float32, device="cpu").clone(),
            target.source_target_mask.detach()
            .to(dtype=torch.bool, device="cpu")
            .clone(),
        )
        del target

    events_by_patient: dict[str, tuple[TUSZIctalEventRecord, ...]] = {
        patient_id: manifest.events_for_patient(patient_id)
        for patient_id in manifest.patient_ids
    }

    def load_patient(patient_id: str) -> IctalTokenPatientBag:
        try:
            patient_events = events_by_patient[patient_id]
        except KeyError as exc:
            raise KeyError(
                f"Patient {patient_id!r} is absent from the frozen token join"
            ) from exc
        token_events: list[LoadedLaBraMConceptTokens] = []
        targets: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for event in patient_events:
            token = load_labram_concept_tokens(
                directories[event.event_id],
                expected_manifest_sha256=pinned_bundle_manifest_sha256s[
                    event.event_id
                ],
            )
            _validate_token_event(
                token,
                event,
                training_manifest_sha256=manifest.manifest_sha256,
                foundation_feature_receipt_sha256=(
                    foundation_feature_receipt_sha256
                ),
                foundation_checkpoint_sha256=foundation_checkpoint_sha256,
            )
            pinned_target, pinned_mask = pinned_targets[event.event_id]
            token_events.append(token)
            targets.append(pinned_target.clone())
            masks.append(pinned_mask.clone())

        event_ids = tuple(event.event_id for event in patient_events)
        return IctalTokenPatientBag(
            patient_id=patient_id,
            event_ids=event_ids,
            expected_event_ids=event_ids,
            training_manifest_sha256=manifest.manifest_sha256,
            expected_event_record_sha256s=tuple(
                event.event_record_sha256 for event in patient_events
            ),
            token_events=tuple(token_events),
            targets=torch.stack(targets, dim=0),
            target_mask=torch.stack(masks, dim=0),
        )

    return IctalTokenBagDataset(
        manifest.patient_ids,
        load_patient,
        training_manifest_sha256=manifest.manifest_sha256,
        token_source_manifest_sha256=manifest.manifest_sha256,
        foundation_feature_receipt_sha256=foundation_feature_receipt_sha256,
        formal_token_corpus_verified=formal_artifact is not None,
        formal_token_corpus_index_sha256=(
            None if formal_artifact is None else formal_artifact.index_sha256
        ),
        formal_token_corpus_training_bundle_manifest_sha256=(
            None
            if formal_artifact is None
            else formal_artifact.training_bundle_manifest_sha256
        ),
        formal_token_corpus_event_roster_sha256=(
            None if formal_artifact is None else formal_artifact.event_roster_sha256
        ),
        formal_token_corpus_patient_roster_sha256=(
            None if formal_artifact is None else formal_artifact.patient_roster_sha256
        ),
        formal_token_corpus_tensor_roster_sha256=(
            None if formal_artifact is None else formal_artifact.tensor_roster_sha256
        ),
    )


def build_tusz_ictal_token_bag_dataset(
    manifest: TUSZIctalTrainingManifest,
    edf_root: str | Path,
    formal_corpus: VerifiedFormalTokenCorpusArtifact,
) -> IctalTokenBagDataset:
    """Build the formal training dataset only from a strict-loader artifact."""

    if not isinstance(formal_corpus, VerifiedFormalTokenCorpusArtifact):
        raise TypeError(
            "formal_corpus must be a VerifiedFormalTokenCorpusArtifact from the strict loader"
        )
    if manifest.manifest_sha256 != formal_corpus.training_source_manifest_sha256:
        raise ValueError("Formal corpus training manifest SHA does not match the dataset")
    if len(manifest) != formal_corpus.event_count or len(manifest.patient_ids) != (
        formal_corpus.patient_count
    ):
        raise ValueError("Formal corpus counts do not match the training manifest")
    event_roster = tuple(
        (
            event.event_id,
            event.patient_id,
            event.event_record_sha256,
            event.signal_preflight_receipt_sha256,
        )
        for event in manifest
    )
    patient_ids = tuple(sorted(manifest.patient_ids))
    patient_events = tuple(
        (
            patient_id,
            tuple(
                sorted(
                    event.event_id for event in manifest.events_for_patient(patient_id)
                )
            ),
        )
        for patient_id in patient_ids
    )
    checks = {
        "event_roster_sha256": (
            _canonical_sha256(event_roster) == formal_corpus.event_roster_sha256
        ),
        "patient_roster_sha256": (
            _canonical_sha256(patient_ids) == formal_corpus.patient_roster_sha256
        ),
        "patient_event_roster_sha256": (
            _canonical_sha256(patient_events)
            == formal_corpus.patient_event_roster_sha256
        ),
        "event_ids": (
            tuple(event.event_id for event in manifest)
            == tuple(binding.event_id for binding in formal_corpus.events)
        ),
    }
    failed = tuple(field for field, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Formal corpus/manifest roster mismatch: {failed}")
    mapping = {
        binding.event_id: binding.bundle_path for binding in formal_corpus.events
    }
    return _build_tusz_ictal_token_bag_dataset_from_mapping(
        manifest,
        edf_root,
        mapping,
        formal_artifact=formal_corpus,
    )


def build_nonformal_tusz_ictal_token_bag_dataset_from_mapping(
    manifest: TUSZIctalTrainingManifest,
    edf_root: str | Path,
    token_bundle_by_event_id: Mapping[str, str | Path],
) -> IctalTokenBagDataset:
    """Explicit synthetic/nonformal path for arbitrary event-to-bundle mappings."""

    return _build_tusz_ictal_token_bag_dataset_from_mapping(
        manifest,
        edf_root,
        token_bundle_by_event_id,
        formal_artifact=None,
    )


__all__: Sequence[str] = (
    "build_nonformal_tusz_ictal_token_bag_dataset_from_mapping",
    "build_tusz_ictal_token_bag_dataset",
)
