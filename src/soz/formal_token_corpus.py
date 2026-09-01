"""Opaque receipts issued only after strict formal token-corpus validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_VERIFIED_FORMAL_CORPUS_MARKER = object()
_VERIFIED_FORMAL_CORPUS_SUBSET_MARKER = object()


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class FormalTokenEventBinding:
    event_id: str
    bundle_path: Path
    bundle_manifest_sha256: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if not isinstance(self.bundle_path, Path) or not self.bundle_path.is_absolute():
            raise ValueError("bundle_path must be an absolute pathlib.Path")
        _sha(self.bundle_manifest_sha256, field="bundle_manifest_sha256")
        _sha(self.tensor_sha256, field="tensor_sha256")


@dataclass(frozen=True)
class FormalTokenSubsetEventBinding:
    """One signal-only event selected from a closed formal corpus index."""

    event_id: str
    patient_id: str
    event_record_sha256: str
    preprocess_receipt_sha256: str
    bundle_path: Path
    bundle_manifest_sha256: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if (
            not isinstance(self.patient_id, str)
            or not re.fullmatch(r"[a-z0-9]{8}", self.patient_id)
            or not self.event_id.startswith(f"{self.patient_id}_")
        ):
            raise ValueError("patient_id must be the canonical event owner")
        if not isinstance(self.bundle_path, Path) or not self.bundle_path.is_absolute():
            raise ValueError("bundle_path must be an absolute pathlib.Path")
        for field in (
            "event_record_sha256",
            "preprocess_receipt_sha256",
            "bundle_manifest_sha256",
            "tensor_sha256",
        ):
            _sha(getattr(self, field), field=field)


def formal_token_subset_roster_sha256(
    events: Sequence[FormalTokenSubsetEventBinding],
) -> str:
    """Bind a selected event order without including machine-local paths."""

    payload = tuple(
        (
            event.patient_id,
            event.event_id,
            event.event_record_sha256,
            event.preprocess_receipt_sha256,
            event.bundle_manifest_sha256,
            event.tensor_sha256,
        )
        for event in events
    )
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, init=False)
class VerifiedFormalTokenCorpusArtifact:
    """Strict-loader attestation; direct public construction is forbidden."""

    path: Path
    index_sha256: str
    master_bundle_manifest_sha256: str
    master_source_manifest_sha256: str
    training_bundle_manifest_sha256: str
    training_source_manifest_sha256: str
    preprocessing_selection_artifact_sha256: str
    preprocessing_selection_bundle_receipt_sha256: str
    preprocessing_protocol_receipt_sha256: str
    preprocessing_selected_arm_result_receipt_sha256: str
    preprocessing_selected_arm_id: str
    event_roster_sha256: str
    patient_roster_sha256: str
    patient_event_roster_sha256: str
    tensor_roster_sha256: str
    event_count: int
    patient_count: int
    events: tuple[FormalTokenEventBinding, ...]

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        index_sha256: str,
        master_bundle_manifest_sha256: str,
        master_source_manifest_sha256: str,
        training_bundle_manifest_sha256: str,
        training_source_manifest_sha256: str,
        preprocessing_selection_artifact_sha256: str,
        preprocessing_selection_bundle_receipt_sha256: str,
        preprocessing_protocol_receipt_sha256: str,
        preprocessing_selected_arm_result_receipt_sha256: str,
        preprocessing_selected_arm_id: str,
        event_roster_sha256: str,
        patient_roster_sha256: str,
        patient_event_roster_sha256: str,
        tensor_roster_sha256: str,
        event_count: int,
        patient_count: int,
        events: Sequence[FormalTokenEventBinding],
    ) -> None:
        if _verification_marker is not _VERIFIED_FORMAL_CORPUS_MARKER:
            raise TypeError(
                "VerifiedFormalTokenCorpusArtifact can only be issued by the strict loader"
            )
        values = {
            "path": path,
            "index_sha256": index_sha256,
            "master_bundle_manifest_sha256": master_bundle_manifest_sha256,
            "master_source_manifest_sha256": master_source_manifest_sha256,
            "training_bundle_manifest_sha256": training_bundle_manifest_sha256,
            "training_source_manifest_sha256": training_source_manifest_sha256,
            "preprocessing_selection_artifact_sha256": (
                preprocessing_selection_artifact_sha256
            ),
            "preprocessing_selection_bundle_receipt_sha256": (
                preprocessing_selection_bundle_receipt_sha256
            ),
            "preprocessing_protocol_receipt_sha256": (
                preprocessing_protocol_receipt_sha256
            ),
            "preprocessing_selected_arm_result_receipt_sha256": (
                preprocessing_selected_arm_result_receipt_sha256
            ),
            "preprocessing_selected_arm_id": preprocessing_selected_arm_id,
            "event_roster_sha256": event_roster_sha256,
            "patient_roster_sha256": patient_roster_sha256,
            "patient_event_roster_sha256": patient_event_roster_sha256,
            "tensor_roster_sha256": tensor_roster_sha256,
            "event_count": event_count,
            "patient_count": patient_count,
            "events": tuple(events),
        }
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("path must be an absolute pathlib.Path")
        for field in (
            "index_sha256",
            "master_bundle_manifest_sha256",
            "master_source_manifest_sha256",
            "training_bundle_manifest_sha256",
            "training_source_manifest_sha256",
            "preprocessing_selection_artifact_sha256",
            "preprocessing_selection_bundle_receipt_sha256",
            "preprocessing_protocol_receipt_sha256",
            "preprocessing_selected_arm_result_receipt_sha256",
            "event_roster_sha256",
            "patient_roster_sha256",
            "patient_event_roster_sha256",
            "tensor_roster_sha256",
        ):
            _sha(values[field], field=field)
        if preprocessing_selected_arm_id != "C-CAR19":
            raise ValueError(
                "Formal TUSZ ictal corpora require selected arm C-CAR19"
            )
        if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 1:
            raise ValueError("event_count must be positive")
        if isinstance(patient_count, bool) or not isinstance(patient_count, int) or patient_count < 1:
            raise ValueError("patient_count must be positive")
        if len(values["events"]) != event_count or any(
            not isinstance(event, FormalTokenEventBinding) for event in values["events"]
        ):
            raise ValueError("events must be the complete verified event roster")
        if len({event.event_id for event in values["events"]}) != event_count:
            raise ValueError("Verified event IDs must be unique")
        for field, value in values.items():
            object.__setattr__(self, field, value)


@dataclass(frozen=True, init=False)
class VerifiedFormalTokenCorpusSubsetArtifact:
    """Attestation for a strict event subset that opened no other bundles.

    The full index is validated, but only the requested event bundle
    manifests and tensors are opened.  This type is intentionally distinct
    from :class:`VerifiedFormalTokenCorpusArtifact`, whose loader opens every
    event in the corpus.
    """

    path: Path
    index_sha256: str
    master_bundle_manifest_sha256: str
    master_source_manifest_sha256: str
    training_bundle_manifest_sha256: str
    training_source_manifest_sha256: str
    preprocessing_selection_artifact_sha256: str
    preprocessing_selection_bundle_receipt_sha256: str
    preprocessing_protocol_receipt_sha256: str
    preprocessing_selected_arm_result_receipt_sha256: str
    preprocessing_selected_arm_id: str
    foundation_feature_receipt_sha256: str
    foundation_checkpoint_sha256: str
    foundation_modeling_sha256: str
    full_event_count: int
    full_patient_count: int
    selected_patient_ids: tuple[str, ...]
    selected_patient_roster_sha256: str
    selected_event_roster_sha256: str
    selected_event_count: int
    events: tuple[FormalTokenSubsetEventBinding, ...]
    unselected_event_bundles_opened: bool

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        index_sha256: str,
        master_bundle_manifest_sha256: str,
        master_source_manifest_sha256: str,
        training_bundle_manifest_sha256: str,
        training_source_manifest_sha256: str,
        preprocessing_selection_artifact_sha256: str,
        preprocessing_selection_bundle_receipt_sha256: str,
        preprocessing_protocol_receipt_sha256: str,
        preprocessing_selected_arm_result_receipt_sha256: str,
        preprocessing_selected_arm_id: str,
        foundation_feature_receipt_sha256: str,
        foundation_checkpoint_sha256: str,
        foundation_modeling_sha256: str,
        full_event_count: int,
        full_patient_count: int,
        selected_patient_ids: Sequence[str],
        selected_patient_roster_sha256: str,
        selected_event_roster_sha256: str,
        selected_event_count: int,
        events: Sequence[FormalTokenSubsetEventBinding],
        unselected_event_bundles_opened: bool,
    ) -> None:
        if _verification_marker is not _VERIFIED_FORMAL_CORPUS_SUBSET_MARKER:
            raise TypeError(
                "VerifiedFormalTokenCorpusSubsetArtifact can only be issued "
                "by the selective strict loader"
            )
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("path must be an absolute pathlib.Path")
        values = {
            "path": path,
            "index_sha256": index_sha256,
            "master_bundle_manifest_sha256": master_bundle_manifest_sha256,
            "master_source_manifest_sha256": master_source_manifest_sha256,
            "training_bundle_manifest_sha256": training_bundle_manifest_sha256,
            "training_source_manifest_sha256": training_source_manifest_sha256,
            "preprocessing_selection_artifact_sha256": (
                preprocessing_selection_artifact_sha256
            ),
            "preprocessing_selection_bundle_receipt_sha256": (
                preprocessing_selection_bundle_receipt_sha256
            ),
            "preprocessing_protocol_receipt_sha256": (
                preprocessing_protocol_receipt_sha256
            ),
            "preprocessing_selected_arm_result_receipt_sha256": (
                preprocessing_selected_arm_result_receipt_sha256
            ),
            "preprocessing_selected_arm_id": preprocessing_selected_arm_id,
            "foundation_feature_receipt_sha256": foundation_feature_receipt_sha256,
            "foundation_checkpoint_sha256": foundation_checkpoint_sha256,
            "foundation_modeling_sha256": foundation_modeling_sha256,
            "full_event_count": full_event_count,
            "full_patient_count": full_patient_count,
            "selected_patient_ids": tuple(selected_patient_ids),
            "selected_patient_roster_sha256": selected_patient_roster_sha256,
            "selected_event_roster_sha256": selected_event_roster_sha256,
            "selected_event_count": selected_event_count,
            "events": tuple(events),
            "unselected_event_bundles_opened": unselected_event_bundles_opened,
        }
        for field in (
            "index_sha256",
            "master_bundle_manifest_sha256",
            "master_source_manifest_sha256",
            "training_bundle_manifest_sha256",
            "training_source_manifest_sha256",
            "preprocessing_selection_artifact_sha256",
            "preprocessing_selection_bundle_receipt_sha256",
            "preprocessing_protocol_receipt_sha256",
            "preprocessing_selected_arm_result_receipt_sha256",
            "foundation_feature_receipt_sha256",
            "foundation_checkpoint_sha256",
            "foundation_modeling_sha256",
            "selected_patient_roster_sha256",
            "selected_event_roster_sha256",
        ):
            _sha(values[field], field=field)
        if preprocessing_selected_arm_id != "C-CAR19":
            raise ValueError("Selective formal corpora require C-CAR19")
        for field in ("full_event_count", "full_patient_count", "selected_event_count"):
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        patients = values["selected_patient_ids"]
        if (
            not patients
            or patients != tuple(sorted(patients))
            or len(set(patients)) != len(patients)
            or any(not re.fullmatch(r"[a-z0-9]{8}", patient) for patient in patients)
        ):
            raise ValueError("selected_patient_ids must be canonical, sorted, and unique")
        events_tuple = values["events"]
        if (
            len(events_tuple) != selected_event_count
            or any(
                not isinstance(event, FormalTokenSubsetEventBinding)
                for event in events_tuple
            )
            or tuple(sorted(events_tuple, key=lambda event: (event.patient_id, event.event_id)))
            != events_tuple
            or len({event.event_id for event in events_tuple}) != len(events_tuple)
        ):
            raise ValueError("events must be the complete canonical selected roster")
        if {event.patient_id for event in events_tuple} != set(patients):
            raise ValueError("selected patients and events do not have exact coverage")
        expected_patient_sha = hashlib.sha256(
            json.dumps(
                patients,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if selected_patient_roster_sha256 != expected_patient_sha:
            raise ValueError("selected patient roster receipt mismatch")
        if selected_event_roster_sha256 != formal_token_subset_roster_sha256(
            events_tuple
        ):
            raise ValueError("selected event roster receipt mismatch")
        if unselected_event_bundles_opened is not False:
            raise ValueError("Selective loader cannot attest opening unselected bundles")
        if full_event_count < selected_event_count or full_patient_count < len(patients):
            raise ValueError("Selected corpus counts cannot exceed full corpus counts")
        for field, value in values.items():
            object.__setattr__(self, field, value)

    @property
    def selected_tensor_roster_sha256(self) -> str:
        """Bind the selected event/tensor roster without touching bundles."""

        payload = tuple(
            (event.event_id, event.tensor_sha256) for event in self.events
        )
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def _issue_verified_formal_token_corpus_artifact(
    **kwargs: object,
) -> VerifiedFormalTokenCorpusArtifact:
    return VerifiedFormalTokenCorpusArtifact(
        _verification_marker=_VERIFIED_FORMAL_CORPUS_MARKER,
        **kwargs,
    )


def _issue_verified_formal_token_corpus_subset_artifact(
    **kwargs: object,
) -> VerifiedFormalTokenCorpusSubsetArtifact:
    return VerifiedFormalTokenCorpusSubsetArtifact(
        _verification_marker=_VERIFIED_FORMAL_CORPUS_SUBSET_MARKER,
        **kwargs,
    )


__all__ = [
    "FormalTokenEventBinding",
    "FormalTokenSubsetEventBinding",
    "VerifiedFormalTokenCorpusArtifact",
    "VerifiedFormalTokenCorpusSubsetArtifact",
    "formal_token_subset_roster_sha256",
]
