"""Cryptographically bound public-event and concept-cache provenance.

This module deliberately keeps target-bearing identity outside model tensors.
The registry owns the event-to-patient join; callers cannot attach an arbitrary
patient ID or split to an evidence tensor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Iterator, Mapping, Sequence

import pandas as pd
import torch

from ..evidence import EvidenceBatch
from ..evidence_schema import (
    EVIDENCE_TENSOR_SEMANTICS_SHA256,
    require_current_evidence_semantics,
)
from ..temporal_masks import (
    OFFSET_AWARE_PHASE_POLICY_SHA256,
    OFFSET_TIME_TOLERANCE_SEC,
    PRIMARY_WINDOW_START_SEC,
)
from .deepsoz import DeepSOZReferenceRegistry, normalize_patient_id


PUBLIC_EVIDENCE_SOURCE = "deepsoz_tusz_overlay"
EVIDENCE_CACHE_SCHEMA = "soz_evidence_cache_v3"
EVENT_TEMPORAL_PROVENANCE_SCHEMA = "soz_event_temporal_provenance_v1"
_MODEL_BY_OFFICIAL_SPLIT = {
    "train": "source_train",
    "dev": "source_dev",
    "eval": "source_eval",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: str | Path) -> str:
    """Hash a file without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return text


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def patient_roster_sha256(patient_ids: Iterable[object]) -> str:
    """Hash a sorted, duplicate-free normalized target-patient roster."""

    normalized = tuple(sorted(normalize_patient_id(value) for value in patient_ids))
    if len(set(normalized)) != len(normalized):
        raise ValueError("Patient roster cannot contain duplicates")
    return _canonical_sha256(normalized)


def evidence_batch_sha256(evidence: EvidenceBatch) -> str:
    """Hash semantic schema plus all values, masks, shapes, and dtypes."""

    digest = hashlib.sha256()
    semantic_metadata = (
        f"evidence_semantics|{EVIDENCE_TENSOR_SEMANTICS_SHA256}"
    ).encode("ascii")
    digest.update(len(semantic_metadata).to_bytes(4, "little"))
    digest.update(semantic_metadata)
    tensors = (
        ("node", evidence.node),
        ("edge", evidence.edge),
        ("node_mask", evidence.node_mask),
        ("edge_mask", evidence.edge_mask),
        ("physical_signal_mask", evidence.physical_signal_mask),
        ("ictal_phase_mask", evidence.ictal_phase_mask),
        ("morphology_mask", evidence.morphology_mask),
        ("morphology_context_mask", evidence.morphology_context_mask),
        ("ictal_mask", evidence.ictal_mask),
    )
    for name, tensor in tensors:
        metadata = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode("ascii")
        digest.update(len(metadata).to_bytes(4, "little"))
        digest.update(metadata)
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def ictal_phase_mask_sha256(mask: torch.Tensor) -> str:
    """Hash one detached boolean ``[15]`` primary phase mask."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError("ictal phase mask must be a torch.Tensor")
    value = mask.detach().cpu().contiguous()
    if value.dtype != torch.bool or tuple(value.shape) != (15,):
        raise ValueError("ictal phase mask must have bool shape [15]")
    return _canonical_sha256(
        {
            "shape": [15],
            "dtype": "bool",
            "values": [bool(item) for item in value.tolist()],
        }
    )


@dataclass(frozen=True)
class EventTemporalProvenanceReceipt:
    """Non-learned timing proof for one cached event's primary phase mask."""

    event_id: str
    global_timeline_receipt_sha256: str
    temporal_phase_policy_sha256: str
    ictal_phase_mask_sha256: str
    offset_trustworthy: bool
    seizure_duration_sec: float
    previous_timeline_trustworthy: bool
    has_previous_seizure: bool
    previous_seizure_overlap: bool
    previous_seizure_gap_sec: float
    current_offset_semantics: str = (
        "official_global_event_stop_minus_t0_never_window_stop"
    )
    pre_anchor_semantics: str = "pre_anchor_context_not_interictal_baseline"
    gap_usage: str = (
        "lt12s_primary_overlap_mask_30s_60s_sensitivity_only_not_model_feature"
    )
    schema_version: str = EVENT_TEMPORAL_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        if not event_id:
            raise ValueError("Temporal provenance event_id cannot be empty")
        object.__setattr__(self, "event_id", event_id)
        for field_name in (
            "global_timeline_receipt_sha256",
            "temporal_phase_policy_sha256",
            "ictal_phase_mask_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field=field_name),
            )
        if self.temporal_phase_policy_sha256 != OFFSET_AWARE_PHASE_POLICY_SHA256:
            raise ValueError("Temporal provenance uses another phase policy")
        for field_name in (
            "offset_trustworthy",
            "previous_timeline_trustworthy",
            "has_previous_seizure",
            "previous_seizure_overlap",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        for field_name in ("seizure_duration_sec", "previous_seizure_gap_sec"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)
        if self.offset_trustworthy and self.seizure_duration_sec <= 0:
            raise ValueError("Trustworthy offset requires positive seizure duration")
        expected_overlap = (
            self.has_previous_seizure
            and self.previous_timeline_trustworthy
            and self.previous_seizure_gap_sec
            < abs(PRIMARY_WINDOW_START_SEC) - OFFSET_TIME_TOLERANCE_SEC
        )
        if self.previous_seizure_overlap != expected_overlap:
            raise ValueError("Previous-seizure overlap disagrees with trust/gap state")
        if self.current_offset_semantics != (
            "official_global_event_stop_minus_t0_never_window_stop"
        ):
            raise ValueError("Window crop stop cannot be used as seizure offset")
        if self.pre_anchor_semantics != "pre_anchor_context_not_interictal_baseline":
            raise ValueError("Pre-anchor context cannot be claimed as interictal")
        if self.gap_usage != (
            "lt12s_primary_overlap_mask_30s_60s_sensitivity_only_not_model_feature"
        ):
            raise ValueError("Previous-seizure gap cannot become a model feature")
        if self.schema_version != EVENT_TEMPORAL_PROVENANCE_SCHEMA:
            raise ValueError("Unsupported event temporal provenance schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ConceptExtractorReceipt:
    """Auditable lineage for one learned evidence-family extractor.

    ``training_target_patient_ids`` records only patients that occur in the
    DeepSOZ target cohort. Auxiliary TUSZ/TUEV patients may be recorded in the
    producer's full training manifest, but cannot weaken this overlap audit.
    """

    concept_family: str
    checkpoint_sha256: str
    scaler_sha256: str
    split_manifest_sha256: str
    oof_fold: int | None
    training_target_patient_ids: tuple[str, ...]
    held_out_target_patient_ids: tuple[str, ...]
    training_target_roster_sha256: str
    held_out_target_roster_sha256: str

    def __post_init__(self) -> None:
        family = str(self.concept_family).strip()
        if not family:
            raise ValueError("concept_family cannot be empty")
        object.__setattr__(self, "concept_family", family)
        for field in ("checkpoint_sha256", "scaler_sha256", "split_manifest_sha256"):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), field=field),
            )
        if self.oof_fold is not None and (
            isinstance(self.oof_fold, bool)
            or not isinstance(self.oof_fold, int)
            or self.oof_fold not in range(5)
        ):
            raise ValueError("oof_fold must be None or an integer in [0,4]")

        training = tuple(
            sorted(normalize_patient_id(value) for value in self.training_target_patient_ids)
        )
        held_out = tuple(
            sorted(normalize_patient_id(value) for value in self.held_out_target_patient_ids)
        )
        if len(set(training)) != len(training) or len(set(held_out)) != len(held_out):
            raise ValueError("Extractor target-patient rosters cannot contain duplicates")
        if set(training) & set(held_out):
            raise ValueError("Extractor training and held-out rosters must be disjoint")
        object.__setattr__(self, "training_target_patient_ids", training)
        object.__setattr__(self, "held_out_target_patient_ids", held_out)

        declared_training = _require_sha256(
            self.training_target_roster_sha256,
            field="training_target_roster_sha256",
        )
        declared_heldout = _require_sha256(
            self.held_out_target_roster_sha256,
            field="held_out_target_roster_sha256",
        )
        if declared_training != patient_roster_sha256(training):
            raise ValueError("training_target_roster_sha256 does not match its roster")
        if declared_heldout != patient_roster_sha256(held_out):
            raise ValueError("held_out_target_roster_sha256 does not match its roster")
        object.__setattr__(self, "training_target_roster_sha256", declared_training)
        object.__setattr__(self, "held_out_target_roster_sha256", declared_heldout)

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class EvidenceCacheReceipt:
    """Bind one target-free evidence tensor to an event and extractor lineage."""

    event_id: str
    event_registry_sha256: str
    event_record_sha256: str
    evidence_sha256: str
    extractors: tuple[ConceptExtractorReceipt, ...]
    evidence_semantics_sha256: str = EVIDENCE_TENSOR_SEMANTICS_SHA256
    authorization_sha256: str | None = None
    temporal_provenance: EventTemporalProvenanceReceipt | None = None
    schema_version: str = EVIDENCE_CACHE_SCHEMA

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        if not event_id:
            raise ValueError("event_id cannot be empty")
        object.__setattr__(self, "event_id", event_id)
        for field in (
            "event_registry_sha256",
            "event_record_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), field=field),
            )
        if self.schema_version != EVIDENCE_CACHE_SCHEMA:
            raise ValueError(f"Unsupported evidence cache schema: {self.schema_version}")
        object.__setattr__(
            self,
            "evidence_semantics_sha256",
            require_current_evidence_semantics(self.evidence_semantics_sha256),
        )
        if self.authorization_sha256 is not None:
            object.__setattr__(
                self,
                "authorization_sha256",
                _require_sha256(
                    self.authorization_sha256,
                    field="authorization_sha256",
                ),
            )
        if self.temporal_provenance is not None:
            if not isinstance(
                self.temporal_provenance, EventTemporalProvenanceReceipt
            ):
                raise TypeError(
                    "temporal_provenance must be EventTemporalProvenanceReceipt"
                )
            if self.temporal_provenance.event_id != event_id:
                raise ValueError("Temporal provenance event ID disagrees with cache")
        families = tuple(receipt.concept_family for receipt in self.extractors)
        if len(set(families)) != len(families):
            raise ValueError("Evidence cache may contain one receipt per concept family")
        if not families:
            raise ValueError("Evidence cache requires at least one active-family receipt")

    def extractor(self, concept_family: str) -> ConceptExtractorReceipt:
        matches = tuple(
            receipt
            for receipt in self.extractors
            if receipt.concept_family == concept_family
        )
        if len(matches) != 1:
            raise KeyError(f"No unique extractor receipt for {concept_family}")
        return matches[0]


@dataclass(frozen=True)
class EventInputRecord:
    """Target-free, frozen event identity issued by :class:`EventInputRegistry`."""

    event_id: str
    patient_id: str
    source: str
    official_split: str
    model_split: str
    local_edf_path: str
    t0_sec: float
    window_start_sec: float
    window_stop_sec: float
    record_sha256: str

    def __post_init__(self) -> None:
        if self.source != PUBLIC_EVIDENCE_SOURCE:
            raise ValueError("Only the public DeepSOZ/TUSZ overlay is accepted")
        if _MODEL_BY_OFFICIAL_SPLIT.get(self.official_split) != self.model_split:
            raise ValueError("Official and model split are inconsistent")
        if not self.event_id or not self.local_edf_path:
            raise ValueError("Event ID and EDF path cannot be empty")
        if not all(
            math.isfinite(value)
            for value in (self.t0_sec, self.window_start_sec, self.window_stop_sec)
        ):
            raise ValueError("Event times must be finite")
        if self.window_stop_sec <= self.window_start_sec:
            raise ValueError("Event window must have positive duration")
        if abs((self.window_stop_sec - self.window_start_sec) - 60.0) > 1e-3:
            raise ValueError("Primary SOZ event windows must be exactly 60 seconds")
        _require_sha256(self.record_sha256, field="record_sha256")


class EventInputRegistry(Sequence[EventInputRecord]):
    """Immutable owner of the event→patient/split/source mapping."""

    def __init__(
        self,
        records: Iterable[EventInputRecord],
        *,
        manifest_sha256: str,
        split_manifest_sha256: str,
    ) -> None:
        self.manifest_sha256 = _require_sha256(
            manifest_sha256, field="manifest_sha256"
        )
        self.split_manifest_sha256 = _require_sha256(
            split_manifest_sha256, field="split_manifest_sha256"
        )
        ordered = tuple(sorted(records, key=lambda item: item.event_id))
        by_event = {record.event_id: record for record in ordered}
        if len(by_event) != len(ordered):
            raise ValueError("Event registry contains duplicate event IDs")
        by_patient: dict[str, list[EventInputRecord]] = {}
        for record in ordered:
            by_patient.setdefault(record.patient_id, []).append(record)
        self._records = ordered
        self._by_event = by_event
        self._by_patient = {
            key: tuple(value) for key, value in sorted(by_patient.items())
        }

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> EventInputRecord:
        return self._records[index]

    def __iter__(self) -> Iterator[EventInputRecord]:
        return iter(self._records)

    def get(self, event_id: object) -> EventInputRecord:
        key = str(event_id).strip()
        try:
            return self._by_event[key]
        except KeyError as exc:
            raise KeyError(f"Unknown registered event: {key}") from exc

    def events_for_patient(self, patient_id: object) -> tuple[EventInputRecord, ...]:
        return self._by_patient.get(normalize_patient_id(patient_id), ())

    def patient_ids_for_split(self, model_split: str) -> tuple[str, ...]:
        return tuple(
            patient_id
            for patient_id, records in self._by_patient.items()
            if records and records[0].model_split == model_split
        )

    def validate_cache_receipt(
        self,
        receipt: EvidenceCacheReceipt,
        references: DeepSOZReferenceRegistry,
    ) -> EventInputRecord:
        record = self.get(receipt.event_id)
        if receipt.event_registry_sha256 != self.manifest_sha256:
            raise ValueError("Evidence cache was built from a different event registry")
        if receipt.event_record_sha256 != record.record_sha256:
            raise ValueError("Evidence cache event-record hash mismatch")

        reference = references.get(record.patient_id)
        if reference.model_split != record.model_split:
            raise ValueError("Event and target registry model splits disagree")

        train_ids = set(self.patient_ids_for_split("source_train"))
        dev_eval_ids = set(self.patient_ids_for_split("source_dev")) | set(
            self.patient_ids_for_split("source_eval")
        )
        all_target_ids = train_ids | dev_eval_ids
        for extractor in receipt.extractors:
            if extractor.split_manifest_sha256 != self.split_manifest_sha256:
                raise ValueError("Extractor used a different target split manifest")
            training = set(extractor.training_target_patient_ids)
            held_out = set(extractor.held_out_target_patient_ids)
            shared_external = (
                extractor.oof_fold is None
                and not training
                and held_out == all_target_ids
            )
            if record.model_split == "source_train":
                fold = reference.concept_oof_fold
                if (
                    isinstance(fold, bool)
                    or not isinstance(fold, int)
                    or fold not in range(5)
                ):
                    raise ValueError(
                        "Source-train references require an OOF fold in [0,4]"
                    )
                fold_held_out = {
                    patient_id
                    for patient_id in train_ids
                    if references.get(patient_id).concept_oof_fold == fold
                }
                fold_specific = (
                    extractor.oof_fold == fold
                    and training == train_ids - fold_held_out
                    and held_out == fold_held_out
                )
                if not (fold_specific or shared_external):
                    raise ValueError(
                        "Extractor roster is not authorized: source-train producer "
                        "is neither the matching OOF fold nor target-independent "
                        "shared-external"
                    )
            else:
                final_train_only = (
                    extractor.oof_fold is None
                    and training == train_ids
                    and held_out == dev_eval_ids
                )
                if not (final_train_only or shared_external):
                    raise ValueError(
                        "Extractor roster is not authorized: dev/eval producer is "
                        "neither final train-only nor target-independent shared-external"
                    )
            if record.patient_id in training:
                raise ValueError("Extractor provenance includes its target patient in training")
            if record.patient_id not in held_out:
                raise ValueError("Extractor provenance does not hold out its target patient")
        return record


def _strict_binary(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and math.isfinite(value) and value in (0.0, 1.0):
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    raise ValueError(f"{field} must be explicitly binary")


def build_event_input_registry(
    frame: pd.DataFrame,
    references: DeepSOZReferenceRegistry,
    *,
    manifest_sha256: str,
    split_manifest_sha256: str,
) -> EventInputRegistry:
    """Build a registry from the target-free ``event_inputs.csv`` contract."""

    required = {
        "source",
        "deepsoz_patient_id",
        "official_split",
        "event_id",
        "local_edf_path",
        "t0_sec",
        "window_start_sec",
        "window_stop_sec",
        "signal_input_eligible",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Event input manifest is missing columns: {missing}")

    records: list[EventInputRecord] = []
    seen: set[str] = set()
    for row_number, row in frame.iterrows():
        source = str(row["source"]).strip()
        if source != PUBLIC_EVIDENCE_SOURCE:
            raise ValueError(f"Row {row_number} has unauthorized source {source}")
        patient_id = normalize_patient_id(row["deepsoz_patient_id"])
        reference = references.get(patient_id)
        official_split = str(row["official_split"]).strip()
        expected_model_split = _MODEL_BY_OFFICIAL_SPLIT.get(official_split)
        if expected_model_split is None or expected_model_split != reference.model_split:
            raise ValueError(f"Row {row_number} official/model split mismatch")
        eligible = _strict_binary(
            row["signal_input_eligible"], field="signal_input_eligible"
        )
        if not eligible or not reference.eligible_for_localization:
            continue

        event_id = str(row["event_id"]).strip()
        if not event_id or event_id in seen:
            raise ValueError(f"Duplicate or empty eligible event ID: {event_id}")
        seen.add(event_id)
        payload: Mapping[str, object] = {
            "event_id": event_id,
            "patient_id": patient_id,
            "source": source,
            "official_split": official_split,
            "model_split": reference.model_split,
            "local_edf_path": str(row["local_edf_path"]).strip(),
            "t0_sec": float(row["t0_sec"]),
            "window_start_sec": float(row["window_start_sec"]),
            "window_stop_sec": float(row["window_stop_sec"]),
        }
        records.append(
            EventInputRecord(
                **payload,
                record_sha256=_canonical_sha256(payload),
            )
        )
    return EventInputRegistry(
        records,
        manifest_sha256=manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )


def load_event_input_registry(
    event_inputs_csv: str | Path,
    split_manifest_csv: str | Path,
    references: DeepSOZReferenceRegistry,
    *,
    expected_event_sha256: str | None = None,
    expected_split_sha256: str | None = None,
) -> EventInputRegistry:
    """Load and optionally pin both source manifests by SHA256."""

    event_path = Path(event_inputs_csv)
    split_path = Path(split_manifest_csv)
    event_hash = file_sha256(event_path)
    split_hash = file_sha256(split_path)
    if expected_event_sha256 is not None and event_hash != _require_sha256(
        expected_event_sha256, field="expected_event_sha256"
    ):
        raise ValueError("event_inputs.csv SHA256 does not match the pinned artifact")
    if expected_split_sha256 is not None and split_hash != _require_sha256(
        expected_split_sha256, field="expected_split_sha256"
    ):
        raise ValueError("split_manifest.csv SHA256 does not match the pinned artifact")
    return build_event_input_registry(
        pd.read_csv(event_path),
        references,
        manifest_sha256=event_hash,
        split_manifest_sha256=split_hash,
    )
