"""Path-independent public-data patient/content overlap ledger.

The ledger accepts only TUEV, TUSZ, and DeepSOZ target-overlay records.  Each
record is bound to a canonical patient key, official split, exact-file SHA256,
and signal-content SHA256; filesystem paths are deliberately absent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Iterable, Iterator, Mapping, Sequence

from .deepsoz import normalize_patient_id


PUBLIC_OVERLAP_LEDGER_SCHEMA = "public_patient_content_overlap_v1.0.0"
CONCEPT_COHORT_SCHEMA = "public_concept_training_cohort_v1.0.0"
PUBLIC_DATASET_SPLITS: Mapping[str, frozenset[str]] = {
    "tuev": frozenset({"train", "eval"}),
    "tusz": frozenset({"train", "dev", "eval"}),
    "deepsoz": frozenset({"train", "dev", "eval"}),
}
CONCEPT_DATASETS = frozenset({"tuev", "tusz"})
_SPLIT_ALIASES = {
    "source_train": "train",
    "source_dev": "dev",
    "source_eval": "eval",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PATIENT_KEY_RE = re.compile(r"[a-z0-9][a-z0-9._:-]*")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a 64-character SHA256 hex digest")
    return text


def normalize_public_patient_key(value: object) -> str:
    key = normalize_patient_id(value).strip().lower()
    if not _PATIENT_KEY_RE.fullmatch(key):
        raise ValueError(
            "patient_key must be a canonical public identifier, not a path or free text"
        )
    return key


def canonical_public_roster_sha256(record_sha256s: Iterable[object]) -> str:
    roster = tuple(sorted(str(value).strip().lower() for value in record_sha256s))
    if any(not _SHA256_RE.fullmatch(value) for value in roster):
        raise ValueError("Public roster entries must be SHA256 record identifiers")
    if len(set(roster)) != len(roster):
        raise ValueError("Public record roster cannot contain duplicates")
    return _canonical_sha256(roster)


def _patient_roster_sha256(patient_keys: Iterable[object]) -> str:
    roster = tuple(sorted(normalize_public_patient_key(value) for value in patient_keys))
    if len(set(roster)) != len(roster):
        raise ValueError("Held-out patient roster cannot contain duplicates")
    return _canonical_sha256(roster)


@dataclass(frozen=True)
class PublicDataRecord:
    dataset: str
    split: str
    patient_key: str
    file_sha256: str
    signal_content_sha256: str

    def __post_init__(self) -> None:
        dataset = str(self.dataset).strip().lower()
        if dataset not in PUBLIC_DATASET_SPLITS:
            raise ValueError(
                f"Unknown or unauthorized public dataset {self.dataset!r}; "
                "only TUEV/TUSZ/DeepSOZ are allowed"
            )
        split = str(self.split).strip().lower()
        split = _SPLIT_ALIASES.get(split, split)
        if split not in PUBLIC_DATASET_SPLITS[dataset]:
            raise ValueError(f"Unknown official split {self.split!r} for {dataset}")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "split", split)
        object.__setattr__(
            self, "patient_key", normalize_public_patient_key(self.patient_key)
        )
        object.__setattr__(
            self,
            "file_sha256",
            _require_sha256(self.file_sha256, field="file_sha256"),
        )
        object.__setattr__(
            self,
            "signal_content_sha256",
            _require_sha256(
                self.signal_content_sha256, field="signal_content_sha256"
            ),
        )

    @property
    def canonical_payload(self) -> tuple[str, str, str, str, str]:
        return (
            self.dataset,
            self.split,
            self.patient_key,
            self.file_sha256,
            self.signal_content_sha256,
        )

    @property
    def record_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


@dataclass(frozen=True)
class PublicOverlapGroup:
    match_kind: str
    match_key: str
    record_sha256s: tuple[str, ...]
    dataset_splits: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.match_kind not in {
            "patient_key",
            "file_sha256",
            "signal_content_sha256",
        }:
            raise ValueError(f"Unknown overlap match kind: {self.match_kind}")
        if len(self.record_sha256s) < 2 or len(set(self.record_sha256s)) != len(
            self.record_sha256s
        ):
            raise ValueError("An overlap group requires at least two unique records")
        if tuple(sorted(self.record_sha256s)) != self.record_sha256s:
            raise ValueError("Overlap record identifiers must be canonically sorted")
        if tuple(sorted(set(self.dataset_splits))) != self.dataset_splits:
            raise ValueError("Overlap dataset/split roles must be unique and sorted")

    @property
    def crosses_dataset(self) -> bool:
        return len({dataset for dataset, _ in self.dataset_splits}) > 1

    @property
    def crosses_split(self) -> bool:
        return len({split for _, split in self.dataset_splits}) > 1


@dataclass(frozen=True)
class PublicOverlapLedgerReceipt:
    ledger_sha256: str
    record_count: int
    dataset_split_counts: tuple[tuple[str, str, int], ...]
    patient_overlap_count: int
    file_sha_overlap_count: int
    signal_content_overlap_count: int
    schema_version: str = PUBLIC_OVERLAP_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ledger_sha256",
            _require_sha256(self.ledger_sha256, field="ledger_sha256"),
        )
        if self.schema_version != PUBLIC_OVERLAP_LEDGER_SCHEMA:
            raise ValueError("Unexpected public overlap ledger schema")
        if self.record_count < 1:
            raise ValueError("Public overlap ledger cannot be empty")
        if any(
            value < 0
            for value in (
                self.patient_overlap_count,
                self.file_sha_overlap_count,
                self.signal_content_overlap_count,
            )
        ):
            raise ValueError("Overlap counts cannot be negative")
        if tuple(sorted(self.dataset_split_counts)) != self.dataset_split_counts:
            raise ValueError("dataset_split_counts must be canonically sorted")
        if sum(count for _, _, count in self.dataset_split_counts) != self.record_count:
            raise ValueError("dataset_split_counts do not sum to record_count")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


class PublicOverlapLedger(Sequence[PublicDataRecord]):
    """Immutable public-data identity ledger and overlap index."""

    def __init__(self, records: Iterable[PublicDataRecord]) -> None:
        materialized = tuple(records)
        if not materialized:
            raise ValueError("Public overlap ledger cannot be empty")
        if any(not isinstance(record, PublicDataRecord) for record in materialized):
            raise TypeError("Ledger entries must be PublicDataRecord objects")
        ordered = tuple(sorted(materialized, key=lambda record: record.canonical_payload))
        record_ids = tuple(record.record_sha256 for record in ordered)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("Duplicate canonical public-data record")

        file_to_content: dict[str, str] = {}
        dataset_file: dict[tuple[str, str], PublicDataRecord] = {}
        dataset_content_role: dict[tuple[str, str], tuple[str, str]] = {}
        dataset_patient_split: dict[tuple[str, str], str] = {}
        for record in ordered:
            previous_content = file_to_content.setdefault(
                record.file_sha256, record.signal_content_sha256
            )
            if previous_content != record.signal_content_sha256:
                raise ValueError(
                    "One file_sha256 is bound to contradictory signal-content SHA values"
                )

            file_key = (record.dataset, record.file_sha256)
            previous_file = dataset_file.setdefault(file_key, record)
            if previous_file.record_sha256 != record.record_sha256:
                raise ValueError(
                    "One dataset/file SHA is assigned contradictory patient or split metadata"
                )

            content_key = (record.dataset, record.signal_content_sha256)
            role = (record.patient_key, record.split)
            previous_role = dataset_content_role.setdefault(content_key, role)
            if previous_role != role:
                raise ValueError(
                    "One dataset/content SHA is assigned contradictory patient or split metadata"
                )

            patient_key = (record.dataset, record.patient_key)
            previous_split = dataset_patient_split.setdefault(patient_key, record.split)
            if previous_split != record.split:
                raise ValueError(
                    "One dataset/patient key is assigned to contradictory official splits"
                )

        self._records = ordered
        self._record_ids = record_ids
        self._patient_overlaps = self._build_groups("patient_key")
        self._file_overlaps = self._build_groups("file_sha256")
        self._content_overlaps = self._build_groups("signal_content_sha256")
        counts = Counter((record.dataset, record.split) for record in ordered)
        self.receipt = PublicOverlapLedgerReceipt(
            ledger_sha256=_canonical_sha256(
                [record.canonical_payload for record in ordered]
            ),
            record_count=len(ordered),
            dataset_split_counts=tuple(
                (dataset, split, count)
                for (dataset, split), count in sorted(counts.items())
            ),
            patient_overlap_count=len(self._patient_overlaps),
            file_sha_overlap_count=len(self._file_overlaps),
            signal_content_overlap_count=len(self._content_overlaps),
        )

    def _build_groups(self, field: str) -> tuple[PublicOverlapGroup, ...]:
        grouped: dict[str, list[PublicDataRecord]] = defaultdict(list)
        for record in self._records:
            grouped[str(getattr(record, field))].append(record)
        groups: list[PublicOverlapGroup] = []
        for key, records in grouped.items():
            if len(records) < 2:
                continue
            roles = tuple(sorted({(record.dataset, record.split) for record in records}))
            if field == "patient_key" and len(roles) < 2:
                continue
            groups.append(
                PublicOverlapGroup(
                    match_kind=field,
                    match_key=key,
                    record_sha256s=tuple(
                        sorted(record.record_sha256 for record in records)
                    ),
                    dataset_splits=roles,
                )
            )
        return tuple(sorted(groups, key=lambda group: (group.match_kind, group.match_key)))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> PublicDataRecord:
        return self._records[index]

    def __iter__(self) -> Iterator[PublicDataRecord]:
        return iter(self._records)

    @property
    def patient_overlaps(self) -> tuple[PublicOverlapGroup, ...]:
        return self._patient_overlaps

    @property
    def file_sha_overlaps(self) -> tuple[PublicOverlapGroup, ...]:
        return self._file_overlaps

    @property
    def signal_content_overlaps(self) -> tuple[PublicOverlapGroup, ...]:
        return self._content_overlaps

    def records_for(
        self,
        *,
        dataset: str | None = None,
        split: str | None = None,
    ) -> tuple[PublicDataRecord, ...]:
        dataset_key = None if dataset is None else str(dataset).strip().lower()
        split_key = None if split is None else str(split).strip().lower()
        split_key = _SPLIT_ALIASES.get(split_key, split_key)
        if dataset_key is not None and dataset_key not in PUBLIC_DATASET_SPLITS:
            raise ValueError(f"Unknown dataset selector: {dataset}")
        if split_key is not None and split_key not in {"train", "dev", "eval"}:
            raise ValueError(f"Unknown split selector: {split}")
        return tuple(
            record
            for record in self._records
            if (dataset_key is None or record.dataset == dataset_key)
            and (split_key is None or record.split == split_key)
        )


@dataclass(frozen=True)
class ExcludedPublicRecord:
    record: PublicDataRecord
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reasons or tuple(sorted(set(self.reasons))) != self.reasons:
            raise ValueError("Exclusion reasons must be non-empty, unique, and sorted")


@dataclass(frozen=True)
class ConceptCohortReceipt:
    ledger_sha256: str
    ledger_receipt_sha256: str
    target_split: str
    concept_datasets: tuple[str, ...]
    heldout_target_patient_keys: tuple[str, ...]
    heldout_target_roster_sha256: str
    allowed_record_sha256s: tuple[str, ...]
    must_exclude_record_sha256s: tuple[str, ...]
    allowed_roster_sha256: str
    must_exclude_roster_sha256: str
    exclusion_reason_roster_sha256: str
    schema_version: str = CONCEPT_COHORT_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "ledger_sha256",
            "ledger_receipt_sha256",
            "heldout_target_roster_sha256",
            "allowed_roster_sha256",
            "must_exclude_roster_sha256",
            "exclusion_reason_roster_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.schema_version != CONCEPT_COHORT_SCHEMA:
            raise ValueError("Unexpected concept cohort receipt schema")
        if self.target_split not in {"train", "dev", "eval"}:
            raise ValueError("target_split must be train/dev/eval")
        if (
            not self.concept_datasets
            or tuple(sorted(set(self.concept_datasets))) != self.concept_datasets
            or any(dataset not in CONCEPT_DATASETS for dataset in self.concept_datasets)
        ):
            raise ValueError("concept_datasets must be unique TUEV/TUSZ names")
        if tuple(sorted(set(self.heldout_target_patient_keys))) != (
            self.heldout_target_patient_keys
        ):
            raise ValueError("heldout_target_patient_keys must be unique and sorted")
        if self.heldout_target_roster_sha256 != _patient_roster_sha256(
            self.heldout_target_patient_keys
        ):
            raise ValueError("heldout_target_roster_sha256 does not match its roster")
        if tuple(sorted(set(self.allowed_record_sha256s))) != self.allowed_record_sha256s:
            raise ValueError("Allowed record roster must be unique and sorted")
        if tuple(sorted(set(self.must_exclude_record_sha256s))) != (
            self.must_exclude_record_sha256s
        ):
            raise ValueError("Must-exclude record roster must be unique and sorted")
        if set(self.allowed_record_sha256s) & set(self.must_exclude_record_sha256s):
            raise ValueError("Allowed and must-exclude record rosters must be disjoint")
        if self.allowed_roster_sha256 != canonical_public_roster_sha256(
            self.allowed_record_sha256s
        ):
            raise ValueError("allowed_roster_sha256 does not match its roster")
        if self.must_exclude_roster_sha256 != canonical_public_roster_sha256(
            self.must_exclude_record_sha256s
        ):
            raise ValueError("must_exclude_roster_sha256 does not match its roster")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ConceptTrainingCohort:
    allowed_records: tuple[PublicDataRecord, ...]
    must_exclude_records: tuple[ExcludedPublicRecord, ...]
    receipt: ConceptCohortReceipt

    def __post_init__(self) -> None:
        allowed_ids = tuple(sorted(record.record_sha256 for record in self.allowed_records))
        excluded_ids = tuple(
            sorted(item.record.record_sha256 for item in self.must_exclude_records)
        )
        if allowed_ids != self.receipt.allowed_record_sha256s:
            raise ValueError("Allowed records disagree with the cohort receipt")
        if excluded_ids != self.receipt.must_exclude_record_sha256s:
            raise ValueError("Must-exclude records disagree with the cohort receipt")
        if any(record.split != "train" for record in self.allowed_records):
            raise ValueError("Concept training may use official-train records only")
        if any(
            record.dataset not in self.receipt.concept_datasets
            for record in self.allowed_records
        ):
            raise ValueError("Allowed record belongs to an unselected concept dataset")

    @property
    def allowed_patient_roster(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted({(record.dataset, record.patient_key) for record in self.allowed_records})
        )

    @property
    def must_exclude_patient_roster(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                {
                    (item.record.dataset, item.record.patient_key)
                    for item in self.must_exclude_records
                }
            )
        )


def build_public_overlap_ledger(
    records: Iterable[PublicDataRecord],
) -> PublicOverlapLedger:
    return PublicOverlapLedger(records)


def build_concept_training_cohort(
    ledger: PublicOverlapLedger,
    *,
    target_split: str,
    heldout_target_patient_keys: Sequence[object],
    concept_datasets: Sequence[str] = ("tuev", "tusz"),
) -> ConceptTrainingCohort:
    """Authorize a globally split-safe auxiliary concept-training cohort.

    All public dev/eval identities and fingerprints are reserved.  For a
    source-train OOF build, the explicitly held-out DeepSOZ target patients are
    added to that protected set.  Dev/eval builds require the complete target
    split roster and still consume auxiliary official-train records only.
    """

    if not isinstance(ledger, PublicOverlapLedger):
        raise TypeError("ledger must be a PublicOverlapLedger")
    split = _SPLIT_ALIASES.get(str(target_split).strip().lower(), str(target_split).strip().lower())
    if split not in {"train", "dev", "eval"}:
        raise ValueError(f"Unknown target split: {target_split}")
    selected_datasets = tuple(sorted(str(value).strip().lower() for value in concept_datasets))
    if (
        not selected_datasets
        or len(set(selected_datasets)) != len(selected_datasets)
        or any(dataset not in CONCEPT_DATASETS for dataset in selected_datasets)
    ):
        raise ValueError("concept_datasets may contain unique TUEV/TUSZ names only")
    heldout = tuple(
        sorted(normalize_public_patient_key(value) for value in heldout_target_patient_keys)
    )
    if not heldout or len(set(heldout)) != len(heldout):
        raise ValueError("heldout_target_patient_keys must be non-empty and unique")

    target_records = ledger.records_for(dataset="deepsoz", split=split)
    target_patients = {record.patient_key for record in target_records}
    unknown_heldout = sorted(set(heldout) - target_patients)
    if unknown_heldout:
        raise ValueError(
            f"Held-out patients are absent from DeepSOZ target split {split}: "
            f"{unknown_heldout}"
        )
    if split in {"dev", "eval"} and set(heldout) != target_patients:
        missing = sorted(target_patients - set(heldout))
        extra = sorted(set(heldout) - target_patients)
        raise ValueError(
            f"{split} cohort requires the complete DeepSOZ target roster; "
            f"missing={missing}, extra={extra}"
        )

    protected_records = [record for record in ledger if record.split in {"dev", "eval"}]
    protected_records.extend(
        record
        for record in target_records
        if record.patient_key in set(heldout) and record not in protected_records
    )
    protected_patients = {record.patient_key for record in protected_records}
    protected_files = {record.file_sha256 for record in protected_records}
    protected_content = {
        record.signal_content_sha256 for record in protected_records
    }

    allowed: list[PublicDataRecord] = []
    excluded: list[ExcludedPublicRecord] = []
    for record in ledger:
        reasons: set[str] = set()
        if record.dataset not in selected_datasets:
            reasons.add("dataset_not_selected_for_concept")
        if record.split != "train":
            reasons.add("not_official_train")
        if record.patient_key in protected_patients:
            reasons.add("protected_patient_overlap")
        if record.file_sha256 in protected_files:
            reasons.add("protected_file_sha_overlap")
        if record.signal_content_sha256 in protected_content:
            reasons.add("protected_signal_content_sha_overlap")
        if reasons:
            excluded.append(
                ExcludedPublicRecord(record=record, reasons=tuple(sorted(reasons)))
            )
        else:
            allowed.append(record)

    allowed = sorted(allowed, key=lambda record: record.canonical_payload)
    excluded = sorted(
        excluded, key=lambda item: item.record.canonical_payload
    )
    if not allowed:
        raise ValueError("Overlap policy leaves no authorized official-train concept records")
    allowed_ids = tuple(sorted(record.record_sha256 for record in allowed))
    excluded_ids = tuple(sorted(item.record.record_sha256 for item in excluded))
    if len(allowed_ids) + len(excluded_ids) != len(ledger):
        raise RuntimeError("Concept cohort did not partition the complete public ledger")
    if any(record.patient_key in heldout for record in allowed):
        raise RuntimeError("A held-out target patient leaked into concept training")
    exclusion_reason_payload = tuple(
        sorted(
            (item.record.record_sha256, item.reasons)
            for item in excluded
        )
    )
    receipt = ConceptCohortReceipt(
        ledger_sha256=ledger.receipt.ledger_sha256,
        ledger_receipt_sha256=ledger.receipt.receipt_sha256,
        target_split=split,
        concept_datasets=selected_datasets,
        heldout_target_patient_keys=heldout,
        heldout_target_roster_sha256=_patient_roster_sha256(heldout),
        allowed_record_sha256s=allowed_ids,
        must_exclude_record_sha256s=excluded_ids,
        allowed_roster_sha256=canonical_public_roster_sha256(allowed_ids),
        must_exclude_roster_sha256=canonical_public_roster_sha256(excluded_ids),
        exclusion_reason_roster_sha256=_canonical_sha256(exclusion_reason_payload),
    )
    return ConceptTrainingCohort(
        allowed_records=tuple(allowed),
        must_exclude_records=tuple(excluded),
        receipt=receipt,
    )


__all__ = [
    "CONCEPT_COHORT_SCHEMA",
    "CONCEPT_DATASETS",
    "PUBLIC_DATASET_SPLITS",
    "PUBLIC_OVERLAP_LEDGER_SCHEMA",
    "ConceptCohortReceipt",
    "ConceptTrainingCohort",
    "ExcludedPublicRecord",
    "PublicDataRecord",
    "PublicOverlapGroup",
    "PublicOverlapLedger",
    "PublicOverlapLedgerReceipt",
    "build_concept_training_cohort",
    "build_public_overlap_ledger",
    "canonical_public_roster_sha256",
    "normalize_public_patient_key",
]
