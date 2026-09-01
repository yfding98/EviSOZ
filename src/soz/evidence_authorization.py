"""Independent authorization for formal patient-out-of-fold evidence caches.

An :class:`EvidenceCacheReceipt` is a claim made by a cache producer.  It is
not, by itself, proof that the named checkpoint, scaler, fold, or patient
rosters were authorized.  This module supplies the missing independent side
of that comparison.  A formal cache is accepted only when every active
M/I/V family exactly matches one lineage in a separately rebuilt OOF
authorization plan.

The authorization object contains provenance only.  It never contains raw
EEG, foundation tokens, DeepSOZ targets, private labels, or source annotation
coverage masks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Mapping, Sequence

import torch

from .data.batching import EvidenceEvent, PatientBagDataset
from .data.deepsoz import DeepSOZReferenceRegistry, normalize_patient_id
from .data.provenance import (
    ConceptExtractorReceipt,
    EventTemporalProvenanceReceipt,
    EventInputRegistry,
    evidence_batch_sha256,
    ictal_phase_mask_sha256,
)
from .evidence_schema import (
    EVIDENCE_TENSOR_SEMANTICS_SHA256,
    EVOLUTION_FAMILY,
    FAMILY_FEATURE_NAMES,
    FAMILY_FEATURE_SCHEMA_SHA256,
    ICTAL_FAMILY,
    MORPHOLOGY_FAMILY,
    require_current_evidence_semantics,
)
from .geometry import (
    N_MORPHOLOGY_FEATURES,
    N_TIME_TILES,
)
from .temporal_masks import (
    OFFSET_AWARE_PHASE_POLICY_SHA256,
    OffsetAwarePhaseMasks,
)


OOF_EVIDENCE_AUTHORIZATION_SCHEMA = "soz_oof_evidence_authorization_v3"
FAMILY_PRODUCER_AUTHORIZATION_SCHEMA = (
    "soz_family_producer_authorization_v2"
)
NATIVE_CLASS_SUPPORT_SCHEMA = "soz_native_binary_class_support_v1"

EVIDENCE_FAMILIES = (
    MORPHOLOGY_FAMILY,
    ICTAL_FAMILY,
    EVOLUTION_FAMILY,
)
_SELECTION_KEYS = (0, 1, 2, 3, 4, None)
_SOURCE_SPLITS = ("source_train", "source_dev", "source_eval")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_AUTHORIZATION_ISSUER = object()

_OUTPUT_SEMANTICS = {
    MORPHOLOGY_FAMILY: "typed_ce6_morphology_evidence",
    ICTAL_FAMILY: "conditional_ictal_involvement_score",
    EVOLUTION_FAMILY: "scaled_explicit_temporal_descriptors",
}

_FAMILY_SELECTION_POLICY = {
    MORPHOLOGY_FAMILY: "shared_external",
    ICTAL_FAMILY: "five_oof_plus_final",
    EVOLUTION_FAMILY: "five_oof_plus_final",
}

def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _normalize_source_ids(
    values: Sequence[object], *, field_name: str, require_nonempty: bool = True
) -> tuple[str, ...]:
    normalized = tuple(sorted(str(value).strip() for value in values))
    if require_nonempty and not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} cannot contain empty identifiers")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return normalized


def source_roster_sha256(values: Sequence[object]) -> str:
    """Hash a canonical source-patient/group roster without guessing identity."""

    return _canonical_sha256(
        _normalize_source_ids(values, field_name="source roster", require_nonempty=False)
    )


def _selection_order(value: int | None) -> int:
    return 5 if value is None else value


@dataclass(frozen=True)
class NativeClassSupportReceipt:
    """Explicit-label support for a binary native concept evaluation.

    Unknown cells are counted separately and are never converted to negative
    labels.  Patient-macro class-sensitive metrics and native probability
    calibration are disabled whenever fewer than half of evaluated patients
    contain both explicit classes.  This deliberately rejects the current
    source-development TUSZ native evaluation (2/16 negative-support
    patients) as a calibration set while still allowing conditional-score
    fidelity reporting.
    """

    event_count: int
    patient_count: int
    positive_label_count: int
    negative_label_count: int
    unknown_label_count: int
    positive_event_count: int
    negative_event_count: int
    mixed_class_event_count: int
    positive_patient_count: int
    negative_patient_count: int
    mixed_class_patient_count: int
    unknown_policy: str = "masked_never_imputed_as_negative"
    class_sensitive_metric_policy: str = (
        "requires_explicit_both_class_support_in_at_least_half_of_patients"
    )
    schema_version: str = NATIVE_CLASS_SUPPORT_SCHEMA

    def __post_init__(self) -> None:
        count_fields = (
            "event_count",
            "patient_count",
            "positive_label_count",
            "negative_label_count",
            "unknown_label_count",
            "positive_event_count",
            "negative_event_count",
            "mixed_class_event_count",
            "positive_patient_count",
            "negative_patient_count",
            "mixed_class_patient_count",
        )
        for name in count_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.event_count < 1 or self.patient_count < 1:
            raise ValueError("Native support requires non-empty event and patient rosters")
        if self.positive_event_count > self.event_count or self.negative_event_count > self.event_count:
            raise ValueError("Class-bearing event counts cannot exceed event_count")
        if self.positive_patient_count > self.patient_count or self.negative_patient_count > self.patient_count:
            raise ValueError("Class-bearing patient counts cannot exceed patient_count")
        if self.mixed_class_event_count > min(
            self.positive_event_count, self.negative_event_count
        ):
            raise ValueError("mixed_class_event_count exceeds class-bearing events")
        if self.mixed_class_patient_count > min(
            self.positive_patient_count, self.negative_patient_count
        ):
            raise ValueError("mixed_class_patient_count exceeds class-bearing patients")
        if (self.positive_label_count == 0) != (self.positive_event_count == 0):
            raise ValueError("Positive label and event support are inconsistent")
        if (self.negative_label_count == 0) != (self.negative_event_count == 0):
            raise ValueError("Negative label and event support are inconsistent")
        if self.positive_label_count < self.positive_event_count:
            raise ValueError("Positive labels cannot be fewer than positive events")
        if self.negative_label_count < self.negative_event_count:
            raise ValueError("Negative labels cannot be fewer than negative events")
        if self.unknown_policy != "masked_never_imputed_as_negative":
            raise ValueError("Unknown native targets must remain masked")
        if self.class_sensitive_metric_policy != (
            "requires_explicit_both_class_support_in_at_least_half_of_patients"
        ):
            raise ValueError("Native metric-support policy cannot be weakened")
        if self.schema_version != NATIVE_CLASS_SUPPORT_SCHEMA:
            raise ValueError("Unsupported native class-support schema")

    @property
    def class_sensitive_metrics_authorized(self) -> bool:
        return (
            self.positive_label_count > 0
            and self.negative_label_count > 0
            and 2 * self.mixed_class_patient_count >= self.patient_count
        )

    @property
    def probability_calibration_authorized(self) -> bool:
        return self.class_sensitive_metrics_authorized

    def require_class_sensitive_metrics(self) -> None:
        if not self.class_sensitive_metrics_authorized:
            raise ValueError(
                "Native AUROC/specificity is not identifiable under the explicit "
                "patient-level class support"
            )

    def require_probability_calibration(self) -> None:
        if not self.probability_calibration_authorized:
            raise ValueError(
                "Native probability calibration is not authorized by explicit "
                "patient-level negative support"
            )

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class FamilyProducerAuthorization:
    """Independent expected lineage for one family and one fold/final key."""

    extractor: ConceptExtractorReceipt
    fit_source_ids: tuple[str, ...]
    held_out_source_ids: tuple[str, ...]
    fit_source_roster_sha256: str
    held_out_source_roster_sha256: str
    source_unit_kind: str
    producer_run_receipt_sha256: str
    source_receipt_sha256: str
    preprocessing_receipt_sha256: str
    mask_semantics_receipt_sha256: str
    native_fidelity_receipt_sha256: str
    scale_alignment_receipt_sha256: str
    fold_identity_probe_receipt_sha256: str
    formal_family_gate_receipt_sha256: str
    native_fidelity_gate_passed: bool
    scale_alignment_gate_passed: bool
    fold_identity_probe_gate_passed: bool
    formal_family_gate_passed: bool
    feature_names: tuple[str, ...]
    feature_schema_sha256: str
    output_semantics: str
    native_class_support: NativeClassSupportReceipt | None = None
    source_target_mask_forbidden_from_cache: bool = True
    schema_version: str = FAMILY_PRODUCER_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.extractor, ConceptExtractorReceipt):
            raise TypeError("extractor must be a ConceptExtractorReceipt")
        family = self.extractor.concept_family
        if family not in EVIDENCE_FAMILIES:
            raise ValueError(f"Unsupported evidence family: {family!r}")
        fit = _normalize_source_ids(
            self.fit_source_ids, field_name="fit_source_ids"
        )
        held = _normalize_source_ids(
            self.held_out_source_ids, field_name="held_out_source_ids"
        )
        if set(fit) & set(held):
            raise ValueError("Producer fit and held-out source rosters overlap")
        object.__setattr__(self, "fit_source_ids", fit)
        object.__setattr__(self, "held_out_source_ids", held)
        declared_fit = _require_sha256(
            self.fit_source_roster_sha256, field_name="fit_source_roster_sha256"
        )
        declared_held = _require_sha256(
            self.held_out_source_roster_sha256,
            field_name="held_out_source_roster_sha256",
        )
        if declared_fit != source_roster_sha256(fit):
            raise ValueError("fit_source_roster_sha256 does not match its roster")
        if declared_held != source_roster_sha256(held):
            raise ValueError("held_out_source_roster_sha256 does not match its roster")
        object.__setattr__(self, "fit_source_roster_sha256", declared_fit)
        object.__setattr__(self, "held_out_source_roster_sha256", declared_held)
        if self.source_unit_kind not in {
            "patient",
            "subject_group",
            "session_group",
        }:
            raise ValueError("source_unit_kind must state the actual grouping unit")
        for name in (
            "producer_run_receipt_sha256",
            "source_receipt_sha256",
            "preprocessing_receipt_sha256",
            "mask_semantics_receipt_sha256",
            "native_fidelity_receipt_sha256",
            "scale_alignment_receipt_sha256",
            "fold_identity_probe_receipt_sha256",
            "formal_family_gate_receipt_sha256",
            "feature_schema_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), field_name=name),
            )
        names = tuple(str(value).strip() for value in self.feature_names)
        if names != FAMILY_FEATURE_NAMES[family]:
            raise ValueError(f"{family} feature names/order changed")
        object.__setattr__(self, "feature_names", names)
        if self.feature_schema_sha256 != FAMILY_FEATURE_SCHEMA_SHA256[family]:
            raise ValueError(f"{family} feature schema SHA changed")
        if self.output_semantics != _OUTPUT_SEMANTICS[family]:
            raise ValueError(
                f"{family} output semantics must be {_OUTPUT_SEMANTICS[family]!r}"
            )
        for gate_name in (
            "native_fidelity_gate_passed",
            "scale_alignment_gate_passed",
            "fold_identity_probe_gate_passed",
            "formal_family_gate_passed",
        ):
            if getattr(self, gate_name) is not True:
                raise ValueError(f"{gate_name} must pass before cache authorization")
        if family == ICTAL_FAMILY:
            if not isinstance(self.native_class_support, NativeClassSupportReceipt):
                raise ValueError("Ictal authorization requires explicit native support")
            # Even a support-rich fit split does not remove the source-selection
            # limitation.  The cache value remains a conditional score.
            if self.output_semantics != "conditional_ictal_involvement_score":
                raise ValueError("Ictal cache output cannot claim calibrated probability")
        elif self.native_class_support is not None:
            raise ValueError("Binary native support is only defined for the ictal family")
        if not self.source_target_mask_forbidden_from_cache:
            raise ValueError("Source annotation coverage may not enter reasoner caches")
        if self.schema_version != FAMILY_PRODUCER_AUTHORIZATION_SCHEMA:
            raise ValueError("Unsupported family producer authorization schema")

    @property
    def concept_family(self) -> str:
        return self.extractor.concept_family

    @property
    def oof_fold(self) -> int | None:
        return self.extractor.oof_fold

    @property
    def lineage_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class OOFEvidenceAuthorization:
    """Exact five-fold/final producer plan for every registered event."""

    event_registry_sha256: str
    split_manifest_sha256: str
    source_train_patient_folds: tuple[tuple[str, int], ...]
    source_dev_patient_ids: tuple[str, ...]
    source_eval_patient_ids: tuple[str, ...]
    event_records: tuple[tuple[str, str, str, str], ...]
    active_families: tuple[str, ...]
    family_selection_policies: tuple[tuple[str, str], ...]
    family_lineages: tuple[FamilyProducerAuthorization, ...]
    event_temporal_provenance: tuple[EventTemporalProvenanceReceipt, ...]
    evidence_semantics_sha256: str = EVIDENCE_TENSOR_SEMANTICS_SHA256
    temporal_phase_policy_sha256: str = OFFSET_AWARE_PHASE_POLICY_SHA256
    pre_anchor_semantics: str = "pre_anchor_context_not_assumed_interictal_baseline"
    annotation_coverage_forbidden: bool = True
    cache_publication_authorized: bool = True
    schema_version: str = OOF_EVIDENCE_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        for name in ("event_registry_sha256", "split_manifest_sha256"):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "evidence_semantics_sha256",
            require_current_evidence_semantics(self.evidence_semantics_sha256),
        )
        phase_policy_sha = _require_sha256(
            self.temporal_phase_policy_sha256,
            field_name="temporal_phase_policy_sha256",
        )
        if phase_policy_sha != OFFSET_AWARE_PHASE_POLICY_SHA256:
            raise ValueError("Temporal offset/previous-overlap phase policy changed")
        object.__setattr__(
            self, "temporal_phase_policy_sha256", phase_policy_sha
        )
        if self.pre_anchor_semantics != (
            "pre_anchor_context_not_assumed_interictal_baseline"
        ):
            raise ValueError("Pre-anchor context cannot be claimed as interictal baseline")
        folds = tuple(
            sorted(
                (normalize_patient_id(patient_id), fold)
                for patient_id, fold in self.source_train_patient_folds
            )
        )
        if not folds or len({patient_id for patient_id, _ in folds}) != len(folds):
            raise ValueError("source_train_patient_folds must be non-empty and unique")
        if any(
            isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5)
            for _, fold in folds
        ):
            raise ValueError("Every source-train patient requires one fold in [0,4]")
        object.__setattr__(self, "source_train_patient_folds", folds)
        dev = tuple(sorted(normalize_patient_id(value) for value in self.source_dev_patient_ids))
        eval_ids = tuple(sorted(normalize_patient_id(value) for value in self.source_eval_patient_ids))
        if not dev or not eval_ids or len(set(dev)) != len(dev) or len(set(eval_ids)) != len(eval_ids):
            raise ValueError("Source dev/eval patient rosters must be non-empty and unique")
        train_ids = {patient_id for patient_id, _ in folds}
        if train_ids & set(dev) or train_ids & set(eval_ids) or set(dev) & set(eval_ids):
            raise ValueError("Source split patient rosters must be disjoint")
        object.__setattr__(self, "source_dev_patient_ids", dev)
        object.__setattr__(self, "source_eval_patient_ids", eval_ids)

        active = tuple(family for family in EVIDENCE_FAMILIES if family in self.active_families)
        if active != self.active_families or not active:
            raise ValueError("active_families must be a canonical non-empty M/I/V subset")
        # Evolution is the minimum executable evidence family in the frozen
        # current design.  Morphology and ictal are optional only because each
        # has an independent, one-shot deployment/promotion gate.  Requiring I
        # here would contradict the preregistered v5 consequence (I is removed
        # after a failed gate); accepting a caller-selected I without a passed
        # ``formal_family_gate`` would be equally invalid.
        if EVOLUTION_FAMILY not in active:
            raise ValueError(
                "Formal primary caches require active temporal evolution; "
                "morphology/ictal are included only after their formal family gate"
            )

        expected_policies = tuple(
            (family, _FAMILY_SELECTION_POLICY[family]) for family in active
        )
        if self.family_selection_policies != expected_policies:
            raise ValueError(
                "family_selection_policies must use shared_external morphology "
                "and five_oof_plus_final ictal/evolution"
            )

        expected_lineage_keys = tuple(
            (family, key)
            for family in active
            for key in (
                (None,)
                if _FAMILY_SELECTION_POLICY[family] == "shared_external"
                else _SELECTION_KEYS
            )
        )
        actual_lineage_keys = tuple(
            (lineage.concept_family, lineage.oof_fold)
            for lineage in self.family_lineages
        )
        if actual_lineage_keys != expected_lineage_keys:
            raise ValueError(
                "family_lineages must contain one shared-external morphology "
                "lineage and canonical folds 0-4 plus final for ictal/evolution"
            )
        if any(
            lineage.extractor.split_manifest_sha256 != self.split_manifest_sha256
            for lineage in self.family_lineages
        ):
            raise ValueError("A family lineage uses a different target split manifest")
        for family in active:
            producer_pairs = {
                (
                    lineage.extractor.checkpoint_sha256,
                    lineage.extractor.scaler_sha256,
                )
                for lineage in self.family_lineages
                if lineage.concept_family == family
            }
            expected_pair_count = (
                1
                if _FAMILY_SELECTION_POLICY[family] == "shared_external"
                else len(_SELECTION_KEYS)
            )
            if len(producer_pairs) != expected_pair_count:
                raise ValueError(
                    f"{family} producer count disagrees with its selection policy"
                )

        records = tuple(sorted(self.event_records, key=lambda row: row[0]))
        if records != self.event_records or not records:
            raise ValueError("event_records must be non-empty and canonically ordered")
        event_ids: set[str] = set()
        temporal = tuple(
            sorted(self.event_temporal_provenance, key=lambda item: item.event_id)
        )
        if temporal != self.event_temporal_provenance or any(
            not isinstance(item, EventTemporalProvenanceReceipt)
            for item in temporal
        ):
            raise ValueError(
                "event_temporal_provenance must be typed and canonically ordered"
            )
        if tuple(item.event_id for item in temporal) != tuple(
            row[0] for row in records
        ):
            raise ValueError(
                "Every authorized event requires exact temporal provenance"
            )
        roster_by_split: dict[str, set[str]] = {split: set() for split in _SOURCE_SPLITS}
        lineage_by_key = {
            (lineage.concept_family, lineage.oof_fold): lineage
            for lineage in self.family_lineages
        }
        fold_by_patient = dict(folds)
        for event_id, record_sha, patient_id, model_split in records:
            event_id = str(event_id).strip()
            patient_id = normalize_patient_id(patient_id)
            if not event_id or event_id in event_ids:
                raise ValueError("event_records contain duplicate or empty event IDs")
            event_ids.add(event_id)
            _require_sha256(record_sha, field_name="event_record_sha256")
            if model_split not in _SOURCE_SPLITS:
                raise ValueError("event_records contain an unsupported model split")
            roster_by_split[model_split].add(patient_id)
            key = fold_by_patient[patient_id] if model_split == "source_train" else None
            for family in active:
                family_key = (
                    None
                    if _FAMILY_SELECTION_POLICY[family] == "shared_external"
                    else key
                )
                extractor = lineage_by_key[(family, family_key)].extractor
                if patient_id in extractor.training_target_patient_ids:
                    raise ValueError(
                        f"{family} fold/final producer includes event patient {patient_id} in fit targets"
                    )
                if patient_id not in extractor.held_out_target_patient_ids:
                    raise ValueError(
                        f"{family} fold/final producer does not explicitly hold out event patient {patient_id}"
                    )
        expected_rosters = {
            "source_train": train_ids,
            "source_dev": set(dev),
            "source_eval": set(eval_ids),
        }
        if roster_by_split != expected_rosters:
            raise ValueError("Authorization event roster does not exactly cover source splits")
        if not self.annotation_coverage_forbidden:
            raise ValueError("Annotation coverage is forbidden from reasoner caches")
        if not self.cache_publication_authorized:
            raise ValueError("An authorization capability must explicitly authorize publication")
        if self.schema_version != OOF_EVIDENCE_AUTHORIZATION_SCHEMA:
            raise ValueError("Unsupported OOF evidence authorization schema")

    @property
    def authorization_sha256(self) -> str:
        return _canonical_sha256(asdict(self))

    def lineage(
        self, concept_family: str, oof_fold: int | None
    ) -> FamilyProducerAuthorization:
        matches = tuple(
            lineage
            for lineage in self.family_lineages
            if lineage.concept_family == concept_family
            and lineage.oof_fold == oof_fold
        )
        if len(matches) != 1:
            raise KeyError(f"No unique authorization for {(concept_family, oof_fold)}")
        return matches[0]

    def lineage_key_for_event(
        self, concept_family: str, event_oof_fold: int | None
    ) -> int | None:
        """Return the authorized routing key without inventing morphology folds."""

        if concept_family not in self.active_families:
            raise KeyError(f"Inactive evidence family: {concept_family!r}")
        policy = dict(self.family_selection_policies)[concept_family]
        if policy == "shared_external":
            return None
        if event_oof_fold is not None and (
            isinstance(event_oof_fold, bool)
            or not isinstance(event_oof_fold, int)
            or event_oof_fold not in range(5)
        ):
            raise ValueError("event_oof_fold must be None or an integer in [0,4]")
        return event_oof_fold

    def event_record(self, event_id: object) -> tuple[str, str, str, str]:
        key = str(event_id).strip()
        matches = tuple(row for row in self.event_records if row[0] == key)
        if len(matches) != 1:
            raise KeyError(f"Unknown authorized event: {key}")
        return matches[0]


def build_oof_evidence_authorization(
    references: DeepSOZReferenceRegistry,
    event_registry: EventInputRegistry,
    *,
    active_families: Sequence[str],
    family_lineages: Sequence[FamilyProducerAuthorization],
    event_temporal_provenance: Sequence[EventTemporalProvenanceReceipt],
) -> OOFEvidenceAuthorization:
    """Build the exact authorization after strict upstream artifacts load."""

    if not isinstance(references, DeepSOZReferenceRegistry):
        raise TypeError("references must be a DeepSOZReferenceRegistry")
    if not isinstance(event_registry, EventInputRegistry):
        raise TypeError("event_registry must be an EventInputRegistry")
    train_folds = tuple(
        sorted(
            (
                patient_id,
                references.get(patient_id).concept_oof_fold,
            )
            for patient_id in event_registry.patient_ids_for_split("source_train")
        )
    )
    if any(fold is None for _, fold in train_folds):
        raise ValueError("Every registered source-train patient requires an OOF fold")
    events = tuple(
        (
            record.event_id,
            record.record_sha256,
            record.patient_id,
            record.model_split,
        )
        for record in event_registry
    )
    return OOFEvidenceAuthorization(
        event_registry_sha256=event_registry.manifest_sha256,
        split_manifest_sha256=event_registry.split_manifest_sha256,
        source_train_patient_folds=tuple(
            (patient_id, int(fold)) for patient_id, fold in train_folds
        ),
        source_dev_patient_ids=event_registry.patient_ids_for_split("source_dev"),
        source_eval_patient_ids=event_registry.patient_ids_for_split("source_eval"),
        event_records=tuple(sorted(events, key=lambda row: row[0])),
        active_families=tuple(active_families),
        family_selection_policies=tuple(
            (family, _FAMILY_SELECTION_POLICY[family])
            for family in active_families
        ),
        family_lineages=tuple(family_lineages),
        event_temporal_provenance=tuple(
            sorted(event_temporal_provenance, key=lambda item: item.event_id)
        ),
    )


def build_event_temporal_provenance_receipts(
    event_ids: Sequence[object],
    phase_masks: OffsetAwarePhaseMasks,
    *,
    global_timeline_receipt_sha256s: Sequence[object],
) -> tuple[EventTemporalProvenanceReceipt, ...]:
    """Project verified timeline-derived masks into non-learned cache receipts."""

    if not isinstance(phase_masks, OffsetAwarePhaseMasks):
        raise TypeError("phase_masks must be OffsetAwarePhaseMasks")
    ids = tuple(str(value).strip() for value in event_ids)
    timeline_shas = tuple(global_timeline_receipt_sha256s)
    if (
        not ids
        or len(ids) != phase_masks.ictal_phase_mask.shape[0]
        or len(ids) != len(timeline_shas)
        or len(set(ids)) != len(ids)
        or any(not value for value in ids)
    ):
        raise ValueError("Event/timeline rosters must exactly align with phase masks")
    receipts = []
    for index, event_id in enumerate(ids):
        receipts.append(
            EventTemporalProvenanceReceipt(
                event_id=event_id,
                global_timeline_receipt_sha256=_require_sha256(
                    timeline_shas[index],
                    field_name="global_timeline_receipt_sha256",
                ),
                temporal_phase_policy_sha256=phase_masks.policy_sha256,
                ictal_phase_mask_sha256=ictal_phase_mask_sha256(
                    phase_masks.ictal_phase_mask[index]
                ),
                offset_trustworthy=bool(
                    phase_masks.offset_trustworthy[index].item()
                ),
                seizure_duration_sec=float(
                    phase_masks.seizure_duration_sec[index].item()
                ),
                previous_timeline_trustworthy=bool(
                    phase_masks.previous_timeline_trustworthy[index].item()
                ),
                has_previous_seizure=bool(
                    phase_masks.has_previous_seizure[index].item()
                ),
                previous_seizure_overlap=bool(
                    phase_masks.previous_seizure_overlap[index].item()
                ),
                previous_seizure_gap_sec=float(
                    phase_masks.previous_seizure_gap_sec[index].item()
                ),
            )
        )
    return tuple(receipts)


@dataclass(frozen=True)
class AuthorizedEvidenceEvent:
    """Capability issued only after tensor, event, and every family match."""

    event: EvidenceEvent
    authorization_sha256: str
    model_split: str
    oof_fold: int | None
    active_families: tuple[str, ...]
    _issuer_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer_token is not _AUTHORIZATION_ISSUER:
            raise PermissionError("Authorized evidence can only be issued by validator")
        object.__setattr__(
            self,
            "authorization_sha256",
            _require_sha256(
                self.authorization_sha256, field_name="authorization_sha256"
            ),
        )
        object.__setattr__(self, "_issuer_token", None)


def _validate_absent_family(
    event: EvidenceEvent, concept_family: str
) -> None:
    evidence = event.evidence
    if concept_family == MORPHOLOGY_FAMILY:
        if (
            evidence.morphology_mask is None
            or evidence.morphology_context_mask is None
        ):
            raise RuntimeError("Morphology masks were not initialized")
        if (
            evidence.morphology_mask.any()
            or evidence.morphology_context_mask.any()
            or torch.any(
            evidence.edge[..., :N_MORPHOLOGY_FEATURES] != 0
            )
        ):
            raise ValueError(
                "Absent morphology must have local/context masks false and "
                "finite zero fill"
            )
    elif concept_family == ICTAL_FAMILY:
        if evidence.ictal_mask is None:
            raise RuntimeError("Ictal mask was not initialized")
        if evidence.ictal_mask.any() or torch.any(
            evidence.edge[..., N_MORPHOLOGY_FEATURES:] != 0
        ):
            raise ValueError("Absent ictal evidence must be masked and finite zero-filled")
    elif concept_family == EVOLUTION_FAMILY:
        if evidence.node_mask.any() or torch.any(evidence.node != 0):
            raise ValueError("Absent evolution must be masked and finite zero-filled")
    else:  # pragma: no cover - callers iterate the closed family vocabulary
        raise ValueError(f"Unsupported evidence family: {concept_family}")


def authorize_evidence_event(
    event: EvidenceEvent,
    references: DeepSOZReferenceRegistry,
    event_registry: EventInputRegistry,
    authorization: OOFEvidenceAuthorization,
) -> AuthorizedEvidenceEvent:
    """Validate one cache against independently rebuilt M/I/V authority."""

    if not isinstance(event, EvidenceEvent):
        raise TypeError("event must be an EvidenceEvent")
    if not isinstance(authorization, OOFEvidenceAuthorization):
        raise TypeError("authorization must be an OOFEvidenceAuthorization")
    if authorization.event_registry_sha256 != event_registry.manifest_sha256:
        raise ValueError("Authorization belongs to a different event registry")
    if authorization.split_manifest_sha256 != event_registry.split_manifest_sha256:
        raise ValueError("Authorization belongs to a different target split manifest")
    record = event_registry.get(event.event_id)
    expected_event = authorization.event_record(event.event_id)
    if expected_event != (
        record.event_id,
        record.record_sha256,
        record.patient_id,
        record.model_split,
    ):
        raise ValueError("Authorized event identity/record/split changed")
    reference = references.get(record.patient_id)
    if reference.model_split != record.model_split:
        raise ValueError("Reference and event split disagree")
    receipt = event.cache_receipt
    if receipt.authorization_sha256 != authorization.authorization_sha256:
        raise ValueError("Cache does not bind the expected OOF authorization")
    if receipt.evidence_semantics_sha256 != authorization.evidence_semantics_sha256:
        raise ValueError("Cache typed evidence semantics are unauthorized")
    expected_temporal = next(
        item
        for item in authorization.event_temporal_provenance
        if item.event_id == event.event_id
    )
    if receipt.temporal_provenance != expected_temporal:
        raise ValueError("Cache temporal provenance is missing or unauthorized")
    if expected_temporal.ictal_phase_mask_sha256 != ictal_phase_mask_sha256(
        event.evidence.ictal_phase_mask[0]
    ):
        raise ValueError("Cached ictal phase mask disagrees with timeline provenance")
    if receipt.event_registry_sha256 != event_registry.manifest_sha256:
        raise ValueError("Cache belongs to a different event registry")
    if receipt.event_record_sha256 != record.record_sha256:
        raise ValueError("Cache event-record lineage changed")
    if receipt.evidence_sha256 != evidence_batch_sha256(event.evidence):
        raise ValueError("Cache evidence content does not match its receipt")
    event.evidence.validate()
    if event.evidence.batch_size != 1 or event.evidence.n_tiles != N_TIME_TILES:
        raise ValueError("Formal evidence requires one fixed 15-tile event")
    if event.evidence.node.requires_grad or event.evidence.edge.requires_grad:
        raise ValueError("Formal reasoner evidence must be detached")

    receipt_families = tuple(extractor.concept_family for extractor in receipt.extractors)
    if set(receipt_families) != set(authorization.active_families) or len(
        receipt_families
    ) != len(authorization.active_families):
        raise ValueError("Cache extractor set does not equal the active-family set")
    key = reference.concept_oof_fold if record.model_split == "source_train" else None
    for family in authorization.active_families:
        family_key = authorization.lineage_key_for_event(family, key)
        expected = authorization.lineage(family, family_key).extractor
        actual = receipt.extractor(family)
        if actual != expected:
            raise ValueError(
                f"Cache {family} fold/checkpoint/scaler/fit-held lineage is unauthorized"
            )
        if record.patient_id in actual.training_target_patient_ids:
            raise ValueError(f"Cache {family} producer trained on its target patient")
        if record.patient_id not in actual.held_out_target_patient_ids:
            raise ValueError(f"Cache {family} producer did not hold out target patient")
    for family in EVIDENCE_FAMILIES:
        if family not in authorization.active_families:
            _validate_absent_family(event, family)
    return AuthorizedEvidenceEvent(
        event=event,
        authorization_sha256=authorization.authorization_sha256,
        model_split=record.model_split,
        oof_fold=key,
        active_families=authorization.active_families,
        _issuer_token=_AUTHORIZATION_ISSUER,
    )


def load_authorized_evidence_cache(
    path: str,
    references: DeepSOZReferenceRegistry,
    event_registry: EventInputRegistry,
    authorization: OOFEvidenceAuthorization,
    *,
    expected_manifest_sha256: str | None = None,
) -> AuthorizedEvidenceEvent:
    """Strictly load and then independently authorize one cache bundle."""

    from .evidence_io import load_evidence_cache

    event = load_evidence_cache(
        path, expected_manifest_sha256=expected_manifest_sha256
    )
    return authorize_evidence_event(
        event, references, event_registry, authorization
    )


class AuthorizedPatientBagDataset(PatientBagDataset):
    """Complete split dataset whose every event carries one authorization."""

    def __init__(
        self,
        events: Sequence[AuthorizedEvidenceEvent],
        references: DeepSOZReferenceRegistry,
        event_registry: EventInputRegistry,
        authorization: OOFEvidenceAuthorization,
        *,
        expected_model_split: str,
    ) -> None:
        if not events:
            raise ValueError("Authorized patient-bag dataset cannot be empty")
        if any(
            not isinstance(event, AuthorizedEvidenceEvent) for event in events
        ):
            raise TypeError("Formal patient bags require AuthorizedEvidenceEvent")
        for event in events:
            if event.authorization_sha256 != authorization.authorization_sha256:
                raise ValueError("Authorized events use different OOF plans")
            if event.model_split != expected_model_split:
                raise ValueError("Authorized event belongs to a different model split")
            # Revalidate rather than trusting a stale in-memory capability after
            # callers may have mutated an underlying tensor through aliasing.
            authorize_evidence_event(
                event.event, references, event_registry, authorization
            )
        super().__init__(
            [event.event for event in events],
            references,
            event_registry,
            expected_model_split=expected_model_split,
        )
        self._evidence_authorization = authorization

    @property
    def evidence_authorization_sha256(self) -> str:
        return self._evidence_authorization.authorization_sha256

    @property
    def active_families(self) -> tuple[str, ...]:
        return self._evidence_authorization.active_families


__all__ = [
    "EVIDENCE_FAMILIES",
    "EVOLUTION_FAMILY",
    "FAMILY_FEATURE_NAMES",
    "FAMILY_FEATURE_SCHEMA_SHA256",
    "FAMILY_PRODUCER_AUTHORIZATION_SCHEMA",
    "ICTAL_FAMILY",
    "MORPHOLOGY_FAMILY",
    "NATIVE_CLASS_SUPPORT_SCHEMA",
    "OOF_EVIDENCE_AUTHORIZATION_SCHEMA",
    "AuthorizedEvidenceEvent",
    "AuthorizedPatientBagDataset",
    "FamilyProducerAuthorization",
    "NativeClassSupportReceipt",
    "OOFEvidenceAuthorization",
    "authorize_evidence_event",
    "build_event_temporal_provenance_receipts",
    "build_oof_evidence_authorization",
    "load_authorized_evidence_cache",
    "source_roster_sha256",
]
