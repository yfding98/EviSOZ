"""Leakage-safe planning for DeepSOZ OOF evidence materialization.

The current evidence-cache v1 format cannot losslessly describe an absent
morphology branch or validate the fit lineage of directly computed evolution
descriptors.  This module therefore freezes only the selection/provenance
layer before any real cache producer is allowed to run.  A saved plan is not
authority to publish formal evidence caches: verified fold-training manifests,
a strict evolution-scaler artifact loader, signal/preprocessing lineage, and
the target-v2 join gate are still deliberately absent.  It does not accept raw
EEG, token tensors, targets, or guessed filesystem paths.

The locked vertical slice is:

* morphology: explicitly absent, finite zero fill, all-false family mask;
* ictal involvement: the patient-specific OOF checkpoint for source-train and
  the final train-only checkpoint for source-dev/source-eval; and
* temporal evolution: deterministic descriptors scaled by a fit-receipted
  patient-excluded robust scaler selected on the same OOF/final key.

Plans are target-free at event level.  Patient identity remains owned by the
supplied :class:`~soz.data.provenance.EventInputRegistry` and is used only
during fail-closed plan construction/validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import torch

from .concept_checkpoint import LoadedIctalConceptCheckpoint
from .concept_oof import IctalConceptOOFPlan, IctalConceptOOFProtocol
from .data.deepsoz import (
    ALLOWED_MODEL_SPLITS,
    DeepSOZReferenceRegistry,
)
from .data.overlap import normalize_public_patient_key
from .data.provenance import (
    ConceptExtractorReceipt,
    EventInputRegistry,
)
from .evidence import EvidenceBatch
from .evolution import (
    EVOLUTION_FEATURE_SCHEMA_SHA256,
    PatientBalancedRobustScaler,
    patient_roster_sha256 as evolution_patient_roster_sha256,
)
from .geometry import N_MORPHOLOGY_FEATURES


OOF_EVIDENCE_MATERIALIZATION_PLAN_SCHEMA = (
    "soz_deepsoz_oof_evidence_materialization_plan_v1"
)
OOF_EVIDENCE_MATERIALIZATION_ARTIFACT_SCHEMA = (
    "soz_deepsoz_oof_evidence_materialization_artifact_v1"
)
OOF_EVIDENCE_MATERIALIZATION_FILENAME = "oof_evidence_materialization_plan.json"

MORPHOLOGY_ABSENCE_POLICY = "absent_finite_zero_fill_all_false_mask"
ICTAL_MATERIALIZATION_POLICY = "learned_oof_probability_four_second_mean_max"
EVOLUTION_MATERIALIZATION_POLICY = (
    "deterministic_computed_descriptors_patient_excluded_robust_scaler"
)
REASONER_INPUT_POLICY = "evidence_batch_only_no_raw_eeg_or_foundation_tokens"
CACHE_AUTHORIZATION_POLICY = "selection_provenance_plan_only_not_cache_authority"
CACHE_AUTHORIZATION_BLOCKERS = (
    "verified_preflight_fold_training_manifest_and_run_binding",
    "verified_evolution_scaler_artifact_loader_and_exact_fit_roster",
    "verified_event_signal_preprocessing_window_lineage",
    "verified_deepsoz_target_v2_loader_and_join",
)
BRANCH_PRESENCE = (
    ("morphology", False),
    ("ictal_involvement", True),
    ("temporal_evolution", True),
)

_SELECTION_KEYS = (0, 1, 2, 3, 4, None)
_SOURCE_SPLITS = ("source_train", "source_dev", "source_eval")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_FIELDS = frozenset({"schema_version", "plan_sha256", "plan"})
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Materialization provenance is not canonical JSON data"
        ) from exc
    return (text + "\n").encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return text


def _normalize_selection_key(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(5):
        raise ValueError(f"{field} must be None or an integer in [0,4]")
    return value


def _selection_key_order(value: int | None) -> int:
    return 5 if value is None else value


def _normalized_public_roster(values: Sequence[object]) -> tuple[str, ...]:
    roster = tuple(sorted(normalize_public_patient_key(value) for value in values))
    if not roster or len(set(roster)) != len(roster):
        raise ValueError("Public scaler-fit roster must be non-empty and unique")
    return roster


@dataclass(frozen=True)
class IctalCheckpointLineage:
    """Closed hash/roster view of one strictly loaded ictal checkpoint."""

    oof_fold: int | None
    checkpoint_manifest_sha256: str
    checkpoint_weights_sha256: str
    foundation_feature_receipt_sha256: str
    foundation_checkpoint_sha256: str
    tusz_annotation_sha256: str
    tusz_manifest_sha256: str
    split_manifest_sha256: str
    oof_plan_receipt_sha256: str
    oof_protocol_receipt_sha256: str
    training_run_receipt_sha256: str
    checkpoint_scaler_sha256: str
    training_target_patient_ids: tuple[str, ...]
    held_out_target_patient_ids: tuple[str, ...]
    training_target_roster_sha256: str
    held_out_target_roster_sha256: str
    schema_version: str = "soz_ictal_checkpoint_selection_lineage_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "oof_fold",
            _normalize_selection_key(self.oof_fold, field="oof_fold"),
        )
        for field in (
            "checkpoint_manifest_sha256",
            "checkpoint_weights_sha256",
            "foundation_feature_receipt_sha256",
            "foundation_checkpoint_sha256",
            "tusz_annotation_sha256",
            "tusz_manifest_sha256",
            "split_manifest_sha256",
            "oof_plan_receipt_sha256",
            "oof_protocol_receipt_sha256",
            "training_run_receipt_sha256",
            "checkpoint_scaler_sha256",
            "training_target_roster_sha256",
            "held_out_target_roster_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )

        # Reuse the cache-facing receipt validator so checkpoint and evidence
        # provenance cannot silently diverge in roster normalization.
        extractor = ConceptExtractorReceipt(
            concept_family="ictal_involvement",
            checkpoint_sha256=self.checkpoint_manifest_sha256,
            scaler_sha256=self.checkpoint_scaler_sha256,
            split_manifest_sha256=self.split_manifest_sha256,
            oof_fold=self.oof_fold,
            training_target_patient_ids=self.training_target_patient_ids,
            held_out_target_patient_ids=self.held_out_target_patient_ids,
            training_target_roster_sha256=self.training_target_roster_sha256,
            held_out_target_roster_sha256=self.held_out_target_roster_sha256,
        )
        object.__setattr__(
            self, "training_target_patient_ids", extractor.training_target_patient_ids
        )
        object.__setattr__(
            self, "held_out_target_patient_ids", extractor.held_out_target_patient_ids
        )
        object.__setattr__(
            self,
            "training_target_roster_sha256",
            extractor.training_target_roster_sha256,
        )
        object.__setattr__(
            self,
            "held_out_target_roster_sha256",
            extractor.held_out_target_roster_sha256,
        )
        if self.schema_version != "soz_ictal_checkpoint_selection_lineage_v1":
            raise ValueError("Unsupported ictal checkpoint selection schema")

    @property
    def lineage_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def ictal_checkpoint_lineage_from_loaded(
    checkpoint: LoadedIctalConceptCheckpoint,
) -> IctalCheckpointLineage:
    """Project a safe loaded checkpoint into path-free selection provenance."""

    if not isinstance(checkpoint, LoadedIctalConceptCheckpoint):
        raise TypeError("checkpoint must be a LoadedIctalConceptCheckpoint")
    metadata = checkpoint.metadata
    return IctalCheckpointLineage(
        oof_fold=metadata["oof_fold"],
        checkpoint_manifest_sha256=checkpoint.manifest_sha256,
        checkpoint_weights_sha256=metadata["checkpoint_sha256"],
        foundation_feature_receipt_sha256=metadata[
            "foundation_feature_receipt_sha256"
        ],
        foundation_checkpoint_sha256=metadata["foundation_checkpoint_sha256"],
        tusz_annotation_sha256=metadata["tusz_annotation_sha256"],
        tusz_manifest_sha256=metadata["tusz_manifest_sha256"],
        split_manifest_sha256=metadata["split_manifest_sha256"],
        oof_plan_receipt_sha256=metadata["oof_plan_receipt_sha256"],
        oof_protocol_receipt_sha256=metadata["oof_protocol_receipt_sha256"],
        training_run_receipt_sha256=metadata["training_run_receipt_sha256"],
        checkpoint_scaler_sha256=metadata["scaler_sha256"],
        training_target_patient_ids=tuple(metadata["training_target_patient_ids"]),
        held_out_target_patient_ids=tuple(metadata["held_out_target_patient_ids"]),
        training_target_roster_sha256=metadata[
            "training_target_roster_sha256"
        ],
        held_out_target_roster_sha256=metadata["held_out_target_roster_sha256"],
    )


@dataclass(frozen=True)
class ComputedEvolutionScalerLineage:
    """Declared fit receipt and external hash for deterministic evolution.

    This is selection lineage, not a verified scaler artifact.  The current
    scaler API has no strict canonical loader, so ``scaler_artifact_sha256``
    and ``fit_manifest_sha256`` cannot yet be independently reloaded and
    cross-checked here.  Formal cache publication remains blocked until that
    loader exists and proves the exact preflight-eligible fit roster.
    """

    oof_fold: int | None
    scaler_artifact_sha256: str
    scaler_receipt_sha256: str
    fit_manifest_sha256: str
    split_manifest_sha256: str
    fit_split_sha256: str
    feature_schema_sha256: str
    fit_patient_public_keys: tuple[str, ...]
    fit_patient_roster_sha256: str
    patient_count: int
    clip: float
    producer: str = "deterministic_computed_temporal_evolution"
    schema_version: str = "soz_computed_evolution_scaler_selection_lineage_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "oof_fold",
            _normalize_selection_key(self.oof_fold, field="oof_fold"),
        )
        for field in (
            "scaler_artifact_sha256",
            "scaler_receipt_sha256",
            "fit_manifest_sha256",
            "split_manifest_sha256",
            "fit_split_sha256",
            "feature_schema_sha256",
            "fit_patient_roster_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.feature_schema_sha256 != EVOLUTION_FEATURE_SCHEMA_SHA256:
            raise ValueError("Computed-evolution feature schema SHA mismatch")
        roster = _normalized_public_roster(self.fit_patient_public_keys)
        object.__setattr__(self, "fit_patient_public_keys", roster)
        if self.fit_patient_roster_sha256 != evolution_patient_roster_sha256(roster):
            raise ValueError("Scaler fit-patient roster SHA does not match its roster")
        if (
            isinstance(self.patient_count, bool)
            or not isinstance(self.patient_count, int)
            or self.patient_count != len(roster)
        ):
            raise ValueError("Scaler patient_count must equal its exact fit roster")
        if not math.isfinite(float(self.clip)) or float(self.clip) <= 0:
            raise ValueError("Scaler clip must be finite and positive")
        object.__setattr__(self, "clip", float(self.clip))
        if self.producer != "deterministic_computed_temporal_evolution":
            raise ValueError("Learned evolution cannot use the computed-V lineage")
        if (
            self.schema_version
            != "soz_computed_evolution_scaler_selection_lineage_v1"
        ):
            raise ValueError("Unsupported computed-evolution scaler selection schema")

    @property
    def lineage_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def build_computed_evolution_scaler_lineage(
    scaler: PatientBalancedRobustScaler,
    *,
    oof_fold: int | None,
    scaler_artifact_sha256: str,
    fit_manifest_sha256: str,
    fit_patient_public_keys: Sequence[object],
) -> ComputedEvolutionScalerLineage:
    """Bind a scaler object to its exact public-patient fit manifest."""

    if not isinstance(scaler, PatientBalancedRobustScaler):
        raise TypeError("scaler must be a PatientBalancedRobustScaler")
    roster = _normalized_public_roster(fit_patient_public_keys)
    receipt = scaler.receipt
    if receipt.patient_roster_sha256 != evolution_patient_roster_sha256(roster):
        raise ValueError("Scaler receipt was fitted on a different patient roster")
    if receipt.patient_count != len(roster):
        raise ValueError("Scaler receipt patient_count disagrees with fit roster")
    return ComputedEvolutionScalerLineage(
        oof_fold=oof_fold,
        scaler_artifact_sha256=scaler_artifact_sha256,
        scaler_receipt_sha256=_canonical_sha256(asdict(receipt)),
        fit_manifest_sha256=fit_manifest_sha256,
        split_manifest_sha256=receipt.split_manifest_sha256,
        fit_split_sha256=receipt.fit_split_sha256,
        feature_schema_sha256=receipt.feature_schema_sha256,
        fit_patient_public_keys=roster,
        fit_patient_roster_sha256=receipt.patient_roster_sha256,
        patient_count=receipt.patient_count,
        clip=receipt.clip,
    )


@dataclass(frozen=True)
class OOFEvidenceEventSelection:
    """Target-free producer selection for one exact registered event."""

    event_id: str
    event_record_sha256: str
    model_split: str
    oof_fold: int | None
    ictal_checkpoint_manifest_sha256: str
    ictal_checkpoint_lineage_sha256: str
    evolution_scaler_artifact_sha256: str
    evolution_scaler_lineage_sha256: str
    morphology_present: bool = False
    ictal_involvement_present: bool = True
    temporal_evolution_present: bool = True
    morphology_mask_policy: str = "all_false"
    schema_version: str = "soz_oof_evidence_event_selection_v1"

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        if not event_id:
            raise ValueError("event_id cannot be empty")
        object.__setattr__(self, "event_id", event_id)
        if self.model_split not in ALLOWED_MODEL_SPLITS:
            raise ValueError("Event selection requires source train/dev/eval")
        fold = _normalize_selection_key(self.oof_fold, field="oof_fold")
        if self.model_split == "source_train" and fold is None:
            raise ValueError("source_train event selection requires an OOF fold")
        if self.model_split != "source_train" and fold is not None:
            raise ValueError("source_dev/source_eval must use final selection")
        object.__setattr__(self, "oof_fold", fold)
        for field in (
            "event_record_sha256",
            "ictal_checkpoint_manifest_sha256",
            "ictal_checkpoint_lineage_sha256",
            "evolution_scaler_artifact_sha256",
            "evolution_scaler_lineage_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if (
            self.morphology_present
            or not self.ictal_involvement_present
            or not self.temporal_evolution_present
        ):
            raise ValueError(
                "Vertical slice branch presence must be M=absent, I/V=present"
            )
        if self.morphology_mask_policy != "all_false":
            raise ValueError("Absent morphology requires an all-false family mask")
        if self.schema_version != "soz_oof_evidence_event_selection_v1":
            raise ValueError("Unsupported OOF evidence event-selection schema")


@dataclass(frozen=True)
class OOFEvidenceMaterializationPlan:
    """Complete event roster and producer *selections* for the vertical slice.

    The artifact is intentionally non-authorizing.  It may be used to audit
    which fold/final producer an event would use and to run the local semantic
    evidence gate, but it must not be treated as permission to publish a
    production evidence cache or train the SOZ reasoner.
    """

    event_registry_sha256: str
    split_manifest_sha256: str
    oof_protocol_receipt_sha256: str
    public_ledger_build_sha256: str
    source_train_roster_sha256: str
    source_dev_roster_sha256: str
    source_eval_roster_sha256: str
    event_roster_sha256: str
    event_count: int
    split_event_counts: tuple[tuple[str, int], ...]
    ictal_checkpoints: tuple[IctalCheckpointLineage, ...]
    evolution_scalers: tuple[ComputedEvolutionScalerLineage, ...]
    events: tuple[OOFEvidenceEventSelection, ...]
    branch_presence: tuple[tuple[str, bool], ...] = BRANCH_PRESENCE
    morphology_absence_policy: str = MORPHOLOGY_ABSENCE_POLICY
    ictal_materialization_policy: str = ICTAL_MATERIALIZATION_POLICY
    evolution_materialization_policy: str = EVOLUTION_MATERIALIZATION_POLICY
    reasoner_input_policy: str = REASONER_INPUT_POLICY
    cache_publication_authorized: bool = False
    cache_authorization_policy: str = CACHE_AUTHORIZATION_POLICY
    cache_authorization_blockers: tuple[str, ...] = CACHE_AUTHORIZATION_BLOCKERS
    schema_version: str = OOF_EVIDENCE_MATERIALIZATION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "event_registry_sha256",
            "split_manifest_sha256",
            "oof_protocol_receipt_sha256",
            "public_ledger_build_sha256",
            "source_train_roster_sha256",
            "source_dev_roster_sha256",
            "source_eval_roster_sha256",
            "event_roster_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.branch_presence != BRANCH_PRESENCE:
            raise ValueError("Materialization branch presence policy cannot be changed")
        if self.morphology_absence_policy != MORPHOLOGY_ABSENCE_POLICY:
            raise ValueError("Morphology absence policy cannot be changed")
        if self.ictal_materialization_policy != ICTAL_MATERIALIZATION_POLICY:
            raise ValueError("Ictal materialization policy cannot be changed")
        if self.evolution_materialization_policy != EVOLUTION_MATERIALIZATION_POLICY:
            raise ValueError("Evolution materialization policy cannot be changed")
        if self.reasoner_input_policy != REASONER_INPUT_POLICY:
            raise ValueError("Reasoner input firewall policy cannot be changed")
        if self.cache_publication_authorized:
            raise ValueError(
                "A selection/provenance plan cannot authorize evidence caches"
            )
        if self.cache_authorization_policy != CACHE_AUTHORIZATION_POLICY:
            raise ValueError("Cache authorization policy cannot be changed")
        if self.cache_authorization_blockers != CACHE_AUTHORIZATION_BLOCKERS:
            raise ValueError("Cache authorization blockers cannot be changed")
        if self.schema_version != OOF_EVIDENCE_MATERIALIZATION_PLAN_SCHEMA:
            raise ValueError("Unsupported OOF evidence materialization plan schema")

        checkpoint_keys = tuple(item.oof_fold for item in self.ictal_checkpoints)
        scaler_keys = tuple(item.oof_fold for item in self.evolution_scalers)
        if checkpoint_keys != _SELECTION_KEYS or scaler_keys != _SELECTION_KEYS:
            raise ValueError("Plan must bind folds 0-4 and one final checkpoint/scaler")
        checkpoint_by_key = {item.oof_fold: item for item in self.ictal_checkpoints}
        scaler_by_key = {item.oof_fold: item for item in self.evolution_scalers}

        # Fold-specific heads and manifests are expected to differ, but the
        # representation and annotation semantics must not.  Without this
        # global invariant, an apparent OOF comparison could silently combine
        # six different feature extractors or label policies.
        shared_checkpoint_fields = (
            "foundation_feature_receipt_sha256",
            "foundation_checkpoint_sha256",
            "tusz_annotation_sha256",
            "split_manifest_sha256",
            "oof_protocol_receipt_sha256",
        )
        for field in shared_checkpoint_fields:
            values = {getattr(item, field) for item in self.ictal_checkpoints}
            if len(values) != 1:
                raise ValueError(
                    "All ictal checkpoints must share one "
                    f"{field.replace('_', ' ')}"
                )
        if any(
            item.split_manifest_sha256 != self.split_manifest_sha256
            for item in self.ictal_checkpoints
        ):
            raise ValueError("Ictal checkpoint split lineage disagrees with plan")
        if any(
            item.oof_protocol_receipt_sha256
            != self.oof_protocol_receipt_sha256
            for item in self.ictal_checkpoints
        ):
            raise ValueError("Ictal checkpoint protocol lineage disagrees with plan")

        if tuple(sorted(self.events, key=lambda item: item.event_id)) != self.events:
            raise ValueError("Plan events must be canonically ordered")
        event_ids = tuple(event.event_id for event in self.events)
        if not event_ids or len(set(event_ids)) != len(event_ids):
            raise ValueError("Plan events must be non-empty and unique")
        if self.event_count != len(self.events):
            raise ValueError("event_count disagrees with exact event roster")
        expected_roster_sha = _canonical_sha256(
            tuple((event.event_id, event.event_record_sha256) for event in self.events)
        )
        if self.event_roster_sha256 != expected_roster_sha:
            raise ValueError("event_roster_sha256 disagrees with exact event roster")
        expected_counts = tuple(
            (split, sum(event.model_split == split for event in self.events))
            for split in _SOURCE_SPLITS
        )
        if self.split_event_counts != expected_counts or any(
            count < 1 for _, count in expected_counts
        ):
            raise ValueError("split_event_counts must exactly cover all three splits")

        for event in self.events:
            checkpoint = checkpoint_by_key[event.oof_fold]
            scaler = scaler_by_key[event.oof_fold]
            if (
                event.ictal_checkpoint_manifest_sha256
                != checkpoint.checkpoint_manifest_sha256
                or event.ictal_checkpoint_lineage_sha256
                != checkpoint.lineage_sha256
            ):
                raise ValueError("Event references the wrong ictal checkpoint lineage")
            if (
                event.evolution_scaler_artifact_sha256
                != scaler.scaler_artifact_sha256
                or event.evolution_scaler_lineage_sha256 != scaler.lineage_sha256
            ):
                raise ValueError("Event references the wrong evolution scaler lineage")

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(asdict(self))

    def event(self, event_id: object) -> OOFEvidenceEventSelection:
        key = str(event_id).strip()
        matches = tuple(event for event in self.events if event.event_id == key)
        if len(matches) != 1:
            raise KeyError(f"Unknown materialization-plan event: {key}")
        return matches[0]


def _exact_resource_mapping(
    values: Mapping[int | None, object], *, field: str
) -> dict[int | None, object]:
    keys = tuple(values.keys())
    if any(isinstance(key, bool) for key in keys) or set(keys) != set(_SELECTION_KEYS):
        raise ValueError(f"{field} must contain exactly fold keys 0-4 and None")
    return {key: values[key] for key in _SELECTION_KEYS}


def _protocol_plan(
    protocol: IctalConceptOOFProtocol, key: int | None
) -> IctalConceptOOFPlan:
    return protocol.final_plan if key is None else protocol.fold_plans[key]


def _validate_checkpoint_against_protocol(
    lineage: IctalCheckpointLineage,
    protocol: IctalConceptOOFProtocol,
) -> None:
    plan = _protocol_plan(protocol, lineage.oof_fold)
    checks = {
        "split manifest": (
            lineage.split_manifest_sha256 == protocol.receipt.split_manifest_sha256
        ),
        "OOF plan receipt": (
            lineage.oof_plan_receipt_sha256 == plan.receipt.receipt_sha256
        ),
        "OOF protocol receipt": (
            lineage.oof_protocol_receipt_sha256 == protocol.receipt.receipt_sha256
        ),
        "training target roster": (
            lineage.training_target_patient_ids
            == plan.training_target_patient_ids
        ),
        "held-out target roster": (
            lineage.held_out_target_patient_ids == plan.held_out_target_patient_ids
        ),
        "training target roster SHA": (
            lineage.training_target_roster_sha256
            == plan.receipt.training_target_roster_sha256
        ),
        "held-out target roster SHA": (
            lineage.held_out_target_roster_sha256
            == plan.receipt.held_out_target_roster_sha256
        ),
    }
    failed = tuple(label for label, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Ictal checkpoint selection disagrees with OOF protocol: {failed}"
        )


def _validate_scaler_against_protocol(
    lineage: ComputedEvolutionScalerLineage,
    protocol: IctalConceptOOFProtocol,
) -> None:
    """Enforce the strongest scaler checks available before artifact IO exists.

    ``fit_patient_public_keys`` is required to be a non-empty subset of the
    fold-authorized TUSZ cohort, not an exact copy.  Exact equality would be a
    false guarantee because the OOF plan is defined before EDF/window
    preflight, while the eventual descriptor manifest may legitimately omit
    unusable records or patients.  Conversely, this caller-supplied subset is
    insufficient for formal materialization.  A future strict scaler loader
    must authenticate the preflighted descriptor manifest and derive the exact
    fit roster from it before cache publication can be enabled.
    """

    plan = _protocol_plan(protocol, lineage.oof_fold)
    if lineage.split_manifest_sha256 != protocol.receipt.split_manifest_sha256:
        raise ValueError("Evolution scaler uses a different split manifest")
    allowed_public = {
        record.patient_key for record in plan.training_cohort.allowed_records
    }
    fit_public = set(lineage.fit_patient_public_keys)
    leaked = tuple(
        sorted(fit_public & set(plan.held_out_public_patient_keys))
    )
    if leaked:
        raise ValueError(
            f"Evolution scaler fit roster contains held-out patients: {leaked}"
        )
    unauthorized = tuple(sorted(fit_public - allowed_public))
    if unauthorized:
        raise ValueError(
            "Evolution scaler fit roster contains patients outside the OOF-authorized "
            f"TUSZ cohort: {unauthorized}"
        )


def build_deepsoz_oof_evidence_materialization_plan(
    references: DeepSOZReferenceRegistry,
    event_registry: EventInputRegistry,
    oof_protocol: IctalConceptOOFProtocol,
    *,
    ictal_checkpoints: Mapping[int | None, LoadedIctalConceptCheckpoint],
    evolution_scalers: Mapping[int | None, ComputedEvolutionScalerLineage],
) -> OOFEvidenceMaterializationPlan:
    """Plan exact OOF/final I and computed-V selection for every event.

    The returned object is an auditable, non-authorizing selection plan.  It
    does not prove the declared scaler bytes, fold-training preflight bundle,
    processed signal window, or target-v2 join and therefore cannot authorize
    formal evidence cache publication.
    """

    if not isinstance(references, DeepSOZReferenceRegistry):
        raise TypeError("references must be a DeepSOZReferenceRegistry")
    if not isinstance(event_registry, EventInputRegistry):
        raise TypeError("event_registry must be an EventInputRegistry")
    if not isinstance(oof_protocol, IctalConceptOOFProtocol):
        raise TypeError("oof_protocol must be an IctalConceptOOFProtocol")
    if (
        event_registry.split_manifest_sha256
        != oof_protocol.receipt.split_manifest_sha256
    ):
        raise ValueError(
            "Event registry and OOF protocol use different split manifests"
        )

    checkpoint_inputs = _exact_resource_mapping(
        ictal_checkpoints, field="ictal_checkpoints"
    )
    scaler_inputs = _exact_resource_mapping(
        evolution_scalers, field="evolution_scalers"
    )
    checkpoint_by_key: dict[int | None, IctalCheckpointLineage] = {}
    scaler_by_key: dict[int | None, ComputedEvolutionScalerLineage] = {}
    for key in _SELECTION_KEYS:
        checkpoint = checkpoint_inputs[key]
        if not isinstance(checkpoint, LoadedIctalConceptCheckpoint):
            raise TypeError(
                "ictal_checkpoints values must be strictly loaded checkpoints"
            )
        checkpoint_lineage = ictal_checkpoint_lineage_from_loaded(checkpoint)
        if checkpoint_lineage.oof_fold != key:
            raise ValueError(
                "Ictal checkpoint mapping key disagrees with checkpoint OOF fold"
            )
        _validate_checkpoint_against_protocol(checkpoint_lineage, oof_protocol)
        checkpoint_by_key[key] = checkpoint_lineage

        scaler_lineage = scaler_inputs[key]
        if not isinstance(scaler_lineage, ComputedEvolutionScalerLineage):
            raise TypeError(
                "evolution_scalers values must be scaler lineage receipts"
            )
        if scaler_lineage.oof_fold != key:
            raise ValueError(
                "Evolution scaler mapping key disagrees with scaler OOF fold"
            )
        _validate_scaler_against_protocol(scaler_lineage, oof_protocol)
        scaler_by_key[key] = scaler_lineage

    for split in _SOURCE_SPLITS:
        expected = set(getattr(oof_protocol.receipt, f"{split}_patient_ids"))
        actual = set(event_registry.patient_ids_for_split(split))
        if actual != expected:
            raise ValueError(
                f"Event registry patient roster is incomplete for {split}; "
                f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
            )
    if not len(event_registry):
        raise ValueError("Evidence materialization requires registered events")

    target_to_public = dict(oof_protocol.receipt.target_public_crosswalk)
    event_selections: list[OOFEvidenceEventSelection] = []
    for record in event_registry:
        reference = references.get(record.patient_id)
        if not reference.eligible_for_localization:
            raise ValueError(
                "Event registry contains an ineligible localization patient"
            )
        if reference.model_split != record.model_split:
            raise ValueError("Event/reference model split mismatch")
        key = (
            reference.concept_oof_fold
            if record.model_split == "source_train"
            else None
        )
        key = _normalize_selection_key(key, field="event selection key")
        checkpoint = checkpoint_by_key[key]
        scaler = scaler_by_key[key]
        public_key = target_to_public[record.patient_id]
        if public_key in scaler.fit_patient_public_keys:
            raise ValueError(
                "Event patient appears in its computed-evolution scaler fit roster"
            )
        if record.patient_id in checkpoint.training_target_patient_ids:
            raise ValueError(
                "Event patient appears in its ictal checkpoint training roster"
            )
        if record.patient_id not in checkpoint.held_out_target_patient_ids:
            raise ValueError("Event patient is absent from ictal held-out provenance")
        event_selections.append(
            OOFEvidenceEventSelection(
                event_id=record.event_id,
                event_record_sha256=record.record_sha256,
                model_split=record.model_split,
                oof_fold=key,
                ictal_checkpoint_manifest_sha256=(
                    checkpoint.checkpoint_manifest_sha256
                ),
                ictal_checkpoint_lineage_sha256=checkpoint.lineage_sha256,
                evolution_scaler_artifact_sha256=(
                    scaler.scaler_artifact_sha256
                ),
                evolution_scaler_lineage_sha256=scaler.lineage_sha256,
            )
        )
    events = tuple(sorted(event_selections, key=lambda event: event.event_id))
    event_roster_sha = _canonical_sha256(
        tuple((event.event_id, event.event_record_sha256) for event in events)
    )
    receipt = oof_protocol.receipt
    return OOFEvidenceMaterializationPlan(
        event_registry_sha256=event_registry.manifest_sha256,
        split_manifest_sha256=event_registry.split_manifest_sha256,
        oof_protocol_receipt_sha256=receipt.receipt_sha256,
        public_ledger_build_sha256=receipt.public_ledger_build_sha256,
        source_train_roster_sha256=receipt.source_train_roster_sha256,
        source_dev_roster_sha256=receipt.source_dev_roster_sha256,
        source_eval_roster_sha256=receipt.source_eval_roster_sha256,
        event_roster_sha256=event_roster_sha,
        event_count=len(events),
        split_event_counts=tuple(
            (split, sum(event.model_split == split for event in events))
            for split in _SOURCE_SPLITS
        ),
        ictal_checkpoints=tuple(checkpoint_by_key[key] for key in _SELECTION_KEYS),
        evolution_scalers=tuple(scaler_by_key[key] for key in _SELECTION_KEYS),
        events=events,
    )


def validate_i_computed_v_evidence(
    plan: OOFEvidenceMaterializationPlan,
    event_id: object,
    evidence: EvidenceBatch,
) -> OOFEvidenceEventSelection:
    """Validate an evidence-only I+computed-V payload before cache publication.

    This gate cannot authenticate an as-yet-unimplemented scaler artifact
    loader.  It does enforce the semantic firewall that is locally decidable:
    morphology is truly absent rather than a zero-valued observed branch, the
    edge availability is exactly the ictal availability, and all masked values
    use finite zero fill.
    """

    if not isinstance(plan, OOFEvidenceMaterializationPlan):
        raise TypeError("plan must be an OOFEvidenceMaterializationPlan")
    if not isinstance(evidence, EvidenceBatch):
        raise TypeError("Reasoner materialization accepts EvidenceBatch only")
    selection = plan.event(event_id)
    evidence.validate()
    if evidence.batch_size != 1 or evidence.n_tiles != 15:
        raise ValueError("Materialized evidence must contain one fixed 15-tile event")
    if evidence.node.requires_grad or evidence.edge.requires_grad:
        raise ValueError("Materialized reasoner evidence must be detached")
    morphology_mask = evidence.morphology_mask
    morphology_context_mask = evidence.morphology_context_mask
    ictal_mask = evidence.ictal_mask
    if (
        morphology_mask is None
        or morphology_context_mask is None
        or ictal_mask is None
    ):
        raise RuntimeError("Evidence family masks were not initialized")
    if morphology_mask.any() or morphology_context_mask.any():
        raise ValueError(
            "Morphology is absent and must use an all-false mask pair "
            "(local/context)"
        )
    morphology_values = evidence.edge[..., :N_MORPHOLOGY_FEATURES]
    if torch.any(morphology_values != 0):
        raise ValueError("Absent morphology must be finite zero-filled")
    if not torch.equal(evidence.edge_mask, ictal_mask):
        raise ValueError("With morphology absent, edge_mask must equal ictal_mask")
    if not ictal_mask.any():
        raise ValueError("Present ictal branch contains no available evidence")
    ictal_values = evidence.edge[..., N_MORPHOLOGY_FEATURES:]
    if torch.any(ictal_values[~ictal_mask] != 0):
        raise ValueError("Masked ictal evidence must use finite zero fill")
    observed_ictal = ictal_values[ictal_mask]
    if torch.any((observed_ictal < 0) | (observed_ictal > 1)):
        raise ValueError("Ictal mean/max evidence must lie in [0,1]")
    if not evidence.node_mask.any():
        raise ValueError("Present computed-evolution branch has no available evidence")
    if torch.any(evidence.node[~evidence.node_mask] != 0):
        raise ValueError("Masked computed-evolution evidence must use finite zero fill")
    scaler = next(
        item for item in plan.evolution_scalers if item.oof_fold == selection.oof_fold
    )
    observed_node = evidence.node[evidence.node_mask]
    if torch.any(observed_node.abs() > scaler.clip + 1e-6):
        raise ValueError("Computed-evolution evidence exceeds its scaler clip")
    return selection


@dataclass(frozen=True)
class OOFEvidenceMaterializationPlanArtifact:
    path: Path
    plan: OOFEvidenceMaterializationPlan
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, OOFEvidenceMaterializationPlan):
            raise TypeError("artifact plan must be an OOFEvidenceMaterializationPlan")
        object.__setattr__(
            self,
            "artifact_sha256",
            _require_sha256(self.artifact_sha256, field="artifact_sha256"),
        )


def _artifact_payload(plan: OOFEvidenceMaterializationPlan) -> dict[str, object]:
    return {
        "schema_version": OOF_EVIDENCE_MATERIALIZATION_ARTIFACT_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "plan": asdict(plan),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_symlink_components(path: Path, *, field: str) -> None:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot contain symlink components")


def save_oof_evidence_materialization_plan(
    plan: OOFEvidenceMaterializationPlan,
    bundle_directory: str | Path,
) -> OOFEvidenceMaterializationPlanArtifact:
    """Atomically publish a canonical plan bundle; overwrite is forbidden."""

    if not isinstance(plan, OOFEvidenceMaterializationPlan):
        raise TypeError("plan must be an OOFEvidenceMaterializationPlan")
    encoded = _canonical_json_bytes(_artifact_payload(plan))
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("Materialization plan exceeds the closed size limit")
    bundle = Path(bundle_directory)
    if bundle.is_symlink() or os.path.lexists(bundle):
        raise FileExistsError("Materialization plan destination already exists")
    if bundle.name in {"", ".", ".."}:
        raise ValueError("Materialization plan requires a concrete directory name")
    parent = bundle.parent
    _reject_symlink_components(parent, field="Materialization plan parent")
    if not parent.is_dir():
        raise FileNotFoundError("Materialization plan parent does not exist")

    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle.name}.tmp-", dir=parent))
    temporary_file = temporary / OOF_EVIDENCE_MATERIALIZATION_FILENAME
    published = False
    try:
        with temporary_file.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        if os.path.lexists(bundle):
            raise FileExistsError("Materialization plan destination already exists")
        os.rename(temporary, bundle)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and temporary.exists():
            if temporary_file.exists() and not temporary_file.is_symlink():
                temporary_file.unlink()
            temporary.rmdir()
    return OOFEvidenceMaterializationPlanArtifact(
        path=bundle,
        plan=plan,
        artifact_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def load_oof_evidence_materialization_plan(
    bundle_directory: str | Path,
    expected_plan: OOFEvidenceMaterializationPlan,
    *,
    expected_artifact_sha256: str | None = None,
    expected_plan_sha256: str | None = None,
) -> OOFEvidenceMaterializationPlanArtifact:
    """Load only when bytes exactly reproduce an independently rebuilt plan."""

    if not isinstance(expected_plan, OOFEvidenceMaterializationPlan):
        raise TypeError("expected_plan must be an OOFEvidenceMaterializationPlan")
    bundle = Path(bundle_directory)
    _reject_symlink_components(bundle, field="Materialization plan bundle")
    if not bundle.is_dir():
        raise FileNotFoundError("Materialization plan bundle does not exist")
    entries = tuple(sorted(bundle.iterdir(), key=lambda item: item.name))
    if (
        len(entries) != 1
        or entries[0].name != OOF_EVIDENCE_MATERIALIZATION_FILENAME
        or entries[0].is_symlink()
        or not entries[0].is_file()
    ):
        raise ValueError("Materialization plan bundle violates its closed file schema")
    artifact_file = entries[0]
    before = artifact_file.stat()
    if before.st_size < 1 or before.st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("Materialization plan artifact has an invalid size")
    encoded = artifact_file.read_bytes()
    after = artifact_file.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("Materialization plan changed while it was read")
    artifact_sha = hashlib.sha256(encoded).hexdigest()
    if expected_artifact_sha256 is not None and artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Materialization artifact SHA does not match expected SHA")
    try:
        payload = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Materialization plan is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Materialization plan artifact must be a JSON object")
    actual_fields = set(payload)
    if actual_fields != _ARTIFACT_FIELDS:
        raise ValueError(
            "Materialization artifact violates the closed schema; "
            f"missing={sorted(_ARTIFACT_FIELDS-actual_fields)}, "
            f"unknown={sorted(actual_fields-_ARTIFACT_FIELDS)}"
        )
    if _canonical_json_bytes(payload) != encoded:
        raise ValueError("Materialization plan bytes are not canonical JSON")
    if payload["schema_version"] != OOF_EVIDENCE_MATERIALIZATION_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported materialization plan artifact schema")
    declared_plan_sha = _require_sha256(
        payload["plan_sha256"], field="plan_sha256"
    )
    if declared_plan_sha != expected_plan.plan_sha256:
        raise ValueError("Materialization plan SHA does not match rebuilt plan")
    if expected_plan_sha256 is not None and declared_plan_sha != _require_sha256(
        expected_plan_sha256, field="expected_plan_sha256"
    ):
        raise ValueError("Materialization plan SHA does not match expected SHA")
    if _canonical_json_bytes(payload["plan"]) != _canonical_json_bytes(
        asdict(expected_plan)
    ):
        raise ValueError(
            "Persisted materialization plan does not exactly match rebuild"
        )
    return OOFEvidenceMaterializationPlanArtifact(
        path=bundle,
        plan=expected_plan,
        artifact_sha256=artifact_sha,
    )


__all__ = [
    "BRANCH_PRESENCE",
    "CACHE_AUTHORIZATION_BLOCKERS",
    "CACHE_AUTHORIZATION_POLICY",
    "EVOLUTION_MATERIALIZATION_POLICY",
    "ICTAL_MATERIALIZATION_POLICY",
    "MORPHOLOGY_ABSENCE_POLICY",
    "OOF_EVIDENCE_MATERIALIZATION_ARTIFACT_SCHEMA",
    "OOF_EVIDENCE_MATERIALIZATION_FILENAME",
    "OOF_EVIDENCE_MATERIALIZATION_PLAN_SCHEMA",
    "REASONER_INPUT_POLICY",
    "ComputedEvolutionScalerLineage",
    "IctalCheckpointLineage",
    "OOFEvidenceEventSelection",
    "OOFEvidenceMaterializationPlan",
    "OOFEvidenceMaterializationPlanArtifact",
    "build_computed_evolution_scaler_lineage",
    "build_deepsoz_oof_evidence_materialization_plan",
    "ictal_checkpoint_lineage_from_loaded",
    "load_oof_evidence_materialization_plan",
    "save_oof_evidence_materialization_plan",
    "validate_i_computed_v_evidence",
]
