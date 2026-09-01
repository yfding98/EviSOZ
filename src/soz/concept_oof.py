"""Frozen patient-OOF protocol for the ictal-involvement concept extractor.

The fold assignment is owned by :class:`DeepSOZReferenceRegistry`; this module
never creates, shuffles, or repairs folds.  Each plan also binds the exact
overlap-audited TUSZ official-train cohort that it is permitted to consume.
Private data cannot enter this protocol because the overlap ledger accepts
only its closed public-data schema and the ictal cohort is fixed to TUSZ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

from .data.deepsoz import (
    EXPECTED_CONCEPT_OOF_FOLDS,
    DeepSOZReferenceRegistry,
    normalize_patient_id,
)
from .data.overlap import (
    ConceptTrainingCohort,
    PublicOverlapLedger,
    build_concept_training_cohort,
    canonical_public_roster_sha256,
    normalize_public_patient_key,
)
from .data.provenance import patient_roster_sha256
from .data.public_ledger_builder import TUSZDeepSOZPublicLedgerArtifact


ICTAL_CONCEPT_FAMILY = "ictal_involvement"
ICTAL_CONCEPT_DATASETS = ("tusz",)
ICTAL_OOF_PLAN_SCHEMA = "soz_ictal_concept_oof_plan_v2"
ICTAL_OOF_PROTOCOL_SCHEMA = "soz_ictal_concept_oof_protocol_v2"
ICTAL_OOF_PROTOCOL_ARTIFACT_SCHEMA = (
    "soz_ictal_concept_oof_protocol_artifact_v1"
)
ICTAL_OOF_PROTOCOL_ARTIFACT_FILENAME = "ictal_concept_oof_protocol.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_SPLITS = ("source_train", "source_dev", "source_eval")
_PROTOCOL_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_sha256",
        "public_ledger_build_sha256",
        "split_manifest_sha256",
        "protocol",
    }
)
_MAX_PROTOCOL_ARTIFACT_BYTES = 128 * 1024 * 1024


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        raise ValueError("OOF protocol artifact is not canonical JSON data") from exc
    return (encoded + "\n").encode("utf-8")


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return text


def _normalize_patient_roster(
    values: tuple[str, ...], *, field: str, allow_empty: bool = False
) -> tuple[str, ...]:
    normalized = tuple(sorted(normalize_patient_id(value) for value in values))
    if (not normalized and not allow_empty) or len(set(normalized)) != len(normalized):
        qualifier = "non-empty and " if not allow_empty else ""
        raise ValueError(f"{field} must be {qualifier}unique")
    return normalized


def _normalize_public_patient_roster(
    values: tuple[str, ...], *, field: str
) -> tuple[str, ...]:
    normalized = tuple(sorted(normalize_public_patient_key(value) for value in values))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must be non-empty and unique")
    return normalized


def _public_patient_roster_sha256(values: tuple[str, ...]) -> str:
    normalized = _normalize_public_patient_roster(
        values, field="public_patient_roster"
    )
    return _canonical_sha256(normalized)


def _normalize_record_roster(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(sorted(str(value).strip().lower() for value in values))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("authorized_record_sha256s must be non-empty and unique")
    if any(not _SHA256_RE.fullmatch(value) for value in normalized):
        raise ValueError("authorized_record_sha256s must contain SHA256 identifiers")
    return normalized


@dataclass(frozen=True)
class IctalConceptOOFPlanReceipt:
    """Canonical lineage for one fold extractor or the final extractor."""

    oof_fold: int | None
    split_manifest_sha256: str
    public_ledger_build_sha256: str
    ledger_sha256: str
    ledger_receipt_sha256: str
    training_target_patient_ids: tuple[str, ...]
    held_out_target_patient_ids: tuple[str, ...]
    held_out_public_patient_keys: tuple[str, ...]
    training_target_roster_sha256: str
    held_out_target_roster_sha256: str
    held_out_public_roster_sha256: str
    cohort_bindings: tuple[tuple[str, str], ...]
    authorized_record_sha256s: tuple[str, ...]
    authorized_record_roster_sha256: str
    concept_family: str = ICTAL_CONCEPT_FAMILY
    schema_version: str = ICTAL_OOF_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ICTAL_OOF_PLAN_SCHEMA:
            raise ValueError("Unexpected ictal OOF plan schema")
        if self.concept_family != ICTAL_CONCEPT_FAMILY:
            raise ValueError("OOF plans are restricted to ictal_involvement")
        if self.oof_fold is not None and (
            isinstance(self.oof_fold, bool)
            or not isinstance(self.oof_fold, int)
            or self.oof_fold not in range(EXPECTED_CONCEPT_OOF_FOLDS)
        ):
            raise ValueError("oof_fold must be None or an integer in [0,4]")

        for field in (
            "split_manifest_sha256",
            "public_ledger_build_sha256",
            "ledger_sha256",
            "ledger_receipt_sha256",
            "training_target_roster_sha256",
            "held_out_target_roster_sha256",
            "held_out_public_roster_sha256",
            "authorized_record_roster_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), field=field),
            )

        training = _normalize_patient_roster(
            self.training_target_patient_ids,
            field="training_target_patient_ids",
        )
        held_out = _normalize_patient_roster(
            self.held_out_target_patient_ids,
            field="held_out_target_patient_ids",
        )
        if set(training) & set(held_out):
            raise ValueError("Training and held-out target rosters must be disjoint")
        object.__setattr__(self, "training_target_patient_ids", training)
        object.__setattr__(self, "held_out_target_patient_ids", held_out)
        if self.training_target_roster_sha256 != patient_roster_sha256(training):
            raise ValueError("training_target_roster_sha256 does not match its roster")
        if self.held_out_target_roster_sha256 != patient_roster_sha256(held_out):
            raise ValueError("held_out_target_roster_sha256 does not match its roster")
        held_out_public = _normalize_public_patient_roster(
            self.held_out_public_patient_keys,
            field="held_out_public_patient_keys",
        )
        if len(held_out_public) != len(held_out):
            raise ValueError(
                "Held-out target IDs and public patient keys must be one-to-one"
            )
        object.__setattr__(
            self, "held_out_public_patient_keys", held_out_public
        )
        if self.held_out_public_roster_sha256 != _public_patient_roster_sha256(
            held_out_public
        ):
            raise ValueError("held_out_public_roster_sha256 does not match its roster")

        bindings = tuple(
            (str(split).strip().lower(), _require_sha256(digest, field="cohort_sha256"))
            for split, digest in self.cohort_bindings
        )
        expected_splits = (
            ("train",)
            if self.oof_fold is not None
            else ("dev", "eval")
        )
        if tuple(split for split, _ in bindings) != expected_splits:
            raise ValueError(
                "Fold plans must bind train; the final plan must bind dev and eval"
            )
        if len({digest for _, digest in bindings}) != len(bindings):
            raise ValueError("Each cohort binding must have a distinct receipt hash")
        object.__setattr__(self, "cohort_bindings", bindings)

        authorized = _normalize_record_roster(self.authorized_record_sha256s)
        object.__setattr__(self, "authorized_record_sha256s", authorized)
        if self.authorized_record_roster_sha256 != canonical_public_roster_sha256(
            authorized
        ):
            raise ValueError(
                "authorized_record_roster_sha256 does not match its roster"
            )

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class IctalConceptOOFPlan:
    """One executable plan with its overlap-audited auxiliary cohort(s)."""

    cohorts: tuple[ConceptTrainingCohort, ...]
    receipt: IctalConceptOOFPlanReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, IctalConceptOOFPlanReceipt):
            raise TypeError("receipt must be an IctalConceptOOFPlanReceipt")
        if not self.cohorts or any(
            not isinstance(cohort, ConceptTrainingCohort) for cohort in self.cohorts
        ):
            raise TypeError("cohorts must contain ConceptTrainingCohort objects")
        bindings = tuple(
            (cohort.receipt.target_split, cohort.receipt.receipt_sha256)
            for cohort in self.cohorts
        )
        if bindings != self.receipt.cohort_bindings:
            raise ValueError("Plan cohorts disagree with their canonical bindings")

        first_allowed = self.cohorts[0].receipt.allowed_record_sha256s
        for cohort in self.cohorts:
            if cohort.receipt.ledger_sha256 != self.receipt.ledger_sha256:
                raise ValueError("Plan cohort was built from a different overlap ledger")
            if cohort.receipt.ledger_receipt_sha256 != self.receipt.ledger_receipt_sha256:
                raise ValueError("Plan cohort ledger receipt does not match")
            if cohort.receipt.concept_datasets != ICTAL_CONCEPT_DATASETS:
                raise ValueError("Ictal concept cohorts are restricted to TUSZ")
            if cohort.receipt.allowed_record_sha256s != first_allowed:
                raise ValueError("All final-plan cohorts must authorize one exact roster")
            if any(
                record.dataset != "tusz" or record.split != "train"
                for record in cohort.allowed_records
            ):
                raise ValueError("Ictal concept training requires TUSZ official-train")
        if first_allowed != self.receipt.authorized_record_sha256s:
            raise ValueError("Authorized records disagree with the plan receipt")

        if self.receipt.oof_fold is not None:
            cohort_heldout = self.cohorts[0].receipt.heldout_target_patient_keys
            if cohort_heldout != self.receipt.held_out_public_patient_keys:
                raise ValueError("Fold cohort does not hold out the exact public roster")
        else:
            protected_public = {
                key
                for cohort in self.cohorts
                for key in cohort.receipt.heldout_target_patient_keys
            }
            if not set(self.receipt.held_out_public_patient_keys) <= protected_public:
                raise ValueError(
                    "Final cohort protection omits an eligible held-out public key"
                )

    @property
    def oof_fold(self) -> int | None:
        return self.receipt.oof_fold

    @property
    def training_target_patient_ids(self) -> tuple[str, ...]:
        return self.receipt.training_target_patient_ids

    @property
    def held_out_target_patient_ids(self) -> tuple[str, ...]:
        return self.receipt.held_out_target_patient_ids

    @property
    def held_out_public_patient_keys(self) -> tuple[str, ...]:
        return self.receipt.held_out_public_patient_keys

    @property
    def training_cohort(self) -> ConceptTrainingCohort:
        """Return the shared authorized cohort (identical for final dev/eval binds)."""

        return self.cohorts[0]


@dataclass(frozen=True)
class IctalConceptOOFProtocolReceipt:
    """Canonical receipt for all five OOF plans and the final plan."""

    split_manifest_sha256: str
    public_ledger_build_sha256: str
    ledger_sha256: str
    ledger_receipt_sha256: str
    source_train_patient_ids: tuple[str, ...]
    source_dev_patient_ids: tuple[str, ...]
    source_eval_patient_ids: tuple[str, ...]
    source_train_roster_sha256: str
    source_dev_roster_sha256: str
    source_eval_roster_sha256: str
    target_public_crosswalk: tuple[tuple[str, str], ...]
    target_public_crosswalk_sha256: str
    fold_plan_receipt_sha256s: tuple[tuple[int, str], ...]
    final_plan_receipt_sha256: str
    concept_family: str = ICTAL_CONCEPT_FAMILY
    schema_version: str = ICTAL_OOF_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ICTAL_OOF_PROTOCOL_SCHEMA:
            raise ValueError("Unexpected ictal OOF protocol schema")
        if self.concept_family != ICTAL_CONCEPT_FAMILY:
            raise ValueError("OOF protocol is restricted to ictal_involvement")
        for field in (
            "split_manifest_sha256",
            "public_ledger_build_sha256",
            "ledger_sha256",
            "ledger_receipt_sha256",
            "source_train_roster_sha256",
            "source_dev_roster_sha256",
            "source_eval_roster_sha256",
            "target_public_crosswalk_sha256",
            "final_plan_receipt_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), field=field),
            )

        rosters = {
            split: _normalize_patient_roster(
                getattr(self, f"{split}_patient_ids"),
                field=f"{split}_patient_ids",
            )
            for split in _SOURCE_SPLITS
        }
        if any(
            set(rosters[left]) & set(rosters[right])
            for index, left in enumerate(_SOURCE_SPLITS)
            for right in _SOURCE_SPLITS[index + 1 :]
        ):
            raise ValueError("Eligible target-patient splits must be disjoint")
        for split, roster in rosters.items():
            object.__setattr__(self, f"{split}_patient_ids", roster)
            if getattr(self, f"{split}_roster_sha256") != patient_roster_sha256(
                roster
            ):
                raise ValueError(f"{split}_roster_sha256 does not match its roster")

        crosswalk = tuple(
            (
                normalize_patient_id(target_id),
                normalize_public_patient_key(public_key),
            )
            for target_id, public_key in self.target_public_crosswalk
        )
        if tuple(sorted(crosswalk)) != crosswalk:
            raise ValueError("target_public_crosswalk must be canonically sorted")
        target_ids = tuple(target_id for target_id, _ in crosswalk)
        public_keys = tuple(public_key for _, public_key in crosswalk)
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("target_public_crosswalk contains duplicate target IDs")
        if len(set(public_keys)) != len(public_keys):
            raise ValueError("target_public_crosswalk must be one-to-one")
        expected_targets = set().union(*(set(roster) for roster in rosters.values()))
        if set(target_ids) != expected_targets:
            raise ValueError(
                "target_public_crosswalk must exactly cover all eligible target IDs"
            )
        object.__setattr__(self, "target_public_crosswalk", crosswalk)
        if self.target_public_crosswalk_sha256 != _canonical_sha256(crosswalk):
            raise ValueError(
                "target_public_crosswalk_sha256 does not match the canonical crosswalk"
            )

        fold_hashes = tuple(
            (fold, _require_sha256(digest, field="fold_plan_receipt_sha256"))
            for fold, digest in self.fold_plan_receipt_sha256s
        )
        if tuple(fold for fold, _ in fold_hashes) != tuple(
            range(EXPECTED_CONCEPT_OOF_FOLDS)
        ):
            raise ValueError("Protocol must bind exactly OOF folds 0 through 4")
        if len({digest for _, digest in fold_hashes}) != len(fold_hashes):
            raise ValueError("Fold plan receipt hashes must be distinct")
        object.__setattr__(self, "fold_plan_receipt_sha256s", fold_hashes)

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class IctalConceptOOFProtocol:
    """The five frozen source-train OOF plans plus one final train-only plan."""

    fold_plans: tuple[IctalConceptOOFPlan, ...]
    final_plan: IctalConceptOOFPlan
    receipt: IctalConceptOOFProtocolReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, IctalConceptOOFProtocolReceipt):
            raise TypeError("receipt must be an IctalConceptOOFProtocolReceipt")
        if len(self.fold_plans) != EXPECTED_CONCEPT_OOF_FOLDS:
            raise ValueError("Protocol requires exactly five fold plans")
        if any(not isinstance(plan, IctalConceptOOFPlan) for plan in self.fold_plans):
            raise TypeError("fold_plans must contain IctalConceptOOFPlan objects")
        if not isinstance(self.final_plan, IctalConceptOOFPlan):
            raise TypeError("final_plan must be an IctalConceptOOFPlan")
        if tuple(plan.oof_fold for plan in self.fold_plans) != tuple(
            range(EXPECTED_CONCEPT_OOF_FOLDS)
        ):
            raise ValueError("Fold plans must be canonically ordered 0 through 4")
        if self.final_plan.oof_fold is not None:
            raise ValueError("Final train-only plan must have oof_fold=None")
        for plan in (*self.fold_plans, self.final_plan):
            if plan.receipt.split_manifest_sha256 != self.receipt.split_manifest_sha256:
                raise ValueError("Plan uses a different target split manifest")
            if (
                plan.receipt.public_ledger_build_sha256
                != self.receipt.public_ledger_build_sha256
            ):
                raise ValueError("Plan uses a different public-ledger build")
            if plan.receipt.ledger_sha256 != self.receipt.ledger_sha256:
                raise ValueError("Plan uses a different public overlap ledger")
            if (
                plan.receipt.ledger_receipt_sha256
                != self.receipt.ledger_receipt_sha256
            ):
                raise ValueError("Plan uses a different overlap-ledger receipt")
        expected_fold_hashes = tuple(
            (int(plan.oof_fold), plan.receipt.receipt_sha256)
            for plan in self.fold_plans
        )
        if expected_fold_hashes != self.receipt.fold_plan_receipt_sha256s:
            raise ValueError("Fold plans disagree with the protocol receipt")
        if (
            self.final_plan.receipt.receipt_sha256
            != self.receipt.final_plan_receipt_sha256
        ):
            raise ValueError("Final plan disagrees with the protocol receipt")
        all_train = set(self.receipt.source_train_patient_ids)
        held_once: list[str] = []
        for plan in self.fold_plans:
            if set(plan.training_target_patient_ids) | set(
                plan.held_out_target_patient_ids
            ) != all_train:
                raise ValueError("A fold plan does not partition source_train")
            held_once.extend(plan.held_out_target_patient_ids)
        if len(held_once) != len(set(held_once)) or set(held_once) != all_train:
            raise ValueError("Each source_train patient must be held out exactly once")
        if set(self.final_plan.training_target_patient_ids) != all_train:
            raise ValueError("Final plan must train on every source_train patient")
        expected_final_heldout = set(self.receipt.source_dev_patient_ids) | set(
            self.receipt.source_eval_patient_ids
        )
        if set(self.final_plan.held_out_target_patient_ids) != expected_final_heldout:
            raise ValueError("Final plan must hold out all source_dev and source_eval")
        crosswalk = dict(self.receipt.target_public_crosswalk)
        for plan in (*self.fold_plans, self.final_plan):
            expected_public = {
                crosswalk[target_id]
                for target_id in plan.held_out_target_patient_ids
            }
            if set(plan.held_out_public_patient_keys) != expected_public:
                raise ValueError("Plan public held-out roster disagrees with the crosswalk")

    def for_fold(self, fold: int) -> IctalConceptOOFPlan:
        if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(
            EXPECTED_CONCEPT_OOF_FOLDS
        ):
            raise ValueError("fold must be an integer in [0,4]")
        return self.fold_plans[fold]


@dataclass(frozen=True)
class IctalConceptOOFProtocolArtifact:
    """Verified in-memory view of one canonical OOF protocol bundle."""

    protocol: IctalConceptOOFProtocol
    artifact_sha256: str
    schema_version: str = ICTAL_OOF_PROTOCOL_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, IctalConceptOOFProtocol):
            raise TypeError("protocol must be an IctalConceptOOFProtocol")
        object.__setattr__(
            self,
            "artifact_sha256",
            _require_sha256(self.artifact_sha256, field="artifact_sha256"),
        )
        if self.schema_version != ICTAL_OOF_PROTOCOL_ARTIFACT_SCHEMA:
            raise ValueError("Unexpected ictal OOF protocol artifact schema")

    @property
    def protocol_sha256(self) -> str:
        return self.protocol.receipt.receipt_sha256

    @property
    def public_ledger_build_sha256(self) -> str:
        return self.protocol.receipt.public_ledger_build_sha256


def _eligible_rosters(
    registry: DeepSOZReferenceRegistry,
) -> Mapping[str, tuple[str, ...]]:
    rosters = {
        split: tuple(
            sorted(reference.patient_id for reference in registry.for_split(split))
        )
        for split in _SOURCE_SPLITS
    }
    if any(not roster for roster in rosters.values()):
        missing = [split for split, roster in rosters.items() if not roster]
        raise ValueError(f"Eligible DeepSOZ target rosters are empty for: {missing}")
    return rosters


def _normalize_target_public_crosswalk(
    target_patient_to_public_key: Mapping[object, object],
    *,
    rosters: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(target_patient_to_public_key, Mapping):
        raise TypeError("target_patient_to_public_key must be a mapping")
    normalized: dict[str, str] = {}
    public_to_target: dict[str, str] = {}
    for raw_target_id, raw_public_key in target_patient_to_public_key.items():
        target_id = normalize_patient_id(raw_target_id)
        public_key = normalize_public_patient_key(raw_public_key)
        if target_id in normalized:
            raise ValueError(
                "target_patient_to_public_key contains duplicate normalized "
                f"target ID {target_id!r}"
            )
        if public_key in public_to_target:
            raise ValueError(
                "target_patient_to_public_key must be one-to-one; public key "
                f"{public_key!r} maps from both {public_to_target[public_key]!r} "
                f"and {target_id!r}"
            )
        normalized[target_id] = public_key
        public_to_target[public_key] = target_id

    expected_targets = set().union(*(set(roster) for roster in rosters.values()))
    supplied_targets = set(normalized)
    missing = sorted(expected_targets - supplied_targets)
    extra = sorted(supplied_targets - expected_targets)
    if missing or extra:
        raise ValueError(
            "target_patient_to_public_key must exactly cover eligible target IDs; "
            f"missing={missing}, extra={extra}"
        )
    return tuple(sorted(normalized.items()))


def _validate_registry_ledger_alignment(
    rosters: Mapping[str, tuple[str, ...]],
    crosswalk: Mapping[str, str],
    ledger: PublicOverlapLedger,
) -> None:
    for source_split, official_split in zip(_SOURCE_SPLITS, ("train", "dev", "eval")):
        ledger_patients = {
            record.patient_key
            for record in ledger.records_for(dataset="deepsoz", split=official_split)
        }
        missing = sorted(
            target_id
            for target_id in rosters[source_split]
            if crosswalk[target_id] not in ledger_patients
        )
        if missing:
            raise ValueError(
                "Mapped eligible DeepSOZ patients are absent from the overlap ledger "
                f"for {source_split}: {missing}"
            )


def _make_plan(
    *,
    oof_fold: int | None,
    split_manifest_sha256: str,
    public_ledger_build_sha256: str,
    ledger: PublicOverlapLedger,
    training: tuple[str, ...],
    held_out: tuple[str, ...],
    held_out_public: tuple[str, ...],
    cohorts: tuple[ConceptTrainingCohort, ...],
) -> IctalConceptOOFPlan:
    first_allowed = cohorts[0].receipt.allowed_record_sha256s
    receipt = IctalConceptOOFPlanReceipt(
        oof_fold=oof_fold,
        split_manifest_sha256=split_manifest_sha256,
        public_ledger_build_sha256=public_ledger_build_sha256,
        ledger_sha256=ledger.receipt.ledger_sha256,
        ledger_receipt_sha256=ledger.receipt.receipt_sha256,
        training_target_patient_ids=training,
        held_out_target_patient_ids=held_out,
        held_out_public_patient_keys=held_out_public,
        training_target_roster_sha256=patient_roster_sha256(training),
        held_out_target_roster_sha256=patient_roster_sha256(held_out),
        held_out_public_roster_sha256=_public_patient_roster_sha256(
            held_out_public
        ),
        cohort_bindings=tuple(
            (cohort.receipt.target_split, cohort.receipt.receipt_sha256)
            for cohort in cohorts
        ),
        authorized_record_sha256s=first_allowed,
        authorized_record_roster_sha256=canonical_public_roster_sha256(
            first_allowed
        ),
    )
    return IctalConceptOOFPlan(cohorts=cohorts, receipt=receipt)


def build_ictal_concept_oof_protocol(
    registry: DeepSOZReferenceRegistry,
    ledger: PublicOverlapLedger,
    *,
    target_patient_to_public_key: Mapping[object, object],
    split_manifest_sha256: str,
    public_ledger_build_sha256: str,
) -> IctalConceptOOFProtocol:
    """Build the immutable five-fold/final ictal concept-training protocol.

    No seed or fold-assignment input is accepted.  There is deliberately no
    identity fallback: ``target_patient_to_public_key`` must map every eligible
    DeepSOZ target ID to its distinct canonical public-data patient key.
    Eligible source-train patients retain the exact ``concept_oof_fold`` stored
    in ``registry``.
    The final extractor trains on the complete eligible source-train roster;
    source-dev and source-eval are both held out and independently receipt-bound.
    ``public_ledger_build_sha256`` must be the build receipt SHA from the exact
    :class:`TUSZDeepSOZPublicLedgerArtifact` that supplied ``ledger`` and the
    crosswalk.  The loader enforces that relationship by reconstruction.
    """

    if not isinstance(registry, DeepSOZReferenceRegistry):
        raise TypeError("registry must be a DeepSOZReferenceRegistry")
    if not isinstance(ledger, PublicOverlapLedger):
        raise TypeError("ledger must be a PublicOverlapLedger")
    split_sha = _require_sha256(
        split_manifest_sha256, field="split_manifest_sha256"
    )
    public_build_sha = _require_sha256(
        public_ledger_build_sha256, field="public_ledger_build_sha256"
    )
    rosters = _eligible_rosters(registry)
    crosswalk_items = _normalize_target_public_crosswalk(
        target_patient_to_public_key, rosters=rosters
    )
    crosswalk = dict(crosswalk_items)
    _validate_registry_ledger_alignment(rosters, crosswalk, ledger)

    source_train = rosters["source_train"]
    fold_plans: list[IctalConceptOOFPlan] = []
    for fold in range(EXPECTED_CONCEPT_OOF_FOLDS):
        held_out = tuple(
            patient_id
            for patient_id in source_train
            if registry.get(patient_id).concept_oof_fold == fold
        )
        if not held_out:
            raise ValueError(
                f"Frozen DeepSOZ assignment has no eligible source_train patient in fold {fold}"
            )
        training = tuple(
            patient_id for patient_id in source_train if patient_id not in set(held_out)
        )
        if not training:
            raise ValueError(f"Fold {fold} leaves no source_train patient for training")
        held_out_public = tuple(sorted(crosswalk[patient_id] for patient_id in held_out))
        cohort = build_concept_training_cohort(
            ledger,
            target_split="source_train",
            heldout_target_patient_keys=held_out_public,
            concept_datasets=ICTAL_CONCEPT_DATASETS,
        )
        fold_plans.append(
            _make_plan(
                oof_fold=fold,
                split_manifest_sha256=split_sha,
                public_ledger_build_sha256=public_build_sha,
                ledger=ledger,
                training=training,
                held_out=held_out,
                held_out_public=held_out_public,
                cohorts=(cohort,),
            )
        )

    # The cohort builder requires a complete roster for dev/eval.  Use the
    # complete ledger rosters here (including any non-targetable audit rows),
    # while the extractor receipt records only eligible target patients.
    dev_protection = tuple(
        sorted(
            {
                record.patient_key
                for record in ledger.records_for(dataset="deepsoz", split="dev")
            }
        )
    )
    eval_protection = tuple(
        sorted(
            {
                record.patient_key
                for record in ledger.records_for(dataset="deepsoz", split="eval")
            }
        )
    )
    if not dev_protection or not eval_protection:
        raise ValueError("Final plan requires non-empty DeepSOZ dev and eval ledgers")
    final_cohorts = (
        build_concept_training_cohort(
            ledger,
            target_split="source_dev",
            heldout_target_patient_keys=dev_protection,
            concept_datasets=ICTAL_CONCEPT_DATASETS,
        ),
        build_concept_training_cohort(
            ledger,
            target_split="source_eval",
            heldout_target_patient_keys=eval_protection,
            concept_datasets=ICTAL_CONCEPT_DATASETS,
        ),
    )
    final_heldout = tuple(
        sorted((*rosters["source_dev"], *rosters["source_eval"]))
    )
    final_heldout_public = tuple(
        sorted(crosswalk[patient_id] for patient_id in final_heldout)
    )
    final_plan = _make_plan(
        oof_fold=None,
        split_manifest_sha256=split_sha,
        public_ledger_build_sha256=public_build_sha,
        ledger=ledger,
        training=source_train,
        held_out=final_heldout,
        held_out_public=final_heldout_public,
        cohorts=final_cohorts,
    )

    protocol_receipt = IctalConceptOOFProtocolReceipt(
        split_manifest_sha256=split_sha,
        public_ledger_build_sha256=public_build_sha,
        ledger_sha256=ledger.receipt.ledger_sha256,
        ledger_receipt_sha256=ledger.receipt.receipt_sha256,
        source_train_patient_ids=source_train,
        source_dev_patient_ids=rosters["source_dev"],
        source_eval_patient_ids=rosters["source_eval"],
        source_train_roster_sha256=patient_roster_sha256(source_train),
        source_dev_roster_sha256=patient_roster_sha256(rosters["source_dev"]),
        source_eval_roster_sha256=patient_roster_sha256(rosters["source_eval"]),
        target_public_crosswalk=crosswalk_items,
        target_public_crosswalk_sha256=_canonical_sha256(crosswalk_items),
        fold_plan_receipt_sha256s=tuple(
            (int(plan.oof_fold), plan.receipt.receipt_sha256)
            for plan in fold_plans
        ),
        final_plan_receipt_sha256=final_plan.receipt.receipt_sha256,
    )
    return IctalConceptOOFProtocol(
        fold_plans=tuple(fold_plans),
        final_plan=final_plan,
        receipt=protocol_receipt,
    )


def _protocol_artifact_payload(
    protocol: IctalConceptOOFProtocol,
) -> dict[str, object]:
    if not isinstance(protocol, IctalConceptOOFProtocol):
        raise TypeError("protocol must be an IctalConceptOOFProtocol")
    return {
        "schema_version": ICTAL_OOF_PROTOCOL_ARTIFACT_SCHEMA,
        "protocol_sha256": protocol.receipt.receipt_sha256,
        "public_ledger_build_sha256": (
            protocol.receipt.public_ledger_build_sha256
        ),
        "split_manifest_sha256": protocol.receipt.split_manifest_sha256,
        "protocol": asdict(protocol),
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


def save_ictal_concept_oof_protocol(
    protocol: IctalConceptOOFProtocol,
    bundle_directory: str | Path,
) -> IctalConceptOOFProtocolArtifact:
    """Atomically publish a new, non-overwriting canonical protocol bundle."""

    payload = _protocol_artifact_payload(protocol)
    encoded = _canonical_json_bytes(payload)
    if len(encoded) > _MAX_PROTOCOL_ARTIFACT_BYTES:
        raise ValueError("OOF protocol artifact exceeds the closed size limit")

    bundle = Path(bundle_directory)
    if bundle.is_symlink():
        raise ValueError("OOF protocol bundle destination cannot be a symlink")
    if os.path.lexists(bundle):
        raise FileExistsError("OOF protocol bundle destination already exists")
    if bundle.name in {"", ".", ".."}:
        raise ValueError("OOF protocol bundle requires a concrete directory name")
    parent = bundle.parent
    _reject_symlink_components(parent, field="OOF protocol bundle parent")
    if not parent.is_dir():
        raise FileNotFoundError(
            "OOF protocol bundle parent directory does not exist"
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle.name}.tmp-", dir=parent))
    temporary_file = temporary / ICTAL_OOF_PROTOCOL_ARTIFACT_FILENAME
    published = False
    try:
        with temporary_file.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        if os.path.lexists(bundle):
            raise FileExistsError("OOF protocol bundle destination already exists")
        os.rename(temporary, bundle)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and temporary.exists():
            if temporary_file.exists() and not temporary_file.is_symlink():
                temporary_file.unlink()
            temporary.rmdir()

    return IctalConceptOOFProtocolArtifact(
        protocol=protocol,
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


def _require_closed_json_object(
    value: object,
    *,
    expected_fields: frozenset[str],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    actual = set(value)
    missing = sorted(expected_fields - actual)
    unknown = sorted(actual - expected_fields)
    if missing or unknown:
        raise ValueError(
            f"{field} violates the closed schema; "
            f"missing={missing}, unknown={unknown}"
        )
    return value


def _require_json_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a JSON string")
    return value


def _read_stable_protocol_artifact_bytes(path: Path) -> bytes:
    before = path.stat()
    if not path.is_file():
        raise ValueError("OOF protocol artifact entry must be a regular file")
    if before.st_size > _MAX_PROTOCOL_ARTIFACT_BYTES:
        raise ValueError("OOF protocol artifact exceeds the closed size limit")
    payload = path.read_bytes()
    after = path.stat()
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
        raise RuntimeError("OOF protocol artifact changed while it was read")
    return payload


def _artifact_crosswalk_for_registry(
    registry: DeepSOZReferenceRegistry,
    artifact: TUSZDeepSOZPublicLedgerArtifact,
) -> dict[str, str]:
    eligible_ids = {
        reference.patient_id
        for split in _SOURCE_SPLITS
        for reference in registry.for_split(split)
    }
    missing = sorted(
        patient_id
        for patient_id in eligible_ids
        if patient_id not in artifact.target_patient_to_public_key
    )
    if missing:
        raise ValueError(
            "Public-ledger artifact crosswalk omits eligible target IDs: "
            f"{missing}"
        )
    return {
        patient_id: artifact.target_patient_to_public_key[patient_id]
        for patient_id in eligible_ids
    }


def load_ictal_concept_oof_protocol(
    bundle_directory: str | Path,
    registry: DeepSOZReferenceRegistry,
    public_ledger_artifact: TUSZDeepSOZPublicLedgerArtifact,
    *,
    expected_artifact_sha256: str | None = None,
    expected_protocol_sha256: str | None = None,
) -> IctalConceptOOFProtocolArtifact:
    """Load by rebuilding from the registry and verified public-ledger artifact."""

    if not isinstance(registry, DeepSOZReferenceRegistry):
        raise TypeError("registry must be a DeepSOZReferenceRegistry")
    if not isinstance(public_ledger_artifact, TUSZDeepSOZPublicLedgerArtifact):
        raise TypeError(
            "public_ledger_artifact must be a TUSZDeepSOZPublicLedgerArtifact"
        )

    bundle = Path(bundle_directory)
    _reject_symlink_components(bundle, field="OOF protocol bundle")
    if not bundle.is_dir():
        raise FileNotFoundError("OOF protocol bundle directory does not exist")
    entries = tuple(sorted(bundle.iterdir(), key=lambda item: item.name))
    if (
        len(entries) != 1
        or entries[0].name != ICTAL_OOF_PROTOCOL_ARTIFACT_FILENAME
    ):
        names = [entry.name for entry in entries]
        raise ValueError(
            "OOF protocol bundle must contain exactly the canonical JSON file; "
            f"found={names}"
        )
    artifact_file = entries[0]
    if artifact_file.is_symlink():
        raise ValueError("OOF protocol artifact JSON cannot be a symlink")
    encoded = _read_stable_protocol_artifact_bytes(artifact_file)
    artifact_sha = hashlib.sha256(encoded).hexdigest()
    if expected_artifact_sha256 is not None:
        expected_artifact = _require_sha256(
            expected_artifact_sha256, field="expected_artifact_sha256"
        )
        if artifact_sha != expected_artifact:
            raise ValueError(
                "OOF protocol artifact SHA does not match the expected SHA"
            )

    try:
        payload = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("OOF protocol artifact is not strict UTF-8 JSON") from exc
    if _canonical_json_bytes(payload) != encoded:
        raise ValueError("OOF protocol artifact bytes are not canonical JSON")
    raw = _require_closed_json_object(
        payload,
        expected_fields=_PROTOCOL_ARTIFACT_FIELDS,
        field="OOF protocol artifact",
    )
    schema = _require_json_string(raw["schema_version"], field="schema_version")
    if schema != ICTAL_OOF_PROTOCOL_ARTIFACT_SCHEMA:
        raise ValueError(f"Unsupported OOF protocol artifact schema: {schema}")
    declared_protocol_sha = _require_sha256(
        _require_json_string(raw["protocol_sha256"], field="protocol_sha256"),
        field="protocol_sha256",
    )
    declared_build_sha = _require_sha256(
        _require_json_string(
            raw["public_ledger_build_sha256"],
            field="public_ledger_build_sha256",
        ),
        field="public_ledger_build_sha256",
    )
    if declared_build_sha != public_ledger_artifact.build_sha256:
        raise ValueError(
            "OOF protocol public-ledger build SHA does not match the supplied artifact"
        )
    split_manifest_sha = _require_sha256(
        _require_json_string(
            raw["split_manifest_sha256"], field="split_manifest_sha256"
        ),
        field="split_manifest_sha256",
    )

    rebuilt = build_ictal_concept_oof_protocol(
        registry,
        public_ledger_artifact.ledger,
        target_patient_to_public_key=_artifact_crosswalk_for_registry(
            registry, public_ledger_artifact
        ),
        split_manifest_sha256=split_manifest_sha,
        public_ledger_build_sha256=public_ledger_artifact.build_sha256,
    )
    if declared_protocol_sha != rebuilt.receipt.receipt_sha256:
        raise ValueError("OOF protocol SHA does not match the rebuilt protocol")
    if expected_protocol_sha256 is not None:
        expected_protocol = _require_sha256(
            expected_protocol_sha256, field="expected_protocol_sha256"
        )
        if rebuilt.receipt.receipt_sha256 != expected_protocol:
            raise ValueError("OOF protocol SHA does not match the expected SHA")
    if _canonical_json_bytes(_protocol_artifact_payload(rebuilt)) != encoded:
        raise ValueError(
            "Persisted OOF protocol payload does not exactly match reconstruction"
        )
    return IctalConceptOOFProtocolArtifact(
        protocol=rebuilt,
        artifact_sha256=artifact_sha,
    )


__all__ = [
    "ICTAL_CONCEPT_DATASETS",
    "ICTAL_CONCEPT_FAMILY",
    "ICTAL_OOF_PLAN_SCHEMA",
    "ICTAL_OOF_PROTOCOL_ARTIFACT_FILENAME",
    "ICTAL_OOF_PROTOCOL_ARTIFACT_SCHEMA",
    "ICTAL_OOF_PROTOCOL_SCHEMA",
    "IctalConceptOOFPlan",
    "IctalConceptOOFPlanReceipt",
    "IctalConceptOOFProtocol",
    "IctalConceptOOFProtocolArtifact",
    "IctalConceptOOFProtocolReceipt",
    "build_ictal_concept_oof_protocol",
    "load_ictal_concept_oof_protocol",
    "save_ictal_concept_oof_protocol",
]
