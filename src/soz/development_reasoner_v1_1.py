"""Explicit signal-evidence eligibility amendment for the I+V candidate.

Version 1.1 does not weaken the v1 target join to a run-time intersection.
Instead, it signs the exact target/header and signal-evidence rosters before
target values are opened, wraps the unchanged v1 evidence capability, and
permits the target join only when all three rosters replay exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from . import development_reasoner as _v1
from .data.deepsoz import normalize_patient_id
from .data.deepsoz_signal_preflight import VerifiedDeepSOZSignalPreflightBundle
from .data.deepsoz_target_v2 import (
    TARGET_V2_POLICY_SHA256,
    VerifiedDeepSOZTargetV2Artifact,
)
from .ictal_recovery_evidence import TargetFreeOOFProtocolView


SIGNAL_EVIDENCE_AMENDMENT_SCHEMA = (
    "soz_development_iv_signal_evidence_eligibility_amendment_v1_1"
)
SIGNAL_EVIDENCE_AMENDMENT_ARTIFACT_SCHEMA = (
    "soz_development_iv_signal_evidence_eligibility_amendment_artifact_v1_1"
)
DEVELOPMENT_IV_CAPABILITY_SCHEMA_V1_1 = (
    "soz_development_iv_evidence_capability_v1_1"
)
DEVELOPMENT_IV_CAPABILITY_AUTHORIZATION_SCHEMA_V1_1 = (
    "soz_development_iv_candidate_evidence_authorization_v1_1"
)

AMENDMENT_FILENAME = "amendment.json"
CAPABILITY_MANIFEST_FILENAME = "manifest.json"
BASE_CAPABILITY_DIRECTORY = "base_v1"
AMENDMENT_DIRECTORY = "eligibility_amendment"
_MAX_JSON_BYTES = 64 * 1024 * 1024

# Trust anchors for the one frozen 2026-08-10 development experiment.  A new
# dataset or upstream artifact requires a new schema/protocol, not new CLI
# values under this schema.
FROZEN_TARGET_V2_ARTIFACT_SHA256 = (
    "5c01591c20328fb60817099cac669032bd743e36f47df77ac390842e9a2c67ed"
)
FROZEN_TARGET_V2_RECEIPT_SHA256 = (
    "80f2b71cfdf23d604849b2d1a52cc36f0b01c593906e3cef74e79d425cc442d3"
)
FROZEN_TARGET_V2_POLICY_SHA256 = (
    "bc953272edf638150a7800b01be01261d7b96dfc6db5def5b98cfd6b93dea237"
)
FROZEN_SPLIT_MANIFEST_SHA256 = (
    "5062e894ec139ffaf7abc1b8f45b326f50a118cfcb8907bb25ff81dbbaa91d57"
)
FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256 = (
    "a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66"
)
FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256 = (
    "10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446"
)
FROZEN_SIGNAL_PREPROCESS_CONFIG_SHA256 = (
    "f95ee10a3f67b6864ed2a87c7347f60668b07b854f82a22335edef5008f0111b"
)
FROZEN_SIGNAL_ELIGIBLE_PATIENT_ROSTER_SHA256 = (
    "49ced5020a7df002b61c0dea523c46ab13f2b9bb4f2978ec3f883b68210c682f"
)
FROZEN_SIGNAL_ELIGIBLE_EVENT_ROSTER_SHA256 = (
    "82453898ec09d1420b0d7de1b15b98cab222a1297ff659093ed6131868bad9e8"
)
FROZEN_SIGNAL_EXCLUDED_EVENT_ROSTER_SHA256 = (
    "379352ab0b6737b112ec391d0e01540a3c20997cdf1cd07d2cbbe480ae6c3b2a"
)
FROZEN_SIGNAL_SOURCE_TRAIN_EVENT_SET_SHA256 = (
    "99bd7b66ab5deae080badca71d9c9d0cc2e02c6f939782c9b81681a12a767af8"
)
FROZEN_SIGNAL_SOURCE_DEV_EVENT_SET_SHA256 = (
    "8062e9531c0bc55950d9f60d9f2850e326725b4b6371a12c875f67c63919c4bb"
)
FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256 = (
    "cd1893031873b81053678316ed36145c1ba572d33ae332d221bc0907e1e0bca0"
)
FROZEN_OOF_PROTOCOL_RECEIPT_SHA256 = (
    "a1668bfaa9b3489851251924d618e2c107503455183bf54e0b44ae1613ed4803"
)
FROZEN_BASE_V1_MANIFEST_SHA256 = (
    "c5391d600fba82be1d1e07796c81e51f9ca3a88979074a340c20a1c574cf9214"
)
FROZEN_BASE_V1_AUTHORIZATION_RECEIPT_SHA256 = (
    "dbc56ff88473ff191aacdd3022784d81b1a441fa80d9690d170f6369cd60c654"
)
FROZEN_AMENDMENT_ARTIFACT_SHA256 = (
    "b8246914fa4103117108de64e9f8244987ce1d3c39b40c10e92d4bdc94744237"
)
FROZEN_AMENDMENT_RECEIPT_SHA256 = (
    "2aea0dc246059d77657a4221f19e8d8056ff4dc743e40534d3933ff8dae6356a"
)
FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256 = (
    "e3ff6f5bfe9e39f0e59a10522cec93cbbee26bb221813a62efeac174fe791a24"
)
FROZEN_V1_1_AUTHORIZATION_RECEIPT_SHA256 = (
    "f1724244b071baa4d16c550c5f5dd16f29dceb16fb26bd3fb45936e91e3668bc"
)

_AMENDMENT_POLICY = {
    "estimand": "deepsoz_signal_evidence_eligible_complete_patient_bags_v1_1",
    "target_header_roster_is_not_training_roster": True,
    "runtime_intersection_forbidden": True,
    "difference_must_have_zero_signal_eligible_events": True,
    "difference_fixed_before_target_values_are_loaded": True,
    "source_dev_requires_complete_signal_coverage": True,
    "source_eval_used": False,
    "private_used": False,
    "formal_promotion": False,
}
SIGNAL_EVIDENCE_AMENDMENT_POLICY_SHA256 = _v1._canonical_sha256(
    _AMENDMENT_POLICY
)

_CAPABILITY_POLICY_V1_1 = {
    "active_families": list(_v1.ACTIVE_EVIDENCE_FAMILIES),
    "absent_families": list(_v1.ABSENT_EVIDENCE_FAMILIES),
    "base_v1_evidence_is_byte_preserved": True,
    "eligibility_amendment_required": True,
    "target_values_loaded_during_authorization": False,
    "raw_eeg_present": False,
    "foundation_latent_present": False,
    "quality_diagnostics_or_burden_present": False,
    "source_eval_used": False,
    "private_used": False,
    "formal_reasoner_authorized": False,
    "formal_promotion": False,
}
DEVELOPMENT_IV_CAPABILITY_POLICY_SHA256_V1_1 = _v1._canonical_sha256(
    _CAPABILITY_POLICY_V1_1
)

_EXPECTED_EXCLUSIONS = {
    "906": {
        "public_patient_id": "aaaaabiw",
        "oof_fold": 0,
        "events": (("aaaaabiw_s005_t000__ev0000", "insufficient_warmup"),),
    },
    "10088": {
        "public_patient_id": "aaaaaoya",
        "oof_fold": 4,
        "events": (("aaaaaoya_s010_t001__ev0000", "signal_qc"),),
    },
    "11321": {
        "public_patient_id": "aaaaaqtl",
        "oof_fold": 3,
        "events": (("aaaaaqtl_s007_t007__ev0000", "insufficient_warmup"),),
    },
    "13407": {
        "public_patient_id": "aaaaatvr",
        "oof_fold": 3,
        "events": (
            ("aaaaatvr_s001_t000__ev0000", "insufficient_warmup"),
            ("aaaaatvr_s001_t004__ev0000", "insufficient_warmup"),
            ("aaaaatvr_s001_t013__ev0000", "insufficient_warmup"),
        ),
    },
}


def _sha(value: object, *, field_name: str) -> str:
    return _v1._require_sha256(value, field_name=field_name)


def _sorted_roster(values: Sequence[object], *, field_name: str) -> tuple[str, ...]:
    roster = tuple(normalize_patient_id(value) for value in values)
    if not roster or roster != tuple(sorted(set(roster))):
        raise ValueError(f"{field_name} must be non-empty, sorted, and unique")
    return roster


def _roster_sha256(values: Sequence[str]) -> str:
    return _v1._canonical_sha256(tuple(values))


@dataclass(frozen=True)
class SignalEvidenceExcludedPatient:
    patient_id: str
    public_patient_id: str
    model_split: str
    oof_fold: int
    event_ids: tuple[str, ...]
    exclusion_codes: tuple[str, ...]
    event_record_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        patient_id = normalize_patient_id(self.patient_id)
        object.__setattr__(self, "patient_id", patient_id)
        expected = _EXPECTED_EXCLUSIONS.get(patient_id)
        if expected is None:
            raise ValueError("Eligibility amendment contains an unauthorized patient")
        if self.public_patient_id != expected["public_patient_id"]:
            raise ValueError("Excluded patient public identity changed")
        if self.model_split != "source_train" or self.oof_fold != expected["oof_fold"]:
            raise ValueError("Excluded patient split/fold changed")
        if not (
            len(self.event_ids)
            == len(self.exclusion_codes)
            == len(self.event_record_sha256s)
        ):
            raise ValueError("Excluded patient event evidence is misaligned")
        observed = tuple(zip(self.event_ids, self.exclusion_codes))
        if observed != expected["events"]:
            raise ValueError("Excluded patient event/reason evidence changed")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("Excluded patient event IDs repeat")
        for index, value in enumerate(self.event_record_sha256s):
            _sha(value, field_name=f"event_record_sha256s[{index}]")


@dataclass(frozen=True)
class SignalEvidenceEligibilityAmendmentReceipt:
    policy_sha256: str
    oof_protocol_artifact_sha256: str
    oof_protocol_receipt_sha256: str
    signal_preflight_artifact_sha256: str
    signal_preflight_receipt_sha256: str
    signal_preflight_policy: str
    signal_preprocess_config_sha256: str
    signal_eligible_patient_roster_sha256: str
    signal_eligible_event_roster_sha256: str
    signal_excluded_event_roster_sha256: str
    verified_target_v2_artifact_sha256: str
    verified_target_v2_receipt_sha256: str
    verified_target_v2_policy_sha256: str
    split_manifest_sha256: str
    target_header_source_train_patient_ids: tuple[str, ...]
    target_header_source_dev_patient_ids: tuple[str, ...]
    signal_evidence_source_train_patient_ids: tuple[str, ...]
    signal_evidence_source_dev_patient_ids: tuple[str, ...]
    target_header_source_train_roster_sha256: str
    target_header_source_dev_roster_sha256: str
    signal_evidence_source_train_roster_sha256: str
    signal_evidence_source_dev_roster_sha256: str
    signal_evidence_source_train_event_set_sha256: str
    signal_evidence_source_dev_event_set_sha256: str
    excluded_source_train_patients: tuple[SignalEvidenceExcludedPatient, ...]
    target_header_source_train_patient_count: int
    signal_evidence_source_train_patient_count: int
    target_header_source_dev_patient_count: int
    signal_evidence_source_dev_patient_count: int
    signal_evidence_source_train_event_count: int
    signal_evidence_source_dev_event_count: int
    target_values_loaded: bool = False
    target_vectors_loaded: bool = False
    source_eval_used: bool = False
    private_used: bool = False
    formal_reasoner_authorized: bool = False
    formal_promotion: bool = False
    schema_version: str = SIGNAL_EVIDENCE_AMENDMENT_SCHEMA

    def __post_init__(self) -> None:
        sha_fields = (
            "policy_sha256",
            "oof_protocol_artifact_sha256",
            "oof_protocol_receipt_sha256",
            "signal_preflight_artifact_sha256",
            "signal_preflight_receipt_sha256",
            "signal_preprocess_config_sha256",
            "signal_eligible_patient_roster_sha256",
            "signal_eligible_event_roster_sha256",
            "signal_excluded_event_roster_sha256",
            "verified_target_v2_artifact_sha256",
            "verified_target_v2_receipt_sha256",
            "verified_target_v2_policy_sha256",
            "split_manifest_sha256",
            "target_header_source_train_roster_sha256",
            "target_header_source_dev_roster_sha256",
            "signal_evidence_source_train_roster_sha256",
            "signal_evidence_source_dev_roster_sha256",
            "signal_evidence_source_train_event_set_sha256",
            "signal_evidence_source_dev_event_set_sha256",
        )
        for name in sha_fields:
            object.__setattr__(self, name, _sha(getattr(self, name), field_name=name))
        if self.policy_sha256 != SIGNAL_EVIDENCE_AMENDMENT_POLICY_SHA256:
            raise ValueError("Eligibility amendment policy changed")
        if self.verified_target_v2_policy_sha256 != TARGET_V2_POLICY_SHA256:
            raise ValueError("Eligibility amendment target policy changed")
        if self.signal_preflight_policy != (
            "verified_target_v2_direct_physical19_causal_replay_only"
        ):
            raise ValueError("Eligibility amendment signal policy changed")
        frozen = {
            "oof_protocol_artifact_sha256": FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
            "oof_protocol_receipt_sha256": FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
            "signal_preflight_artifact_sha256": FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
            "signal_preflight_receipt_sha256": FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
            "signal_preprocess_config_sha256": FROZEN_SIGNAL_PREPROCESS_CONFIG_SHA256,
            "signal_eligible_patient_roster_sha256": FROZEN_SIGNAL_ELIGIBLE_PATIENT_ROSTER_SHA256,
            "signal_eligible_event_roster_sha256": FROZEN_SIGNAL_ELIGIBLE_EVENT_ROSTER_SHA256,
            "signal_excluded_event_roster_sha256": FROZEN_SIGNAL_EXCLUDED_EVENT_ROSTER_SHA256,
            "signal_evidence_source_train_event_set_sha256": FROZEN_SIGNAL_SOURCE_TRAIN_EVENT_SET_SHA256,
            "signal_evidence_source_dev_event_set_sha256": FROZEN_SIGNAL_SOURCE_DEV_EVENT_SET_SHA256,
            "verified_target_v2_artifact_sha256": FROZEN_TARGET_V2_ARTIFACT_SHA256,
            "verified_target_v2_receipt_sha256": FROZEN_TARGET_V2_RECEIPT_SHA256,
            "verified_target_v2_policy_sha256": FROZEN_TARGET_V2_POLICY_SHA256,
            "split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
        }
        changed = tuple(
            name for name, expected in frozen.items() if getattr(self, name) != expected
        )
        if changed:
            raise ValueError(f"Frozen eligibility lineage changed: {changed}")
        rosters = {}
        for name in (
            "target_header_source_train_patient_ids",
            "target_header_source_dev_patient_ids",
            "signal_evidence_source_train_patient_ids",
            "signal_evidence_source_dev_patient_ids",
        ):
            roster = _sorted_roster(getattr(self, name), field_name=name)
            object.__setattr__(self, name, roster)
            rosters[name] = roster
        expected_counts = {
            "target_header_source_train_patient_count": 69,
            "signal_evidence_source_train_patient_count": 65,
            "target_header_source_dev_patient_count": 16,
            "signal_evidence_source_dev_patient_count": 16,
            "signal_evidence_source_train_event_count": 582,
            "signal_evidence_source_dev_event_count": 221,
        }
        for name, expected in expected_counts.items():
            if getattr(self, name) != expected:
                raise ValueError(f"Eligibility amendment count changed: {name}")
        if len(rosters["target_header_source_train_patient_ids"]) != 69 or len(
            rosters["signal_evidence_source_train_patient_ids"]
        ) != 65:
            raise ValueError("Eligibility amendment source-train roster length changed")
        if len(rosters["target_header_source_dev_patient_ids"]) != 16 or len(
            rosters["signal_evidence_source_dev_patient_ids"]
        ) != 16:
            raise ValueError("Eligibility amendment source-dev roster length changed")
        if rosters["target_header_source_dev_patient_ids"] != rosters[
            "signal_evidence_source_dev_patient_ids"
        ]:
            raise ValueError("Source-dev must retain complete signal coverage")
        train_target = set(rosters["target_header_source_train_patient_ids"])
        train_signal = set(rosters["signal_evidence_source_train_patient_ids"])
        if not train_signal < train_target:
            raise ValueError("Signal source-train roster must be a strict target subset")
        excluded = tuple(self.excluded_source_train_patients)
        if tuple(row.patient_id for row in excluded) != tuple(sorted(_EXPECTED_EXCLUSIONS)):
            raise ValueError("Eligibility amendment excluded-patient roster changed")
        if train_target - train_signal != set(_EXPECTED_EXCLUSIONS):
            raise ValueError("Eligibility amendment is not the frozen 69-to-65 change")
        hash_checks = {
            "target_header_source_train_roster_sha256": rosters[
                "target_header_source_train_patient_ids"
            ],
            "target_header_source_dev_roster_sha256": rosters[
                "target_header_source_dev_patient_ids"
            ],
            "signal_evidence_source_train_roster_sha256": rosters[
                "signal_evidence_source_train_patient_ids"
            ],
            "signal_evidence_source_dev_roster_sha256": rosters[
                "signal_evidence_source_dev_patient_ids"
            ],
        }
        for name, roster in hash_checks.items():
            if getattr(self, name) != _roster_sha256(roster):
                raise ValueError(f"Eligibility amendment roster hash changed: {name}")
        if any(
            (
                self.target_values_loaded,
                self.target_vectors_loaded,
                self.source_eval_used,
                self.private_used,
                self.formal_reasoner_authorized,
                self.formal_promotion,
            )
        ):
            raise ValueError("Eligibility amendment development boundary changed")
        if self.schema_version != SIGNAL_EVIDENCE_AMENDMENT_SCHEMA:
            raise ValueError("Unsupported eligibility amendment schema")

    @property
    def receipt_sha256(self) -> str:
        return _v1._canonical_sha256(asdict(self))


@dataclass(frozen=True)
class PublishedSignalEvidenceEligibilityAmendment:
    path: Path
    artifact_sha256: str
    receipt: SignalEvidenceEligibilityAmendmentReceipt

    @property
    def receipt_sha256(self) -> str:
        return self.receipt.receipt_sha256


def _derive_amendment_receipt(
    *,
    signal_receipt: Mapping[str, object],
    signal_artifact_sha256: str,
    signal_receipt_sha256: str,
    protocol_artifact_sha256: str,
    protocol_receipt_sha256: str,
    source_train_patient_ids: Sequence[str],
    source_dev_patient_ids: Sequence[str],
    public_crosswalk: Mapping[str, str],
    folds_by_patient: Mapping[str, int],
) -> SignalEvidenceEligibilityAmendmentReceipt:
    target_train = _sorted_roster(
        source_train_patient_ids, field_name="OOF source-train roster"
    )
    target_dev = _sorted_roster(
        source_dev_patient_ids, field_name="OOF source-dev roster"
    )
    raw_split_rosters = signal_receipt.get("eligible_split_patient_ids")
    if not isinstance(raw_split_rosters, list):
        raise ValueError("Signal receipt split roster is missing")
    split_rosters: dict[str, tuple[str, ...]] = {}
    for raw in raw_split_rosters:
        if not isinstance(raw, list) or len(raw) != 2 or not isinstance(raw[1], list):
            raise ValueError("Signal receipt split roster has invalid structure")
        name = str(raw[0])
        if name in split_rosters:
            raise ValueError("Signal receipt repeats a split roster")
        split_rosters[name] = _sorted_roster(
            raw[1], field_name=f"signal {name} roster"
        )
    if set(split_rosters) != {"source_train", "source_dev", "source_eval"}:
        raise ValueError("Signal receipt split keys changed")
    signal_train = split_rosters["source_train"]
    signal_dev = split_rosters["source_dev"]
    if not set(signal_train) <= set(target_train):
        raise ValueError("Signal source-train roster is outside OOF target roster")
    if signal_dev != target_dev:
        raise ValueError("Signal source-dev roster is not complete")
    omitted = tuple(sorted(set(target_train) - set(signal_train)))
    if omitted != tuple(sorted(_EXPECTED_EXCLUSIONS)):
        raise ValueError("Signal receipt does not produce the frozen four-patient difference")

    accepted_rows = signal_receipt.get("events")
    excluded_rows = signal_receipt.get("exclusions")
    if not isinstance(accepted_rows, list) or not isinstance(excluded_rows, list):
        raise ValueError("Signal receipt event evidence is missing")
    accepted_by_patient: dict[str, list[Mapping[str, object]]] = {}
    excluded_by_patient: dict[str, list[Mapping[str, object]]] = {}
    all_event_ids: set[str] = set()
    for raw, destination in (
        *((row, accepted_by_patient) for row in accepted_rows),
        *((row, excluded_by_patient) for row in excluded_rows),
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("Signal receipt event row is invalid")
        event_id = str(raw.get("event_id", ""))
        if not event_id or event_id in all_event_ids:
            raise ValueError("Signal receipt event IDs are empty or duplicated")
        all_event_ids.add(event_id)
        patient_id = normalize_patient_id(raw.get("patient_id", ""))
        destination.setdefault(patient_id, []).append(raw)

    exclusions: list[SignalEvidenceExcludedPatient] = []
    for patient_id in omitted:
        if accepted_by_patient.get(patient_id):
            raise ValueError("Amendment cannot exclude a patient with accepted signal")
        rows = sorted(
            excluded_by_patient.get(patient_id, []), key=lambda row: str(row["event_id"])
        )
        if not rows:
            raise ValueError("Amendment exclusion lacks candidate-event evidence")
        if any(str(row.get("model_split")) != "source_train" for row in rows):
            raise ValueError("Amendment exclusion crossed model split")
        expected_public = str(public_crosswalk.get(patient_id, ""))
        if not expected_public or any(
            str(row.get("local_patient_id")) != expected_public for row in rows
        ):
            raise ValueError("Amendment exclusion public crosswalk changed")
        exclusions.append(
            SignalEvidenceExcludedPatient(
                patient_id=patient_id,
                public_patient_id=expected_public,
                model_split="source_train",
                oof_fold=int(folds_by_patient[patient_id]),
                event_ids=tuple(str(row["event_id"]) for row in rows),
                exclusion_codes=tuple(str(row["eligibility_code"]) for row in rows),
                event_record_sha256s=tuple(
                    str(row["event_record_sha256"]) for row in rows
                ),
            )
        )

    accepted_train_ids = tuple(
        sorted(
            str(row["event_id"])
            for row in accepted_rows
            if str(row.get("model_split")) == "source_train"
        )
    )
    accepted_dev_ids = tuple(
        sorted(
            str(row["event_id"])
            for row in accepted_rows
            if str(row.get("model_split")) == "source_dev"
        )
    )
    if len(accepted_train_ids) != len(set(accepted_train_ids)) or len(
        accepted_dev_ids
    ) != len(set(accepted_dev_ids)):
        raise ValueError("Signal accepted-event set contains duplicates")

    return SignalEvidenceEligibilityAmendmentReceipt(
        policy_sha256=SIGNAL_EVIDENCE_AMENDMENT_POLICY_SHA256,
        oof_protocol_artifact_sha256=protocol_artifact_sha256,
        oof_protocol_receipt_sha256=protocol_receipt_sha256,
        signal_preflight_artifact_sha256=signal_artifact_sha256,
        signal_preflight_receipt_sha256=signal_receipt_sha256,
        signal_preflight_policy=str(signal_receipt["policy"]),
        signal_preprocess_config_sha256=str(
            signal_receipt["preprocess_config_sha256"]
        ),
        signal_eligible_patient_roster_sha256=str(
            signal_receipt["eligible_patient_roster_sha256"]
        ),
        signal_eligible_event_roster_sha256=str(
            signal_receipt["eligible_event_roster_sha256"]
        ),
        signal_excluded_event_roster_sha256=str(
            signal_receipt["excluded_event_roster_sha256"]
        ),
        verified_target_v2_artifact_sha256=str(
            signal_receipt["verified_target_v2_artifact_sha256"]
        ),
        verified_target_v2_receipt_sha256=str(
            signal_receipt["verified_target_v2_receipt_sha256"]
        ),
        verified_target_v2_policy_sha256=str(
            signal_receipt["verified_target_v2_policy_sha256"]
        ),
        split_manifest_sha256=str(signal_receipt["split_manifest_sha256"]),
        target_header_source_train_patient_ids=target_train,
        target_header_source_dev_patient_ids=target_dev,
        signal_evidence_source_train_patient_ids=signal_train,
        signal_evidence_source_dev_patient_ids=signal_dev,
        target_header_source_train_roster_sha256=_roster_sha256(target_train),
        target_header_source_dev_roster_sha256=_roster_sha256(target_dev),
        signal_evidence_source_train_roster_sha256=_roster_sha256(signal_train),
        signal_evidence_source_dev_roster_sha256=_roster_sha256(signal_dev),
        signal_evidence_source_train_event_set_sha256=_v1._canonical_sha256(
            accepted_train_ids
        ),
        signal_evidence_source_dev_event_set_sha256=_v1._canonical_sha256(
            accepted_dev_ids
        ),
        excluded_source_train_patients=tuple(exclusions),
        target_header_source_train_patient_count=len(target_train),
        signal_evidence_source_train_patient_count=len(signal_train),
        target_header_source_dev_patient_count=len(target_dev),
        signal_evidence_source_dev_patient_count=len(signal_dev),
        signal_evidence_source_train_event_count=len(accepted_train_ids),
        signal_evidence_source_dev_event_count=len(accepted_dev_ids),
    )


def build_signal_evidence_eligibility_amendment(
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    *,
    expected_target_v2_artifact_sha256: str,
    expected_target_v2_receipt_sha256: str,
    expected_target_v2_policy_sha256: str,
) -> SignalEvidenceEligibilityAmendmentReceipt:
    """Derive the frozen 69-to-65 amendment without opening target values."""

    if type(signal) is not VerifiedDeepSOZSignalPreflightBundle:
        raise TypeError("Amendment requires a strictly replayed signal bundle")
    if type(protocol) is not TargetFreeOOFProtocolView:
        raise TypeError("Amendment requires the target-free OOF protocol")
    protocol.assert_unchanged()
    if _v1._canonical_sha256(dict(signal.receipt)) != signal.receipt_sha256:
        raise ValueError("Signal receipt changed in memory after strict verification")
    frozen_inputs = {
        "signal artifact": (
            signal.artifact_sha256,
            FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
        ),
        "signal receipt": (
            signal.receipt_sha256,
            FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
        ),
        "OOF artifact": (
            protocol.artifact_sha256,
            FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
        ),
        "OOF receipt": (
            protocol.receipt_sha256,
            FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
        ),
    }
    changed = tuple(
        name for name, (actual, expected) in frozen_inputs.items() if actual != expected
    )
    if changed:
        raise ValueError(f"Frozen amendment input changed: {changed}")
    target_bindings = {
        "verified_target_v2_artifact_sha256": _sha(
            expected_target_v2_artifact_sha256,
            field_name="expected_target_v2_artifact_sha256",
        ),
        "verified_target_v2_receipt_sha256": _sha(
            expected_target_v2_receipt_sha256,
            field_name="expected_target_v2_receipt_sha256",
        ),
        "verified_target_v2_policy_sha256": _sha(
            expected_target_v2_policy_sha256,
            field_name="expected_target_v2_policy_sha256",
        ),
    }
    if target_bindings != {
        "verified_target_v2_artifact_sha256": FROZEN_TARGET_V2_ARTIFACT_SHA256,
        "verified_target_v2_receipt_sha256": FROZEN_TARGET_V2_RECEIPT_SHA256,
        "verified_target_v2_policy_sha256": FROZEN_TARGET_V2_POLICY_SHA256,
    }:
        raise ValueError("CLI target trust anchors differ from the frozen experiment")
    for name, expected in target_bindings.items():
        if signal.receipt.get(name) != expected:
            raise ValueError(f"Amendment target-free binding changed: {name}")
    if signal.receipt.get("split_manifest_sha256") != (
        protocol.receipt.split_manifest_sha256
    ):
        raise ValueError("Amendment signal and OOF split lineage disagree")
    folds = {
        patient_id: protocol.fold_for_target(patient_id)
        for patient_id in protocol.receipt.source_train_patient_ids
    }
    return _derive_amendment_receipt(
        signal_receipt=signal.receipt,
        signal_artifact_sha256=signal.artifact_sha256,
        signal_receipt_sha256=signal.receipt_sha256,
        protocol_artifact_sha256=protocol.artifact_sha256,
        protocol_receipt_sha256=protocol.receipt_sha256,
        source_train_patient_ids=protocol.receipt.source_train_patient_ids,
        source_dev_patient_ids=protocol.receipt.source_dev_patient_ids,
        public_crosswalk=protocol.crosswalk,
        folds_by_patient=folds,
    )


def _safe_new_directory(path: str | Path, *, field_name: str) -> Path:
    target = Path(os.path.abspath(path))
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"{field_name} already exists")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise ValueError(f"{field_name} parent must be a regular directory")
    return target


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _amendment_payload(
    receipt: SignalEvidenceEligibilityAmendmentReceipt,
) -> dict[str, object]:
    return {
        "schema_version": SIGNAL_EVIDENCE_AMENDMENT_ARTIFACT_SCHEMA,
        "purpose": "target_free_signal_evidence_eligibility_amendment_only",
        "serialization": "canonical_json_utf8_no_pickle",
        "policy": dict(_AMENDMENT_POLICY),
        "policy_sha256": SIGNAL_EVIDENCE_AMENDMENT_POLICY_SHA256,
        "receipt": asdict(receipt),
        "receipt_sha256": receipt.receipt_sha256,
        "target_values_loaded": False,
        "target_vectors_loaded": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
    }


def publish_signal_evidence_eligibility_amendment(
    receipt: SignalEvidenceEligibilityAmendmentReceipt,
    output_directory: str | Path,
) -> PublishedSignalEvidenceEligibilityAmendment:
    if type(receipt) is not SignalEvidenceEligibilityAmendmentReceipt:
        raise TypeError("Only a verified amendment receipt may be published")
    target = _safe_new_directory(output_directory, field_name="Amendment output")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        raw = _v1._canonical_json_bytes(_amendment_payload(receipt))
        path = temporary / AMENDMENT_FILENAME
        path.write_bytes(raw)
        _fsync_file(path)
        _fsync_directory(temporary)
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        artifact_sha = hashlib.sha256(raw).hexdigest()
        if artifact_sha != FROZEN_AMENDMENT_ARTIFACT_SHA256 or (
            receipt.receipt_sha256 != FROZEN_AMENDMENT_RECEIPT_SHA256
        ):
            raise RuntimeError("Published amendment differs from the frozen v1.1 artifact")
        return PublishedSignalEvidenceEligibilityAmendment(
            path=target,
            artifact_sha256=artifact_sha,
            receipt=receipt,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _receipt_from_payload(
    value: object,
) -> SignalEvidenceEligibilityAmendmentReceipt:
    if not isinstance(value, dict):
        raise ValueError("Amendment receipt is not an object")
    expected = set(SignalEvidenceEligibilityAmendmentReceipt.__dataclass_fields__)
    if set(value) != expected:
        raise ValueError("Amendment receipt violates its closed schema")
    payload = dict(value)
    raw_exclusions = payload.get("excluded_source_train_patients")
    if not isinstance(raw_exclusions, list):
        raise ValueError("Amendment excluded-patient evidence is not an array")
    try:
        normalized_exclusions = []
        for row in raw_exclusions:
            if not isinstance(row, dict) or set(row) != set(
                SignalEvidenceExcludedPatient.__dataclass_fields__
            ):
                raise ValueError("Excluded-patient row violates its closed schema")
            normalized = dict(row)
            for name in ("event_ids", "exclusion_codes", "event_record_sha256s"):
                normalized[name] = tuple(normalized[name])
            normalized_exclusions.append(SignalEvidenceExcludedPatient(**normalized))
        payload["excluded_source_train_patients"] = tuple(normalized_exclusions)
        for name in (
            "target_header_source_train_patient_ids",
            "target_header_source_dev_patient_ids",
            "signal_evidence_source_train_patient_ids",
            "signal_evidence_source_dev_patient_ids",
        ):
            payload[name] = tuple(payload[name])
        return SignalEvidenceEligibilityAmendmentReceipt(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Amendment receipt failed reconstruction") from exc


def load_signal_evidence_eligibility_amendment(
    bundle_directory: str | Path,
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    *,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> PublishedSignalEvidenceEligibilityAmendment:
    source = Path(os.path.abspath(bundle_directory))
    if source.is_symlink() or not source.is_dir() or {
        path.name for path in source.iterdir()
    } != {AMENDMENT_FILENAME}:
        raise ValueError("Amendment bundle violates its closed directory schema")
    path = source / AMENDMENT_FILENAME
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= _MAX_JSON_BYTES:
        raise ValueError("Amendment artifact must be a bounded regular file")
    raw = path.read_bytes()
    artifact_sha = hashlib.sha256(raw).hexdigest()
    if artifact_sha != _sha(
        expected_artifact_sha256, field_name="expected_artifact_sha256"
    ):
        raise ValueError("Amendment artifact SHA mismatch")
    if artifact_sha != FROZEN_AMENDMENT_ARTIFACT_SHA256:
        raise ValueError("Amendment artifact is not the frozen v1.1 artifact")
    payload = _v1._strict_json(raw, field_name="Eligibility amendment")
    if not isinstance(payload, dict) or _v1._canonical_json_bytes(payload) != raw:
        raise ValueError("Amendment artifact is not canonical JSON")
    expected_fields = {
        "schema_version",
        "purpose",
        "serialization",
        "policy",
        "policy_sha256",
        "receipt",
        "receipt_sha256",
        "target_values_loaded",
        "target_vectors_loaded",
        "source_eval_used",
        "private_used",
        "formal_reasoner_authorized",
        "formal_promotion",
    }
    if set(payload) != expected_fields:
        raise ValueError("Amendment artifact violates its closed schema")
    fixed = {
        "schema_version": SIGNAL_EVIDENCE_AMENDMENT_ARTIFACT_SCHEMA,
        "purpose": "target_free_signal_evidence_eligibility_amendment_only",
        "serialization": "canonical_json_utf8_no_pickle",
        "policy": _AMENDMENT_POLICY,
        "policy_sha256": SIGNAL_EVIDENCE_AMENDMENT_POLICY_SHA256,
        "target_values_loaded": False,
        "target_vectors_loaded": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
    }
    if any(payload.get(name) != value for name, value in fixed.items()):
        raise ValueError("Amendment artifact scientific boundary changed")
    receipt = _receipt_from_payload(payload["receipt"])
    expected_receipt = _sha(
        expected_receipt_sha256, field_name="expected_receipt_sha256"
    )
    if payload["receipt_sha256"] != expected_receipt or (
        receipt.receipt_sha256 != expected_receipt
    ):
        raise ValueError("Amendment receipt SHA mismatch")
    if expected_receipt != FROZEN_AMENDMENT_RECEIPT_SHA256:
        raise ValueError("Amendment receipt is not the frozen v1.1 receipt")
    replay = build_signal_evidence_eligibility_amendment(
        signal,
        protocol,
        expected_target_v2_artifact_sha256=receipt.verified_target_v2_artifact_sha256,
        expected_target_v2_receipt_sha256=receipt.verified_target_v2_receipt_sha256,
        expected_target_v2_policy_sha256=receipt.verified_target_v2_policy_sha256,
    )
    if replay != receipt:
        raise ValueError("Amendment did not replay from target-free sources")
    return PublishedSignalEvidenceEligibilityAmendment(
        path=source, artifact_sha256=artifact_sha, receipt=receipt
    )


@dataclass(frozen=True)
class DevelopmentIVEvidenceAuthorizationReceiptV11:
    policy_sha256: str
    base_v1_manifest_sha256: str
    base_v1_authorization_receipt_sha256: str
    amendment_artifact_sha256: str
    amendment_receipt_sha256: str
    signal_preflight_artifact_sha256: str
    signal_preflight_receipt_sha256: str
    oof_protocol_artifact_sha256: str
    oof_protocol_receipt_sha256: str
    verified_target_v2_artifact_sha256: str
    verified_target_v2_receipt_sha256: str
    verified_target_v2_policy_sha256: str
    source_train_evidence_receipt_sha256: str
    source_dev_evidence_receipt_sha256: str
    source_train_event_roster_sha256: str
    source_dev_event_roster_sha256: str
    source_train_event_set_sha256: str
    source_dev_event_set_sha256: str
    source_train_event_count: int
    source_dev_event_count: int
    source_train_patient_ids: tuple[str, ...]
    source_dev_patient_ids: tuple[str, ...]
    source_train_patient_roster_sha256: str
    source_dev_patient_roster_sha256: str
    target_values_loaded: bool = False
    candidate_reasoner_input_authorized: bool = True
    upstream_ictal_reasoner_authorized: bool = False
    morphology_present: bool = False
    source_eval_used: bool = False
    private_used: bool = False
    formal_reasoner_authorized: bool = False
    formal_promotion: bool = False
    schema_version: str = DEVELOPMENT_IV_CAPABILITY_AUTHORIZATION_SCHEMA_V1_1

    def __post_init__(self) -> None:
        for name in (
            "policy_sha256",
            "base_v1_manifest_sha256",
            "base_v1_authorization_receipt_sha256",
            "amendment_artifact_sha256",
            "amendment_receipt_sha256",
            "signal_preflight_artifact_sha256",
            "signal_preflight_receipt_sha256",
            "oof_protocol_artifact_sha256",
            "oof_protocol_receipt_sha256",
            "verified_target_v2_artifact_sha256",
            "verified_target_v2_receipt_sha256",
            "verified_target_v2_policy_sha256",
            "source_train_evidence_receipt_sha256",
            "source_dev_evidence_receipt_sha256",
            "source_train_event_roster_sha256",
            "source_dev_event_roster_sha256",
            "source_train_event_set_sha256",
            "source_dev_event_set_sha256",
            "source_train_patient_roster_sha256",
            "source_dev_patient_roster_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), field_name=name))
        if self.policy_sha256 != DEVELOPMENT_IV_CAPABILITY_POLICY_SHA256_V1_1:
            raise ValueError("v1.1 capability policy changed")
        frozen = {
            "base_v1_manifest_sha256": FROZEN_BASE_V1_MANIFEST_SHA256,
            "base_v1_authorization_receipt_sha256": FROZEN_BASE_V1_AUTHORIZATION_RECEIPT_SHA256,
            "signal_preflight_artifact_sha256": FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
            "signal_preflight_receipt_sha256": FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
            "oof_protocol_artifact_sha256": FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
            "oof_protocol_receipt_sha256": FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
            "verified_target_v2_artifact_sha256": FROZEN_TARGET_V2_ARTIFACT_SHA256,
            "verified_target_v2_receipt_sha256": FROZEN_TARGET_V2_RECEIPT_SHA256,
            "verified_target_v2_policy_sha256": FROZEN_TARGET_V2_POLICY_SHA256,
            "amendment_artifact_sha256": FROZEN_AMENDMENT_ARTIFACT_SHA256,
            "amendment_receipt_sha256": FROZEN_AMENDMENT_RECEIPT_SHA256,
        }
        changed = tuple(
            name for name, expected in frozen.items() if getattr(self, name) != expected
        )
        if changed:
            raise ValueError(f"Frozen v1.1 capability lineage changed: {changed}")
        train = _sorted_roster(self.source_train_patient_ids, field_name="source_train_patient_ids")
        dev = _sorted_roster(self.source_dev_patient_ids, field_name="source_dev_patient_ids")
        object.__setattr__(self, "source_train_patient_ids", train)
        object.__setattr__(self, "source_dev_patient_ids", dev)
        if len(train) != 65 or len(dev) != 16 or set(train) & set(dev):
            raise ValueError("v1.1 capability patient roster changed")
        if self.source_train_event_count != 582 or self.source_dev_event_count != 221:
            raise ValueError("v1.1 capability event count changed")
        if self.source_train_event_set_sha256 != (
            FROZEN_SIGNAL_SOURCE_TRAIN_EVENT_SET_SHA256
        ) or self.source_dev_event_set_sha256 != (
            FROZEN_SIGNAL_SOURCE_DEV_EVENT_SET_SHA256
        ):
            raise ValueError("v1.1 capability event set changed")
        if self.source_train_patient_roster_sha256 != _roster_sha256(train) or (
            self.source_dev_patient_roster_sha256 != _roster_sha256(dev)
        ):
            raise ValueError("v1.1 capability patient roster SHA changed")
        if self.verified_target_v2_policy_sha256 != TARGET_V2_POLICY_SHA256:
            raise ValueError("v1.1 capability target policy changed")
        if any(
            (
                self.target_values_loaded,
                self.upstream_ictal_reasoner_authorized,
                self.morphology_present,
                self.source_eval_used,
                self.private_used,
                self.formal_reasoner_authorized,
                self.formal_promotion,
            )
        ) or not self.candidate_reasoner_input_authorized:
            raise ValueError("v1.1 capability development boundary changed")
        if self.schema_version != DEVELOPMENT_IV_CAPABILITY_AUTHORIZATION_SCHEMA_V1_1:
            raise ValueError("Unsupported v1.1 capability authorization schema")

    @property
    def receipt_sha256(self) -> str:
        return _v1._canonical_sha256(asdict(self))


@dataclass(frozen=True)
class VerifiedDevelopmentIVEvidenceCapabilityV11:
    base: _v1.PublishedDevelopmentIVEvidenceCapability = field(repr=False)
    amendment: PublishedSignalEvidenceEligibilityAmendment
    receipt: DevelopmentIVEvidenceAuthorizationReceiptV11

    def assert_unchanged(self) -> None:
        self.base.capability.assert_unchanged()
        base_receipt = self.base.capability.receipt
        amendment = self.amendment.receipt
        checks = {
            "base manifest": self.base.manifest_sha256
            == self.receipt.base_v1_manifest_sha256,
            "base authorization": base_receipt.receipt_sha256
            == self.receipt.base_v1_authorization_receipt_sha256,
            "amendment artifact": self.amendment.artifact_sha256
            == self.receipt.amendment_artifact_sha256,
            "amendment receipt": amendment.receipt_sha256
            == self.receipt.amendment_receipt_sha256,
            "train evidence": base_receipt.source_train_evidence_receipt_sha256
            == self.receipt.source_train_evidence_receipt_sha256,
            "dev evidence": base_receipt.source_dev_evidence_receipt_sha256
            == self.receipt.source_dev_evidence_receipt_sha256,
            "train events": base_receipt.source_train_event_roster_sha256
            == self.receipt.source_train_event_roster_sha256,
            "dev events": base_receipt.source_dev_event_roster_sha256
            == self.receipt.source_dev_event_roster_sha256,
            "train event set": _v1._canonical_sha256(
                tuple(sorted(self.base.capability.source_train.event_ids))
            )
            == self.receipt.source_train_event_set_sha256
            == amendment.signal_evidence_source_train_event_set_sha256,
            "dev event set": _v1._canonical_sha256(
                tuple(sorted(self.base.capability.source_dev.event_ids))
            )
            == self.receipt.source_dev_event_set_sha256
            == amendment.signal_evidence_source_dev_event_set_sha256,
            "train patients": self.base.capability.source_train.patient_ids
            == self.receipt.source_train_patient_ids
            == amendment.signal_evidence_source_train_patient_ids,
            "dev patients": self.base.capability.source_dev.patient_ids
            == self.receipt.source_dev_patient_ids
            == amendment.signal_evidence_source_dev_patient_ids,
        }
        failed = tuple(name for name, value in checks.items() if not value)
        if failed:
            raise ValueError(f"v1.1 capability changed after authorization: {failed}")


@dataclass(frozen=True)
class PublishedDevelopmentIVEvidenceCapabilityV11:
    path: Path
    manifest_sha256: str
    authorization_receipt_sha256: str
    capability: VerifiedDevelopmentIVEvidenceCapabilityV11 = field(repr=False)


def issue_development_iv_evidence_capability_v1_1(
    base: _v1.PublishedDevelopmentIVEvidenceCapability,
    amendment: PublishedSignalEvidenceEligibilityAmendment,
) -> VerifiedDevelopmentIVEvidenceCapabilityV11:
    if type(base) is not _v1.PublishedDevelopmentIVEvidenceCapability:
        raise TypeError("v1.1 authorization requires the strict published v1 capability")
    if type(amendment) is not PublishedSignalEvidenceEligibilityAmendment:
        raise TypeError("v1.1 authorization requires the strict eligibility amendment")
    base.capability.assert_unchanged()
    if base.manifest_sha256 != FROZEN_BASE_V1_MANIFEST_SHA256 or (
        base.authorization_receipt_sha256
        != FROZEN_BASE_V1_AUTHORIZATION_RECEIPT_SHA256
    ):
        raise ValueError("Frozen base-v1 capability trust anchor changed")
    old = base.capability.receipt
    amended = amendment.receipt
    lineage = {
        "target artifact": old.verified_target_v2_artifact_sha256
        == amended.verified_target_v2_artifact_sha256,
        "target receipt": old.verified_target_v2_receipt_sha256
        == amended.verified_target_v2_receipt_sha256,
        "target policy": old.verified_target_v2_policy_sha256
        == amended.verified_target_v2_policy_sha256,
        "train evidence roster": base.capability.source_train.patient_ids
        == amended.signal_evidence_source_train_patient_ids,
        "dev evidence roster": base.capability.source_dev.patient_ids
        == amended.signal_evidence_source_dev_patient_ids,
    }
    failed = tuple(name for name, value in lineage.items() if not value)
    if failed:
        raise ValueError(f"Base v1 capability and amendment disagree: {failed}")
    receipt = DevelopmentIVEvidenceAuthorizationReceiptV11(
        policy_sha256=DEVELOPMENT_IV_CAPABILITY_POLICY_SHA256_V1_1,
        base_v1_manifest_sha256=base.manifest_sha256,
        base_v1_authorization_receipt_sha256=old.receipt_sha256,
        amendment_artifact_sha256=amendment.artifact_sha256,
        amendment_receipt_sha256=amended.receipt_sha256,
        signal_preflight_artifact_sha256=amended.signal_preflight_artifact_sha256,
        signal_preflight_receipt_sha256=amended.signal_preflight_receipt_sha256,
        oof_protocol_artifact_sha256=amended.oof_protocol_artifact_sha256,
        oof_protocol_receipt_sha256=amended.oof_protocol_receipt_sha256,
        verified_target_v2_artifact_sha256=amended.verified_target_v2_artifact_sha256,
        verified_target_v2_receipt_sha256=amended.verified_target_v2_receipt_sha256,
        verified_target_v2_policy_sha256=amended.verified_target_v2_policy_sha256,
        source_train_evidence_receipt_sha256=old.source_train_evidence_receipt_sha256,
        source_dev_evidence_receipt_sha256=old.source_dev_evidence_receipt_sha256,
        source_train_event_roster_sha256=old.source_train_event_roster_sha256,
        source_dev_event_roster_sha256=old.source_dev_event_roster_sha256,
        source_train_event_set_sha256=_v1._canonical_sha256(
            tuple(sorted(base.capability.source_train.event_ids))
        ),
        source_dev_event_set_sha256=_v1._canonical_sha256(
            tuple(sorted(base.capability.source_dev.event_ids))
        ),
        source_train_event_count=len(base.capability.source_train.event_ids),
        source_dev_event_count=len(base.capability.source_dev.event_ids),
        source_train_patient_ids=base.capability.source_train.patient_ids,
        source_dev_patient_ids=base.capability.source_dev.patient_ids,
        source_train_patient_roster_sha256=_roster_sha256(
            base.capability.source_train.patient_ids
        ),
        source_dev_patient_roster_sha256=_roster_sha256(
            base.capability.source_dev.patient_ids
        ),
    )
    result = VerifiedDevelopmentIVEvidenceCapabilityV11(
        base=base, amendment=amendment, receipt=receipt
    )
    result.assert_unchanged()
    return result


def _copy_regular(source: Path, destination: Path) -> dict[str, object]:
    if source.is_symlink() or not source.is_file():
        raise ValueError("v1.1 capability source must be a regular file")
    shutil.copyfile(source, destination)
    return {"sha256": _v1._file_sha256(destination), "size_bytes": destination.stat().st_size}


def _capability_manifest_v1_1(
    capability: VerifiedDevelopmentIVEvidenceCapabilityV11,
    *,
    files: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": DEVELOPMENT_IV_CAPABILITY_SCHEMA_V1_1,
        "purpose": "development_iv_candidate_reasoner_evidence_with_signed_signal_eligibility",
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "authorization_policy": dict(_CAPABILITY_POLICY_V1_1),
        "authorization_policy_sha256": DEVELOPMENT_IV_CAPABILITY_POLICY_SHA256_V1_1,
        "authorization_receipt": asdict(capability.receipt),
        "authorization_receipt_sha256": capability.receipt.receipt_sha256,
        "active_evidence_families": list(_v1.ACTIVE_EVIDENCE_FAMILIES),
        "absent_evidence_families": list(_v1.ABSENT_EVIDENCE_FAMILIES),
        "development_only": True,
        "target_values_loaded": False,
        "candidate_reasoner_input_authorized": True,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
        "source_eval_used": False,
        "private_used": False,
        "base_v1_directory": BASE_CAPABILITY_DIRECTORY,
        "amendment_directory": AMENDMENT_DIRECTORY,
        "files": dict(files),
    }


def publish_development_iv_evidence_capability_v1_1(
    capability: VerifiedDevelopmentIVEvidenceCapabilityV11,
    output_directory: str | Path,
) -> PublishedDevelopmentIVEvidenceCapabilityV11:
    if type(capability) is not VerifiedDevelopmentIVEvidenceCapabilityV11:
        raise TypeError("Only the closed v1.1 issuer may publish a capability")
    capability.assert_unchanged()
    target = _safe_new_directory(output_directory, field_name="v1.1 capability output")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        base_dir = temporary / BASE_CAPABILITY_DIRECTORY
        amendment_dir = temporary / AMENDMENT_DIRECTORY
        base_dir.mkdir()
        amendment_dir.mkdir()
        files: dict[str, object] = {}
        for name in (
            _v1.DEVELOPMENT_IV_CAPABILITY_MANIFEST_FILENAME,
            _v1.DEVELOPMENT_IV_CAPABILITY_EVENTS_FILENAME,
            _v1.DEVELOPMENT_IV_CAPABILITY_TENSORS_FILENAME,
        ):
            relative = f"{BASE_CAPABILITY_DIRECTORY}/{name}"
            files[relative] = _copy_regular(
                capability.base.path / name, base_dir / name
            )
        files[f"{AMENDMENT_DIRECTORY}/{AMENDMENT_FILENAME}"] = _copy_regular(
            capability.amendment.path / AMENDMENT_FILENAME,
            amendment_dir / AMENDMENT_FILENAME,
        )
        manifest = _capability_manifest_v1_1(capability, files=files)
        raw = _v1._canonical_json_bytes(manifest)
        manifest_path = temporary / CAPABILITY_MANIFEST_FILENAME
        manifest_path.write_bytes(raw)
        for path in (
            manifest_path,
            *(base_dir.iterdir()),
            *(amendment_dir.iterdir()),
        ):
            _fsync_file(path)
        _fsync_directory(base_dir)
        _fsync_directory(amendment_dir)
        _fsync_directory(temporary)
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        manifest_sha = hashlib.sha256(raw).hexdigest()
        if manifest_sha != FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256 or (
            capability.receipt.receipt_sha256
            != FROZEN_V1_1_AUTHORIZATION_RECEIPT_SHA256
        ):
            raise RuntimeError("Published capability differs from frozen v1.1")
        return PublishedDevelopmentIVEvidenceCapabilityV11(
            path=target,
            manifest_sha256=manifest_sha,
            authorization_receipt_sha256=capability.receipt.receipt_sha256,
            capability=capability,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _authorization_v1_1_from_payload(
    value: object,
) -> DevelopmentIVEvidenceAuthorizationReceiptV11:
    if not isinstance(value, dict) or set(value) != set(
        DevelopmentIVEvidenceAuthorizationReceiptV11.__dataclass_fields__
    ):
        raise ValueError("v1.1 authorization receipt violates its closed schema")
    payload = dict(value)
    payload["source_train_patient_ids"] = tuple(payload["source_train_patient_ids"])
    payload["source_dev_patient_ids"] = tuple(payload["source_dev_patient_ids"])
    try:
        return DevelopmentIVEvidenceAuthorizationReceiptV11(**payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("v1.1 authorization receipt failed reconstruction") from exc


def load_development_iv_evidence_capability_v1_1(
    bundle_directory: str | Path,
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    *,
    expected_manifest_sha256: str,
) -> PublishedDevelopmentIVEvidenceCapabilityV11:
    source = Path(os.path.abspath(bundle_directory))
    if source.is_symlink() or not source.is_dir() or {
        path.name for path in source.iterdir()
    } != {CAPABILITY_MANIFEST_FILENAME, BASE_CAPABILITY_DIRECTORY, AMENDMENT_DIRECTORY}:
        raise ValueError("v1.1 capability violates its closed directory schema")
    if (source / BASE_CAPABILITY_DIRECTORY).is_symlink() or (
        source / AMENDMENT_DIRECTORY
    ).is_symlink():
        raise ValueError("v1.1 capability nested directories cannot be symlinks")
    manifest_path = source / CAPABILITY_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file() or not (
        1 <= manifest_path.stat().st_size <= _MAX_JSON_BYTES
    ):
        raise ValueError("v1.1 capability manifest must be a bounded regular file")
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha != _sha(
        expected_manifest_sha256, field_name="expected_manifest_sha256"
    ):
        raise ValueError("v1.1 capability manifest SHA mismatch")
    if manifest_sha != FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256:
        raise ValueError("Capability manifest is not the frozen v1.1 artifact")
    manifest = _v1._strict_json(raw, field_name="v1.1 capability manifest")
    if not isinstance(manifest, dict) or _v1._canonical_json_bytes(manifest) != raw:
        raise ValueError("v1.1 capability manifest is not canonical JSON")
    keys = {
        "schema_version",
        "purpose",
        "serialization",
        "authorization_policy",
        "authorization_policy_sha256",
        "authorization_receipt",
        "authorization_receipt_sha256",
        "active_evidence_families",
        "absent_evidence_families",
        "development_only",
        "target_values_loaded",
        "candidate_reasoner_input_authorized",
        "formal_reasoner_authorized",
        "formal_promotion",
        "source_eval_used",
        "private_used",
        "base_v1_directory",
        "amendment_directory",
        "files",
    }
    if set(manifest) != keys:
        raise ValueError("v1.1 capability manifest violates its closed schema")
    fixed = {
        "schema_version": DEVELOPMENT_IV_CAPABILITY_SCHEMA_V1_1,
        "purpose": "development_iv_candidate_reasoner_evidence_with_signed_signal_eligibility",
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "authorization_policy": _CAPABILITY_POLICY_V1_1,
        "authorization_policy_sha256": DEVELOPMENT_IV_CAPABILITY_POLICY_SHA256_V1_1,
        "active_evidence_families": list(_v1.ACTIVE_EVIDENCE_FAMILIES),
        "absent_evidence_families": list(_v1.ABSENT_EVIDENCE_FAMILIES),
        "development_only": True,
        "target_values_loaded": False,
        "candidate_reasoner_input_authorized": True,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
        "source_eval_used": False,
        "private_used": False,
        "base_v1_directory": BASE_CAPABILITY_DIRECTORY,
        "amendment_directory": AMENDMENT_DIRECTORY,
    }
    if any(manifest.get(name) != value for name, value in fixed.items()):
        raise ValueError("v1.1 capability scientific boundary changed")
    receipt = _authorization_v1_1_from_payload(manifest["authorization_receipt"])
    if manifest["authorization_receipt_sha256"] != receipt.receipt_sha256:
        raise ValueError("v1.1 capability authorization SHA mismatch")
    if receipt.receipt_sha256 != FROZEN_V1_1_AUTHORIZATION_RECEIPT_SHA256:
        raise ValueError("Capability authorization is not the frozen v1.1 receipt")
    expected_files = {
        f"{BASE_CAPABILITY_DIRECTORY}/{name}"
        for name in (
            _v1.DEVELOPMENT_IV_CAPABILITY_MANIFEST_FILENAME,
            _v1.DEVELOPMENT_IV_CAPABILITY_EVENTS_FILENAME,
            _v1.DEVELOPMENT_IV_CAPABILITY_TENSORS_FILENAME,
        )
    } | {f"{AMENDMENT_DIRECTORY}/{AMENDMENT_FILENAME}"}
    records = manifest["files"]
    if not isinstance(records, dict) or set(records) != expected_files:
        raise ValueError("v1.1 capability file receipt changed")
    for relative, record in records.items():
        path = source / relative
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256", "size_bytes"}
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _v1._file_sha256(path) != record["sha256"]
        ):
            raise ValueError(f"v1.1 capability payload changed: {relative}")
    base = _v1.load_development_iv_evidence_capability(
        source / BASE_CAPABILITY_DIRECTORY,
        expected_manifest_sha256=receipt.base_v1_manifest_sha256,
    )
    amendment = load_signal_evidence_eligibility_amendment(
        source / AMENDMENT_DIRECTORY,
        signal,
        protocol,
        expected_artifact_sha256=receipt.amendment_artifact_sha256,
        expected_receipt_sha256=receipt.amendment_receipt_sha256,
    )
    replay = issue_development_iv_evidence_capability_v1_1(base, amendment)
    if replay.receipt != receipt:
        raise ValueError("v1.1 capability did not replay from nested artifacts")
    return PublishedDevelopmentIVEvidenceCapabilityV11(
        path=source,
        manifest_sha256=manifest_sha,
        authorization_receipt_sha256=receipt.receipt_sha256,
        capability=replay,
    )


def _dataset_from_signed_roster(
    split: _v1._DevelopmentSplitEvidence,
    *,
    signed_patient_ids: tuple[str, ...],
    authorization_sha256: str,
    target: VerifiedDeepSOZTargetV2Artifact,
) -> _v1.DevelopmentReasonerDataset:
    if split.patient_ids != signed_patient_ids:
        raise ValueError("Evidence does not equal the signed signal-eligibility roster")
    target_batch = target.registry.target_batch(signed_patient_ids)
    patient_to_index = {
        patient_id: index for index, patient_id in enumerate(signed_patient_ids)
    }
    event_patient_index = torch.tensor(
        [patient_to_index[value] for value in split.patient_ids_by_event], dtype=torch.long
    )
    counts = torch.bincount(event_patient_index, minlength=len(signed_patient_ids))
    full = _v1.DevelopmentReasonerPatientBatch(
        _verification_marker=_v1._PATIENT_BATCH_MARKER,
        evidence=split.evidence,
        event_patient_index=event_patient_index,
        patient_ids=signed_patient_ids,
        event_ids=split.event_ids,
        expected_event_counts=counts,
        targets=target_batch.values.to(torch.float32),
        target_mask=target_batch.mask,
    )
    return _v1.DevelopmentReasonerDataset(
        _verification_marker=_v1._DATASET_MARKER,
        model_split=split.model_split,
        full_batch=full,
        evidence_authorization_sha256=authorization_sha256,
        verified_target_v2_receipt_sha256=target.receipt.receipt_sha256,
    )


def join_development_iv_targets_v1_1(
    capability: VerifiedDevelopmentIVEvidenceCapabilityV11,
    target: VerifiedDeepSOZTargetV2Artifact,
) -> _v1.VerifiedDevelopmentReasonerDataBundle:
    """Join targets only after exact signed target and evidence rosters replay."""

    if type(capability) is not VerifiedDevelopmentIVEvidenceCapabilityV11:
        raise TypeError("v1.1 target join requires the closed v1.1 capability")
    capability.assert_unchanged()
    _v1._assert_verified_target_unchanged(target)
    amendment = capability.amendment.receipt
    receipt = capability.receipt
    bindings = {
        "target artifact": target.receipt.target_artifact_sha256
        == receipt.verified_target_v2_artifact_sha256,
        "target receipt": target.receipt.receipt_sha256
        == receipt.verified_target_v2_receipt_sha256,
        "target policy": target.receipt.policy_sha256
        == receipt.verified_target_v2_policy_sha256,
    }
    failed = tuple(name for name, value in bindings.items() if not value)
    if failed:
        raise ValueError(f"v1.1 capability and target-v2 lineage disagree: {failed}")
    target_rosters = {
        name: tuple(values) for name, values in target.receipt.eligible_split_patient_ids
    }
    if target_rosters.get("source_train") != (
        amendment.target_header_source_train_patient_ids
    ) or target_rosters.get("source_dev") != (
        amendment.target_header_source_dev_patient_ids
    ):
        raise ValueError("Target-v2 does not equal the signed target/header roster")
    authorization_sha = receipt.receipt_sha256
    train = _dataset_from_signed_roster(
        capability.base.capability.source_train,
        signed_patient_ids=amendment.signal_evidence_source_train_patient_ids,
        authorization_sha256=authorization_sha,
        target=target,
    )
    dev = _dataset_from_signed_roster(
        capability.base.capability.source_dev,
        signed_patient_ids=amendment.signal_evidence_source_dev_patient_ids,
        authorization_sha256=authorization_sha,
        target=target,
    )
    return _v1.VerifiedDevelopmentReasonerDataBundle(
        _verification_marker=_v1._BUNDLE_MARKER,
        source_train=train,
        source_dev=dev,
        evidence_authorization_sha256=authorization_sha,
        verified_target_v2_receipt_sha256=target.receipt.receipt_sha256,
    )


__all__ = [
    "AMENDMENT_FILENAME",
    "DEVELOPMENT_IV_CAPABILITY_POLICY_SHA256_V1_1",
    "DEVELOPMENT_IV_CAPABILITY_SCHEMA_V1_1",
    "DevelopmentIVEvidenceAuthorizationReceiptV11",
    "PublishedDevelopmentIVEvidenceCapabilityV11",
    "PublishedSignalEvidenceEligibilityAmendment",
    "SIGNAL_EVIDENCE_AMENDMENT_POLICY_SHA256",
    "SIGNAL_EVIDENCE_AMENDMENT_SCHEMA",
    "SignalEvidenceEligibilityAmendmentReceipt",
    "SignalEvidenceExcludedPatient",
    "VerifiedDevelopmentIVEvidenceCapabilityV11",
    "build_signal_evidence_eligibility_amendment",
    "issue_development_iv_evidence_capability_v1_1",
    "join_development_iv_targets_v1_1",
    "load_development_iv_evidence_capability_v1_1",
    "load_signal_evidence_eligibility_amendment",
    "publish_development_iv_evidence_capability_v1_1",
    "publish_signal_evidence_eligibility_amendment",
]
