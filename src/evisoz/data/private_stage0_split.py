"""Target-free deterministic patient split for the private EviSOZ cohort."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Mapping, Sequence

from .artifact_ref import build_json_artifact_ref, canonical_json_sha256
from .split_ledger import (
    SPLIT_ROSTER_SCHEMA_VERSION,
    build_patient_linkage_group,
    build_split_roster,
)


PRIVATE_STAGE0_SPLIT_MATERIALIZATION_SCHEMA_VERSION = (
    "evisoz_private_stage0_split_materialization_v1"
)
PRIVATE_SPLIT_POLICY_ID = "private_target_free_hash_locked20_balanced_cv5_v1"
_SPLIT_SALT = b"evisoz-private-stage0-split-v1\x00"
_PATIENT_SALT = b"evisoz-private-source-patient-v1\x00"


def _patient_hash(patient_key: str) -> str:
    return hashlib.sha256(_PATIENT_SALT + patient_key.encode("ascii")).hexdigest()


def _rank(patient_key: str) -> str:
    return hashlib.sha256(_SPLIT_SALT + patient_key.encode("ascii")).hexdigest()


def build_private_patient_linkage_group(patient_key: str) -> dict[str, Any]:
    """Replay the singleton linkage group used by the frozen private split."""

    if (
        not isinstance(patient_key, str)
        or not patient_key
        or not patient_key.isascii()
        or any(character.isspace() for character in patient_key)
    ):
        raise ValueError("private patient_key is invalid")
    return build_patient_linkage_group(
        members=[
            {
                "dataset_id": "private",
                "patient_key": patient_key,
                "source_patient_sha256": _patient_hash(patient_key),
            }
        ],
        linkage_status="singleton",
    )


def build_private_stage0_split(
    signal_roster_rows: Sequence[Mapping[str, object]],
    *,
    signal_roster_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a split ledger and a compact target-free materialization receipt."""

    if not isinstance(signal_roster_rows, Sequence) or not signal_roster_rows:
        raise ValueError("private signal roster must be non-empty")
    if len(signal_roster_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in signal_roster_sha256
    ):
        raise ValueError("signal_roster_sha256 must be a lowercase SHA256")
    patient_keys = sorted({str(row.get("patient_id", "")) for row in signal_roster_rows})
    if (
        not patient_keys
        or any(
            not key
            or not key.isascii()
            or any(character.isspace() for character in key)
            for key in patient_keys
        )
    ):
        raise ValueError("private signal roster patient keys are invalid")
    if any(not str(row.get("event_id", "")) for row in signal_roster_rows):
        raise ValueError("private signal roster contains an empty event_id")

    groups_by_patient: dict[str, dict[str, Any]] = {}
    for patient_key in patient_keys:
        groups_by_patient[patient_key] = build_private_patient_linkage_group(
            patient_key
        )

    ranked = sorted(patient_keys, key=lambda key: (_rank(key), key))
    locked_count = max(1, (len(ranked) + 2) // 5)
    locked = set(ranked[:locked_count])
    development = ranked[locked_count:]
    fold_by_patient = {
        patient_key: ordinal % 5
        for ordinal, patient_key in enumerate(development)
    }
    assignments: list[dict[str, object]] = []
    for patient_key in patient_keys:
        group = groups_by_patient[patient_key]
        if patient_key in locked:
            assignments.append(
                {
                    "linkage_group_id": group["linkage_group_id"],
                    "official_splits": [
                        {
                            "dataset_id": "private",
                            "official_split": (
                                "evisoz_locked_test_prior_frozen_v29_exposure"
                            ),
                        }
                    ],
                    "evisoz_role": "locked_test",
                    "outer_holdout_fold": None,
                    "locked": True,
                }
            )
        else:
            assignments.append(
                {
                    "linkage_group_id": group["linkage_group_id"],
                    "official_splits": [
                        {
                            "dataset_id": "private",
                            "official_split": "evisoz_development_prior_frozen_v29_exposure",
                        }
                    ],
                    "evisoz_role": "development_cv",
                    "outer_holdout_fold": fold_by_patient[patient_key],
                    "locked": False,
                }
            )
    split = build_split_roster(
        linkage_groups=list(groups_by_patient.values()),
        assignments=assignments,
    )
    role_counts = Counter(row["evisoz_role"] for row in split["assignments"])
    fold_counts = Counter(
        row["outer_holdout_fold"]
        for row in split["assignments"]
        if row["evisoz_role"] == "development_cv"
    )
    split_ref = build_json_artifact_ref(
        split,
        artifact_kind="split_roster",
        payload_schema_version=SPLIT_ROSTER_SCHEMA_VERSION,
    )
    summary: dict[str, Any] = {
        "schema_version": PRIVATE_STAGE0_SPLIT_MATERIALIZATION_SCHEMA_VERSION,
        "status": "frozen_target_free_patient_split",
        "policy_id": PRIVATE_SPLIT_POLICY_ID,
        "input_signal_roster_sha256": signal_roster_sha256,
        "patient_count": len(patient_keys),
        "event_count": len(signal_roster_rows),
        "role_patient_counts": dict(sorted(role_counts.items())),
        "development_outer_fold_patient_counts": {
            str(key): fold_counts[key] for key in sorted(fold_counts)
        },
        "split_roster_ref": split_ref,
        "selection_contract": {
            "patient_level": True,
            "labels_or_reports_used_for_assignment": False,
            "deterministic_salted_sha256_rank": True,
            "locked_test_fraction_target": 0.2,
            "development_outer_fold_count": 5,
            "same_patient_events_share_one_assignment": True,
        },
        "prior_exposure_boundary": {
            "frozen_private_v29_may_have_processed_these_patients": True,
            "locked_test_means_excluded_from_new_evisoz_fitting": True,
            "locked_test_is_pristine_external_validation": False,
        },
        "receipt_sha256": "0" * 64,
    }
    summary["receipt_sha256"] = canonical_json_sha256(summary)
    return split, summary


__all__ = [
    "PRIVATE_SPLIT_POLICY_ID",
    "PRIVATE_STAGE0_SPLIT_MATERIALIZATION_SCHEMA_VERSION",
    "build_private_patient_linkage_group",
    "build_private_stage0_split",
]
