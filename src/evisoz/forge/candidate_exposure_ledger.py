"""Exposure ledger for offline EviSOZ teacher/candidate components."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from src.evisoz.data.artifact_ref import build_json_artifact_ref, canonical_json_sha256, validate_artifact_ref


CANDIDATE_EXPOSURE_LEDGER_SCHEMA_VERSION = "evisoz_candidate_exposure_ledger_v1"
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_ID_PREFIX = "EVISOZ-EXPOSURE-"


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = _hash_source(value)
    result["ledger_id"] = _PENDING_ID
    return result


def build_candidate_exposure_ledger(
    *,
    candidate_manifest: Mapping[str, object],
) -> dict[str, Any]:
    """Build a component-level exposure receipt from a candidate manifest."""

    if type(candidate_manifest) is not dict:
        raise TypeError("candidate manifest must be an object")
    if candidate_manifest.get("schema_version") != "evisoz_deterministic_signal_candidate_materialization_v1":
        raise ValueError("candidate manifest schema drifted")
    events = candidate_manifest.get("events")
    counts = candidate_manifest.get("counts")
    if not isinstance(events, list) or not events or not isinstance(counts, dict):
        raise ValueError("candidate manifest lacks event/count materialization")
    event_rows: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    fold_counts: Counter[str] = Counter()
    for event in events:
        if type(event) is not dict:
            raise ValueError("candidate manifest event row is invalid")
        required = {
            "event_id",
            "evisoz_role",
            "outer_holdout_fold",
            "linkage_group_id",
            "candidate_cache_ref",
            "source_dual_montage_cache_ref",
            "candidate_count",
            "candidate_concept_counts",
            "relative_candidate_cache_path",
        }
        if set(event) != required:
            raise ValueError("candidate manifest event row fields drifted")
        validate_artifact_ref(event["candidate_cache_ref"])
        validate_artifact_ref(event["source_dual_montage_cache_ref"])
        if event["evisoz_role"] not in {"development_cv", "locked_test"}:
            raise ValueError("candidate manifest event role drifted")
        fold = event["outer_holdout_fold"]
        # Development events carry their numeric outer fold; locked-test
        # events intentionally have no fold assignment.  Preserve that
        # distinction instead of coercing null to a synthetic fold.
        if event["evisoz_role"] == "development_cv":
            if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
                raise ValueError("candidate manifest development fold drifted")
        elif fold is not None:
            raise ValueError("candidate manifest locked-test fold drifted")
        candidate_count = event["candidate_count"]
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 0:
            raise ValueError("candidate manifest candidate count drifted")
        concept_counts = event["candidate_concept_counts"]
        if type(concept_counts) is not dict or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0
            for v in concept_counts.values()
        ) or sum(concept_counts.values()) != candidate_count:
            raise ValueError("candidate manifest concept counts drifted")
        event_rows.append({
            "event_id": str(event["event_id"]),
            "linkage_group_id": str(event["linkage_group_id"]),
            "evisoz_role": event["evisoz_role"],
            "outer_holdout_fold": fold,
            "candidate_count": candidate_count,
            "candidate_cache_ref": deepcopy(event["candidate_cache_ref"]),
            "source_dual_montage_cache_ref": deepcopy(event["source_dual_montage_cache_ref"]),
        })
        role_counts[str(event["evisoz_role"])] += 1
        fold_counts[str(fold) if fold is not None else "locked_test"] += 1
    event_rows.sort(key=lambda row: row["event_id"])
    source_ref = build_json_artifact_ref(
        candidate_manifest,
        artifact_kind="deterministic_signal_candidate_materialization",
        payload_schema_version="evisoz_deterministic_signal_candidate_materialization_v1",
    )
    body: dict[str, Any] = {
        "schema_version": CANDIDATE_EXPOSURE_LEDGER_SCHEMA_VERSION,
        "ledger_id": _PENDING_ID,
        "status": "deterministic_candidate_lineage_closed_teachers_absent",
        "candidate_materialization_ref": source_ref,
        "components": [
            {
                "component_id": "deterministic_signal_candidates",
                "component_role": "deterministic",
                "status": "materialized",
                "target_relation": "training_only_or_soft_auxiliary",
                "exposure_relation": "same_event_source_signal",
                "calibration_state": "uncalibrated",
                "node_localization_supervision": False,
            },
            {
                "component_id": "cerebragloss",
                "component_role": "teacher_programmatic",
                "status": "absent",
                "target_relation": "not_applicable",
                "exposure_relation": "unknown",
                "calibration_state": "not_applicable",
                "node_localization_supervision": False,
            },
            {
                "component_id": "elm",
                "component_role": "teacher_programmatic",
                "status": "absent",
                "target_relation": "not_applicable",
                "exposure_relation": "unknown",
                "calibration_state": "not_applicable",
                "node_localization_supervision": False,
            },
        ],
        "events": event_rows,
        "counts": {
            "event_count": len(event_rows),
            "candidate_count": int(counts["candidate_count"]),
            "development_event_count": role_counts["development_cv"],
            "locked_event_count": role_counts["locked_test"],
            "outer_fold_event_counts": dict(sorted(fold_counts.items())),
        },
        "permissions": {
            "teacher_runtime_required_at_deployment": False,
            "candidate_soft_auxiliary_only": True,
            "node_localization_supervision_authorized": False,
            "calibration_authorized": False,
            "training_authorized": False,
            "raw_patient_identifiers_stored": False,
        },
        "missing_closure_codes": [
            "cerebragloss_candidate_artifact_missing",
            "elm_candidate_artifact_missing",
            "fold_local_calibration_receipts_missing",
        ],
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["ledger_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_candidate_exposure_ledger(body)


def validate_candidate_exposure_ledger(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version", "ledger_id", "status", "candidate_materialization_ref",
        "components", "events", "counts", "permissions", "missing_closure_codes", "receipt_sha256",
    }:
        raise ValueError("candidate exposure ledger fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != CANDIDATE_EXPOSURE_LEDGER_SCHEMA_VERSION:
        raise ValueError("candidate exposure ledger schema drifted")
    if data["status"] != "deterministic_candidate_lineage_closed_teachers_absent":
        raise ValueError("candidate exposure ledger status drifted")
    validate_artifact_ref(data["candidate_materialization_ref"])
    components = data["components"]
    if not isinstance(components, list) or [row.get("component_id") for row in components] != [
        "deterministic_signal_candidates", "cerebragloss", "elm"
    ]:
        raise ValueError("candidate exposure component roster drifted")
    for row in components:
        if type(row) is not dict or set(row) != {
            "component_id", "component_role", "status", "target_relation",
            "exposure_relation", "calibration_state", "node_localization_supervision",
        }:
            raise ValueError("candidate exposure component fields drifted")
        if row["node_localization_supervision"] is not False:
            raise ValueError("candidate exposure promoted node supervision")
    if components[0]["status"] != "materialized" or any(row["status"] != "absent" for row in components[1:]):
        raise ValueError("candidate exposure teacher status drifted")
    events = data["events"]
    if not isinstance(events, list) or events != sorted(events, key=lambda row: row["event_id"]):
        raise ValueError("candidate exposure events are not sorted")
    role_counts: Counter[str] = Counter()
    fold_counts: Counter[str] = Counter()
    candidate_count = 0
    seen_events: set[str] = set()
    for row in events:
        if type(row) is not dict or set(row) != {
            "event_id", "linkage_group_id", "evisoz_role", "outer_holdout_fold",
            "candidate_count", "candidate_cache_ref", "source_dual_montage_cache_ref",
        }:
            raise ValueError("candidate exposure event fields drifted")
        if row["event_id"] in seen_events:
            raise ValueError("candidate exposure event duplicated")
        seen_events.add(row["event_id"])
        validate_artifact_ref(row["candidate_cache_ref"])
        validate_artifact_ref(row["source_dual_montage_cache_ref"])
        if row["evisoz_role"] not in {"development_cv", "locked_test"}:
            raise ValueError("candidate exposure event role drifted")
        role_counts[row["evisoz_role"]] += 1
        fold = row["outer_holdout_fold"]
        if row["evisoz_role"] == "development_cv":
            if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
                raise ValueError("candidate exposure development fold drifted")
        elif fold is not None:
            raise ValueError("candidate exposure locked-test fold drifted")
        fold_counts[str(fold) if fold is not None else "locked_test"] += 1
        candidate_count += row["candidate_count"]
    expected_counts = {
        "event_count": len(events),
        "candidate_count": candidate_count,
        "development_event_count": role_counts["development_cv"],
        "locked_event_count": role_counts["locked_test"],
        "outer_fold_event_counts": dict(sorted(fold_counts.items())),
    }
    if data["counts"] != expected_counts:
        raise ValueError("candidate exposure counts drifted")
    if data["permissions"] != {
        "teacher_runtime_required_at_deployment": False,
        "candidate_soft_auxiliary_only": True,
        "node_localization_supervision_authorized": False,
        "calibration_authorized": False,
        "training_authorized": False,
        "raw_patient_identifiers_stored": False,
    }:
        raise ValueError("candidate exposure permissions drifted")
    if data["missing_closure_codes"] != [
        "cerebragloss_candidate_artifact_missing",
        "elm_candidate_artifact_missing",
        "fold_local_calibration_receipts_missing",
    ]:
        raise ValueError("candidate exposure missing closures drifted")
    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["ledger_id"] != expected_id:
        raise ValueError("candidate exposure ledger ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("candidate exposure ledger receipt drifted")
    return data


__all__ = [
    "CANDIDATE_EXPOSURE_LEDGER_SCHEMA_VERSION",
    "build_candidate_exposure_ledger",
    "validate_candidate_exposure_ledger",
]
