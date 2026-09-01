"""Versioned public-development union for identity-recovered DeepSOZ signals.

The v11 102-patient fold assignment is immutable.  Its 988 event rows are
copied as an exact ordered prefix; only the 161 signal-eligible recovered
events are appended.  Any newly admitted patient is assigned without target
values, private data, or predictions, using event burden, patient count, and
a frozen salted hash solely as a deterministic tie breaker.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

from .deepsoz_signal_identity_recovery import (
    DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_SCHEMA,
    load_deepsoz_signal_identity_recovery_bundle,
)


PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_SCHEMA = (
    "soz_public_development_union_identity_v12"
)
PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FILENAME = "manifest.json"
PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_POLICY = (
    "immutable_v11_folds_and_988_event_prefix_append_recovered_events_"
    "target_free_new_patient_assignment"
)
N_OUTER_FOLDS = 5
NEW_PATIENT_FOLD_SALT = (
    "public-development-union-identity-v12-new-patient-fold-20260812"
)
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_EXPECTED_OLD_SCHEMA = "soz_public_development_union_v11"
EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256 = (
    "645c55541c37dfc204fdd48c21e0a3c81fe7201f76b862556d1c4dc3bfa4d429"
)
EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256 = (
    "cab51b090ae45f7dee5c3a0b8a9d89143f3731be25cf46631e225a8291056aad"
)
EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PATIENTS = 103
EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_EVENTS = 1149
EXPECTED_LEGACY_EVENT_PREFIX_ROWS_SHA256 = (
    "7ae731d6cd246341b182b3be875d42f1bea0fce82b775b0441149716f11b070c"
)
EXPECTED_LEGACY_EVENT_PREFIX_IDS_SHA256 = (
    "82453898ec09d1420b0d7de1b15b98cab222a1297ff659093ed6131868bad9e8"
)
EXPECTED_RECOVERED_APPEND_EVENT_IDS_SHA256 = (
    "da4a6dbda114aac8ca2117ea37a6e86932bef25994c8958b06714172a4892875"
)
EXPECTED_V12_EVENT_ORDER_SHA256 = (
    "df849209580c65c3cdf70eb6bcc6912f8f82560a9ab676b36c9bdae297aaac4f"
)
EXPECTED_V12_PATIENT_ROSTER_SHA256 = (
    "81dd925132ce11bfbd68f65fea1db530f3c1f772b4ebcd0145a15838751d82a5"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_FIELDS = frozenset(
    {
        "access_receipt",
        "claim_boundary",
        "cohort_name",
        "event_count",
        "event_ids",
        "event_order_sha256",
        "events",
        "immutability_receipt",
        "legacy_outer_fold_assignment_policy",
        "legacy_split_event_counts",
        "legacy_v11_event_prefix_count",
        "legacy_v11_event_prefix_ids_sha256",
        "legacy_v11_event_prefix_rows_sha256",
        "legacy_v11_patient_count",
        "lineage",
        "manifest_payload_sha256",
        "new_patient_assignment_receipts",
        "new_patient_count",
        "new_patient_ids",
        "new_patient_outer_fold_assignment_policy",
        "new_patient_outer_fold_salt",
        "outer_fold_count",
        "outer_fold_event_counts",
        "outer_fold_patient_counts",
        "patient_count",
        "patient_ids",
        "patient_roster_sha256",
        "patients",
        "policy",
        "purpose",
        "recovered_append_event_count",
        "recovered_append_event_ids_sha256",
        "recovered_append_patient_count",
        "schema_version",
    }
)
_PATIENT_FIELDS = frozenset(
    {
        "patient_id",
        "outer_fold",
        "event_count",
        "legacy_v11_event_count",
        "recovered_appended_event_count",
        "legacy_model_split",
        "patient_origin",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "ordinal",
        "event_id",
        "patient_id",
        "outer_fold",
        "legacy_model_split",
        "official_split",
        "relative_edf_path",
        "global_event_index",
        "global_t0_sec",
        "global_stop_sec",
        "event_record_sha256",
        "edf_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
        "processed_window_shape",
        "processed_window_dtype",
    }
)


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_old_union(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_payload_sha256: str,
) -> tuple[dict[str, object], str]:
    root = Path(os.path.abspath(directory))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Legacy public-development union must be a regular directory")
    entries = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    if (
        len(entries) != 1
        or entries[0].name != PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FILENAME
        or entries[0].is_symlink()
        or not entries[0].is_file()
    ):
        raise ValueError("Legacy public-development union violates its closed schema")
    path = entries[0]
    if path.stat().st_size < 1 or path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("Legacy union manifest has an invalid size")
    encoded = path.read_bytes()
    manifest_sha = hashlib.sha256(encoded).hexdigest()
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("Legacy union manifest SHA mismatch")
    try:
        manifest = json.loads(encoded.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Legacy union manifest is not strict JSON") from exc
    if _canonical_bytes(manifest, newline=True) != encoded:
        raise ValueError("Legacy union manifest is not canonical JSON")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _EXPECTED_OLD_SCHEMA:
        raise ValueError("Legacy union schema mismatch")
    declared_payload_sha = str(manifest.get("manifest_payload_sha256", ""))
    payload_without_sha = dict(manifest)
    payload_without_sha.pop("manifest_payload_sha256", None)
    if (
        declared_payload_sha != expected_payload_sha256
        or _canonical_sha256(payload_without_sha) != declared_payload_sha
    ):
        raise ValueError("Legacy union payload SHA mismatch")
    if manifest.get("patient_count") != 102 or manifest.get("event_count") != 988:
        raise ValueError("Legacy union is not the frozen 102-patient/988-event cohort")
    patients = manifest.get("patients")
    events = manifest.get("events")
    patient_ids = manifest.get("patient_ids")
    event_ids = manifest.get("event_ids")
    if not all(isinstance(value, list) for value in (patients, events, patient_ids, event_ids)):
        raise ValueError("Legacy union rosters have invalid types")
    if len(patients) != 102 or len(events) != 988:
        raise ValueError("Legacy union row counts drifted")
    if [str(row["patient_id"]) for row in patients] != [str(value) for value in patient_ids]:
        raise ValueError("Legacy patient row order differs from patient_ids")
    if [str(row["event_id"]) for row in events] != [str(value) for value in event_ids]:
        raise ValueError("Legacy event row order differs from event_ids")
    if [int(row["ordinal"]) for row in events] != list(range(988)):
        raise ValueError("Legacy event ordinals are not the frozen 0..987 prefix")
    if len(set(str(value) for value in patient_ids)) != 102:
        raise ValueError("Legacy union patient IDs are not unique")
    if len(set(str(value) for value in event_ids)) != 988:
        raise ValueError("Legacy union event IDs are not unique")
    return manifest, manifest_sha


def _project_recovered_event(
    row: Mapping[str, object], *, ordinal: int, outer_fold: int
) -> dict[str, object]:
    return {
        "ordinal": int(ordinal),
        "event_id": str(row["event_id"]),
        "patient_id": str(row["patient_id"]),
        "outer_fold": int(outer_fold),
        "legacy_model_split": str(row["model_split"]),
        "official_split": str(row["official_split"]),
        "relative_edf_path": str(row["relative_edf_path"]),
        "global_event_index": int(row["global_event_index"]),
        "global_t0_sec": float(row["global_t0_sec"]),
        "global_stop_sec": float(row["global_stop_sec"]),
        "event_record_sha256": str(row["event_record_sha256"]),
        "edf_sha256": str(row["edf_sha256"]),
        "edf_receipt_sha256": str(row["edf_receipt_sha256"]),
        "signal_receipt_sha256": str(row["signal_receipt_sha256"]),
        "processed_window_sha256": str(row["processed_window_sha256"]),
        "processed_window_shape": list(row["processed_window_shape"]),
        "processed_window_dtype": str(row["processed_window_dtype"]),
    }


def _assign_new_patient_folds(
    new_patient_event_counts: Mapping[str, int],
    *,
    existing_fold_event_counts: Mapping[int, int],
    existing_fold_patient_counts: Mapping[int, int],
) -> tuple[dict[str, int], list[dict[str, object]]]:
    fold_events = {fold: int(existing_fold_event_counts[fold]) for fold in range(N_OUTER_FOLDS)}
    fold_patients = {
        fold: int(existing_fold_patient_counts[fold]) for fold in range(N_OUTER_FOLDS)
    }
    assignment: dict[str, int] = {}
    receipts: list[dict[str, object]] = []

    def patient_order(item: tuple[str, int]) -> tuple[int, str]:
        patient_id, count = item
        digest = hashlib.sha256(
            f"{NEW_PATIENT_FOLD_SALT}|patient={patient_id}".encode("ascii")
        ).hexdigest()
        return -int(count), digest

    for patient_id, event_count in sorted(
        new_patient_event_counts.items(), key=patient_order
    ):
        before_events = dict(fold_events)
        before_patients = dict(fold_patients)
        tie_hashes = {
            fold: hashlib.sha256(
                f"{NEW_PATIENT_FOLD_SALT}|{patient_id}|fold={fold}".encode("ascii")
            ).hexdigest()
            for fold in range(N_OUTER_FOLDS)
        }
        fold = min(
            range(N_OUTER_FOLDS),
            key=lambda candidate: (
                fold_events[candidate],
                fold_patients[candidate],
                tie_hashes[candidate],
            ),
        )
        assignment[patient_id] = fold
        fold_events[fold] += int(event_count)
        fold_patients[fold] += 1
        receipts.append(
            {
                "patient_id": patient_id,
                "event_count": int(event_count),
                "selected_outer_fold": fold,
                "fold_event_counts_before": {
                    str(key): before_events[key] for key in range(N_OUTER_FOLDS)
                },
                "fold_patient_counts_before": {
                    str(key): before_patients[key] for key in range(N_OUTER_FOLDS)
                },
                "selection_key": (
                    "minimum_event_burden_then_minimum_patient_count_then_"
                    "frozen_salted_patient_fold_hash"
                ),
            }
        )
    return assignment, receipts


def build_public_development_union_identity_v12(
    legacy_union_directory: str | Path,
    signal_recovery_directory: str | Path,
    *,
    expected_legacy_manifest_sha256: str,
    expected_legacy_payload_sha256: str,
    expected_signal_recovery_artifact_sha256: str,
) -> dict[str, object]:
    """Build, but do not publish, the target-free versioned union manifest."""

    legacy, legacy_manifest_sha = _read_old_union(
        legacy_union_directory,
        expected_manifest_sha256=expected_legacy_manifest_sha256,
        expected_payload_sha256=expected_legacy_payload_sha256,
    )
    recovery = load_deepsoz_signal_identity_recovery_bundle(
        signal_recovery_directory,
        expected_artifact_sha256=expected_signal_recovery_artifact_sha256,
    )
    receipt = recovery.receipt
    if receipt["schema_version"] != DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_SCHEMA:
        raise ValueError("Signal recovery receipt schema mismatch")
    if (
        receipt["base_eligible_event_count"] != 988
        or receipt["recovered_eligible_event_count"] != 161
        or receipt["combined_eligible_event_count"] != 1149
        or receipt["combined_eligible_patient_count"] != 103
    ):
        raise ValueError("Signal recovery does not close to 103 patients / 1149 events")

    legacy_events = [dict(row) for row in legacy["events"]]
    legacy_patients = [dict(row) for row in legacy["patients"]]
    legacy_event_ids = [str(value) for value in legacy["event_ids"]]
    legacy_patient_ids = [str(value) for value in legacy["patient_ids"]]
    legacy_fold_by_patient = {
        str(row["patient_id"]): int(row["outer_fold"]) for row in legacy_patients
    }
    if len(legacy_fold_by_patient) != 102:
        raise ValueError("Legacy patient fold mapping is not one-to-one")

    recovery_events_by_id = {
        str(row["event_id"]): row for row in receipt["events"]
    }
    recovered_event_ids = [str(value) for value in receipt["recovered_eligible_event_ids"]]
    if recovered_event_ids != sorted(set(recovered_event_ids)):
        raise ValueError("Recovered event append roster must be sorted and unique")
    if set(recovered_event_ids) & set(legacy_event_ids):
        raise ValueError("Recovered append roster overlaps the immutable event prefix")
    try:
        recovered_rows = [recovery_events_by_id[event_id] for event_id in recovered_event_ids]
    except KeyError as exc:
        raise ValueError("A recovered append event is absent from the signal receipt") from exc

    recovered_counts = Counter(str(row["patient_id"]) for row in recovered_rows)
    recovered_existing = {
        patient_id: count
        for patient_id, count in recovered_counts.items()
        if patient_id in legacy_fold_by_patient
    }
    recovered_new = {
        patient_id: count
        for patient_id, count in recovered_counts.items()
        if patient_id not in legacy_fold_by_patient
    }
    if recovered_new != {"10489": 27}:
        raise ValueError("The identity recovery must add exactly patient 10489 with 27 events")

    fold_event_burden = {
        fold: int(legacy["outer_fold_event_counts"][str(fold)])
        for fold in range(N_OUTER_FOLDS)
    }
    fold_patient_burden = {
        fold: int(legacy["outer_fold_patient_counts"][str(fold)])
        for fold in range(N_OUTER_FOLDS)
    }
    for patient_id, count in recovered_existing.items():
        fold_event_burden[legacy_fold_by_patient[patient_id]] += int(count)
    new_folds, new_assignment_receipts = _assign_new_patient_folds(
        recovered_new,
        existing_fold_event_counts=fold_event_burden,
        existing_fold_patient_counts=fold_patient_burden,
    )
    fold_by_patient = {**legacy_fold_by_patient, **new_folds}

    appended_events = [
        _project_recovered_event(
            row,
            ordinal=988 + index,
            outer_fold=fold_by_patient[str(row["patient_id"])],
        )
        for index, row in enumerate(recovered_rows)
    ]
    events = [*legacy_events, *appended_events]
    if events[:988] != legacy_events:
        raise RuntimeError("The legacy 988-event prefix changed during append")
    if [int(row["ordinal"]) for row in events] != list(range(1149)):
        raise RuntimeError("Versioned union event ordinals are not contiguous")
    event_ids = [str(row["event_id"]) for row in events]
    if event_ids[:988] != legacy_event_ids or event_ids[988:] != recovered_event_ids:
        raise RuntimeError("Versioned union event ordering contract failed")
    if len(set(event_ids)) != 1149:
        raise ValueError("Versioned union event IDs are not unique")

    events_by_patient: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in events:
        patient_id = str(row["patient_id"])
        if int(row["outer_fold"]) != fold_by_patient[patient_id]:
            raise ValueError("One patient appears in multiple outer folds")
        events_by_patient[patient_id].append(row)
    new_patient_ids = sorted(set(events_by_patient) - set(legacy_patient_ids))
    if new_patient_ids != ["10489"]:
        raise ValueError("Versioned union has an unexpected new patient roster")
    patient_ids = [*legacy_patient_ids, *new_patient_ids]
    if len(patient_ids) != 103 or len(set(patient_ids)) != 103:
        raise ValueError("Versioned union patient roster is not 103 unique patients")

    legacy_patient_by_id = {
        str(row["patient_id"]): row for row in legacy_patients
    }
    patients: list[dict[str, object]] = []
    for patient_id in patient_ids:
        rows = events_by_patient[patient_id]
        if patient_id in legacy_patient_by_id:
            legacy_row = legacy_patient_by_id[patient_id]
            if int(legacy_row["outer_fold"]) != fold_by_patient[patient_id]:
                raise RuntimeError("A legacy patient's outer fold changed")
            patients.append(
                {
                    "patient_id": patient_id,
                    "outer_fold": fold_by_patient[patient_id],
                    "event_count": len(rows),
                    "legacy_v11_event_count": int(legacy_row["event_count"]),
                    "recovered_appended_event_count": int(
                        recovered_counts.get(patient_id, 0)
                    ),
                    "legacy_model_split": str(legacy_row["legacy_model_split"]),
                    "patient_origin": "legacy_v11",
                }
            )
        else:
            patients.append(
                {
                    "patient_id": patient_id,
                    "outer_fold": fold_by_patient[patient_id],
                    "event_count": len(rows),
                    "legacy_v11_event_count": 0,
                    "recovered_appended_event_count": len(rows),
                    "legacy_model_split": str(rows[0]["legacy_model_split"]),
                    "patient_origin": "identity_recovery_v3",
                }
            )

    fold_patient_counts = Counter(int(row["outer_fold"]) for row in patients)
    fold_event_counts = Counter(int(row["outer_fold"]) for row in events)
    legacy_split_counts = Counter(str(row["legacy_model_split"]) for row in events)
    manifest: dict[str, object] = {
        "schema_version": PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_SCHEMA,
        "policy": PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_POLICY,
        "purpose": "identity_recovered_public_developmental_nested_cv_only",
        "claim_boundary": {
            "public_confirmation_forbidden": True,
            "public_external_validation_forbidden": True,
            "legacy_source_eval_retry_claim_forbidden": True,
            "legacy_splits_retained_for_audit_only": True,
            "all_103_patients_are_developmental": True,
            "nested_oof_is_internal_developmental_estimate_only": True,
            "private_data_must_remain_unread_for_training_selection_calibration": True,
            "private_is_reserved_for_frozen_zero_adaptation_transfer": True,
        },
        "cohort_name": "public_development_union_identity_v12",
        "patient_count": len(patient_ids),
        "event_count": len(events),
        "legacy_v11_patient_count": 102,
        "legacy_v11_event_prefix_count": 988,
        "recovered_append_patient_count": len(recovered_counts),
        "recovered_append_event_count": len(recovered_event_ids),
        "new_patient_count": len(new_patient_ids),
        "new_patient_ids": new_patient_ids,
        "legacy_split_event_counts": dict(sorted(legacy_split_counts.items())),
        "outer_fold_count": N_OUTER_FOLDS,
        "legacy_outer_fold_assignment_policy": str(
            legacy["outer_fold_assignment_policy"]
        ),
        "new_patient_outer_fold_assignment_policy": (
            "target_free_existing_burden_then_patient_count_then_salted_hash_v1"
        ),
        "new_patient_outer_fold_salt": NEW_PATIENT_FOLD_SALT,
        "new_patient_assignment_receipts": new_assignment_receipts,
        "outer_fold_patient_counts": {
            str(key): fold_patient_counts[key] for key in range(N_OUTER_FOLDS)
        },
        "outer_fold_event_counts": {
            str(key): fold_event_counts[key] for key in range(N_OUTER_FOLDS)
        },
        "patient_ids": patient_ids,
        "patient_roster_sha256": _canonical_sha256(patient_ids),
        "event_ids": event_ids,
        "event_order_sha256": _canonical_sha256(event_ids),
        "legacy_v11_event_prefix_rows_sha256": _canonical_sha256(legacy_events),
        "legacy_v11_event_prefix_ids_sha256": _canonical_sha256(legacy_event_ids),
        "recovered_append_event_ids_sha256": _canonical_sha256(recovered_event_ids),
        "patients": patients,
        "events": events,
        "lineage": {
            "legacy_union_path": str(Path(legacy_union_directory).resolve()),
            "legacy_union_manifest_sha256": legacy_manifest_sha,
            "legacy_union_payload_sha256": expected_legacy_payload_sha256,
            "signal_identity_recovery_path": str(
                Path(signal_recovery_directory).resolve()
            ),
            "signal_identity_recovery_artifact_sha256": recovery.artifact_sha256,
            "signal_identity_recovery_receipt_sha256": recovery.receipt_sha256,
            "base_signal_preflight_artifact_sha256": str(
                receipt["base_signal_preflight_artifact_sha256"]
            ),
            "base_signal_preflight_receipt_sha256": str(
                receipt["base_signal_preflight_receipt_sha256"]
            ),
            "preprocess_config_sha256": str(receipt["preprocess_config_sha256"]),
            "combined_signal_event_roster_sha256": str(
                receipt["combined_eligible_event_roster_sha256"]
            ),
            "recovered_signal_event_ids_sha256": _canonical_sha256(
                recovered_event_ids
            ),
        },
        "immutability_receipt": {
            "legacy_102_patient_outer_folds_preserved": all(
                fold_by_patient[patient_id] == legacy_fold_by_patient[patient_id]
                for patient_id in legacy_patient_ids
            ),
            "legacy_988_event_rows_exact_prefix": events[:988] == legacy_events,
            "legacy_988_event_ids_exact_prefix": event_ids[:988] == legacy_event_ids,
            "recovered_events_append_only": event_ids[988:] == recovered_event_ids,
            "legacy_recovered_event_overlap_count": len(
                set(legacy_event_ids) & set(recovered_event_ids)
            ),
            "new_patient_10489_event_count": len(events_by_patient["10489"]),
            "new_patient_10489_outer_fold": fold_by_patient["10489"],
        },
        "access_receipt": {
            "legacy_union_metadata_loaded": True,
            "signal_recovery_metadata_loaded": True,
            "raw_eeg_loaded": False,
            "deepsoz_target_values_loaded": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "prediction_artifacts_loaded": False,
        },
    }
    if not all(
        bool(manifest["immutability_receipt"][field])
        for field in (
            "legacy_102_patient_outer_folds_preserved",
            "legacy_988_event_rows_exact_prefix",
            "legacy_988_event_ids_exact_prefix",
            "recovered_events_append_only",
        )
    ):
        raise RuntimeError("Versioned public union immutability contract failed")
    if manifest["immutability_receipt"]["legacy_recovered_event_overlap_count"] != 0:
        raise RuntimeError("Versioned public union event rosters overlap")
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    return manifest


def publish_public_development_union_identity_v12(
    manifest: Mapping[str, object], output_directory: str | Path
) -> Path:
    """Atomically publish a new one-file union directory; never overwrite."""

    target = Path(os.path.abspath(output_directory))
    if target.name in {"", ".", ".."}:
        raise ValueError("Output requires a concrete directory name")
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise ValueError("Output parent must be a regular existing directory")
    encoded = _canonical_bytes(manifest, newline=True)
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise ValueError("Versioned public union manifest exceeds its size limit")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        output = staging / PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FILENAME
        with output.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(target):
            raise FileExistsError(target)
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def _closed_mapping(
    value: object, *, fields: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{name} violates its closed schema; missing={missing}, unknown={unknown}"
        )
    return value


def _strict_v12_json(raw: bytes) -> dict[str, object]:
    def reject_duplicate(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate manifest field: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v12 union manifest is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError("v12 union manifest must be an object")
    if raw != _canonical_bytes(payload, newline=True):
        raise ValueError("v12 union manifest is not canonical JSON")
    return payload


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class PublicDevelopmentUnionIdentityV12Event:
    ordinal: int
    event_id: str
    patient_id: str
    outer_fold: int
    legacy_model_split: str
    official_split: str
    relative_edf_path: str
    global_event_index: int
    global_t0_sec: float
    global_stop_sec: float
    event_record_sha256: str
    edf_sha256: str
    edf_receipt_sha256: str
    signal_receipt_sha256: str
    processed_window_sha256: str
    processed_window_shape: tuple[int, int]
    processed_window_dtype: str

    def __post_init__(self) -> None:
        if self.ordinal < 0 or self.global_event_index < 0:
            raise ValueError("v12 event ordinal/index must be non-negative")
        if not self.event_id or not self.patient_id:
            raise ValueError("v12 event identity cannot be empty")
        if self.outer_fold not in range(N_OUTER_FOLDS):
            raise ValueError("v12 event outer fold is invalid")
        if self.legacy_model_split not in {
            "source_train",
            "source_dev",
            "source_eval",
        }:
            raise ValueError("v12 event legacy split is invalid")
        if self.official_split not in {"train", "dev", "eval"}:
            raise ValueError("v12 event official split is invalid")
        relative = Path(self.relative_edf_path)
        if (
            relative.is_absolute()
            or len(relative.parts) != 5
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.suffix != ".edf"
        ):
            raise ValueError("v12 event EDF path is not canonical relative TUSZ identity")
        if (
            not math.isfinite(self.global_t0_sec)
            or not math.isfinite(self.global_stop_sec)
            or self.global_t0_sec < 0
            or self.global_stop_sec <= self.global_t0_sec
        ):
            raise ValueError("v12 event timing is invalid")
        for field in (
            "event_record_sha256",
            "edf_sha256",
            "edf_receipt_sha256",
            "signal_receipt_sha256",
            "processed_window_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.processed_window_shape != (19, 12000):
            raise ValueError("v12 event signal shape changed")
        if self.processed_window_dtype != "torch.float32":
            raise ValueError("v12 event signal dtype changed")


@dataclass(frozen=True)
class PublicDevelopmentUnionIdentityV12:
    path: Path
    manifest_sha256: str
    payload: Mapping[str, object]
    patient_ids: tuple[str, ...]
    patient_folds: tuple[int, ...]
    events: tuple[PublicDevelopmentUnionIdentityV12Event, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_sha256, field="manifest_sha256")
        if len(self.patient_ids) != EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PATIENTS:
            raise ValueError("v12 union must contain 103 patients")
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("v12 union repeats a patient")
        if len(self.patient_folds) != len(self.patient_ids):
            raise ValueError("v12 patient folds do not align with patient IDs")
        if set(self.patient_folds) != set(range(N_OUTER_FOLDS)):
            raise ValueError("v12 union must cover five outer folds")
        if len(self.events) != EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_EVENTS:
            raise ValueError("v12 union must contain 1149 events")
        if tuple(event.ordinal for event in self.events) != tuple(range(len(self.events))):
            raise ValueError("v12 event ordinals are not contiguous")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("v12 union repeats an event")
        patient_to_fold = dict(zip(self.patient_ids, self.patient_folds))
        for event in self.events:
            if patient_to_fold.get(event.patient_id) != event.outer_fold:
                raise ValueError("v12 event fold disagrees with its complete patient bag")

    @property
    def patient_index(self) -> dict[str, int]:
        return {patient: index for index, patient in enumerate(self.patient_ids)}

    @property
    def event_patient_index(self) -> tuple[int, ...]:
        index = self.patient_index
        return tuple(index[event.patient_id] for event in self.events)

    @property
    def recovered_append_events(
        self,
    ) -> tuple[PublicDevelopmentUnionIdentityV12Event, ...]:
        return self.events[988:]


def load_public_development_union_identity_v12(
    directory: str | Path,
    *,
    expected_manifest_sha256: str = (
        EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256
    ),
    expected_payload_sha256: str = (
        EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256
    ),
) -> PublicDevelopmentUnionIdentityV12:
    """Strictly load the closed append-only v12 union carrier."""

    declared = Path(os.path.abspath(directory))
    if declared.is_symlink() or not declared.is_dir():
        raise ValueError("v12 union directory must be regular and non-symlinked")
    entries = tuple(sorted(declared.iterdir(), key=lambda path: path.name))
    if (
        len(entries) != 1
        or entries[0].name != PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FILENAME
        or entries[0].is_symlink()
        or not entries[0].is_file()
    ):
        raise ValueError("v12 union directory violates its closed file schema")
    manifest_path = entries[0]
    before = manifest_path.stat()
    if before.st_size < 1 or before.st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("v12 union manifest has an invalid size")
    raw = manifest_path.read_bytes()
    after = manifest_path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("v12 union manifest changed while it was read")
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("v12 union manifest SHA mismatch")
    payload = _strict_v12_json(raw)
    _closed_mapping(payload, fields=_MANIFEST_FIELDS, name="v12 manifest")
    declared_payload_sha = _require_sha256(
        payload["manifest_payload_sha256"], field="manifest_payload_sha256"
    )
    without_payload_sha = dict(payload)
    without_payload_sha.pop("manifest_payload_sha256")
    if (
        declared_payload_sha
        != _require_sha256(expected_payload_sha256, field="expected_payload_sha256")
        or _canonical_sha256(without_payload_sha) != declared_payload_sha
    ):
        raise ValueError("v12 union payload digest changed")

    scalar_checks = {
        "schema_version": PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_SCHEMA,
        "policy": PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_POLICY,
        "purpose": "identity_recovered_public_developmental_nested_cv_only",
        "cohort_name": "public_development_union_identity_v12",
        "patient_count": 103,
        "event_count": 1149,
        "legacy_v11_patient_count": 102,
        "legacy_v11_event_prefix_count": 988,
        "recovered_append_patient_count": 9,
        "recovered_append_event_count": 161,
        "new_patient_count": 1,
        "outer_fold_count": 5,
        "new_patient_outer_fold_salt": NEW_PATIENT_FOLD_SALT,
        "new_patient_outer_fold_assignment_policy": (
            "target_free_existing_burden_then_patient_count_then_salted_hash_v1"
        ),
        "legacy_v11_event_prefix_rows_sha256": (
            EXPECTED_LEGACY_EVENT_PREFIX_ROWS_SHA256
        ),
        "legacy_v11_event_prefix_ids_sha256": (
            EXPECTED_LEGACY_EVENT_PREFIX_IDS_SHA256
        ),
        "recovered_append_event_ids_sha256": (
            EXPECTED_RECOVERED_APPEND_EVENT_IDS_SHA256
        ),
        "event_order_sha256": EXPECTED_V12_EVENT_ORDER_SHA256,
        "patient_roster_sha256": EXPECTED_V12_PATIENT_ROSTER_SHA256,
    }
    failed_scalars = sorted(
        field for field, expected in scalar_checks.items() if payload[field] != expected
    )
    if failed_scalars:
        raise ValueError(f"v12 union schema/count/hash fields changed: {failed_scalars}")

    claim = _closed_mapping(
        payload["claim_boundary"],
        fields=frozenset(
            {
                "public_confirmation_forbidden",
                "public_external_validation_forbidden",
                "legacy_source_eval_retry_claim_forbidden",
                "legacy_splits_retained_for_audit_only",
                "all_103_patients_are_developmental",
                "nested_oof_is_internal_developmental_estimate_only",
                "private_data_must_remain_unread_for_training_selection_calibration",
                "private_is_reserved_for_frozen_zero_adaptation_transfer",
            }
        ),
        name="claim_boundary",
    )
    if any(value is not True for value in claim.values()):
        raise ValueError("v12 public developmental claim boundary was weakened")
    access = _closed_mapping(
        payload["access_receipt"],
        fields=frozenset(
            {
                "legacy_union_metadata_loaded",
                "signal_recovery_metadata_loaded",
                "raw_eeg_loaded",
                "deepsoz_target_values_loaded",
                "private_eeg_loaded",
                "private_target_values_loaded",
                "prediction_artifacts_loaded",
            }
        ),
        name="access_receipt",
    )
    expected_access = {
        "legacy_union_metadata_loaded": True,
        "signal_recovery_metadata_loaded": True,
        "raw_eeg_loaded": False,
        "deepsoz_target_values_loaded": False,
        "private_eeg_loaded": False,
        "private_target_values_loaded": False,
        "prediction_artifacts_loaded": False,
    }
    if dict(access) != expected_access:
        raise ValueError("v12 access receipt is not target/private/prediction free")
    immutability = _closed_mapping(
        payload["immutability_receipt"],
        fields=frozenset(
            {
                "legacy_102_patient_outer_folds_preserved",
                "legacy_988_event_ids_exact_prefix",
                "legacy_988_event_rows_exact_prefix",
                "legacy_recovered_event_overlap_count",
                "new_patient_10489_event_count",
                "new_patient_10489_outer_fold",
                "recovered_events_append_only",
            }
        ),
        name="immutability_receipt",
    )
    expected_immutability = {
        "legacy_102_patient_outer_folds_preserved": True,
        "legacy_988_event_ids_exact_prefix": True,
        "legacy_988_event_rows_exact_prefix": True,
        "legacy_recovered_event_overlap_count": 0,
        "new_patient_10489_event_count": 27,
        "new_patient_10489_outer_fold": 1,
        "recovered_events_append_only": True,
    }
    if dict(immutability) != expected_immutability:
        raise ValueError("v12 append-only immutability receipt changed")

    lineage = _closed_mapping(
        payload["lineage"],
        fields=frozenset(
            {
                "legacy_union_path",
                "legacy_union_manifest_sha256",
                "legacy_union_payload_sha256",
                "signal_identity_recovery_path",
                "signal_identity_recovery_artifact_sha256",
                "signal_identity_recovery_receipt_sha256",
                "base_signal_preflight_artifact_sha256",
                "base_signal_preflight_receipt_sha256",
                "preprocess_config_sha256",
                "combined_signal_event_roster_sha256",
                "recovered_signal_event_ids_sha256",
            }
        ),
        name="lineage",
    )
    lineage_hashes = {
        key: value
        for key, value in lineage.items()
        if key not in {"legacy_union_path", "signal_identity_recovery_path"}
    }
    for field, value in lineage_hashes.items():
        _require_sha256(value, field=f"lineage.{field}")
    expected_lineage = {
        "legacy_union_manifest_sha256": (
            "89a9ca456c724c2dee4d14a2c0da5a1190e58f97ad602060f6dda5f619b97232"
        ),
        "legacy_union_payload_sha256": (
            "8ca1a4af04f6fdb9e2e4bd6a7f0270ef312ceb341bc6d5ba34156ee18903ba1f"
        ),
        "signal_identity_recovery_artifact_sha256": (
            "2a6bb8a7be20993949e7250b10c83d11fe027ff1afc0fa0919124f7fa371ef8e"
        ),
        "signal_identity_recovery_receipt_sha256": (
            "be79e8dc70f553976864a3b8ac6d85a24ff7d7a9769e9e11739ecff43ad701e3"
        ),
        "base_signal_preflight_artifact_sha256": (
            "a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66"
        ),
        "base_signal_preflight_receipt_sha256": (
            "10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446"
        ),
        "recovered_signal_event_ids_sha256": (
            EXPECTED_RECOVERED_APPEND_EVENT_IDS_SHA256
        ),
    }
    if any(lineage[field] != expected for field, expected in expected_lineage.items()):
        raise ValueError("v12 lineage pins changed")

    if payload["new_patient_ids"] != ["10489"]:
        raise ValueError("v12 new-patient roster changed")
    assignments = payload["new_patient_assignment_receipts"]
    if not isinstance(assignments, list) or len(assignments) != 1:
        raise ValueError("v12 new-patient assignment receipt changed")
    assignment = _closed_mapping(
        assignments[0],
        fields=frozenset(
            {
                "patient_id",
                "event_count",
                "selected_outer_fold",
                "fold_event_counts_before",
                "fold_patient_counts_before",
                "selection_key",
            }
        ),
        name="new_patient_assignment_receipt",
    )
    if (
        assignment["patient_id"] != "10489"
        or assignment["event_count"] != 27
        or assignment["selected_outer_fold"] != 1
        or assignment["fold_event_counts_before"]
        != {"0": 203, "1": 199, "2": 270, "3": 228, "4": 222}
        or assignment["fold_patient_counts_before"]
        != {"0": 20, "1": 21, "2": 20, "3": 21, "4": 20}
    ):
        raise ValueError("v12 target-free new-patient fold receipt changed")

    raw_patient_ids = payload["patient_ids"]
    raw_patients = payload["patients"]
    raw_event_ids = payload["event_ids"]
    raw_events = payload["events"]
    if not all(
        isinstance(value, list)
        for value in (raw_patient_ids, raw_patients, raw_event_ids, raw_events)
    ):
        raise TypeError("v12 manifest rosters are missing")
    patient_ids = tuple(str(value) for value in raw_patient_ids)
    if (
        len(patient_ids) != 103
        or len(set(patient_ids)) != 103
        or tuple(sorted(patient_ids[:102])) != patient_ids[:102]
        or patient_ids[-1] != "10489"
        or _canonical_sha256(list(patient_ids)) != payload["patient_roster_sha256"]
    ):
        raise ValueError("v12 patient roster/order/hash changed")
    if len(raw_patients) != 103:
        raise ValueError("v12 patient row count changed")
    patient_rows: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(raw_patients):
        row = _closed_mapping(value, fields=_PATIENT_FIELDS, name=f"patients[{index}]")
        patient_id = str(row["patient_id"])
        if patient_id in patient_rows:
            raise ValueError("v12 repeats a patient row")
        if int(row["outer_fold"]) not in range(N_OUTER_FOLDS):
            raise ValueError("v12 patient outer fold is invalid")
        if int(row["event_count"]) < 1:
            raise ValueError("v12 patient event count must be positive")
        if int(row["legacy_v11_event_count"]) + int(
            row["recovered_appended_event_count"]
        ) != int(row["event_count"]):
            raise ValueError("v12 patient legacy/recovered counts do not close")
        patient_rows[patient_id] = row
    if tuple(patient_rows) != patient_ids:
        raise ValueError("v12 patient rows do not follow the frozen patient order")
    if (
        patient_rows["10489"]["patient_origin"] != "identity_recovery_v3"
        or int(patient_rows["10489"]["outer_fold"]) != 1
        or int(patient_rows["10489"]["event_count"]) != 27
    ):
        raise ValueError("v12 patient 10489 carrier changed")
    patient_folds = tuple(int(patient_rows[patient]["outer_fold"]) for patient in patient_ids)

    event_ids = tuple(str(value) for value in raw_event_ids)
    if len(event_ids) != 1149 or len(set(event_ids)) != 1149:
        raise ValueError("v12 event ID roster is not 1149 unique events")
    if (
        _canonical_sha256(list(event_ids)) != payload["event_order_sha256"]
        or _canonical_sha256(list(event_ids[:988]))
        != payload["legacy_v11_event_prefix_ids_sha256"]
        or _canonical_sha256(list(event_ids[988:]))
        != payload["recovered_append_event_ids_sha256"]
    ):
        raise ValueError("v12 event order/prefix/append hash changed")
    if len(raw_events) != 1149:
        raise ValueError("v12 event row count changed")
    events: list[PublicDevelopmentUnionIdentityV12Event] = []
    event_patient_counts: Counter[str] = Counter()
    fold_event_counts: Counter[int] = Counter()
    split_event_counts: Counter[str] = Counter()
    for index, value in enumerate(raw_events):
        row = _closed_mapping(value, fields=_EVENT_FIELDS, name=f"events[{index}]")
        if str(row["event_id"]) != event_ids[index]:
            raise ValueError("v12 event rows differ from event_ids order")
        event = PublicDevelopmentUnionIdentityV12Event(
            ordinal=int(row["ordinal"]),
            event_id=str(row["event_id"]),
            patient_id=str(row["patient_id"]),
            outer_fold=int(row["outer_fold"]),
            legacy_model_split=str(row["legacy_model_split"]),
            official_split=str(row["official_split"]),
            relative_edf_path=str(row["relative_edf_path"]),
            global_event_index=int(row["global_event_index"]),
            global_t0_sec=float(row["global_t0_sec"]),
            global_stop_sec=float(row["global_stop_sec"]),
            event_record_sha256=str(row["event_record_sha256"]),
            edf_sha256=str(row["edf_sha256"]),
            edf_receipt_sha256=str(row["edf_receipt_sha256"]),
            signal_receipt_sha256=str(row["signal_receipt_sha256"]),
            processed_window_sha256=str(row["processed_window_sha256"]),
            processed_window_shape=tuple(row["processed_window_shape"]),
            processed_window_dtype=str(row["processed_window_dtype"]),
        )
        if event.ordinal != index:
            raise ValueError("v12 event ordinal/order changed")
        events.append(event)
        event_patient_counts[event.patient_id] += 1
        fold_event_counts[event.outer_fold] += 1
        split_event_counts[event.legacy_model_split] += 1
    if _canonical_sha256(raw_events[:988]) != payload[
        "legacy_v11_event_prefix_rows_sha256"
    ]:
        raise ValueError("v12 legacy 988 event rows are not the exact frozen prefix")
    if event_patient_counts != Counter(
        {patient: int(patient_rows[patient]["event_count"]) for patient in patient_ids}
    ):
        raise ValueError("v12 patient event counts disagree with event rows")
    expected_fold_patients = Counter(patient_folds)
    if payload["outer_fold_patient_counts"] != {
        str(fold): expected_fold_patients[fold] for fold in range(N_OUTER_FOLDS)
    }:
        raise ValueError("v12 outer-fold patient counts changed")
    if payload["outer_fold_event_counts"] != {
        str(fold): fold_event_counts[fold] for fold in range(N_OUTER_FOLDS)
    }:
        raise ValueError("v12 outer-fold event counts changed")
    if payload["legacy_split_event_counts"] != dict(sorted(split_event_counts.items())):
        raise ValueError("v12 legacy split event counts changed")
    if event_patient_counts["10489"] != 27 or any(
        event.outer_fold != 1 for event in events if event.patient_id == "10489"
    ):
        raise ValueError("v12 patient 10489 event bag/fold changed")
    return PublicDevelopmentUnionIdentityV12(
        path=declared,
        manifest_sha256=manifest_sha,
        payload=payload,
        patient_ids=patient_ids,
        patient_folds=patient_folds,
        events=tuple(events),
    )


__all__ = [
    "EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_EVENTS",
    "EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256",
    "EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PATIENTS",
    "NEW_PATIENT_FOLD_SALT",
    "N_OUTER_FOLDS",
    "PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FILENAME",
    "PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_POLICY",
    "PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_SCHEMA",
    "PublicDevelopmentUnionIdentityV12",
    "PublicDevelopmentUnionIdentityV12Event",
    "build_public_development_union_identity_v12",
    "load_public_development_union_identity_v12",
    "publish_public_development_union_identity_v12",
]
