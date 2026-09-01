#!/usr/bin/env python3
"""Freeze the 102-patient public developmental union for v11.

This is an irreversible *claim-boundary* change, not a new public split.  The
historical source-train/source-dev/source-eval identities are retained only
for audit.  All 102 signal-eligible patients become one developmental cohort;
none can subsequently be described as an untouched public test cohort.

The builder reads target-free signal receipts only.  It does not open DeepSOZ
target values, raw EEG, predictions, or private files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL = ROOT / "outputs/deepsoz_signal_preflight_v2_20260809_current"
DEFAULT_OUTPUT = ROOT / "outputs/public_development_union_v11_20260811"

EXPECTED_SIGNAL_ARTIFACT_SHA256 = (
    "a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66"
)
EXPECTED_SIGNAL_RECEIPT_SHA256 = (
    "10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446"
)
EXPECTED_PATIENT_ROSTER_SHA256 = (
    "49ced5020a7df002b61c0dea523c46ab13f2b9bb4f2978ec3f883b68210c682f"
)
EXPECTED_EVENT_ROSTER_SHA256 = (
    "82453898ec09d1420b0d7de1b15b98cab222a1297ff659093ed6131868bad9e8"
)
SCHEMA = "soz_public_development_union_v11"
MANIFEST_NAME = "manifest.json"
N_OUTER_FOLDS = 5
FOLD_SALT = "public-development-union-v11-outer-folds-20260811"


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_signal(path: Path) -> tuple[dict[str, object], str, str]:
    artifact = path / "deepsoz_signal_preflight.json"
    raw = artifact.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("receipt"), dict):
        raise TypeError("signal preflight has an invalid structure")
    receipt = payload["receipt"]
    receipt_sha = str(payload.get("receipt_sha256"))
    artifact_sha = _file_sha256(artifact)
    checks = {
        "artifact": artifact_sha == EXPECTED_SIGNAL_ARTIFACT_SHA256,
        "receipt": receipt_sha == EXPECTED_SIGNAL_RECEIPT_SHA256,
        "receipt replay": _sha256(receipt) == receipt_sha,
        "patient roster": receipt.get("eligible_patient_roster_sha256")
        == EXPECTED_PATIENT_ROSTER_SHA256,
        "event roster": receipt.get("eligible_event_roster_sha256")
        == EXPECTED_EVENT_ROSTER_SHA256,
        "patient count": receipt.get("eligible_patient_count") == 102,
        "event count": receipt.get("eligible_event_count") == 988,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"signal preflight trust boundary failed: {failed}")
    return receipt, artifact_sha, receipt_sha


def _balanced_patient_folds(
    patient_event_counts: Mapping[str, int],
    *,
    n_folds: int = N_OUTER_FOLDS,
) -> dict[str, int]:
    """Assign patient folds without reading target values.

    Large event bags are placed first into the fold with the smallest current
    event burden, then smallest patient count.  A salted patient hash is the
    deterministic tie breaker.
    """

    if len(patient_event_counts) < n_folds:
        raise ValueError("not enough patients for the requested fold count")

    def patient_key(item: tuple[str, int]) -> tuple[int, str]:
        patient_id, count = item
        digest = hashlib.sha256(f"{FOLD_SALT}|{patient_id}".encode("ascii")).hexdigest()
        return -int(count), digest

    fold_events = [0] * n_folds
    fold_patients = [0] * n_folds
    assignment: dict[str, int] = {}
    for patient_id, event_count in sorted(patient_event_counts.items(), key=patient_key):
        tie_hashes = [
            hashlib.sha256(
                f"{FOLD_SALT}|{patient_id}|fold={fold}".encode("ascii")
            ).hexdigest()
            for fold in range(n_folds)
        ]
        fold = min(
            range(n_folds),
            key=lambda value: (
                fold_events[value],
                fold_patients[value],
                tie_hashes[value],
            ),
        )
        assignment[patient_id] = fold
        fold_events[fold] += int(event_count)
        fold_patients[fold] += 1
    if set(assignment) != set(patient_event_counts):
        raise RuntimeError("patient fold assignment lost a patient")
    return assignment


def build_manifest(signal_directory: Path) -> dict[str, object]:
    receipt, artifact_sha, receipt_sha = _load_signal(signal_directory)
    raw_events = receipt.get("events")
    if not isinstance(raw_events, list) or len(raw_events) != 988:
        raise ValueError("signal preflight does not contain the expected event roster")

    patient_events: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in raw_events:
        if not isinstance(row, Mapping):
            raise TypeError("signal event row must be an object")
        patient_events[str(row["patient_id"])].append(row)
    event_counts = {patient: len(rows) for patient, rows in patient_events.items()}
    folds = _balanced_patient_folds(event_counts)
    patient_ids = tuple(sorted(patient_events))
    event_ids = tuple(str(row["event_id"]) for row in raw_events)
    if len(patient_ids) != 102 or len(set(event_ids)) != 988:
        raise ValueError("developmental union roster is not 102 patients / 988 events")

    patients = [
        {
            "patient_id": patient_id,
            "outer_fold": folds[patient_id],
            "event_count": event_counts[patient_id],
            "legacy_model_split": str(patient_events[patient_id][0]["model_split"]),
        }
        for patient_id in patient_ids
    ]
    events = []
    for ordinal, row in enumerate(raw_events):
        patient_id = str(row["patient_id"])
        events.append(
            {
                "ordinal": ordinal,
                "event_id": str(row["event_id"]),
                "patient_id": patient_id,
                "outer_fold": folds[patient_id],
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
        )
    split_counts = Counter(str(row["legacy_model_split"]) for row in events)
    fold_patient_counts = Counter(row["outer_fold"] for row in patients)
    fold_event_counts = Counter(row["outer_fold"] for row in events)
    manifest = {
        "schema_version": SCHEMA,
        "purpose": "post_v10_public_developmental_nested_cv_only",
        "authorization_basis": (
            "user_requested_execution_of_the_post_v10_recovery_plan_on_2026-08-11"
        ),
        "claim_boundary": {
            "public_confirmation_forbidden": True,
            "public_external_validation_forbidden": True,
            "legacy_source_eval_retry_claim_forbidden": True,
            "legacy_splits_retained_for_audit_only": True,
            "all_102_patients_are_developmental": True,
            "nested_oof_is_internal_developmental_estimate_only": True,
            "private_data_must_remain_unread_for_training_selection_calibration": True,
            "private_is_reserved_for_frozen_zero_adaptation_transfer": True,
        },
        "cohort_name": "public_development_union",
        "patient_count": len(patient_ids),
        "event_count": len(events),
        "legacy_split_event_counts": dict(sorted(split_counts.items())),
        "outer_fold_count": N_OUTER_FOLDS,
        "outer_fold_assignment_policy": (
            "target_free_greedy_event_burden_then_patient_count_then_salted_hash_v1"
        ),
        "outer_fold_salt": FOLD_SALT,
        "outer_fold_patient_counts": {
            str(key): value for key, value in sorted(fold_patient_counts.items())
        },
        "outer_fold_event_counts": {
            str(key): value for key, value in sorted(fold_event_counts.items())
        },
        "patient_ids": list(patient_ids),
        "patient_roster_sha256": _sha256(patient_ids),
        "event_ids": list(event_ids),
        "event_order_sha256": _sha256(event_ids),
        "patients": patients,
        "events": events,
        "lineage": {
            "signal_preflight_path": str(signal_directory.resolve()),
            "signal_preflight_artifact_sha256": artifact_sha,
            "signal_preflight_receipt_sha256": receipt_sha,
            "signal_eligible_patient_roster_sha256": EXPECTED_PATIENT_ROSTER_SHA256,
            "signal_eligible_event_roster_sha256": EXPECTED_EVENT_ROSTER_SHA256,
            "verified_target_v2_artifact_sha256": str(
                receipt["verified_target_v2_artifact_sha256"]
            ),
            "verified_target_v2_policy_sha256": str(
                receipt["verified_target_v2_policy_sha256"]
            ),
            "verified_target_v2_receipt_sha256": str(
                receipt["verified_target_v2_receipt_sha256"]
            ),
            "preprocess_config_sha256": str(receipt["preprocess_config_sha256"]),
        },
        "access_receipt": {
            "signal_metadata_loaded": True,
            "raw_eeg_loaded": False,
            "deepsoz_target_values_loaded": False,
            "source_eval_target_values_loaded": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "prediction_artifacts_loaded": False,
        },
    }
    manifest["manifest_payload_sha256"] = _sha256(manifest)
    return manifest


def publish(manifest: Mapping[str, object], output_directory: Path) -> Path:
    target = Path(os.path.abspath(output_directory))
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        output = staging / MANIFEST_NAME
        output.write_bytes(_canonical_bytes(manifest, newline=True))
        with output.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--signal-directory", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args.signal_directory)
    path = publish(manifest, args.output_directory)
    print(
        json.dumps(
            {
                "status": "PUBLIC_DEVELOPMENT_UNION_V11_FROZEN",
                "path": str(path),
                "manifest_file_sha256": _file_sha256(path / MANIFEST_NAME),
                "patient_count": manifest["patient_count"],
                "event_count": manifest["event_count"],
                "fold_patient_counts": manifest["outer_fold_patient_counts"],
                "fold_event_counts": manifest["outer_fold_event_counts"],
                "public_confirmation_forbidden": True,
                "target_values_loaded": False,
                "private_loaded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
