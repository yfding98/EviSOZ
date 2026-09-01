#!/usr/bin/env python3
"""Audit EEG-only DeepSOZ detector-input compatibility without inference.

Only the three identity/split columns projected by the posterior materializer
and EDF signal headers/samples are read.  The output replaces recording and
patient identifiers with deterministic opaque hashes and never reads EDF
annotations, seizure intervals, SOZ labels, spreadsheets, or clinical text.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_deepsoz_continuous_posteriors import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_TUSZ_ROOT,
    PUBLISHED_DEEPSOZ_MISSING_CHANNEL_POLICY,
    PUBLISHED_DEEPSOZ_UTILS_PREPROCESS_SHA256,
    _atomic_text,
    _canonical_sha256,
    _read_complete_standard19,
    _safe_edf,
    _selected_manifest_rows,
)


SCHEMA_VERSION = "deepsoz_detector_input_compatibility_audit_v1"


def _opaque(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = _selected_manifest_rows(
        args.manifest,
        split=args.split,
        recording_id=None,
        max_records=args.max_records,
    )
    root = args.tusz_root.resolve(strict=True)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        recording_id = str(row["local_edf_path"])
        patient_id = str(row["deepsoz_patient_id"])
        recording_key = _opaque("DSZREC-", recording_id)
        patient_key = _opaque("DSZPAT-", patient_id)
        try:
            eeg, sampling_rate, receipt = _read_complete_standard19(
                _safe_edf(root, recording_id)
            )
        except Exception as exc:
            failures.append(
                {
                    "recording_key": recording_key,
                    "patient_key": patient_key,
                    "failure_type": type(exc).__name__,
                    "failure_code": str(exc),
                }
            )
            continue
        successes.append(
            {
                "recording_key": recording_key,
                "patient_key": patient_key,
                "sampling_rate_hz": sampling_rate,
                "sample_count": int(eeg.shape[1]),
                "duration_seconds": float(eeg.shape[1] / sampling_rate),
                "observed_channel_count": receipt["observed_channel_count"],
                "imputed_channel_ids": receipt["imputed_channel_ids"],
                "input_channel_receipt_id": receipt["receipt_id"],
                "input_channel_receipt_sha256": _canonical_sha256(receipt),
            }
        )

    pattern_counts = Counter(
        ",".join(row["imputed_channel_ids"]) or "none" for row in successes
    )
    rate_counts = Counter(str(row["sampling_rate_hz"]) for row in successes)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": "DEEPSOZ-INPUT-AUDIT-PENDING",
        "selected_split": args.split,
        "inventory_scope": (
            "full_selected_split" if args.max_records == 0 else "explicit_subset"
        ),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selected_recording_count": len(rows),
        "selected_patient_count": len(
            {str(row["deepsoz_patient_id"]) for row in rows}
        ),
        "compatible_recording_count": len(successes),
        "failed_recording_count": len(failures),
        "records_with_detector_imputation": sum(
            bool(row["imputed_channel_ids"]) for row in successes
        ),
        "detector_imputed_channel_total": sum(
            len(row["imputed_channel_ids"]) for row in successes
        ),
        "missing_channel_pattern_counts": dict(sorted(pattern_counts.items())),
        "sampling_rate_counts": dict(sorted(rate_counts.items())),
        "total_eeg_hours": sum(row["duration_seconds"] for row in successes)
        / 3600.0,
        "published_missing_channel_policy": (
            PUBLISHED_DEEPSOZ_MISSING_CHANNEL_POLICY
        ),
        "published_policy_source_sha256": (
            PUBLISHED_DEEPSOZ_UTILS_PREPROCESS_SHA256
        ),
        "imputed_channels_clinical_evidence_eligible": False,
        "successes": successes,
        "failures": failures,
        "scope_receipt": {
            "eeg_samples_and_signal_headers_used": True,
            "edf_annotations_used": False,
            "seizure_or_soz_labels_used": False,
            "excel_or_doctor_labels_used": False,
            "recording_paths_or_patient_ids_exported": False,
            "model_inference_run": False,
            "sota_claim_authorized": False,
        },
    }
    body["audit_id"] = "DSZINPUTAUDIT-" + _canonical_sha256(body)[:24]
    _atomic_text(
        args.output,
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return body


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DeepSOZ detector-input compatibility without inference"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--split", default="source_dev")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = audit(_parse_args())
    summary = {
        key: result[key]
        for key in (
            "audit_id",
            "selected_split",
            "selected_recording_count",
            "compatible_recording_count",
            "failed_recording_count",
            "records_with_detector_imputation",
            "missing_channel_pattern_counts",
            "total_eeg_hours",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
