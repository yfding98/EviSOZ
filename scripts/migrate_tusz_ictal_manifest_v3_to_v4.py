#!/usr/bin/env python3
"""Strict schema-only migration for a non-preflighted TUSZ ictal v3 bundle.

Signal-preflight receipts are schema-sensitive and cannot be promoted by tag
replacement.  Preflighted v3 inputs are therefore rejected and must instead
be replayed from EDF with ``repreflight_tusz_ictal_manifest.py``.  This command
only migrates planning manifests with no signal receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data import tusz_training as tt  # noqa: E402


OLD_MANIFEST_SCHEMA = "tusz_ictal_training_manifest_v3.0.0"
OLD_BUNDLE_SCHEMA = "tusz_ictal_training_bundle_v3.0.0"


def _sha(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256")
    return normalized


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", type=_sha, required=True)
    parser.add_argument("--expected-receipt-sha256", type=_sha, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source_bundle.absolute()
    if source.is_symlink() or not source.is_dir() or source.resolve(strict=True) != source:
        raise ValueError("source bundle must be one canonical regular directory")
    if {item.name for item in source.iterdir()} != {"manifest.json", "receipt.json"}:
        raise ValueError("source bundle must contain only manifest.json and receipt.json")

    manifest_bytes = (source / "manifest.json").read_bytes()
    receipt_bytes = (source / "receipt.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != args.expected_bundle_sha256:
        raise ValueError("source bundle SHA-256 mismatch")
    if hashlib.sha256(receipt_bytes).hexdigest() != args.expected_receipt_sha256:
        raise ValueError("source receipt SHA-256 mismatch")
    manifest_payload = json.loads(manifest_bytes)
    receipt_payload = json.loads(receipt_bytes)
    if _canonical_json(manifest_payload) != manifest_bytes:
        raise ValueError("source manifest is not canonical JSON")
    if _canonical_json(receipt_payload) != receipt_bytes:
        raise ValueError("source receipt is not canonical JSON")
    if manifest_payload.get("schema_version") != OLD_BUNDLE_SCHEMA:
        raise ValueError("source is not the frozen v3 bundle schema")
    if receipt_payload.get("schema_version") != OLD_MANIFEST_SCHEMA:
        raise ValueError("source is not the frozen v3 receipt schema")
    if manifest_payload.get("receipt_sha256") != args.expected_receipt_sha256:
        raise ValueError("source bundle does not bind the expected receipt")
    if manifest_payload.get("source_manifest_sha256") != args.expected_receipt_sha256:
        raise ValueError("source bundle source-manifest binding is inconsistent")

    required_v4_fields = {
        "authorized_source_record_sha256s",
        "excluded_source_record_sha256s",
        "discovered_source_count",
        "preprocess_config",
        "preflight_performed",
    }
    missing = sorted(required_v4_fields - set(receipt_payload))
    if missing:
        raise ValueError(f"v3 receipt lacks v4-required fields: {missing}")
    events = receipt_payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("v3 receipt must contain a non-empty event roster")
    if any(event.get("schema_version") != OLD_MANIFEST_SCHEMA for event in events):
        raise ValueError("v3 event roster contains mixed or unexpected schemas")
    if receipt_payload.get("preflight_performed") is not False or any(
        event.get("signal_preflight_receipt_sha256") is not None for event in events
    ):
        raise ValueError(
            "Preflighted v3 receipts cannot be schema-only promoted; replay "
            "them with repreflight_tusz_ictal_manifest.py"
        )

    migrated = dict(receipt_payload)
    migrated["schema_version"] = tt.TUSZ_ICTAL_TRAINING_MANIFEST_SCHEMA
    migrated["events"] = [
        {**event, "schema_version": tt.TUSZ_ICTAL_TRAINING_MANIFEST_SCHEMA}
        for event in events
    ]
    reconstructed = tt._manifest_from_receipt_payload(migrated)
    artifact = tt.save_tusz_ictal_training_manifest(
        args.output_bundle, reconstructed
    )
    verified = tt.load_tusz_ictal_training_manifest(
        artifact.path,
        expected_bundle_manifest_sha256=artifact.bundle_manifest_sha256,
        expected_source_manifest_sha256=artifact.source_manifest_sha256,
    )
    if verified != reconstructed:
        raise RuntimeError("strict v4 reload disagrees with migrated manifest")
    summary = {
        "schema_version": "tusz_ictal_manifest_schema_migration_receipt_v1",
        "migration_semantics": "schema_tags_only",
        "source_bundle_sha256": args.expected_bundle_sha256,
        "source_receipt_sha256": args.expected_receipt_sha256,
        "output_bundle": str(artifact.path),
        "output_bundle_sha256": artifact.bundle_manifest_sha256,
        "output_receipt_sha256": artifact.receipt_sha256,
        "event_count": len(verified.events),
        "patient_count": len(verified.patient_ids),
        "preflight_performed": verified.preflight_performed,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
