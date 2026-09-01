#!/usr/bin/env python3
"""Publish the append-only 103-patient public-development union v12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.public_development_union_identity_v12 import (  # noqa: E402
    PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FILENAME,
    build_public_development_union_identity_v12,
    publish_public_development_union_identity_v12,
)


DEFAULT_LEGACY_UNION = ROOT / "outputs/public_development_union_v11_20260811"
DEFAULT_SIGNAL_RECOVERY = (
    ROOT / "outputs/deepsoz_signal_preflight_identity_v3_20260812"
)
DEFAULT_OUTPUT = ROOT / "outputs/public_development_union_identity_v12_20260812"
EXPECTED_LEGACY_MANIFEST_SHA256 = (
    "89a9ca456c724c2dee4d14a2c0da5a1190e58f97ad602060f6dda5f619b97232"
)
EXPECTED_LEGACY_PAYLOAD_SHA256 = (
    "8ca1a4af04f6fdb9e2e4bd6a7f0270ef312ceb341bc6d5ba34156ee18903ba1f"
)
EXPECTED_SIGNAL_RECOVERY_ARTIFACT_SHA256 = (
    "2a6bb8a7be20993949e7250b10c83d11fe027ff1afc0fa0919124f7fa371ef8e"
)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--legacy-union-directory", type=Path, default=DEFAULT_LEGACY_UNION
    )
    parser.add_argument(
        "--signal-recovery-directory", type=Path, default=DEFAULT_SIGNAL_RECOVERY
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_public_development_union_identity_v12(
        args.legacy_union_directory,
        args.signal_recovery_directory,
        expected_legacy_manifest_sha256=EXPECTED_LEGACY_MANIFEST_SHA256,
        expected_legacy_payload_sha256=EXPECTED_LEGACY_PAYLOAD_SHA256,
        expected_signal_recovery_artifact_sha256=(
            EXPECTED_SIGNAL_RECOVERY_ARTIFACT_SHA256
        ),
    )
    output = publish_public_development_union_identity_v12(
        manifest, args.output_directory
    )
    artifact = output / PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FILENAME
    import hashlib

    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    summary = {
        "status": "PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FROZEN",
        "path": str(output),
        "manifest_file_sha256": artifact_sha,
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "patient_count": manifest["patient_count"],
        "event_count": manifest["event_count"],
        "legacy_event_prefix_count": manifest["legacy_v11_event_prefix_count"],
        "recovered_append_event_count": manifest["recovered_append_event_count"],
        "new_patient_ids": manifest["new_patient_ids"],
        "new_patient_assignment_receipts": manifest[
            "new_patient_assignment_receipts"
        ],
        "outer_fold_patient_counts": manifest["outer_fold_patient_counts"],
        "outer_fold_event_counts": manifest["outer_fold_event_counts"],
        "immutability_receipt": manifest["immutability_receipt"],
        "target_values_loaded": False,
        "private_loaded": False,
        "predictions_loaded": False,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
