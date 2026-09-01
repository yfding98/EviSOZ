#!/usr/bin/env python3
"""Replay an external private clinical-label training authorization receipt.

This command only validates references and scope.  It never opens EDF/DOCX,
creates a loader/optimizer, enables training, or writes a receipt bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.artifact_ref import build_json_artifact_ref, build_raw_artifact_ref  # noqa: E402
from src.evisoz.data.private_training_authorization import (  # noqa: E402
    PRIVATE_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
    validate_private_training_authorization,
)
from src.evisoz.data.split_ledger import SPLIT_ROSTER_SCHEMA_VERSION  # noqa: E402


def _json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"JSON input must be an object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--split-roster", type=Path, required=True)
    parser.add_argument("--signal-roster", type=Path, required=True)
    parser.add_argument("--target-ledger", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--as-of-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authorization_path = args.authorization.resolve(strict=True)
    if authorization_path.is_symlink() or not authorization_path.is_file():
        raise ValueError("authorization must be a regular JSON file")
    split_path = args.split_roster.resolve(strict=True)
    signal_path = args.signal_roster.resolve(strict=True)
    target_path = args.target_ledger.resolve(strict=True)
    source_path = args.source_manifest.resolve(strict=True)
    split = _json(split_path)
    bindings = {
        "dataset_id": "private",
        "patient_roster_sha256": str(split["receipt_sha256"]),
        "split_roster_ref": build_json_artifact_ref(
            split,
            artifact_kind="split_roster",
            payload_schema_version=SPLIT_ROSTER_SCHEMA_VERSION,
        ),
        "signal_roster_ref": build_raw_artifact_ref(
            signal_path.read_bytes(),
            artifact_kind="private_signal_roster",
            media_type="text/csv",
        ),
        "target_ledger_ref": build_raw_artifact_ref(
            target_path.read_bytes(),
            artifact_kind="private_target_ledger",
            media_type="text/csv",
        ),
        "source_manifest_ref": build_raw_artifact_ref(
            source_path.read_bytes(),
            artifact_kind="private_label_authority_manifest",
            media_type="text/csv",
        ),
    }
    authorization = validate_private_training_authorization(
        _json(authorization_path),
        expected_bindings=bindings,
        expected_field_ids={
            "PRIVATE-DIFFUSE-SPREAD",
            "PRIVATE-EARLY-SPREAD-NODES",
            "PRIVATE-EVOLUTION",
            "PRIVATE-LATERALITY",
            "PRIVATE-LOCALIZABILITY",
            "PRIVATE-MORPHOLOGY",
            "PRIVATE-ONSET-NODES",
            "PRIVATE-ONSET-REGIONS",
            "PRIVATE-PHYSICIAN-REPORT-TEXT",
            "PRIVATE-QUALITY",
        },
        as_of_utc=args.as_of_utc,
    )
    print(
        json.dumps(
            {
                "status": "validated",
                "schema_version": PRIVATE_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
                "authorization_id": authorization["authorization_id"],
                "receipt_sha256": authorization["receipt_sha256"],
                "allowed_evisoz_roles": authorization["field_scope"][
                    "allowed_evisoz_roles"
                ],
                "field_permissions": authorization["field_scope"][
                    "field_permissions"
                ],
                "locked_test_training_allowed": authorization["field_scope"][
                    "locked_test_training_allowed"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
