#!/usr/bin/env python3
"""Atomically publish the unique pre-result ictal promotion-gate policy.

This CLI deliberately exposes no numerical threshold and no policy-document
path.  It can only publish the repository-pinned policy document, then it
strictly reloads the resulting bundle before printing the two hashes that a
formal training command must pin independently.
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

from src.soz.ictal_gate_policy import (  # noqa: E402
    ICTAL_LOCKED_PROMOTION_GATE_POLICY_DOCUMENT_RELATIVE_PATH,
    materialize_ictal_promotion_gate_policy,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the unique code-pinned ictal promotion-gate policy; "
            "threshold overrides are intentionally unavailable"
        )
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifact = materialize_ictal_promotion_gate_policy(
        policy_document_path=(
            ROOT / ICTAL_LOCKED_PROMOTION_GATE_POLICY_DOCUMENT_RELATIVE_PATH
        ),
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "artifact_sha256": artifact.artifact_sha256,
                "bundle_receipt_sha256": artifact.receipt_sha256,
                "policy_receipt_sha256": artifact.policy_receipt_sha256,
                "policy_document_sha256": artifact.policy_document_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
