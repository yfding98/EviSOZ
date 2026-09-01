#!/usr/bin/env python3
"""Materialize a non-authorizing EviSOZ clean-freeze audit.

This command only reads Git metadata and a fixed repository-relative contract
roster.  A ``GO`` result means that a reproducible snapshot is available; it
does not authorize Stage-0 training or clinical deployment.
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

from src.evisoz.data.artifact_ref import canonical_json_bytes  # noqa: E402
from src.evisoz.data.clean_freeze import (  # noqa: E402
    DEFAULT_CONTRACT_PATHS,
    build_clean_freeze_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage0-gate", type=Path, default=None)
    parser.add_argument(
        "--contract-path",
        action="append",
        dest="contract_paths",
        help="additional repository-relative contract path; may be repeated",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(args.contract_paths or DEFAULT_CONTRACT_PATHS)
    audit = build_clean_freeze_audit(
        repository_root=ROOT,
        contract_paths=paths,
        stage0_gate_path=args.stage0_gate,
        excluded_status_paths=(
            (args.output.resolve().relative_to(ROOT.resolve()),)
            if args.output.resolve().is_relative_to(ROOT.resolve())
            else ()
        ),
    )
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(audit) + b"\n")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "audit_id": audit["audit_id"],
                "receipt_sha256": audit["receipt_sha256"],
                "git_clean": audit["git_snapshot"]["clean"],
                "contract_count": len(audit["required_contracts"]),
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
