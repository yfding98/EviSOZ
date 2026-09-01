#!/usr/bin/env python3
"""Inventory possible CerebraGloss/ELM files without admitting them.

The command is intentionally read-only.  It records missing or
unvalidated candidates and never produces teacher caches or training
authorization.
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
from src.evisoz.forge.teacher_discovery import (  # noqa: E402
    build_teacher_artifact_discovery,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--teacher-id", choices=("cerebragloss", "elm"), required=True)
    parser.add_argument("--root", action="append", required=True, help="directory to scan; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--hash-files", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    discovery = build_teacher_artifact_discovery(
        teacher_id=args.teacher_id,
        roots=tuple(args.root),
        max_depth=args.max_depth,
        max_candidates=args.max_candidates,
        hash_files=args.hash_files,
    )
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(discovery) + b"\n")
    print(json.dumps({
        "status": discovery["status"],
        "teacher_id": discovery["teacher_id"],
        "candidate_count": discovery["counts"]["candidate_count"],
        "missing_closure_codes": discovery["missing_closure_codes"],
        "training_authorized": discovery["permissions"]["training_authorized"],
        "output": str(output),
        "receipt_sha256": discovery["receipt_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
