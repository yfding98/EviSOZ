#!/usr/bin/env python3
"""Materialize an independently authorized physician-report text release.

This command intentionally requires two external JSON inputs: a governance
authorization reference and a manual-review matrix.  It cannot promote the
current pending de-identification candidates by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.private_physician_report_release import (  # noqa: E402
    materialize_private_physician_report_release,
)


DEFAULT_CANDIDATE_ROOT = (
    ROOT / "outputs/evisoz_stage0_private_report_deid_candidates_v1_20260831"
)
DEFAULT_AUTHORIZATION = ROOT / "inputs/private_report_release_authorization.json"
DEFAULT_REVIEW = ROOT / "inputs/private_report_manual_review.json"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_private_physician_report_release_v1"


def _json(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"release input must be a regular JSON file: {path}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"release input must be a JSON object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate_root = args.candidate_root.resolve(strict=True)
    manifest_path = (
        args.candidate_manifest.resolve(strict=True)
        if args.candidate_manifest is not None
        else candidate_root / "manifest.json"
    )
    candidate_bundle = _json(manifest_path)
    authorization = _json(args.authorization)
    review = _json(args.manual_review)
    reviewed_rows = review.get("rows")
    if not isinstance(reviewed_rows, list):
        raise ValueError("manual-review input must contain a rows array")
    release = materialize_private_physician_report_release(
        candidate_bundle=candidate_bundle,
        candidate_output_root=candidate_root,
        authorization=authorization,
        reviewed_rows=reviewed_rows,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": "private_physician_report_release_materialized",
                "release_id": release["release_id"],
                "counts": release["counts"],
                "receipt_sha256": release["receipt_sha256"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
