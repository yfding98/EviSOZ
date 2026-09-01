#!/usr/bin/env python3
"""Materialize the deterministic candidate exposure ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.forge.candidate_exposure_ledger import (  # noqa: E402
    build_candidate_exposure_ledger,
)


DEFAULT_MANIFEST = ROOT / "outputs/evisoz_stage0_deterministic_signal_candidates_v1_20260831/manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_candidate_exposure_ledger_v1_20260831"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.candidate_manifest.is_symlink() or not args.candidate_manifest.is_file():
        raise ValueError("candidate manifest must be a regular file")
    manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    ledger = build_candidate_exposure_ledger(candidate_manifest=manifest)
    args.output.mkdir(parents=True)
    (args.output / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": ledger["status"],
        "ledger_id": ledger["ledger_id"],
        "counts": ledger["counts"],
        "missing_closure_codes": ledger["missing_closure_codes"],
        "training_authorized": ledger["permissions"]["training_authorized"],
        "receipt_sha256": ledger["receipt_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
