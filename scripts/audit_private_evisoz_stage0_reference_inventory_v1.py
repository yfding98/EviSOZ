#!/usr/bin/env python3
"""Materialize the privacy-safe private EviSOZ Stage-0 reference audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.real_stage0_reference_audit import (  # noqa: E402
    audit_private_stage0_reference_inventory,
)


DEFAULT_MANIFEST = ROOT / "outputs/soz_pre/private_edf_soz_manifest.csv"
DEFAULT_EEG_ROOT = Path("/mnt/hd1/dyf/dataset/EEG")
DEFAULT_OUTPUT = (
    ROOT / "outputs/evisoz_stage0_private_reference_audit_v1_20260831"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = audit_private_stage0_reference_inventory(
        args.manifest,
        args.eeg_root,
    )
    args.output.mkdir(parents=True)
    (args.output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "unique_edf_count": result["unique_edf_count"],
                "aggregate": result["aggregate"],
                "receipt_sha256": result["receipt_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
