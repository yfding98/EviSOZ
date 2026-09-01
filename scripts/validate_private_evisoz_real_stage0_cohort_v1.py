#!/usr/bin/env python3
"""Replay every materialized private EviSOZ Stage-0 event cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.private_stage0_cohort_materializer import (  # noqa: E402
    validate_private_stage0_cohort_artifact,
)


DEFAULT_COHORT = ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_stage0_private_real_dual_montage_validation_v1_20260831"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = validate_private_stage0_cohort_artifact(args.cohort)
    args.output.mkdir(parents=True)
    (args.output / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
