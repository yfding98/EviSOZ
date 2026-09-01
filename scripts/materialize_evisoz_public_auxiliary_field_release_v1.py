#!/usr/bin/env python3
"""Materialize the capability-only public auxiliary field release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.public_auxiliary_field_release import (  # noqa: E402
    build_public_auxiliary_field_release,
)


DEFAULT_PROJECTION = ROOT / "outputs/evisoz_public_auxiliary_exposure_projection_v1_20260831/projection.json"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_public_auxiliary_field_release_v1_20260831"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--projection", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.projection.is_symlink() or not args.projection.is_file():
        raise ValueError("public auxiliary projection must be a regular file")
    projection = json.loads(args.projection.read_text(encoding="utf-8"))
    release = build_public_auxiliary_field_release(projection=projection)
    args.output.mkdir(parents=True)
    (args.output / "field_release.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": release["status"],
        "release_id": release["release_id"],
        "counts": release["counts"],
        "training_authorized": release["permissions"]["field_values_training_authorized"],
        "receipt_sha256": release["receipt_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
