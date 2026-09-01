#!/usr/bin/env python3
"""Materialize the privacy-safe public auxiliary patient/exposure projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.public_exposure_projection import (  # noqa: E402
    build_public_auxiliary_exposure_projection,
)


DEFAULT_SOURCE = (
    ROOT
    / "outputs/clinical_eeg_full_stack_nested_exposure_graph_v1_20260824r1/exposure_registry.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/evisoz_public_auxiliary_exposure_projection_v1_20260831"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    source = json.loads(args.source.resolve(strict=True).read_text(encoding="utf-8"))
    projection = build_public_auxiliary_exposure_projection(source)
    args.output.mkdir(parents=True)
    (args.output / "projection.json").write_text(
        json.dumps(projection, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": projection["status"],
                "patient_count": projection["counts"][
                    "tusz_source_train_patient_count"
                ],
                "outer_fold_patient_counts": projection["counts"][
                    "outer_fold_patient_counts"
                ],
                "deepsoz_overlap_count": projection["counts"][
                    "deepsoz_source_train_overlap_patient_count"
                ],
                "tuev_overlap_count": projection["counts"][
                    "tuev_train_visible_overlap_patient_count"
                ],
                "training_authorized": projection["permissions"][
                    "training_authorized_by_projection"
                ],
                "receipt_sha256": projection["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
