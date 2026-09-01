#!/usr/bin/env python3
"""Atomically publish the frozen 102-patient public v29 held-fold cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.baseline.v29_public_cache_materializer import (  # noqa: E402
    DEFAULT_PUBLIC_CACHE_DIRECTORY,
    materialize_public_v29_cache_to_disk,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_PUBLIC_CACHE_DIRECTORY,
        help="New output directory; an existing path is always rejected.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_public_v29_cache_to_disk(args.output_directory)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_directory": str(result.path),
                "cache_id": result.cache_id,
                "patient_count": result.materialization_receipt["patient_count"],
                "p0_c18_tensor_sha256": result.materialization_receipt[
                    "p0_c18_tensor_sha256"
                ],
                "source": result.materialization_receipt["source"],
                "independently_recomputed_from_targets": result.materialization_receipt[
                    "independently_recomputed_from_targets"
                ],
                "target_tensor_values_deserialized": result.materialization_receipt[
                    "target_tensor_values_deserialized"
                ],
                "targets_or_target_mask_get_tensor_calls": (
                    result.materialization_receipt[
                        "targets_or_target_mask_get_tensor_calls"
                    ]
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
