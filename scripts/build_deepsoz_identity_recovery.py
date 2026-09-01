#!/usr/bin/env python3
"""Materialize the deterministic DeepSOZ-to-TUSZ identity recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.deepsoz_identity_recovery import (  # noqa: E402
    materialize_identity_recovery,
)


DEFAULT_SOURCE = (
    ROOT
    / "outputs/deepsoz_tusz_adapted_manifest_20260803/source/TUH_manifest_final.csv"
)
DEFAULT_MAPPING = (
    ROOT / "outputs/deepsoz_tusz_adapted_manifest_20260803/mapping.csv"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")


def main() -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=__doc__,
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--conservative-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--time-tolerance-sec", type=float, default=0.25)
    args = parser.parse_args()
    if args.time_tolerance_sec < 0:
        parser.error("--time-tolerance-sec must be non-negative")
    summary = materialize_identity_recovery(
        args.source,
        args.conservative_mapping,
        args.tusz_root,
        args.output_directory,
        tolerance_sec=args.time_tolerance_sec,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
