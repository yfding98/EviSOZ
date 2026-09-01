#!/usr/bin/env python3
"""Build the independent DeepSOZ patient-target v2 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    build_deepsoz_target_v2_artifact,
)


DEFAULT_SOURCE = Path(
    "outputs/deepsoz_llm_tusz_all_607_20260801/source/TUH_manifest_final.csv"
)
DEFAULT_SPLIT = Path("outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv")
DEFAULT_OUTPUT = Path("outputs/deepsoz_target_v2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the DeepSOZ benchmark-complement target-v2 policy "
            "without modifying the frozen v1 split package"
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only existing target-v2 files; v1 package writes stay forbidden",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_deepsoz_target_v2_artifact(
        args.source,
        args.split,
        args.output_dir,
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "policy_version": summary["policy_version"],
                "patients_total": summary["counts"]["patients_total"],
                "patients_eligible": summary["counts"]["patients_eligible"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
