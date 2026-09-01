#!/usr/bin/env python3
"""Evaluate a frozen exploratory ST16 source-dev dense prediction inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.st16_common17_source_dev_evaluation_v1 import (  # noqa: E402
    evaluate_st16_source_dev,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Post-freeze source-dev-only ST16 strict and pinned SzCORE-compatible "
            "evaluation. There is intentionally no source-eval option."
        )
    )
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument(
        "--canonical-physical-projection", type=Path, required=True
    )
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_st16_source_dev(
        prediction_manifest_path=args.prediction_manifest,
        canonical_physical_projection_path=args.canonical_physical_projection,
        reference_manifest_path=args.reference_manifest,
        output_dir=args.output_dir,
        project_root=ROOT,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_threshold": result["threshold_selection"][
                    "selected_threshold"
                ],
                "receipt_sha256": result["receipt_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
