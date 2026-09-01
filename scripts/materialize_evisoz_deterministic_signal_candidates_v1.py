#!/usr/bin/env python3
"""Materialize deterministic Stage-0 candidates from real dual-montage caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.forge.deterministic_signal_candidates import (  # noqa: E402
    materialize_deterministic_signal_candidates,
)


DEFAULT_REAL_COHORT = (
    ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/evisoz_stage0_deterministic_signal_candidates_v1_20260831"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--real-cohort", type=Path, default=DEFAULT_REAL_COHORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_deterministic_signal_candidates(
        real_cohort_root=args.real_cohort,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "materialization_id": result["materialization_id"],
                "event_count": result["counts"]["event_count"],
                "candidate_count": result["counts"]["candidate_count"],
                "candidate_concept_counts": result["counts"][
                    "candidate_concept_counts"
                ],
                "fold_local_calibration_receipt_count": result["counts"][
                    "fold_local_calibration_receipt_count"
                ],
                "node_localization_supervision_candidate_count": result["counts"][
                    "node_localization_supervision_candidate_count"
                ],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
