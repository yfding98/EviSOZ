#!/usr/bin/env python3
"""Evaluate frozen private long-recording SOZ Top-5 against post-freeze GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.private_recording_soz_postfreeze_evaluation_v1 import (  # noqa: E402
    evaluate_private_recording_soz_postfreeze,
    write_evaluation_artifacts_append_only,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-cohort", type=Path, required=True)
    parser.add_argument("--doctor-label-bundle", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = evaluate_private_recording_soz_postfreeze(
        prediction_cohort_path=args.prediction_cohort,
        doctor_bundle_path=args.doctor_label_bundle,
    )
    writes = write_evaluation_artifacts_append_only(
        artifact,
        output_json=args.output_json,
        output_report=args.output_report,
    )
    summary = {
        "status": artifact["status"],
        "evaluation_id": artifact["evaluation_id"],
        "content_sha256": artifact["content_sha256"],
        "coverage": artifact["coverage"],
        "hard_metrics": artifact["metrics"]["hard_significant_electrodes"],
        "laterality": artifact["metrics"]["laterality_compatible"],
        "region": artifact["metrics"]["region_compatible"],
        "writes": writes,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
