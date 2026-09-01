#!/usr/bin/env python3
"""Build a PHI-free, review-gated private EEG annotation ledger.

Raw EDF annotation descriptions and Excel cell values are read only by this
offline ingestion step.  The published ledger contains controlled point-marker
types, pseudonymous event bindings, hashes, and an unbound Excel review queue;
it contains no patient names, raw paths, descriptions, diagnoses, or inferred
seizure intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.private_clinical_eeg_annotations import (  # noqa: E402
    materialize_private_annotation_ledger,
)


DEFAULT_DATASET_ROOT = Path("/mnt/hd1/dyf/dataset/EEG")
DEFAULT_BUNDLE = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814"
DEFAULT_EDF_ANNOTATIONS = DEFAULT_DATASET_ROOT / "edf_annotations.csv"
DEFAULT_WORKBOOKS = (
    DEFAULT_DATASET_ROOT / "EEG-fMRI颞叶癫痫(1).xls",
    DEFAULT_DATASET_ROOT / "头皮扩散.xlsx",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--private-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--edf-annotations", type=Path, default=DEFAULT_EDF_ANNOTATIONS
    )
    parser.add_argument(
        "--workbook",
        action="append",
        type=Path,
        default=None,
        help=(
            "Repeat to override the two canonical source workbooks; versioned "
            "or subset copies must be supplied explicitly for separate review"
        ),
    )
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--eeg-root", type=Path, default=None)
    parser.add_argument(
        "--event-id",
        action="append",
        required=True,
        help=(
            "Required pseudonymous private event selection; repeat as needed. "
            "This prevents accidental all-dataset EDF hashing."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ledger = materialize_private_annotation_ledger(
        private_bundle_directory=args.private_bundle,
        edf_annotations_path=args.edf_annotations,
        workbook_paths=tuple(args.workbook or DEFAULT_WORKBOOKS),
        output_path=args.output,
        source_manifest_path=args.source_manifest,
        eeg_root=args.eeg_root,
        selected_event_ids=args.event_id,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": ledger["status"],
                "event_count": len(ledger["events"]),
                "point_marker_count": ledger["exclusion_summary"][
                    "recognized_point_markers"
                ],
                "excel_pending_review_count": len(
                    ledger["pending_excel_review"]
                ),
                "raw_content_released": ledger["claim_boundary"][
                    "raw_description_or_path_released"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
