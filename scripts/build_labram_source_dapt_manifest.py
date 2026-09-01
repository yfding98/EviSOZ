#!/usr/bin/env python3
"""Build the immutable annotation-free LaBraM source-DAPT manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.labram_source_dapt import (  # noqa: E402
    build_source_dapt_manifest,
    write_source_dapt_manifest,
)


DEFAULT_TUSZ_EDF_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_DEEPSOZ_SPLIT = ROOT / "outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv"
DEFAULT_DEEPSOZ_CROSSWALK = ROOT / "outputs/deepsoz_tusz_patient_splits_v1/record_crosswalk.csv"
DEFAULT_MASTER = ROOT / "outputs/tusz_ictal_token_corpus_formal_v4_20260809/master/index.json"
DEFAULT_OUTPUT = ROOT / "outputs/labram_source_only_dapt_manifest_v1_20260811/manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tusz-edf-root", type=Path, default=DEFAULT_TUSZ_EDF_ROOT)
    parser.add_argument("--deepsoz-split-roster", type=Path, default=DEFAULT_DEEPSOZ_SPLIT)
    parser.add_argument("--deepsoz-record-crosswalk", type=Path, default=DEFAULT_DEEPSOZ_CROSSWALK)
    parser.add_argument("--historical-master-index", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_source_dapt_manifest(
        tusz_root=args.tusz_edf_root,
        deepsoz_split_roster=args.deepsoz_split_roster,
        deepsoz_record_crosswalk=args.deepsoz_record_crosswalk,
        historical_master_index=args.historical_master_index,
        split_seed=args.split_seed,
    )
    digest = write_source_dapt_manifest(payload, args.output)
    summary = {
        "manifest": str(args.output.resolve()),
        "manifest_sha256": digest,
        "source_patient_count": payload["source_pool_audit"]["source_patient_count"],
        "source_index_row_count": payload["source_pool_audit"]["source_index_row_count"],
        **payload["counts"],
        "deepsoz_identity_overlap": 0,
        "deepsoz_path_overlap": payload["deepsoz_exclusion_contract"][
            "source_resolved_path_overlap_count"
        ],
        "deepsoz_content_overlap": payload["deepsoz_exclusion_contract"][
            "source_content_overlap_count"
        ],
        "target_values_loaded": False,
        "private_data_loaded": False,
        "annotation_sidecars_opened": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
