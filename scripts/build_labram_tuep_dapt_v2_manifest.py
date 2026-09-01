#!/usr/bin/env python3
"""Build the locked target-free TUEP manifest for LaBraM DAPT-v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.labram_tuep_dapt_v2 import (  # noqa: E402
    SPLIT_SEED,
    build_tuep_dapt_v2_manifest,
    write_tuep_dapt_v2_manifest,
)


DEFAULT_TUEP_ROOT = Path("/mnt/hd1/dyf/dataset/tuh_eeg_epilepsy/v2.0.1")
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_AUDITED_SOURCE_MANIFEST = (
    ROOT / "outputs/labram_source_only_dapt_manifest_v1_20260811/manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/labram_tuep_dapt_v2_manifest_20260811/manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuep-root", type=Path, default=DEFAULT_TUEP_ROOT)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument(
        "--audited-source-manifest",
        type=Path,
        default=DEFAULT_AUDITED_SOURCE_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_tuep_dapt_v2_manifest(
        tuep_root=args.tuep_root,
        tusz_edf_root=args.tusz_root,
        audited_source_manifest=args.audited_source_manifest,
        split_seed=SPLIT_SEED,
    )
    digest = write_tuep_dapt_v2_manifest(payload, args.output)
    summary = {
        "manifest": str(args.output.resolve()),
        "manifest_sha256": digest,
        "counts": payload["counts"],
        "split_patient_counts": {
            "pretext_train": len(
                payload["pretext_split_contract"]["train_patient_ids"]
            ),
            "pretext_dev": len(
                payload["pretext_split_contract"]["dev_patient_ids"]
            ),
            "pretext_qualification": len(
                payload["pretext_split_contract"]["qualification_patient_ids"]
            ),
        },
        "target_values_loaded": False,
        "private_data_loaded": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
