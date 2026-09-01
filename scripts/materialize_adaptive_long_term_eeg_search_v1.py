#!/usr/bin/env python3
"""Materialize EEG-only adaptive searches around long-recording candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.adaptive_search_materialization import (  # noqa: E402
    materialize_adaptive_eeg_search,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--detection-manifest", type=Path, required=True)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--event-map",
        type=Path,
        help="Optional JSON object mapping detector candidate IDs to EEG event IDs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    event_map = None
    if args.event_map is not None:
        value = json.loads(args.event_map.resolve(strict=True).read_text(encoding="utf-8"))
        if type(value) is not dict or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in value.items()
        ):
            raise TypeError("--event-map must contain a JSON string-to-string object")
        event_map = value
    artifact = materialize_adaptive_eeg_search(
        detection_manifest_path=args.detection_manifest,
        edf_path=args.edf,
        output_path=args.output,
        event_id_by_candidate=event_map,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "recording_id": artifact["recording_id"],
                "event_count": artifact["event_count"],
                "qualified_complete": sum(
                    item["status"] == "qualified_complete"
                    for item in artifact["events"]
                ),
                "edf_annotations_used": False,
                "excel_used": False,
                "labels_or_ground_truth_used": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
