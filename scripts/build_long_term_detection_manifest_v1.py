#!/usr/bin/env python3
"""Build a de-identified long-recording detector manifest from frozen alarms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.detection import (  # noqa: E402
    build_long_term_detection_manifest,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError("detector alarm input must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "JSON object containing recording identity, detector_receipt, "
            "raw_alarm_observations and merge policy."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = _json(args.input)
    required = {
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "recording_duration_seconds",
        "detector_receipt",
        "raw_alarm_observations",
        "merge_gap_seconds",
        "max_selected_candidates",
    }
    if set(source) != required:
        raise ValueError("detector alarm input has missing or unknown keys")
    manifest = build_long_term_detection_manifest(**source)
    _atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_id": manifest["manifest_id"],
                "detector_role": manifest["detector_receipt"]["detector_role"],
                "raw_alarm_count": len(manifest["raw_alarms"]),
                "selected_candidate_count": sum(
                    item["decision"] == "selected_for_event_analysis"
                    for item in manifest["merge_candidates"]
                ),
                "candidate_semantics": manifest["candidate_semantics"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
