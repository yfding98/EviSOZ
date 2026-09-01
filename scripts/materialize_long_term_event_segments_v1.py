#!/usr/bin/env python3
"""Replay and materialize all detector-selected long-EEG event segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.event_materialization import (  # noqa: E402
    materialize_long_term_event_segments,
)


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"JSON input must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"JSON input must be a regular file: {path}")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"JSON contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"JSON contains invalid constant {value!r}")

    payload = json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )
    if type(payload) is not dict:
        raise TypeError(f"JSON input must contain an object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--recording-edf", type=Path, required=True)
    parser.add_argument("--detection-manifest", type=Path, required=True)
    parser.add_argument(
        "--event-registry",
        "--event-id-assignment",
        dest="event_registry",
        type=Path,
        required=True,
        help=(
            "Detector-aligned frozen event registry or strict v29 candidate/event "
            "assignment JSON."
        ),
    )
    parser.add_argument(
        "--ranking-manifest",
        type=Path,
        required=True,
        help="manifest.json produced by materialize_v29_long_recording_rankings.py",
    )
    parser.add_argument(
        "--analysis-selection",
        type=Path,
        help=(
            "Signal-only candidate partition produced by the filtered v29 stage; "
            "required when ranking excludes signal-ineligible candidates."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = materialize_long_term_event_segments(
        recording_path=args.recording_edf,
        detection_manifest=_strict_json(args.detection_manifest),
        event_registry=_strict_json(args.event_registry),
        ranking_manifest=_strict_json(args.ranking_manifest),
        analysis_selection=(
            _strict_json(args.analysis_selection)
            if args.analysis_selection is not None
            else None
        ),
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recording_id": manifest["recording_id"],
                "event_count": manifest["event_count"],
                "waveform_root": str(args.output),
                "unsigned_research_ai_draft": True,
                "candidate_is_confirmed_seizure": False,
                "annotation_or_excel_loaded": False,
                "raw_edf_path_persisted": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
