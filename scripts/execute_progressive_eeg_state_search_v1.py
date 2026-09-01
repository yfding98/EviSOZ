#!/usr/bin/env python3
"""Execute an opt-in EEG-only progressive event state-search plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.progressive_state_search_executor import (  # noqa: E402
    execute_progressive_eeg_state_search,
)


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("progressive executor JSON input must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("progressive executor JSON input must be a regular file")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"progressive executor JSON repeats key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"progressive executor JSON contains {value!r}")

    payload = json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )
    if type(payload) is not dict:
        raise TypeError("progressive executor JSON input must contain an object")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--continuous-decoding", type=Path, required=True)
    parser.add_argument("--progressive-plan", type=Path, required=True)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = execute_progressive_eeg_state_search(
        continuous_decoding_receipt=_strict_json(args.continuous_decoding),
        progressive_search_plan=_strict_json(args.progressive_plan),
        edf_path=args.edf,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "recording_id": receipt["recording_id"],
                "event_count": receipt["event_count"],
                "qualified_complete_event_count": receipt["summary"][
                    "qualified_complete_event_count"
                ],
                "censored_unresolved_event_count": receipt["summary"][
                    "censored_unresolved_event_count"
                ],
                "default_batch_integration_enabled": False,
                "edf_annotations_used": False,
                "excel_used": False,
                "labels_or_ground_truth_used": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

