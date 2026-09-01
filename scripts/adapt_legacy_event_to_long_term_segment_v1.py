#!/usr/bin/env python3
"""Bind one existing trustworthy event artifact to a long-recording segment."""

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

from src.clinical_eeg_long_recording.adapters import (  # noqa: E402
    adapt_legacy_event_to_long_term_segment,
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _qualified_row(path: Path, event_id: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if type(value) is not dict:
                raise TypeError(f"qualified report row {line_number} is not an object")
            if value.get("unit_id") == event_id:
                matches.append(value)
    if len(matches) != 1:
        raise ValueError("qualified report source must contain exactly one event row")
    return matches[0]


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
    parser.add_argument("--clinical-facts", type=Path, required=True)
    parser.add_argument("--waveform-manifest", type=Path, required=True)
    parser.add_argument("--qualified-reports-jsonl", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--patient-pseudonym", required=True)
    parser.add_argument("--source-signal-sha256", required=True)
    parser.add_argument("--recording-duration-seconds", type=float, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-anchor-seconds", type=float, required=True)
    parser.add_argument("--eeg-event-id", required=True)
    parser.add_argument("--portable-figure-file", required=True)
    parser.add_argument(
        "--ranker-method-id", default="v29_equal_H_D_probability_ensemble"
    )
    parser.add_argument("--ranker-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    waveform_manifest = _object(args.waveform_manifest)
    attachments = waveform_manifest.get("attachments")
    if not isinstance(attachments, list):
        raise TypeError("waveform manifest has no attachments array")
    matches = [
        item
        for item in attachments
        if isinstance(item, Mapping) and item.get("eeg_event_id") == args.eeg_event_id
    ]
    if len(matches) != 1:
        raise ValueError("waveform manifest must contain exactly one event attachment")
    segment = adapt_legacy_event_to_long_term_segment(
        event_report_payload=_object(args.clinical_facts),
        legacy_waveform_attachment=matches[0],
        qualified_report=_qualified_row(
            args.qualified_reports_jsonl, args.eeg_event_id
        ),
        recording_id=args.recording_id,
        patient_pseudonym=args.patient_pseudonym,
        source_signal_sha256=args.source_signal_sha256,
        recording_duration_seconds=args.recording_duration_seconds,
        candidate_id=args.candidate_id,
        candidate_anchor_offset_seconds=args.candidate_anchor_seconds,
        eeg_event_id=args.eeg_event_id,
        portable_figure_file=args.portable_figure_file,
        ranker_method_id=args.ranker_method_id,
        ranker_model_sha256=args.ranker_model_sha256,
    )
    _atomic_json(args.output, segment)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "segment_receipt_id": segment["segment_receipt_id"],
                "eeg_event_id": segment["eeg_event_id"],
                "candidate_anchor_seconds": segment[
                    "candidate_anchor_offset_seconds"
                ],
                "source_context_in_clinical_facts": False,
                "research_soz_in_clinical_facts": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
