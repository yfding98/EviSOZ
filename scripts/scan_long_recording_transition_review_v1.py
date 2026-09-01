#!/usr/bin/env python3
"""Run the label-neutral transition preselector and publish only safe receipts.

This is a heuristic review-queue provider, not a clinically validated seizure
detector.  It accepts one explicitly selected EDF, never loads an onset
manifest, discards raw paths/patient identity from output, and wraps the
candidate scores in ``long_term_seizure_detection_manifest_v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.auto_annotate.full_record_soz_scan import (  # noqa: E402
    DEFAULT_WEIGHTS,
    RecordingSource,
    extract_full_record_feature_timeline,
    load_full_record_segment,
    recording_duration_s,
    select_transition_candidates,
)
from src.clinical_eeg_long_recording.detection import (  # noqa: E402
    build_long_term_detection_manifest,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--patient-pseudonym", required=True)
    parser.add_argument("--source-signal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-sfreq", type=float, default=100.0)
    parser.add_argument("--highpass-hz", type=float, default=0.5)
    parser.add_argument("--lowpass-hz", type=float, default=40.0)
    parser.add_argument("--window-s", type=float, default=4.0)
    parser.add_argument("--step-s", type=float, default=2.0)
    parser.add_argument("--rolling-baseline-s", type=float, default=120.0)
    parser.add_argument("--future-s", type=float, default=8.0)
    parser.add_argument("--min-history-s", type=float, default=20.0)
    parser.add_argument("--candidates-per-hour", type=float, default=24.0)
    parser.add_argument("--min-candidates", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--min-gap-s", type=float, default=12.0)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    edf = args.edf.resolve(strict=True)
    if edf.is_symlink() or not edf.is_file() or edf.suffix.lower() != ".edf":
        raise ValueError("--edf must be a regular non-symlink EDF file")
    actual_sha = _sha256_file(edf)
    if actual_sha != args.source_signal_sha256:
        raise ValueError("EDF SHA-256 does not match the de-identified binding")
    duration = recording_duration_s(edf)
    recording = RecordingSource(
        ordinal=0,
        dataset="private_long_recording",
        split="inference_only",
        patient_id=args.patient_pseudonym,
        edf_path=str(edf),
        relative_edf_path="withheld.edf",
        reference_onsets_s=(),
        manifest_rows=(),
    )
    segment = load_full_record_segment(
        recording,
        duration_s=duration,
        analysis_sfreq=args.analysis_sfreq,
        highpass_hz=args.highpass_hz,
        lowpass_hz=args.lowpass_hz,
    )
    timeline = extract_full_record_feature_timeline(
        segment,
        window_s=args.window_s,
        step_s=args.step_s,
        rolling_baseline_s=args.rolling_baseline_s,
        future_s=args.future_s,
        min_history_s=args.min_history_s,
    )
    candidates = select_transition_candidates(
        timeline,
        segment,
        candidates_per_hour=args.candidates_per_hour,
        min_candidates_per_file=args.min_candidates,
        max_candidates_per_file=args.max_candidates,
        min_candidate_gap_s=args.min_gap_s,
        min_confidence=args.min_confidence,
    )
    observations = []
    half_window = args.window_s / 2.0
    for index, candidate in enumerate(candidates, start=1):
        anchor = float(candidate["coarse_candidate_s"])
        support_start = max(0.0, anchor - half_window)
        support_stop = min(duration, anchor + half_window + args.future_s)
        observations.append(
            {
                "start_offset_seconds": support_start,
                "stop_offset_seconds": support_stop,
                "anchor_offset_seconds": anchor,
                "score": float(candidate["coarse_confidence"]),
                "decision_available_offset_seconds": support_stop,
                "support_window_ids": [f"SCANWIN-{index:04d}"],
            }
        )
    policy = {
        "analysis_sfreq": args.analysis_sfreq,
        "highpass_hz": args.highpass_hz,
        "lowpass_hz": args.lowpass_hz,
        "window_s": args.window_s,
        "step_s": args.step_s,
        "rolling_baseline_s": args.rolling_baseline_s,
        "future_s": args.future_s,
        "min_history_s": args.min_history_s,
        "candidates_per_hour": args.candidates_per_hour,
        "min_candidates": args.min_candidates,
        "max_candidates": args.max_candidates,
        "min_gap_s": args.min_gap_s,
        "min_confidence": args.min_confidence,
        "selection_function_can_force_minimum_or_fallback": True,
    }
    scanner_path = ROOT / "code/auto_annotate/full_record_soz_scan.py"
    detector_receipt = {
        "detector_id": "transition_review_preselector_v1",
        "detector_role": "heuristic_preselector",
        "weights_sha256": _canonical_sha256(DEFAULT_WEIGHTS),
        "code_sha256": _sha256_file(scanner_path),
        "policy_sha256": _canonical_sha256(policy),
        "operating_point": {
            "operating_point_id": "engineering_min_confidence_v1",
            "threshold": args.min_confidence,
            "score_direction": "greater_or_equal",
            "selection_source": "engineering_heuristic_frozen",
            "frozen_before_recording": True,
        },
        "promotion_status": "not_evaluated_for_deployment",
        "promotion_receipt_sha256": None,
        "annotations_used": False,
        "labels_used": False,
    }
    manifest = build_long_term_detection_manifest(
        recording_id=args.recording_id,
        patient_pseudonym=args.patient_pseudonym,
        source_signal_sha256=actual_sha,
        recording_duration_seconds=duration,
        detector_receipt=detector_receipt,
        raw_alarm_observations=observations,
        merge_gap_seconds=args.min_gap_s,
        max_selected_candidates=args.max_candidates,
    )
    _atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recording_id": manifest["recording_id"],
                "recording_duration_seconds": manifest[
                    "recording_duration_seconds"
                ],
                "review_candidates": len(manifest["merge_candidates"]),
                "selected_for_event_analysis": sum(
                    item["decision"] == "selected_for_event_analysis"
                    for item in manifest["merge_candidates"]
                ),
                "detector_role": "heuristic_preselector",
                "candidate_is_confirmed_seizure": False,
                "raw_path_released": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
