#!/usr/bin/env python3
"""Benchmark frozen continuous-detector alarms on a patient-level TUSZ split.

The script never runs a detector and never exposes reference events to model
code.  It joins already materialized prediction intervals to either:

1. a complete JSONL benchmark row file; or
2. the local DeepSOZ/TUSZ record manifest (or a column-compatible CSV).

Every selected recording, including zero-alarm recordings, must have exactly
one prediction row.  The output is metrics-only and can never itself promote a
provider to production.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_detection_benchmark import (  # noqa: E402
    evaluate_patient_level_continuous_detection,
)


DEFAULT_REFERENCE_CSV = ROOT / "deepsoz_tusz_652_record_manifest.csv"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if type(payload) is not dict:
                raise TypeError(f"JSONL row {line_number} must be an object")
            rows.append(payload)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _number_list(value: object, *, context: str) -> list[float]:
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} must be a JSON number array") from error
    if not isinstance(parsed, list):
        raise TypeError(f"{context} must be a JSON number array")
    result: list[float] = []
    for item in parsed:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{context} contains a non-number")
        result.append(float(item))
    return result


def _normalize_prediction_events(row: Mapping[str, Any]) -> list[dict[str, float]]:
    raw_events = row.get("predicted_events", row.get("event_proposals"))
    if not isinstance(raw_events, list):
        raise TypeError("prediction row lacks predicted_events/event_proposals")
    events: list[dict[str, float]] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise TypeError(f"prediction event {index} must be an object")
        if "start_seconds" in raw and "stop_seconds" in raw:
            start = raw["start_seconds"]
            stop = raw["stop_seconds"]
        elif "start_offset_seconds" in raw and "stop_offset_seconds" in raw:
            start = raw["start_offset_seconds"]
            stop = raw["stop_offset_seconds"]
        else:
            raise ValueError(f"prediction event {index} lacks interval fields")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise TypeError("prediction start must be numeric")
        if isinstance(stop, bool) or not isinstance(stop, (int, float)):
            raise TypeError("prediction stop must be numeric")
        events.append({"start_seconds": float(start), "stop_seconds": float(stop)})
    events.sort(key=lambda event: (event["start_seconds"], event["stop_seconds"]))
    return events


def _prediction_lookup(path: Path) -> dict[str, list[dict[str, float]]]:
    output: dict[str, list[dict[str, float]]] = {}
    for row in _read_jsonl(path):
        recording_id = row.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise TypeError("prediction recording_id must be a non-empty string")
        if recording_id in output:
            raise ValueError(f"duplicate prediction recording_id: {recording_id}")
        output[recording_id] = _normalize_prediction_events(row)
    return output


def _rows_from_tusz_csv(
    *,
    reference_csv: Path,
    predictions_jsonl: Path,
    selected_split: str,
    patient_field: str,
    recording_field: str,
    duration_field: str,
    starts_field: str,
    stops_field: str,
    split_field: str,
) -> list[dict[str, Any]]:
    predictions = _prediction_lookup(predictions_jsonl)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    patient_splits: dict[str, set[str]] = {}
    with reference_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            patient_field,
            recording_field,
            duration_field,
            starts_field,
            stops_field,
            split_field,
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"reference CSV lacks fields: {missing}")
        for line_number, raw in enumerate(reader, start=2):
            patient_id = str(raw[patient_field]).strip()
            recording_id = str(raw[recording_field]).strip()
            split = str(raw[split_field]).strip()
            if not patient_id or not recording_id or not split:
                raise ValueError(
                    f"blank identity/split at reference line {line_number}"
                )
            patient_splits.setdefault(patient_id, set()).add(split)
            if split != selected_split:
                continue
            if recording_id in selected_ids:
                raise ValueError(
                    f"duplicate selected reference recording: {recording_id}"
                )
            selected_ids.add(recording_id)
            try:
                duration = float(raw[duration_field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid duration at reference line {line_number}"
                ) from error
            starts = _number_list(
                raw[starts_field], context=f"reference line {line_number} starts"
            )
            stops = _number_list(
                raw[stops_field], context=f"reference line {line_number} stops"
            )
            if len(starts) != len(stops):
                raise ValueError(
                    f"reference start/stop count differs at line {line_number}"
                )
            if recording_id not in predictions:
                raise ValueError(f"missing prediction row for {recording_id}")
            selected.append(
                {
                    "patient_id": patient_id,
                    "recording_id": recording_id,
                    "split": split,
                    "duration_seconds": duration,
                    "reference_events": [
                        {"start_seconds": start, "stop_seconds": stop}
                        for start, stop in zip(starts, stops)
                    ],
                    "predicted_events": predictions[recording_id],
                }
            )
    if not selected:
        raise ValueError(f"reference split has no rows: {selected_split}")
    cross_split = sorted(
        patient for patient, splits in patient_splits.items() if len(splits) > 1
    )
    if cross_split:
        raise ValueError(
            "reference CSV violates patient-level split isolation; "
            f"{len(cross_split)} patients cross splits"
        )
    extra_predictions = sorted(set(predictions).difference(selected_ids))
    if extra_predictions:
        raise ValueError(
            "prediction file contains records outside the selected split; "
            f"first extra={extra_predictions[0]!r}"
        )
    return selected


def _roster(path: Path | None, *, context: str) -> Iterable[str] | None:
    if path is None:
        return None
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"{context} roster is empty")
    return values


def _write_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is None:
        sys.stdout.write(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-level continuous seizure-detector benchmark"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-jsonl",
        type=Path,
        help="Complete rows with patient/recording/split/reference/predicted events",
    )
    source.add_argument(
        "--predictions-jsonl",
        type=Path,
        help="Predictions to join to --tusz-reference-csv",
    )
    parser.add_argument(
        "--tusz-reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV
    )
    parser.add_argument("--split", default="source_eval")
    parser.add_argument("--patient-field", default="local_patient_id")
    parser.add_argument("--recording-field", default="local_edf_path")
    parser.add_argument("--duration-field", default="duration_sec")
    parser.add_argument("--starts-field", default="seizure_starts_sec")
    parser.add_argument("--stops-field", default="seizure_ends_sec")
    parser.add_argument("--split-field", default="model_split")
    parser.add_argument("--provider-id", required=True)
    parser.add_argument("--operating-point-id", required=True)
    parser.add_argument(
        "--operating-point-frozen-before-evaluation", action="store_true"
    )
    parser.add_argument(
        "--development-patient-roster",
        type=Path,
        help="One development patient ID per line; used only for overlap audit",
    )
    parser.add_argument(
        "--expected-evaluation-recording-roster",
        type=Path,
        help=(
            "One frozen evaluation recording ID per line; required to prove that "
            "different detector receipts scored the identical complete inventory"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.input_jsonl is not None:
        rows = _read_jsonl(args.input_jsonl)
    else:
        rows = _rows_from_tusz_csv(
            reference_csv=args.tusz_reference_csv,
            predictions_jsonl=args.predictions_jsonl,
            selected_split=args.split,
            patient_field=args.patient_field,
            recording_field=args.recording_field,
            duration_field=args.duration_field,
            starts_field=args.starts_field,
            stops_field=args.stops_field,
            split_field=args.split_field,
        )
    result = evaluate_patient_level_continuous_detection(
        provider_id=args.provider_id,
        operating_point_id=args.operating_point_id,
        rows=rows,
        operating_point_frozen_before_evaluation=(
            args.operating_point_frozen_before_evaluation
        ),
        development_patient_ids=_roster(
            args.development_patient_roster,
            context="development patient",
        ),
        expected_evaluation_recording_ids=_roster(
            args.expected_evaluation_recording_roster,
            context="expected evaluation recording",
        ),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_json(args.output, result)


if __name__ == "__main__":
    main()
