#!/usr/bin/env python3
"""Provider-neutral prediction-first CLI for Alarm and Navigation OP scoring.

``freeze`` never opens a reference file.  ``score`` first validates the frozen
inventory receipt and only then opens the reference rows and performs the
exact one-to-one joins/scoring implemented by ``detector_dual_operating_point_v1``.
The command performs no detector inference and never overwrites an output.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.detector_dual_operating_point_v1 import (  # noqa: E402
    freeze_detector_prediction_inventory_v1,
    score_detector_dual_op_v1,
    validate_detector_prediction_inventory_v1,
)


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_json(path: Path, context: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_object,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{context} contains non-finite token {token}")
        ),
    )


def _load_array(path: Path, context: str) -> list[Any]:
    value = _load_json(path, context)
    if not isinstance(value, list):
        raise TypeError(f"{context} must contain a JSON array")
    return value


def _write_new_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    prediction_rows = _load_array(args.prediction_rows, "prediction rows")
    expected_recording_ids = _load_array(
        args.expected_recording_ids, "expected recording IDs"
    )
    expected_policy_ids = _load_array(args.expected_policy_ids, "expected policy IDs")
    inventory = freeze_detector_prediction_inventory_v1(
        provider_id=args.provider_id,
        rows=prediction_rows,
        expected_recording_ids=expected_recording_ids,
        expected_policy_ids=expected_policy_ids,
    )
    _write_new_json(args.output, inventory)
    return inventory


def _score(args: argparse.Namespace) -> dict[str, Any]:
    # Fail before the reference path is opened if the prediction receipt is not
    # authentic and complete.
    frozen = validate_detector_prediction_inventory_v1(
        _load_json(args.frozen_inventory, "frozen prediction inventory")
    )
    references = _load_array(args.reference_rows, "reference rows")
    reference_by_recording: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(references):
        if type(raw) is not dict:
            raise TypeError(f"reference row {index} must be an object")
        recording_id = raw.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id:
            raise TypeError(f"reference row {index} recording_id is invalid")
        if recording_id in reference_by_recording:
            raise ValueError("reference recording IDs must be unique")
        reference_by_recording[recording_id] = deepcopy(raw)
    expected = set(frozen["expected_recording_ids"])
    if set(reference_by_recording) != expected:
        missing = sorted(expected - set(reference_by_recording))
        extra = sorted(set(reference_by_recording) - expected)
        raise ValueError(
            "reference recording inventory mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    joined_rows = [
        {
            "prediction_row": deepcopy(row),
            "reference_row": deepcopy(reference_by_recording[row["recording_id"]]),
        }
        for row in frozen["prediction_rows"]
    ]
    diagnostic = score_detector_dual_op_v1(
        frozen_prediction_inventory=frozen,
        joined_rows=joined_rows,
    )
    _write_new_json(args.output, diagnostic)
    return diagnostic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze reference-free full-record predictions, then score the "
            "separate Alarm and Navigation operating-point diagnostics."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--provider-id", required=True)
    freeze.add_argument("--prediction-rows", type=Path, required=True)
    freeze.add_argument("--expected-recording-ids", type=Path, required=True)
    freeze.add_argument("--expected-policy-ids", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=_freeze)

    score = subparsers.add_parser("score")
    score.add_argument("--frozen-inventory", type=Path, required=True)
    score.add_argument("--reference-rows", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = args.handler(args)
    summary = {
        "schema_version": result["schema_version"],
        "provider_id": result["provider_id"],
        "receipt_sha256": result.get("receipt_sha256"),
        "output": str(args.output),
        "reference_accessed": args.command == "score",
        "model_inference_performed": False,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
