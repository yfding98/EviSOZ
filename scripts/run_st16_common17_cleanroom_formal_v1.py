#!/usr/bin/env python3
"""Execute target-isolated ST16 formal-entry dry-runs (never GPU training)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.st16_common17_cleanroom_formal_entry_v1 import (  # noqa: E402
    audit_materialized_dry_runs,
    build_dev_prediction_dry_run,
    build_training_dry_run,
)


DEFAULT_CONFIG = ROOT / "configs/clinical_eeg_st16_common17_cleanroom_formal_entry_v1.json"
DEFAULT_TRAIN_MANIFEST = (
    ROOT
    / "outputs/clinical_eeg_detector_cleanroom_physical_isolation_v1_20260825"
    / "source_train_labeled_manifest.json"
)
DEFAULT_DEV_ROSTER = (
    ROOT
    / "outputs/clinical_eeg_detector_cleanroom_physical_isolation_v1_20260825"
    / "source_dev_eeg_only_prediction_roster.json"
)


def _write_json_atomic(path: Path, value: object) -> None:
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ST16 common17/LB16 clean-room formal-entry dry-run. This program "
            "does not start CUDA or training."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "training-dry-run",
        help="open only the labelled source-train manifest and audit all 4664 rows",
    )
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    train.add_argument("--output-dir", type=Path, required=True)

    dev = subparsers.add_parser(
        "dev-prediction-dry-run",
        help="open only the EEG-only source-dev roster and build its denominator",
    )
    dev.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    dev.add_argument("--dev-roster", type=Path, default=DEFAULT_DEV_ROSTER)
    dev.add_argument("--output-dir", type=Path, required=True)

    audit = subparsers.add_parser(
        "audit",
        help="replay already-materialized dry-run receipts without loading datasets",
    )
    audit.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    audit.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.resolve(strict=False)
    if args.command == "training-dry-run":
        receipt, ledger = build_training_dry_run(
            config_path=args.config,
            train_manifest_path=args.train_manifest,
        )
        _write_json_atomic(output / "training_record_admission_ledger.json", ledger)
        _write_json_atomic(output / "training_dry_run_receipt.json", receipt)
        result = {
            "command": args.command,
            "status": receipt["status"],
            "source_train_recording_count": receipt["source_train_recording_count"],
            "short_context_arm_record_count": receipt[
                "short_context_arm_record_count"
            ],
            "content_sha256": receipt["content_sha256"],
        }
    elif args.command == "dev-prediction-dry-run":
        receipt = build_dev_prediction_dry_run(
            config_path=args.config,
            dev_roster_path=args.dev_roster,
        )
        _write_json_atomic(output / "dev_prediction_dry_run_receipt.json", receipt)
        result = {
            "command": args.command,
            "status": receipt["status"],
            "expected_recording_count": receipt["expected_recording_count"],
            "target_bearing_field_or_value_count": receipt[
                "target_bearing_field_or_value_count"
            ],
            "content_sha256": receipt["content_sha256"],
        }
    elif args.command == "audit":
        receipt = audit_materialized_dry_runs(
            config_path=args.config,
            output_dir=output,
        )
        _write_json_atomic(output / "receipt.json", receipt)
        result = {
            "command": args.command,
            "status": receipt["status"],
            "formal_launch_gate": receipt["formal_launch_gate"]["status"],
            "content_sha256": receipt["content_sha256"],
        }
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
