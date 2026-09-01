#!/usr/bin/env python3
"""Build, predict, and score the research-only external ST common17 adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.clinical_eeg_long_recording.external_seizuretransformer_common17_v1 import (
    DEFAULT_THRESHOLDS,
    audit_prediction_inventory,
    build_target_free_source_dev_projection,
    predict_source_dev,
    score_source_dev_predictions,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    projection = subparsers.add_parser("build-projection", allow_abbrev=False)
    projection.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json",
    )
    projection.add_argument("--output", type=Path, required=True)

    predict = subparsers.add_parser("predict", allow_abbrev=False)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--projection", type=Path, required=True)
    predict.add_argument("--tusz-root", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--batch-size", type=int, default=8)
    predict.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="float32"
    )
    predict.add_argument("--maximum-records", type=int)
    predict.add_argument(
        "--retry-failures",
        action="store_true",
        help="Atomically retry only compatible typed-failure rows; completed rows still require an exact run contract.",
    )
    predict.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )

    score = subparsers.add_parser("score", allow_abbrev=False)
    score.add_argument("--prediction-manifest", type=Path, required=True)
    score.add_argument(
        "--reference-manifest",
        type=Path,
        default=ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json",
    )
    score.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit", allow_abbrev=False)
    audit.add_argument("--prediction-manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--observed-device-name")
    audit.add_argument(
        "--fresh-single-configuration-run-asserted", action="store_true"
    )

    args = parser.parse_args()
    if args.command == "build-projection":
        result = build_target_free_source_dev_projection(args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "predict":
        result = predict_source_dev(
            checkpoint_path=args.checkpoint,
            projection_path=args.projection,
            tusz_root=args.tusz_root,
            output_dir=args.output_dir,
            device_name=args.device,
            batch_size=args.batch_size,
            precision=args.precision,
            thresholds=args.thresholds,
            maximum_records=args.maximum_records,
            retry_failures=args.retry_failures,
        )
    elif args.command == "score":
        result = score_source_dev_predictions(
            prediction_manifest_path=args.prediction_manifest,
            reference_manifest_path=args.reference_manifest,
            project_root=ROOT,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        result = audit_prediction_inventory(
            prediction_manifest_path=args.prediction_manifest,
            observed_device_name=args.observed_device_name,
            fresh_single_configuration_run_asserted=(
                args.fresh_single_configuration_run_asserted
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "command": args.command,
                "schema_version": result["schema_version"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
