#!/usr/bin/env python3
"""Plan, train, or predict with the non-promotable common17 ST16 challenger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.st16_common17_exploratory_runner_v1 import (  # noqa: E402
    build_exploratory_epoch_plan,
    predict_source_dev_dense,
    train_exploratory_st16,
)


DEFAULT_MANIFEST = (
    ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json"
)
DEFAULT_PROJECTION = (
    ROOT
    / "outputs/tusz_complete_detector_roster_v2_20260823/analysis_projection.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory/non-promotable ST16 common17 runner. Source-eval is "
            "intentionally unavailable."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="build a target-only training plan")
    plan.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    plan.add_argument("--epoch-index", type=int, default=0)
    plan.add_argument("--batch-size", type=int, required=True)
    plan.add_argument(
        "--partial-batch-policy",
        required=True,
        choices=("fail", "emit_explicit"),
    )
    plan.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train", help="run scratch source-train ST16")
    train.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, required=True)
    train.add_argument("--batch-size", type=int, required=True)
    train.add_argument(
        "--partial-batch-policy",
        required=True,
        choices=("fail", "emit_explicit"),
    )
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=2e-5)
    train.add_argument(
        "--resume",
        action="store_true",
        help="resume exact model/optimizer/RNG/cursor state from output-dir/last.pt",
    )
    train.add_argument("--checkpoint-every-batches", type=int, default=25)
    train.add_argument(
        "--maximum-steps",
        type=int,
        default=None,
        help="smoke-only early stop; no partial-epoch checkpoint is emitted",
    )

    predict = commands.add_parser(
        "predict-source-dev",
        help="save complete pre-threshold weighted-OLA source-dev posteriors",
    )
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument(
        "--analysis-projection", type=Path, default=DEFAULT_PROJECTION
    )
    predict.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--inference-batch-size", type=int, required=True)
    predict.add_argument(
        "--maximum-records",
        type=int,
        default=None,
        help="smoke-only incomplete roster; omitted for comparable source-dev",
    )
    predict.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help=(
            "CPU full-record transform workers; 1 preserves the legacy serial "
            "path, 2-4 enables bounded spawn prefetch with parent-only GPU inference"
        ),
    )
    predict.add_argument(
        "--preprocess-prefetch",
        type=int,
        default=1,
        help="hard bound on staged transform futures (parallel: workers..2*workers)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        result = build_exploratory_epoch_plan(
            args.manifest,
            epoch_index=args.epoch_index,
            batch_size=args.batch_size,
            partial_batch_policy=args.partial_batch_policy,
        )
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "train":
        result = train_exploratory_st16(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            partial_batch_policy=args.partial_batch_policy,
            device_name=args.device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            maximum_steps=args.maximum_steps,
            resume=args.resume,
            checkpoint_every_batches=args.checkpoint_every_batches,
        )
    elif args.command == "predict-source-dev":
        result = predict_source_dev_dense(
            checkpoint_path=args.checkpoint,
            analysis_projection_path=args.analysis_projection,
            tusz_root=args.tusz_root,
            output_dir=args.output_dir,
            device_name=args.device,
            inference_batch_size=args.inference_batch_size,
            maximum_records=args.maximum_records,
            preprocess_workers=args.preprocess_workers,
            preprocess_prefetch=args.preprocess_prefetch,
        )
    else:  # pragma: no cover - argparse owns the command domain
        raise AssertionError(args.command)
    print(
        json.dumps(
            {
                "command": args.command,
                "claim_status": result.get("claim_status"),
                "receipt_sha256": result["receipt_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
