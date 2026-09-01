#!/usr/bin/env python3
"""CLI for the additive common-17 TUSZ EventNet experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.eventnet_common17_streaming_v1 import (
    evaluate_common17_source_dev,
    load_common17_manifest,
    materialize_common17_manifest,
    streaming_full_parity,
    train_common17,
)


DEFAULT_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_FOLD = ROOT / "outputs/tusz_canonical_physical_signal_audit_v1_full_20260824r2/detector_cleanroom_fold_plan.json"
DEFAULT_AUDIT = ROOT / "outputs/tusz_canonical_physical_signal_audit_v1_full_20260824r2/audit.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-manifest")
    build.add_argument("--fold-plan", type=Path, default=DEFAULT_FOLD)
    build.add_argument("--canonical-audit", type=Path, default=DEFAULT_AUDIT)
    build.add_argument("--tusz-root", type=Path, default=DEFAULT_ROOT)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--maximum-records-per-split", type=int)

    train = sub.add_parser("train")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=2e-5)
    train.add_argument("--seed", type=int, default=20260824)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--checkpoint-every-steps", type=int, default=50)
    train.add_argument("--maximum-steps", type=int)
    train.add_argument("--allow-smoke-manifest", action="store_true")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--inference-batch-size", type=int, default=8)
    evaluate.add_argument(
        "--thresholds",
        default=(
            "0.001,0.002,0.005,0.01,0.02,0.03,0.05,0.07,"
            "0.09,0.10,0.15,0.20,0.30,0.44,0.60,0.80"
        ),
    )
    evaluate.add_argument("--maximum-records", type=int)
    evaluate.add_argument("--reverse-record-order", action="store_true")
    evaluate.add_argument("--allow-smoke-manifest", action="store_true")

    parity = sub.add_parser("parity")
    parity.add_argument("--manifest", type=Path, required=True)
    parity.add_argument("--record-index", type=int, default=0)
    parity.add_argument("--target-start-sample", type=int, default=128)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build-manifest":
        result = materialize_common17_manifest(
            fold_plan_path=args.fold_plan,
            canonical_audit_path=args.canonical_audit,
            tusz_root=args.tusz_root,
            output_path=args.output,
            maximum_records_per_split=args.maximum_records_per_split,
        )
    elif args.command == "train":
        result = train_common17(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            resume=args.resume,
            checkpoint_every_steps=args.checkpoint_every_steps,
            maximum_steps=args.maximum_steps,
            require_complete_manifest=not args.allow_smoke_manifest,
        )
    elif args.command == "evaluate":
        thresholds = [float(value) for value in args.thresholds.split(",")]
        result = evaluate_common17_source_dev(
            manifest_path=args.manifest,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            inference_batch_size=args.inference_batch_size,
            thresholds=thresholds,
            maximum_records=args.maximum_records,
            reverse_record_order=args.reverse_record_order,
            require_complete_manifest=not args.allow_smoke_manifest,
        )
    else:
        manifest = load_common17_manifest(args.manifest, require_complete=False)
        record = manifest["records"][args.record_index]
        path = Path(manifest["source_bindings"]["tusz_root"]) / record["edf_relative_path"]
        result = streaming_full_parity(
            path, target_start_sample=args.target_start_sample
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
