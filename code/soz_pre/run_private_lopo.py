#!/usr/bin/env python3
"""Run patient-level leave-one-patient-out evaluation on private samples."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from soz_pre.dataset import UnifiedSOZDataset, list_private_patients  # noqa: E402
from soz_pre.train_region_soz import set_seed, train_once  # noqa: E402


DEFAULT_PREPROCESSED = "outputs/soz_pre/preprocessed"
DEFAULT_OUTPUT = "outputs/soz_pre/private_lopo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run private LOPO for SOZ pretraining model")
    parser.add_argument("--preprocessed_dir", default=DEFAULT_PREPROCESSED)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--patients", default="", help="Optional comma/semicolon-separated patient list")
    parser.add_argument("--max_patients", type=int, default=0)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--model", choices=["deepsoz", "eegnet"], default="deepsoz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--lstm_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--attention_temperature", type=float, default=1.0)
    parser.add_argument("--eegnet_temporal_filters", type=int, default=16)
    parser.add_argument("--eegnet_depth_multiplier", type=int, default=2)
    parser.add_argument("--eegnet_pointwise_filters", type=int, default=32)
    parser.add_argument("--eegnet_kernel_length", type=int, default=64)
    parser.add_argument("--eegnet_separable_kernel_length", type=int, default=16)
    parser.add_argument("--eegnet_pool1", type=int, default=4)
    parser.add_argument("--eegnet_pool2", type=int, default=8)
    parser.add_argument("--channel_loss_weight", type=float, default=1.0)
    parser.add_argument("--region_loss_weight", type=float, default=1.5)
    parser.add_argument("--propagation_loss_weight", type=float, default=0.5)
    parser.add_argument("--seizure_loss_weight", type=float, default=0.5)
    parser.add_argument("--hemisphere_loss_weight", type=float, default=0.7)
    parser.add_argument("--channel_ranking_loss_weight", type=float, default=0.2)
    parser.add_argument("--region_ranking_loss_weight", type=float, default=0.1)
    parser.add_argument("--channel_region_loss_weight", type=float, default=0.3)
    parser.add_argument("--ranking_margin", type=float, default=0.0)
    parser.add_argument("--region_pool_blend", type=float, default=0.5)
    parser.add_argument("--region_pos_weight_mode", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--channel_pos_weight_mode", choices=["none", "balanced"], default="none")
    parser.add_argument("--max_pos_weight", type=float, default=5.0)
    parser.add_argument("--tusz_spatial_weight_scale", type=float, default=0.25)
    parser.add_argument("--private_spatial_weight_scale", type=float, default=1.0)
    parser.add_argument("--other_spatial_weight_scale", type=float, default=1.0)
    parser.add_argument("--source_balance", choices=["none", "source"], default="source")
    parser.add_argument("--sampler_weight_cap", type=float, default=10.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patient_region_threshold", type=float, default=0.5)
    parser.add_argument("--patient_channel_threshold", type=float, default=0.5)
    parser.add_argument("--patient_min_regions", type=int, default=1)
    parser.add_argument("--patient_max_regions", type=int, default=3)
    parser.add_argument("--patient_min_channels", type=int, default=1)
    parser.add_argument("--patient_max_channels", type=int, default=8)
    parser.add_argument("--selection_metric", default="region_macro_f1")
    parser.add_argument("--device", default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_patient_list(value: str) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patients = parse_patient_list(args.patients) or list_private_patients(args.preprocessed_dir)
    if int(args.max_patients) > 0:
        patients = patients[: int(args.max_patients)]
    args.train_splits = "private"
    args.val_splits = "private"
    args.train_sources = "private"
    args.val_sources = "private"
    args.exclude_patients = "<held_out_patient_per_fold>"
    args.val_patients = "<held_out_patient_per_fold>"
    summary_rows: List[Dict[str, object]] = []
    for patient in patients:
        fold_dir = output_dir / patient
        try:
            train_ds = UnifiedSOZDataset(
                args.preprocessed_dir,
                splits=["private"],
                sources=["private"],
                exclude_patients=[patient],
            )
            val_ds = UnifiedSOZDataset(
                args.preprocessed_dir,
                splits=["private"],
                sources=["private"],
                include_patients=[patient],
            )
            fold_args = copy.copy(args)
            fold_args.train_splits = "private"
            fold_args.val_splits = "private"
            fold_args.train_sources = "private"
            fold_args.val_sources = "private"
            fold_args.exclude_patients = patient
            fold_args.val_patients = patient
            result = train_once(train_ds, val_ds, fold_dir, fold_args)
            row = {"patient": patient, **result["metrics"]}
            summary_rows.append(row)
        except Exception as exc:
            summary_rows.append({"patient": patient, "error": str(exc)})
            print(json.dumps({"patient": patient, "error": str(exc)}, ensure_ascii=False))
    numeric: Dict[str, List[float]] = {}
    for row in summary_rows:
        for key, value in row.items():
            if key in {"patient", "error"}:
                continue
            try:
                numeric.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                pass
    aggregate = {
        key: {
            "mean": float(sum(values) / max(len(values), 1)),
            "n": len(values),
        }
        for key, values in numeric.items()
    }
    summary = {
        "patients": patients,
        "folds": summary_rows,
        "aggregate": aggregate,
        "config": vars(args),
    }
    (output_dir / "lopo_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
