#!/usr/bin/env python3
"""Summarize SOZ prediction CSVs from single runs or LOPO directories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_set(value: object) -> set[str]:
    return {item for item in str(value or "").replace(",", ";").split(";") if item}


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_prediction_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    n_region = 0
    n_region_hit = 0
    n_hemi = 0
    n_hemi_hit = 0
    n_channel = 0
    channel_hit_sum = 0.0
    by_patient: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in rows:
        patient = row.get("base_patient_id") or row.get("patient_id") or ""
        true_regions = split_set(row.get("true_regions"))
        pred_region = str(row.get("pred_top_region", ""))
        if true_regions:
            hit = float(pred_region in true_regions)
            n_region += 1
            n_region_hit += int(hit)
            by_patient[patient]["region_n"] += 1
            by_patient[patient]["region_hit"] += hit
        hemi_true = str(row.get("hemisphere_true", ""))
        hemi_pred = str(row.get("hemisphere_pred", ""))
        if hemi_true:
            hit = float(hemi_true == hemi_pred)
            n_hemi += 1
            n_hemi_hit += int(hit)
            by_patient[patient]["hemi_n"] += 1
            by_patient[patient]["hemi_hit"] += hit
        true_channels = split_set(row.get("true_channels"))
        pred_channels = split_set(row.get("pred_top_channels"))
        if true_channels:
            hit_rate = len(true_channels & pred_channels) / float(len(true_channels))
            n_channel += 1
            channel_hit_sum += hit_rate
            by_patient[patient]["channel_n"] += 1
            by_patient[patient]["channel_hit_sum"] += hit_rate

    patient_rows: List[Dict[str, object]] = []
    for patient, values in sorted(by_patient.items()):
        patient_rows.append({
            "patient": patient,
            "region_top1_hit": values["region_hit"] / max(values["region_n"], 1.0),
            "hemisphere_accuracy": values["hemi_hit"] / max(values["hemi_n"], 1.0),
            "channel_topk_hit": values["channel_hit_sum"] / max(values["channel_n"], 1.0),
            "n_region_samples": int(values["region_n"]),
            "n_channel_samples": int(values["channel_n"]),
        })
    macro_region = sum(float(row["region_top1_hit"]) for row in patient_rows) / max(len(patient_rows), 1)
    macro_hemi = sum(float(row["hemisphere_accuracy"]) for row in patient_rows) / max(len(patient_rows), 1)
    macro_channel = sum(float(row["channel_topk_hit"]) for row in patient_rows) / max(len(patient_rows), 1)
    return {
        "n_samples": len(rows),
        "sample_region_top1_hit": n_region_hit / max(n_region, 1),
        "sample_hemisphere_accuracy": n_hemi_hit / max(n_hemi, 1),
        "sample_channel_topk_hit": channel_hit_sum / max(n_channel, 1),
        "patient_macro_region_top1_hit": macro_region,
        "patient_macro_hemisphere_accuracy": macro_hemi,
        "patient_macro_channel_topk_hit": macro_channel,
        "n_patients": len(patient_rows),
        "patients": patient_rows,
    }


def find_prediction_files(run_dir: Path) -> List[Path]:
    if (run_dir / "val_predictions.csv").is_file():
        return [run_dir / "val_predictions.csv"]
    return sorted(run_dir.glob("*/val_predictions.csv"))


def write_patient_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report SOZ prediction metrics")
    parser.add_argument("--run_dir", default="outputs/soz_pre/private_lopo")
    parser.add_argument("--output", default="", help="Default: <run_dir>/soz_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    pred_files = find_prediction_files(run_dir)
    rows: List[Dict[str, str]] = []
    file_summaries = []
    for path in pred_files:
        file_rows = read_csv(path)
        summary = summarize_prediction_rows(file_rows)
        file_summaries.append({"path": str(path), **{k: v for k, v in summary.items() if k != "patients"}})
        rows.extend(file_rows)
    summary = summarize_prediction_rows(rows)
    summary["prediction_files"] = file_summaries
    output = Path(args.output) if args.output else run_dir / "soz_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_patient_csv(output.with_suffix(".patients.csv"), summary.get("patients", []))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

