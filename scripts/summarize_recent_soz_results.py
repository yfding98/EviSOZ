#!/usr/bin/env python3
"""Summarize recent private SOZ experiments for work-report figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


EXPERIMENTS = (
    {
        "name": "Mamba-128s TUSZ-init",
        "short_name": "Mamba-128s",
        "path": "outputs/soz_mamba_private_adapt_128s/private_lopo_encoder",
        "status": "complete",
        "note": "128s event sequence model, TUSZ encoder init, private LOPO.",
    },
    {
        "name": "DeepSOZ private",
        "short_name": "DeepSOZ",
        "path": "outputs/soz_baselines_soft_private/deepsoz/private_lopo",
        "status": "complete",
        "note": "DeepSOZ-style baseline trained on private folds.",
    },
    {
        "name": "DeepSOZ TUSZ-init",
        "short_name": "DeepSOZ+TUSZ",
        "path": "outputs/soz_baselines_soft_private/deepsoz/private_lopo_from_tusz",
        "status": "complete",
        "note": "DeepSOZ baseline initialized from TUSZ pretraining.",
    },
    {
        "name": "EEGNet private",
        "short_name": "EEGNet",
        "path": "outputs/soz_baselines_soft_private/eegnet/private_lopo",
        "status": "complete",
        "note": "EEGNet baseline trained on private folds.",
    },
    {
        "name": "EEGNet TUSZ-init",
        "short_name": "EEGNet+TUSZ",
        "path": "outputs/soz_baselines_soft_private/eegnet/private_lopo_from_tusz",
        "status": "complete",
        "note": "EEGNet baseline initialized from TUSZ pretraining.",
    },
)

DATASET_RUNS = (
    {
        "dataset": "Private",
        "name": "Mamba-128s TUSZ-init",
        "short_name": "Mamba-128s",
        "path": "outputs/soz_mamba_private_adapt_128s/private_lopo_encoder",
        "mode": "lopo",
        "status": "complete",
    },
    {
        "dataset": "Private",
        "name": "DeepSOZ private",
        "short_name": "DeepSOZ",
        "path": "outputs/soz_baselines_soft_private/deepsoz/private_lopo",
        "mode": "lopo",
        "status": "complete",
    },
    {
        "dataset": "Private",
        "name": "DeepSOZ TUSZ-init",
        "short_name": "DeepSOZ+TUSZ",
        "path": "outputs/soz_baselines_soft_private/deepsoz/private_lopo_from_tusz",
        "mode": "lopo",
        "status": "complete",
    },
    {
        "dataset": "Private",
        "name": "EEGNet private",
        "short_name": "EEGNet",
        "path": "outputs/soz_baselines_soft_private/eegnet/private_lopo",
        "mode": "lopo",
        "status": "complete",
    },
    {
        "dataset": "Private",
        "name": "EEGNet TUSZ-init",
        "short_name": "EEGNet+TUSZ",
        "path": "outputs/soz_baselines_soft_private/eegnet/private_lopo_from_tusz",
        "mode": "lopo",
        "status": "complete",
    },
    {
        "dataset": "TUSZ",
        "name": "Mamba-128s",
        "short_name": "Mamba-128s",
        "path": "outputs/soz_mamba_long_128s/tusz_mamba",
        "mode": "single",
        "status": "complete",
    },
    {
        "dataset": "TUSZ",
        "name": "DeepSOZ",
        "short_name": "DeepSOZ",
        "path": "outputs/soz_baselines_soft_private/deepsoz/tusz",
        "mode": "single",
        "status": "complete",
    },
    {
        "dataset": "TUSZ",
        "name": "EEGNet",
        "short_name": "EEGNet",
        "path": "outputs/soz_baselines_soft_private/eegnet/tusz",
        "mode": "single",
        "status": "complete",
    },
)

METRICS = (
    "patient_region_top1_hit",
    "patient_region_top2_hit",
    "patient_region_threshold_f1",
    "patient_channel_top1_hit",
    "patient_channel_top2_hit",
    "patient_channel_topk_hit",
    "region_macro_f1",
    "channel_macro_f1",
    "seizure_f1",
)

PLOT_METRICS = (
    ("patient_region_top1_hit", "Region top-1"),
    ("patient_region_threshold_f1", "Region threshold F1"),
    ("patient_channel_top1_hit", "Channel top-1"),
    ("patient_channel_topk_hit", "Channel top-k"),
    ("channel_macro_f1", "Channel macro F1"),
    ("seizure_f1", "Seizure F1"),
)

PALETTE = {
    "Mamba-128s": "#3B82F6",
    "DeepSOZ": "#10B981",
    "DeepSOZ+TUSZ": "#F59E0B",
    "EEGNet": "#8B5CF6",
    "EEGNet+TUSZ": "#EF4444",
}

REGION_NAMES = (
    "left_frontal",
    "right_frontal",
    "left_temporal",
    "right_temporal",
    "central_parietal",
)

TCP_CHANNELS = (
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
)

CONFUSION_TASKS = (
    ("patient_region", "Patient region"),
    ("patient_channel", "Patient channel"),
    ("sample_region", "Sample region"),
    ("sample_channel", "Sample channel"),
    ("seizure_window", "Seizure window"),
)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def mean(values: Iterable[Any]) -> float:
    nums = [finite_float(value) for value in values]
    nums = [value for value in nums if math.isfinite(value)]
    return float(np.mean(nums)) if nums else math.nan


def pct(value: Any) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100:.1f}%"


def split_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, float) and math.isnan(value):
        return set()
    return {
        item.strip()
        for item in str(value).replace(",", ";").split(";")
        if item.strip()
    }


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def empty_counts() -> Dict[str, float]:
    return {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}


def add_counts(dst: Dict[str, float], src: Dict[str, float]) -> Dict[str, float]:
    for key in ("tp", "fp", "tn", "fn"):
        dst[key] = float(dst.get(key, 0.0)) + float(src.get(key, 0.0))
    return dst


def confusion_metrics_from_counts(counts: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    tp = max(0.0, finite_float(counts.get("tp")))
    fp = max(0.0, finite_float(counts.get("fp")))
    tn = max(0.0, finite_float(counts.get("tn")))
    fn = max(0.0, finite_float(counts.get("fn")))
    total = tp + fp + tn + fn
    precision = tp / max(tp + fp, 1.0)
    sensitivity = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    npv = tn / max(tn + fn, 1.0)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-8)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    metrics = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support_positive": tp + fn,
        "support_negative": tn + fp,
        "accuracy": (tp + tn) / max(total, 1.0),
        "precision": precision,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }
    if prefix:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
    return metrics


def set_confusion_counts(true_set: set[str], pred_set: set[str], labels: Sequence[str]) -> Dict[str, float]:
    counts = empty_counts()
    for label in labels:
        truth = label in true_set
        pred = label in pred_set
        if truth and pred:
            counts["tp"] += 1.0
        elif (not truth) and pred:
            counts["fp"] += 1.0
        elif truth and not pred:
            counts["fn"] += 1.0
        else:
            counts["tn"] += 1.0
    return counts


def set_localization_metrics(rows: Sequence[Dict[str, str]], true_col: str, pred_col: str) -> Dict[str, float]:
    top1_hits: List[float] = []
    top2_hits: List[float] = []
    topk_recalls: List[float] = []
    exact_matches: List[float] = []
    jaccards: List[float] = []
    for row in rows:
        true_set = split_set(row.get(true_col))
        pred_list = [item.strip() for item in str(row.get(pred_col) or "").replace(",", ";").split(";") if item.strip()]
        pred_set = set(pred_list)
        if not true_set:
            continue
        top1_hits.append(float(bool(pred_list) and pred_list[0] in true_set))
        top2_hits.append(float(bool(set(pred_list[:2]) & true_set)))
        topk_recalls.append(float(len(pred_set & true_set) / max(len(true_set), 1)))
        exact_matches.append(float(pred_set == true_set))
        union = pred_set | true_set
        jaccards.append(float(len(pred_set & true_set) / max(len(union), 1)))
    return {
        "n_patients": float(len(top1_hits)),
        "localization_top1_accuracy": mean(top1_hits),
        "localization_top2_accuracy": mean(top2_hits),
        "localization_topk_recall": mean(topk_recalls),
        "exact_match_accuracy": mean(exact_matches),
        "jaccard": mean(jaccards),
    }


def patient_level_counts(rows: Sequence[Dict[str, str]], true_col: str, pred_col: str, labels: Sequence[str]) -> Dict[str, float]:
    counts = empty_counts()
    for row in rows:
        true_set = split_set(row.get(true_col))
        if not true_set:
            continue
        pred_set = split_set(row.get(pred_col))
        add_counts(counts, set_confusion_counts(true_set, pred_set, labels))
    return counts


def sample_region_counts(rows: Sequence[Dict[str, str]], threshold: float) -> Dict[str, float]:
    counts = empty_counts()
    for row in rows:
        for label in REGION_NAMES:
            y = finite_float(row.get(f"label_region_{label}"))
            prob = finite_float(row.get(f"prob_region_{label}"))
            if not math.isfinite(y) or not math.isfinite(prob):
                continue
            truth = y >= 0.5
            pred = prob >= threshold
            add_counts(counts, set_confusion_counts({label} if truth else set(), {label} if pred else set(), [label]))
    return counts


def sample_channel_counts(rows: Sequence[Dict[str, str]], threshold: float) -> Dict[str, float]:
    counts = empty_counts()
    for row in rows:
        true_channels = split_set(row.get("true_channels"))
        if not true_channels:
            continue
        for channel in TCP_CHANNELS:
            prob = finite_float(row.get(f"prob_channel_{channel.replace('-', '_')}"))
            if not math.isfinite(prob):
                continue
            truth = channel in true_channels
            pred = prob >= threshold
            add_counts(counts, set_confusion_counts({channel} if truth else set(), {channel} if pred else set(), [channel]))
    return counts


def counts_from_saved_binary_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, float]:
    pos = finite_float(metrics.get(f"{prefix}_support_positive"))
    neg = finite_float(metrics.get(f"{prefix}_support_negative"))
    acc = finite_float(metrics.get(f"{prefix}_accuracy"))
    sens = finite_float(metrics.get(f"{prefix}_recall"))
    if not all(math.isfinite(value) for value in (pos, neg, acc, sens)):
        return empty_counts()
    tp = sens * pos
    fn = pos - tp
    tn = acc * (pos + neg) - tp
    fp = neg - tn
    return {
        "tp": max(0.0, tp),
        "fp": max(0.0, fp),
        "tn": max(0.0, tn),
        "fn": max(0.0, fn),
    }


def load_folds(exp: Dict[str, str]) -> List[Dict[str, Any]]:
    root = Path(exp["path"])
    summary = read_json(root / "lopo_summary.json")
    folds = summary.get("folds")
    if isinstance(folds, list) and folds:
        out = []
        for row in folds:
            if isinstance(row, dict):
                item = dict(row)
                item.setdefault("source_file", str(root / str(item.get("patient", "")) / "val_metrics.json"))
                out.append(item)
        return out

    out = []
    for metrics_path in sorted(root.glob("*/val_metrics.json"), key=lambda p: p.stat().st_mtime):
        row = read_json(metrics_path)
        if row:
            row = dict(row)
            row["patient"] = metrics_path.parent.name
            row["source_file"] = str(metrics_path)
            row["metric_mtime"] = metrics_path.stat().st_mtime
            out.append(row)
    return out


def is_valid_spatial(row: Dict[str, Any]) -> bool:
    if "patient_n_patients" in row:
        return finite_float(row.get("patient_n_patients")) > 0
    if "private_patient_n_patients" in row:
        return finite_float(row.get("private_patient_n_patients")) > 0
    return True


def rows_for_experiment(exp: Dict[str, str]) -> List[Dict[str, Any]]:
    rows = []
    for fold in load_folds(exp):
        row = dict(fold)
        row["dataset"] = exp.get("dataset", "Private")
        row["experiment"] = exp["name"]
        row["short_name"] = exp["short_name"]
        row["status"] = exp["status"]
        row["valid_spatial"] = is_valid_spatial(row)
        rows.append(row)
    return rows


def rows_for_run_spec(spec: Dict[str, str]) -> List[Dict[str, Any]]:
    root = Path(spec["path"])
    if spec.get("mode") == "single":
        metrics_path = root / "val_metrics.json"
        row = read_json(metrics_path)
        if not row:
            return []
        row = dict(row)
        row["dataset"] = spec["dataset"]
        row["experiment"] = spec["name"]
        row["short_name"] = spec["short_name"]
        row["status"] = spec["status"]
        row["patient"] = spec["dataset"]
        row["source_file"] = str(metrics_path)
        row["metric_mtime"] = metrics_path.stat().st_mtime if metrics_path.is_file() else math.nan
        row["valid_spatial"] = is_valid_spatial(row)
        return [row]
    return rows_for_experiment(spec)


def summarize_experiments(folds_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp in EXPERIMENTS:
        sub = folds_df[folds_df["short_name"] == exp["short_name"]]
        valid = sub[sub["valid_spatial"]]
        row: Dict[str, Any] = {
            "experiment": exp["name"],
            "short_name": exp["short_name"],
            "status": exp["status"],
            "n_folds": int(len(sub)),
            "valid_spatial_folds": int(len(valid)),
            "note": exp["note"],
        }
        for metric in METRICS:
            row[f"all_{metric}"] = mean(sub.get(metric, []))
            row[f"valid_{metric}"] = mean(valid.get(metric, []))
        rows.append(row)
    return pd.DataFrame(rows)


def common_valid_summary(folds_df: pd.DataFrame) -> pd.DataFrame:
    patient_sets = []
    for exp in EXPERIMENTS:
        sub = folds_df[(folds_df["short_name"] == exp["short_name"]) & (folds_df["valid_spatial"])]
        if len(sub):
            patient_sets.append(set(sub["patient"].dropna().astype(str)))
    common = set.intersection(*patient_sets) if patient_sets else set()

    rows = []
    for exp in EXPERIMENTS:
        sub = folds_df[
            (folds_df["short_name"] == exp["short_name"])
            & (folds_df["valid_spatial"])
            & (folds_df["patient"].astype(str).isin(common))
        ]
        row: Dict[str, Any] = {
            "experiment": exp["name"],
            "short_name": exp["short_name"],
            "status": exp["status"],
            "common_valid_patients": len(common),
            "n_rows": int(len(sub)),
        }
        for metric in METRICS:
            row[metric] = mean(sub.get(metric, []))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_deepsoz_style_fold_metrics(folds_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, fold in folds_df.iterrows():
        source_file = str(fold.get("source_file") or "")
        metrics_path = Path(source_file)
        if not metrics_path.is_file():
            continue
        run_dir = metrics_path.parent
        run_config = read_json(run_dir / "run_config.json")
        threshold = finite_float(run_config.get("threshold"))
        if not math.isfinite(threshold):
            threshold = 0.5

        val_metrics = read_json(metrics_path)
        pred_rows = read_csv_rows(run_dir / "val_predictions.csv")
        patient_rows = read_csv_rows(run_dir / "val_patient_predictions.csv")

        task_counts = {
            "patient_region": patient_level_counts(patient_rows, "true_regions", "pred_regions", REGION_NAMES),
            "patient_channel": patient_level_counts(patient_rows, "true_channels", "pred_channels", TCP_CHANNELS),
            "sample_region": sample_region_counts(pred_rows, threshold),
            "sample_channel": sample_channel_counts(pred_rows, threshold),
            "seizure_window": counts_from_saved_binary_metrics(val_metrics, "seizure"),
        }

        row: Dict[str, Any] = {
            "dataset": fold.get("dataset", "Private"),
            "experiment": fold.get("experiment"),
            "short_name": fold.get("short_name"),
            "status": fold.get("status"),
            "patient": fold.get("patient"),
            "valid_spatial": bool(fold.get("valid_spatial")),
            "threshold": threshold,
            "run_dir": str(run_dir),
        }
        for task, counts in task_counts.items():
            row.update(confusion_metrics_from_counts(counts, prefix=task))

        region_loc = set_localization_metrics(patient_rows, "true_regions", "pred_regions")
        channel_loc = set_localization_metrics(patient_rows, "true_channels", "pred_channels")
        for key, value in region_loc.items():
            row[f"patient_region_{key}"] = value
        for key, value in channel_loc.items():
            row[f"patient_channel_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_deepsoz_style_metrics(
    deepsoz_df: pd.DataFrame,
    specs: Sequence[Dict[str, Any]] = EXPERIMENTS,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for exp in specs:
        sub = deepsoz_df[deepsoz_df["short_name"] == exp["short_name"]]
        if "dataset" in exp and "dataset" in deepsoz_df:
            sub = sub[sub["dataset"] == exp["dataset"]]
        row: Dict[str, Any] = {
            "dataset": exp.get("dataset", "Private"),
            "experiment": exp["name"],
            "short_name": exp["short_name"],
            "status": exp["status"],
            "n_folds": int(len(sub)),
            "valid_spatial_folds": int(sub["valid_spatial"].sum()) if "valid_spatial" in sub else 0,
        }
        for task, _ in CONFUSION_TASKS:
            counts = {
                "tp": sub.get(f"{task}_tp", pd.Series(dtype=float)).sum(),
                "fp": sub.get(f"{task}_fp", pd.Series(dtype=float)).sum(),
                "tn": sub.get(f"{task}_tn", pd.Series(dtype=float)).sum(),
                "fn": sub.get(f"{task}_fn", pd.Series(dtype=float)).sum(),
            }
            row.update(confusion_metrics_from_counts(counts, prefix=task))

        for task in ("patient_region", "patient_channel"):
            n_col = f"{task}_n_patients"
            row[f"{task}_n_eval_patients"] = int(np.nansum(sub.get(n_col, pd.Series(dtype=float)).to_numpy(dtype=float)))
            for metric in (
                "localization_top1_accuracy",
                "localization_top2_accuracy",
                "localization_topk_recall",
                "exact_match_accuracy",
                "jaccard",
            ):
                row[f"{task}_{metric}"] = mean(sub.get(f"{task}_{metric}", []))
        rows.append(row)
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    font_name = "DejaVu Sans"
    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        path = Path(font_path)
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            font_name = font_manager.FontProperties(fname=str(path)).get_name()
            break

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "font.family": "sans-serif",
        "font.sans-serif": [font_name, "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    if sns is not None:
        sns.set_theme(style="whitegrid", font=font_name, rc={"axes.grid": True})


def save_fig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_metric_overview(summary_df: pd.DataFrame, output_dir: Path) -> str:
    rows = []
    for _, row in summary_df.iterrows():
        for metric, label in PLOT_METRICS:
            rows.append({
                "short_name": row["short_name"],
                "metric": label,
                "value": row[f"valid_{metric}"],
            })
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    x = np.arange(len(PLOT_METRICS))
    width = 0.15
    names = [exp["short_name"] for exp in EXPERIMENTS]
    for idx, name in enumerate(names):
        vals = [
            finite_float(plot_df[(plot_df["short_name"] == name) & (plot_df["metric"] == label)]["value"].iloc[0])
            for _, label in PLOT_METRICS
        ]
        bars = ax.bar(
            x + (idx - (len(names) - 1) / 2) * width,
            vals,
            width=width,
            label=name,
            color=PALETTE.get(name),
            alpha=0.92,
        )
        for bar, val in zip(bars, vals):
            if math.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(val + 0.018, 1.02),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6.8,
                )
    ax.set_title("Recent private SOZ LOPO metrics")
    ax.set_ylabel("Mean score over valid spatial folds")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in PLOT_METRICS], rotation=18, ha="right")
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return save_fig(fig, output_dir / "experiment_metric_overview.png")


def plot_common_comparison(common_df: pd.DataFrame, output_dir: Path) -> str:
    metrics = (
        ("patient_region_top1_hit", "Region top-1"),
        ("patient_region_threshold_f1", "Region threshold F1"),
        ("patient_channel_top1_hit", "Channel top-1"),
        ("patient_channel_topk_hit", "Channel top-k"),
        ("seizure_f1", "Seizure F1"),
    )
    rows = []
    for _, row in common_df.iterrows():
        for metric, label in metrics:
            rows.append({
                "short_name": row["short_name"],
                "metric": label,
                "value": row[metric],
            })
    plot_df = pd.DataFrame(rows)
    n_common = int(common_df["common_valid_patients"].max()) if len(common_df) else 0

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    x = np.arange(len(metrics))
    width = 0.15
    names = [exp["short_name"] for exp in EXPERIMENTS]
    for idx, name in enumerate(names):
        vals = [
            finite_float(plot_df[(plot_df["short_name"] == name) & (plot_df["metric"] == label)]["value"].iloc[0])
            for _, label in metrics
        ]
        ax.bar(
            x + (idx - (len(names) - 1) / 2) * width,
            vals,
            width=width,
            label=name,
            color=PALETTE.get(name),
            alpha=0.92,
        )
    ax.set_title(f"Common valid-patient comparison (n={n_common})")
    ax.set_ylabel("Mean score")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return save_fig(fig, output_dir / "common_valid_patient_comparison.png")


def plot_patient_heatmap(folds_df: pd.DataFrame, output_dir: Path, metric: str, filename: str, title: str) -> str:
    names = [exp["short_name"] for exp in EXPERIMENTS]
    pivot = folds_df.pivot_table(index="patient", columns="short_name", values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=names)
    if "Mamba-128s" in pivot:
        pivot = pivot.sort_values(["Mamba-128s"], ascending=False, na_position="last")
    fig_height = max(5.0, 0.26 * max(len(pivot), 1))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#E5E7EB")
    if sns is not None:
        sns.heatmap(
            pivot,
            cmap=cmap,
            vmin=0,
            vmax=1,
            annot=False,
            linewidths=0.35,
            linecolor="white",
            cbar_kws={"label": "score"},
            ax=ax,
        )
    else:
        matrix = pivot.to_numpy(dtype=float)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(image, ax=ax, label="score")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    return save_fig(fig, output_dir / filename)


def plot_mamba_progress(folds_df: pd.DataFrame, output_dir: Path) -> str | None:
    sub = folds_df[folds_df["short_name"] == "Mamba-128s"].copy()
    if sub.empty:
        return None
    if "metric_mtime" in sub.columns:
        sub = sub.sort_values("metric_mtime")
    sub["fold_order"] = np.arange(1, len(sub) + 1)
    metrics = (
        ("patient_region_top1_hit", "Region top-1"),
        ("patient_channel_top1_hit", "Channel top-1"),
        ("seizure_f1", "Seizure F1"),
    )
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    colors = ["#2563EB", "#059669", "#DC2626"]
    for (metric, label), color in zip(metrics, colors):
        ax.plot(sub["fold_order"], sub[metric], marker="o", linewidth=1.6, label=label, color=color)
    ax.set_title(f"Mamba-128s fold trajectory ({len(sub)} folds completed)")
    ax.set_xlabel("Completed fold order")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(sub["fold_order"])
    ax.set_xticklabels(sub["patient"], rotation=45, ha="right")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    return save_fig(fig, output_dir / "mamba_partial_fold_progress.png")


def plot_dataset_pipeline(output_dir: Path) -> str:
    private_manifest = read_json(Path("outputs/soz_pre/private_edf_soz_manifest_soft_ica.summary.json"))
    unified = read_json(Path("outputs/soz_pre/unified_region_soz_manifest_tusz_fnsz_ica_private_soft_ica.summary.json"))
    private_pre = read_json(Path("outputs/soz_pre/preprocessed_ica_fnsz_soft_private_only/preprocess_summary.json"))
    mamba_pre = read_json(Path("outputs/soz_mamba/event_sequences_128s/preprocess_summary.json"))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    source_counts = unified.get("source_counts", {})
    axes[0].bar(source_counts.keys(), source_counts.values(), color=["#2563EB", "#F59E0B"])
    axes[0].set_title("Unified manifest rows")
    axes[0].set_ylabel("Rows")
    for idx, value in enumerate(source_counts.values()):
        axes[0].text(idx, value, str(value), ha="center", va="bottom", fontsize=8)

    stats = private_pre.get("stats", {})
    role_counts = {
        "onset": stats.get("role_onset", 0),
        "early": stats.get("role_early_ictal", 0),
        "prop.": stats.get("role_propagation", 0),
        "bg pre": stats.get("role_background_pre", 0),
        "bg post": stats.get("role_background_post", 0),
    }
    axes[1].bar(role_counts.keys(), role_counts.values(), color="#10B981")
    axes[1].set_title("Private 10s samples by role")
    axes[1].set_ylabel("Samples")
    for idx, value in enumerate(role_counts.values()):
        axes[1].text(idx, value, str(value), ha="center", va="bottom", fontsize=8)

    mamba_stats = mamba_pre.get("stats", {})
    mamba_counts = {
        "TUSZ": mamba_stats.get("source_tusz", 0),
        "private": mamba_stats.get("source_private", 0),
        "onset crops": mamba_stats.get("crop_onset", 0),
        "long crops": mamba_stats.get("crop_long_01", 0),
    }
    axes[2].bar(mamba_counts.keys(), mamba_counts.values(), color=["#3B82F6", "#F59E0B", "#8B5CF6", "#EF4444"])
    axes[2].set_title("Mamba 128s sequence samples")
    axes[2].set_ylabel("Samples")
    axes[2].tick_params(axis="x", rotation=15)
    for idx, value in enumerate(mamba_counts.values()):
        axes[2].text(idx, value, str(value), ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Recent data pipeline snapshot: "
        f"{private_manifest.get('files_selected', 'NA')} private files, "
        f"{private_pre.get('n_samples', 'NA')} private 10s samples",
        y=1.02,
        fontsize=12,
        fontweight="semibold",
    )
    return save_fig(fig, output_dir / "private_data_pipeline_snapshot.png")


def plot_deepsoz_style_patient_metrics(summary_df: pd.DataFrame, output_dir: Path) -> str:
    metrics = (
        ("localization_top1_accuracy", "Top-1 loc. acc."),
        ("accuracy", "Binary acc."),
        ("precision", "Precision"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("f1", "F1"),
    )
    tasks = (("patient_region", "Patient-level region"), ("patient_channel", "Patient-level channel"))
    names = [exp["short_name"] for exp in EXPERIMENTS]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    width = 0.15
    x = np.arange(len(metrics))
    for ax, (task, title) in zip(axes, tasks):
        for idx, name in enumerate(names):
            row = summary_df[summary_df["short_name"] == name]
            vals = []
            for metric, _ in metrics:
                col = f"{task}_{metric}"
                vals.append(finite_float(row[col].iloc[0]) if len(row) and col in row else math.nan)
            ax.bar(
                x + (idx - (len(names) - 1) / 2) * width,
                vals,
                width=width,
                label=name,
                color=PALETTE.get(name),
                alpha=0.92,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], rotation=25, ha="right")
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(-0.08, -0.25), frameon=False)
    fig.suptitle("DeepSOZ-style patient-level localization and confusion metrics", y=1.02, fontweight="semibold")
    return save_fig(fig, output_dir / "deepsoz_style_patient_metrics.png")


def plot_deepsoz_style_sample_metrics(summary_df: pd.DataFrame, output_dir: Path) -> str:
    metrics = (
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("f1", "F1"),
    )
    tasks = (
        ("sample_region", "Sample-level region"),
        ("sample_channel", "Sample-level channel"),
        ("seizure_window", "Seizure window"),
    )
    names = [exp["short_name"] for exp in EXPERIMENTS]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    width = 0.15
    x = np.arange(len(metrics))
    for ax, (task, title) in zip(axes, tasks):
        for idx, name in enumerate(names):
            row = summary_df[summary_df["short_name"] == name]
            vals = []
            for metric, _ in metrics:
                col = f"{task}_{metric}"
                vals.append(finite_float(row[col].iloc[0]) if len(row) and col in row else math.nan)
            ax.bar(
                x + (idx - (len(names) - 1) / 2) * width,
                vals,
                width=width,
                label=name,
                color=PALETTE.get(name),
                alpha=0.92,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], rotation=25, ha="right")
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.26), frameon=False)
    fig.suptitle("DeepSOZ-style sample/window confusion metrics", y=1.02, fontweight="semibold")
    return save_fig(fig, output_dir / "deepsoz_style_sample_window_metrics.png")


def dataset_names(summary_df: pd.DataFrame) -> List[str]:
    preferred = ["Private", "TUSZ"]
    present = [str(value) for value in summary_df.get("dataset", pd.Series(dtype=str)).dropna().unique()]
    ordered = [name for name in preferred if name in present]
    ordered.extend(sorted(name for name in present if name not in ordered))
    return ordered


def model_names_for_dataset(dataset: str) -> List[str]:
    names: List[str] = []
    for spec in DATASET_RUNS:
        if spec["dataset"] == dataset and spec["short_name"] not in names:
            names.append(spec["short_name"])
    return names


def legend_names_for_datasets(datasets: Sequence[str]) -> List[str]:
    names: List[str] = []
    for dataset in datasets:
        for name in model_names_for_dataset(dataset):
            if name not in names:
                names.append(name)
    return names


def plot_dataset_patient_localization_metrics(summary_df: pd.DataFrame, output_dir: Path) -> str:
    metrics = (
        ("patient_region_localization_top1_accuracy", "Region top-1"),
        ("patient_region_f1", "Region binary F1"),
        ("patient_channel_localization_top1_accuracy", "Channel top-1"),
        ("patient_channel_f1", "Channel binary F1"),
    )
    datasets = dataset_names(summary_df)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.8 * len(datasets), 4.9), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    x = np.arange(len(metrics))
    for ax, dataset in zip(axes, datasets):
        names = model_names_for_dataset(dataset)
        width = min(0.18, 0.78 / max(len(names), 1))
        for idx, name in enumerate(names):
            row = summary_df[(summary_df["dataset"] == dataset) & (summary_df["short_name"] == name)]
            vals = [
                finite_float(row[col].iloc[0]) if len(row) and col in row else math.nan
                for col, _ in metrics
            ]
            ax.bar(
                x + (idx - (len(names) - 1) / 2) * width,
                vals,
                width=width,
                label=name,
                color=PALETTE.get(name),
                alpha=0.92,
            )
        protocol = "33-fold LOPO" if dataset == "Private" else "TUSZ train/dev"
        ax.set_title(f"{dataset} ({protocol})")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], rotation=22, ha="right")
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    legend_names = legend_names_for_datasets(datasets)
    handles = [Patch(facecolor=PALETTE.get(name), label=name, alpha=0.92) for name in legend_names]
    fig.legend(handles=handles, ncol=min(5, len(handles)), loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Dataset-wise patient-level localization metrics", y=1.02, fontweight="semibold")
    return save_fig(fig, output_dir / "dataset_patient_localization_metrics.png")


def plot_dataset_confusion_metrics(summary_df: pd.DataFrame, output_dir: Path) -> str:
    metrics = (
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("f1", "F1"),
    )
    tasks = (
        ("patient_region", "Patient region"),
        ("patient_channel", "Patient channel"),
        ("seizure_window", "Seizure window"),
    )
    datasets = dataset_names(summary_df)
    fig, axes = plt.subplots(
        len(tasks),
        len(datasets),
        figsize=(6.9 * len(datasets), 3.7 * len(tasks)),
        sharey=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape((len(tasks), len(datasets)))

    x = np.arange(len(metrics))
    for row_idx, (task, task_label) in enumerate(tasks):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            names = model_names_for_dataset(dataset)
            width = min(0.16, 0.78 / max(len(names), 1))
            for idx, name in enumerate(names):
                row = summary_df[(summary_df["dataset"] == dataset) & (summary_df["short_name"] == name)]
                vals = []
                for metric, _ in metrics:
                    col = f"{task}_{metric}"
                    vals.append(finite_float(row[col].iloc[0]) if len(row) and col in row else math.nan)
                ax.bar(
                    x + (idx - (len(names) - 1) / 2) * width,
                    vals,
                    width=width,
                    label=name,
                    color=PALETTE.get(name),
                    alpha=0.92,
                )
            ax.set_title(f"{dataset}: {task_label}")
            ax.set_xticks(x)
            ax.set_xticklabels([label for _, label in metrics], rotation=22, ha="right")
            ax.set_ylim(0, 1.03)
            ax.grid(axis="y", alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("Score")
    legend_names = legend_names_for_datasets(datasets)
    handles = [Patch(facecolor=PALETTE.get(name), label=name, alpha=0.92) for name in legend_names]
    fig.legend(handles=handles, ncol=min(5, len(handles)), loc="lower center", bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("Dataset-wise DeepSOZ-style confusion metrics", y=1.01, fontweight="semibold")
    return save_fig(fig, output_dir / "dataset_confusion_metrics.png")


def plot_confusion_matrix_grid(
    summary_df: pd.DataFrame,
    task: str,
    title: str,
    filename: str,
    output_dir: Path,
) -> str:
    names = [exp["short_name"] for exp in EXPERIMENTS]
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 2.7), sharex=True, sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        row = summary_df[summary_df["short_name"] == name]
        if row.empty:
            matrix = np.zeros((2, 2), dtype=float)
        else:
            r = row.iloc[0]
            matrix = np.array([
                [finite_float(r.get(f"{task}_tn")), finite_float(r.get(f"{task}_fp"))],
                [finite_float(r.get(f"{task}_fn")), finite_float(r.get(f"{task}_tp"))],
            ], dtype=float)
            matrix = np.nan_to_num(matrix, nan=0.0)
        total = matrix.sum()
        norm = matrix / max(total, 1.0)
        ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=0.5)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"], rotation=30, ha="right")
        ax.set_yticklabels(["true 0", "true 1"])
        for i in range(2):
            for j in range(2):
                text = f"{int(round(matrix[i, j]))}\n{norm[i, j]:.1%}"
                color = "white" if norm[i, j] > 0.25 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)
    fig.suptitle(title, y=1.08, fontweight="semibold")
    return save_fig(fig, output_dir / filename)


def plot_all_deepsoz_confusions(summary_df: pd.DataFrame, output_dir: Path) -> List[str]:
    return [
        plot_confusion_matrix_grid(
            summary_df,
            "patient_region",
            "Pooled patient-level region confusion matrices",
            "deepsoz_style_patient_region_confusion_matrices.png",
            output_dir,
        ),
        plot_confusion_matrix_grid(
            summary_df,
            "patient_channel",
            "Pooled patient-level channel confusion matrices",
            "deepsoz_style_patient_channel_confusion_matrices.png",
            output_dir,
        ),
        plot_confusion_matrix_grid(
            summary_df,
            "seizure_window",
            "Pooled seizure-window confusion matrices",
            "deepsoz_style_seizure_window_confusion_matrices.png",
            output_dir,
        ),
    ]


def write_recommendations(
    summary_df: pd.DataFrame,
    common_df: pd.DataFrame,
    deepsoz_summary_df: pd.DataFrame,
    dataset_deepsoz_summary_df: pd.DataFrame,
    figures: Sequence[str],
    output_dir: Path,
) -> str:
    def row_for(short_name: str) -> pd.Series:
        return summary_df[summary_df["short_name"] == short_name].iloc[0]

    mamba = row_for("Mamba-128s")
    deepsoz_tusz = row_for("DeepSOZ+TUSZ")
    eegnet_tusz = row_for("EEGNet+TUSZ")

    table_cols = [
        "short_name",
        "status",
        "n_folds",
        "valid_spatial_folds",
        "valid_patient_region_top1_hit",
        "valid_patient_region_threshold_f1",
        "valid_patient_channel_top1_hit",
        "valid_patient_channel_topk_hit",
        "valid_seizure_f1",
    ]
    table = summary_df[table_cols].copy()
    for col in table_cols[4:]:
        table[col] = table[col].map(pct)

    common_table_cols = [
        "short_name",
        "n_rows",
        "common_valid_patients",
        "patient_region_top1_hit",
        "patient_region_threshold_f1",
        "patient_channel_top1_hit",
        "patient_channel_topk_hit",
        "seizure_f1",
    ]
    common_table = common_df[common_table_cols].copy()
    for col in common_table_cols[3:]:
        common_table[col] = common_table[col].map(pct)

    deepsoz_patient_cols = [
        "short_name",
        "n_folds",
        "patient_region_n_eval_patients",
        "patient_region_localization_top1_accuracy",
        "patient_region_accuracy",
        "patient_region_precision",
        "patient_region_sensitivity",
        "patient_region_specificity",
        "patient_region_f1",
        "patient_channel_n_eval_patients",
        "patient_channel_localization_top1_accuracy",
        "patient_channel_accuracy",
        "patient_channel_precision",
        "patient_channel_sensitivity",
        "patient_channel_specificity",
        "patient_channel_f1",
    ]
    deepsoz_patient_table = deepsoz_summary_df[deepsoz_patient_cols].copy()
    for col in deepsoz_patient_cols[2:]:
        if col.endswith("_n_eval_patients"):
            deepsoz_patient_table[col] = deepsoz_patient_table[col].astype(int)
        else:
            deepsoz_patient_table[col] = deepsoz_patient_table[col].map(pct)

    deepsoz_sample_cols = [
        "short_name",
        "sample_region_accuracy",
        "sample_region_precision",
        "sample_region_sensitivity",
        "sample_region_specificity",
        "sample_region_f1",
        "sample_channel_accuracy",
        "sample_channel_precision",
        "sample_channel_sensitivity",
        "sample_channel_specificity",
        "sample_channel_f1",
        "seizure_window_accuracy",
        "seizure_window_precision",
        "seizure_window_sensitivity",
        "seizure_window_specificity",
        "seizure_window_f1",
    ]
    deepsoz_sample_table = deepsoz_summary_df[deepsoz_sample_cols].copy()
    for col in deepsoz_sample_cols[1:]:
        deepsoz_sample_table[col] = deepsoz_sample_table[col].map(pct)

    dataset_patient_cols = [
        "dataset",
        "short_name",
        "n_folds",
        "valid_spatial_folds",
        "patient_region_n_eval_patients",
        "patient_region_localization_top1_accuracy",
        "patient_region_precision",
        "patient_region_sensitivity",
        "patient_region_specificity",
        "patient_region_f1",
        "patient_channel_n_eval_patients",
        "patient_channel_localization_top1_accuracy",
        "patient_channel_precision",
        "patient_channel_sensitivity",
        "patient_channel_specificity",
        "patient_channel_f1",
    ]
    dataset_patient_table = dataset_deepsoz_summary_df[dataset_patient_cols].copy()
    for col in dataset_patient_cols[4:]:
        if col.endswith("_n_eval_patients"):
            dataset_patient_table[col] = dataset_patient_table[col].astype(int)
        else:
            dataset_patient_table[col] = dataset_patient_table[col].map(pct)

    dataset_seizure_cols = [
        "dataset",
        "short_name",
        "seizure_window_accuracy",
        "seizure_window_precision",
        "seizure_window_sensitivity",
        "seizure_window_specificity",
        "seizure_window_f1",
    ]
    dataset_seizure_table = dataset_deepsoz_summary_df[dataset_seizure_cols].copy()
    for col in dataset_seizure_cols[2:]:
        dataset_seizure_table[col] = dataset_seizure_table[col].map(pct)

    lines = [
        "# 近期 SOZ 实验汇报建议",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 建议重点",
        "",
        (
            "1. 数据与预处理链路已经可以汇报：最新 private soft-label + ICA manifest "
            "覆盖 120 个私有 EDF 文件、122 条事件，生成 462 个私有 10s 样本；"
            "统一 manifest 共 4086 行，其中 TUSZ 3964 行、private 122 行。"
        ),
        (
            "2. 完整 baseline 对照值得作为本次主结果：DeepSOZ/EEGNet 的 private LOPO "
            "和 TUSZ-init LOPO 均已完成 33 folds，其中 27 folds 有有效病人级空间标签。"
        ),
        (
            "3. 迁移学习现象值得讲：DeepSOZ+TUSZ 在有效空间 fold 上的病人级 channel top-1 "
            f"为 {pct(deepsoz_tusz['valid_patient_channel_top1_hit'])}，"
            f"高于 DeepSOZ private 的 {pct(row_for('DeepSOZ')['valid_patient_channel_top1_hit'])}；"
            "EEGNet+TUSZ 的 region top-1 最高，但 channel 定位相对弱。"
        ),
        (
            "4. Mamba-128s 已完成 private LOPO：共 "
            f"{int(mamba['n_folds'])} folds，region top-1 为 {pct(mamba['valid_patient_region_top1_hit'])}，"
            f"channel top-k 为 {pct(mamba['valid_patient_channel_topk_hit'])}。"
            "它和 10s baseline 输入粒度不同，建议作为 128s 序列建模路线展示，并把定位与 seizure-window 结果分开讲。"
        ),
        "",
        "## 不建议重点讲",
        "",
        "- `outputs/soz_mamba_private_adapt_128s_soft_private/private_lopo_encoder/lopo_summary.json` 目前 folds 为空，不适合作为结果。",
        "- Mamba-128s 和 10s baseline 的窗口粒度不同；不建议只用 seizure-window F1 包装成单一胜负结论。",
        "- 单病人过高或过低的结果可以放在备份页，不建议在主线中过度解释。",
        "",
        "## 关键数值：有效空间 fold 均值",
        "",
        table.to_markdown(index=False),
        "",
        "## 关键数值：所有模型共同有效病人子集",
        "",
        common_table.to_markdown(index=False),
        "",
        "## 两个数据集上的表现：病人级定位与混淆矩阵指标",
        "",
        "说明：Private 是 private 数据集 33-fold LOPO；TUSZ 是 TUSZ train/dev 单次验证。`n_folds=1` 的 TUSZ 行不是 LOPO，不能和 Private 的 fold 方差直接比较。",
        "",
        dataset_patient_table.to_markdown(index=False),
        "",
        "## 两个数据集上的表现：seizure-window 指标",
        "",
        dataset_seizure_table.to_markdown(index=False),
        "",
        "## DeepSOZ-style 病人级指标",
        "",
        "说明：`localization_top1_accuracy` 是 DeepSOZ 风格的最终定位命中率；其余 accuracy/precision/sensitivity/specificity/F1 是把每个脑区或通道当作二分类标签后池化 TP/FP/TN/FN 得到。",
        "",
        deepsoz_patient_table.to_markdown(index=False),
        "",
        "## DeepSOZ-style sample/window 混淆矩阵指标",
        "",
        "说明：脑区与通道来自 `val_predictions.csv` 的标签/概率，阈值使用各 fold 的 `run_config.threshold`；seizure window 指标由已保存的 fold-level `val_metrics.json` 反推 TP/FP/TN/FN。",
        "",
        deepsoz_sample_table.to_markdown(index=False),
        "",
        "## 输出图",
        "",
    ]
    lines.extend([f"- `{path}`" for path in figures if path])
    lines.append("")

    path = output_dir / "report_recommendations.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/work_report_recent_soz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    folds = []
    for exp in EXPERIMENTS:
        folds.extend(rows_for_experiment(exp))
    folds_df = pd.DataFrame(folds)
    if folds_df.empty:
        raise SystemExit("No experiment folds found.")

    summary_df = summarize_experiments(folds_df)
    common_df = common_valid_summary(folds_df)
    deepsoz_fold_df = collect_deepsoz_style_fold_metrics(folds_df)
    deepsoz_summary_df = summarize_deepsoz_style_metrics(deepsoz_fold_df)

    dataset_folds = []
    for spec in DATASET_RUNS:
        dataset_folds.extend(rows_for_run_spec(spec))
    dataset_folds_df = pd.DataFrame(dataset_folds)
    dataset_deepsoz_fold_df = collect_deepsoz_style_fold_metrics(dataset_folds_df)
    dataset_deepsoz_summary_df = summarize_deepsoz_style_metrics(dataset_deepsoz_fold_df, DATASET_RUNS)

    folds_df.to_csv(output_dir / "fold_metrics_long.csv", index=False)
    summary_df.to_csv(output_dir / "experiment_summary.csv", index=False)
    common_df.to_csv(output_dir / "common_valid_patient_summary.csv", index=False)
    deepsoz_fold_df.to_csv(output_dir / "deepsoz_style_fold_metrics.csv", index=False)
    deepsoz_summary_df.to_csv(output_dir / "deepsoz_style_summary.csv", index=False)
    dataset_folds_df.to_csv(output_dir / "dataset_fold_metrics_long.csv", index=False)
    dataset_deepsoz_fold_df.to_csv(output_dir / "dataset_deepsoz_style_fold_metrics.csv", index=False)
    dataset_deepsoz_summary_df.to_csv(output_dir / "dataset_deepsoz_style_summary.csv", index=False)

    figures: List[str] = []
    figures.append(plot_metric_overview(summary_df, output_dir))
    figures.append(plot_common_comparison(common_df, output_dir))
    figures.append(plot_deepsoz_style_patient_metrics(deepsoz_summary_df, output_dir))
    figures.append(plot_deepsoz_style_sample_metrics(deepsoz_summary_df, output_dir))
    figures.append(plot_dataset_patient_localization_metrics(dataset_deepsoz_summary_df, output_dir))
    figures.append(plot_dataset_confusion_metrics(dataset_deepsoz_summary_df, output_dir))
    figures.extend(plot_all_deepsoz_confusions(deepsoz_summary_df, output_dir))
    figures.append(plot_patient_heatmap(
        folds_df,
        output_dir,
        "patient_region_top1_hit",
        "patient_region_top1_heatmap.png",
        "Patient-level region top-1 hit",
    ))
    figures.append(plot_patient_heatmap(
        folds_df,
        output_dir,
        "patient_channel_top1_hit",
        "patient_channel_top1_heatmap.png",
        "Patient-level channel top-1 hit",
    ))
    progress = plot_mamba_progress(folds_df, output_dir)
    if progress:
        figures.append(progress)
    figures.append(plot_dataset_pipeline(output_dir))

    recommendations = write_recommendations(
        summary_df,
        common_df,
        deepsoz_summary_df,
        dataset_deepsoz_summary_df,
        figures,
        output_dir,
    )
    print(f"Wrote {output_dir}")
    print(f"Wrote {recommendations}")


if __name__ == "__main__":
    main()
