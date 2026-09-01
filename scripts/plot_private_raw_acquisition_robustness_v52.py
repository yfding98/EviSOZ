#!/usr/bin/env python3
"""Plot the frozen private v29 raw acquisition robustness audit v52."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_AUDIT = (
    ROOT / "outputs/trustworthy_soz_private_v29_raw_acquisition_robustness_audit_v52_20260816"
)
DEFAULT_OUTPUT = ROOT / "figures/trustworthy_soz_private_raw_acquisition_robustness_v52_20260816"
CONDITIONS = (
    ("identity", "Identity"),
    ("anchor_shift_m5s", "Anchor −5 s"),
    ("anchor_shift_m2s", "Anchor −2 s"),
    ("anchor_shift_p2s", "Anchor +2 s"),
    ("anchor_shift_p5s", "Anchor +5 s"),
    ("amplitude_scale_0p5", "Amplitude 0.5×"),
    ("amplitude_scale_2p0", "Amplitude 2.0×"),
    ("dynamic_top1_dropout", "Selected channel\nmissing"),
)
COLORS = {
    "identity": "#235789",
    "anchor": "#C44536",
    "amplitude": "#E6A23C",
    "dropout": "#6A4C93",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _condition_color(name: str) -> str:
    if name == "identity":
        return COLORS["identity"]
    if name.startswith("anchor_"):
        return COLORS["anchor"]
    if name.startswith("amplitude_"):
        return COLORS["amplitude"]
    return COLORS["dropout"]


def plot(audit_directory: Path, output_prefix: Path) -> tuple[Path, Path]:
    result = json.loads((audit_directory / "result.json").read_text(encoding="utf-8"))
    condition_rows = {
        row["intervention"]: row
        for row in _read_csv(audit_directory / "condition_summary.csv")
    }
    dropout_rows = _read_csv(audit_directory / "single_channel_dropout_summary.csv")
    dynamic = result["exhaustive_single_candidate_channel_dropout"]

    labels = [label for _, label in CONDITIONS]
    names = [name for name, _ in CONDITIONS]
    strict = []
    n4 = []
    top1_retention = []
    top3_jaccard = []
    for name in names:
        if name == "dynamic_top1_dropout":
            summary = dynamic["dynamic_original_top1_dropout_summary"]
            stability = dynamic["dynamic_original_top1_dropout_stability_all_88"]
            strict.append(100 * float(summary["event_micro"]["strict"]))
            n4.append(100 * float(summary["event_micro"]["relaxed"]))
            top1_retention.append(100 * float(stability["top1_retention"]))
            top3_jaccard.append(100 * float(stability["top3_jaccard"]))
        else:
            row = condition_rows[name]
            strict.append(100 * float(row["strict_event_micro"]))
            n4.append(100 * float(row["neighborhood4_event_micro"]))
            top1_retention.append(100 * float(row["top1_retention"]))
            top3_jaccard.append(100 * float(row["top3_jaccard"]))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)
    x = np.arange(len(names))
    width = 0.38
    colors = [_condition_color(name) for name in names]

    axis = axes[0, 0]
    axis.bar(x - width / 2, strict, width, label="Strict Top-1", color=colors, alpha=0.98)
    axis.bar(
        x + width / 2,
        n4,
        width,
        label="Neighborhood-4",
        color=colors,
        alpha=0.42,
        edgecolor=colors,
    )
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Reference agreement (%)")
    axis.set_title("A  Private localization under raw acquisition perturbations")
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[0, 1]
    axis.plot(x, top1_retention, marker="o", linewidth=2, color="#C44536", label="Top-1 retention")
    axis.plot(x, top3_jaccard, marker="s", linewidth=2, color="#2A9D8F", label="Top-3 Jaccard")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylim(0, 105)
    axis.set_ylabel("Ranking stability (%)")
    axis.set_title("B  Prediction stability across all 88 target-blind events")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)

    channels = [row["channel"] for row in dropout_rows]
    channel_strict = np.asarray([100 * float(row["strict_event_micro"]) for row in dropout_rows])
    channel_n4 = np.asarray([100 * float(row["neighborhood4_event_micro"]) for row in dropout_rows])
    order = np.argsort(channel_strict)
    y = np.arange(len(channels))
    axis = axes[1, 0]
    axis.scatter(channel_strict[order], y, color="#6A4C93", s=30, label="Strict Top-1")
    axis.scatter(channel_n4[order], y, color="#2A9D8F", s=30, label="Neighborhood-4")
    axis.hlines(y, channel_strict[order], channel_n4[order], color="#B8B8B8", linewidth=1)
    axis.axvline(49.02, color="#235789", linestyle="--", linewidth=1, label="Identity strict")
    axis.axvline(74.51, color="#235789", linestyle=":", linewidth=1, label="Identity N4")
    axis.set_yticks(y, np.asarray(channels)[order])
    axis.set_xlim(20, 82)
    axis.set_xlabel("Reference agreement after channel removal (%)")
    axis.set_title("C  Exhaustive single-candidate-channel removal")
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    axis.grid(axis="x", alpha=0.2)

    selected = dynamic["original_top1_probability_drop"]
    nonselected = dynamic["nonselected_channel_mean_probability_drop"]
    contrast = dynamic["selected_minus_nonselected_probability_drop"]
    points = [
        float(selected["patient_equal_mean"]),
        float(nonselected["patient_equal_mean"]),
        float(contrast["patient_equal_mean"]),
    ]
    intervals = [
        selected["patient_cluster_bootstrap_ci95"],
        nonselected["patient_cluster_bootstrap_ci95"],
        contrast["patient_cluster_bootstrap_ci95"],
    ]
    lower = [point - float(interval[0]) for point, interval in zip(points, intervals)]
    upper = [float(interval[1]) - point for point, interval in zip(points, intervals)]
    axis = axes[1, 1]
    px = np.arange(3)
    axis.errorbar(
        px,
        points,
        yerr=np.asarray([lower, upper]),
        fmt="o",
        markersize=7,
        capsize=5,
        color="#6A4C93",
        linewidth=2,
    )
    axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xticks(
        px,
        ("Selected channel\nremoved", "Mean of other 17\nremoved", "Selected − other"),
    )
    axis.set_ylabel("Original Top-1 probability drop")
    axis.set_title("D  Candidate-specific raw channel reliance")
    axis.grid(axis="y", alpha=0.2)
    axis.text(
        0.02,
        0.96,
        "Patient-equal mean and patient-cluster 95% interval",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        color="#444444",
    )

    figure.suptitle(
        "Frozen v29 private raw acquisition robustness audit (post-open descriptive)",
        fontsize=13,
        fontweight="bold",
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    png, pdf = plot(args.audit, args.output)
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
