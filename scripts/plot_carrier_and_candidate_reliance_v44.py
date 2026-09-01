#!/usr/bin/env python3
"""Render the v43/v44 frozen carrier audit as a paper-ready four-panel figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H = ROOT / "outputs/trustworthy_soz_labram_v29_h_carrier_stress_v43_20260816"
DEFAULT_CANDIDATE = ROOT / "outputs/trustworthy_soz_v29_candidate_channel_reliance_v44_20260816"
DEFAULT_OUTPUT = ROOT / "figures/trustworthy_soz_carrier_reliance_v44_20260816"


H_LABELS = {
    "identity_replay": "Identity",
    "zero_H_content": "Zero H",
    "channel_mean_locality_removed": "Channel mean",
    "left_right_channel_swap": "L/R swap",
    "remove_onset_minus_baseline": "Remove O−B",
    "remove_early_minus_baseline": "Remove E−B",
    "remove_late_minus_early": "Remove L−E",
    "swap_first_third_contrast": "Swap 1st/3rd",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _percent(value: str) -> float:
    return 100.0 * float(value)


def _metric_bars(
    axis: plt.Axes,
    labels: Sequence[str],
    strict: Sequence[float],
    relaxed: Sequence[float],
    *,
    title: str,
    ylim: tuple[float, float] = (0, 100),
) -> None:
    x = np.arange(len(labels))
    width = 0.37
    axis.bar(x - width / 2, strict, width, color="#38598c", label="Strict Top-1")
    axis.bar(x + width / 2, relaxed, width, color="#7a9e7e", label="Neighborhood-4")
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylim(*ylim)
    axis.set_ylabel("Agreement (%)")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)


def run(h_directory: Path, candidate_directory: Path, output: Path) -> None:
    public_h = [
        row
        for row in _rows(h_directory / "public_stress_table.csv")
        if row["scope"] == "stressed_H_plus_frozen_D_equal"
    ]
    private_h = [
        row
        for row in _rows(h_directory / "private_stress_table.csv")
        if row["scope"] == "stressed_H_plus_frozen_D_equal"
    ]
    candidate = _rows(candidate_directory / "intervention_summary.csv")
    public_candidate = [
        row for row in candidate if row["dataset"] == "public_consumed_development"
    ]
    private_candidate = [
        row for row in candidate if row["dataset"] == "private_post_open_transport"
    ]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 7.5), constrained_layout=True)
    labels = [H_LABELS[row["perturbation"]] for row in public_h]
    _metric_bars(
        axes[0, 0],
        labels,
        [_percent(row["strict_top1"]) for row in public_h],
        [_percent(row["neighborhood4_top1"]) for row in public_h],
        title="A  Public consumed-development H stress",
    )
    _metric_bars(
        axes[0, 1],
        labels,
        [_percent(row["strict_event_micro"]) for row in private_h],
        [_percent(row["neighborhood4_event_micro"]) for row in private_h],
        title="B  Private post-open transport H stress",
    )

    intervention_labels = ["Original", "Top-1 removed", "Top-1 only"]
    _metric_bars(
        axes[1, 0],
        intervention_labels,
        [_percent(row["strict"]) for row in public_candidate],
        [_percent(row["neighborhood4"]) for row in public_candidate],
        title="C  Public candidate-channel intervention",
    )
    _metric_bars(
        axes[1, 1],
        intervention_labels,
        [_percent(row["strict"]) for row in private_candidate],
        [_percent(row["neighborhood4"]) for row in private_candidate],
        title="D  Private candidate-channel intervention",
    )
    for index, row in enumerate(private_candidate):
        axes[1, 1].text(
            index,
            3,
            f"contra-far={row['contralateral_far_count']}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
            color="#7d2020",
        )

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    figure.suptitle(
        "Frozen LaBraM carrier reliance: channel-local dependence transports, phase semantics do not qualify",
        fontsize=12,
        fontweight="bold",
        y=1.07,
    )
    figure.text(
        0.5,
        -0.015,
        "Representation-level interventions; not raw-EEG causal explanations. Private outcomes are post-open descriptive transport.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--h-directory", type=Path, default=DEFAULT_H)
    parser.add_argument("--candidate-directory", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    run(args.h_directory, args.candidate_directory, args.output)
    print(str(args.output.with_suffix(".pdf")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
