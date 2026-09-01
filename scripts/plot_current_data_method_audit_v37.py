#!/usr/bin/env python3
"""Plot the frozen current-data public method audit for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT
    / "outputs/trustworthy_soz_current_data_method_audit_v37_20260816/result.json"
)
DEFAULT_PNG = ROOT / "figures/trustworthy_soz_method_audit_v37_20260816.png"
DEFAULT_PDF = ROOT / "figures/trustworthy_soz_method_audit_v37_20260816.pdf"


ARM_ORDER = (
    "v29_equal_H_D",
    "H_only",
    "D_only",
    "DeepSOZ_local_replay",
    "fold_local_prevalence_only",
)
ARM_LABELS = ("H/D", "H", "D", "DeepSOZ", "Spatial prior")
COLORS = {
    "strict": "#205493",
    "neighbor": "#58A5A5",
    "far": "#D66B5D",
    "neutral": "#6F7782",
}


def _percent(values: list[float]) -> np.ndarray:
    return 100.0 * np.asarray(values, dtype=float)


def plot(result_path: Path, png: Path, pdf: Path) -> None:
    result = json.loads(result_path.resolve(strict=True).read_text(encoding="utf-8"))
    metrics = result["metrics"]
    paired = result["paired_v29_minus_comparator_patient_bootstrap"]
    stress = result["carrier_replacement_stress"]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 150,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)

    # A: absolute metrics.
    axis = axes[0, 0]
    strict = _percent([metrics[name]["top1"]["strict_accuracy"] for name in ARM_ORDER])
    relaxed = _percent([metrics[name]["top1"]["relaxed_accuracy"] for name in ARM_ORDER])
    positions = np.arange(len(ARM_ORDER))
    width = 0.36
    axis.bar(positions - width / 2, strict, width, color=COLORS["strict"], label="Strict Top-1")
    axis.bar(
        positions + width / 2,
        relaxed,
        width,
        color=COLORS["neighbor"],
        label="Neighborhood-4",
    )
    axis.set_xticks(positions, ARM_LABELS)
    axis.set_ylim(0, 88)
    axis.set_ylabel("Patients (%)")
    axis.set_title("A  Frozen public development comparison")
    axis.legend(frameon=False, ncols=2, loc="upper right")
    axis.grid(axis="y", alpha=0.2)

    # B: paired patient bootstrap differences.
    axis = axes[0, 1]
    comparator_order = (
        "H_only",
        "D_only",
        "DeepSOZ_local_replay",
        "fold_local_prevalence_only",
    )
    comparator_labels = ("H", "D", "DeepSOZ", "Spatial prior")
    endpoints = (("strict", "Strict", COLORS["strict"]), ("relaxed", "Neighborhood-4", COLORS["neighbor"]))
    offsets = (-0.12, 0.12)
    y_positions = np.arange(len(comparator_order))
    for (endpoint, label, color), offset in zip(endpoints, offsets):
        values = np.asarray([paired[name][endpoint]["delta"] for name in comparator_order]) * 100
        low = np.asarray([paired[name][endpoint]["ci95"][0] for name in comparator_order]) * 100
        high = np.asarray([paired[name][endpoint]["ci95"][1] for name in comparator_order]) * 100
        axis.errorbar(
            values,
            y_positions + offset,
            xerr=np.vstack((values - low, high - values)),
            fmt="o",
            capsize=3,
            color=color,
            label=label,
        )
    axis.axvline(0, color="#333333", linewidth=1, linestyle="--")
    axis.set_yticks(y_positions, comparator_labels)
    axis.invert_yaxis()
    axis.set_xlabel("v29 minus comparator (percentage points, 95% CI)")
    axis.set_title("B  Paired patient-level differences")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="x", alpha=0.2)

    # C: exact / neighbor-only / far composition.
    axis = axes[1, 0]
    exact = strict
    neighbor_only = relaxed - strict
    far = 100.0 - relaxed
    axis.bar(positions, exact, color=COLORS["strict"], label="Exact")
    axis.bar(positions, neighbor_only, bottom=exact, color=COLORS["neighbor"], label="Neighbor-only")
    axis.bar(positions, far, bottom=relaxed, color=COLORS["far"], label="Far")
    axis.set_xticks(positions, ARM_LABELS)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Patients (%)")
    axis.set_title("C  Top-1 error-distance composition")
    axis.legend(frameon=False, ncols=3, loc="upper center")

    # D: post-hoc branch replacement stability.
    axis = axes[1, 1]
    stress_order = (
        "replace_D_with_prevalence",
        "replace_H_with_prevalence",
        "replace_both_with_prevalence",
    )
    stress_labels = ("Replace D", "Replace H", "Replace H & D")
    top1_retention = _percent([stress[name]["top1_retention"] for name in stress_order])
    top3_jaccard = _percent([stress[name]["top3_jaccard"] for name in stress_order])
    positions_d = np.arange(len(stress_order))
    axis.bar(
        positions_d - width / 2,
        top1_retention,
        width,
        color=COLORS["strict"],
        label="Top-1 retained",
    )
    axis.bar(
        positions_d + width / 2,
        top3_jaccard,
        width,
        color=COLORS["neutral"],
        label="Top-3 Jaccard",
    )
    axis.set_xticks(positions_d, stress_labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Stability (%)")
    axis.set_title("D  Replace a frozen carrier with spatial prior")
    axis.legend(frameon=False, ncols=2, loc="upper right")
    axis.grid(axis="y", alpha=0.2)

    figure.text(
        0.5,
        -0.01,
        "All public results are adaptive development evidence on 102 patients; "
        "neighborhood-4 is a sensitivity endpoint, not strict electrode accuracy.",
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
    )
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, bbox_inches="tight", dpi=300)
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot(args.result, args.png, args.pdf)
    print(args.png)
    print(args.pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
