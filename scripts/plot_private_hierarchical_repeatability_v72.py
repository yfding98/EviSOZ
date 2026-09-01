#!/usr/bin/env python3
"""Plot target-blind hierarchical repeatability for v29 and Raw200."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "outputs/trustworthy_soz_dual_model_private_hierarchical_repeatability_v72_20260816/result.json"
DEFAULT_PDF = ROOT / "figures/trustworthy_soz_private_hierarchical_repeatability_v72_20260816.pdf"
DEFAULT_PNG = ROOT / "figures/trustworthy_soz_private_hierarchical_repeatability_v72_20260816.png"
COLORS = {
    "v29": "#315C9B",
    "v29_between": "#9CB4D4",
    "raw200": "#D28735",
    "raw200_between": "#E9BC84",
}


def _bar_labels(ax: plt.Axes, bars: object, values: Sequence[float]) -> None:
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=7.5)


def plot(result: dict[str, object], *, pdf: Path, png: Path) -> None:
    models = result["models"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), constrained_layout=True)

    ax = axes[0]
    metrics = ["top1_agreement", "top3_jaccard", "rank_spearman"]
    labels = ["Top-1\nagreement", "Top-3\nJaccard", "Full-rank\nSpearman"]
    x = np.arange(len(metrics))
    width = 0.2
    bars_to_plot = (
        ("v29", "within_patient", -1.5 * width, "v29 within", COLORS["v29"]),
        ("v29", "between_patient", -0.5 * width, "v29 between", COLORS["v29_between"]),
        ("raw200", "within_patient", 0.5 * width, "Raw200 within", COLORS["raw200"]),
        ("raw200", "between_patient", 1.5 * width, "Raw200 between", COLORS["raw200_between"]),
    )
    for model, scope, offset, name, color in bars_to_plot:
        key = "patient_equal" if scope == "within_patient" else "patient_pair_equal"
        values = [float(models[model][scope][key][metric]) for metric in metrics]
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        _bar_labels(ax, bars, values)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("Similarity")
    ax.set_title("A  Same-patient versus different-patient")
    ax.legend(frameon=False, fontsize=7.5, ncols=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    x = np.arange(2)
    width = 0.34
    within = [
        float(models[model]["within_patient"]["patient_equal"]["jensen_shannon_distance"])
        for model in ("v29", "raw200")
    ]
    between = [
        float(models[model]["between_patient"]["patient_pair_equal"]["jensen_shannon_distance"])
        for model in ("v29", "raw200")
    ]
    first = ax.bar(x - width / 2, within, width, color=[COLORS["v29"], COLORS["raw200"]], label="Within patient")
    second = ax.bar(x + width / 2, between, width, color=[COLORS["v29_between"], COLORS["raw200_between"]], label="Between patients")
    _bar_labels(ax, first, within)
    _bar_labels(ax, second, between)
    ax.set_xticks(x, ["v29", "Raw200"])
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("Jensen–Shannon distance (lower = closer)")
    ax.set_title("B  Complete C18 probability distance")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    metrics = [
        "top1_agreement",
        "top3_jaccard",
        "rank_spearman",
        "jensen_shannon_distance",
    ]
    labels = ["Top-1", "Top-3", "Rank ρ", "JS reduction"]
    x = np.arange(len(metrics))
    width = 0.34
    for offset, model, label, color in (
        (-width / 2, "v29", "v29", COLORS["v29"]),
        (width / 2, "raw200", "Raw200", COLORS["raw200"]),
    ):
        values = [
            float(models[model]["same_patient_similarity_advantage_over_null"][metric])
            for metric in metrics
        ]
        bars = ax.bar(x + offset, values, width, color=color, label=label)
        _bar_labels(ax, bars, values)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.32)
    ax.set_ylabel("Advantage over event-count null")
    ax.set_title("C  Patient-structure permutation audit")
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.02,
        0.97,
        "10,000 permutations\nall rank/JS tails ≤ 1/10,001",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
        color="#444444",
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Target-blind patient structure in frozen private predictions",
        fontsize=12.5,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.035,
        "Same-patient similarity may include acquisition/session fingerprints and is not qualified as SOZ stability.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = json.loads(args.result.resolve(strict=True).read_text(encoding="utf-8"))
    if result.get("schema_version") != "trustworthy_soz_dual_model_private_hierarchical_repeatability_v72":
        raise ValueError("unexpected v72 result schema")
    plot(result, pdf=args.pdf.resolve(), png=args.png.resolve())
    print(json.dumps({"pdf": str(args.pdf.resolve()), "png": str(args.png.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
