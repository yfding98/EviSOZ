#!/usr/bin/env python3
"""Plot private longitudinal reference stability and event-pairing audit v74."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "outputs/trustworthy_soz_private_event_reference_alignment_v74_20260816/result.json"
DEFAULT_PDF = ROOT / "figures/trustworthy_soz_private_event_reference_alignment_v74_20260816.pdf"
DEFAULT_PNG = ROOT / "figures/trustworthy_soz_private_event_reference_alignment_v74_20260816.png"
COLORS = {"v29": "#315C9B", "raw200": "#D28735", "null": "#B9BEC7"}


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _alignment_panel(
    ax: plt.Axes,
    *,
    model: str,
    result: Mapping[str, object],
    title: str,
) -> None:
    audit = result["event_specific_prediction_reference_alignment"][model]
    formal = audit["formal_event_pairing"]["multi_event_only_patient_equal"]
    null = audit["within_patient_event_permutation_null"]["multi_event_only_patient_equal"]
    metrics = ("strict", "positive_mass", "reciprocal_first_positive_rank")
    labels = ("Strict\nTop-1", "Positive-set\nmass", "First-positive\nreciprocal rank")
    x = np.arange(len(metrics))
    null_mean = np.asarray([float(null[metric]["mean"]) for metric in metrics])
    null_low = np.asarray([float(null[metric]["quantile_025_50_975"][0]) for metric in metrics])
    null_high = np.asarray([float(null[metric]["quantile_025_50_975"][2]) for metric in metrics])
    formal_values = np.asarray([float(formal[metric]) for metric in metrics])
    bars = ax.bar(x, null_mean, width=0.58, color=COLORS["null"], label="Within-patient null mean")
    ax.errorbar(
        x,
        null_mean,
        yerr=np.vstack((null_mean - null_low, null_high - null_mean)),
        fmt="none",
        color="#4F555C",
        capsize=4,
        linewidth=1.2,
        label="Null 2.5–97.5%",
    )
    ax.scatter(x, formal_values, s=50, color=COLORS[model], zorder=3, label="Formal event pairing")
    for index, value in enumerate(formal_values):
        ax.text(index, min(value + 0.045, 0.76), f"{value:.3f}", ha="center", fontsize=8, color=COLORS[model])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Patient-equal value")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")


def plot(result: Mapping[str, object], *, pdf: Path, png: Path) -> None:
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
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.65), constrained_layout=True)

    reference = result["reference_stability"]["reference"]
    metrics = (
        "reference_exact",
        "reference_any_overlap",
        "reference_jaccard",
        "reference_laterality_equal",
        "reference_cardinality_equal",
    )
    labels = ("Exact set", "Any overlap", "Jaccard", "Laterality", "Cardinality")
    values = np.asarray([float(reference[metric]["patient_equal"]) for metric in metrics])
    low = np.asarray([float(reference[metric]["patient_bootstrap_ci95"][0]) for metric in metrics])
    high = np.asarray([float(reference[metric]["patient_bootstrap_ci95"][1]) for metric in metrics])
    x = np.arange(len(metrics))
    bars = axes[0].bar(x, values, color="#4B8B78", width=0.68)
    axes[0].errorbar(
        x,
        values,
        yerr=np.vstack((values - low, high - values)),
        fmt="none",
        color="#27332F",
        capsize=3,
        linewidth=1.1,
    )
    axes[0].bar_label(bars, labels=[_percent(value) for value in values], padding=3, fontsize=7.5)
    axes[0].set_xticks(x, labels, rotation=24, ha="right")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Patient-equal within-patient agreement")
    axes[0].set_title("A  Clinician reference stability", loc="left", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    _alignment_panel(
        axes[1],
        model="v29",
        result=result,
        title="B  v29 event-specific alignment",
    )
    _alignment_panel(
        axes[2],
        model="raw200",
        result=result,
        title="C  Raw200 event-specific alignment",
    )
    fig.suptitle(
        "Private longitudinal target audit: frozen predictions versus within-patient event permutation",
        fontsize=12,
        fontweight="bold",
    )
    for output in (pdf, png):
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = json.loads(args.result.resolve(strict=True).read_text(encoding="utf-8"))
    plot(result, pdf=args.pdf, png=args.png)
    print(args.pdf)
    print(args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
