#!/usr/bin/env python3
"""Plot the frozen dual-model private construct/repeatability audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "outputs/trustworthy_soz_dual_model_private_construct_repeatability_v71_20260816/result.json"
DEFAULT_PDF = ROOT / "figures/trustworthy_soz_dual_model_private_audit_v71_20260816.pdf"
DEFAULT_PNG = ROOT / "figures/trustworthy_soz_dual_model_private_audit_v71_20260816.png"
COLORS = {"v29": "#315C9B", "raw200": "#D28735"}


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def plot(result: dict[str, object], *, pdf: Path, png: Path) -> None:
    construct = result["reference_construct"]["models"]
    repeatability = result["target_blind_repeatability"]["models"]
    concordance = result["cross_model_concordance"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.5), constrained_layout=True)

    ax = axes[0]
    labels = ["Complete set\nstrict", "Documented-singleton\nsensitivity"]
    x = np.arange(len(labels))
    width = 0.34
    for offset, model in ((-width / 2, "v29"), (width / 2, "raw200")):
        summary = construct[model]["summary"]
        values = [
            summary["original_set_strict"]["unit_micro"],
            summary["documented_singleton_uniform_top1"]["unit_micro"],
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label="v29" if model == "v29" else "Raw200",
            color=COLORS[model],
        )
        ax.bar_label(bars, labels=[_percent(value) for value in values], padding=2, fontsize=8)
    ax.set_ylim(0, 0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Private event-level functional")
    ax.set_title("A  Multi-positive reference sensitivity")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    labels = ["Pairwise Top-1\nagreement", "Pairwise Top-3\nJaccard"]
    x = np.arange(len(labels))
    for offset, model in ((-width / 2, "v29"), (width / 2, "raw200")):
        summary = repeatability[model]["multi_event_patients"]
        values = [
            summary["patient_equal_mean_pairwise_top1_agreement"],
            summary["patient_equal_mean_pairwise_top3_jaccard"],
        ]
        bars = ax.bar(x + offset, values, width, color=COLORS[model])
        ax.bar_label(bars, labels=[_percent(value) for value in values], padding=2, fontsize=8)
    ax.set_ylim(0, 0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Patient-equal repeatability")
    ax.set_title("B  Within-patient seizure repeatability")
    ax.text(
        0.02,
        0.98,
        "28 multi-seizure patients / 85 events",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#444444",
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    counts = concordance["evaluated_51_strict_overlap_counts"]
    names = ["Both", "v29 only", "Raw200 only", "Neither"]
    values = [
        counts["both_strict"],
        counts["v29_only_strict"],
        counts["raw200_only_strict"],
        counts["neither_strict"],
    ]
    colors = ["#4C8C6B", COLORS["v29"], COLORS["raw200"], "#A7A7A7"]
    bottom = 0
    for name, value, color in zip(names, values, colors, strict=True):
        ax.bar([0], [value], bottom=bottom, width=0.58, color=color, label=f"{name}: {value}")
        if value >= 5:
            ax.text(0, bottom + value / 2, str(value), ha="center", va="center", color="white", weight="bold")
        bottom += value
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(0, 55)
    ax.set_xticks([0], ["51 reference-evaluable events"])
    ax.set_ylabel("Event count")
    ax.set_title("C  Strict-hit overlap and model discordance")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.text(
        0.98,
        0.03,
        f"All-88 Top-1 agreement: {_percent(concordance['all_88_event_micro_top1_agreement'])}\n"
        f"All-88 Top-3 Jaccard: {_percent(concordance['all_88_event_micro_top3_jaccard'])}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Frozen dual-model audit on the private cohort",
        fontsize=12.5,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.035,
        "Read-only post-open sensitivity analysis; no model selection, patient-consensus target, or alternative accuracy claim.",
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
    if result.get("schema_version") != "trustworthy_soz_dual_model_private_construct_repeatability_v71":
        raise ValueError("unexpected v71 result schema")
    plot(result, pdf=args.pdf.resolve(), png=args.png.resolve())
    print(json.dumps({"pdf": str(args.pdf.resolve()), "png": str(args.png.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
