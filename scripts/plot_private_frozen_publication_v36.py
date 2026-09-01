#!/usr/bin/env python3
"""Render the frozen private-transfer result and trade-off panels for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT / "outputs/trustworthy_soz_private_frozen_publication_v36_20260816/result.json"
)
DEFAULT_OUTPUT = ROOT / "figures/trustworthy_soz_private_transfer_v36_20260816"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != (
        "trustworthy_soz_private_frozen_publication_audit_v36"
    ):
        raise ValueError("Unexpected private publication audit")
    return value


def render(result: dict[str, object], output_prefix: Path) -> None:
    arms = result["frozen_arms"]
    arm_keys = ["H_fold_mean", "D_fold_mean", "v29_equal_H_D"]
    arm_labels = ["H", "D", "H/D"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.3), constrained_layout=True)

    ax = axes[0, 0]
    metric_keys = ["strict", "relaxed", "laterality_agreement"]
    metric_labels = ["Strict Top-1", "Neighborhood-4", "Laterality"]
    x = np.arange(len(metric_keys), dtype=float)
    width = 0.23
    for arm_index, (arm, label, color) in enumerate(zip(arm_keys, arm_labels, colors)):
        summary = arms[arm]
        values = np.asarray([summary["patient_equal_event_macro"][metric] for metric in metric_keys])
        intervals = [
            summary["patient_cluster_bootstrap_ci95"][metric]["patient_equal_ci95"]
            for metric in metric_keys
        ]
        lower = values - np.asarray([interval[0] for interval in intervals])
        upper = np.asarray([interval[1] for interval in intervals]) - values
        ax.bar(
            x + (arm_index - 1) * width,
            values * 100,
            width,
            color=color,
            label=label,
            yerr=np.vstack([lower, upper]) * 100,
            capsize=3,
            linewidth=0.5,
            edgecolor="white",
        )
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Patient-equal event rate (%)")
    ax.set_title("a  Private zero-adaptation transfer")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)

    ax = axes[0, 1]
    bottom = np.zeros(len(arm_keys))
    for key, label, color in (
        ("strict_exact", "Exact", "#54A24B"),
        ("neighbor_only", "Neighbor only", "#ECA82C"),
        ("far", "Far", "#E45756"),
    ):
        values = np.asarray([arms[arm]["endpoint_counts"][key] for arm in arm_keys])
        ax.bar(arm_labels, values, bottom=bottom, label=label, color=color)
        for index, (value, base) in enumerate(zip(values, bottom)):
            if value:
                ax.text(index, base + value / 2, str(int(value)), ha="center", va="center", fontsize=9)
        bottom += values
    ax.set_ylim(0, 55)
    ax.set_ylabel("Events (n=51; 23 patient clusters)")
    ax.set_title("b  Clinical-distance error composition")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.7)

    ax = axes[1, 0]
    paired = result["paired_proposed_minus_H"]
    paired_keys = ["strict", "relaxed", "hit_at_3", "hit_at_5"]
    paired_labels = ["Strict Top-1", "Neighborhood-4", "Hit@3", "Hit@5"]
    values = np.asarray([paired[key]["patient_equal_difference"] for key in paired_keys])
    intervals = [paired[key]["patient_cluster_bootstrap_ci95"] for key in paired_keys]
    lower = values - np.asarray([interval[0] for interval in intervals])
    upper = np.asarray([interval[1] for interval in intervals]) - values
    y = np.arange(len(values))
    ax.errorbar(
        values * 100,
        y,
        xerr=np.vstack([lower, upper]) * 100,
        fmt="o",
        color="#333333",
        ecolor="#4C78A8",
        capsize=4,
    )
    ax.axvline(0, color="#999999", linewidth=1, linestyle="--")
    ax.set_yticks(y, paired_labels)
    ax.invert_yaxis()
    ax.set_xlabel("H/D minus H (percentage points)")
    ax.set_title("c  Paired patient-clustered trade-off")
    ax.grid(axis="x", color="#eeeeee", linewidth=0.7)

    ax = axes[1, 1]
    laterality = result["private_proposed_strata"]["reference_laterality"]
    strata_keys = [key for key in ("left_only", "right_only", "bilateral_or_mixed") if key in laterality]
    labels = [
        {"left_only": "Left", "right_only": "Right", "bilateral_or_mixed": "Bilateral/mixed"}[key]
        for key in strata_keys
    ]
    strict = [laterality[key]["event_micro"]["strict"] * 100 for key in strata_keys]
    neighbor = [
        laterality[key]["event_micro"]["neighbor_only"] * 100 for key in strata_keys
    ]
    far = [laterality[key]["event_micro"]["far"] * 100 for key in strata_keys]
    ax.bar(labels, strict, color="#54A24B", label="Exact")
    ax.bar(labels, neighbor, bottom=strict, color="#ECA82C", label="Neighbor only")
    ax.bar(
        labels,
        far,
        bottom=np.asarray(strict) + np.asarray(neighbor),
        color="#E45756",
        label="Far",
    )
    counts = [laterality[key]["event_count"] for key in strata_keys]
    for index, count in enumerate(counts):
        ax.text(index, 102, f"n={count}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Events (%)")
    ax.set_title("d  Outcome by reference laterality")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.7)

    fig.suptitle(
        "Frozen LaBraM private transfer: accuracy, error distance, and ranking trade-offs",
        fontsize=14,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(_load(args.result), args.output_prefix)
    print(f"png={args.output_prefix.with_suffix('.png')}")
    print(f"pdf={args.output_prefix.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
