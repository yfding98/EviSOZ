#!/usr/bin/env python3
"""Plot frozen v29 endpoint-definition sensitivity v59."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT
    / "outputs/trustworthy_soz_v29_spatial_endpoint_sensitivity_v59_20260816/result.json"
)
DEFAULT_OUTPUT = (
    ROOT / "figures/trustworthy_soz_spatial_endpoint_sensitivity_v59_20260816.pdf"
)


def plot(result_path: Path, output: Path) -> Path:
    result = json.loads(result_path.resolve(strict=True).read_text(encoding="utf-8"))
    if result.get("schema_version") != "trustworthy_soz_v29_spatial_endpoint_sensitivity_v59":
        raise ValueError("unexpected v59 result schema")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 160})
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.15), constrained_layout=True)

    endpoints = ("strict", "official_N2", "official_N4")
    labels = ("Strict", "Official N2", "Official N4")
    x = np.arange(3)
    width = 0.36
    public = result["public"]["summary"]
    private = result["private"]["summary"]
    public_point = np.asarray([100 * public[key]["patient_equal"] for key in endpoints])
    private_point = np.asarray([100 * private[key]["patient_equal"] for key in endpoints])
    public_ci = np.asarray([public[key]["patient_cluster_bootstrap_ci95"] for key in endpoints]) * 100
    private_ci = np.asarray([private[key]["patient_cluster_bootstrap_ci95"] for key in endpoints]) * 100
    axes[0].bar(x - width / 2, public_point, width, color="#4C78A8", label="Public (102 patients)")
    axes[0].bar(x + width / 2, private_point, width, color="#F58518", label="Private patient-equal (23 clusters)")
    axes[0].errorbar(
        x - width / 2,
        public_point,
        yerr=np.vstack((public_point - public_ci[:, 0], public_ci[:, 1] - public_point)),
        fmt="none",
        ecolor="#222222",
        capsize=2,
        linewidth=0.8,
    )
    axes[0].errorbar(
        x + width / 2,
        private_point,
        yerr=np.vstack((private_point - private_ci[:, 0], private_ci[:, 1] - private_point)),
        fmt="none",
        ecolor="#222222",
        capsize=2,
        linewidth=0.8,
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Patient-equal agreement (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("A  Agreement depends on tolerance definition")
    axes[0].legend(frameon=False, fontsize=7.7, loc="upper left")
    axes[0].grid(axis="y", alpha=0.2)

    strata = ("1", "2", "3-4", ">=5")
    public_counts = np.asarray(
        [result["public"]["positive_set_size_strata"].get(key, {}).get("units", 0) for key in strata]
    )
    private_counts = np.asarray(
        [result["private"]["positive_set_size_strata"].get(key, {}).get("units", 0) for key in strata]
    )
    axes[1].bar(x=np.arange(4) - width / 2, height=100 * public_counts / public_counts.sum(), width=width, color="#4C78A8")
    axes[1].bar(x=np.arange(4) + width / 2, height=100 * private_counts / private_counts.sum(), width=width, color="#F58518")
    axes[1].set_xticks(np.arange(4), strata)
    axes[1].set_xlabel("Reference-positive set size")
    axes[1].set_ylabel("Units (%)")
    axes[1].set_ylim(0, 70)
    axes[1].set_title("B  Private N4 is often enabled only at 3–4 positives")
    axes[1].grid(axis="y", alpha=0.2)
    for index, (left, right) in enumerate(zip(public_counts, private_counts)):
        axes[1].text(index - width / 2, 100 * left / public_counts.sum() + 1, str(left), ha="center", fontsize=7.5)
        axes[1].text(index + width / 2, 100 * right / private_counts.sum() + 1, str(right), ha="center", fontsize=7.5)

    distance_keys = ("0", "1", "2", ">=3")
    distance_labels = ("0 exact", "1 hop", "2 hops", "≥3 hops")
    colors = ("#59A14F", "#EDC948", "#B279A2", "#E15759")
    for row, (name, summary) in enumerate((("Public", public), ("Private", private))):
        counts = summary["minimum_undirected_graph_hops"]["counts"]
        denominator = sum(counts[key] for key in distance_keys)
        left = 0.0
        for key, label, color in zip(distance_keys, distance_labels, colors):
            width_value = 100.0 * counts[key] / denominator
            axes[2].barh(row, width_value, left=left, color=color, label=label if row == 0 else None)
            if width_value >= 8:
                axes[2].text(left + width_value / 2, row, str(counts[key]), ha="center", va="center", fontsize=8)
            left += width_value
    axes[2].set_yticks((0, 1), ("Public", "Private"))
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel("Units (%)")
    axes[2].set_title("C  Distance to nearest reference-positive electrode")
    axes[2].legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center")

    figure.suptitle(
        "Frozen v29 spatial endpoint sensitivity\n"
        "Strict is primary; N2/N4 and graph hops are spatial-tolerance descriptions",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(figure)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = plot(args.result, args.output)
    print(json.dumps({"output": str(output), "png": str(output.with_suffix('.png'))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
