#!/usr/bin/env python3
"""Plot the frozen v29 target-blind public/private shift audit v56."""

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
    / "outputs/trustworthy_soz_v29_target_blind_input_shift_v56_20260816/result.json"
)
DEFAULT_OUTPUT = (
    ROOT / "figures/trustworthy_soz_target_blind_input_shift_v56_20260816.pdf"
)


SHORT_NAMES = {
    "channel_mean::node_change_detected": "Node change detected (mean)",
    "channel_mean::node_relative_recruitment_delay_sec_censored": "Relative change delay (mean)",
    "channel_mean::node_late_change_persistence_12_36s": "Late change persistence (mean)",
    "channel_mean::node_change_latency_sec_censored": "Change latency, censored (mean)",
    "channel_mean::bipolar_incident_change_detected": "Bipolar change detected (mean)",
    "channel_mean::bipolar_incident_relative_delay_sec_censored": "Bipolar relative delay (mean)",
    "channel_mean::bipolar_incident_change_latency_sec_censored": "Bipolar latency, censored (mean)",
    "channel_mean::node_change_persistence_0_12s": "Early change persistence (mean)",
    "channel_mean::node_late_minus_early_change": "Late minus early change (mean)",
    "channel_sd::node_late_minus_early_change": "Late minus early change (spatial SD)",
}


def _mass(value: dict[str, float], categories: Sequence[str]) -> list[float]:
    return [100.0 * float(value.get(category, 0.0)) for category in categories]


def plot(result_path: Path, output: Path) -> Path:
    result = json.loads(result_path.resolve(strict=True).read_text(encoding="utf-8"))
    if result.get("schema_version") != "trustworthy_soz_v29_target_blind_input_shift_v56":
        raise ValueError("unexpected v56 result schema")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), constrained_layout=True)

    acquisition = result["acquisition"]
    public_mass = acquisition["public"]["patient_equal_source_sfreq_mass"]
    private_mass = acquisition["private"]["patient_equal_source_sfreq_mass"]
    categories = sorted(
        set(public_mass) | set(private_mass), key=lambda value: float(value)
    )
    x = np.arange(len(categories))
    width = 0.38
    axes[0].bar(
        x - width / 2,
        _mass(public_mass, categories),
        width,
        label="Public development (102 patients)",
        color="#4C78A8",
    )
    axes[0].bar(
        x + width / 2,
        _mass(private_mass, categories),
        width,
        label="Private transport (31 patients)",
        color="#F58518",
    )
    axes[0].set_xticks(x, categories)
    axes[0].set_xlabel("Source sampling rate (Hz)")
    axes[0].set_ylabel("Patient-equal mass (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("A  Acquisition metadata differs before frozen 200-Hz CAR19")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    rows = result["shift_summary"]["largest_absolute_smd_descriptors"]
    values = np.asarray(
        [float(row["standardized_mean_difference_private_minus_public"]) for row in rows]
    )
    lower = np.asarray([float(row["smd_patient_bootstrap_ci95_low"]) for row in rows])
    upper = np.asarray([float(row["smd_patient_bootstrap_ci95_high"]) for row in rows])
    labels = [SHORT_NAMES.get(str(row["descriptor"]), str(row["descriptor"])) for row in rows]
    order = np.arange(len(rows))[::-1]
    colors = np.where(values >= 0, "#F58518", "#4C78A8")
    axes[1].barh(order, values, color=colors, alpha=0.88)
    axes[1].errorbar(
        values,
        order,
        xerr=np.vstack((values - lower, upper - values)),
        fmt="none",
        ecolor="#333333",
        elinewidth=0.8,
        capsize=2,
    )
    axes[1].axvline(0, color="#222222", linewidth=0.8)
    axes[1].axvline(-1, color="#777777", linewidth=0.7, linestyle="--")
    axes[1].axvline(1, color="#777777", linewidth=0.7, linestyle="--")
    axes[1].set_yticks(order, labels)
    axes[1].set_xlabel("Standardized mean difference (private − public)")
    axes[1].set_title("B  Largest patient-level target-blind evidence shifts")
    axes[1].grid(axis="x", alpha=0.2)
    summary = result["shift_summary"]
    axes[1].text(
        0.99,
        0.02,
        (
            f"|SMD|≥0.5: {summary['descriptor_count_abs_smd_ge_0_5']}/40\n"
            f"|SMD|≥1.0: {summary['descriptor_count_abs_smd_ge_1_0']}/40"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
    )

    figure.suptitle(
        "Frozen v29 target-blind public-to-private input/evidence shift\n"
        "Patient-equal description only; no SOZ reference, model selection, OOD threshold, or abstention rule",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    png = output.with_suffix(".png")
    figure.savefig(png, bbox_inches="tight")
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
