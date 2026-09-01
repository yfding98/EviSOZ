#!/usr/bin/env python3
"""Plot the v46-v54 trustworthy SOZ method-audit results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "figures/trustworthy_soz_current_data_v55_20260816"
V46 = ROOT / "outputs/trustworthy_soz_v29_patient_bag_event_consistency_v46_20260816/result.json"
V47 = ROOT / "outputs/trustworthy_soz_v29_public_private_construct_shift_v47_20260816/result.json"
V48 = ROOT / "outputs/trustworthy_soz_v29_uncertainty_proxies_v48_20260816/result.json"
V49 = ROOT / "outputs/trustworthy_soz_private_v29_raw_channel_time_audit_v49_20260816/result.json"
V50 = ROOT / "outputs/trustworthy_soz_v29_nonfoundation_fine_baseline_v50_20260816/result.json"
V54 = ROOT / "outputs/trustworthy_soz_raw25_tcn_baseline_audit_v54_20260816/result.json"

BLUE = "#2A6FBB"
ORANGE = "#E07A2D"
GREEN = "#2C9A78"
RED = "#C64545"
PURPLE = "#7656A5"
GRAY = "#7A7F87"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _percent_axis(axis: plt.Axes, *, maximum: float = 100.0) -> None:
    axis.set_ylim(0, maximum)
    axis.set_ylabel("Percent (%)")
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def main() -> int:
    v46, v47, v48, v49, v50, v54 = map(_json, (V46, V47, V48, V49, V50, V54))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "legend.fontsize": 8.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 8.8), constrained_layout=True)

    # A: foundation representation increment against the non-foundation baseline.
    axis = axes[0, 0]
    labels = ("Public\nStrict", "Public\nN4", "Private\nStrict", "Private\nN4")
    v29_values = np.asarray(
        [
            v50["public"]["v29_metrics"]["top1"]["strict_accuracy"],
            v50["public"]["v29_metrics"]["top1"]["relaxed_accuracy"],
            v50["private"]["v29_summary"]["event_micro"]["strict"],
            v50["private"]["v29_summary"]["event_micro"]["relaxed"],
        ]
    ) * 100
    fine_values = np.asarray(
        [
            v50["public"]["fine_only_metrics"]["top1"]["strict_accuracy"],
            v50["public"]["fine_only_metrics"]["top1"]["relaxed_accuracy"],
            v50["private"]["fine_only_summary"]["event_micro"]["strict"],
            v50["private"]["fine_only_summary"]["event_micro"]["relaxed"],
        ]
    ) * 100
    raw25_values = np.asarray(
        [
            v54["public"]["raw25_tcn"]["top1"]["strict_accuracy"],
            v54["public"]["raw25_tcn"]["top1"]["relaxed_accuracy"],
            v54["private"]["raw25_tcn"]["event_micro"]["strict"],
            v54["private"]["raw25_tcn"]["event_micro"]["relaxed"],
        ]
    ) * 100
    x = np.arange(len(labels))
    width = 0.25
    axis.bar(x - width, v29_values, width, color=BLUE, label="Frozen LaBraM v29")
    axis.bar(x, fine_values, width, color=GRAY, label="Non-FM fine-only")
    axis.bar(x + width, raw25_values, width, color=PURPLE, label="Raw25 TCN")
    for positions, values in (
        (x - width, v29_values),
        (x, fine_values),
        (x + width, raw25_values),
    ):
        for position, value in zip(positions, values):
            axis.text(position, value + 1.6, f"{value:.1f}", ha="center", va="bottom", fontsize=7.6)
    axis.set_xticks(x, labels)
    _percent_axis(axis, maximum=100)
    axis.set_title("A  Foundation-representation increment", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper left", fontsize=7.4)

    # B: public event-bag sensitivity.
    axis = axes[0, 1]
    sizes = ("1", "2", "4", "8", "All")
    strict = []
    relaxed = []
    retention = []
    for key in ("1", "2", "4", "8"):
        value = v46["public"]["bag_subsampling"][key]["metrics_and_stability"]
        strict.append(value["strict"]["mean"] * 100)
        relaxed.append(value["neighborhood4"]["mean"] * 100)
        retention.append(value["top1_retention"]["mean"] * 100)
    full = v46["public"]["bag_subsampling"]["all"]["metrics"]
    strict.append(full["strict"] * 100)
    relaxed.append(full["neighborhood4"] * 100)
    retention.append(100)
    axis.plot(sizes, strict, marker="o", color=BLUE, linewidth=2, label="Strict")
    axis.plot(sizes, relaxed, marker="s", color=ORANGE, linewidth=2, label="N4")
    axis.plot(sizes, retention, marker="^", color=GREEN, linewidth=2, label="Top-1 retention")
    _percent_axis(axis, maximum=105)
    axis.set_xlabel("Maximum seizures per public patient")
    axis.set_title("B  Patient-bag sensitivity", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=3, loc="lower right")

    # C: private raw intervention performance.
    axis = axes[0, 2]
    raw_labels = ("Identity", "Pre", "Early", "Late", "Full")
    raw_keys = ("identity", "top1_pre_removed", "top1_early_removed", "top1_late_removed", "top1_full_removed")
    raw_strict = [v49["intervention_summaries"][key]["event_micro"]["strict"] * 100 for key in raw_keys]
    raw_n4 = [v49["intervention_summaries"][key]["event_micro"]["relaxed"] * 100 for key in raw_keys]
    raw_x = np.arange(len(raw_labels))
    raw_width = 0.36
    axis.bar(raw_x - raw_width / 2, raw_strict, raw_width, color=BLUE, label="Strict")
    axis.bar(raw_x + raw_width / 2, raw_n4, raw_width, color=ORANGE, label="N4")
    axis.set_xticks(raw_x, raw_labels)
    _percent_axis(axis, maximum=100)
    axis.set_title("C  Private raw Top-1-channel intervention", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper right")

    # D: candidate-specific effect versus matched channel controls.
    axis = axes[1, 0]
    phase_order = ("full", "pre", "early", "late")
    means = []
    lower = []
    upper = []
    for phase in phase_order:
        value = v49["phase_matched_selected_vs_control"][phase][
            "selected_minus_matched_control_probability_drop"
        ]
        means.append(value["patient_equal_mean"])
        lower.append(value["patient_equal_mean"] - value["patient_cluster_bootstrap_ci95"][0])
        upper.append(value["patient_cluster_bootstrap_ci95"][1] - value["patient_equal_mean"])
    colors = [PURPLE, GRAY, RED, ORANGE]
    axis.errorbar(
        np.arange(4),
        means,
        yerr=np.asarray([lower, upper]),
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1.2,
    )
    axis.scatter(np.arange(4), means, s=65, c=colors, zorder=3)
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.set_xticks(np.arange(4), ("Full", "Pre", "Early", "Late"))
    axis.set_ylabel("Selected − control probability drop")
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.7)
    axis.set_title("D  Raw candidate-specific reliance", loc="left", fontweight="bold")

    # E: private laterality failure profile.
    axis = axes[1, 1]
    laterality_order = ("left_only", "right_only", "bilateral_or_mixed")
    laterality_labels = ("Left-only", "Right-only", "Bilateral/mixed")
    private_laterality = v47["private"]["reference_laterality"]
    strict = np.asarray([private_laterality[key]["strict"]["unit_micro"] for key in laterality_order]) * 100
    n4 = np.asarray([private_laterality[key]["relaxed"]["unit_micro"] for key in laterality_order]) * 100
    contra = np.asarray([private_laterality[key]["contralateral_far"]["unit_micro"] for key in laterality_order]) * 100
    x = np.arange(3)
    w = 0.25
    axis.bar(x - w, strict, w, color=BLUE, label="Strict")
    axis.bar(x, n4, w, color=ORANGE, label="N4")
    axis.bar(x + w, contra, w, color=RED, label="Contralateral far")
    axis.set_xticks(x, laterality_labels, rotation=12)
    _percent_axis(axis, maximum=100)
    axis.set_title("E  Private laterality audit", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper right")

    # F: fixed uncertainty proxies; no selection.
    axis = axes[1, 2]
    proxy_keys = (
        "negative_top1_top2_margin",
        "normalized_predictive_entropy",
        "H_D_jensen_shannon",
        "H_D_top1_disagreement",
    )
    proxy_labels = ("Margin", "Entropy", "H/D JS", "H/D Top-1")
    public_auc = [v48["public"]["proxies"][key]["strict_failure_detection"]["patient_equal_weighted_AUROC"] for key in proxy_keys]
    private_auc = [v48["private"]["proxies"][key]["strict_failure_detection"]["patient_equal_weighted_AUROC"] for key in proxy_keys]
    x = np.arange(len(proxy_keys))
    uncertainty_width = 0.36
    axis.bar(x - uncertainty_width / 2, public_auc, uncertainty_width, color=BLUE, label="Public")
    axis.bar(x + uncertainty_width / 2, private_auc, uncertainty_width, color=GREEN, label="Private")
    axis.axhline(0.5, color="#333333", linewidth=1, linestyle="--", label="Chance")
    axis.set_xticks(x, proxy_labels, rotation=15)
    axis.set_ylim(0.25, 0.8)
    axis.set_ylabel("Strict-error detection AUROC")
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.7)
    axis.set_title("F  Uncertainty does not uniformly transport", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=3, loc="upper center")

    figure.suptitle(
        "Frozen qualification-aware scalp-electrode SOZ-reference ranking: current-data method audit",
        fontsize=13.2,
        fontweight="bold",
    )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(FIGURE.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(FIGURE.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
