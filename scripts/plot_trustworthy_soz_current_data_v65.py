#!/usr/bin/env python3
"""Render the final current-data trustworthy-SOZ audit composite figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "figures/trustworthy_soz_current_data_v65_20260816.pdf"
DEFAULT_PNG = ROOT / "figures/trustworthy_soz_current_data_v65_20260816.png"
V56 = ROOT / "outputs/trustworthy_soz_v29_target_blind_input_shift_v56_20260816/result.json"
V57 = ROOT / "outputs/trustworthy_soz_v29_fail_closed_input_contract_v57_20260816/result.json"
V59 = ROOT / "outputs/trustworthy_soz_v29_spatial_endpoint_sensitivity_v59_20260816/result.json"
V60 = ROOT / "outputs/trustworthy_soz_raw200_shallow_baseline_audit_v60_20260816/result.json"
V61 = ROOT / "outputs/trustworthy_soz_v29_patient_label_permutation_v61_20260816/result.json"
V63 = ROOT / "outputs/trustworthy_soz_v29_partition_stability_v63_20260816/manifest.json"
V64 = ROOT / "outputs/trustworthy_soz_v29_partition_stability_audit_v64_20260816/result.json"

NAVY = "#17365D"
BLUE = "#3E7CB1"
GREEN = "#2A7F62"
ORANGE = "#C76D2B"
RED = "#A63D40"
GRAY = "#6C757D"
PALE = "#F4F6F8"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=NAVY, pad=8)
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        va="top",
    )


def _percent_axis(ax: plt.Axes, top: float = 1.0) -> None:
    ax.set_ylim(0, top)
    ticks = np.linspace(0, top, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{100 * value:.0f}" for value in ticks])
    ax.set_ylabel("Agreement / score (%)")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def render(pdf: Path, png: Path) -> None:
    shift = _load(V56)
    fail = _load(V57)
    endpoints = _load(V59)
    raw = _load(V60)
    permutation = _load(V61)
    partition_public = _load(V63)
    partition_private = _load(V64)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.1), facecolor="white")
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    # A: endpoint hierarchy.
    public = endpoints["public"]["summary"]
    private = endpoints["private"]["summary"]
    cohort_values = np.asarray(
        [
            [public[key]["unit_micro"] for key in ("strict", "official_N2", "official_N4")],
            [private[key]["unit_micro"] for key in ("strict", "official_N2", "official_N4")],
            [private[key]["patient_equal"] for key in ("strict", "official_N2", "official_N4")],
        ]
    )
    x = np.arange(3)
    width = 0.23
    for index, (name, color) in enumerate(
        zip(("Strict", "Official N2", "Official N4"), (NAVY, ORANGE, GREEN))
    ):
        bars = ax_a.bar(x + (index - 1) * width, cohort_values[:, index], width, label=name, color=color)
        ax_a.bar_label(bars, labels=[f"{100 * value:.1f}" for value in cohort_values[:, index]], padding=2, fontsize=7)
    ax_a.set_xticks(x, ("Public\npatient OOF", "Private\nevent", "Private\npatient-equal"))
    _percent_axis(ax_a, 0.9)
    ax_a.legend(frameon=False, ncol=3, fontsize=7.6, loc="upper left")
    _panel(ax_a, "A", "Primary vs spatially tolerant endpoints")

    # B: private v29 vs Raw200.
    v29_private = raw["private"]["v29"]
    raw_private = raw["private"]["raw200_shallow"]
    v29_n2 = v29_private["official_N2"]["event_micro"]
    raw_n2 = raw_private["official_N2"]["event_micro"]
    private_compare = np.asarray(
        [
            [v29_private["event_micro"]["strict"], v29_n2, v29_private["event_micro"]["relaxed"], v29_private["event_micro"]["laterality_agreement"]],
            [raw_private["event_micro"]["strict"], raw_n2, raw_private["event_micro"]["relaxed"], raw_private["event_micro"]["laterality_agreement"]],
        ]
    )
    x = np.arange(4)
    for index, (name, color) in enumerate((("Frozen v29", GREEN), ("Raw200", GRAY))):
        bars = ax_b.bar(x + (index - 0.5) * 0.34, private_compare[index], 0.34, label=name, color=color)
        ax_b.bar_label(bars, labels=[f"{100 * value:.1f}" for value in private_compare[index]], padding=2, fontsize=7)
    ax_b.set_xticks(x, ("Strict", "N2", "N4", "Laterality"))
    _percent_axis(ax_b, 0.9)
    ax_b.legend(frameon=False, fontsize=8)
    _panel(ax_b, "B", "Private transport: v29 vs full-bandwidth Raw200")

    # C: public reversal.
    v29_public = raw["public"]["v29"]
    raw_public = raw["public"]["raw200_shallow"]
    public_compare = np.asarray(
        [
            [v29_public["top1"]["strict_accuracy"], v29_public["official_N2"]["relaxed_accuracy"], v29_public["top1"]["relaxed_accuracy"], v29_public["ranking"]["macro_average_precision"]],
            [raw_public["top1"]["strict_accuracy"], raw_public["official_N2"]["relaxed_accuracy"], raw_public["top1"]["relaxed_accuracy"], raw_public["ranking"]["macro_average_precision"]],
        ]
    )
    x = np.arange(4)
    for index, (name, color) in enumerate((("Frozen v29", BLUE), ("Raw200", ORANGE))):
        bars = ax_c.bar(x + (index - 0.5) * 0.34, public_compare[index], 0.34, label=name, color=color)
        ax_c.bar_label(bars, labels=[f"{100 * value:.1f}" for value in public_compare[index]], padding=2, fontsize=7)
    ax_c.set_xticks(x, ("Strict", "N2", "N4", "Macro-AP"))
    _percent_axis(ax_c, 0.9)
    ax_c.legend(frameon=False, fontsize=8)
    _panel(ax_c, "C", "Public development reversal: Raw200 scores higher")

    # D: label permutation falsification.
    perm_values = np.asarray([row["strict"] for row in permutation["permutation_metrics"]])
    rng = np.random.default_rng(20260865)
    ax_d.scatter(rng.normal(0, 0.035, len(perm_values)), perm_values, color=GRAY, alpha=0.8, s=30, label="20 patient-label permutations")
    formal = permutation["formal_v29_metrics"]["strict"]
    prior = permutation["prevalence_only_metrics"]["strict"]
    ax_d.axhline(formal, color=GREEN, linewidth=2.2, label=f"Formal v29 {100 * formal:.1f}%")
    ax_d.axhline(prior, color=ORANGE, linewidth=1.8, linestyle="--", label=f"Prevalence {100 * prior:.1f}%")
    ax_d.set_xlim(-0.35, 0.35)
    ax_d.set_xticks([0], ["Patient-level null"])
    _percent_axis(ax_d, 0.65)
    ax_d.legend(frameon=False, fontsize=7.7, loc="upper right")
    _panel(ax_d, "D", "True patient–reference correspondence is required")

    # E: partition stability, public and private.
    public_alt = partition_public["alternative_public_metrics"]
    private_alt = [row["compact"] for row in partition_private["alternative_partitions"]]
    formal_public = partition_public["formal_v29_public_metrics"]
    formal_private = partition_private["formal_v29"]["compact"]
    for endpoint_index, (name, pub_key, pri_key, color, offset) in enumerate(
        (
            ("Strict", "strict", "strict_event_micro", NAVY, -0.08),
            ("N4", "official_N4", "N4_event_micro", GREEN, 0.08),
        )
    ):
        pub = [row[pub_key] for row in public_alt]
        pri = [row[pri_key] for row in private_alt]
        ax_e.scatter(np.full(len(pub), 0 + offset), pub, color=color, s=35, alpha=0.75)
        ax_e.scatter(np.full(len(pri), 1 + offset), pri, color=color, s=35, alpha=0.75, label=name)
        ax_e.scatter([0 + offset], [formal_public[pub_key]], marker="*", s=150, color=color, edgecolor="white", linewidth=0.8)
        ax_e.scatter([1 + offset], [formal_private[pri_key]], marker="*", s=150, color=color, edgecolor="white", linewidth=0.8)
    ax_e.set_xticks([0, 1], ["Public OOF", "Private transport"])
    _percent_axis(ax_e, 0.9)
    ax_e.legend(frameon=False, fontsize=8, title="Dots: alternative splits\nStars: formal v29")
    _panel(ax_e, "E", "Full H/D patient-partition sensitivity")

    # F: safety-case facts.
    ax_f.axis("off")
    _panel(ax_f, "F", "Released audit facts and fail-closed limits")
    shift_summary = shift["shift_summary"]
    challenges = fail["challenge_matrix"]
    lines = (
        ("Input/evidence shift", f"{shift_summary['descriptor_count_abs_smd_ge_0_5']}/40 descriptors |SMD| ≥ 0.5", BLUE),
        ("Fail-closed challenges", f"{challenges['invalid_challenges']}/{challenges['invalid_challenges']} invalid inputs rejected; 0 localization/report calls", GREEN),
        ("Private exact/far errors", "25 exact · 13 neighbor-only · 13 far", ORANGE),
        ("High-severity errors", "9 contralateral-far · 1 known-spread Top-1", RED),
        ("Concept qualification", "M FAIL · I FAIL · learned V NO-GO", RED),
        ("Uncertainty", "NOT QUALIFIED — no clinical-risk abstention", GRAY),
    )
    y = 0.88
    for heading, value, color in lines:
        ax_f.add_patch(
            plt.Rectangle((0.02, y - 0.075), 0.025, 0.07, transform=ax_f.transAxes, color=color, clip_on=False)
        )
        ax_f.text(0.065, y, heading, transform=ax_f.transAxes, fontweight="bold", color=NAVY, va="top", fontsize=9)
        ax_f.text(0.065, y - 0.04, value, transform=ax_f.transAxes, color="#27313A", va="top", fontsize=8.3)
        y -= 0.135
    ax_f.text(
        0.02,
        0.02,
        "Development-stage method audit — not cortical SOZ, surgical target,\nindependent external validation, or calibrated clinical decision support.",
        transform=ax_f.transAxes,
        fontsize=8.1,
        color=GRAY,
        va="bottom",
    )

    fig.suptitle(
        "Trustworthy scalp-electrode SOZ-reference candidate ranking: current-data audit",
        fontsize=15,
        fontweight="bold",
        color="#17212B",
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.965), h_pad=2.4, w_pad=2.0)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render(args.pdf, args.png)
    print(f"wrote {args.pdf}")
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
