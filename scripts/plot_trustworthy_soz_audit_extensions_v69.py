#!/usr/bin/env python3
"""Render the v66-v68 trustworthy-method audit extension figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "figures/trustworthy_soz_audit_extensions_v69_20260816.pdf"
DEFAULT_PNG = ROOT / "figures/trustworthy_soz_audit_extensions_v69_20260816.png"
V60 = ROOT / "outputs/trustworthy_soz_raw200_shallow_baseline_audit_v60_20260816/result.json"
V61 = ROOT / "outputs/trustworthy_soz_v29_patient_label_permutation_v61_20260816/result.json"
V66 = ROOT / "outputs/trustworthy_soz_v29_evidence_admission_noninterference_v66_20260816/result.json"
V67 = ROOT / "outputs/trustworthy_soz_v29_reference_set_perturbation_v67_20260816/result.json"
V68 = ROOT / "outputs/trustworthy_soz_v29_stratified_patient_label_permutation_v68_20260816/result.json"

NAVY = "#17365D"
BLUE = "#3E7CB1"
GREEN = "#2A7F62"
ORANGE = "#C76D2B"
RED = "#A63D40"
GRAY = "#6C757D"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=NAVY, pad=8)
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=NAVY, va="top")


def _percent_axis(ax: plt.Axes, top: float = 1.0) -> None:
    ax.set_ylim(0, top)
    ticks = np.linspace(0, top, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{100 * value:.0f}" for value in ticks])
    ax.set_ylabel("Percent")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def _reference_panel(ax: plt.Axes, summary: dict[str, object], title: str, letter: str) -> None:
    complete = summary["original_set_strict"]["unit_micro"]
    singleton = summary["documented_singleton_uniform_top1"]["unit_micro"]
    lower = summary["documented_singleton_top1_lower"]["unit_micro"]
    upper = summary["documented_singleton_top1_upper"]["unit_micro"]
    bars = ax.bar([0, 1], [complete, singleton], color=[GREEN, ORANGE], width=0.62)
    ax.bar_label(bars, labels=[f"{100 * complete:.1f}", f"{100 * singleton:.1f}"], padding=3, fontsize=8)
    ax.errorbar(
        [1], [singleton],
        yerr=np.asarray([[singleton - lower], [upper - singleton]]),
        fmt="none", ecolor=NAVY, capsize=5, linewidth=1.5,
        label="documented-singleton min–max",
    )
    ax.set_xticks([0, 1], ["Complete\npositive set", "Uniform documented\nsingleton sensitivity"])
    _percent_axis(ax, 0.7)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    _panel(ax, letter, title)


def render(pdf: Path, png: Path) -> None:
    raw = _load(V60)
    unconditional = _load(V61)
    admission = _load(V66)
    reference = _load(V67)
    stratified = _load(V68)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.1), facecolor="white")
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    # A: keep private transport as the clinical-data emphasis.
    v29 = raw["private"]["v29"]["event_micro"]
    raw200 = raw["private"]["raw200_shallow"]["event_micro"]
    values = np.asarray([
        [v29["strict"], v29["relaxed"], v29["laterality_agreement"]],
        [raw200["strict"], raw200["relaxed"], raw200["laterality_agreement"]],
    ])
    x = np.arange(3)
    for index, (name, color) in enumerate((("Frozen v29", GREEN), ("Raw200", GRAY))):
        bars = ax_a.bar(x + (index - 0.5) * 0.34, values[index], 0.34, label=name, color=color)
        ax_a.bar_label(bars, labels=[f"{100 * value:.1f}" for value in values[index]], padding=2, fontsize=7.5)
    ax_a.set_xticks(x, ("Strict", "N4", "Laterality"))
    _percent_axis(ax_a, 0.9)
    ax_a.legend(frameon=False, fontsize=8)
    _panel(ax_a, "A", "Private post-open transport remains the emphasis")

    _reference_panel(ax_b, reference["private"]["summary"], "Private agreement depends on set semantics", "B")
    _reference_panel(ax_c, reference["public"]["summary"], "Public agreement shows the same construct sensitivity", "C")

    # D: actual non-interference result and finite-test uncertainty.
    public = admission["noninterference"]["public"]
    private = admission["noninterference"]["private"]
    ax_d.axis("off")
    _panel(ax_d, "D", "Failed evidence cannot re-enter the frozen ranking")
    ax_d.text(0.5, 0.70, "0 / 8,192", ha="center", va="center", transform=ax_d.transAxes, fontsize=31, fontweight="bold", color=GREEN)
    ax_d.text(0.5, 0.56, "public + private ranking violations", ha="center", transform=ax_d.transAxes, color=NAVY, fontsize=10)
    ax_d.text(0.5, 0.41, f"4,096 trials/cohort · upper 95% bound {100 * public['zero_violation_exact_binomial_upper95']:.3f}%", ha="center", transform=ax_d.transAxes, color=GRAY, fontsize=8.5)
    ax_d.text(0.5, 0.25, "M FAIL · I FAIL · learned V NO-GO\nuncertainty NOT QUALIFIED · direct V DESCRIPTION ONLY", ha="center", transform=ax_d.transAxes, color=RED, fontsize=9, linespacing=1.5)

    # E: controller actions under known truth.
    scenarios = admission["controlled_policy_validation"]["scenarios"]
    labels = ("Qualified", "Chance", "Shift\nreversal", "Low\ncoverage", "Shortcut", "Description")
    keys = ("qualified_signal", "native_chance", "source_valid_transport_reversal", "insufficient_coverage", "shortcut_contaminated", "target_blind_description")
    rates = [scenarios[key]["expected_decision_rate"] for key in keys]
    bars = ax_e.bar(np.arange(len(keys)), rates, color=[GREEN, ORANGE, RED, RED, RED, BLUE])
    ax_e.bar_label(bars, labels=[f"{100 * value:.1f}" for value in rates], padding=2, fontsize=7)
    ax_e.set_xticks(np.arange(len(keys)), labels, fontsize=7.3)
    _percent_axis(ax_e, 1.08)
    ax_e.set_ylabel("Expected policy action (%)")
    ax_e.text(
        0.02,
        0.12,
        "Unsafe false-admission rate: 0.03%",
        transform=ax_e.transAxes,
        color=RED,
        fontsize=8.2,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )
    _panel(ax_e, "E", "Controlled qualification mechanism validation")

    # F: unconditional vs prior-preserving conditional patient-label nulls.
    rng = np.random.default_rng(20260869)
    u = np.asarray([row["strict"] for row in unconditional["permutation_metrics"]])
    s = np.asarray([row["strict"] for row in stratified["permutation_metrics"]])
    ax_f.scatter(rng.normal(0, 0.035, len(u)), u, s=23, color=GRAY, alpha=0.65, label="20 unconditional")
    ax_f.scatter(rng.normal(1, 0.045, len(s)), s, s=22, color=BLUE, alpha=0.65, label="99 cardinality/laterality-stratified")
    formal = stratified["formal_v29_metrics"]["strict"]
    prevalence = stratified["prevalence_only_metrics"]["strict"]
    ax_f.axhline(formal, color=GREEN, linewidth=2.0, label=f"Formal {100 * formal:.1f}%")
    ax_f.axhline(prevalence, color=ORANGE, linewidth=1.5, linestyle="--", label=f"Prevalence {100 * prevalence:.1f}%")
    ax_f.set_xlim(-0.35, 1.35)
    ax_f.set_xticks([0, 1], ["Unconditional", "Prior-preserving"])
    _percent_axis(ax_f, 0.7)
    ax_f.legend(frameon=False, fontsize=7.2, loc="lower right")
    _panel(ax_f, "F", "Patient–reference falsification beyond coarse priors")

    fig.suptitle("Trustworthy SOZ-reference audit extensions: admission, construct and falsification", fontsize=15, fontweight="bold", color="#17212B", y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.965), h_pad=2.5, w_pad=2.0)
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
