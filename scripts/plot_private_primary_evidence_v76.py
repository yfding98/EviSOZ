#!/usr/bin/env python3
"""Create the private-centered primary evidence figure for submission v76.

The figure is a read-only composition of frozen v60, v71, and v74 artifacts.
It performs no model fitting, selection, calibration, aggregation, or new
endpoint computation beyond deterministic plotting summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V60 = ROOT / "outputs/trustworthy_soz_raw200_shallow_baseline_audit_v60_20260816/result.json"
DEFAULT_V71 = ROOT / "outputs/trustworthy_soz_dual_model_private_construct_repeatability_v71_20260816/result.json"
DEFAULT_V74 = ROOT / "outputs/trustworthy_soz_private_event_reference_alignment_v74_20260816/result.json"
DEFAULT_PDF = ROOT / "figures/trustworthy_soz_private_primary_evidence_v76_20260816.pdf"
DEFAULT_PNG = ROOT / "figures/trustworthy_soz_private_primary_evidence_v76_20260816.png"
COLORS = {
    "v29": "#245A88",
    "raw200": "#D17A2B",
    "exact": "#2E7D64",
    "neighbor": "#E2A14A",
    "far": "#B54444",
    "reference": "#4B8B78",
    "null": "#B9BEC7",
    "ink": "#172330",
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _private_patient_equal(
    model: Mapping[str, object],
) -> tuple[list[float], list[list[float]]]:
    values = [
        float(model["patient_equal_event_macro"]["strict"]),
        float(model["official_N2"]["patient_equal"]),
        float(model["patient_equal_event_macro"]["relaxed"]),
        float(model["patient_equal_event_macro"]["laterality_agreement"]),
    ]
    intervals = [
        [float(value) for value in model["patient_cluster_bootstrap_ci95"]["strict"]["patient_equal_ci95"]],
        [float(value) for value in model["official_N2"]["patient_cluster_bootstrap_ci95"]],
        [float(value) for value in model["patient_cluster_bootstrap_ci95"]["relaxed"]["patient_equal_ci95"]],
        [float(value) for value in model["patient_cluster_bootstrap_ci95"]["laterality_agreement"]["patient_equal_ci95"]],
    ]
    return values, intervals


def _asymmetric_error(values: Sequence[float], intervals: Sequence[Sequence[float]]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    low = np.asarray([interval[0] for interval in intervals], dtype=np.float64)
    high = np.asarray([interval[1] for interval in intervals], dtype=np.float64)
    return np.vstack((values_array - low, high - values_array))


def plot(
    *,
    v60: Mapping[str, object],
    v71: Mapping[str, object],
    v74: Mapping[str, object],
    pdf: Path,
    png: Path,
) -> None:
    if v60.get("schema_version") != "trustworthy_soz_raw200_shallow_baseline_audit_v60":
        raise ValueError("unexpected v60 schema")
    if v71.get("schema_version") != "trustworthy_soz_dual_model_private_construct_repeatability_v71":
        raise ValueError("unexpected v71 schema")
    if v74.get("schema_version") != "trustworthy_soz_private_event_reference_alignment_v74":
        raise ValueError("unexpected v74 schema")

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
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.5), constrained_layout=True)

    # A: private patient-equal agreement with cluster bootstrap intervals.
    ax = axes[0, 0]
    labels = ("Strict", "N2", "N4", "Laterality")
    x = np.arange(len(labels))
    width = 0.34
    for offset, model_name, label in (
        (-width / 2, "v29", "Frozen v29"),
        (width / 2, "raw200_shallow", "Raw200"),
    ):
        color_key = "v29" if model_name == "v29" else "raw200"
        values, intervals = _private_patient_equal(v60["private"][model_name])
        bars = ax.bar(x + offset, values, width, color=COLORS[color_key], label=label)
        ax.errorbar(
            x + offset,
            values,
            yerr=_asymmetric_error(values, intervals),
            fmt="none",
            color=COLORS["ink"],
            capsize=3,
            linewidth=1.0,
        )
        ax.bar_label(bars, labels=[f"{100 * value:.1f}" for value in values], padding=2, fontsize=7.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Patient-equal agreement")
    ax.set_title("A  Private transport is the primary result", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # B: exact, neighbor-only, and far error spectrum.
    ax = axes[0, 1]
    model_labels = ("Frozen v29", "Raw200")
    counts = [
        v60["private"][model]["endpoint_counts"]
        for model in ("v29", "raw200_shallow")
    ]
    exact = np.asarray([float(row["strict_exact"]) for row in counts])
    neighbor = np.asarray([float(row["neighbor_only"]) for row in counts])
    far = np.asarray([float(row["far"]) for row in counts])
    x = np.arange(2)
    ax.bar(x, exact, color=COLORS["exact"], label="Exact")
    ax.bar(x, neighbor, bottom=exact, color=COLORS["neighbor"], label="Neighbor-only")
    ax.bar(x, far, bottom=exact + neighbor, color=COLORS["far"], label="Far")
    for index in range(2):
        segments = ((exact[index], exact[index] / 2), (neighbor[index], exact[index] + neighbor[index] / 2), (far[index], exact[index] + neighbor[index] + far[index] / 2))
        for value, y in segments:
            if value >= 5:
                ax.text(index, y, f"{int(value)}", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xticks(x, model_labels)
    ax.set_ylim(0, 56)
    ax.set_ylabel("Private event count")
    ax.set_title("B  Exact-to-far error burden remains visible", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncols=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)

    # C: complete-set versus documented-singleton sensitivity.
    ax = axes[1, 0]
    construct = v71["reference_construct"]["models"]
    labels = ("Complete documented set", "Uniform documented-singleton")
    x = np.arange(2)
    for offset, model in ((-width / 2, "v29"), (width / 2, "raw200")):
        summary = construct[model]["summary"]
        keys = ("original_set_strict", "documented_singleton_uniform_top1")
        values = [float(summary[key]["patient_equal"]) for key in keys]
        intervals = [summary[key]["patient_cluster_bootstrap_ci95"] for key in keys]
        bars = ax.bar(x + offset, values, width, color=COLORS[model], label="Frozen v29" if model == "v29" else "Raw200")
        ax.errorbar(
            x + offset,
            values,
            yerr=_asymmetric_error(values, intervals),
            fmt="none",
            color=COLORS["ink"],
            capsize=3,
            linewidth=1.0,
        )
        ax.bar_label(bars, labels=[f"{100 * value:.1f}" for value in values], padding=2, fontsize=7.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Private patient-equal functional")
    ax.set_title("C  Multi-positive target construction changes agreement", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # D: clinician-reference stability and within-patient event-pairing null.
    ax = axes[1, 1]
    reference = v74["reference_stability"]["reference"]
    reference_keys = ("reference_exact", "reference_any_overlap", "reference_jaccard")
    reference_labels = ("Reference\nexact set", "Reference\nany overlap", "Reference\nJaccard")
    ref_x = np.arange(3)
    ref_values = [float(reference[key]["patient_equal"]) for key in reference_keys]
    ref_intervals = [reference[key]["patient_bootstrap_ci95"] for key in reference_keys]
    bars = ax.bar(ref_x, ref_values, width=0.62, color=COLORS["reference"], label="Clinician-reference stability")
    ax.errorbar(
        ref_x,
        ref_values,
        yerr=_asymmetric_error(ref_values, ref_intervals),
        fmt="none",
        color=COLORS["ink"],
        capsize=3,
        linewidth=1.0,
    )
    ax.bar_label(bars, labels=[f"{100 * value:.1f}" for value in ref_values], padding=2, fontsize=7.5)

    model_x = np.asarray([4.0, 5.0])
    formal = []
    null_mean = []
    null_interval = []
    for model in ("v29", "raw200"):
        alignment = v74["event_specific_prediction_reference_alignment"][model]
        formal.append(float(alignment["formal_event_pairing"]["multi_event_only_patient_equal"]["strict"]))
        null = alignment["within_patient_event_permutation_null"]["multi_event_only_patient_equal"]["strict"]
        null_mean.append(float(null["mean"]))
        null_interval.append([float(null["quantile_025_50_975"][0]), float(null["quantile_025_50_975"][2])])
    ax.bar(model_x, null_mean, width=0.62, color=COLORS["null"], label="Within-patient event-pairing null")
    ax.errorbar(
        model_x,
        null_mean,
        yerr=_asymmetric_error(null_mean, null_interval),
        fmt="none",
        color=COLORS["ink"],
        capsize=3,
        linewidth=1.0,
    )
    ax.scatter(model_x, formal, s=52, color=[COLORS["v29"], COLORS["raw200"]], zorder=3, label="Formal event pairing")
    for position, value, color in zip(model_x, formal, (COLORS["v29"], COLORS["raw200"]), strict=True):
        ax.text(position, value + 0.055, f"{100 * value:.1f}", ha="center", fontsize=7.5, color=color)
    ax.axvline(3.25, color="#8A8F96", linestyle="--", linewidth=1)
    ax.set_xticks([*ref_x, *model_x], [*reference_labels, "v29\nstrict", "Raw200\nstrict"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Patient-equal value")
    ax.set_title("D  Stable references do not imply event-specific tracking", loc="left", fontweight="bold")
    ax.text(
        3.38,
        0.20,
        "Gray bars: within-patient null\nColored dots: formal pairing",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#4B5560",
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Private post-open transport: performance, errors, target construction, and longitudinal meaning",
        fontsize=13,
        fontweight="bold",
        color=COLORS["ink"],
    )
    for output in (pdf, png):
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--v60", type=Path, default=DEFAULT_V60)
    parser.add_argument("--v71", type=Path, default=DEFAULT_V71)
    parser.add_argument("--v74", type=Path, default=DEFAULT_V74)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot(
        v60=_load(args.v60),
        v71=_load(args.v71),
        v74=_load(args.v74),
        pdf=args.pdf,
        png=args.png,
    )
    print(args.pdf)
    print(args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
