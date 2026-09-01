#!/usr/bin/env python3
"""Plot public-to-private v29 margin risk transport audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT / "outputs/trustworthy_soz_v29_margin_transport_v42_20260816/result.json"
)
DEFAULT_PNG = ROOT / "figures/trustworthy_soz_v29_margin_transport_v42_20260816.png"
DEFAULT_PDF = ROOT / "figures/trustworthy_soz_v29_margin_transport_v42_20260816.pdf"


def _series(policies, cohort_key: str, metric: str):
    x = []
    y = []
    low = []
    high = []
    coverage_key = "public_coverage" if cohort_key.startswith("public") else "private_coverage"
    for policy in policies:
        summary = policy[cohort_key]["metrics"][metric]
        x.append(100.0 * policy[coverage_key])
        y.append(100.0 * summary["event_micro"])
        low.append(100.0 * summary["cluster_event_micro_ci95"][0])
        high.append(100.0 * summary["cluster_event_micro_ci95"][1])
    return np.asarray(x), np.asarray(y), np.asarray(low), np.asarray(high)


def _errorbar(axis, values, label, color, marker):
    x, y, low, high = values
    order = np.argsort(x)
    axis.errorbar(
        x[order],
        y[order],
        yerr=np.vstack((y[order] - low[order], high[order] - y[order])),
        color=color,
        marker=marker,
        linewidth=1.7,
        capsize=2.5,
        label=label,
    )


def plot(result_path: Path, png: Path, pdf: Path) -> None:
    result = json.loads(result_path.resolve(strict=True).read_text(encoding="utf-8"))
    policies = result["policies"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)

    for axis, metric, title in (
        (axes[0], "strict", "A  Strict Top-1"),
        (axes[1], "neighborhood4", "B  Neighborhood-4"),
    ):
        _errorbar(
            axis,
            _series(policies, "public_accepted", metric),
            "Public development",
            "#245B9E",
            "o",
        )
        _errorbar(
            axis,
            _series(policies, "private_accepted", metric),
            "Private transport",
            "#C86834",
            "s",
        )
        axis.set_xlim(35, 103)
        axis.set_ylim(15 if metric == "strict" else 35, 100)
        axis.set_xlabel("Actual coverage (%)")
        axis.set_ylabel("Agreement among displayed cases (%)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, loc="lower right")

    axis = axes[2]
    private_coverage = np.asarray([100.0 * policy["private_coverage"] for policy in policies])
    contra = np.asarray(
        [
            100.0
            * policy["private_accepted"]["metrics"]["contralateral_far"]["event_micro"]
            for policy in policies
        ]
    )
    spread = np.asarray(
        [
            100.0
            * policy["private_accepted"]["metrics"]["known_spread_top1"]["event_micro"]
            for policy in policies
        ]
    )
    order = np.argsort(private_coverage)
    axis.plot(
        private_coverage[order],
        contra[order],
        marker="o",
        color="#A23E48",
        label="Contralateral-far Top-1",
    )
    axis.plot(
        private_coverage[order],
        spread[order],
        marker="s",
        color="#7B6BA8",
        label="Known-spread Top-1",
    )
    axis.set_xlim(35, 103)
    axis.set_ylim(0, max(30, float(contra.max()) + 5))
    axis.set_xlabel("Actual private coverage (%)")
    axis.set_ylabel("Observed private error rate (%)")
    axis.set_title("C  Safety-relevant errors")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, loc="upper right")

    figure.text(
        0.5,
        -0.03,
        "Thresholds use fixed public score-coverages and are applied unchanged to private scores. "
        "Both cohorts are historically opened; curves are descriptive, not calibrated risk guarantees.",
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
    )
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, bbox_inches="tight", dpi=300)
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot(args.result, args.png, args.pdf)
    print(args.png)
    print(args.pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
