#!/usr/bin/env python3
"""Render the final qualification-aware trustworthy-SOZ method architecture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "figures/trustworthy_soz_final_architecture_v62_20260816.pdf"
DEFAULT_PNG = ROOT / "figures/trustworthy_soz_final_architecture_v62_20260816.png"

NAVY = "#17365D"
BLUE = "#3E7CB1"
PALE_BLUE = "#EAF2F8"
GREEN = "#2A7F62"
PALE_GREEN = "#E8F5EF"
ORANGE = "#C76D2B"
PALE_ORANGE = "#FFF1E6"
RED = "#A63D40"
PALE_RED = "#FBEAEC"
GRAY = "#5E6872"
PALE_GRAY = "#F1F3F5"
INK = "#15202B"


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 9.5,
    linewidth: float = 1.4,
    style: str = "round,pad=0.018,rounding_size=0.018",
) -> FancyBboxPatch:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=style,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        transform=ax.transAxes,
        linespacing=1.15,
    )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    dashed: bool = False,
    width: float = 1.5,
    mutation: float = 11,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=width,
        color=color,
        linestyle="--" if dashed else "-",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(arrow)


def render(pdf: Path, png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(17.5, 8.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.55, 1.0, 1.05), wspace=0.14)
    ax_flow, ax_gate, ax_eval = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in (ax_flow, ax_gate, ax_eval):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax_flow.text(
        0.0,
        0.985,
        "A  Fail-closed localization and reporting flow",
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=NAVY,
        transform=ax_flow.transAxes,
    )
    x, w, h = 0.20, 0.60, 0.075
    y_rows = (0.875, 0.745, 0.615, 0.485, 0.355, 0.225, 0.095)
    labels = (
        "Standard-19 CAR EEG\n200 Hz · [19 × 12,000] · [−12,+48) s",
        "Input contract + technical rejection\nchannels · reference · units · anchor · QC",
        "Frozen official LaBraM block-9\n0 trainable foundation parameters",
        "H carrier  +  D carrier\nfixed low-capacity adaptation heads",
        "Fixed probability fusion (0.5 / 0.5)\nC18 set-valued candidate ranker",
        "Scalp-electrode SOZ-reference candidates\nstrict Top-1 primary · ranked list for review",
        "Typed fact ledger → deterministic report\nwaveform context · optional constrained language layer",
    )
    faces = (
        PALE_BLUE,
        PALE_ORANGE,
        PALE_BLUE,
        PALE_GREEN,
        PALE_GREEN,
        PALE_GREEN,
        PALE_BLUE,
    )
    edges = (BLUE, ORANGE, BLUE, GREEN, GREEN, GREEN, BLUE)
    for index, (y, label, face, edge) in enumerate(zip(y_rows, labels, faces, edges)):
        _box(
            ax_flow,
            (x, y),
            w,
            h,
            label,
            face=face,
            edge=edge,
            fontsize=8.5 if index == len(labels) - 1 else 9.1,
        )
    for upper, lower in zip(y_rows[:-1], y_rows[1:]):
        _arrow(ax_flow, (0.50, upper), (0.50, lower + h), color=NAVY)
    _box(
        ax_flow,
        (0.825, 0.735),
        0.15,
        0.095,
        "REJECT\nno localization\nno report",
        face=PALE_RED,
        edge=RED,
        fontsize=8.3,
    )
    _arrow(ax_flow, (x + w, 0.782), (0.825, 0.782), color=RED)
    ax_flow.text(
        0.52,
        0.707,
        "valid only",
        fontsize=7.8,
        color=GRAY,
        transform=ax_flow.transAxes,
    )

    ax_gate.text(
        0.0,
        0.985,
        "B  Evidence qualification controller",
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=NAVY,
        transform=ax_gate.transAxes,
    )
    _box(
        ax_gate,
        (0.08, 0.855),
        0.84,
        0.075,
        "Only qualified outputs may cross\nthe localization/report firewall",
        face=PALE_ORANGE,
        edge=ORANGE,
        fontsize=9.2,
    )
    gate_rows = (
        (0.700, "Morphology M", "FAIL_NATIVE", "blocked", PALE_RED, RED),
        (0.565, "Ictal involvement I", "FAIL_NATIVE", "blocked", PALE_RED, RED),
        (0.430, "Learned evolution V_F", "NO_GO", "blocked", PALE_RED, RED),
        (0.295, "Direct waveform change V", "DESCRIPTION_\nONLY", "report context", PALE_ORANGE, ORANGE),
        (0.160, "Uncertainty proxies", "NOT_\nQUALIFIED", "no clinical abstention", PALE_GRAY, GRAY),
    )
    for y, name, status, route, face, edge in gate_rows:
        _box(ax_gate, (0.07, y), 0.50, 0.085, name, face=face, edge=edge, fontsize=8.9)
        _box(ax_gate, (0.62, y), 0.32, 0.085, f"{status}\n{route}", face=face, edge=edge, fontsize=7.25)
        _arrow(ax_gate, (0.57, y + 0.0425), (0.62, y + 0.0425), color=edge, mutation=9)
    ax_gate.text(
        0.08,
        0.07,
        "Negative qualification is a released result, not a hidden failed arm.",
        ha="left",
        va="center",
        fontsize=8.2,
        color=GRAY,
        transform=ax_gate.transAxes,
    )

    ax_eval.text(
        0.0,
        0.985,
        "C  Evaluation and claim governance",
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=NAVY,
        transform=ax_eval.transAxes,
    )
    _box(
        ax_eval,
        (0.05, 0.805),
        0.90,
        0.12,
        "PUBLIC DEVELOPMENT\n102 patients · 1,145 events\npatient-OOF · adaptively consumed\npossible foundation-pretraining exposure",
        face=PALE_BLUE,
        edge=BLUE,
        fontsize=8.8,
    )
    _arrow(ax_eval, (0.50, 0.805), (0.50, 0.720), color=BLUE, dashed=True)
    _box(
        ax_eval,
        (0.05, 0.590),
        0.90,
        0.13,
        "PRIVATE TRANSPORT (emphasized)\n88 reference-isolated predictions\n51 evaluable events · 23 patient clusters\nzero adaptation · post-open descriptive audit",
        face=PALE_GREEN,
        edge=GREEN,
        fontsize=8.8,
    )
    _arrow(ax_eval, (0.50, 0.590), (0.50, 0.505), color=GREEN, dashed=True)
    _box(
        ax_eval,
        (0.05, 0.365),
        0.90,
        0.14,
        "Endpoint hierarchy\nPRIMARY: strict physical-electrode Top-1\nSECONDARY: official N2 / N4, laterality, rank burden\nSpread never becomes a positive",
        face=PALE_ORANGE,
        edge=ORANGE,
        fontsize=8.8,
    )
    _arrow(ax_eval, (0.50, 0.365), (0.50, 0.280), color=ORANGE, dashed=True)
    _box(
        ax_eval,
        (0.05, 0.095),
        0.90,
        0.185,
        "Permitted claim\nDevelopment-stage, qualification-aware candidate ranking\nunder heterogeneous set-valued references\n\nNot cortical SOZ/EZ · not surgical target\nnot independent external or clinical validation",
        face=PALE_GRAY,
        edge=GRAY,
        fontsize=8.7,
    )

    fig.suptitle(
        "Qualification-aware, fail-closed scalp-EEG SOZ-reference candidate ranking",
        x=0.5,
        y=0.995,
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.012,
        "The reporting layer cannot alter candidate scores or introduce new patient-level facts.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=GRAY,
    )
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
