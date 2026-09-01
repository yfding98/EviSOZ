#!/usr/bin/env python3
"""Summarize locked VEPiSet IED/SOZ-like results without test-time selection.

This report intentionally treats VEPiSet labels as IED spatial-distribution
labels rather than clinical SOZ ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


MetricDict = Mapping[str, Any]


RUNS: Tuple[Tuple[str, str, str], ...] = (
    (
        "raw_full6_weighted",
        "Full 6-class v2, weighted-F1 validation selection",
        "outputs/vepiset_ied_v2_full6_seed2026_weighted_noamp20/test_metrics.json",
    ),
    (
        "calibrated_single_bias",
        "Validation-only single Non-IED bias calibration",
        "outputs/vepiset_ied_v2_full6_seed2026_logitadj025_weighted_noamp20_calibrated_macro_valacc90/calibrated_metrics.json",
    ),
    (
        "best_macro_acc80",
        "Validation-only smoothing plus single Non-IED bias calibration",
        "outputs/vepiset_ied_v2_full6_seed2026_logitadj025_weighted_noamp20_smooth_macro_valacc905_noniedbias_macro_valacc90/calibrated_metrics.json",
    ),
    (
        "joint_smoothing_audit",
        "Validation-only joint smoothing/bias audit, not selected as main result",
        "outputs/vepiset_ied_v2_full6_seed2026_logitadj025_weighted_noamp20_smooth_joint_macro_valacc90/smoothed_metrics.json",
    ),
    (
        "state_conditioned_audit",
        "Wake/sleep-conditioned v2 audit, validation-only single-bias calibrated",
        "outputs/vepiset_ied_v2_full6_seed2026_logitadj025_weighted_state_noamp20_calibrated_macro_valacc90/calibrated_metrics.json",
    ),
    (
        "state_bias_weighted_audit",
        "Wake/sleep-specific Non-IED bias audit, weighted-F1 validation selection",
        "outputs/vepiset_ied_v2_full6_seed2026_logitadj025_weighted_noamp20_statebias_weighted/calibrated_metrics.json",
    ),
    (
        "smooth_state_bias_fine_audit",
        "Smoothing plus wake/sleep-specific small-bias audit",
        "outputs/vepiset_ied_v2_full6_seed2026_logitadj025_weighted_noamp20_smooth_macro_valacc905_statebias_fine_macro_valacc909/calibrated_metrics.json",
    ),
    (
        "feature_ensemble_near_miss",
        "v2 plus morphology-feature ensemble, macro selector, below 80% accuracy",
        "outputs/vepiset_ied_v2_full6_seed2026_smooth_feature_ensemble_cap3000_macro_valacc90/ensemble_metrics.json",
    ),
    (
        "feature_ensemble_acc80",
        "v2 plus morphology-feature ensemble, accuracy-constrained fine grid",
        "outputs/vepiset_ied_v2_full6_seed2026_smooth_feature_ensemble_cap3000_fine_macro_valacc909/ensemble_metrics.json",
    ),
    (
        "feature_gate_acc80",
        "v2 spatial probabilities plus train-only morphology binary gate",
        "outputs/vepiset_ied_v2_full6_seed2026_smooth_feature_gate_cap3000_macro_valacc91/feature_gate_metrics.json",
    ),
    (
        "positive_spatial_gate_acc80",
        "v2 gate/spatial probabilities plus train-only binary and positive-spatial feature gates",
        "outputs/vepiset_ied_v2_full6_seed2026_positive_spatial_gate_cap3000_macro_valacc91/positive_spatial_gate_metrics.json",
    ),
)

CLASS_ORDER: Tuple[str, ...] = (
    "Non-IED",
    "Generalized-IED",
    "Frontal-IED",
    "Temporal-IED",
    "Centro-Parietal-IED",
    "Occipital-IED",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_metrics(payload: MetricDict) -> MetricDict:
    return payload.get("test_metrics", payload)


def val_metrics(payload: MetricDict) -> MetricDict | None:
    value = payload.get("val_metrics")
    return value if isinstance(value, Mapping) else None


def compact_metrics(metrics: MetricDict) -> Dict[str, float]:
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    return {key: float(metrics[key]) for key in keys if key in metrics}


def compact_per_class(metrics: MetricDict) -> Dict[str, float]:
    per_class = metrics.get("per_class", {})
    return {
        name: float(per_class.get(name, {}).get("f1", 0.0))
        for name in CLASS_ORDER
    }


def split_patients(split_summary: MetricDict) -> Dict[str, set[str]]:
    meta = split_summary.get("train_split_meta", {})
    patients = meta.get("patients", {})
    return {
        split: set(str(pid) for pid in patients.get(split, []))
        for split in ("train", "val", "test")
    }


def split_overlap(patients: Mapping[str, set[str]]) -> Dict[str, List[str]]:
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    return {
        f"{left}_{right}": sorted(patients.get(left, set()) & patients.get(right, set()))
        for left, right in pairs
    }


def split_counts(split_summary: MetricDict) -> Dict[str, Any]:
    return {
        split: split_summary.get(split, {})
        for split in ("train", "val", "test")
    }


def format_pct(value: float) -> str:
    return f"{value * 100.0:.2f}"


def render_markdown(report: MetricDict) -> str:
    lines: List[str] = []
    lines.append("# Locked VEPiSet IED / SOZ-like Report")
    lines.append("")
    lines.append("VEPiSet is used here as an IED detection and IED spatial-distribution benchmark. The spatial labels are a coarse SOZ-like proxy, not clinical SOZ ground truth.")
    lines.append("")
    lines.append("## Split Audit")
    lines.append("")
    split = report["split"]
    lines.append(f"- Train/val/test patients: {split['train']['patients']} / {split['val']['patients']} / {split['test']['patients']}")
    overlap_text = json.dumps(report["patient_overlap"], ensure_ascii=False)
    lines.append(f"- Patient overlaps: `{overlap_text}`")
    class_counts = json.dumps(split["test"]["classes"], ensure_ascii=False)
    lines.append(f"- Test samples: {split['test']['rows']}, classes: `{class_counts}`")
    lines.append("")
    lines.append("## Main Metrics")
    lines.append("")
    lines.append("| Run | Accuracy | Balanced Acc. | Macro-F1 | Weighted-F1 | Note |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for run in report["runs"]:
        m = run["test"]
        lines.append(
            "| {name} | {acc} | {bal} | {macro} | {weighted} | {note} |".format(
                name=run["name"],
                acc=format_pct(m.get("accuracy", 0.0)),
                bal=format_pct(m.get("balanced_accuracy", 0.0)),
                macro=format_pct(m.get("macro_f1", 0.0)),
                weighted=format_pct(m.get("weighted_f1", 0.0)),
                note=run["description"],
            )
        )
    lines.append("")
    lines.append("## Per-class F1")
    lines.append("")
    lines.append("| Run | Non-IED | Generalized | Frontal | Temporal | Centro-Parietal | Occipital |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in report["runs"]:
        pc = run["per_class_f1"]
        lines.append(
            "| {name} | {values} |".format(
                name=run["name"],
                values=" | ".join(format_pct(pc[class_name]) for class_name in CLASS_ORDER),
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The locked patient-wise run exceeds 80% accuracy, but this is strongly affected by the Non-IED majority class.")
    lines.append("- The best accuracy-constrained macro-F1 result is still weak on Temporal and Centro-Parietal IEDs.")
    lines.append("- The locked split is especially sparse for Centro-Parietal IED at patient level: train/val/test contain 1 / 1 / 2 patients with that label, so cross-patient generalization for this class is weak evidence rather than a stable SOZ-localization result.")
    lines.append("- These results should not be described as scalp-EEG SOZ SOTA; they are VEPiSet IED/SOZ-like proxy results.")
    lines.append("- Hyperparameters/calibration parameters listed here are validation-selected; test metrics are report-only.")
    lines.append("")
    return "\n".join(lines)


def build_report(repo_root: Path) -> Dict[str, Any]:
    split_path = repo_root / "outputs/vepiset_ied_v2_full6_seed2026_logitadj025_weighted_noamp20/split_summary.json"
    split_summary = load_json(split_path)
    patients = split_patients(split_summary)
    runs = []
    for key, description, relative_path in RUNS:
        path = repo_root / relative_path
        payload = load_json(path)
        tm = test_metrics(payload)
        vm = val_metrics(payload)
        runs.append(
            {
                "key": key,
                "name": key,
                "description": description,
                "path": str(path.relative_to(repo_root)),
                "validation": compact_metrics(vm) if vm is not None else {},
                "test": compact_metrics(tm),
                "per_class_f1": compact_per_class(tm),
                "selection": {
                    "selector": payload.get("selector"),
                    "min_val_accuracy": payload.get("min_val_accuracy"),
                    "min_val_weighted_f1": payload.get("min_val_weighted_f1"),
                    "non_ied_logit_bias": payload.get("non_ied_logit_bias"),
                    "params": payload.get("params"),
                },
            }
        )
    return {
        "dataset": "VEPiSet",
        "task": "IED classification / SOZ-like proxy",
        "split_source": str(split_path.relative_to(repo_root)),
        "split": split_counts(split_summary),
        "patient_overlap": split_overlap(patients),
        "runs": runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root containing outputs/")
    parser.add_argument("--output-json", default="outputs/vepiset_ied_locked_seed2026_report.json")
    parser.add_argument("--output-md", default="outputs/vepiset_ied_locked_seed2026_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    report = build_report(repo_root)
    output_json = repo_root / args.output_json
    output_md = repo_root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
