#!/usr/bin/env python3
"""Audit distance-error enrichment under the already frozen margin gate.

The threshold is loaded from the immutable v21.1 selective result.  This audit
does not choose a threshold, change a prediction, or expose hidden rankings in
clinical reports.  Hidden abstained rankings are read only for benchmark error
analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_trustworthy_soz_ranking_distance_v22_6 import (  # noqa: E402
    DEFAULT_PRIVATE,
    DEFAULT_PRIVATE_SIGNAL,
    DEFAULT_PRIVATE_TARGET,
    DEFAULT_PUBLIC,
    NUMERIC_METRICS,
    _private_rows,
    _public_rows,
    _rank_burden,
)
from scripts.audit_trustworthy_soz_selective_v21_1 import (  # noqa: E402
    _probability_and_margin,
)


DEFAULT_SELECTIVE = ROOT / "outputs/trustworthy_soz_selective_v21_1_20260815/result.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_selective_distance_v22_7_20260815/result.json"
)
SCHEMA = "trustworthy_soz_selective_distance_audit_v22_7"
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260815

GAP_METRICS = (
    "strict_error",
    "far",
    "contralateral_far",
    "known_spread_top1",
    "rank_gt5",
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _augment(row: Mapping[str, object], *, margin: float, threshold: float) -> dict[str, object]:
    result = dict(row)
    result["margin"] = margin
    result["accepted"] = margin >= threshold
    result["strict_error"] = 1.0 - float(row["strict"])
    result["rank_gt5"] = float(int(row["first_positive_rank"]) > 5)
    return result


def _rows_with_frozen_route(
    public_directory: Path,
    private_directory: Path,
    private_signal_manifest: Path,
    private_target: Path,
    threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    public_rows = _public_rows(public_directory)
    public_tensors = load_file(
        str((public_directory / "oof_predictions.safetensors").resolve(strict=True))
    )
    candidate_mask = public_tensors["config.candidate_mask"].bool()
    _, public_margin = _probability_and_margin(
        public_tensors["oof.frozen_labram_only"].float(),
        candidate_mask,
        values_are_logits=True,
    )
    if len(public_rows) != public_margin.numel():
        raise ValueError("public row/margin roster drifted")
    public_augmented = [
        _augment(row, margin=float(public_margin[index]), threshold=threshold)
        for index, row in enumerate(public_rows)
    ]

    private_rows = _private_rows(
        private_directory, private_signal_manifest, private_target
    )
    private_tensors = load_file(
        str((private_directory / "predictions.safetensors").resolve(strict=True))
    )
    _, private_margin = _probability_and_margin(
        private_tensors["private_h_only_probability"].float(),
        candidate_mask,
        values_are_logits=False,
    )
    signal = _read_json(private_signal_manifest)
    events = signal.get("events")
    if not isinstance(events, list) or len(events) != private_margin.numel():
        raise ValueError("private signal/margin roster drifted")
    margin_by_event = {
        str(event["event_id"]): float(private_margin[index])
        for index, event in enumerate(events)
    }
    private_augmented = []
    for row in private_rows:
        event_id = str(row["unit_id"])
        if event_id not in margin_by_event:
            raise ValueError(f"private primary event has no margin: {event_id}")
        private_augmented.append(
            _augment(row, margin=margin_by_event[event_id], threshold=threshold)
        )
    return public_augmented, private_augmented


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _subset(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("selective subset cannot be empty")
    categories = Counter(
        "exact" if row["strict"] else "neighbor_only" if row["neighbor_only"] else "far"
        for row in rows
    )
    far = Counter(
        str(row["far_subtype"]) for row in rows if row["far_subtype"] is not None
    )
    metrics = {name: _mean(rows, name) for name in NUMERIC_METRICS}
    metrics.update(
        {
            "strict_error": _mean(rows, "strict_error"),
            "rank_gt5": _mean(rows, "rank_gt5"),
            "n": len(rows),
            "patient_count": len({str(row["patient_id"]) for row in rows}),
        }
    )
    return {
        "metrics": metrics,
        "category_counts": dict(sorted(categories.items())),
        "far_subtype_counts": dict(sorted(far.items())),
        "rank_burden": _rank_burden(rows),
    }


def _gap(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    accepted = [row for row in rows if bool(row["accepted"])]
    abstained = [row for row in rows if not bool(row["accepted"])]
    if not accepted or not abstained:
        raise ValueError("gap requires accepted and abstained rows")
    return _mean(abstained, metric) - _mean(accepted, metric)


def _bootstrap_gaps(
    rows: Sequence[Mapping[str, object]], *, groups: Sequence[str], seed: int
) -> dict[str, object]:
    if len(rows) != len(groups):
        raise ValueError("rows/groups length differs")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    unique = sorted(by_group)
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in GAP_METRICS}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        indices = [index for group in sampled_groups for index in by_group[str(group)]]
        sampled = [rows[index] for index in indices]
        if not any(bool(row["accepted"]) for row in sampled) or not any(
            not bool(row["accepted"]) for row in sampled
        ):
            continue
        for metric in GAP_METRICS:
            values[metric].append(_gap(sampled, metric))
    result = {}
    for metric, samples in values.items():
        if not samples:
            raise RuntimeError(f"bootstrap produced no valid gap for {metric}")
        interval = np.quantile(samples, [0.025, 0.975])
        result[metric] = {
            "abstained_minus_accepted_rate": _gap(rows, metric),
            "ci95": [float(interval[0]), float(interval[1])],
            "bootstrap_interval_excludes_zero": bool(
                float(interval[0]) > 0.0 or float(interval[1]) < 0.0
            ),
            "confirmatory_significance_claim_allowed": False,
            "valid_replicates": len(samples),
        }
    return result


def _cohort(
    rows: Sequence[Mapping[str, object]], *, groups: Sequence[str], seed: int
) -> dict[str, object]:
    accepted = [row for row in rows if bool(row["accepted"])]
    abstained = [row for row in rows if not bool(row["accepted"])]
    return {
        "total_count": len(rows),
        "accepted_count": len(accepted),
        "abstained_count": len(abstained),
        "coverage": len(accepted) / len(rows),
        "accepted_displayed_candidates": _subset(accepted),
        "abstained_hidden_ranking_audit": _subset(abstained),
        "risk_enrichment": _bootstrap_gaps(rows, groups=groups, seed=seed),
    }


def audit(
    public_directory: Path,
    private_directory: Path,
    private_signal_manifest: Path,
    private_target: Path,
    selective_result: Path,
) -> dict[str, object]:
    selective = _read_json(selective_result)
    if selective.get("schema_version") != "trustworthy_soz_selective_result_v21_1":
        raise ValueError("frozen selective result schema drifted")
    selected = selective.get("selected_public_operating_point")
    if not isinstance(selected, Mapping):
        raise ValueError("frozen selective operating point is unavailable")
    threshold = float(selected["threshold"])
    public_rows, private_rows = _rows_with_frozen_route(
        public_directory,
        private_directory,
        private_signal_manifest,
        private_target,
        threshold,
    )
    public = _cohort(
        public_rows,
        groups=[str(row["patient_id"]) for row in public_rows],
        seed=BOOTSTRAP_SEED,
    )
    private = _cohort(
        private_rows,
        groups=[str(row["patient_id"]) for row in private_rows],
        seed=BOOTSTRAP_SEED + 1,
    )
    if (public["accepted_count"], private["accepted_count"]) != (81, 43):
        raise RuntimeError("frozen selective accepted counts drifted")
    return {
        "schema_version": SCHEMA,
        "status": "completed_frozen_selective_distance_error_audit",
        "arm": "portable_frozen_labram_h_only_v16",
        "frozen_threshold": threshold,
        "public_patient_level": public,
        "private_event_level_patient_clustered": private,
        "scientific_boundary": {
            "threshold_reselected": False,
            "private_used_for_threshold_selection": False,
            "hidden_abstained_ranking_displayed_in_clinical_report": False,
            "risk_gap_is_calibrated_or_conformal_guarantee": False,
            "public_is_repeatedly_used_development": True,
            "private_is_post_open_descriptive": True,
            "public_and_private_units_match": False,
            "distance_subtype_analysis_is_post_hoc": True,
            "multiple_metric_confirmatory_test_performed": False,
            "bootstrap_interval_exclusion_is_confirmatory_significance": False,
        },
        "access_receipt": {
            "raw_eeg_loaded": False,
            "model_weights_loaded": False,
            "existing_frozen_predictions_loaded": True,
            "existing_public_and_opened_private_targets_loaded_for_evaluation": True,
            "training_performed": False,
            "model_threshold_or_report_selection_performed": False,
            "candidate_or_report_changed": False,
            "llm_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument(
        "--private-signal-manifest", type=Path, default=DEFAULT_PRIVATE_SIGNAL
    )
    parser.add_argument("--private-target", type=Path, default=DEFAULT_PRIVATE_TARGET)
    parser.add_argument("--selective-result", type=Path, default=DEFAULT_SELECTIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(
        args.public,
        args.private,
        args.private_signal_manifest,
        args.private_target,
        args.selective_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "public": result["public_patient_level"],
                "private": result["private_event_level_patient_clustered"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
