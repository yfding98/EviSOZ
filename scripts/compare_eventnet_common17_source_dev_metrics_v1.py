#!/usr/bin/env python3
"""Compare two complete common17 EventNet source-dev evaluations.

This is a metrics-only audit.  It neither imports torch nor reads EEG samples.
The comparison fails closed unless both receipts use the same complete
source-dev denominator, manifest, channel contract and threshold grid.

The fixed-threshold comparison is the primary model-to-model diagnostic.  A
second, explicitly labelled best-on-the-same-dev comparison is retained only
for navigation; it is not a held-out performance estimate.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PENDING = "CONTENT-ADDRESS-PENDING"
SCHEMA_VERSION = "eventnet_common17_source_dev_comparison_v1"
EXPECTED_STAGE = "source_dev_full_record_evaluation"
EXPECTED_THRESHOLD_STATUS = "source_dev_diagnostic_grid_not_source_eval"
EXPECTED_CHANNELS = (
    "FP1", "F3", "C3", "P3", "O1", "F7", "T7", "P7", "CZ",
    "FP2", "F4", "C4", "P4", "O2", "F8", "T8", "P8",
)
DEFAULT_BUDGETS_FA_PER_24H = (1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = PENDING
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def read_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"metrics JSON root must be an object: {path}")
    claimed = value.get("receipt_sha256")
    replay = deepcopy(value)
    replay["receipt_sha256"] = PENDING
    if not isinstance(claimed, str) or claimed != canonical_sha256(replay):
        raise ValueError(f"metrics receipt content address is invalid: {path}")
    return value


def finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def grid_by_threshold(metrics: Mapping[str, Any]) -> dict[float, Mapping[str, Any]]:
    raw = metrics.get("metric_grid")
    if not isinstance(raw, list) or not raw:
        raise ValueError("metrics receipt has no threshold grid")
    result: dict[float, Mapping[str, Any]] = {}
    previous = -math.inf
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping) or not isinstance(row.get("pooled"), Mapping):
            raise TypeError(f"threshold grid row {index} is malformed")
        threshold = finite_number(row.get("center_threshold"), name="center_threshold")
        if not 0.0 < threshold < 1.0 or threshold <= previous:
            raise ValueError("threshold grid must be unique and strictly increasing")
        result[threshold] = row
        previous = threshold
    return result


def selected_best(grid: Mapping[float, Mapping[str, Any]]) -> Mapping[str, Any]:
    def selection_key(row: Mapping[str, Any]) -> tuple[float, float]:
        pooled = row["pooled"]
        f1 = pooled.get("event_f1")
        fa24 = pooled.get("alarm_false_alarms_per_24h")
        return (
            -1.0 if f1 is None else finite_number(f1, name="event_f1"),
            -math.inf if fa24 is None else -finite_number(fa24, name="FA/24h"),
        )

    return max(grid.values(), key=selection_key)


def validate_evaluation(metrics: Mapping[str, Any], *, label: str) -> dict[float, Mapping[str, Any]]:
    if metrics.get("stage") != EXPECTED_STAGE:
        raise ValueError(f"{label}: evaluation stage drifted")
    if metrics.get("complete_source_dev_denominator") is not True:
        raise ValueError(f"{label}: source-dev denominator is incomplete")
    if metrics.get("threshold_selection_status") != EXPECTED_THRESHOLD_STATUS:
        raise ValueError(f"{label}: threshold-selection scope drifted")
    if tuple(metrics.get("common17_channel_order", ())) != EXPECTED_CHANNELS:
        raise ValueError(f"{label}: common17 channel order drifted")
    if metrics.get("FZ_or_PZ_model_axis_present") is not False:
        raise ValueError(f"{label}: FZ/PZ leaked into the model axes")
    grid = grid_by_threshold(metrics)
    replay_best = selected_best(grid)
    stored_best = metrics.get("best_source_dev_diagnostic_operating_point")
    if not isinstance(stored_best, Mapping):
        raise TypeError(f"{label}: stored best operating point is malformed")
    if finite_number(stored_best.get("center_threshold"), name="stored best threshold") != finite_number(
        replay_best.get("center_threshold"), name="replayed best threshold"
    ):
        raise ValueError(f"{label}: stored best threshold does not replay")
    return grid


def rate(metric: Mapping[str, Any], *path: str) -> float | None:
    value: object = metric
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    if value is None:
        return None
    return finite_number(value, name=".".join(path))


METRIC_PATHS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("event_sensitivity", ("event_sensitivity",), "higher_is_better"),
    ("event_precision", ("event_precision",), "higher_is_better"),
    ("event_f1", ("event_f1",), "higher_is_better"),
    (
        "false_alarms_per_24h",
        ("alarm_false_alarms_per_24h",),
        "lower_is_better",
    ),
    (
        "seizure_record_recall",
        ("seizure_record_recall", "rate"),
        "higher_is_better",
    ),
    (
        "onset_hit_at_1s",
        ("onset_absolute_hit_rate", "1s", "rate"),
        "higher_is_better",
    ),
    (
        "onset_hit_at_3s",
        ("onset_absolute_hit_rate", "3s", "rate"),
        "higher_is_better",
    ),
    (
        "onset_hit_at_5s",
        ("onset_absolute_hit_rate", "5s", "rate"),
        "higher_is_better",
    ),
    (
        "onset_hit_at_10s",
        ("onset_absolute_hit_rate", "10s", "rate"),
        "higher_is_better",
    ),
    (
        "absolute_onset_error_median_matched_only_seconds",
        ("onset_latency_seconds", "absolute_median_matched_only"),
        "lower_is_better_but_matched_only",
    ),
)


def metric_comparison(
    old_pooled: Mapping[str, Any], new_pooled: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path, direction in METRIC_PATHS:
        old_value = rate(old_pooled, *path)
        new_value = rate(new_pooled, *path)
        result[name] = {
            "old": old_value,
            "new": new_value,
            "delta_new_minus_old": (
                None if old_value is None or new_value is None else new_value - old_value
            ),
            "direction": direction,
        }
    for name in (
        "reference_event_count",
        "predicted_alarm_count",
        "matched_event_count",
        "false_alarm_count",
    ):
        old_value = int(old_pooled[name])
        new_value = int(new_pooled[name])
        result[name] = {
            "old": old_value,
            "new": new_value,
            "delta_new_minus_old": new_value - old_value,
            "direction": "descriptive_count",
        }
    return result


def constrained_point(
    grid: Mapping[float, Mapping[str, Any]], budget: float
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in grid.values()
        if row["pooled"].get("alarm_false_alarms_per_24h") is not None
        and float(row["pooled"]["alarm_false_alarms_per_24h"]) <= budget
    ]
    if not eligible:
        return None
    chosen = max(
        eligible,
        key=lambda row: (
            -1.0 if row["pooled"].get("event_sensitivity") is None else row["pooled"]["event_sensitivity"],
            -1.0 if row["pooled"].get("event_f1") is None else row["pooled"]["event_f1"],
            -row["center_threshold"],
        ),
    )
    pooled = chosen["pooled"]
    return {
        "threshold": chosen["center_threshold"],
        "event_sensitivity": pooled["event_sensitivity"],
        "event_precision": pooled["event_precision"],
        "event_f1": pooled["event_f1"],
        "false_alarms_per_24h": pooled["alarm_false_alarms_per_24h"],
        "onset_hit_at_10s": pooled["onset_absolute_hit_rate"]["10s"]["rate"],
    }


def compare(
    *,
    old_path: Path,
    new_path: Path,
    fixed_threshold: float,
    budgets_fa_per_24h: Sequence[float] = DEFAULT_BUDGETS_FA_PER_24H,
) -> dict[str, Any]:
    old = read_receipt(old_path)
    new = read_receipt(new_path)
    old_grid = validate_evaluation(old, label="old")
    new_grid = validate_evaluation(new, label="new")

    lineage_fields = (
        "manifest_receipt_sha256",
        "common17_channel_order",
        "FZ_or_PZ_model_axis_present",
        "recording_count",
        "reference_event_count",
        "recording_hours",
        "complete_source_dev_denominator",
        "threshold_selection_status",
    )
    lineage_equal = {field: old.get(field) == new.get(field) for field in lineage_fields}
    if not all(lineage_equal.values()):
        raise ValueError(f"evaluation denominator or lineage differs: {lineage_equal}")
    if tuple(old_grid) != tuple(new_grid):
        raise ValueError("old and new evaluations use different threshold grids")
    fixed = finite_number(fixed_threshold, name="fixed_threshold")
    if fixed not in old_grid:
        raise ValueError(f"fixed threshold {fixed:g} is absent from the shared grid")

    old_best = selected_best(old_grid)
    new_best = selected_best(new_grid)
    threshold_curve = []
    for threshold in old_grid:
        old_pooled = old_grid[threshold]["pooled"]
        new_pooled = new_grid[threshold]["pooled"]
        threshold_curve.append(
            {
                "threshold": threshold,
                "old": {
                    "event_sensitivity": old_pooled["event_sensitivity"],
                    "event_precision": old_pooled["event_precision"],
                    "event_f1": old_pooled["event_f1"],
                    "false_alarms_per_24h": old_pooled["alarm_false_alarms_per_24h"],
                    "onset_hit_at_10s": old_pooled["onset_absolute_hit_rate"]["10s"]["rate"],
                },
                "new": {
                    "event_sensitivity": new_pooled["event_sensitivity"],
                    "event_precision": new_pooled["event_precision"],
                    "event_f1": new_pooled["event_f1"],
                    "false_alarms_per_24h": new_pooled["alarm_false_alarms_per_24h"],
                    "onset_hit_at_10s": new_pooled["onset_absolute_hit_rate"]["10s"]["rate"],
                },
            }
        )

    budgets = sorted({finite_number(value, name="FA/24h budget") for value in budgets_fa_per_24h})
    if not budgets or budgets[0] <= 0:
        raise ValueError("FA/24h budgets must be positive")
    return content_address(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_same_complete_source_dev_contract_metrics_only",
            "interpretation_scope": {
                "primary_comparison": "same_source_dev_same_fixed_threshold_diagnostic",
                "best_grid_comparison": "optimistic_same_source_dev_navigation_not_held_out",
                "source_eval_or_external_test_used": False,
                "inference_performed": False,
                "EEG_samples_read": False,
                "paired_record_level_uncertainty_estimated": False,
            },
            "lineage_equal": lineage_equal,
            "shared_denominator": {
                "manifest_receipt_sha256": old["manifest_receipt_sha256"],
                "recording_count": old["recording_count"],
                "reference_event_count": old["reference_event_count"],
                "recording_hours": old["recording_hours"],
                "common17_channel_order": old["common17_channel_order"],
                "FZ_or_PZ_model_axis_present": False,
                "threshold_grid": list(old_grid),
            },
            "models": {
                "old": {
                    "metrics_path": str(old_path.resolve()),
                    "metrics_file_sha256": file_sha256(old_path.resolve()),
                    "metrics_receipt_sha256": old["receipt_sha256"],
                    "checkpoint_path": old["checkpoint_path"],
                    "checkpoint_file_sha256": old["checkpoint_file_sha256"],
                    "checkpoint_global_step": old["checkpoint_global_step"],
                },
                "new": {
                    "metrics_path": str(new_path.resolve()),
                    "metrics_file_sha256": file_sha256(new_path.resolve()),
                    "metrics_receipt_sha256": new["receipt_sha256"],
                    "checkpoint_path": new["checkpoint_path"],
                    "checkpoint_file_sha256": new["checkpoint_file_sha256"],
                    "checkpoint_global_step": new["checkpoint_global_step"],
                },
            },
            "fixed_threshold_comparison": {
                "threshold": fixed,
                "metrics": metric_comparison(
                    old_grid[fixed]["pooled"], new_grid[fixed]["pooled"]
                ),
            },
            "best_on_same_source_dev_comparison": {
                "old_threshold": old_best["center_threshold"],
                "new_threshold": new_best["center_threshold"],
                "metrics": metric_comparison(old_best["pooled"], new_best["pooled"]),
            },
            "fa24_budget_navigation": [
                {
                    "budget": budget,
                    "old": constrained_point(old_grid, budget),
                    "new": constrained_point(new_grid, budget),
                }
                for budget in budgets
            ],
            "threshold_curve": threshold_curve,
            "receipt_sha256": PENDING,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-metrics", type=Path, required=True)
    parser.add_argument("--new-metrics", type=Path, required=True)
    parser.add_argument("--fixed-threshold", type=float, default=0.02)
    parser.add_argument(
        "--fa24-budgets",
        default=",".join(f"{value:g}" for value in DEFAULT_BUDGETS_FA_PER_24H),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    budgets = [float(value) for value in args.fa24_budgets.split(",")]
    result = compare(
        old_path=args.old_metrics,
        new_path=args.new_metrics,
        fixed_threshold=args.fixed_threshold,
        budgets_fa_per_24h=budgets,
    )
    payload = canonical_bytes(result) + b"\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(args.output)
    print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
