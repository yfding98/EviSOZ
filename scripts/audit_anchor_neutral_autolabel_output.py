#!/usr/bin/env python3
"""Audit an anchor-neutral nearby-onset AutoLabel output directory.

The audit is intentionally independent from the inference path.  It names
each output contract separately so that a failure identifies the first event
and field instead of disappearing inside one compound assertion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.auto_annotate.nearby_onset_autolabel import REVIEW_CHANNELS_32  # noqa: E402


AUTO_KINDS = {"autolabel_candidate", "autolabel_feature_candidate"}
EPSILON_S = 1e-6


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _same_time(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return (
        left_number is not None
        and right_number is not None
        and abs(left_number - right_number) <= EPSILON_S
    )


def _event_input_hash(
    run_fingerprint: str,
    item: Mapping[str, Any],
) -> str:
    novice_anchor = item.get("novice_anchor") or {}
    payload = {
        "run_fingerprint": run_fingerprint,
        "event_id": item.get("event_id"),
        "source_file_identity": item.get("source_file_identity"),
        "anchor_start_s": novice_anchor.get("start_s"),
        "anchor_end_s": novice_anchor.get("end_s"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _channel_names(rows: Any) -> list[str]:
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("channel", ""))
        for row in rows
        if isinstance(row, Mapping) and row.get("channel")
    ]


class Audit:
    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}

    def check(
        self,
        name: str,
        condition: bool,
        *,
        event_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        state = self.checks.setdefault(
            name,
            {"checked": 0, "violations": 0, "first_violation": None},
        )
        state["checked"] += 1
        if condition:
            return
        state["violations"] += 1
        if state["first_violation"] is None:
            state["first_violation"] = {
                "event_id": event_id,
                **dict(details or {}),
            }

    @property
    def violation_count(self) -> int:
        return sum(int(state["violations"]) for state in self.checks.values())


def _audit_channel_contract(audit: Audit, item: Mapping[str, Any]) -> None:
    event_id = str(item.get("event_id", ""))
    labels = item.get("channel_labels")
    rows = labels if isinstance(labels, list) else []
    names = _channel_names(rows)
    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    expected_names = set(REVIEW_CHANNELS_32)
    actual_names = set(names)

    audit.check(
        "channel_labels_are_exactly_32_unique_review_channels",
        len(rows) == 32 and len(names) == 32 and not duplicate_names and actual_names == expected_names,
        event_id=event_id,
        details={
            "count": len(rows),
            "duplicates": duplicate_names,
            "missing": sorted(expected_names - actual_names),
            "unexpected": sorted(actual_names - expected_names),
        },
    )
    unmasked = [
        row.get("channel")
        for row in rows
        if not isinstance(row, Mapping) or row.get("mask_for_soz_loss") is not True
    ]
    audit.check(
        "all_32_channel_labels_are_loss_masked_before_review",
        len(rows) == 32 and not unmasked,
        event_id=event_id,
        details={"unmasked_or_malformed_channels": unmasked},
    )
    legacy_train_mask_fields = [
        row.get("channel")
        for row in rows
        if isinstance(row, Mapping) and "train_mask" in row
    ]
    audit.check(
        "channel_mask_schema_uses_mask_for_soz_loss_not_train_mask",
        len(rows) == 32
        and not legacy_train_mask_fields
        and all(
            isinstance(row, Mapping) and "mask_for_soz_loss" in row
            for row in rows
        ),
        event_id=event_id,
        details={
            "canonical_field": "mask_for_soz_loss",
            "unexpected_train_mask_channels": legacy_train_mask_fields,
        },
    )


def _audit_auto_annotation(audit: Audit, item: Mapping[str, Any]) -> None:
    event_id = str(item.get("event_id", ""))
    t0 = _number((item.get("t0") or {}).get("time_sec"))
    t_spread = _number((item.get("t_spread") or {}).get("time_sec"))
    t_end = _number((item.get("t_end") or {}).get("time_sec"))
    model_evidence = item.get("model_evidence") or {}
    summary = item.get("candidate_summary") or {}
    onset_bounds = model_evidence.get("onset_search_bounds") or {}
    forward_bounds = model_evidence.get("forward_followup_bounds") or {}
    onset_start = _number(onset_bounds.get("start_s"))
    onset_end = _number(onset_bounds.get("end_s"))
    forward_end = _number(forward_bounds.get("end_s"))
    timeline = item.get("feature_timeline")
    timeline_rows = timeline if isinstance(timeline, list) else []

    audit.check(
        "t0_is_inside_eeg_onset_search_bounds",
        t0 is not None
        and onset_start is not None
        and onset_end is not None
        and onset_start - EPSILON_S <= t0 <= onset_end + EPSILON_S,
        event_id=event_id,
        details={"t0_s": t0, "onset_search_bounds": onset_bounds},
    )

    overlap_times = [
        row.get("time_s")
        for row in timeline_rows
        if isinstance(row, Mapping) and row.get("is_baseline") is True and row.get("is_search") is True
    ]
    audit.check(
        "feature_timeline_baseline_and_search_flags_do_not_overlap",
        not overlap_times,
        event_id=event_id,
        details={"overlap_times_s": overlap_times[:10]},
    )

    selected_t0_rows = [
        row
        for row in timeline_rows
        if isinstance(row, Mapping) and row.get("is_selected_t0") is True
    ]
    audit.check(
        "timeline_has_one_search_marked_t0_matching_annotation",
        len(selected_t0_rows) == 1
        and _same_time(selected_t0_rows[0].get("time_s"), t0)
        and selected_t0_rows[0].get("is_search") is True,
        event_id=event_id,
        details={
            "t0_s": t0,
            "marked_rows": [
                {"time_s": row.get("time_s"), "is_search": row.get("is_search")}
                for row in selected_t0_rows
            ],
        },
    )

    # supporting_channels is a list of strings, unlike other channel objects.
    raw_spread_support = (item.get("t_spread") or {}).get("supporting_channels")
    spread_support = (
        [str(value) for value in raw_spread_support if value]
        if isinstance(raw_spread_support, list)
        else []
    )
    exported_spread = _channel_names(item.get("spread_channels"))
    summary_spread = summary.get("spread_channels")
    summary_spread_names = (
        [str(value) for value in summary_spread if value]
        if isinstance(summary_spread, list)
        else []
    )
    spread_markers = [
        row.get("time_s")
        for row in timeline_rows
        if isinstance(row, Mapping) and row.get("is_selected_spread") is True
    ]
    if t_spread is None:
        spread_condition = not spread_support and not exported_spread and not summary_spread_names
        marker_condition = not spread_markers
    else:
        spread_condition = (
            bool(spread_support)
            and set(spread_support) == set(exported_spread) == set(summary_spread_names)
        )
        marker_condition = len(spread_markers) == 1 and _same_time(spread_markers[0], t_spread)
    audit.check(
        "spread_supporting_channels_obey_optional_time_semantics",
        spread_condition,
        event_id=event_id,
        details={
            "t_spread_s": t_spread,
            "supporting_channels": spread_support,
            "exported_spread_channels": exported_spread,
            "summary_spread_channels": summary_spread_names,
        },
    )
    audit.check(
        "timeline_spread_marker_matches_optional_spread_time",
        marker_condition,
        event_id=event_id,
        details={"t_spread_s": t_spread, "marked_times_s": spread_markers},
    )
    if t_spread is not None:
        audit.check(
            "spread_is_after_t0_and_inside_forward_horizon",
            t0 is not None
            and forward_end is not None
            and t_spread >= t0 + 3.0 - EPSILON_S
            and t_spread <= t0 + 120.0 + EPSILON_S
            and t_spread <= forward_end + EPSILON_S,
            event_id=event_id,
            details={
                "t0_s": t0,
                "t_spread_s": t_spread,
                "forward_end_s": forward_end,
            },
        )

    end_markers = [
        row.get("time_s")
        for row in timeline_rows
        if isinstance(row, Mapping) and row.get("is_selected_end") is True
    ]
    if t_end is None:
        end_condition = not end_markers
    else:
        end_condition = (
            t0 is not None
            and forward_end is not None
            and t_end >= t0 + 10.0 - EPSILON_S
            and t_end <= t0 + 120.0 + EPSILON_S
            and t_end <= forward_end + EPSILON_S
            and len(end_markers) == 1
            and _same_time(end_markers[0], t_end)
        )
    audit.check(
        "optional_end_is_at_most_t0_plus_120_and_matches_timeline",
        end_condition,
        event_id=event_id,
        details={
            "t0_s": t0,
            "t_end_s": t_end,
            "forward_end_s": forward_end,
            "marked_times_s": end_markers,
        },
    )

    audit.check(
        "candidate_summary_times_match_annotation_times",
        _same_time(summary.get("selected_t0_s"), t0)
        and (
            (summary.get("t_spread_candidate_s") is None and t_spread is None)
            or _same_time(summary.get("t_spread_candidate_s"), t_spread)
        )
        and (
            (summary.get("t_end_candidate_s") is None and t_end is None)
            or _same_time(summary.get("t_end_candidate_s"), t_end)
        ),
        event_id=event_id,
        details={
            "annotation_times": {"t0": t0, "t_spread": t_spread, "t_end": t_end},
            "summary_times": {
                "t0": summary.get("selected_t0_s"),
                "t_spread": summary.get("t_spread_candidate_s"),
                "t_end": summary.get("t_end_candidate_s"),
            },
        },
    )
    novice_anchor = item.get("novice_anchor") or {}
    audit.check(
        "novice_anchor_is_navigation_only_not_scored_or_ranked",
        novice_anchor.get("ground_truth") is False
        and novice_anchor.get("used_in_scoring") is False
        and novice_anchor.get("used_in_ranking") is False
        and model_evidence.get("anchor_used_in_scoring") is False
        and model_evidence.get("anchor_used_in_ranking") is False,
        event_id=event_id,
        details={
            "novice_anchor": novice_anchor,
            "model_anchor_flags": {
                "used_in_scoring": model_evidence.get("anchor_used_in_scoring"),
                "used_in_ranking": model_evidence.get("anchor_used_in_ranking"),
            },
        },
    )


def audit_output(output_dir: Path) -> dict[str, Any]:
    summary = _read_json(output_dir / "summary.json")
    viewer_payload = _read_json(output_dir / "viewer_annotations.json")
    annotations = viewer_payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("viewer_annotations.json annotations must be a list")
    records = _read_jsonl(output_dir / "nearby_onset_records.jsonl")
    candidates = _read_jsonl(output_dir / "nearby_onset_candidates.jsonl")
    errors = _read_jsonl(output_dir / "errors.jsonl")

    audit = Audit()
    annotation_ids = [str(item.get("event_id", "")) for item in annotations]
    record_ids = [str(item.get("event_id", "")) for item in records]
    auto_annotations = [item for item in annotations if item.get("annotation_kind") in AUTO_KINDS]
    manual_annotations = [item for item in annotations if item.get("annotation_kind") == "manual_review_task"]
    run_fingerprint = str(summary.get("run_fingerprint", ""))

    audit.check(
        "run_completed_without_recorded_errors",
        summary.get("status") == "completed" and not errors and int(summary.get("errors", -1)) == 0,
        details={
            "status": summary.get("status"),
            "summary_errors": summary.get("errors"),
            "error_rows": len(errors),
        },
    )
    audit.check(
        "one_unique_annotation_and_record_per_selected_event",
        len(annotation_ids) == len(set(annotation_ids))
        and len(record_ids) == len(set(record_ids))
        and set(annotation_ids) == set(record_ids)
        and len(annotations) == int(summary.get("selected_event_count", -1))
        and len(records) == int(summary.get("processed_events", -1)),
        details={
            "annotations": len(annotations),
            "unique_annotation_ids": len(set(annotation_ids)),
            "records": len(records),
            "unique_record_ids": len(set(record_ids)),
            "selected_event_count": summary.get("selected_event_count"),
            "processed_events": summary.get("processed_events"),
        },
    )
    audit.check(
        "aggregate_candidates_match_automatic_annotations",
        len(candidates) == len(auto_annotations)
        and [row.get("candidate_id") for row in candidates]
        == [row.get("candidate_id") for row in auto_annotations],
        details={"candidate_rows": len(candidates), "automatic_annotations": len(auto_annotations)},
    )
    audit.check(
        "run_fingerprint_is_consistent_across_outputs",
        bool(run_fingerprint)
        and all(item.get("run_fingerprint") == run_fingerprint for item in records),
        details={
            "run_fingerprint": run_fingerprint,
            "annotation_input_hash_semantics": "per_event_hash_not_run_fingerprint",
        },
    )

    record_by_id = {str(record.get("event_id")): record for record in records}
    for item in annotations:
        event_id = str(item.get("event_id", ""))
        record = record_by_id.get(event_id) or {}
        record_annotations = record.get("viewer_annotations")
        audit.check(
            "aggregate_annotation_matches_per_event_record",
            isinstance(record_annotations, list)
            and len(record_annotations) == 1
            and record_annotations[0] == item,
            event_id=event_id,
            details={"record_annotation_count": len(record_annotations) if isinstance(record_annotations, list) else None},
        )
        expected_input_hash = _event_input_hash(run_fingerprint, item)
        audit.check(
            "annotation_input_hash_matches_per_event_input_identity",
            item.get("input_hash") == expected_input_hash,
            event_id=event_id,
            details={
                "actual_input_hash": item.get("input_hash"),
                "expected_input_hash": expected_input_hash,
            },
        )
        _audit_channel_contract(audit, item)
        if item.get("annotation_kind") in AUTO_KINDS:
            _audit_auto_annotation(audit, item)
        else:
            novice_anchor = item.get("novice_anchor") or {}
            quality_flags = item.get("quality_flags") or []
            audit.check(
                "nonautomatic_annotations_are_explicit_manual_fail_closed_tasks",
                item.get("annotation_kind") == "manual_review_task"
                and (item.get("t_spread") or {}).get("time_sec") is None
                and (item.get("t_end") or {}).get("time_sec") is None
                and item.get("overall_confidence") == 0.0,
                event_id=event_id,
                details={"annotation_kind": item.get("annotation_kind")},
            )
            audit.check(
                "manual_task_t0_is_navigation_line_not_automatic_detection",
                item.get("annotation_kind") == "manual_review_task"
                and str(item.get("candidate_id", "")).endswith("-manual-review")
                and "manual_review_task" in quality_flags
                and "no_localizable_autolabel_candidate" in quality_flags
                and _same_time((item.get("t0") or {}).get("time_sec"), novice_anchor.get("start_s"))
                and (item.get("t0") or {}).get("confidence") == 0.0
                and not (item.get("t0") or {}).get("supporting_channels"),
                event_id=event_id,
                details={
                    "candidate_id": item.get("candidate_id"),
                    "t0_s": (item.get("t0") or {}).get("time_sec"),
                    "novice_navigation_start_s": novice_anchor.get("start_s"),
                    "quality_flags": quality_flags,
                },
            )

    spread_present = sum(
        (item.get("t_spread") or {}).get("time_sec") is not None for item in auto_annotations
    )
    end_present = sum((item.get("t_end") or {}).get("time_sec") is not None for item in auto_annotations)
    first_missing_spread = next(
        (
            {
                "event_id": item.get("event_id"),
                "event_row_index": item.get("event_row_index"),
            }
            for item in auto_annotations
            if (item.get("t_spread") or {}).get("time_sec") is None
        ),
        None,
    )
    result = {
        "status": "passed" if audit.violation_count == 0 else "failed",
        "output_dir": str(output_dir.resolve()),
        "run_fingerprint": run_fingerprint,
        "counts": {
            "annotations": len(annotations),
            "automatic_annotations": len(auto_annotations),
            "manual_fail_closed_annotations": len(manual_annotations),
            "records": len(records),
            "spread_present": spread_present,
            "spread_absent": len(auto_annotations) - spread_present,
            "end_present": end_present,
            "end_absent_right_censored": len(auto_annotations) - end_present,
        },
        "optional_time_semantics": {
            "spread_supporting_channels_are_required_only_when_t_spread_is_present": True,
            "first_automatic_annotation_without_spread": first_missing_spread,
            "null_end_is_right_censored_not_negative": True,
        },
        "schema_semantics": {
            "canonical_channel_loss_mask_field": "mask_for_soz_loss",
            "train_mask_field_is_not_part_of_viewer_annotation_schema": True,
            "annotation_input_hash_is_per_event_not_run_fingerprint": True,
            "manual_task_t0_is_sz_start_navigation_line_not_auto_t0": True,
        },
        "violation_count": audit.violation_count,
        "checks": audit.checks,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Completed nearby-onset output directory")
    parser.add_argument("--output", type=Path, help="Optional JSON path for the audit result")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = audit_output(args.output_dir.expanduser().resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
