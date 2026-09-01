#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

from src.soz.metrics import STANDARD_19


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "outputs/trustworthy_soz_workflow_mrmc_study_v1_1_20260816"
DEFAULT_OUTPUT = DEFAULT_PACK / "preflight_receipt.json"
READERS = ("reader_a", "reader_b", "reader_c")
ARMS = {"raw_only", "candidate_only", "candidate_plus_report"}
ACTIONS = {"display_candidate", "abstain", "indeterminate"}


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object row: {path}")
                rows.append(value)
    return rows


def _unique(rows: list[dict[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {name}: {value}")
        result[value] = row
    return result


def _channels(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or len(value) != len(set(value)):
        raise ValueError(f"{name} must be a unique channel list")
    if any(channel not in STANDARD_19 for channel in value):
        raise ValueError(f"{name} contains a non-standard channel")
    return [str(channel) for channel in value]


def _validate_completed(row: Mapping[str, Any]) -> None:
    if row.get("raw_phase_locked") is not True or not row.get("raw_phase_locked_at"):
        raise ValueError("Completed row lacks an immutable raw-phase lock")
    if row.get("raw_signal_assessable") not in {True, False}:
        raise ValueError("Completed row lacks raw assessability")
    for field in ("raw_review_time_sec", "post_intervention_review_time_sec"):
        value = row.get(field)
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Completed row has invalid {field}")
    for field in ("raw_candidate_action", "final_candidate_action"):
        if row.get(field) not in ACTIONS:
            raise ValueError(f"Completed row has invalid {field}")
    _channels(row.get("raw_candidate_channels"), "raw_candidate_channels")
    _channels(row.get("final_candidate_channels"), "final_candidate_channels")
    reviewed = row.get("reviewed_event_case_ids")
    if not isinstance(reviewed, list) or not reviewed:
        raise ValueError("Completed row must record every reviewed event")
    for field in ("raw_confidence_1_to_5", "final_confidence_1_to_5"):
        if row.get(field) not in {1, 2, 3, 4, 5}:
            raise ValueError(f"Completed row has invalid {field}")
    if not row.get("intervention_revealed_at") or not row.get("review_completed_at"):
        raise ValueError("Completed row lacks intervention/completion timestamps")
    arm = str(row["assigned_arm"])
    if arm == "raw_only":
        if row.get("assistance_helpfulness_1_to_5") is not None or row.get("assistance_harmfulness_1_to_5") is not None:
            raise ValueError("Raw-only arm may not rate nonexistent assistance")
        if row.get("would_use_in_research_review") is not None or row.get("report_overstatement_present") is not None:
            raise ValueError("Raw-only arm may not rate a hidden report")
    else:
        for field in ("assistance_helpfulness_1_to_5", "assistance_harmfulness_1_to_5"):
            if row.get(field) not in {1, 2, 3, 4, 5}:
                raise ValueError(f"Assisted row has invalid {field}")
        if row.get("would_use_in_research_review") not in {True, False}:
            raise ValueError("Assisted row lacks use decision")
        if arm == "candidate_plus_report" and row.get("report_overstatement_present") not in {True, False}:
            raise ValueError("Report arm lacks overstatement judgement")
        if arm == "candidate_only" and row.get("report_overstatement_present") is not None:
            raise ValueError("Candidate-only arm may not rate a hidden report")


def _validate_unreviewed(row: Mapping[str, Any]) -> None:
    if row.get("raw_phase_locked") is not False or row.get("review_completed_at") is not None:
        raise ValueError("Unreviewed row contains locked/completed state")
    if row.get("raw_candidate_channels") != [] or row.get("final_candidate_channels") != []:
        raise ValueError("Unreviewed row contains reader candidates")
    if row.get("reviewed_event_case_ids") != []:
        raise ValueError("Unreviewed row contains reviewed-event state")


def audit_pack(pack: Path) -> dict[str, Any]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "trustworthy_soz_workflow_mrmc_pack_v1_1":
        raise ValueError("Workflow manifest schema drifted")
    raw_cards = _unique(_rows(pack / "raw_case_cards.jsonl"), "case_id", "raw case")
    allocation = _unique(_rows(pack / "data_manager_allocation_key.jsonl"), "case_id", "allocation")
    if set(raw_cards) != set(allocation) or len(raw_cards) != 27:
        raise ValueError("Raw/allocation case rosters do not close")

    all_annotations: list[dict[str, Any]] = []
    arms_by_case: dict[str, set[str]] = defaultdict(set)
    reader_counts: dict[str, dict[str, int]] = {}
    completion: dict[str, dict[str, int]] = {}
    for reader in READERS:
        interventions = _unique(_rows(pack / f"{reader}_interventions.jsonl"), "case_id", f"{reader} intervention")
        annotations = _unique(_rows(pack / f"{reader}_annotations.jsonl"), "case_id", f"{reader} annotation")
        if set(interventions) != set(raw_cards) or set(annotations) != set(raw_cards):
            raise ValueError(f"{reader} case roster does not close")
        arm_counts: Counter[str] = Counter()
        completed = 0
        for case_id in raw_cards:
            intervention = interventions[case_id]
            annotation = annotations[case_id]
            arm = str(intervention.get("assigned_arm"))
            if arm not in ARMS or annotation.get("assigned_arm") != arm or annotation.get("reviewer_id") != reader:
                raise ValueError(f"{reader}/{case_id} assignment mismatch")
            if intervention.get("must_remain_hidden_until_raw_phase_lock") is not True:
                raise ValueError("Intervention is not phase-locked")
            if arm == "raw_only" and (intervention.get("candidate") is not None or intervention.get("report_text_zh") is not None):
                raise ValueError("Raw-only arm leaks assistance")
            if arm == "candidate_only" and (intervention.get("candidate") is None or intervention.get("report_text_zh") is not None):
                raise ValueError("Candidate-only arm content mismatch")
            if arm == "candidate_plus_report" and (intervention.get("candidate") is None or not intervention.get("report_text_zh")):
                raise ValueError("Candidate+report arm content mismatch")
            arms_by_case[case_id].add(arm)
            arm_counts[arm] += 1
            status = annotation.get("review_status")
            if status == "completed":
                expected_events = {
                    f"{case_id}-E{index:03d}"
                    for index in range(1, int(raw_cards[case_id]["linked_event_count"]) + 1)
                }
                if set(annotation.get("reviewed_event_case_ids", [])) != expected_events:
                    raise ValueError(f"{reader}/{case_id} did not review the complete event bag")
                _validate_completed(annotation)
                completed += 1
            elif status == "unreviewed":
                _validate_unreviewed(annotation)
            else:
                raise ValueError(f"Invalid review_status for {reader}/{case_id}")
            all_annotations.append(annotation)
        if dict(sorted(arm_counts.items())) != {"candidate_only": 9, "candidate_plus_report": 9, "raw_only": 9}:
            raise ValueError(f"{reader} arm balance failed")
        reader_counts[reader] = dict(sorted(arm_counts.items()))
        completion[reader] = {"completed": completed, "total": len(annotations)}
    if any(arms != ARMS for arms in arms_by_case.values()):
        raise ValueError("At least one case does not receive all three arms")

    complete = all(item["completed"] == item["total"] for item in completion.values())
    result: dict[str, Any] = {
        "schema_version": "trustworthy_soz_workflow_mrmc_audit_v1",
        "status": "COMPLETE" if complete else "PENDING_INDEPENDENT_CLINICIAN_REVIEWS",
        "preflight_passed": True,
        "outcome_metrics_computed": complete,
        "counts": {
            "cases": len(raw_cards),
            "reader_case_rows": len(all_annotations),
            "per_reader_arm_counts": reader_counts,
            "completion": completion,
        },
        "safety_boundary": {
            "targets_are_hidden_from_readers": True,
            "reader_candidates_do_not_return_to_training": True,
            "empty_or_partial_annotations_are_not_clinical_results": True,
        },
    }
    if complete:
        outcome_rows: list[dict[str, Any]] = []
        for annotation in all_annotations:
            reference = allocation[str(annotation["case_id"])]
            positives = set(reference["hidden_reference_positive_channels"])
            raw_channels = set(annotation["raw_candidate_channels"])
            final_channels = set(annotation["final_candidate_channels"])
            model_top1 = reference.get("hidden_model_top1")
            outcome_rows.append(
                {
                    "case_id": annotation["case_id"],
                    "reader_id": annotation["reviewer_id"],
                    "arm": annotation["assigned_arm"],
                    "stratum": reference["hidden_outcome_stratum"],
                    "raw_strict_hit": bool(raw_channels.intersection(positives)),
                    "final_strict_hit": bool(final_channels.intersection(positives)),
                    "model_top1_added": bool(model_top1 and model_top1 not in raw_channels and model_top1 in final_channels),
                    "unsafe_far_model_top1_adoption": bool(
                        reference["hidden_outcome_stratum"] == "far"
                        and model_top1
                        and model_top1 not in raw_channels
                        and model_top1 in final_channels
                    ),
                    "raw_review_time_sec": annotation["raw_review_time_sec"],
                    "post_intervention_review_time_sec": annotation["post_intervention_review_time_sec"],
                }
            )
        aggregates: dict[str, Any] = {}
        for arm in sorted(ARMS):
            arm_rows = [row for row in outcome_rows if row["arm"] == arm]
            far_rows = [row for row in arm_rows if row["stratum"] == "far"]
            exact_rows = [row for row in arm_rows if row["stratum"] == "exact"]
            aggregates[arm] = {
                "n": len(arm_rows),
                "far_unsafe_adoption": sum(row["unsafe_far_model_top1_adoption"] for row in far_rows),
                "far_denominator": len(far_rows),
                "exact_final_strict_hit": sum(row["final_strict_hit"] for row in exact_rows),
                "exact_denominator": len(exact_rows),
            }
        result["descriptive_outcomes"] = aggregates
        result["interpretation"] = "descriptive_MRMC_panel_not_cohort_prevalence_or_independent_confirmation"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the three-arm SOZ workflow MRMC pack")
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    pack = args.pack if args.pack.is_absolute() else ROOT / args.pack
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = audit_pack(pack)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status={result['status']}")
    print(f"preflight_passed={str(result['preflight_passed']).lower()}")
    print(f"outcome_metrics_computed={str(result['outcome_metrics_computed']).lower()}")
    print(f"output={output}")
    return 2 if args.require_complete and not result["outcome_metrics_computed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
