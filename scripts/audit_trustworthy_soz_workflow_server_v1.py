#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from scripts.serve_trustworthy_soz_report_reader_study_v1 import DEFAULT_TUSZ_ROOT
from scripts.serve_trustworthy_soz_workflow_mrmc_study_v1 import WorkflowMRMCStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "outputs/trustworthy_soz_workflow_mrmc_study_v1_1_20260816"
DEFAULT_OUTPUT = DEFAULT_PACK / "server_preflight_receipt.json"
ROLES = ("reader_a", "reader_b", "reader_c")


def _copy_server_view(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in ("manifest.json", "raw_case_cards.jsonl", "case_linkage.csv"):
        shutil.copy2(source / name, destination / name)
    for role in ROLES:
        shutil.copy2(source / f"{role}_interventions.jsonl", destination / f"{role}_interventions.jsonl")
        shutil.copy2(source / f"{role}_annotations.jsonl", destination / f"{role}_annotations.jsonl")


def _raw_payload(store: WorkflowMRMCStore, case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "raw_signal_assessable": True,
        "raw_review_time_sec": 1.0,
        "reviewed_event_case_ids": sorted(store._expected_events(case_id)),
        "raw_candidate_action": "indeterminate",
        "raw_candidate_channels": [],
        "raw_confidence_1_to_5": 3,
    }


def audit_server(*, pack: Path, tusz_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="workflow-mrmc-server-preflight-") as temporary_name:
        isolated = Path(temporary_name) / "server_view"
        _copy_server_view(pack, isolated)
        if (isolated / "data_manager_allocation_key.jsonl").exists():
            raise AssertionError("target-bearing allocation key entered the server view")
        role_results: dict[str, object] = {}
        for role in ROLES:
            store = WorkflowMRMCStore(
                reader_pack=isolated,
                tusz_root=tusz_root,
                role=role,
                reviewer_id=f"PREFLIGHT_{role}",
            )
            prelock_with_intervention = 0
            prelock_with_arm = 0
            for case_id in store.raw_card_by_case:
                payload = store.case_payload(case_id)
                prelock_with_intervention += int("intervention" in payload)
                prelock_with_arm += int("assigned_arm" in payload["annotation"])
            postlock_checks: dict[str, bool] = {}
            for arm in ("raw_only", "candidate_only", "candidate_plus_report"):
                case_id = next(
                    case_id
                    for case_id, intervention in store.intervention_by_case.items()
                    if intervention["assigned_arm"] == arm
                    and store.annotation_by_case[case_id]["raw_phase_locked"] is False
                )
                result = store.save_raw(_raw_payload(store, case_id), lock_phase=True)
                intervention = result["intervention"]
                postlock_checks[arm] = (
                    intervention["assigned_arm"] == arm
                    and (
                        (arm == "raw_only" and intervention["candidate"] is None and intervention["report_text_zh"] is None)
                        or (arm == "candidate_only" and intervention["candidate"] is not None and intervention["report_text_zh"] is None)
                        or (arm == "candidate_plus_report" and intervention["candidate"] is not None and bool(intervention["report_text_zh"]))
                    )
                )
            role_results[role] = {
                "case_count": len(store.raw_card_by_case),
                "prelock_intervention_exposure_count": prelock_with_intervention,
                "prelock_arm_exposure_count": prelock_with_arm,
                "postlock_arm_content_checks": postlock_checks,
                "other_reader_annotation_loaded": False,
                "target_allocation_loaded": False,
            }
        passed = all(
            result["prelock_intervention_exposure_count"] == 0
            and result["prelock_arm_exposure_count"] == 0
            and all(result["postlock_arm_content_checks"].values())
            for result in role_results.values()
        )
        return {
            "schema_version": "trustworthy_soz_workflow_mrmc_server_preflight_v1",
            "status": "PASS" if passed else "FAIL",
            "server_enforced_raw_phase_lock": passed,
            "target_bearing_allocation_absent_from_server_view": True,
            "roles": role_results,
            "clinical_annotations_modified": False,
            "clinical_outcomes_computed": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the target-free workflow MRMC server view")
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pack = args.pack if args.pack.is_absolute() else ROOT / args.pack
    tusz_root = args.tusz_root if args.tusz_root.is_absolute() else ROOT / args.tusz_root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = audit_server(pack=pack.resolve(strict=True), tusz_root=tusz_root.resolve(strict=True))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status={result['status']}")
    print(f"server_enforced_raw_phase_lock={str(result['server_enforced_raw_phase_lock']).lower()}")
    print(f"output={output}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
