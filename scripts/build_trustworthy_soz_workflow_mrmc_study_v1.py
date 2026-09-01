#!/usr/bin/env python3
"""Build a three-reader, three-arm MRMC workflow and automation-bias study pack."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from safetensors.torch import load_file

from scripts.build_trustworthy_soz_report_reader_study_v1 import _signal_events
from src.soz.metrics import DEEPSOZ_STANDARD19_NEIGHBORS, STANDARD_19


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("raw_only", "candidate_only", "candidate_plus_report")
OUTCOME_QUOTAS = {"exact": 10, "neighbor_only": 7, "far": 7, "abstain": 3}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected JSON object row: {path}")
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded_patients(factuality_pack: Path, language_pack: Path) -> set[str]:
    excluded: set[str] = set()
    with (factuality_pack / "case_linkage.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            excluded.add(str(row["public_patient_id"]))
    for row in _read_jsonl(language_pack / "data_manager_allocation_key.jsonl"):
        if str(row["source_cohort"]).startswith("public_"):
            excluded.add(str(row["source_patient_id"]))
    return excluded


def _qwen_text(narrative: Mapping[str, Any]) -> str:
    sections = narrative.get("sections")
    if not isinstance(sections, Sequence) or not sections:
        raise ValueError("Qwen narrative sections are missing")
    parts = [f"{section['heading_zh']}\n{section['text_zh']}" for section in sections]
    notes = narrative.get("knowledge_notes", [])
    if notes:
        parts.append("一般医学知识说明\n" + "\n".join(f"- {note['text_zh']}" for note in notes))
    return "\n\n".join(parts)


def _outcome(
    *,
    report: Mapping[str, Any],
    target: torch.Tensor,
) -> tuple[str, list[str], str | None]:
    localization = report.get("localization")
    if not isinstance(localization, Mapping):
        raise ValueError("Report localization is missing")
    positive_indices = torch.nonzero(target > 0, as_tuple=False).flatten().tolist()
    positives = [STANDARD_19[index] for index in positive_indices]
    if not positives:
        raise ValueError("Every study patient must have a non-empty reference-positive set")
    if localization.get("action") == "localization_abstain":
        return "abstain", positives, None
    candidates = localization.get("displayed_candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        raise ValueError("Displayed report lacks candidates")
    top1 = str(candidates[0]["channel"])
    top1_index = STANDARD_19.index(top1)
    if top1_index in positive_indices:
        return "exact", positives, top1
    acceptable: set[int] = set()
    for index in positive_indices:
        acceptable.update(DEEPSOZ_STANDARD19_NEIGHBORS[index])
    if top1_index in acceptable:
        return "neighbor_only", positives, top1
    return "far", positives, top1


def _select_cases(
    *,
    reports: Mapping[str, dict[str, Any]],
    qwen_reports: Mapping[str, dict[str, Any]],
    signal_by_patient: Mapping[str, Sequence[dict[str, Any]]],
    patient_ids: Sequence[str],
    targets: torch.Tensor,
    h_only_probability: torch.Tensor,
    candidate_mask: torch.Tensor,
    excluded: set[str],
    far_subtypes: Mapping[str, str],
    max_events: int,
    quotas: Mapping[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        if patient_id in excluded:
            continue
        events = signal_by_patient.get(patient_id)
        if not events or len(events) > max_events:
            continue
        report = reports[patient_id]
        qwen = qwen_reports[patient_id]
        category, positives, top1 = _outcome(report=report, target=targets[index])
        localization = report["localization"]
        if localization["action"] == "display_candidate":
            masked = h_only_probability[index].clone()
            masked[~candidate_mask.bool()] = -torch.inf
            order = torch.argsort(masked, descending=True).tolist()
            expected = [STANDARD_19[channel_index] for channel_index in order[:5]]
            observed = [str(item["channel"]) for item in localization["displayed_candidates"]]
            if expected != observed:
                raise ValueError(f"v21 report/H-only lineage mismatch for patient {patient_id}")
        if qwen["localization"] != localization:
            raise ValueError(f"Qwen changed localization for patient {patient_id}")
        grouped[category].append(
            {
                "patient_id": patient_id,
                "report": report,
                "qwen": qwen,
                "events": list(events),
                "outcome": category,
                "positive_channels": positives,
                "top1": top1,
                "far_subtype": far_subtypes.get(patient_id) if category == "far" else None,
            }
        )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for category, quota in quotas.items():
        rows = sorted(
            grouped.get(category, []),
            key=lambda row: (
                len(row["events"]),
                str(row.get("far_subtype")),
                str(row["patient_id"]),
            ),
        )
        if len(rows) < quota:
            raise ValueError(f"Insufficient {category} cases: {len(rows)} < {quota}")
        # Draw within event-count/subtype blocks without outcome-score tuning.
        selected.extend(rng.sample(rows, quota))
    rng.shuffle(selected)
    return selected


def _candidate_card(localization: Mapping[str, Any]) -> dict[str, Any]:
    action = str(localization["action"])
    channels = [str(item["channel"]) for item in localization.get("displayed_candidates", [])]
    return {
        "action": action,
        "candidate_channels": channels,
        "top1_region_zh": localization.get("top1_region_projection_zh"),
        "display_text_zh": (
            "系统对本病例不显示定位候选；该弃权不表示不存在SOZ。"
            if action == "localization_abstain"
            else "供复核的头皮电极SOZ-reference候选依次为" + "、".join(channels) + "。"
        ),
        "scores_margin_or_threshold_exposed": False,
    }


def _blank_annotation(case_id: str, reviewer: str, arm: str, order: int) -> dict[str, Any]:
    return {
        "schema_version": "trustworthy_soz_workflow_mrmc_annotation_v1_1",
        "case_id": case_id,
        "reviewer_id": reviewer,
        "assigned_arm": arm,
        "presentation_order": order,
        "review_status": "unreviewed",
        "raw_phase_locked": False,
        "raw_phase_locked_at": None,
        "raw_signal_assessable": None,
        "raw_review_time_sec": None,
        "reviewed_event_case_ids": [],
        "raw_candidate_action": None,
        "raw_candidate_channels": [],
        "raw_confidence_1_to_5": None,
        "intervention_revealed_at": None,
        "post_intervention_review_time_sec": None,
        "final_candidate_action": None,
        "final_candidate_channels": [],
        "final_confidence_1_to_5": None,
        "assistance_helpfulness_1_to_5": None,
        "assistance_harmfulness_1_to_5": None,
        "report_overstatement_present": None,
        "would_use_in_research_review": None,
        "review_completed_at": None,
        "free_text_not_for_training": "",
    }


def build_workflow_pack(
    *,
    report_dir: Path,
    qwen_dir: Path,
    signal_universe_path: Path,
    identity_manifest_path: Path,
    v16_manifest_path: Path,
    v29_oof_path: Path,
    far_audit_path: Path,
    factuality_pack: Path,
    language_pack: Path,
    output_dir: Path,
    seed: int = 20260816,
    max_events: int = 10,
    quotas: Mapping[str, int] = OUTCOME_QUOTAS,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    reports = {row["patient_id"]: row for row in _read_jsonl(report_dir / "public_patient_reports.jsonl")}
    qwen_reports = {row["patient_id"]: row for row in _read_jsonl(qwen_dir / "public_patient_reports.jsonl")}
    if set(reports) != set(qwen_reports):
        raise ValueError("Deterministic and Qwen public patient rosters differ")
    signal_universe = _read_json(signal_universe_path)
    identity = _read_json(identity_manifest_path)
    _, signal_by_patient = _signal_events(signal_universe, identity)
    v16 = _read_json(v16_manifest_path)
    patient_ids = [str(value) for value in v16["patient_ids"]]
    tensors = load_file(str(v29_oof_path))
    targets = tensors["targets"]
    h_only_probability = tensors["oof.h_only_probability"]
    candidate_mask = tensors["candidate_mask"]
    if targets.shape != h_only_probability.shape or targets.shape[0] != len(patient_ids):
        raise ValueError("Patient/tensor axes do not align")
    far_audit = _read_json(far_audit_path)
    far_subtypes = {
        str(row["patient_id"]): str(row["far_subtype"])
        for row in far_audit["public_patient_level"]["far_cases"]
    }
    excluded = _excluded_patients(factuality_pack, language_pack)
    selected = _select_cases(
        reports=reports,
        qwen_reports=qwen_reports,
        signal_by_patient=signal_by_patient,
        patient_ids=patient_ids,
        targets=targets,
        h_only_probability=h_only_probability,
        candidate_mask=candidate_mask,
        excluded=excluded,
        far_subtypes=far_subtypes,
        max_events=max_events,
        quotas=quotas,
        seed=seed,
    )
    if len(selected) % 3 != 0:
        raise ValueError("Case count must be divisible by three for balanced Latin assignment")

    output_dir.mkdir(parents=True)
    raw_cards: list[dict[str, Any]] = []
    allocation: list[dict[str, Any]] = []
    linkages: list[dict[str, Any]] = []
    intervention_by_reader: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotation_by_reader: dict[str, list[dict[str, Any]]] = defaultdict(list)
    readers = ("reader_a", "reader_b", "reader_c")
    per_reader_arm_counts: dict[str, dict[str, int]] = {reader: defaultdict(int) for reader in readers}

    for case_index, row in enumerate(selected, start=1):
        case_id = f"WF-{case_index:03d}"
        patient_id = str(row["patient_id"])
        localization = row["report"]["localization"]
        raw_cards.append(
            {
                "schema_version": "trustworthy_soz_workflow_raw_card_v1",
                "case_id": case_id,
                "linked_event_count": len(row["events"]),
                "raw_phase_instruction_zh": "先独立复核全部关联发作事件并锁定初始候选，随后系统才显示分配的辅助条件。",
            }
        )
        allocation.append(
            {
                "case_id": case_id,
                "public_patient_id": patient_id,
                "hidden_outcome_stratum": row["outcome"],
                "hidden_reference_positive_channels": row["positive_channels"],
                "hidden_model_top1": row["top1"],
                "hidden_far_subtype": row["far_subtype"],
                "event_count": len(row["events"]),
            }
        )
        for event_index, event in enumerate(row["events"], start=1):
            linkages.append(
                {
                    "case_id": case_id,
                    "public_patient_id": patient_id,
                    "event_bundle_index": event_index,
                    "public_event_id": event["event_id"],
                    "relative_edf_path": event["relative_edf_path"],
                    "global_event_t0_sec": float(event["global_t0_sec"]),
                    "global_event_stop_sec": float(event["global_stop_sec"]),
                    "suggested_review_window_start_sec": max(0.0, float(event["global_t0_sec"]) - 30.0),
                    "suggested_review_window_stop_sec_unclamped": float(event["global_stop_sec"]) + 60.0,
                }
            )
        for reader_index, reader in enumerate(readers):
            arm = ARMS[(case_index - 1 + reader_index) % len(ARMS)]
            per_reader_arm_counts[reader][arm] += 1
            intervention: dict[str, Any] = {
                "schema_version": "trustworthy_soz_workflow_intervention_card_v1",
                "case_id": case_id,
                "assigned_arm": arm,
                "candidate": None,
                "report_text_zh": None,
                "must_remain_hidden_until_raw_phase_lock": True,
            }
            if arm in {"candidate_only", "candidate_plus_report"}:
                intervention["candidate"] = _candidate_card(localization)
            if arm == "candidate_plus_report":
                intervention["report_text_zh"] = _qwen_text(row["qwen"]["published_narrative"])
            intervention_by_reader[reader].append(intervention)
            annotation_by_reader[reader].append(_blank_annotation(case_id, reader, arm, case_index))

    rng = random.Random(seed + 70000)
    for reader_index, reader in enumerate(readers):
        pairs = list(zip(intervention_by_reader[reader], annotation_by_reader[reader]))
        random.Random(rng.randrange(2**31) + reader_index).shuffle(pairs)
        for order, (_, annotation) in enumerate(pairs, start=1):
            annotation["presentation_order"] = order
        intervention_by_reader[reader] = [pair[0] for pair in pairs]
        annotation_by_reader[reader] = [pair[1] for pair in pairs]

    _write_jsonl(output_dir / "raw_case_cards.jsonl", raw_cards)
    _write_jsonl(output_dir / "data_manager_allocation_key.jsonl", allocation)
    with (output_dir / "case_linkage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(linkages[0]))
        writer.writeheader()
        writer.writerows(linkages)
    for reader in readers:
        _write_jsonl(output_dir / f"{reader}_interventions.jsonl", intervention_by_reader[reader])
        _write_jsonl(output_dir / f"{reader}_annotations.jsonl", annotation_by_reader[reader])

    manifest = {
        "schema_version": "trustworthy_soz_workflow_mrmc_pack_v1_1",
        "status": "empty_three_reader_three_arm_workflow_pack_ready",
        "sampling_seed": seed,
        "counts": {
            "cases": len(selected),
            "readers": len(readers),
            "linked_events": len(linkages),
            "hidden_outcome_strata": dict(sorted((key, sum(row["outcome"] == key for row in selected)) for key in quotas)),
            "per_reader_arm_counts": {reader: dict(sorted(counts.items())) for reader, counts in per_reader_arm_counts.items()},
        },
        "design": {
            "each_case_seen_once_per_reader": True,
            "each_case_receives_all_three_arms_across_readers": True,
            "raw_phase_locked_before_intervention": True,
            "factuality_and_language_study_patient_overlap": False,
            "outcome_stratification_is_for_safety_not_prevalence_estimation": True,
            "primary_safety_contrast": "far_model_top1_adoption_candidate_or_report_vs_raw_only",
            "benefit_contrast": "strict_reference_concordance_change_candidate_or_report_vs_raw_only",
        },
        "access_receipt": {
            "deepsoz_target_loaded_for_safety_stratification": True,
            "target_used_for_training_calibration_or_model_selection": False,
            "private_data_loaded": False,
            "tusz_channel_time_involvement_loaded": False,
            "candidate_or_report_changed_after_target_read": False,
            "reader_output_returns_to_training": False,
        },
        "blinding": {
            "reader_sees_hidden_outcome_or_reference": False,
            "reader_sees_other_reader_assignment_or_annotation": False,
            "raw_only_arm_exposes_model_candidate": False,
            "candidate_only_arm_exposes_report": False,
            "scores_margin_threshold_hidden": True,
            "data_manager_allocation_and_linkage_are_not_reader_files": True,
        },
        "source_hashes": {
            "v29_oof": _sha256(v29_oof_path),
            "v16_manifest": _sha256(v16_manifest_path),
            "deterministic_report_manifest": _sha256(report_dir / "manifest.json"),
            "qwen_report_manifest": _sha256(qwen_dir / "manifest.json"),
            "far_audit": _sha256(far_audit_path),
        },
        "analysis_boundary": {
            "selected_outcome_strata_are_not_cohort_prevalence": True,
            "reader_candidate_is_not_new_gold": True,
            "workflow_result_is_not_independent_SOZ_confirmation": True,
            "clinical_results_pending": True,
        },
        "execution": {
            "server": "scripts/serve_trustworthy_soz_workflow_mrmc_study_v1.py",
            "launcher": "scripts/run_trustworthy_soz_workflow_mrmc_study_v1.sh",
            "reader_ui": "research/00_problem_definition/trustworthy_soz_workflow_mrmc_reader.html",
            "completion_audit": "scripts/audit_trustworthy_soz_workflow_mrmc_study_v1.py",
            "server_preflight_audit": "scripts/audit_trustworthy_soz_workflow_server_v1.py",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trustworthy_soz_workflow_mrmc_study_v1_1_20260816"))
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    manifest = build_workflow_pack(
        report_dir=ROOT / "outputs/trustworthy_soz_clinical_reports_v32_20260816",
        qwen_dir=ROOT / "outputs/constrained_llm_soz_reports_v34_qwen36_20260816",
        signal_universe_path=ROOT / "outputs/deepsoz_target_independent_signal_universe_v1_20260812/deepsoz_target_independent_signal_universe.json",
        identity_manifest_path=ROOT / "outputs/labram_mrsc_target_free_identity_v16_20260812/manifest.json",
        v16_manifest_path=ROOT / "outputs/labram_identity_recovery_closed_replay_v16_20260812/manifest.json",
        v29_oof_path=ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815/oof_predictions.safetensors",
        far_audit_path=ROOT / "outputs/trustworthy_soz_ranking_distance_v22_6_20260815/result.json",
        factuality_pack=ROOT / "outputs/trustworthy_soz_report_reader_study_v1_20260815",
        language_pack=ROOT / "outputs/trustworthy_soz_template_vs_qwen_reader_study_v1_20260816",
        output_dir=output_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    print(f"output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
