#!/usr/bin/env python3
"""Build a target-blind two-reader study pack for qualified SOZ reports.

This builder reads only already-sealed target-free report artifacts, the
target-independent TUSZ signal universe, and the frozen identity-v16 event
roster.  It has no input for DeepSOZ labels, private labels, or correctness
metrics.  The resulting pack separates event-clause factuality from
patient-level candidate usefulness and keeps the two patient samples disjoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIRECTORY = (
    ROOT / "outputs/trustworthy_soz_qualified_reports_v22_20260815"
)
DEFAULT_SIGNAL_UNIVERSE = (
    ROOT
    / "outputs/deepsoz_target_independent_signal_universe_v1_20260812"
    / "deepsoz_target_independent_signal_universe.json"
)
DEFAULT_IDENTITY_MANIFEST = (
    ROOT / "outputs/labram_mrsc_target_free_identity_v16_20260812/manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_report_reader_study_v1_20260815"
)

PACK_SCHEMA = "trustworthy_soz_report_reader_study_pack_v1"
ANNOTATION_SCHEMA = "trustworthy_soz_report_reader_annotation_v1"
REPORT_SCHEMA = "trustworthy_soz_qualified_report_v22"

C18 = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
)

EVENT_STATUS_QUOTAS = {
    "event_phenotype_reportable_display_candidate_facts_locked": 16,
    "event_phenotype_reportable_localization_abstain_facts_locked": 6,
    "event_phenotype_abstained_display_candidate_facts_locked": 6,
    "event_phenotype_abstained_localization_abstain_facts_locked": 4,
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.resolve(strict=True).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _objects_by_unique_key(
    rows: Iterable[Mapping[str, object]], key: str, *, name: str
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for raw in rows:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} has invalid {key}")
        if value in result:
            raise ValueError(f"duplicate {name} {key}: {value}")
        result[value] = dict(raw)
    return result


def _shuffled(values: Iterable[object], seed: int) -> list[object]:
    result = list(values)
    result.sort(key=lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True))
    random.Random(seed).shuffle(result)
    return result


def _round_robin_select(
    grouped: Mapping[tuple[str, ...], Sequence[dict[str, object]]],
    count: int,
    *,
    seed: int,
) -> list[dict[str, object]]:
    queues: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for group_index, (group, values) in enumerate(sorted(grouped.items())):
        queues[group] = [
            dict(value)
            for value in _shuffled(values, seed + 101 * (group_index + 1))
        ]
    selected: list[dict[str, object]] = []
    group_order = list(_shuffled(queues, seed + 17))
    while len(selected) < count:
        progressed = False
        for group in group_order:
            queue = queues[group]
            if queue:
                selected.append(queue.pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"only {len(selected)} rows available for quota {count}")
    return selected


def _validate_report(row: Mapping[str, object], *, expected_cohort: str) -> None:
    if row.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("qualified report schema drifted")
    if row.get("cohort") != expected_cohort:
        raise ValueError("qualified report cohort drifted")
    if row.get("facts_locked") is not True or row.get("llm_used") is not False:
        raise ValueError("report must be fact-locked and non-LLM-materialized")
    text = row.get("clinical_text_zh")
    clauses = row.get("clauses")
    if not isinstance(text, str) or not text or not isinstance(clauses, list) or not clauses:
        raise ValueError("qualified report lacks text or clauses")
    if len(clauses) != len(row.get("sentence_fact_map", [])):
        raise ValueError("clause-to-fact mapping is incomplete")
    clause_texts: list[str] = []
    for clause in clauses:
        if not isinstance(clause, dict):
            raise TypeError("qualified report clause is not an object")
        clause_text = clause.get("text")
        fact_paths = clause.get("fact_paths")
        if (
            not isinstance(clause_text, str)
            or not clause_text
            or not isinstance(fact_paths, list)
            or not fact_paths
            or not all(isinstance(path, str) and path for path in fact_paths)
        ):
            raise ValueError("qualified report clause is not fact-grounded")
        clause_texts.append(clause_text.rstrip("。"))
    if "。".join(clause_texts) + "。" != text:
        raise ValueError("qualified report text does not reconstruct from clauses")


def _signal_events(
    signal_universe: Mapping[str, object], identity: Mapping[str, object]
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    if signal_universe.get("schema_version") != (
        "soz_deepsoz_target_independent_signal_universe_artifact_v1"
    ):
        raise ValueError("signal universe schema drifted")
    receipt = signal_universe.get("receipt")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != (
        "soz_deepsoz_target_independent_signal_universe_v1"
    ):
        raise ValueError("signal universe receipt schema drifted")
    axes = receipt.get("lineage_axes")
    if not isinstance(axes, dict):
        raise ValueError("signal universe lacks target-independence axes")
    for axis in ("direct_target_values", "target_supervised_model"):
        state = axes.get(axis)
        if not isinstance(state, dict) or state.get("used") is not False:
            raise ValueError(f"signal universe is not target-independent: {axis}")
    if identity.get("schema_version") != (
        "soz_labram_mrsc_target_free_identity_v16_descriptive_v1"
    ):
        raise ValueError("identity-v16 schema drifted")
    access = identity.get("access_receipt")
    if not isinstance(access, dict):
        raise ValueError("identity-v16 access receipt missing")
    forbidden_access = (
        "deepsoz_target_values_loaded",
        "private_target_values_loaded",
        "target_tensor_values_loaded",
    )
    if any(access.get(field) is not False for field in forbidden_access):
        raise ValueError("identity-v16 target access boundary failed")
    event_ids_raw = identity.get("event_ids")
    if not isinstance(event_ids_raw, list) or not all(
        isinstance(value, str) and value for value in event_ids_raw
    ):
        raise ValueError("identity-v16 event roster missing")
    event_ids = set(event_ids_raw)
    events_raw = receipt.get("events")
    if not isinstance(events_raw, list):
        raise ValueError("signal universe events missing")
    by_event: dict[str, dict[str, object]] = {}
    by_patient: dict[str, list[dict[str, object]]] = defaultdict(list)
    for raw in events_raw:
        if not isinstance(raw, dict) or raw.get("event_id") not in event_ids:
            continue
        event_id = raw.get("event_id")
        patient_id = raw.get("patient_id")
        relative = raw.get("relative_edf_path")
        if not all(isinstance(value, str) and value for value in (event_id, patient_id, relative)):
            raise ValueError("signal event identity/path is invalid")
        if event_id in by_event:
            raise ValueError(f"duplicate signal event: {event_id}")
        row = dict(raw)
        by_event[event_id] = row
        by_patient[patient_id].append(row)
    if len(by_event) != identity.get("event_count"):
        raise ValueError("identity-v16 event roster does not close against signal universe")
    if len(by_patient) != identity.get("patient_count"):
        raise ValueError("identity-v16 patient roster does not close against signal universe")
    for events in by_patient.values():
        events.sort(key=lambda row: (float(row["global_t0_sec"]), str(row["event_id"])))
    return by_event, dict(by_patient)


def _candidate_sample(
    patient_reports: Mapping[str, dict[str, object]],
    signal_by_patient: Mapping[str, Sequence[dict[str, object]]],
    *,
    total: int,
    abstain_count: int,
    max_events: int,
    seed: int,
) -> list[dict[str, object]]:
    if not 0 < abstain_count < total:
        raise ValueError("candidate abstain quota must be between zero and total")
    grouped: dict[str, dict[tuple[str, ...], list[dict[str, object]]]] = {
        "display_candidate": defaultdict(list),
        "localization_abstain": defaultdict(list),
    }
    for patient_id, report in patient_reports.items():
        events = signal_by_patient.get(patient_id)
        if not events or len(events) > max_events:
            continue
        localization = report.get("localization")
        if not isinstance(localization, dict):
            raise ValueError("patient report localization is invalid")
        action = localization.get("action")
        if action not in grouped:
            continue
        event_count_group = str(min(len(events), 3))
        if action == "display_candidate":
            region = localization.get("top1_region_projection_zh")
            if not isinstance(region, str) or not region:
                raise ValueError("displayed candidate lacks a region projection")
            group = (region, event_count_group)
        else:
            group = (event_count_group,)
        grouped[action][group].append(
            {"patient_id": patient_id, "report": report, "events": list(events)}
        )
    selected = _round_robin_select(
        grouped["localization_abstain"], abstain_count, seed=seed + 1
    )
    selected.extend(
        _round_robin_select(
            grouped["display_candidate"], total - abstain_count, seed=seed + 2
        )
    )
    return [dict(value) for value in _shuffled(selected, seed + 3)]


def _event_sample(
    event_reports: Sequence[dict[str, object]],
    signal_by_event: Mapping[str, dict[str, object]],
    excluded_patients: set[str],
    *,
    quotas: Mapping[str, int],
    seed: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    consumed_patients = set(excluded_patients)
    for status_index, (status, quota) in enumerate(quotas.items()):
        per_patient: dict[str, list[dict[str, object]]] = defaultdict(list)
        for report in event_reports:
            if report.get("report_status") != status:
                continue
            patient_id = report.get("patient_id")
            event_id = report.get("unit_id")
            if (
                not isinstance(patient_id, str)
                or patient_id in consumed_patients
                or not isinstance(event_id, str)
                or event_id not in signal_by_event
            ):
                continue
            per_patient[patient_id].append(report)
        patient_order = _shuffled(per_patient, seed + 100 * (status_index + 1))
        chosen_for_status = 0
        for patient_id_raw in patient_order:
            patient_id = str(patient_id_raw)
            if patient_id in consumed_patients:
                continue
            reports = _shuffled(
                per_patient[patient_id], seed + 1000 * (status_index + 1) + chosen_for_status
            )
            report = dict(reports[0])
            selected.append(
                {
                    "patient_id": patient_id,
                    "report": report,
                    "events": [signal_by_event[str(report["unit_id"])]],
                }
            )
            consumed_patients.add(patient_id)
            chosen_for_status += 1
            if chosen_for_status == quota:
                break
        if chosen_for_status != quota:
            raise ValueError(
                f"event status {status} supplies {chosen_for_status}, requires {quota}"
            )
    return [dict(value) for value in _shuffled(selected, seed + 9999)]


def _report_card(
    case_id: str, layer: str, report: Mapping[str, object], linked_event_count: int
) -> dict[str, object]:
    raw_clauses = report.get("clauses")
    if not isinstance(raw_clauses, list):
        raise TypeError("report clauses must be a list")
    clauses: list[dict[str, object]] = []
    for index, raw in enumerate(raw_clauses, start=1):
        if not isinstance(raw, dict):
            raise TypeError("report clause must be an object")
        clause_type = raw.get("type")
        text = raw.get("text")
        if not isinstance(clause_type, str) or not isinstance(text, str):
            raise ValueError("report clause lacks type/text")
        clauses.append(
            {
                "clause_id": f"{case_id}-C{index:02d}",
                "clause_type": clause_type,
                "text_zh": text,
            }
        )
    localization = report.get("localization")
    if not isinstance(localization, dict) or localization.get("action") not in {
        "display_candidate",
        "localization_abstain",
    }:
        raise ValueError("report card localization action is invalid")
    return {
        "schema_version": "trustworthy_soz_report_reader_card_v1",
        "case_id": case_id,
        "layer": layer,
        "linked_event_count": linked_event_count,
        "report_status": report.get("report_status"),
        "candidate_display_action": localization["action"],
        "clinical_text_zh": report.get("clinical_text_zh"),
        "clauses": clauses,
        "display_note_zh": (
            "先完成原始EEG盲评并锁定第一阶段字段，再显示本报告。"
            "候选有用性不是SOZ正确率，医生填写内容不得回流训练。"
        ),
    }


def _blank_annotation(
    card: Mapping[str, object], *, reviewer_id: str, order: int
) -> dict[str, object]:
    clauses = card.get("clauses")
    if not isinstance(clauses, list):
        raise TypeError("reader card clauses must be a list")
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "case_id": card["case_id"],
        "layer": card["layer"],
        "presentation_order": order,
        "reviewer_id": reviewer_id,
        "review_status": "unreviewed",
        "raw_phase_locked": False,
        "raw_phase_locked_at": None,
        "report_revealed_at": None,
        "reviewed_event_case_ids": [],
        "raw_only_signal_assessable": None,
        "raw_only_assessability_reason": None,
        "raw_only_review_duration_sec": None,
        "raw_only_key_findings_not_for_model": "",
        "raw_only_candidate_action": None,
        "raw_only_candidate_channels": [],
        "report_review_duration_sec": None,
        "clause_ratings": [
            {
                "clause_id": clause["clause_id"],
                "clause_type": clause["clause_type"],
                "support": None,
                "clinically_material_error": None,
                "proposed_action": None,
                "correction_text_not_for_model": "",
            }
            for clause in clauses
        ],
        "important_omission": None,
        "omission_categories": [],
        "omission_text_not_for_model": "",
        "candidate_eeg_consistency_likert_1_to_5": None,
        "candidate_review_usefulness_likert_1_to_5": None,
        "candidate_burden_acceptable": None,
        "abstention_display_appropriate": None,
        "candidate_action_after_report": None,
        "candidate_channels_after_report": [],
        "safe_without_edit": None,
        "overall_modification_count": None,
        "overstatement_present": None,
        "review_completed_at": None,
        "free_text_note_not_for_model": "",
    }


def _annotation_json_schema() -> dict[str, object]:
    nullable_bool = {"type": ["boolean", "null"]}
    nullable_number = {"type": ["number", "null"], "minimum": 0}
    nullable_action = {
        "type": ["string", "null"],
        "enum": ["display_candidate", "abstain", "indeterminate", None],
    }
    channel_array = {
        "type": "array",
        "items": {"type": "string", "enum": list(C18)},
        "uniqueItems": True,
    }
    properties: dict[str, object] = {
        "schema_version": {"const": ANNOTATION_SCHEMA},
        "case_id": {"type": "string", "minLength": 1},
        "layer": {"enum": ["event_clause_factuality", "patient_candidate_utility"]},
        "presentation_order": {"type": "integer", "minimum": 1},
        "reviewer_id": {"type": "string", "minLength": 1},
        "review_status": {"enum": ["unreviewed", "completed"]},
        "raw_phase_locked": {"type": "boolean"},
        "raw_phase_locked_at": {"type": ["string", "null"]},
        "report_revealed_at": {"type": ["string", "null"]},
        "reviewed_event_case_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "raw_only_signal_assessable": nullable_bool,
        "raw_only_assessability_reason": {
            "type": ["string", "null"],
            "enum": [
                "assessable",
                "artifact_obscured",
                "recording_truncated",
                "missing_channel_or_reference",
                "insufficient_context",
                "other",
                None,
            ],
        },
        "raw_only_review_duration_sec": nullable_number,
        "raw_only_key_findings_not_for_model": {"type": "string"},
        "raw_only_candidate_action": nullable_action,
        "raw_only_candidate_channels": channel_array,
        "report_review_duration_sec": nullable_number,
        "clause_ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "clause_id",
                    "clause_type",
                    "support",
                    "clinically_material_error",
                    "proposed_action",
                    "correction_text_not_for_model",
                ],
                "properties": {
                    "clause_id": {"type": "string"},
                    "clause_type": {"type": "string"},
                    "support": {
                        "type": ["string", "null"],
                        "enum": [
                            "supported",
                            "partially_supported",
                            "unsupported",
                            "not_assessable",
                            None,
                        ],
                    },
                    "clinically_material_error": nullable_bool,
                    "proposed_action": {
                        "type": ["string", "null"],
                        "enum": [
                            "retain",
                            "minor_edit",
                            "major_edit",
                            "delete",
                            "not_assessable",
                            None,
                        ],
                    },
                    "correction_text_not_for_model": {"type": "string"},
                },
            },
        },
        "important_omission": nullable_bool,
        "omission_categories": {
            "type": "array",
            "uniqueItems": True,
            "items": {
                "enum": [
                    "signal_quality",
                    "event_timing",
                    "spatial_pattern",
                    "rhythm_frequency",
                    "evolution",
                    "later_visible_change",
                    "artifact",
                    "uncertainty",
                    "clinical_boundary",
                    "other",
                ]
            },
        },
        "omission_text_not_for_model": {"type": "string"},
        "candidate_eeg_consistency_likert_1_to_5": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": 5,
        },
        "candidate_review_usefulness_likert_1_to_5": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": 5,
        },
        "candidate_burden_acceptable": nullable_bool,
        "abstention_display_appropriate": {
            "type": ["string", "null"],
            "enum": ["yes", "no", "indeterminate", "not_applicable", None],
        },
        "candidate_action_after_report": nullable_action,
        "candidate_channels_after_report": channel_array,
        "safe_without_edit": nullable_bool,
        "overall_modification_count": {
            "type": ["integer", "null"],
            "minimum": 0,
        },
        "overstatement_present": nullable_bool,
        "review_completed_at": {"type": ["string", "null"]},
        "free_text_note_not_for_model": {"type": "string"},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": ANNOTATION_SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def build_reader_study_pack(
    *,
    report_directory: Path,
    signal_universe_path: Path,
    identity_manifest_path: Path,
    output_directory: Path,
    candidate_count: int = 24,
    candidate_abstain_count: int = 6,
    candidate_max_events: int = 5,
    event_status_quotas: Mapping[str, int] = EVENT_STATUS_QUOTAS,
    seed: int = 20260815,
) -> dict[str, object]:
    report_manifest = _read_json(report_directory / "manifest.json")
    if report_manifest.get("schema_version") != (
        "trustworthy_soz_qualified_reporting_manifest_v22"
    ):
        raise ValueError("qualified report manifest schema drifted")
    access = report_manifest.get("access_receipt")
    if not isinstance(access, dict) or any(
        access.get(field) is not False
        for field in ("deepsoz_target_values_loaded", "private_target_ledger_loaded")
    ):
        raise ValueError("qualified report target-access boundary failed")
    patient_rows = _read_jsonl(report_directory / "public_patient_reports.jsonl")
    event_rows = _read_jsonl(report_directory / "public_event_reports.jsonl")
    for row in patient_rows:
        _validate_report(row, expected_cohort="public_deepsoz_development_patient_report")
    for row in event_rows:
        _validate_report(row, expected_cohort="public_deepsoz_development_event_report")
    patient_reports = _objects_by_unique_key(
        patient_rows, "patient_id", name="public patient report"
    )
    signal_universe = _read_json(signal_universe_path)
    identity = _read_json(identity_manifest_path)
    signal_by_event, signal_by_patient = _signal_events(signal_universe, identity)
    if set(patient_reports) != set(signal_by_patient):
        raise ValueError("patient report and identity-v16 signal rosters differ")

    candidate = _candidate_sample(
        patient_reports,
        signal_by_patient,
        total=candidate_count,
        abstain_count=candidate_abstain_count,
        max_events=candidate_max_events,
        seed=seed,
    )
    candidate_patient_ids = {str(row["patient_id"]) for row in candidate}
    event = _event_sample(
        event_rows,
        signal_by_event,
        candidate_patient_ids,
        quotas=event_status_quotas,
        seed=seed + 50000,
    )
    event_patient_ids = {str(row["patient_id"]) for row in event}
    if candidate_patient_ids & event_patient_ids:
        raise AssertionError("reader-study layers are not patient-disjoint")

    target = output_directory
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        cards: list[dict[str, object]] = []
        linkages: list[dict[str, object]] = []
        case_patient_ids: dict[str, str] = {}
        for layer, prefix, rows in (
            ("event_clause_factuality", "RPT-E", event),
            ("patient_candidate_utility", "RPT-P", candidate),
        ):
            for case_index, row in enumerate(rows, start=1):
                case_id = f"{prefix}-{case_index:03d}"
                patient_id = str(row["patient_id"])
                report = row["report"]
                events = row["events"]
                if not isinstance(report, dict) or not isinstance(events, list):
                    raise TypeError("selected reader case is malformed")
                card = _report_card(case_id, layer, report, len(events))
                cards.append(card)
                case_patient_ids[case_id] = patient_id
                for event_index, signal in enumerate(events, start=1):
                    event_id = signal.get("event_id")
                    relative = signal.get("relative_edf_path")
                    t0 = signal.get("global_t0_sec")
                    stop = signal.get("global_stop_sec")
                    if (
                        not isinstance(event_id, str)
                        or not isinstance(relative, str)
                        or not isinstance(t0, (int, float))
                        or not isinstance(stop, (int, float))
                    ):
                        raise ValueError("selected signal linkage is malformed")
                    linkages.append(
                        {
                            "case_id": case_id,
                            "layer": layer,
                            "public_patient_id": patient_id,
                            "event_bundle_index": event_index,
                            "public_event_id": event_id,
                            "relative_edf_path": relative,
                            "global_event_t0_sec": float(t0),
                            "global_event_stop_sec": float(stop),
                            "suggested_review_window_start_sec": max(0.0, float(t0) - 30.0),
                            "suggested_review_window_stop_sec_unclamped": float(stop) + 60.0,
                            "index_event_for_event_clause": (
                                layer == "event_clause_factuality"
                            ),
                        }
                    )

        _write_jsonl(temporary / "report_cards.jsonl", cards)
        with (temporary / "case_linkage.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(linkages[0]))
            writer.writeheader()
            writer.writerows(linkages)
        _write_json(temporary / "annotation_schema.json", _annotation_json_schema())
        for reader_index, reviewer_id in enumerate(("reader_a", "reader_b"), start=1):
            ordered = [dict(card) for card in cards]
            random.Random(seed + 70000 + reader_index).shuffle(ordered)
            annotations = [
                _blank_annotation(card, reviewer_id=reviewer_id, order=index)
                for index, card in enumerate(ordered, start=1)
            ]
            _write_jsonl(temporary / f"{reviewer_id}_annotations.jsonl", annotations)

        event_status_counts: dict[str, int] = defaultdict(int)
        candidate_status_counts: dict[str, int] = defaultdict(int)
        action_counts: dict[str, int] = defaultdict(int)
        for card in cards:
            destination = (
                event_status_counts
                if card["layer"] == "event_clause_factuality"
                else candidate_status_counts
            )
            destination[str(card["report_status"])] += 1
        for row in candidate:
            localization = row["report"].get("localization")
            if not isinstance(localization, dict):
                raise TypeError("candidate localization is malformed")
            action_counts[str(localization.get("action"))] += 1
        manifest: dict[str, object] = {
            "schema_version": PACK_SCHEMA,
            "status": "empty_target_blind_two_reader_pack_ready",
            "sampling_seed": seed,
            "counts": {
                "event_clause_factuality_cases": len(event),
                "patient_candidate_utility_cases": len(candidate),
                "unique_patients": len(case_patient_ids),
                "linked_signal_events": len(linkages),
                "readers": 2,
                "event_report_statuses": dict(sorted(event_status_counts.items())),
                "candidate_report_statuses": dict(
                    sorted(candidate_status_counts.items())
                ),
                "candidate_actions": dict(sorted(action_counts.items())),
            },
            "sampling_contract": {
                "event_layer": (
                    "one_event_per_patient_stratified_by_reportability_and_"
                    "candidate_or_abstention_status"
                ),
                "candidate_layer": (
                    "complete_identity_v16_event_bag_for_patients_with_at_most_"
                    f"{candidate_max_events}_events;stratified_by_action_region_and_event_count"
                ),
                "event_status_quotas": dict(event_status_quotas),
                "candidate_total": candidate_count,
                "candidate_abstain_count": candidate_abstain_count,
                "layers_patient_disjoint": True,
                "correctness_or_target_stratification": False,
            },
            "blinding_contract": {
                "raw_eeg_review_precedes_report_reveal": True,
                "deepsoz_target_hidden": True,
                "tusz_channel_time_annotation_hidden": True,
                "private_data_hidden": True,
                "model_scores_margin_threshold_hidden": True,
                "other_reader_annotation_hidden": True,
                "case_linkage_is_data_manager_only": True,
            },
            "access_receipt": {
                "qualified_target_free_report_rows_loaded": True,
                "target_independent_signal_identity_and_paths_loaded": True,
                "identity_v16_target_free_event_roster_loaded": True,
                "raw_eeg_bytes_loaded": False,
                "deepsoz_target_values_loaded": False,
                "private_eeg_or_target_loaded": False,
                "tusz_channel_time_target_values_loaded": False,
                "model_correctness_or_outcome_metrics_loaded": False,
                "training_calibration_or_model_selection_performed": False,
                "llm_annotation_performed": False,
            },
            "analysis_boundary": {
                "event_clause_support_is_clinical_factuality_not_soz_accuracy": True,
                "candidate_usefulness_is_not_candidate_correctness": True,
                "reader_candidates_are_not_gold_or_training_labels": True,
                "current_reports_may_not_be_rewritten_from_private_or_public_outcomes": True,
                "independent_clinician_results_pending": True,
            },
            "files": {
                "report_cards": "report_cards.jsonl",
                "data_manager_case_linkage": "case_linkage.csv",
                "annotation_schema": "annotation_schema.json",
                "reader_templates": [
                    "reader_a_annotations.jsonl",
                    "reader_b_annotations.jsonl",
                ],
                "local_reader_server": (
                    "scripts/serve_trustworthy_soz_report_reader_study_v1.py"
                ),
                "local_reader_launcher": (
                    "scripts/run_trustworthy_soz_report_reader_study_v1.sh"
                ),
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            "# Trustworthy SOZ qualified-report reader study v1\n\n"
            "This is an empty, target-blind two-reader pack. It contains 32 "
            "event-clause factuality cases and 24 patient-candidate utility "
            "cases. The two layers are patient-disjoint.\n\n"
            "For every case, the viewer must first show raw EEG only. Lock the "
            "raw-only fields before revealing `report_cards.jsonl`. Never show "
            "DeepSOZ labels, TUSZ channel annotations, correctness metrics, "
            "private labels, model scores, or the other reader's annotation.\n\n"
            "`case_linkage.csv` is data-manager-only and must not be distributed "
            "with TUSZ data. Candidate usefulness is not SOZ accuracy, and any "
            "reader-entered candidate or free text is never a model input, gold "
            "label, calibration target, or report-rewrite source. Suggested "
            "window stops are unclamped hints; the viewer must clamp them to "
            "the real EDF duration.\n\n"
            "Launch independent local readers from the repository root:\n\n"
            "```bash\n"
            "scripts/run_trustworthy_soz_report_reader_study_v1.sh reader_a READER_A_ID\n"
            "scripts/run_trustworthy_soz_report_reader_study_v1.sh reader_b READER_B_ID\n"
            "```\n\n"
            "Reader A uses http://127.0.0.1:8781 and reader B uses "
            "http://127.0.0.1:8782. The server withholds the report card until "
            "all linked events are marked reviewed and the raw-only phase is "
            "validated and locked. Completed records are immutable.\n",
            encoding="utf-8",
        )
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    parser.add_argument("--signal-universe", type=Path, default=DEFAULT_SIGNAL_UNIVERSE)
    parser.add_argument("--identity-manifest", type=Path, default=DEFAULT_IDENTITY_MANIFEST)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--candidate-abstain-count", type=int, default=6)
    parser.add_argument("--candidate-max-events", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    result = build_reader_study_pack(
        report_directory=args.report_directory,
        signal_universe_path=args.signal_universe,
        identity_manifest_path=args.identity_manifest,
        output_directory=args.output_directory,
        candidate_count=args.candidate_count,
        candidate_abstain_count=args.candidate_abstain_count,
        candidate_max_events=args.candidate_max_events,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
