#!/usr/bin/env python3
"""Audit target-free event laterality against frozen displayed SOZ candidates.

This is a transparency audit, not a localization metric or a routing rule.  It
reads the sealed public event facts and already materialized qualified reports;
it never reads SOZ targets, private data, raw EEG, or model weights and cannot
change a score, threshold, candidate, or report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/target_free_oof_reports_v3_recovered_20260813.json"
DEFAULT_REPORTS = ROOT / "outputs/trustworthy_soz_qualified_reports_v22_20260815"
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_cross_layer_concordance_v22_5_20260815"
    / "result.json"
)

SCHEMA = "trustworthy_soz_cross_layer_concordance_audit_v22_5"

LEFT = frozenset(("FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"))
RIGHT = frozenset(("FP2", "F4", "F8", "T8", "C4", "P4", "P8", "O2"))
MIDLINE = frozenset(("FZ", "CZ", "PZ"))
STANDARD_19 = LEFT | RIGHT | MIDLINE


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


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


def channel_laterality(channel: str) -> str:
    if channel in LEFT:
        return "left"
    if channel in RIGHT:
        return "right"
    if channel in MIDLINE:
        return "midline"
    raise ValueError(f"unknown standard-19 channel: {channel!r}")


def first_visible_laterality(derivations: Sequence[object]) -> str:
    """Reduce first-visible bipolar edges without expanding them into SOZ labels.

    Midline endpoints do not erase a lateral endpoint (for example CZ-C4 is
    right-lateralized).  Evidence on both hemispheres is retained as bilateral.
    An empty edge list is unavailable rather than negative.
    """

    if not derivations:
        return "no_first_visible_derivation"
    lateral: set[str] = set()
    saw_midline = False
    for raw in derivations:
        edge = _text(raw, name="first-visible derivation")
        endpoints = edge.split("-")
        if len(endpoints) != 2 or any(endpoint not in STANDARD_19 for endpoint in endpoints):
            raise ValueError(f"invalid standard-19 bipolar derivation: {edge!r}")
        for endpoint in endpoints:
            side = channel_laterality(endpoint)
            if side == "midline":
                saw_midline = True
            else:
                lateral.add(side)
    if lateral == {"left"}:
        return "left"
    if lateral == {"right"}:
        return "right"
    if lateral == {"left", "right"}:
        return "bilateral"
    if saw_midline:
        return "midline_only"
    raise RuntimeError("derivation laterality reduction reached an impossible state")


def comparison_status(
    *, evidence_laterality: str, candidate_action: str, candidate_laterality: str | None
) -> str:
    if evidence_laterality == "evidence_unavailable":
        return "evidence_unavailable"
    if candidate_action != "display_candidate":
        return "candidate_not_displayed"
    if evidence_laterality in {
        "no_first_visible_derivation",
        "no_sustained_bipolar_change",
        "bilateral",
        "midline_only",
    }:
        return f"evidence_{evidence_laterality}"
    if candidate_laterality == "midline":
        return "candidate_midline_against_lateralized_evidence"
    if candidate_laterality == evidence_laterality:
        return "same_side_descriptive_concordance"
    if candidate_laterality in {"left", "right"}:
        return "contralateral_descriptive_tension"
    raise ValueError("displayed candidate laterality is missing or invalid")


def _patient_status(rows: Sequence[Mapping[str, object]], action: str) -> str:
    if action != "display_candidate":
        return "candidate_not_displayed"
    comparable = [
        str(row["comparison_status"])
        for row in rows
        if row["comparison_status"]
        in {
            "same_side_descriptive_concordance",
            "contralateral_descriptive_tension",
        }
    ]
    if not comparable:
        return "no_comparable_lateralized_event"
    values = set(comparable)
    if values == {"same_side_descriptive_concordance"}:
        return "concordance_only"
    if values == {"contralateral_descriptive_tension"}:
        return "tension_only"
    return "mixed_concordance_and_tension"


def audit(source: Path, report_directory: Path) -> dict[str, object]:
    source_value = _read_json(source)
    if source_value.get("schema_version") != "soz_target_free_oof_report_assembler_v3":
        raise ValueError("target-free event source schema drifted")
    access = _object(source_value.get("access_receipt"), name="source access receipt")
    for field in (
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "localization_scores_used_by_event_producer",
        "calibration_performed",
        "model_selection_performed",
        "raw_eeg_loaded_by_assembler",
        "training_performed",
    ):
        if access.get(field) is not False:
            raise ValueError(f"target-free source access boundary drifted: {field}")
    source_rows_raw = source_value.get("records")
    if not isinstance(source_rows_raw, list):
        raise TypeError("source records must be a list")

    manifest = _read_json(report_directory / "manifest.json")
    if manifest.get("schema_version") != "trustworthy_soz_qualified_reporting_manifest_v22":
        raise ValueError("qualified-report manifest schema drifted")
    report_access = _object(manifest.get("access_receipt"), name="report access receipt")
    for field in (
        "raw_eeg_loaded",
        "deepsoz_target_values_loaded",
        "private_target_ledger_loaded",
        "private_evaluation_rows_loaded",
        "training_performed",
        "model_selection_performed",
        "llm_used",
    ):
        if report_access.get(field) is not False:
            raise ValueError(f"qualified-report access boundary drifted: {field}")

    reports = _read_jsonl(report_directory / "public_event_reports.jsonl")
    report_by_event: dict[str, dict[str, object]] = {}
    for row in reports:
        event_id = _text(row.get("unit_id"), name="qualified report unit_id")
        if event_id in report_by_event:
            raise ValueError(f"duplicate qualified event report: {event_id}")
        report_by_event[event_id] = row

    event_rows: list[dict[str, object]] = []
    seen_source: set[str] = set()
    for raw in source_rows_raw:
        source_row = _object(raw, name="source row")
        event_id = _text(source_row.get("event_id"), name="source event_id")
        patient_id = _text(source_row.get("patient_id"), name="source patient_id")
        if event_id in seen_source:
            raise ValueError(f"duplicate source event: {event_id}")
        seen_source.add(event_id)
        report = report_by_event.get(event_id)
        if report is None or report.get("patient_id") != patient_id:
            raise ValueError(f"source/report identity mismatch: {event_id}")

        localization = _object(report.get("localization"), name="localization")
        action = _text(localization.get("action"), name="candidate action")
        displayed = localization.get("displayed_candidates")
        if not isinstance(displayed, list):
            raise TypeError("displayed_candidates must be a list")
        candidate_channel: str | None = None
        candidate_side: str | None = None
        if action == "display_candidate":
            if not displayed:
                raise ValueError("display action has no candidate")
            first = _object(displayed[0], name="displayed candidate")
            candidate_channel = _text(first.get("channel"), name="candidate channel")
            candidate_side = channel_laterality(candidate_channel)
        elif action not in {"localization_abstain", "localization_unavailable"}:
            raise ValueError(f"unknown localization action: {action}")
        elif displayed:
            raise ValueError("non-display action leaked candidates")

        facts_value = source_row.get("typed_facts")
        if facts_value is None:
            evidence_side = "evidence_unavailable"
            derivations: list[object] = []
        else:
            facts = _object(facts_value, name="typed_facts")
            phenotype = _object(facts.get("event_phenotype"), name="event phenotype")
            derivations_value = phenotype.get("first_visible_derivations")
            if derivations_value is None:
                reason_codes = phenotype.get("reason_codes")
                if (
                    phenotype.get("detected_bipolar_edge_count") != 0
                    or not isinstance(reason_codes, list)
                    or "no_sustained_bipolar_change" not in reason_codes
                ):
                    raise ValueError("null first-visible evidence lacks its explicit reason")
                derivations = []
                evidence_side = "no_sustained_bipolar_change"
            else:
                if not isinstance(derivations_value, list):
                    raise TypeError("first_visible_derivations must be a list or null")
                derivations = derivations_value
                evidence_side = first_visible_laterality(derivations)
        status = comparison_status(
            evidence_laterality=evidence_side,
            candidate_action=action,
            candidate_laterality=candidate_side,
        )
        event_rows.append(
            {
                "event_id": event_id,
                "patient_id": patient_id,
                "first_visible_derivations": derivations,
                "evidence_laterality": evidence_side,
                "candidate_action": action,
                "candidate_top1_channel": candidate_channel,
                "candidate_laterality": candidate_side,
                "comparison_status": status,
            }
        )
    if seen_source != set(report_by_event):
        raise ValueError("source and qualified event-report rosters differ")

    by_patient: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in event_rows:
        by_patient[str(row["patient_id"])].append(row)
    patient_rows: list[dict[str, object]] = []
    for patient_id, rows in sorted(by_patient.items()):
        actions = {str(row["candidate_action"]) for row in rows}
        channels = {row["candidate_top1_channel"] for row in rows}
        if len(actions) != 1 or len(channels) != 1:
            raise ValueError(f"patient candidate decision changes across events: {patient_id}")
        action = next(iter(actions))
        counts = Counter(str(row["comparison_status"]) for row in rows)
        patient_rows.append(
            {
                "patient_id": patient_id,
                "event_count": len(rows),
                "candidate_action": action,
                "candidate_top1_channel": next(iter(channels)),
                "patient_comparison_status": _patient_status(rows, action),
                "event_comparison_counts": dict(sorted(counts.items())),
            }
        )

    event_counts = Counter(str(row["comparison_status"]) for row in event_rows)
    evidence_counts = Counter(str(row["evidence_laterality"]) for row in event_rows)
    action_counts = Counter(str(row["candidate_action"]) for row in event_rows)
    patient_counts = Counter(
        str(row["patient_comparison_status"]) for row in patient_rows
    )
    comparable = (
        event_counts["same_side_descriptive_concordance"]
        + event_counts["contralateral_descriptive_tension"]
    )
    return {
        "schema_version": SCHEMA,
        "status": "completed_target_free_cross_layer_transparency_audit",
        "classification_policy": {
            "object_compared": (
                "first_visible_bipolar_edge_laterality_vs_frozen_patient_top1_"
                "scalp_electrode_candidate_laterality"
            ),
            "midline_endpoint_with_one_lateral_endpoint": "use_lateral_side",
            "both_hemispheres_present": "bilateral_indeterminate",
            "null_or_empty_first_visible_derivations": "unavailable_or_no_sustained_change_not_negative",
            "comparison_changes_candidate_or_abstention": False,
        },
        "roster": {
            "legacy_event_reports": len(event_rows),
            "legacy_event_report_patients": len(patient_rows),
            "current_localization_patients_with_legacy_events": sum(
                row["candidate_action"] != "localization_unavailable"
                for row in patient_rows
            ),
            "localization_unavailable_legacy_patients": sum(
                row["candidate_action"] == "localization_unavailable"
                for row in patient_rows
            ),
        },
        "event_level": {
            "comparison_status_counts": dict(sorted(event_counts.items())),
            "evidence_laterality_counts": dict(sorted(evidence_counts.items())),
            "candidate_action_counts": dict(sorted(action_counts.items())),
            "comparable_lateralized_display_events": comparable,
            "same_side_fraction_among_comparable_events": (
                event_counts["same_side_descriptive_concordance"] / comparable
                if comparable
                else None
            ),
            "unit_warning": "events_repeat_patient_level_candidate_and_are_not_independent",
        },
        "patient_level": {
            "comparison_status_counts": dict(sorted(patient_counts.items())),
            "patients": patient_rows,
        },
        "access_receipt": {
            "raw_eeg_loaded": False,
            "deepsoz_target_values_loaded": False,
            "private_data_loaded": False,
            "model_weights_loaded": False,
            "training_performed": False,
            "threshold_or_candidate_changed": False,
            "report_text_changed": False,
            "llm_used": False,
        },
        "scientific_boundary": {
            "is_localization_accuracy": False,
            "event_facts_causally_explain_h_only_candidate": False,
            "contralateral_tension_proves_candidate_error": False,
            "same_side_concordance_proves_clinical_factuality": False,
            "may_route_or_suppress_candidates": False,
            "allowed_use": "descriptive_report_transparency_and_reader_study_stratification_only",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.source, args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event_level": result["event_level"], "roster": result["roster"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
