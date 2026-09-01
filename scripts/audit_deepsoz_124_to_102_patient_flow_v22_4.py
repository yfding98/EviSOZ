#!/usr/bin/env python3
"""Close the DeepSOZ 124-to-102 patient flow and disambiguate the two 102 rosters."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL_UNIVERSE = (
    ROOT
    / "outputs/deepsoz_target_independent_signal_universe_v1_20260812/"
    "deepsoz_target_independent_signal_universe.json"
)
DEFAULT_TARGETS = (
    ROOT
    / "outputs/deepsoz_target_v2_identity_recovery_20260812/"
    "patient_targets_v2.csv"
)
DEFAULT_LOCALIZATION = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_20260812/manifest.json"
)
DEFAULT_AUXILIARY = (
    ROOT
    / "outputs/deepsoz_masked_variable_auxiliary_join_v1_20260812/"
    "deepsoz_masked_variable_auxiliary_target_join_v17.json"
)
DEFAULT_LEGACY_CORE = (
    ROOT
    / "outputs/deepsoz_signal_preflight_v1_20260808/"
    "deepsoz_signal_preflight.json"
)
DEFAULT_REPORTS = ROOT / "outputs/trustworthy_soz_qualified_reports_v22_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/deepsoz_124_to_102_patient_flow_v22_4_20260815"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected JSONL object: {path}")
            rows.append(value)
    return rows


def _read_targets(path: Path) -> dict[str, dict[str, str]]:
    with path.resolve(strict=True).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["deepsoz_patient_id"]: row for row in rows}
    if len(rows) != 124 or len(result) != 124:
        raise RuntimeError("DeepSOZ target roster must contain 124 unique patients")
    return result


def _legacy_patient_ids(receipt: dict[str, object]) -> set[str]:
    split_rows = receipt.get("eligible_split_patient_ids")
    if not isinstance(split_rows, list):
        raise TypeError("legacy eligible_split_patient_ids must be a list")
    result: set[str] = set()
    for row in split_rows:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], list):
            raise TypeError("legacy split patient row is malformed")
        result.update(str(value) for value in row[1])
    return result


def _reason_counts_by_patient(
    exclusions: list[object], patient_ids: set[str]
) -> dict[str, dict[str, int]]:
    result = {patient_id: {} for patient_id in patient_ids}
    for value in exclusions:
        if not isinstance(value, dict):
            raise TypeError("signal exclusion row must be an object")
        patient_id = str(value["patient_id"])
        if patient_id not in result:
            continue
        code = str(value["eligibility_code"])
        result[patient_id][code] = result[patient_id].get(code, 0) + 1
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    signal_artifact = _read_json(args.signal_universe)
    signal_receipt = signal_artifact.get("receipt")
    if not isinstance(signal_receipt, dict):
        raise TypeError("signal universe receipt must be an object")
    if signal_receipt.get("identity_patient_count") != 124:
        raise RuntimeError("signal universe identity roster drifted")
    identity_ids = {str(value) for value in signal_receipt["identity_patient_ids"]}
    events = signal_receipt.get("events")
    exclusions = signal_receipt.get("exclusions")
    if not isinstance(events, list) or not isinstance(exclusions, list):
        raise TypeError("signal universe event/exclusion rows must be lists")
    signal_ids = {str(row["patient_id"]) for row in events if isinstance(row, dict)}
    no_signal_ids = identity_ids - signal_ids
    if len(signal_ids) != 114 or len(no_signal_ids) != 10:
        raise RuntimeError("target-independent 114/10 signal flow drifted")

    targets = _read_targets(args.targets)
    if set(targets) != identity_ids:
        raise RuntimeError("signal and target identity rosters differ")
    target_eligible_ids = {
        patient_id
        for patient_id, row in targets.items()
        if row["eligible_for_localization"] == "1"
    }
    variable_ids = {
        patient_id
        for patient_id, row in targets.items()
        if "quarantine_variable_label" in row["exclusion_reason"]
    }
    no_strict_input_ids = {
        patient_id
        for patient_id, row in targets.items()
        if row["exclusion_reason"] == "quarantine_no_strict_input_event"
    }
    if len(target_eligible_ids) != 107 or len(variable_ids) != 11 or len(no_strict_input_ids) != 6:
        raise RuntimeError("DeepSOZ target-state counts drifted")

    localization = _read_json(args.localization)
    primary_ids = {str(value) for value in localization["patient_ids"]}
    partial_ids = {str(value) for value in localization["excluded_partial_reference_patients"]}
    if (
        localization.get("primary_patient_count") != 102
        or localization.get("primary_event_count") != 1145
        or partial_ids != {"258"}
    ):
        raise RuntimeError("v16 localization primary roster drifted")

    auxiliary = _read_json(args.auxiliary)
    auxiliary_receipt = auxiliary.get("receipt")
    if not isinstance(auxiliary_receipt, dict):
        raise TypeError("auxiliary join receipt must be an object")
    auxiliary_candidate_ids = {
        str(value) for value in auxiliary_receipt["candidate_patient_ids"]
    }
    auxiliary_admitted_ids = {
        str(value) for value in auxiliary_receipt["admitted_patient_ids"]
    }
    auxiliary_excluded_ids = {
        str(value) for value in auxiliary_receipt["excluded_patient_ids"]
    }
    if (
        auxiliary_candidate_ids != signal_ids & variable_ids
        or auxiliary_admitted_ids | auxiliary_excluded_ids != auxiliary_candidate_ids
        or auxiliary_admitted_ids & auxiliary_excluded_ids
        or len(auxiliary_admitted_ids) != 9
        or len(auxiliary_excluded_ids) != 2
    ):
        raise RuntimeError("variable-target auxiliary flow drifted")

    legacy = _read_json(args.legacy_core)
    legacy_receipt = legacy.get("receipt")
    if not isinstance(legacy_receipt, dict):
        raise TypeError("legacy signal receipt must be an object")
    legacy_ids = _legacy_patient_ids(legacy_receipt)
    if legacy_receipt.get("eligible_event_count") != 988 or len(legacy_ids) != 102:
        raise RuntimeError("legacy 102/988 core drifted")

    report_patient_rows = _read_jsonl(args.reports / "public_patient_reports.jsonl")
    report_event_rows = _read_jsonl(args.reports / "public_event_reports.jsonl")
    report_patient_ids = {str(row["patient_id"]) for row in report_patient_rows}
    report_event_patient_ids = {str(row["patient_id"]) for row in report_event_rows}
    if (
        report_patient_ids != primary_ids
        or report_event_patient_ids != legacy_ids
        or len(report_patient_rows) != 102
        or len(report_event_rows) != 988
    ):
        raise RuntimeError("v22 patient/event report roster contract drifted")

    categories = {
        "current_localization_primary_c18_stable": primary_ids,
        "signal_eligible_target_stable_but_partial_c18": partial_ids,
        "signal_eligible_variable_target_auxiliary_admitted": auxiliary_admitted_ids,
        "signal_eligible_variable_target_not_admissible": auxiliary_excluded_ids,
        "no_signal_but_target_stable": no_signal_ids & target_eligible_ids,
        "no_signal_and_no_strict_target_input": no_signal_ids & no_strict_input_ids,
    }
    union: set[str] = set()
    for values in categories.values():
        if union & values:
            raise RuntimeError("124-patient flow categories overlap")
        union |= values
    if union != identity_ids:
        raise RuntimeError("124-patient flow categories are not exhaustive")

    no_signal_reasons = _reason_counts_by_patient(exclusions, no_signal_ids)
    if sum(sum(row.values()) for row in no_signal_reasons.values()) != 19:
        raise RuntimeError("no-signal candidate-event reasons drifted")
    if targets["258"]["benchmark_state_O2"] != "missing":
        raise RuntimeError("patient 258 is no longer the missing-O2 partial reference")

    payload: dict[str, object] = {
        "schema_version": "deepsoz_124_to_102_patient_flow_audit_v22_4",
        "status": "completed_exhaustive_patient_flow_and_dual_102_roster_audit",
        "source_overlay": {
            "records": int(signal_receipt["identity_record_count"]),
            "patients": len(identity_ids),
        },
        "target_independent_signal_view": {
            "candidate_events": int(signal_receipt["candidate_event_count"]),
            "eligible_events": int(signal_receipt["eligible_event_count"]),
            "eligible_patients": len(signal_ids),
            "patients_without_eligible_signal": len(no_signal_ids),
        },
        "target_view": {
            "stable_eligible_patients": len(target_eligible_ids),
            "variable_label_patients": len(variable_ids),
            "no_strict_input_patients": len(no_strict_input_ids),
        },
        "mutually_exclusive_patient_flow": {
            name: {"count": len(values), "patient_ids": sorted(values)}
            for name, values in categories.items()
        },
        "no_signal_patient_reason_counts": no_signal_reasons,
        "current_localization_primary": {
            "patients": len(primary_ids),
            "events": int(localization["primary_event_count"]),
            "roster": "identity_v16_c18_complete_excludes_258_includes_10489",
        },
        "legacy_event_evidence_core": {
            "patients": len(legacy_ids),
            "events": int(legacy_receipt["eligible_event_count"]),
            "roster": "legacy_preflight_includes_258_excludes_10489",
        },
        "dual_102_roster_relation": {
            "intersection_count": len(primary_ids & legacy_ids),
            "localization_only_patient_ids": sorted(primary_ids - legacy_ids),
            "legacy_event_core_only_patient_ids": sorted(legacy_ids - primary_ids),
            "same_roster": primary_ids == legacy_ids,
        },
        "v22_report_join": {
            "patient_reports": len(report_patient_rows),
            "patient_report_roster": "current_localization_primary",
            "event_reports": len(report_event_rows),
            "event_report_roster": "legacy_event_evidence_core",
            "localization_only_patient_has_no_event_report": sorted(
                report_patient_ids - report_event_patient_ids
            ),
            "legacy_only_patient_events_are_localization_unavailable": all(
                row["localization"]["action"] == "localization_unavailable"
                for row in report_event_rows
                if str(row["patient_id"]) in legacy_ids - primary_ids
            ),
        },
        "scientific_decision": {
            "unrecovered_stable_complete_c18_patients_remaining": 0,
            "variable_labels_may_be_converted_to_gold": False,
            "partial_reference_may_be_zero_filled": False,
            "no_signal_patients_may_enter_localization": False,
            "additional_current_cohort_training_or_threshold_search_authorized": False,
            "new_same_endpoint_patients_still_required": True,
        },
        "access_receipt": {
            "raw_eeg_loaded": False,
            "model_predictions_loaded": False,
            "training_performed": False,
            "model_or_threshold_selection_performed": False,
            "private_data_accessed": False,
        },
    }

    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--signal-universe", type=Path, default=DEFAULT_SIGNAL_UNIVERSE)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--localization", type=Path, default=DEFAULT_LOCALIZATION)
    parser.add_argument("--auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    parser.add_argument("--legacy-core", type=Path, default=DEFAULT_LEGACY_CORE)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "patient_flow_counts": {
                    key: value["count"]
                    for key, value in result["mutually_exclusive_patient_flow"].items()
                },
                "dual_102_roster_relation": result["dual_102_roster_relation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
