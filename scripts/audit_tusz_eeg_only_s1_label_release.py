#!/usr/bin/env python3
"""Audit whether one TUSZ EEG-only S1 cohort may release supervision.

This is a fail-closed read-only gate.  It validates two independent complete
patient-bag reads and a later third-reader adjudication for every patient in
the requested cohort.  It does not materialize model targets, train a model,
or inspect another cohort's annotation files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_tusz_eeg_only_s1_reader_pack import (  # noqa: E402
    ADJUDICATION_SCHEMA,
    ANNOTATION_SCHEMA,
    COHORT_SIZES,
    DEFAULT_OUTPUT,
    PACK_SCHEMA,
    validate_completed_adjudication,
    validate_completed_annotation,
)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Expected JSONL object: {path}")
        case_id = str(value.get("case_id", "")).strip()
        if not case_id or case_id in result:
            raise ValueError(f"Empty or duplicated case ID: {path}")
        result[case_id] = value
    return result


def audit_cohort(reader_pack: Path, cohort: str) -> dict[str, object]:
    if cohort not in COHORT_SIZES:
        raise ValueError(f"Unknown S1 cohort: {cohort}")
    root = reader_pack.resolve(strict=True)
    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema_version") != PACK_SCHEMA:
        raise ValueError("Unexpected S1 reader-pack schema")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping) or access.get("new_reader_labels_opened") is not False:
        raise ValueError("Immutable blank-pack receipt changed")

    with (root / "patient_linkage.csv").open(encoding="utf-8", newline="") as stream:
        patient_rows = [
            row for row in csv.DictReader(stream) if row.get("cohort") == cohort
        ]
    expected_cases = {str(row["case_id"]) for row in patient_rows}
    if len(patient_rows) != COHORT_SIZES[cohort] or len(expected_cases) != len(patient_rows):
        raise ValueError("Cohort patient roster changed")
    roster = _load_json(root / "event_case_roster.json")
    if set(roster) != {
        str(row["case_id"])
        for row in csv.DictReader((root / "patient_linkage.csv").open(encoding="utf-8"))
    }:
        raise ValueError("Event case roster and patient linkage disagree")

    reader_a = _load_jsonl(root / f"reader_a_{cohort}.jsonl")
    reader_b = _load_jsonl(root / f"reader_b_{cohort}.jsonl")
    adjudication = _load_jsonl(root / f"adjudication_{cohort}.jsonl")
    for name, values in (
        ("reader_a", reader_a),
        ("reader_b", reader_b),
        ("adjudication", adjudication),
    ):
        if set(values) != expected_cases:
            raise ValueError(f"{name} case roster disagrees with the cohort")

    incomplete = []
    invalid: list[dict[str, str]] = []
    available = 0
    indeterminate_or_unavailable = 0
    exact_reader_positive_set_agreement = 0
    for case_id in sorted(expected_cases):
        left = reader_a[case_id]
        right = reader_b[case_id]
        final = adjudication[case_id]
        for row, schema, status_name in (
            (left, ANNOTATION_SCHEMA, "review_status"),
            (right, ANNOTATION_SCHEMA, "review_status"),
            (final, ADJUDICATION_SCHEMA, "adjudication_status"),
        ):
            if row.get("schema_version") != schema or row.get("cohort") != cohort:
                raise ValueError("Annotation schema/cohort binding changed")
            if row.get(status_name) != "completed":
                incomplete.append(case_id)
                break
        else:
            try:
                event_ids = set(str(value) for value in roster[case_id])
                validate_completed_annotation(left, event_ids)
                validate_completed_annotation(right, event_ids)
                validate_completed_adjudication(final, left, right)
            except (TypeError, ValueError) as error:
                invalid.append({"case_id": case_id, "error": str(error)})
                continue
            if final.get("target_availability") == "available":
                available += 1
            else:
                indeterminate_or_unavailable += 1
            if set(left["candidate_positive_electrodes"]) == set(
                right["candidate_positive_electrodes"]
            ):
                exact_reader_positive_set_agreement += 1

    complete_count = len(expected_cases) - len(set(incomplete)) - len(invalid)
    ready = not incomplete and not invalid
    return {
        "schema_version": "tusz_eeg_only_s1_label_release_audit_v1",
        "status": (
            "ready_for_cohort_specific_target_release"
            if ready
            else "not_ready_for_supervised_training"
        ),
        "cohort": cohort,
        "expected_patient_count": len(expected_cases),
        "valid_completed_patient_count": complete_count,
        "incomplete_patient_count": len(set(incomplete)),
        "invalid_patient_count": len(invalid),
        "invalid_records": invalid,
        "available_target_count": available,
        "indeterminate_or_unavailable_target_count": indeterminate_or_unavailable,
        "exact_reader_positive_set_agreement_count": (
            exact_reader_positive_set_agreement
        ),
        "ready": ready,
        "training_performed": False,
        "target_artifact_materialized": False,
        "other_cohort_annotation_files_opened": False,
        "private_loaded": False,
        "deepsoz_targets_loaded": False,
        "tusz_involvement_targets_loaded": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--reader-pack", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort", choices=tuple(COHORT_SIZES), default="s1_development")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(audit_cohort(args.reader_pack, args.cohort), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
