#!/usr/bin/env python3
"""Build a label-fresh locked S1 reader pack from official TUSZ dev/eval.

The source is the completed target-free signal-eligibility audit.  All
eligible non-DeepSOZ patients from official dev/eval are placed in one locked
extension before any expert label is opened.  The builder exports event
navigation and blank two-reader/third-adjudicator templates only; it does not
open TUSZ channel involvement, DeepSOZ target values, model predictions, or
private data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_tusz_non_deepsoz_s1_signal_eligibility import (  # noqa: E402
    SCHEMA_VERSION as AUDIT_SCHEMA,
    STATUS_COMPLETE as AUDIT_COMPLETE,
)
from scripts.build_tusz_eeg_only_s1_reader_pack import (  # noqa: E402
    ADJUDICATION_SCHEMA,
    ANNOTATION_SCHEMA,
    STANDARD_19,
    TARGET_SEMANTICS,
    _adjudication_json_schema,
    _annotation_json_schema,
    _blank_adjudication,
    _blank_annotation,
    _write_json,
    _write_jsonl,
)


DEFAULT_AUDIT = ROOT / "outputs/tusz_non_deepsoz_s1_signal_eligibility_v1_20260813.json"
DEFAULT_OUTPUT = ROOT / "outputs/tusz_non_deepsoz_s1_locked_extension_reader_pack_v2_20260813"
PACK_SCHEMA = "tusz_non_deepsoz_s1_locked_extension_reader_pack_v1"
COHORT = "s1_locked_extension"
CASE_PREFIX = "S1-X"
ORDER_SEED = 20260813
EXPECTED_PATIENTS = 25
EXPECTED_EVENTS = 330


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Eligibility audit must be a JSON object")
    return value


def _extension_schema(value: dict[str, object]) -> dict[str, object]:
    """Bind the shared field schema to the pre-frozen extension cohort."""

    properties = value.get("properties")
    if not isinstance(properties, dict) or not isinstance(properties.get("cohort"), dict):
        raise TypeError("Shared reader schema lacks a cohort property")
    properties["cohort"] = {"const": COHORT}
    return value


def _load_eligible(audit_path: Path) -> tuple[list[dict[str, object]], Mapping[str, object]]:
    audit = _read_json(audit_path)
    if audit.get("schema_version") != AUDIT_SCHEMA or audit.get("status") != AUDIT_COMPLETE:
        raise ValueError("Locked extension requires the completed target-free audit")
    access = audit.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "deepsoz_soz_values_loaded",
            "tusz_per_channel_involvement_annotations_opened",
            "tusz_involvement_values_loaded",
            "model_or_pseudolabel_predictions_loaded",
            "private_eeg_loaded",
            "private_targets_loaded",
            "training_performed",
        )
    ):
        raise ValueError("Eligibility audit violates the target-free firewall")
    records = audit.get("records")
    if not isinstance(records, list):
        raise TypeError("Eligibility audit lacks records")
    events: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Eligibility audit record must be an object")
        if record.get("official_split") not in {"dev", "eval"} or (
            record.get("excluded_deepsoz_identity") is not False
        ):
            continue
        for raw_event in record.get("events", []):
            if isinstance(raw_event, Mapping) and raw_event.get("signal_eligible") is True:
                events.append(dict(raw_event))
    patients = {str(row["patient_id"]) for row in events}
    event_ids = {str(row["event_id"]) for row in events}
    if (
        len(patients) != EXPECTED_PATIENTS
        or len(events) != EXPECTED_EVENTS
        or len(event_ids) != EXPECTED_EVENTS
    ):
        raise ValueError("Locked extension eligibility scope changed")
    return events, audit


def build(
    *, audit_path: Path, output_directory: Path
) -> tuple[Path, Mapping[str, object]]:
    events, audit = _load_eligible(audit_path)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in events:
        grouped.setdefault(str(row["patient_id"]), []).append(row)
    ordered_patients = sorted(grouped)
    random.Random(ORDER_SEED).shuffle(ordered_patients)
    case_by_patient = {
        patient: f"{CASE_PREFIX}-{index:03d}"
        for index, patient in enumerate(ordered_patients, start=1)
    }

    target = output_directory.absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    published = False
    try:
        patients: list[dict[str, object]] = []
        linkage: list[dict[str, object]] = []
        roster: dict[str, list[str]] = {}
        for presentation_order, patient in enumerate(ordered_patients, start=1):
            case_id = case_by_patient[patient]
            patient_events = sorted(
                grouped[patient],
                key=lambda row: (
                    str(row["official_split"]),
                    str(row["relative_edf_path"]),
                    float(row["global_event_start_sec"]),
                    str(row["event_id"]),
                ),
            )
            patients.append(
                {
                    "case_id": case_id,
                    "cohort": COHORT,
                    "patient_pseudonym": patient,
                    "official_split": str(patient_events[0]["official_split"]),
                    "available_event_count": len(patient_events),
                    "target_semantics": TARGET_SEMANTICS,
                }
            )
            roster[case_id] = []
            for event_index, event in enumerate(patient_events, start=1):
                event_case = f"{case_id}-E{event_index:03d}"
                roster[case_id].append(event_case)
                linkage.append(
                    {
                        "case_id": case_id,
                        "event_case_id": event_case,
                        "cohort": COHORT,
                        "patient_pseudonym": patient,
                        "event_pseudonym": event["event_id"],
                        "official_split": event["official_split"],
                        "relative_edf_path": event["relative_edf_path"],
                        "global_event_t0_sec": event["global_event_start_sec"],
                        "global_event_stop_sec": event["global_event_stop_sec"],
                        "suggested_review_window_start_sec": max(
                            0.0, float(event["global_event_start_sec"]) - 30.0
                        ),
                        "suggested_review_window_stop_sec_unclamped": float(
                            event["global_event_stop_sec"]
                        )
                        + 60.0,
                        "event_anchor_semantics": event["event_anchor_semantics"],
                    }
                )
        for filename, rows in (
            ("patient_linkage.csv", patients),
            ("event_linkage.csv", linkage),
        ):
            with (staging / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        ids = [str(row["case_id"]) for row in patients]
        for reader_index, reader in enumerate(("reader_a", "reader_b")):
            ordered = ids.copy()
            random.Random(ORDER_SEED + reader_index + 1).shuffle(ordered)
            _write_jsonl(
                staging / f"{reader}.jsonl",
                [
                    _blank_annotation(case_id, COHORT, position)
                    for position, case_id in enumerate(ordered, start=1)
                ],
            )
        _write_jsonl(
            staging / "adjudication.jsonl",
            [_blank_adjudication(case_id, COHORT) for case_id in ids],
        )
        _write_json(
            staging / "annotation_schema.json",
            _extension_schema(_annotation_json_schema()),
        )
        _write_json(
            staging / "adjudication_schema.json",
            _extension_schema(_adjudication_json_schema()),
        )
        _write_json(staging / "event_case_roster.json", roster)
        split_counts = {
            split: {
                "patient_count": len(
                    {
                        str(row["patient_pseudonym"])
                        for row in patients
                        if row["official_split"] == split
                    }
                ),
                "event_count": sum(row["official_split"] == split for row in linkage),
            }
            for split in ("dev", "eval")
        }
        manifest: dict[str, object] = {
            "schema_version": PACK_SCHEMA,
            "status": "empty_label_fresh_locked_extension_reader_pack_ready",
            "cohort": COHORT,
            "patient_count": len(patients),
            "event_count": len(linkage),
            "official_split_counts": split_counts,
            "presentation_order_seed": ORDER_SEED,
            "target_semantics": TARGET_SEMANTICS,
            "label_release_policy": (
                "remain sealed until the current S1 model, uncertainty rule, region map, "
                "and report slots are frozen; open once as a confirmatory locked extension"
            ),
            "target_is_not": [
                "deepsoz_clinical_note_integrated_reference",
                "cortical_or_invasive_soz",
                "earliest_scalp_visible_electrode",
                "tusz_ictal_involvement",
                "spread_electrode",
                "surgical_target",
            ],
            "signal_contract": audit["signal_contract"],
            "signal_preprocessing_contract": {
                "preprocess_config": {
                    "output_sfreq_hz": 200.0,
                    "highpass_hz": 0.5,
                    "lowpass_hz": 45.0,
                    "butterworth_order": 4,
                    "warmup_sec": 30.0,
                    "pre_onset_sec": 12.0,
                    "post_onset_sec": 48.0,
                    "fir_half_length_per_rate": 10,
                    "flatline_run_sec": 2.0,
                    "clipping_run_sec": 0.5,
                    "qc_tolerance_volts": 1e-12,
                    "reference_policy": "primary_ref",
                    "sensitivity_reference": None,
                    "apply_car19": True,
                },
                "input_channels": list(STANDARD_19),
                "output_shape_per_event": [19, 12000],
                "sampling_frequency_hz": 200.0,
                "event_interval_sec": [-12.0, 48.0],
                "reference_representation": "primary_reference_then_CAR19",
            },
            "access_receipt": {
                "completed_target_free_signal_eligibility_audit_loaded": True,
                "deepsoz_identity_exclusion_already_applied": True,
                "deepsoz_target_values_loaded": False,
                "deepsoz_soz_values_loaded": False,
                "tusz_per_channel_involvement_annotations_opened": False,
                "tusz_involvement_values_loaded": False,
                "tusz_channel_time_target_values_used_for_selection_or_s1_labels": False,
                "tusz_channel_time_target_values_exported": False,
                "model_predictions_loaded": False,
                "model_or_pseudolabel_predictions_loaded": False,
                "private_eeg_loaded": False,
                "private_target_loaded": False,
                "private_targets_loaded": False,
                "automatic_soz_annotation_performed": False,
                "new_reader_labels_opened": False,
                "training_performed": False,
            },
            "files": {
                "patient_linkage": "patient_linkage.csv",
                "event_linkage": "event_linkage.csv",
                "event_case_roster": "event_case_roster.json",
                "annotation_schema": "annotation_schema.json",
                "adjudication_schema": "adjudication_schema.json",
                "independent_reader_templates": ["reader_a.jsonl", "reader_b.jsonl"],
                "adjudication_templates": ["adjudication.jsonl"],
            },
            "source_audit": str(audit_path.resolve(strict=True)),
            "annotation_schema_version": ANNOTATION_SCHEMA,
            "adjudication_schema_version": ADJUDICATION_SCHEMA,
            "channels": list(STANDARD_19),
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "README.md").write_text(
            "# TUSZ non-DeepSOZ locked S1 extension\n\n"
            "This label-fresh extension contains every signal-eligible non-DeepSOZ "
            "patient from official TUSZ dev/eval. It was frozen before reader labels. "
            "Keep all labels sealed until the current S1-development model and the "
            "single calibration rule are frozen. Two experts independently review every "
            "event in each patient bag; a third expert adjudicates after both reads close.\n\n"
            "Do not show TUSZ channel involvement, DeepSOZ labels, model predictions, "
            "private data, or the other reader's decision. Spread remains separate from "
            "the candidate-positive set.\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return target, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path, manifest = build(
        audit_path=args.audit, output_directory=args.output_directory
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "patient_count": manifest["patient_count"],
                "event_count": manifest["event_count"],
                "output": str(path),
                "labels_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
