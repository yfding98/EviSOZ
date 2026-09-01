#!/usr/bin/env python3
"""Build a target-blind two-reader TUSZ event-phenotype study pack.

The pack deliberately contains no DeepSOZ target, TUSZ channel-time target,
private datum, or model prediction.  It reproduces the already-consumed
lexical 64-patient development roster and reserves every remaining source
patient as a patient-disjoint locked reader cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "outputs/tusz_ictal_master_manifest_v4_20260809_preflight/receipt.json"
)
DEFAULT_CONSUMED_AUDIT = (
    ROOT / "outputs/event_phenotype_source_only_n64_20260811.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/tusz_event_phenotype_reader_study_v1"

PACK_SCHEMA = "tusz_event_phenotype_reader_study_pack_v1"
ANNOTATION_SCHEMA = "tusz_event_phenotype_reader_annotation_v1"
EXPECTED_SOURCE_SCHEMA = "tusz_ictal_training_manifest_v4.0.0"
EXPECTED_AUDIT_SCHEMA = "soz_event_phenotype_source_only_audit_v1"

TCP_DERIVATIONS = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FZ-CZ",
    "CZ-PZ",
    "T7-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T8",
)
LATERALITY = ("left", "right", "bilateral", "midline", "indeterminate")
REGION = (
    "frontal",
    "temporal",
    "central",
    "parietal",
    "occipital",
    "multiregional",
    "indeterminate",
)
TRISTATE = ("present", "absent", "indeterminate")
ARTIFACT_TYPES = (
    "muscle",
    "ocular",
    "movement",
    "electrode_transient",
    "line_noise",
    "other",
)
MONTAGES = ("longitudinal_bipolar", "transverse_bipolar", "average_reference")


def _load_json(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Expected a canonical JSON file: {path}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _safe_relative_edf(value: object) -> str:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("Unsafe relative EDF path in source receipt")
    return relative.as_posix()


def _identity_rows(events: Sequence[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_events: set[str] = set()
    for value in events:
        if not isinstance(value, Mapping):
            raise TypeError("Every source event must be an object")
        patient = str(value.get("patient_id", "")).strip()
        event = str(value.get("event_id", "")).strip()
        if not patient or not event or event in seen_events:
            raise ValueError("Source event identity is empty or duplicated")
        seen_events.add(event)
        t0 = float(value.get("event_t0_sec"))
        stop = float(value.get("event_stop_sec"))
        if not math.isfinite(t0) or not math.isfinite(stop) or stop < t0:
            raise ValueError("Source event timing is invalid")
        if value.get("dataset") != "tusz":
            raise ValueError("Reader study source must be TUSZ")
        rows.append(
            {
                "patient_id": patient,
                "event_id": event,
                "relative_edf_path": _safe_relative_edf(
                    value.get("relative_edf_path")
                ),
                "global_event_t0_sec": t0,
                "global_event_stop_sec": stop,
                "event_anchor_semantics": str(
                    value.get("event_anchor_semantics", "")
                ),
            }
        )
    return rows


def _one_lexical_event_per_patient(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    chosen: list[dict[str, object]] = []
    patients: set[str] = set()
    for row in sorted(
        rows,
        key=lambda item: (str(item["patient_id"]), str(item["event_id"])),
    ):
        patient = str(row["patient_id"])
        if patient in patients:
            continue
        patients.add(patient)
        chosen.append(dict(row))
    return chosen


def _validate_consumed_development_roster(
    audit: Mapping[str, object],
    development: Sequence[Mapping[str, object]],
) -> None:
    if audit.get("schema_version") != EXPECTED_AUDIT_SCHEMA:
        raise ValueError("Unexpected consumed event-phenotype audit schema")
    access = audit.get("access_receipt")
    events = audit.get("events")
    if not isinstance(access, Mapping) or not isinstance(events, list):
        raise TypeError("Consumed audit lacks access receipt or events")
    forbidden_true = (
        "tusz_native_target_values_loaded",
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "training_performed",
        "threshold_selection_performed",
    )
    if any(access.get(name) is not False for name in forbidden_true):
        raise ValueError("Consumed audit is not target-blind and source-only")
    audit_pairs = {
        (str(row.get("patient_id", "")), str(row.get("event_id", "")))
        for row in events
        if isinstance(row, Mapping)
    }
    development_pairs = {
        (str(row["patient_id"]), str(row["event_id"])) for row in development
    }
    if len(audit_pairs) != len(events) or audit_pairs != development_pairs:
        raise ValueError(
            "Lexical development roster disagrees with the consumed audit"
        )


def _blank_montage() -> dict[str, object]:
    return {
        "assessable": None,
        "dominant_derivations": [],
        "laterality": None,
        "region": None,
    }


def _blank_annotation(case_id: str, cohort: str, order: int) -> dict[str, object]:
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "case_id": case_id,
        "cohort": cohort,
        "presentation_order": order,
        "reviewer_id": "",
        "review_status": "unreviewed",
        "scalp_onset_visible": None,
        "first_visible_start_offset_sec": None,
        "first_visible_end_offset_sec": None,
        "earliest_derivations": [],
        "rhythmic_activity": None,
        "dominant_frequency_lower_hz": None,
        "dominant_frequency_upper_hz": None,
        "evolution_frequency": None,
        "evolution_amplitude": None,
        "evolution_morphology": None,
        "evolution_spatial_recruitment": None,
        "later_visible_state": None,
        "later_visible_delay_sec": None,
        "later_visible_derivations": [],
        "later_visible_laterality": None,
        "later_visible_region": None,
        "montage_observations": {name: _blank_montage() for name in MONTAGES},
        "artifact_assessable": None,
        "artifact_types": [],
        "artifact_burden": None,
        "event_onset_laterality": None,
        "event_onset_region": None,
        "no_scalp_visible_reason": None,
        "reader_confidence": None,
        "review_completed_at": None,
        "free_text_note_not_for_model": "",
    }


def _json_schema() -> dict[str, object]:
    nullable_number = {"type": ["number", "null"]}
    nullable_boolean = {"type": ["boolean", "null"]}
    nullable_tristate = {"type": ["string", "null"], "enum": [*TRISTATE, None]}
    nullable_laterality = {
        "type": ["string", "null"],
        "enum": [*LATERALITY, None],
    }
    nullable_region = {"type": ["string", "null"], "enum": [*REGION, None]}
    edge_array = {
        "type": "array",
        "items": {"type": "string", "enum": list(TCP_DERIVATIONS)},
        "uniqueItems": True,
    }
    montage_properties = {
        "assessable": nullable_boolean,
        "dominant_derivations": edge_array,
        "laterality": nullable_laterality,
        "region": nullable_region,
    }
    properties: dict[str, object] = {
        "schema_version": {"const": ANNOTATION_SCHEMA},
        "case_id": {"type": "string", "minLength": 1},
        "cohort": {"enum": ["development", "locked"]},
        "presentation_order": {"type": "integer", "minimum": 1},
        "reviewer_id": {"type": "string"},
        "review_status": {"enum": ["unreviewed", "completed"]},
        "scalp_onset_visible": {
            "type": ["string", "null"],
            "enum": ["yes", "no", "indeterminate", None],
        },
        "first_visible_start_offset_sec": nullable_number,
        "first_visible_end_offset_sec": nullable_number,
        "earliest_derivations": edge_array,
        "rhythmic_activity": nullable_tristate,
        "dominant_frequency_lower_hz": nullable_number,
        "dominant_frequency_upper_hz": nullable_number,
        "evolution_frequency": nullable_tristate,
        "evolution_amplitude": nullable_tristate,
        "evolution_morphology": nullable_tristate,
        "evolution_spatial_recruitment": nullable_tristate,
        "later_visible_state": nullable_tristate,
        "later_visible_delay_sec": nullable_number,
        "later_visible_derivations": edge_array,
        "later_visible_laterality": nullable_laterality,
        "later_visible_region": nullable_region,
        "montage_observations": {
            "type": "object",
            "additionalProperties": False,
            "required": list(MONTAGES),
            "properties": {
                name: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(montage_properties),
                    "properties": montage_properties,
                }
                for name in MONTAGES
            },
        },
        "artifact_assessable": nullable_boolean,
        "artifact_types": {
            "type": "array",
            "items": {"type": "string", "enum": list(ARTIFACT_TYPES)},
            "uniqueItems": True,
        },
        "artifact_burden": {
            "type": ["string", "null"],
            "enum": ["none", "mild", "moderate", "severe", "indeterminate", None],
        },
        "event_onset_laterality": nullable_laterality,
        "event_onset_region": nullable_region,
        "no_scalp_visible_reason": {
            "type": ["string", "null"],
            "enum": [
                "no_scalp_visible_change",
                "artifact_obscured",
                "recording_truncated",
                "montage_disagreement",
                "other",
                None,
            ],
        },
        "reader_confidence": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1,
        },
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


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def build_reader_study_pack(
    *,
    source_receipt: Path,
    consumed_audit: Path,
    output_dir: Path,
    development_patient_count: int = 64,
    expected_source_patient_count: int = 129,
    expected_source_event_count: int = 1519,
) -> dict[str, object]:
    source = _load_json(source_receipt)
    audit = _load_json(consumed_audit)
    if source.get("schema_version") != EXPECTED_SOURCE_SCHEMA:
        raise ValueError("Unexpected TUSZ source receipt schema")
    raw_events = source.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("TUSZ source receipt lacks events")
    if len(raw_events) != expected_source_event_count:
        raise ValueError("Unexpected TUSZ source event count")
    identity_rows = _identity_rows(raw_events)
    selected = _one_lexical_event_per_patient(identity_rows)
    if len(selected) != expected_source_patient_count:
        raise ValueError("Unexpected TUSZ source patient count")
    if not 1 <= development_patient_count < len(selected):
        raise ValueError("development_patient_count cannot form two cohorts")
    development = selected[:development_patient_count]
    locked = selected[development_patient_count:]
    _validate_consumed_development_roster(audit, development)
    if {str(row["patient_id"]) for row in development} & {
        str(row["patient_id"]) for row in locked
    }:
        raise RuntimeError("Development and locked patients overlap")

    target = output_dir.absolute()
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        linkage: list[dict[str, object]] = []
        case_ids: dict[str, list[str]] = {"development": [], "locked": []}
        for cohort, rows, seed in (
            ("development", development, 20260811),
            ("locked", locked, 20260812),
        ):
            shuffled = [dict(row) for row in rows]
            random.Random(seed).shuffle(shuffled)
            prefix = "EPH-D" if cohort == "development" else "EPH-L"
            for index, row in enumerate(shuffled, start=1):
                case_id = f"{prefix}-{index:03d}"
                case_ids[cohort].append(case_id)
                linkage.append(
                    {
                        "case_id": case_id,
                        "cohort": cohort,
                        "patient_pseudonym": row["patient_id"],
                        "event_pseudonym": row["event_id"],
                        "relative_edf_path": row["relative_edf_path"],
                        "global_event_t0_sec": row["global_event_t0_sec"],
                        "global_event_stop_sec": row["global_event_stop_sec"],
                        "suggested_review_window_start_sec": max(
                            0.0, float(row["global_event_t0_sec"]) - 30.0
                        ),
                        "suggested_review_window_stop_sec_unclamped": (
                            float(row["global_event_stop_sec"]) + 60.0
                        ),
                        "event_anchor_semantics": row["event_anchor_semantics"],
                    }
                )

        with (temporary / "case_linkage.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(linkage[0]))
            writer.writeheader()
            writer.writerows(linkage)

        for reader_index, reader in enumerate(("reader_a", "reader_b"), start=1):
            for cohort in ("development", "locked"):
                ids = list(case_ids[cohort])
                random.Random(20260900 + reader_index + (100 if cohort == "locked" else 0)).shuffle(ids)
                annotations = [
                    _blank_annotation(case_id, cohort, order)
                    for order, case_id in enumerate(ids, start=1)
                ]
                _write_jsonl(temporary / f"{reader}_{cohort}.jsonl", annotations)

        _write_json(temporary / "annotation_schema.json", _json_schema())
        manifest: dict[str, object] = {
            "schema_version": PACK_SCHEMA,
            "status": "empty_target_blind_reader_templates_ready",
            "source_event_count": len(identity_rows),
            "source_patient_count": len(selected),
            "development_patient_count": len(development),
            "locked_patient_count": len(locked),
            "events_per_patient": 1,
            "selection": (
                "lexical_first_event_per_patient;first_64_lexical_patients_match_"
                "consumed_development_audit;remaining_patients_locked"
            ),
            "reader_count": 2,
            "required_montages": list(MONTAGES),
            "annotation_time_coordinate": "seconds_relative_to_global_event_t0",
            "suggested_review_window_stop_is_unclamped_hint": True,
            "access_receipt": {
                "consumed_model_audit_loaded_for_identity_validation_only": True,
                "model_predictions_exported": False,
                "tusz_channel_time_target_values_loaded": False,
                "tusz_channel_time_target_values_exported": False,
                "deepsoz_identity_or_target_loaded": False,
                "deepsoz_identity_or_target_exported": False,
                "private_eeg_loaded": False,
                "private_target_loaded": False,
                "automatic_annotation_performed": False,
                "locked_annotations_opened": False,
            },
            "files": {
                "case_linkage": "case_linkage.csv",
                "annotation_schema": "annotation_schema.json",
                "reader_templates": [
                    "reader_a_development.jsonl",
                    "reader_b_development.jsonl",
                    "reader_a_locked.jsonl",
                    "reader_b_locked.jsonl",
                ],
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            "# TUSZ event-phenotype reader study v1\n\n"
            "This directory contains empty, target-blind templates for two "
            "independent epilepsy-EEG readers. `case_linkage.csv` is the only "
            "file that links opaque case IDs to public TUSZ EDF/event anchors. "
            "Do not show model output, DeepSOZ labels, TUSZ channel labels, or "
            "the other reader's annotation during independent review.\n\n"
            "Times are recorded relative to the global TUSZ event marker. "
            "First-visible and later-visible fields are scalp display facts, "
            "not SOZ or propagation labels. The suggested review-window stop "
            "is an unclamped hint; the viewer must clamp it to the real EDF "
            "duration. Free text is never a model input.\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--consumed-audit", type=Path, default=DEFAULT_CONSUMED_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--development-patients", type=int, default=64)
    parser.add_argument("--expected-source-patients", type=int, default=129)
    parser.add_argument("--expected-source-events", type=int, default=1519)
    args = parser.parse_args()
    result = build_reader_study_pack(
        source_receipt=args.source_receipt,
        consumed_audit=args.consumed_audit,
        output_dir=args.output_dir,
        development_patient_count=args.development_patients,
        expected_source_patient_count=args.expected_source_patients,
        expected_source_event_count=args.expected_source_events,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
