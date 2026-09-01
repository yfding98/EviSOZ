#!/usr/bin/env python3
"""Build a label-fresh patient-level EEG-only scalp-SOZ reader pack.

The pack contains the 60 official-train TUSZ patients excluded from the local
DeepSOZ identity roster and all 433 signal-eligible seizure events belonging
to them.  It never exports TUSZ channel-time involvement targets, DeepSOZ
targets, private data, or model predictions.  Two epilepsy-EEG experts review
all available events for one patient and independently provide one
patient-level standard-19 scalp-electrode candidate set.  A third expert uses
the separate adjudication template only after both independent reads close.

This creates label-fresh EEG-only S1 supervision.  It is not an integrated
clinical SOZ reference because clinical notes, semiology, imaging, invasive
EEG, resection, and outcome are unavailable.  The waveforms have also been
used in target-free/source-native development, so the cohort is not a
waveform-unseen external test.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = (
    ROOT / "outputs/labram_source_only_dapt_manifest_v1_20260811/manifest.json"
)
DEFAULT_MASTER_RECEIPT = (
    ROOT / "outputs/tusz_ictal_master_manifest_v4_20260809_preflight/receipt.json"
)
DEFAULT_DEEPSOZ_SPLIT = (
    ROOT / "outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs/tusz_eeg_only_s1_reader_pack_v1_20260813"

PACK_SCHEMA = "tusz_eeg_only_patient_s1_reader_pack_v1"
ANNOTATION_SCHEMA = "tusz_eeg_only_patient_s1_annotation_v1"
ADJUDICATION_SCHEMA = "tusz_eeg_only_patient_s1_adjudication_v1"
EXPECTED_SOURCE_SCHEMA = "soz_labram_source_only_continuous_dapt_v1"
EXPECTED_MASTER_SCHEMA = "tusz_ictal_training_manifest_v4.0.0"
EXPECTED_PATIENT_COUNT = 60
EXPECTED_EVENT_COUNT = 433
EXPECTED_ELIGIBLE_RECORD_COUNT = 748
EXPECTED_DEEPSOZ_IDENTITY_COUNT = 124
SPLIT_SEED = 20260813
ORDER_SEED = 20260814

STANDARD_19 = (
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
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
)
ELECTRODE_STATES = (
    "candidate_positive",
    "reviewed_not_candidate",
    "unknown",
    "unavailable",
)
TARGET_AVAILABILITY = (
    "available",
    "indeterminate",
    "unavailable_no_scalp_visible_localizing_evidence",
    "unavailable_signal_quality",
    "unavailable_other",
)
TARGET_SEMANTICS = (
    "expert_adjudicated_eeg_only_all_event_patient_level_"
    "scalp_electrode_soz_candidate"
)
SET_EXHAUSTIVENESS = ("yes", "no", "indeterminate")
SCALP_VISIBILITY = ("yes", "no", "indeterminate")
EVIDENCE_BASES = (
    "ictal_onset_pattern",
    "epileptiform_morphology",
    "temporal_evolution",
    "repeated_across_events",
    "montage_consistency",
    "other_eeg_evidence",
)
COHORT_SIZES = {"s1_development": 36, "s1_calibration": 12, "s1_locked": 12}


def _load_json(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _safe_relative_edf(value: object) -> str:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("Unsafe relative EDF path")
    return relative.as_posix()


def _deepsoz_local_ids(path: Path, *, expected_count: int) -> set[str]:
    if expected_count <= 0:
        raise ValueError("Expected DeepSOZ identity count must be positive")
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_count:
        raise ValueError("DeepSOZ local identity roster changed")
    result = {str(row.get("local_patient_id", "")).strip() for row in rows}
    if "" in result or len(result) != expected_count:
        raise ValueError("DeepSOZ local identities are empty or duplicated")
    return result


def _eligible_source_patients(
    payload: Mapping[str, object],
) -> tuple[tuple[str, ...], set[str]]:
    if payload.get("schema_version") != EXPECTED_SOURCE_SCHEMA:
        raise ValueError("Unexpected source-only manifest schema")
    for flag in ("target_values_loaded", "private_data_loaded", "annotation_sidecars_opened"):
        if payload.get(flag) is not False:
            raise ValueError("Source-only safety contract changed")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("Source-only manifest lacks records")
    eligible = [row for row in records if isinstance(row, Mapping) and row.get("eligibility") == "eligible"]
    if len(eligible) != EXPECTED_ELIGIBLE_RECORD_COUNT:
        raise ValueError("Eligible source-only record count changed")
    patients = tuple(sorted({str(row.get("patient_id", "")).strip() for row in eligible}))
    paths = {_safe_relative_edf(row.get("relative_edf_path")) for row in eligible}
    if len(patients) != EXPECTED_PATIENT_COUNT or "" in patients:
        raise ValueError("Expected exactly 60 source-only patients")
    return patients, paths


def _events_for_patients(
    payload: Mapping[str, object],
    patients: Sequence[str],
    eligible_paths: set[str],
) -> dict[str, list[dict[str, object]]]:
    if payload.get("schema_version") != EXPECTED_MASTER_SCHEMA:
        raise ValueError("Unexpected TUSZ master receipt schema")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("TUSZ master receipt lacks events")
    patient_set = set(patients)
    grouped = {patient: [] for patient in patients}
    seen_ids: set[str] = set()
    for value in raw_events:
        if not isinstance(value, Mapping):
            raise TypeError("TUSZ event must be an object")
        patient = str(value.get("patient_id", "")).strip()
        if patient not in patient_set:
            continue
        event_id = str(value.get("event_id", "")).strip()
        relative = _safe_relative_edf(value.get("relative_edf_path"))
        if not event_id or event_id in seen_ids or relative not in eligible_paths:
            raise ValueError("Source S1 event identity/path contract failed")
        seen_ids.add(event_id)
        start = float(value.get("event_t0_sec"))
        stop = float(value.get("event_stop_sec"))
        if not 0.0 <= start <= stop:
            raise ValueError("Invalid event interval")
        grouped[patient].append(
            {
                "event_id": event_id,
                "relative_edf_path": relative,
                "global_event_t0_sec": start,
                "global_event_stop_sec": stop,
                "event_anchor_semantics": str(value.get("event_anchor_semantics", "")),
                "source_montage": str(value.get("montage", "")),
                "session_id": str(value.get("session_id", "")),
            }
        )
    for patient, events in grouped.items():
        events.sort(key=lambda row: (str(row["relative_edf_path"]), float(row["global_event_t0_sec"]), str(row["event_id"])))
        if not events:
            raise ValueError(f"Patient lost all events: {patient}")
    if sum(map(len, grouped.values())) != EXPECTED_EVENT_COUNT:
        raise ValueError("Expected 433 source-only events")
    return grouped


def _preprocess_config(payload: Mapping[str, object]) -> dict[str, object]:
    value = payload.get("preprocess_config")
    if not isinstance(value, Mapping):
        raise TypeError("TUSZ master receipt lacks preprocessing configuration")
    required = {
        "apply_car19",
        "highpass_hz",
        "lowpass_hz",
        "output_sfreq_hz",
        "post_onset_sec",
        "pre_onset_sec",
        "reference_policy",
        "warmup_sec",
    }
    if not required.issubset(value):
        raise ValueError("TUSZ preprocessing configuration is incomplete")
    result = dict(value)
    if (
        result["apply_car19"] is not True
        or float(result["output_sfreq_hz"]) != 200.0
        or float(result["pre_onset_sec"]) != 12.0
        or float(result["post_onset_sec"]) != 48.0
        or str(result["reference_policy"]) != "primary_ref"
    ):
        raise ValueError("S1 requires the frozen C-REF19/CAR19 60-second contract")
    return result


def _cohort_assignment(
    patients: Sequence[str], patient_event_counts: Mapping[str, int]
) -> dict[str, str]:
    """Allocate patients while balancing target-free event-count burden.

    Event count is known before S1 annotation and is needed to avoid making
    cross-event consistency almost synonymous with cohort membership.  The
    greedy rule assigns largest bags first to the cohort with the lowest
    event burden relative to its 60/20/20 patient share, while enforcing the
    exact 36/12/12 patient capacities.  Seeded tie breaking is frozen before
    any reader labels exist.
    """

    if set(patient_event_counts) != set(patients) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in patient_event_counts.values()
    ):
        raise ValueError("Patient event counts must cover the S1 roster exactly")
    tie_rng = random.Random(SPLIT_SEED)
    tie_break = {patient: tie_rng.random() for patient in patients}
    ordered = sorted(
        patients,
        key=lambda patient: (
            -patient_event_counts[patient],
            tie_break[patient],
            patient,
        ),
    )
    total_capacity = sum(COHORT_SIZES.values())
    assigned_patient_count = {cohort: 0 for cohort in COHORT_SIZES}
    assigned_event_count = {cohort: 0 for cohort in COHORT_SIZES}
    result: dict[str, str] = {}
    for patient in ordered:
        available = [
            cohort
            for cohort, capacity in COHORT_SIZES.items()
            if assigned_patient_count[cohort] < capacity
        ]
        if not available:
            raise RuntimeError("S1 cohort capacities were exhausted early")
        cohort = min(
            available,
            key=lambda name: (
                assigned_event_count[name]
                / (COHORT_SIZES[name] / total_capacity),
                assigned_patient_count[name] / COHORT_SIZES[name],
                name,
            ),
        )
        result[patient] = cohort
        assigned_patient_count[cohort] += 1
        assigned_event_count[cohort] += patient_event_counts[patient]
    if (
        assigned_patient_count != COHORT_SIZES
        or len(result) != EXPECTED_PATIENT_COUNT
    ):
        raise RuntimeError("S1 cohort partition is incomplete")
    return result


def _electrode_states() -> dict[str, object]:
    return {electrode: None for electrode in STANDARD_19}


def _blank_annotation(case_id: str, cohort: str, presentation_order: int) -> dict[str, object]:
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "case_id": case_id,
        "cohort": cohort,
        "presentation_order": presentation_order,
        "reviewer_id": "",
        "review_status": "unreviewed",
        "all_available_events_reviewed": None,
        "reviewed_event_case_ids": [],
        "scalp_visible_localizing_evidence": None,
        "target_availability": None,
        "electrode_states": _electrode_states(),
        "candidate_positive_electrodes": [],
        "set_exhaustive": None,
        "evidence_bases": [],
        "known_spread_assessable": None,
        "known_spread_electrodes": [],
        "patient_event_consistency": None,
        "label_confidence": None,
        "unavailability_reason": "",
        "review_completed_at": None,
        "free_text_note_not_for_model": "",
    }


def _blank_adjudication(case_id: str, cohort: str) -> dict[str, object]:
    return {
        "schema_version": ADJUDICATION_SCHEMA,
        "case_id": case_id,
        "cohort": cohort,
        "adjudicator_id": "",
        "adjudication_status": "unreviewed",
        "reader_a_record_available": None,
        "reader_b_record_available": None,
        "all_available_events_reviewed": None,
        "scalp_visible_localizing_evidence": None,
        "target_availability": None,
        "electrode_states": _electrode_states(),
        "candidate_positive_electrodes": [],
        "set_exhaustive": None,
        "evidence_bases": [],
        "known_spread_assessable": None,
        "known_spread_electrodes": [],
        "patient_event_consistency": None,
        "label_confidence": None,
        "disagreement_domains": [],
        "adjudication_rationale_not_for_model": "",
        "adjudication_completed_at": None,
    }


def _electrode_state_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(STANDARD_19),
        "properties": {
            electrode: {"type": ["string", "null"], "enum": [*ELECTRODE_STATES, None]}
            for electrode in STANDARD_19
        },
    }


def _electrode_array() -> dict[str, object]:
    return {
        "type": "array",
        "items": {"type": "string", "enum": list(STANDARD_19)},
        "uniqueItems": True,
    }


def _common_annotation_properties() -> dict[str, object]:
    return {
        "all_available_events_reviewed": {"type": ["boolean", "null"]},
        "scalp_visible_localizing_evidence": {"type": ["string", "null"], "enum": [*SCALP_VISIBILITY, None]},
        "target_availability": {"type": ["string", "null"], "enum": [*TARGET_AVAILABILITY, None]},
        "electrode_states": _electrode_state_schema(),
        "candidate_positive_electrodes": _electrode_array(),
        "set_exhaustive": {"type": ["string", "null"], "enum": [*SET_EXHAUSTIVENESS, None]},
        "evidence_bases": {
            "type": "array",
            "items": {"type": "string", "enum": list(EVIDENCE_BASES)},
            "uniqueItems": True,
        },
        "known_spread_assessable": {"type": ["boolean", "null"]},
        "known_spread_electrodes": _electrode_array(),
        "patient_event_consistency": {
            "type": ["string", "null"],
            "enum": ["consistent", "partially_consistent", "heterogeneous", "indeterminate", None],
        },
        "label_confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
    }


def _annotation_json_schema() -> dict[str, object]:
    properties = {
        "schema_version": {"const": ANNOTATION_SCHEMA},
        "case_id": {"type": "string", "minLength": 1},
        "cohort": {"enum": list(COHORT_SIZES)},
        "presentation_order": {"type": "integer", "minimum": 1},
        "reviewer_id": {"type": "string"},
        "review_status": {"enum": ["unreviewed", "completed"]},
        **_common_annotation_properties(),
        "reviewed_event_case_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "unavailability_reason": {"type": "string"},
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


def _adjudication_json_schema() -> dict[str, object]:
    properties = {
        "schema_version": {"const": ADJUDICATION_SCHEMA},
        "case_id": {"type": "string", "minLength": 1},
        "cohort": {"enum": list(COHORT_SIZES)},
        "adjudicator_id": {"type": "string"},
        "adjudication_status": {"enum": ["unreviewed", "completed"]},
        "reader_a_record_available": {"type": ["boolean", "null"]},
        "reader_b_record_available": {"type": ["boolean", "null"]},
        **_common_annotation_properties(),
        "disagreement_domains": {
            "type": "array",
            "items": {
                "enum": [
                    "target_availability",
                    "candidate_positive_set",
                    "reviewed_negative_or_unknown",
                    "set_exhaustiveness",
                    "spread",
                    "event_consistency",
                    "confidence",
                ]
            },
            "uniqueItems": True,
        },
        "adjudication_rationale_not_for_model": {"type": "string"},
        "adjudication_completed_at": {"type": ["string", "null"]},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": ADJUDICATION_SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def validate_completed_annotation(row: Mapping[str, object], expected_event_ids: set[str]) -> None:
    """Validate semantic invariants that JSON Schema alone cannot express."""

    if row.get("review_status") != "completed":
        raise ValueError("Annotation is not completed")
    if not str(row.get("reviewer_id", "")).strip():
        raise ValueError("Completed annotation requires a reviewer ID")
    _parse_completed_timestamp(row.get("review_completed_at"), field="review_completed_at")
    states = row.get("electrode_states")
    if not isinstance(states, Mapping) or set(states) != set(STANDARD_19):
        raise ValueError("Annotation must contain all standard-19 electrode states")
    positives = row.get("candidate_positive_electrodes")
    if not isinstance(positives, list) or set(positives) != {
        electrode for electrode, state in states.items() if state == "candidate_positive"
    }:
        raise ValueError("Positive set disagrees with electrode states")
    reviewed = row.get("reviewed_event_case_ids")
    if row.get("all_available_events_reviewed") is not True:
        raise ValueError("Completed annotation requires all available events reviewed")
    if set(reviewed or ()) != expected_event_ids:
        raise ValueError("All-events-reviewed flag disagrees with reviewed event IDs")
    if set(reviewed or ()) - expected_event_ids:
        raise ValueError("Annotation references an event outside this patient bag")
    available = row.get("target_availability") == "available"
    if available and not positives:
        raise ValueError("Available S1 target requires at least one positive electrode")
    if not available and positives:
        raise ValueError("Unavailable/indeterminate S1 target cannot carry positives")
    _validate_common_completed_fields(row, available=available)
    if row.get("known_spread_assessable") is not True and row.get("known_spread_electrodes"):
        raise ValueError("Unassessable spread cannot carry spread electrodes")


def _parse_completed_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Completed record requires {field}")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _validate_common_completed_fields(
    row: Mapping[str, object], *, available: bool
) -> None:
    states = row.get("electrode_states")
    if not isinstance(states, Mapping) or set(states) != set(STANDARD_19):
        raise ValueError("Completed record must contain all standard-19 states")
    if any(state not in ELECTRODE_STATES for state in states.values()):
        raise ValueError("Completed record contains an invalid electrode state")
    positives = row.get("candidate_positive_electrodes")
    spread = row.get("known_spread_electrodes")
    if not isinstance(positives, list) or not isinstance(spread, list):
        raise TypeError("Candidate and spread electrodes must be arrays")
    if set(positives) & set(spread):
        raise ValueError("SOZ candidate and known-spread sets must be disjoint")
    if available:
        if row.get("scalp_visible_localizing_evidence") != "yes":
            raise ValueError("Available S1 target requires scalp-visible localizing evidence")
        if row.get("set_exhaustive") not in SET_EXHAUSTIVENESS:
            raise ValueError("Available S1 target requires set exhaustiveness")
        evidence = row.get("evidence_bases")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Available S1 target requires at least one evidence basis")
    elif row.get("set_exhaustive") not in (None, "indeterminate"):
        raise ValueError("Unavailable target cannot claim an exhaustive candidate set")
    confidence = row.get("label_confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("Completed record requires label confidence in [0,1]")
    availability = row.get("target_availability")
    if availability not in TARGET_AVAILABILITY:
        raise ValueError("Completed record requires a valid target availability")
    visibility = row.get("scalp_visible_localizing_evidence")
    if visibility not in SCALP_VISIBILITY:
        raise ValueError("Completed record requires scalp-evidence visibility")
    if availability == "unavailable_no_scalp_visible_localizing_evidence" and (
        row.get("scalp_visible_localizing_evidence") != "no"
    ):
        raise ValueError("No-scalp-evidence unavailability requires scalp visibility=no")


def validate_completed_adjudication(
    row: Mapping[str, object],
    reader_a: Mapping[str, object],
    reader_b: Mapping[str, object],
) -> None:
    """Validate a third-reader decision and its independence chronology."""

    if row.get("adjudication_status") != "completed":
        raise ValueError("Adjudication is not completed")
    adjudicator = str(row.get("adjudicator_id", "")).strip()
    reader_ids = {
        str(reader_a.get("reviewer_id", "")).strip(),
        str(reader_b.get("reviewer_id", "")).strip(),
    }
    if not adjudicator or "" in reader_ids or len(reader_ids) != 2:
        raise ValueError("Adjudication requires two distinct readers and one adjudicator")
    if adjudicator in reader_ids:
        raise ValueError("Adjudicator must differ from both independent readers")
    if row.get("reader_a_record_available") is not True or (
        row.get("reader_b_record_available") is not True
    ):
        raise ValueError("Completed adjudication requires both reader records")
    if row.get("all_available_events_reviewed") is not True:
        raise ValueError("Completed adjudication requires the full patient event bag")
    states = row.get("electrode_states")
    positives = row.get("candidate_positive_electrodes")
    if not isinstance(states, Mapping) or not isinstance(positives, list) or set(positives) != {
        electrode for electrode, state in states.items() if state == "candidate_positive"
    }:
        raise ValueError("Adjudicated positive set disagrees with electrode states")
    available = row.get("target_availability") == "available"
    if available and not positives:
        raise ValueError("Available adjudication requires at least one candidate")
    if not available and positives:
        raise ValueError("Unavailable adjudication cannot carry candidates")
    _validate_common_completed_fields(row, available=available)
    if row.get("known_spread_assessable") is not True and row.get("known_spread_electrodes"):
        raise ValueError("Unassessable adjudicated spread cannot carry electrodes")

    completed = _parse_completed_timestamp(
        row.get("adjudication_completed_at"), field="adjudication_completed_at"
    )
    for reader in (reader_a, reader_b):
        reader_completed = _parse_completed_timestamp(
            reader.get("review_completed_at"), field="review_completed_at"
        )
        if completed < reader_completed:
            raise ValueError("Adjudication predates an independent reader completion")

    expected_domains: set[str] = set()
    comparisons = {
        "target_availability": "target_availability",
        "candidate_positive_set": "candidate_positive_electrodes",
        "set_exhaustiveness": "set_exhaustive",
        "spread": "known_spread_electrodes",
        "event_consistency": "patient_event_consistency",
        "confidence": "label_confidence",
    }
    for domain, field in comparisons.items():
        left = reader_a.get(field)
        right = reader_b.get(field)
        if isinstance(left, list) and isinstance(right, list):
            differs = set(left) != set(right)
        else:
            differs = left != right
        if differs:
            expected_domains.add(domain)
    left_states = reader_a.get("electrode_states")
    right_states = reader_b.get("electrode_states")
    if isinstance(left_states, Mapping) and isinstance(right_states, Mapping) and any(
        {left_states.get(electrode), right_states.get(electrode)}
        & {"reviewed_not_candidate", "unknown", "unavailable"}
        and left_states.get(electrode) != right_states.get(electrode)
        for electrode in STANDARD_19
    ):
        expected_domains.add("reviewed_negative_or_unknown")
    declared = row.get("disagreement_domains")
    if not isinstance(declared, list) or set(declared) != expected_domains:
        raise ValueError("Adjudication disagreement domains do not match reader records")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def build_s1_reader_pack(
    *,
    source_manifest_path: Path,
    master_receipt_path: Path,
    deepsoz_split_path: Path,
    output_dir: Path,
    expected_deepsoz_identity_count: int = EXPECTED_DEEPSOZ_IDENTITY_COUNT,
) -> dict[str, object]:
    source = _load_json(source_manifest_path)
    master = _load_json(master_receipt_path)
    patients, eligible_paths = _eligible_source_patients(source)
    deepsoz_ids = _deepsoz_local_ids(
        deepsoz_split_path,
        expected_count=expected_deepsoz_identity_count,
    )
    overlap = set(patients) & deepsoz_ids
    if overlap:
        raise ValueError(f"S1 candidates overlap DeepSOZ identities: {sorted(overlap)}")
    grouped = _events_for_patients(master, patients, eligible_paths)
    preprocess_config = _preprocess_config(master)
    assignment = _cohort_assignment(
        patients,
        {patient: len(grouped[patient]) for patient in patients},
    )

    target = output_dir.absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        patient_linkage: list[dict[str, object]] = []
        event_linkage: list[dict[str, object]] = []
        case_by_patient: dict[str, str] = {}
        for cohort, prefix in (
            ("s1_development", "S1-D"),
            ("s1_calibration", "S1-C"),
            ("s1_locked", "S1-L"),
        ):
            cohort_patients = sorted(patient for patient in patients if assignment[patient] == cohort)
            random.Random(ORDER_SEED + list(COHORT_SIZES).index(cohort)).shuffle(cohort_patients)
            for index, patient in enumerate(cohort_patients, start=1):
                case_id = f"{prefix}-{index:03d}"
                case_by_patient[patient] = case_id
                patient_linkage.append(
                    {
                        "case_id": case_id,
                        "cohort": cohort,
                        "patient_pseudonym": patient,
                        "available_event_count": len(grouped[patient]),
                        "target_semantics": TARGET_SEMANTICS,
                    }
                )
                for event_index, event in enumerate(grouped[patient], start=1):
                    event_case_id = f"{case_id}-E{event_index:03d}"
                    event_linkage.append(
                        {
                            "case_id": case_id,
                            "event_case_id": event_case_id,
                            "cohort": cohort,
                            "patient_pseudonym": patient,
                            "event_pseudonym": event["event_id"],
                            "relative_edf_path": event["relative_edf_path"],
                            "global_event_t0_sec": event["global_event_t0_sec"],
                            "global_event_stop_sec": event["global_event_stop_sec"],
                            "suggested_review_window_start_sec": max(0.0, float(event["global_event_t0_sec"]) - 30.0),
                            "suggested_review_window_stop_sec_unclamped": float(event["global_event_stop_sec"]) + 60.0,
                            "event_anchor_semantics": event["event_anchor_semantics"],
                            "source_montage": event["source_montage"],
                            "session_id": event["session_id"],
                        }
                    )

        for filename, rows in (("patient_linkage.csv", patient_linkage), ("event_linkage.csv", event_linkage)):
            with (temporary / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        event_ids_by_case: dict[str, list[str]] = {row["case_id"]: [] for row in patient_linkage}
        for row in event_linkage:
            event_ids_by_case[str(row["case_id"])].append(str(row["event_case_id"]))

        for reader_index, reader in enumerate(("reader_a", "reader_b")):
            for cohort in COHORT_SIZES:
                ids = [str(row["case_id"]) for row in patient_linkage if row["cohort"] == cohort]
                random.Random(ORDER_SEED + 100 * (reader_index + 1) + list(COHORT_SIZES).index(cohort)).shuffle(ids)
                rows = [_blank_annotation(case_id, cohort, order) for order, case_id in enumerate(ids, start=1)]
                _write_jsonl(temporary / f"{reader}_{cohort}.jsonl", rows)
        for cohort in COHORT_SIZES:
            ids = [str(row["case_id"]) for row in patient_linkage if row["cohort"] == cohort]
            _write_jsonl(
                temporary / f"adjudication_{cohort}.jsonl",
                [_blank_adjudication(case_id, cohort) for case_id in ids],
            )

        _write_json(temporary / "annotation_schema.json", _annotation_json_schema())
        _write_json(temporary / "adjudication_schema.json", _adjudication_json_schema())
        _write_json(temporary / "event_case_roster.json", event_ids_by_case)
        manifest: dict[str, object] = {
            "schema_version": PACK_SCHEMA,
            "status": "empty_label_fresh_eeg_only_s1_reader_pack_ready",
            "patient_count": len(patients),
            "event_count": sum(map(len, grouped.values())),
            "split_seed": SPLIT_SEED,
            "presentation_order_seed": ORDER_SEED,
            "cohort_counts": dict(COHORT_SIZES),
            "cohort_event_counts": {
                cohort: sum(
                    len(grouped[patient])
                    for patient in patients
                    if assignment[patient] == cohort
                )
                for cohort in COHORT_SIZES
            },
            "split_policy": (
                "prelabel_seeded_greedy_patient_event_count_balanced_"
                "exact_36_12_12_v1"
            ),
            "reader_count": 2,
            "adjudicator_count": 1,
            "target_semantics": TARGET_SEMANTICS,
            "signal_preprocessing_contract": {
                "preprocess_config": preprocess_config,
                "input_channels": list(STANDARD_19),
                "output_shape_per_event": [19, 12000],
                "sampling_frequency_hz": 200.0,
                "event_interval_sec": [-12.0, 48.0],
                "reference_representation": "primary_reference_then_CAR19",
            },
            "target_is_not": [
                "deepsoz_clinical_note_integrated_reference",
                "cortical_or_invasive_soz",
                "earliest_scalp_visible_electrode",
                "tusz_ictal_involvement",
                "spread_electrode",
                "surgical_target",
            ],
            "exposure_contract": {
                "new_s1_target_labels_previously_available": False,
                "waveforms_previously_used_in_target_free_dapt": True,
                "waveforms_previously_used_in_source_native_concept_development": True,
                "waveform_unseen_external_test": False,
                "label_fresh_patient_level_s1": True,
            },
            "access_receipt": {
                "deepsoz_identity_roster_loaded_for_exclusion_only": True,
                "deepsoz_target_values_loaded": False,
                "target_bearing_tusz_master_receipt_loaded_for_event_navigation": True,
                "tusz_channel_time_target_fields_present_in_source_receipt": True,
                "tusz_channel_time_target_values_used_for_selection_or_s1_labels": False,
                "tusz_channel_time_target_values_exported": False,
                "model_predictions_loaded": False,
                "model_predictions_exported": False,
                "private_eeg_loaded": False,
                "private_target_loaded": False,
                "automatic_soz_annotation_performed": False,
                "new_reader_labels_opened": False,
            },
            "label_release_policy": {
                "s1_development": "may_open_after_both_independent_reads_complete;used_for_preregistered_A0_A1_A2_development_only",
                "s1_calibration": "open_only_after_one_model_is_frozen;threshold_or_uncertainty_calibration_only",
                "s1_locked": "open_once_after_model_threshold_region_map_and_report_slots_are_frozen",
            },
            "files": {
                "patient_linkage": "patient_linkage.csv",
                "event_linkage": "event_linkage.csv",
                "event_case_roster": "event_case_roster.json",
                "annotation_schema": "annotation_schema.json",
                "adjudication_schema": "adjudication_schema.json",
                "independent_reader_templates": [
                    f"{reader}_{cohort}.jsonl"
                    for reader in ("reader_a", "reader_b")
                    for cohort in COHORT_SIZES
                ],
                "adjudication_templates": [f"adjudication_{cohort}.jsonl" for cohort in COHORT_SIZES],
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            "# TUSZ EEG-only patient-level S1 reader pack v1\n\n"
            "This pack contains 60 patients and all 433 signal-eligible seizure events. "
            "Two epilepsy-EEG experts independently review every available event in one patient bag "
            "and issue one standard-19 EEG-only scalp-electrode SOZ-candidate set. A third expert "
            "adjudicates only after both reads close.\n\n"
            "Do not show DeepSOZ targets, TUSZ channel involvement annotations, model predictions, "
            "private data, or the other reader's result during independent reading. The global TUSZ "
            "event time is a navigation anchor, not an onset-channel or SOZ answer.\n\n"
            "The upstream monolithic TUSZ master receipt contains involvement fields, so it was read "
            "to project event identity/time/path into this target-free pack. Those fields were not used "
            "for patient selection, S1 labeling, or export.\n\n"
            "Every electrode must be marked candidate_positive, reviewed_not_candidate, unknown, or "
            "unavailable. PZ is explicit. Spread is stored separately and never enters the SOZ-positive "
            "set. If all events were not reviewed, or no stable scalp-localizing evidence exists, the "
            "patient target must remain unavailable/indeterminate rather than guessed.\n\n"
            "These labels are fresh, but the waveforms are not: the same identities were used in "
            "target-free/source-native development. This is an EEG-only S1 development/calibration/"
            "locked-label resource, not a waveform-unseen external or integrated clinical validation cohort.\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--master-receipt", type=Path, default=DEFAULT_MASTER_RECEIPT)
    parser.add_argument("--deepsoz-split", type=Path, default=DEFAULT_DEEPSOZ_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_s1_reader_pack(
        source_manifest_path=args.source_manifest,
        master_receipt_path=args.master_receipt,
        deepsoz_split_path=args.deepsoz_split,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
