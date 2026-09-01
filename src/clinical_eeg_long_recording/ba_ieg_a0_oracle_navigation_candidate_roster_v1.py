"""Content-addressed BA-IEG A0 oracle-navigation candidate roster.

The roster is a *localization-target-independent* navigation artifact for the
70-patient, 318-record DeepSOZ/TUSZ source-train signal universe.  It is not
independent of seizure-detection targets: public TUSZ seizure intervals are
used as an oracle.  Its only semantic inputs are:

1. the already validated source-train signal identity binding; and
2. five explicitly whitelisted columns from ``event_inputs.csv`` describing
   public TUSZ global seizure intervals.

Every other CSV column is discarded before semantic parsing.  In particular,
seizure type, channel involvement, signal eligibility, DeepSOZ targets,
private/doctor data, embedded EDF annotations, and clinical text cannot enter
the roster.  Changes to non-whitelisted columns therefore cannot change its
content receipt.

A0 is explicitly a conditional-on-seizure-interval upper bound.  It is not a
detector-frozen end-to-end arm, and its events contain no localization-channel
target or seizure-type model field.
"""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, Sequence

from .ba_ieg_complete_patient_positive_set_bridge_v1 import (
    BA_IEG_NAVIGATION_ARM_A0,
    BA_IEG_RECORD_HAS_CANDIDATES,
    BA_IEG_RECORD_ZERO_CANDIDATE,
    BAIEGCompletePatientRecordRosterV1,
    BAIEGCompleteRecordRosterEntryV1,
)
from .deepsoz_tusz_identity_binding_v1 import (
    load_deepsoz_tusz_source_train_identity_binding_v1,
    validate_deepsoz_tusz_source_train_identity_binding_v1,
)


BA_IEG_A0_ORACLE_CANDIDATE_ROSTER_SCHEMA_V1: Final[str] = (
    "ba_ieg_a0_oracle_navigation_candidate_roster_v1"
)
BA_IEG_A0_ORACLE_CANDIDATE_ROSTER_METHOD_V1: Final[str] = (
    "source_train_identity_plus_public_tusz_global_seizure_intervals_v1"
)
BA_IEG_A0_EVENT_INTERVAL_WHITELIST_V1: Final[tuple[str, ...]] = (
    "local_edf_path",
    "event_id",
    "event_index",
    "t0_sec",
    "seizure_end_sec",
)
BA_IEG_A0_EXPECTED_PATIENTS_V1: Final[int] = 70
BA_IEG_A0_EXPECTED_RECORDS_V1: Final[int] = 318
BA_IEG_A0_EXPECTED_EVENTS_V1: Final[int] = 908
BA_IEG_A0_EXPECTED_ZERO_EVENT_RECORDS_V1: Final[int] = 2
BA_IEG_A0_RECORD_HAS_ORACLE_INTERVALS: Final[str] = (
    "oracle_global_seizure_intervals_present"
)
BA_IEG_A0_RECORD_ZERO_ORACLE_INTERVAL: Final[str] = (
    "zero_oracle_global_seizure_interval"
)

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "method_id",
        "roster_id",
        "model_split",
        "navigation_arm",
        "identity_binding_sha256",
        "event_interval_projection_sha256",
        "oracle_navigation_receipt_sha256",
        "denominator_contract",
        "scope_receipt",
        "counts",
        "patients",
        "records",
        "events",
        "receipt_sha256",
    }
)
_PATIENT_KEYS = frozenset(
    {
        "patient_uid",
        "source_recording_ids",
        "model_recording_ids",
        "source_record_count",
        "event_count",
        "patient_roster_receipt_sha256",
    }
)
_RECORD_KEYS = frozenset(
    {
        "patient_uid",
        "source_recording_id",
        "model_recording_id",
        "source_container_sha256",
        "exact_container_equivalence_id",
        "recording_duration_fraction",
        "model_source_binding_sha256",
        "candidate_status",
        "expected_unique_occurrence_count",
        "expected_qualified_unique_occurrence_count",
        "model_event_ids",
        "record_roster_receipt_sha256",
    }
)
_EVENT_KEYS = frozenset(
    {
        "patient_uid",
        "source_recording_id",
        "model_recording_id",
        "source_event_id",
        "model_event_id",
        "event_index",
        "seizure_interval_seconds",
        "occurrence_equivalence_id",
        "event_receipt_sha256",
    }
)
_DENOMINATOR_CONTRACT = {
    "patients": BA_IEG_A0_EXPECTED_PATIENTS_V1,
    "records": BA_IEG_A0_EXPECTED_RECORDS_V1,
    "events": BA_IEG_A0_EXPECTED_EVENTS_V1,
    "zero_event_records": BA_IEG_A0_EXPECTED_ZERO_EVENT_RECORDS_V1,
}


def _scope_receipt_v1() -> dict[str, Any]:
    # Return a fresh nested value so caller mutation cannot alter the validator's
    # frozen comparison authority in the current Python process.
    return {
        "source_train_signal_identity_binding_used": True,
        "public_tusz_global_seizure_intervals_used_for_navigation": True,
        "event_inputs_whitelist_columns": list(
            BA_IEG_A0_EVENT_INTERVAL_WHITELIST_V1
        ),
        "non_whitelisted_event_input_columns_discarded_before_semantic_parsing": True,
        "event_seizure_type_used": False,
        "tusz_channel_involvement_opened": False,
        "deepsoz_target_opened": False,
        "private_data_opened": False,
        "doctor_labels_opened": False,
        "edf_embedded_annotations_opened": False,
        "clinical_text_opened": False,
        "candidate_selection_localization_target_independent": True,
        "seizure_detection_interval_oracle_used": True,
        "localization_channel_target_used": False,
        "evaluation_semantics": "conditional_on_seizure_interval_upper_bound",
        "navigation_is_oracle_conditional_not_detector_frozen": True,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or set(text) - _SHA256_ALPHABET:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _exact_keys(value: object, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} keys drifted from the whitelist")
    return value


def _strict_file(path: str | Path, name: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a regular file: {resolved}")
    return resolved


def _read_whitelisted_interval_rows(path: str | Path) -> list[dict[str, str]]:
    """Read exactly five semantic fields; all other columns remain opaque."""

    resolved = _strict_file(path, "event_inputs.csv")
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("event_inputs.csv is empty") from exc
        if len(header) != len(set(header)):
            raise ValueError("event_inputs.csv repeats a header name")
        missing = [
            item
            for item in BA_IEG_A0_EVENT_INTERVAL_WHITELIST_V1
            if item not in header
        ]
        if missing:
            raise ValueError(
                "event_inputs.csv lacks whitelisted global interval fields: "
                + ",".join(missing)
            )
        indices = {
            item: header.index(item)
            for item in BA_IEG_A0_EVENT_INTERVAL_WHITELIST_V1
        }
        projected: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"event_inputs.csv row {line_number} does not align with its header"
                )
            # Do not construct a mapping for any non-whitelisted column.
            projected.append(
                {
                    item: row[index]
                    for item, index in indices.items()
                }
            )
    return projected


def _finite_seconds(value: object, name: str) -> float:
    text = _identifier(value, name)
    try:
        decimal = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a decimal number") from exc
    if not decimal.is_finite():
        raise ValueError(f"{name} must be finite")
    result = float(decimal)
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds finite float range")
    return 0.0 if result == 0.0 else result


def _record_duration(record: Mapping[str, Any]) -> Fraction:
    value = record.get("recording_duration_fraction")
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise ValueError("identity record has an invalid duration fraction")
    return Fraction(value[0], value[1])


def _model_source_binding(
    *, identity_binding_sha256: str, record: Mapping[str, Any]
) -> str:
    return _canonical_sha256(
        {
            "schema": "ba_ieg_a0_model_source_record_binding_v1",
            "identity_binding_sha256": identity_binding_sha256,
            "patient_uid": record["patient_uid"],
            "source_recording_id": record["tusz_recording_id"],
            "model_recording_id": record["detector_recording_id"],
            "source_container_sha256": record["source_container_sha256"],
            "exact_container_equivalence_id": record[
                "exact_container_equivalence_id"
            ],
        }
    )


def build_ba_ieg_a0_oracle_navigation_candidate_roster_v1(
    *,
    identity_binding: Mapping[str, Any],
    event_inputs_csv_path: str | Path,
) -> dict[str, Any]:
    """Build the frozen 70/318/908 A0 navigation roster."""

    validate_deepsoz_tusz_source_train_identity_binding_v1(identity_binding)
    identity_receipt = _sha256(
        identity_binding.get("receipt_sha256"), "identity binding receipt"
    )
    identity_counts = identity_binding.get("counts")
    if not isinstance(identity_counts, Mapping) or (
        identity_counts.get("patients") != BA_IEG_A0_EXPECTED_PATIENTS_V1
        or identity_counts.get("source_records") != BA_IEG_A0_EXPECTED_RECORDS_V1
        or identity_counts.get("unique_exact_containers")
        != BA_IEG_A0_EXPECTED_RECORDS_V1
    ):
        raise ValueError("A0 identity binding denominator is not frozen at 70/318")

    identity_records = sorted(
        identity_binding["records"],
        key=lambda item: (str(item["patient_uid"]), str(item["tusz_recording_id"])),
    )
    records_by_source = {
        str(item["tusz_recording_id"]): item for item in identity_records
    }
    if len(records_by_source) != BA_IEG_A0_EXPECTED_RECORDS_V1:
        raise ValueError("A0 identity binding repeats a source recording")

    projected_rows = _read_whitelisted_interval_rows(event_inputs_csv_path)
    selected_rows = [
        row
        for row in projected_rows
        if row["local_edf_path"] in records_by_source
    ]
    # Only selected source-train rows and the exact whitelist enter this hash;
    # the raw CSV file hash is intentionally absent.
    interval_projection_receipt = _canonical_sha256(
        {
            "schema": "ba_ieg_a0_source_train_global_interval_projection_v1",
            "whitelist_columns": list(BA_IEG_A0_EVENT_INTERVAL_WHITELIST_V1),
            "rows": sorted(
                selected_rows,
                key=lambda item: (
                    item["local_edf_path"],
                    int(item["event_index"]),
                    item["event_id"],
                ),
            ),
        }
    )

    events_by_record: dict[str, list[dict[str, Any]]] = {
        item: [] for item in records_by_source
    }
    seen_source_event_ids: set[str] = set()
    for row in selected_rows:
        source_recording_id = _identifier(
            row["local_edf_path"], "source recording ID"
        )
        record = records_by_source[source_recording_id]
        source_event_id = _identifier(row["event_id"], "source event ID")
        try:
            event_index = int(row["event_index"])
        except ValueError as exc:
            raise ValueError("event index must be an integer") from exc
        if event_index < 0 or event_index >= 10000 or str(event_index) != row[
            "event_index"
        ]:
            raise ValueError("event index must be canonical non-negative decimal")
        expected_event_id = (
            PurePosixPath(source_recording_id).stem
            + "__ev"
            + str(event_index).zfill(4)
        )
        if source_event_id != expected_event_id:
            raise ValueError("source event ID disagrees with record/event index")
        if source_event_id in seen_source_event_ids:
            raise ValueError("A0 interval projection repeats a source event")
        seen_source_event_ids.add(source_event_id)
        start = _finite_seconds(row["t0_sec"], "seizure start")
        stop = _finite_seconds(row["seizure_end_sec"], "seizure stop")
        duration = _record_duration(record)
        if start < 0.0 or stop <= start or Fraction(str(stop)) > duration:
            raise ValueError("global seizure interval is outside its source record")
        model_source_binding_sha256 = _model_source_binding(
            identity_binding_sha256=identity_receipt, record=record
        )
        event_body = {
            "schema": "ba_ieg_a0_oracle_navigation_event_receipt_v1",
            "identity_binding_sha256": identity_receipt,
            "event_interval_projection_sha256": interval_projection_receipt,
            "model_source_binding_sha256": model_source_binding_sha256,
            "patient_uid": record["patient_uid"],
            "source_recording_id": source_recording_id,
            "model_recording_id": record["detector_recording_id"],
            "source_event_id": source_event_id,
            "event_index": event_index,
            "seizure_interval_seconds": [start, stop],
        }
        event_receipt = _canonical_sha256(event_body)
        events_by_record[source_recording_id].append(
            {
                "patient_uid": str(record["patient_uid"]),
                "source_recording_id": source_recording_id,
                "model_recording_id": str(record["detector_recording_id"]),
                "source_event_id": source_event_id,
                "model_event_id": "BAIEG-A0EVT-" + event_receipt[:24],
                "event_index": event_index,
                "seizure_interval_seconds": [start, stop],
                "occurrence_equivalence_id": (
                    "TUSZ-GLOBAL-SEIZURE-" + event_receipt
                ),
                "event_receipt_sha256": event_receipt,
            }
        )

    event_count = sum(len(items) for items in events_by_record.values())
    if event_count != BA_IEG_A0_EXPECTED_EVENTS_V1:
        raise ValueError(
            f"A0 global interval denominator drifted: events={event_count}"
        )
    for source_recording_id, rows in events_by_record.items():
        rows.sort(key=lambda item: int(item["event_index"]))
        if [item["event_index"] for item in rows] != list(range(len(rows))):
            raise ValueError(
                f"A0 event indices are not contiguous for {source_recording_id}"
            )

    event_rows = [
        item
        for source_recording_id in sorted(events_by_record)
        for item in events_by_record[source_recording_id]
    ]
    oracle_navigation_receipt = _canonical_sha256(
        {
            "schema": "ba_ieg_a0_oracle_navigation_receipt_v1",
            "identity_binding_sha256": identity_receipt,
            "event_interval_projection_sha256": interval_projection_receipt,
            "event_receipt_sha256s": [
                item["event_receipt_sha256"] for item in event_rows
            ],
            "seizure_detection_interval_oracle_used": True,
            "localization_channel_target_used": False,
            "event_seizure_type_used": False,
        }
    )

    record_rows: list[dict[str, Any]] = []
    for record in identity_records:
        source_recording_id = str(record["tusz_recording_id"])
        model_recording_id = str(record["detector_recording_id"])
        events = events_by_record[source_recording_id]
        model_source_binding_sha256 = _model_source_binding(
            identity_binding_sha256=identity_receipt, record=record
        )
        candidate_count = len(events)
        candidate_status = (
            BA_IEG_A0_RECORD_HAS_ORACLE_INTERVALS
            if candidate_count
            else BA_IEG_A0_RECORD_ZERO_ORACLE_INTERVAL
        )
        body = {
            "schema": "ba_ieg_a0_oracle_navigation_record_receipt_v1",
            "identity_binding_sha256": identity_receipt,
            "oracle_navigation_receipt_sha256": oracle_navigation_receipt,
            "patient_uid": record["patient_uid"],
            "source_recording_id": source_recording_id,
            "model_recording_id": model_recording_id,
            "source_container_sha256": record["source_container_sha256"],
            "exact_container_equivalence_id": record[
                "exact_container_equivalence_id"
            ],
            "recording_duration_fraction": record["recording_duration_fraction"],
            "model_source_binding_sha256": model_source_binding_sha256,
            "candidate_status": candidate_status,
            "expected_unique_occurrence_count": candidate_count,
            "expected_qualified_unique_occurrence_count": candidate_count,
            "model_event_ids": [item["model_event_id"] for item in events],
        }
        receipt = _canonical_sha256(body)
        body.pop("schema")
        body.pop("identity_binding_sha256")
        body.pop("oracle_navigation_receipt_sha256")
        body["record_roster_receipt_sha256"] = receipt
        record_rows.append(body)

    records_by_patient: dict[str, list[dict[str, Any]]] = {}
    for item in record_rows:
        records_by_patient.setdefault(str(item["patient_uid"]), []).append(item)
    patient_rows: list[dict[str, Any]] = []
    for patient_uid in sorted(records_by_patient):
        records = records_by_patient[patient_uid]
        patient_body = {
            "schema": "ba_ieg_a0_oracle_navigation_patient_roster_receipt_v1",
            "identity_binding_sha256": identity_receipt,
            "oracle_navigation_receipt_sha256": oracle_navigation_receipt,
            "patient_uid": patient_uid,
            "source_recording_ids": [
                item["source_recording_id"] for item in records
            ],
            "model_recording_ids": [item["model_recording_id"] for item in records],
            "source_record_count": len(records),
            "event_count": sum(
                int(item["expected_unique_occurrence_count"]) for item in records
            ),
        }
        receipt = _canonical_sha256(patient_body)
        patient_body.pop("schema")
        patient_body.pop("identity_binding_sha256")
        patient_body.pop("oracle_navigation_receipt_sha256")
        patient_body["patient_roster_receipt_sha256"] = receipt
        patient_rows.append(patient_body)

    zero_event_records = sum(
        item["candidate_status"] == BA_IEG_A0_RECORD_ZERO_ORACLE_INTERVAL
        for item in record_rows
    )
    counts = {
        "patients": len(patient_rows),
        "records": len(record_rows),
        "events": len(event_rows),
        "records_with_events": len(record_rows) - zero_event_records,
        "zero_event_records": zero_event_records,
    }
    if counts != {
        "patients": 70,
        "records": 318,
        "events": 908,
        "records_with_events": 316,
        "zero_event_records": 2,
    }:
        raise ValueError(f"A0 candidate denominator drifted: {counts}")

    body: dict[str, Any] = {
        "schema_version": BA_IEG_A0_ORACLE_CANDIDATE_ROSTER_SCHEMA_V1,
        "method_id": BA_IEG_A0_ORACLE_CANDIDATE_ROSTER_METHOD_V1,
        "roster_id": "BAIEG-A0-ORACLE-" + oracle_navigation_receipt[:24],
        "model_split": "source_train",
        "navigation_arm": BA_IEG_NAVIGATION_ARM_A0,
        "identity_binding_sha256": identity_receipt,
        "event_interval_projection_sha256": interval_projection_receipt,
        "oracle_navigation_receipt_sha256": oracle_navigation_receipt,
        "denominator_contract": dict(_DENOMINATOR_CONTRACT),
        "scope_receipt": _scope_receipt_v1(),
        "counts": counts,
        "patients": patient_rows,
        "records": record_rows,
        "events": event_rows,
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(body)
    return body


def validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(
    payload: Mapping[str, Any],
) -> None:
    top = _exact_keys(payload, _TOP_LEVEL_KEYS, "A0 candidate roster")
    if top.get("schema_version") != BA_IEG_A0_ORACLE_CANDIDATE_ROSTER_SCHEMA_V1 or (
        top.get("method_id") != BA_IEG_A0_ORACLE_CANDIDATE_ROSTER_METHOD_V1
    ):
        raise ValueError("A0 candidate roster schema/method drifted")
    if top.get("model_split") != "source_train" or top.get(
        "navigation_arm"
    ) != BA_IEG_NAVIGATION_ARM_A0:
        raise ValueError("A0 candidate roster crossed split/navigation arm")
    receipt = _sha256(top.get("receipt_sha256"), "A0 candidate roster receipt")
    body = dict(top)
    body.pop("receipt_sha256")
    if _canonical_sha256(body) != receipt:
        raise ValueError("A0 candidate roster receipt does not replay")
    identity_receipt = _sha256(
        top.get("identity_binding_sha256"), "identity binding receipt"
    )
    projection_receipt = _sha256(
        top.get("event_interval_projection_sha256"), "event interval projection"
    )
    oracle_receipt = _sha256(
        top.get("oracle_navigation_receipt_sha256"), "oracle navigation receipt"
    )
    if top.get("roster_id") != "BAIEG-A0-ORACLE-" + oracle_receipt[:24]:
        raise ValueError("A0 roster ID does not derive from navigation receipt")
    if top.get("denominator_contract") != _DENOMINATOR_CONTRACT:
        raise ValueError("A0 denominator contract drifted")
    if top.get("scope_receipt") != _scope_receipt_v1():
        raise ValueError("A0 scope/permission receipt drifted")
    counts = top.get("counts")
    expected_counts = {
        "patients": 70,
        "records": 318,
        "events": 908,
        "records_with_events": 316,
        "zero_event_records": 2,
    }
    if counts != expected_counts:
        raise ValueError("A0 roster counts drifted from 70/318/908/2")

    patients = top.get("patients")
    records = top.get("records")
    events = top.get("events")
    if not isinstance(patients, list) or not isinstance(records, list) or not isinstance(
        events, list
    ):
        raise TypeError("A0 patient/record/event rosters must be arrays")
    if (len(patients), len(records), len(events)) != (70, 318, 908):
        raise ValueError("A0 roster arrays disagree with frozen counts")

    patient_by_uid: dict[str, Mapping[str, Any]] = {}
    for item in patients:
        row = _exact_keys(item, _PATIENT_KEYS, "A0 patient row")
        patient_uid = _identifier(row.get("patient_uid"), "patient UID")
        if patient_uid in patient_by_uid:
            raise ValueError("A0 roster repeats a patient UID")
        source_ids = row.get("source_recording_ids")
        model_ids = row.get("model_recording_ids")
        if (
            not isinstance(source_ids, list)
            or not isinstance(model_ids, list)
            or row.get("source_record_count") != len(source_ids)
            or len(source_ids) != len(model_ids)
            or not source_ids
        ):
            raise ValueError("A0 patient row has an invalid record roster")
        patient_body = {
            "schema": "ba_ieg_a0_oracle_navigation_patient_roster_receipt_v1",
            "identity_binding_sha256": identity_receipt,
            "oracle_navigation_receipt_sha256": oracle_receipt,
            **{key: value for key, value in row.items() if key != "patient_roster_receipt_sha256"},
        }
        if _canonical_sha256(patient_body) != _sha256(
            row.get("patient_roster_receipt_sha256"), "patient roster receipt"
        ):
            raise ValueError("A0 patient roster receipt does not replay")
        patient_by_uid[patient_uid] = row

    record_by_model: dict[str, Mapping[str, Any]] = {}
    record_by_source: dict[str, Mapping[str, Any]] = {}
    event_ids_from_records: list[str] = []
    zero_records = 0
    records_by_patient: dict[str, list[Mapping[str, Any]]] = {
        item: [] for item in patient_by_uid
    }
    for item in records:
        row = _exact_keys(item, _RECORD_KEYS, "A0 record row")
        patient_uid = _identifier(row.get("patient_uid"), "record patient UID")
        if patient_uid not in patient_by_uid:
            raise ValueError("A0 record has no patient row")
        source_id = _identifier(row.get("source_recording_id"), "source recording ID")
        model_id = _identifier(row.get("model_recording_id"), "model recording ID")
        if source_id in record_by_source or model_id in record_by_model:
            raise ValueError("A0 roster repeats a source/model recording")
        container = _sha256(row.get("source_container_sha256"), "source container")
        if row.get("exact_container_equivalence_id") != (
            "TUSZ-EDF-CONTAINER-" + container
        ):
            raise ValueError("A0 record exact-container identity drifted")
        _sha256(row.get("model_source_binding_sha256"), "model/source binding")
        _record_duration(row)
        event_ids = row.get("model_event_ids")
        total = row.get("expected_unique_occurrence_count")
        qualified = row.get("expected_qualified_unique_occurrence_count")
        if (
            not isinstance(event_ids, list)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or qualified != total
            or total != len(event_ids)
        ):
            raise ValueError("A0 record occurrence counts/event IDs drifted")
        expected_status = (
            BA_IEG_A0_RECORD_HAS_ORACLE_INTERVALS
            if total > 0
            else BA_IEG_A0_RECORD_ZERO_ORACLE_INTERVAL
        )
        if row.get("candidate_status") != expected_status:
            raise ValueError("A0 record candidate status disagrees with event count")
        zero_records += total == 0
        record_body = {
            "schema": "ba_ieg_a0_oracle_navigation_record_receipt_v1",
            "identity_binding_sha256": identity_receipt,
            "oracle_navigation_receipt_sha256": oracle_receipt,
            **{key: value for key, value in row.items() if key != "record_roster_receipt_sha256"},
        }
        if _canonical_sha256(record_body) != _sha256(
            row.get("record_roster_receipt_sha256"), "record roster receipt"
        ):
            raise ValueError("A0 record roster receipt does not replay")
        record_by_source[source_id] = row
        record_by_model[model_id] = row
        records_by_patient[patient_uid].append(row)
        event_ids_from_records.extend(str(value) for value in event_ids)
    if zero_records != 2:
        raise ValueError("A0 roster did not preserve two zero-event records")

    seen_source_events: set[str] = set()
    seen_model_events: set[str] = set()
    event_receipts: list[str] = []
    events_by_record: dict[str, list[Mapping[str, Any]]] = {
        item: [] for item in record_by_source
    }
    for item in events:
        row = _exact_keys(item, _EVENT_KEYS, "A0 event row")
        if "seizure_type" in row:
            raise ValueError("seizure type cannot enter an A0 model event")
        patient_uid = _identifier(row.get("patient_uid"), "event patient UID")
        source_id = _identifier(row.get("source_recording_id"), "event source record")
        model_id = _identifier(row.get("model_recording_id"), "event model record")
        source_event = _identifier(row.get("source_event_id"), "source event ID")
        model_event = _identifier(row.get("model_event_id"), "model event ID")
        record = record_by_source.get(source_id)
        if (
            record is None
            or record["patient_uid"] != patient_uid
            or record["model_recording_id"] != model_id
        ):
            raise ValueError("A0 event crosses patient/record identity")
        if source_event in seen_source_events or model_event in seen_model_events:
            raise ValueError("A0 roster repeats an event identity")
        seen_source_events.add(source_event)
        seen_model_events.add(model_event)
        event_index = row.get("event_index")
        interval = row.get("seizure_interval_seconds")
        if (
            isinstance(event_index, bool)
            or not isinstance(event_index, int)
            or event_index < 0
            or not isinstance(interval, list)
            or len(interval) != 2
        ):
            raise ValueError("A0 event index/interval shape is invalid")
        start, stop = (float(value) for value in interval)
        if (
            not math.isfinite(start)
            or not math.isfinite(stop)
            or start < 0.0
            or stop <= start
            or Fraction(str(stop)) > _record_duration(record)
        ):
            raise ValueError("A0 event interval is outside its source record")
        expected_source_event = (
            PurePosixPath(source_id).stem + "__ev" + str(event_index).zfill(4)
        )
        if source_event != expected_source_event:
            raise ValueError("A0 source event identity drifted")
        event_body = {
            "schema": "ba_ieg_a0_oracle_navigation_event_receipt_v1",
            "identity_binding_sha256": identity_receipt,
            "event_interval_projection_sha256": projection_receipt,
            "model_source_binding_sha256": record["model_source_binding_sha256"],
            "patient_uid": patient_uid,
            "source_recording_id": source_id,
            "model_recording_id": model_id,
            "source_event_id": source_event,
            "event_index": event_index,
            "seizure_interval_seconds": [start, stop],
        }
        event_receipt = _sha256(row.get("event_receipt_sha256"), "event receipt")
        if _canonical_sha256(event_body) != event_receipt or model_event != (
            "BAIEG-A0EVT-" + event_receipt[:24]
        ) or row.get("occurrence_equivalence_id") != (
            "TUSZ-GLOBAL-SEIZURE-" + event_receipt
        ):
            raise ValueError("A0 event content-addressed identity drifted")
        event_receipts.append(event_receipt)
        events_by_record[source_id].append(row)

    if set(event_ids_from_records) != seen_model_events or len(
        event_ids_from_records
    ) != len(seen_model_events):
        raise ValueError("A0 record event rosters do not cover events exactly once")
    ordered_event_receipts: list[str] = []
    for source_id in sorted(events_by_record):
        rows = sorted(events_by_record[source_id], key=lambda item: item["event_index"])
        if [item["event_index"] for item in rows] != list(range(len(rows))):
            raise ValueError("A0 event indices are not contiguous per record")
        expected_ids = record_by_source[source_id]["model_event_ids"]
        if [item["model_event_id"] for item in rows] != expected_ids:
            raise ValueError("A0 record/event order roster drifted")
        ordered_event_receipts.extend(item["event_receipt_sha256"] for item in rows)
    expected_oracle = _canonical_sha256(
        {
            "schema": "ba_ieg_a0_oracle_navigation_receipt_v1",
            "identity_binding_sha256": identity_receipt,
            "event_interval_projection_sha256": projection_receipt,
            "event_receipt_sha256s": ordered_event_receipts,
            "seizure_detection_interval_oracle_used": True,
            "localization_channel_target_used": False,
            "event_seizure_type_used": False,
        }
    )
    if oracle_receipt != expected_oracle:
        raise ValueError("A0 oracle navigation receipt does not replay")

    for patient_uid, patient in patient_by_uid.items():
        rows = records_by_patient[patient_uid]
        if patient["source_recording_ids"] != [
            item["source_recording_id"] for item in rows
        ] or patient["model_recording_ids"] != [
            item["model_recording_id"] for item in rows
        ] or patient["source_record_count"] != len(rows) or patient[
            "event_count"
        ] != sum(item["expected_unique_occurrence_count"] for item in rows):
            raise ValueError("A0 patient roster does not replay from records")


def load_ba_ieg_a0_oracle_navigation_candidate_roster_v1(
    path: str | Path,
) -> dict[str, Any]:
    resolved = _strict_file(path, "A0 oracle candidate roster")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(payload)
    return payload


def materialize_ba_ieg_a0_oracle_navigation_candidate_roster_v1(
    payload: Mapping[str, Any], output_path: str | Path
) -> Path:
    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(payload)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise FileExistsError("refusing to overwrite a different A0 roster")
        return destination
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def ba_ieg_a0_candidate_roster_to_complete_patient_roster_v1(
    payload: Mapping[str, Any],
) -> BAIEGCompletePatientRecordRosterV1:
    """Convert A0 only; this adapter has no A1 detector-receipt surface."""

    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(payload)
    records = tuple(
        BAIEGCompleteRecordRosterEntryV1(
            patient_uid=str(item["patient_uid"]),
            source_recording_id=str(item["source_recording_id"]),
            model_recording_id=str(item["model_recording_id"]),
            source_container_sha256=str(item["source_container_sha256"]),
            exact_container_equivalence_id=str(
                item["exact_container_equivalence_id"]
            ),
            model_source_binding_sha256=str(item["model_source_binding_sha256"]),
            candidate_status=(
                BA_IEG_RECORD_HAS_CANDIDATES
                if int(item["expected_unique_occurrence_count"]) > 0
                else BA_IEG_RECORD_ZERO_CANDIDATE
            ),
            expected_unique_occurrence_count=int(
                item["expected_unique_occurrence_count"]
            ),
            expected_qualified_unique_occurrence_count=int(
                item["expected_qualified_unique_occurrence_count"]
            ),
        )
        for item in payload["records"]
    )
    return BAIEGCompletePatientRecordRosterV1(
        identity_binding_sha256=str(payload["identity_binding_sha256"]),
        candidate_roster_receipt_sha256=str(payload["receipt_sha256"]),
        navigation_arm=BA_IEG_NAVIGATION_ARM_A0,
        records=records,
        oracle_navigation_receipt_sha256=str(
            payload["oracle_navigation_receipt_sha256"]
        ),
        oracle_event_intervals_used_for_navigation=True,
    )


def build_ba_ieg_a0_oracle_navigation_candidate_roster_from_paths_v1(
    *, identity_binding_path: str | Path, event_inputs_csv_path: str | Path
) -> dict[str, Any]:
    return build_ba_ieg_a0_oracle_navigation_candidate_roster_v1(
        identity_binding=load_deepsoz_tusz_source_train_identity_binding_v1(
            identity_binding_path
        ),
        event_inputs_csv_path=event_inputs_csv_path,
    )


__all__ = [
    "BA_IEG_A0_ORACLE_CANDIDATE_ROSTER_SCHEMA_V1",
    "BA_IEG_A0_ORACLE_CANDIDATE_ROSTER_METHOD_V1",
    "BA_IEG_A0_EVENT_INTERVAL_WHITELIST_V1",
    "BA_IEG_A0_EXPECTED_PATIENTS_V1",
    "BA_IEG_A0_EXPECTED_RECORDS_V1",
    "BA_IEG_A0_EXPECTED_EVENTS_V1",
    "BA_IEG_A0_EXPECTED_ZERO_EVENT_RECORDS_V1",
    "BA_IEG_A0_RECORD_HAS_ORACLE_INTERVALS",
    "BA_IEG_A0_RECORD_ZERO_ORACLE_INTERVAL",
    "build_ba_ieg_a0_oracle_navigation_candidate_roster_v1",
    "build_ba_ieg_a0_oracle_navigation_candidate_roster_from_paths_v1",
    "validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1",
    "load_ba_ieg_a0_oracle_navigation_candidate_roster_v1",
    "materialize_ba_ieg_a0_oracle_navigation_candidate_roster_v1",
    "ba_ieg_a0_candidate_roster_to_complete_patient_roster_v1",
]
