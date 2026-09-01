"""Target-free DeepSOZ numeric-patient to TUSZ signal identity binding.

This module closes an identity seam only.  It reads the released DeepSOZ/TUSZ
split and record crosswalk plus the target-free complete TUSZ EDF inventory.
It never opens DeepSOZ channel targets, TUSZ event annotations, private labels,
clinical text, or source-eval labels.  The v1 builder is deliberately restricted
to ``source_train`` so a localization target cannot be silently rebound to a
different local patient or recording identity.

The exact EDF-container SHA-256 is used as the occurrence-equivalence authority.
It is an identity key, not a claim that differently encoded EDF containers have
different physiological signals.  A later P0 runner must additionally bind its
canonical sample receipt to this source-container hash.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, Mapping, Sequence


DEEPSOZ_TUSZ_SOURCE_TRAIN_IDENTITY_BINDING_SCHEMA_V1: Final[str] = (
    "deepsoz_tusz_source_train_signal_identity_binding_v1"
)
DEEPSOZ_TUSZ_SOURCE_TRAIN_IDENTITY_BINDING_METHOD_V1: Final[str] = (
    "numeric_deepsoz_to_exact_tusz_edf_container_identity_v1"
)
DEEPSOZ_TUSZ_SOURCE_TRAIN_EXPECTED_PATIENTS_V1: Final[int] = 70
DEEPSOZ_TUSZ_SOURCE_TRAIN_EXPECTED_RECORDS_V1: Final[int] = 318

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_LOCAL_PATIENT_RE = re.compile(r"^[a-z0-9]+$")
_RECORD_SESSION_TAIL_RE = re.compile(r"^[^_]+(_s[0-9]+_t[0-9]+\.edf)$")


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_file(path: str | Path, name: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _read_csv(path: Path, required: Sequence[str], name: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = sorted(set(required) - set(fields))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        # Project immediately to the identity allowlist.  The source assets
        # contain unrelated event/label-derived columns that are deliberately
        # unavailable to this builder even though they share the CSV file.
        rows = [{field: row[field] for field in required} for row in reader]
    if not rows:
        raise ValueError(f"{name} must not be empty")
    return rows


def _normal_numeric_patient_id(value: object) -> str:
    text = str(value)
    if not text or text != text.strip() or not text.isdigit():
        raise ValueError("deepsoz_patient_id must be a trimmed numeric identifier")
    result = str(int(text))
    if result != text:
        raise ValueError("deepsoz_patient_id must use canonical decimal form")
    return result


def _local_patient_id(value: object) -> str:
    text = str(value)
    if text != text.strip() or _LOCAL_PATIENT_RE.fullmatch(text) is None:
        raise ValueError("local_patient_id must be a canonical lowercase TUSZ ID")
    return text


def _source_train_patient_uid(local_patient_id: str) -> str:
    return f"TUSZ-SOURCE_TRAIN-{local_patient_id.upper()}"


def _safe_source_train_recording_id(value: object, local_patient_id: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError("local_edf_path must be a trimmed relative path")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix.lower() != ".edf"
        or len(relative.parts) < 3
        or relative.parts[0] != "train"
        or relative.parts[1] != local_patient_id
    ):
        raise ValueError("local_edf_path is not a safe source-train TUSZ identity")
    return relative.as_posix()


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _record_session_tail(value: object, name: str) -> str:
    basename = PurePosixPath(str(value)).name
    match = _RECORD_SESSION_TAIL_RE.fullmatch(basename)
    if match is None:
        raise ValueError(f"{name} has no canonical session/task suffix")
    return match.group(1)


def _read_roster(path: Path) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("complete TUSZ roster must contain a records array")
    rows: dict[str, Mapping[str, Any]] = {}
    for item in payload["records"]:
        if not isinstance(item, Mapping):
            raise TypeError("complete TUSZ roster records must be mappings")
        recording_id = str(item.get("recording_id", ""))
        if not recording_id or recording_id in rows:
            raise ValueError("complete TUSZ roster repeats or omits recording_id")
        rows[recording_id] = item
    return payload, rows


def build_deepsoz_tusz_source_train_identity_binding_v1(
    *,
    split_manifest_path: str | Path,
    record_crosswalk_path: str | Path,
    complete_tusz_roster_path: str | Path,
    expected_patient_count: int = DEEPSOZ_TUSZ_SOURCE_TRAIN_EXPECTED_PATIENTS_V1,
    expected_record_count: int = DEEPSOZ_TUSZ_SOURCE_TRAIN_EXPECTED_RECORDS_V1,
) -> dict[str, Any]:
    """Build the target-free, source-train-only identity receipt."""

    if expected_patient_count < 1 or expected_record_count < 1:
        raise ValueError("expected source-train counts must be positive")
    split_path = _strict_file(split_manifest_path, "DeepSOZ split manifest")
    crosswalk_path = _strict_file(record_crosswalk_path, "DeepSOZ record crosswalk")
    roster_path = _strict_file(complete_tusz_roster_path, "complete TUSZ roster")

    split_required = (
        "deepsoz_patient_id",
        "local_patient_id",
        "official_split",
        "model_split",
        "cohort_status",
        "source_record_count",
        "unique_mapped_record_count",
    )
    split_rows = _read_csv(split_path, split_required, "DeepSOZ split manifest")
    source_train_rows = [row for row in split_rows if row["model_split"] == "source_train"]
    if not source_train_rows:
        raise ValueError("DeepSOZ split manifest has no source_train patients")

    patients_by_deepsoz: dict[str, dict[str, Any]] = {}
    local_to_deepsoz: dict[str, str] = {}
    for row in source_train_rows:
        deepsoz_id = _normal_numeric_patient_id(row["deepsoz_patient_id"])
        local_id = _local_patient_id(row["local_patient_id"])
        if row["official_split"] != "train" or row["cohort_status"] != (
            "included_positive_only"
        ):
            raise ValueError("source_train identity row is not an included train patient")
        if deepsoz_id in patients_by_deepsoz:
            raise ValueError("DeepSOZ source_train patient ID is duplicated")
        previous = local_to_deepsoz.setdefault(local_id, deepsoz_id)
        if previous != deepsoz_id:
            raise ValueError("one local TUSZ patient maps to multiple DeepSOZ IDs")
        try:
            source_records = int(row["source_record_count"])
            unique_records = int(row["unique_mapped_record_count"])
        except ValueError as exc:
            raise ValueError("source-train record counts must be integers") from exc
        if source_records < 1 or unique_records != source_records:
            raise ValueError("source_train identity requires complete unique record mapping")
        patients_by_deepsoz[deepsoz_id] = {
            "deepsoz_patient_id": deepsoz_id,
            "local_patient_id": local_id,
            "patient_uid": _source_train_patient_uid(local_id),
            "model_split": "source_train",
            "expected_source_record_count": source_records,
        }

    crosswalk_required = (
        "deepsoz_patient_id",
        "deepsoz_record",
        "mapping_status",
        "candidate_count",
        "local_patient_id",
        "local_official_split",
        "split_agreement",
        "local_edf_path",
        "local_edf_exists",
        "header_read_ok",
    )
    crosswalk_rows = _read_csv(
        crosswalk_path, crosswalk_required, "DeepSOZ record crosswalk"
    )
    _, roster_by_recording = _read_roster(roster_path)

    records: list[dict[str, Any]] = []
    seen_recordings: set[str] = set()
    for row in crosswalk_rows:
        raw_id = str(row["deepsoz_patient_id"])
        if not raw_id.isdigit() or str(int(raw_id)) not in patients_by_deepsoz:
            continue
        deepsoz_id = _normal_numeric_patient_id(raw_id)
        patient = patients_by_deepsoz[deepsoz_id]
        local_id = _local_patient_id(row["local_patient_id"])
        if local_id != patient["local_patient_id"]:
            raise ValueError("record crosswalk disagrees with patient identity")
        if (
            row["mapping_status"] != "unique"
            or row["candidate_count"] != "1"
            or row["local_official_split"] != "train"
            or row["split_agreement"] != "1"
            or row["local_edf_exists"] != "1"
            or row["header_read_ok"] != "1"
        ):
            raise ValueError("source_train record crosswalk is not uniquely verified")
        recording_id = _safe_source_train_recording_id(row["local_edf_path"], local_id)
        if recording_id in seen_recordings:
            raise ValueError("source_train record crosswalk repeats a recording")
        seen_recordings.add(recording_id)
        inventory = roster_by_recording.get(recording_id)
        if inventory is None:
            raise ValueError("source_train recording is absent from complete TUSZ roster")
        if inventory.get("patient_id") != local_id or inventory.get(
            "benchmark_split"
        ) != "source_train" or inventory.get("official_split") != "train":
            raise ValueError("complete TUSZ roster disagrees with source_train identity")
        container_sha256 = _sha256(
            inventory.get("container_sha256"), "EDF container SHA-256"
        )
        header_sha256 = _sha256(inventory.get("header_sha256"), "EDF header SHA-256")
        # DeepSOZ uses a numeric patient prefix whereas local TUSZ paths use a
        # pseudonymous alphabetic prefix.  Session/task identity must agree;
        # comparing full basenames would incorrectly reject the intended join.
        if _record_session_tail(
            row["deepsoz_record"], "DeepSOZ recording"
        ) != _record_session_tail(recording_id, "TUSZ recording"):
            raise ValueError("DeepSOZ recording suffix disagrees with TUSZ identity")
        records.append(
            {
                "deepsoz_patient_id": deepsoz_id,
                "local_patient_id": local_id,
                "patient_uid": patient["patient_uid"],
                "model_split": "source_train",
                "deepsoz_record": str(row["deepsoz_record"]),
                "tusz_recording_id": recording_id,
                "detector_recording_id": (
                    "TUSZREC-" + hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:24]
                ),
                "exact_container_equivalence_id": (
                    "TUSZ-EDF-CONTAINER-" + container_sha256
                ),
                "source_container_sha256": container_sha256,
                "source_header_sha256": header_sha256,
                "container_bytes": int(inventory["container_bytes"]),
                "recording_duration_fraction": list(
                    inventory["recording_duration_fraction"]
                ),
            }
        )

    records.sort(key=lambda item: (int(item["deepsoz_patient_id"]), item["tusz_recording_id"]))
    records_by_patient: dict[str, list[dict[str, Any]]] = {
        deepsoz_id: [] for deepsoz_id in patients_by_deepsoz
    }
    for record in records:
        records_by_patient[record["deepsoz_patient_id"]].append(record)

    patients: list[dict[str, Any]] = []
    for deepsoz_id in sorted(patients_by_deepsoz, key=int):
        patient = patients_by_deepsoz[deepsoz_id]
        patient_records = records_by_patient[deepsoz_id]
        if len(patient_records) != patient["expected_source_record_count"]:
            raise ValueError("identity binding lost a source_train patient recording")
        equivalence_ids = [
            item["exact_container_equivalence_id"] for item in patient_records
        ]
        patients.append(
            {
                **patient,
                "source_record_count": len(patient_records),
                "unique_exact_container_count": len(set(equivalence_ids)),
                "tusz_recording_ids": [
                    item["tusz_recording_id"] for item in patient_records
                ],
                "exact_container_equivalence_ids": sorted(set(equivalence_ids)),
            }
        )

    patient_count = len(patients)
    record_count = len(records)
    unique_container_count = len(
        {item["exact_container_equivalence_id"] for item in records}
    )
    if patient_count != expected_patient_count or record_count != expected_record_count:
        raise ValueError(
            "source_train identity denominator drifted: "
            f"patients={patient_count}, records={record_count}"
        )
    if unique_container_count != record_count:
        raise ValueError(
            "v1 source_train binding requires one exact container per source record"
        )

    body: dict[str, Any] = {
        "schema_version": DEEPSOZ_TUSZ_SOURCE_TRAIN_IDENTITY_BINDING_SCHEMA_V1,
        "method_id": DEEPSOZ_TUSZ_SOURCE_TRAIN_IDENTITY_BINDING_METHOD_V1,
        "binding_id": "DSZ-TUSZ-ST-ID-" + _canonical_sha256(records)[:24],
        "model_split": "source_train",
        "identity_semantics": (
            "numeric_deepsoz_patient_to_local_tusz_patient_and_exact_edf_container"
        ),
        "inputs": {
            "split_manifest": {
                "path": str(split_path),
                "sha256": _file_sha256(split_path),
            },
            "record_crosswalk": {
                "path": str(crosswalk_path),
                "sha256": _file_sha256(crosswalk_path),
            },
            "complete_tusz_roster": {
                "path": str(roster_path),
                "sha256": _file_sha256(roster_path),
            },
        },
        "counts": {
            "patients": patient_count,
            "source_records": record_count,
            "unique_exact_containers": unique_container_count,
        },
        "scope_receipt": {
            "source_train_identity_rows_used": True,
            "source_dev_identity_rows_used": False,
            "source_eval_identity_rows_used": False,
            "deepsoz_channel_targets_opened": False,
            "tusz_event_annotations_opened": False,
            "tusz_channel_annotations_opened": False,
            "private_labels_opened": False,
            "clinical_text_opened": False,
            "identity_binding_is_model_input": False,
            "identity_binding_is_training_join_authority": True,
        },
        "patients": patients,
        "records": records,
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def validate_deepsoz_tusz_source_train_identity_binding_v1(
    payload: Mapping[str, Any],
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("identity binding must be a mapping")
    if payload.get("schema_version") != (
        DEEPSOZ_TUSZ_SOURCE_TRAIN_IDENTITY_BINDING_SCHEMA_V1
    ) or payload.get("method_id") != DEEPSOZ_TUSZ_SOURCE_TRAIN_IDENTITY_BINDING_METHOD_V1:
        raise ValueError("identity binding schema or method drifted")
    if payload.get("model_split") != "source_train":
        raise ValueError("identity binding must remain source_train-only")
    receipt = _sha256(payload.get("receipt_sha256"), "identity receipt")
    body = dict(payload)
    body.pop("receipt_sha256", None)
    if _canonical_sha256(body) != receipt:
        raise ValueError("identity binding receipt does not replay")
    scope = payload.get("scope_receipt")
    expected_scope = {
        "source_train_identity_rows_used": True,
        "source_dev_identity_rows_used": False,
        "source_eval_identity_rows_used": False,
        "deepsoz_channel_targets_opened": False,
        "tusz_event_annotations_opened": False,
        "tusz_channel_annotations_opened": False,
        "private_labels_opened": False,
        "clinical_text_opened": False,
        "identity_binding_is_model_input": False,
        "identity_binding_is_training_join_authority": True,
    }
    if scope != expected_scope:
        raise ValueError("identity binding scope receipt drifted")
    patients = payload.get("patients")
    records = payload.get("records")
    if not isinstance(patients, list) or not isinstance(records, list):
        raise TypeError("identity binding patient/record rosters must be arrays")
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or counts.get("patients") != len(
        patients
    ) or counts.get("source_records") != len(records):
        raise ValueError("identity binding counts disagree with rosters")
    deepsoz_ids: set[str] = set()
    patient_uids: set[str] = set()
    local_ids: set[str] = set()
    patient_by_id: dict[str, Mapping[str, Any]] = {}
    for item in patients:
        if not isinstance(item, Mapping):
            raise TypeError("identity binding patient row must be a mapping")
        deepsoz_id = _normal_numeric_patient_id(item.get("deepsoz_patient_id"))
        local_id = _local_patient_id(item.get("local_patient_id"))
        patient_uid = _source_train_patient_uid(local_id)
        if item.get("patient_uid") != patient_uid or item.get("model_split") != (
            "source_train"
        ):
            raise ValueError("identity binding patient row is not canonical")
        if deepsoz_id in deepsoz_ids or local_id in local_ids or patient_uid in patient_uids:
            raise ValueError("identity binding repeats a patient identity")
        deepsoz_ids.add(deepsoz_id)
        local_ids.add(local_id)
        patient_uids.add(patient_uid)
        patient_by_id[deepsoz_id] = item
    recording_ids: set[str] = set()
    container_ids: set[str] = set()
    record_counts = {item: 0 for item in deepsoz_ids}
    for item in records:
        if not isinstance(item, Mapping):
            raise TypeError("identity binding record row must be a mapping")
        deepsoz_id = _normal_numeric_patient_id(item.get("deepsoz_patient_id"))
        if deepsoz_id not in patient_by_id:
            raise ValueError("identity binding record has no patient row")
        patient = patient_by_id[deepsoz_id]
        if item.get("patient_uid") != patient["patient_uid"] or item.get(
            "local_patient_id"
        ) != patient["local_patient_id"]:
            raise ValueError("identity binding record crosses patient identities")
        recording_id = _safe_source_train_recording_id(
            item.get("tusz_recording_id"), str(patient["local_patient_id"])
        )
        container_sha = _sha256(
            item.get("source_container_sha256"), "source container SHA-256"
        )
        expected_equivalence = "TUSZ-EDF-CONTAINER-" + container_sha
        expected_detector_id = (
            "TUSZREC-" + hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:24]
        )
        if item.get("exact_container_equivalence_id") != expected_equivalence or item.get(
            "detector_recording_id"
        ) != expected_detector_id:
            raise ValueError("identity binding record derivation drifted")
        if recording_id in recording_ids or expected_equivalence in container_ids:
            raise ValueError("identity binding repeats a record or exact container")
        recording_ids.add(recording_id)
        container_ids.add(expected_equivalence)
        record_counts[deepsoz_id] += 1
    for deepsoz_id, patient in patient_by_id.items():
        if patient.get("source_record_count") != record_counts[deepsoz_id]:
            raise ValueError("identity binding patient record count drifted")
    if counts.get("unique_exact_containers") != len(container_ids):
        raise ValueError("identity binding unique-container count drifted")


def load_deepsoz_tusz_source_train_identity_binding_v1(
    path: str | Path,
) -> dict[str, Any]:
    resolved = _strict_file(path, "DeepSOZ/TUSZ identity binding")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_deepsoz_tusz_source_train_identity_binding_v1(payload)
    return payload


def materialize_deepsoz_tusz_source_train_identity_binding_v1(
    payload: Mapping[str, Any], output_path: str | Path
) -> Path:
    validate_deepsoz_tusz_source_train_identity_binding_v1(payload)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise FileExistsError("refusing to overwrite a different identity binding")
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


def deepsoz_patient_uid_lookup_v1(payload: Mapping[str, Any]) -> dict[str, str]:
    validate_deepsoz_tusz_source_train_identity_binding_v1(payload)
    return {
        str(item["deepsoz_patient_id"]): str(item["patient_uid"])
        for item in payload["patients"]
    }


__all__ = [
    "DEEPSOZ_TUSZ_SOURCE_TRAIN_EXPECTED_PATIENTS_V1",
    "DEEPSOZ_TUSZ_SOURCE_TRAIN_EXPECTED_RECORDS_V1",
    "DEEPSOZ_TUSZ_SOURCE_TRAIN_IDENTITY_BINDING_METHOD_V1",
    "DEEPSOZ_TUSZ_SOURCE_TRAIN_IDENTITY_BINDING_SCHEMA_V1",
    "build_deepsoz_tusz_source_train_identity_binding_v1",
    "deepsoz_patient_uid_lookup_v1",
    "load_deepsoz_tusz_source_train_identity_binding_v1",
    "materialize_deepsoz_tusz_source_train_identity_binding_v1",
    "validate_deepsoz_tusz_source_train_identity_binding_v1",
]
