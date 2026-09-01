"""Verified carriers for the post-v10 public developmental union.

This module encodes an important claim boundary: the historical train/dev/eval
patients are all developmental after v10.  Loading this carrier can never
turn any of them back into an untouched public test set.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


PUBLIC_DEVELOPMENT_UNION_SCHEMA = "soz_public_development_union_v11"
PUBLIC_DEVELOPMENT_UNION_MANIFEST = "manifest.json"
EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256 = (
    "89a9ca456c724c2dee4d14a2c0da5a1190e58f97ad602060f6dda5f619b97232"
)
EXPECTED_PUBLIC_DEVELOPMENT_UNION_PATIENTS = 102
EXPECTED_PUBLIC_DEVELOPMENT_UNION_EVENTS = 988
EXPECTED_PUBLIC_DEVELOPMENT_UNION_FOLDS = 5


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class PublicDevelopmentUnionEvent:
    ordinal: int
    event_id: str
    patient_id: str
    outer_fold: int
    legacy_model_split: str
    official_split: str
    relative_edf_path: str
    global_event_index: int
    global_t0_sec: float
    global_stop_sec: float
    event_record_sha256: str
    edf_sha256: str
    edf_receipt_sha256: str
    signal_receipt_sha256: str
    processed_window_sha256: str


@dataclass(frozen=True)
class PublicDevelopmentUnion:
    path: Path
    manifest_sha256: str
    payload: Mapping[str, object]
    patient_ids: tuple[str, ...]
    patient_folds: tuple[int, ...]
    events: tuple[PublicDevelopmentUnionEvent, ...]

    def __post_init__(self) -> None:
        if len(self.patient_ids) != EXPECTED_PUBLIC_DEVELOPMENT_UNION_PATIENTS:
            raise ValueError("public developmental union must contain 102 patients")
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("public developmental union repeats a patient")
        if len(self.patient_folds) != len(self.patient_ids):
            raise ValueError("patient folds do not align with patient IDs")
        if set(self.patient_folds) != set(range(EXPECTED_PUBLIC_DEVELOPMENT_UNION_FOLDS)):
            raise ValueError("public developmental union must cover five folds")
        if len(self.events) != EXPECTED_PUBLIC_DEVELOPMENT_UNION_EVENTS:
            raise ValueError("public developmental union must contain 988 events")
        if tuple(event.ordinal for event in self.events) != tuple(range(len(self.events))):
            raise ValueError("public developmental event ordinals are not contiguous")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("public developmental union repeats an event")
        patient_to_fold = dict(zip(self.patient_ids, self.patient_folds))
        for event in self.events:
            if patient_to_fold.get(event.patient_id) != event.outer_fold:
                raise ValueError("event fold disagrees with its complete patient bag")

    @property
    def patient_index(self) -> dict[str, int]:
        return {patient: index for index, patient in enumerate(self.patient_ids)}

    @property
    def event_patient_index(self) -> tuple[int, ...]:
        index = self.patient_index
        return tuple(index[event.patient_id] for event in self.events)


def _strict_json(raw: bytes) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate manifest field: {key}")
            result[key] = value
        return result

    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    if not isinstance(payload, dict):
        raise TypeError("public developmental manifest must be an object")
    if raw != _canonical_bytes(payload, newline=True):
        raise ValueError("public developmental manifest is not canonical JSON")
    return payload


def load_public_development_union(
    directory: str | Path,
    *,
    expected_manifest_sha256: str = EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
) -> PublicDevelopmentUnion:
    path = Path(directory).resolve(strict=True)
    manifest_path = path / PUBLIC_DEVELOPMENT_UNION_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    raw = manifest_path.read_bytes()
    actual_sha = _sha256_bytes(raw)
    if actual_sha != str(expected_manifest_sha256):
        raise ValueError(
            "public developmental manifest SHA mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_sha}"
        )
    payload = _strict_json(raw)
    payload_digest = payload.get("manifest_payload_sha256")
    without_digest = dict(payload)
    without_digest.pop("manifest_payload_sha256", None)
    if payload_digest != _sha256_bytes(_canonical_bytes(without_digest)):
        raise ValueError("public developmental manifest payload digest changed")
    boundary = payload.get("claim_boundary")
    required_boundary = {
        "public_confirmation_forbidden": True,
        "public_external_validation_forbidden": True,
        "legacy_source_eval_retry_claim_forbidden": True,
        "all_102_patients_are_developmental": True,
        "nested_oof_is_internal_developmental_estimate_only": True,
        "private_data_must_remain_unread_for_training_selection_calibration": True,
        "private_is_reserved_for_frozen_zero_adaptation_transfer": True,
    }
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) is not expected for key, expected in required_boundary.items()
    ):
        raise ValueError("public developmental claim boundary was weakened")
    if (
        payload.get("schema_version") != PUBLIC_DEVELOPMENT_UNION_SCHEMA
        or payload.get("patient_count") != EXPECTED_PUBLIC_DEVELOPMENT_UNION_PATIENTS
        or payload.get("event_count") != EXPECTED_PUBLIC_DEVELOPMENT_UNION_EVENTS
        or payload.get("outer_fold_count") != EXPECTED_PUBLIC_DEVELOPMENT_UNION_FOLDS
    ):
        raise ValueError("public developmental manifest schema/counts changed")

    raw_patient_ids = payload.get("patient_ids")
    raw_patients = payload.get("patients")
    raw_events = payload.get("events")
    if not isinstance(raw_patient_ids, list) or not isinstance(raw_patients, list) or (
        not isinstance(raw_events, list)
    ):
        raise TypeError("public developmental manifest rosters are missing")
    patient_ids = tuple(str(value) for value in raw_patient_ids)
    patient_rows: dict[str, Mapping[str, object]] = {}
    for value in raw_patients:
        if not isinstance(value, Mapping):
            raise TypeError("public developmental patient row must be an object")
        patient_id = str(value["patient_id"])
        if patient_id in patient_rows:
            raise ValueError("public developmental manifest repeats a patient row")
        patient_rows[patient_id] = value
    if tuple(sorted(patient_rows)) != patient_ids:
        raise ValueError("patient IDs and patient rows disagree")
    patient_folds = tuple(int(patient_rows[patient]["outer_fold"]) for patient in patient_ids)

    events = []
    for value in raw_events:
        if not isinstance(value, Mapping):
            raise TypeError("public developmental event row must be an object")
        shape = value.get("processed_window_shape")
        if shape != [19, 12000] or value.get("processed_window_dtype") != "torch.float32":
            raise ValueError("public developmental event signal shape/dtype changed")
        events.append(
            PublicDevelopmentUnionEvent(
                ordinal=int(value["ordinal"]),
                event_id=str(value["event_id"]),
                patient_id=str(value["patient_id"]),
                outer_fold=int(value["outer_fold"]),
                legacy_model_split=str(value["legacy_model_split"]),
                official_split=str(value["official_split"]),
                relative_edf_path=str(value["relative_edf_path"]),
                global_event_index=int(value["global_event_index"]),
                global_t0_sec=float(value["global_t0_sec"]),
                global_stop_sec=float(value["global_stop_sec"]),
                event_record_sha256=str(value["event_record_sha256"]),
                edf_sha256=str(value["edf_sha256"]),
                edf_receipt_sha256=str(value["edf_receipt_sha256"]),
                signal_receipt_sha256=str(value["signal_receipt_sha256"]),
                processed_window_sha256=str(value["processed_window_sha256"]),
            )
        )
    return PublicDevelopmentUnion(
        path=path,
        manifest_sha256=actual_sha,
        payload=payload,
        patient_ids=patient_ids,
        patient_folds=patient_folds,
        events=tuple(events),
    )


__all__ = [
    "EXPECTED_PUBLIC_DEVELOPMENT_UNION_EVENTS",
    "EXPECTED_PUBLIC_DEVELOPMENT_UNION_FOLDS",
    "EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256",
    "EXPECTED_PUBLIC_DEVELOPMENT_UNION_PATIENTS",
    "PUBLIC_DEVELOPMENT_UNION_MANIFEST",
    "PUBLIC_DEVELOPMENT_UNION_SCHEMA",
    "PublicDevelopmentUnion",
    "PublicDevelopmentUnionEvent",
    "load_public_development_union",
]
