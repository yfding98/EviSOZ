"""Locked, target-free source-eval event roster for frozen SOZ inference.

This module is deliberately narrower than the full DeepSOZ signal-preflight
artifact.  It projects only the already verified ``source_eval`` signal rows
needed for feature production and drops every target-v2 provenance field.  It
does not import a target loader, accept a target path, or expose a target
value.  The resulting roster is therefore safe to materialize before the
single locked SOZ-label evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from .ictal_native_eval import load_bound_deepsoz_signal_preflight_artifact


LOCKED_SOURCE_EVAL_ROSTER_SCHEMA = "soz_locked_target_free_source_eval_roster_v1"
LOCKED_SOURCE_EVAL_ROSTER_ARTIFACT_SCHEMA = (
    "soz_locked_target_free_source_eval_roster_artifact_v1"
)
LOCKED_SOURCE_EVAL_ROSTER_PURPOSE = (
    "frozen_v9_source_eval_signal_feature_production_only"
)
LOCKED_SOURCE_EVAL_ROSTER_SERIALIZATION = "canonical_json_utf8_no_pickle"
LOCKED_SOURCE_EVAL_ROSTER_FILENAME = "manifest.json"
LOCKED_SOURCE_EVAL_MODEL_SPLIT = "source_eval"
LOCKED_SOURCE_EVAL_OFFICIAL_SPLIT = "eval"
EXPECTED_SOURCE_EVAL_EVENT_COUNT = 185
EXPECTED_SOURCE_EVAL_PATIENT_COUNT = 21

_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_SHA256_HEX = frozenset("0123456789abcdef")
_ARTIFACT_FIELDS = frozenset(
    {"schema_version", "serialization", "receipt_sha256", "receipt"}
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "model_split",
        "official_split",
        "locked_evaluation",
        "training_authorized",
        "model_selection_authorized",
        "threshold_tuning_authorized",
        "contains_soz_labels",
        "contains_tusz_channel_targets_or_masks",
        "target_values_loaded",
        "target_paths_accepted",
        "signal_preflight_schema",
        "signal_preflight_artifact_sha256",
        "signal_preflight_receipt_sha256",
        "split_manifest_sha256",
        "preprocess_config",
        "preprocess_config_sha256",
        "event_count",
        "patient_count",
        "event_order_sha256",
        "patient_roster_sha256",
        "events",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "ordinal",
        "event_id",
        "patient_id",
        "model_split",
        "official_split",
        "signal_event_record_sha256",
        "deepsoz_source_record_sha256",
        "relative_edf_path",
        "global_event_index",
        "global_t0_sec",
        "global_stop_sec",
        "seizure_duration_sec",
        "previous_seizure_gap_sec",
        "global_seizure_type",
        "record_timeline_sha256",
        "edf_sha256",
        "preprocess_config_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
        "processed_window_shape",
        "processed_window_dtype",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Locked source-eval roster is not canonical JSON data") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _require_closed_fields(
    value: Mapping[str, object], expected: frozenset[str], *, field: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise ValueError(
            f"{field} violates its closed schema; "
            f"missing={sorted(set(expected)-actual)}, "
            f"unknown={sorted(actual-set(expected))}"
        )


def _ordered_roster_sha256(values: Sequence[object], *, field: str) -> str:
    normalized = tuple(str(value) for value in values)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must be non-empty and unique")
    return _canonical_sha256(normalized)


def _strict_json(raw: bytes, *, field: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"{field} contains forbidden constant {value}")

    try:
        result = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{field} must be a JSON object")
    if _canonical_json_bytes(result) != raw:
        raise ValueError(f"{field} is not canonical JSON")
    return result


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return result


def _stable_file(path: Path, *, field: str) -> tuple[bytes, str]:
    source = _absolute_no_symlink(path, field=field)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field} must be a regular file")
    before = source.stat()
    if not 1 <= before.st_size <= _MAX_ARTIFACT_BYTES:
        raise ValueError(f"{field} has an invalid size")
    raw = source.read_bytes()
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{field} changed while it was read")
    return raw, _bytes_sha256(raw)


def _safe_new_directory(path: str | Path) -> Path:
    target = _absolute_no_symlink(path, field="locked source-eval roster output")
    if target.name in {"", ".", ".."}:
        raise ValueError("Locked source-eval roster output needs a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class LockedSourceEvalEvent:
    ordinal: int
    event_id: str
    patient_id: str
    relative_edf_path: str
    global_event_index: int
    global_t0_sec: float
    global_stop_sec: float
    seizure_duration_sec: float
    previous_seizure_gap_sec: float | None
    global_seizure_type: str
    record_timeline_sha256: str
    signal_event_record_sha256: str
    deepsoz_source_record_sha256: str
    edf_sha256: str
    preprocess_config_sha256: str
    edf_receipt_sha256: str
    signal_receipt_sha256: str
    processed_window_sha256: str
    processed_window_shape: tuple[int, int]
    processed_window_dtype: str


def _validate_event(value: object, *, ordinal: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"events[{ordinal}] must be an object")
    row = dict(value)
    _require_closed_fields(row, _EVENT_FIELDS, field=f"events[{ordinal}]")
    if row["ordinal"] != ordinal:
        raise ValueError("Locked source-eval event ordinal changed")
    if row["model_split"] != LOCKED_SOURCE_EVAL_MODEL_SPLIT or row[
        "official_split"
    ] != LOCKED_SOURCE_EVAL_OFFICIAL_SPLIT:
        raise ValueError("Locked source-eval event escaped the eval split")
    for field in (
        "event_id",
        "patient_id",
        "relative_edf_path",
        "record_timeline_sha256",
        "signal_event_record_sha256",
        "deepsoz_source_record_sha256",
        "edf_sha256",
        "preprocess_config_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
    ):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"events[{ordinal}].{field} must be non-empty")
    for field in (
        "record_timeline_sha256",
        "signal_event_record_sha256",
        "deepsoz_source_record_sha256",
        "edf_sha256",
        "preprocess_config_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
    ):
        _require_sha256(row[field], field=f"events[{ordinal}].{field}")
    index = row["global_event_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Locked source-eval global event index is invalid")
    start = row["global_t0_sec"]
    stop = row["global_stop_sec"]
    duration = row["seizure_duration_sec"]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (start, stop, duration)
    ):
        raise TypeError("Locked source-eval timing must be numeric")
    start, stop, duration = float(start), float(stop), float(duration)
    if not all(math.isfinite(value) for value in (start, stop, duration)) or (
        start < 0 or stop <= start or abs(duration - (stop - start)) > 1e-6
    ):
        raise ValueError("Locked source-eval timing is invalid")
    gap = row["previous_seizure_gap_sec"]
    if gap is not None and (
        isinstance(gap, bool)
        or not isinstance(gap, (int, float))
        or not math.isfinite(float(gap))
        or float(gap) < 0
    ):
        raise ValueError("Locked source-eval previous-seizure gap is invalid")
    if row["processed_window_shape"] != [19, 12000] or row[
        "processed_window_dtype"
    ] != "torch.float32":
        raise ValueError("Locked source-eval processed signal contract changed")
    if not isinstance(row["global_seizure_type"], str):
        raise TypeError("Locked source-eval seizure type must be a string")
    return row


def _validate_receipt(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Locked source-eval receipt must be an object")
    receipt = dict(value)
    _require_closed_fields(receipt, _RECEIPT_FIELDS, field="receipt")
    fixed = {
        "schema_version": LOCKED_SOURCE_EVAL_ROSTER_SCHEMA,
        "purpose": LOCKED_SOURCE_EVAL_ROSTER_PURPOSE,
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "official_split": LOCKED_SOURCE_EVAL_OFFICIAL_SPLIT,
        "locked_evaluation": True,
        "training_authorized": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "target_values_loaded": False,
        "target_paths_accepted": False,
        "event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
        "patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
    }
    changed = tuple(field for field, expected in fixed.items() if receipt[field] != expected)
    if changed:
        raise ValueError(f"Locked source-eval boundary changed: {changed}")
    for field in (
        "signal_preflight_artifact_sha256",
        "signal_preflight_receipt_sha256",
        "split_manifest_sha256",
        "preprocess_config_sha256",
        "event_order_sha256",
        "patient_roster_sha256",
    ):
        _require_sha256(receipt[field], field=field)
    if not isinstance(receipt["signal_preflight_schema"], str) or not receipt[
        "signal_preflight_schema"
    ]:
        raise ValueError("Signal-preflight schema must be explicit")
    if not isinstance(receipt["preprocess_config"], dict):
        raise TypeError("Preprocess config must be an object")
    events_value = receipt["events"]
    if not isinstance(events_value, list) or len(events_value) != receipt["event_count"]:
        raise ValueError("Locked source-eval events disagree with event_count")
    events = [_validate_event(row, ordinal=index) for index, row in enumerate(events_value)]
    event_ids = tuple(str(row["event_id"]) for row in events)
    patients = tuple(sorted({str(row["patient_id"]) for row in events}))
    if len(patients) != receipt["patient_count"]:
        raise ValueError("Locked source-eval patients disagree with patient_count")
    if receipt["event_order_sha256"] != _ordered_roster_sha256(
        event_ids, field="event roster"
    ):
        raise ValueError("Locked source-eval event roster SHA changed")
    if receipt["patient_roster_sha256"] != _ordered_roster_sha256(
        patients, field="patient roster"
    ):
        raise ValueError("Locked source-eval patient roster SHA changed")
    receipt["events"] = events
    return receipt


def _runtime_events(receipt: Mapping[str, object]) -> tuple[LockedSourceEvalEvent, ...]:
    result = []
    for row_value in receipt["events"]:
        row = dict(row_value)
        result.append(
            LockedSourceEvalEvent(
                ordinal=int(row["ordinal"]),
                event_id=str(row["event_id"]),
                patient_id=str(row["patient_id"]),
                relative_edf_path=str(row["relative_edf_path"]),
                global_event_index=int(row["global_event_index"]),
                global_t0_sec=float(row["global_t0_sec"]),
                global_stop_sec=float(row["global_stop_sec"]),
                seizure_duration_sec=float(row["seizure_duration_sec"]),
                previous_seizure_gap_sec=(
                    None
                    if row["previous_seizure_gap_sec"] is None
                    else float(row["previous_seizure_gap_sec"])
                ),
                global_seizure_type=str(row["global_seizure_type"]),
                record_timeline_sha256=str(row["record_timeline_sha256"]),
                signal_event_record_sha256=str(row["signal_event_record_sha256"]),
                deepsoz_source_record_sha256=str(row["deepsoz_source_record_sha256"]),
                edf_sha256=str(row["edf_sha256"]),
                preprocess_config_sha256=str(row["preprocess_config_sha256"]),
                edf_receipt_sha256=str(row["edf_receipt_sha256"]),
                signal_receipt_sha256=str(row["signal_receipt_sha256"]),
                processed_window_sha256=str(row["processed_window_sha256"]),
                processed_window_shape=tuple(row["processed_window_shape"]),
                processed_window_dtype=str(row["processed_window_dtype"]),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class VerifiedLockedSourceEvalRoster:
    path: Path | None
    artifact_sha256: str
    receipt_sha256: str
    receipt: Mapping[str, object]
    events: tuple[LockedSourceEvalEvent, ...]

    def __post_init__(self) -> None:
        validated = _validate_receipt(dict(self.receipt))
        _require_sha256(self.artifact_sha256, field="artifact_sha256")
        if self.receipt_sha256 != _canonical_sha256(validated):
            raise ValueError("Locked source-eval receipt SHA mismatch")
        if len(self.events) != EXPECTED_SOURCE_EVAL_EVENT_COUNT:
            raise ValueError("Locked source-eval runtime event count changed")
        object.__setattr__(self, "receipt", validated)

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.events)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({event.patient_id for event in self.events}))


def derive_locked_source_eval_roster_receipt(
    signal_receipt: Mapping[str, object],
    *,
    signal_artifact_sha256: str,
    signal_receipt_sha256: str,
) -> dict[str, object]:
    """Project a verified signal receipt into the fixed label-free eval roster."""

    artifact_sha = _require_sha256(
        signal_artifact_sha256, field="signal_artifact_sha256"
    )
    receipt_sha = _require_sha256(
        signal_receipt_sha256, field="signal_receipt_sha256"
    )
    accepted = signal_receipt.get("events")
    excluded = signal_receipt.get("exclusions")
    if not isinstance(accepted, list) or not isinstance(excluded, list):
        raise TypeError("Signal receipt must expose accepted and excluded event lists")
    all_rows = (*accepted, *excluded)
    groups: dict[str, list[Mapping[str, object]]] = {}
    for value in all_rows:
        if not isinstance(value, dict):
            raise TypeError("Signal event rows must be objects")
        source_record = _require_sha256(
            value.get("deepsoz_source_record_sha256"),
            field="deepsoz_source_record_sha256",
        )
        groups.setdefault(source_record, []).append(value)

    previous_stop: dict[str, float | None] = {}
    timeline_sha: dict[str, str] = {}
    for source_record, group in groups.items():
        ordered = tuple(sorted(group, key=lambda row: int(row["global_event_index"])))
        indices = tuple(int(row["global_event_index"]) for row in ordered)
        if indices != tuple(range(len(ordered))):
            raise ValueError("Signal receipt does not contain a complete record timeline")
        starts = tuple(float(row["global_t0_sec"]) for row in ordered)
        if any(right < left - 1e-6 for left, right in zip(starts, starts[1:])):
            raise ValueError("Signal receipt record timeline is not chronological")
        payload = {
            "schema_version": "locked_target_free_record_timeline_v1",
            "signal_preflight_artifact_sha256": artifact_sha,
            "signal_preflight_receipt_sha256": receipt_sha,
            "deepsoz_source_record_sha256": source_record,
            "events": [
                {
                    "event_id": str(row["event_id"]),
                    "event_record_sha256": str(row["event_record_sha256"]),
                    "global_event_index": int(row["global_event_index"]),
                    "global_t0_sec": float(row["global_t0_sec"]),
                    "global_stop_sec": float(row["global_stop_sec"]),
                }
                for row in ordered
            ],
        }
        group_sha = _canonical_sha256(payload)
        running_max: float | None = None
        for row in ordered:
            event_id = str(row["event_id"])
            if event_id in previous_stop:
                raise ValueError("Signal receipt repeats an event ID")
            previous_stop[event_id] = running_max
            timeline_sha[event_id] = group_sha
            stop = float(row["global_stop_sec"])
            running_max = stop if running_max is None else max(running_max, stop)

    selected = sorted(
        (
            row
            for row in accepted
            if row.get("model_split") == LOCKED_SOURCE_EVAL_MODEL_SPLIT
        ),
        key=lambda row: str(row["event_id"]),
    )
    if len(selected) != EXPECTED_SOURCE_EVAL_EVENT_COUNT:
        raise ValueError(
            "Locked source-eval event count changed: "
            f"expected {EXPECTED_SOURCE_EVAL_EVENT_COUNT}, got {len(selected)}"
        )
    rows: list[dict[str, object]] = []
    for ordinal, row in enumerate(selected):
        if row.get("official_split") != LOCKED_SOURCE_EVAL_OFFICIAL_SPLIT:
            raise ValueError("Source-eval signal row is not in official eval")
        event_id = str(row["event_id"])
        start = float(row["global_t0_sec"])
        stop = float(row["global_stop_sec"])
        prior = previous_stop[event_id]
        event = {
            "ordinal": ordinal,
            "event_id": event_id,
            "patient_id": str(row["patient_id"]),
            "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
            "official_split": LOCKED_SOURCE_EVAL_OFFICIAL_SPLIT,
            "signal_event_record_sha256": str(row["event_record_sha256"]),
            "deepsoz_source_record_sha256": str(
                row["deepsoz_source_record_sha256"]
            ),
            "relative_edf_path": str(row["relative_edf_path"]),
            "global_event_index": int(row["global_event_index"]),
            "global_t0_sec": start,
            "global_stop_sec": stop,
            "seizure_duration_sec": stop - start,
            "previous_seizure_gap_sec": (
                None if prior is None else max(0.0, start - prior)
            ),
            "global_seizure_type": str(row.get("global_seizure_type", "")),
            "record_timeline_sha256": timeline_sha[event_id],
            "edf_sha256": str(row["edf_sha256"]),
            "preprocess_config_sha256": str(row["preprocess_config_sha256"]),
            "edf_receipt_sha256": str(row["edf_receipt_sha256"]),
            "signal_receipt_sha256": str(row["signal_receipt_sha256"]),
            "processed_window_sha256": str(row["processed_window_sha256"]),
            "processed_window_shape": list(row["processed_window_shape"]),
            "processed_window_dtype": str(row["processed_window_dtype"]),
        }
        rows.append(_validate_event(event, ordinal=ordinal))

    patients = tuple(sorted({str(row["patient_id"]) for row in rows}))
    if len(patients) != EXPECTED_SOURCE_EVAL_PATIENT_COUNT:
        raise ValueError(
            "Locked source-eval patient count changed: "
            f"expected {EXPECTED_SOURCE_EVAL_PATIENT_COUNT}, got {len(patients)}"
        )
    split_rows = signal_receipt.get("eligible_split_patient_ids")
    if not isinstance(split_rows, list):
        raise TypeError("Signal receipt lacks eligible split patient rosters")
    split_rosters = {str(row[0]): tuple(str(value) for value in row[1]) for row in split_rows}
    if split_rosters.get(LOCKED_SOURCE_EVAL_MODEL_SPLIT) != patients:
        raise ValueError("Signal-preflight and locked source-eval patient rosters differ")
    preprocess_config = signal_receipt.get("preprocess_config")
    if not isinstance(preprocess_config, dict):
        raise TypeError("Signal receipt lacks a closed preprocess config")
    receipt = {
        "schema_version": LOCKED_SOURCE_EVAL_ROSTER_SCHEMA,
        "purpose": LOCKED_SOURCE_EVAL_ROSTER_PURPOSE,
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "official_split": LOCKED_SOURCE_EVAL_OFFICIAL_SPLIT,
        "locked_evaluation": True,
        "training_authorized": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "target_values_loaded": False,
        "target_paths_accepted": False,
        "signal_preflight_schema": str(signal_receipt["schema_version"]),
        "signal_preflight_artifact_sha256": artifact_sha,
        "signal_preflight_receipt_sha256": receipt_sha,
        "split_manifest_sha256": str(signal_receipt["split_manifest_sha256"]),
        "preprocess_config": dict(preprocess_config),
        "preprocess_config_sha256": str(signal_receipt["preprocess_config_sha256"]),
        "event_count": len(rows),
        "patient_count": len(patients),
        "event_order_sha256": _ordered_roster_sha256(
            tuple(str(row["event_id"]) for row in rows), field="event roster"
        ),
        "patient_roster_sha256": _ordered_roster_sha256(
            patients, field="patient roster"
        ),
        "events": rows,
    }
    return _validate_receipt(receipt)


def _in_memory_artifact(receipt: Mapping[str, object]) -> VerifiedLockedSourceEvalRoster:
    validated = _validate_receipt(dict(receipt))
    receipt_sha = _canonical_sha256(validated)
    payload = {
        "schema_version": LOCKED_SOURCE_EVAL_ROSTER_ARTIFACT_SCHEMA,
        "serialization": LOCKED_SOURCE_EVAL_ROSTER_SERIALIZATION,
        "receipt_sha256": receipt_sha,
        "receipt": validated,
    }
    artifact_sha = _bytes_sha256(_canonical_json_bytes(payload))
    return VerifiedLockedSourceEvalRoster(
        path=None,
        artifact_sha256=artifact_sha,
        receipt_sha256=receipt_sha,
        receipt=validated,
        events=_runtime_events(validated),
    )


def preflight_locked_source_eval_roster(
    *,
    signal_preflight_bundle: str | Path,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
) -> dict[str, object]:
    """Validate the complete target-free source-eval roster without writing."""

    signal = load_bound_deepsoz_signal_preflight_artifact(
        signal_preflight_bundle,
        expected_artifact_sha256=expected_signal_artifact_sha256,
        expected_receipt_sha256=expected_signal_receipt_sha256,
    )
    receipt = derive_locked_source_eval_roster_receipt(
        signal.receipt,
        signal_artifact_sha256=signal.artifact_sha256,
        signal_receipt_sha256=signal.receipt_sha256,
    )
    return {
        "schema_version": LOCKED_SOURCE_EVAL_ROSTER_SCHEMA,
        "status": "ready_target_free_source_eval_roster",
        "purpose": LOCKED_SOURCE_EVAL_ROSTER_PURPOSE,
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "event_count": receipt["event_count"],
        "patient_count": receipt["patient_count"],
        "event_order_sha256": receipt["event_order_sha256"],
        "patient_roster_sha256": receipt["patient_roster_sha256"],
        "contains_soz_labels": False,
        "contains_tusz_channel_targets_or_masks": False,
        "target_values_loaded": False,
        "target_paths_accepted": False,
        "training_authorized": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
    }


def build_locked_source_eval_roster(
    *,
    signal_preflight_bundle: str | Path,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
    output_directory: str | Path,
) -> VerifiedLockedSourceEvalRoster:
    """Atomically publish the fixed 21-patient/185-event target-free roster."""

    signal = load_bound_deepsoz_signal_preflight_artifact(
        signal_preflight_bundle,
        expected_artifact_sha256=expected_signal_artifact_sha256,
        expected_receipt_sha256=expected_signal_receipt_sha256,
    )
    receipt = derive_locked_source_eval_roster_receipt(
        signal.receipt,
        signal_artifact_sha256=signal.artifact_sha256,
        signal_receipt_sha256=signal.receipt_sha256,
    )
    memory = _in_memory_artifact(receipt)
    payload = {
        "schema_version": LOCKED_SOURCE_EVAL_ROSTER_ARTIFACT_SCHEMA,
        "serialization": LOCKED_SOURCE_EVAL_ROSTER_SERIALIZATION,
        "receipt_sha256": memory.receipt_sha256,
        "receipt": memory.receipt,
    }
    raw = _canonical_json_bytes(payload)
    if not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        raise ValueError("Locked source-eval roster artifact has an invalid size")
    output = _safe_new_directory(output_directory)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        artifact_path = temporary / LOCKED_SOURCE_EVAL_ROSTER_FILENAME
        with artifact_path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        if os.path.lexists(output):
            raise FileExistsError(output)
        os.rename(temporary, output)
        published = True
        _fsync_directory(output.parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return VerifiedLockedSourceEvalRoster(
        path=output,
        artifact_sha256=_bytes_sha256(raw),
        receipt_sha256=memory.receipt_sha256,
        receipt=memory.receipt,
        events=memory.events,
    )


def load_locked_source_eval_roster(
    bundle_directory: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
) -> VerifiedLockedSourceEvalRoster:
    """Strictly load a closed target-free source-eval roster artifact."""

    bundle = _absolute_no_symlink(bundle_directory, field="locked source-eval roster")
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("Locked source-eval roster must be a regular directory")
    entries = tuple(bundle.iterdir())
    if len(entries) != 1 or entries[0].name != LOCKED_SOURCE_EVAL_ROSTER_FILENAME:
        raise ValueError("Locked source-eval roster violates its closed file schema")
    raw, artifact_sha = _stable_file(entries[0], field="locked source-eval artifact")
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Locked source-eval roster artifact SHA mismatch")
    payload = _strict_json(raw, field="locked source-eval artifact")
    _require_closed_fields(payload, _ARTIFACT_FIELDS, field="artifact")
    if payload["schema_version"] != LOCKED_SOURCE_EVAL_ROSTER_ARTIFACT_SCHEMA or payload[
        "serialization"
    ] != LOCKED_SOURCE_EVAL_ROSTER_SERIALIZATION:
        raise ValueError("Unsupported locked source-eval roster artifact")
    receipt = _validate_receipt(payload["receipt"])
    receipt_sha = _canonical_sha256(receipt)
    if receipt_sha != payload["receipt_sha256"]:
        raise ValueError("Locked source-eval roster receipt SHA mismatch")
    if receipt["signal_preflight_artifact_sha256"] != _require_sha256(
        expected_signal_artifact_sha256,
        field="expected_signal_artifact_sha256",
    ) or receipt["signal_preflight_receipt_sha256"] != _require_sha256(
        expected_signal_receipt_sha256,
        field="expected_signal_receipt_sha256",
    ):
        raise ValueError("Locked source-eval roster is bound to another signal artifact")
    return VerifiedLockedSourceEvalRoster(
        path=bundle,
        artifact_sha256=artifact_sha,
        receipt_sha256=receipt_sha,
        receipt=receipt,
        events=_runtime_events(receipt),
    )


__all__ = [
    "EXPECTED_SOURCE_EVAL_EVENT_COUNT",
    "EXPECTED_SOURCE_EVAL_PATIENT_COUNT",
    "LOCKED_SOURCE_EVAL_MODEL_SPLIT",
    "LOCKED_SOURCE_EVAL_ROSTER_ARTIFACT_SCHEMA",
    "LOCKED_SOURCE_EVAL_ROSTER_PURPOSE",
    "LOCKED_SOURCE_EVAL_ROSTER_SCHEMA",
    "LockedSourceEvalEvent",
    "VerifiedLockedSourceEvalRoster",
    "build_locked_source_eval_roster",
    "derive_locked_source_eval_roster_receipt",
    "load_locked_source_eval_roster",
    "preflight_locked_source_eval_roster",
]
