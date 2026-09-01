"""Typed offline CerebraGloss/ELM candidate-cache contracts.

The teacher models are intentionally represented as *candidate* producers.
This module never promotes a teacher probability to a clinical finding or a
node-level SOZ target.  It accepts already materialized, content-addressed
outputs so that model-specific inference can remain outside the deployment
graph while the Stage-0 gate can replay the resulting lineage.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from typing import Any, Mapping, Sequence

from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)


TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION = "evisoz_teacher_candidate_cache_v1"
TEACHER_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION = (
    "evisoz_teacher_candidate_materialization_v1"
)
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_CACHE_ID_PREFIX = "EVISOZ-TEACHER-CACHE-"
_MATERIALIZATION_ID_PREFIX = "EVISOZ-TEACHER-MAT-"

TEACHER_IDS = ("cerebragloss", "elm")
_SUPPORT_KINDS = ("edge_interval", "node_interval", "crop", "global")
_SUPPORT_VIEWS = {
    "cerebragloss": {"tcp22_edge_context", "car19_context"},
    "elm": {"elm_tcp20_crop", "tcp22_edge_context", "car19_context"},
}
_PROHIBITED_USES = (
    "clinical_label",
    "measured_fact",
    "node_localization_supervision",
    "endpoint_expansion_from_edge",
)
_POLICY = {
    "authority": "offline_teacher",
    "status": "candidate_only",
    "calibration_state": "uncalibrated",
    "soft_auxiliary_only": True,
    "may_create_clinical_label": False,
    "may_be_treated_as_measured_fact": False,
    "may_supervise_node_localization": False,
    "teacher_runtime_required_at_deployment": False,
}


def _finite_tree(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _cache_id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["cache_id"] = _PENDING_ID
    return body


def _materialization_id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["materialization_id"] = _PENDING_ID
    return body


def _require_ref(value: object, *, kind: str, context: str) -> dict[str, Any]:
    ref = validate_artifact_ref(value)
    if ref["artifact_kind"] != kind:
        raise ValueError(f"{context} reference kind drifted")
    return ref


def _validate_event_identity_binding(
    event_identity_ref: Mapping[str, object],
    source_dual_montage_cache_ref: Mapping[str, object],
) -> None:
    event = _require_ref(
        event_identity_ref, kind="event_identity", context="event identity"
    )
    dual = _require_ref(
        source_dual_montage_cache_ref,
        kind="dual_montage_cache_materialization_receipt",
        context="dual montage",
    )
    # The dual-cache receipt is allowed to be opaque at ingestion time, but if
    # its payload is opened by a caller the event identity must be the same.
    # Keeping this check structural avoids reading private waveform payloads.
    if event["payload_schema_version"] != "evisoz_event_identity_v1":
        raise ValueError("teacher candidate event identity schema drifted")
    if dual["payload_schema_version"] != (
        "evisoz_dual_montage_cache_materialization_receipt_v1"
    ):
        raise ValueError("teacher candidate dual montage schema drifted")


def _candidate_identity(row: Mapping[str, object]) -> str:
    body = deepcopy(dict(row))
    body["candidate_id"] = _PENDING_ID
    return "EVISOZ-TEACHER-CAND-" + canonical_json_sha256(body)[:24]


def _validate_candidate_row(
    row: Mapping[str, object],
    *,
    teacher_id: str,
) -> dict[str, Any]:
    required = {
        "candidate_id",
        "concept",
        "support_kind",
        "support_view",
        "support_units",
        "support_interval_seconds",
        "confidence",
        "probability_semantics",
        "authority",
        "status",
        "calibration_state",
        "permitted_uses",
        "prohibited_uses",
    }
    if type(row) is not dict or set(row) != required:
        raise ValueError("teacher candidate row fields drifted")
    data = deepcopy(row)
    if data["candidate_id"] != _candidate_identity(data):
        raise ValueError("teacher candidate identity drifted")
    if not isinstance(data["concept"], str) or not data["concept"]:
        raise ValueError("teacher candidate concept is invalid")
    if data["support_kind"] not in _SUPPORT_KINDS:
        raise ValueError("teacher candidate support kind is invalid")
    if data["support_view"] not in _SUPPORT_VIEWS[teacher_id]:
        raise ValueError("teacher candidate support view is not allowed")
    units = data["support_units"]
    if not isinstance(units, list) or any(
        not isinstance(unit, str) or not unit for unit in units
    ) or units != sorted(set(units)):
        raise ValueError("teacher candidate support units are invalid")
    interval = data["support_interval_seconds"]
    if interval is not None:
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in interval)
            or not float(interval[0]) < float(interval[1])
        ):
            raise ValueError("teacher candidate interval is invalid")
    if data["support_kind"] in {"edge_interval", "node_interval"}:
        if not units or interval is None:
            raise ValueError("localized teacher candidate lacks support")
    elif data["support_kind"] == "crop" and units:
        raise ValueError("crop teacher candidate must not claim channel units")
    elif data["support_kind"] == "global" and (units or interval is not None):
        raise ValueError("global teacher candidate must not claim local support")
    confidence = data["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("teacher candidate confidence is invalid")
    if not isinstance(data["probability_semantics"], str) or not data[
        "probability_semantics"
    ]:
        raise ValueError("teacher candidate probability semantics are invalid")
    if data["authority"] != "offline_teacher" or data["status"] != "candidate_only":
        raise ValueError("teacher candidate authority/status drifted")
    if data["calibration_state"] != "uncalibrated":
        raise ValueError("teacher candidates must remain uncalibrated at ingestion")
    if data["permitted_uses"] != ["soft_auxiliary"]:
        raise ValueError("teacher candidate permitted uses drifted")
    if data["prohibited_uses"] != list(_PROHIBITED_USES):
        raise ValueError("teacher candidate prohibited uses drifted")
    return data


def build_teacher_candidate_cache(
    *,
    teacher_id: str,
    event_id: str,
    linkage_group_id: str,
    evisoz_role: str,
    outer_holdout_fold: int,
    event_identity_ref: Mapping[str, object],
    source_dual_montage_cache_ref: Mapping[str, object],
    teacher_model_ref: Mapping[str, object],
    input_view: str,
    input_sampling_rate_hz: int,
    input_window_seconds: list[float],
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Build one development-only, candidate-only teacher cache."""

    if teacher_id not in TEACHER_IDS:
        raise ValueError("unknown EviSOZ teacher ID")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("teacher candidate event ID is invalid")
    if not isinstance(linkage_group_id, str) or not linkage_group_id:
        raise ValueError("teacher candidate linkage group is invalid")
    if evisoz_role != "development_cv":
        raise ValueError("teacher candidate caches may not contain locked-test events")
    if isinstance(outer_holdout_fold, bool) or not isinstance(outer_holdout_fold, int) or outer_holdout_fold < 0:
        raise ValueError("teacher candidate outer fold is invalid")
    _validate_event_identity_binding(event_identity_ref, source_dual_montage_cache_ref)
    model_ref = validate_artifact_ref(teacher_model_ref)
    if model_ref["artifact_kind"] not in {"teacher_model_checkpoint", "teacher_model_manifest"}:
        raise ValueError("teacher model reference kind drifted")
    if not isinstance(input_view, str) or input_view not in _SUPPORT_VIEWS[teacher_id]:
        raise ValueError("teacher input view is not allowed")
    if isinstance(input_sampling_rate_hz, bool) or not isinstance(input_sampling_rate_hz, int) or input_sampling_rate_hz <= 0:
        raise ValueError("teacher input sampling rate is invalid")
    if (
        not isinstance(input_window_seconds, list)
        or len(input_window_seconds) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in input_window_seconds)
        or not float(input_window_seconds[0]) < float(input_window_seconds[1])
    ):
        raise ValueError("teacher input window is invalid")
    rows = [_validate_candidate_row(row, teacher_id=teacher_id) for row in candidate_rows]
    rows.sort(key=lambda row: row["candidate_id"])
    body: dict[str, Any] = {
        "schema_version": TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION,
        "cache_id": _PENDING_ID,
        "teacher_id": teacher_id,
        "status": "complete_uncalibrated_candidate_only",
        "event_id": event_id,
        "linkage_group_id": linkage_group_id,
        "evisoz_role": evisoz_role,
        "outer_holdout_fold": outer_holdout_fold,
        "event_identity_ref": deepcopy(dict(event_identity_ref)),
        "source_dual_montage_cache_ref": deepcopy(dict(source_dual_montage_cache_ref)),
        "teacher_model_ref": deepcopy(dict(teacher_model_ref)),
        "input_view": input_view,
        "input_sampling_rate_hz": input_sampling_rate_hz,
        "input_window_seconds": list(input_window_seconds),
        "candidate_rows": rows,
        "authority_policy": deepcopy(_POLICY),
        "counts": {
            "candidate_count": len(rows),
            "candidate_concept_counts": dict(
                sorted(Counter(str(row["concept"]) for row in rows).items())
            ),
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["cache_id"] = _CACHE_ID_PREFIX + canonical_json_sha256(
        _cache_id_source(body)
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_teacher_candidate_cache(body)


def validate_teacher_candidate_cache(value: object) -> dict[str, Any]:
    """Validate source bindings and candidate-only safety invariants."""

    required = {
        "schema_version", "cache_id", "teacher_id", "status", "event_id",
        "linkage_group_id", "evisoz_role", "outer_holdout_fold",
        "event_identity_ref", "source_dual_montage_cache_ref", "teacher_model_ref",
        "input_view", "input_sampling_rate_hz", "input_window_seconds",
        "candidate_rows", "authority_policy", "counts", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("teacher candidate cache fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION:
        raise ValueError("teacher candidate cache schema drifted")
    if data["teacher_id"] not in TEACHER_IDS:
        raise ValueError("teacher candidate cache teacher ID drifted")
    if data["status"] != "complete_uncalibrated_candidate_only":
        raise ValueError("teacher candidate cache status drifted")
    _validate_event_identity_binding(
        data["event_identity_ref"], data["source_dual_montage_cache_ref"]
    )
    model_ref = validate_artifact_ref(data["teacher_model_ref"])
    if model_ref["artifact_kind"] not in {"teacher_model_checkpoint", "teacher_model_manifest"}:
        raise ValueError("teacher model reference kind drifted")
    if data["evisoz_role"] != "development_cv":
        raise ValueError("teacher candidate cache contains non-development data")
    if isinstance(data["outer_holdout_fold"], bool) or not isinstance(data["outer_holdout_fold"], int) or data["outer_holdout_fold"] < 0:
        raise ValueError("teacher candidate cache fold drifted")
    if data["input_view"] not in _SUPPORT_VIEWS[data["teacher_id"]]:
        raise ValueError("teacher candidate cache input view drifted")
    if isinstance(data["input_sampling_rate_hz"], bool) or not isinstance(data["input_sampling_rate_hz"], int) or data["input_sampling_rate_hz"] <= 0:
        raise ValueError("teacher candidate cache sampling rate drifted")
    window = data["input_window_seconds"]
    if not isinstance(window, list) or len(window) != 2 or not float(window[0]) < float(window[1]):
        raise ValueError("teacher candidate cache input window drifted")
    rows = data["candidate_rows"]
    if not isinstance(rows, list) or rows != sorted(rows, key=lambda row: row["candidate_id"]):
        raise ValueError("teacher candidate rows are not sorted")
    ids: set[str] = set()
    concepts: Counter[str] = Counter()
    for row in rows:
        checked = _validate_candidate_row(row, teacher_id=str(data["teacher_id"]))
        if checked["candidate_id"] in ids:
            raise ValueError("teacher candidate row duplicated")
        ids.add(checked["candidate_id"])
        concepts[str(checked["concept"])] += 1
    if data["authority_policy"] != _POLICY:
        raise ValueError("teacher candidate authority policy drifted")
    if data["counts"] != {
        "candidate_count": len(rows),
        "candidate_concept_counts": dict(sorted(concepts.items())),
    }:
        raise ValueError("teacher candidate counts drifted")
    expected_id = _CACHE_ID_PREFIX + canonical_json_sha256(_cache_id_source(data))[:24]
    if data["cache_id"] != expected_id:
        raise ValueError("teacher candidate cache ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("teacher candidate cache receipt drifted")
    return data


def build_teacher_candidate_materialization(
    *,
    teacher_id: str,
    source_split_roster_ref: Mapping[str, object],
    teacher_model_ref: Mapping[str, object],
    caches: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Build a development-only manifest from validated teacher caches.

    File materialization is deliberately left to the caller so this function
    can also be used by a controlled importer that writes canonical JSON and
    then records each cache path.
    """

    if teacher_id not in TEACHER_IDS:
        raise ValueError("unknown EviSOZ teacher ID")
    split_ref = validate_artifact_ref(source_split_roster_ref)
    if split_ref["artifact_kind"] != "split_roster":
        raise ValueError("teacher materialization split reference kind drifted")
    model_ref = validate_artifact_ref(teacher_model_ref)
    if model_ref["artifact_kind"] not in {"teacher_model_checkpoint", "teacher_model_manifest"}:
        raise ValueError("teacher materialization model reference kind drifted")
    rows: list[dict[str, Any]] = []
    for cache in caches:
        checked = validate_teacher_candidate_cache(cache)
        if checked["teacher_id"] != teacher_id or checked["teacher_model_ref"] != model_ref:
            raise ValueError("teacher materialization cache binding drifted")
        rows.append({
            "event_id": checked["event_id"],
            "linkage_group_id": checked["linkage_group_id"],
            "outer_holdout_fold": checked["outer_holdout_fold"],
            "candidate_count": checked["counts"]["candidate_count"],
            "relative_cache_path": f"events/{checked['event_id']}/candidate_cache.json",
            "candidate_cache_ref": build_json_artifact_ref(
                checked,
                artifact_kind="teacher_candidate_cache",
                payload_schema_version=TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION,
            ),
        })
    if not rows:
        raise ValueError("teacher materialization requires at least one cache")
    rows.sort(key=lambda row: row["event_id"])
    if len({row["event_id"] for row in rows}) != len(rows):
        raise ValueError("teacher materialization event duplicated")
    body: dict[str, Any] = {
        "schema_version": TEACHER_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION,
        "materialization_id": _PENDING_ID,
        "teacher_id": teacher_id,
        "status": "complete_development_only_teacher_candidates_uncalibrated",
        "source_split_roster_ref": deepcopy(dict(source_split_roster_ref)),
        "teacher_model_ref": deepcopy(dict(teacher_model_ref)),
        "events": rows,
        "counts": {
            "event_count": len(rows),
            "candidate_count": sum(row["candidate_count"] for row in rows),
            "outer_fold_event_counts": dict(
                sorted(Counter(str(row["outer_holdout_fold"]) for row in rows).items())
            ),
        },
        "permissions": {
            "teacher_runtime_required_at_deployment": False,
            "candidate_soft_auxiliary_only": True,
            "node_localization_supervision_authorized": False,
            "calibration_authorized": False,
            "training_authorized": False,
            "locked_test_included": False,
        },
        "missing_closure_codes": ["fold_local_calibration_receipts_missing"],
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["materialization_id"] = _MATERIALIZATION_ID_PREFIX + canonical_json_sha256(
        _materialization_id_source(body)
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_teacher_candidate_materialization(body)


def validate_teacher_candidate_materialization(
    value: object,
    *,
    output_root: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version", "materialization_id", "teacher_id", "status",
        "source_split_roster_ref", "teacher_model_ref", "events", "counts",
        "permissions", "missing_closure_codes", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("teacher materialization fields drifted")
    data = deepcopy(value)
    _finite_tree(data)
    if data["schema_version"] != TEACHER_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("teacher materialization schema drifted")
    if data["teacher_id"] not in TEACHER_IDS:
        raise ValueError("teacher materialization teacher ID drifted")
    if data["status"] != "complete_development_only_teacher_candidates_uncalibrated":
        raise ValueError("teacher materialization status drifted")
    split_ref = validate_artifact_ref(data["source_split_roster_ref"])
    if split_ref["artifact_kind"] != "split_roster":
        raise ValueError("teacher materialization split reference drifted")
    model_ref = validate_artifact_ref(data["teacher_model_ref"])
    if model_ref["artifact_kind"] not in {"teacher_model_checkpoint", "teacher_model_manifest"}:
        raise ValueError("teacher materialization model reference drifted")
    events = data["events"]
    if not isinstance(events, list) or not events or events != sorted(events, key=lambda row: row["event_id"]):
        raise ValueError("teacher materialization event roster drifted")
    seen: set[str] = set()
    folds: Counter[str] = Counter()
    candidate_count = 0
    for row in events:
        if type(row) is not dict or set(row) != {
            "event_id", "linkage_group_id", "outer_holdout_fold", "candidate_count",
            "relative_cache_path", "candidate_cache_ref"
        }:
            raise ValueError("teacher materialization event fields drifted")
        if not isinstance(row["event_id"], str) or not row["event_id"] or row["event_id"] in seen:
            raise ValueError("teacher materialization event identity drifted")
        seen.add(row["event_id"])
        if isinstance(row["outer_holdout_fold"], bool) or not isinstance(row["outer_holdout_fold"], int) or row["outer_holdout_fold"] < 0:
            raise ValueError("teacher materialization fold drifted")
        if isinstance(row["candidate_count"], bool) or not isinstance(row["candidate_count"], int) or row["candidate_count"] < 0:
            raise ValueError("teacher materialization candidate count drifted")
        ref = _require_ref(row["candidate_cache_ref"], kind="teacher_candidate_cache", context="teacher cache")
        if ref["payload_schema_version"] != TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION:
            raise ValueError("teacher materialization cache schema drifted")
        relative = row["relative_cache_path"]
        if (
            not isinstance(relative, str)
            or not relative.startswith("events/")
            or ".." in relative.split("/")
            or relative != f"events/{row['event_id']}/candidate_cache.json"
        ):
            raise ValueError("teacher materialization cache path is unsafe")
        if output_root is not None:
            from pathlib import Path, PurePosixPath
            root = Path(output_root).resolve(strict=True)
            rel = PurePosixPath(relative)
            path = root.joinpath(*rel.parts)
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            if path.is_symlink() or not path.is_file():
                raise ValueError("teacher materialization cache is not a regular file")
            raw = path.read_bytes()
            cache = json.loads(raw.decode("utf-8"))
            if raw != canonical_json_bytes(cache):
                raise ValueError("teacher materialization cache is not canonical JSON")
            checked = validate_teacher_candidate_cache(cache)
            cache_ref = build_json_artifact_ref(
                checked,
                artifact_kind="teacher_candidate_cache",
                payload_schema_version=TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION,
            )
            if cache_ref != ref:
                raise ValueError("teacher materialization cache reference drifted")
            if (
                checked["teacher_id"] != data["teacher_id"]
                or checked["teacher_model_ref"] != model_ref
                or checked["event_id"] != row["event_id"]
                or checked["linkage_group_id"] != row["linkage_group_id"]
                or checked["outer_holdout_fold"] != row["outer_holdout_fold"]
                or checked["counts"]["candidate_count"] != row["candidate_count"]
            ):
                raise ValueError("teacher materialization cache metadata drifted")
        candidate_count += row["candidate_count"]
        folds[str(row["outer_holdout_fold"])] += 1
    if data["counts"] != {
        "event_count": len(events),
        "candidate_count": candidate_count,
        "outer_fold_event_counts": dict(sorted(folds.items())),
    }:
        raise ValueError("teacher materialization counts drifted")
    if data["permissions"] != {
        "teacher_runtime_required_at_deployment": False,
        "candidate_soft_auxiliary_only": True,
        "node_localization_supervision_authorized": False,
        "calibration_authorized": False,
        "training_authorized": False,
        "locked_test_included": False,
    }:
        raise ValueError("teacher materialization permissions drifted")
    if data["missing_closure_codes"] != ["fold_local_calibration_receipts_missing"]:
        raise ValueError("teacher materialization missing closures drifted")
    expected_id = _MATERIALIZATION_ID_PREFIX + canonical_json_sha256(_materialization_id_source(data))[:24]
    if data["materialization_id"] != expected_id:
        raise ValueError("teacher materialization ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("teacher materialization receipt drifted")
    return data


__all__ = [
    "TEACHER_CANDIDATE_CACHE_SCHEMA_VERSION",
    "TEACHER_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION",
    "TEACHER_IDS",
    "build_teacher_candidate_cache",
    "validate_teacher_candidate_cache",
    "build_teacher_candidate_materialization",
    "validate_teacher_candidate_materialization",
]
