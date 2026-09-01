"""Versioned EEG-only adaptive event-analysis release.

This module binds the signal-only adaptive-search artifact to one independent
variable window per event.  The variable window is the *only* permitted input
for event Findings, evolution analysis, waveform rendering and report-language
generation in the v2 profile.

The legacy ``[-12,+48]`` crop is retained only as an explicitly named
``compatibility_core`` for the frozen v29 SOZ ranker.  It is never promoted to
the primary analysis window and is not evidence for a confirmed seizure,
cortical SOZ, epileptogenic zone or treatment target.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .adaptive_event_window import (
    derive_adaptive_event_analysis_window,
    validate_adaptive_event_analysis_window,
)
from .adaptive_search_materialization import (
    AdaptiveEnvelopeLoader,
    load_standard19_adaptive_envelope,
    materialize_adaptive_eeg_search,
    validate_adaptive_materialization_artifact,
)


ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID = "adaptive_event_findings_v2"
ADAPTIVE_EVENT_ANALYSIS_PLAN_SCHEMA_VERSION = "adaptive_event_analysis_plan_v2"
ADAPTIVE_EVENT_ANALYSIS_RELEASE_SCHEMA_VERSION = (
    "adaptive_event_analysis_release_v2"
)

_PRIMARY_CONSUMERS = [
    "signal_findings",
    "event_evolution",
    "waveform_rendering",
    "llm_report_generation",
]
_COMPATIBILITY_CONSUMERS = ["legacy_v29_soz_ranker"]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.chmod(path, 0o600)


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"adaptive profile JSON must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"adaptive profile JSON is not a regular file: {path}")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"adaptive profile JSON repeats key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"adaptive profile JSON contains {value!r}")

    payload = json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )
    if type(payload) is not dict:
        raise TypeError("adaptive profile JSON must contain an object")
    return payload


def build_adaptive_event_analysis_plan(
    adaptive_search_artifact: object,
) -> dict[str, Any]:
    """Bind each adaptive search event to its own variable Findings window."""

    search_artifact = validate_adaptive_materialization_artifact(
        adaptive_search_artifact
    )
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(search_artifact["events"], start=1):
        search_receipt = event["adaptive_search_receipt"]
        if search_receipt is None:
            primary_window = None
            compatibility_projection = None
            source_search_status = str(event["status"])
        else:
            primary_window = derive_adaptive_event_analysis_window(search_receipt)
            primary_window = validate_adaptive_event_analysis_window(primary_window)
            compatibility_projection = deepcopy(search_receipt["v29_projection"])
            source_search_status = str(search_receipt["status"])

        compatibility_interval = (
            None
            if compatibility_projection is None
            else compatibility_projection["fixed_window_recording_seconds"]
        )
        compatibility_decision = (
            "unavailable_adaptive_search"
            if compatibility_projection is None
            else str(compatibility_projection["decision"])
        )
        rows.append(
            {
                "event_index": event_index,
                "candidate_id": event["candidate_id"],
                "eeg_event_id": event["eeg_event_id"],
                "source_plan_id": event["plan"]["plan_id"],
                "source_search_status": source_search_status,
                "primary_findings_window": {
                    "role": (
                        "primary_signal_findings_evolution_waveform_and_language"
                    ),
                    "status": (
                        "unavailable_plan_context"
                        if primary_window is None
                        else str(primary_window["status"])
                    ),
                    "window_receipt": primary_window,
                    "allowed_consumers": list(_PRIMARY_CONSUMERS),
                    "compatibility_core_used": False,
                },
                "compatibility_core": {
                    "role": "legacy_v29_soz_ranker_only",
                    "decision": compatibility_decision,
                    "fixed_window_recording_seconds": compatibility_interval,
                    "source_projection": compatibility_projection,
                    "allowed_consumers": list(_COMPATIBILITY_CONSUMERS),
                    "forbidden_consumers": list(_PRIMARY_CONSUMERS),
                    "used_as_primary_findings_window": False,
                },
            }
        )

    body: dict[str, Any] = {
        "schema_version": ADAPTIVE_EVENT_ANALYSIS_PLAN_SCHEMA_VERSION,
        "profile_id": ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID,
        "recording_id": search_artifact["recording_id"],
        "patient_pseudonym": search_artifact["patient_pseudonym"],
        "source_signal_sha256": search_artifact["source_signal_sha256"],
        "recording_duration_seconds": search_artifact[
            "recording_duration_seconds"
        ],
        "source_adaptive_search_artifact_sha256": search_artifact[
            "artifact_sha256"
        ],
        "event_count": len(rows),
        "events": rows,
        "route_contract": {
            "event_window_cardinality": (
                "one_independent_window_slot_per_event_with_explicit_unavailable_state"
            ),
            "primary_findings_input": (
                "events[].primary_findings_window.window_receipt"
            ),
            "compatibility_soz_input": (
                "events[].compatibility_core.fixed_window_recording_seconds"
            ),
            "legacy_fixed_crop_is_primary_findings_input": False,
            "silent_padding_permitted": False,
        },
        "scope_receipt": {
            "eeg_signal_only": True,
            "edf_annotation_api_called": False,
            "excel_used": False,
            "clinical_context_used": False,
            "labels_or_ground_truth_used": False,
            "fixed_crop_used_for_findings_or_evolution": False,
        },
        "artifact_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["artifact_sha256"] = _canonical_sha256(body)
    return validate_adaptive_event_analysis_plan(body)


def validate_adaptive_event_analysis_plan(payload: object) -> dict[str, Any]:
    """Fail closed if primary and compatibility window roles are conflated."""

    if type(payload) is not dict:
        raise TypeError("adaptive event analysis plan must be an object")
    required = {
        "schema_version",
        "profile_id",
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "recording_duration_seconds",
        "source_adaptive_search_artifact_sha256",
        "event_count",
        "events",
        "route_contract",
        "scope_receipt",
        "artifact_sha256",
    }
    if set(payload) != required:
        raise ValueError("adaptive event analysis plan has missing or unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != ADAPTIVE_EVENT_ANALYSIS_PLAN_SCHEMA_VERSION
        or data["profile_id"] != ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID
    ):
        raise ValueError("adaptive event analysis plan schema/profile drifted")
    for field in ("recording_id", "patient_pseudonym"):
        if not isinstance(data[field], str) or not data[field]:
            raise ValueError(f"adaptive event analysis {field} is invalid")
    for field in (
        "source_signal_sha256",
        "source_adaptive_search_artifact_sha256",
        "artifact_sha256",
    ):
        value = data[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"adaptive event analysis {field} is invalid")
    events = data["events"]
    if not isinstance(events, list) or data["event_count"] != len(events):
        raise ValueError("adaptive event analysis event count drifted")
    candidate_ids: set[str] = set()
    event_ids: set[str] = set()
    window_ids: set[str] = set()
    for index, event in enumerate(events, start=1):
        if type(event) is not dict or set(event) != {
            "event_index",
            "candidate_id",
            "eeg_event_id",
            "source_plan_id",
            "source_search_status",
            "primary_findings_window",
            "compatibility_core",
        }:
            raise ValueError("adaptive event analysis event is malformed")
        if event["event_index"] != index:
            raise ValueError("adaptive event analysis event order drifted")
        for field, seen in (
            ("candidate_id", candidate_ids),
            ("eeg_event_id", event_ids),
        ):
            value = event[field]
            if not isinstance(value, str) or not value or value in seen:
                raise ValueError(f"adaptive event analysis {field} is invalid")
            seen.add(value)
        if not isinstance(event["source_plan_id"], str) or not event[
            "source_plan_id"
        ]:
            raise ValueError("adaptive event analysis source plan ID is invalid")
        if not isinstance(event["source_search_status"], str) or not event[
            "source_search_status"
        ]:
            raise ValueError("adaptive event analysis search status is invalid")

        primary = event["primary_findings_window"]
        if type(primary) is not dict or primary != {
            "role": "primary_signal_findings_evolution_waveform_and_language",
            "status": primary.get("status"),
            "window_receipt": primary.get("window_receipt"),
            "allowed_consumers": _PRIMARY_CONSUMERS,
            "compatibility_core_used": False,
        }:
            raise ValueError("adaptive primary Findings window role drifted")
        window_raw = primary["window_receipt"]
        if window_raw is not None:
            window = validate_adaptive_event_analysis_window(window_raw)
            if primary["status"] != window["status"]:
                raise ValueError("adaptive primary window status drifted")
            window_id = window["window_receipt_id"]
            if window_id in window_ids:
                raise ValueError("adaptive event windows are not event-independent")
            window_ids.add(window_id)
            if window["policy"]["legacy_fixed_minus12_plus48_used"] is not False:
                raise ValueError("adaptive primary window reused the fixed crop")

        compatibility = event["compatibility_core"]
        if type(compatibility) is not dict or set(compatibility) != {
            "role",
            "decision",
            "fixed_window_recording_seconds",
            "source_projection",
            "allowed_consumers",
            "forbidden_consumers",
            "used_as_primary_findings_window",
        }:
            raise ValueError("adaptive compatibility core is malformed")
        if (
            compatibility["role"] != "legacy_v29_soz_ranker_only"
            or compatibility["allowed_consumers"] != _COMPATIBILITY_CONSUMERS
            or compatibility["forbidden_consumers"] != _PRIMARY_CONSUMERS
            or compatibility["used_as_primary_findings_window"] is not False
        ):
            raise ValueError("adaptive compatibility core escaped its role")
        projection = compatibility["source_projection"]
        interval = compatibility["fixed_window_recording_seconds"]
        if projection is None:
            if (
                compatibility["decision"] != "unavailable_adaptive_search"
                or interval is not None
                or window_raw is not None
                or primary["status"] != "unavailable_plan_context"
            ):
                raise ValueError("unavailable adaptive event retained evidence")
        else:
            if type(projection) is not dict or set(projection) != {
                "decision",
                "refined_anchor_recording_seconds",
                "fixed_window_recording_seconds",
                "reason",
            }:
                raise ValueError("adaptive compatibility projection is malformed")
            if (
                compatibility["decision"] != projection["decision"]
                or interval != projection["fixed_window_recording_seconds"]
            ):
                raise ValueError("adaptive compatibility projection drifted")

    expected_route = {
        "event_window_cardinality": (
            "one_independent_window_slot_per_event_with_explicit_unavailable_state"
        ),
        "primary_findings_input": (
            "events[].primary_findings_window.window_receipt"
        ),
        "compatibility_soz_input": (
            "events[].compatibility_core.fixed_window_recording_seconds"
        ),
        "legacy_fixed_crop_is_primary_findings_input": False,
        "silent_padding_permitted": False,
    }
    if data["route_contract"] != expected_route:
        raise ValueError("adaptive event analysis route contract drifted")
    expected_scope = {
        "eeg_signal_only": True,
        "edf_annotation_api_called": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
        "fixed_crop_used_for_findings_or_evolution": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("adaptive event analysis violated EEG-only scope")
    digest = deepcopy(data)
    digest["artifact_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["artifact_sha256"] != _canonical_sha256(digest):
        raise ValueError("adaptive event analysis hash does not bind content")
    return data


def materialize_adaptive_event_analysis_profile(
    *,
    detection_manifest_path: Path,
    edf_path: Path,
    output_dir: Path,
    envelope_loader: AdaptiveEnvelopeLoader = load_standard19_adaptive_envelope,
) -> dict[str, Any]:
    """Atomically publish adaptive search plus the v2 per-event window plan."""

    target = output_dir.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        search_path = staging / "adaptive_search.json"
        search = materialize_adaptive_eeg_search(
            detection_manifest_path=detection_manifest_path,
            edf_path=edf_path,
            output_path=search_path,
            envelope_loader=envelope_loader,
        )
        plan = build_adaptive_event_analysis_plan(search)
        plan_path = staging / "event_analysis_plan.json"
        _write_json(plan_path, plan)
        manifest = {
            "schema_version": ADAPTIVE_EVENT_ANALYSIS_RELEASE_SCHEMA_VERSION,
            "profile_id": ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID,
            "status": "completed_eeg_only_adaptive_event_analysis",
            "recording_id": plan["recording_id"],
            "patient_pseudonym": plan["patient_pseudonym"],
            "source_signal_sha256": plan["source_signal_sha256"],
            "event_count": plan["event_count"],
            "artifacts": {
                "adaptive_search.json": _file_sha256(search_path),
                "event_analysis_plan.json": _file_sha256(plan_path),
            },
            "scope_receipt": {
                "eeg_signal_only": True,
                "edf_annotation_api_called": False,
                "excel_used": False,
                "clinical_context_used": False,
                "labels_or_ground_truth_used": False,
                "primary_findings_window_is_adaptive": True,
                "fixed_crop_role": "compatibility_core_only",
            },
        }
        _write_json(staging / "manifest.json", manifest)
        for path in staging.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        os.chmod(staging, 0o700)
        os.replace(staging, target)
        os.chmod(target, 0o700)
        published = True
        return validate_materialized_adaptive_event_analysis_profile(target)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def validate_materialized_adaptive_event_analysis_profile(
    output_dir: Path,
) -> dict[str, Any]:
    """Validate a published v2 directory and every content/file binding."""

    root = output_dir.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("adaptive event analysis release is not a directory")
    manifest = _strict_json(root / "manifest.json")
    required = {
        "schema_version",
        "profile_id",
        "status",
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "event_count",
        "artifacts",
        "scope_receipt",
    }
    if set(manifest) != required:
        raise ValueError("adaptive event analysis release manifest is malformed")
    if (
        manifest["schema_version"]
        != ADAPTIVE_EVENT_ANALYSIS_RELEASE_SCHEMA_VERSION
        or manifest["profile_id"] != ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID
        or manifest["status"] != "completed_eeg_only_adaptive_event_analysis"
    ):
        raise ValueError("adaptive event analysis release status drifted")
    artifacts = manifest["artifacts"]
    if type(artifacts) is not dict or set(artifacts) != {
        "adaptive_search.json",
        "event_analysis_plan.json",
    }:
        raise ValueError("adaptive event analysis release artifacts drifted")
    loaded: dict[str, dict[str, Any]] = {}
    for name, expected_hash in artifacts.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("adaptive event analysis artifact is not a regular file")
        if not isinstance(expected_hash, str) or _file_sha256(path) != expected_hash:
            raise ValueError("adaptive event analysis artifact hash drifted")
        loaded[name] = _strict_json(path)
    search = validate_adaptive_materialization_artifact(
        loaded["adaptive_search.json"]
    )
    plan = validate_adaptive_event_analysis_plan(
        loaded["event_analysis_plan.json"]
    )
    for field in (
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "event_count",
    ):
        if manifest[field] != plan[field] or manifest[field] != search[field]:
            raise ValueError("adaptive event analysis release identity drifted")
    if plan["source_adaptive_search_artifact_sha256"] != search["artifact_sha256"]:
        raise ValueError("adaptive event analysis plan is not bound to its search")
    expected_scope = {
        "eeg_signal_only": True,
        "edf_annotation_api_called": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
        "primary_findings_window_is_adaptive": True,
        "fixed_crop_role": "compatibility_core_only",
    }
    if manifest["scope_receipt"] != expected_scope:
        raise ValueError("adaptive event analysis release violated EEG-only scope")
    return manifest


__all__ = [
    "ADAPTIVE_EVENT_ANALYSIS_PLAN_SCHEMA_VERSION",
    "ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID",
    "ADAPTIVE_EVENT_ANALYSIS_RELEASE_SCHEMA_VERSION",
    "build_adaptive_event_analysis_plan",
    "materialize_adaptive_event_analysis_profile",
    "validate_adaptive_event_analysis_plan",
    "validate_materialized_adaptive_event_analysis_profile",
]
