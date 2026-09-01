"""Project progressive terminal searches into primary Findings windows.

Each event consumes only the terminal adaptive-search receipt from a validated
progressive execution.  ``derive_adaptive_event_analysis_window`` converts that
receipt into the event's primary variable Findings/evolution/waveform window.
The terminal search's fixed v29 projection remains an explicitly isolated
compatibility core for the legacy ranker and cannot enter Findings or language.

This opt-in projection is not wired into the default private batch.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from .adaptive_event_window import (
    derive_adaptive_event_analysis_window,
    validate_adaptive_event_analysis_window,
)
from .adaptive_search import validate_adaptive_search_receipt
from .progressive_state_search_executor import (
    validate_progressive_eeg_state_search_execution,
)


PROGRESSIVE_FINDINGS_PROJECTION_SCHEMA_VERSION = (
    "progressive_execution_primary_findings_projection_v1"
)
PROGRESSIVE_FINDINGS_PROJECTION_METHOD_ID = (
    "terminal_adaptive_receipt_to_variable_findings_window_v1"
)

_PRIMARY_CONSUMERS = [
    "signal_findings",
    "event_evolution",
    "waveform_rendering",
    "llm_report_generation",
]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_progressive_primary_findings_projection(
    progressive_execution_receipt: object,
) -> dict[str, Any]:
    """Derive one primary variable-window slot per progressive event."""

    execution = validate_progressive_eeg_state_search_execution(
        progressive_execution_receipt
    )
    rows: list[dict[str, Any]] = []
    for event in execution["events"]:
        terminal_search = validate_adaptive_search_receipt(
            event["executed_levels"][-1]["adaptive_search_receipt"]
        )
        window = derive_adaptive_event_analysis_window(terminal_search)
        window = validate_adaptive_event_analysis_window(window)
        compatibility_projection = deepcopy(terminal_search["v29_projection"])
        rows.append(
            {
                "event_index": event["event_index"],
                "candidate_id": event["candidate_id"],
                "terminal_execution_status": event["terminal_status"],
                "terminal_execution_decision": event["terminal_decision"],
                "terminal_censoring": deepcopy(event["censoring"]),
                "terminal_adaptive_search_receipt": terminal_search,
                "primary_findings_window": {
                    "role": (
                        "primary_signal_findings_evolution_waveform_and_language"
                    ),
                    "status": window["status"],
                    "window_receipt": window,
                    "allowed_consumers": list(_PRIMARY_CONSUMERS),
                    "compatibility_core_used": False,
                },
                "compatibility_core": {
                    "role": "legacy_v29_soz_ranker_only",
                    "decision": compatibility_projection["decision"],
                    "fixed_window_recording_seconds": compatibility_projection[
                        "fixed_window_recording_seconds"
                    ],
                    "source_projection": compatibility_projection,
                    "allowed_consumers": ["legacy_v29_soz_ranker"],
                    "forbidden_consumers": list(_PRIMARY_CONSUMERS),
                    "used_as_primary_findings_window": False,
                },
            }
        )
    body: dict[str, Any] = {
        "schema_version": PROGRESSIVE_FINDINGS_PROJECTION_SCHEMA_VERSION,
        "projection_id": "CONTENT-ADDRESS-PENDING",
        "method_id": PROGRESSIVE_FINDINGS_PROJECTION_METHOD_ID,
        "source_progressive_execution_id": execution["execution_id"],
        "source_progressive_execution_sha256": _canonical_sha256(execution),
        "recording_id": execution["recording_id"],
        "source_signal_sha256": execution["source_signal_sha256"],
        "recording_duration_seconds": execution["recording_duration_seconds"],
        "event_count": len(rows),
        "events": rows,
        "route_contract": {
            "source_event_receipt": (
                "events[].terminal_adaptive_search_receipt"
            ),
            "primary_findings_input": (
                "events[].primary_findings_window.window_receipt"
            ),
            "compatibility_ranker_input": (
                "events[].compatibility_core.fixed_window_recording_seconds"
            ),
            "legacy_fixed_crop_is_primary_findings_input": False,
            "default_batch_consumer_connected": False,
        },
        "scope_receipt": {
            "progressive_execution_only": True,
            "eeg_signal_only_upstream": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "clinical_context_used": False,
            "labels_or_ground_truth_used": False,
            "fixed_crop_used_for_findings_or_evolution": False,
            "joint_state_segmentation_claimed_complete": False,
        },
    }
    body["projection_id"] = "PROGRESSIVE-FINDINGS-" + _canonical_sha256(body)[:20]
    return validate_progressive_primary_findings_projection(body)


def validate_progressive_primary_findings_projection(
    payload: object,
) -> dict[str, Any]:
    """Re-derive every variable window and enforce compatibility isolation."""

    if type(payload) is not dict:
        raise TypeError("progressive Findings projection must be an object")
    required = {
        "schema_version",
        "projection_id",
        "method_id",
        "source_progressive_execution_id",
        "source_progressive_execution_sha256",
        "recording_id",
        "source_signal_sha256",
        "recording_duration_seconds",
        "event_count",
        "events",
        "route_contract",
        "scope_receipt",
    }
    if set(payload) != required:
        raise ValueError("progressive Findings projection has invalid fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != PROGRESSIVE_FINDINGS_PROJECTION_SCHEMA_VERSION
        or data["method_id"] != PROGRESSIVE_FINDINGS_PROJECTION_METHOD_ID
    ):
        raise ValueError("progressive Findings projection schema/method drifted")
    for field in ("source_progressive_execution_sha256", "source_signal_sha256"):
        value = data[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"progressive Findings projection {field} is invalid")
    for field in (
        "projection_id",
        "source_progressive_execution_id",
        "recording_id",
    ):
        if not isinstance(data[field], str) or not data[field]:
            raise ValueError(f"progressive Findings projection {field} is invalid")
    events = data["events"]
    if not isinstance(events, list) or data["event_count"] != len(events):
        raise ValueError("progressive Findings projection event count drifted")
    seen: set[str] = set()
    window_ids: set[str] = set()
    for event_index, event in enumerate(events, start=1):
        if type(event) is not dict or set(event) != {
            "event_index",
            "candidate_id",
            "terminal_execution_status",
            "terminal_execution_decision",
            "terminal_censoring",
            "terminal_adaptive_search_receipt",
            "primary_findings_window",
            "compatibility_core",
        }:
            raise ValueError("progressive Findings event is malformed")
        candidate_id = event["candidate_id"]
        if (
            event["event_index"] != event_index
            or not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen
        ):
            raise ValueError("progressive Findings event identity drifted")
        seen.add(candidate_id)
        search = validate_adaptive_search_receipt(
            event["terminal_adaptive_search_receipt"]
        )
        expected_window = derive_adaptive_event_analysis_window(search)
        expected_window = validate_adaptive_event_analysis_window(expected_window)
        primary = event["primary_findings_window"]
        if type(primary) is not dict or primary != {
            "role": "primary_signal_findings_evolution_waveform_and_language",
            "status": expected_window["status"],
            "window_receipt": expected_window,
            "allowed_consumers": _PRIMARY_CONSUMERS,
            "compatibility_core_used": False,
        }:
            raise ValueError("progressive primary Findings window drifted")
        window_id = expected_window["window_receipt_id"]
        if window_id in window_ids:
            raise ValueError("progressive Findings windows are not event-independent")
        window_ids.add(window_id)
        if expected_window["policy"]["legacy_fixed_minus12_plus48_used"] is not False:
            raise ValueError("progressive primary Findings window reused fixed crop")
        compatibility = event["compatibility_core"]
        projection = search["v29_projection"]
        if type(compatibility) is not dict or compatibility != {
            "role": "legacy_v29_soz_ranker_only",
            "decision": projection["decision"],
            "fixed_window_recording_seconds": projection[
                "fixed_window_recording_seconds"
            ],
            "source_projection": projection,
            "allowed_consumers": ["legacy_v29_soz_ranker"],
            "forbidden_consumers": _PRIMARY_CONSUMERS,
            "used_as_primary_findings_window": False,
        }:
            raise ValueError("progressive compatibility core escaped its role")
        terminal_status = event["terminal_execution_status"]
        terminal_decision = event["terminal_execution_decision"]
        censoring = event["terminal_censoring"]
        if terminal_status == "completed_qualified_state_closed":
            if (
                terminal_decision != "stop_qualified_complete"
                or search["status"] != "qualified_complete"
                or censoring != {"is_censored": False, "reasons": []}
            ):
                raise ValueError("progressive qualified terminal binding drifted")
        elif terminal_status == "completed_censored_state_unresolved":
            if (
                terminal_decision
                not in {
                    "stop_recording_boundary_censored",
                    "stop_search_cap_censored",
                }
                or search["status"] == "qualified_complete"
                or type(censoring) is not dict
                or censoring.get("is_censored") is not True
                or not isinstance(censoring.get("reasons"), list)
                or not censoring["reasons"]
            ):
                raise ValueError("progressive censored terminal binding drifted")
        else:
            raise ValueError("progressive terminal execution status is unsupported")

    expected_route = {
        "source_event_receipt": "events[].terminal_adaptive_search_receipt",
        "primary_findings_input": (
            "events[].primary_findings_window.window_receipt"
        ),
        "compatibility_ranker_input": (
            "events[].compatibility_core.fixed_window_recording_seconds"
        ),
        "legacy_fixed_crop_is_primary_findings_input": False,
        "default_batch_consumer_connected": False,
    }
    if data["route_contract"] != expected_route:
        raise ValueError("progressive Findings route contract drifted")
    expected_scope = {
        "progressive_execution_only": True,
        "eeg_signal_only_upstream": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
        "fixed_crop_used_for_findings_or_evolution": False,
        "joint_state_segmentation_claimed_complete": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("progressive Findings projection violated EEG-only scope")
    digest = deepcopy(data)
    digest["projection_id"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "PROGRESSIVE-FINDINGS-" + _canonical_sha256(digest)[:20]
    if data["projection_id"] != expected_id:
        raise ValueError("progressive Findings projection ID does not bind content")
    return data


__all__ = [
    "PROGRESSIVE_FINDINGS_PROJECTION_METHOD_ID",
    "PROGRESSIVE_FINDINGS_PROJECTION_SCHEMA_VERSION",
    "build_progressive_primary_findings_projection",
    "validate_progressive_primary_findings_projection",
]
