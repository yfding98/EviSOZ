"""Validate a legacy source-context sidecar for post-freeze audit only.

The sidecar is not an EEG fact ledger and is ineligible for report rendering,
Findings, SOZ inference or language generation.  The legacy binding API is
retained for read-only evaluation compatibility; active long-recording
materializers and renderers reject its output.  A binding mismatch fails
closed.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from src.soz.long_term_clinical_context import (
    validate_long_term_clinical_context,
)


SOURCE_CONTEXT_BINDING_SCHEMA = "long_term_report_source_context_binding_v1"
SOURCE_CONTEXT_DISPLAY_POLICY = (
    "legacy_name_postfreeze_audit_only_never_rendered_v2"
)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def bind_source_context_to_frozen_bundle(
    bundle: object,
    context: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one context sidecar against the exact frozen report events.

    The association registry must contain exactly the bundle event IDs and the
    same coarse/refined anchors.  Re-association is therefore required after
    any adaptive-anchor or event-selection change; an older coarse-anchor
    context cannot silently decorate a refined-anchor report.
    """

    source = _mapping(bundle, "frozen long-recording EEG bundle")
    required_bundle_keys = {
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "recording_duration_seconds",
        "events",
    }
    missing = required_bundle_keys - set(source)
    if missing:
        raise ValueError(f"frozen bundle missing context binding keys: {sorted(missing)}")

    validated = validate_long_term_clinical_context(
        context,
        expected_recording_id=str(source["recording_id"]),
        expected_patient_id=str(source["patient_pseudonym"]),
        expected_source_signal_sha256=str(source["source_signal_sha256"]),
    )
    bundle_duration = _finite(
        source["recording_duration_seconds"], "bundle recording duration"
    )
    if not math.isclose(
        bundle_duration,
        _finite(validated["recording_duration_seconds"], "context recording duration"),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("source context recording duration does not match bundle")

    raw_events = source["events"]
    if not isinstance(raw_events, list):
        raise TypeError("frozen bundle events must be a list")
    expected_anchors: dict[str, float] = {}
    for index, raw_event in enumerate(raw_events):
        event = _mapping(raw_event, f"frozen bundle event {index}")
        event_id = event.get("eeg_event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("frozen bundle event has no valid eeg_event_id")
        if event_id in expected_anchors:
            raise ValueError("frozen bundle repeats an eeg_event_id")
        expected_anchors[event_id] = _finite(
            event.get("candidate_anchor_offset_seconds"),
            f"frozen bundle event {event_id} anchor",
        )

    associations = validated["event_associations"]
    association_by_id = {
        str(item["eeg_event_id"]): item for item in associations
    }
    if len(association_by_id) != len(associations):
        raise ValueError("source context repeats an event association")
    if set(association_by_id) != set(expected_anchors):
        missing_context = sorted(set(expected_anchors) - set(association_by_id))
        excess_context = sorted(set(association_by_id) - set(expected_anchors))
        raise ValueError(
            "source context event registry does not match frozen report events "
            f"(missing={missing_context}, excess={excess_context})"
        )
    for event_id, expected_anchor in expected_anchors.items():
        actual_anchor = _finite(
            association_by_id[event_id]["event_anchor_recording_seconds"],
            f"source context event {event_id} anchor",
        )
        if not math.isclose(
            expected_anchor, actual_anchor, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "source context event anchor does not match frozen report event"
            )

    boundary = _mapping(validated["claim_boundary"], "context claim boundary")
    required_false = {
        "raw_annotation_text_released",
        "source_path_released",
        "direct_identity_released",
        "sleep_context_included",
        "provocation_context_included",
        "cardiac_or_emg_context_included",
        "annotations_used_for_detection",
        "annotations_used_for_ranking",
        "annotations_used_for_narrative",
        "annotations_used_for_impression",
        "excel_used_for_detection",
        "excel_used_for_ranking",
        "excel_used_for_narrative",
        "excel_used_for_impression",
        "unbound_annotations_create_events",
        "excel_automatically_bound",
        "physician_verification_inferred",
        "llm_access_allowed",
    }
    if any(boundary.get(key) is not False for key in required_false):
        raise ValueError("source context claim boundary is not report-display safe")

    bound_annotation_ids = {
        str(link["annotation_id"])
        for association in associations
        for link in association["annotation_links"]
    }
    bound_excel_ids = {
        str(observation_id)
        for association in associations
        for observation_id in association["excel_observation_ids"]
    }
    receipt = {
        "schema_version": SOURCE_CONTEXT_BINDING_SCHEMA,
        "display_policy": SOURCE_CONTEXT_DISPLAY_POLICY,
        "recording_id": validated["recording_id"],
        "source_signal_sha256": validated["source_signal_sha256"],
        "frozen_report_event_count": len(expected_anchors),
        "annotation_count": len(validated["annotations"]),
        "bound_annotation_count": len(bound_annotation_ids),
        "unbound_annotation_count": len(validated["unbound_annotation_ids"]),
        "excel_onset_observation_count": len(
            validated["excel_onset_observations"]
        ),
        "bound_excel_onset_observation_count": len(bound_excel_ids),
        "event_registry_exact_match": True,
        "event_anchors_exact_match": True,
        "raw_source_text_or_paths_present": False,
        "excluded_domains_present": False,
        "may_create_or_remove_events": False,
        "may_change_event_coordinates": False,
        "may_change_eeg_facts": False,
        "may_change_research_ranking": False,
        "may_change_automatic_eeg_impression": False,
        "sent_to_language_model": False,
        "displayed_as_separate_attributed_source_context": False,
        "eligible_for_findings_or_soz": False,
        "eligible_for_language_model": False,
        "eligible_for_report_rendering": False,
        "postfreeze_evaluation_only": True,
    }
    return deepcopy(validated), receipt


__all__ = [
    "SOURCE_CONTEXT_BINDING_SCHEMA",
    "SOURCE_CONTEXT_DISPLAY_POLICY",
    "bind_source_context_to_frozen_bundle",
]
