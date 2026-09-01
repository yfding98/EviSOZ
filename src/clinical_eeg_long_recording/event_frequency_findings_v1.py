"""Replayable S03 frequency Findings projected from native EEG measurements.

This module is a deliberately small semantic projection over
``BAIEGDenseMeasurementSidecar``.  The sidecar is the native-sample producer:
it validates the canonical signal/view hashes, maps physical time to the view
clock, applies per-family QC and bandwidth masks, and measures the spectrum.
This module retains that row-level lineage and gives the three native spectral
measurements needed by S03 a stable event-card representation:

* dominant-frequency range and median in hertz;
* spectral concentration and normalized spectral entropy;
* event-minus-comparison-context deltas when both opportunities are present;
* a time-ordered, per-unit trajectory of the measured values.

No frequency band is called pathological, rhythmic, periodic, ictal or
evolving.  A comparison context is only a signal-selected within-record
reference and is never described as normal background.  Missing/QC-masked
opportunities remain ``not_evaluable`` with null summaries; they are not zero
measurements or negative clinical findings.  Bipolar derivations remain whole
``lead`` units and are never attributed to either endpoint.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .ba_ieg_dense_measurement_sidecar import (
    BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
    DEFAULT_BA_IEG_DENSE_MEASUREMENT_POLICY,
    BAIEGDenseMeasurementPolicy,
    BAIEGDenseMeasurementSidecar,
    BAIEGDenseMeasurementViewInput,
    materialize_ba_ieg_dense_measurement_sidecar,
)
from .ba_ieg_training_contract import BA_IEG_DETERMINISTIC_TARGETS


EVENT_FREQUENCY_FINDINGS_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_event_frequency_findings_v1"
)
EVENT_FREQUENCY_FINDINGS_METHOD_ID: Final[str] = (
    "DETERMINISTIC-EVENT-FREQUENCY-FINDINGS-V1"
)
EVENT_FREQUENCY_FINDINGS_POLICY_ID: Final[str] = (
    "DETERMINISTIC-EVENT-FREQUENCY-PROJECTION-POLICY-V1"
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL = 1e-8
_SPECTRAL_TARGETS: Final[tuple[str, ...]] = (
    "dominant_frequency_hz",
    "spectral_concentration",
    "spectral_entropy",
)
_TARGET_INDEX = {
    name: BA_IEG_DETERMINISTIC_TARGETS.index(name) for name in _SPECTRAL_TARGETS
}
_QUERY_AUTHORITIES = frozenset(
    {
        "deterministic_signal_proposal",
        "frozen_model_proposal",
        "synthetic_signal_injection",
    }
)

_FIREWALL: Final[dict[str, bool]] = {
    "eeg_samples_used_by_native_source": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_or_reports_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_or_activation_information_used": False,
    "ecg_emg_eog_used": False,
    "llm_used": False,
}

_AUTHORIZATION: Final[dict[str, bool | str]] = {
    "event_card_slot_id": "S03_FREQUENCY",
    "projection_scope": "native_spectral_measurements_only",
    "comparison_context_is_normal_background": False,
    "pathological_frequency_term_authorized": False,
    "rhythmic_or_periodic_term_authorized": False,
    "definite_evolution_term_authorized": False,
    "electrographic_seizure_term_authorized": False,
    "onset_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
    "clinical_term_qualification_authorized": False,
    "negative_clinical_assertion_authorized": False,
    "report_text_authorized": False,
    "future_dependent_measurement_may_support_onset": False,
    "reference_change_may_create_clinical_fact": False,
    "proposal_or_latent_promoted_to_clinical_fact": False,
    "native_values_are_direct_signal_measurements": True,
    "whole_bipolar_lead_identity_required": True,
    "bipolar_endpoint_fact_projection_authorized": False,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _self_hash(value: Mapping[str, object]) -> str:
    body = deepcopy(dict(value))
    body.pop("receipt_sha256", None)
    return _canonical_sha256(body)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a contract-compatible identifier")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain two numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name}[{index}] must be finite and non-negative")
        result.append(number)
    if result[1] <= result[0] + _TOL:
        raise ValueError(f"{name} must be non-empty")
    return result[0], result[1]


def _contains(carrier: tuple[float, float], item: tuple[float, float]) -> bool:
    return item[0] >= carrier[0] - _TOL and item[1] <= carrier[1] + _TOL


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] - _TOL and right[0] < left[1] - _TOL


def _metric_summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {
            "status": "not_evaluable",
            "count": None,
            "minimum": None,
            "median": None,
            "maximum": None,
            "reason_codes": ["no_evaluable_native_spectral_measurement"],
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("native spectral values must be finite")
    return {
        "status": "measured",
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "reason_codes": [],
    }


def _context_delta(
    event: Mapping[str, object], context: Mapping[str, object]
) -> dict[str, object]:
    if event["status"] != "measured" or context["status"] != "measured":
        return {
            "status": "not_evaluable",
            "event_median_minus_context_median": None,
            "reason_codes": ["paired_event_context_measurement_unavailable"],
        }
    return {
        "status": "measured",
        "event_median_minus_context_median": float(event["median"])
        - float(context["median"]),
        "reason_codes": [],
    }


def _row_partition(
    interval: tuple[float, float],
    *,
    event_interval: tuple[float, float],
    context_intervals: Sequence[tuple[float, float]],
) -> str:
    in_event = _contains(event_interval, interval)
    in_context = any(_contains(item, interval) for item in context_intervals)
    if in_event and in_context:
        raise ValueError("one native spectral row cannot be event and context")
    if in_event:
        return "event_course"
    if in_context:
        return "signal_selected_comparison_context"
    return "outside_selected_scope"


def _window_row(
    sidecar: BAIEGDenseMeasurementSidecar,
    row: object,
    *,
    role: str,
) -> dict[str, object]:
    values: list[float] | None = None
    masks: list[bool] | None = None
    if row.training_row_index is not None:
        values = sidecar.targets.values[row.training_row_index].tolist()
        masks = sidecar.targets.value_mask[row.training_row_index].tolist()
    measurements: dict[str, object] = {}
    unavailable_reasons: set[str] = set()
    for name in _SPECTRAL_TARGETS:
        index = _TARGET_INDEX[name]
        available = bool(masks is not None and masks[index])
        reasons = list(row.target_reason_codes[index])
        if not available:
            unavailable_reasons.update(reasons)
        measurements[name] = float(values[index]) if available else None
    status = (
        "measured"
        if all(measurements[name] is not None for name in _SPECTRAL_TARGETS)
        else "not_evaluable"
    )
    return {
        "requested_row_index": int(row.requested_row_index),
        "training_row_index": row.training_row_index,
        "measurement_role": role,
        "requested_recording_interval_seconds": list(
            row.requested_recording_interval_seconds
        ),
        "recording_interval_seconds": list(row.recording_interval_seconds),
        "tensor_sample_interval": list(row.tensor_sample_interval),
        "status": status,
        **measurements,
        "reason_codes": [] if status == "measured" else sorted(unavailable_reasons),
        "source_binding_sha256": row.source_binding_sha256,
        "quality_mask_sha256": row.quality_mask_sha256,
        "reference_row_sha256": row.reference_row_sha256,
        "canonical_source_channel_ids": list(row.canonical_source_channel_ids),
    }


def _derived_unit_fields(
    windows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    event_rows = [
        row for row in windows if row["measurement_role"] == "event_course"
    ]
    context_rows = [
        row
        for row in windows
        if row["measurement_role"] == "signal_selected_comparison_context"
    ]
    event_summaries = {
        name: _metric_summary(
            [float(row[name]) for row in event_rows if row[name] is not None]
        )
        for name in _SPECTRAL_TARGETS
    }
    context_summaries = {
        name: _metric_summary(
            [float(row[name]) for row in context_rows if row[name] is not None]
        )
        for name in _SPECTRAL_TARGETS
    }
    deltas = {
        name: _context_delta(event_summaries[name], context_summaries[name])
        for name in _SPECTRAL_TARGETS
    }
    trajectory_points = [
        {
            "recording_interval_seconds": deepcopy(row["recording_interval_seconds"]),
            **{name: row[name] for name in _SPECTRAL_TARGETS},
            "source_binding_sha256": row["source_binding_sha256"],
        }
        for row in event_rows
        if row["status"] == "measured"
    ]
    transitions = []
    for left, right in zip(trajectory_points, trajectory_points[1:]):
        transitions.append(
            {
                "from_interval_seconds": deepcopy(left["recording_interval_seconds"]),
                "to_interval_seconds": deepcopy(right["recording_interval_seconds"]),
                "dominant_frequency_delta_hz": float(right["dominant_frequency_hz"])
                - float(left["dominant_frequency_hz"]),
                "spectral_concentration_delta": float(right["spectral_concentration"])
                - float(left["spectral_concentration"]),
                "spectral_entropy_delta": float(right["spectral_entropy"])
                - float(left["spectral_entropy"]),
                "clinical_evolution_qualified": False,
            }
        )
    trajectory_status = "measured" if trajectory_points else "not_evaluable"
    requested_event_windows = len(event_rows)
    evaluable_event_windows = sum(
        row["status"] == "measured" for row in event_rows
    )
    if requested_event_windows == 0 or evaluable_event_windows == 0:
        four_state = "not_evaluable"
        four_state_reasons = ["no_evaluable_native_event_spectral_opportunity"]
    elif evaluable_event_windows < requested_event_windows:
        four_state = "uncertain"
        four_state_reasons = ["partial_native_event_spectral_opportunity"]
    else:
        four_state = "present"
        four_state_reasons = []
    return {
        "four_state_qualification": {
            "target": "native_frequency_spectral_measurement_availability",
            "state_semantics": (
                "metric_opportunity_only_not_frequency_band_or_pathology_presence"
            ),
            "state": four_state,
            "state_vocabulary": [
                "present",
                "absent_with_opportunity",
                "uncertain",
                "not_evaluable",
            ],
            "absent_with_opportunity_authorized": False,
            "clinical_term_qualified": False,
            "clinical_term_state_authorized": False,
            "frequency_band_presence_state_authorized": False,
            "pathological_rhythm_presence_state_authorized": False,
            "complete_opportunity": bool(
                requested_event_windows > 0
                and evaluable_event_windows == requested_event_windows
            ),
            "reason_codes": four_state_reasons,
        },
        "event_opportunity": {
            "status": "measured"
            if any(row["status"] == "measured" for row in event_rows)
            else "not_evaluable",
            "requested_window_count": len(event_rows),
            "evaluable_window_count": sum(
                row["status"] == "measured" for row in event_rows
            ),
            "not_evaluable_is_negative": False,
        },
        "comparison_context_opportunity": {
            "status": "measured"
            if any(row["status"] == "measured" for row in context_rows)
            else "not_evaluable",
            "requested_window_count": len(context_rows),
            "evaluable_window_count": sum(
                row["status"] == "measured" for row in context_rows
            ),
            "context_is_normal_background": False,
        },
        "event_summary": event_summaries,
        "comparison_context_summary": context_summaries,
        "event_minus_context_delta": deltas,
        "event_trajectory": {
            "status": trajectory_status,
            "points": trajectory_points,
            "adjacent_transitions": transitions,
            "ordered_change_is_clinical_evolution": False,
            "reason_codes": []
            if trajectory_status == "measured"
            else ["no_evaluable_event_frequency_points"],
        },
    }


def _unit_result(
    sidecar: BAIEGDenseMeasurementSidecar,
    rows: Sequence[object],
    *,
    event_interval: tuple[float, float],
    context_intervals: Sequence[tuple[float, float]],
) -> dict[str, object]:
    first = rows[0]
    windows: list[dict[str, object]] = []
    outside_hashes: list[str] = []
    for row in rows:
        interval = tuple(float(item) for item in row.recording_interval_seconds)
        role = _row_partition(
            interval,
            event_interval=event_interval,
            context_intervals=context_intervals,
        )
        if role == "outside_selected_scope":
            outside_hashes.append(row.source_binding_sha256)
            continue
        windows.append(_window_row(sidecar, row, role=role))
    windows.sort(
        key=lambda item: (
            0 if item["measurement_role"] == "signal_selected_comparison_context" else 1,
            float(item["recording_interval_seconds"][0]),
            int(item["requested_row_index"]),
        )
    )
    derived = _derived_unit_fields(windows)
    return {
        "view_index": int(first.view_index),
        "unit_index": int(first.unit_index),
        "view_id": str(first.view_id),
        "unit_id": str(first.unit_id),
        "unit_type": str(first.unit_type),
        "reference_type": str(first.reference_type),
        "whole_output_unit_identity_preserved": True,
        "bipolar_endpoint_fact_projection_authorized": False,
        "window_measurements": windows,
        **derived,
        "outside_selected_scope_row_count": len(outside_hashes),
        "outside_selected_scope_row_source_binding_sha256s": sorted(outside_hashes),
    }


def materialize_event_frequency_findings_v1(
    *,
    event_id: str,
    dense_measurement_sidecar: BAIEGDenseMeasurementSidecar,
    event_course_interval_seconds: Sequence[float],
    comparison_set_id: str,
    selection_receipt_sha256: str,
    query_authority: str = "frozen_model_proposal",
) -> dict[str, Any]:
    """Project native sidecar rows into a factorized S03 event Finding."""

    _identifier(event_id, "event_id")
    _identifier(comparison_set_id, "comparison_set_id")
    _sha(selection_receipt_sha256, "selection_receipt_sha256")
    if query_authority not in _QUERY_AUTHORITIES:
        raise ValueError("query_authority must be signal-only or synthetic")
    if not isinstance(dense_measurement_sidecar, BAIEGDenseMeasurementSidecar):
        raise TypeError(
            "dense_measurement_sidecar must be BAIEGDenseMeasurementSidecar"
        )
    dense_measurement_sidecar.verify_integrity()
    event_interval = _interval(
        event_course_interval_seconds, "event_course_interval_seconds"
    )
    analysis = dense_measurement_sidecar.analysis_interval_seconds
    if not _contains(analysis, event_interval):
        raise ValueError("event course lies outside the native sidecar analysis")
    contexts = tuple(dense_measurement_sidecar.background_intervals_seconds)
    if any(_overlaps(event_interval, item) for item in contexts):
        raise ValueError("event course overlaps its signal-selected context")

    grouped: dict[tuple[int, int], list[object]] = {}
    for row in dense_measurement_sidecar.row_bindings:
        grouped.setdefault((row.view_index, row.unit_index), []).append(row)
    units = [
        _unit_result(
            dense_measurement_sidecar,
            sorted(rows, key=lambda row: row.requested_row_index),
            event_interval=event_interval,
            context_intervals=contexts,
        )
        for _, rows in sorted(grouped.items())
    ]
    body: dict[str, Any] = {
        "schema_version": EVENT_FREQUENCY_FINDINGS_SCHEMA_VERSION,
        "method_id": EVENT_FREQUENCY_FINDINGS_METHOD_ID,
        "policy_id": EVENT_FREQUENCY_FINDINGS_POLICY_ID,
        "event_id": event_id,
        "event_card_slot_id": "S03_FREQUENCY",
        "source": {
            "schema_version": BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
            "recording_id": dense_measurement_sidecar.recording_id,
            "canonical_signal_id": dense_measurement_sidecar.canonical_signal_id,
            "canonical_receipt_sha256": (
                dense_measurement_sidecar.canonical_receipt_sha256
            ),
            "source_signal_sha256": dense_measurement_sidecar.source_signal_sha256,
            "source_binding_sha256": dense_measurement_sidecar.source_binding_sha256,
            "sidecar_receipt_sha256": dense_measurement_sidecar.receipt_sha256,
            "target_receipt_sha256": dense_measurement_sidecar.targets.receipt_sha256,
            "native_target_names": list(BA_IEG_DETERMINISTIC_TARGETS),
        },
        "selection": {
            "analysis_interval_seconds": list(analysis),
            "event_course_interval_seconds": list(event_interval),
            "comparison_context_intervals_seconds": [list(item) for item in contexts],
            "comparison_set_id": comparison_set_id,
            "query_authority": query_authority,
            "selection_receipt_sha256": selection_receipt_sha256,
            "selection_must_be_eeg_signal_only": True,
            "comparison_context_is_normal_background": False,
        },
        "measurement_definitions": {
            "dominant_frequency_hz": "maximum_power_fft_bin_within_effective_analysis_band",
            "spectral_concentration": "peak_bin_power_over_total_analysis_band_power",
            "spectral_entropy": "normalized_shannon_entropy_over_analysis_band_bins",
            "frequency_range": "minimum_to_maximum_dominant_frequency_across_evaluable_windows",
            "context_delta": "event_median_minus_signal_selected_context_median",
        },
        "units": units,
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_event_frequency_findings_v1(body)


def materialize_event_frequency_findings_from_native_signal_v1(
    *,
    event_id: str,
    canonical_receipt: object,
    views: Sequence[BAIEGDenseMeasurementViewInput],
    analysis_interval_seconds: Sequence[float],
    event_course_interval_seconds: Sequence[float],
    comparison_context_intervals_seconds: Sequence[Sequence[float]],
    comparison_set_id: str,
    selection_receipt_sha256: str,
    query_authority: str = "frozen_model_proposal",
    measurement_policy: BAIEGDenseMeasurementPolicy = (
        DEFAULT_BA_IEG_DENSE_MEASUREMENT_POLICY
    ),
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[dict[str, Any], BAIEGDenseMeasurementSidecar]:
    """Remeasure native EEG first, then project the resulting S03 receipt.

    Returning the immutable dense sidecar makes the native numerical source a
    first-class artifact rather than hiding it behind the semantic projection.
    The projection never consumes model logits, proposals or latent vectors as
    frequency facts.
    """

    sidecar = materialize_ba_ieg_dense_measurement_sidecar(
        canonical_receipt=canonical_receipt,
        views=views,
        analysis_interval_seconds=analysis_interval_seconds,
        background_intervals_seconds=comparison_context_intervals_seconds,
        policy=measurement_policy,
        trusted_parent_views=trusted_parent_views,
    )
    receipt = materialize_event_frequency_findings_v1(
        event_id=event_id,
        dense_measurement_sidecar=sidecar,
        event_course_interval_seconds=event_course_interval_seconds,
        comparison_set_id=comparison_set_id,
        selection_receipt_sha256=selection_receipt_sha256,
        query_authority=query_authority,
    )
    return receipt, sidecar


def validate_event_frequency_findings_v1(value: object) -> dict[str, Any]:
    """Validate the closed S03 projection contract without source replay."""

    if not isinstance(value, Mapping):
        raise TypeError("event frequency Findings must be an object")
    payload = deepcopy(dict(value))
    expected_top = {
        "schema_version",
        "method_id",
        "policy_id",
        "event_id",
        "event_card_slot_id",
        "source",
        "selection",
        "measurement_definitions",
        "units",
        "firewall",
        "authorization",
        "receipt_sha256",
    }
    if set(payload) != expected_top:
        raise ValueError("event frequency Findings top-level fields drifted")
    if payload["schema_version"] != EVENT_FREQUENCY_FINDINGS_SCHEMA_VERSION:
        raise ValueError("event frequency Findings schema version drifted")
    if payload["method_id"] != EVENT_FREQUENCY_FINDINGS_METHOD_ID:
        raise ValueError("event frequency Findings method drifted")
    if payload["policy_id"] != EVENT_FREQUENCY_FINDINGS_POLICY_ID:
        raise ValueError("event frequency Findings policy drifted")
    _identifier(payload["event_id"], "event_id")
    if payload["event_card_slot_id"] != "S03_FREQUENCY":
        raise ValueError("event frequency Findings must bind S03")
    if payload["firewall"] != _FIREWALL or payload["authorization"] != _AUTHORIZATION:
        raise ValueError("event frequency Findings permissions drifted")
    if payload["measurement_definitions"] != {
        "dominant_frequency_hz": (
            "maximum_power_fft_bin_within_effective_analysis_band"
        ),
        "spectral_concentration": (
            "peak_bin_power_over_total_analysis_band_power"
        ),
        "spectral_entropy": (
            "normalized_shannon_entropy_over_analysis_band_bins"
        ),
        "frequency_range": (
            "minimum_to_maximum_dominant_frequency_across_evaluable_windows"
        ),
        "context_delta": (
            "event_median_minus_signal_selected_context_median"
        ),
    }:
        raise ValueError("event frequency measurement definitions drifted")
    source = payload["source"]
    if not isinstance(source, Mapping):
        raise TypeError("event frequency Findings source must be an object")
    if set(source) != {
        "schema_version",
        "recording_id",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "source_binding_sha256",
        "sidecar_receipt_sha256",
        "target_receipt_sha256",
        "native_target_names",
    }:
        raise ValueError("event frequency Findings source fields drifted")
    if source["schema_version"] != BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION:
        raise ValueError("event frequency native source schema drifted")
    _identifier(source["recording_id"], "source.recording_id")
    _identifier(source["canonical_signal_id"], "source.canonical_signal_id")
    if source["native_target_names"] != list(BA_IEG_DETERMINISTIC_TARGETS):
        raise ValueError("event frequency native target vocabulary drifted")
    for name in (
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "source_binding_sha256",
        "sidecar_receipt_sha256",
        "target_receipt_sha256",
    ):
        _sha(source[name], f"source.{name}")
    selection = payload["selection"]
    if not isinstance(selection, Mapping):
        raise TypeError("event frequency Findings selection must be an object")
    if set(selection) != {
        "analysis_interval_seconds",
        "event_course_interval_seconds",
        "comparison_context_intervals_seconds",
        "comparison_set_id",
        "query_authority",
        "selection_receipt_sha256",
        "selection_must_be_eeg_signal_only",
        "comparison_context_is_normal_background",
    }:
        raise ValueError("event frequency Findings selection fields drifted")
    analysis = _interval(
        selection["analysis_interval_seconds"], "analysis_interval_seconds"
    )
    event_interval = _interval(
        selection["event_course_interval_seconds"], "event_course_interval"
    )
    if not _contains(analysis, event_interval):
        raise ValueError("event frequency course lies outside its analysis")
    raw_contexts = selection["comparison_context_intervals_seconds"]
    if not isinstance(raw_contexts, list):
        raise TypeError("comparison context intervals must be an array")
    contexts = [
        _interval(value, f"comparison_context_intervals_seconds[{index}]")
        for index, value in enumerate(raw_contexts)
    ]
    if contexts != sorted(contexts) or any(
        _overlaps(left, right) for left, right in zip(contexts, contexts[1:])
    ):
        raise ValueError("comparison context intervals must be ordered and disjoint")
    if any(
        not _contains(analysis, context) or _overlaps(event_interval, context)
        for context in contexts
    ):
        raise ValueError("comparison context is outside analysis or overlaps event")
    _identifier(selection["comparison_set_id"], "comparison_set_id")
    _sha(selection["selection_receipt_sha256"], "selection_receipt_sha256")
    if selection["query_authority"] not in _QUERY_AUTHORITIES:
        raise ValueError("event frequency Findings has non-signal query authority")
    if (
        selection["selection_must_be_eeg_signal_only"] is not True
        or selection["comparison_context_is_normal_background"] is not False
    ):
        raise ValueError("event frequency Findings context permissions drifted")
    units = payload["units"]
    if not isinstance(units, list) or not units:
        raise ValueError("event frequency Findings requires a unit ledger")
    keys = [(row["view_index"], row["unit_index"]) for row in units]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("event frequency unit ledger order drifted")
    for row in units:
        if set(row) != {
            "view_index",
            "unit_index",
            "view_id",
            "unit_id",
            "unit_type",
            "reference_type",
            "whole_output_unit_identity_preserved",
            "bipolar_endpoint_fact_projection_authorized",
            "window_measurements",
            "event_opportunity",
            "four_state_qualification",
            "comparison_context_opportunity",
            "event_summary",
            "comparison_context_summary",
            "event_minus_context_delta",
            "event_trajectory",
            "outside_selected_scope_row_count",
            "outside_selected_scope_row_source_binding_sha256s",
        }:
            raise ValueError("event frequency unit fields drifted")
        _identifier(row["view_id"], "unit.view_id")
        _identifier(row["unit_id"], "unit.unit_id")
        _identifier(row["reference_type"], "unit.reference_type")
        if row["unit_type"] not in {"electrode", "lead"}:
            raise ValueError("frequency unit type must be electrode or whole lead")
        if (
            row["whole_output_unit_identity_preserved"] is not True
            or row["bipolar_endpoint_fact_projection_authorized"] is not False
        ):
            raise ValueError("frequency unit identity permissions drifted")
        windows = row["window_measurements"]
        if not isinstance(windows, list):
            raise TypeError("frequency window measurements must be an array")
        expected_window_order = sorted(
            windows,
            key=lambda item: (
                0
                if item["measurement_role"]
                == "signal_selected_comparison_context"
                else 1,
                float(item["recording_interval_seconds"][0]),
                int(item["requested_row_index"]),
            ),
        )
        if windows != expected_window_order:
            raise ValueError("frequency window measurement order drifted")
        requested_indices: set[int] = set()
        for window in windows:
            if set(window) != {
                "requested_row_index",
                "training_row_index",
                "measurement_role",
                "requested_recording_interval_seconds",
                "recording_interval_seconds",
                "tensor_sample_interval",
                "status",
                "dominant_frequency_hz",
                "spectral_concentration",
                "spectral_entropy",
                "reason_codes",
                "source_binding_sha256",
                "quality_mask_sha256",
                "reference_row_sha256",
                "canonical_source_channel_ids",
            }:
                raise ValueError("event frequency window fields drifted")
            requested_index = window["requested_row_index"]
            if (
                isinstance(requested_index, bool)
                or not isinstance(requested_index, int)
                or requested_index < 0
                or requested_index in requested_indices
            ):
                raise ValueError("frequency requested row indices must be unique")
            requested_indices.add(requested_index)
            if window["measurement_role"] not in {
                "event_course",
                "signal_selected_comparison_context",
            }:
                raise ValueError("frequency measurement role drifted")
            requested = _interval(
                window["requested_recording_interval_seconds"],
                "requested_recording_interval_seconds",
            )
            measured_interval = _interval(
                window["recording_interval_seconds"],
                "recording_interval_seconds",
            )
            if not _contains(requested, measured_interval):
                raise ValueError("frequency measured interval exceeds its request")
            expected_role = _row_partition(
                measured_interval,
                event_interval=event_interval,
                context_intervals=contexts,
            )
            if window["measurement_role"] != expected_role:
                raise ValueError("frequency window role disagrees with physical time")
            if window["status"] not in {"measured", "not_evaluable"}:
                raise ValueError("frequency window status drifted")
            for name in _SPECTRAL_TARGETS:
                measurement = window[name]
                if measurement is not None and (
                    isinstance(measurement, bool)
                    or not isinstance(measurement, (int, float))
                    or not math.isfinite(float(measurement))
                ):
                    raise ValueError("frequency window has a non-finite measurement")
            all_measured = all(window[name] is not None for name in _SPECTRAL_TARGETS)
            if (window["status"] == "measured") is not all_measured:
                raise ValueError("frequency window status disagrees with measurements")
            if all_measured and window["reason_codes"]:
                raise ValueError("measured frequency window carries failure reasons")
            if not all_measured and not window["reason_codes"]:
                raise ValueError("unevaluable frequency window lacks a reason")
            for name in (
                "source_binding_sha256",
                "quality_mask_sha256",
                "reference_row_sha256",
            ):
                _sha(window[name], f"window.{name}")
        derived = _derived_unit_fields(windows)
        for name, expected in derived.items():
            if row[name] != expected:
                raise ValueError(f"event frequency derived field {name} drifted")
        outside_hashes = row[
            "outside_selected_scope_row_source_binding_sha256s"
        ]
        if (
            not isinstance(outside_hashes, list)
            or outside_hashes != sorted(outside_hashes)
            or row["outside_selected_scope_row_count"] != len(outside_hashes)
        ):
            raise ValueError("frequency outside-scope row ledger drifted")
        for item in outside_hashes:
            _sha(item, "outside-scope row source binding")
    _sha(payload["receipt_sha256"], "receipt_sha256")
    if payload["receipt_sha256"] != _self_hash(payload):
        raise ValueError("event frequency Findings receipt hash drifted")
    return payload


def replay_event_frequency_findings_v1(
    expected: object,
    *,
    dense_measurement_sidecar: BAIEGDenseMeasurementSidecar,
) -> dict[str, Any]:
    """Recompose the S03 projection from its immutable native sidecar."""

    payload = validate_event_frequency_findings_v1(expected)
    selection = payload["selection"]
    replayed = materialize_event_frequency_findings_v1(
        event_id=payload["event_id"],
        dense_measurement_sidecar=dense_measurement_sidecar,
        event_course_interval_seconds=selection["event_course_interval_seconds"],
        comparison_set_id=selection["comparison_set_id"],
        selection_receipt_sha256=selection["selection_receipt_sha256"],
        query_authority=selection["query_authority"],
    )
    if replayed != payload:
        raise ValueError("event frequency Findings do not replay from native source")
    return replayed


def replay_event_frequency_findings_from_native_signal_v1(
    expected: object,
    *,
    canonical_receipt: object,
    views: Sequence[BAIEGDenseMeasurementViewInput],
    measurement_policy: BAIEGDenseMeasurementPolicy = (
        DEFAULT_BA_IEG_DENSE_MEASUREMENT_POLICY
    ),
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Re-run native spectral measurement and require exact S03 equality."""

    payload = validate_event_frequency_findings_v1(expected)
    selection = payload["selection"]
    replayed, _ = materialize_event_frequency_findings_from_native_signal_v1(
        event_id=payload["event_id"],
        canonical_receipt=canonical_receipt,
        views=views,
        analysis_interval_seconds=selection["analysis_interval_seconds"],
        event_course_interval_seconds=selection["event_course_interval_seconds"],
        comparison_context_intervals_seconds=(
            selection["comparison_context_intervals_seconds"]
        ),
        comparison_set_id=selection["comparison_set_id"],
        selection_receipt_sha256=selection["selection_receipt_sha256"],
        query_authority=selection["query_authority"],
        measurement_policy=measurement_policy,
        trusted_parent_views=trusted_parent_views,
    )
    if replayed != payload:
        raise ValueError("event frequency Findings do not replay from native EEG")
    return replayed


__all__ = [
    "EVENT_FREQUENCY_FINDINGS_METHOD_ID",
    "EVENT_FREQUENCY_FINDINGS_POLICY_ID",
    "EVENT_FREQUENCY_FINDINGS_SCHEMA_VERSION",
    "materialize_event_frequency_findings_from_native_signal_v1",
    "materialize_event_frequency_findings_v1",
    "replay_event_frequency_findings_from_native_signal_v1",
    "replay_event_frequency_findings_v1",
    "validate_event_frequency_findings_v1",
]
