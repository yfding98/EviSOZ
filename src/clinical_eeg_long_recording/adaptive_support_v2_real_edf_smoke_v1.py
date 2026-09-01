"""Smoke-only detector-cache to adaptive-support-v2 real-EDF bridge.

This module is intentionally non-promotable.  It selects the unique highest
probability item from an already frozen post-NMS common-17 peak cache for each
predeclared recording.  The selected peak center is only a navigation
coordinate, and its decoded interval is only a detector-envelope exclusion
input.  The pre-existing formal operating-point candidate count is preserved
separately and is never replaced by the smoke argmax.

Runtime inputs are limited to the frozen peak gzip, the EDF signal/header
allowlist needed by the direct-common17 reader, acquisition parameters, and
EEG-derived QC.  No reference interval, channel target, annotation stream,
clinical text, spreadsheet, or source-evaluation asset has an API route.
"""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .adaptive_native_evidence_common17 import COMMON17_CHANNELS
from .adaptive_support_v2 import (
    ADAPTIVE_SUPPORT_V2_METHOD_ID,
    DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY,
    _materialize_common17_adaptive_support_v2_with_state,
    evaluate_common17_anchor_jitter_shadow_v2,
    validate_common17_adaptive_support_v2,
    validate_common17_anchor_jitter_shadow_v2,
)
from .tusz_real_edf_adaptive_findings_v1 import (
    DirectObservedCommon17EDFQueryReader,
)


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA: Final[str] = (
    "clinical_eeg_adaptive_support_v2_real_edf_smoke_manifest_v1"
)
EVENT_SCHEMA: Final[str] = (
    "clinical_eeg_adaptive_support_v2_real_edf_smoke_event_v1"
)
COHORT_SCHEMA: Final[str] = (
    "clinical_eeg_adaptive_support_v2_real_edf_smoke_cohort_v1"
)
METHOD_ID: Final[str] = "COMMON17-FROZEN-POSTNMS-ARGMAX-V2-REAL-EDF-SMOKE-V1"

_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "smoke_id",
        "status",
        "selection_firewall",
        "detector_cache_contract",
        "adaptive_and_jitter_contract",
        "entries",
    }
)
_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "recording_id",
        "official_split",
        "relative_edf_path",
        "edf_sha256",
        "analysis_identity_id",
        "prediction_relative_path",
        "prediction_file_sha256",
        "expected_source_sampling_rate_hz",
        "expected_recording_sample_count",
        "expected_FZ_PZ_observation_state",
    }
)
_PREDICTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "analysis_identity_id",
        "checkpoint_file_sha256",
        "checkpoint_global_step",
        "common17_channel_order",
        "FZ_or_PZ_model_axis_present",
        "minimum_peak_threshold",
        "minimum_peak_distance_seconds",
        "smoothing_sigma_samples",
        "patient_id",
        "peaks",
        "recording_duration_seconds",
        "runtime",
    }
)
_PEAK_FIELDS: Final[frozenset[str]] = frozenset(
    {"center_probability", "center_seconds", "duration_fraction"}
)
_SELECTION_FIREWALL: Final[dict[str, object]] = {
    "asset_roster_source": "user_requested_preexisting_real_edf_smoke_assets",
    "performance_or_efficacy_cohort": False,
    "TERM_or_seizure_interval_opened_at_runtime": False,
    "SOZ_or_channel_target_opened_at_runtime": False,
    "EDF_annotations_opened_at_runtime": False,
    "clinical_text_or_spreadsheet_opened_at_runtime": False,
    "source_eval_opened": False,
}
_SCOPE: Final[dict[str, object]] = {
    "frozen_common17_post_NMS_peak_cache_used": True,
    "direct_common17_EEG_samples_used": True,
    "acquisition_parameters_and_EEG_QC_used": True,
    "formal_operating_point_candidate_count_preserved": True,
    "smoke_argmax_substitutes_for_formal_zero_candidate": False,
    "TERM_or_seizure_interval_used": False,
    "SOZ_or_channel_target_used": False,
    "EDF_annotations_used": False,
    "clinical_text_or_spreadsheet_used": False,
    "source_eval_used": False,
    "FZ_or_PZ_samples_read": False,
    "zero_fill_interpolation_or_montage_synthesis_used": False,
}
_CLAIMS: Final[dict[str, object]] = {
    "real_EEG_engineering_smoke_completed": True,
    "detector_provider_qualified": False,
    "formal_alarm_operating_point_evaluated_here": False,
    "adaptive_superiority_authorized": False,
    "detector_or_SOZ_efficacy_claim_authorized": False,
    "clinical_deployment_allowed": False,
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _identifier(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 180
        or any(character in value for character in ("/", "\\"))
    ):
        raise ValueError(f"{field} must be a safe identifier")
    return value


def _safe_relative(root: Path, value: object, *, suffix: str) -> tuple[str, Path]:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != suffix:
        raise ValueError("smoke source path is unsafe")
    resolved = root.joinpath(*relative.parts).resolve(strict=True)
    resolved.relative_to(root)
    return relative.as_posix(), resolved


def _validate_detector_contract(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("smoke detector cache contract must be an object")
    data = deepcopy(value)
    expected_keys = {
        "prediction_schema",
        "checkpoint_file_sha256",
        "checkpoint_global_step",
        "common17_detector_axis_order",
        "post_NMS_peak_cache_floor",
        "post_NMS_minimum_peak_distance_seconds",
        "post_NMS_smoothing_sigma_samples",
        "maximum_decoded_duration_seconds",
        "formal_source_dev_diagnostic_center_threshold",
        "formal_threshold_is_held_out_or_promotable",
        "smoke_navigation_rule",
        "smoke_navigation_rule_is_formal_alarm_operating_point",
        "navigation_anchor_field",
        "candidate_envelope_rule",
        "decoder_source_relative_path",
        "decoder_source_file_sha256",
        "prediction_roster_audit_relative_path",
        "prediction_roster_audit_file_sha256",
    }
    if set(data) != expected_keys:
        raise ValueError("smoke detector cache contract fields drifted")
    if (
        data["prediction_schema"]
        != "eventnet_common17_dev_prediction_global_posterior_runtime_v3"
        or data["checkpoint_file_sha256"]
        != "314a6f25f6c28d0e227f6ec9ab95960d5ba3e2d83f1308c6a8bbdd70b5b8e259"
        or data["checkpoint_global_step"] != 11680
        or data["post_NMS_peak_cache_floor"] != 0.001
        or data["post_NMS_minimum_peak_distance_seconds"] != 60
        or data["post_NMS_smoothing_sigma_samples"] != 100
        or data["maximum_decoded_duration_seconds"] != 300.0
        or data["formal_source_dev_diagnostic_center_threshold"] != 0.05
        or data["formal_threshold_is_held_out_or_promotable"] is not False
        or data["smoke_navigation_rule"]
        != "unique_argmax_center_probability_from_frozen_post_NMS_peak_cache"
        or data["smoke_navigation_rule_is_formal_alarm_operating_point"] is not False
        or data["navigation_anchor_field"] != "center_seconds_not_clinical_onset"
    ):
        raise ValueError("smoke detector cache semantics drifted")
    if data["common17_detector_axis_order"] != [
        "FP1",
        "F3",
        "C3",
        "P3",
        "O1",
        "F7",
        "T7",
        "P7",
        "CZ",
        "FP2",
        "F4",
        "C4",
        "P4",
        "O2",
        "F8",
        "T8",
        "P8",
    ]:
        raise ValueError("smoke detector common17 axis drifted")
    _sha256(data["decoder_source_file_sha256"], "decoder source SHA")
    _sha256(data["prediction_roster_audit_file_sha256"], "roster audit SHA")
    return data


def validate_real_edf_smoke_manifest_v1(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("adaptive-v2 real-EDF smoke manifest fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != MANIFEST_SCHEMA
        or data["status"] != "frozen_before_adaptive_v2_real_edf_query"
        or data["selection_firewall"] != _SELECTION_FIREWALL
    ):
        raise ValueError("adaptive-v2 real-EDF smoke manifest contract drifted")
    _identifier(data["smoke_id"], "smoke_id")
    _validate_detector_contract(data["detector_cache_contract"])
    adaptive = data["adaptive_and_jitter_contract"]
    if adaptive != {
        "adaptive_support_method_id": ADAPTIVE_SUPPORT_V2_METHOD_ID,
        "jitter_offsets_seconds": [-10.0, -5.0, 0.0, 5.0, 10.0],
        "jitter_same_physical_shadow_seconds": 120.0,
        "jitter_reruns_support_selection": False,
        "jitter_may_tune_policy": False,
    }:
        raise ValueError("adaptive-v2 smoke jitter contract drifted")
    entries = data["entries"]
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("adaptive-v2 real-EDF smoke requires the frozen two assets")
    seen: set[str] = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != _ENTRY_FIELDS:
            raise ValueError("adaptive-v2 smoke entry fields drifted")
        recording = _identifier(entry["recording_id"], "recording_id")
        if recording in seen:
            raise ValueError("adaptive-v2 smoke recording IDs are duplicated")
        seen.add(recording)
        if entry["official_split"] != "dev":
            raise ValueError("adaptive-v2 smoke must remain source-dev only")
        if not str(entry["relative_edf_path"]).startswith("dev/"):
            raise ValueError("adaptive-v2 smoke EDF escaped source-dev")
        _sha256(entry["edf_sha256"], "EDF SHA")
        identity = _identifier(entry["analysis_identity_id"], "analysis identity")
        if identity != f"TUSZANALYSIS-{entry['edf_sha256']}":
            raise ValueError("adaptive-v2 smoke identity is not EDF-content bound")
        _sha256(entry["prediction_file_sha256"], "prediction file SHA")
        if not str(entry["prediction_relative_path"]).endswith(f"/{identity}.json.gz"):
            raise ValueError("adaptive-v2 smoke prediction path/identity drifted")
        rate = float(entry["expected_source_sampling_rate_hz"])
        samples = entry["expected_recording_sample_count"]
        if (
            not math.isfinite(rate)
            or rate < 10.0
            or isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples < int(120.0 * rate)
        ):
            raise ValueError("adaptive-v2 smoke acquisition expectation is invalid")
        if entry["expected_FZ_PZ_observation_state"] not in {
            "both_observed_but_excluded",
            "both_naturally_absent",
        }:
            raise ValueError("adaptive-v2 smoke midline observation state drifted")
    return data


def load_real_edf_smoke_manifest_v1(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    return validate_real_edf_smoke_manifest_v1(
        json.loads(source.read_text(encoding="utf-8"))
    )


def _load_prediction(
    path: Path,
    *,
    entry: Mapping[str, object],
    contract: Mapping[str, object],
) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if type(value) is not dict or set(value) != _PREDICTION_FIELDS:
        raise ValueError("frozen detector peak payload fields drifted")
    data = deepcopy(value)
    expected_patient = str(entry["recording_id"]).split("_s", 1)[0]
    if (
        data["schema_version"] != contract["prediction_schema"]
        or data["analysis_identity_id"] != entry["analysis_identity_id"]
        or data["checkpoint_file_sha256"] != contract["checkpoint_file_sha256"]
        or data["checkpoint_global_step"] != contract["checkpoint_global_step"]
        or data["common17_channel_order"] != contract["common17_detector_axis_order"]
        or data["FZ_or_PZ_model_axis_present"] is not False
        or data["minimum_peak_threshold"] != contract["post_NMS_peak_cache_floor"]
        or data["minimum_peak_distance_seconds"]
        != contract["post_NMS_minimum_peak_distance_seconds"]
        or data["smoothing_sigma_samples"]
        != contract["post_NMS_smoothing_sigma_samples"]
        or data["patient_id"] != expected_patient
    ):
        raise ValueError("frozen detector peak payload binding drifted")
    duration = float(data["recording_duration_seconds"])
    if not math.isfinite(duration) or duration <= 120.0:
        raise ValueError("frozen detector recording duration is invalid")
    peaks = data["peaks"]
    if not isinstance(peaks, list) or not peaks:
        raise ValueError("smoke argmax requires a nonempty frozen peak cache")
    for peak in peaks:
        if type(peak) is not dict or set(peak) != _PEAK_FIELDS:
            raise ValueError("frozen detector peak fields drifted")
        probability = float(peak["center_probability"])
        center = float(peak["center_seconds"])
        fraction = float(peak["duration_fraction"])
        if (
            not all(math.isfinite(item) for item in (probability, center, fraction))
            or not 0.0 <= probability <= 1.0
            or not 0.0 <= center <= duration
            or not 0.0 <= fraction <= 1.0
        ):
            raise ValueError("frozen detector peak value is invalid")
    return data


def _select_smoke_peak(
    prediction: Mapping[str, object], contract: Mapping[str, object]
) -> tuple[dict[str, float], dict[str, Any]]:
    peaks = prediction["peaks"]
    maximum = max(float(peak["center_probability"]) for peak in peaks)
    selected_rows = [
        peak for peak in peaks if float(peak["center_probability"]) == maximum
    ]
    if len(selected_rows) != 1:
        raise ValueError("frozen post-NMS peak argmax is not unique")
    peak = selected_rows[0]
    recording_duration = float(prediction["recording_duration_seconds"])
    maximum_duration = float(contract["maximum_decoded_duration_seconds"])
    decoded_duration = min(
        maximum_duration,
        max(1.0 / 256.0, float(peak["duration_fraction"]) * maximum_duration),
    )
    center = float(peak["center_seconds"])
    start = max(0.0, center - decoded_duration / 2.0)
    stop = min(recording_duration, center + decoded_duration / 2.0)
    formal_threshold = float(contract["formal_source_dev_diagnostic_center_threshold"])
    formal_candidates = [
        row
        for row in peaks
        if float(row["center_probability"]) >= formal_threshold
    ]
    selected = {
        "center_probability": float(peak["center_probability"]),
        "center_seconds": center,
        "duration_fraction": float(peak["duration_fraction"]),
        "decoded_duration_seconds": decoded_duration,
        "decoded_envelope_start_seconds": start,
        "decoded_envelope_stop_seconds": stop,
    }
    distinction = {
        "formal_source_dev_diagnostic_center_threshold": formal_threshold,
        "formal_candidate_count": len(formal_candidates),
        "formal_zero_candidate": not formal_candidates,
        "smoke_argmax_survives_formal_threshold": maximum >= formal_threshold,
        "smoke_argmax_is_formal_alarm_substitute": False,
    }
    return selected, distinction


def materialize_real_edf_smoke_entry_v1(
    *,
    entry: Mapping[str, object],
    detector_contract: Mapping[str, object],
    manifest_sha256: str,
    tusz_root: str | Path,
    workspace_root: str | Path = ROOT,
) -> dict[str, Any]:
    contract = _validate_detector_contract(detector_contract)
    root = Path(workspace_root).resolve(strict=True)
    eeg_root = Path(tusz_root).resolve(strict=True)
    prediction_relative, prediction_path = _safe_relative(
        root, entry["prediction_relative_path"], suffix=".gz"
    )
    if _file_sha256(prediction_path) != entry["prediction_file_sha256"]:
        raise ValueError("frozen detector peak gzip SHA-256 drifted")
    edf_relative, edf_path = _safe_relative(
        eeg_root, entry["relative_edf_path"], suffix=".edf"
    )
    prediction = _load_prediction(
        prediction_path, entry=entry, contract=contract
    )
    selected_peak, formal_distinction = _select_smoke_peak(prediction, contract)
    anchor = float(selected_peak["center_seconds"])
    event_id = f"{entry['recording_id']}__frozen_postnms_argmax_smoke"
    envelope = (
        (
            float(selected_peak["decoded_envelope_start_seconds"]),
            float(selected_peak["decoded_envelope_stop_seconds"]),
        ),
    )

    with DirectObservedCommon17EDFQueryReader(
        edf_path,
        expected_edf_sha256=str(entry["edf_sha256"]),
    ) as reader:
        if (
            reader.sampling_rate_hz
            != float(entry["expected_source_sampling_rate_hz"])
            or reader.recording_sample_count
            != int(entry["expected_recording_sample_count"])
            or reader.fz_pz_observation_state
            != entry["expected_FZ_PZ_observation_state"]
        ):
            raise ValueError("real EDF acquisition contract differs from frozen entry")
        if abs(reader.recording_duration_seconds - float(prediction["recording_duration_seconds"])) > 1.0e-9:
            raise ValueError("detector cache and EDF recording durations differ")
        state = _materialize_common17_adaptive_support_v2_with_state(
            event_id=event_id,
            recording_id=str(entry["recording_id"]),
            navigation_anchor_recording_seconds=anchor,
            sampling_rate_hz=reader.sampling_rate_hz,
            recording_sample_count=reader.recording_sample_count,
            query_reader=reader,
            frozen_detector_candidate_envelopes_recording_seconds=envelope,
        )
        base_reader_receipt = reader.receipt()
    adaptive_receipt = validate_common17_adaptive_support_v2(state.receipt)
    base_freeze = {
        "adaptive_support_receipt_sha256": adaptive_receipt["receipt_sha256"],
        "reader_receipt_sha256": _canonical_sha256(base_reader_receipt),
        "frozen_before_jitter_shadow_query": True,
    }
    base_freeze["base_freeze_sha256"] = _canonical_sha256(base_freeze)

    rate = float(entry["expected_source_sampling_rate_hz"])
    anchor_sample = int(round(anchor * rate))
    shadow_start = anchor_sample - int(round(60.0 * rate))
    shadow_stop = anchor_sample + int(round(60.0 * rate))
    if state.baseline is None:
        jitter: dict[str, Any] = {
            "status": "not_evaluable_background_censored",
            "reason": "adaptive_v2_remote_baseline_not_qualified",
            "fixed_shadow_query_performed": False,
        }
        shadow_reader_receipt = None
    elif shadow_start < 0 or shadow_stop > int(entry["expected_recording_sample_count"]):
        jitter = {
            "status": "not_evaluable_record_geometry",
            "reason": "exact_centered_120s_shadow_outside_recording",
            "fixed_shadow_query_performed": False,
        }
        shadow_reader_receipt = None
    else:
        with DirectObservedCommon17EDFQueryReader(
            edf_path,
            expected_edf_sha256=str(entry["edf_sha256"]),
        ) as shadow_reader:
            shadow = shadow_reader(shadow_start, shadow_stop)
            shadow_reader_receipt = shadow_reader.receipt()
        jitter = evaluate_common17_anchor_jitter_shadow_v2(
            event_id=event_id,
            recording_id=str(entry["recording_id"]),
            baseline=state.baseline,
            shadow_signal_volts=shadow.signal_volts,
            shadow_qc=shadow.valid_sample_mask,
            shadow_start_sample=shadow_start,
            navigation_anchor_sample=anchor_sample,
            rate=rate,
            policy=DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY,
            morphology_cache=state.morphology_cache,
        )
        validate_common17_anchor_jitter_shadow_v2(jitter)

    base_candidate_payload = adaptive_receipt["final_evidence"].get(
        "onset_candidate"
    )
    base_candidate = (
        float(base_candidate_payload["recording_seconds"])
        if base_candidate_payload is not None
        else None
    )
    base_ranked = (
        [
            str(row["channel"])
            for row in adaptive_receipt["final_evidence"].get(
                "per_channel_evidence", []
            )
        ]
        if base_candidate is not None
        else []
    )
    if jitter.get("schema_version") is not None:
        zero_replay = next(
            row for row in jitter["replays"] if row["anchor_offset_seconds"] == 0.0
        )
        shadow_candidate = zero_replay["candidate_recording_seconds"]
        shadow_top3 = list(zero_replay["top3_channels"])
        base_top3 = base_ranked[:3]
        union = set(base_top3) | set(shadow_top3)
        top3_jaccard = (
            len(set(base_top3) & set(shadow_top3)) / len(union) if union else 0.0
        )
        candidate_absolute_difference = (
            abs(float(base_candidate) - float(shadow_candidate))
            if base_candidate is not None and shadow_candidate is not None
            else None
        )
        candidate_limit = float(
            jitter["summary"]["engineering_gate_thresholds"][
                "candidate_recording_time_range_maximum_seconds"
            ]
        )
        top3_limit = float(
            jitter["summary"]["engineering_gate_thresholds"][
                "top3_pairwise_jaccard_minimum"
            ]
        )
        base_shadow_pass = bool(
            candidate_absolute_difference is not None
            and candidate_absolute_difference <= candidate_limit
            and bool(base_ranked)
            and base_ranked[0] == zero_replay["top1_channel"]
            and top3_jaccard >= top3_limit
        )
        base_shadow_comparison: dict[str, Any] = {
            "status": "evaluated",
            "base_adaptive_candidate_recording_seconds": base_candidate,
            "fixed_shadow_zero_offset_candidate_recording_seconds": shadow_candidate,
            "candidate_absolute_difference_seconds": candidate_absolute_difference,
            "base_adaptive_top1_channel": base_ranked[0] if base_ranked else None,
            "fixed_shadow_zero_offset_top1_channel": zero_replay["top1_channel"],
            "top1_exact_agreement": bool(base_ranked)
            and base_ranked[0] == zero_replay["top1_channel"],
            "base_adaptive_top3_channels": base_top3,
            "fixed_shadow_zero_offset_top3_channels": shadow_top3,
            "top3_jaccard": top3_jaccard,
            "agreement_thresholds": {
                "candidate_absolute_difference_maximum_seconds": candidate_limit,
                "top1_exact_agreement_required": True,
                "top3_jaccard_minimum": top3_limit,
            },
            "base_vs_fixed_shadow_agreement_pass": base_shadow_pass,
            "overall_cross_jitter_and_base_shadow_pass": bool(
                jitter["summary"]["descriptive_engineering_gate_pass"]
                and base_shadow_pass
            ),
            "promotion_or_efficacy_permission_granted": False,
        }
    else:
        base_shadow_comparison = {
            "status": "not_evaluable",
            "reason": jitter["reason"],
            "base_vs_fixed_shadow_agreement_pass": False,
            "overall_cross_jitter_and_base_shadow_pass": False,
            "promotion_or_efficacy_permission_granted": False,
        }

    body: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": METHOD_ID,
        "manifest_sha256": _sha256(manifest_sha256, "manifest SHA"),
        "recording_id": str(entry["recording_id"]),
        "event_id": event_id,
        "source_bindings": {
            "official_split": "dev",
            "relative_edf_path": edf_relative,
            "edf_sha256": str(entry["edf_sha256"]),
            "prediction_relative_path": prediction_relative,
            "prediction_file_sha256": str(entry["prediction_file_sha256"]),
            "analysis_identity_id": str(entry["analysis_identity_id"]),
            "runtime_reference_or_annotation_files_opened": [],
        },
        "detector_cache_binding": {
            "prediction_schema": prediction["schema_version"],
            "checkpoint_file_sha256": prediction["checkpoint_file_sha256"],
            "checkpoint_global_step": prediction["checkpoint_global_step"],
            "cached_peak_count": len(prediction["peaks"]),
            "cached_peak_floor_threshold": prediction["minimum_peak_threshold"],
            "post_NMS_minimum_peak_distance_seconds": prediction[
                "minimum_peak_distance_seconds"
            ],
            "post_NMS_smoothing_sigma_samples": prediction[
                "smoothing_sigma_samples"
            ],
            "decoder_source_file_sha256": contract[
                "decoder_source_file_sha256"
            ],
        },
        "formal_operating_point_distinction": formal_distinction,
        "smoke_navigation": {
            "selection_rule": contract["smoke_navigation_rule"],
            "selection_is_formal_alarm_operating_point": False,
            "selected_peak": selected_peak,
            "navigation_anchor_is_clinical_onset": False,
            "decoded_envelope_used_for_baseline_exclusion_only": True,
        },
        "adaptive_v2_base_freeze": base_freeze,
        "adaptive_v2_reader_receipt": base_reader_receipt,
        "adaptive_v2_receipt": adaptive_receipt,
        "fixed_shadow_reader_receipt": shadow_reader_receipt,
        "anchor_jitter_fixed_shadow_receipt": jitter,
        "base_vs_fixed_shadow_zero_offset": base_shadow_comparison,
        "scope_receipt": deepcopy(_SCOPE),
        "claim_limits": deepcopy(_CLAIMS),
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return validate_real_edf_smoke_event_v1(body)


def validate_real_edf_smoke_event_v1(payload: object) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("adaptive-v2 real-EDF smoke event must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "receipt_sha256",
        "method_id",
        "manifest_sha256",
        "recording_id",
        "event_id",
        "source_bindings",
        "detector_cache_binding",
        "formal_operating_point_distinction",
        "smoke_navigation",
        "adaptive_v2_base_freeze",
        "adaptive_v2_reader_receipt",
        "adaptive_v2_receipt",
        "fixed_shadow_reader_receipt",
        "anchor_jitter_fixed_shadow_receipt",
        "base_vs_fixed_shadow_zero_offset",
        "scope_receipt",
        "claim_limits",
    }
    if set(data) != required:
        raise ValueError("adaptive-v2 real-EDF smoke event fields drifted")
    if data["schema_version"] != EVENT_SCHEMA or data["method_id"] != METHOD_ID:
        raise ValueError("adaptive-v2 real-EDF smoke event method drifted")
    _identifier(data["recording_id"], "recording_id")
    _identifier(data["event_id"], "event_id")
    _sha256(data["manifest_sha256"], "manifest SHA")
    adaptive = validate_common17_adaptive_support_v2(data["adaptive_v2_receipt"])
    source = data["source_bindings"]
    if (
        source.get("official_split") != "dev"
        or source.get("runtime_reference_or_annotation_files_opened") != []
        or source.get("edf_sha256")
        != str(source.get("analysis_identity_id", "")).removeprefix("TUSZANALYSIS-")
    ):
        raise ValueError("adaptive-v2 smoke runtime source firewall drifted")
    freeze = data["adaptive_v2_base_freeze"]
    if (
        freeze.get("adaptive_support_receipt_sha256") != adaptive["receipt_sha256"]
        or freeze.get("reader_receipt_sha256")
        != _canonical_sha256(data["adaptive_v2_reader_receipt"])
        or freeze.get("frozen_before_jitter_shadow_query") is not True
        or freeze.get("base_freeze_sha256")
        != _canonical_sha256(
            {key: value for key, value in freeze.items() if key != "base_freeze_sha256"}
        )
    ):
        raise ValueError("adaptive-v2 base freeze binding drifted")
    formal = data["formal_operating_point_distinction"]
    smoke = data["smoke_navigation"]
    if (
        formal.get("smoke_argmax_is_formal_alarm_substitute") is not False
        or smoke.get("selection_is_formal_alarm_operating_point") is not False
        or smoke.get("navigation_anchor_is_clinical_onset") is not False
        or smoke.get("decoded_envelope_used_for_baseline_exclusion_only") is not True
    ):
        raise ValueError("adaptive-v2 smoke/formal OP distinction drifted")
    selected_peak = smoke["selected_peak"]
    adaptive_rate = float(adaptive["acquisition"]["sampling_rate_hz"])
    expected_envelope = [
        [
            round(
                round(
                    float(selected_peak["decoded_envelope_start_seconds"])
                    * adaptive_rate
                )
                / adaptive_rate,
                6,
            ),
            round(
                round(
                    float(selected_peak["decoded_envelope_stop_seconds"])
                    * adaptive_rate
                )
                / adaptive_rate,
                6,
            ),
        ]
    ]
    if (
        adaptive["navigation_anchor_recording_seconds"]
        != round(
            round(float(selected_peak["center_seconds"]) * adaptive_rate)
            / adaptive_rate,
            6,
        )
        or adaptive["frozen_detector_candidate_envelopes"][
            "intervals_recording_seconds"
        ]
        != expected_envelope
        or adaptive["frozen_detector_candidate_envelopes"]["source"]
        != "explicit_frozen_detector_provider"
    ):
        raise ValueError("adaptive-v2 smoke peak/anchor/envelope binding drifted")
    if bool(formal.get("formal_zero_candidate")) != (
        int(formal.get("formal_candidate_count", -1)) == 0
    ):
        raise ValueError("adaptive-v2 formal zero-candidate distinction drifted")
    jitter = data["anchor_jitter_fixed_shadow_receipt"]
    comparison = data["base_vs_fixed_shadow_zero_offset"]
    if jitter.get("schema_version") is not None:
        validate_common17_anchor_jitter_shadow_v2(jitter)
        if data["fixed_shadow_reader_receipt"] is None:
            raise ValueError("adaptive-v2 jitter lacks its fixed-shadow reader receipt")
        if jitter["frozen_baseline_receipt_sha256"] != _canonical_sha256(
            adaptive["remote_baseline_bank"]
        ):
            raise ValueError("adaptive-v2 jitter baseline binding drifted")
        shadow_reader = data["fixed_shadow_reader_receipt"]
        if (
            shadow_reader.get("queried_intervals_samples")
            != [jitter["fixed_shadow"]["interval_samples"]]
            or shadow_reader.get("FZ_PZ_samples_read") is not False
            or shadow_reader.get("non_common17_signal_samples_read") is not False
            or shadow_reader.get("EDF_annotation_API_called") is not False
            or shadow_reader.get("patient_header_API_called") is not False
            or shadow_reader.get("target_sidecar_opened") is not False
        ):
            raise ValueError("adaptive-v2 fixed-shadow reader firewall drifted")
        zero_replay = next(
            row for row in jitter["replays"] if row["anchor_offset_seconds"] == 0.0
        )
        if (
            comparison.get("status") != "evaluated"
            or comparison.get("fixed_shadow_zero_offset_candidate_recording_seconds")
            != zero_replay["candidate_recording_seconds"]
            or comparison.get("fixed_shadow_zero_offset_top1_channel")
            != zero_replay["top1_channel"]
            or comparison.get("promotion_or_efficacy_permission_granted") is not False
        ):
            raise ValueError("adaptive-v2 base/fixed-shadow comparison drifted")
    elif jitter.get("status") not in {
        "not_evaluable_background_censored",
        "not_evaluable_record_geometry",
    }:
        raise ValueError("adaptive-v2 jitter typed censor drifted")
    elif (
        comparison.get("status") != "not_evaluable"
        or comparison.get("overall_cross_jitter_and_base_shadow_pass") is not False
        or comparison.get("promotion_or_efficacy_permission_granted") is not False
    ):
        raise ValueError("adaptive-v2 typed-censor comparison drifted")
    if data["scope_receipt"] != _SCOPE or data["claim_limits"] != _CLAIMS:
        raise ValueError("adaptive-v2 real-EDF smoke claim boundary drifted")
    expected = _canonical_sha256(
        {key: value for key, value in data.items() if key != "receipt_sha256"}
    )
    if data["receipt_sha256"] != expected:
        raise ValueError("adaptive-v2 real-EDF smoke event content hash mismatch")
    return data


def summarize_real_edf_smoke_v1(
    *, manifest_sha256: str, events: Sequence[Mapping[str, object]]
) -> dict[str, Any]:
    rows = [validate_real_edf_smoke_event_v1(event) for event in events]
    if len(rows) != 2 or len({row["recording_id"] for row in rows}) != 2:
        raise ValueError("adaptive-v2 smoke cohort does not contain two unique records")
    geometries: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for row in rows:
        support = row["adaptive_v2_receipt"]["adaptive_analysis_support"]
        geometry = json.dumps(
            [
                support["left_extent_seconds"],
                support["right_extent_seconds"],
                support["left_terminal_reason"],
                support["right_terminal_reason"],
            ],
            separators=(",", ":"),
        )
        geometries[geometry] = geometries.get(geometry, 0) + 1
        jitter = row["anchor_jitter_fixed_shadow_receipt"]
        summaries.append(
            {
                "recording_id": row["recording_id"],
                "event_receipt_sha256": row["receipt_sha256"],
                "formal_candidate_count": row[
                    "formal_operating_point_distinction"
                ]["formal_candidate_count"],
                "formal_zero_candidate": row[
                    "formal_operating_point_distinction"
                ]["formal_zero_candidate"],
                "smoke_argmax_survives_formal_threshold": row[
                    "formal_operating_point_distinction"
                ]["smoke_argmax_survives_formal_threshold"],
                "navigation_anchor_recording_seconds": row["smoke_navigation"][
                    "selected_peak"
                ]["center_seconds"],
                "background_status": row["adaptive_v2_receipt"][
                    "remote_baseline_bank"
                ]["status"],
                "support_geometry": json.loads(geometry),
                "jitter_status": (
                    "evaluated_fixed_120s_shadow"
                    if jitter.get("schema_version") is not None
                    else jitter["status"]
                ),
                "jitter_descriptive_engineering_gate_pass": (
                    jitter["summary"]["descriptive_engineering_gate_pass"]
                    if jitter.get("schema_version") is not None
                    else False
                ),
                "base_vs_fixed_shadow_agreement_pass": row[
                    "base_vs_fixed_shadow_zero_offset"
                ]["base_vs_fixed_shadow_agreement_pass"],
                "overall_cross_jitter_and_base_shadow_pass": row[
                    "base_vs_fixed_shadow_zero_offset"
                ]["overall_cross_jitter_and_base_shadow_pass"],
            }
        )
    maximum_geometry_fraction = max(geometries.values()) / len(rows)
    body: dict[str, Any] = {
        "schema_version": COHORT_SCHEMA,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": METHOD_ID,
        "manifest_sha256": _sha256(manifest_sha256, "manifest SHA"),
        "record_count": len(rows),
        "records": summaries,
        "descriptive_geometry": {
            "unique_geometry_count": len(geometries),
            "maximum_single_geometry_fraction": maximum_geometry_fraction,
            "two_record_smoke_is_cohort_degeneracy_gate": False,
        },
        "scope_receipt": deepcopy(_SCOPE),
        "claim_limits": deepcopy(_CLAIMS),
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return body


__all__ = [
    "COHORT_SCHEMA",
    "EVENT_SCHEMA",
    "MANIFEST_SCHEMA",
    "METHOD_ID",
    "load_real_edf_smoke_manifest_v1",
    "materialize_real_edf_smoke_entry_v1",
    "summarize_real_edf_smoke_v1",
    "validate_real_edf_smoke_event_v1",
    "validate_real_edf_smoke_manifest_v1",
]
