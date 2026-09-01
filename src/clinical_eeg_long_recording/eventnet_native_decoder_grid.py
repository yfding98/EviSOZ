"""Provider-native EventNet decoder grid over frozen raw prediction tensors.

Unlike the provider-neutral one-second hysteresis decoder, EventNet predicts a
sample-level event-center probability together with a duration fraction.  Its
native calibration space must therefore replay the direct-event decoder:
per-tile smoothing, peak selection, duration conversion, record clipping and
cross-tile interval union.  This module materializes that grid without any
reference, annotation, spreadsheet, clinical-text, private-data or
source-evaluation input.

The output is a prediction-only gzip JSONL inventory.  Raw peaks retain tile
and sample provenance; merged alarms never replace that raw ranking roster.
Source-development references may be joined only by the separate calibration
module after this complete bundle validates and freezes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .eventnet_full_record_adapter import (
    EVENTNET_PROVIDER_ID,
    EVENTNET_SAMPLING_RATE_HZ,
)
from .eventnet_raw_prediction_bundle import (
    ValidatedEventNetRawPredictionBundle,
    revalidate_eventnet_raw_prediction_bundle_without_references,
)
from .tusz_complete_detector_roster_v1 import TUSZ_V203_EXPECTED_INVENTORY


EVENTNET_DECODER_GRID_SCHEMA_VERSION = "eventnet_native_decoder_grid_v1"
EVENTNET_DECODER_GRID_METHOD_ID = (
    "eventnet_per_tile_center_duration_reference_free_decoder_grid_v1"
)
EVENTNET_DECODER_GRID_ROW_SCHEMA_VERSION = (
    "eventnet_native_decoder_grid_prediction_row_v1"
)
EVENTNET_DECODER_GRID_BUNDLE_SCHEMA_VERSION = (
    "eventnet_native_decoder_grid_prediction_bundle_v1"
)
EVENTNET_DECODER_GRID_FILENAME = "prediction_grid.jsonl.gz"
EVENTNET_DECODER_GRID_RECEIPT_FILENAME = "grid_receipt.json"
EVENTNET_RAW_VALIDATION_RECEIPT_FILENAME = "raw_bundle_validation_receipt.json"
EVENTNET_TUSZ_ANALYSIS_PROJECTION_BINDING_SCHEMA_VERSION = (
    "eventnet_tusz_analysis_projection_source_dev_binding_v1"
)

_POLICY_FIELDS = {
    "center_smoothing_sigma_samples",
    "center_threshold",
    "minimum_peak_distance_seconds",
    "duration_multiplier_seconds",
    "minimum_decoded_duration_seconds",
    "maximum_decoded_duration_seconds",
    "cross_tile_merge_gap_seconds",
    "maximum_alarms_per_record",
}
_GRID_FIELDS = {
    "schema_version",
    "grid_id",
    "method_id",
    "provider_id",
    "calibration_split",
    "candidate_policies",
    "selection_definition",
    "scope_receipt",
}
_SELECTION_FIELDS = {
    "minimum_pooled_event_sensitivity",
    "minimum_patient_macro_event_sensitivity",
    "false_alarm_budgets_per_24h",
    "onset_tie_tolerance_seconds",
}
_GRID_SCOPE = {
    "raw_prediction_bundle_must_validate_before_grid_decode": True,
    "reference_or_annotation_path_accepted": False,
    "source_dev_reference_used": False,
    "source_eval_used": False,
    "private_data_used": False,
    "edf_annotations_used": False,
    "excel_or_clinical_text_used": False,
    "prediction_only_grid": True,
    "production_or_sota_claim_authorized": False,
}
_REFERENCE_ACCESS = {
    "reference_path_argument_accepted": False,
    "reference_files_opened": 0,
    "edf_annotations_opened": 0,
    "excel_files_opened": 0,
    "clinical_text_opened": 0,
    "private_data_opened": 0,
    "source_eval_opened": 0,
}
_LEGACY_ROSTER_BINDING_FIELDS = {
    "roster_id",
    "roster_receipt_sha256",
    "source_manifest_file_sha256",
    "inventory_scope",
    "complete_split_inventory_verified",
    "source_dev_recording_roster_sha256",
}
_ANALYSIS_PROJECTION_BINDING_FIELDS = {
    "binding_schema_version",
    "analysis_projection_schema_version",
    "analysis_projection_id",
    "analysis_projection_receipt_sha256",
    "analysis_projection_file_sha256",
    "source_roster_binding",
    "source_dev_split_summary",
    "source_dev_recording_count",
    "source_dev_recording_roster_sha256",
    "inventory_scope",
    "completeness_receipt",
    "complete_split_inventory_verified",
}
_TUSZ_SOURCE_ROSTER_BINDING_FIELDS = {
    "source_schema_version",
    "source_roster_id",
    "source_roster_receipt_sha256",
    "source_release_id",
    "source_records_payload_sha256",
    "source_equivalence_class_roster_sha256",
    "source_audit_recording_count",
    "source_equivalence_class_count",
    "source_analysis_eligible_class_count",
    "source_same_patient_alias_excluded_path_count",
    "source_quarantined_path_count",
    "source_split_accounting_sha256",
}
_TUSZ_SOURCE_DEV_SUMMARY_FIELDS = {
    "official_split",
    "audit_official_path_count",
    "analysis_identity_count",
    "analysis_patient_alias_count",
    "same_patient_alias_excluded_path_count",
    "quarantined_path_count",
    "analysis_identity_roster_sha256",
    "analysis_patient_alias_roster_sha256",
    "path_count_closure_verified",
}
_ANALYSIS_PROJECTION_COMPLETENESS_FIELDS = {
    "analysis_projection_schema_validated",
    "projection_receipt_matches_stage_p_input",
    "source_roster_schema_v2_bound",
    "source_release_matches_tusz_v203_expected_inventory",
    "source_global_audit_recording_count_matches_tusz_v203",
    "source_dev_audit_recording_count_matches_tusz_v203",
    "source_dev_patient_alias_count_matches_tusz_v203",
    "source_dev_analysis_identity_count_matches_tusz_v203",
    "source_dev_has_no_projection_exclusions_or_quarantine",
    "source_dev_path_count_closure_verified",
    "source_dev_recording_count_matches_analysis_identity_count",
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eventnet_native_decoder_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _finite(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{context} must be finite and >= {minimum}")
    return result


def _unit_interval(value: object, context: str, *, open_zero: bool = False) -> float:
    result = _finite(value, context)
    if result > 1 or (open_zero and result <= 0):
        raise ValueError(f"{context} must lie in {'(0,1]' if open_zero else '[0,1]'}")
    return result


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{context} must be an integer >= {minimum}")
    return value


def validate_eventnet_source_dev_roster_binding(
    value: object,
    *,
    expected_recording_roster_sha256: str,
    expected_analysis_projection_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate either the legacy roster or complete v2 identity projection."""

    if type(value) is not dict:
        raise TypeError("EventNet source-dev roster binding must be an object")
    binding = deepcopy(value)
    if set(binding) == _LEGACY_ROSTER_BINDING_FIELDS:
        if expected_analysis_projection_receipt_sha256 is not None:
            raise ValueError(
                "legacy EventNet roster binding cannot satisfy frozen Stage-P "
                "analysis-projection lineage"
            )
        _identifier(binding["roster_id"], "EventNet roster ID")
        for field in (
            "roster_receipt_sha256",
            "source_manifest_file_sha256",
            "source_dev_recording_roster_sha256",
        ):
            if not _is_sha256(binding[field]):
                raise ValueError(f"EventNet roster binding {field} is invalid")
        _identifier(binding["inventory_scope"], "EventNet inventory scope")
        if type(binding["complete_split_inventory_verified"]) is not bool:
            raise TypeError("EventNet roster completeness flag must be boolean")
    elif set(binding) == _ANALYSIS_PROJECTION_BINDING_FIELDS:
        if (
            binding["binding_schema_version"]
            != EVENTNET_TUSZ_ANALYSIS_PROJECTION_BINDING_SCHEMA_VERSION
            or binding["analysis_projection_schema_version"]
            != "tusz_analysis_identity_projection_v2"
            or binding["inventory_scope"]
            != "complete_tusz_v2_source_dev_analysis_identity_projection"
        ):
            raise ValueError("EventNet analysis-projection binding schema drifted")
        _identifier(
            binding["analysis_projection_id"],
            "EventNet analysis projection ID",
        )
        for field in (
            "analysis_projection_receipt_sha256",
            "analysis_projection_file_sha256",
            "source_dev_recording_roster_sha256",
        ):
            if not _is_sha256(binding[field]):
                raise ValueError(
                    f"EventNet analysis-projection binding {field} is invalid"
                )
        source = binding["source_roster_binding"]
        if type(source) is not dict or set(source) != _TUSZ_SOURCE_ROSTER_BINDING_FIELDS:
            raise ValueError("EventNet bound TUSZ source-roster fields drifted")
        if source["source_schema_version"] != "tusz_complete_detector_roster_v2":
            raise ValueError("EventNet bound TUSZ source-roster schema drifted")
        _identifier(source["source_roster_id"], "bound TUSZ source-roster ID")
        _identifier(source["source_release_id"], "bound TUSZ source release ID")
        for field in (
            "source_roster_receipt_sha256",
            "source_records_payload_sha256",
            "source_equivalence_class_roster_sha256",
            "source_split_accounting_sha256",
        ):
            if not _is_sha256(source[field]):
                raise ValueError(f"EventNet bound TUSZ {field} is invalid")
        for field in (
            "source_audit_recording_count",
            "source_equivalence_class_count",
            "source_analysis_eligible_class_count",
            "source_same_patient_alias_excluded_path_count",
            "source_quarantined_path_count",
        ):
            _integer(source[field], f"EventNet bound TUSZ {field}")

        summary = binding["source_dev_split_summary"]
        if type(summary) is not dict or set(summary) != _TUSZ_SOURCE_DEV_SUMMARY_FIELDS:
            raise ValueError("EventNet bound source-dev summary fields drifted")
        if summary["official_split"] != "dev":
            raise ValueError("EventNet bound source-dev official split drifted")
        for field in (
            "audit_official_path_count",
            "analysis_identity_count",
            "analysis_patient_alias_count",
            "same_patient_alias_excluded_path_count",
            "quarantined_path_count",
        ):
            _integer(summary[field], f"EventNet bound source-dev {field}")
        for field in (
            "analysis_identity_roster_sha256",
            "analysis_patient_alias_roster_sha256",
        ):
            if not _is_sha256(summary[field]):
                raise ValueError(f"EventNet bound source-dev {field} is invalid")
        if (
            summary["path_count_closure_verified"] is not True
            or summary["audit_official_path_count"]
            != summary["analysis_identity_count"]
            + summary["same_patient_alias_excluded_path_count"]
            + summary["quarantined_path_count"]
        ):
            raise ValueError("EventNet bound source-dev path accounting drifted")
        count = _integer(
            binding["source_dev_recording_count"],
            "EventNet source-dev recording count",
            minimum=1,
        )
        if count != summary["analysis_identity_count"]:
            raise ValueError("EventNet source-dev projected count drifted")
        completeness = binding["completeness_receipt"]
        if (
            type(completeness) is not dict
            or set(completeness) != _ANALYSIS_PROJECTION_COMPLETENESS_FIELDS
            or any(type(item) is not bool for item in completeness.values())
        ):
            raise ValueError("EventNet projection completeness receipt drifted")
        if (
            not _is_sha256(expected_analysis_projection_receipt_sha256)
            or binding["analysis_projection_receipt_sha256"]
            != expected_analysis_projection_receipt_sha256
        ):
            raise ValueError(
                "EventNet analysis projection disagrees with Stage-P lineage"
            )
        expected_dev = TUSZ_V203_EXPECTED_INVENTORY["split_expectations"]["dev"]
        expected_completeness = {
            "analysis_projection_schema_validated": True,
            "projection_receipt_matches_stage_p_input": True,
            "source_roster_schema_v2_bound": True,
            "source_release_matches_tusz_v203_expected_inventory": (
                source["source_release_id"]
                == TUSZ_V203_EXPECTED_INVENTORY["release_id"]
            ),
            "source_global_audit_recording_count_matches_tusz_v203": (
                source["source_audit_recording_count"]
                == TUSZ_V203_EXPECTED_INVENTORY["total_recording_count"]
            ),
            "source_dev_audit_recording_count_matches_tusz_v203": (
                summary["audit_official_path_count"]
                == expected_dev["recording_count"]
            ),
            "source_dev_patient_alias_count_matches_tusz_v203": (
                summary["analysis_patient_alias_count"]
                == expected_dev["patient_count"]
            ),
            "source_dev_analysis_identity_count_matches_tusz_v203": (
                summary["analysis_identity_count"]
                == expected_dev["recording_count"]
            ),
            "source_dev_has_no_projection_exclusions_or_quarantine": (
                summary["same_patient_alias_excluded_path_count"] == 0
                and summary["quarantined_path_count"] == 0
            ),
            "source_dev_path_count_closure_verified": True,
            "source_dev_recording_count_matches_analysis_identity_count": (
                count == summary["analysis_identity_count"]
            ),
        }
        if completeness != expected_completeness:
            raise ValueError("EventNet projection completeness receipt is not replayable")
        derived_complete = all(expected_completeness.values())
        if (
            type(binding["complete_split_inventory_verified"]) is not bool
            or binding["complete_split_inventory_verified"] is not derived_complete
        ):
            raise ValueError("EventNet projection completeness claim is not derived")
    else:
        raise ValueError("EventNet source-dev roster binding fields drifted")
    if (
        not _is_sha256(expected_recording_roster_sha256)
        or binding["source_dev_recording_roster_sha256"]
        != expected_recording_roster_sha256
    ):
        raise ValueError("EventNet roster binding and raw-bundle inventory disagree")
    return binding


def _loads_json(payload: bytes | str, context: str) -> Any:
    def no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def no_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=no_duplicate,
            parse_constant=no_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error


def _regular_file(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return path


def _regular_directory(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{context} must be a regular non-symlink directory")
    return path


def _tensor_payload_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def validate_eventnet_native_decoder_policy(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _POLICY_FIELDS:
        raise ValueError("EventNet native decoder policy fields drifted")
    data = deepcopy(payload)
    sigma = _integer(
        data["center_smoothing_sigma_samples"],
        "EventNet smoothing sigma",
        minimum=1,
    )
    threshold = _unit_interval(
        data["center_threshold"], "EventNet center threshold", open_zero=True
    )
    distance = _finite(
        data["minimum_peak_distance_seconds"],
        "EventNet minimum peak distance",
        minimum=1 / EVENTNET_SAMPLING_RATE_HZ,
    )
    distance_samples = distance * EVENTNET_SAMPLING_RATE_HZ
    if abs(distance_samples - round(distance_samples)) > 1e-9:
        raise ValueError("EventNet minimum peak distance must align to 256 Hz")
    duration_multiplier = _finite(
        data["duration_multiplier_seconds"],
        "EventNet duration multiplier",
        minimum=1 / EVENTNET_SAMPLING_RATE_HZ,
    )
    minimum_duration = _finite(
        data["minimum_decoded_duration_seconds"],
        "EventNet minimum decoded duration",
        minimum=1 / EVENTNET_SAMPLING_RATE_HZ,
    )
    maximum_duration = _finite(
        data["maximum_decoded_duration_seconds"],
        "EventNet maximum decoded duration",
        minimum=minimum_duration,
    )
    if maximum_duration < minimum_duration:
        raise ValueError("EventNet maximum duration is below minimum duration")
    merge_gap = _finite(data["cross_tile_merge_gap_seconds"], "EventNet merge gap")
    maximum_alarms = data["maximum_alarms_per_record"]
    if maximum_alarms is not None:
        maximum_alarms = _integer(
            maximum_alarms, "EventNet maximum alarms per record", minimum=1
        )
    return {
        "center_smoothing_sigma_samples": sigma,
        "center_threshold": threshold,
        "minimum_peak_distance_seconds": distance,
        "duration_multiplier_seconds": duration_multiplier,
        "minimum_decoded_duration_seconds": minimum_duration,
        "maximum_decoded_duration_seconds": maximum_duration,
        "cross_tile_merge_gap_seconds": merge_gap,
        "maximum_alarms_per_record": maximum_alarms,
    }


def _normalize_policy_rows(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
    ):
        raise TypeError("EventNet decoder grid must contain candidate policies")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        policy = validate_eventnet_native_decoder_policy(dict(raw))
        policy_sha256 = _canonical_sha256(policy)
        if policy_sha256 in seen:
            raise ValueError("EventNet decoder grid contains duplicate policies")
        seen.add(policy_sha256)
        rows.append(
            {
                "policy_id": "EVNPOL-" + policy_sha256[:20],
                "policy_sha256": policy_sha256,
                "decoder_policy": policy,
            }
        )
    rows.sort(key=lambda row: row["policy_id"])
    return rows


def validate_eventnet_native_decoder_grid(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _GRID_FIELDS:
        raise ValueError("EventNet decoder-grid definition fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != EVENTNET_DECODER_GRID_SCHEMA_VERSION
        or data["method_id"] != EVENTNET_DECODER_GRID_METHOD_ID
        or data["provider_id"] != EVENTNET_PROVIDER_ID
        or data["calibration_split"] != "source_dev"
        or data["scope_receipt"] != _GRID_SCOPE
    ):
        raise ValueError("EventNet decoder-grid identity, split, or scope drifted")
    policies = _normalize_policy_rows(data["candidate_policies"])
    # Definitions store only the normalized policies, in canonical policy-ID order.
    if data["candidate_policies"] != [row["decoder_policy"] for row in policies]:
        raise ValueError("EventNet decoder-grid policies are not canonical")
    selection = data["selection_definition"]
    if type(selection) is not dict or set(selection) != _SELECTION_FIELDS:
        raise ValueError("EventNet decoder-grid selection definition drifted")
    _unit_interval(
        selection["minimum_pooled_event_sensitivity"],
        "minimum pooled event sensitivity",
        open_zero=True,
    )
    _unit_interval(
        selection["minimum_patient_macro_event_sensitivity"],
        "minimum patient-macro sensitivity",
        open_zero=True,
    )
    budgets = selection["false_alarm_budgets_per_24h"]
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("EventNet decoder-grid FA budgets are empty")
    normalized_budgets = [
        _finite(value, "false-alarm budget", minimum=1e-12) for value in budgets
    ]
    if normalized_budgets != sorted(set(normalized_budgets)):
        raise ValueError("EventNet decoder-grid FA budgets are not canonical")
    _finite(
        selection["onset_tie_tolerance_seconds"],
        "onset tie tolerance",
        minimum=1e-12,
    )
    digest = deepcopy(data)
    digest["grid_id"] = "EVENTNET-DECODER-GRID-PENDING"
    if data["grid_id"] != "EVNGRID-" + _canonical_sha256(digest)[:24]:
        raise ValueError("EventNet decoder-grid ID is not content-bound")
    return data


def eventnet_decoder_policy_rows(
    grid: Mapping[str, Any],
) -> list[dict[str, Any]]:
    value = validate_eventnet_native_decoder_grid(dict(grid))
    return _normalize_policy_rows(value["candidate_policies"])


def smooth_eventnet_center_per_tile(
    center_probability: np.ndarray,
    tile_receipts: Sequence[Mapping[str, Any]],
    *,
    sigma_samples: int,
) -> np.ndarray:
    center = np.asarray(center_probability)
    if center.dtype != np.dtype("<f4") or center.ndim != 1 or center.size < 1:
        raise ValueError("EventNet center tensor must be one-dimensional float32")
    sigma = _integer(sigma_samples, "EventNet smoothing sigma", minimum=1)
    smoothed = np.empty_like(center, dtype=np.float32)
    cursor = 0
    for index, tile in enumerate(tile_receipts):
        start = _integer(tile.get("target_start_sample"), "tile start")
        stop = _integer(
            tile.get("target_stop_sample_exclusive"), "tile stop", minimum=1
        )
        if start != cursor or stop <= start or stop > center.size:
            raise ValueError("EventNet target tiles are not complete and contiguous")
        if tile.get("tile_id") != f"EVNTILE-{index:06d}":
            raise ValueError("EventNet target tile ID drifted")
        smoothed[start:stop] = gaussian_filter1d(center[start:stop], sigma).astype(
            np.float32, copy=False
        )
        cursor = stop
    if cursor != center.size:
        raise ValueError("EventNet target tiles do not cover the tensor")
    return smoothed


def decode_eventnet_native_policy(
    *,
    center_probability: np.ndarray,
    duration_fraction: np.ndarray,
    tile_receipts: Sequence[Mapping[str, Any]],
    recording_duration_seconds: float,
    policy: Mapping[str, Any],
    smoothed_center_probability: np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay one direct-event decoder policy on frozen sample-level outputs."""

    normalized = validate_eventnet_native_decoder_policy(dict(policy))
    center = np.asarray(center_probability)
    duration = np.asarray(duration_fraction)
    if (
        center.dtype != np.dtype("<f4")
        or duration.dtype != np.dtype("<f4")
        or center.ndim != 1
        or duration.shape != center.shape
        or center.size < 1
        or not np.isfinite(center).all()
        or not np.isfinite(duration).all()
        or np.any(center < 0)
        or np.any(center > 1)
        or np.any(duration < 0)
        or np.any(duration > 1)
    ):
        raise ValueError("EventNet native decoder received invalid frozen tensors")
    recording_duration = _finite(
        recording_duration_seconds, "recording duration", minimum=1e-12
    )
    if center.size != int(round(recording_duration * EVENTNET_SAMPLING_RATE_HZ)):
        raise ValueError("EventNet native decoder tensor clock drifted")
    if smoothed_center_probability is None:
        smoothed = smooth_eventnet_center_per_tile(
            center,
            tile_receipts,
            sigma_samples=normalized["center_smoothing_sigma_samples"],
        )
    else:
        smoothed = np.asarray(smoothed_center_probability)
        if (
            smoothed.dtype != np.dtype("<f4")
            or smoothed.shape != center.shape
            or not np.isfinite(smoothed).all()
            or np.any(smoothed < 0)
            or np.any(smoothed > 1)
        ):
            raise ValueError("EventNet cached smoothed tensor is invalid")

    policy_sha256 = _canonical_sha256(normalized)
    policy_id = "EVNPOL-" + policy_sha256[:20]
    distance_samples = int(
        round(normalized["minimum_peak_distance_seconds"] * EVENTNET_SAMPLING_RATE_HZ)
    )
    proposals: list[dict[str, Any]] = []
    for tile in tile_receipts:
        start = int(tile["target_start_sample"])
        stop = int(tile["target_stop_sample_exclusive"])
        peaks, properties = find_peaks(
            smoothed[start:stop],
            height=normalized["center_threshold"],
            distance=distance_samples,
        )
        for local_peak, probability in zip(peaks, properties["peak_heights"]):
            sample = start + int(local_peak)
            raw_duration = (
                float(duration[sample]) * normalized["duration_multiplier_seconds"]
            )
            decoded_duration = min(
                normalized["maximum_decoded_duration_seconds"],
                max(normalized["minimum_decoded_duration_seconds"], raw_duration),
            )
            center_seconds = sample / EVENTNET_SAMPLING_RATE_HZ
            event_start = max(0.0, center_seconds - 0.5 * decoded_duration)
            event_stop = min(
                recording_duration, center_seconds + 0.5 * decoded_duration
            )
            if event_stop <= event_start:
                continue
            proposals.append(
                {
                    "proposal_id": f"EVNPEAK-{len(proposals):07d}",
                    "tile_id": tile["tile_id"],
                    "peak_sample": sample,
                    "center_offset_seconds": center_seconds,
                    "smoothed_center_probability": float(probability),
                    "duration_fraction": float(duration[sample]),
                    "raw_duration_seconds": raw_duration,
                    "decoded_duration_seconds_before_record_clipping": decoded_duration,
                    "start_offset_seconds": event_start,
                    "stop_offset_seconds": event_stop,
                }
            )
    maximum_alarms = normalized["maximum_alarms_per_record"]
    top_k_applied = False
    if maximum_alarms is not None and len(proposals) > maximum_alarms:
        selected_ids = {
            row["proposal_id"]
            for row in sorted(
                proposals,
                key=lambda row: (
                    -row["smoothed_center_probability"],
                    row["center_offset_seconds"],
                    row["proposal_id"],
                ),
            )[:maximum_alarms]
        }
        proposals = [row for row in proposals if row["proposal_id"] in selected_ids]
        top_k_applied = True

    alarms: list[dict[str, Any]] = []
    merge_gap = normalized["cross_tile_merge_gap_seconds"]
    for proposal in sorted(
        proposals,
        key=lambda row: (
            row["start_offset_seconds"],
            row["stop_offset_seconds"],
            row["center_offset_seconds"],
        ),
    ):
        if (
            not alarms
            or proposal["start_offset_seconds"]
            > alarms[-1]["stop_offset_seconds"] + merge_gap
        ):
            alarms.append(
                {
                    "alarm_id": f"EVNALARM-{len(alarms):07d}",
                    "start_offset_seconds": proposal["start_offset_seconds"],
                    "stop_offset_seconds": proposal["stop_offset_seconds"],
                    "maximum_center_probability": proposal[
                        "smoothed_center_probability"
                    ],
                    "contributing_proposal_ids": [proposal["proposal_id"]],
                }
            )
        else:
            alarms[-1]["stop_offset_seconds"] = max(
                alarms[-1]["stop_offset_seconds"], proposal["stop_offset_seconds"]
            )
            alarms[-1]["maximum_center_probability"] = max(
                alarms[-1]["maximum_center_probability"],
                proposal["smoothed_center_probability"],
            )
            alarms[-1]["contributing_proposal_ids"].append(proposal["proposal_id"])
    return {
        "policy_id": policy_id,
        "policy_sha256": policy_sha256,
        "decoder_policy": normalized,
        "smoothed_center_payload_sha256": _tensor_payload_sha256(smoothed),
        "raw_proposals": proposals,
        "merged_alarms": alarms,
        "top_k_cap_applied": top_k_applied,
        "decoder_semantics": {
            "center_smoothing_scope": "independently_within_each_120_second_target_tile",
            "peak_search_scope": "independently_within_each_target_tile",
            "alarm_interval": "center_plus_minus_half_clipped_decoded_duration",
            "cross_tile_policy": "chronological_union_with_configured_gap",
            "alarm_start_is_clinical_onset": False,
            "prediction_is_confirmed_seizure": False,
        },
    }


def _build_prediction_grid_row(
    *,
    grid_id: str,
    recording: Any,
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": EVENTNET_DECODER_GRID_ROW_SCHEMA_VERSION,
        "row_id": "EVENTNET-DECODER-GRID-ROW-PENDING",
        "provider_id": EVENTNET_PROVIDER_ID,
        "decoder_code_sha256": eventnet_native_decoder_code_sha256(),
        "grid_id": grid_id,
        "policy_id": decoded["policy_id"],
        "policy_sha256": decoded["policy_sha256"],
        "decoder_policy": deepcopy(decoded["decoder_policy"]),
        "recording_id": recording.recording_id,
        "patient_alias": recording.patient_alias,
        "prediction_id": recording.prediction_id,
        "prediction_receipt_sha256": recording.prediction_receipt_sha256,
        "source_signal_sha256": recording.source_signal_sha256,
        "tensor_file_sha256": recording.tensor_file_sha256,
        "recording_duration_seconds": recording.recording_duration_seconds,
        "provider_sample_count": recording.sample_count,
        "provider_sampling_rate_hz": EVENTNET_SAMPLING_RATE_HZ,
        "smoothed_center_payload_sha256": decoded["smoothed_center_payload_sha256"],
        "raw_proposal_count": len(decoded["raw_proposals"]),
        "raw_proposals": deepcopy(decoded["raw_proposals"]),
        "merged_alarm_count": len(decoded["merged_alarms"]),
        "merged_alarms": deepcopy(decoded["merged_alarms"]),
        "outcome_status": (
            "completed_with_alarms"
            if decoded["merged_alarms"]
            else "completed_zero_alarm"
        ),
        "top_k_cap_applied": decoded["top_k_cap_applied"],
        "decoder_semantics": deepcopy(decoded["decoder_semantics"]),
        "reference_access": deepcopy(_REFERENCE_ACCESS),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["row_id"] = "EVNGRIDROW-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_eventnet_decoder_grid_prediction_row(body)


def validate_eventnet_decoder_grid_prediction_row(payload: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "row_id",
        "provider_id",
        "decoder_code_sha256",
        "grid_id",
        "policy_id",
        "policy_sha256",
        "decoder_policy",
        "recording_id",
        "patient_alias",
        "prediction_id",
        "prediction_receipt_sha256",
        "source_signal_sha256",
        "tensor_file_sha256",
        "recording_duration_seconds",
        "provider_sample_count",
        "provider_sampling_rate_hz",
        "smoothed_center_payload_sha256",
        "raw_proposal_count",
        "raw_proposals",
        "merged_alarm_count",
        "merged_alarms",
        "outcome_status",
        "top_k_cap_applied",
        "decoder_semantics",
        "reference_access",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("EventNet decoder-grid prediction row fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != EVENTNET_DECODER_GRID_ROW_SCHEMA_VERSION
        or data["provider_id"] != EVENTNET_PROVIDER_ID
        or data["provider_sampling_rate_hz"] != EVENTNET_SAMPLING_RATE_HZ
        or data["reference_access"] != _REFERENCE_ACCESS
    ):
        raise ValueError("EventNet decoder-grid row identity or firewall drifted")
    for field in ("grid_id", "recording_id", "patient_alias", "prediction_id"):
        _identifier(data[field], f"EventNet decoder-grid {field}")
    for field in (
        "decoder_code_sha256",
        "policy_sha256",
        "prediction_receipt_sha256",
        "source_signal_sha256",
        "tensor_file_sha256",
        "smoothed_center_payload_sha256",
        "receipt_sha256",
    ):
        if not _is_sha256(data[field]):
            raise ValueError(f"EventNet decoder-grid {field} is invalid")
    policy = validate_eventnet_native_decoder_policy(data["decoder_policy"])
    policy_sha256 = _canonical_sha256(policy)
    if (
        data["policy_sha256"] != policy_sha256
        or data["policy_id"] != "EVNPOL-" + policy_sha256[:20]
    ):
        raise ValueError("EventNet decoder-grid row policy binding drifted")
    duration = _finite(
        data["recording_duration_seconds"], "EventNet grid duration", minimum=1e-12
    )
    sample_count = _integer(
        data["provider_sample_count"], "EventNet grid sample count", minimum=1
    )
    if sample_count != int(round(duration * EVENTNET_SAMPLING_RATE_HZ)):
        raise ValueError("EventNet decoder-grid row clock drifted")
    proposals = data["raw_proposals"]
    alarms = data["merged_alarms"]
    if not isinstance(proposals, list) or data["raw_proposal_count"] != len(proposals):
        raise ValueError("EventNet decoder-grid raw proposal count drifted")
    proposal_ids: set[str] = set()
    previous_center = -math.inf
    for index, proposal in enumerate(proposals):
        required_proposal = {
            "proposal_id",
            "tile_id",
            "peak_sample",
            "center_offset_seconds",
            "smoothed_center_probability",
            "duration_fraction",
            "raw_duration_seconds",
            "decoded_duration_seconds_before_record_clipping",
            "start_offset_seconds",
            "stop_offset_seconds",
        }
        if type(proposal) is not dict or set(proposal) != required_proposal:
            raise ValueError(f"EventNet decoder-grid proposal {index} drifted")
        proposal_id = _identifier(proposal["proposal_id"], "proposal ID")
        if proposal_id in proposal_ids:
            raise ValueError("EventNet decoder-grid proposal IDs are not unique")
        proposal_ids.add(proposal_id)
        peak_sample = _integer(proposal["peak_sample"], "peak sample")
        center_seconds = _finite(proposal["center_offset_seconds"], "peak time")
        start = _finite(proposal["start_offset_seconds"], "proposal start")
        stop = _finite(proposal["stop_offset_seconds"], "proposal stop")
        if (
            abs(center_seconds - peak_sample / EVENTNET_SAMPLING_RATE_HZ) > 1e-12
            or center_seconds < previous_center
            or stop <= start
            or stop > duration + 1e-9
        ):
            raise ValueError("EventNet decoder-grid proposal timing drifted")
        _unit_interval(
            proposal["smoothed_center_probability"], "proposal center probability"
        )
        _unit_interval(proposal["duration_fraction"], "proposal duration fraction")
        previous_center = center_seconds
    if not isinstance(alarms, list) or data["merged_alarm_count"] != len(alarms):
        raise ValueError("EventNet decoder-grid merged alarm count drifted")
    previous_stop = 0.0
    contributed: set[str] = set()
    for index, alarm in enumerate(alarms):
        required_alarm = {
            "alarm_id",
            "start_offset_seconds",
            "stop_offset_seconds",
            "maximum_center_probability",
            "contributing_proposal_ids",
        }
        if type(alarm) is not dict or set(alarm) != required_alarm:
            raise ValueError(f"EventNet decoder-grid alarm {index} drifted")
        start = _finite(alarm["start_offset_seconds"], "alarm start")
        stop = _finite(alarm["stop_offset_seconds"], "alarm stop")
        if stop <= start or stop > duration + 1e-9 or (index and start < previous_stop):
            raise ValueError("EventNet decoder-grid alarms overlap or lie off-record")
        ids = alarm["contributing_proposal_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(item not in proposal_ids for item in ids)
        ):
            raise ValueError("EventNet decoder-grid alarm provenance drifted")
        if any(item in contributed for item in ids):
            raise ValueError("EventNet proposal contributes to multiple merged alarms")
        contributed.update(ids)
        previous_stop = stop
    if contributed != proposal_ids:
        raise ValueError("EventNet merged alarms do not account for every proposal")
    if data["outcome_status"] != (
        "completed_with_alarms" if alarms else "completed_zero_alarm"
    ):
        raise ValueError("EventNet decoder-grid row outcome drifted")
    if type(data["top_k_cap_applied"]) is not bool:
        raise TypeError("EventNet decoder-grid top-k flag must be boolean")
    digest = deepcopy(data)
    digest["row_id"] = "EVENTNET-DECODER-GRID-ROW-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["row_id"] != "EVNGRIDROW-" + _canonical_sha256(digest)[:24]:
        raise ValueError("EventNet decoder-grid row ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet decoder-grid row receipt hash drifted")
    return data


@dataclass(frozen=True, slots=True)
class ValidatedEventNetDecoderGridPrediction:
    recording_id: str
    patient_alias: str
    policy_id: str
    policy_sha256: str
    recording_duration_seconds: float
    merged_alarms_json: str
    prediction_row_receipt_sha256: str

    def merged_alarms(self) -> list[dict[str, Any]]:
        value = _loads_json(self.merged_alarms_json, "sealed EventNet merged alarms")
        if not isinstance(value, list):
            raise RuntimeError("sealed EventNet merged alarms are corrupted")
        return value


@dataclass(frozen=True, slots=True)
class ValidatedEventNetDecoderGridBundle:
    grid_root: str
    predictions: tuple[ValidatedEventNetDecoderGridPrediction, ...]
    grid_definition_json: str
    bundle_receipt_json: str

    def grid_definition(self) -> dict[str, Any]:
        return validate_eventnet_native_decoder_grid(
            _loads_json(self.grid_definition_json, "sealed decoder-grid definition")
        )

    def bundle_receipt(self) -> dict[str, Any]:
        value = _loads_json(self.bundle_receipt_json, "sealed decoder-grid receipt")
        if type(value) is not dict:
            raise RuntimeError("sealed decoder-grid receipt is corrupted")
        return value


def materialize_eventnet_native_decoder_grid(
    raw_bundle: ValidatedEventNetRawPredictionBundle,
    *,
    grid_definition: Mapping[str, Any],
    output_directory: str | Path,
    source_dev_roster_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a complete prediction-only policy grid before references open."""

    frozen = revalidate_eventnet_raw_prediction_bundle_without_references(raw_bundle)
    grid = validate_eventnet_native_decoder_grid(dict(grid_definition))
    policies = eventnet_decoder_policy_rows(grid)
    raw_receipt = frozen.validation_receipt()
    binding = validate_eventnet_source_dev_roster_binding(
        dict(source_dev_roster_binding),
        expected_recording_roster_sha256=raw_receipt[
            "expected_recording_roster_sha256"
        ],
        expected_analysis_projection_receipt_sha256=(
            raw_receipt.get("stage_p_lineage", {}).get(
                "upstream_complete_projection_receipt_sha256"
            )
            if isinstance(raw_receipt.get("stage_p_lineage"), dict)
            else None
        ),
    )

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("EventNet decoder-grid output must be a new path")
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / EVENTNET_RAW_VALIDATION_RECEIPT_FILENAME
    raw_path.write_text(
        json.dumps(raw_receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    prediction_path = output / EVENTNET_DECODER_GRID_FILENAME
    uncompressed_digest = hashlib.sha256()
    row_receipt_inventory: list[list[str]] = []
    row_count = 0
    raw_proposal_count = 0
    merged_alarm_count = 0
    zero_alarm_rows = 0
    with tempfile.NamedTemporaryFile("wb", dir=output, delete=False) as handle:
        temporary = Path(handle.name)
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=handle, mtime=0
        ) as compressed:
            for recording in frozen.recordings:
                arrays = recording.load_tensors()
                prediction = recording.prediction_receipt()
                tiles = prediction["tile_receipts"]
                smoothed_cache: dict[int, np.ndarray] = {}
                for policy_row in policies:
                    policy = policy_row["decoder_policy"]
                    sigma = int(policy["center_smoothing_sigma_samples"])
                    if sigma not in smoothed_cache:
                        smoothed_cache[sigma] = smooth_eventnet_center_per_tile(
                            arrays["center_probability"],
                            tiles,
                            sigma_samples=sigma,
                        )
                    decoded = decode_eventnet_native_policy(
                        center_probability=arrays["center_probability"],
                        duration_fraction=arrays["duration_fraction"],
                        tile_receipts=tiles,
                        recording_duration_seconds=recording.recording_duration_seconds,
                        policy=policy,
                        smoothed_center_probability=smoothed_cache[sigma],
                    )
                    row = _build_prediction_grid_row(
                        grid_id=grid["grid_id"],
                        recording=recording,
                        decoded=decoded,
                    )
                    payload = (_canonical_json(row) + "\n").encode("utf-8")
                    compressed.write(payload)
                    uncompressed_digest.update(payload)
                    row_receipt_inventory.append(
                        [row["recording_id"], row["policy_id"], row["receipt_sha256"]]
                    )
                    row_count += 1
                    raw_proposal_count += row["raw_proposal_count"]
                    merged_alarm_count += row["merged_alarm_count"]
                    zero_alarm_rows += int(row["merged_alarm_count"] == 0)
    temporary.replace(prediction_path)
    expected_rows = len(frozen.recordings) * len(policies)
    if row_count != expected_rows:
        raise RuntimeError("EventNet decoder-grid row cardinality drifted")
    body: dict[str, Any] = {
        "schema_version": EVENTNET_DECODER_GRID_BUNDLE_SCHEMA_VERSION,
        "bundle_id": "EVENTNET-DECODER-GRID-BUNDLE-PENDING",
        "method_id": EVENTNET_DECODER_GRID_METHOD_ID,
        "provider_id": EVENTNET_PROVIDER_ID,
        "decoder_code_sha256": eventnet_native_decoder_code_sha256(),
        "source_split": "source_dev",
        "grid_definition": grid,
        "grid_definition_sha256": _canonical_sha256(grid),
        "source_dev_roster_binding": binding,
        "raw_bundle_validation_id": raw_receipt["validation_id"],
        "raw_bundle_validation_receipt_sha256": raw_receipt["receipt_sha256"],
        "raw_bundle_validation_file_sha256": _file_sha256(raw_path),
        "record_count": len(frozen.recordings),
        "policy_count": len(policies),
        "prediction_row_count": row_count,
        "raw_proposal_count": raw_proposal_count,
        "merged_alarm_count": merged_alarm_count,
        "zero_alarm_policy_record_row_count": zero_alarm_rows,
        "prediction_grid_filename": EVENTNET_DECODER_GRID_FILENAME,
        "prediction_grid_gzip_file_sha256": _file_sha256(prediction_path),
        "prediction_grid_uncompressed_jsonl_sha256": uncompressed_digest.hexdigest(),
        "row_receipt_inventory_sha256": _canonical_sha256(row_receipt_inventory),
        "recording_roster_sha256": _canonical_sha256(
            [row.recording_id for row in frozen.recordings]
        ),
        "policy_roster_sha256": _canonical_sha256(
            [row["policy_id"] for row in policies]
        ),
        "all_records_decoded_under_every_policy": True,
        "raw_predictions_validated_before_first_grid_decode": True,
        "reference_access": deepcopy(_REFERENCE_ACCESS),
        "scope_receipt": deepcopy(_GRID_SCOPE),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["bundle_id"] = "EVNGRIDBUNDLE-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    receipt_path = output / EVENTNET_DECODER_GRID_RECEIPT_FILENAME
    receipt_path.write_text(
        json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return body


def validate_eventnet_native_decoder_grid_bundle_without_references(
    root: str | Path,
) -> ValidatedEventNetDecoderGridBundle:
    """Validate the complete prediction grid before any reference is opened."""

    directory = _regular_directory(Path(root).resolve(strict=True), "decoder-grid root")
    if {path.name for path in directory.iterdir()} != {
        EVENTNET_RAW_VALIDATION_RECEIPT_FILENAME,
        EVENTNET_DECODER_GRID_FILENAME,
        EVENTNET_DECODER_GRID_RECEIPT_FILENAME,
    }:
        raise ValueError("EventNet decoder-grid directory has missing or unknown files")
    receipt_path = _regular_file(
        directory / EVENTNET_DECODER_GRID_RECEIPT_FILENAME,
        "EventNet decoder-grid receipt",
    )
    receipt = _loads_json(receipt_path.read_bytes(), "EventNet decoder-grid receipt")
    required = {
        "schema_version",
        "bundle_id",
        "method_id",
        "provider_id",
        "decoder_code_sha256",
        "source_split",
        "grid_definition",
        "grid_definition_sha256",
        "source_dev_roster_binding",
        "raw_bundle_validation_id",
        "raw_bundle_validation_receipt_sha256",
        "raw_bundle_validation_file_sha256",
        "record_count",
        "policy_count",
        "prediction_row_count",
        "raw_proposal_count",
        "merged_alarm_count",
        "zero_alarm_policy_record_row_count",
        "prediction_grid_filename",
        "prediction_grid_gzip_file_sha256",
        "prediction_grid_uncompressed_jsonl_sha256",
        "row_receipt_inventory_sha256",
        "recording_roster_sha256",
        "policy_roster_sha256",
        "all_records_decoded_under_every_policy",
        "raw_predictions_validated_before_first_grid_decode",
        "reference_access",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(receipt) is not dict or set(receipt) != required:
        raise ValueError("EventNet decoder-grid bundle receipt fields drifted")
    if (
        receipt["schema_version"] != EVENTNET_DECODER_GRID_BUNDLE_SCHEMA_VERSION
        or receipt["method_id"] != EVENTNET_DECODER_GRID_METHOD_ID
        or receipt["provider_id"] != EVENTNET_PROVIDER_ID
        or not _is_sha256(receipt["decoder_code_sha256"])
        or receipt["source_split"] != "source_dev"
        or receipt["reference_access"] != _REFERENCE_ACCESS
        or receipt["scope_receipt"] != _GRID_SCOPE
        or receipt["all_records_decoded_under_every_policy"] is not True
        or receipt["raw_predictions_validated_before_first_grid_decode"] is not True
    ):
        raise ValueError("EventNet decoder-grid bundle identity or firewall drifted")
    grid = validate_eventnet_native_decoder_grid(receipt["grid_definition"])
    if receipt["grid_definition_sha256"] != _canonical_sha256(grid):
        raise ValueError("EventNet decoder-grid definition hash drifted")
    raw_path = _regular_file(
        directory / EVENTNET_RAW_VALIDATION_RECEIPT_FILENAME,
        "EventNet raw validation receipt sidecar",
    )
    if _file_sha256(raw_path) != receipt["raw_bundle_validation_file_sha256"]:
        raise ValueError("EventNet raw validation receipt file hash drifted")
    raw_receipt = _loads_json(raw_path.read_bytes(), "EventNet raw validation receipt")
    if (
        raw_receipt.get("validation_id") != receipt["raw_bundle_validation_id"]
        or raw_receipt.get("receipt_sha256")
        != receipt["raw_bundle_validation_receipt_sha256"]
        or raw_receipt.get("reference_access") != _REFERENCE_ACCESS
    ):
        raise ValueError("EventNet raw validation and decoder-grid bundle disagree")
    validate_eventnet_source_dev_roster_binding(
        receipt["source_dev_roster_binding"],
        expected_recording_roster_sha256=raw_receipt.get(
            "expected_recording_roster_sha256"
        ),
        expected_analysis_projection_receipt_sha256=(
            raw_receipt.get("stage_p_lineage", {}).get(
                "upstream_complete_projection_receipt_sha256"
            )
            if isinstance(raw_receipt.get("stage_p_lineage"), dict)
            else None
        ),
    )
    prediction_path = _regular_file(
        directory / EVENTNET_DECODER_GRID_FILENAME,
        "EventNet decoder-grid prediction inventory",
    )
    if (
        receipt["prediction_grid_filename"] != EVENTNET_DECODER_GRID_FILENAME
        or _file_sha256(prediction_path) != receipt["prediction_grid_gzip_file_sha256"]
    ):
        raise ValueError("EventNet decoder-grid compressed file hash drifted")
    predictions: list[ValidatedEventNetDecoderGridPrediction] = []
    uncompressed_digest = hashlib.sha256()
    row_inventory: list[list[str]] = []
    observed_pairs: list[tuple[str, str]] = []
    raw_count = 0
    alarm_count = 0
    zero_count = 0
    try:
        with gzip.open(prediction_path, "rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith(b"\n") or not line.strip():
                    raise ValueError("EventNet decoder-grid JSONL framing drifted")
                uncompressed_digest.update(line)
                row = validate_eventnet_decoder_grid_prediction_row(
                    _loads_json(line, f"EventNet decoder-grid row {line_number}")
                )
                if (
                    row["grid_id"] != grid["grid_id"]
                    or row["decoder_code_sha256"] != receipt["decoder_code_sha256"]
                ):
                    raise ValueError(
                        "EventNet decoder-grid row grid ID or decoder binding drifted"
                    )
                pair = (row["recording_id"], row["policy_id"])
                if observed_pairs and pair <= observed_pairs[-1]:
                    raise ValueError(
                        "EventNet decoder-grid rows are not canonical and unique"
                    )
                observed_pairs.append(pair)
                row_inventory.append(
                    [row["recording_id"], row["policy_id"], row["receipt_sha256"]]
                )
                raw_count += row["raw_proposal_count"]
                alarm_count += row["merged_alarm_count"]
                zero_count += int(row["merged_alarm_count"] == 0)
                predictions.append(
                    ValidatedEventNetDecoderGridPrediction(
                        recording_id=row["recording_id"],
                        patient_alias=row["patient_alias"],
                        policy_id=row["policy_id"],
                        policy_sha256=row["policy_sha256"],
                        recording_duration_seconds=row["recording_duration_seconds"],
                        merged_alarms_json=_canonical_json(row["merged_alarms"]),
                        prediction_row_receipt_sha256=row["receipt_sha256"],
                    )
                )
    except (OSError, EOFError) as error:
        raise ValueError("EventNet decoder-grid gzip is invalid") from error
    policies = eventnet_decoder_policy_rows(grid)
    recordings = sorted({row.recording_id for row in predictions})
    expected_pairs = [
        (recording_id, policy["policy_id"])
        for recording_id in recordings
        for policy in policies
    ]
    if observed_pairs != expected_pairs:
        raise ValueError(
            "EventNet decoder-grid policy-by-record inventory is incomplete"
        )
    checks = {
        "record_count": len(recordings),
        "policy_count": len(policies),
        "prediction_row_count": len(predictions),
        "raw_proposal_count": raw_count,
        "merged_alarm_count": alarm_count,
        "zero_alarm_policy_record_row_count": zero_count,
        "prediction_grid_uncompressed_jsonl_sha256": uncompressed_digest.hexdigest(),
        "row_receipt_inventory_sha256": _canonical_sha256(row_inventory),
        "recording_roster_sha256": _canonical_sha256(recordings),
        "policy_roster_sha256": _canonical_sha256(
            [row["policy_id"] for row in policies]
        ),
    }
    for field, expected in checks.items():
        if receipt[field] != expected:
            raise ValueError(f"EventNet decoder-grid bundle {field} drifted")
    digest = deepcopy(receipt)
    digest["bundle_id"] = "EVENTNET-DECODER-GRID-BUNDLE-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if receipt["bundle_id"] != "EVNGRIDBUNDLE-" + _canonical_sha256(digest)[:24]:
        raise ValueError("EventNet decoder-grid bundle ID is not content-bound")
    digest = deepcopy(receipt)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if receipt["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet decoder-grid bundle receipt hash drifted")
    return ValidatedEventNetDecoderGridBundle(
        grid_root=str(directory),
        predictions=tuple(predictions),
        grid_definition_json=_canonical_json(grid),
        bundle_receipt_json=_canonical_json(receipt),
    )


def revalidate_eventnet_native_decoder_grid_bundle_without_references(
    value: object,
) -> ValidatedEventNetDecoderGridBundle:
    if type(value) is not ValidatedEventNetDecoderGridBundle:
        raise TypeError("source-dev calibration requires the sealed decoder-grid type")
    receipt = value.bundle_receipt()
    grid = value.grid_definition()
    if (
        receipt.get("schema_version") != EVENTNET_DECODER_GRID_BUNDLE_SCHEMA_VERSION
        or receipt.get("grid_definition") != grid
        or receipt.get("prediction_row_count") != len(value.predictions)
        or receipt.get("reference_access") != _REFERENCE_ACCESS
    ):
        raise ValueError("sealed EventNet decoder-grid bundle drifted")
    return value


__all__ = [
    "EVENTNET_DECODER_GRID_BUNDLE_SCHEMA_VERSION",
    "EVENTNET_DECODER_GRID_METHOD_ID",
    "EVENTNET_DECODER_GRID_SCHEMA_VERSION",
    "EVENTNET_TUSZ_ANALYSIS_PROJECTION_BINDING_SCHEMA_VERSION",
    "ValidatedEventNetDecoderGridBundle",
    "ValidatedEventNetDecoderGridPrediction",
    "decode_eventnet_native_policy",
    "eventnet_native_decoder_code_sha256",
    "eventnet_decoder_policy_rows",
    "materialize_eventnet_native_decoder_grid",
    "revalidate_eventnet_native_decoder_grid_bundle_without_references",
    "smooth_eventnet_center_per_tile",
    "validate_eventnet_decoder_grid_prediction_row",
    "validate_eventnet_native_decoder_grid",
    "validate_eventnet_native_decoder_grid_bundle_without_references",
    "validate_eventnet_native_decoder_policy",
    "validate_eventnet_source_dev_roster_binding",
]
