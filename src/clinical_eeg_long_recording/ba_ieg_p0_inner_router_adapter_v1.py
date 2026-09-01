"""Target-free P0 to BA-IEG inner-router candidate adapter.

This module is the executable seam between the deterministic physical-time P0
tokens and :mod:`ba_ieg_inner_ragged_router_v1`.  It does not select an EEG
channel.  One router cell is one ``physical interval x scale x temporal
permission`` group and contains *all* signal-eligible source rows from every
eligible analysis unit and reference view in that group.

The current P0 producer materializes onset-causal and offline-context views but
does not materialize a native-morphology view.  Permissions are inherited from
``BAIEGEventTokens.view_effective_temporal_roles``; this adapter never relabels
causal or offline samples as native morphology.  Consequently the native lane
has a typed no-source-view status today and will begin producing candidates
without an adapter change when a qualified native P0 view is added.

P0 stores the inward-mapped actual support of every token, not its nominal
physical tile.  Adapter v1 therefore accepts only the frozen non-overlapping
1/4/16-second P0 grid, reconstructs that grid from the event start, uniquely
maps every source token to it, and records every per-row actual support.  A
cell's actual interval is the conservative intersection shared by all of its
active source rows.  Unsupported overlapping or non-nested policies return a
typed not-evaluable artifact instead of silently sampling a convenient tree.

Routing features and scores are deterministic signal-only navigation values.
They are not clinical Findings and cannot authorize onset, SOZ, EZ, diagnosis
or report claims.  Successful production consumes a target-free event-model
projection.  The convenience P0 entry point first detaches the dense
measurement target sidecar through the registered projection boundary; no
target value or label is read while constructing cells or scores.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch

from .ba_ieg_event_model_input_projection_v1 import (
    BAIEGEventModelInputProjectionV1,
    project_ba_ieg_event_model_input_v1,
)
from .ba_ieg_inner_ragged_router_v1 import (
    BA_IEG_INNER_ROUTER_PERMISSIONS,
    BA_IEG_INNER_ROUTER_SCALES,
    BAIEGInnerRaggedRouterPolicyV1,
    materialize_ba_ieg_inner_ragged_router_v1,
)
from .ba_ieg_training_contract import (
    BA_IEG_P0_TOKEN_FEATURES,
    BA_IEG_TOKEN_SCALES,
    BAIEGEventTokens,
    BAIEGP0MaterializationResult,
    BAIEGP0TokenizationPolicy,
)
from .canonical_signal_views import validate_canonical_signal_receipt


BA_IEG_P0_INNER_ROUTER_ADAPTER_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_ba_ieg_p0_inner_router_candidate_adapter_v1"
BA_IEG_P0_INNER_ROUTER_ADAPTER_METHOD_ID: Final[
    str
] = "ba_ieg_target_free_p0_channel_neutral_candidate_adapter_v1"
BA_IEG_P0_INNER_ROUTER_DETERMINISTIC_SCORE_METHOD_ID: Final[
    str
] = "ba_ieg_p0_lane_local_deterministic_utility_density_v1"

_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TOLERANCE_SECONDS: Final[float] = 1e-5
_FIXED_SCALE_DURATIONS: Final[dict[str, float]] = {
    "fine": 1.0,
    "coarse": 4.0,
    "context": 16.0,
}
_ADAPTER_REASON_CODES: Final[frozenset[str | None]] = frozenset(
    {
        None,
        "p0_materialization_failed",
        "target_free_projection_required",
        "unsupported_p0_physical_grid",
        "outer_support_binding_mismatch",
        "token_nominal_tile_mapping_failed",
        "duplicate_source_token_coordinate",
        "empty_shared_actual_support",
        "router_incompatible_cell_topology",
        "no_signal_eligible_router_cells",
    }
)
_SCOPE_RECEIPT: Final[dict[str, bool]] = {
    "eeg_signal_rows_used": True,
    "target_free_model_input_required": True,
    "deterministic_measurement_target_sidecar_used_for_features": False,
    "public_or_private_label_used": False,
    "edf_annotation_used": False,
    "spreadsheet_used": False,
    "clinical_text_used": False,
    "channel_or_reference_subset_selected": False,
    "new_physical_eeg_support_acquired": False,
    "transitive_p0_token_dependency_receipt_used": True,
    "interval_raw_sample_dependency_closure_proven": False,
    "onset_permission_inherited_not_granted": True,
    "clinical_finding_or_soz_claim_authorized": False,
}


class _TypedAdapterFailure(RuntimeError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        if reason_code not in _ADAPTER_REASON_CODES or reason_code is None:
            raise ValueError("unknown P0 inner-router adapter failure code")
        self.reason_code = reason_code
        self.detail = detail


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in _SHA256_CHARACTERS for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _finite_interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item interval")
    start, stop = float(value[0]), float(value[1])
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError(f"{name} must be finite and have positive duration")
    return start, stop


def _same_interval(
    left: Sequence[float], right: Sequence[float], *, tolerance: float = 1e-6
) -> bool:
    return bool(
        abs(float(left[0]) - float(right[0])) <= tolerance
        and abs(float(left[1]) - float(right[1])) <= tolerance
    )


def _contains(outer: Sequence[float], inner: Sequence[float]) -> bool:
    return bool(
        float(inner[0]) >= float(outer[0]) - _TOLERANCE_SECONDS
        and float(inner[1]) <= float(outer[1]) + _TOLERANCE_SECONDS
    )


def _overlaps(left: Sequence[float], right: Sequence[float]) -> bool:
    return bool(
        min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0]))
        > _TOLERANCE_SECONDS
    )


def _normalize_union(intervals: Sequence[Sequence[float]]) -> list[list[float]]:
    ordered = sorted(
        (_finite_interval(item, "support interval") for item in intervals),
        key=lambda item: (item[0], item[1]),
    )
    result: list[list[float]] = []
    for start, stop in ordered:
        if not result or start > result[-1][1] + _TOLERANCE_SECONDS:
            result.append([start, stop])
        else:
            result[-1][1] = max(result[-1][1], stop)
    return result


def _same_union(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> bool:
    first, second = _normalize_union(left), _normalize_union(right)
    return len(first) == len(second) and all(
        _same_interval(a, b, tolerance=_TOLERANCE_SECONDS)
        for a, b in zip(first, second)
    )


def _policy_from_dict(value: object) -> BAIEGP0TokenizationPolicy:
    if type(value) is not dict:
        raise ValueError("P0 tokenization policy receipt must be an object")
    durations = value.get("scale_duration_seconds")
    steps = value.get("scale_step_seconds")
    spectral = value.get("spectral_interval_hz")
    if (
        type(durations) is not dict
        or type(steps) is not dict
        or not isinstance(spectral, list)
        or len(spectral) != 2
    ):
        raise ValueError("P0 tokenization policy receipt is incomplete")
    result = BAIEGP0TokenizationPolicy(
        fine_duration_seconds=durations["fine"],
        fine_step_seconds=steps["fine"],
        coarse_duration_seconds=durations["coarse"],
        coarse_step_seconds=steps["coarse"],
        context_duration_seconds=durations["context"],
        context_step_seconds=steps["context"],
        minimum_fine_samples=value["minimum_fine_samples"],
        spectral_low_hz=spectral[0],
        spectral_high_hz=spectral[1],
        rhythmic_half_bandwidth_hz=value["rhythmic_half_bandwidth_hz"],
    )
    if result.to_dict() != value:
        raise ValueError("P0 tokenization policy receipt is not reproducible")
    return result


def _require_v1_grid(policy: BAIEGP0TokenizationPolicy) -> None:
    supplied = {
        "fine": (
            float(policy.fine_duration_seconds),
            float(policy.fine_step_seconds),
        ),
        "coarse": (
            float(policy.coarse_duration_seconds),
            float(policy.coarse_step_seconds),
        ),
        "context": (
            float(policy.context_duration_seconds),
            float(policy.context_step_seconds),
        ),
    }
    if any(
        not math.isclose(duration, _FIXED_SCALE_DURATIONS[scale], abs_tol=1e-12)
        or not math.isclose(step, duration, abs_tol=1e-12)
        for scale, (duration, step) in supplied.items()
    ):
        raise _TypedAdapterFailure(
            "unsupported_p0_physical_grid",
            "adapter v1 requires non-overlapping 1/4/16-second P0 tiles",
        )


def _nominal_tiles(
    event_interval: tuple[float, float],
) -> dict[str, list[tuple[float, float]]]:
    start, stop = event_interval
    result: dict[str, list[tuple[float, float]]] = {
        scale: [] for scale in BA_IEG_INNER_ROUTER_SCALES
    }
    for scale in BA_IEG_INNER_ROUTER_SCALES:
        duration = _FIXED_SCALE_DURATIONS[scale]
        index = 0
        while True:
            tile_start = start + index * duration
            if tile_start >= stop - 1e-10:
                break
            tile_stop = tile_start + duration
            if tile_stop > stop + 1e-10:
                if scale != "fine":
                    break
                tile_stop = stop
            result[scale].append((float(tile_start), float(tile_stop)))
            index += 1
    return result


def _match_nominal_tile(
    actual: tuple[float, float],
    tiles: Sequence[tuple[float, float]],
) -> int:
    candidates = [
        index for index, nominal in enumerate(tiles) if _contains(nominal, actual)
    ]
    if not candidates:
        raise _TypedAdapterFailure(
            "token_nominal_tile_mapping_failed",
            "a P0 actual support does not map inside its reconstructed nominal tile",
        )
    candidates.sort(
        key=lambda index: (
            abs(tiles[index][0] - actual[0]) + abs(tiles[index][1] - actual[1]),
            index,
        )
    )
    if len(candidates) > 1:
        first_distance = abs(tiles[candidates[0]][0] - actual[0]) + abs(
            tiles[candidates[0]][1] - actual[1]
        )
        second_distance = abs(tiles[candidates[1]][0] - actual[0]) + abs(
            tiles[candidates[1]][1] - actual[1]
        )
        if abs(first_distance - second_distance) <= 1e-10:
            raise _TypedAdapterFailure(
                "token_nominal_tile_mapping_failed",
                "a P0 actual support ambiguously maps to multiple nominal tiles",
            )
    return candidates[0]


def _row_sha256(values: torch.Tensor) -> str:
    return _canonical_sha256([float(item) for item in values.to(torch.float64)])


def _mask_sha256(values: torch.Tensor) -> str:
    return _canonical_sha256([bool(item) for item in values])


def _reference_row_fingerprint(values: torch.Tensor) -> str:
    row = values.detach().cpu().to(torch.float64)
    norm = float(torch.sum(torch.abs(row)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise _TypedAdapterFailure(
            "router_incompatible_cell_topology",
            "an eligible P0 unit has an empty physical reference row",
        )
    normalized = row / norm
    nonzero = torch.nonzero(torch.abs(normalized) > 1e-12, as_tuple=False).flatten()
    if int(nonzero.numel()) and float(normalized[int(nonzero[0])]) < 0.0:
        normalized = -normalized
    return _canonical_sha256([round(float(item), 12) for item in normalized])


def _transitive_token_dependency_sha256(
    *,
    recording_id: str,
    event_id: str,
    permission: str,
    scale: str,
    nominal_interval_seconds: Sequence[float],
    actual_interval_seconds: Sequence[float],
    view_id: str,
    view_receipt_sha256: str,
    transform_sha256: str,
    physical_reference_row_sha256: str,
    unit_id: str,
    unit_source_id: str,
    reference_family: str,
    token_values_sha256: str,
    token_feature_mask_sha256: str,
    canonical_signal_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "ba_ieg_p0_transitive_token_dependency_v1",
            "recording_id": recording_id,
            "event_id": event_id,
            "permission": permission,
            "scale": scale,
            "nominal_interval_seconds": list(nominal_interval_seconds),
            "actual_interval_seconds": list(actual_interval_seconds),
            "view_id": view_id,
            "view_receipt_sha256": view_receipt_sha256,
            "transform_sha256": transform_sha256,
            "physical_reference_row_sha256": physical_reference_row_sha256,
            "unit_id": unit_id,
            "unit_source_id": unit_source_id,
            "reference_family": reference_family,
            "token_values_sha256": token_values_sha256,
            "token_feature_mask_sha256": token_feature_mask_sha256,
            "canonical_signal_sha256": canonical_signal_sha256,
            "raw_sample_interval_closure_claimed": False,
        }
    )


def _unique_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Remove exact facet copies for metrics, never for token cost/provenance."""

    by_fingerprint: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        by_fingerprint.setdefault(str(row["facet_fingerprint_sha256"]), row)
    return [by_fingerprint[key] for key in sorted(by_fingerprint)]


def _pairwise_normalized_disagreement(
    rows: Sequence[Mapping[str, Any]],
) -> float | None:
    terms: list[float] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            common = left["feature_mask"] & right["feature_mask"]
            if not bool(common.any()):
                continue
            a = left["values"][common].to(torch.float64)
            b = right["values"][common].to(torch.float64)
            ratio = torch.abs(a - b) / (torch.abs(a) + torch.abs(b) + 1e-9)
            terms.extend(float(item) for item in ratio)
    if not terms:
        return None
    return min(1.0, max(0.0, math.fsum(terms) / len(terms)))


def _cross_channel_disagreement(rows: Sequence[Mapping[str, Any]]) -> float:
    by_view: dict[int, list[Mapping[str, Any]]] = {}
    for row in _unique_metric_rows(rows):
        by_view.setdefault(int(row["view_index"]), []).append(row)
    by_physical_transform: dict[str, float] = {}
    for view_rows in by_view.values():
        value = _pairwise_normalized_disagreement(view_rows)
        if value is not None:
            by_physical_transform.setdefault(
                str(view_rows[0]["reference_transform_fingerprint_sha256"]),
                value,
            )
    values = list(by_physical_transform.values())
    return 0.0 if not values else math.fsum(values) / len(values)


def _family_summary(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = len(BA_IEG_P0_TOKEN_FEATURES)
    summary = torch.zeros(dimension, dtype=torch.float64)
    mask = torch.zeros(dimension, dtype=torch.bool)
    for feature_index in range(dimension):
        values = [
            float(row["values"][feature_index])
            for row in rows
            if bool(row["feature_mask"][feature_index])
        ]
        if values:
            summary[feature_index] = torch.median(
                torch.tensor(sorted(set(values)), dtype=torch.float64)
            )
            mask[feature_index] = True
    return summary, mask


def _reference_instability(rows: Sequence[Mapping[str, Any]]) -> float:
    by_transform: dict[str, list[Mapping[str, Any]]] = {}
    for row in _unique_metric_rows(rows):
        by_transform.setdefault(
            str(row["reference_transform_fingerprint_sha256"]), []
        ).append(row)
    if len(by_transform) < 2:
        # Missing cross-reference opportunity must not masquerade as stability.
        return 1.0
    summaries = {
        transform: _family_summary(transform_rows)
        for transform, transform_rows in sorted(by_transform.items())
    }
    terms: list[float] = []
    families = sorted(summaries)
    for left_index, left_family in enumerate(families):
        for right_family in families[left_index + 1 :]:
            left, left_mask = summaries[left_family]
            right, right_mask = summaries[right_family]
            common = left_mask & right_mask
            if not bool(common.any()):
                continue
            ratio = torch.abs(left[common] - right[common]) / (
                torch.abs(left[common]) + torch.abs(right[common]) + 1e-9
            )
            terms.extend(float(item) for item in ratio)
    return 1.0 if not terms else min(1.0, math.fsum(terms) / len(terms))


def _cell_features(
    active_rows: Sequence[Mapping[str, Any]],
    *,
    nominal_interval: tuple[float, float],
    present_row_count: int,
    expected_row_count: int,
) -> dict[str, float]:
    duration = nominal_interval[1] - nominal_interval[0]
    change_index = BA_IEG_P0_TOKEN_FEATURES.index("robust_previous_tile_change_score")
    change_values = sorted(
        {
            max(0.0, float(row["values"][change_index]))
            for row in _unique_metric_rows(active_rows)
            if bool(row["feature_mask"][change_index])
        }
    )
    if change_values:
        change_level = float(
            torch.median(torch.tensor(change_values, dtype=torch.float64))
        )
    else:
        change_level = 0.0
    quality = len(active_rows) / max(1, present_row_count)
    opportunity = present_row_count / max(1, expected_row_count)
    cross = _cross_channel_disagreement(active_rows)
    reference = _reference_instability(active_rows)
    # P0 exposes no calibrated causal boundary posterior.  Keep this at
    # maximal uncertainty instead of relabelling a change statistic as a
    # calibrated boundary probability.
    boundary_uncertainty = 1.0
    change_density = change_level / max(duration, 1e-12)
    score = (
        quality
        * opportunity
        * (
            0.55 * math.log1p(change_density)
            + 0.20 * cross
            + 0.15 * (1.0 - reference)
            + 0.10 * (1.0 - boundary_uncertainty)
        )
    )
    return {
        "boundary_uncertainty": float(boundary_uncertainty),
        "change_density": float(change_density),
        "cross_channel_disagreement": float(cross),
        "reference_instability": float(reference),
        "quality_fraction": float(quality),
        "opportunity_fraction": float(opportunity),
        "router_score": float(score),
    }


def _score_policy_receipt() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "ba_ieg_p0_deterministic_router_score_policy_v1",
        "method_id": BA_IEG_P0_INNER_ROUTER_DETERMINISTIC_SCORE_METHOD_ID,
        "router_score_semantics": "per_physical_second_utility_density",
        "formula": (
            "quality*opportunity*(0.55*log1p(change_density)+"
            "0.20*cross_channel_disagreement+0.15*(1-reference_instability)+"
            "0.10*(1-boundary_uncertainty))"
        ),
        "aggregation": "exact_facet_deduplicated_then_reference_family_balanced_v1",
        "permission_lane_isolation": True,
        "source_token_copy_can_increase_utility": False,
        "labels_annotations_spreadsheets_or_clinical_text_used": False,
        "clinical_finding_onset_soz_or_claim_authorized": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _cell_id(
    recording_id: str,
    event_id: str,
    permission: str,
    scale: str,
    nominal: tuple[float, float],
) -> str:
    digest = _canonical_sha256(
        {
            "schema_version": "ba_ieg_p0_physical_cell_identity_v1",
            "recording_id": recording_id,
            "event_id": event_id,
            "permission": permission,
            "scale": scale,
            "nominal_interval_seconds": [nominal[0], nominal[1]],
        }
    )
    return "P0-CELL-" + digest[:24]


def _build_source_rows(
    event: BAIEGEventTokens,
    tiles: Mapping[str, Sequence[tuple[float, float]]],
    *,
    canonical_signal_sha256: str,
) -> tuple[dict[tuple[str, str, int], list[dict[str, Any]]], dict[str, int],]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    expected_units: dict[str, int] = {}
    for permission in BA_IEG_INNER_ROUTER_PERMISSIONS:
        expected_units[permission] = sum(
            bool(event.unit_evidence_mask[unit_index])
            and event.view_effective_temporal_roles[
                int(event.unit_view_index[unit_index])
            ]
            == permission
            for unit_index in range(len(event.unit_ids))
        )
    view_reference_fingerprints: dict[int, str] = {}
    for view_index in range(len(event.view_ids)):
        physical_rows = sorted(
            {
                _reference_row_fingerprint(event.unit_reference_matrix[unit_index])
                for unit_index in range(len(event.unit_ids))
                if int(event.unit_view_index[unit_index]) == view_index
                and bool(event.unit_evidence_mask[unit_index])
            }
        )
        if physical_rows:
            view_reference_fingerprints[view_index] = _canonical_sha256(
                {
                    "schema_version": "ba_ieg_physical_reference_transform_v1",
                    "physical_reference_row_fingerprints": physical_rows,
                }
            )

    seen_coordinates: set[tuple[str, str, int, int]] = set()
    for token_index in range(int(event.token_values.shape[0])):
        unit_index = int(event.token_unit_index[token_index])
        view_index = int(event.token_view_index[token_index])
        if not bool(event.unit_evidence_mask[unit_index]):
            continue
        permission = event.view_effective_temporal_roles[view_index]
        if permission not in BA_IEG_INNER_ROUTER_PERMISSIONS:
            raise _TypedAdapterFailure(
                "router_incompatible_cell_topology",
                "P0 event contains an unregistered temporal permission",
            )
        scale = BA_IEG_TOKEN_SCALES[int(event.token_scale_index[token_index])]
        actual = _finite_interval(
            event.token_time_bounds_seconds[token_index].tolist(),
            "P0 token actual support",
        )
        tile_index = _match_nominal_tile(actual, tiles[scale])
        coordinate = (permission, scale, tile_index, unit_index)
        if coordinate in seen_coordinates:
            raise _TypedAdapterFailure(
                "duplicate_source_token_coordinate",
                "multiple P0 tokens occupy one permission/scale/tile/unit coordinate",
            )
        seen_coordinates.add(coordinate)
        reference_family = event.reference_families[view_index]
        values = event.token_values[token_index]
        feature_mask = event.token_feature_mask[token_index]
        physical_reference_row_sha256 = _reference_row_fingerprint(
            event.unit_reference_matrix[unit_index]
        )
        physical_transform_sha256 = view_reference_fingerprints[view_index]
        facet_source = {
            "physical_reference_row_sha256": physical_reference_row_sha256,
            "unit_source_id": event.unit_source_ids[unit_index],
            "actual_interval_seconds": [actual[0], actual[1]],
            "token_values_sha256": _row_sha256(values),
            "token_feature_mask_sha256": _mask_sha256(feature_mask),
        }
        token_dependency_sha256 = _transitive_token_dependency_sha256(
            recording_id=event.recording_id,
            event_id=event.event_id,
            permission=permission,
            scale=scale,
            nominal_interval_seconds=tiles[scale][tile_index],
            actual_interval_seconds=actual,
            view_id=event.view_ids[view_index],
            view_receipt_sha256=event.view_receipt_sha256s[view_index],
            transform_sha256=event.view_transform_sha256s[view_index],
            physical_reference_row_sha256=physical_reference_row_sha256,
            unit_id=event.unit_ids[unit_index],
            unit_source_id=event.unit_source_ids[unit_index],
            reference_family=reference_family,
            token_values_sha256=facet_source["token_values_sha256"],
            token_feature_mask_sha256=facet_source["token_feature_mask_sha256"],
            canonical_signal_sha256=canonical_signal_sha256,
        )
        row = {
            "source_token_index": token_index,
            "view_index": view_index,
            "view_id": event.view_ids[view_index],
            "view_receipt_sha256": event.view_receipt_sha256s[view_index],
            "transform_sha256": event.view_transform_sha256s[view_index],
            "temporal_evidence_sha256": (
                event.view_temporal_evidence_sha256s[view_index]
            ),
            "reference_family": reference_family,
            "physical_reference_row_sha256": physical_reference_row_sha256,
            "reference_transform_fingerprint_sha256": (physical_transform_sha256),
            "unit_index": unit_index,
            "unit_id": event.unit_ids[unit_index],
            "unit_source_id": event.unit_source_ids[unit_index],
            "unit_type": event.unit_types[unit_index],
            "actual_interval_seconds": [actual[0], actual[1]],
            "signal_eligible": bool(event.token_signal_mask[token_index]),
            "future_sample_access": bool(event.token_future_sample_access[token_index]),
            "onset_evidence_authorized": bool(
                event.token_onset_evidence_mask[token_index]
            ),
            "canonical_signal_sha256": canonical_signal_sha256,
            "canonical_receipt_sha256": event.canonical_receipt_sha256,
            "token_values_sha256": facet_source["token_values_sha256"],
            "token_feature_mask_sha256": facet_source["token_feature_mask_sha256"],
            "facet_fingerprint_sha256": _canonical_sha256(facet_source),
            "p0_token_dependency_sha256": token_dependency_sha256,
            # Tensors are internal-only and removed from persisted ledgers.
            "values": values,
            "feature_mask": feature_mask,
        }
        if permission == "onset_causal" and (
            event.view_dependency_policies[view_index] != "past_and_present_only"
            or row["future_sample_access"] is not False
        ):
            raise _TypedAdapterFailure(
                "router_incompatible_cell_topology",
                "an onset-causal P0 row failed the future-free permission firewall",
            )
        if permission != "onset_causal" and row["onset_evidence_authorized"]:
            raise _TypedAdapterFailure(
                "router_incompatible_cell_topology",
                "a non-onset P0 row acquired onset authorization",
            )
        groups.setdefault((permission, scale, tile_index), []).append(row)
    return groups, expected_units


def _persisted_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in row.items()
        if key not in {"values", "feature_mask"}
    }


def _raw_cells(
    event: BAIEGEventTokens,
    groups: Mapping[tuple[str, str, int], Sequence[Mapping[str, Any]]],
    expected_units: Mapping[str, int],
    tiles: Mapping[str, Sequence[tuple[float, float]]],
    *,
    canonical_signal_sha256: str,
    score_policy_receipt_sha256: str,
    router_policy: BAIEGInnerRaggedRouterPolicyV1,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cells: dict[str, dict[str, Any]] = {}
    ledgers: dict[str, dict[str, Any]] = {}
    for (permission, scale, tile_index), rows in sorted(groups.items()):
        active = [row for row in rows if bool(row["signal_eligible"])]
        if not active:
            continue
        nominal = tiles[scale][tile_index]
        actual_start = max(float(row["actual_interval_seconds"][0]) for row in active)
        actual_stop = min(float(row["actual_interval_seconds"][1]) for row in active)
        if actual_stop <= actual_start + _TOLERANCE_SECONDS:
            raise _TypedAdapterFailure(
                "empty_shared_actual_support",
                "eligible multi-clock rows have no common physical support",
            )
        cell_id = _cell_id(
            event.recording_id, event.event_id, permission, scale, nominal
        )
        if cell_id in cells:
            raise RuntimeError("content-addressed P0 cell IDs collided")
        features = _cell_features(
            active,
            nominal_interval=nominal,
            present_row_count=len(rows),
            expected_row_count=int(expected_units[permission]),
        )
        source_indices = sorted(int(row["source_token_index"]) for row in active)
        future_values = {bool(row["future_sample_access"]) for row in active}
        onset_values = {bool(row["onset_evidence_authorized"]) for row in active}
        if len(future_values) != 1:
            raise _TypedAdapterFailure(
                "router_incompatible_cell_topology",
                "one physical cell mixes temporal permissions",
            )
        # Authorization is inherited conservatively, never granted by the
        # lane name.  A clinically unqualified but future-free causal view is
        # still a valid model-candidate routing lane; it simply cannot support
        # an onset-positive report claim downstream.
        onset_authorized = bool(onset_values) and all(onset_values)
        cells[cell_id] = {
            "cell_id": cell_id,
            "parent_cell_id": None,
            "scale": scale,
            "permission": permission,
            "nominal_interval_seconds": [nominal[0], nominal[1]],
            "actual_interval_seconds": [actual_start, actual_stop],
            "future_sample_access": next(iter(future_values)),
            "onset_evidence_authorized": onset_authorized,
            "view_ids": sorted({str(row["view_id"]) for row in active}),
            "unit_ids": sorted({str(row["unit_id"]) for row in active}),
            "reference_families": sorted(
                {str(row["reference_family"]) for row in active}
            ),
            "source_token_indices": source_indices,
            "raw_dependency_sha256s": sorted(
                {str(row["p0_token_dependency_sha256"]) for row in active}
            ),
            "view_receipt_sha256s": sorted(
                {str(row["view_receipt_sha256"]) for row in active}
            ),
            "transform_sha256s": sorted(
                {str(row["transform_sha256"]) for row in active}
            ),
            **features,
            "score_source": "deterministic_signal_policy",
            "score_receipt_sha256": score_policy_receipt_sha256,
            "token_cost": len(source_indices),
            "resolution_weighted_eeg_seconds_cost": (
                (nominal[1] - nominal[0])
                * router_policy.scale_resolution_weights[scale]
            ),
        }
        ledger_rows = sorted(
            (_persisted_row(row) for row in rows),
            key=lambda row: int(row["source_token_index"]),
        )
        ledgers[cell_id] = {
            "cell_id": cell_id,
            "nominal_interval_seconds": [nominal[0], nominal[1]],
            "shared_actual_interval_seconds": [actual_start, actual_stop],
            "expected_eligible_unit_count": int(expected_units[permission]),
            "present_source_row_count": len(rows),
            "active_source_row_count": len(active),
            "all_eligible_rows_grouped_before_qc": True,
            "source_rows": ledger_rows,
            "ledger_sha256": "CONTENT-ADDRESS-PENDING",
        }
        ledgers[cell_id]["ledger_sha256"] = _canonical_sha256(ledgers[cell_id])
    return cells, ledgers


def _nonoverlapping(cells: Sequence[Mapping[str, Any]]) -> bool:
    ordered = sorted(
        cells,
        key=lambda item: (
            float(item["nominal_interval_seconds"][0]),
            float(item["nominal_interval_seconds"][1]),
        ),
    )
    return not any(
        _overlaps(left["nominal_interval_seconds"], right["nominal_interval_seconds"])
        for left, right in zip(ordered, ordered[1:])
    )


def _complete_cover(
    parent: Mapping[str, Any], children: Sequence[Mapping[str, Any]]
) -> bool:
    return bool(
        children
        and _nonoverlapping(children)
        and _same_union(
            [child["nominal_interval_seconds"] for child in children],
            [parent["nominal_interval_seconds"]],
        )
    )


def _build_complete_tree_for_permission(
    raw_cells: Mapping[str, Mapping[str, Any]], permission: str
) -> tuple[set[str], dict[str, str], dict[str, str]]:
    """Return included cells, parent assignments and excluded-cell reasons."""

    by_scale: dict[str, list[Mapping[str, Any]]] = {
        scale: sorted(
            (
                cell
                for cell in raw_cells.values()
                if cell["permission"] == permission and cell["scale"] == scale
            ),
            key=lambda cell: (
                float(cell["nominal_interval_seconds"][0]),
                float(cell["nominal_interval_seconds"][1]),
                str(cell["cell_id"]),
            ),
        )
        for scale in BA_IEG_INNER_ROUTER_SCALES
    }
    if any(not _nonoverlapping(items) for items in by_scale.values()):
        raise _TypedAdapterFailure(
            "router_incompatible_cell_topology",
            "same-scale P0 candidate cells overlap",
        )

    included: set[str] = set()
    parents: dict[str, str] = {}
    excluded: dict[str, str] = {}
    root_ids: set[str] = set()

    # Every available full context cell is a root.
    for cell in by_scale["context"]:
        cell_id = str(cell["cell_id"])
        included.add(cell_id)
        root_ids.add(cell_id)

    # Coarse cells either form a complete cover of one context root or become
    # non-overlapping tail/gap roots outside all context support.
    coarse_by_context: dict[str, list[Mapping[str, Any]]] = {}
    for cell in by_scale["coarse"]:
        overlaps = [
            root_id
            for root_id in root_ids
            if _overlaps(
                raw_cells[root_id]["nominal_interval_seconds"],
                cell["nominal_interval_seconds"],
            )
        ]
        containing = [
            root_id
            for root_id in overlaps
            if _contains(
                raw_cells[root_id]["nominal_interval_seconds"],
                cell["nominal_interval_seconds"],
            )
        ]
        if overlaps and len(containing) != 1:
            raise _TypedAdapterFailure(
                "router_incompatible_cell_topology",
                "a coarse P0 cell partially crosses context-root support",
            )
        if containing:
            coarse_by_context.setdefault(containing[0], []).append(cell)
        else:
            cell_id = str(cell["cell_id"])
            included.add(cell_id)
            root_ids.add(cell_id)

    for parent_id, children in coarse_by_context.items():
        if _complete_cover(raw_cells[parent_id], children):
            for child in children:
                child_id = str(child["cell_id"])
                included.add(child_id)
                parents[child_id] = parent_id
        else:
            for child in children:
                excluded[str(child["cell_id"])] = "incomplete_context_to_coarse_cover"

    # Fine cells refine an included coarse cell only as a complete cover.  A
    # fine tail outside all coarser root support remains a root, preserving
    # event-end opportunity when fewer than four seconds remain.
    included_coarse = [
        raw_cells[cell_id]
        for cell_id in included
        if raw_cells[cell_id]["scale"] == "coarse"
    ]
    fine_by_coarse: dict[str, list[Mapping[str, Any]]] = {}
    for cell in by_scale["fine"]:
        containing_coarse = [
            str(parent["cell_id"])
            for parent in included_coarse
            if _contains(
                parent["nominal_interval_seconds"],
                cell["nominal_interval_seconds"],
            )
        ]
        if len(containing_coarse) > 1:
            raise _TypedAdapterFailure(
                "router_incompatible_cell_topology",
                "a fine P0 cell maps to multiple coarse parents",
            )
        if containing_coarse:
            fine_by_coarse.setdefault(containing_coarse[0], []).append(cell)
            continue
        overlapping_roots = [
            root_id
            for root_id in root_ids
            if _overlaps(
                raw_cells[root_id]["nominal_interval_seconds"],
                cell["nominal_interval_seconds"],
            )
        ]
        if overlapping_roots:
            if not any(
                _contains(
                    raw_cells[root_id]["nominal_interval_seconds"],
                    cell["nominal_interval_seconds"],
                )
                for root_id in overlapping_roots
            ):
                raise _TypedAdapterFailure(
                    "router_incompatible_cell_topology",
                    "a fine P0 cell partially crosses coarser root support",
                )
            excluded[str(cell["cell_id"])] = "coarser_root_without_complete_path"
        else:
            cell_id = str(cell["cell_id"])
            included.add(cell_id)
            root_ids.add(cell_id)

    for parent_id, children in fine_by_coarse.items():
        if _complete_cover(raw_cells[parent_id], children):
            for child in children:
                child_id = str(child["cell_id"])
                included.add(child_id)
                parents[child_id] = parent_id
        else:
            for child in children:
                excluded[str(child["cell_id"])] = "incomplete_coarse_to_fine_cover"

    roots = [raw_cells[cell_id]["nominal_interval_seconds"] for cell_id in root_ids]
    all_available = [
        cell["nominal_interval_seconds"]
        for cell in raw_cells.values()
        if cell["permission"] == permission
    ]
    if all_available and not _same_union(roots, all_available):
        raise _TypedAdapterFailure(
            "router_incompatible_cell_topology",
            "coarsest roots do not preserve all available P0 physical support",
        )
    return included, parents, excluded


def _lane_execution_receipt(
    permission: str,
    cells: Sequence[Mapping[str, Any]],
    score_policy_sha256: str,
) -> dict[str, Any]:
    rows = [
        {
            "cell_id": cell["cell_id"],
            "boundary_uncertainty": cell["boundary_uncertainty"],
            "change_density": cell["change_density"],
            "cross_channel_disagreement": cell["cross_channel_disagreement"],
            "reference_instability": cell["reference_instability"],
            "quality_fraction": cell["quality_fraction"],
            "opportunity_fraction": cell["opportunity_fraction"],
            "router_score": cell["router_score"],
            "source_token_indices": cell["source_token_indices"],
        }
        for cell in sorted(cells, key=lambda item: str(item["cell_id"]))
    ]
    body: dict[str, Any] = {
        "schema_version": "ba_ieg_p0_lane_score_execution_receipt_v1",
        "method_id": BA_IEG_P0_INNER_ROUTER_DETERMINISTIC_SCORE_METHOD_ID,
        "permission": permission,
        "score_policy_receipt_sha256": score_policy_sha256,
        "candidate_feature_and_score_rows": rows,
        "other_permission_rows_read": False,
        "labels_annotations_spreadsheets_or_clinical_text_used": False,
        "clinical_claim_authorized": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _empty_lane_metadata() -> list[dict[str, Any]]:
    return [
        {
            "permission": permission,
            "status": "not_evaluable_no_source_view",
            "source_view_ids": [],
            "expected_eligible_unit_count": 0,
            "source_signal_token_count": 0,
            "raw_cell_count": 0,
            "candidate_cell_count": 0,
            "root_cell_ids": [],
            "available_support_union": [],
            "root_support_union": [],
            "excluded_refinement_cells": [],
        }
        for permission in BA_IEG_INNER_ROUTER_PERMISSIONS
    ]


def _finalize_artifact(body: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(body)
    result["artifact_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["artifact_sha256"] = _canonical_sha256(result)
    return validate_ba_ieg_p0_inner_router_candidate_materialization_v1(result)


def _typed_empty_artifact(
    *,
    status_reason: str,
    detail: str,
    event_identity: Mapping[str, Any],
    p0_materialization_receipt_sha256: str,
    p0_policy: BAIEGP0TokenizationPolicy,
    router_policy: BAIEGInnerRaggedRouterPolicyV1,
    outer_support_union: Sequence[Sequence[float]] = (),
    canonical_signal_sha256: str | None = None,
    canonical_receipt_sha256: str | None = None,
    event_input_receipt_sha256: str | None = None,
    outer_support_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    score_policy = _score_policy_receipt()
    return _finalize_artifact(
        {
            "schema_version": BA_IEG_P0_INNER_ROUTER_ADAPTER_SCHEMA_VERSION,
            "method_id": BA_IEG_P0_INNER_ROUTER_ADAPTER_METHOD_ID,
            "status": "not_evaluable",
            "reason_code": status_reason,
            "reason_detail": detail,
            "source_binding": {
                **deepcopy(dict(event_identity)),
                "p0_materialization_receipt_sha256": (
                    p0_materialization_receipt_sha256
                ),
                "event_input_receipt_sha256": event_input_receipt_sha256,
                "canonical_signal_sha256": canonical_signal_sha256,
                "canonical_receipt_sha256": canonical_receipt_sha256,
                "outer_support_receipt_sha256": outer_support_receipt_sha256,
                "target_free_model_input_used": False,
            },
            "policy_binding": {
                "p0_tokenization_policy": p0_policy.to_dict(),
                "p0_tokenization_policy_sha256": p0_policy.sha256,
                "inner_router_policy": router_policy.to_dict(),
                "inner_router_policy_sha256": router_policy.sha256,
            },
            "outer_support_union": [list(item) for item in outer_support_union],
            "candidate_cells": [],
            "cell_source_ledgers": [],
            "permission_lanes": _empty_lane_metadata(),
            "deterministic_score_policy_receipt": score_policy,
            "lane_score_execution_receipts": [
                _lane_execution_receipt(permission, [], score_policy["receipt_sha256"])
                for permission in BA_IEG_INNER_ROUTER_PERMISSIONS
            ],
            "diagnostics": {
                "source_token_count": 0,
                "signal_eligible_source_token_count": 0,
                "candidate_source_token_count": 0,
                "candidate_cell_count": 0,
                "root_support_preserves_available_support": False,
            },
            "scope_receipt": deepcopy(_SCOPE_RECEIPT),
            "artifact_sha256": "CONTENT-ADDRESS-PENDING",
        }
    )


def _materialize_target_free_event(
    event: BAIEGEventTokens,
    *,
    source_p0_materialization_receipt_sha256: str,
    p0_policy: BAIEGP0TokenizationPolicy,
    canonical_signal_receipt: Mapping[str, Any],
    outer_support_receipt_sha256: str,
    outer_support_union: Sequence[Sequence[float]],
    router_policy: BAIEGInnerRaggedRouterPolicyV1,
) -> dict[str, Any]:
    if not isinstance(event, BAIEGEventTokens):
        raise TypeError("P0 inner-router adapter requires BAIEGEventTokens")
    event.verify_integrity()
    if event.deterministic_targets is not None:
        raise ValueError(
            "P0 inner-router adapter requires a target-free model-input event"
        )
    p0_receipt = _sha256(
        source_p0_materialization_receipt_sha256,
        "source_p0_materialization_receipt_sha256",
    )
    canonical = validate_canonical_signal_receipt(canonical_signal_receipt)
    if (
        canonical["receipt_sha256"] != event.canonical_receipt_sha256
        or canonical["recording_id"] != event.recording_id
        or float(event.analysis_interval_seconds[0]) < -_TOLERANCE_SECONDS
        or float(event.analysis_interval_seconds[1])
        > float(canonical["recording_duration_seconds"]) + _TOLERANCE_SECONDS
    ):
        raise ValueError(
            "canonical signal receipt does not bind the target-free P0 event"
        )
    canonical_signal = str(canonical["source_signal_sha256"])
    outer_receipt = _sha256(
        outer_support_receipt_sha256, "outer_support_receipt_sha256"
    )
    if not isinstance(p0_policy, BAIEGP0TokenizationPolicy):
        raise TypeError("p0_policy must be BAIEGP0TokenizationPolicy")
    if not isinstance(router_policy, BAIEGInnerRaggedRouterPolicyV1):
        raise TypeError("router_policy must be BAIEGInnerRaggedRouterPolicyV1")
    event_interval = _finite_interval(
        event.analysis_interval_seconds, "event analysis interval"
    )
    support = _normalize_union(outer_support_union)
    identity = {
        "event_id": event.event_id,
        "recording_id": event.recording_id,
        "patient_uid": event.patient_uid,
        "model_split": event.model_split,
    }
    try:
        _require_v1_grid(p0_policy)
        if len(support) != 1 or not _same_interval(
            support[0], event_interval, tolerance=_TOLERANCE_SECONDS
        ):
            raise _TypedAdapterFailure(
                "outer_support_binding_mismatch",
                "adapter v1 must bind exactly the one P0 analysis interval",
            )
        tiles = _nominal_tiles(event_interval)
        groups, expected_units = _build_source_rows(
            event, tiles, canonical_signal_sha256=canonical_signal
        )
        score_policy = _score_policy_receipt()
        raw_cells, raw_ledgers = _raw_cells(
            event,
            groups,
            expected_units,
            tiles,
            canonical_signal_sha256=canonical_signal,
            score_policy_receipt_sha256=score_policy["receipt_sha256"],
            router_policy=router_policy,
        )
        included: set[str] = set()
        parents: dict[str, str] = {}
        excluded_by_permission: dict[str, dict[str, str]] = {}
        for permission in BA_IEG_INNER_ROUTER_PERMISSIONS:
            (
                lane_included,
                lane_parents,
                lane_excluded,
            ) = _build_complete_tree_for_permission(raw_cells, permission)
            included.update(lane_included)
            parents.update(lane_parents)
            excluded_by_permission[permission] = lane_excluded
        if not included:
            raise _TypedAdapterFailure(
                "no_signal_eligible_router_cells",
                "P0 contains no signal-eligible cell on the registered grid",
            )
    except _TypedAdapterFailure as failure:
        return _typed_empty_artifact(
            status_reason=failure.reason_code,
            detail=failure.detail,
            event_identity=identity,
            p0_materialization_receipt_sha256=p0_receipt,
            p0_policy=p0_policy,
            router_policy=router_policy,
            outer_support_union=support,
            canonical_signal_sha256=canonical_signal,
            canonical_receipt_sha256=event.canonical_receipt_sha256,
            event_input_receipt_sha256=event.input_receipt_sha256,
            outer_support_receipt_sha256=outer_receipt,
        )

    candidates: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    for cell_id in included:
        candidate = deepcopy(raw_cells[cell_id])
        candidate["parent_cell_id"] = parents.get(cell_id)
        candidates.append(candidate)
        ledgers.append(deepcopy(raw_ledgers[cell_id]))
    candidates.sort(
        key=lambda cell: (
            str(cell["permission"]),
            float(cell["nominal_interval_seconds"][0]),
            float(cell["nominal_interval_seconds"][1]),
            str(cell["scale"]),
            str(cell["cell_id"]),
        )
    )
    ledgers.sort(key=lambda ledger: str(ledger["cell_id"]))
    permission_lanes: list[dict[str, Any]] = []
    lane_execution_receipts: list[dict[str, Any]] = []
    for permission in BA_IEG_INNER_ROUTER_PERMISSIONS:
        source_views = sorted(
            event.view_ids[index]
            for index, role in enumerate(event.view_effective_temporal_roles)
            if role == permission
        )
        raw_lane = [
            cell for cell in raw_cells.values() if cell["permission"] == permission
        ]
        lane = [cell for cell in candidates if cell["permission"] == permission]
        roots = [cell for cell in lane if cell["parent_cell_id"] is None]
        if not source_views:
            lane_status = "not_evaluable_no_source_view"
        elif not raw_lane:
            lane_status = "not_evaluable_no_signal_opportunity"
        elif not lane:
            lane_status = "not_evaluable_no_router_compatible_tree"
        else:
            lane_status = "materialized"
        excluded = excluded_by_permission[permission]
        permission_lanes.append(
            {
                "permission": permission,
                "status": lane_status,
                "source_view_ids": source_views,
                "expected_eligible_unit_count": int(expected_units[permission]),
                "source_signal_token_count": sum(
                    bool(event.token_signal_mask[index])
                    and event.view_effective_temporal_roles[
                        int(event.token_view_index[index])
                    ]
                    == permission
                    for index in range(int(event.token_values.shape[0]))
                ),
                "raw_cell_count": len(raw_lane),
                "candidate_cell_count": len(lane),
                "root_cell_ids": sorted(str(cell["cell_id"]) for cell in roots),
                "available_support_union": _normalize_union(
                    [cell["nominal_interval_seconds"] for cell in raw_lane]
                ),
                "root_support_union": _normalize_union(
                    [cell["nominal_interval_seconds"] for cell in roots]
                ),
                "excluded_refinement_cells": [
                    {
                        "cell_id": cell_id,
                        "reason": reason,
                        "source_token_indices": raw_cells[cell_id][
                            "source_token_indices"
                        ],
                    }
                    for cell_id, reason in sorted(excluded.items())
                ],
            }
        )
        lane_execution_receipts.append(
            _lane_execution_receipt(permission, lane, score_policy["receipt_sha256"])
        )

    candidate_indices = [
        int(index)
        for candidate in candidates
        for index in candidate["source_token_indices"]
    ]
    if len(candidate_indices) != len(set(candidate_indices)):
        raise RuntimeError("one P0 source token entered multiple router cells")
    body = {
        "schema_version": BA_IEG_P0_INNER_ROUTER_ADAPTER_SCHEMA_VERSION,
        "method_id": BA_IEG_P0_INNER_ROUTER_ADAPTER_METHOD_ID,
        "status": "materialized",
        "reason_code": None,
        "reason_detail": None,
        "source_binding": {
            **identity,
            "p0_materialization_receipt_sha256": p0_receipt,
            "event_input_receipt_sha256": event.input_receipt_sha256,
            "canonical_signal_sha256": canonical_signal,
            "canonical_receipt_sha256": event.canonical_receipt_sha256,
            "outer_support_receipt_sha256": outer_receipt,
            "target_free_model_input_used": True,
        },
        "policy_binding": {
            "p0_tokenization_policy": p0_policy.to_dict(),
            "p0_tokenization_policy_sha256": p0_policy.sha256,
            "inner_router_policy": router_policy.to_dict(),
            "inner_router_policy_sha256": router_policy.sha256,
        },
        "outer_support_union": support,
        "candidate_cells": candidates,
        "cell_source_ledgers": ledgers,
        "permission_lanes": permission_lanes,
        "deterministic_score_policy_receipt": score_policy,
        "lane_score_execution_receipts": lane_execution_receipts,
        "diagnostics": {
            "source_token_count": int(event.token_values.shape[0]),
            "signal_eligible_source_token_count": int(event.token_signal_mask.sum()),
            "candidate_source_token_count": len(candidate_indices),
            "candidate_cell_count": len(candidates),
            "root_support_preserves_available_support": all(
                _same_union(lane["available_support_union"], lane["root_support_union"])
                for lane in permission_lanes
            ),
        },
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "artifact_sha256": "CONTENT-ADDRESS-PENDING",
    }
    return _finalize_artifact(body)


def materialize_ba_ieg_target_free_p0_inner_router_candidates_v1(
    event: BAIEGEventTokens,
    *,
    source_p0_materialization_receipt_sha256: str,
    p0_policy: BAIEGP0TokenizationPolicy,
    canonical_signal_receipt: Mapping[str, Any],
    outer_support_receipt_sha256: str,
    outer_support_union: Sequence[Sequence[float]],
    router_policy: BAIEGInnerRaggedRouterPolicyV1 = BAIEGInnerRaggedRouterPolicyV1(),
) -> dict[str, Any]:
    """Adapt one already target-detached P0 event into router candidates."""

    return _materialize_target_free_event(
        event,
        source_p0_materialization_receipt_sha256=(
            source_p0_materialization_receipt_sha256
        ),
        p0_policy=p0_policy,
        canonical_signal_receipt=canonical_signal_receipt,
        outer_support_receipt_sha256=outer_support_receipt_sha256,
        outer_support_union=outer_support_union,
        router_policy=router_policy,
    )


def materialize_ba_ieg_p0_inner_router_candidates_v1(
    source: BAIEGEventModelInputProjectionV1 | BAIEGP0MaterializationResult,
    *,
    canonical_signal_receipt: Mapping[str, Any] | None = None,
    outer_support_receipt_sha256: str | None = None,
    outer_support_union: Sequence[Sequence[float]] | None = None,
    p0_policy: BAIEGP0TokenizationPolicy | None = None,
    router_policy: BAIEGInnerRaggedRouterPolicyV1 = BAIEGInnerRaggedRouterPolicyV1(),
) -> dict[str, Any]:
    """Project a real P0 result and materialize channel-neutral router cells.

    A failed P0 result returns a typed empty artifact.  A successful raw P0
    result is first passed through the registered target-detachment projection.
    Callers with an existing projection must supply its frozen P0 policy.
    ``canonical_signal_receipt`` is explicit because its receipt hash is not
    interchangeable with the source-signal hash required by the router.
    """

    if not isinstance(router_policy, BAIEGInnerRaggedRouterPolicyV1):
        raise TypeError("router_policy must be BAIEGInnerRaggedRouterPolicyV1")
    projection: BAIEGEventModelInputProjectionV1
    source_p0_receipt: Mapping[str, Any] | None = None
    if isinstance(source, BAIEGP0MaterializationResult):
        receipt = source.receipt
        source_p0_receipt = receipt
        policy = _policy_from_dict(receipt["policy"])
        if receipt["status"] != "materialized" or source.event_tokens is None:
            timing = receipt.get("timing")
            support: list[list[float]] = []
            if isinstance(timing, Mapping):
                interval = timing.get("requested_analysis_interval_seconds")
                if isinstance(interval, list) and len(interval) == 2:
                    try:
                        support = [
                            list(_finite_interval(interval, "failed P0 interval"))
                        ]
                    except ValueError:
                        support = []
            return _typed_empty_artifact(
                status_reason="p0_materialization_failed",
                detail=(f"{receipt['failure_code']} at {receipt['failure_stage']}"),
                event_identity=receipt["event_identity"],
                p0_materialization_receipt_sha256=receipt["receipt_sha256"],
                p0_policy=policy,
                router_policy=router_policy,
                outer_support_union=support,
            )
        projection = project_ba_ieg_event_model_input_v1(source)
        if p0_policy is not None and p0_policy.to_dict() != policy.to_dict():
            raise ValueError("caller P0 policy disagrees with materialization receipt")
        p0_policy = policy
    elif isinstance(source, BAIEGEventModelInputProjectionV1):
        source.verify_integrity()
        projection = source
        if not isinstance(p0_policy, BAIEGP0TokenizationPolicy):
            raise TypeError("an existing P0 projection requires p0_policy")
    else:
        raise TypeError(
            "source must be BAIEGP0MaterializationResult or "
            "BAIEGEventModelInputProjectionV1"
        )

    if canonical_signal_receipt is None:
        raise ValueError("canonical_signal_receipt is required for a materialized P0")
    canonical = validate_canonical_signal_receipt(canonical_signal_receipt)
    if source_p0_receipt is not None and (
        source_p0_receipt["lineage"].get("canonical_receipt_sha256")
        != canonical["receipt_sha256"]
    ):
        raise ValueError("P0 lineage and canonical signal receipt disagree")
    if outer_support_receipt_sha256 is None:
        raise ValueError(
            "outer_support_receipt_sha256 is required for a materialized P0"
        )
    event = projection.model_input_event
    if outer_support_union is None:
        outer_support_union = [list(event.analysis_interval_seconds)]
    return _materialize_target_free_event(
        event,
        source_p0_materialization_receipt_sha256=(
            projection.source_p0_materialization_receipt_sha256
        ),
        p0_policy=p0_policy,
        canonical_signal_receipt=canonical_signal_receipt,
        outer_support_receipt_sha256=outer_support_receipt_sha256,
        outer_support_union=outer_support_union,
        router_policy=router_policy,
    )


def route_ba_ieg_p0_inner_router_candidates_v1(
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay a successful adapter artifact through the registered router."""

    artifact = validate_ba_ieg_p0_inner_router_candidate_materialization_v1(
        materialization
    )
    if artifact["status"] != "materialized":
        raise ValueError("a not-evaluable adapter artifact cannot enter the router")
    source = artifact["source_binding"]
    policy = BAIEGInnerRaggedRouterPolicyV1.from_dict(
        artifact["policy_binding"]["inner_router_policy"]
    )
    return materialize_ba_ieg_inner_ragged_router_v1(
        event_id=source["event_id"],
        canonical_signal_sha256=source["canonical_signal_sha256"],
        outer_support_receipt_sha256=source["outer_support_receipt_sha256"],
        outer_support_union=artifact["outer_support_union"],
        candidate_cells=artifact["candidate_cells"],
        policy=policy,
    )


def validate_ba_ieg_p0_inner_router_candidate_materialization_v1(
    payload: object,
) -> dict[str, Any]:
    """Validate content binding and the downstream router contract."""

    required = {
        "schema_version",
        "method_id",
        "status",
        "reason_code",
        "reason_detail",
        "source_binding",
        "policy_binding",
        "outer_support_union",
        "candidate_cells",
        "cell_source_ledgers",
        "permission_lanes",
        "deterministic_score_policy_receipt",
        "lane_score_execution_receipts",
        "diagnostics",
        "scope_receipt",
        "artifact_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("P0 inner-router adapter artifact has missing/unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != BA_IEG_P0_INNER_ROUTER_ADAPTER_SCHEMA_VERSION:
        raise ValueError("P0 inner-router adapter schema drifted")
    if data["method_id"] != BA_IEG_P0_INNER_ROUTER_ADAPTER_METHOD_ID:
        raise ValueError("P0 inner-router adapter method drifted")
    if data["status"] not in {"materialized", "not_evaluable"}:
        raise ValueError("P0 inner-router adapter status is invalid")
    if data["reason_code"] not in _ADAPTER_REASON_CODES:
        raise ValueError("P0 inner-router adapter reason code is invalid")
    if (data["status"] == "materialized") != (data["reason_code"] is None):
        raise ValueError("adapter status and reason code disagree")
    if data["scope_receipt"] != _SCOPE_RECEIPT:
        raise ValueError("P0 inner-router adapter violated its EEG-only scope")
    source = data["source_binding"]
    if type(source) is not dict or set(source) != {
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
        "p0_materialization_receipt_sha256",
        "event_input_receipt_sha256",
        "canonical_signal_sha256",
        "canonical_receipt_sha256",
        "outer_support_receipt_sha256",
        "target_free_model_input_used",
    }:
        raise ValueError("P0 inner-router adapter source binding is invalid")
    _sha256(
        source["p0_materialization_receipt_sha256"],
        "source.p0_materialization_receipt_sha256",
    )
    if data["status"] == "materialized":
        for name in (
            "event_input_receipt_sha256",
            "canonical_signal_sha256",
            "canonical_receipt_sha256",
            "outer_support_receipt_sha256",
        ):
            _sha256(source[name], f"source.{name}")
        if source["target_free_model_input_used"] is not True:
            raise ValueError("materialized adapter did not use target-free input")
    elif source["target_free_model_input_used"] is not False:
        raise ValueError("empty adapter cannot claim a target-free model forward")
    policy_binding = data["policy_binding"]
    if type(policy_binding) is not dict or set(policy_binding) != {
        "p0_tokenization_policy",
        "p0_tokenization_policy_sha256",
        "inner_router_policy",
        "inner_router_policy_sha256",
    }:
        raise ValueError("P0 inner-router adapter policy binding is invalid")
    p0_policy = _policy_from_dict(policy_binding["p0_tokenization_policy"])
    if p0_policy.sha256 != policy_binding["p0_tokenization_policy_sha256"]:
        raise ValueError("P0 adapter tokenization policy hash drifted")
    router_policy = BAIEGInnerRaggedRouterPolicyV1.from_dict(
        policy_binding["inner_router_policy"]
    )
    if router_policy.sha256 != policy_binding["inner_router_policy_sha256"]:
        raise ValueError("P0 adapter router policy hash drifted")
    support = data["outer_support_union"]
    if not isinstance(support, list):
        raise TypeError("P0 adapter outer support must be an array")
    if support and _normalize_union(support) != support:
        raise ValueError("P0 adapter outer support union is not canonical")
    candidates = data["candidate_cells"]
    ledgers = data["cell_source_ledgers"]
    if not isinstance(candidates, list) or not isinstance(ledgers, list):
        raise TypeError("P0 adapter candidates/ledgers must be arrays")
    score_policy = data["deterministic_score_policy_receipt"]
    if score_policy != _score_policy_receipt():
        raise ValueError("P0 adapter deterministic score policy receipt drifted")
    policy_sha = score_policy["receipt_sha256"]
    if any(cell.get("score_receipt_sha256") != policy_sha for cell in candidates):
        raise ValueError("P0 adapter candidate lost deterministic score policy")
    ledger_by_id: dict[str, Mapping[str, Any]] = {}
    ledger_token_indices: set[int] = set()
    for ledger in ledgers:
        if type(ledger) is not dict or ledger.get("cell_id") in ledger_by_id:
            raise ValueError("P0 adapter cell ledgers are invalid")
        digest_source = deepcopy(ledger)
        supplied = digest_source.get("ledger_sha256")
        digest_source["ledger_sha256"] = "CONTENT-ADDRESS-PENDING"
        if supplied != _canonical_sha256(digest_source):
            raise ValueError("P0 adapter source ledger hash drifted")
        ledger_by_id[str(ledger["cell_id"])] = ledger
    for cell in candidates:
        cell_id = str(cell["cell_id"])
        expected_cell_id = _cell_id(
            source["recording_id"],
            source["event_id"],
            str(cell["permission"]),
            str(cell["scale"]),
            tuple(float(item) for item in cell["nominal_interval_seconds"]),
        )
        if cell_id != expected_cell_id:
            raise ValueError("P0 adapter cell identity is not coordinate-stable")
        if cell_id not in ledger_by_id:
            raise ValueError("P0 adapter candidate lacks its source-row ledger")
        ledger = ledger_by_id[cell_id]
        for row in ledger["source_rows"]:
            if (
                row["canonical_signal_sha256"] != source["canonical_signal_sha256"]
                or row["canonical_receipt_sha256"] != source["canonical_receipt_sha256"]
            ):
                raise ValueError("P0 adapter row escaped its canonical signal root")
            expected_dependency = _transitive_token_dependency_sha256(
                recording_id=source["recording_id"],
                event_id=source["event_id"],
                permission=str(cell["permission"]),
                scale=str(cell["scale"]),
                nominal_interval_seconds=ledger["nominal_interval_seconds"],
                actual_interval_seconds=row["actual_interval_seconds"],
                view_id=str(row["view_id"]),
                view_receipt_sha256=str(row["view_receipt_sha256"]),
                transform_sha256=str(row["transform_sha256"]),
                physical_reference_row_sha256=str(row["physical_reference_row_sha256"]),
                unit_id=str(row["unit_id"]),
                unit_source_id=str(row["unit_source_id"]),
                reference_family=str(row["reference_family"]),
                token_values_sha256=str(row["token_values_sha256"]),
                token_feature_mask_sha256=str(row["token_feature_mask_sha256"]),
                canonical_signal_sha256=source["canonical_signal_sha256"],
            )
            if row["p0_token_dependency_sha256"] != expected_dependency:
                raise ValueError("P0 adapter token dependency receipt drifted")
        active = sorted(
            int(row["source_token_index"])
            for row in ledger["source_rows"]
            if row["signal_eligible"]
        )
        if active != cell["source_token_indices"]:
            raise ValueError("P0 adapter candidate/source ledger indices drifted")
        dependencies = sorted(
            str(row["p0_token_dependency_sha256"])
            for row in ledger["source_rows"]
            if row["signal_eligible"]
        )
        if dependencies != cell["raw_dependency_sha256s"]:
            raise ValueError("P0 adapter candidate/source dependency ledger drifted")
        if ledger_token_indices.intersection(active):
            raise ValueError("one P0 source token belongs to multiple candidate cells")
        ledger_token_indices.update(active)
    if set(ledger_by_id) != {str(cell["cell_id"]) for cell in candidates}:
        raise ValueError("P0 adapter persisted a ledger for an absent candidate")
    lane_receipts = data["lane_score_execution_receipts"]
    if not isinstance(lane_receipts, list) or len(lane_receipts) != len(
        BA_IEG_INNER_ROUTER_PERMISSIONS
    ):
        raise ValueError("P0 adapter lane score receipts are incomplete")
    for permission, receipt in zip(BA_IEG_INNER_ROUTER_PERMISSIONS, lane_receipts):
        digest_source = deepcopy(receipt)
        supplied = digest_source.get("receipt_sha256")
        digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
        if supplied != _canonical_sha256(digest_source):
            raise ValueError("P0 adapter lane score receipt hash drifted")
        if receipt.get("score_policy_receipt_sha256") != policy_sha:
            raise ValueError("P0 adapter lane score receipt lost its policy")
        if receipt.get("other_permission_rows_read") is not False:
            raise ValueError("P0 adapter lane score crossed a permission boundary")
        expected_receipt = _lane_execution_receipt(
            permission,
            [cell for cell in candidates if cell["permission"] == permission],
            policy_sha,
        )
        if receipt != expected_receipt:
            raise ValueError("P0 adapter lane score execution does not replay")
    permissions = data["permission_lanes"]
    if not isinstance(permissions, list) or [
        item.get("permission") for item in permissions
    ] != list(BA_IEG_INNER_ROUTER_PERMISSIONS):
        raise ValueError("P0 adapter permission-lane ledger is incomplete")
    if data["status"] == "materialized":
        if not candidates or not support:
            raise ValueError("materialized P0 adapter needs candidates and support")
        # This call performs the authoritative candidate normalization, unique
        # token ownership and complete parent-child-cover validation.
        materialize_ba_ieg_inner_ragged_router_v1(
            event_id=source["event_id"],
            canonical_signal_sha256=source["canonical_signal_sha256"],
            outer_support_receipt_sha256=source["outer_support_receipt_sha256"],
            outer_support_union=support,
            candidate_cells=candidates,
            policy=router_policy,
        )
    elif candidates or ledgers:
        raise ValueError("not-evaluable P0 adapter cannot claim router candidates")
    supplied_hash = _sha256(data["artifact_sha256"], "artifact_sha256")
    digest_source = deepcopy(data)
    digest_source["artifact_sha256"] = "CONTENT-ADDRESS-PENDING"
    if supplied_hash != _canonical_sha256(digest_source):
        raise ValueError("P0 inner-router adapter artifact hash drifted")
    return data


__all__ = [
    "BA_IEG_P0_INNER_ROUTER_ADAPTER_METHOD_ID",
    "BA_IEG_P0_INNER_ROUTER_ADAPTER_SCHEMA_VERSION",
    "BA_IEG_P0_INNER_ROUTER_DETERMINISTIC_SCORE_METHOD_ID",
    "materialize_ba_ieg_p0_inner_router_candidates_v1",
    "materialize_ba_ieg_target_free_p0_inner_router_candidates_v1",
    "route_ba_ieg_p0_inner_router_candidates_v1",
    "validate_ba_ieg_p0_inner_router_candidate_materialization_v1",
]
