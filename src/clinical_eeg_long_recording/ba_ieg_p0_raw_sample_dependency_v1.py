"""Host-replayed raw-sample dependency sidecar for BA-IEG P0 tokens.

This artifact is provenance metadata, not a model input.  It binds every P0
token to its trusted canonical/view receipts and keeps three intervals
separate:

* the event token's float recording-time interval;
* the exact integer output/tensor support replayed on the view clock;
* the raw canonical-channel samples on which that output depends.

Native morphology has exact instantaneous support.  A causal FIR view has a
finite past-and-present support expanded by the trusted filter order.  Offline
zero-phase/resampled context deliberately uses the complete recording as a
conservative closure.  Clinical onset authorization is recorded but never
participates in choosing or validating the raw-support rule.

The validator requires the event, canonical receipt, and complete trusted view
registry from the host.  Embedded hashes alone are not treated as authority.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch

from .ba_ieg_training_contract import BA_IEG_TOKEN_SCALES, BAIEGEventTokens
from .canonical_signal_views import (
    validate_canonical_signal_receipt,
    validate_signal_view_receipt,
)


BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_SCHEMA_VERSION_V1: Final[
    str
] = "ba_ieg_p0_raw_sample_dependency_v1"
BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_SIDECAR_SCHEMA_VERSION_V1: Final[
    str
] = "ba_ieg_p0_raw_sample_dependency_sidecar_v1"
BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_METHOD_ID_V1: Final[
    str
] = "ba_ieg_p0_host_replayed_raw_sample_dependency_v1"

_DEPENDENCY_ID_DOMAIN: Final[str] = "ba-ieg-p0-raw-dependency-id-v1"
_DEPENDENCY_SHA_DOMAIN: Final[str] = "ba-ieg-p0-raw-dependency-sha-v1"
_SIDECAR_ID_DOMAIN: Final[str] = "ba-ieg-p0-raw-dependency-sidecar-id-v1"
_SIDECAR_SHA_DOMAIN: Final[str] = "ba-ieg-p0-raw-dependency-sidecar-sha-v1"
_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_COEFFICIENT_TOLERANCE: Final[float] = 1e-12
_OUTPUT_EDGE_TOLERANCE_SAMPLES: Final[float] = 0.25

_SCOPE_RECEIPT: Final[dict[str, bool]] = {
    "artifact_supplied_to_model": False,
    "event_model_input_receipt_bound": True,
    "event_token_coordinates_read": True,
    "event_model_feature_values_read": False,
    "deterministic_measurement_targets_read": False,
    "raw_signal_samples_read": False,
    "host_canonical_receipt_required": True,
    "host_view_receipt_registry_required": True,
    "edf_annotation_used": False,
    "spreadsheet_used": False,
    "clinical_text_used": False,
    "clinical_onset_authorization_used_for_raw_closure": False,
}

_DEPENDENCY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "dependency_id",
        "dependency_sha256",
        "source_token_index",
        "source_binding",
        "token_coordinate",
        "output_support",
        "temporal_contract",
        "reference_lineage",
        "raw_support",
    }
)
_SIDECAR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "method_id",
        "sidecar_id",
        "sidecar_sha256",
        "source_binding",
        "dependencies",
        "dependency_roster_sha256",
        "scope_receipt",
    }
)


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


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _canonical_float(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("reference coefficients must be finite")
    return 0.0 if abs(result) <= _COEFFICIENT_TOLERANCE else result


def _matrix_multiply(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    if not left or not right or not right[0]:
        raise ValueError("reference matrices must be non-empty")
    inner = len(right)
    if any(len(row) != inner for row in left):
        raise ValueError("reference matrices have incompatible dimensions")
    columns = len(right[0])
    if any(len(row) != columns for row in right):
        raise ValueError("reference matrix rows must have equal length")
    result: list[list[float]] = []
    for left_row in left:
        output_row = []
        for column in range(columns):
            value = sum(
                float(left_row[index]) * float(right[index][column])
                for index in range(inner)
            )
            output_row.append(_canonical_float(value))
        result.append(output_row)
    return result


def _finalize_dependency(body: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result["dependency_id"] = "CONTENT-ADDRESS-PENDING"
    result["dependency_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["dependency_id"] = (
        "P0-RAWDEP-"
        + _canonical_sha256({"domain": _DEPENDENCY_ID_DOMAIN, "dependency": result})[
            :24
        ]
    )
    result["dependency_sha256"] = _canonical_sha256(
        {"domain": _DEPENDENCY_SHA_DOMAIN, "dependency": result}
    )
    return result


def _dependency_roster_sha256(
    dependencies: Sequence[Mapping[str, Any]],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "ba_ieg_p0_raw_dependency_roster_v1",
            "rows": [
                {
                    "source_token_index": int(row["source_token_index"]),
                    "dependency_id": str(row["dependency_id"]),
                    "dependency_sha256": str(row["dependency_sha256"]),
                }
                for row in dependencies
            ],
        }
    )


def _finalize_sidecar(body: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result["sidecar_id"] = "CONTENT-ADDRESS-PENDING"
    result["sidecar_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["sidecar_id"] = (
        "P0-RAWDEPS-"
        + _canonical_sha256({"domain": _SIDECAR_ID_DOMAIN, "sidecar": result})[:24]
    )
    result["sidecar_sha256"] = _canonical_sha256(
        {"domain": _SIDECAR_SHA_DOMAIN, "sidecar": result}
    )
    return result


def _validate_event_model_input(event: BAIEGEventTokens) -> None:
    if not isinstance(event, BAIEGEventTokens):
        raise TypeError("event_tokens must be BAIEGEventTokens")
    # The model-input digest intentionally excludes deterministic supervision.
    # Do not call verify_integrity(), which also traverses the optional target
    # sidecar and would make this provenance artifact target-dependent.
    if event.input_receipt_sha256 != event._compute_input_sha256():
        raise ValueError("BA-IEG event model input changed after registration")


def _validate_trusted_view_registry(
    event: BAIEGEventTokens,
    canonical: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(registry, Mapping):
        raise TypeError("trusted_view_receipts must be a host-supplied mapping")
    raw: dict[str, dict[str, Any]] = {}
    for key, value in registry.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise TypeError("trusted view registry entries are invalid")
        receipt = deepcopy(dict(value))
        if receipt.get("view_id") != key:
            raise ValueError("trusted view registry key/view_id mismatch")
        raw[key] = receipt
    if set(raw) != set(event.view_ids):
        raise ValueError(
            "trusted view registry must exactly cover the P0 event view roster"
        )

    validated: dict[str, dict[str, Any]] = {}
    visiting: set[str] = set()

    def visit(view_id: str) -> dict[str, Any]:
        if view_id in validated:
            return validated[view_id]
        if view_id in visiting:
            raise ValueError("trusted view registry contains a parent cycle")
        visiting.add(view_id)
        receipt = raw[view_id]
        bindings = receipt.get("parent_view_bindings")
        if not isinstance(bindings, list):
            raise TypeError("trusted view parent bindings must be an array")
        parent_ids: list[str] = []
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise TypeError("trusted view parent binding is invalid")
            parent_id = _identifier(binding.get("view_id"), "parent view_id")
            if parent_id not in raw:
                raise ValueError("trusted view parent is absent from the host registry")
            parent_ids.append(parent_id)
            visit(parent_id)
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("trusted view repeats a parent")
        trusted_parents = {parent_id: validated[parent_id] for parent_id in parent_ids}
        normalized = validate_signal_view_receipt(
            receipt,
            canonical,
            trusted_parent_views=trusted_parents,
        )
        validated[view_id] = normalized
        visiting.remove(view_id)
        return normalized

    for view_id in event.view_ids:
        visit(str(view_id))
    return validated


def _reference_family(view: Mapping[str, Any]) -> str:
    if not view["parent_view_bindings"]:
        return "referential"
    reference_type = str(view["transform_spec"]["reference"]["reference_type"])
    mapping = {
        "longitudinal_bipolar_tcp20_frozen_carriers_v1": "bipolar",
        "common_average_standard19_frozen_all_carriers_v1": "common_average",
        "surface_laplacian_standard19_frozen_neighbour_graph_v1": "laplacian",
    }
    if reference_type not in mapping:
        raise ValueError("P0 view has an unsupported reference transform")
    return mapping[reference_type]


def _root_view_ids(
    view_id: str, views: Mapping[str, Mapping[str, Any]]
) -> tuple[str, ...]:
    parents = [str(row["view_id"]) for row in views[view_id]["parent_view_bindings"]]
    if not parents:
        return (view_id,)
    roots = sorted(
        {root for parent_id in parents for root in _root_view_ids(parent_id, views)}
    )
    return tuple(roots)


def _view_lineage(
    view_id: str, views: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    visited: set[str] = set()

    def append(view_key: str) -> None:
        if view_key in visited:
            return
        for binding in views[view_key]["parent_view_bindings"]:
            append(str(binding["view_id"]))
        receipt = views[view_key]
        result.append(
            {
                "view_id": view_key,
                "view_receipt_id": str(receipt["view_receipt_id"]),
                "view_receipt_sha256": str(receipt["receipt_sha256"]),
                "transform_spec_sha256": str(
                    receipt["transform_spec"]["transform_spec_sha256"]
                ),
            }
        )
        visited.add(view_key)

    append(view_id)
    return result


def _effective_reference_matrices(
    views: Mapping[str, Mapping[str, Any]],
    physical_ids: Sequence[str],
) -> dict[str, tuple[tuple[str, ...], list[list[float]]]]:
    physical_index = {name: index for index, name in enumerate(physical_ids)}
    if len(physical_index) != len(physical_ids):
        raise ValueError("physical EEG basis contains duplicate IDs")
    memo: dict[str, tuple[tuple[str, ...], list[list[float]]]] = {}
    visiting: set[str] = set()

    def compute(view_id: str) -> tuple[tuple[str, ...], list[list[float]]]:
        if view_id in memo:
            return memo[view_id]
        if view_id in visiting:
            raise ValueError("view reference graph contains a cycle")
        visiting.add(view_id)
        view = views[view_id]
        transform = view["transform_spec"]
        input_ids = tuple(str(item) for item in transform["input_unit_ids"])
        output_ids = tuple(str(item) for item in transform["output_unit_ids"])
        local_matrix = [
            [_canonical_float(value) for value in row]
            for row in transform["reference"]["matrix"]
        ]
        parents = [str(item["view_id"]) for item in view["parent_view_bindings"]]
        source_rows: dict[str, list[float]] = {}
        if not parents:
            for input_id in input_ids:
                if input_id not in physical_index:
                    raise ValueError(
                        "direct P0 task view input is outside the physical EEG basis"
                    )
                row = [0.0] * len(physical_ids)
                row[physical_index[input_id]] = 1.0
                source_rows[input_id] = row
        else:
            for parent_id in parents:
                parent_outputs, parent_matrix = compute(parent_id)
                for unit_id, row in zip(parent_outputs, parent_matrix):
                    if unit_id in source_rows:
                        raise ValueError(
                            "P0 parent views expose duplicate source unit IDs"
                        )
                    source_rows[unit_id] = row
            if any(input_id not in source_rows for input_id in input_ids):
                raise ValueError(
                    "P0 reference transform lacks a trusted parent carrier"
                )
        input_matrix = [source_rows[input_id] for input_id in input_ids]
        effective = _matrix_multiply(local_matrix, input_matrix)
        memo[view_id] = (output_ids, effective)
        visiting.remove(view_id)
        return memo[view_id]

    for key in views:
        compute(key)
    return memo


def _validate_event_view_bindings(
    event: BAIEGEventTokens,
    views: Mapping[str, Mapping[str, Any]],
) -> None:
    root_role = {
        "findings_native": "morphology_native",
        "findings_native_morphology": "morphology_native",
        "onset_causal": "onset_causal",
        "findings_clinical": "context_offline",
        "context_offline": "context_offline",
    }
    for index, view_id in enumerate(event.view_ids):
        view = views[str(view_id)]
        temporal = view["temporal_evidence"]
        roots = _root_view_ids(str(view_id), views)
        if len(roots) != 1:
            raise ValueError("one P0 view cannot mix temporal task roots")
        root_task_role = str(views[roots[0]]["task_role"])
        expected_effective_role = root_role.get(root_task_role)
        if expected_effective_role is None:
            raise ValueError("P0 view has an unsupported task root")
        expected = {
            "task_role": str(view["task_role"]),
            "effective_temporal_role": expected_effective_role,
            "dependency_policy": str(temporal["dependency_policy"]),
            "future_sample_access": bool(temporal["future_sample_access"]),
            "onset_evidence_authorized": bool(temporal["onset_evidence_authorized"]),
            "temporal_evidence_sha256": _canonical_sha256(temporal),
            "receipt_sha256": str(view["receipt_sha256"]),
            "transform_sha256": str(view["transform_spec"]["transform_spec_sha256"]),
            "reference_family": _reference_family(view),
        }
        actual = {
            "task_role": event.view_roles[index],
            "effective_temporal_role": event.view_effective_temporal_roles[index],
            "dependency_policy": event.view_dependency_policies[index],
            "future_sample_access": bool(event.view_future_sample_access[index]),
            "onset_evidence_authorized": bool(
                event.view_onset_evidence_authorized[index]
            ),
            "temporal_evidence_sha256": event.view_temporal_evidence_sha256s[index],
            "receipt_sha256": event.view_receipt_sha256s[index],
            "transform_sha256": event.view_transform_sha256s[index],
            "reference_family": event.reference_families[index],
        }
        if actual != expected:
            raise ValueError("P0 event view binding disagrees with the trusted receipt")


def _support_contract(
    view_id: str,
    effective_role: str,
    views: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, int | None]:
    lineage_ids = [row["view_id"] for row in _view_lineage(view_id, views)]
    roots = _root_view_ids(view_id, views)
    if len(roots) != 1:
        raise ValueError("P0 temporal support has multiple task roots")
    root = views[roots[0]]
    root_transform = root["transform_spec"]

    if effective_role == "context_offline":
        return (
            "conservative_whole_record_future_dependent_v1",
            "conservative_complete_recording",
            None,
        )

    # Native and causal closure are exact only when every transform after the
    # task root is instantaneous and introduces no resampling.
    for lineage_id in lineage_ids:
        transform = views[lineage_id]["transform_spec"]
        if lineage_id == roots[0] and effective_role == "onset_causal":
            continue
        if (
            transform["filter"]["family"] != "none"
            or transform["filter"]["phase_policy"] != "none"
            or transform["resampler"]["implementation"] != "none"
            or int(transform["resampler"]["up"]) != 1
            or int(transform["resampler"]["down"]) != 1
        ):
            raise ValueError("instantaneous P0 reference path has temporal support")

    if effective_role == "morphology_native":
        return "exact_instantaneous_identity_v1", "exact", None

    if effective_role != "onset_causal":
        raise ValueError("P0 token has an unsupported temporal permission")
    filter_spec = root_transform["filter"]
    resampler = root_transform["resampler"]
    order = filter_spec["order"]
    if (
        filter_spec["family"] != "fir"
        or filter_spec["phase_policy"] != "causal_with_group_delay_receipt"
        or isinstance(order, bool)
        or not isinstance(order, int)
        or order < 1
        or resampler["implementation"] != "none"
        or int(resampler["up"]) != 1
        or int(resampler["down"]) != 1
        or int(root["temporal_evidence"]["warm_up_samples"]) != order
    ):
        raise ValueError("causal P0 root lacks an exact finite FIR receipt")
    return "bounded_past_and_present_exact_fir_v1", "exact_finite", order


def _nearest_output_edge(seconds: float, numerator: int, denominator: int) -> int:
    position = float(seconds) * numerator / denominator
    edge = int(round(position))
    if abs(position - edge) > _OUTPUT_EDGE_TOLERANCE_SAMPLES:
        raise ValueError("P0 token time is not bound to a view output sample edge")
    return edge


def _outward_raw_interval(
    output_start: int,
    output_stop: int,
    *,
    output_numerator: int,
    output_denominator: int,
    raw_numerator: int,
    raw_denominator: int,
) -> tuple[int, int]:
    denominator = output_numerator * raw_denominator
    start_numerator = output_start * output_denominator * raw_numerator
    stop_numerator = output_stop * output_denominator * raw_numerator
    start = start_numerator // denominator
    stop = (stop_numerator + denominator - 1) // denominator
    return int(start), int(stop)


def _copy_neutral_reference_fingerprint(row: Sequence[float]) -> str:
    norm = sum(abs(float(value)) for value in row)
    if not math.isfinite(norm) or norm <= _COEFFICIENT_TOLERANCE:
        raise ValueError("effective physical reference row is empty")
    normalized = [float(value) / norm for value in row]
    first = next(value for value in normalized if abs(value) > _COEFFICIENT_TOLERANCE)
    if first < 0.0:
        normalized = [-value for value in normalized]
    return _canonical_sha256([round(value, 12) for value in normalized])


def _prepare_host(
    event: BAIEGEventTokens,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, tuple[tuple[str, ...], list[list[float]]]],
]:
    _validate_event_model_input(event)
    canonical = validate_canonical_signal_receipt(canonical_signal_receipt)
    if (
        event.recording_id != canonical["recording_id"]
        or event.canonical_receipt_sha256 != canonical["receipt_sha256"]
    ):
        raise ValueError("P0 event is not bound to the host canonical signal")
    channel_catalog = {
        str(channel["channel_id"]): channel for channel in canonical["channels"]
    }
    if set(event.physical_electrode_ids) != set(channel_catalog):
        raise ValueError("P0 physical basis and canonical channel roster disagree")
    expected_physical_mask = torch.tensor(
        [
            bool(channel_catalog[channel_id]["observed"])
            for channel_id in event.physical_electrode_ids
        ],
        dtype=torch.bool,
    )
    if not torch.equal(event.physical_evidence_mask.cpu(), expected_physical_mask):
        raise ValueError("P0 physical evidence mask drifted from the canonical receipt")
    views = _validate_trusted_view_registry(event, canonical, trusted_view_receipts)
    _validate_event_view_bindings(event, views)
    matrices = _effective_reference_matrices(views, event.physical_electrode_ids)
    return canonical, views, matrices


def _unit_binding(
    event: BAIEGEventTokens,
    unit_index: int,
    views: Mapping[str, Mapping[str, Any]],
    matrices: Mapping[str, tuple[tuple[str, ...], list[list[float]]]],
) -> dict[str, Any]:
    view_index = int(event.unit_view_index[unit_index])
    view_id = str(event.view_ids[view_index])
    view = views[view_id]
    output_ids, effective_matrix = matrices[view_id]
    source_unit_id = str(event.unit_source_ids[unit_index])
    if output_ids.count(source_unit_id) != 1:
        raise ValueError("P0 unit does not map uniquely to a trusted view output")
    row_index = output_ids.index(source_unit_id)
    output = view["output_units"][row_index]
    if (
        event.unit_ids[unit_index] != f"{view_id}::{source_unit_id}"
        or event.unit_types[unit_index] != output["unit_type"]
        or bool(event.unit_evidence_mask[unit_index])
        and not bool(output["evidence_eligible"])
    ):
        raise ValueError("P0 unit metadata was rebound from its trusted view row")
    effective_row = [_canonical_float(value) for value in effective_matrix[row_index]]
    expected_tensor_row = torch.tensor(effective_row, dtype=torch.float32)
    actual_tensor_row = event.unit_reference_matrix[unit_index].cpu().to(torch.float32)
    if not torch.equal(actual_tensor_row, expected_tensor_row):
        raise ValueError("P0 unit physical reference row drifted")
    carriers = [
        {
            "carrier_ordinal": ordinal,
            "physical_column_index": column,
            "physical_channel_id": str(event.physical_electrode_ids[column]),
            "coefficient": float(coefficient),
        }
        for ordinal, (column, coefficient) in enumerate(
            (
                item
                for item in enumerate(effective_row)
                if abs(item[1]) > _COEFFICIENT_TOLERANCE
            )
        )
    ]
    carrier_ids = {str(row["physical_channel_id"]) for row in carriers}
    if carrier_ids != set(str(item) for item in output["canonical_source_channel_ids"]):
        raise ValueError(
            "trusted output-unit carrier catalog disagrees with its matrix"
        )
    matrix_payload = {
        "view_id": view_id,
        "physical_input_unit_ids": list(event.physical_electrode_ids),
        "output_unit_ids": list(output_ids),
        "matrix": effective_matrix,
    }
    signed_row_payload = {
        "view_id": view_id,
        "output_unit_id": source_unit_id,
        "output_row_index": row_index,
        "physical_input_unit_ids": list(event.physical_electrode_ids),
        "signed_coefficients": effective_row,
    }
    return {
        "view_index": view_index,
        "view_id": view_id,
        "source_unit_id": source_unit_id,
        "unit_id": str(event.unit_ids[unit_index]),
        "unit_type": str(event.unit_types[unit_index]),
        "output_row_index": row_index,
        "effective_row": effective_row,
        "carriers": carriers,
        "effective_matrix_sha256": _canonical_sha256(matrix_payload),
        "signed_row_sha256": _canonical_sha256(signed_row_payload),
        "carrier_set_sha256": _canonical_sha256(carriers),
        "copy_neutral_fingerprint_sha256": (
            _copy_neutral_reference_fingerprint(effective_row)
        ),
        "source_reference_matrix_sha256": str(
            view["transform_spec"]["reference"]["matrix_sha256"]
        ),
    }


def _build_dependency(
    event: BAIEGEventTokens,
    token_index: int,
    canonical: Mapping[str, Any],
    views: Mapping[str, Mapping[str, Any]],
    unit: Mapping[str, Any],
    *,
    source_p0_materialization_receipt_sha256: str,
) -> dict[str, Any]:
    view_index = int(unit["view_index"])
    view_id = str(unit["view_id"])
    view = views[view_id]
    effective_role = str(event.view_effective_temporal_roles[view_index])
    policy, precision, fir_order = _support_contract(view_id, effective_role, views)
    interval = [
        float(item) for item in event.token_time_bounds_seconds[token_index].tolist()
    ]
    clock = view["transform_spec"]["output_clock"]
    output_num = int(clock["sampling_rate_numerator"])
    output_den = int(clock["sampling_rate_denominator"])
    global_start = _nearest_output_edge(interval[0], output_num, output_den)
    global_stop = _nearest_output_edge(interval[1], output_num, output_den)
    if global_stop <= global_start:
        raise ValueError("P0 token output support is empty")
    selected_start, selected_stop = (
        int(item)
        for item in view["coordinates"]["selected_global_output_sample_interval"]
    )
    if global_start < selected_start or global_stop > selected_stop:
        raise ValueError("P0 token output support escapes the trusted view")
    valid_start, valid_stop = (
        int(item) for item in view["tensor_layout"]["valid_data_tensor_sample_interval"]
    )
    tensor_start = valid_start + global_start - selected_start
    tensor_stop = valid_start + global_stop - selected_start
    if tensor_start < valid_start or tensor_stop > valid_stop:
        raise ValueError("P0 token tensor support escapes observed view data")
    exact_output_seconds = [
        global_start * output_den / output_num,
        global_stop * output_den / output_num,
    ]

    canonical_channels = {str(row["channel_id"]): row for row in canonical["channels"]}
    raw_rows: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for carrier in unit["carriers"]:
        channel_id = str(carrier["physical_channel_id"])
        channel = canonical_channels[channel_id]
        raw_num = int(channel["sample_rate_numerator"])
        raw_den = int(channel["sample_rate_denominator"])
        raw_count = int(channel["sample_count"])
        equivalent_start, equivalent_stop = _outward_raw_interval(
            global_start,
            global_stop,
            output_numerator=output_num,
            output_denominator=output_den,
            raw_numerator=raw_num,
            raw_denominator=raw_den,
        )
        observed = bool(channel["observed"])
        dependency_interval: list[int] | None
        recording_interval: list[float] | None
        if not observed:
            dependency_interval = None
            recording_interval = None
            unavailable.append(channel_id)
        elif effective_role == "context_offline":
            dependency_interval = [0, raw_count]
            recording_interval = [0.0, float(canonical["recording_duration_seconds"])]
        else:
            if raw_num * output_den != output_num * raw_den:
                raise ValueError(
                    "exact native/causal P0 closure requires the canonical output clock"
                )
            start = equivalent_start
            if fir_order is not None:
                start = max(0, start - fir_order)
            stop = equivalent_stop
            if not (0 <= start < stop <= raw_count):
                raise ValueError(
                    "P0 raw dependency support escapes a canonical channel"
                )
            dependency_interval = [start, stop]
            recording_interval = [
                start * raw_den / raw_num,
                stop * raw_den / raw_num,
            ]
        raw_rows.append(
            {
                "carrier_ordinal": int(carrier["carrier_ordinal"]),
                "physical_channel_id": channel_id,
                "coefficient": float(carrier["coefficient"]),
                "observed": observed,
                "sample_rate_numerator": raw_num,
                "sample_rate_denominator": raw_den,
                "channel_sample_count": raw_count,
                "output_equivalent_raw_sample_interval": [
                    equivalent_start,
                    equivalent_stop,
                ],
                "raw_dependency_sample_interval": dependency_interval,
                "raw_dependency_recording_interval_seconds": recording_interval,
            }
        )

    temporal = view["temporal_evidence"]
    body = {
        "schema_version": BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_SCHEMA_VERSION_V1,
        "source_token_index": token_index,
        "source_binding": {
            "event_model_input_receipt_sha256": event.input_receipt_sha256,
            "source_p0_materialization_receipt_sha256": (
                source_p0_materialization_receipt_sha256
            ),
            "canonical_signal_id": str(canonical["canonical_signal_id"]),
            "canonical_receipt_sha256": str(canonical["receipt_sha256"]),
            "source_signal_sha256": str(canonical["source_signal_sha256"]),
            "view_receipt_sha256": str(view["receipt_sha256"]),
            "view_transform_spec_sha256": str(
                view["transform_spec"]["transform_spec_sha256"]
            ),
        },
        "token_coordinate": {
            "event_id": event.event_id,
            "recording_id": event.recording_id,
            "view_index": view_index,
            "view_id": view_id,
            "unit_index": int(event.token_unit_index[token_index]),
            "unit_id": str(unit["unit_id"]),
            "unit_source_id": str(unit["source_unit_id"]),
            "scale_index": int(event.token_scale_index[token_index]),
            "scale": BA_IEG_TOKEN_SCALES[int(event.token_scale_index[token_index])],
            "signal_eligible": bool(event.token_signal_mask[token_index]),
        },
        "output_support": {
            "event_token_recording_interval_seconds": interval,
            "exact_output_recording_interval_seconds": exact_output_seconds,
            "output_clock": {
                "sampling_rate_numerator": output_num,
                "sampling_rate_denominator": output_den,
                "global_origin_recording_seconds": 0.0,
            },
            "global_output_sample_interval": [global_start, global_stop],
            "view_tensor_sample_interval": [tensor_start, tensor_stop],
            "event_time_precision_policy": (
                "float32_token_time_replayed_to_nearest_trusted_output_edge_v1"
            ),
        },
        "temporal_contract": {
            "effective_temporal_role": effective_role,
            "dependency_policy": str(temporal["dependency_policy"]),
            "future_sample_access": bool(temporal["future_sample_access"]),
            "clinical_onset_evidence_authorized": bool(
                temporal["onset_evidence_authorized"]
            ),
            "token_onset_evidence_eligible": bool(
                event.token_onset_evidence_mask[token_index]
            ),
            "clinical_onset_authorization_used_for_raw_closure": False,
            "temporal_evidence_sha256": _canonical_sha256(temporal),
        },
        "reference_lineage": {
            "physical_input_unit_ids": list(event.physical_electrode_ids),
            "source_reference_matrix_sha256": str(
                unit["source_reference_matrix_sha256"]
            ),
            "effective_physical_reference_matrix_sha256": str(
                unit["effective_matrix_sha256"]
            ),
            "output_row_index": int(unit["output_row_index"]),
            "output_unit_id": str(unit["source_unit_id"]),
            "exact_signed_reference_row": list(unit["effective_row"]),
            "exact_signed_reference_row_sha256": str(unit["signed_row_sha256"]),
            "ordered_carriers": deepcopy(unit["carriers"]),
            "ordered_carrier_set_sha256": str(unit["carrier_set_sha256"]),
            "copy_neutral_reference_fingerprint_sha256": str(
                unit["copy_neutral_fingerprint_sha256"]
            ),
            "view_lineage": _view_lineage(view_id, views),
            "processed_view_sha256": str(view["processed_view_sha256"]),
        },
        "raw_support": {
            "support_policy": policy,
            "support_precision": precision,
            "fir_filter_order_samples": fir_order,
            "raw_dependency_closure_proven": not unavailable,
            "raw_dependency_minimality_proven": (
                not unavailable and effective_role != "context_offline"
            ),
            "unavailable_carrier_ids": sorted(unavailable),
            "per_carrier_raw_sample_intervals": raw_rows,
        },
    }
    return _finalize_dependency(body)


def _build_sidecar(
    event: BAIEGEventTokens,
    *,
    source_p0_materialization_receipt_sha256: str,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    source_p0_sha256 = _sha256(
        source_p0_materialization_receipt_sha256,
        "source_p0_materialization_receipt_sha256",
    )
    canonical, views, matrices = _prepare_host(
        event, canonical_signal_receipt, trusted_view_receipts
    )
    unit_bindings = {
        unit_index: _unit_binding(event, unit_index, views, matrices)
        for unit_index in range(len(event.unit_ids))
    }
    dependencies = [
        _build_dependency(
            event,
            token_index,
            canonical,
            views,
            unit_bindings[int(event.token_unit_index[token_index])],
            source_p0_materialization_receipt_sha256=source_p0_sha256,
        )
        for token_index in range(int(event.token_values.shape[0]))
    ]
    body = {
        "schema_version": (BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_SIDECAR_SCHEMA_VERSION_V1),
        "method_id": BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_METHOD_ID_V1,
        "source_binding": {
            "event_id": event.event_id,
            "recording_id": event.recording_id,
            "patient_uid": event.patient_uid,
            "model_split": event.model_split,
            "event_model_input_receipt_sha256": event.input_receipt_sha256,
            "source_p0_materialization_receipt_sha256": source_p0_sha256,
            "canonical_signal_id": str(canonical["canonical_signal_id"]),
            "canonical_receipt_sha256": str(canonical["receipt_sha256"]),
            "source_signal_sha256": str(canonical["source_signal_sha256"]),
            "trusted_view_receipt_sha256s": [
                str(views[view_id]["receipt_sha256"]) for view_id in event.view_ids
            ],
            "token_count": int(event.token_values.shape[0]),
        },
        "dependencies": dependencies,
        "dependency_roster_sha256": _dependency_roster_sha256(dependencies),
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
    }
    return _finalize_sidecar(body)


def _validate_embedded_content(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _SIDECAR_KEYS:
        raise ValueError("P0 raw-dependency sidecar has missing/unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"]
        != BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_SIDECAR_SCHEMA_VERSION_V1
        or data["method_id"] != BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_METHOD_ID_V1
        or data["scope_receipt"] != _SCOPE_RECEIPT
    ):
        raise ValueError("P0 raw-dependency sidecar contract drifted")
    dependencies = data["dependencies"]
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("P0 raw-dependency sidecar requires dependencies")
    indices: list[int] = []
    for index, dependency in enumerate(dependencies):
        if type(dependency) is not dict or set(dependency) != _DEPENDENCY_KEYS:
            raise ValueError("P0 raw dependency has missing/unknown fields")
        if (
            dependency["schema_version"]
            != BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_SCHEMA_VERSION_V1
        ):
            raise ValueError("P0 raw dependency schema drifted")
        source_index = int(dependency["source_token_index"])
        indices.append(source_index)
        body = deepcopy(dependency)
        supplied_id = _identifier(body.pop("dependency_id"), "dependency_id")
        supplied_sha = _sha256(body.pop("dependency_sha256"), "dependency_sha256")
        expected = _finalize_dependency(body)
        if (
            supplied_id != expected["dependency_id"]
            or supplied_sha != expected["dependency_sha256"]
        ):
            raise ValueError(f"P0 raw dependency {index} content hash drifted")
    if indices != list(range(len(dependencies))):
        raise ValueError("P0 raw dependencies must exactly follow token order")
    if data["dependency_roster_sha256"] != _dependency_roster_sha256(dependencies):
        raise ValueError("P0 raw dependency roster hash drifted")
    body = deepcopy(data)
    supplied_id = _identifier(body.pop("sidecar_id"), "sidecar_id")
    supplied_sha = _sha256(body.pop("sidecar_sha256"), "sidecar_sha256")
    expected = _finalize_sidecar(body)
    if (
        supplied_id != expected["sidecar_id"]
        or supplied_sha != expected["sidecar_sha256"]
    ):
        raise ValueError("P0 raw-dependency sidecar content hash drifted")
    return data


def materialize_ba_ieg_p0_raw_sample_dependency_sidecar_v1(
    event_tokens: BAIEGEventTokens,
    *,
    source_p0_materialization_receipt_sha256: str,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Materialize one non-model-input raw dependency receipt per P0 token."""

    artifact = _build_sidecar(
        event_tokens,
        source_p0_materialization_receipt_sha256=(
            source_p0_materialization_receipt_sha256
        ),
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    return _validate_embedded_content(artifact)


def validate_ba_ieg_p0_raw_sample_dependency_sidecar_v1(
    payload: object,
    *,
    event_tokens: BAIEGEventTokens,
    source_p0_materialization_receipt_sha256: str,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Validate hashes and replay every dependency from host-supplied roots."""

    data = _validate_embedded_content(payload)
    expected = _build_sidecar(
        event_tokens,
        source_p0_materialization_receipt_sha256=(
            source_p0_materialization_receipt_sha256
        ),
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    if data != expected:
        raise ValueError(
            "P0 raw-dependency sidecar does not replay from host-supplied roots"
        )
    return data


def replay_ba_ieg_p0_raw_sample_dependency_sidecar_v1(
    payload: object,
    *,
    event_tokens: BAIEGEventTokens,
    source_p0_materialization_receipt_sha256: str,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Explicit replay alias for integration boundaries and registries."""

    return validate_ba_ieg_p0_raw_sample_dependency_sidecar_v1(
        payload,
        event_tokens=event_tokens,
        source_p0_materialization_receipt_sha256=(
            source_p0_materialization_receipt_sha256
        ),
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )


__all__ = [
    "BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_METHOD_ID_V1",
    "BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_SCHEMA_VERSION_V1",
    "BA_IEG_P0_RAW_SAMPLE_DEPENDENCY_SIDECAR_SCHEMA_VERSION_V1",
    "materialize_ba_ieg_p0_raw_sample_dependency_sidecar_v1",
    "replay_ba_ieg_p0_raw_sample_dependency_sidecar_v1",
    "validate_ba_ieg_p0_raw_sample_dependency_sidecar_v1",
]
