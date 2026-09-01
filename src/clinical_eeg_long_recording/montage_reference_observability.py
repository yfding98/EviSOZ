"""Fail-closed acquisition-montage and derived-reference observability.

This module answers a deliberately narrower question than source
localisation: do the EDF *signal labels* make a common-reference electrode
field observable enough to construct a frozen linear montage?  It never
opens an EDF, reads an annotation, looks at a physician label, or inspects a
clinical report.

The contract is conservative:

* a shared reference token in every directly observed Standard-19 signal is
  called ``common_compatible_referential``; this is header compatibility, not
  physical verification of the acquisition reference;
* already-bipolar, mixed-reference, and unknown-reference inputs cannot
  authorize a CAR, Laplacian, or a second bipolar electrode field;
* every reference matrix is content addressed and carries numerical rank,
  non-zero-spectrum condition, and carrier-graph connectivity diagnostics;
* structural eligibility is per output row and requires every non-zero
  carrier to be directly observed; and
* a bad-channel, step, gap, or other canonical quality primitive contaminates
  every output row with a non-zero coefficient on that source, over the same
  interval for an instantaneous reference transform.  A reference child may
  add masks but may never clear a parent/source mask.

The receipt does not assert that a scalp field equals cortical SOZ, an
epileptogenic zone, or a surgical target.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from src.soz.geometry import STANDARD_19, TCP_20_EDGES, normalize_electrode_name


MONTAGE_REFERENCE_OBSERVABILITY_SCHEMA_VERSION = (
    "clinical_eeg_montage_reference_observability_v1"
)
REFERENCE_MATRIX_OBSERVABILITY_SCHEMA_VERSION = (
    "clinical_eeg_reference_matrix_observability_v1"
)
MONTAGE_CLASSES = (
    "common_compatible_referential",
    "already_bipolar",
    "mixed",
    "unknown",
)
DERIVED_REFERENCE_KINDS = ("tcp_bipolar", "car", "laplacian")
QUALITY_KINDS = (
    "missing",
    "flat",
    "clipping",
    "step",
    "line_noise",
    "gap",
    "other_signal_quality",
)
QUALITY_SEVERITIES = ("limited", "unusable")
QUALITY_EVIDENCE_FAMILIES = (
    "amplitude",
    "morphology",
    "spectral",
    "spatial_field",
    "high_frequency",
    "waveform",
)
QUALITY_PROPAGATION_POLICY: dict[str, object] = {
    "source_support_rule": "every_nonzero_reference_matrix_carrier_union_v1",
    "instantaneous_interval_rule": (
        "same_recording_relative_half_open_interval_no_shrink_v1"
    ),
    "child_mask_rule": "derived_child_may_add_but_never_clear_source_or_parent_mask_v1",
    "bad_channel_rule": (
        "whole_channel_or_marked_interval_disables_every_dependent_output_row_v1"
    ),
    "step_gap_rule": (
        "step_and_gap_disable_every_dependent_output_row_over_parent_qualified_interval_v1"
    ),
    "parent_filter_rule": (
        "temporal_filter_influence_must_be_qualified_and_masked_in_parent_before_reference_v1"
    ),
    "covered_quality_kinds": list(QUALITY_KINDS),
}
MONTAGE_SCOPE_RECEIPT: dict[str, object] = {
    "edf_signal_labels_used": True,
    "eeg_samples_used": False,
    "edf_patient_or_recording_header_used": False,
    "edf_annotation_api_called": False,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "video_used": False,
    "research_only": True,
    "source_localization_authorized": False,
}

_REFERENCE_TOKENS = (
    "LINKED-EARS",
    "LINKED-EAR",
    "A1A2",
    "M1M2",
    "REF",
    "LE",
    "AR",
    "AVG",
    "AV",
    "CAR",
    "A1",
    "A2",
    "M1",
    "M2",
)
_TOL = 1e-10
_NON_ALNUM = re.compile(r"[^A-Z0-9-]+")
_QUALITY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")

_LAPLACIAN_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "FP1": ("F7", "F3", "FZ"),
    "FP2": ("FZ", "F4", "F8"),
    "F7": ("FP1", "F3", "T7"),
    "F3": ("FP1", "F7", "FZ", "C3"),
    "FZ": ("FP1", "FP2", "F3", "F4", "CZ"),
    "F4": ("FP2", "FZ", "F8", "C4"),
    "F8": ("FP2", "F4", "T8"),
    "T7": ("F7", "C3", "P7"),
    "C3": ("F3", "T7", "CZ", "P3"),
    "CZ": ("FZ", "C3", "C4", "PZ"),
    "C4": ("F4", "CZ", "T8", "P4"),
    "T8": ("F8", "C4", "P8"),
    "P7": ("T7", "P3", "O1"),
    "P3": ("C3", "P7", "PZ", "O1"),
    "PZ": ("CZ", "P3", "P4", "O1", "O2"),
    "P4": ("C4", "PZ", "P8", "O2"),
    "P8": ("T8", "P4", "O2"),
    "O1": ("P7", "P3", "PZ"),
    "O2": ("PZ", "P4", "P8"),
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def signal_labels_sha256(signal_labels: Sequence[object]) -> str:
    """Hash the complete ordered EDF signal-label vector."""

    if isinstance(signal_labels, (str, bytes)) or not isinstance(
        signal_labels, Sequence
    ):
        raise TypeError("signal_labels must be an ordered array")
    labels = [str(label).strip() for label in signal_labels]
    if not labels or any(not label for label in labels):
        raise ValueError("signal_labels must be non-empty trimmed strings")
    return _canonical_sha256(
        {"domain": "edf-ordered-signal-labels-v1", "signal_labels": labels}
    )


def _label_core(raw_label: object) -> tuple[str, bool]:
    raw = str(raw_label).strip()
    upper = raw.upper().replace("_", "-")
    eeg_prefixed = upper.startswith("EEG ") or upper.startswith("EEG-")
    if eeg_prefixed:
        upper = upper[4:]
    upper = _NON_ALNUM.sub("", upper).strip("-")
    return upper, eeg_prefixed


def _standard_endpoint(value: str) -> str | None:
    endpoint = normalize_electrode_name(value)
    return endpoint if endpoint in STANDARD_19 else None


def _parse_signal_label(raw_label: object, signal_index: int) -> dict[str, object]:
    raw = str(raw_label).strip()
    core, eeg_prefixed = _label_core(raw)

    # A known suffix is interpreted as an acquisition-reference token before
    # trying a bipolar split.  Thus FP1-REF is not mistaken for a lead.
    for token in _REFERENCE_TOKENS:
        marker = f"-{token}"
        if core.endswith(marker):
            positive = _standard_endpoint(core[: -len(marker)])
            if positive is not None:
                return {
                    "signal_index": signal_index,
                    "raw_label": raw,
                    "normalized_label": core,
                    "signal_role": "direct_standard_electrode",
                    "positive_electrode": positive,
                    "negative_electrode": None,
                    "reference_token": token,
                    "modeled_reference_node": f"REF::{token}",
                }
            # TUSZ commonly stores auxiliary EOG/ECG, auricular, photic and
            # numbered technical channels with the same ``-REF`` suffix as
            # the scalp channels (for example ``EEG ROC-REF`` and
            # ``EEG 26-REF``).  Their presence does not make the reference of
            # the directly observed Standard-19 carriers unknowable.  Keep an
            # explicit ignored observation rather than treating an auxiliary
            # channel as an unparsed scalp EEG signal.  A label without a
            # recognized suffix still follows the fail-closed ``eeg_unknown``
            # route below.
            return {
                "signal_index": signal_index,
                "raw_label": raw,
                "normalized_label": core,
                "signal_role": "non_standard_referential_ignored",
                "positive_electrode": None,
                "negative_electrode": None,
                "reference_token": token,
                "modeled_reference_node": None,
            }

    parts = core.split("-")
    if len(parts) == 2:
        positive = _standard_endpoint(parts[0])
        negative = _standard_endpoint(parts[1])
        if positive is not None and negative is not None and positive != negative:
            return {
                "signal_index": signal_index,
                "raw_label": raw,
                "normalized_label": core,
                "signal_role": "standard_bipolar",
                "positive_electrode": positive,
                "negative_electrode": negative,
                "reference_token": None,
                "modeled_reference_node": negative,
            }

    direct = _standard_endpoint(core)
    if direct is not None:
        return {
            "signal_index": signal_index,
            "raw_label": raw,
            "normalized_label": core,
            "signal_role": "direct_standard_electrode_unknown_reference",
            "positive_electrode": direct,
            "negative_electrode": None,
            "reference_token": None,
            "modeled_reference_node": f"OPAQUE-REF::SIG-{signal_index:04d}",
        }

    tokens = [part for part in core.split("-") if part]
    has_standard_token = any(_standard_endpoint(token) is not None for token in tokens)
    role = "eeg_unknown" if eeg_prefixed or has_standard_token else "non_eeg_ignored"
    return {
        "signal_index": signal_index,
        "raw_label": raw,
        "normalized_label": core,
        "signal_role": role,
        "positive_electrode": None,
        "negative_electrode": None,
        "reference_token": None,
        "modeled_reference_node": None,
    }


def classify_signal_labels(signal_labels: Sequence[object]) -> dict[str, Any]:
    """Return deterministic signal-label observations and montage class."""

    labels_hash = signal_labels_sha256(signal_labels)
    observations = [
        _parse_signal_label(label, index) for index, label in enumerate(signal_labels)
    ]
    direct_known = [
        row
        for row in observations
        if row["signal_role"] == "direct_standard_electrode"
    ]
    direct_unknown = [
        row
        for row in observations
        if row["signal_role"] == "direct_standard_electrode_unknown_reference"
    ]
    bipolar = [
        row for row in observations if row["signal_role"] == "standard_bipolar"
    ]
    eeg_unknown = [row for row in observations if row["signal_role"] == "eeg_unknown"]
    direct_all = [*direct_known, *direct_unknown]
    direct_ids = [str(row["positive_electrode"]) for row in direct_all]
    duplicate_direct = sorted(
        {channel for channel in direct_ids if direct_ids.count(channel) > 1},
        key=STANDARD_19.index,
    )
    reference_tokens = sorted(
        {str(row["reference_token"]) for row in direct_known}
    )

    common = (
        bool(direct_known)
        and not direct_unknown
        and not bipolar
        and not eeg_unknown
        and not duplicate_direct
        and len(reference_tokens) == 1
    )
    reasons: list[str] = []
    if common:
        montage_class = "common_compatible_referential"
        reasons.append("common_reference_token_consistent_across_direct_channels")
    elif bipolar and not direct_all and not eeg_unknown:
        montage_class = "already_bipolar"
        reasons.append("standard_bipolar_signal_labels_without_direct_electrode_channels")
    elif (
        (bipolar and (direct_all or eeg_unknown))
        or len(reference_tokens) > 1
        or (direct_known and direct_unknown)
        or bool(duplicate_direct)
    ):
        montage_class = "mixed"
        if bipolar and (direct_all or eeg_unknown):
            reasons.append("referential_bipolar_or_unknown_eeg_labels_coexist")
        if len(reference_tokens) > 1:
            reasons.append("multiple_acquisition_reference_tokens")
        if direct_known and direct_unknown:
            reasons.append("known_and_unknown_direct_references_coexist")
        if duplicate_direct:
            reasons.append("duplicate_direct_electrode_observations")
    else:
        montage_class = "unknown"
        if direct_unknown:
            reasons.append("direct_electrode_reference_token_unobservable")
        if eeg_unknown:
            reasons.append("unparsed_eeg_signal_labels_present")
        if not direct_all and not bipolar and not eeg_unknown:
            reasons.append("no_observable_standard19_eeg_montage")
    reasons = sorted(set(reasons))

    observed_unique = [
        channel for channel in STANDARD_19 if channel in set(direct_ids)
    ]
    return {
        "signal_labels_sha256": labels_hash,
        "signal_label_observations": observations,
        "montage_class": montage_class,
        "classification_reason_codes": reasons,
        "direct_electrode_ids": observed_unique,
        "duplicate_direct_electrode_ids": duplicate_direct,
        "common_reference_token": reference_tokens[0] if common else None,
        "common_reference_compatible": common,
    }


def _rounded(value: float) -> float:
    return float(format(float(value), ".12g"))


def _matrix_diagnostics(matrix: Sequence[Sequence[float]]) -> dict[str, object]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("reference matrix must be two-dimensional")
    rows, columns = (int(value) for value in values.shape)
    if rows == 0 or columns == 0:
        rank = 0
        tolerance = 0.0
        condition: float | None = None
    else:
        singular = np.linalg.svd(values, compute_uv=False)
        maximum = float(singular[0]) if singular.size else 0.0
        tolerance = max(rows, columns) * np.finfo(np.float64).eps * maximum
        nonzero = singular[singular > tolerance]
        rank = int(nonzero.size)
        condition = (
            None
            if not nonzero.size
            else _rounded(float(nonzero[0]) / float(nonzero[-1]))
        )
    return {
        "row_count": rows,
        "column_count": columns,
        "numerical_rank": rank,
        "row_nullity": rows - rank,
        "column_nullity": columns - rank,
        "full_row_rank": rank == rows,
        "full_column_rank": rank == columns,
        "rank_tolerance": _rounded(tolerance),
        "nonzero_singular_value_condition_number": condition,
        "condition_semantics": "largest_over_smallest_nonzero_singular_value_v1",
    }


def _support_connectivity(
    column_ids: Sequence[str], matrix: Sequence[Sequence[float]]
) -> dict[str, object]:
    ids = list(column_ids)
    adjacency: dict[str, set[str]] = {item: set() for item in ids}
    edge_pairs: set[tuple[str, str]] = set()
    for row in matrix:
        support = [
            item for item, coefficient in zip(ids, row) if abs(float(coefficient)) > _TOL
        ]
        for left_index, left in enumerate(support):
            for right in support[left_index + 1 :]:
                pair = tuple(sorted((left, right)))
                edge_pairs.add(pair)
                adjacency[left].add(right)
                adjacency[right].add(left)
    components: list[list[str]] = []
    remaining = set(ids)
    order = {item: index for index, item in enumerate(ids)}
    while remaining:
        root = min(remaining, key=order.__getitem__)
        stack = [root]
        component: list[str] = []
        remaining.remove(root)
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = sorted(adjacency[node].intersection(remaining), key=order.__getitem__)
            for neighbour in reversed(neighbours):
                remaining.remove(neighbour)
                stack.append(neighbour)
        components.append(sorted(component, key=order.__getitem__))
    return {
        "node_count": len(ids),
        "edge_count": len(edge_pairs),
        "connected_component_count": len(components),
        "components": components,
        "all_modeled_nodes_connected": len(components) <= 1,
        "connectivity_semantics": "shared_nonzero_output_row_carrier_graph_v1",
    }


def build_reference_matrix_observability(
    *,
    row_unit_ids: Sequence[str],
    column_unit_ids: Sequence[str],
    matrix: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Build a content-addressed numerical/support receipt for a matrix."""

    rows = [str(item) for item in row_unit_ids]
    columns = [str(item) for item in column_unit_ids]
    if not rows or not columns or len(rows) != len(set(rows)) or len(columns) != len(set(columns)):
        raise ValueError("reference matrix row/column IDs must be unique and non-empty")
    values = [[float(item) for item in row] for row in matrix]
    if len(values) != len(rows) or any(len(row) != len(columns) for row in values):
        raise ValueError("reference matrix shape disagrees with unit IDs")
    if any(
        not math.isfinite(item) for row in values for item in row
    ) or any(not any(abs(item) > _TOL for item in row) for row in values):
        raise ValueError("reference matrix must be finite with non-zero output rows")
    payload = {
        "row_unit_ids": rows,
        "column_unit_ids": columns,
        "matrix": values,
    }
    body = {
        "schema_version": REFERENCE_MATRIX_OBSERVABILITY_SCHEMA_VERSION,
        **payload,
        "matrix_sha256": _canonical_sha256(
            {"domain": "linear-reference-matrix-v1", **payload}
        ),
        "numerical_diagnostics": _matrix_diagnostics(values),
        "support_connectivity": _support_connectivity(columns, values),
    }
    return validate_reference_matrix_observability(body)


def validate_reference_matrix_observability(payload: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "row_unit_ids",
        "column_unit_ids",
        "matrix",
        "matrix_sha256",
        "numerical_diagnostics",
        "support_connectivity",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("reference matrix observability has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != REFERENCE_MATRIX_OBSERVABILITY_SCHEMA_VERSION:
        raise ValueError("unsupported reference matrix observability schema")
    rows = data["row_unit_ids"]
    columns = data["column_unit_ids"]
    if (
        not isinstance(rows, list)
        or not isinstance(columns, list)
        or not rows
        or not columns
        or any(not isinstance(item, str) or not item for item in [*rows, *columns])
        or len(rows) != len(set(rows))
        or len(columns) != len(set(columns))
    ):
        raise ValueError("reference matrix observability unit IDs are invalid")
    matrix = data["matrix"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(rows)
        or any(not isinstance(row, list) or len(row) != len(columns) for row in matrix)
    ):
        raise ValueError("reference matrix observability shape is invalid")
    values = [[float(item) for item in row] for row in matrix]
    if any(not math.isfinite(item) for row in values for item in row) or any(
        not any(abs(item) > _TOL for item in row) for row in values
    ):
        raise ValueError("reference matrix observability values are invalid")
    expected_hash = _canonical_sha256(
        {
            "domain": "linear-reference-matrix-v1",
            "row_unit_ids": rows,
            "column_unit_ids": columns,
            "matrix": values,
        }
    )
    if data["matrix_sha256"] != expected_hash:
        raise ValueError("reference matrix observability hash drifted")
    if data["numerical_diagnostics"] != _matrix_diagnostics(values):
        raise ValueError("reference matrix numerical diagnostics drifted")
    if data["support_connectivity"] != _support_connectivity(columns, values):
        raise ValueError("reference matrix support connectivity drifted")
    return data


def _acquisition_reference_model(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    modeled = [
        row
        for row in observations
        if row["signal_role"]
        in {
            "direct_standard_electrode",
            "direct_standard_electrode_unknown_reference",
            "standard_bipolar",
        }
    ]
    electrode_nodes = [
        channel
        for channel in STANDARD_19
        if any(
            row["positive_electrode"] == channel
            or row["negative_electrode"] == channel
            for row in modeled
        )
    ]
    reference_nodes = sorted(
        {
            str(row["modeled_reference_node"])
            for row in modeled
            if row["modeled_reference_node"] not in STANDARD_19
        }
    )
    columns = [*electrode_nodes, *reference_nodes]
    index = {node: position for position, node in enumerate(columns)}
    rows: list[list[float]] = []
    row_ids: list[str] = []
    for observation in modeled:
        positive = str(observation["positive_electrode"])
        negative = str(observation["modeled_reference_node"])
        row = [0.0] * len(columns)
        row[index[positive]] = 1.0
        row[index[negative]] -= 1.0
        rows.append(row)
        row_ids.append(f"SIG-{int(observation['signal_index']):04d}")
    matrix_receipt = (
        None
        if not rows
        else build_reference_matrix_observability(
            row_unit_ids=row_ids,
            column_unit_ids=columns,
            matrix=rows,
        )
    )
    unmodeled = [
        int(row["signal_index"])
        for row in observations
        if row["signal_role"] == "eeg_unknown"
    ]
    return {
        "interpretation_policy": (
            "signal_label_incidence_only_opaque_unknown_reference_per_signal_v1"
        ),
        "matrix_coverage_complete": not unmodeled,
        "unmodeled_eeg_signal_indices": unmodeled,
        "reference_matrix_observability": matrix_receipt,
    }


def _reference_recipe(
    reference_kind: str,
) -> tuple[tuple[str, ...], list[list[float]], dict[str, list[str]], str]:
    channel_index = {name: index for index, name in enumerate(STANDARD_19)}
    if reference_kind == "tcp_bipolar":
        output_ids = tuple(f"{left}-{right}" for left, right in TCP_20_EDGES)
        sources = {
            output_id: [left, right]
            for output_id, (left, right) in zip(output_ids, TCP_20_EDGES)
        }
        matrix: list[list[float]] = []
        for left, right in TCP_20_EDGES:
            row = [0.0] * len(STANDARD_19)
            row[channel_index[left]] = 1.0
            row[channel_index[right]] = -1.0
            matrix.append(row)
        return output_ids, matrix, sources, "longitudinal_bipolar_tcp20_v1"
    if reference_kind == "car":
        output_ids = tuple(f"{channel}-CAR" for channel in STANDARD_19)
        scale = 1.0 / len(STANDARD_19)
        matrix = []
        for target in STANDARD_19:
            row = [-scale] * len(STANDARD_19)
            row[channel_index[target]] += 1.0
            matrix.append(row)
        sources = {output_id: list(STANDARD_19) for output_id in output_ids}
        return output_ids, matrix, sources, "common_average_standard19_frozen_v1"
    if reference_kind == "laplacian":
        output_ids = tuple(f"{channel}-LAP" for channel in STANDARD_19)
        matrix = []
        sources: dict[str, list[str]] = {}
        for target, output_id in zip(STANDARD_19, output_ids):
            neighbours = _LAPLACIAN_NEIGHBORS[target]
            row = [0.0] * len(STANDARD_19)
            row[channel_index[target]] = 1.0
            for neighbour in neighbours:
                row[channel_index[neighbour]] = -1.0 / len(neighbours)
            matrix.append(row)
            sources[output_id] = [target, *neighbours]
        return output_ids, matrix, sources, "surface_laplacian_standard19_graph_v1"
    raise ValueError(f"unsupported derived reference kind: {reference_kind}")


def _derived_reference_contract(
    *,
    reference_kind: str,
    common_reference_compatible: bool,
    direct_electrode_ids: Sequence[str],
) -> dict[str, Any]:
    output_ids, matrix, sources, reference_type = _reference_recipe(reference_kind)
    matrix_receipt = build_reference_matrix_observability(
        row_unit_ids=output_ids,
        column_unit_ids=STANDARD_19,
        matrix=matrix,
    )
    observed = set(direct_electrode_ids)
    output_support: list[dict[str, object]] = []
    for unit_id in output_ids:
        carriers = sources[unit_id]
        reasons: list[str] = []
        if not common_reference_compatible:
            reasons.append("acquisition_montage_not_common_reference_compatible")
        missing = [channel for channel in carriers if channel not in observed]
        if missing:
            reasons.append("required_reference_carrier_unobserved")
        output_support.append(
            {
                "unit_id": unit_id,
                "quality_dependency_channel_ids": carriers,
                "missing_carrier_ids": missing,
                "evidence_eligible": not reasons,
                "reason_codes": reasons,
            }
        )
    any_eligible = any(bool(row["evidence_eligible"]) for row in output_support)
    all_eligible = all(bool(row["evidence_eligible"]) for row in output_support)
    global_reasons: list[str] = []
    if not common_reference_compatible:
        global_reasons.append("acquisition_montage_not_common_reference_compatible")
    if common_reference_compatible and not any_eligible:
        global_reasons.append("no_output_has_complete_observed_carrier_support")
    return {
        "reference_kind": reference_kind,
        "reference_type": reference_type,
        "tensor_materialization_authorized": common_reference_compatible,
        "any_output_evidence_eligible": any_eligible,
        "all_outputs_evidence_eligible": all_eligible,
        "global_reason_codes": global_reasons,
        "reference_matrix_observability": matrix_receipt,
        "output_support": output_support,
        "interpretation_limit": (
            "scalp_montage_consistency_only_not_source_localization_or_cortical_soz_v1"
        ),
    }


def _receipt_core(signal_labels: Sequence[object], source_signal_sha256: str) -> dict[str, Any]:
    if not _is_sha256(source_signal_sha256):
        raise ValueError("source_signal_sha256 must be lowercase SHA-256")
    classification = classify_signal_labels(signal_labels)
    common = bool(classification["common_reference_compatible"])
    direct = list(classification["direct_electrode_ids"])
    return {
        "schema_version": MONTAGE_REFERENCE_OBSERVABILITY_SCHEMA_VERSION,
        "source_signal_sha256": source_signal_sha256,
        "signal_labels_sha256": classification["signal_labels_sha256"],
        "observation_policy": (
            "ordered_edf_signal_labels_only_no_signal_or_clinical_context_v1"
        ),
        "montage_class": classification["montage_class"],
        "classification_reason_codes": classification["classification_reason_codes"],
        "signal_label_observations": classification["signal_label_observations"],
        "direct_electrode_ids": direct,
        "duplicate_direct_electrode_ids": classification[
            "duplicate_direct_electrode_ids"
        ],
        "common_reference_compatibility": {
            "compatible": common,
            "reference_token": classification["common_reference_token"],
            "semantics": (
                "shared_edf_label_token_compatibility_not_physical_reference_verification_v1"
            ),
            "reason_codes": (
                []
                if common
                else ["common_reference_compatibility_not_established"]
            ),
        },
        "acquisition_reference_model": _acquisition_reference_model(
            classification["signal_label_observations"]
        ),
        "derived_reference_contracts": {
            kind: _derived_reference_contract(
                reference_kind=kind,
                common_reference_compatible=common,
                direct_electrode_ids=direct,
            )
            for kind in DERIVED_REFERENCE_KINDS
        },
        "quality_propagation_policy": deepcopy(QUALITY_PROPAGATION_POLICY),
        "scope_receipt": deepcopy(MONTAGE_SCOPE_RECEIPT),
    }


def build_montage_reference_observability_receipt(
    *, signal_labels: Sequence[object], source_signal_sha256: str
) -> dict[str, Any]:
    """Materialize the acquisition/derived-reference qualification receipt."""

    body = {
        **_receipt_core(signal_labels, source_signal_sha256),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_montage_reference_observability_receipt(body)


def validate_montage_reference_observability_receipt(payload: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "source_signal_sha256",
        "signal_labels_sha256",
        "observation_policy",
        "montage_class",
        "classification_reason_codes",
        "signal_label_observations",
        "direct_electrode_ids",
        "duplicate_direct_electrode_ids",
        "common_reference_compatibility",
        "acquisition_reference_model",
        "derived_reference_contracts",
        "quality_propagation_policy",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("montage/reference observability has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != MONTAGE_REFERENCE_OBSERVABILITY_SCHEMA_VERSION:
        raise ValueError("unsupported montage/reference observability schema")
    if not _is_sha256(data["source_signal_sha256"]) or not _is_sha256(
        data["signal_labels_sha256"]
    ):
        raise ValueError("montage/reference observability hashes are invalid")
    observations = data["signal_label_observations"]
    if not isinstance(observations, list) or not observations:
        raise ValueError("montage signal-label observations must be non-empty")
    if [row.get("signal_index") for row in observations] != list(range(len(observations))):
        raise ValueError("montage signal-label indices drifted")
    labels = [row.get("raw_label") for row in observations]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("montage raw signal labels are invalid")
    expected = _receipt_core(labels, str(data["source_signal_sha256"]))
    supplied_core = {key: data[key] for key in expected}
    if supplied_core != expected:
        raise ValueError("montage/reference classification or diagnostics drifted")
    for contract in data["derived_reference_contracts"].values():
        validate_reference_matrix_observability(
            contract["reference_matrix_observability"]
        )
    acquisition_matrix = data["acquisition_reference_model"][
        "reference_matrix_observability"
    ]
    if acquisition_matrix is not None:
        validate_reference_matrix_observability(acquisition_matrix)
    if not _is_sha256(data["receipt_sha256"]):
        raise ValueError("montage/reference receipt hash is invalid")
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("montage/reference receipt hash does not bind content")
    return data


def require_reference_materialization_authorized(
    receipt: object, *, reference_kinds: Sequence[str] = DERIVED_REFERENCE_KINDS
) -> dict[str, Any]:
    """Reject any requested derived tensor not authorized by acquisition labels."""

    data = validate_montage_reference_observability_receipt(receipt)
    requested = [str(kind) for kind in reference_kinds]
    if not requested or any(kind not in DERIVED_REFERENCE_KINDS for kind in requested):
        raise ValueError("requested derived reference kinds are invalid")
    denied = [
        kind
        for kind in requested
        if not data["derived_reference_contracts"][kind][
            "tensor_materialization_authorized"
        ]
    ]
    if denied:
        raise ValueError(
            "derived reference materialization denied for acquisition montage "
            f"{data['montage_class']}: {denied}"
        )
    return data


def project_quality_contamination_to_reference(
    receipt: object,
    *,
    reference_kind: str,
    quality_primitives: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Project source quality intervals through non-zero reference support.

    The function performs only the instantaneous spatial projection.  Any
    temporal ringing/support introduced by a filter must already be expressed
    in the parent mask, as required by ``QUALITY_PROPAGATION_POLICY``.
    """

    data = validate_montage_reference_observability_receipt(receipt)
    if reference_kind not in DERIVED_REFERENCE_KINDS:
        raise ValueError("unsupported reference kind for quality projection")
    contract = data["derived_reference_contracts"][reference_kind]
    rows: list[dict[str, Any]] = []
    quality_ids: set[str] = set()
    for index, primitive in enumerate(quality_primitives):
        required = {
            "quality_id",
            "channel_ids",
            "start_recording_seconds",
            "stop_recording_seconds",
            "kind",
            "severity",
            "disabled_evidence_families",
        }
        if type(primitive) is not dict or set(primitive) != required:
            raise ValueError(f"quality_primitives[{index}] has an invalid shape")
        quality_id = primitive["quality_id"]
        if not isinstance(quality_id, str) or not _QUALITY_ID.fullmatch(quality_id):
            raise ValueError(f"quality_primitives[{index}] has an invalid quality_id")
        if quality_id in quality_ids:
            raise ValueError("quality primitive IDs must be unique")
        quality_ids.add(quality_id)
        source_ids = primitive["channel_ids"]
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or any(
                not isinstance(channel, str) or channel not in STANDARD_19
                for channel in source_ids
            )
            or len(source_ids) != len(set(source_ids))
        ):
            raise ValueError(f"quality_primitives[{index}] has invalid channels")
        start_value = primitive["start_recording_seconds"]
        stop_value = primitive["stop_recording_seconds"]
        if (
            isinstance(start_value, bool)
            or not isinstance(start_value, (int, float))
            or isinstance(stop_value, bool)
            or not isinstance(stop_value, (int, float))
        ):
            raise TypeError(
                f"quality_primitives[{index}] interval endpoints must be numeric"
            )
        start = float(start_value)
        stop = float(stop_value)
        if not math.isfinite(start) or not math.isfinite(stop) or start < 0 or stop <= start:
            raise ValueError(f"quality_primitives[{index}] has an invalid interval")
        if primitive["kind"] not in QUALITY_KINDS:
            raise ValueError(f"quality_primitives[{index}] has an invalid kind")
        severity = primitive["severity"]
        if severity not in QUALITY_SEVERITIES:
            raise ValueError(f"quality_primitives[{index}] has an invalid severity")
        disabled = primitive["disabled_evidence_families"]
        if (
            not isinstance(disabled, list)
            or not disabled
            or any(
                not isinstance(family, str)
                or family not in QUALITY_EVIDENCE_FAMILIES
                for family in disabled
            )
            or len(disabled) != len(set(disabled))
        ):
            raise ValueError(
                f"quality_primitives[{index}] has invalid disabled evidence families"
            )
        if severity == "unusable" and set(disabled) != set(
            QUALITY_EVIDENCE_FAMILIES
        ):
            raise ValueError(
                "unusable quality primitives must disable every evidence family"
            )
        affected = [
            str(output["unit_id"])
            for output in contract["output_support"]
            if not set(source_ids).isdisjoint(
                output["quality_dependency_channel_ids"]
            )
        ]
        if not affected:
            continue
        rows.append(
            {
                "quality_id": quality_id,
                "kind": str(primitive["kind"]),
                "source_channel_ids": list(source_ids),
                "affected_output_unit_ids": affected,
                "start_recording_seconds": start,
                "stop_recording_seconds": stop,
                "severity": severity,
                "disabled_evidence_families": list(disabled),
                "propagation_reason_code": (
                    "derived_reference_nonzero_carrier_contamination_v1"
                ),
            }
        )
    return rows


def direct_electrode_index_by_signal(receipt: object) -> dict[str, int]:
    """Return the unique direct-electrode signal index map or fail closed."""

    data = validate_montage_reference_observability_receipt(receipt)
    result: dict[str, int] = {}
    for row in data["signal_label_observations"]:
        if row["signal_role"] not in {
            "direct_standard_electrode",
            "direct_standard_electrode_unknown_reference",
        }:
            continue
        channel = str(row["positive_electrode"])
        if channel in result:
            raise ValueError("direct electrode signal mapping is ambiguous")
        result[channel] = int(row["signal_index"])
    return result


__all__ = [
    "DERIVED_REFERENCE_KINDS",
    "MONTAGE_CLASSES",
    "MONTAGE_REFERENCE_OBSERVABILITY_SCHEMA_VERSION",
    "MONTAGE_SCOPE_RECEIPT",
    "QUALITY_PROPAGATION_POLICY",
    "REFERENCE_MATRIX_OBSERVABILITY_SCHEMA_VERSION",
    "build_montage_reference_observability_receipt",
    "build_reference_matrix_observability",
    "classify_signal_labels",
    "direct_electrode_index_by_signal",
    "project_quality_contamination_to_reference",
    "require_reference_materialization_authorized",
    "signal_labels_sha256",
    "validate_montage_reference_observability_receipt",
    "validate_reference_matrix_observability",
]
