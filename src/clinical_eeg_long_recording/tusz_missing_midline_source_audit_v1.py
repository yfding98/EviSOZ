"""Source-EDF audit for canonical TUSZ records that appear to miss FZ/PZ.

The audit distinguishes a genuinely absent scalp electrode from a label alias
or a recoverable reference transform.  It uses the strict acquisition-header
parser and never opens EDF samples, EDF annotation payload, csv/csv_bi target
files, spreadsheets, or clinical text.  Output is aggregate and path-free.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .eeg_acquisition_header_allowlist_v1 import parse_eeg_acquisition_header_v1


SCHEMA_VERSION = "clinical_eeg_tusz_missing_midline_source_audit_v1"
METHOD_ID = "strict_header_graph_derivability_and_sibling_source_check_v1"

_OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
_SCALP_TOKEN = re.compile(
    r"^(?:FP|AF|F|FT|FC|C|T|TP|CP|P|PO|O)[Z0-9]{1,2}$|^(?:A|M)[12]$|^IZ$"
)
_REFERENCE_TOKENS = frozenset({"REF", "LE", "AR", "AVG", "AV", "A1", "A2", "M1", "M2"})
_STANDARD_19 = frozenset(
    {
        "FP1",
        "FP2",
        "F7",
        "F3",
        "FZ",
        "F4",
        "F8",
        "T7",
        "C3",
        "CZ",
        "C4",
        "T8",
        "P7",
        "P3",
        "PZ",
        "P4",
        "P8",
        "O1",
        "O2",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_token(value: str) -> str:
    token = value.strip().upper()
    return _OLD_TO_NEW.get(token, token)


def parse_eeg_channel_label_graph_edge(label: str) -> tuple[str, str] | None:
    """Project an EEG label to one voltage-difference graph edge.

    The returned orientation follows the EDF label: ``A-B`` means ``V_A-V_B``.
    A single scalp label receives an explicit ``IMPLICIT_REF`` node; that node
    is never equated to REF/LE/AR without source evidence.
    """

    if not isinstance(label, str) or not label.strip():
        raise TypeError("EEG channel label must be a non-empty string")
    text = " ".join(label.strip().upper().split())
    if text.startswith("EEG "):
        text = text[4:].strip()
    compact = text.replace(" ", "")
    parts = [_normalize_token(part) for part in compact.split("-") if part]
    if len(parts) == 1 and _SCALP_TOKEN.fullmatch(parts[0]):
        return parts[0], "IMPLICIT_REF"
    if len(parts) != 2:
        return None
    left, right = parts
    if not _SCALP_TOKEN.fullmatch(left):
        return None
    if not (_SCALP_TOKEN.fullmatch(right) or right in _REFERENCE_TOKENS):
        return None
    return left, right


def _graph(edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return adjacency


def _connected(adjacency: Mapping[str, set[str]], source: str, target: str) -> bool:
    if source not in adjacency or target not in adjacency:
        return False
    pending = [source]
    seen = {source}
    while pending:
        node = pending.pop()
        if node == target:
            return True
        for neighbor in adjacency.get(node, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return False


def classify_midline_derivability(channel_labels: Sequence[str]) -> dict[str, Any]:
    edges = [
        edge
        for edge in (parse_eeg_channel_label_graph_edge(label) for label in channel_labels)
        if edge is not None
    ]
    adjacency = _graph(edges)
    nodes = set(adjacency)
    midline_nodes = sorted(nodes.intersection({"FPZ", "FZ", "CZ", "PZ", "OZ"}))
    st_fz_cz = _connected(adjacency, "FZ", "CZ")
    st_cz_pz = _connected(adjacency, "CZ", "PZ")
    reference_nodes = sorted(
        node
        for node in nodes
        if node in _REFERENCE_TOKENS or node == "IMPLICIT_REF"
    )
    eventnet_references = [
        reference
        for reference in reference_nodes
        if _connected(adjacency, "FZ", reference)
        and _connected(adjacency, "PZ", reference)
    ]
    return {
        "edge_count": len(edges),
        "normalized_graph_nodes": sorted(nodes),
        "midline_nodes": midline_nodes,
        "raw_FZ_node_present": "FZ" in nodes,
        "raw_PZ_node_present": "PZ" in nodes,
        "raw_FPZ_node_present_not_equivalent_to_FZ": "FPZ" in nodes,
        "ST18_FZ_CZ_derivable": st_fz_cz,
        "ST18_CZ_PZ_derivable": st_cz_pz,
        "ST18_both_midline_edges_derivable": st_fz_cz and st_cz_pz,
        "EventNet19_FZ_and_PZ_to_common_reference_derivable": bool(
            eventnet_references
        ),
        "common_reference_nodes_for_EventNet19": eventnet_references,
        "standard19_nodes_present": sorted(nodes.intersection(_STANDARD_19)),
    }


def _load_object(path: Path, context: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def audit_tusz_missing_midline_sources_v1(
    *,
    canonical_audit_path: str | Path,
    physical_projection_path: str | Path,
    tusz_root: str | Path,
) -> dict[str, Any]:
    audit_path = Path(canonical_audit_path).resolve(strict=True)
    projection_path = Path(physical_projection_path).resolve(strict=True)
    source_root = Path(tusz_root).resolve(strict=True)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    audit = _load_object(audit_path, "canonical audit")
    projection = _load_object(projection_path, "physical projection")

    outcomes = audit.get("outcomes")
    records = projection.get("records")
    if not isinstance(outcomes, list) or not isinstance(records, list):
        raise ValueError("canonical audit/projection inventory is missing")
    outcome_by_id: dict[str, Mapping[str, Any]] = {}
    for row in outcomes:
        if not isinstance(row, Mapping) or row.get("failure") is not None:
            continue
        identity = row.get("analysis_identity_id")
        if not isinstance(identity, str) or identity in outcome_by_id:
            raise ValueError("canonical audit identity is invalid or duplicated")
        outcome_by_id[identity] = row

    selected: list[tuple[str, Mapping[str, Any]]] = []
    for record in records:
        if not isinstance(record, Mapping) or record.get("analysis_unit_weight") != 1:
            continue
        identity = record.get("analysis_identity_id")
        if not isinstance(identity, str) or identity not in outcome_by_id:
            raise ValueError("weight-one projection identity lacks a successful outcome")
        outcome = outcome_by_id[identity]
        physical = outcome.get("physical_signal")
        if not isinstance(physical, Mapping):
            raise ValueError("canonical physical signal receipt is missing")
        channels = physical.get("observed_channel_ids")
        if not isinstance(channels, list):
            raise ValueError("canonical observed channel roster is missing")
        if "FZ" not in channels or "PZ" not in channels:
            selected.append((str(record.get("model_split")), outcome))

    split_counts: Counter[str] = Counter()
    montage_counts: Counter[str] = Counter()
    raw_roster_counts: Counter[tuple[str, ...]] = Counter()
    sampling_rate_counts: Counter[str] = Counter()
    source_class_counts: Counter[str] = Counter()
    sibling_counts: Counter[str] = Counter()
    parser_scope_sums: Counter[str] = Counter()
    all_receipts_valid = True
    all_paths_beneath_root = True

    for split, outcome in selected:
        relative = Path(str(outcome.get("local_edf_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("canonical local EDF path escapes source root")
        source = (source_root / relative).resolve(strict=True)
        try:
            source.relative_to(source_root)
        except ValueError as error:
            all_paths_beneath_root = False
            raise ValueError("canonical local EDF path escapes source root") from error
        physical = outcome["physical_signal"]
        external_binding = physical.get("canonical_source_signal_sha256")
        if not isinstance(external_binding, str):
            raise ValueError("canonical source signal binding is missing")
        receipt = parse_eeg_acquisition_header_v1(
            source,
            external_source_binding_sha256=external_binding,
        )
        labels = tuple(str(row["channel_label"]) for row in receipt["channels"])
        classification = classify_midline_derivability(labels)

        split_counts[split] += 1
        montage_counts[relative.parent.name] += 1
        raw_roster_counts[labels] += 1
        rates = sorted(
            {
                tuple(row["sampling_rate_hz_fraction"])
                for row in receipt["channels"]
            }
        )
        sampling_rate_counts[";".join(f"{n}/{d}" for n, d in rates)] += 1
        if (
            not classification["raw_FZ_node_present"]
            and not classification["raw_PZ_node_present"]
        ):
            source_class = "FZ_and_PZ_nodes_genuinely_absent_from_raw_signal_labels"
        elif classification["ST18_both_midline_edges_derivable"]:
            source_class = "ST18_midline_recoverable_by_reference_graph"
        else:
            source_class = "midline_nodes_partial_or_not_jointly_derivable"
        source_class_counts[source_class] += 1

        sibling_candidates = [
            candidate
            for candidate in source.parent.parent.glob(f"*/{source.name}")
            if candidate.resolve() != source
        ]
        if sibling_candidates:
            sibling_counts["same_recording_basename_in_alternate_montage_present"] += 1
        else:
            sibling_counts["no_same_recording_basename_in_alternate_montage"] += 1

        scope = receipt["scope_receipt"]
        for key in (
            "patient_identity_bytes_read",
            "recording_identity_or_free_text_bytes_read",
            "start_date_or_time_bytes_read",
            "transducer_free_text_bytes_read",
            "raw_prefilter_free_text_bytes_read",
            "reserved_free_text_bytes_read",
            "eeg_sample_payload_bytes_read",
            "annotation_payload_bytes_read",
        ):
            parser_scope_sums[key] += int(scope[key])
        all_receipts_valid = all_receipts_valid and len(receipt["receipt_sha256"]) == 64

    roster_summary = [
        {
            "roster_sha256": hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest(),
            "record_count": count,
            "signal_label_count": len(labels),
            "contains_FZ_token": classify_midline_derivability(labels)[
                "raw_FZ_node_present"
            ],
            "contains_PZ_token": classify_midline_derivability(labels)[
                "raw_PZ_node_present"
            ],
            "contains_FPZ_token_not_equivalent": classify_midline_derivability(labels)[
                "raw_FPZ_node_present_not_equivalent_to_FZ"
            ],
            "ST18_both_midline_edges_derivable": classify_midline_derivability(labels)[
                "ST18_both_midline_edges_derivable"
            ],
            "EventNet19_midline_referential_derivable": classify_midline_derivability(labels)[
                "EventNet19_FZ_and_PZ_to_common_reference_derivable"
            ],
        }
        for labels, count in sorted(
            raw_roster_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "canonical_audit_file_sha256": _file_sha256(audit_path),
        "physical_projection_file_sha256": _file_sha256(projection_path),
        "missing_midline_canonical_record_count": len(selected),
        "split_counts": dict(sorted(split_counts.items())),
        "source_montage_directory_counts": dict(sorted(montage_counts.items())),
        "source_sampling_rate_profile_counts": dict(sorted(sampling_rate_counts.items())),
        "source_classification_counts": dict(sorted(source_class_counts.items())),
        "same_basename_alternate_montage_counts": dict(sorted(sibling_counts.items())),
        "unique_raw_signal_label_roster_count": len(raw_roster_counts),
        "raw_signal_label_roster_summaries": roster_summary,
        "all_missing_records_ST18_midline_derivable": all(
            row["ST18_both_midline_edges_derivable"] for row in roster_summary
        ),
        "all_missing_records_EventNet19_midline_referential_derivable": all(
            row["EventNet19_midline_referential_derivable"] for row in roster_summary
        ),
        "all_source_paths_resolved_beneath_TUSZ_root": all_paths_beneath_root,
        "all_header_receipts_valid": all_receipts_valid,
        "aggregate_forbidden_byte_read_counts": dict(sorted(parser_scope_sums.items())),
        "csv_or_csv_bi_opened": False,
        "EDF_annotation_payload_opened": False,
        "EEG_sample_payload_opened": False,
        "spreadsheet_or_doctor_text_opened": False,
        "path_patient_session_or_filename_emitted": False,
        "scientific_interpretation": {
            "FPZ_may_be_substituted_for_FZ": False,
            "absent_electrode_may_be_reconstructed_from_unrelated_neighbors": False,
            "reference_graph_conversion_requires_target_nodes_to_exist": True,
            "support_route_should_remain_if_raw_FZ_and_PZ_nodes_are_absent": True,
        },
    }
    result["receipt_sha256"] = _sha256(result)
    return result


def validate_tusz_missing_midline_source_audit_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("missing-midline source audit must be an object")
    row = deepcopy(dict(value))
    supplied = row.pop("receipt_sha256", None)
    if supplied != _sha256(row):
        raise ValueError("missing-midline source audit hash mismatch")
    if row.get("schema_version") != SCHEMA_VERSION or row.get("method_id") != METHOD_ID:
        raise ValueError("missing-midline source audit schema/method drifted")
    if row.get("missing_midline_canonical_record_count") != sum(
        row.get("split_counts", {}).values()
    ):
        raise ValueError("missing-midline split denominator does not close")
    forbidden = row.get("aggregate_forbidden_byte_read_counts")
    if not isinstance(forbidden, Mapping) or any(value != 0 for value in forbidden.values()):
        raise ValueError("missing-midline audit read forbidden EDF bytes")
    for key in (
        "csv_or_csv_bi_opened",
        "EDF_annotation_payload_opened",
        "EEG_sample_payload_opened",
        "spreadsheet_or_doctor_text_opened",
        "path_patient_session_or_filename_emitted",
    ):
        if row.get(key) is not False:
            raise ValueError(f"missing-midline audit source firewall opened: {key}")
    return deepcopy(dict(value))


__all__ = [
    "METHOD_ID",
    "SCHEMA_VERSION",
    "audit_tusz_missing_midline_sources_v1",
    "classify_midline_derivability",
    "parse_eeg_channel_label_graph_edge",
    "validate_tusz_missing_midline_source_audit_v1",
]
