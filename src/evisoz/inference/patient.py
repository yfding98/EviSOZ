"""Patient-level aggregation for loader-backed candidate predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch

from src.evisoz.data.bound_evidence_loader import BoundEvidenceRecord
from src.evisoz.models.clinical_evidence import PatientEvidenceAggregator
from src.evisoz.models.predicted_evidence import validate_predicted_evidence_packet
from src.evisoz.reporting.qwen_patient_input import build_qwen_patient_input


def aggregate_bound_shadow_predictions(
    records: Iterable[BoundEvidenceRecord],
    predictions: Iterable[object],
    *,
    localizing_quality_threshold: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Aggregate event packets by opaque linkage group, never by raw patient ID.

    Quality/localizability values are model probabilities, not labels.  The
    helper therefore emits a shadow aggregate and retains abstention state;
    it never upgrades a non-localizing event into a localization target.
    """

    if not 0 <= localizing_quality_threshold <= 1:
        raise ValueError("localizing_quality_threshold must be in [0,1]")
    record_rows = list(records)
    prediction_rows = list(predictions)
    if not record_rows or len(record_rows) != len(prediction_rows):
        raise ValueError("records and predictions must be non-empty and aligned")
    grouped: dict[str, list[tuple[BoundEvidenceRecord, Mapping[str, Any]]]] = defaultdict(list)
    seen: set[str] = set()
    for record, prediction in zip(record_rows, prediction_rows):
        if not isinstance(record, BoundEvidenceRecord):
            raise TypeError("records must come from bound_evidence_loader")
        packet = getattr(prediction, "predicted_evidence", prediction)
        if not isinstance(packet, Mapping):
            raise ValueError("prediction lacks predicted_evidence packet")
        packet = validate_predicted_evidence_packet(dict(packet))
        if packet["event_id"] != record.event_id or record.event_id in seen:
            raise ValueError("patient aggregation event identity/uniqueness drifted")
        seen.add(record.event_id)
        grouped[record.linkage_group_id].append((record, packet))
    if not grouped:
        raise ValueError("patient aggregation received no events")

    output: dict[str, dict[str, Any]] = {}
    for linkage_group_id, rows in sorted(grouped.items()):
        event_ids = [record.event_id for record, _ in rows]
        probabilities = torch.tensor(
            [packet["node_probabilities"] for _, packet in rows], dtype=torch.float32
        )
        signal_quality = torch.tensor(
            [max(packet["quality_probabilities"]) for _, packet in rows], dtype=torch.float32
        )
        localization_quality = torch.tensor(
            [max(packet["localizability_probabilities"]) for _, packet in rows], dtype=torch.float32
        )
        uncertainty = torch.tensor(
            [packet["uncertainty"] for _, packet in rows], dtype=torch.float32
        )
        localizing = localization_quality >= localizing_quality_threshold
        aggregate = PatientEvidenceAggregator.aggregate(
            probabilities,
            localization_quality,
            signal_quality,
            uncertainty,
            localizing_event_mask=localizing,
        )
        aggregate["event_ids"] = event_ids
        aggregate["linkage_group_id"] = linkage_group_id
        aggregate["status"] = "model_candidate_shadow"
        output[linkage_group_id] = aggregate
    return output


def build_bound_patient_qwen_shadow_inputs(
    records: Iterable[BoundEvidenceRecord],
) -> dict[str, dict[str, Any]]:
    """Build patient-level Qwen shadow packets from loader-bound report graphs.

    The packets are derived from the already materialized signal-candidate
    graph, canonical shadow report, and knowledge selection.  They do not use
    model predictions to create facts and remain no-generation inputs.
    """

    rows = list(records)
    if not rows:
        raise ValueError("patient Qwen shadow inputs require non-empty records")
    grouped: dict[str, list[BoundEvidenceRecord]] = defaultdict(list)
    for record in rows:
        if not isinstance(record, BoundEvidenceRecord):
            raise TypeError("records must come from bound_evidence_loader")
        grouped[record.linkage_group_id].append(record)
    output: dict[str, dict[str, Any]] = {}
    for linkage_group_id, group in sorted(grouped.items()):
        first = group[0]
        graph = first.signal_candidate_claim_graph
        report = first.canonical_report
        selection = first.knowledge_selection
        if graph is None or report is None or selection is None:
            raise ValueError(
                f"patient {linkage_group_id} lacks a fully bound shadow report route"
            )
        for record in group[1:]:
            if (
                record.signal_candidate_claim_graph is None
                or record.canonical_report is None
                or record.knowledge_selection is None
                or record.signal_candidate_claim_graph["graph_id"] != graph["graph_id"]
                or record.canonical_report["report_id"] != report["report_id"]
                or record.knowledge_selection["selection_id"] != selection["selection_id"]
            ):
                raise ValueError("patient-level bound report authorities drifted")
        output[linkage_group_id] = build_qwen_patient_input(
            signal_graph=graph,
            canonical_report=report,
            knowledge_selection=selection,
        )
    return output


__all__ = [
    "aggregate_bound_shadow_predictions",
    "build_bound_patient_qwen_shadow_inputs",
]
