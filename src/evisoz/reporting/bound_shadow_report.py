"""Loader-bound report-plan construction with knowledge/eeg safety limits."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.evisoz.data.bound_evidence_loader import BoundEvidenceRecord
from src.evisoz.models.predicted_evidence import validate_predicted_evidence_packet

from .predicted_report_plan import build_predicted_report_plan


def selected_knowledge_card_ids(record: BoundEvidenceRecord) -> tuple[str, ...]:
    """Return only card IDs already fixed by the validated selection receipt."""

    if not isinstance(record, BoundEvidenceRecord):
        raise TypeError("record must come from bound_evidence_loader")
    selection = record.knowledge_selection
    if selection is None:
        return ()
    if selection.get("patient_fact_creation_allowed") is not False or selection.get("can_add_patient_fact") is not False:
        raise PermissionError("knowledge selection is not non-patient-fact")
    cards = selection.get("selected_cards")
    if not isinstance(cards, list):
        raise ValueError("knowledge selection cards are missing")
    result: list[str] = []
    for row in cards:
        if not isinstance(row, Mapping) or not isinstance(row.get("card_id"), str) or not row["card_id"]:
            raise ValueError("knowledge selection card identity drifted")
        result.append(str(row["card_id"]))
    return tuple(sorted(set(result)))


def build_bound_shadow_report_plan(
    record: BoundEvidenceRecord,
    packet: Mapping[str, object],
    *,
    knowledge_card_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a candidate-only plan using the record's fixed knowledge receipt.

    Explicit card IDs are accepted only as an equality-checked replay of the
    bound selection.  Callers cannot inject arbitrary knowledge cards or use
    knowledge to create patient facts.
    """

    evidence = validate_predicted_evidence_packet(dict(packet))
    if evidence["event_id"] != record.event_id:
        raise ValueError("packet event does not match bound record")
    selected = selected_knowledge_card_ids(record)
    if knowledge_card_ids is not None:
        supplied = tuple(sorted({str(item) for item in knowledge_card_ids if str(item)}))
        if supplied != selected:
            raise ValueError("explicit knowledge cards do not replay bound selection")
    return build_predicted_report_plan(evidence, knowledge_card_ids=selected)


__all__ = ["build_bound_shadow_report_plan", "selected_knowledge_card_ids"]
