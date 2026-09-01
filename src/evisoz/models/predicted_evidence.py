"""Serialize Clinical Evidence decoder outputs into a typed JSON packet.

The packet is a research-shadow prediction, not a replacement for physician or
dataset labels.  It is the narrow bridge consumed by report planning and
evaluation: node and edge probabilities remain separate, masks are preserved,
and no text/knowledge source is consulted.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from src.evisoz.data.artifact_ref import canonical_json_sha256

from .clinical_evidence import (
    ClinicalEvidenceOutput,
    MOTIF_NAMES,
    N_EDGES,
    N_MOTIFS,
    N_NODES,
    QUERY_NAMES,
)


PREDICTED_EVIDENCE_SCHEMA_VERSION = "evisoz_predicted_evidence_v1"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-EVID-"


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["packet_id"] = "CONTENT-ADDRESS-PENDING"
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _finite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")


def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.ndim != 1 or mask.ndim != 1 or logits.shape != mask.shape:
        raise ValueError("masked softmax inputs must be matching vectors")
    if not torch.isfinite(logits).all():
        raise ValueError("decoder logits must be finite")
    if not mask.any():
        return torch.zeros_like(logits)
    values = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(values, dim=-1) * mask.to(logits.dtype)


def _probability_vector(value: Tensor, *, size: int, name: str) -> list[float]:
    if tuple(value.shape) != (1, size) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must have shape [1,{size}] and be finite")
    return torch.softmax(value[0], dim=-1).detach().cpu().tolist()


def build_predicted_evidence_packet(
    *,
    event_id: str,
    output: ClinicalEvidenceOutput,
    node_mask: Tensor,
    edge_mask: Tensor,
    candidate_node_mask: Tensor,
    node_units: Sequence[str],
    edge_units: Sequence[str],
    stage0_status: str = "NO_GO",
) -> dict[str, Any]:
    """Build one packet from a batch-one decoder output.

    The node and edge masks are reduced over time only for the event-level
    probability vectors; the original unit masks are retained as booleans.
    ``candidate_node_mask`` is an eligibility mask, never a label.
    """

    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")
    if stage0_status not in {"GO", "NO_GO"}:
        raise ValueError("stage0_status must be GO or NO_GO")
    for name, value, shape in (
        ("node_mask", node_mask, (1, N_NODES, node_mask.shape[-1] if isinstance(node_mask, Tensor) and node_mask.ndim == 3 else -1)),
        ("edge_mask", edge_mask, (1, N_EDGES, edge_mask.shape[-1] if isinstance(edge_mask, Tensor) and edge_mask.ndim == 3 else -1)),
    ):
        if not isinstance(value, Tensor) or value.ndim != 3 or value.dtype != torch.bool:
            raise ValueError(f"{name} must be bool with shape [1,units,T]")
        if value.shape[0] != shape[0] or value.shape[1] != shape[1] or value.shape[2] < 1:
            raise ValueError(f"{name} has invalid shape")
    if not isinstance(candidate_node_mask, Tensor) or candidate_node_mask.dtype != torch.bool or tuple(candidate_node_mask.shape) != (N_NODES,):
        raise ValueError("candidate_node_mask must have shape [19] and bool dtype")
    if len(node_units) != N_NODES or len(edge_units) != N_EDGES:
        raise ValueError("node/edge unit rosters have drifted")
    if output.onset_logits.shape != (1, N_NODES) or output.spread_logits.shape != (1, N_EDGES):
        raise ValueError("decoder output must be batch one with node/edge heads")
    if output.motif_logits.shape[0] != 1 or output.motif_logits.shape[-1] != N_MOTIFS:
        raise ValueError("motif logits shape drifted")
    if output.query_names != QUERY_NAMES:
        raise ValueError("query roster drifted")

    node_observed = node_mask[0].any(dim=-1)
    edge_observed = edge_mask[0].any(dim=-1)
    node_prob = _masked_softmax(output.onset_logits[0], node_observed & candidate_node_mask)
    edge_prob = _masked_softmax(output.spread_logits[0], edge_observed)
    cell_mask = node_mask[0]
    motif = output.motif_logits[0]
    if tuple(motif.shape[:2]) != tuple(cell_mask.shape):
        raise ValueError("motif logits and node mask time geometry drifted")
    if not torch.isfinite(motif).all():
        raise ValueError("motif logits must be finite")
    if cell_mask.any():
        pooled_motif = motif[cell_mask].mean(dim=0, keepdim=True)
        motif_prob = torch.softmax(pooled_motif[0], dim=-1)
    else:
        motif_prob = torch.zeros(N_MOTIFS, dtype=motif.dtype, device=motif.device)

    quality_prob = _probability_vector(output.quality_logits, size=3, name="quality_logits")
    evolution_prob = _probability_vector(output.evolution_logits, size=4, name="evolution_logits")
    localizability_prob = _probability_vector(output.localizability_logits, size=3, name="localizability_logits")
    uncertainty = 1.0 - max(localizability_prob)
    body: dict[str, Any] = {
        "schema_version": PREDICTED_EVIDENCE_SCHEMA_VERSION,
        "packet_id": _HASH_PLACEHOLDER,
        "event_id": event_id,
        "stage0_status": stage0_status,
        "status": "research_shadow",
        "node_units": [str(item) for item in node_units],
        "edge_units": [str(item) for item in edge_units],
        "candidate_node_mask": candidate_node_mask.detach().cpu().tolist(),
        "observed_node_mask": node_observed.detach().cpu().tolist(),
        "observed_edge_mask": edge_observed.detach().cpu().tolist(),
        "node_probabilities": [float(item) for item in node_prob.detach().cpu().tolist()],
        "edge_probabilities": [float(item) for item in edge_prob.detach().cpu().tolist()],
        "motif_probabilities": [float(item) for item in motif_prob.detach().cpu().tolist()],
        "quality_probabilities": quality_prob,
        "evolution_probabilities": evolution_prob,
        "localizability_probabilities": localizability_prob,
        "uncertainty": float(uncertainty),
        "permissions": {
            "can_create_patient_fact": False,
            "can_supervise_node_localization": False,
            "report_mode": "candidate_only",
            "knowledge_can_create_patient_fact": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["packet_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_predicted_evidence_packet(body)


def validate_predicted_evidence_packet(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "packet_id", "event_id", "stage0_status", "status",
        "node_units", "edge_units", "candidate_node_mask", "observed_node_mask",
        "observed_edge_mask", "node_probabilities", "edge_probabilities",
        "motif_probabilities", "quality_probabilities", "evolution_probabilities",
        "localizability_probabilities", "uncertainty", "permissions", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("predicted evidence packet fields drifted")
    data = deepcopy(value)
    _finite(data)
    if data["schema_version"] != PREDICTED_EVIDENCE_SCHEMA_VERSION or data["status"] != "research_shadow":
        raise ValueError("predicted evidence packet identity drifted")
    if data["stage0_status"] not in {"GO", "NO_GO"}:
        raise ValueError("predicted evidence packet Stage-0 status drifted")
    for key, size in (
        ("node_units", N_NODES), ("edge_units", N_EDGES),
        ("candidate_node_mask", N_NODES), ("observed_node_mask", N_NODES),
        ("observed_edge_mask", N_EDGES), ("node_probabilities", N_NODES),
        ("edge_probabilities", N_EDGES), ("motif_probabilities", N_MOTIFS),
        ("quality_probabilities", 3), ("evolution_probabilities", 4),
        ("localizability_probabilities", 3),
    ):
        if not isinstance(data[key], list) or len(data[key]) != size:
            raise ValueError(f"predicted evidence {key} length drifted")
    for key in ("candidate_node_mask", "observed_node_mask", "observed_edge_mask"):
        if any(type(item) is not bool for item in data[key]):
            raise ValueError(f"predicted evidence {key} must be boolean")
    for key in (
        "node_probabilities", "edge_probabilities", "motif_probabilities",
        "quality_probabilities", "evolution_probabilities", "localizability_probabilities",
    ):
        values = data[key]
        if any(not isinstance(item, (int, float)) or not 0 <= float(item) <= 1 for item in values):
            raise ValueError(f"predicted evidence {key} contains invalid probabilities")
        if values and sum(float(item) for item in values) > 1.00001 and key != "node_probabilities" and key != "edge_probabilities":
            raise ValueError(f"predicted evidence {key} is not a probability vector")
    if not 0 <= float(data["uncertainty"]) <= 1:
        raise ValueError("predicted evidence uncertainty is invalid")
    if data["permissions"] != {
        "can_create_patient_fact": False,
        "can_supervise_node_localization": False,
        "report_mode": "candidate_only",
        "knowledge_can_create_patient_fact": False,
    }:
        raise ValueError("predicted evidence permission policy drifted")
    if data["packet_id"] != _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]:
        raise ValueError("predicted evidence packet ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("predicted evidence packet receipt drifted")
    return data


__all__ = [
    "PREDICTED_EVIDENCE_SCHEMA_VERSION",
    "build_predicted_evidence_packet",
    "validate_predicted_evidence_packet",
]
