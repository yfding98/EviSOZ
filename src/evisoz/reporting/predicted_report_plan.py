"""Deterministic report-plan projection from a predicted evidence packet.

The plan is intentionally lexical and candidate-only.  It is suitable as a
future Qwen input target, but it is not a clinical report and never creates a
SOZ fact from a probability vector alone.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

from src.evisoz.data.artifact_ref import canonical_json_sha256
from src.evisoz.models.predicted_evidence import validate_predicted_evidence_packet


PREDICTED_REPORT_PLAN_SCHEMA_VERSION = "evisoz_predicted_report_plan_v1"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-RPLAN-"


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["plan_id"] = "CONTENT-ADDRESS-PENDING"
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _top_indices(values: Sequence[float], *, k: int, allowed: Sequence[bool]) -> list[int]:
    rows = [(float(value), index) for index, value in enumerate(values) if allowed[index] and float(value) > 0]
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [index for _, index in rows[:k]]


def build_predicted_report_plan(
    packet: Mapping[str, object],
    *,
    knowledge_card_ids: Sequence[str] = (),
    top_k_nodes: int = 3,
    top_k_edges: int = 3,
) -> dict[str, Any]:
    """Build a candidate-only report plan from one validated packet."""

    evidence = validate_predicted_evidence_packet(packet)
    if top_k_nodes < 1 or top_k_edges < 1:
        raise ValueError("top-k values must be positive")
    cards = sorted({str(item) for item in knowledge_card_ids if str(item)})
    node_indices = _top_indices(
        evidence["node_probabilities"],
        k=top_k_nodes,
        allowed=[
            bool(a) and bool(b)
            for a, b in zip(evidence["candidate_node_mask"], evidence["observed_node_mask"])
        ],
    )
    edge_indices = _top_indices(
        evidence["edge_probabilities"],
        k=top_k_edges,
        allowed=evidence["observed_edge_mask"],
    )
    motif_indices = _top_indices(
        evidence["motif_probabilities"],
        k=1,
        allowed=[True] * len(evidence["motif_probabilities"]),
    )
    claims: list[dict[str, object]] = []
    for index in node_indices:
        claims.append({
            "claim_id": f"{evidence['packet_id']}:onset_node:{index}",
            "concept": "onset_candidate_node",
            "support_units": [evidence["node_units"][index]],
            "confidence": float(evidence["node_probabilities"][index]),
            "assertion_level": "model_candidate",
            "status": "candidate_only",
        })
    for index in edge_indices:
        claims.append({
            "claim_id": f"{evidence['packet_id']}:spread_candidate_edge:{index}",
            "concept": "spread_candidate_edge",
            "support_units": [evidence["edge_units"][index]],
            "confidence": float(evidence["edge_probabilities"][index]),
            "assertion_level": "model_candidate",
            "status": "candidate_only",
        })
    if motif_indices:
        index = motif_indices[0]
        claims.append({
            "claim_id": f"{evidence['packet_id']}:motif:{index}",
            "concept": "motif_candidate",
            "support_units": [],
            "motif": index,
            "confidence": float(evidence["motif_probabilities"][index]),
            "assertion_level": "model_candidate",
            "status": "candidate_only",
        })
    claims.sort(key=lambda row: str(row["claim_id"]))
    node_text = "、".join(
        f"{evidence['node_units'][i]}（{float(evidence['node_probabilities'][i]):.3f}）"
        for i in node_indices
    ) or "未形成可用节点候选"
    edge_text = "、".join(
        f"{evidence['edge_units'][i]}（{float(evidence['edge_probabilities'][i]):.3f}）"
        for i in edge_indices
    ) or "未形成可用边候选"
    motif_text = (
        f"{motif_indices[0]}（{float(evidence['motif_probabilities'][motif_indices[0]]):.3f}）"
        if motif_indices else "未形成可用形态候选"
    )
    sections = [
        {
            "section_id": "analysis_scope",
            "text_zh": "本计划仅基于已给定发作片段的模型证据，属于 research-shadow candidate，不是临床签署报告。",
            "claim_ids": [],
            "knowledge_card_ids": [],
        },
        {
            "section_id": "candidate_onset",
            "text_zh": f"模型给出的起始节点候选（不等同于 SOZ 标签）：{node_text}。",
            "claim_ids": [str(row["claim_id"]) for row in claims if row["concept"] == "onset_candidate_node"],
            "knowledge_card_ids": cards,
        },
        {
            "section_id": "candidate_spread",
            "text_zh": f"模型给出的有符号 TCP22 边候选：{edge_text}；边不能展开为端点阳性。",
            "claim_ids": [str(row["claim_id"]) for row in claims if row["concept"] == "spread_candidate_edge"],
            "knowledge_card_ids": cards,
        },
        {
            "section_id": "candidate_morphology",
            "text_zh": f"模型形态候选索引：{motif_text}；需由独立信号证据和伪迹检查复核。",
            "claim_ids": [str(row["claim_id"]) for row in claims if row["concept"] == "motif_candidate"],
            "knowledge_card_ids": cards,
        },
        {
            "section_id": "uncertainty",
            "text_zh": f"模型不确定性为 {float(evidence['uncertainty']):.3f}；当前计划不升级 certainty。",
            "claim_ids": [],
            "knowledge_card_ids": cards,
        },
        {
            "section_id": "limitations",
            "text_zh": "仅凭头皮 EEG 模型候选不能确认皮层起始、临床 SOZ、致痫区或手术靶点；需要冻结证据、校准和医师复核。",
            "claim_ids": [],
            "knowledge_card_ids": cards,
        },
    ]
    body: dict[str, Any] = {
        "schema_version": PREDICTED_REPORT_PLAN_SCHEMA_VERSION,
        "plan_id": _HASH_PLACEHOLDER,
        "event_id": evidence["event_id"],
        "source_packet": {
            "packet_id": evidence["packet_id"],
            "receipt_sha256": evidence["receipt_sha256"],
        },
        "stage0_status": evidence["stage0_status"],
        "status": "model_candidate_shadow",
        "sections": sections,
        "claims": claims,
        "permissions": {
            "can_create_patient_fact": False,
            "can_supervise_node_localization": False,
            "knowledge_can_create_patient_fact": False,
            "qwen_may_lexicalize_only": True,
            "requires_physician_review": True,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["plan_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_predicted_report_plan(body)


def validate_predicted_report_plan(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "plan_id", "event_id", "source_packet", "stage0_status",
        "status", "sections", "claims", "permissions", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("predicted report plan fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PREDICTED_REPORT_PLAN_SCHEMA_VERSION or data["status"] != "model_candidate_shadow":
        raise ValueError("predicted report plan identity drifted")
    if data["stage0_status"] not in {"GO", "NO_GO"}:
        raise ValueError("predicted report plan Stage-0 status drifted")
    source = data["source_packet"]
    if type(source) is not dict or set(source) != {"packet_id", "receipt_sha256"}:
        raise ValueError("predicted report plan source packet drifted")
    if not isinstance(source["packet_id"], str) or not source["packet_id"].startswith("EVISOZ-EVID-"):
        raise ValueError("predicted report plan source packet ID drifted")
    if not isinstance(source["receipt_sha256"], str) or len(source["receipt_sha256"]) != 64:
        raise ValueError("predicted report plan source packet receipt drifted")
    sections = data["sections"]
    if not isinstance(sections, list) or len(sections) < 6:
        raise ValueError("predicted report plan section roster is incomplete")
    section_ids = [row.get("section_id") for row in sections if isinstance(row, dict)]
    if len(section_ids) != len(sections) or len(section_ids) != len(set(section_ids)):
        raise ValueError("predicted report plan section IDs drifted")
    for row in sections:
        if set(row) != {"section_id", "text_zh", "claim_ids", "knowledge_card_ids"}:
            raise ValueError("predicted report plan section fields drifted")
        if not isinstance(row["text_zh"], str) or not row["text_zh"]:
            raise ValueError("predicted report plan section text is empty")
        if not isinstance(row["claim_ids"], list) or not isinstance(row["knowledge_card_ids"], list):
            raise ValueError("predicted report plan section references drifted")
    claims = data["claims"]
    if not isinstance(claims, list):
        raise ValueError("predicted report plan claims must be an array")
    claim_ids: set[str] = set()
    for claim in claims:
        if type(claim) is not dict or not {"claim_id", "concept", "support_units", "confidence", "assertion_level", "status"}.issubset(claim):
            raise ValueError("predicted report plan claim fields drifted")
        claim_id = claim["claim_id"]
        if not isinstance(claim_id, str) or claim_id in claim_ids:
            raise ValueError("predicted report plan claim IDs drifted")
        claim_ids.add(claim_id)
        if claim["assertion_level"] != "model_candidate" or claim["status"] != "candidate_only":
            raise ValueError("predicted report plan claim permission drifted")
        if not isinstance(claim["support_units"], list) or not 0 <= float(claim["confidence"]) <= 1:
            raise ValueError("predicted report plan claim confidence drifted")
    if data["permissions"] != {
        "can_create_patient_fact": False,
        "can_supervise_node_localization": False,
        "knowledge_can_create_patient_fact": False,
        "qwen_may_lexicalize_only": True,
        "requires_physician_review": True,
    }:
        raise ValueError("predicted report plan permission policy drifted")
    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["plan_id"] != expected_id:
        raise ValueError("predicted report plan ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("predicted report plan receipt drifted")
    return data


__all__ = [
    "PREDICTED_REPORT_PLAN_SCHEMA_VERSION",
    "build_predicted_report_plan",
    "validate_predicted_report_plan",
]
