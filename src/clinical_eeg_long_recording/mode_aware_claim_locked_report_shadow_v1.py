"""Deterministic claim-locked report preview for the public/synthetic MIL path.

This module closes one research-only replay path across event Findings v3,
mode-aware MIL, report graph v2, atomic claims, Chinese text, and waveform-panel
metadata.  It deliberately does *not* promote the existing MIL/report-graph
bridge: its authorized claim, Qwen, and formal renderer overlays remain empty.

The output is an audit preview, not a clinical report.  Candidate ranking
values are copied as uncalibrated probability-like scores, every signal
sentence has exactly one content-addressed claim owner, and Qwen receives only
immutable lexical slots whose proposals cannot replace the canonical
deterministic text.  No raw EDF, annotation, spreadsheet, doctor label,
clinical narrative, or private path is opened here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .mode_aware_hierarchical_positive_set_mil_v1 import (
    CompleteRecordModeAwareMILBagV1,
    ModeAwareMILForwardV1,
    ModeAwareMILPolicyV1,
)
from .mode_aware_mil_report_graph_v2_bridge_shadow_v1 import (
    MIL_HARD_INPUT_ROLES,
    ModeAwareMILHardInputBindingV1,
    validate_mode_aware_mil_report_graph_v2_bridge_shadow_v1,
)


MODE_AWARE_CLAIM_LOCKED_REPORT_SHADOW_SCHEMA_VERSION = (
    "clinical_eeg_mode_aware_claim_locked_report_shadow_v1"
)
MODE_AWARE_CLAIM_LOCKED_REPORT_SHADOW_ROUTE_CONNECTED = False
MODE_AWARE_CLAIM_LOCKED_REPORT_SHADOW_RENDERER_ID = (
    "deterministic_public_synthetic_claim_locked_preview_zh_v1"
)

_SECTION_ORDER = (
    "boundary",
    "event_findings",
    "multievent_summary",
    "impression",
    "provenance",
    "waveform_index",
    "limitations",
)
_SECTION_LABELS = {
    "boundary": "输出边界",
    "event_findings": "事件脑电所见（研究性影子）",
    "multievent_summary": "多事件与模式汇总",
    "impression": "脑电图印象（研究性候选）",
    "provenance": "证据链",
    "waveform_index": "波形证据索引",
    "limitations": "局限性",
}
_TERM_LABELS = {
    "theta_band_power_increase": "θ频带功率增加",
    "rhythmic_theta_activity": "θ频段节律活动候选",
    "left_temporal_recruitment_candidate": "左侧颞区随后累及候选",
    "right_temporal_recruitment_candidate": "右侧颞区随后累及候选",
    "left_temporal_onset_candidate": "左侧颞区头皮可见起始候选",
    "right_temporal_onset_candidate": "右侧颞区头皮可见起始候选",
}
_ENTITY_LABELS = {
    "left": "左侧",
    "right": "右侧",
    "bilateral": "双侧",
    "midline": "中线",
    "indeterminate": "侧别不定",
    "left_temporal": "左侧颞区",
    "right_temporal": "右侧颞区",
    "left_frontal": "左侧额区",
    "right_frontal": "右侧额区",
    "localized_or_lateralized_scalp_visible_onset_pattern": ("局灶或侧化的头皮可见起始模式"),
    "widespread_bilateral_near_synchronous_scalp_onset_pattern": ("广泛双侧近同步头皮起始模式"),
    "scalp_onset_nonlocalizable": "头皮起始不可定位",
}
_ASSERTION_LABELS = {
    "measured": "可重放测量",
    "model_candidate": "模型候选",
    "report_eligible_automated": "来源自动报告资格项",
}
_STATUS_LABELS = {
    "present": "见",
    "absent_with_opportunity": "在完整评价机会下未见",
    "uncertain": "是否存在不确定",
    "not_evaluable": "不可评价",
}


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _bounded_id(prefix: str, value: object) -> str:
    return f"{prefix}-{_canonical_sha256({'domain': prefix, 'value': value})[:24]}"


def _seal(value: dict[str, Any], field: str, domain: str) -> None:
    value[field] = "0" * 64
    value[field] = _canonical_sha256({"binding_domain": domain, "value": value})


def _format_time(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("recording-relative time must be finite and non-negative")
    milliseconds = int(round(seconds * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _score(value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("candidate probability-like score must lie in [0, 1]")
    return result


def _normalize_ranking(value: object, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping) or set(item) != {
            "rank",
            "candidate_id",
            "score",
        }:
            raise ValueError(f"{context} has an invalid ranking row")
        if int(item["rank"]) != index:
            raise ValueError(f"{context} ranks must be consecutive")
        candidate_id = str(item["candidate_id"])
        if not candidate_id:
            raise ValueError(f"{context} candidate_id is empty")
        rows.append(
            {
                "rank": index,
                "candidate_id": candidate_id,
                "uncalibrated_probability_like_score": _score(item["score"]),
            }
        )
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError(f"{context} repeats a candidate")
    return rows


def _measurement_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    projected = {
        "measurement_id": str(value["measurement_id"]),
        "name_id": str(value["name_id"]),
        "value": float(value["value"]),
        "unit_id": str(value["unit_id"]),
        "baseline_delta": (
            None if value["baseline_delta"] is None else float(value["baseline_delta"])
        ),
        "numerical_uncertainty": deepcopy(value["numerical_uncertainty"]),
        "producer_type": str(value["producer_type"]),
        "source_binding_sha256": _canonical_sha256(value["source_binding"]),
        "source_view_id": str(value["source_binding"]["source_view_id"]),
        "view_role": str(value["source_binding"]["view_role"]),
        "reference_type": str(value["source_binding"]["reference_type"]),
        "effective_bandwidth_hz": [
            float(item) for item in value["source_binding"]["effective_bandwidth_hz"]
        ],
        "raw_sample_dependency_id": str(
            value["source_binding"]["raw_sample_dependency"]["dependency_id"]
        ),
    }
    projected["measurement_projection_sha256"] = _canonical_sha256(projected)
    return projected


def _atomic_claim_projections(
    graph: Mapping[str, Any], sources: Sequence[object]
) -> list[dict[str, Any]]:
    source_by_event: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise TypeError("trusted event Findings v3 source must be an object")
        event_id = str(source["event_id"])
        if event_id in source_by_event:
            raise ValueError("trusted event Findings v3 repeats an event")
        source_by_event[event_id] = source
    graph_event_order = [str(item["event_id"]) for item in graph["source_event_graphs"]]
    if set(source_by_event) != set(graph_event_order):
        raise ValueError("trusted Findings and report graph disagree on event roster")

    graph_claim_by_key = {
        (str(item["event_id"]), str(item["source_evidence_ids"][0])): item
        for item in graph["claims"]
        if item["claim_kind"] == "finding_state"
        and len(item["source_evidence_ids"]) == 1
    }
    graph_node_by_key = {
        (str(item["event_id"]), str(item["evidence_id"])): item
        for item in graph["finding_evidence_nodes"]
    }
    rows: list[dict[str, Any]] = []
    for event_id in graph_event_order:
        source = source_by_event[event_id]
        for finding in source["findings"]:
            evidence_id = str(finding["evidence_id"])
            key = (event_id, evidence_id)
            claim = graph_claim_by_key.get(key)
            node = graph_node_by_key.get(key)
            if claim is None or node is None:
                raise ValueError("one source Finding lacks an atomic graph claim/node")
            source_sha = _canonical_sha256(finding)
            if (
                source_sha != node["source_finding_sha256"]
                or source_sha != claim["source_refs"][0]["object_sha256"]
            ):
                raise ValueError("source Finding hash is not claim/node closed")
            row = {
                "claim_id": str(claim["claim_id"]),
                "claim_sha256": str(claim["claim_sha256"]),
                "claim_kind": "atomic_event_finding",
                "record_id": str(graph["record"]["record_id"]),
                "event_id": event_id,
                "evidence_id": evidence_id,
                "source_finding_sha256": source_sha,
                "term_id": str(finding["term"]["term_id"]),
                "family": str(finding["family"]),
                "status": str(finding["status"]),
                "assertion_level": str(finding["assertion_level"]),
                "intrinsic_evidence_role": str(finding["intrinsic_evidence_role"]),
                "signal_temporal_context": str(finding["signal_temporal_context"]),
                "time_interval": deepcopy(finding["time_interval"]),
                "spatial_support": deepcopy(finding["spatial_support"]),
                "measurements": [
                    _measurement_projection(item) for item in finding["measurements"]
                ],
                "uncertainty": deepcopy(finding["uncertainty"]),
                "permission_edge_ids": [
                    str(item) for item in claim["permission_edge_ids"]
                ],
                "waveform_evidence_ids": [
                    str(item) for item in finding["waveform_evidence_ids"]
                ],
                "raw_sample_dependency_ids": [
                    str(item) for item in finding["raw_sample_dependency_ids"]
                ],
                "formal_clinical_claim_authorized": False,
                "clinical_correctness_claimed": False,
                "projection_sha256": "",
            }
            _seal(
                row,
                "projection_sha256",
                "clinical-eeg-atomic-finding-shadow-claim-v1",
            )
            rows.append(row)
    return rows


def _mode_claims(
    bridge: Mapping[str, Any],
    graph: Mapping[str, Any],
    atomic_claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    overlay_modes = {
        str(item["mode_id"]): item
        for item in bridge["evaluation_only_candidate_overlay"]["modes"]
    }
    mode_bindings = {
        str(item["mil_mode_id"]): item for item in bridge["mil_mode_bindings"]
    }
    if set(overlay_modes) != set(mode_bindings):
        raise ValueError("MIL mode ranking and source mode binding rosters differ")
    hard_bindings = bridge["hard_onset_input_role_bindings"]
    panels_by_event: dict[str, list[str]] = defaultdict(list)
    for panel in graph["waveform_panels"]:
        panels_by_event[str(panel["event_id"])].append(str(panel["panel_id"]))
    atomic_by_key = {
        (str(item["event_id"]), str(item["evidence_id"])): str(item["claim_id"])
        for item in atomic_claims
    }

    rows: list[dict[str, Any]] = []
    for mode_id in sorted(overlay_modes):
        candidate = overlay_modes[mode_id]
        binding = mode_bindings[mode_id]
        event_ids = [str(item) for item in binding["event_ids"]]
        bound_inputs = [
            item for item in hard_bindings if str(item["event_id"]) in event_ids
        ]
        expected_count = len(event_ids) * len(MIL_HARD_INPUT_ROLES)
        if len(bound_inputs) != expected_count:
            raise ValueError("one mode lacks the exact hard-input role roster")
        evidence_keys = sorted(
            {
                (str(key[0]), str(key[1]))
                for item in bound_inputs
                for key in item["source_evidence_keys"]
            }
        )
        source_atomic_claim_ids = sorted(
            {atomic_by_key[key] for key in evidence_keys if key in atomic_by_key}
        )
        evidence_chain = {
            "event_ids": event_ids,
            "event_scoped_evidence_keys": [list(item) for item in evidence_keys],
            "source_atomic_claim_ids": source_atomic_claim_ids,
            "hard_input_binding_sha256s": sorted(
                str(item["binding_sha256"]) for item in bound_inputs
            ),
            "hard_input_roles": sorted(
                {str(item["input_role"]) for item in bound_inputs}
            ),
            "permission_edge_ids": sorted(
                {
                    str(edge_id)
                    for item in bound_inputs
                    for edge_id in item["permission_edge_ids"]
                }
            ),
            "raw_sample_dependency_ids": sorted(
                {
                    str(dependency_id)
                    for item in bound_inputs
                    for dependency_id in item["raw_sample_dependency_ids"]
                }
            ),
            "constructive_spatial_receipt_ids": sorted(
                {
                    str(receipt_id)
                    for item in bound_inputs
                    for receipt_id in item["constructive_spatial_receipt_ids"]
                }
            ),
            "waveform_panel_ids": sorted(
                {
                    panel_id
                    for event_id in event_ids
                    for panel_id in panels_by_event[event_id]
                }
            ),
        }
        evidence_chain["evidence_chain_sha256"] = _canonical_sha256(evidence_chain)
        identity = {
            "mode_id": mode_id,
            "decode_sha256": bridge["source_binding"]["decode_sha256"],
            "evidence_chain_sha256": evidence_chain["evidence_chain_sha256"],
        }
        row = {
            "claim_id": _bounded_id("SHADOWMODECLAIM", identity),
            "claim_kind": "uncalibrated_mode_onset_candidate_ranking",
            "record_id": str(graph["record"]["record_id"]),
            "mode_id": mode_id,
            "event_ids": event_ids,
            "physical_occurrence_sha256s": deepcopy(
                binding["physical_occurrence_sha256s"]
            ),
            "phenotype_probability_like_ranking": _normalize_ranking(
                candidate["phenotype_ranking"], f"mode {mode_id} phenotype"
            ),
            "channel_probability_like_ranking": _normalize_ranking(
                candidate["channel_hard_onset_ranking"],
                f"mode {mode_id} channel",
            ),
            "region_probability_like_ranking": _normalize_ranking(
                candidate["region_hard_onset_ranking"], f"mode {mode_id} region"
            ),
            "laterality_probability_like_ranking": _normalize_ranking(
                candidate["laterality_ranking"], f"mode {mode_id} laterality"
            ),
            "candidate_unit_type": str(candidate["candidate_unit_type"]),
            "score_semantics": "uncalibrated_probability_like_candidate_score",
            "assertion_status": "research_candidate",
            "evidence_chain": evidence_chain,
            "formal_clinical_claim_authorized": False,
            "qwen_may_change_claim": False,
            "claim_sha256": "",
        }
        _seal(row, "claim_sha256", "clinical-eeg-mode-shadow-claim-v1")
        rows.append(row)
    return rows


def _record_claim(
    bridge: Mapping[str, Any],
    graph: Mapping[str, Any],
    mode_claims: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    overlay = bridge["evaluation_only_candidate_overlay"]
    row = {
        "claim_id": _bounded_id(
            "SHADOWRECORDCLAIM",
            {
                "record_id": graph["record"]["record_id"],
                "mode_claim_sha256s": [item["claim_sha256"] for item in mode_claims],
                "decode_sha256": overlay["decode_sha256"],
            },
        ),
        "claim_kind": "mode_aware_record_candidate_summary",
        "record_id": str(graph["record"]["record_id"]),
        "record_phenotype": str(overlay["record_phenotype"]),
        "mode_claim_ids": [str(item["claim_id"]) for item in mode_claims],
        "mode_count": len(mode_claims),
        "multiple_mode_record_average_withheld": bool(
            overlay["multiple_mode_record_average_withheld"]
        ),
        "record_axis_candidate_available": bool(
            overlay["record_hard_onset"]["record_axis_candidate_available"]
        ),
        "formal_clinical_claim_authorized": False,
        "claim_sha256": "",
    }
    _seal(row, "claim_sha256", "clinical-eeg-record-shadow-claim-v1")
    return row


def _provenance_edges(
    atomic_claims: Sequence[Mapping[str, Any]],
    mode_claims: Sequence[Mapping[str, Any]],
    record_claim: Mapping[str, Any],
) -> list[dict[str, Any]]:
    atomic_ids = {str(item["claim_id"]) for item in atomic_claims}
    rows: list[dict[str, Any]] = []
    for mode in mode_claims:
        mode_id = str(mode["claim_id"])
        for atomic_id in mode["evidence_chain"]["source_atomic_claim_ids"]:
            if atomic_id not in atomic_ids:
                raise ValueError(
                    "mode evidence chain references an unknown atomic claim"
                )
            rows.append(
                {
                    "edge_id": _bounded_id(
                        "SHADOWEDGE",
                        {"source": atomic_id, "target": mode_id},
                    ),
                    "source_claim_id": atomic_id,
                    "target_claim_id": mode_id,
                    "relation": "event_evidence_contributes_to_mode_candidate",
                    "formal_derivation_authorized": False,
                    "edge_sha256": "",
                }
            )
        rows.append(
            {
                "edge_id": _bounded_id(
                    "SHADOWEDGE",
                    {"source": mode_id, "target": record_claim["claim_id"]},
                ),
                "source_claim_id": mode_id,
                "target_claim_id": str(record_claim["claim_id"]),
                "relation": "mode_candidate_included_in_record_summary",
                "formal_derivation_authorized": False,
                "edge_sha256": "",
            }
        )
    for row in rows:
        _seal(row, "edge_sha256", "clinical-eeg-shadow-provenance-edge-v1")
    return sorted(rows, key=lambda item: str(item["edge_id"]))


def _entity_label(identifier: str) -> str:
    return _ENTITY_LABELS.get(identifier, identifier)


def _ranking_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "无"
    return "、".join(
        f"{_entity_label(str(item['candidate_id']))}="
        f"{float(item['uncalibrated_probability_like_score']):.6f}"
        for item in rows
    )


def _atomic_text(claim: Mapping[str, Any], event_ordinal: int) -> str:
    interval = claim["time_interval"]
    when = (
        f"相对记录 {_format_time(float(interval['start']))}–"
        f"{_format_time(float(interval['stop']))}"
    )
    term = _TERM_LABELS.get(str(claim["term_id"]), str(claim["term_id"]))
    spatial = [
        str(item["id"]) + ("（空间证据可用）" if item["evidence_eligible"] else "（空间证据不可用）")
        for item in claim["spatial_support"]
    ]
    measurements = [
        f"{item['name_id']}={float(item['value']):.6g} {item['unit_id']}"
        + (
            ""
            if item["baseline_delta"] is None
            else f"，baseline_delta={float(item['baseline_delta']):.6g}"
        )
        for item in claim["measurements"]
    ]
    details = []
    if spatial:
        details.append("空间=" + "、".join(spatial))
    if measurements:
        details.append("测量=" + "；".join(measurements))
    detail_text = "；" + "；".join(details) if details else ""
    assertion = _ASSERTION_LABELS[str(claim["assertion_level"])]
    status = _STATUS_LABELS[str(claim["status"])]
    route_suffix = (
        "；但本影子路线不继承报告授权"
        if claim["assertion_level"] == "report_eligible_automated"
        else ""
    )
    return (
        f"第 {event_ordinal} 次事件于{when}，{status}{term}（{assertion}；"
        f"证据角色={claim['intrinsic_evidence_role']}）{detail_text}{route_suffix}。"
    )


def _mode_text(claim: Mapping[str, Any], ordinal: int) -> str:
    events = "、".join(str(item) for item in claim["event_ids"])
    return (
        f"模式 {ordinal}（{claim['mode_id']}；事件 {events}）的未校准概率样候选排序："
        f"{claim['candidate_unit_type']}={_ranking_text(claim['channel_probability_like_ranking'])}；"
        f"脑区={_ranking_text(claim['region_probability_like_ranking'])}；"
        f"侧别={_ranking_text(claim['laterality_probability_like_ranking'])}；"
        f"表型={_ranking_text(claim['phenotype_probability_like_ranking'])}。"
    )


def _waveform_panel_projections(
    graph: Mapping[str, Any], mode_claims: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    mode_by_event = {
        str(event_id): str(mode["claim_id"])
        for mode in mode_claims
        for event_id in mode["event_ids"]
    }
    rows = []
    for panel in graph["waveform_panels"]:
        event_id = str(panel["event_id"])
        if event_id not in mode_by_event:
            raise ValueError("waveform panel event lacks a mode claim")
        row = {
            "panel_id": str(panel["panel_id"]),
            "source_panel_sha256": str(panel["panel_sha256"]),
            "event_id": event_id,
            "source_graph_claim_ids": [str(item) for item in panel["claim_ids"]],
            "source_finding_evidence_ids": [
                str(item) for item in panel["finding_evidence_ids"]
            ],
            "source_claim_evidence_bindings": deepcopy(
                panel["claim_evidence_bindings"]
            ),
            "shadow_mode_claim_id": mode_by_event[event_id],
            "waveform_evidence_id": str(panel["waveform_evidence_id"]),
            "recording_interval": [float(item) for item in panel["interval"]],
            "unit_ids": [str(item) for item in panel["unit_ids"]],
            "view_role": str(panel["view_role"]),
            "raw_sample_dependency_id": panel["raw_sample_dependency_id"],
            "pixel_or_png_payload_present": False,
            "panel_semantics": (
                "claim_bound_waveform_metadata_index_not_a_rendered_signal_image"
            ),
            "projection_sha256": "",
        }
        _seal(row, "projection_sha256", "clinical-eeg-shadow-waveform-panel-v1")
        rows.append(row)
    return rows


def _render(
    graph: Mapping[str, Any],
    atomic_claims: Sequence[Mapping[str, Any]],
    mode_claims: Sequence[Mapping[str, Any]],
    record_claim: Mapping[str, Any],
    edges: Sequence[Mapping[str, Any]],
    waveform_panels: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], str, list[dict[str, Any]]]:
    sections: dict[str, list[str]] = {key: [] for key in _SECTION_ORDER}
    sentences: list[dict[str, Any]] = []
    known_claims = {
        str(item["claim_id"]): str(item["claim_sha256"])
        for item in [*atomic_claims, *mode_claims, record_claim]
    }
    known_edges = {str(item["edge_id"]) for item in edges}

    def add(
        section: str,
        template: str,
        text: str,
        *,
        claim_ids: Sequence[str] = (),
        edge_ids: Sequence[str] = (),
        panel_ids: Sequence[str] = (),
        signal_assertion: bool = False,
    ) -> None:
        if signal_assertion and len(claim_ids) != 1:
            raise ValueError(
                "each signal assertion sentence needs exactly one claim owner"
            )
        if not set(claim_ids).issubset(known_claims):
            raise ValueError("sentence references an unknown claim")
        if not set(edge_ids).issubset(known_edges):
            raise ValueError("sentence references an unknown provenance edge")
        sentence = {
            "sentence_id": _bounded_id(
                "SHADOWSENTENCE",
                {
                    "section": section,
                    "ordinal": len(sentences) + 1,
                    "template": template,
                    "claim_ids": list(claim_ids),
                    "text": text,
                },
            ),
            "section_id": section,
            "section_sentence_ordinal": len(sections[section]) + 1,
            "template_id": template,
            "text_zh": text,
            "text_sha256": _canonical_sha256(text),
            "locked_claim_ids": list(claim_ids),
            "locked_claim_sha256s": [known_claims[item] for item in claim_ids],
            "locked_provenance_edge_ids": list(edge_ids),
            "locked_waveform_panel_ids": list(panel_ids),
            "signal_assertion": signal_assertion,
            "sentence_binding_sha256": "",
        }
        _seal(
            sentence,
            "sentence_binding_sha256",
            "clinical-eeg-shadow-claim-locked-sentence-v1",
        )
        sentences.append(sentence)
        sections[section].append(text)

    add(
        "boundary",
        "research_shadow_boundary_v1",
        ("本页仅为公共/合成数据上的可重放研究影子，不是临床诊断报告；" "所有模式分数均未校准，不代表个体诊断概率、皮层 SOZ、致痫区或手术靶点。"),
    )
    event_order = [str(item["event_id"]) for item in graph["source_event_graphs"]]
    event_ordinals = {event_id: index for index, event_id in enumerate(event_order, 1)}
    for claim in atomic_claims:
        add(
            "event_findings",
            "atomic_event_finding_v1",
            _atomic_text(claim, event_ordinals[str(claim["event_id"])]),
            claim_ids=[str(claim["claim_id"])],
            signal_assertion=True,
        )
    add(
        "multievent_summary",
        "record_mode_summary_v1",
        (
            f"本记录纳入 {len(event_order)} 个唯一事件，形成 {len(mode_claims)} 个研究性模式；"
            + (
                "因存在多个模式，跨模式平均通道、脑区和侧别排序已强制留空。"
                if record_claim["multiple_mode_record_average_withheld"]
                else "记录级候选仅保留当前影子解码允许的分辨率。"
            )
        ),
        claim_ids=[str(record_claim["claim_id"])],
        signal_assertion=True,
    )
    for index, claim in enumerate(mode_claims, 1):
        add(
            "impression",
            "mode_probability_like_ranking_v1",
            _mode_text(claim, index),
            claim_ids=[str(claim["claim_id"])],
            signal_assertion=True,
        )
        mode_edge_ids = [
            str(item["edge_id"])
            for item in edges
            if item["target_claim_id"] == claim["claim_id"]
        ]
        chain = claim["evidence_chain"]
        add(
            "provenance",
            "mode_evidence_chain_v1",
            (
                f"{claim['mode_id']} 绑定 {len(chain['event_ids'])} 个事件、"
                f"{len(chain['hard_input_binding_sha256s'])} 个硬输入、"
                f"{len(chain['event_scoped_evidence_keys'])} 个 event-scoped evidence key、"
                f"{len(chain['raw_sample_dependency_ids'])} 个原始采样依赖和"
                f"{len(chain['constructive_spatial_receipt_ids'])} 个构造性空间回执。"
            ),
            edge_ids=mode_edge_ids,
        )
    for panel in waveform_panels:
        interval = panel["recording_interval"]
        add(
            "waveform_index",
            "claim_bound_waveform_metadata_v1",
            (
                f"面板 {panel['panel_id']}：事件 {panel['event_id']}，相对记录 "
                f"{_format_time(interval[0])}–{_format_time(interval[1])}，"
                f"单元 {'、'.join(panel['unit_ids'])}，视图 {panel['view_role']}；"
                "当前仅闭合波形元数据，未在本工件中生成 PNG。"
            ),
            panel_ids=[str(panel["panel_id"])],
        )
    add(
        "limitations",
        "research_shadow_limitations_v1",
        (
            "该闭环只证明 Findings→事件→模式→记录→文字/面板索引的序列化、数值复制和"
            "provenance 可重放；不证明上游测量临床正确、模型校准、患者级事实一致性或语言质量。"
        ),
    )

    rendered_sections = {key: "\n".join(sections[key]) for key in _SECTION_ORDER}
    report_text = "\n\n".join(
        f"【{_SECTION_LABELS[key]}】\n{rendered_sections[key]}"
        for key in _SECTION_ORDER
        if rendered_sections[key]
    )
    qwen_slots = [
        {
            "slot_id": _bounded_id(
                "QWENLEXICALSLOT",
                {"sentence_id": item["sentence_id"], "text": item["text_sha256"]},
            ),
            "sentence_id": item["sentence_id"],
            "locked_claim_ids": deepcopy(item["locked_claim_ids"]),
            "deterministic_text_sha256": item["text_sha256"],
            "allowed_action": "propose_lexical_surface_for_separate_audit_only",
            "structural_or_numeric_edit_authorized": False,
            "canonical_report_replacement_authorized": False,
            "proposal_text_zh": None,
        }
        for item in sentences
        if item["signal_assertion"]
    ]
    return sentences, rendered_sections, report_text, qwen_slots


def materialize_mode_aware_claim_locked_report_shadow_v1(
    bridge_shadow: object,
    *,
    report_graph_v2: object,
    trusted_source_event_findings_v3: Sequence[object],
    event_processing_ledger_v2: object,
    mil_bag: CompleteRecordModeAwareMILBagV1,
    mil_policy: ModeAwareMILPolicyV1,
    mil_forward: ModeAwareMILForwardV1,
    hard_input_bindings: Sequence[ModeAwareMILHardInputBindingV1],
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: (Mapping[str, Mapping[str, object]] | None) = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Rebuild a deterministic, non-authoritative Chinese report preview."""

    bridge = validate_mode_aware_mil_report_graph_v2_bridge_shadow_v1(
        bridge_shadow,
        report_graph_v2=report_graph_v2,
        trusted_source_event_findings_v3=trusted_source_event_findings_v3,
        event_processing_ledger_v2=event_processing_ledger_v2,
        mil_bag=mil_bag,
        mil_policy=mil_policy,
        mil_forward=mil_forward,
        hard_input_bindings=hard_input_bindings,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    if bridge["route_boundary"]["route_scope"] not in {"public", "synthetic"}:
        raise ValueError("claim-locked report shadow requires public/synthetic scope")
    if (
        bridge["authorized_claim_overlay"]
        or bridge["qwen_lexicalization_slots"]
        or bridge["renderer_projection"]
    ):
        raise ValueError("formal bridge surfaces must remain empty")
    if not isinstance(report_graph_v2, Mapping):
        raise TypeError("report_graph_v2 must be an object")
    graph = report_graph_v2

    atomic_claims = _atomic_claim_projections(graph, trusted_source_event_findings_v3)
    mode_claims = _mode_claims(bridge, graph, atomic_claims)
    record_claim = _record_claim(bridge, graph, mode_claims)
    edges = _provenance_edges(atomic_claims, mode_claims, record_claim)
    waveform_panels = _waveform_panel_projections(graph, mode_claims)
    sentences, sections, report_text, qwen_slots = _render(
        graph,
        atomic_claims,
        mode_claims,
        record_claim,
        edges,
        waveform_panels,
    )

    claim_owner_counts = Counter(
        claim_id
        for sentence in sentences
        if sentence["signal_assertion"]
        for claim_id in sentence["locked_claim_ids"]
    )
    all_claim_ids = {
        str(item["claim_id"]) for item in [*atomic_claims, *mode_claims, record_claim]
    }
    if claim_owner_counts != Counter({claim_id: 1 for claim_id in all_claim_ids}):
        raise ValueError("signal claim surface ownership is not exact-once")
    source_finding_count = sum(
        len(source["findings"])
        for source in trusted_source_event_findings_v3
        if isinstance(source, Mapping)
    )
    if source_finding_count != len(atomic_claims):
        raise ValueError("source Finding denominator was reduced during projection")

    family_counts = Counter(str(item["family"]) for item in atomic_claims)
    status_counts = Counter(str(item["status"]) for item in atomic_claims)
    assertion_counts = Counter(str(item["assertion_level"]) for item in atomic_claims)
    artifact: dict[str, Any] = {
        "schema_version": MODE_AWARE_CLAIM_LOCKED_REPORT_SHADOW_SCHEMA_VERSION,
        "artifact_id": _bounded_id(
            "CLAIMLOCKEDSHADOWREPORT",
            {
                "bridge_sha256": bridge["bridge_sha256"],
                "record_id": graph["record"]["record_id"],
                "renderer_id": MODE_AWARE_CLAIM_LOCKED_REPORT_SHADOW_RENDERER_ID,
            },
        ),
        "route_boundary": {
            "route_scope": bridge["route_boundary"]["route_scope"],
            "public_or_synthetic_shadow_only": True,
            "formal_report_route_connected": (
                MODE_AWARE_CLAIM_LOCKED_REPORT_SHADOW_ROUTE_CONNECTED
            ),
            "clinical_use_authorized": False,
            "private_data_used": False,
            "edf_annotations_used": False,
            "spreadsheet_used": False,
            "doctor_labels_used": False,
            "clinical_reports_used": False,
            "qwen_used_for_canonical_report": False,
            "qwen_may_replace_canonical_report": False,
            "source_bridge_formal_surfaces_remain_empty": True,
        },
        "source_binding": {
            "bridge_id": bridge["bridge_id"],
            "bridge_sha256": bridge["bridge_sha256"],
            "report_graph_id": graph["graph_id"],
            "report_graph_sha256": graph["graph_sha256"],
            "record_id": graph["record"]["record_id"],
            "canonical_signal_sha256": graph["record"]["canonical_signal_sha256"],
            "decode_sha256": bridge["source_binding"]["decode_sha256"],
            "onset_decision_sha256": bridge["source_binding"]["onset_decision_sha256"],
            "renderer_id": MODE_AWARE_CLAIM_LOCKED_REPORT_SHADOW_RENDERER_ID,
        },
        "atomic_finding_claims": atomic_claims,
        "mode_candidate_claims": mode_claims,
        "record_candidate_claim": record_claim,
        "event_to_mode_to_record_provenance_edges": edges,
        "waveform_panel_projections": waveform_panels,
        "sentence_ledger": sentences,
        "sections_zh": sections,
        "canonical_report_text_zh": report_text,
        "qwen_lexical_overlay_contract": {
            "slots": qwen_slots,
            "proposal_surface_is_noncanonical": True,
            "proposal_may_add_claim_entity_time_channel_number_or_epistemic_status": False,
            "canonical_output_is_always_deterministic": True,
        },
        "faithful_rendering_evaluation": {
            "source_finding_denominator": source_finding_count,
            "atomic_finding_claim_count": len(atomic_claims),
            "atomic_finding_claim_rendered_exactly_once_count": len(atomic_claims),
            "mode_claim_count": len(mode_claims),
            "mode_claim_rendered_exactly_once_count": len(mode_claims),
            "record_claim_rendered_exactly_once": True,
            "signal_assertion_claim_ownership_exact_once": True,
            "numeric_values_copied_from_frozen_claim_projection": True,
            "physical_times_copied_from_frozen_claim_projection": True,
            "channels_and_spatial_eligibility_copied_from_frozen_claim_projection": True,
            "status_negation_and_uncertainty_copied_from_frozen_claim_projection": True,
            "event_to_mode_to_record_provenance_explicit": True,
            "waveform_panels_claim_and_evidence_bound": True,
            "clinical_correctness_or_calibration_claimed": False,
            "claim_surface_precision_semantics": (
                "self_replay_serialization_only_not_independent_clinical_factuality"
            ),
        },
        "facet_coverage_receipt": {
            "finding_family_counts": dict(sorted(family_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "assertion_level_counts": dict(sorted(assertion_counts.items())),
            "claims_with_numeric_measurements": sum(
                bool(item["measurements"]) for item in atomic_claims
            ),
            "claims_with_physical_time": sum(
                item["time_interval"] is not None for item in atomic_claims
            ),
            "claims_with_spatial_support": sum(
                bool(item["spatial_support"]) for item in atomic_claims
            ),
            "claims_with_waveform_evidence": sum(
                bool(item["waveform_evidence_ids"]) for item in atomic_claims
            ),
            "morphology_and_evolution_coverage_semantics": (
                "all_frozen_term_families_are_projected_without_inventing_missing_heads"
            ),
        },
        "artifact_sha256": "",
    }
    _seal(
        artifact,
        "artifact_sha256",
        "clinical-eeg-mode-aware-claim-locked-report-shadow-v1",
    )
    return artifact


def validate_mode_aware_claim_locked_report_shadow_v1(
    payload: object,
    **replay_inputs: Any,
) -> dict[str, Any]:
    """Rebuild the full preview; coordinated payload resealing cannot bypass it."""

    if not isinstance(payload, Mapping):
        raise TypeError("claim-locked report shadow must be an object")
    rebuilt = materialize_mode_aware_claim_locked_report_shadow_v1(**replay_inputs)
    if payload != rebuilt:
        raise ValueError(
            "claim-locked report shadow does not replay from frozen public/synthetic inputs"
        )
    return rebuilt


__all__ = [
    "MODE_AWARE_CLAIM_LOCKED_REPORT_SHADOW_RENDERER_ID",
    "MODE_AWARE_CLAIM_LOCKED_REPORT_SHADOW_ROUTE_CONNECTED",
    "MODE_AWARE_CLAIM_LOCKED_REPORT_SHADOW_SCHEMA_VERSION",
    "materialize_mode_aware_claim_locked_report_shadow_v1",
    "validate_mode_aware_claim_locked_report_shadow_v1",
]
