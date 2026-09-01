"""Replayable report-graph-v2 to atomic-claim factuality bridge.

The legacy source-bound factuality materializer intentionally consumes the
frozen four-role report format.  This independent adapter consumes only the
lossless Findings-v3 report graph and preserves its five permission roles
verbatim.  It never translates onset time/topography to ``onset_support`` and
never promotes course/spread evidence into an onset role.

Every portable evidence identifier is scoped to one permission edge and one
source finding.  This is necessary because one v3 finding may legitimately
participate in several named permission edges while the portable claim case
requires one role per evidence-temporal binding.  Validation replays the
entire report graph from its independently validated embedded v3 sources and
then rematerializes this artifact byte-for-byte.

This module is EEG-only and performs no file I/O.  It does not read EDF
annotations, spreadsheets, physician labels, clinical text or patient
metadata.  The bridge is an evaluation/audit sidecar, not a clinical-use
authorization or a text renderer.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .claim_factuality_evaluation import (
    CLAIM_FACTUALITY_CASE_SCHEMA_VERSION,
    evaluate_claim_factuality_case,
    validate_claim_factuality_case,
)
from .multievent_soz_report_graph_v2 import (
    validate_multievent_soz_report_graph_v2,
)
from .eeg_only_event_outcome_semantics import normalize_eeg_only_event_outcome


REPORT_GRAPH_V2_CLAIM_FACTUALITY_BRIDGE_SCHEMA_VERSION = (
    "eeg_report_graph_v2_claim_factuality_bridge_v1"
)
REPORT_GRAPH_V2_CLAIM_FACTUALITY_BRIDGE_ID = (
    "lossless_five_role_report_graph_v2_claim_factuality_bridge_v1"
)

_ROLE_ORDER = (
    "ictal_pattern_qualification",
    "onset_time_support",
    "onset_topography_support",
    "course_or_spread_support",
    "counterevidence",
)
_ROLE_SET = set(_ROLE_ORDER)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_CLAIM_BOUNDARY = {
    "eeg_signal_claims_only": True,
    "private_data_loaded_by_evaluator": False,
    "excel_loaded_by_evaluator": False,
    "edf_annotations_loaded_by_evaluator": False,
    "doctor_labels_loaded_by_evaluator": False,
    "clinical_text_loaded_by_evaluator": False,
    "source_eval_loaded_by_evaluator": False,
}
_TRANSLATION_POLICY_BODY: Mapping[str, Any] = {
    "policy_id": "report_graph_v2_five_role_claim_projection_v1",
    "source_permission_vocabulary": list(_ROLE_ORDER),
    "permission_roles_preserved_verbatim": True,
    "permission_role_remapping_allowed": False,
    "edge_scoped_case_evidence_ids": True,
    "course_or_spread_may_support_positive_onset": False,
    "counterevidence_may_support_positive_onset": False,
    "ictal_qualification_alone_may_support_positive_onset": False,
    "localized_claim_requires_onset_time_and_topography": True,
    "report_eligible_automated_projection": (
        "model_candidate_without_clinical_qualification_uplift"
    ),
    "unauthorized_mode_or_record_conclusion_projection": "not_evaluable",
    "predicted_claim_set_semantics": (
        "closed_structured_graph_projection_before_text_lexicalization"
    ),
}


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


def _domain_sha256(domain: str, value: object) -> str:
    return _canonical_sha256({"binding_domain": domain, "value": value})


def _bounded_id(prefix: str, value: object) -> str:
    return f"{prefix}:{_domain_sha256(prefix, value)[:32]}"


def _seal(value: dict[str, Any], field: str, domain: str) -> None:
    value[field] = "0" * 64
    value[field] = _domain_sha256(domain, value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be an opaque identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _strict_object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    actual = set(value)
    missing = keys.difference(actual)
    extra = actual.difference(keys)
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _translation_policy() -> dict[str, Any]:
    result = deepcopy(dict(_TRANSLATION_POLICY_BODY))
    result["policy_sha256"] = _domain_sha256(
        "clinical-eeg-report-graph-v2-claim-translation-policy-v1",
        result,
    )
    return result


def _trusted_kwargs(
    *,
    trusted_source_event_findings_v3: Mapping[str, object] | Sequence[object] | None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ),
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    return {
        "trusted_source_event_findings_v3": trusted_source_event_findings_v3,
        "trusted_producer_receipts": trusted_producer_receipts,
        "trusted_calibration_receipts": trusted_calibration_receipts,
        "trusted_capability_qualification_receipts": (
            trusted_capability_qualification_receipts
        ),
        "trusted_sensitivity_receipts": trusted_sensitivity_receipts,
        "trusted_term_decision_receipts": trusted_term_decision_receipts,
        "trusted_registry_bindings": trusted_registry_bindings,
    }


def _source_indexes(
    graph: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    findings: dict[tuple[str, str], Mapping[str, Any]] = {}
    for wrapper in graph["source_event_graphs"]:
        source = wrapper["event_findings_v3"]
        event_id = str(source["event_id"])
        for finding in source["findings"]:
            key = (event_id, str(finding["evidence_id"]))
            if key in findings:
                raise ValueError("report graph repeats an event-scoped finding ID")
            findings[key] = finding
    nodes: dict[tuple[str, str], Mapping[str, Any]] = {}
    for node in graph["finding_evidence_nodes"]:
        key = (str(node["event_id"]), str(node["evidence_id"]))
        if key in nodes:
            raise ValueError("report graph repeats an event-scoped finding node")
        nodes[key] = node
    if set(findings) != set(nodes):
        raise ValueError("report graph finding nodes do not close embedded findings")
    return findings, nodes


def _aggregate_dependency_binding(
    *,
    role: str,
    case_evidence_id: str,
    node: Mapping[str, Any],
) -> dict[str, Any]:
    dependencies = list(node["raw_sample_dependencies"])
    view_roles = {str(item["view_role"]) for item in dependencies}
    view_role = next(iter(view_roles)) if len(view_roles) == 1 else "unknown"
    return {
        "evidence_id": case_evidence_id,
        "evidence_role": role,
        "intrinsic_evidence_role": str(node["intrinsic_evidence_role"]),
        "view_role": view_role,
        "future_sample_access": (
            any(bool(item["future_sample_access"]) for item in dependencies)
            if dependencies
            else False
        ),
        "onset_evidence_authorized": bool(dependencies)
        and all(bool(item["onset_evidence_authorized"]) for item in dependencies),
        "onset_support_eligible": bool(dependencies)
        and all(bool(item["onset_support_eligible"]) for item in dependencies),
    }


def _permission_projection(
    graph: Mapping[str, Any],
    nodes: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    role_roster = deepcopy(graph["evidence_permission_role_roster"])
    if [str(item["role"]) for item in role_roster] != list(_ROLE_ORDER):
        raise ValueError("report graph five-role roster order is not closed")
    edge_ids_by_role: dict[str, list[str]] = defaultdict(list)
    edge_bindings: list[dict[str, Any]] = []
    binding_by_case_id: dict[str, Mapping[str, Any]] = {}
    edge_by_id: dict[str, Mapping[str, Any]] = {}
    for edge in graph["evidence_permission_edges"]:
        edge_id = str(edge["permission_edge_id"])
        role = str(edge["role"])
        event_id = str(edge["event_id"])
        if role not in _ROLE_SET or edge_id in edge_by_id:
            raise ValueError("report graph permission edge roster is invalid")
        edge_by_id[edge_id] = edge
        edge_ids_by_role[role].append(edge_id)
        for evidence_id_value in edge["evidence_ids"]:
            evidence_id = str(evidence_id_value)
            node = nodes.get((event_id, evidence_id))
            if node is None:
                raise ValueError("permission edge references an unknown event finding")
            case_evidence_id = _bounded_id(
                "CASEEVIDENCE",
                {
                    "permission_edge_id": edge_id,
                    "event_id": event_id,
                    "source_evidence_id": evidence_id,
                },
            )
            temporal = _aggregate_dependency_binding(
                role=role,
                case_evidence_id=case_evidence_id,
                node=node,
            )
            row = {
                "case_evidence_id": case_evidence_id,
                "source_permission_edge_id": edge_id,
                "source_permission_edge_sha256": _domain_sha256(
                    "clinical-eeg-report-graph-v2-permission-edge-source-v1",
                    edge,
                ),
                "event_id": event_id,
                "source_evidence_id": evidence_id,
                "source_evidence_sha256": str(node["source_finding_sha256"]),
                "role": role,
                "derivation_rule_id": str(edge["derivation_rule_id"]),
                "raw_sample_dependency_ids": sorted(
                    str(item["dependency_id"])
                    for item in node["raw_sample_dependencies"]
                ),
                "constructive_spatial_receipt_id": edge[
                    "constructive_spatial_receipt_id"
                ],
                "temporal_binding": temporal,
                "binding_sha256": "",
            }
            _seal(
                row,
                "binding_sha256",
                "clinical-eeg-report-graph-v2-edge-scoped-case-evidence-v1",
            )
            if case_evidence_id in binding_by_case_id:
                raise ValueError("edge-scoped case evidence ID collision")
            binding_by_case_id[case_evidence_id] = row
            edge_bindings.append(row)
    for roster_row in role_roster:
        role = str(roster_row["role"])
        expected_ids = edge_ids_by_role[role]
        if list(roster_row["edge_ids"]) != expected_ids:
            raise ValueError("permission role roster does not equal source edges")
        expected_status = "materialized" if expected_ids else "not_expressed_by_source"
        if roster_row["status"] != expected_status:
            raise ValueError("permission role status does not equal source expression")

    claim_bindings: list[dict[str, Any]] = []
    edge_binding_ids: dict[str, list[str]] = defaultdict(list)
    for row in edge_bindings:
        edge_binding_ids[str(row["source_permission_edge_id"])].append(
            str(row["case_evidence_id"])
        )
    for claim in graph["claims"]:
        edge_ids = [str(item) for item in claim["permission_edge_ids"]]
        roles = sorted({str(edge_by_id[item]["role"]) for item in edge_ids})
        if roles != sorted(str(item) for item in claim["required_permission_roles"]):
            raise ValueError("claim permission roles differ from referenced edges")
        case_evidence_ids = sorted(
            case_id for edge_id in edge_ids for case_id in edge_binding_ids[edge_id]
        )
        row = {
            "claim_id": str(claim["claim_id"]),
            "source_claim_sha256": str(claim["claim_sha256"]),
            "source_permission_edge_ids": edge_ids,
            "source_permission_roles": roles,
            "case_evidence_ids": case_evidence_ids,
            "case_evidence_roles": roles,
            "permission_roles_preserved_verbatim": True,
            "claim_binding_sha256": "",
        }
        _seal(
            row,
            "claim_binding_sha256",
            "clinical-eeg-report-graph-v2-claim-permission-binding-v1",
        )
        claim_bindings.append(row)

    role_roster_sha = _domain_sha256(
        "clinical-eeg-report-graph-v2-five-role-roster-v1", role_roster
    )
    permission_edges_sha = _domain_sha256(
        "clinical-eeg-report-graph-v2-permission-edge-roster-v1",
        graph["evidence_permission_edges"],
    )
    edge_bindings_sha = _domain_sha256(
        "clinical-eeg-report-graph-v2-edge-scoped-case-evidence-roster-v1",
        edge_bindings,
    )
    claim_bindings_sha = _domain_sha256(
        "clinical-eeg-report-graph-v2-claim-permission-binding-roster-v1",
        claim_bindings,
    )
    combined = _domain_sha256(
        "clinical-eeg-report-graph-v2-combined-permission-binding-v1",
        {
            "graph_id": graph["graph_id"],
            "role_roster_sha256": role_roster_sha,
            "permission_edges_sha256": permission_edges_sha,
            "edge_bindings_sha256": edge_bindings_sha,
            "claim_permission_bindings_sha256": claim_bindings_sha,
        },
    )
    return (
        {
            "role_roster": role_roster,
            "edge_bindings": edge_bindings,
            "claim_permission_bindings": claim_bindings,
            "role_roster_sha256": role_roster_sha,
            "permission_edges_sha256": permission_edges_sha,
            "edge_bindings_sha256": edge_bindings_sha,
            "claim_permission_bindings_sha256": claim_bindings_sha,
            "combined_permission_binding_sha256": combined,
        },
        binding_by_case_id,
    )


def _entity_type(source_type: str) -> str:
    return {
        "lead": "bipolar_derivation",
        "electrode": "electrode",
        "region": "region",
        "laterality": "laterality",
    }[source_type]


def _deduplicated_entities(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["type"]), str(row["id"]))
        if key in seen:
            continue
        seen.add(key)
        result.append({"type": key[0], "id": key[1]})
    return result


def _none_time() -> dict[str, Any]:
    return {
        "kind": "none",
        "timebase": "not_applicable",
        "lower": None,
        "upper": None,
        "left_censored": False,
        "right_censored": False,
    }


def _finding_time(finding: Mapping[str, Any]) -> dict[str, Any]:
    interval = finding["time_interval"]
    if interval is None:
        return _none_time()
    return {
        "kind": "recording_interval",
        "timebase": "recording_relative_seconds",
        "lower": float(interval["start"]),
        "upper": float(interval["stop"]),
        "left_censored": False,
        "right_censored": False,
    }


def _claim_time(
    claim: Mapping[str, Any],
    findings: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    if claim["claim_kind"] == "finding_state" and claim["event_id"] is not None:
        evidence_ids = list(claim["source_evidence_ids"])
        if len(evidence_ids) == 1:
            return _finding_time(
                findings[(str(claim["event_id"]), str(evidence_ids[0]))]
            )
    if claim["claim_kind"] != "scalp_onset_spatial_candidate":
        return _none_time()
    intervals = [
        findings[(str(claim["event_id"]), str(evidence_id))]["time_interval"]
        for evidence_id in claim["source_evidence_ids"]
    ]
    intervals = [item for item in intervals if item is not None]
    if not intervals:
        return _none_time()
    return {
        "kind": "recording_interval",
        "timebase": "recording_relative_seconds",
        "lower": min(float(item["start"]) for item in intervals),
        "upper": max(float(item["stop"]) for item in intervals),
        "left_censored": False,
        "right_censored": False,
    }


def _claim_semantics(
    claim: Mapping[str, Any],
    findings: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str, list[dict[str, str]], list[dict[str, Any]], str | None]:
    kind = str(claim["claim_kind"])
    event_id = claim["event_id"]
    if kind == "finding_state":
        finding = findings[(str(event_id), str(claim["source_evidence_ids"][0]))]
        entities = _deduplicated_entities(
            [
                {
                    "type": _entity_type(str(item["unit_type"])),
                    "id": str(item["id"]),
                }
                for item in finding["spatial_support"]
                if bool(item["evidence_eligible"])
                and item["observation_status"] in {"observed", "derived"}
            ]
        )
        measurements = [
            {
                "name": str(item["measurement_id"]),
                "value": float(item["value"]),
                "unit": str(item["unit_id"]),
            }
            for item in finding["measurements"]
        ]
        return (
            str(finding["term"]["term_id"]),
            entities,
            measurements,
            str(finding["term"]["term_id"]),
        )
    if kind == "event_outcome":
        # The typed outcome is recovered from the embedded source by the
        # caller; render disposition is deliberately not used as physiology.
        return "event_outcome", [], [], None
    if kind == "competing_hypothesis":
        return "competing_signal_hypothesis", [], [], None
    if kind == "scalp_onset_spatial_candidate":
        return (
            "event_supports_soz_candidate",
            [
                {
                    "type": _entity_type(str(claim["target_resolution"])),
                    "id": str(claim["target_entity_id"]),
                }
            ],
            [],
            None,
        )
    if kind == "event_scalp_onset_hypothesis":
        return "event_scalp_onset_hypothesis_status", [], [], None
    if kind == "mode_conclusion_status":
        return "mode_conclusion_authorization_status", [], [], None
    if kind == "record_impression_status":
        return "record_impression_authorization_status", [], [], None
    raise ValueError(f"unsupported report graph claim kind: {kind}")


def _source_claim_lookup(
    graph: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    event_outcomes: dict[str, str] = {}
    competing_terms: dict[str, str] = {}
    for wrapper in graph["source_event_graphs"]:
        source = wrapper["event_findings_v3"]
        event_id = str(source["event_id"])
        event_outcomes[event_id] = normalize_eeg_only_event_outcome(
            str(source["event_outcome"]["outcome"])
        )
        for hypothesis in source["competing_hypotheses"]["hypotheses"]:
            competing_terms[str(hypothesis["hypothesis_id"])] = str(
                hypothesis["term_id"]
            )
    event_mode: dict[str, str] = {}
    node_by_id = {
        str(item["node_id"]): item for item in graph["derivation_dag"]["nodes"]
    }
    for edge in graph["derivation_dag"]["edges"]:
        if edge["relation"] != "event_to_mode":
            continue
        source_node = node_by_id[str(edge["source_node_id"])]
        event_mode[str(source_node["source_ids"][0])] = str(edge["target_node_id"])
    return event_outcomes, competing_terms, event_mode


def _portable_claim(
    *,
    graph_claim: Mapping[str, Any],
    claim_binding: Mapping[str, Any],
    binding_by_case_id: Mapping[str, Mapping[str, Any]],
    findings: Mapping[tuple[str, str], Mapping[str, Any]],
    event_outcomes: Mapping[str, str],
    competing_terms: Mapping[str, str],
    event_mode: Mapping[str, str],
) -> dict[str, Any]:
    predicate, entities, measurements, code = _claim_semantics(graph_claim, findings)
    kind = str(graph_claim["claim_kind"])
    event_id = graph_claim["event_id"]
    if kind == "event_outcome":
        code = event_outcomes[str(event_id)]
    elif kind == "competing_hypothesis":
        hypothesis_ref = next(
            item
            for item in graph_claim["source_refs"]
            if item["object_kind"] == "competing_hypothesis"
        )
        code = competing_terms[str(hypothesis_ref["object_id"])]

    layer = str(graph_claim["layer"])
    if kind == "finding_state":
        subject = {
            "type": "finding",
            "id": str(graph_claim["source_evidence_ids"][0]),
        }
        claim_kind = "observation"
    elif layer == "event":
        subject = {"type": "eeg_event", "id": str(event_id)}
        claim_kind = "event_inference"
    elif layer == "mode":
        subject = {"type": "mode", "id": str(graph_claim["mode_id"])}
        claim_kind = "mode_inference"
    else:
        subject = {"type": "eeg_record", "id": str(graph_claim["record_id"])}
        claim_kind = "record_hypothesis"

    status = graph_claim["finding_status"]
    if status is None:
        status = (
            "not_evaluable"
            if graph_claim["conclusion_authorization"]
            == "not_authorized_missing_mode_aware_mil_receipt"
            else "present"
        )
    assertion_status = str(status)
    assertion_level = graph_claim["assertion_level"]
    if assertion_status == "not_evaluable":
        epistemic = "not_evaluable"
    elif kind in {
        "scalp_onset_spatial_candidate",
        "event_scalp_onset_hypothesis",
        "competing_hypothesis",
    }:
        epistemic = "research_ai_hypothesis"
    elif assertion_level == "measured":
        epistemic = "measured"
    else:
        # ``report_eligible_automated`` is intentionally not upgraded to the
        # evaluator's clinically-qualified state.
        epistemic = "model_candidate"

    case_evidence_ids = list(claim_binding["case_evidence_ids"])
    temporal_bindings = [
        deepcopy(binding_by_case_id[item]["temporal_binding"])
        for item in case_evidence_ids
    ]
    roles = sorted({str(item["evidence_role"]) for item in temporal_bindings})
    if roles != list(claim_binding["case_evidence_roles"]):
        raise ValueError("portable claim roles do not replay source claim roles")

    surface = str(graph_claim["render_disposition"])
    salient = surface in {
        "positive_surface_allowed",
        "explicit_absence_surface_allowed",
        "uncertainty_surface_only",
        "research_candidate_surface_allowed",
    }
    critical = kind == "scalp_onset_spatial_candidate"
    severity = 3.0 if critical else 1.0
    if assertion_status == "absent_with_opportunity":
        polarity = "negated"
        negation_scope = "predicate"
    else:
        polarity = "affirmed"
        negation_scope = "none"
    mode_id = graph_claim["mode_id"]
    if layer == "event" and event_id is not None:
        mode_id = event_mode.get(str(event_id))
    return {
        "claim_id": str(graph_claim["claim_id"]),
        "claim_kind": claim_kind,
        "subject": subject,
        "predicate": predicate,
        "object_or_value": {
            "entities": entities,
            "measurements": measurements,
            "code": code,
        },
        "event_id": event_id,
        "mode_id": mode_id,
        "time": _claim_time(graph_claim, findings),
        "polarity": polarity,
        "negation_scope": negation_scope,
        "assertion_status": assertion_status,
        "epistemic_status": epistemic,
        "evidence_ids": case_evidence_ids,
        "evidence_roles": roles,
        "evidence_temporal_bindings": temporal_bindings,
        "relation_endpoints": None,
        "salient": salient,
        "salience_weight": severity if salient else 0.0,
        "severity_weight": severity,
        "critical": critical,
    }


def _derivations(
    graph: Mapping[str, Any], portable_by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    canonical_claim_by_node: dict[str, str] = {}
    desired_kind = {
        "event": "event_scalp_onset_hypothesis",
        "mode": "mode_conclusion_status",
        "record": "record_impression_status",
    }
    node_type = {
        str(item["node_id"]): str(item["node_type"])
        for item in graph["derivation_dag"]["nodes"]
    }
    for claim in graph["claims"]:
        dag_node_id = claim["dag_node_id"]
        if dag_node_id is None:
            continue
        expected = desired_kind[node_type[str(dag_node_id)]]
        if claim["claim_kind"] == expected:
            canonical_claim_by_node[str(dag_node_id)] = str(claim["claim_id"])
    if set(canonical_claim_by_node) != set(node_type):
        raise ValueError("report graph DAG lacks canonical portable claims")
    premises_by_target: dict[str, list[str]] = defaultdict(list)
    relation_by_target: dict[str, str] = {}
    for edge in graph["derivation_dag"]["edges"]:
        target = str(edge["target_node_id"])
        premises_by_target[target].append(
            canonical_claim_by_node[str(edge["source_node_id"])]
        )
        relation_by_target[target] = str(edge["relation"])
    result: list[dict[str, Any]] = []
    for target in graph["derivation_dag"]["topological_order"]:
        target_id = str(target)
        if target_id not in premises_by_target:
            continue
        conclusion = canonical_claim_by_node[target_id]
        premises = premises_by_target[target_id]
        if conclusion not in portable_by_id or not set(premises).issubset(
            portable_by_id
        ):
            raise ValueError("portable derivation does not close canonical claims")
        relation = relation_by_target[target_id]
        rule = {
            "event_to_mode": "event_hypothesis_to_mode_hypothesis_v1",
            "mode_to_record": "mode_hypothesis_to_record_hypothesis_v1",
        }[relation]
        result.append(
            {
                "derivation_id": _bounded_id(
                    "CASE-DERIVATION",
                    {
                        "target_node_id": target_id,
                        "conclusion_claim_id": conclusion,
                        "premise_claim_ids": premises,
                    },
                ),
                "conclusion_claim_id": conclusion,
                "premise_claim_ids": premises,
                "rule_id": rule,
                "weight": 1.0,
            }
        )
    return result


def _materialize_case(
    *,
    graph: Mapping[str, Any],
    permission_projection: Mapping[str, Any],
    binding_by_case_id: Mapping[str, Mapping[str, Any]],
    findings: Mapping[tuple[str, str], Mapping[str, Any]],
    source_bundle_sha256: str,
) -> dict[str, Any]:
    event_outcomes, competing_terms, event_mode = _source_claim_lookup(graph)
    claim_binding_by_id = {
        str(item["claim_id"]): item
        for item in permission_projection["claim_permission_bindings"]
    }
    portable_claims = [
        _portable_claim(
            graph_claim=claim,
            claim_binding=claim_binding_by_id[str(claim["claim_id"])],
            binding_by_case_id=binding_by_case_id,
            findings=findings,
            event_outcomes=event_outcomes,
            competing_terms=competing_terms,
            event_mode=event_mode,
        )
        for claim in graph["claims"]
    ]
    portable_by_id = {str(item["claim_id"]): item for item in portable_claims}
    cited_ids = {
        str(evidence_id)
        for claim in portable_claims
        for evidence_id in claim["evidence_ids"]
    }
    all_ids = set(binding_by_case_id)
    if cited_ids != all_ids:
        raise ValueError("portable claims do not close every permission-edge evidence")
    evidence_flow = [
        {
            "evidence_id": evidence_id,
            "weight": 1.0,
            "detector_recovered": True,
            "adaptive_window_retained": True,
            "finding_emitted": True,
            "record_claim_retained": True,
            "rendered_claim_retained": True,
        }
        for evidence_id in sorted(all_ids)
    ]
    canonical_signal = str(graph["record"]["canonical_signal_sha256"])
    return validate_claim_factuality_case(
        {
            "schema_version": CLAIM_FACTUALITY_CASE_SCHEMA_VERSION,
            "case_id": f"CASE:{source_bundle_sha256[:32]}",
            "patient_id": f"EEGSIGNAL:{canonical_signal[:32]}",
            "record_id": str(graph["record"]["record_id"]),
            "predicted_claims": deepcopy(portable_claims),
            "reference_claims": deepcopy(portable_claims),
            "derivations": _derivations(graph, portable_by_id),
            "evidence_flow": evidence_flow,
            "claim_boundary": dict(_CLAIM_BOUNDARY),
        }
    )


def _source_bindings(
    graph: Mapping[str, Any], permission_projection: Mapping[str, Any]
) -> dict[str, Any]:
    base = {
        "report_graph_id": str(graph["graph_id"]),
        "report_graph_self_sha256": str(graph["graph_sha256"]),
        "canonical_report_graph_sha256": _domain_sha256(
            "clinical-eeg-report-graph-v2-canonical-source-v1", graph
        ),
        "source_event_roster_sha256": str(graph["source_event_roster_sha256"]),
        "canonical_signal_sha256": str(graph["record"]["canonical_signal_sha256"]),
        "permission_role_roster_sha256": str(
            permission_projection["role_roster_sha256"]
        ),
        "permission_edges_sha256": str(
            permission_projection["permission_edges_sha256"]
        ),
        "derivation_dag_self_sha256": str(graph["derivation_dag"]["dag_sha256"]),
        "derivation_dag_sha256": _domain_sha256(
            "clinical-eeg-report-graph-v2-derivation-dag-source-v1",
            graph["derivation_dag"],
        ),
        "claims_sha256": _domain_sha256(
            "clinical-eeg-report-graph-v2-claim-roster-source-v1", graph["claims"]
        ),
        "combined_permission_binding_sha256": str(
            permission_projection["combined_permission_binding_sha256"]
        ),
        "source_event_graph_sha256s": [
            {
                "event_id": str(item["event_id"]),
                "sha256": str(item["source_event_findings_v3_sha256"]),
            }
            for item in graph["source_event_graphs"]
        ],
    }
    base["source_bundle_sha256"] = _domain_sha256(
        "clinical-eeg-report-graph-v2-factuality-source-bundle-v1", base
    )
    return base


def _artifact_shape(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        {
            "schema_version",
            "bridge_id",
            "translation_policy",
            "source_bindings",
            "permission_projection",
            "case",
            "case_sha256",
            "bridge_sha256",
        },
        "report-graph-v2 factuality bridge",
    )
    if data["schema_version"] != REPORT_GRAPH_V2_CLAIM_FACTUALITY_BRIDGE_SCHEMA_VERSION:
        raise ValueError("report-graph-v2 factuality bridge schema_version mismatch")
    if data["bridge_id"] != REPORT_GRAPH_V2_CLAIM_FACTUALITY_BRIDGE_ID:
        raise ValueError("report-graph-v2 factuality bridge_id mismatch")
    if data["translation_policy"] != _translation_policy():
        raise ValueError("report-graph-v2 factuality translation policy drifted")

    source = _strict_object(
        data["source_bindings"],
        {
            "report_graph_id",
            "report_graph_self_sha256",
            "canonical_report_graph_sha256",
            "source_event_roster_sha256",
            "canonical_signal_sha256",
            "permission_role_roster_sha256",
            "permission_edges_sha256",
            "derivation_dag_self_sha256",
            "derivation_dag_sha256",
            "claims_sha256",
            "combined_permission_binding_sha256",
            "source_event_graph_sha256s",
            "source_bundle_sha256",
        },
        "report-graph-v2 factuality source_bindings",
    )
    source["report_graph_id"] = _identifier(
        source["report_graph_id"], "source_bindings.report_graph_id"
    )
    for key in (
        "report_graph_self_sha256",
        "canonical_report_graph_sha256",
        "source_event_roster_sha256",
        "canonical_signal_sha256",
        "permission_role_roster_sha256",
        "permission_edges_sha256",
        "derivation_dag_self_sha256",
        "derivation_dag_sha256",
        "claims_sha256",
        "combined_permission_binding_sha256",
        "source_bundle_sha256",
    ):
        source[key] = _sha256(source[key], f"source_bindings.{key}")
    if not isinstance(source["source_event_graph_sha256s"], list):
        raise TypeError("source event graph hashes must be a list")
    seen_events: set[str] = set()
    for index, item in enumerate(source["source_event_graph_sha256s"]):
        row = _strict_object(
            item, {"event_id", "sha256"}, f"source_event_graph_sha256s[{index}]"
        )
        event_id = _identifier(row["event_id"], f"event hash {index}.event_id")
        _sha256(row["sha256"], f"event hash {index}.sha256")
        if event_id in seen_events:
            raise ValueError("source event hash roster contains duplicates")
        seen_events.add(event_id)
    data["source_bindings"] = source

    projection = _strict_object(
        data["permission_projection"],
        {
            "role_roster",
            "edge_bindings",
            "claim_permission_bindings",
            "role_roster_sha256",
            "permission_edges_sha256",
            "edge_bindings_sha256",
            "claim_permission_bindings_sha256",
            "combined_permission_binding_sha256",
        },
        "report-graph-v2 permission_projection",
    )
    if not isinstance(projection["role_roster"], list) or [
        str(item.get("role")) if isinstance(item, Mapping) else ""
        for item in projection["role_roster"]
    ] != list(_ROLE_ORDER):
        raise ValueError("permission projection does not preserve the five-role roster")
    for name in ("edge_bindings", "claim_permission_bindings"):
        if not isinstance(projection[name], list):
            raise TypeError(f"permission_projection.{name} must be a list")
    expected_hashes = {
        "role_roster_sha256": _domain_sha256(
            "clinical-eeg-report-graph-v2-five-role-roster-v1",
            projection["role_roster"],
        ),
        "edge_bindings_sha256": _domain_sha256(
            "clinical-eeg-report-graph-v2-edge-scoped-case-evidence-roster-v1",
            projection["edge_bindings"],
        ),
        "claim_permission_bindings_sha256": _domain_sha256(
            "clinical-eeg-report-graph-v2-claim-permission-binding-roster-v1",
            projection["claim_permission_bindings"],
        ),
    }
    for key, expected in expected_hashes.items():
        if projection[key] != expected:
            raise ValueError(f"permission projection {key} drifted")
    if projection["permission_edges_sha256"] != source["permission_edges_sha256"]:
        raise ValueError("permission edge hash differs across source bindings")
    combined_expected = _domain_sha256(
        "clinical-eeg-report-graph-v2-combined-permission-binding-v1",
        {
            "graph_id": source["report_graph_id"],
            "role_roster_sha256": projection["role_roster_sha256"],
            "permission_edges_sha256": projection["permission_edges_sha256"],
            "edge_bindings_sha256": projection["edge_bindings_sha256"],
            "claim_permission_bindings_sha256": projection[
                "claim_permission_bindings_sha256"
            ],
        },
    )
    if projection["combined_permission_binding_sha256"] != combined_expected:
        raise ValueError("combined permission binding hash drifted")
    if source["permission_role_roster_sha256"] != projection["role_roster_sha256"]:
        raise ValueError("role roster hash differs across source bindings")
    if source["combined_permission_binding_sha256"] != combined_expected:
        raise ValueError("combined permission hash differs across source bindings")
    data["permission_projection"] = projection

    case = validate_claim_factuality_case(data["case"])
    if case["predicted_claims"] != case["reference_claims"]:
        raise ValueError("bridge case must be a closed structured graph projection")
    case_roles = {
        str(role)
        for claim in case["predicted_claims"]
        for role in claim["evidence_roles"]
    }
    if not case_roles.issubset(_ROLE_SET):
        raise ValueError("bridge case contains a downgraded or foreign evidence role")
    edge_case_ids = {
        str(item["case_evidence_id"]) for item in projection["edge_bindings"]
    }
    flow_ids = {str(item["evidence_id"]) for item in case["evidence_flow"]}
    if edge_case_ids != flow_ids:
        raise ValueError("case evidence flow does not close edge-scoped evidence")
    data["case"] = case
    data["case_sha256"] = _sha256(data["case_sha256"], "case_sha256")
    if data["case_sha256"] != _domain_sha256(
        "clinical-eeg-report-graph-v2-portable-factuality-case-v1", case
    ):
        raise ValueError("portable factuality case hash drifted")
    data["bridge_sha256"] = _sha256(data["bridge_sha256"], "bridge_sha256")
    digest = deepcopy(data)
    digest["bridge_sha256"] = "0" * 64
    if data["bridge_sha256"] != _domain_sha256(
        "clinical-eeg-report-graph-v2-claim-factuality-bridge-v1", digest
    ):
        raise ValueError("report-graph-v2 factuality bridge hash drifted")
    return data


def materialize_report_graph_v2_claim_factuality_bridge(
    report_graph_v2: object,
    *,
    trusted_source_event_findings_v3: (
        Mapping[str, object] | Sequence[object] | None
    ) = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Materialize a lossless five-role portable factuality case."""

    trusted = _trusted_kwargs(
        trusted_source_event_findings_v3=trusted_source_event_findings_v3,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    graph = validate_multievent_soz_report_graph_v2(report_graph_v2, **trusted)
    findings, nodes = _source_indexes(graph)
    projection, binding_by_case_id = _permission_projection(graph, nodes)
    source_bindings = _source_bindings(graph, projection)
    case = _materialize_case(
        graph=graph,
        permission_projection=projection,
        binding_by_case_id=binding_by_case_id,
        findings=findings,
        source_bundle_sha256=str(source_bindings["source_bundle_sha256"]),
    )
    body: dict[str, Any] = {
        "schema_version": REPORT_GRAPH_V2_CLAIM_FACTUALITY_BRIDGE_SCHEMA_VERSION,
        "bridge_id": REPORT_GRAPH_V2_CLAIM_FACTUALITY_BRIDGE_ID,
        "translation_policy": _translation_policy(),
        "source_bindings": source_bindings,
        "permission_projection": projection,
        "case": case,
        "case_sha256": _domain_sha256(
            "clinical-eeg-report-graph-v2-portable-factuality-case-v1", case
        ),
        "bridge_sha256": "",
    }
    _seal(
        body,
        "bridge_sha256",
        "clinical-eeg-report-graph-v2-claim-factuality-bridge-v1",
    )
    return _artifact_shape(body)


def validate_report_graph_v2_claim_factuality_bridge(
    value: object,
    *,
    report_graph_v2: object,
    trusted_source_event_findings_v3: (
        Mapping[str, object] | Sequence[object] | None
    ) = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Validate by replaying the supplied report graph and every v3 source."""

    data = _artifact_shape(value)
    expected = materialize_report_graph_v2_claim_factuality_bridge(
        report_graph_v2,
        trusted_source_event_findings_v3=trusted_source_event_findings_v3,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    if _canonical_json(data) != _canonical_json(expected):
        raise ValueError("report-graph-v2 factuality bridge differs from source replay")
    return data


def evaluate_report_graph_v2_claim_factuality_bridge(
    value: object,
    *,
    report_graph_v2: object,
    trusted_source_event_findings_v3: (
        Mapping[str, object] | Sequence[object] | None
    ) = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
    policy: object | None = None,
) -> dict[str, Any]:
    """Replay the bridge, then run the independent atomic-claim evaluator."""

    validated = validate_report_graph_v2_claim_factuality_bridge(
        value,
        report_graph_v2=report_graph_v2,
        trusted_source_event_findings_v3=trusted_source_event_findings_v3,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    return evaluate_claim_factuality_case(validated["case"], policy=policy)


__all__ = [
    "REPORT_GRAPH_V2_CLAIM_FACTUALITY_BRIDGE_ID",
    "REPORT_GRAPH_V2_CLAIM_FACTUALITY_BRIDGE_SCHEMA_VERSION",
    "evaluate_report_graph_v2_claim_factuality_bridge",
    "materialize_report_graph_v2_claim_factuality_bridge",
    "validate_report_graph_v2_claim_factuality_bridge",
]
