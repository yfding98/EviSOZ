"""Replayable public/synthetic MIL -> report-graph-v2 audit bridge.

The report graph and the mode-aware MIL reference model deliberately have
different authority boundaries.  This module joins them without weakening
either boundary.  It replays the trusted ordered v3 source roster, the event
processing ledger, the typed MIL bag/forward path, every hard-onset input
value hash, event-scoped evidence keys, temporal permissions and constructive
spatial receipts.  The result is an audit/evaluation sidecar only.

There is intentionally no argument for a caller-provided decode.  Candidate
scores are recomputed locally from the replayed forward with no calibration or
formal receipts.  The authorized claim, Qwen and renderer overlays are always
empty.  A later host-only promotion adapter may consume the sealed closure,
but it must not turn this shadow artifact into its own authority.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import torch

from .event_processing_ledger_v2 import validate_event_processing_ledger_v2
from .mode_aware_hierarchical_positive_set_mil_v1 import (
    CompleteRecordModeAwareMILBagV1,
    MODE_AWARE_HIERARCHICAL_MIL_DECODE_SCHEMA_VERSION,
    MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
    MODE_AWARE_HIERARCHICAL_MIL_TRUSTED_REGISTRY_ROUTE_CONNECTED,
    MULTIPLE_MODE_PHENOTYPE,
    ModeAwareMILEventV1,
    ModeAwareMILPolicyV1,
    ModeAwareMILReferenceViewV1,
    ModeAwareMILForwardV1,
    decode_mode_aware_hierarchical_mil_v1,
    forward_mode_aware_hierarchical_mil_v1,
)
from .multievent_soz_report_graph_v2 import (
    MULTIEVENT_SOZ_REPORT_GRAPH_V2_PRIVATE_ROUTE_CONNECTED,
    MULTIEVENT_SOZ_REPORT_GRAPH_V2_QWEN_PRODUCTION_ROUTE_CONNECTED,
    _canonical_electrode,
    _canonical_lead_endpoints,
    validate_multievent_soz_report_graph_v2,
)


MODE_AWARE_MIL_REPORT_GRAPH_V2_BRIDGE_SHADOW_SCHEMA_VERSION = (
    "clinical_eeg_mode_aware_mil_report_graph_v2_bridge_shadow_v1"
)
MODE_AWARE_MIL_REPORT_GRAPH_V2_BRIDGE_SHADOW_ROUTE_ID = (
    "public_synthetic_mode_aware_mil_report_graph_v2_audit_overlay_v1"
)
MODE_AWARE_MIL_REPORT_GRAPH_V2_BRIDGE_FORMAL_ROUTE_CONNECTED = False

REPORT_PERMISSION_ROLES = (
    "ictal_pattern_qualification",
    "onset_time_support",
    "onset_topography_support",
    "course_or_spread_support",
    "counterevidence",
)
MIL_HARD_INPUT_ROLES = (
    "phenotype_logits",
    "onset_channel_logits",
    "quality_weight",
    "onset_safe_mode_embedding",
    "pattern_group_assignment",
)
AGGREGATION_CONTROL_ROLES = (
    "physical_occurrence_assignment",
    "alias_assignment",
    "reference_observation_identity",
    "resolution_ceiling_assignment",
    "channel_region_laterality_ontology",
)

_VIEW_SCOPED_HARD_ROLES = frozenset(
    {
        "phenotype_logits",
        "onset_channel_logits",
        "quality_weight",
        "onset_safe_mode_embedding",
    }
)
_FORBIDDEN_POSITIVE_PERMISSION_ROLES = frozenset(
    {"course_or_spread_support", "counterevidence"}
)
_BASE_REQUIRED_REPORT_ROLES: Mapping[str, frozenset[str]] = {
    "phenotype_logits": frozenset({"onset_time_support"}),
    "onset_channel_logits": frozenset(
        {"onset_time_support", "onset_topography_support"}
    ),
    "quality_weight": frozenset(),
    "onset_safe_mode_embedding": frozenset({"onset_time_support"}),
    "pattern_group_assignment": frozenset({"onset_time_support"}),
}
_REPORT_ROLE_TO_HARD_INPUT: Mapping[str, tuple[str, ...]] = {
    "ictal_pattern_qualification": (),
    "onset_time_support": (
        "phenotype_logits",
        "onset_channel_logits",
        "onset_safe_mode_embedding",
        "pattern_group_assignment",
    ),
    "onset_topography_support": ("onset_channel_logits",),
    "course_or_spread_support": (),
    "counterevidence": (),
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be an opaque identifier")
    return value


def _sha256_string(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode("ascii")
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def _seal(value: dict[str, Any], field: str, domain: str) -> None:
    value[field] = "0" * 64
    value[field] = _canonical_sha256({"binding_domain": domain, "value": value})


def _unique_ids(values: Sequence[str], context: str) -> tuple[str, ...]:
    result = tuple(_identifier(item, context) for item in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicates")
    return result


@dataclass(frozen=True)
class ModeAwareMILHardInputBindingV1:
    """One replayable MIL hard-input -> event EvidenceGraph binding.

    ``source_evidence_keys`` are deliberately composite.  A bare evidence ID
    is not globally unique across events and must never be used as a record
    authority key.
    """

    event_id: str
    physical_occurrence_sha256: str
    reference_id: str | None
    input_role: str
    input_value_sha256: str
    producer_artifact_sha256: str
    source_event_graph_sha256: str
    source_evidence_keys: tuple[tuple[str, str], ...]
    permission_edge_ids: tuple[str, ...]
    raw_sample_dependency_ids: tuple[str, ...]
    source_view_ids: tuple[str, ...]
    constructive_spatial_receipt_ids: tuple[str, ...] = ()
    candidate_unit_type: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.event_id, "hard-input event_id")
        _sha256_string(
            self.physical_occurrence_sha256,
            "hard-input physical_occurrence_sha256",
        )
        if self.input_role not in MIL_HARD_INPUT_ROLES:
            raise ValueError("hard-input role is unsupported")
        if self.input_role in _VIEW_SCOPED_HARD_ROLES:
            if self.reference_id is None:
                raise ValueError("view-scoped hard input requires reference_id")
            _identifier(self.reference_id, "hard-input reference_id")
        elif self.reference_id is not None:
            raise ValueError("pattern-group hard input must not carry reference_id")
        _sha256_string(self.input_value_sha256, "hard-input value hash")
        _sha256_string(
            self.producer_artifact_sha256,
            "hard-input producer artifact hash",
        )
        _sha256_string(
            self.source_event_graph_sha256,
            "hard-input source event graph hash",
        )
        if not self.source_evidence_keys:
            raise ValueError("hard input requires event-scoped source evidence")
        normalized_keys = tuple(
            (
                _identifier(item[0], "hard-input evidence event_id"),
                _identifier(item[1], "hard-input evidence_id"),
            )
            for item in self.source_evidence_keys
            if isinstance(item, tuple) and len(item) == 2
        )
        if len(normalized_keys) != len(self.source_evidence_keys):
            raise TypeError("source_evidence_keys must contain two-item tuples")
        if len(normalized_keys) != len(set(normalized_keys)):
            raise ValueError("hard-input source evidence keys contain duplicates")
        _unique_ids(self.permission_edge_ids, "hard-input permission edge")
        if not self.raw_sample_dependency_ids:
            raise ValueError("hard input requires raw-sample dependencies")
        _unique_ids(
            self.raw_sample_dependency_ids,
            "hard-input raw-sample dependency",
        )
        if not self.source_view_ids:
            raise ValueError("hard input requires source view identities")
        _unique_ids(self.source_view_ids, "hard-input source view")
        _unique_ids(
            self.constructive_spatial_receipt_ids,
            "hard-input constructive spatial receipt",
        )
        if self.input_role == "onset_channel_logits":
            if self.candidate_unit_type not in {"lead", "electrode"}:
                raise ValueError(
                    "onset-channel binding requires lead/electrode unit semantics"
                )
        elif self.candidate_unit_type is not None:
            raise ValueError(
                "candidate_unit_type belongs only to onset-channel bindings"
            )


def mode_aware_mil_hard_input_value_sha256_v1(
    event: ModeAwareMILEventV1,
    *,
    input_role: str,
    reference_id: str | None,
) -> str:
    """Hash the exact tensor/value used by one MIL hard-input role."""

    if not isinstance(event, ModeAwareMILEventV1):
        raise TypeError("hard-input value hashing requires ModeAwareMILEventV1")
    if input_role not in MIL_HARD_INPUT_ROLES:
        raise ValueError("hard-input value hashing role is unsupported")
    if input_role == "pattern_group_assignment":
        if reference_id is not None:
            raise ValueError("pattern-group value is event-scoped")
        value: object = {
            "pattern_group_sha256": event.pattern_group_sha256,
        }
    else:
        if reference_id is None:
            raise ValueError("view-scoped hard-input value requires reference_id")
        view = next(
            (item for item in event.reference_views if item.reference_id == reference_id),
            None,
        )
        if view is None:
            raise ValueError("hard-input reference_id is absent from the event")
        if input_role == "phenotype_logits":
            value = {"tensor_sha256": _tensor_sha256(view.phenotype_logits)}
        elif input_role == "onset_channel_logits":
            value = {"tensor_sha256": _tensor_sha256(view.onset_channel_logits)}
        elif input_role == "onset_safe_mode_embedding":
            value = {
                "tensor_sha256": _tensor_sha256(view.onset_safe_mode_embedding)
            }
        else:
            if not math.isfinite(float(view.quality)):
                raise ValueError("hard-input quality must be finite")
            value = {"quality": float(view.quality)}
    return _canonical_sha256(
        {
            "binding_domain": "clinical-eeg-mode-aware-mil-hard-input-value-v1",
            "input_role": input_role,
            "value": value,
        }
    )


def _canonical_observations(
    event: ModeAwareMILEventV1,
) -> tuple[ModeAwareMILReferenceViewV1, ...]:
    """Mirror the MIL's correlated-copy collapse for binding cardinality."""

    observations: dict[tuple[str, str], ModeAwareMILReferenceViewV1] = {}
    for view in sorted(event.reference_views, key=lambda item: item.reference_id):
        key = (view.producer_sha256, view.mode_embedding_source_sha256)
        observations.setdefault(key, view)
    return tuple(sorted(observations.values(), key=lambda item: item.reference_id))


def _expected_producer(
    event: ModeAwareMILEventV1,
    view: ModeAwareMILReferenceViewV1 | None,
    role: str,
) -> str:
    if role == "pattern_group_assignment":
        return event.pattern_group_sha256
    assert view is not None
    if role == "onset_safe_mode_embedding":
        return view.mode_embedding_source_sha256
    return view.producer_sha256


def _future_free_dependency(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("dependency_status")
        in {"bounded_past_and_present", "exact_instantaneous"}
        and value.get("view_role") == "onset_causal"
        and value.get("dependency_policy") == "past_and_present_only"
        and value.get("future_sample_access") is False
        and value.get("onset_evidence_authorized") is True
        and value.get("onset_support_eligible") is True
    )


def _validate_candidate_axis(
    policy: ModeAwareMILPolicyV1, candidate_unit_type: str
) -> None:
    if candidate_unit_type == "electrode":
        if any(_canonical_electrode(item) is None for item in policy.channel_ids):
            raise ValueError(
                "electrode candidate axis contains a lead or unknown electrode"
            )
    elif candidate_unit_type == "lead":
        if any(_canonical_lead_endpoints(item) is None for item in policy.channel_ids):
            raise ValueError("lead candidate axis contains a non-lead channel")
    else:
        raise ValueError("candidate channel axis unit type is unsupported")


def _replay_forward_and_candidate(
    bag: CompleteRecordModeAwareMILBagV1,
    policy: ModeAwareMILPolicyV1,
    supplied: ModeAwareMILForwardV1,
) -> tuple[ModeAwareMILForwardV1, dict[str, Any]]:
    if not isinstance(supplied, ModeAwareMILForwardV1):
        raise TypeError("bridge requires a typed ModeAwareMILForwardV1")
    replayed = forward_mode_aware_hierarchical_mil_v1(bag, policy)
    if (
        replayed.onset_decision_sha256 != supplied.onset_decision_sha256
        or replayed.spread_decision_sha256 != supplied.spread_decision_sha256
    ):
        raise ValueError("supplied MIL forward does not replay from the typed bag")
    # Decode the supplied object as well so its tensor seals are independently
    # replayed before the sidecar accepts it.
    decoded = decode_mode_aware_hierarchical_mil_v1(
        supplied,
        policy,
        model_receipt=None,
        calibration_receipt=None,
        risk_receipt=None,
        loeo_receipt=None,
        loeo_replay_bag=None,
        input_provenance_receipt=None,
        host_trusted_receipt_maps=None,
    )
    if (
        decoded["schema_version"]
        != MODE_AWARE_HIERARCHICAL_MIL_DECODE_SCHEMA_VERSION
        or decoded["formal_report_authorized"] is not False
        or decoded["qwen_or_report_use_authorized"] is not False
        or decoded["candidate_rankings_are_nonclinical"] is not True
    ):
        raise ValueError("shadow bridge received a non-candidate MIL decode")
    return replayed, decoded


def _validate_ledger_and_order(
    graph: Mapping[str, Any], ledger_value: object
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    ledger = validate_event_processing_ledger_v2(ledger_value)
    source = ledger["source_binding"]
    record = graph["record"]
    if (
        source["recording_id"] != record["record_id"]
        or source["canonical_signal_sha256"] != record["canonical_signal_sha256"]
        or not math.isclose(
            float(source["recording_duration_seconds"]),
            float(record["recording_duration_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("event ledger is not bound to the report-graph record")
    outcomes = {str(item["event_id"]): item for item in ledger["event_outcomes"]}
    ordered_eligible = [
        str(item["event_id"])
        for item in ledger["detector_selected_roster"]
        if outcomes[str(item["event_id"])]["eligibility"][
            "eligible_for_record_aggregation"
        ]
    ]
    graph_order = [str(item["event_id"]) for item in graph["source_event_graphs"]]
    if ordered_eligible != graph_order:
        raise ValueError(
            "ordered eligible detector roster does not equal report-graph sources"
        )
    wrapper_by_event = {
        str(item["event_id"]): item for item in graph["source_event_graphs"]
    }
    for event_id in ordered_eligible:
        if (
            outcomes[event_id]["stage_hashes"]["event_findings_sha256"]
            != wrapper_by_event[event_id]["source_event_findings_v3_sha256"]
        ):
            raise ValueError(
                "event ledger Findings hash does not bind the report-graph source"
            )
    return ledger, outcomes


def _event_contexts(
    graph: Mapping[str, Any],
    bag: CompleteRecordModeAwareMILBagV1,
    forward: ModeAwareMILForwardV1,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    wrappers_by_hash: dict[str, Mapping[str, Any]] = {}
    for wrapper in graph["source_event_graphs"]:
        source_hash = str(wrapper["source_event_findings_v3_sha256"])
        if source_hash in wrappers_by_hash:
            raise ValueError("report graph repeats a source-event content hash")
        wrappers_by_hash[source_hash] = wrapper
    groups: dict[str, list[ModeAwareMILEventV1]] = defaultdict(list)
    for event in bag.events:
        groups[event.physical_occurrence_sha256].append(event)
    if set(wrappers_by_hash) != {
        group[0].source_event_graph_sha256 for group in groups.values()
    }:
        raise ValueError("MIL and report graph do not close the same event sources")
    membership = {
        str(physical): (str(graph_sha), str(mode_id))
        for physical, graph_sha, mode_id in forward.event_mode_membership
    }
    event_node_by_id = {
        str(node["source_ids"][0]): str(node["node_id"])
        for node in graph["derivation_dag"]["nodes"]
        if node["node_type"] == "event"
    }
    rows_by_event: dict[str, dict[str, Any]] = {}
    for physical_sha, aliases in groups.items():
        graph_hashes = {item.source_event_graph_sha256 for item in aliases}
        if len(graph_hashes) != 1:
            raise ValueError("one physical event has conflicting source graph hashes")
        graph_sha = next(iter(graph_hashes))
        wrapper = wrappers_by_hash[graph_sha]
        event_id = str(wrapper["event_id"])
        alias_ids = sorted(item.event_id for item in aliases)
        if event_id not in alias_ids:
            raise ValueError("report-graph event_id is absent from its MIL alias group")
        if physical_sha not in membership or membership[physical_sha][0] != graph_sha:
            raise ValueError("MIL event-to-mode membership is not source-bound")
        representative = next(item for item in aliases if item.event_id == event_id)
        rows_by_event[event_id] = {
            "event_id": event_id,
            "source_event_graph_sha256": graph_sha,
            "physical_occurrence_sha256": physical_sha,
            "alias_event_ids": alias_ids,
            "mode_id": membership[physical_sha][1],
            "event_dag_node_id": event_node_by_id[event_id],
            "representative": representative,
        }
    ordered = [
        rows_by_event[str(wrapper["event_id"])]
        for wrapper in graph["source_event_graphs"]
    ]
    return ordered, rows_by_event


def _report_role_matrix(
    graph: Mapping[str, Any], event_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[Mapping[str, Any]]]]:
    edges_by_event_role: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for edge in graph["evidence_permission_edges"]:
        edges_by_event_role[(str(edge["event_id"]), str(edge["role"]))].append(edge)
    rows: list[dict[str, Any]] = []
    for event_id in event_ids:
        for role in REPORT_PERMISSION_ROLES:
            edges = edges_by_event_role[(event_id, role)]
            rows.append(
                {
                    "event_id": event_id,
                    "report_permission_role": role,
                    "status": "materialized" if edges else "not_expressed_by_source",
                    "permission_edge_ids": sorted(
                        str(item["permission_edge_id"]) for item in edges
                    ),
                    "event_scoped_evidence_keys": sorted(
                        [event_id, str(evidence_id)]
                        for edge in edges
                        for evidence_id in edge["evidence_ids"]
                    ),
                    "permitted_mil_hard_input_roles": list(
                        _REPORT_ROLE_TO_HARD_INPUT[role]
                    ),
                    "positive_hard_input_authorization": role
                    not in _FORBIDDEN_POSITIVE_PERMISSION_ROLES,
                    "ictal_role_is_event_gate_not_tensor_source": role
                    == "ictal_pattern_qualification",
                }
            )
    return rows, edges_by_event_role


def _validate_hard_input_bindings(
    *,
    graph: Mapping[str, Any],
    policy: ModeAwareMILPolicyV1,
    event_contexts: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
    bindings: Sequence[ModeAwareMILHardInputBindingV1],
    edges_by_event_role: Mapping[
        tuple[str, str], Sequence[Mapping[str, Any]]
    ],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if not bindings or not all(
        isinstance(item, ModeAwareMILHardInputBindingV1) for item in bindings
    ):
        raise TypeError("bridge requires typed hard-input bindings")
    candidate_types = {
        str(item.candidate_unit_type)
        for item in bindings
        if item.input_role == "onset_channel_logits"
    }
    if len(candidate_types) != 1:
        raise ValueError("record hard-onset channel axis has mixed unit semantics")
    candidate_unit_type = next(iter(candidate_types))
    _validate_candidate_axis(policy, candidate_unit_type)

    node_by_key = {
        (str(item["event_id"]), str(item["evidence_id"])): item
        for item in graph["finding_evidence_nodes"]
    }
    edge_by_id = {
        str(item["permission_edge_id"]): item
        for item in graph["evidence_permission_edges"]
    }
    receipt_by_id = {
        str(item["receipt_id"]): item
        for item in graph["constructive_spatial_resolution_receipts"]
    }
    context_by_event = {str(item["event_id"]): item for item in event_contexts}

    expected: dict[
        tuple[str, str | None, str],
        tuple[Mapping[str, Any], ModeAwareMILReferenceViewV1 | None],
    ] = {}
    for context in event_contexts:
        event = context["representative"]
        assert isinstance(event, ModeAwareMILEventV1)
        for view in _canonical_observations(event):
            for role in sorted(_VIEW_SCOPED_HARD_ROLES):
                expected[(event.event_id, view.reference_id, role)] = (context, view)
        expected[(event.event_id, None, "pattern_group_assignment")] = (
            context,
            None,
        )
    supplied: dict[tuple[str, str | None, str], ModeAwareMILHardInputBindingV1] = {}
    for binding in bindings:
        key = (binding.event_id, binding.reference_id, binding.input_role)
        if key in supplied:
            raise ValueError("hard-input binding ledger repeats one role input")
        supplied[key] = binding
    if set(supplied) != set(expected):
        missing = sorted(str(item) for item in set(expected).difference(supplied))
        extra = sorted(str(item) for item in set(supplied).difference(expected))
        raise ValueError(
            f"hard-input role roster is not exact; missing={missing}, extra={extra}"
        )

    normalized: list[dict[str, Any]] = []
    constructive_rows: dict[str, dict[str, Any]] = {}
    positive_evidence_by_event: dict[str, set[str]] = defaultdict(set)
    for key in sorted(supplied, key=lambda item: (item[0], str(item[1]), item[2])):
        binding = supplied[key]
        context, view = expected[key]
        event = context["representative"]
        assert isinstance(event, ModeAwareMILEventV1)
        if (
            binding.physical_occurrence_sha256
            != context["physical_occurrence_sha256"]
            or binding.source_event_graph_sha256
            != context["source_event_graph_sha256"]
        ):
            raise ValueError("hard-input binding is attached to the wrong event source")
        expected_value_sha = mode_aware_mil_hard_input_value_sha256_v1(
            event,
            input_role=binding.input_role,
            reference_id=binding.reference_id,
        )
        if binding.input_value_sha256 != expected_value_sha:
            raise ValueError("hard-input tensor/value binding hash is stale")
        if binding.producer_artifact_sha256 != _expected_producer(
            event, view, binding.input_role
        ):
            raise ValueError("hard-input producer artifact binding is stale")
        if any(item[0] != binding.event_id for item in binding.source_evidence_keys):
            raise ValueError(
                "bare evidence ID was laundered across an event boundary"
            )
        evidence_ids = {item[1] for item in binding.source_evidence_keys}
        nodes = []
        for evidence_key in binding.source_evidence_keys:
            node = node_by_key.get(evidence_key)
            if node is None:
                raise ValueError("hard-input binding references unknown event evidence")
            if node["finding_status"] != "present":
                raise ValueError("hard-input binding requires present source evidence")
            nodes.append(node)
        expected_raw_ids = {
            str(dependency["dependency_id"])
            for node in nodes
            for dependency in node["raw_sample_dependencies"]
        }
        if expected_raw_ids != set(binding.raw_sample_dependency_ids):
            raise ValueError("hard-input raw dependency roster is not exact")
        dependencies = [
            dependency
            for node in nodes
            for dependency in node["raw_sample_dependencies"]
        ]
        if not dependencies or any(
            not _future_free_dependency(item) for item in dependencies
        ):
            raise ValueError(
                "future/offline dependency cannot enter a hard-onset MIL input"
            )
        expected_view_ids = {str(item["source_view_id"]) for item in dependencies}
        if expected_view_ids != set(binding.source_view_ids):
            raise ValueError("hard-input source-view roster is not exact")
        if evidence_ids.intersection(event.spread_evidence_ids):
            raise ValueError("spread evidence cannot enter a hard-onset MIL input")

        cited_edges: list[Mapping[str, Any]] = []
        for edge_id in binding.permission_edge_ids:
            edge = edge_by_id.get(edge_id)
            if edge is None or edge["event_id"] != binding.event_id:
                raise ValueError("hard-input permission edge is not event-scoped")
            if not set(str(item) for item in edge["evidence_ids"]).issubset(
                evidence_ids
            ):
                raise ValueError("permission edge evidence is absent from the input binding")
            cited_edges.append(edge)
        cited_roles = {str(item["role"]) for item in cited_edges}
        forbidden = cited_roles.intersection(_FORBIDDEN_POSITIVE_PERMISSION_ROLES)
        if forbidden:
            raise ValueError(
                "course/spread or counterevidence cannot authorize a hard-onset input"
            )
        required_roles = set(_BASE_REQUIRED_REPORT_ROLES[binding.input_role])
        if (
            binding.input_role == "onset_channel_logits"
            and event.resolution_ceiling == "phenotype_only"
        ):
            required_roles.discard("onset_topography_support")
        if not required_roles.issubset(cited_roles):
            raise ValueError("hard-input binding misses required report permission roles")

        if binding.input_role == "onset_channel_logits":
            if binding.candidate_unit_type != candidate_unit_type:
                raise ValueError("onset-channel candidate unit type is inconsistent")
            if event.resolution_ceiling != "phenotype_only" and not (
                binding.constructive_spatial_receipt_ids
            ):
                raise ValueError(
                    "localized onset-channel input lacks constructive spatial receipt"
                )
            for receipt_id in binding.constructive_spatial_receipt_ids:
                receipt = receipt_by_id.get(receipt_id)
                if receipt is None or receipt["event_id"] != binding.event_id:
                    raise ValueError("constructive spatial receipt is not event-scoped")
                if (
                    receipt["target_resolution"] != candidate_unit_type
                    or receipt["target_entity_id"] not in policy.channel_ids
                ):
                    raise ValueError(
                        "constructive spatial receipt does not match the channel axis"
                    )
                if not set(receipt["supporting_evidence_ids"]).issubset(evidence_ids):
                    raise ValueError(
                        "constructive spatial receipt lacks bound input evidence"
                    )
                if not any(
                    edge["role"] == "onset_topography_support"
                    and edge["constructive_spatial_receipt_id"] == receipt_id
                    for edge in cited_edges
                ):
                    raise ValueError(
                        "constructive receipt lacks its cited topography permission edge"
                    )
                constructive_rows[receipt_id] = {
                    "event_id": binding.event_id,
                    "candidate_unit_type": candidate_unit_type,
                    "candidate_id": str(receipt["target_entity_id"]),
                    "constructive_spatial_receipt_id": receipt_id,
                    "event_scoped_evidence_keys": sorted(
                        [binding.event_id, str(item)]
                        for item in receipt["supporting_evidence_ids"]
                    ),
                    "may_authorize_formal_claim_in_this_shadow": False,
                }
        elif binding.constructive_spatial_receipt_ids:
            raise ValueError("non-spatial hard input carries a spatial receipt")

        if binding.input_role != "quality_weight":
            positive_evidence_by_event[binding.event_id].update(evidence_ids)
        row = {
            "event_id": binding.event_id,
            "physical_occurrence_sha256": binding.physical_occurrence_sha256,
            "reference_id": binding.reference_id,
            "input_role": binding.input_role,
            "input_value_sha256": binding.input_value_sha256,
            "producer_artifact_sha256": binding.producer_artifact_sha256,
            "source_event_graph_sha256": binding.source_event_graph_sha256,
            "source_evidence_keys": [list(item) for item in binding.source_evidence_keys],
            "permission_edge_ids": list(binding.permission_edge_ids),
            "report_permission_roles": sorted(cited_roles),
            "raw_sample_dependency_ids": list(binding.raw_sample_dependency_ids),
            "source_view_ids": list(binding.source_view_ids),
            "constructive_spatial_receipt_ids": list(
                binding.constructive_spatial_receipt_ids
            ),
            "candidate_unit_type": binding.candidate_unit_type,
            "future_free_replayed": True,
            "course_or_spread_path_present": False,
            "counterevidence_path_present": False,
            "binding_sha256": "",
        }
        _seal(row, "binding_sha256", "clinical-eeg-mil-hard-input-binding-v1")
        normalized.append(row)

    for context in event_contexts:
        event_id = str(context["event_id"])
        event = context["representative"]
        assert isinstance(event, ModeAwareMILEventV1)
        if set(event.onset_evidence_ids) != positive_evidence_by_event[event_id]:
            raise ValueError(
                "MIL onset evidence roster does not equal event-scoped hard-input evidence"
            )
        course_ids = {
            str(evidence_id)
            for edge in edges_by_event_role[(event_id, "course_or_spread_support")]
            for evidence_id in edge["evidence_ids"]
        }
        if not set(event.spread_evidence_ids).issubset(course_ids):
            raise ValueError("MIL spread evidence lacks course/spread permission")
        if outcomes[event_id]["eligibility"][
            "permitted_to_contribute_onset_positive_evidence"
        ] and not edges_by_event_role[(event_id, "ictal_pattern_qualification")]:
            raise ValueError("positive MIL event lacks ictal-pattern qualification")

    return (
        normalized,
        [constructive_rows[key] for key in sorted(constructive_rows)],
        candidate_unit_type,
    )


def _aggregation_controls(
    contexts: Sequence[Mapping[str, Any]],
    policy: ModeAwareMILPolicyV1,
    candidate_unit_type: str,
) -> list[dict[str, Any]]:
    payloads: Mapping[str, object] = {
        "physical_occurrence_assignment": [
            [item["event_id"], item["physical_occurrence_sha256"]]
            for item in contexts
        ],
        "alias_assignment": [
            [item["physical_occurrence_sha256"], item["alias_event_ids"]]
            for item in contexts
        ],
        "reference_observation_identity": [
            {
                "event_id": item["event_id"],
                "observations": [
                    {
                        "reference_id": view.reference_id,
                        "producer_sha256": view.producer_sha256,
                        "mode_embedding_source_sha256": (
                            view.mode_embedding_source_sha256
                        ),
                    }
                    for view in _canonical_observations(item["representative"])
                ],
            }
            for item in contexts
        ],
        "resolution_ceiling_assignment": [
            [item["event_id"], item["representative"].resolution_ceiling]
            for item in contexts
        ],
        "channel_region_laterality_ontology": {
            "candidate_unit_type": candidate_unit_type,
            "channel_ids": list(policy.channel_ids),
            "channel_to_region": list(policy.channel_to_region),
            "region_to_laterality": [list(item) for item in policy.region_to_laterality],
            "policy_sha256": policy.policy_sha256,
        },
    }
    return [
        {
            "control_role": role,
            "control_value_sha256": _canonical_sha256(
                {
                    "binding_domain": "clinical-eeg-mil-aggregation-control-v1",
                    "control_role": role,
                    "value": payloads[role],
                }
            ),
            "replay_status": "derived_from_typed_bag_policy_and_source_graph",
        }
        for role in AGGREGATION_CONTROL_ROLES
    ]


def _mode_bindings(
    graph: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    forward: ModeAwareMILForwardV1,
    candidate_unit_type: str,
) -> list[dict[str, Any]]:
    context_by_physical = {
        str(item["physical_occurrence_sha256"]): item for item in contexts
    }
    structural_target_by_event = {
        str(next(node["source_ids"][0] for node in graph["derivation_dag"]["nodes"] if node["node_id"] == edge["source_node_id"])): str(edge["target_node_id"])
        for edge in graph["derivation_dag"]["edges"]
        if edge["relation"] == "event_to_mode"
    }
    rows = []
    for mode_id, physical_ids in zip(
        forward.mode_ids, forward.mode_physical_occurrence_sha256s
    ):
        members = [context_by_physical[str(item)] for item in physical_ids]
        rows.append(
            {
                "mil_mode_id": str(mode_id),
                "physical_occurrence_sha256s": [str(item) for item in physical_ids],
                "event_ids": [str(item["event_id"]) for item in members],
                "event_dag_node_ids": [
                    str(item["event_dag_node_id"]) for item in members
                ],
                "source_event_graph_sha256s": [
                    str(item["source_event_graph_sha256"]) for item in members
                ],
                "structural_report_graph_mode_ids": sorted(
                    {
                        structural_target_by_event[str(item["event_id"])]
                        for item in members
                    }
                ),
                "structural_and_mil_mode_semantics_are_identical": False,
                "candidate_unit_type": candidate_unit_type,
                "formal_report_authorized": False,
            }
        )
    return rows


def _candidate_overlay(
    decoded: Mapping[str, Any], candidate_unit_type: str
) -> dict[str, Any]:
    hard = decoded["hard_onset"]
    spread = decoded["soft_spread"]
    modes = []
    for item in decoded["modes"]:
        modes.append(
            {
                "mode_id": str(item["mode_id"]),
                "physical_occurrence_sha256s": deepcopy(
                    item["physical_occurrence_sha256s"]
                ),
                "phenotype_ranking": deepcopy(item["phenotype_ranking"]),
                "channel_hard_onset_ranking": deepcopy(
                    item["electrode_hard_onset_ranking"]
                ),
                "region_hard_onset_ranking": deepcopy(
                    item["region_hard_onset_ranking"]
                ),
                "laterality_ranking": deepcopy(item["laterality_ranking"]),
                "channel_soft_spread_ranking": deepcopy(
                    item["electrode_soft_spread_ranking"]
                ),
                "candidate_unit_type": candidate_unit_type,
                "formal_report_authorized": False,
            }
        )
    if decoded["record_phenotype"] == MULTIPLE_MODE_PHENOTYPE:
        if any(
            (
                hard["electrode_ranking"],
                hard["region_ranking"],
                hard["laterality_ranking"],
                hard["phenotype_ranking"],
                spread["electrode_ranking"],
            )
        ):
            raise ValueError("multi-mode candidate exposes a record-average ranking")
    return {
        "visibility": "audit_and_evaluation_only",
        "may_enter_report_claims": False,
        "may_enter_qwen": False,
        "may_enter_renderer": False,
        "decode_sha256": str(decoded["decode_sha256"]),
        "onset_decision_sha256": str(decoded["onset_decision_sha256"]),
        "spread_decision_sha256": str(decoded["spread_decision_sha256"]),
        "score_semantics": str(decoded["score_semantics"]),
        "record_phenotype": str(decoded["record_phenotype"]),
        "selected_resolution": str(decoded["selected_resolution"]),
        "candidate_unit_type": candidate_unit_type,
        "record_hard_onset": {
            "record_axis_candidate_available": bool(
                hard["record_axis_candidate_available"]
            ),
            "channel_ranking": deepcopy(hard["electrode_ranking"]),
            "region_ranking": deepcopy(hard["region_ranking"]),
            "laterality_ranking": deepcopy(hard["laterality_ranking"]),
            "phenotype_ranking": deepcopy(hard["phenotype_ranking"]),
            "prediction_sets": deepcopy(hard["prediction_sets"]),
        },
        "record_soft_spread": {
            "channel_ranking": deepcopy(spread["electrode_ranking"]),
            "may_support_hard_onset": False,
        },
        "modes": modes,
        "multiple_mode_record_average_withheld": bool(
            decoded["claim_boundary"]["multiple_mode_record_average_withheld"]
        ),
    }


def materialize_mode_aware_mil_report_graph_v2_bridge_shadow_v1(
    report_graph_v2: object,
    *,
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
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Build a sealed audit overlay; never promote a report claim."""

    if isinstance(trusted_source_event_findings_v3, (str, bytes, Mapping)) or not isinstance(
        trusted_source_event_findings_v3, Sequence
    ):
        raise TypeError("bridge requires a trusted ordered v3 source sequence")
    graph = validate_multievent_soz_report_graph_v2(
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
    route_scope = str(graph["route_boundary"]["route_scope"])
    if route_scope not in {"public", "synthetic"}:
        raise ValueError("bridge route must remain public or synthetic")
    if not isinstance(mil_bag, CompleteRecordModeAwareMILBagV1):
        raise TypeError("bridge requires CompleteRecordModeAwareMILBagV1")
    if not isinstance(mil_policy, ModeAwareMILPolicyV1):
        raise TypeError("bridge requires ModeAwareMILPolicyV1")
    expected_scope = "public_source" if route_scope == "public" else "synthetic"
    if mil_bag.source_scope != expected_scope:
        raise ValueError("MIL bag source scope conflicts with report-graph route")
    if (
        mil_bag.record_id != graph["record"]["record_id"]
        or mil_bag.canonical_signal_sha256
        != graph["record"]["canonical_signal_sha256"]
    ):
        raise ValueError("MIL bag is not bound to the report-graph record")

    replayed_forward, decoded = _replay_forward_and_candidate(
        mil_bag, mil_policy, mil_forward
    )
    ledger, outcomes = _validate_ledger_and_order(
        graph, event_processing_ledger_v2
    )
    event_contexts, _context_by_event = _event_contexts(
        graph, mil_bag, replayed_forward
    )
    graph_event_ids = [str(item["event_id"]) for item in event_contexts]
    role_matrix, edges_by_event_role = _report_role_matrix(graph, graph_event_ids)
    normalized_bindings, spatial_bindings, candidate_unit_type = (
        _validate_hard_input_bindings(
            graph=graph,
            policy=mil_policy,
            event_contexts=event_contexts,
            outcomes=outcomes,
            bindings=hard_input_bindings,
            edges_by_event_role=edges_by_event_role,
        )
    )
    controls = _aggregation_controls(
        event_contexts, mil_policy, candidate_unit_type
    )
    modes = _mode_bindings(
        graph, event_contexts, replayed_forward, candidate_unit_type
    )
    event_binding_rows = [
        {
            key: deepcopy(item[key])
            for key in (
                "event_id",
                "source_event_graph_sha256",
                "physical_occurrence_sha256",
                "alias_event_ids",
                "mode_id",
                "event_dag_node_id",
            )
        }
        for item in event_contexts
    ]
    source_closure_payload = {
        "report_graph_sha256": graph["graph_sha256"],
        "source_event_roster_sha256": graph["source_event_roster_sha256"],
        "event_processing_ledger_sha256": ledger["ledger_sha256"],
        "record_id": graph["record"]["record_id"],
        "canonical_signal_sha256": graph["record"]["canonical_signal_sha256"],
        "policy_sha256": replayed_forward.policy_sha256,
        "mil_model_artifact_sha256": replayed_forward.mil_model_artifact_sha256,
        "onset_decision_sha256": replayed_forward.onset_decision_sha256,
        "ordered_event_bindings": event_binding_rows,
        "hard_input_binding_sha256s": [
            item["binding_sha256"] for item in normalized_bindings
        ],
        "aggregation_controls": controls,
    }
    source_graph_closure_sha256 = _canonical_sha256(
        {
            "binding_domain": (
                "clinical-eeg-mode-aware-mil-report-graph-source-closure-v1"
            ),
            "value": source_closure_payload,
        }
    )
    candidate_overlay = _candidate_overlay(decoded, candidate_unit_type)
    bridge: dict[str, Any] = {
        "schema_version": (
            MODE_AWARE_MIL_REPORT_GRAPH_V2_BRIDGE_SHADOW_SCHEMA_VERSION
        ),
        "bridge_id": (
            "MILGRAPHBRIDGE-"
            + _canonical_sha256(
                {
                    "report_graph_sha256": graph["graph_sha256"],
                    "onset_decision_sha256": replayed_forward.onset_decision_sha256,
                    "source_graph_closure_sha256": source_graph_closure_sha256,
                }
            )[:24]
        ),
        "route_boundary": {
            "route_id": MODE_AWARE_MIL_REPORT_GRAPH_V2_BRIDGE_SHADOW_ROUTE_ID,
            "route_scope": route_scope,
            "public_or_synthetic_audit_only": True,
            "private_route_connected": False,
            "clinical_use_authorized": False,
            "formal_report_route_connected": (
                MODE_AWARE_MIL_REPORT_GRAPH_V2_BRIDGE_FORMAL_ROUTE_CONNECTED
            ),
            "qwen_route_connected": False,
            "renderer_route_connected": False,
            "source_report_graph_mutated": False,
        },
        "source_binding": {
            "report_graph_id": graph["graph_id"],
            "report_graph_sha256": graph["graph_sha256"],
            "source_event_roster_sha256": graph["source_event_roster_sha256"],
            "event_processing_ledger_id": ledger["ledger_id"],
            "event_processing_ledger_sha256": ledger["ledger_sha256"],
            "detector_selected_roster_sha256": ledger["source_binding"][
                "detector_selected_roster_sha256"
            ],
            "record_id": graph["record"]["record_id"],
            "canonical_signal_sha256": graph["record"][
                "canonical_signal_sha256"
            ],
            "mil_method_id": MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
            "policy_sha256": replayed_forward.policy_sha256,
            "mil_model_artifact_sha256": (
                replayed_forward.mil_model_artifact_sha256
            ),
            "onset_decision_sha256": replayed_forward.onset_decision_sha256,
            "spread_decision_sha256": replayed_forward.spread_decision_sha256,
            "decode_sha256": decoded["decode_sha256"],
            "source_event_graph_closure_sha256": source_graph_closure_sha256,
        },
        "complete_ordered_event_roster_closure": {
            "detector_event_ids": [
                str(item["event_id"])
                for item in ledger["detector_selected_roster"]
            ],
            "record_aggregation_event_ids": graph_event_ids,
            "report_graph_event_ids": graph_event_ids,
            "mil_unique_physical_event_count": (
                replayed_forward.unique_physical_event_count
            ),
            "status": "closed",
        },
        "event_bindings": event_binding_rows,
        "report_permission_role_matrix": role_matrix,
        "hard_onset_input_role_bindings": normalized_bindings,
        "aggregation_control_bindings": controls,
        "constructive_spatial_candidate_bindings": spatial_bindings,
        "mil_mode_bindings": modes,
        "evaluation_only_candidate_overlay": candidate_overlay,
        "authorized_claim_overlay": [],
        "qwen_lexicalization_slots": [],
        "renderer_projection": [],
        "promotion_gate": {
            "validated_graph_and_trusted_source_replay_closed": True,
            "complete_ordered_event_roster_closed": True,
            "event_scoped_evidence_keys_closed": True,
            "five_report_permission_roles_materialized_without_defaulting": True,
            "five_mil_hard_input_roles_closed": True,
            "tensor_and_value_hashes_replayed": True,
            "future_free_hard_input_provenance_closed": True,
            "constructive_spatial_receipts_bound": True,
            "aggregation_controls_bound": True,
            "candidate_unit_semantics_bound": True,
            "trusted_mil_registry_route_connected": (
                MODE_AWARE_HIERARCHICAL_MIL_TRUSTED_REGISTRY_ROUTE_CONNECTED
            ),
            "report_graph_private_route_connected": (
                MULTIEVENT_SOZ_REPORT_GRAPH_V2_PRIVATE_ROUTE_CONNECTED
            ),
            "report_graph_qwen_production_route_connected": (
                MULTIEVENT_SOZ_REPORT_GRAPH_V2_QWEN_PRODUCTION_ROUTE_CONNECTED
            ),
            "formal_report_authorized": False,
            "qwen_or_report_use_authorized": False,
            "renderer_use_authorized": False,
            "reason_codes": [
                "public_synthetic_shadow_audit_overlay_only",
                "host_only_formal_authority_not_connected",
                "candidate_scores_must_not_enter_claim_qwen_or_renderer",
            ],
        },
        "bridge_sha256": "",
    }
    _seal(
        bridge,
        "bridge_sha256",
        "clinical-eeg-mode-aware-mil-report-graph-v2-bridge-shadow-v1",
    )
    return bridge


def validate_mode_aware_mil_report_graph_v2_bridge_shadow_v1(
    payload: object,
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
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Rebuild the complete bridge and reject any sidecar mutation."""

    if type(payload) is not dict:
        raise TypeError("MIL/report-graph bridge payload must be an object")
    rebuilt = materialize_mode_aware_mil_report_graph_v2_bridge_shadow_v1(
        report_graph_v2,
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
    if payload != rebuilt:
        raise ValueError("MIL/report-graph bridge does not replay from trusted inputs")
    return rebuilt


__all__ = [
    "AGGREGATION_CONTROL_ROLES",
    "MIL_HARD_INPUT_ROLES",
    "MODE_AWARE_MIL_REPORT_GRAPH_V2_BRIDGE_FORMAL_ROUTE_CONNECTED",
    "MODE_AWARE_MIL_REPORT_GRAPH_V2_BRIDGE_SHADOW_ROUTE_ID",
    "MODE_AWARE_MIL_REPORT_GRAPH_V2_BRIDGE_SHADOW_SCHEMA_VERSION",
    "ModeAwareMILHardInputBindingV1",
    "REPORT_PERMISSION_ROLES",
    "materialize_mode_aware_mil_report_graph_v2_bridge_shadow_v1",
    "mode_aware_mil_hard_input_value_sha256_v1",
    "validate_mode_aware_mil_report_graph_v2_bridge_shadow_v1",
]
