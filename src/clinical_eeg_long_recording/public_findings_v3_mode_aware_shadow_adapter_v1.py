"""Target-free Findings-v3 -> mode-aware MIL shadow adapter.

This module closes a deliberately non-clinical public replay path.  It turns
validated, signal-derived ``event_eeg_findings_v3`` ledgers into the typed
inputs required by :mod:`mode_aware_hierarchical_positive_set_mil_v1` and its
report-graph audit bridge.

The adapter is *not* a trained event classifier or SOZ head.  In particular,
an ``unqualified_candidate`` event is forced to ``phenotype_only`` with a
uniform channel axis and a nonlocalizable phenotype candidate.  A causal
field measurement therefore remains visible as an atomic Finding, but cannot
be laundered into a localized onset claim.  Only a separately qualified v3
event carrying constructive lead receipts may obtain a non-uniform channel
candidate in this shadow route.

Inputs are in-memory public/synthetic EEG evidence graphs.  There is no I/O
surface for EDF annotations, spreadsheets, clinical text, doctor labels,
patient metadata or private paths.  Course/spread Findings are intentionally
not projected into the v1 MIL tensor because the current bridge lacks an
exact value/provenance binding for ``spread_channel_logits``; the typed spread
vector is therefore uniform and its evidence roster is empty.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch

from src.soz.geometry import TCP_20_EDGES

from .mode_aware_hierarchical_positive_set_mil_v1 import (
    CompleteRecordModeAwareMILBagV1,
    ModeAwareMILEventV1,
    ModeAwareMILForwardV1,
    ModeAwareMILPolicyV1,
    ModeAwareMILReferenceViewV1,
    forward_mode_aware_hierarchical_mil_v1,
)
from .mode_aware_mil_report_graph_v2_bridge_shadow_v1 import (
    MIL_HARD_INPUT_ROLES,
    ModeAwareMILHardInputBindingV1,
    mode_aware_mil_hard_input_value_sha256_v1,
)
from .multievent_soz_report_graph_v2 import (
    validate_multievent_soz_report_graph_v2,
)


PUBLIC_FINDINGS_V3_MODE_AWARE_SHADOW_ADAPTER_SCHEMA_VERSION = (
    "clinical_eeg_public_findings_v3_mode_aware_shadow_adapter_v1"
)
PUBLIC_FINDINGS_V3_MODE_AWARE_SHADOW_ADAPTER_METHOD_ID = (
    "target_free_untrained_causal_findings_to_mode_mil_shadow_v1"
)

_QUALIFIED_EVENT_STATUSES = {
    "qualified_electrographic_event",
    "qualified_electrographic_seizure",
}
_FUTURE_FREE_DEPENDENCY_STATUSES = {
    "bounded_past_and_present",
    "exact_instantaneous",
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


def _tensor_values(value: torch.Tensor) -> list[float]:
    return [
        round(float(item), 12)
        for item in value.detach().cpu().to(dtype=torch.float64).tolist()
    ]


def _lead_id(edge: Sequence[str]) -> str:
    if len(edge) != 2:
        raise ValueError("TCP edge must contain two electrodes")
    return f"{edge[0]}-{edge[1]}"


def _electrode_side(electrode: str) -> str | None:
    token = str(electrode).upper()
    if token.endswith("Z"):
        return None
    digits = "".join(character for character in token if character.isdigit())
    if not digits:
        return None
    return "left" if int(digits[-1]) % 2 else "right"


def _lead_region(lead_id: str) -> tuple[str, str]:
    endpoints = lead_id.split("-")
    if len(endpoints) != 2:
        raise ValueError("candidate lead ID is malformed")
    sides = {item for item in map(_electrode_side, endpoints) if item is not None}
    if sides == {"left"}:
        return "left_scalp", "left"
    if sides == {"right"}:
        return "right_scalp", "right"
    return "indeterminate_scalp", "indeterminate"


def default_tcp20_mode_aware_shadow_policy_v1() -> ModeAwareMILPolicyV1:
    """Return a coarse side-only ontology for the untrained TCP-20 shadow.

    No lobar label is inferred from a bipolar edge.  This keeps the adapter's
    ontology below the spatial resolution that would require a trained and
    clinically qualified event head.
    """

    channel_ids = tuple(_lead_id(edge) for edge in TCP_20_EDGES)
    channel_regions = tuple(_lead_region(item)[0] for item in channel_ids)
    region_to_laterality: list[tuple[str, str]] = []
    for channel_id, region in zip(channel_ids, channel_regions):
        pair = (region, _lead_region(channel_id)[1])
        if pair not in region_to_laterality:
            region_to_laterality.append(pair)
    return ModeAwareMILPolicyV1(
        channel_ids=channel_ids,
        channel_to_region=channel_regions,
        region_to_laterality=tuple(region_to_laterality),
    )


def _future_free_dependency(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("dependency_status") in _FUTURE_FREE_DEPENDENCY_STATUSES
        and value.get("view_role") == "onset_causal"
        and value.get("dependency_policy") == "past_and_present_only"
        and value.get("future_sample_access") is False
        and value.get("onset_evidence_authorized") is True
        and value.get("onset_support_eligible") is True
    )


def _future_free_onset_ids(
    source: Mapping[str, Any],
    node_by_evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    ids: list[str] = []
    for finding in source["findings"]:
        evidence_id = str(finding["evidence_id"])
        node = node_by_evidence[evidence_id]
        dependencies = node["raw_sample_dependencies"]
        if (
            finding["status"] == "present"
            and finding["intrinsic_evidence_role"] == "onset_eligible"
            and dependencies
            and all(_future_free_dependency(item) for item in dependencies)
        ):
            ids.append(evidence_id)
    result = tuple(sorted(set(ids)))
    if not result:
        raise ValueError(
            "public MIL shadow requires at least one present future-free onset "
            "candidate Finding"
        )
    return result


def _spatial_support_logits(
    source: Mapping[str, Any],
    evidence_ids: Sequence[str],
    policy: ModeAwareMILPolicyV1,
) -> torch.Tensor:
    by_id = {str(item["evidence_id"]): item for item in source["findings"]}
    logits = torch.full((len(policy.channel_ids),), -4.0, dtype=torch.float32)
    observed = False
    for evidence_id in evidence_ids:
        finding = by_id[evidence_id]
        for spatial in finding["spatial_support"]:
            candidate_id = str(spatial["id"])
            if (
                spatial["unit_type"] != "lead"
                or candidate_id not in policy.channel_ids
                or spatial["evidence_eligible"] is not True
                or spatial["observation_status"] not in {"observed", "derived"}
                or spatial["mapping_status"] not in {"direct", "field_qualified"}
            ):
                continue
            score = spatial["support_score"]
            numeric = 1.0 if score is None else float(score)
            if not math.isfinite(numeric):
                raise ValueError("spatial support score must be finite")
            index = policy.channel_ids.index(candidate_id)
            logits[index] = max(
                float(logits[index]),
                float(max(-4.0, min(12.0, math.copysign(math.log1p(abs(numeric)), numeric)))),
            )
            observed = True
    return logits if observed else torch.zeros_like(logits)


def _onset_embedding(
    source: Mapping[str, Any],
    onset_ids: Sequence[str],
    phenotype_logits: torch.Tensor,
    onset_logits: torch.Tensor,
    quality: float,
) -> torch.Tensor:
    findings = {
        str(item["evidence_id"]): item
        for item in source["findings"]
        if str(item["evidence_id"]) in set(onset_ids)
    }
    # Keep physical time relative to the adaptive event window.  Absolute
    # recording time must not define a seizure mode.
    window_start, window_stop = map(
        float, source["window"]["final_interval"]
    )
    duration = max(window_stop - window_start, 1e-12)
    starts: list[float] = []
    stops: list[float] = []
    for finding in findings.values():
        interval = finding["time_interval"]
        if interval is None:
            continue
        starts.append(float(interval["start"]))
        stops.append(float(interval["stop"]))
    relative_start = (
        min(1.0, max(0.0, (min(starts) - window_start) / duration))
        if starts
        else 0.0
    )
    relative_span = (
        min(1.0, max(0.0, (max(stops) - min(starts)) / duration))
        if starts and stops
        else 0.0
    )
    scalar = torch.tensor(
        [
            relative_start,
            relative_span,
            min(1.0, len(onset_ids) / 8.0),
            float(quality),
        ],
        dtype=torch.float32,
    )
    return torch.cat(
        (
            torch.softmax(phenotype_logits, dim=-1),
            torch.softmax(onset_logits, dim=-1),
            scalar,
        )
    )


def _permission_edges_for_ids(
    graph: Mapping[str, Any],
    *,
    event_id: str,
    evidence_ids: Sequence[str],
    roles: set[str],
) -> tuple[str, ...]:
    allowed = set(evidence_ids)
    return tuple(
        sorted(
            str(edge["permission_edge_id"])
            for edge in graph["evidence_permission_edges"]
            if edge["event_id"] == event_id
            and edge["role"] in roles
            and set(str(item) for item in edge["evidence_ids"]).issubset(allowed)
        )
    )


def _constructive_lead_receipts(
    graph: Mapping[str, Any],
    *,
    event_id: str,
    evidence_ids: Sequence[str],
    policy: ModeAwareMILPolicyV1,
) -> tuple[str, ...]:
    allowed = set(evidence_ids)
    return tuple(
        sorted(
            str(receipt["receipt_id"])
            for receipt in graph["constructive_spatial_resolution_receipts"]
            if receipt["event_id"] == event_id
            and receipt["target_resolution"] == "lead"
            and receipt["target_entity_id"] in policy.channel_ids
            and set(
                str(item) for item in receipt["supporting_evidence_ids"]
            ).issubset(allowed)
        )
    )


def _event_and_bindings(
    *,
    graph: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    source: Mapping[str, Any],
    policy: ModeAwareMILPolicyV1,
) -> tuple[ModeAwareMILEventV1, tuple[ModeAwareMILHardInputBindingV1, ...], dict[str, Any]]:
    event_id = str(source["event_id"])
    source_sha = str(wrapper["source_event_findings_v3_sha256"])
    node_by_evidence = {
        str(item["evidence_id"]): item
        for item in graph["finding_evidence_nodes"]
        if item["event_id"] == event_id
    }
    onset_ids = _future_free_onset_ids(source, node_by_evidence)
    time_edges = _permission_edges_for_ids(
        graph,
        event_id=event_id,
        evidence_ids=onset_ids,
        roles={"onset_time_support"},
    )
    if not time_edges:
        raise ValueError("future-free onset candidate lacks an onset-time permission edge")

    qualification_status = str(source["event_qualification"]["status"])
    constructive_receipts = _constructive_lead_receipts(
        graph,
        event_id=event_id,
        evidence_ids=onset_ids,
        policy=policy,
    )
    localized_candidate_available = bool(
        qualification_status in _QUALIFIED_EVENT_STATUSES
        and constructive_receipts
    )
    if localized_candidate_available:
        onset_logits = _spatial_support_logits(source, onset_ids, policy)
        phenotype_logits = torch.tensor([4.0, -2.0, -4.0], dtype=torch.float32)
        resolution_ceiling = "laterality"
    else:
        # The uniform vector is deliberate.  It prevents a measured but
        # unqualified causal field from being rendered as a channel ranking.
        onset_logits = torch.zeros(len(policy.channel_ids), dtype=torch.float32)
        phenotype_logits = torch.tensor([-4.0, -4.0, 4.0], dtype=torch.float32)
        resolution_ceiling = "phenotype_only"
        constructive_receipts = ()

    # The current bridge does not bind the spread-logit value to course
    # evidence.  Keep this axis neutral until that contract is closed.
    spread_logits = torch.zeros(len(policy.channel_ids), dtype=torch.float32)
    quality = 1.0
    embedding = _onset_embedding(
        source, onset_ids, phenotype_logits, onset_logits, quality
    )
    producer_sha = _canonical_sha256(
        {
            "method_id": PUBLIC_FINDINGS_V3_MODE_AWARE_SHADOW_ADAPTER_METHOD_ID,
            "source_event_graph_sha256": source_sha,
            "onset_evidence_ids": list(onset_ids),
            "phenotype_logits": _tensor_values(phenotype_logits),
            "onset_channel_logits": _tensor_values(onset_logits),
            "quality": quality,
            "spread_axis_intentionally_neutral": True,
        }
    )
    embedding_source_sha = _canonical_sha256(
        {
            "method_id": PUBLIC_FINDINGS_V3_MODE_AWARE_SHADOW_ADAPTER_METHOD_ID,
            "source_event_graph_sha256": source_sha,
            "onset_evidence_ids": list(onset_ids),
            "onset_safe_mode_embedding": _tensor_values(embedding),
        }
    )
    pattern_group_sha = _canonical_sha256(
        {
            "method_id": PUBLIC_FINDINGS_V3_MODE_AWARE_SHADOW_ADAPTER_METHOD_ID,
            "quantization": "round_2_decimal_target_free_onset_embedding_v1",
            "embedding": [round(item, 2) for item in _tensor_values(embedding)],
        }
    )
    physical_occurrence_sha = _canonical_sha256(
        {
            "canonical_signal_sha256": graph["record"]["canonical_signal_sha256"],
            "event_id": event_id,
            "analysis_interval_recording_seconds": source["window"][
                "final_interval"
            ],
        }
    )
    reference_id = "TCP20-SHADOW-" + producer_sha[:16]
    view = ModeAwareMILReferenceViewV1(
        reference_id=reference_id,
        producer_sha256=producer_sha,
        phenotype_logits=phenotype_logits,
        onset_channel_logits=onset_logits,
        spread_channel_logits=spread_logits,
        onset_safe_mode_embedding=embedding,
        mode_embedding_source_sha256=embedding_source_sha,
        quality=quality,
    )
    event = ModeAwareMILEventV1(
        event_id=event_id,
        source_event_graph_sha256=source_sha,
        physical_occurrence_sha256=physical_occurrence_sha,
        pattern_group_sha256=pattern_group_sha,
        reference_views=(view,),
        onset_evidence_ids=onset_ids,
        spread_evidence_ids=(),
        resolution_ceiling=resolution_ceiling,
    )

    dependencies = [
        dependency
        for evidence_id in onset_ids
        for dependency in node_by_evidence[evidence_id]["raw_sample_dependencies"]
    ]
    raw_ids = tuple(sorted({str(item["dependency_id"]) for item in dependencies}))
    source_view_ids = tuple(sorted({str(item["source_view_id"]) for item in dependencies}))
    evidence_keys = tuple((event_id, evidence_id) for evidence_id in onset_ids)
    topography_edges = (
        _permission_edges_for_ids(
            graph,
            event_id=event_id,
            evidence_ids=onset_ids,
            roles={"onset_topography_support"},
        )
        if localized_candidate_available
        else ()
    )
    bindings: list[ModeAwareMILHardInputBindingV1] = []
    for role in MIL_HARD_INPUT_ROLES:
        reference = None if role == "pattern_group_assignment" else reference_id
        if role == "onset_channel_logits":
            permission_edges = tuple(sorted(set((*time_edges, *topography_edges))))
            receipt_ids = constructive_receipts
            candidate_unit_type: str | None = "lead"
        elif role in {
            "phenotype_logits",
            "onset_safe_mode_embedding",
            "pattern_group_assignment",
        }:
            permission_edges = time_edges
            receipt_ids = ()
            candidate_unit_type = None
        else:
            permission_edges = ()
            receipt_ids = ()
            candidate_unit_type = None
        if role == "pattern_group_assignment":
            role_producer = pattern_group_sha
        elif role == "onset_safe_mode_embedding":
            role_producer = embedding_source_sha
        else:
            role_producer = producer_sha
        bindings.append(
            ModeAwareMILHardInputBindingV1(
                event_id=event_id,
                physical_occurrence_sha256=physical_occurrence_sha,
                reference_id=reference,
                input_role=role,
                input_value_sha256=mode_aware_mil_hard_input_value_sha256_v1(
                    event,
                    input_role=role,
                    reference_id=reference,
                ),
                producer_artifact_sha256=role_producer,
                source_event_graph_sha256=source_sha,
                source_evidence_keys=evidence_keys,
                permission_edge_ids=permission_edges,
                raw_sample_dependency_ids=raw_ids,
                source_view_ids=source_view_ids,
                constructive_spatial_receipt_ids=receipt_ids,
                candidate_unit_type=candidate_unit_type,
            )
        )
    receipt = {
        "event_id": event_id,
        "source_event_graph_sha256": source_sha,
        "qualification_status": qualification_status,
        "future_free_onset_evidence_ids": list(onset_ids),
        "localized_candidate_available": localized_candidate_available,
        "resolution_ceiling": resolution_ceiling,
        "channel_logits_uniform": not localized_candidate_available,
        "spread_logits_uniform_and_unbound": True,
        "producer_artifact_sha256": producer_sha,
        "mode_embedding_source_sha256": embedding_source_sha,
        "pattern_group_sha256": pattern_group_sha,
        "physical_occurrence_sha256": physical_occurrence_sha,
        "recommended_event_ledger_status": (
            "completed_findings"
            if localized_candidate_available
            else "completed_findings_onset_nonlocalizable"
        ),
    }
    return event, tuple(bindings), receipt


def build_public_findings_v3_mode_aware_shadow_inputs_v1(
    report_graph_v2: object,
    *,
    trusted_source_event_findings_v3: Sequence[object],
    record_pseudonym: str,
    mil_policy: ModeAwareMILPolicyV1 | None = None,
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
) -> tuple[
    CompleteRecordModeAwareMILBagV1,
    ModeAwareMILPolicyV1,
    ModeAwareMILForwardV1,
    tuple[ModeAwareMILHardInputBindingV1, ...],
    dict[str, Any],
]:
    """Build replayable untrained MIL inputs for a public report-graph shadow."""

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
    if graph["route_boundary"]["route_scope"] != "public":
        raise ValueError("public Findings adapter requires a public report graph")
    if not isinstance(record_pseudonym, str) or not record_pseudonym:
        raise ValueError("record_pseudonym must be a non-empty opaque identifier")
    policy = (
        default_tcp20_mode_aware_shadow_policy_v1()
        if mil_policy is None
        else mil_policy
    )
    if not isinstance(policy, ModeAwareMILPolicyV1):
        raise TypeError("mil_policy must be ModeAwareMILPolicyV1")
    sources = [deepcopy(dict(item)) for item in trusted_source_event_findings_v3]
    sources_by_event = {str(item["event_id"]): item for item in sources}
    wrappers = graph["source_event_graphs"]
    if set(sources_by_event) != {str(item["event_id"]) for item in wrappers}:
        raise ValueError("trusted source roster and public report graph differ")

    events: list[ModeAwareMILEventV1] = []
    bindings: list[ModeAwareMILHardInputBindingV1] = []
    event_receipts: list[dict[str, Any]] = []
    for wrapper in wrappers:
        source = sources_by_event[str(wrapper["event_id"])]
        event, event_bindings, receipt = _event_and_bindings(
            graph=graph,
            wrapper=wrapper,
            source=source,
            policy=policy,
        )
        events.append(event)
        bindings.extend(event_bindings)
        event_receipts.append(receipt)

    model_sha = _canonical_sha256(
        {
            "method_id": PUBLIC_FINDINGS_V3_MODE_AWARE_SHADOW_ADAPTER_METHOD_ID,
            "policy_sha256": policy.policy_sha256,
            "trained_weights_present": False,
            "clinical_term_qualification_head_present": False,
            "spread_logit_value_binding_present": False,
        }
    )
    bag = CompleteRecordModeAwareMILBagV1(
        patient_uid=record_pseudonym,
        record_id=str(graph["record"]["record_id"]),
        canonical_signal_sha256=str(graph["record"]["canonical_signal_sha256"]),
        mil_model_artifact_sha256=model_sha,
        events=tuple(events),
        source_scope="public_source",
    )
    forward = forward_mode_aware_hierarchical_mil_v1(bag, policy)
    receipt: dict[str, Any] = {
        "schema_version": (
            PUBLIC_FINDINGS_V3_MODE_AWARE_SHADOW_ADAPTER_SCHEMA_VERSION
        ),
        "method_id": PUBLIC_FINDINGS_V3_MODE_AWARE_SHADOW_ADAPTER_METHOD_ID,
        "record_id": graph["record"]["record_id"],
        "canonical_signal_sha256": graph["record"]["canonical_signal_sha256"],
        "source_event_roster_sha256": graph["source_event_roster_sha256"],
        "policy_sha256": policy.policy_sha256,
        "mil_model_artifact_sha256": model_sha,
        "onset_decision_sha256": forward.onset_decision_sha256,
        "spread_decision_sha256": forward.spread_decision_sha256,
        "event_receipts": event_receipts,
        "scope_receipt": {
            "eeg_findings_only": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "doctor_labels_used": False,
            "clinical_text_used": False,
            "private_source_used": False,
            "trained_clinical_head_present": False,
            "candidate_shadow_only": True,
            "formal_clinical_claim_authorized": False,
            "unqualified_event_forces_uniform_channel_axis": True,
            "spread_logits_forced_uniform_until_value_binding_exists": True,
        },
        "known_contract_gaps": [
            "no_trained_or_term_qualified_event_phenotype_head",
            "spread_channel_logits_lack_exact_value_to_course_evidence_binding",
            "candidate_scores_are_uncalibrated_and_nonclinical",
        ],
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = _canonical_sha256(
        {
            "binding_domain": (
                "clinical-eeg-public-findings-v3-mode-aware-shadow-adapter-v1"
            ),
            "value": receipt,
        }
    )
    return bag, policy, forward, tuple(bindings), receipt


__all__ = [
    "PUBLIC_FINDINGS_V3_MODE_AWARE_SHADOW_ADAPTER_METHOD_ID",
    "PUBLIC_FINDINGS_V3_MODE_AWARE_SHADOW_ADAPTER_SCHEMA_VERSION",
    "build_public_findings_v3_mode_aware_shadow_inputs_v1",
    "default_tcp20_mode_aware_shadow_policy_v1",
]
