"""Additive, event-scope-only projection for the Minimum Event Evidence Card.

The frozen v1 twelve-slot registry intentionally retains historical
``record_non_event_context`` atoms and queries for hash-stable accounting.
Those objects must not be copied into a per-event card.  This module validates
an ``event_eeg_findings_v3`` payload and the frozen registry-closure receipt,
then creates a separate v2 projection containing only event-scoped roster
items, operational queries, and Findings.

The record context is represented by one opaque ``record_context_card_id``.
No context payload is accepted by the public API or embedded in the result.
Validation is source-bound: the complete projection is rebuilt from the
validated Findings, closure receipt, checked-in policies, and the supplied
context-card ID.  Recomputing the projection self hash after tampering cannot
make a modified object valid.

This is a structural shadow contract.  It does not mutate source assertions,
bind Findings to operational queries, authorize clinical absence or report
promotion, connect Qwen, or support a cortical SOZ/EZ claim.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .event_findings_atom_roster import load_event_findings_atom_roster_policy
from .event_findings_term_query_denominator_v2 import (
    load_event_findings_term_query_denominator_policy_v2,
)
from .event_findings_v3_validation import validate_event_eeg_findings_v3_payload
from .minimum_event_evidence_card_registry_v1 import (
    validate_minimum_event_evidence_card_closure_receipt_v1,
)


EVENT_CARD_PROJECTION_SCHEMA_VERSION_V2 = "clinical_eeg_event_card_projection_v2"
EVENT_CARD_PROJECTOR_ID_V2 = "clinical_eeg_event_card_event_scope_projector_v2"
EVENT_CARD_SCOPE_FILTER_POLICY_ID_V2 = "EVENT-CARD-EVENT-SCOPE-FILTER-V2"

_ROOT = Path(__file__).resolve().parents[2]
EVENT_CARD_PROJECTION_SCHEMA_PATH_V2 = (
    _ROOT / "schemas" / "clinical_eeg_event_card_projection_v2.schema.json"
)

_FORBIDDEN_TEMPORAL_CONTEXT = "record_non_event_context"
_FORBIDDEN_INTRINSIC_ROLE = "non_event_context"
_FORBIDDEN_SIGNAL_CONTEXT = "outside_candidate_protection"
_EVENT_INTRINSIC_ROLES = {
    "onset_eligible",
    "early_context",
    "later_involvement",
    "limitation",
}
_RECORD_CONTEXT_ID_RE = re.compile(r"^RNECARD-[a-f0-9]{24}$")

_SOURCE_FIREWALL: Mapping[str, object] = {
    "eeg_signal_and_allowlisted_acquisition_metadata_only": True,
    "source_inference_exclusions_replayed": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_reports_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "ecg_emg_eog_used": False,
    "qwen_used": False,
    "record_context_payload_used": False,
}
_AUTHORIZATION: Mapping[str, object] = {
    "additive_projection_only": True,
    "source_finding_mutation_authorized": False,
    "non_event_context_materialization_authorized": False,
    "record_context_payload_embedding_authorized": False,
    "finding_to_query_binding_claimed": False,
    "clinical_absence_authorized": False,
    "report_promotion_authorized": False,
    "qwen_authorized": False,
    "production_connection_authorized": False,
    "clinical_correctness_claimed": False,
    "soz_or_ez_claim_authorized": False,
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
    return _canonical_sha256({"domain": domain, "value": value})


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if type(value) is not dict:
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _schema_errors(value: object) -> list[str]:
    validator = Draft202012Validator(_read_json(EVENT_CARD_PROJECTION_SCHEMA_PATH_V2))
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    rendered: list[str] = []
    for error in errors[:12]:
        pointer = "/" + "/".join(str(part) for part in error.path)
        rendered.append(f"{pointer}: {error.message}")
    if len(errors) > 12:
        rendered.append(f"... {len(errors) - 12} more error(s)")
    return rendered


def _reject_nonfinite(value: object, context: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{context}[{index}]")


def _require_record_context_card_id(value: object) -> str:
    if type(value) is not str or _RECORD_CONTEXT_ID_RE.fullmatch(value) is None:
        raise ValueError(
            "record_context_card_id must match RNECARD- followed by 24 lowercase hex digits"
        )
    return value


def _source_bundle_sha256(source_bindings: Mapping[str, object]) -> str:
    body = deepcopy(dict(source_bindings))
    body.pop("source_bundle_sha256", None)
    return _domain_sha256("clinical-eeg-event-card-projection-v2-sources", body)


def _is_record_context_roster_spec(spec: Mapping[str, object]) -> bool:
    roles = {str(item) for item in spec.get("allowed_intrinsic_evidence_roles", [])}
    if roles and roles <= {_FORBIDDEN_INTRINSIC_ROLE}:
        return True
    return str(spec.get("enumeration_scope", "")) == _FORBIDDEN_TEMPORAL_CONTEXT


def _validated_sources(
    *,
    event_findings_v3: object,
    registry_closure_receipt_v1: object,
    trusted_registry_closure_receipt_sha256: str | None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ),
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    event_source = validate_event_eeg_findings_v3_payload(
        event_findings_v3,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    exclusions = event_source["provenance"]["inference_exclusions"]
    non_eeg_or_unknown = sorted(
        str(key) for key, state in exclusions.items() if state is not False
    )
    if non_eeg_or_unknown:
        raise ValueError(
            "event-card projection requires explicit EEG-only source exclusions; "
            f"non-false states={non_eeg_or_unknown}"
        )

    closure = validate_minimum_event_evidence_card_closure_receipt_v1(
        registry_closure_receipt_v1,
        trusted_receipt_sha256=trusted_registry_closure_receipt_sha256,
    )
    atom_policy = load_event_findings_atom_roster_policy()
    query_policy = load_event_findings_term_query_denominator_policy_v2()
    source_contracts = closure["source_contracts"]
    if (
        source_contracts["atom_roster_policy_sha256"]
        != atom_policy["policy_sha256"]
        or source_contracts["term_query_denominator_policy_sha256"]
        != query_policy["policy_sha256"]
    ):
        raise ValueError("registry closure source-policy binding drifted")
    return event_source, closure, atom_policy, query_policy


def _scope_partition(
    closure: Mapping[str, object],
    atom_policy: Mapping[str, object],
    query_policy: Mapping[str, object],
) -> tuple[
    list[dict[str, object]],
    set[str],
    dict[str, set[str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    query_by_id = {
        str(row["term_query_id"]): row for row in query_policy["query_specs"]
    }
    core_by_id = {
        str(row["atom_id"]): row for row in atom_policy["core_atom_specs"]
    }
    child_by_id = {
        str(row["child_roster_id"]): row
        for row in atom_policy["child_roster_specs"]
    }

    forbidden_query_specs = [
        row
        for row in query_policy["query_specs"]
        if str(row["temporal_context"]) == _FORBIDDEN_TEMPORAL_CONTEXT
    ]
    forbidden_terms = {str(row["term_id"]) for row in forbidden_query_specs}
    allowed_term_roles: dict[str, set[str]] = {}
    for row in query_policy["query_specs"]:
        if str(row["temporal_context"]) == _FORBIDDEN_TEMPORAL_CONTEXT:
            continue
        allowed_term_roles.setdefault(str(row["term_id"]), set()).add(
            str(row["intrinsic_evidence_role"])
        )
    overlap = sorted(forbidden_terms & set(allowed_term_roles))
    if overlap:
        raise ValueError(
            "term-query policy assigns terms to both event and record context: "
            f"{overlap}"
        )

    forbidden_primary_keys = {
        (
            str(row["primary_roster_item"]["roster_item_kind"]),
            str(row["primary_roster_item"]["roster_item_id"]),
        )
        for row in forbidden_query_specs
    }
    slots: list[dict[str, object]] = []
    excluded_roster: list[dict[str, str]] = []
    excluded_queries: list[dict[str, str]] = []
    included_roster_keys: set[tuple[str, str]] = set()
    roster_by_slot_id: dict[str, list[dict[str, object]]] = {}

    # First close the complete projected roster.  Query validation happens in
    # a second pass so it cannot depend on slot order, even though the frozen
    # v1 closure also requires every query to share a slot with its primary.
    for source_slot in closure["slots"]:
        roster_items: list[dict[str, object]] = []
        for source_row in source_slot["roster_items"]:
            kind = str(source_row["roster_item_kind"])
            item_id = str(source_row["roster_item_id"])
            spec = core_by_id[item_id] if kind == "core_atom" else child_by_id[item_id]
            record_scoped = (
                _is_record_context_roster_spec(spec)
                or (kind, item_id) in forbidden_primary_keys
            )
            if record_scoped:
                excluded_roster.append(
                    {"roster_item_kind": kind, "roster_item_id": item_id}
                )
                continue
            projected_roster = {
                "roster_item_kind": kind,
                "roster_item_id": item_id,
                "source_group": str(source_row["source_group"]),
                "source_implementation_status": str(
                    source_row["source_implementation_status"]
                ),
            }
            roster_items.append(projected_roster)
            included_roster_keys.add((kind, item_id))
        roster_by_slot_id[str(source_slot["slot_id"])] = roster_items

    for source_slot in closure["slots"]:
        operational_queries: list[dict[str, object]] = []
        for closure_query in source_slot["operational_queries"]:
            query_id = str(closure_query["term_query_id"])
            spec = query_by_id[query_id]
            if str(spec["temporal_context"]) == _FORBIDDEN_TEMPORAL_CONTEXT:
                excluded_queries.append(
                    {"term_query_id": query_id, "term_id": str(spec["term_id"])}
                )
                continue
            primary = spec["primary_roster_item"]
            primary_key = (
                str(primary["roster_item_kind"]),
                str(primary["roster_item_id"]),
            )
            if primary_key not in included_roster_keys:
                raise ValueError(
                    f"event query {query_id!r} lost its primary event roster item"
                )
            operational_queries.append(
                {
                    "term_query_id": query_id,
                    "term_id": str(spec["term_id"]),
                    "claim_kind": str(spec["claim_kind"]),
                    "temporal_context": str(spec["temporal_context"]),
                    "intrinsic_evidence_role": str(spec["intrinsic_evidence_role"]),
                    "scope_id": str(spec["scope_id"]),
                    "primary_roster_item_kind": primary_key[0],
                    "primary_roster_item_id": primary_key[1],
                    "source_implementation_status": str(
                        closure_query["source_implementation_status"]
                    ),
                    "source_report_promotion_authorized": False,
                }
            )

        slots.append(
            {
                "slot_index": int(source_slot["slot_index"]),
                "slot_id": str(source_slot["slot_id"]),
                "roster_items": roster_by_slot_id[str(source_slot["slot_id"])],
                "operational_queries": operational_queries,
            }
        )
    return (
        slots,
        forbidden_terms,
        allowed_term_roles,
        excluded_roster,
        excluded_queries,
    )


def _project_findings(
    source_findings: list[Mapping[str, object]],
    *,
    forbidden_terms: set[str],
    allowed_term_roles: Mapping[str, set[str]],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    projected: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for source_row in source_findings:
        evidence_id = str(source_row["evidence_id"])
        term_id = str(source_row["term"]["term_id"])
        role = str(source_row["intrinsic_evidence_role"])
        signal_context = str(source_row["signal_temporal_context"])
        finding_sha256 = _domain_sha256(
            "clinical-eeg-event-card-v2-source-finding", source_row
        )
        registered_event_roles = allowed_term_roles.get(term_id)
        if registered_event_roles is not None and role not in registered_event_roles:
            raise ValueError(
                f"finding {evidence_id!r} role does not match its event query term"
            )
        record_scoped = term_id in forbidden_terms or (
            registered_event_roles is None
            and (
                role == _FORBIDDEN_INTRINSIC_ROLE
                or signal_context == _FORBIDDEN_SIGNAL_CONTEXT
            )
        )
        if record_scoped:
            excluded.append(
                {"evidence_id": evidence_id, "source_finding_sha256": finding_sha256}
            )
            continue
        if registered_event_roles is None and role not in _EVENT_INTRINSIC_ROLES:
            raise ValueError(
                f"finding {evidence_id!r} has no authorized event-scope role"
            )
        projected.append(
            {
                "evidence_id": evidence_id,
                "term_id": term_id,
                "scope_basis": (
                    "trusted_event_query_term"
                    if registered_event_roles is not None
                    else "validated_event_intrinsic_role_no_query_binding"
                ),
                "source_finding_sha256": finding_sha256,
                "finding": deepcopy(dict(source_row)),
            }
        )
    return projected, excluded


def _build_projection(
    *,
    event_source: Mapping[str, object],
    closure: Mapping[str, object],
    atom_policy: Mapping[str, object],
    query_policy: Mapping[str, object],
    record_context_card_id: str,
) -> dict[str, Any]:
    (
        slots,
        forbidden_terms,
        allowed_term_roles,
        excluded_roster,
        excluded_queries,
    ) = _scope_partition(closure, atom_policy, query_policy)
    projected_findings, excluded_findings = _project_findings(
        list(event_source["findings"]),
        forbidden_terms=forbidden_terms,
        allowed_term_roles=allowed_term_roles,
    )

    source_bindings: dict[str, object] = {
        "event_findings_schema_version": str(event_source["schema_version"]),
        "event_findings_sha256": _domain_sha256(
            "clinical-eeg-event-card-v2-event-findings-v3", event_source
        ),
        "registry_closure_schema_version": str(closure["schema_version"]),
        "registry_closure_receipt_sha256": str(closure["receipt_sha256"]),
        "minimum_event_card_registry_id": str(closure["registry_id"]),
        "minimum_event_card_registry_sha256": str(closure["registry_sha256"]),
        "atom_roster_id": str(atom_policy["roster_id"]),
        "atom_roster_policy_sha256": str(atom_policy["policy_sha256"]),
        "term_query_policy_id": str(query_policy["policy_id"]),
        "term_query_policy_sha256": str(query_policy["policy_sha256"]),
        "record_context_card_id": record_context_card_id,
        "source_bundle_sha256": "",
    }
    source_bindings["source_bundle_sha256"] = _source_bundle_sha256(source_bindings)
    projection_id = f"EVCARDV2-{str(source_bindings['source_bundle_sha256'])[:24]}"

    source_roster_count = sum(len(slot["roster_items"]) for slot in closure["slots"])
    event_roster_count = sum(len(slot["roster_items"]) for slot in slots)
    source_query_count = sum(
        len(slot["operational_queries"]) for slot in closure["slots"]
    )
    event_query_count = sum(len(slot["operational_queries"]) for slot in slots)
    source_finding_count = len(event_source["findings"])
    event_finding_count = len(projected_findings)

    projection: dict[str, Any] = {
        "schema_version": EVENT_CARD_PROJECTION_SCHEMA_VERSION_V2,
        "projector_id": EVENT_CARD_PROJECTOR_ID_V2,
        "projection_id": projection_id,
        "owner": {
            "owner_kind": "event",
            "event_id": str(event_source["event_id"]),
            "recording_id": str(event_source["provenance"]["record_id"]),
            "canonical_signal_sha256": str(
                event_source["provenance"]["canonical_signal_sha256"]
            ),
        },
        "source_bindings": source_bindings,
        "record_context_reference": {
            "card_id": record_context_card_id,
            "reference_only": True,
            "payload_embedded": False,
        },
        "scope_filter": {
            "policy_id": EVENT_CARD_SCOPE_FILTER_POLICY_ID_V2,
            "forbidden_temporal_context": _FORBIDDEN_TEMPORAL_CONTEXT,
            "fallback_forbidden_intrinsic_evidence_role": (
                _FORBIDDEN_INTRINSIC_ROLE
            ),
            "fallback_forbidden_signal_temporal_context": (
                _FORBIDDEN_SIGNAL_CONTEXT
            ),
            "unknown_query_term_scope_policy": (
                "retain_only_with_validated_event_intrinsic_role"
            ),
        },
        "slots": slots,
        "event_findings": projected_findings,
        "finding_scope_accounting": {
            "source_count": source_finding_count,
            "materialized_event_count": event_finding_count,
            "excluded_record_context_count": len(excluded_findings),
            "excluded_source_digest_sha256": _domain_sha256(
                "clinical-eeg-event-card-v2-excluded-findings", excluded_findings
            ),
        },
        "closure": {
            "expected_slot_count": 12,
            "observed_slot_count": len(slots),
            "source_roster_item_count": source_roster_count,
            "materialized_event_roster_item_count": event_roster_count,
            "excluded_record_context_roster_item_count": len(excluded_roster),
            "source_query_count": source_query_count,
            "materialized_event_query_count": event_query_count,
            "excluded_record_context_query_count": len(excluded_queries),
            "source_finding_count": source_finding_count,
            "materialized_event_finding_count": event_finding_count,
            "excluded_record_context_finding_count": len(excluded_findings),
            "registry_closure_replayed": True,
            "source_findings_validated": True,
            "twelve_slots_preserved": True,
            "event_queries_only": True,
            "event_roster_items_only": True,
            "event_findings_only": True,
            "every_materialized_finding_exactly_once": True,
            "record_context_reference_only": True,
            "record_context_payload_embedded": False,
            "finding_to_query_binding_claimed": False,
        },
        "source_firewall": deepcopy(dict(_SOURCE_FIREWALL)),
        "authorization": deepcopy(dict(_AUTHORIZATION)),
        "projection_sha256": "",
    }
    projection["projection_sha256"] = _self_hash(projection, "projection_sha256")
    return projection


def _validate_projection_structure(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("event card projection v2 must be an object")
    candidate = deepcopy(value)
    _reject_nonfinite(candidate)
    errors = _schema_errors(candidate)
    if errors:
        raise ValueError(
            "event card projection v2 schema validation failed: " + "; ".join(errors)
        )
    if candidate["projection_sha256"] != _self_hash(
        candidate, "projection_sha256"
    ):
        raise ValueError("event card projection v2 self hash drifted")
    if candidate["source_bindings"]["source_bundle_sha256"] != _source_bundle_sha256(
        candidate["source_bindings"]
    ):
        raise ValueError("event card projection v2 source bundle hash drifted")
    expected_projection_id = (
        "EVCARDV2-"
        + str(candidate["source_bindings"]["source_bundle_sha256"])[:24]
    )
    if candidate["projection_id"] != expected_projection_id:
        raise ValueError("event card projection v2 projection ID drifted")
    if (
        candidate["record_context_reference"]["card_id"]
        != candidate["source_bindings"]["record_context_card_id"]
    ):
        raise ValueError("record context ID source binding drifted")
    if candidate["source_firewall"] != _SOURCE_FIREWALL:
        raise ValueError("event card projection v2 EEG-only firewall drifted")
    if candidate["authorization"] != _AUTHORIZATION:
        raise ValueError("event card projection v2 authorization drifted")

    atom_policy = load_event_findings_atom_roster_policy()
    query_policy = load_event_findings_term_query_denominator_policy_v2()
    forbidden_terms = {
        str(row["term_id"])
        for row in query_policy["query_specs"]
        if str(row["temporal_context"]) == _FORBIDDEN_TEMPORAL_CONTEXT
    }
    slot_indices = [int(row["slot_index"]) for row in candidate["slots"]]
    if slot_indices != list(range(1, 13)):
        raise ValueError("event card projection v2 slot order drifted")

    roster_keys: list[tuple[str, str]] = []
    query_ids: list[str] = []
    for slot in candidate["slots"]:
        for row in slot["roster_items"]:
            key = (str(row["roster_item_kind"]), str(row["roster_item_id"]))
            roster_keys.append(key)
            source_specs = (
                atom_policy["core_atom_specs"]
                if key[0] == "core_atom"
                else atom_policy["child_roster_specs"]
            )
            id_field = "atom_id" if key[0] == "core_atom" else "child_roster_id"
            spec = next(item for item in source_specs if str(item[id_field]) == key[1])
            if _is_record_context_roster_spec(spec):
                raise ValueError("record-context roster item entered Event Card")
        for row in slot["operational_queries"]:
            query_ids.append(str(row["term_query_id"]))
            if (
                str(row["temporal_context"]) == _FORBIDDEN_TEMPORAL_CONTEXT
                or str(row["term_id"]) in forbidden_terms
            ):
                raise ValueError("record-context query or term entered Event Card")
    if len(roster_keys) != len(set(roster_keys)):
        raise ValueError("event card projection v2 duplicates a roster item")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("event card projection v2 duplicates an operational query")

    allowed_term_roles: dict[str, set[str]] = {}
    for source_query in query_policy["query_specs"]:
        if str(source_query["temporal_context"]) == _FORBIDDEN_TEMPORAL_CONTEXT:
            continue
        allowed_term_roles.setdefault(str(source_query["term_id"]), set()).add(
            str(source_query["intrinsic_evidence_role"])
        )

    evidence_ids: list[str] = []
    for index, row in enumerate(candidate["event_findings"]):
        finding = row["finding"]
        evidence_id = str(row["evidence_id"])
        term_id = str(row["term_id"])
        evidence_ids.append(evidence_id)
        if evidence_id != str(finding.get("evidence_id")):
            raise ValueError(f"event_findings[{index}] evidence ID drifted")
        if term_id != str(finding.get("term", {}).get("term_id")):
            raise ValueError(f"event_findings[{index}] term ID drifted")
        if row["source_finding_sha256"] != _domain_sha256(
            "clinical-eeg-event-card-v2-source-finding", finding
        ):
            raise ValueError(f"event_findings[{index}] source Finding hash drifted")
        registered_event_roles = allowed_term_roles.get(term_id)
        role = str(finding.get("intrinsic_evidence_role"))
        signal_context = str(finding.get("signal_temporal_context"))
        if term_id in forbidden_terms:
            raise ValueError("record-context Finding entered Event Card")
        if registered_event_roles is not None:
            if role not in registered_event_roles:
                raise ValueError("event-query Finding role drifted")
        elif (
            role == _FORBIDDEN_INTRINSIC_ROLE
            or signal_context == _FORBIDDEN_SIGNAL_CONTEXT
        ):
            raise ValueError("unregistered record-context Finding entered Event Card")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("event card projection v2 duplicates a Finding")

    closure = candidate["closure"]
    accounting = candidate["finding_scope_accounting"]
    observed_roster = len(roster_keys)
    observed_queries = len(query_ids)
    observed_findings = len(evidence_ids)
    if (
        closure["observed_slot_count"] != len(candidate["slots"])
        or closure["materialized_event_roster_item_count"] != observed_roster
        or closure["materialized_event_query_count"] != observed_queries
        or closure["materialized_event_finding_count"] != observed_findings
        or accounting["materialized_event_count"] != observed_findings
        or closure["source_finding_count"] != accounting["source_count"]
        or closure["excluded_record_context_finding_count"]
        != accounting["excluded_record_context_count"]
        or accounting["source_count"]
        != accounting["materialized_event_count"]
        + accounting["excluded_record_context_count"]
        or closure["source_roster_item_count"]
        != closure["materialized_event_roster_item_count"]
        + closure["excluded_record_context_roster_item_count"]
        or closure["source_query_count"]
        != closure["materialized_event_query_count"]
        + closure["excluded_record_context_query_count"]
    ):
        raise ValueError("event card projection v2 closure counts drifted")
    return candidate


def materialize_event_card_projection_v2(
    *,
    event_findings_v3: object,
    registry_closure_receipt_v1: object,
    record_context_card_id: str,
    trusted_registry_closure_receipt_sha256: str | None = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Build an event-only card projection from validated, source-bound inputs."""

    context_card_id = _require_record_context_card_id(record_context_card_id)
    event_source, closure, atom_policy, query_policy = _validated_sources(
        event_findings_v3=event_findings_v3,
        registry_closure_receipt_v1=registry_closure_receipt_v1,
        trusted_registry_closure_receipt_sha256=(
            trusted_registry_closure_receipt_sha256
        ),
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    projection = _build_projection(
        event_source=event_source,
        closure=closure,
        atom_policy=atom_policy,
        query_policy=query_policy,
        record_context_card_id=context_card_id,
    )
    return _validate_projection_structure(projection)


def validate_event_card_projection_v2(
    value: object,
    *,
    source_event_findings_v3: object,
    source_registry_closure_receipt_v1: object,
    record_context_card_id: str,
    trusted_projection_sha256: str | None = None,
    trusted_registry_closure_receipt_sha256: str | None = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Replay the complete projection and reject self-rehashed source drift."""

    candidate = _validate_projection_structure(value)
    if (
        trusted_projection_sha256 is not None
        and candidate["projection_sha256"] != trusted_projection_sha256
    ):
        raise ValueError("event card projection v2 is not host trusted")
    expected = materialize_event_card_projection_v2(
        event_findings_v3=source_event_findings_v3,
        registry_closure_receipt_v1=source_registry_closure_receipt_v1,
        record_context_card_id=record_context_card_id,
        trusted_registry_closure_receipt_sha256=(
            trusted_registry_closure_receipt_sha256
        ),
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    if candidate != expected:
        raise ValueError(
            "event card projection v2 does not replay exactly from its sources"
        )
    return deepcopy(expected)


__all__ = [
    "EVENT_CARD_PROJECTION_SCHEMA_PATH_V2",
    "EVENT_CARD_PROJECTION_SCHEMA_VERSION_V2",
    "EVENT_CARD_PROJECTOR_ID_V2",
    "EVENT_CARD_SCOPE_FILTER_POLICY_ID_V2",
    "materialize_event_card_projection_v2",
    "validate_event_card_projection_v2",
]
