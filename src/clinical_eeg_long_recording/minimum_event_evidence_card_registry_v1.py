"""Candidate-blind twelve-slot Minimum Event Evidence Card registry.

The existing event-Findings contracts close two useful denominators: 28 core
atoms plus 12 repeatable child rosters, and 41 operational term queries.  They
do not, by themselves, prove that those objects populate the twelve clinical
review slots of the Minimum Event Evidence Card.  This additive module freezes
that semantic partition and emits a replayable closure receipt.

Only checked, host-trusted policy documents are inputs.  No event Findings
payload, candidate, model output, annotation, spreadsheet, private label, or
report text is accepted.  Every source object is retained exactly once,
including ``unimplemented_not_evaluable`` objects.  The registry is a
classification/accounting layer only: it cannot change a status or assertion
level, authorize a clinical absence, promote report text, establish clinical
correctness, or authorize an SOZ/EZ claim.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .event_findings_atom_roster import (
    load_event_findings_atom_roster_policy,
    validate_event_findings_atom_roster_policy,
)
from .event_findings_denominator import (
    load_event_findings_denominator_policy,
    validate_event_findings_denominator_policy,
)
from .event_findings_term_query_denominator_v2 import (
    load_event_findings_term_query_denominator_policy_v2,
    validate_event_findings_term_query_denominator_policy_v2,
)


MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SCHEMA_VERSION_V1 = (
    "clinical_eeg_minimum_event_evidence_card_registry_v1"
)
MINIMUM_EVENT_EVIDENCE_CARD_CLOSURE_RECEIPT_SCHEMA_VERSION_V1 = (
    "clinical_eeg_minimum_event_evidence_card_closure_receipt_v1"
)
MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_ID_V1 = (
    "CLINICAL-EEG-MINIMUM-EVENT-EVIDENCE-CARD-REGISTRY-V1"
)
MINIMUM_EVENT_EVIDENCE_CARD_CLOSURE_METHOD_ID_V1 = (
    "CANDIDATE-BLIND-MINIMUM-EVENT-EVIDENCE-CARD-CLOSURE-V1"
)
DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SHA256_V1 = (
    "ddc07f65e471b619848d40728cf7785d4b10447c9318127b4f8006fda1312146"
)
_EXPECTED_SLOT_MAPPING_SHA256_V1 = (
    "19f434ff04ed2e999ad89fbfcb7f148634b7d4d3663c5ba3203d01a9d2f6b2ed"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_PATH_V1 = (
    _ROOT / "configs" / "clinical_eeg_minimum_event_evidence_card_registry_v1.json"
)
MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SCHEMA_PATH_V1 = (
    _ROOT
    / "schemas"
    / "clinical_eeg_minimum_event_evidence_card_registry_v1.schema.json"
)
MINIMUM_EVENT_EVIDENCE_CARD_CLOSURE_RECEIPT_SCHEMA_PATH_V1 = (
    _ROOT
    / "schemas"
    / "clinical_eeg_minimum_event_evidence_card_closure_receipt_v1.schema.json"
)

_EXPECTED_SLOT_IDS = (
    "S01_SOURCE_EVALUABILITY",
    "S02_EVENT_BOUNDARY",
    "S03_FREQUENCY",
    "S04_PHYSICAL_AMPLITUDE",
    "S05_WAVEFORM_MORPHOLOGY",
    "S06_RHYTHMICITY_PERIODICITY",
    "S07_EARLIEST_VISIBLE_SET",
    "S08_SPATIAL_FIELD_REFERENCE_STABILITY",
    "S09_CHANGE_POINTS_EVOLUTION",
    "S10_LATER_INVOLVEMENT",
    "S11_TERMINATION_POST_EVENT",
    "S12_COMPARABLE_BACKGROUND_RECOVERY",
)
_SOURCE_FIREWALL = {
    "private_data_used": False,
    "event_findings_payload_used": False,
    "findings_candidates_used": False,
    "payload_evaluation_opportunities_used": False,
    "event_outcome_used": False,
    "scalp_onset_hypothesis_used": False,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "qwen_used": False,
}
_AUTHORIZATION = {
    "mapping_is_semantic_partition_only": True,
    "status_or_assertion_mutation_authorized": False,
    "clinical_absence_authorized": False,
    "report_promotion_authorized": False,
    "clinical_correctness_claimed": False,
    "soz_or_ez_claim_authorized": False,
}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


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


def _schema_errors(value: object, path: Path) -> list[str]:
    validator = Draft202012Validator(_read_json(path))
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    rendered: list[str] = []
    for error in errors[:12]:
        pointer = "/" + "/".join(str(part) for part in error.path)
        rendered.append(f"{pointer}: {error.message}")
    if len(errors) > 12:
        rendered.append(f"... {len(errors) - 12} more error(s)")
    return rendered


def _require_sha256(value: object, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _require_canonical_ids(value: object, context: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be an ID array")
    result: list[str] = []
    for raw in value:
        if type(raw) is not str or _ID_RE.fullmatch(raw) is None:
            raise ValueError(f"{context} contains a non-canonical ID")
        result.append(raw)
    if result != sorted(result):
        raise ValueError(f"{context} must be sorted")
    if len(result) != len(set(result)):
        raise ValueError(f"{context} must be unique")
    return result


def _resolve_source_contracts(
    *,
    atom_roster_policy: Mapping[str, object] | None,
    trusted_atom_roster_policy_sha256: str | None,
    item_denominator_policy: Mapping[str, object] | None,
    trusted_item_denominator_policy_sha256: str | None,
    term_query_denominator_policy: Mapping[str, object] | None,
    trusted_term_query_denominator_policy_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    default_atom = load_event_findings_atom_roster_policy()
    if atom_roster_policy is None:
        atom = default_atom
    else:
        atom_trust = trusted_atom_roster_policy_sha256
        if atom_trust is None:
            atom_trust = str(default_atom["policy_sha256"])
        atom = validate_event_findings_atom_roster_policy(
            dict(atom_roster_policy),
            trusted_policy_sha256=atom_trust,
        )

    default_item = load_event_findings_denominator_policy()
    if item_denominator_policy is None:
        item = default_item
    else:
        item_trust = trusted_item_denominator_policy_sha256
        if item_trust is None:
            item_trust = str(default_item["policy_sha256"])
        item = validate_event_findings_denominator_policy(
            dict(item_denominator_policy),
            trusted_policy_sha256=item_trust,
            atom_roster_policy=atom,
            trusted_atom_roster_policy_sha256=str(atom["policy_sha256"]),
        )

    default_query = load_event_findings_term_query_denominator_policy_v2()
    if term_query_denominator_policy is None:
        query = default_query
    else:
        query_trust = trusted_term_query_denominator_policy_sha256
        if query_trust is None:
            query_trust = str(default_query["policy_sha256"])
        query = validate_event_findings_term_query_denominator_policy_v2(
            dict(term_query_denominator_policy),
            trusted_policy_sha256=query_trust,
            atom_roster_policy=atom,
            trusted_atom_roster_policy_sha256=str(atom["policy_sha256"]),
            v1_denominator_policy=item,
            trusted_v1_denominator_policy_sha256=str(item["policy_sha256"]),
        )

    # The checked term-query policy already validates these joins.  Keep the
    # explicit checks here so a future source-contract version cannot silently
    # weaken the twelve-slot trust root.
    if item["atom_roster_policy_sha256"] != atom["policy_sha256"]:
        raise ValueError("item denominator is not bound to the atom roster")
    if (
        query["atom_roster_policy_sha256"] != atom["policy_sha256"]
        or query["v1_denominator_policy_sha256"] != item["policy_sha256"]
    ):
        raise ValueError("term-query denominator source-contract join drifted")
    return atom, item, query


def _source_contract_binding(
    atom: Mapping[str, object],
    item: Mapping[str, object],
    query: Mapping[str, object],
) -> dict[str, str]:
    return {
        "atom_roster_id": str(atom["roster_id"]),
        "atom_roster_policy_sha256": str(atom["policy_sha256"]),
        "item_denominator_policy_id": str(item["policy_id"]),
        "item_denominator_policy_sha256": str(item["policy_sha256"]),
        "term_query_denominator_policy_id": str(query["policy_id"]),
        "term_query_denominator_policy_sha256": str(query["policy_sha256"]),
    }


def _require_exact_partition(
    values: Sequence[str],
    expected: set[str],
    context: str,
) -> None:
    counts = Counter(values)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(expected - set(values))
    extra = sorted(set(values) - expected)
    if duplicates or missing or extra or len(values) != len(expected):
        raise ValueError(
            f"{context} is not an exact once-only partition: "
            f"duplicates={duplicates}, missing={missing}, extra={extra}"
        )


def _mapping_projection(
    slots: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    return [
        {
            "slot_index": row["slot_index"],
            "slot_id": row["slot_id"],
            "core_atom_ids": list(row["core_atom_ids"]),
            "child_roster_ids": list(row["child_roster_ids"]),
            "operational_query_ids": list(row["operational_query_ids"]),
        }
        for row in slots
    ]


def _validate_registry_with_sources(
    value: object,
    *,
    trusted_registry_sha256: str | None,
    atom_roster_policy: Mapping[str, object] | None,
    trusted_atom_roster_policy_sha256: str | None,
    item_denominator_policy: Mapping[str, object] | None,
    trusted_item_denominator_policy_sha256: str | None,
    term_query_denominator_policy: Mapping[str, object] | None,
    trusted_term_query_denominator_policy_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if type(value) is not dict:
        raise TypeError("Minimum Event Evidence Card registry must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(
        candidate,
        MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SCHEMA_PATH_V1,
    )
    if errors:
        raise ValueError(
            "Minimum Event Evidence Card registry schema validation failed: "
            + "; ".join(errors)
        )
    expected_hash = _self_hash(candidate, "registry_sha256")
    if candidate["registry_sha256"] != expected_hash:
        raise ValueError("Minimum Event Evidence Card registry SHA-256 mismatch")
    if trusted_registry_sha256 is not None and expected_hash != _require_sha256(
        trusted_registry_sha256, "trusted_registry_sha256"
    ):
        raise ValueError("Minimum Event Evidence Card registry is not host trusted")

    atom, item, query = _resolve_source_contracts(
        atom_roster_policy=atom_roster_policy,
        trusted_atom_roster_policy_sha256=trusted_atom_roster_policy_sha256,
        item_denominator_policy=item_denominator_policy,
        trusted_item_denominator_policy_sha256=(trusted_item_denominator_policy_sha256),
        term_query_denominator_policy=term_query_denominator_policy,
        trusted_term_query_denominator_policy_sha256=(
            trusted_term_query_denominator_policy_sha256
        ),
    )
    if candidate["source_contracts"] != _source_contract_binding(atom, item, query):
        raise ValueError("Minimum Event Evidence Card source bindings drifted")
    if candidate["source_firewall"] != _SOURCE_FIREWALL:
        raise ValueError("Minimum Event Evidence Card candidate-blind firewall drifted")
    if candidate["authorization"] != _AUTHORIZATION:
        raise ValueError(
            "Minimum Event Evidence Card no-promotion authorization drifted"
        )

    slots = list(candidate["slots"])
    slot_ids = tuple(str(row["slot_id"]) for row in slots)
    slot_indices = tuple(int(row["slot_index"]) for row in slots)
    if tuple(candidate["slot_order"]) != _EXPECTED_SLOT_IDS:
        raise ValueError("Minimum Event Evidence Card slot order drifted")
    if slot_ids != _EXPECTED_SLOT_IDS or slot_indices != tuple(range(1, 13)):
        raise ValueError("Minimum Event Evidence Card must contain ordered slots 1--12")

    flattened_core: list[str] = []
    flattened_child: list[str] = []
    flattened_queries: list[str] = []
    item_to_slot: dict[tuple[str, str], str] = {}
    query_to_slot: dict[str, str] = {}
    for row in slots:
        slot_id = str(row["slot_id"])
        core_ids = _require_canonical_ids(
            row["core_atom_ids"], f"{slot_id}.core_atom_ids"
        )
        child_ids = _require_canonical_ids(
            row["child_roster_ids"], f"{slot_id}.child_roster_ids"
        )
        query_ids = _require_canonical_ids(
            row["operational_query_ids"],
            f"{slot_id}.operational_query_ids",
        )
        flattened_core.extend(core_ids)
        flattened_child.extend(child_ids)
        flattened_queries.extend(query_ids)
        for roster_id in core_ids:
            item_to_slot[("core_atom", roster_id)] = slot_id
        for roster_id in child_ids:
            item_to_slot[("child_roster", roster_id)] = slot_id
        for query_id in query_ids:
            query_to_slot[query_id] = slot_id

    source_core = {str(row["atom_id"]) for row in atom["core_atom_specs"]}
    source_child = {str(row["child_roster_id"]) for row in atom["child_roster_specs"]}
    source_queries = {str(row["term_query_id"]) for row in query["query_specs"]}
    _require_exact_partition(flattened_core, source_core, "core atom mapping")
    _require_exact_partition(flattened_child, source_child, "child roster mapping")
    _require_exact_partition(
        flattened_queries,
        source_queries,
        "operational query mapping",
    )

    source_item_keys = {
        (str(row["roster_item_kind"]), str(row["roster_item_id"]))
        for row in item["item_scopes"]
    }
    if source_item_keys != set(item_to_slot):
        raise ValueError(
            "twelve-slot mapping and candidate-blind item denominator differ"
        )

    query_by_id = {str(row["term_query_id"]): row for row in query["query_specs"]}
    for query_id, slot_id in query_to_slot.items():
        source_query = query_by_id[query_id]
        primary = source_query["primary_roster_item"]
        primary_key = (
            str(primary["roster_item_kind"]),
            str(primary["roster_item_id"]),
        )
        if item_to_slot.get(primary_key) != slot_id:
            raise ValueError(
                f"{query_id} is not in the same slot as its primary roster item"
            )
        if source_query["report_promotion_authorized"] is not False:
            raise ValueError(f"{query_id} unexpectedly authorizes report promotion")

    mapping_hash = _canonical_sha256(_mapping_projection(slots))
    if mapping_hash != _EXPECTED_SLOT_MAPPING_SHA256_V1:
        raise ValueError("Minimum Event Evidence Card frozen slot mapping drifted")
    return candidate, atom, item, query


def validate_minimum_event_evidence_card_registry_v1(
    value: object,
    *,
    trusted_registry_sha256: str | None = None,
    atom_roster_policy: Mapping[str, object] | None = None,
    trusted_atom_roster_policy_sha256: str | None = None,
    item_denominator_policy: Mapping[str, object] | None = None,
    trusted_item_denominator_policy_sha256: str | None = None,
    term_query_denominator_policy: Mapping[str, object] | None = None,
    trusted_term_query_denominator_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the static twelve-slot partition against all source contracts."""

    candidate, _, _, _ = _validate_registry_with_sources(
        value,
        trusted_registry_sha256=trusted_registry_sha256,
        atom_roster_policy=atom_roster_policy,
        trusted_atom_roster_policy_sha256=trusted_atom_roster_policy_sha256,
        item_denominator_policy=item_denominator_policy,
        trusted_item_denominator_policy_sha256=(trusted_item_denominator_policy_sha256),
        term_query_denominator_policy=term_query_denominator_policy,
        trusted_term_query_denominator_policy_sha256=(
            trusted_term_query_denominator_policy_sha256
        ),
    )
    return candidate


def load_minimum_event_evidence_card_registry_v1(
    path: str | Path = DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_PATH_V1,
    *,
    trusted_registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the checked-in registry under the default host trust anchor."""

    path_value = Path(path)
    if trusted_registry_sha256 is None:
        trusted_registry_sha256 = DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SHA256_V1
    return validate_minimum_event_evidence_card_registry_v1(
        _read_json(path_value),
        trusted_registry_sha256=trusted_registry_sha256,
    )


def _registry_bundle(
    registry: Mapping[str, object] | None,
    *,
    trusted_registry_sha256: str | None,
    atom_roster_policy: Mapping[str, object] | None,
    trusted_atom_roster_policy_sha256: str | None,
    item_denominator_policy: Mapping[str, object] | None,
    trusted_item_denominator_policy_sha256: str | None,
    term_query_denominator_policy: Mapping[str, object] | None,
    trusted_term_query_denominator_policy_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if registry is None:
        registry_value = _read_json(
            DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_PATH_V1
        )
        if trusted_registry_sha256 is None:
            trusted_registry_sha256 = (
                DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SHA256_V1
            )
    else:
        registry_value = dict(registry)
        if trusted_registry_sha256 is None:
            trusted_registry_sha256 = (
                DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SHA256_V1
            )
    return _validate_registry_with_sources(
        registry_value,
        trusted_registry_sha256=trusted_registry_sha256,
        atom_roster_policy=atom_roster_policy,
        trusted_atom_roster_policy_sha256=trusted_atom_roster_policy_sha256,
        item_denominator_policy=item_denominator_policy,
        trusted_item_denominator_policy_sha256=(trusted_item_denominator_policy_sha256),
        term_query_denominator_policy=term_query_denominator_policy,
        trusted_term_query_denominator_policy_sha256=(
            trusted_term_query_denominator_policy_sha256
        ),
    )


def _closure_receipt_body(
    registry: Mapping[str, object],
    atom: Mapping[str, object],
    item: Mapping[str, object],
    query: Mapping[str, object],
) -> dict[str, Any]:
    del item  # Its exact 40-item key set was checked during registry validation.
    core_by_id = {str(row["atom_id"]): row for row in atom["core_atom_specs"]}
    child_by_id = {
        str(row["child_roster_id"]): row for row in atom["child_roster_specs"]
    }
    query_by_id = {str(row["term_query_id"]): row for row in query["query_specs"]}
    slots: list[dict[str, object]] = []
    roster_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    for slot in registry["slots"]:
        current_roster: list[dict[str, object]] = []
        for roster_id in slot["core_atom_ids"]:
            source = core_by_id[str(roster_id)]
            current_roster.append(
                {
                    "roster_item_kind": "core_atom",
                    "roster_item_id": str(roster_id),
                    "source_group": str(source["group"]),
                    "source_implementation_status": str(
                        source["current_implementation_status"]
                    ),
                }
            )
        for roster_id in slot["child_roster_ids"]:
            source = child_by_id[str(roster_id)]
            current_roster.append(
                {
                    "roster_item_kind": "child_roster",
                    "roster_item_id": str(roster_id),
                    "source_group": str(source["group"]),
                    "source_implementation_status": str(
                        source["current_implementation_status"]
                    ),
                }
            )
        current_queries: list[dict[str, object]] = []
        for query_id in slot["operational_query_ids"]:
            source = query_by_id[str(query_id)]
            primary = source["primary_roster_item"]
            current_queries.append(
                {
                    "term_query_id": str(query_id),
                    "primary_roster_item_kind": str(primary["roster_item_kind"]),
                    "primary_roster_item_id": str(primary["roster_item_id"]),
                    "source_implementation_status": str(
                        source["implementation_status"]
                    ),
                    "source_report_promotion_authorized": bool(
                        source["report_promotion_authorized"]
                    ),
                }
            )
        roster_rows.extend(current_roster)
        query_rows.extend(current_queries)
        slots.append(
            {
                "slot_index": int(slot["slot_index"]),
                "slot_id": str(slot["slot_id"]),
                "roster_items": current_roster,
                "operational_queries": current_queries,
            }
        )

    unimplemented_roster = sum(
        row["source_implementation_status"] == "unimplemented_not_evaluable"
        for row in roster_rows
    )
    not_evaluable_queries = sum(
        row["source_implementation_status"] == "unimplemented_not_evaluable"
        for row in query_rows
    )
    promoted_queries = sum(
        row["source_report_promotion_authorized"] is not False for row in query_rows
    )
    return {
        "schema_version": (
            MINIMUM_EVENT_EVIDENCE_CARD_CLOSURE_RECEIPT_SCHEMA_VERSION_V1
        ),
        "method_id": MINIMUM_EVENT_EVIDENCE_CARD_CLOSURE_METHOD_ID_V1,
        "registry_id": str(registry["registry_id"]),
        "registry_sha256": str(registry["registry_sha256"]),
        "source_contracts": deepcopy(registry["source_contracts"]),
        "source_firewall": deepcopy(_SOURCE_FIREWALL),
        "candidate_blind_denominator": True,
        "slots": slots,
        "summary": {
            "slot_count": len(slots),
            "core_atom_count": sum(
                row["roster_item_kind"] == "core_atom" for row in roster_rows
            ),
            "child_roster_count": sum(
                row["roster_item_kind"] == "child_roster" for row in roster_rows
            ),
            "roster_item_count": len(roster_rows),
            "operational_query_count": len(query_rows),
            "retained_unimplemented_roster_item_count": unimplemented_roster,
            "retained_not_evaluable_query_count": not_evaluable_queries,
            "report_promoted_query_count": promoted_queries,
            "all_source_roster_items_mapped_exactly_once": True,
            "all_source_queries_mapped_exactly_once": True,
            "all_queries_share_slot_with_primary_roster_item": True,
            "unimplemented_or_not_evaluable_items_retained": True,
            "source_statuses_preserved": True,
        },
        "authorization": deepcopy(_AUTHORIZATION),
    }


def materialize_minimum_event_evidence_card_closure_receipt_v1(
    *,
    registry: Mapping[str, object] | None = None,
    trusted_registry_sha256: str | None = None,
    atom_roster_policy: Mapping[str, object] | None = None,
    trusted_atom_roster_policy_sha256: str | None = None,
    item_denominator_policy: Mapping[str, object] | None = None,
    trusted_item_denominator_policy_sha256: str | None = None,
    term_query_denominator_policy: Mapping[str, object] | None = None,
    trusted_term_query_denominator_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Materialize a deterministic closure receipt without event candidates."""

    registry_value, atom, item, query = _registry_bundle(
        registry,
        trusted_registry_sha256=trusted_registry_sha256,
        atom_roster_policy=atom_roster_policy,
        trusted_atom_roster_policy_sha256=trusted_atom_roster_policy_sha256,
        item_denominator_policy=item_denominator_policy,
        trusted_item_denominator_policy_sha256=(trusted_item_denominator_policy_sha256),
        term_query_denominator_policy=term_query_denominator_policy,
        trusted_term_query_denominator_policy_sha256=(
            trusted_term_query_denominator_policy_sha256
        ),
    )
    receipt = _closure_receipt_body(registry_value, atom, item, query)
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return validate_minimum_event_evidence_card_closure_receipt_v1(
        receipt,
        registry=registry_value,
        trusted_registry_sha256=str(registry_value["registry_sha256"]),
        atom_roster_policy=atom,
        trusted_atom_roster_policy_sha256=str(atom["policy_sha256"]),
        item_denominator_policy=item,
        trusted_item_denominator_policy_sha256=str(item["policy_sha256"]),
        term_query_denominator_policy=query,
        trusted_term_query_denominator_policy_sha256=str(query["policy_sha256"]),
    )


def validate_minimum_event_evidence_card_closure_receipt_v1(
    value: object,
    *,
    registry: Mapping[str, object] | None = None,
    trusted_registry_sha256: str | None = None,
    atom_roster_policy: Mapping[str, object] | None = None,
    trusted_atom_roster_policy_sha256: str | None = None,
    item_denominator_policy: Mapping[str, object] | None = None,
    trusted_item_denominator_policy_sha256: str | None = None,
    term_query_denominator_policy: Mapping[str, object] | None = None,
    trusted_term_query_denominator_policy_sha256: str | None = None,
    trusted_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay source contracts and fail closed on any receipt mutation."""

    if type(value) is not dict:
        raise TypeError("Minimum Event Evidence Card closure receipt must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(
        candidate,
        MINIMUM_EVENT_EVIDENCE_CARD_CLOSURE_RECEIPT_SCHEMA_PATH_V1,
    )
    if errors:
        raise ValueError(
            "Minimum Event Evidence Card closure receipt schema validation failed: "
            + "; ".join(errors)
        )
    expected_hash = _self_hash(candidate, "receipt_sha256")
    if candidate["receipt_sha256"] != expected_hash:
        raise ValueError("Minimum Event Evidence Card closure receipt SHA-256 mismatch")
    if trusted_receipt_sha256 is not None and expected_hash != _require_sha256(
        trusted_receipt_sha256, "trusted_receipt_sha256"
    ):
        raise ValueError("Minimum Event Evidence Card closure receipt is not trusted")

    registry_value, atom, item, query = _registry_bundle(
        registry,
        trusted_registry_sha256=trusted_registry_sha256,
        atom_roster_policy=atom_roster_policy,
        trusted_atom_roster_policy_sha256=trusted_atom_roster_policy_sha256,
        item_denominator_policy=item_denominator_policy,
        trusted_item_denominator_policy_sha256=(trusted_item_denominator_policy_sha256),
        term_query_denominator_policy=term_query_denominator_policy,
        trusted_term_query_denominator_policy_sha256=(
            trusted_term_query_denominator_policy_sha256
        ),
    )
    expected = _closure_receipt_body(registry_value, atom, item, query)
    expected["receipt_sha256"] = _self_hash(expected, "receipt_sha256")
    if candidate != expected:
        raise ValueError(
            "Minimum Event Evidence Card closure receipt does not exactly replay"
        )
    return candidate
