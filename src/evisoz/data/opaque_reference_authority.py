"""Protocol authority for suffix-free private Standard19/A1/A2 recordings.

This module does not claim that a shared acquisition reference is observable
from the EDF labels.  It qualifies a separate, explicit protocol route that
replays the already frozen ``unlabeled_common_car19`` private v29 lineage.
Every event authorization is self-contained, binds the source EDF and the
processed parent tensor, and keeps the header-exact and protocol-authorized
routes distinguishable downstream.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.clinical_eeg_long_recording.montage_reference_observability import (
    MONTAGE_REFERENCE_OBSERVABILITY_SCHEMA_VERSION,
    validate_montage_reference_observability_receipt,
)
from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
)
from src.evisoz.data.real_stage0_reference_audit import (
    PARENT_ELECTRODES,
    REAL_STAGE0_REFERENCE_AUDIT_SCHEMA_VERSION,
)
from src.soz.geometry import STANDARD_19, normalize_electrode_name


OPAQUE_REFERENCE_AUTHORITY_SCHEMA_VERSION = (
    "evisoz_private_opaque_common_reference_authority_v1"
)
OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION = (
    "evisoz_opaque_common_reference_event_authorization_v1"
)
OPAQUE_REFERENCE_ROUTE_ID = "private_suffix_free_standard19_a1_a2_opaque_common_v1"
_AUTHORITY_PREFIX = "EVISOZ-OPAQUE-REF-AUTH-"
_EVENT_PREFIX = "EVISOZ-OPAQUE-REF-EVENT-"
_PENDING = "CONTENT-ADDRESS-PENDING"
_PLACEHOLDER = "0" * 64


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_binding(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be a lowercase SHA256")
    return str(value)


def _id_source(value: Mapping[str, object], key: str) -> dict[str, object]:
    body = deepcopy(dict(value))
    body[key] = _PENDING
    body["receipt_sha256"] = _PLACEHOLDER
    return body


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _PLACEHOLDER
    return body


def _validate_audit(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("reference audit must be an object")
    audit = deepcopy(value)
    if audit.get("schema_version") != REAL_STAGE0_REFERENCE_AUDIT_SCHEMA_VERSION:
        raise ValueError("private reference audit schema mismatch")
    observed_hash = audit.get("receipt_sha256")
    replay = deepcopy(audit)
    replay["receipt_sha256"] = _PLACEHOLDER
    if observed_hash != canonical_json_sha256(replay):
        raise ValueError("private reference audit receipt hash drifted")
    count = audit.get("unique_edf_count")
    roster = audit.get("source_edf_sha256_roster")
    aggregate = audit.get("aggregate")
    if (
        type(count) is not int
        or count < 1
        or not isinstance(roster, list)
        or len(roster) != count
        or roster != sorted(set(roster))
        or any(not _is_sha256(item) for item in roster)
        or type(aggregate) is not dict
    ):
        raise ValueError("private reference audit roster is invalid")
    required_equal = {
        "complete_standard19_edf_count": count,
        "opaque_common_reference_candidate_edf_count": count,
        "explicit_common_reference_candidate_edf_count": 0,
        "mixed_selected_sampling_clock_edf_count": 0,
    }
    if any(aggregate.get(key) != expected for key, expected in required_equal.items()):
        raise ValueError("private reference audit does not qualify the opaque route")
    if (
        aggregate.get("montage_class_counts") != {"unknown": count}
        or aggregate.get("classification_reason_code_counts")
        != {"direct_electrode_reference_token_unobservable": count}
        or aggregate.get("standard19_observed_count_distribution") != {"19": count}
        or aggregate.get("parent_electrode_observed_count_distribution") != {"21": count}
        or aggregate.get("auxiliary_coverage_counts") != {"both": count}
        or aggregate.get("derivable_tcp22_edge_count_distribution") != {"22": count}
    ):
        raise ValueError("private reference audit geometry is not uniformly qualified")
    if audit.get("interpretation", {}).get(
        "header_audit_alone_authorizes_opaque_common_reference"
    ) is not False:
        raise ValueError("private reference audit overstates header authority")
    return audit


def _historical_lineage(
    evidence: Mapping[str, object],
    v29: Mapping[str, object],
    *,
    evidence_file_sha256: str,
    v29_file_sha256: str,
) -> dict[str, object]:
    if evidence.get("schema_version") != "soz_private_target_blind_labram_evidence_v18":
        raise ValueError("historical private evidence manifest schema mismatch")
    preprocessing = evidence.get("preprocessing")
    access = evidence.get("access_receipt")
    evidence_events = evidence.get("events")
    if (
        type(preprocessing) is not dict
        or preprocessing.get("reference_policy") != "unlabeled_common_car19"
        or preprocessing.get("common_reference_assumption_proven_by_header") is not False
        or preprocessing.get("car19_required") is not True
        or type(access) is not dict
        or access.get("target_ledger_opened") is not False
        or not isinstance(evidence_events, list)
        or not evidence_events
    ):
        raise ValueError("historical private evidence reference contract drifted")
    if any(
        row.get("source_reference_policy") != "unlabeled_common_car19"
        or row.get("output_reference") != "common_average_standard19"
        for row in evidence_events
    ):
        raise ValueError("historical private evidence contains a mixed reference route")

    if (
        v29.get("schema_version")
        != "soz_private_target_blind_labram_portable_equal_v29"
        or v29.get("status") != "completed_frozen_target_blind_private_inference"
    ):
        raise ValueError("historical private v29 manifest schema/status mismatch")
    v29_events = v29.get("events")
    v29_access = v29.get("access_receipt")
    if (
        not isinstance(v29_events, list)
        or len(v29_events) != len(evidence_events)
        or type(v29_access) is not dict
        or v29_access.get("private_target_values_loaded") is not False
        or any(
            row.get("source_reference_policy") != "unlabeled_common_car19"
            or row.get("output_reference") != "common_average_standard19"
            for row in v29_events
        )
    ):
        raise ValueError("historical private v29 route/access contract drifted")
    if v29.get("channels") != list(STANDARD_19):
        raise ValueError("historical private v29 channel order drifted")
    return {
        "historical_evidence_manifest_sha256": _file_binding(
            evidence_file_sha256, "historical evidence manifest binding"
        ),
        "historical_private_v29_manifest_sha256": _file_binding(
            v29_file_sha256, "historical private v29 manifest binding"
        ),
        "target_blind_event_count": len(v29_events),
        "reference_policy": "unlabeled_common_car19",
        "output_reference": "common_average_standard19",
        "common_reference_assumption_proven_by_header": False,
        "private_targets_loaded_for_reference_authority": False,
    }


def build_private_opaque_reference_authority(
    reference_audit: Mapping[str, object],
    historical_evidence_manifest: Mapping[str, object],
    historical_private_v29_manifest: Mapping[str, object],
    *,
    evidence_file_sha256: str,
    v29_file_sha256: str,
) -> dict[str, Any]:
    """Build a cohort authority from the audited roster and frozen lineage."""

    audit = _validate_audit(reference_audit)
    lineage = _historical_lineage(
        historical_evidence_manifest,
        historical_private_v29_manifest,
        evidence_file_sha256=evidence_file_sha256,
        v29_file_sha256=v29_file_sha256,
    )
    body: dict[str, Any] = {
        "schema_version": OPAQUE_REFERENCE_AUTHORITY_SCHEMA_VERSION,
        "authority_id": _PENDING,
        "status": "protocol_authorized_opaque_common_reference",
        "dataset_id": "private",
        "route_id": OPAQUE_REFERENCE_ROUTE_ID,
        "source_inventory_binding": {
            "reference_audit_receipt_sha256": audit["receipt_sha256"],
            "source_manifest_sha256": audit["source_manifest_sha256"],
            "source_edf_sha256_roster_sha256": audit[
                "source_edf_sha256_roster_sha256"
            ],
            "authorized_source_edf_sha256": audit["source_edf_sha256_roster"],
            "authorized_source_edf_count": audit["unique_edf_count"],
        },
        "historical_v29_lineage": lineage,
        "authority_decision": {
            "decision_id": "EVISOZ-PRIVATE-OPAQUE-COMMON-ROUTE-20260831",
            "scope": (
                "replay frozen CAR19 and derive signed endpoint differences from the same suffix-free direct field"
            ),
            "authority_level": "protocol_and_frozen_lineage_not_header_exact",
        },
        "permissions": {
            "car19_replay_authorized": True,
            "signed_tcp22_difference_authorized": True,
            "tcp22_direct_evidence_authorized_when_both_endpoints_observed": True,
            "missing_endpoint_interpolation_as_direct_evidence": False,
            "edf_discontinuity_clock_authorized": False,
        },
        "claim_boundary": {
            "shared_reference_proven_by_edf_labels": False,
            "shared_reference_is_protocol_assumption": True,
            "route_is_header_exact_common_reference": False,
            "tcp22_difference_is_cortical_source_localization": False,
            "result_is_clinical_or_surgical_soz": False,
        },
        "receipt_sha256": _PLACEHOLDER,
    }
    body["authority_id"] = _AUTHORITY_PREFIX + canonical_json_sha256(
        _id_source(body, "authority_id")
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_private_opaque_reference_authority(body)


def validate_private_opaque_reference_authority(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "authority_id",
        "status",
        "dataset_id",
        "route_id",
        "source_inventory_binding",
        "historical_v29_lineage",
        "authority_decision",
        "permissions",
        "claim_boundary",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("opaque reference authority fields drifted")
    data = deepcopy(value)
    if (
        data["schema_version"] != OPAQUE_REFERENCE_AUTHORITY_SCHEMA_VERSION
        or data["status"] != "protocol_authorized_opaque_common_reference"
        or data["dataset_id"] != "private"
        or data["route_id"] != OPAQUE_REFERENCE_ROUTE_ID
    ):
        raise ValueError("opaque reference authority identity/status drifted")
    inventory = data["source_inventory_binding"]
    roster = inventory.get("authorized_source_edf_sha256") if type(inventory) is dict else None
    if (
        not isinstance(roster, list)
        or not roster
        or roster != sorted(set(roster))
        or any(not _is_sha256(item) for item in roster)
        or inventory.get("authorized_source_edf_count") != len(roster)
        or any(
            not _is_sha256(inventory.get(key))
            for key in (
                "reference_audit_receipt_sha256",
                "source_manifest_sha256",
                "source_edf_sha256_roster_sha256",
            )
        )
    ):
        raise ValueError("opaque reference authority source binding is invalid")
    lineage = data["historical_v29_lineage"]
    if (
        type(lineage) is not dict
        or lineage.get("reference_policy") != "unlabeled_common_car19"
        or lineage.get("output_reference") != "common_average_standard19"
        or lineage.get("common_reference_assumption_proven_by_header") is not False
        or lineage.get("private_targets_loaded_for_reference_authority") is not False
        or not _is_sha256(lineage.get("historical_evidence_manifest_sha256"))
        or not _is_sha256(lineage.get("historical_private_v29_manifest_sha256"))
    ):
        raise ValueError("opaque reference historical lineage is invalid")
    if data["permissions"] != {
        "car19_replay_authorized": True,
        "signed_tcp22_difference_authorized": True,
        "tcp22_direct_evidence_authorized_when_both_endpoints_observed": True,
        "missing_endpoint_interpolation_as_direct_evidence": False,
        "edf_discontinuity_clock_authorized": False,
    }:
        raise ValueError("opaque reference permissions drifted")
    if data["claim_boundary"] != {
        "shared_reference_proven_by_edf_labels": False,
        "shared_reference_is_protocol_assumption": True,
        "route_is_header_exact_common_reference": False,
        "tcp22_difference_is_cortical_source_localization": False,
        "result_is_clinical_or_surgical_soz": False,
    }:
        raise ValueError("opaque reference claim boundary drifted")
    expected_id = _AUTHORITY_PREFIX + canonical_json_sha256(
        _id_source(data, "authority_id")
    )[:24]
    if data["authority_id"] != expected_id:
        raise ValueError("opaque reference authority_id drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("opaque reference authority receipt hash drifted")
    return data


def build_opaque_reference_event_authorization(
    authority: Mapping[str, object],
    *,
    source_edf_sha256: str,
    parent_signal_sha256: str,
    montage_reference_observability_receipt: Mapping[str, object],
) -> dict[str, Any]:
    """Authorize one source-bound processed event without persisting raw labels."""

    trusted = validate_private_opaque_reference_authority(authority)
    source_hash = _file_binding(source_edf_sha256, "source EDF binding")
    parent_hash = _file_binding(parent_signal_sha256, "parent signal binding")
    if source_hash not in trusted["source_inventory_binding"][
        "authorized_source_edf_sha256"
    ]:
        raise ValueError("source EDF is outside the opaque reference authority roster")
    observed = validate_montage_reference_observability_receipt(
        montage_reference_observability_receipt
    )
    if (
        observed["source_signal_sha256"] != parent_hash
        or observed["montage_class"] != "unknown"
        or observed["classification_reason_codes"]
        != ["direct_electrode_reference_token_unobservable"]
        or observed["direct_electrode_ids"] != list(STANDARD_19)
        or observed["duplicate_direct_electrode_ids"]
        or observed["common_reference_compatibility"]["compatible"] is not False
    ):
        raise ValueError("event signal-label receipt is not the authorized opaque profile")
    parent_rows: dict[str, int] = {}
    for row in observed["signal_label_observations"]:
        normalized = normalize_electrode_name(row["raw_label"])
        if normalized not in PARENT_ELECTRODES:
            continue
        if row["reference_token"] is not None or normalized in parent_rows:
            raise ValueError("opaque event parent endpoints are referenced or duplicated")
        parent_rows[normalized] = int(row["signal_index"])
    if list(name for name in PARENT_ELECTRODES if name in parent_rows) != list(
        PARENT_ELECTRODES
    ):
        raise ValueError("opaque event requires complete Standard19+A1+A2 endpoints")

    body: dict[str, Any] = {
        "schema_version": OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION,
        "authorization_id": _PENDING,
        "route_id": OPAQUE_REFERENCE_ROUTE_ID,
        "authority_receipt": trusted,
        "authority_ref": build_json_artifact_ref(
            trusted,
            artifact_kind="reference_authority",
            payload_schema_version=OPAQUE_REFERENCE_AUTHORITY_SCHEMA_VERSION,
        ),
        "source_edf_sha256": source_hash,
        "parent_signal_sha256": parent_hash,
        "signal_labels_sha256": observed["signal_labels_sha256"],
        "observed_parent_electrodes": list(PARENT_ELECTRODES),
        "reference_semantics": (
            "shared_opaque_reference_protocol_authorized_not_header_proven"
        ),
        "permissions": {
            "car19_replay_authorized": True,
            "signed_tcp22_difference_authorized": True,
            "tcp22_direct_evidence_authorized": True,
            "interpolation_used": False,
        },
        "claim_boundary": deepcopy(trusted["claim_boundary"]),
        "receipt_sha256": _PLACEHOLDER,
    }
    body["authorization_id"] = _EVENT_PREFIX + canonical_json_sha256(
        _id_source(body, "authorization_id")
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_opaque_reference_event_authorization(
        body,
        expected_parent_signal_sha256=parent_hash,
    )


def validate_opaque_reference_event_authorization(
    value: object,
    *,
    expected_parent_signal_sha256: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "authorization_id",
        "route_id",
        "authority_receipt",
        "authority_ref",
        "source_edf_sha256",
        "parent_signal_sha256",
        "signal_labels_sha256",
        "observed_parent_electrodes",
        "reference_semantics",
        "permissions",
        "claim_boundary",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("opaque event reference authorization fields drifted")
    data = deepcopy(value)
    authority = validate_private_opaque_reference_authority(data["authority_receipt"])
    expected_ref = build_json_artifact_ref(
        authority,
        artifact_kind="reference_authority",
        payload_schema_version=OPAQUE_REFERENCE_AUTHORITY_SCHEMA_VERSION,
    )
    if data["authority_ref"] != expected_ref:
        raise ValueError("opaque event authority binding drifted")
    if (
        data["schema_version"]
        != OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION
        or data["route_id"] != OPAQUE_REFERENCE_ROUTE_ID
        or data["source_edf_sha256"]
        not in authority["source_inventory_binding"]["authorized_source_edf_sha256"]
        or not _is_sha256(data["parent_signal_sha256"])
        or not _is_sha256(data["signal_labels_sha256"])
        or data["observed_parent_electrodes"] != list(PARENT_ELECTRODES)
        or data["reference_semantics"]
        != "shared_opaque_reference_protocol_authorized_not_header_proven"
    ):
        raise ValueError("opaque event authorization identity/source drifted")
    if (
        expected_parent_signal_sha256 is not None
        and data["parent_signal_sha256"] != expected_parent_signal_sha256
    ):
        raise ValueError("opaque event authorization belongs to another parent signal")
    if data["permissions"] != {
        "car19_replay_authorized": True,
        "signed_tcp22_difference_authorized": True,
        "tcp22_direct_evidence_authorized": True,
        "interpolation_used": False,
    } or data["claim_boundary"] != authority["claim_boundary"]:
        raise ValueError("opaque event authorization permissions/boundary drifted")
    expected_id = _EVENT_PREFIX + canonical_json_sha256(
        _id_source(data, "authorization_id")
    )[:24]
    if data["authorization_id"] != expected_id:
        raise ValueError("opaque event authorization_id drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("opaque event authorization receipt hash drifted")
    return data


__all__ = [
    "OPAQUE_REFERENCE_AUTHORITY_SCHEMA_VERSION",
    "OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION",
    "OPAQUE_REFERENCE_ROUTE_ID",
    "build_opaque_reference_event_authorization",
    "build_private_opaque_reference_authority",
    "validate_opaque_reference_event_authorization",
    "validate_private_opaque_reference_authority",
]
