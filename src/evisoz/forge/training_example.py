"""Content-closed Stage-0 training envelope for one known seizure segment."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re
from typing import Any, Mapping

from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
)
from src.evisoz.data.dataset_policy import (
    FIELD_RELEASE_SCHEMA_VERSION,
    LOSS_PORTS,
    validate_field_release,
)
from src.evisoz.data.event_identity import (
    EVENT_IDENTITY_SCHEMA_VERSION,
    validate_event_identity,
)
from src.evisoz.data.split_ledger import (
    SPLIT_ROSTER_SCHEMA_VERSION,
    validate_patient_linkage_group,
    validate_split_roster,
)
from src.evisoz.data.tcp22_views import (
    MONTAGE_DERIVATION_RECEIPT_SCHEMA_VERSION,
    validate_montage_derivation_receipt,
)


TRAINING_EXAMPLE_SCHEMA_VERSION = "evisoz_training_example_v1"
ANCHOR_QUALITIES = ("exact", "approximate", "provided_interval")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64
_TOP_KEYS = {
    "schema_version",
    "example_id",
    "sample_id",
    "event_id",
    "dataset_id",
    "linkage_group_id",
    "anchor",
    "split_assignment",
    "report_scope",
    "artifact_refs",
    "field_state_counts",
    "unavailable_field_ids",
    "enabled_loss_ports",
    "safety_contract",
    "receipt_sha256",
}


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a stable identifier")
    return value


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["example_id"] = _PENDING_ID
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _trusted_refs(
    *,
    event_identity: Mapping[str, object],
    split_roster: Mapping[str, object],
    montage_receipt: Mapping[str, object],
    field_release: Mapping[str, object],
) -> dict[str, dict[str, Any]]:
    return {
        "event_identity": build_json_artifact_ref(
            event_identity,
            artifact_kind="event_identity",
            payload_schema_version=EVENT_IDENTITY_SCHEMA_VERSION,
        ),
        "split_roster": build_json_artifact_ref(
            split_roster,
            artifact_kind="split_roster",
            payload_schema_version=SPLIT_ROSTER_SCHEMA_VERSION,
        ),
        "montage_derivation": build_json_artifact_ref(
            montage_receipt,
            artifact_kind="montage_derivation_receipt",
            payload_schema_version=MONTAGE_DERIVATION_RECEIPT_SCHEMA_VERSION,
        ),
        "field_release": build_json_artifact_ref(
            field_release,
            artifact_kind="field_release",
            payload_schema_version=FIELD_RELEASE_SCHEMA_VERSION,
        ),
    }


def _assignment(
    split_roster: Mapping[str, object],
    linkage_group_id: str,
) -> Mapping[str, object]:
    rows = [
        row
        for row in split_roster["assignments"]
        if row["linkage_group_id"] == linkage_group_id
    ]
    if len(rows) != 1:
        raise ValueError("training example linkage group has no unique split assignment")
    return rows[0]


def build_training_example(
    *,
    sample_id: str,
    event_id: str,
    dataset_id: str,
    linkage_group_id: str,
    anchor_quality: str,
    event_identity: Mapping[str, object],
    split_roster: Mapping[str, object],
    trusted_linkage_groups: Mapping[str, Mapping[str, object]],
    trusted_linkage_evidence_payloads: Mapping[str, object] | None = None,
    montage_receipt: Mapping[str, object],
    field_release: Mapping[str, object],
) -> dict[str, Any]:
    """Build one envelope after all Stage-0 authorities are already frozen."""

    sample = _identifier(sample_id, "sample_id")
    event = _identifier(event_id, "event_id")
    dataset = _identifier(dataset_id, "dataset_id")
    group_id = _identifier(linkage_group_id, "linkage_group_id")
    if anchor_quality not in ANCHOR_QUALITIES:
        raise ValueError("unsupported anchor_quality")
    identity = validate_event_identity(event_identity)
    if identity["sample_id"] != sample or identity["event_id"] != event:
        raise ValueError("training example and event identity sample/event drifted")
    if identity["dataset_id"] != dataset or identity["linkage_group_id"] != group_id:
        raise ValueError("training example and event identity dataset/linkage drifted")
    if identity["anchor"]["quality"] != anchor_quality:
        raise ValueError("training example anchor quality drifted from event identity")
    roster = validate_split_roster(
        split_roster,
        trusted_linkage_groups=trusted_linkage_groups,
        trusted_linkage_evidence_payloads=trusted_linkage_evidence_payloads,
    )
    if group_id not in trusted_linkage_groups:
        raise ValueError("training example linkage group is untrusted")
    group = validate_patient_linkage_group(
        trusted_linkage_groups[group_id],
        trusted_linkage_evidence_payloads=trusted_linkage_evidence_payloads,
    )
    member_matches = [
        row
        for row in group["members"]
        if row["dataset_id"] == dataset
        and row["source_patient_sha256"] == identity["source_patient_sha256"]
    ]
    if len(member_matches) != 1:
        raise ValueError("event identity does not uniquely match its trusted patient member")
    montage = validate_montage_derivation_receipt(
        montage_receipt,
        trusted_event_identity=identity,
    )
    release = validate_field_release(
        field_release,
        trusted_event_identity=identity,
    )
    assignment = _assignment(roster, group_id)
    if dataset not in {row["dataset_id"] for row in assignment["official_splits"]}:
        raise ValueError("training example dataset is not in the patient linkage assignment")
    if release["sample_id"] != sample or release["dataset_id"] != dataset:
        raise ValueError("training example and field release identity drifted")
    if release["dataset_capability"]["patient_roster_sha256"] != roster["receipt_sha256"]:
        raise ValueError("field release capability is bound to a different patient roster")
    capability_by_id = {
        row["field_id"]: row
        for row in release["dataset_capability"]["field_roster"]
    }
    if any(
        assignment["evisoz_role"] not in capability_by_id[row["field_id"]]["allowed_roles"]
        for row in release["fields"]
    ):
        raise ValueError("field release is not authorized for this EviSOZ split role")
    counts = Counter(str(row["state"]) for row in release["fields"])
    unavailable = sorted(
        row["field_id"] for row in release["fields"] if row["state"] != "provided"
    )
    enabled = sorted(
        {
            port
            for row in release["fields"]
            for port in LOSS_PORTS
            if row["loss_permissions"][port]
        }
    )
    if assignment["evisoz_role"] != "development_cv" and enabled:
        raise ValueError(
            "locked/external examples cannot enable any training loss"
        )
    if (
        "typed_slot_loss" in enabled
        and not montage["permissions"]["tcp22_standalone_evidence_available"]
    ):
        raise ValueError("typed_slot_loss requires materialized eligible TCP22 evidence")
    if (
        "node_localization_loss" in enabled
        and not montage["permissions"]["residual_main_analysis_eligible"]
    ):
        raise ValueError(
            "node_localization_loss requires exact v29 plus eligible TCP22 onset evidence"
        )
    refs = _trusted_refs(
        event_identity=identity,
        split_roster=roster,
        montage_receipt=montage,
        field_release=release,
    )
    body: dict[str, Any] = {
        "schema_version": TRAINING_EXAMPLE_SCHEMA_VERSION,
        "example_id": _PENDING_ID,
        "sample_id": sample,
        "event_id": event,
        "dataset_id": dataset,
        "linkage_group_id": group_id,
        "anchor": {
            "condition": "known_seizure_segment",
            "quality": anchor_quality,
            "t0_seconds": 0.0,
            "analysis_interval_seconds": [-12.0, 48.0],
        },
        "split_assignment": {
            "evisoz_role": assignment["evisoz_role"],
            "outer_holdout_fold": assignment["outer_holdout_fold"],
            "locked": assignment["locked"],
        },
        "report_scope": release["report_scope"],
        "artifact_refs": refs,
        "field_state_counts": {
            "provided": counts.get("provided", 0),
            "not_provided": counts.get("not_provided", 0),
            "not_evaluable": counts.get("not_evaluable", 0),
            "technical_failure": counts.get("technical_failure", 0),
        },
        "unavailable_field_ids": unavailable,
        "enabled_loss_ports": enabled,
        "safety_contract": {
            "generated_text_can_supervise_localization": False,
            "knowledge_can_create_patient_facts": False,
            "teacher_runtime_required_at_deployment": False,
            "node_and_edge_coordinates_interchangeable": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["example_id"] = "EVISOZ-EX-" + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_training_example(
        body,
        split_roster=roster,
        trusted_linkage_groups=trusted_linkage_groups,
        trusted_linkage_evidence_payloads=trusted_linkage_evidence_payloads,
        event_identity=identity,
        montage_receipt=montage,
        field_release=release,
    )


def validate_training_example(
    value: object,
    *,
    split_roster: Mapping[str, object],
    trusted_linkage_groups: Mapping[str, Mapping[str, object]],
    event_identity: Mapping[str, object],
    trusted_linkage_evidence_payloads: Mapping[str, object] | None = None,
    montage_receipt: Mapping[str, object],
    field_release: Mapping[str, object],
) -> dict[str, Any]:
    """Validate all links against trusted source objects, not embedded hashes."""

    if type(value) is not dict or set(value) != _TOP_KEYS:
        raise ValueError("training example fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != TRAINING_EXAMPLE_SCHEMA_VERSION:
        raise ValueError("training example schema_version drifted")
    sample = _identifier(data["sample_id"], "sample_id")
    event = _identifier(data["event_id"], "event_id")
    dataset = _identifier(data["dataset_id"], "dataset_id")
    group_id = _identifier(data["linkage_group_id"], "linkage_group_id")
    identity = validate_event_identity(event_identity)
    if (
        identity["sample_id"] != sample
        or identity["event_id"] != event
        or identity["dataset_id"] != dataset
        or identity["linkage_group_id"] != group_id
    ):
        raise ValueError("training example identity differs from trusted event identity")
    roster = validate_split_roster(
        split_roster,
        trusted_linkage_groups=trusted_linkage_groups,
        trusted_linkage_evidence_payloads=trusted_linkage_evidence_payloads,
    )
    if group_id not in trusted_linkage_groups:
        raise ValueError("training example linkage group is untrusted")
    group = validate_patient_linkage_group(
        trusted_linkage_groups[group_id],
        trusted_linkage_evidence_payloads=trusted_linkage_evidence_payloads,
    )
    member_matches = [
        row
        for row in group["members"]
        if row["dataset_id"] == dataset
        and row["source_patient_sha256"] == identity["source_patient_sha256"]
    ]
    if len(member_matches) != 1:
        raise ValueError("trusted event identity has no unique patient member")
    montage = validate_montage_derivation_receipt(
        montage_receipt,
        trusted_event_identity=identity,
    )
    release = validate_field_release(
        field_release,
        trusted_event_identity=identity,
    )
    assignment = _assignment(roster, group_id)
    if dataset not in {row["dataset_id"] for row in assignment["official_splits"]}:
        raise ValueError("training example dataset is outside its split assignment")
    if release["sample_id"] != sample or release["dataset_id"] != dataset:
        raise ValueError("training example and field release identity drifted")
    if release["dataset_capability"]["patient_roster_sha256"] != roster["receipt_sha256"]:
        raise ValueError("field release capability patient roster drifted")
    capability_by_id = {
        row["field_id"]: row
        for row in release["dataset_capability"]["field_roster"]
    }
    if any(
        assignment["evisoz_role"] not in capability_by_id[row["field_id"]]["allowed_roles"]
        for row in release["fields"]
    ):
        raise ValueError("field release is unauthorized for this split role")
    if data["split_assignment"] != {
        "evisoz_role": assignment["evisoz_role"],
        "outer_holdout_fold": assignment["outer_holdout_fold"],
        "locked": assignment["locked"],
    }:
        raise ValueError("training example split assignment drifted")
    anchor = data["anchor"]
    if type(anchor) is not dict or anchor != {
        "condition": "known_seizure_segment",
        "quality": anchor.get("quality") if isinstance(anchor, dict) else None,
        "t0_seconds": 0.0,
        "analysis_interval_seconds": [-12.0, 48.0],
    }:
        raise ValueError("training example anchor contract drifted")
    if anchor["quality"] not in ANCHOR_QUALITIES:
        raise ValueError("unsupported anchor quality")
    if anchor["quality"] != identity["anchor"]["quality"]:
        raise ValueError("training example anchor differs from event identity")
    if data["report_scope"] != release["report_scope"]:
        raise ValueError("training example report_scope drifted from field release")
    expected_refs = _trusted_refs(
        event_identity=identity,
        split_roster=roster,
        montage_receipt=montage,
        field_release=release,
    )
    refs = data["artifact_refs"]
    if type(refs) is not dict or set(refs) != set(expected_refs):
        raise ValueError("training example artifact reference set drifted")
    if {key: validate_artifact_ref(ref) for key, ref in refs.items()} != expected_refs:
        raise ValueError("training example artifact reference does not bind trusted content")
    counts = Counter(str(row["state"]) for row in release["fields"])
    expected_counts = {
        "provided": counts.get("provided", 0),
        "not_provided": counts.get("not_provided", 0),
        "not_evaluable": counts.get("not_evaluable", 0),
        "technical_failure": counts.get("technical_failure", 0),
    }
    if data["field_state_counts"] != expected_counts:
        raise ValueError("training example field-state denominator drifted")
    expected_unavailable = sorted(
        row["field_id"] for row in release["fields"] if row["state"] != "provided"
    )
    if data["unavailable_field_ids"] != expected_unavailable:
        raise ValueError("training example unavailable field roster drifted")
    expected_ports = sorted(
        {
            port
            for row in release["fields"]
            for port in LOSS_PORTS
            if row["loss_permissions"][port]
        }
    )
    if data["enabled_loss_ports"] != expected_ports:
        raise ValueError("training example enabled loss ports drifted")
    if assignment["evisoz_role"] != "development_cv" and expected_ports:
        raise ValueError("locked/external examples cannot enable any training loss")
    if (
        "typed_slot_loss" in expected_ports
        and not montage["permissions"]["tcp22_standalone_evidence_available"]
    ):
        raise ValueError("typed_slot_loss lacks eligible materialized TCP22 evidence")
    if (
        "node_localization_loss" in expected_ports
        and not montage["permissions"]["residual_main_analysis_eligible"]
    ):
        raise ValueError("node_localization_loss lacks its montage/residual permission")
    if data["safety_contract"] != {
        "generated_text_can_supervise_localization": False,
        "knowledge_can_create_patient_facts": False,
        "teacher_runtime_required_at_deployment": False,
        "node_and_edge_coordinates_interchangeable": False,
    }:
        raise ValueError("training example safety contract drifted")
    expected_id = "EVISOZ-EX-" + canonical_json_sha256(_id_source(data))[:24]
    if data["example_id"] != expected_id:
        raise ValueError("training example_id does not bind its content")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("training example receipt hash drifted")
    return data


__all__ = [
    "TRAINING_EXAMPLE_SCHEMA_VERSION",
    "ANCHOR_QUALITIES",
    "build_training_example",
    "validate_training_example",
]
