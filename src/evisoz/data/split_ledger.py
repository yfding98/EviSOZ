"""Patient linkage and split-isolation ledgers for EviSOZ Stage 0."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from .artifact_ref import (
    CANONICAL_JSON_HASH_DOMAIN,
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)


PATIENT_LINKAGE_EVIDENCE_SCHEMA_VERSION = "evisoz_patient_linkage_evidence_v1"
PATIENT_LINKAGE_GROUP_SCHEMA_VERSION = "evisoz_patient_linkage_group_v1"
SPLIT_ROSTER_SCHEMA_VERSION = "evisoz_split_roster_v1"

LINKAGE_STATUSES = ("singleton", "verified_cross_dataset")
EVISOZ_ROLES = ("development_cv", "locked_test", "external_evaluation")
LINKAGE_EVIDENCE_ARTIFACT_KIND = "patient_linkage_evidence"
LINKAGE_EVIDENCE_ASSERTION = "same_patient_across_datasets"
LINKAGE_VERIFICATION_METHODS = (
    "privacy_preserving_record_linkage",
    "authorized_manual_crosswalk",
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_HELDOUT_OFFICIAL_SPLIT_RE = re.compile(
    r"(?:^|[._:-])(?:test(?:ing)?|eval(?:uation)?|held[-_]?out|hold[-_]?out|external)(?:$|[._:-])",
    re.IGNORECASE,
)
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a stable ASCII identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _id_source(value: Mapping[str, object], id_field: str, hash_field: str) -> dict[str, object]:
    source = deepcopy(dict(value))
    source[id_field] = _PENDING_ID
    source[hash_field] = _HASH_PLACEHOLDER
    return source


def _hash_source(value: Mapping[str, object], hash_field: str) -> dict[str, object]:
    source = deepcopy(dict(value))
    source[hash_field] = _HASH_PLACEHOLDER
    return source


def _validate_member_rows(
    value: object,
    *,
    context: str,
    require_sorted: bool,
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty array")
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(value):
        if type(row) is not dict or set(row) != {
            "dataset_id",
            "patient_key",
            "source_patient_sha256",
        }:
            raise ValueError(f"{context}[{index}] fields drifted")
        normalized.append(
            {
                "dataset_id": _identifier(
                    row["dataset_id"], f"{context}[{index}].dataset_id"
                ),
                "patient_key": _identifier(
                    row["patient_key"], f"{context}[{index}].patient_key"
                ),
                "source_patient_sha256": _sha256(
                    row["source_patient_sha256"],
                    f"{context}[{index}].source_patient_sha256",
                ),
            }
        )
    ordered = sorted(normalized, key=lambda row: (row["dataset_id"], row["patient_key"]))
    if require_sorted and normalized != ordered:
        raise ValueError(f"{context} must be canonically sorted")
    if len({(row["dataset_id"], row["patient_key"]) for row in normalized}) != len(
        normalized
    ):
        raise ValueError(f"{context} contains duplicate dataset/patient members")
    if len({row["source_patient_sha256"] for row in normalized}) != len(normalized):
        raise ValueError(f"{context} contains duplicate source patient hashes")
    return normalized if require_sorted else ordered


def validate_patient_linkage_evidence(
    value: object,
    *,
    expected_members: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Validate one trusted same-patient assertion bound by an ArtifactRef."""

    if type(value) is not dict or set(value) != {
        "schema_version",
        "assertion",
        "verification_method",
        "issuer_id",
        "members",
        "raw_patient_identifiers_stored",
    }:
        raise ValueError("patient linkage evidence fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PATIENT_LINKAGE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("patient linkage evidence schema_version drifted")
    if data["assertion"] != LINKAGE_EVIDENCE_ASSERTION:
        raise ValueError("patient linkage evidence assertion drifted")
    if data["verification_method"] not in LINKAGE_VERIFICATION_METHODS:
        raise ValueError("patient linkage evidence verification method is unsupported")
    _identifier(data["issuer_id"], "patient linkage evidence issuer_id")
    members = _validate_member_rows(
        data["members"], context="patient linkage evidence members", require_sorted=True
    )
    if len(members) < 2 or len({row["dataset_id"] for row in members}) < 2:
        raise ValueError("patient linkage evidence must bind multiple datasets")
    if data["raw_patient_identifiers_stored"] is not False:
        raise ValueError("raw patient identifiers are forbidden in linkage evidence")
    if expected_members is not None:
        expected = _validate_member_rows(
            [dict(row) for row in expected_members],
            context="expected patient linkage members",
            require_sorted=True,
        )
        if members != expected:
            raise ValueError("patient linkage evidence does not bind the linkage members")
    return data


def _validate_linkage_evidence_refs(
    refs: object,
    *,
    trusted_payloads: Mapping[str, object] | None,
    expected_members: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    if not isinstance(refs, list):
        raise ValueError("patient linkage evidence references must be an array")
    if trusted_payloads is not None and not isinstance(trusted_payloads, Mapping):
        raise TypeError("trusted linkage evidence payloads must be a mapping")
    evidence: list[dict[str, Any]] = []
    for index, raw_ref in enumerate(refs):
        ref = validate_artifact_ref(raw_ref)
        if ref["artifact_kind"] != LINKAGE_EVIDENCE_ARTIFACT_KIND:
            raise ValueError(
                f"linkage_evidence_refs[{index}] has an unauthorized artifact kind"
            )
        if ref["payload_schema_version"] != PATIENT_LINKAGE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"linkage_evidence_refs[{index}] has an unauthorized payload schema"
            )
        if ref["media_type"] != "application/json":
            raise ValueError(
                f"linkage_evidence_refs[{index}] must use application/json"
            )
        if ref["content_hash"]["domain"] != CANONICAL_JSON_HASH_DOMAIN:
            raise ValueError(
                f"linkage_evidence_refs[{index}] must use canonical_json_v1"
            )
        artifact_id = ref["artifact_id"]
        if trusted_payloads is None or artifact_id not in trusted_payloads:
            raise ValueError(
                f"linkage_evidence_refs[{index}] has no caller-trusted evidence payload"
            )
        payload = trusted_payloads[artifact_id]
        verify_artifact_content(ref, payload)
        validate_patient_linkage_evidence(payload, expected_members=expected_members)
        evidence.append(ref)
    if len({ref["artifact_id"] for ref in evidence}) != len(evidence):
        raise ValueError("patient linkage evidence references contain duplicates")
    return evidence


def _validate_global_patient_hash_partition(
    groups: Mapping[str, Mapping[str, object]],
) -> None:
    owners: dict[str, str] = {}
    for group_id, group in groups.items():
        for member in group["members"]:
            patient_hash = member["source_patient_sha256"]
            previous = owners.get(patient_hash)
            if previous is not None and previous != group_id:
                raise ValueError(
                    "source_patient_sha256 appears in multiple trusted linkage groups"
                )
            owners[patient_hash] = group_id


def _validate_official_split_role(
    official_splits: Sequence[Mapping[str, object]],
    *,
    evisoz_role: str,
) -> None:
    if evisoz_role != "development_cv":
        return
    for row in official_splits:
        split = row["official_split"]
        if isinstance(split, str) and _HELDOUT_OFFICIAL_SPLIT_RE.search(split):
            raise ValueError(
                "development_cv cannot include a dataset official held-out/test/eval/external split"
            )


def build_patient_linkage_group(
    *,
    members: Sequence[Mapping[str, object]],
    linkage_status: str,
    linkage_evidence_refs: Sequence[Mapping[str, object]] = (),
    trusted_linkage_evidence_payloads: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build one linkage group without storing raw patient identifiers."""

    if linkage_status not in LINKAGE_STATUSES:
        raise ValueError("unsupported linkage_status")
    if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
        raise TypeError("members must be an array")
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(members):
        if type(row) is not dict or set(row) != {
            "dataset_id",
            "patient_key",
            "source_patient_sha256",
        }:
            raise ValueError(f"members[{index}] fields drifted")
        normalized.append(
            {
                "dataset_id": _identifier(row["dataset_id"], f"members[{index}].dataset_id"),
                "patient_key": _identifier(row["patient_key"], f"members[{index}].patient_key"),
                "source_patient_sha256": _sha256(
                    row["source_patient_sha256"],
                    f"members[{index}].source_patient_sha256",
                ),
            }
        )
    normalized.sort(key=lambda row: (row["dataset_id"], row["patient_key"]))
    if not normalized or len({(row["dataset_id"], row["patient_key"]) for row in normalized}) != len(normalized):
        raise ValueError("linkage members must be non-empty and unique")
    if len({row["source_patient_sha256"] for row in normalized}) != len(normalized):
        raise ValueError("source patient hashes must be unique within one group")
    evidence = _validate_linkage_evidence_refs(
        list(linkage_evidence_refs),
        trusted_payloads=trusted_linkage_evidence_payloads,
        expected_members=normalized,
    )
    if linkage_status == "singleton" and (len(normalized) != 1 or evidence):
        raise ValueError("singleton linkage requires one member and no merge evidence")
    if linkage_status == "verified_cross_dataset":
        if len(normalized) < 2 or len({row["dataset_id"] for row in normalized}) < 2:
            raise ValueError("verified cross-dataset linkage requires multiple datasets")
        if not evidence:
            raise ValueError("verified cross-dataset linkage requires evidence refs")
    body: dict[str, Any] = {
        "schema_version": PATIENT_LINKAGE_GROUP_SCHEMA_VERSION,
        "linkage_group_id": _PENDING_ID,
        "linkage_status": linkage_status,
        "members": normalized,
        "linkage_evidence_refs": evidence,
        "raw_patient_identifiers_stored": False,
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["linkage_group_id"] = "EVISOZ-PAT-" + canonical_json_sha256(
        _id_source(body, "linkage_group_id", "receipt_sha256")
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(
        _hash_source(body, "receipt_sha256")
    )
    return validate_patient_linkage_group(
        body,
        trusted_linkage_evidence_payloads=trusted_linkage_evidence_payloads,
    )


def validate_patient_linkage_group(
    value: object,
    *,
    trusted_linkage_evidence_payloads: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "linkage_group_id",
        "linkage_status",
        "members",
        "linkage_evidence_refs",
        "raw_patient_identifiers_stored",
        "receipt_sha256",
    }:
        raise ValueError("patient linkage group fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PATIENT_LINKAGE_GROUP_SCHEMA_VERSION:
        raise ValueError("patient linkage schema_version drifted")
    # Validate members and policy without trusting the recorded seals.
    members = data["members"]
    if not isinstance(members, list) or not members:
        raise ValueError("patient linkage members must be non-empty")
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(members):
        if type(row) is not dict or set(row) != {
            "dataset_id",
            "patient_key",
            "source_patient_sha256",
        }:
            raise ValueError(f"members[{index}] fields drifted")
        normalized.append(
            {
                "dataset_id": _identifier(row["dataset_id"], f"members[{index}].dataset_id"),
                "patient_key": _identifier(row["patient_key"], f"members[{index}].patient_key"),
                "source_patient_sha256": _sha256(row["source_patient_sha256"], f"members[{index}].source_patient_sha256"),
            }
        )
    if normalized != sorted(normalized, key=lambda row: (row["dataset_id"], row["patient_key"])):
        raise ValueError("patient linkage members must be canonically sorted")
    if len({(row["dataset_id"], row["patient_key"]) for row in normalized}) != len(normalized):
        raise ValueError("patient linkage members contain duplicates")
    if len({row["source_patient_sha256"] for row in normalized}) != len(normalized):
        raise ValueError("patient linkage members contain duplicate source patient hashes")
    evidence = _validate_linkage_evidence_refs(
        data["linkage_evidence_refs"],
        trusted_payloads=trusted_linkage_evidence_payloads,
        expected_members=normalized,
    )
    status = data["linkage_status"]
    if status == "singleton":
        if len(normalized) != 1 or evidence:
            raise ValueError("singleton linkage policy violated")
    elif status == "verified_cross_dataset":
        if len(normalized) < 2 or len({row["dataset_id"] for row in normalized}) < 2 or not evidence:
            raise ValueError("verified cross-dataset linkage policy violated")
    else:
        raise ValueError("unsupported linkage_status")
    if data["raw_patient_identifiers_stored"] is not False:
        raise ValueError("raw patient identifiers are forbidden in the linkage receipt")
    expected_id = "EVISOZ-PAT-" + canonical_json_sha256(
        _id_source(data, "linkage_group_id", "receipt_sha256")
    )[:24]
    if data["linkage_group_id"] != expected_id:
        raise ValueError("linkage_group_id does not bind its content")
    if data["receipt_sha256"] != canonical_json_sha256(
        _hash_source(data, "receipt_sha256")
    ):
        raise ValueError("patient linkage receipt hash drifted")
    return data


def build_split_roster(
    *,
    linkage_groups: Sequence[Mapping[str, object]],
    assignments: Sequence[Mapping[str, object]],
    trusted_linkage_evidence_payloads: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Bind every linkage group to exactly one EviSOZ cohort role."""

    groups = [
        validate_patient_linkage_group(
            group,
            trusted_linkage_evidence_payloads=trusted_linkage_evidence_payloads,
        )
        for group in linkage_groups
    ]
    group_by_id = {group["linkage_group_id"]: group for group in groups}
    if len(group_by_id) != len(groups):
        raise ValueError("split roster contains duplicate linkage groups")
    _validate_global_patient_hash_partition(group_by_id)
    if isinstance(assignments, (str, bytes)) or not isinstance(assignments, Sequence):
        raise TypeError("assignments must be an array")
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(assignments):
        if type(raw) is not dict or set(raw) != {
            "linkage_group_id",
            "official_splits",
            "evisoz_role",
            "outer_holdout_fold",
            "locked",
        }:
            raise ValueError(f"assignments[{index}] fields drifted")
        group_id = _identifier(raw["linkage_group_id"], f"assignments[{index}].linkage_group_id")
        if group_id not in group_by_id:
            raise ValueError(f"assignments[{index}] references an unknown linkage group")
        role = raw["evisoz_role"]
        if role not in EVISOZ_ROLES:
            raise ValueError(f"assignments[{index}] has an unsupported EviSOZ role")
        fold = raw["outer_holdout_fold"]
        locked = raw["locked"]
        if type(locked) is not bool:
            raise TypeError(f"assignments[{index}].locked must be boolean")
        if role == "development_cv":
            if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0 or locked:
                raise ValueError("development_cv requires a non-negative outer fold and locked=false")
        else:
            if fold is not None or not locked:
                raise ValueError("locked/external evaluation requires fold=null and locked=true")
        official = raw["official_splits"]
        if not isinstance(official, list) or not official:
            raise ValueError("official_splits must be non-empty")
        expected_datasets = sorted({member["dataset_id"] for member in group_by_id[group_id]["members"]})
        normalized_official: list[dict[str, str | None]] = []
        for split_index, item in enumerate(official):
            if type(item) is not dict or set(item) != {"dataset_id", "official_split"}:
                raise ValueError(f"official_splits[{split_index}] fields drifted")
            dataset_id = _identifier(item["dataset_id"], "official_splits.dataset_id")
            split = item["official_split"]
            if split is not None:
                split = _identifier(split, "official_splits.official_split")
            normalized_official.append({"dataset_id": dataset_id, "official_split": split})
        normalized_official.sort(key=lambda row: row["dataset_id"])
        if [row["dataset_id"] for row in normalized_official] != expected_datasets:
            raise ValueError("official split datasets do not match linkage members")
        _validate_official_split_role(normalized_official, evisoz_role=role)
        rows.append(
            {
                "linkage_group_id": group_id,
                "linkage_group_ref": build_json_artifact_ref(
                    group_by_id[group_id],
                    artifact_kind="patient_linkage_group",
                    payload_schema_version=PATIENT_LINKAGE_GROUP_SCHEMA_VERSION,
                ),
                "official_splits": normalized_official,
                "evisoz_role": role,
                "outer_holdout_fold": fold,
                "locked": locked,
            }
        )
    rows.sort(key=lambda row: row["linkage_group_id"])
    if len(rows) != len(groups) or len({row["linkage_group_id"] for row in rows}) != len(rows):
        raise ValueError("every linkage group must have exactly one split assignment")
    body: dict[str, Any] = {
        "schema_version": SPLIT_ROSTER_SCHEMA_VERSION,
        "roster_id": _PENDING_ID,
        "assignments": rows,
        "patient_level_isolation": True,
        "synthetic_text_inherits_patient_split": True,
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["roster_id"] = "EVISOZ-SPLIT-" + canonical_json_sha256(
        _id_source(body, "roster_id", "receipt_sha256")
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(
        _hash_source(body, "receipt_sha256")
    )
    return validate_split_roster(
        body,
        trusted_linkage_groups=group_by_id,
        trusted_linkage_evidence_payloads=trusted_linkage_evidence_payloads,
    )


def validate_split_roster(
    value: object,
    *,
    trusted_linkage_groups: Mapping[str, Mapping[str, object]],
    trusted_linkage_evidence_payloads: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "roster_id",
        "assignments",
        "patient_level_isolation",
        "synthetic_text_inherits_patient_split",
        "receipt_sha256",
    }:
        raise ValueError("split roster fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != SPLIT_ROSTER_SCHEMA_VERSION:
        raise ValueError("split roster schema_version drifted")
    if not isinstance(trusted_linkage_groups, Mapping):
        raise TypeError("trusted_linkage_groups must be a mapping")
    trusted: dict[str, dict[str, Any]] = {}
    for raw_group_id, raw_group in trusted_linkage_groups.items():
        group_id = _identifier(raw_group_id, "trusted linkage group mapping key")
        group = validate_patient_linkage_group(
            raw_group,
            trusted_linkage_evidence_payloads=trusted_linkage_evidence_payloads,
        )
        if group_id != group["linkage_group_id"]:
            raise ValueError(
                "trusted linkage group mapping key does not match linkage_group_id"
            )
        if group_id in trusted:
            raise ValueError("trusted linkage groups contain duplicate internal IDs")
        trusted[group_id] = group
    _validate_global_patient_hash_partition(trusted)
    assignments = data["assignments"]
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("split roster assignments must be non-empty")
    ids: list[str] = []
    for index, row in enumerate(assignments):
        if type(row) is not dict or set(row) != {
            "linkage_group_id",
            "linkage_group_ref",
            "official_splits",
            "evisoz_role",
            "outer_holdout_fold",
            "locked",
        }:
            raise ValueError(f"assignments[{index}] fields drifted")
        group_id = _identifier(row["linkage_group_id"], f"assignments[{index}].linkage_group_id")
        group = trusted.get(group_id)
        if group is None:
            raise ValueError("split roster references an untrusted linkage group")
        expected_ref = build_json_artifact_ref(
            group,
            artifact_kind="patient_linkage_group",
            payload_schema_version=PATIENT_LINKAGE_GROUP_SCHEMA_VERSION,
        )
        if validate_artifact_ref(row["linkage_group_ref"]) != expected_ref:
            raise ValueError("split roster linkage reference does not bind trusted content")
        role = row["evisoz_role"]
        fold = row["outer_holdout_fold"]
        locked = row["locked"]
        if role == "development_cv":
            if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0 or locked is not False:
                raise ValueError("development_cv split assignment is invalid")
        elif role in {"locked_test", "external_evaluation"}:
            if fold is not None or locked is not True:
                raise ValueError("locked split assignment is invalid")
        else:
            raise ValueError("unsupported EviSOZ role")
        official = row["official_splits"]
        if not isinstance(official, list):
            raise ValueError("official split bindings must be an array")
        for item in official:
            if type(item) is not dict or set(item) != {"dataset_id", "official_split"}:
                raise ValueError("official split fields drifted")
        expected_datasets = sorted({member["dataset_id"] for member in group["members"]})
        if [item["dataset_id"] for item in official] != expected_datasets:
            raise ValueError("official split bindings do not match the linkage group")
        for item in official:
            _identifier(item["dataset_id"], "official split dataset_id")
            if item["official_split"] is not None:
                _identifier(item["official_split"], "official split value")
        _validate_official_split_role(official, evisoz_role=role)
        ids.append(group_id)
    if ids != sorted(ids) or len(ids) != len(set(ids)) or set(ids) != set(trusted):
        raise ValueError("split roster must contain one sorted row per trusted linkage group")
    if data["patient_level_isolation"] is not True or data["synthetic_text_inherits_patient_split"] is not True:
        raise ValueError("split roster isolation policy drifted")
    expected_id = "EVISOZ-SPLIT-" + canonical_json_sha256(
        _id_source(data, "roster_id", "receipt_sha256")
    )[:24]
    if data["roster_id"] != expected_id:
        raise ValueError("split roster_id does not bind its content")
    if data["receipt_sha256"] != canonical_json_sha256(
        _hash_source(data, "receipt_sha256")
    ):
        raise ValueError("split roster receipt hash drifted")
    return data


__all__ = [
    "PATIENT_LINKAGE_EVIDENCE_SCHEMA_VERSION",
    "PATIENT_LINKAGE_GROUP_SCHEMA_VERSION",
    "SPLIT_ROSTER_SCHEMA_VERSION",
    "LINKAGE_STATUSES",
    "EVISOZ_ROLES",
    "LINKAGE_EVIDENCE_ARTIFACT_KIND",
    "LINKAGE_EVIDENCE_ASSERTION",
    "LINKAGE_VERIFICATION_METHODS",
    "validate_patient_linkage_evidence",
    "build_patient_linkage_group",
    "validate_patient_linkage_group",
    "build_split_roster",
    "validate_split_roster",
]
