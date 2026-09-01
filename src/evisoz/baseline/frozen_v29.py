"""Fail-closed access to the frozen canonical v29 H/D reference.

This module deliberately stops at two responsibilities:

* verify and selectively read the historical v29 resources; and
* resolve whether a patient may use the historical public held-fold row or
  the frozen five-fold new-event route.

The legacy state containers also contain development targets.  They are
therefore never deserialized wholesale: :func:`safe_open` is used with an
explicit per-resource tensor-key allowlist, and ``targets``/``target_mask``
are forbidden at the API boundary.  Raw-byte SHA verification reads the file
only as an opaque byte stream and does not deserialize any tensor.

Cache schemas, signal materialization and model execution belong to later P1
layers and are intentionally absent here.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import torch
from safetensors import safe_open

from src.evisoz.data.artifact_ref import (
    build_raw_artifact_ref,
    validate_artifact_ref,
)


RESOURCE_CONFIG_SCHEMA_VERSION = "evisoz_v29_frozen_resources_v1"
METHOD_ID = "canonical_v29_equal_H_D_probability_ensemble"
HISTORICAL_PUBLIC_ROUTE = "historical_public_oof_held_fold"
FROZEN_FIVE_FOLD_ROUTE = "frozen_five_fold_event_mean"
EVALUATOR_ONLY = "evaluator_only"
DEVELOPMENT_ONLY = "development_only"
INFERENCE_ONLY = "inference_only"
N_FOLDS = 5
PUBLIC_PATIENT_COUNT = 102
PUBLIC_METRIC_RECEIPT_KIND = "historical_public_v29_manifest_receipt_replay_v1"
PUBLIC_METRIC_MAPPING_VALIDATION_KIND = "public_v29_metric_mapping_validation_v1"
FROZEN_PUBLIC_IDENTITY_NAMESPACE = "deepsoz_tusz_public_v29"
FROZEN_MEMBER_RELATION = "frozen_public_roster_member"
PROVEN_ABSENT_RELATION = "proven_absent"
UNKNOWN_RELATION = "unknown"
FROZEN_MEMBER_PROOF_KIND = "frozen_public_roster_index"
CALLER_ABSENCE_PROOF_KIND = "caller_verified_namespace_absence"
P0_LINKAGE_PROOF_ARTIFACT_KIND = "patient_linkage_group"
P0_LINKAGE_PROOF_SCHEMA_VERSION = "evisoz_patient_linkage_group_v1"

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESOURCE_CONFIG = ROOT / "configs/evisoz_v29_frozen_resources_v1.json"
_KNOWN_RESOURCE_CONFIG_SHA256 = (
    "bb93d9a37723ccf338b5ca95599f42ed9acc80391144380005de51017082675b"
)

STANDARD_19 = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
)
PZ_INDEX = STANDARD_19.index("PZ")
V29_CANDIDATE_MASK = tuple(index != PZ_INDEX for index in range(len(STANDARD_19)))

PUBLIC_MANIFEST_RESOURCE = "public_v29_manifest"
PUBLIC_OOF_RESOURCE = "public_v29_oof"
DIRECT_STATES_RESOURCE = "direct_fold_states"
H_STATES_RESOURCE = "h_fold_states"
H_MANIFEST_RESOURCE = "h_fold_manifest"
PUBLIC_UNION_RESOURCE = "public_union_manifest"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_FORBIDDEN_TENSOR_KEYS = frozenset(("targets", "target_mask"))
_PUBLIC_SOURCE_ROLES = frozenset(("source_train", "source_dev", "source_eval"))
_CANONICAL_PUBLIC_METRICS = MappingProxyType(
    {
        "denominator": 102,
        "strict_hits": 54,
        "strict_value": 0.529411792755127,
        "n4_hits": 78,
        "n4_value": 0.7647058963775635,
        "hit_at_3_hits": 79,
        "hit_at_3_value": 0.7745097875595093,
        "hit_at_5_hits": 90,
        "hit_at_5_value": 0.8823529481887817,
        "mrr": 0.6669437885284424,
    }
)


def _direct_state_keys() -> tuple[str, ...]:
    suffixes = (
        "candidate_mask",
        "phase_weights",
        "prior_logits",
        "tile_scorer.bias",
        "tile_scorer.weight",
    )
    return tuple(
        f"outer_state.fold{fold}.{suffix}"
        for fold in range(N_FOLDS)
        for suffix in suffixes
    )


def _h_state_keys() -> tuple[str, ...]:
    result: list[str] = []
    for fold in range(N_FOLDS):
        result.extend(
            (
                f"outer{fold}.frozen_labram_only.candidate_mask",
                f"outer{fold}.frozen_labram_only.h_weight",
                f"outer{fold}.frozen_labram_only.prior_logits",
                f"outer{fold}.transform.h_center",
                f"outer{fold}.transform.h_components",
                f"outer{fold}.transform.h_pca_mean",
                f"outer{fold}.transform.h_scale",
            )
        )
    return tuple(result)


PUBLIC_OOF_TENSOR_KEYS = (
    "candidate_mask",
    "oof.h_only_probability",
    "oof.portable_equal_ensemble_probability",
    "oof.rank1_direct_probability",
    "patient_folds",
)
DIRECT_STATE_TENSOR_KEYS = _direct_state_keys()
H_STATE_TENSOR_KEYS = _h_state_keys()
_TENSOR_KEY_PROFILES = {
    "public_oof_reference_v1": frozenset(PUBLIC_OOF_TENSOR_KEYS),
    "direct_fold_states_v1": frozenset(DIRECT_STATE_TENSOR_KEYS),
    "h_fold_states_v1": frozenset(H_STATE_TENSOR_KEYS),
}

# These identities are code-frozen so editing the JSON config cannot silently
# authorize a different historical artifact.
_KNOWN_RESOURCES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        PUBLIC_MANIFEST_RESOURCE: {
            "path": "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815/manifest.json",
            "sha256": "2db07cbc1319eac8a90c3f5cdf45ebfaf784ee383eb65ae33e6de7e110ea7906",
            "size_bytes": 14053,
            "format": "json",
            "tensor_key_profile": None,
        },
        PUBLIC_OOF_RESOURCE: {
            "path": "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815/oof_predictions.safetensors",
            "sha256": "9ded9de54090d8f7b4ffc5e71cf7f4f86405f9db2265c8a60f0614634f8d62cf",
            "size_bytes": 50053,
            "format": "safetensors",
            "tensor_key_profile": "public_oof_reference_v1",
        },
        DIRECT_STATES_RESOURCE: {
            "path": "outputs/labram_rank1_direct_token_oof_v28_20260815/model_and_oof.safetensors",
            "sha256": "0b5ffaf0ed504c36e01a0be28676f7797a703a6ef57079dc743362984a4351a9",
            "size_bytes": 33448,
            "format": "safetensors",
            "tensor_key_profile": "direct_fold_states_v1",
        },
        H_STATES_RESOURCE: {
            "path": "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/outer_fold_states.safetensors",
            "sha256": "18b69f5e2fc718d2668b3a727f9a3f7bf0da33a613896d939559260ad3009b98",
            "size_bytes": 239497,
            "format": "safetensors",
            "tensor_key_profile": "h_fold_states_v1",
        },
        H_MANIFEST_RESOURCE: {
            "path": "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/manifest.json",
            "sha256": "b12c84f071ae83849bff4263ad7fdc8285444719c31263ffc5d3c5336b21faed",
            "size_bytes": 210886,
            "format": "json",
            "tensor_key_profile": None,
        },
        PUBLIC_UNION_RESOURCE: {
            "path": "outputs/public_development_union_identity_v12_20260812/manifest.json",
            "sha256": "645c55541c37dfc204fdd48c21e0a3c81fe7201f76b862556d1c4dc3bfa4d429",
            "size_bytes": 1003084,
            "format": "json",
            "tensor_key_profile": None,
        },
    }
)


@dataclass(frozen=True)
class FrozenV29Resource:
    resource_id: str
    path: Path
    relative_path: str
    sha256: str
    size_bytes: int
    format: str
    tensor_key_profile: str | None
    tensor_sha256: Mapping[str, str]


@dataclass(frozen=True)
class FrozenV29ResourceRegistry:
    repository_root: Path
    config_path: Path
    config_sha256: str
    config_bytes: bytes
    resources: Mapping[str, FrozenV29Resource]

    def require(self, resource_id: str) -> FrozenV29Resource:
        try:
            return self.resources[resource_id]
        except KeyError as exc:
            raise KeyError(f"unknown frozen v29 resource: {resource_id!r}") from exc


@dataclass(frozen=True)
class PublicV29PatientBinding:
    patient_id: str
    patient_index: int
    held_out_fold: int
    source_role: str
    access_role: str
    route: str = HISTORICAL_PUBLIC_ROUTE

    @property
    def evaluator_only(self) -> bool:
        return self.access_role == EVALUATOR_ONLY


@dataclass(frozen=True)
class PublicV29RosterIndex:
    patient_ids: tuple[str, ...]
    by_patient: Mapping[str, PublicV29PatientBinding]
    fold_counts: Mapping[int, int]
    source_role_counts: Mapping[str, int]
    resource_config_sha256: str
    resource_registry_projection_sha256: str
    authority_sha256: str

    def require(self, patient_id: str) -> PublicV29PatientBinding:
        try:
            return self.by_patient[patient_id]
        except KeyError as exc:
            raise KeyError(f"patient is not in the historical public v29 roster: {patient_id!r}") from exc


@dataclass(frozen=True)
class V29PatientIdentity:
    namespace: str
    patient_id: str


@dataclass(frozen=True)
class V29PublicRosterRelation:
    identity_sha256: str
    state: str
    proof_kind: str | None
    proof_sha256: str | None
    proof_ref: Mapping[str, Any] | None
    relation_sha256: str


@dataclass(frozen=True)
class V29RouteDecision:
    patient_id: str
    identity_namespace: str
    identity_sha256: str
    public_roster_relation: str
    public_roster_relation_proof_kind: str
    public_roster_relation_proof_sha256: str
    public_roster_relation_proof_ref: Mapping[str, Any] | None
    public_roster_relation_sha256: str
    route: str
    unit_kind: str
    fold_indices: tuple[int, ...]
    public_patient_index: int | None
    historical_source_role: str | None
    access_role: str
    historical_development_eligible: bool
    route_layer_training_authorized: bool

    @property
    def evaluator_only(self) -> bool:
        return self.access_role == EVALUATOR_ONLY


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _patient_id(value: object) -> str:
    if not isinstance(value, str) or _PATIENT_ID_RE.fullmatch(value) is None:
        raise ValueError("patient_id must be a stable ASCII identifier")
    return value


def _identity_namespace(value: object) -> str:
    if not isinstance(value, str) or _NAMESPACE_RE.fullmatch(value) is None:
        raise ValueError("identity namespace must be a lowercase stable identifier")
    return value


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_v29_patient_identity(
    *, namespace: object, patient_id: object
) -> V29PatientIdentity:
    return V29PatientIdentity(
        namespace=_identity_namespace(namespace),
        patient_id=_patient_id(patient_id),
    )


def v29_patient_identity_sha256(identity: V29PatientIdentity) -> str:
    if not isinstance(identity, V29PatientIdentity):
        raise TypeError("identity must be V29PatientIdentity")
    namespace = _identity_namespace(identity.namespace)
    patient = _patient_id(identity.patient_id)
    return _canonical_json_sha256(
        {
            "domain": "evisoz_v29_typed_patient_identity_v1",
            "namespace": namespace,
            "patient_id": patient,
        }
    )


def _relation_sha256(
    *,
    identity_sha256: str,
    state: str,
    proof_kind: str | None,
    proof_sha256: str | None,
    proof_ref: Mapping[str, Any] | None,
) -> str:
    return _canonical_json_sha256(
        {
            "domain": "evisoz_v29_public_roster_relation_v1",
            "identity_sha256": identity_sha256,
            "state": state,
            "proof_kind": proof_kind,
            "proof_sha256": proof_sha256,
            "proof_ref": deepcopy(dict(proof_ref)) if proof_ref is not None else None,
        }
    )


def _validated_p0_linkage_proof_ref(value: object) -> dict[str, Any]:
    reference = validate_artifact_ref(value)
    if (
        reference["artifact_kind"] != P0_LINKAGE_PROOF_ARTIFACT_KIND
        or reference["media_type"] != "application/json"
        or reference["content_hash"]["domain"] != "canonical_json_v1"
        or reference["payload_schema_version"]
        != P0_LINKAGE_PROOF_SCHEMA_VERSION
    ):
        raise ValueError(
            "proven absence requires a validated P0 patient-linkage-group ArtifactRef"
        )
    return reference


def _validate_public_roster_relation(
    relation: V29PublicRosterRelation,
    *,
    identity: V29PatientIdentity,
) -> V29PublicRosterRelation:
    if not isinstance(relation, V29PublicRosterRelation):
        raise TypeError("public_roster_relation must be V29PublicRosterRelation")
    identity_sha = v29_patient_identity_sha256(identity)
    if relation.identity_sha256 != identity_sha:
        raise ValueError("public roster relation is bound to another typed identity")
    _sha256(relation.identity_sha256, "relation.identity_sha256")
    if relation.state == FROZEN_MEMBER_RELATION:
        if relation.proof_kind != FROZEN_MEMBER_PROOF_KIND:
            raise ValueError("frozen member relation proof kind drifted")
        proof_sha = _sha256(relation.proof_sha256, "relation.proof_sha256")
        if relation.proof_ref is not None:
            raise ValueError("frozen member relation cannot carry a linkage proof ref")
        proof_ref = None
    elif relation.state == PROVEN_ABSENT_RELATION:
        if relation.proof_kind != CALLER_ABSENCE_PROOF_KIND:
            raise ValueError("proven-absent relation proof kind drifted")
        proof_sha = _sha256(relation.proof_sha256, "relation.proof_sha256")
        proof_ref = _validated_p0_linkage_proof_ref(relation.proof_ref)
        if proof_sha != proof_ref["ref_sha256"]:
            raise ValueError("proven-absent proof SHA does not bind its ArtifactRef")
    elif relation.state == UNKNOWN_RELATION:
        if (
            relation.proof_kind is not None
            or relation.proof_sha256 is not None
            or relation.proof_ref is not None
        ):
            raise ValueError("unknown relation cannot carry proof fields")
        proof_sha = None
        proof_ref = None
    else:
        raise ValueError("unsupported public roster relation state")
    expected = _relation_sha256(
        identity_sha256=identity_sha,
        state=relation.state,
        proof_kind=relation.proof_kind,
        proof_sha256=proof_sha,
        proof_ref=proof_ref,
    )
    if relation.relation_sha256 != expected:
        raise ValueError("public roster relation content digest drifted")
    return relation


def build_proven_absent_public_roster_relation(
    identity: V29PatientIdentity,
    *,
    p0_linkage_proof_ref: object,
) -> V29PublicRosterRelation:
    """Bind caller-verified namespace absence to its full P0 linkage ref.

    The caller must first validate the referenced patient-linkage-group
    payload in the P0 ledger.  This boundary intentionally accepts the full
    typed ArtifactRef, never a caller-chosen naked digest.
    """

    if not isinstance(identity, V29PatientIdentity):
        raise TypeError("identity must be V29PatientIdentity")
    if _identity_namespace(identity.namespace) == FROZEN_PUBLIC_IDENTITY_NAMESPACE:
        raise ValueError("frozen public namespace cannot use caller-proven absence")
    identity_sha = v29_patient_identity_sha256(identity)
    proof_ref = _validated_p0_linkage_proof_ref(p0_linkage_proof_ref)
    proof_sha = proof_ref["ref_sha256"]
    relation = V29PublicRosterRelation(
        identity_sha256=identity_sha,
        state=PROVEN_ABSENT_RELATION,
        proof_kind=CALLER_ABSENCE_PROOF_KIND,
        proof_sha256=proof_sha,
        proof_ref=proof_ref,
        relation_sha256=_relation_sha256(
            identity_sha256=identity_sha,
            state=PROVEN_ABSENT_RELATION,
            proof_kind=CALLER_ABSENCE_PROOF_KIND,
            proof_sha256=proof_sha,
            proof_ref=proof_ref,
        ),
    )
    return _validate_public_roster_relation(relation, identity=identity)


def build_unknown_public_roster_relation(
    identity: V29PatientIdentity,
) -> V29PublicRosterRelation:
    identity_sha = v29_patient_identity_sha256(identity)
    relation = V29PublicRosterRelation(
        identity_sha256=identity_sha,
        state=UNKNOWN_RELATION,
        proof_kind=None,
        proof_sha256=None,
        proof_ref=None,
        relation_sha256=_relation_sha256(
            identity_sha256=identity_sha,
            state=UNKNOWN_RELATION,
            proof_kind=None,
            proof_sha256=None,
            proof_ref=None,
        ),
    )
    return _validate_public_roster_relation(relation, identity=identity)


def _strict_json_bytes(payload: bytes, context: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"{context} contains invalid constant {value!r}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{context} is not UTF-8") from exc
    if type(value) is not dict:
        raise TypeError(f"{context} must be a JSON object")
    return value


def _secure_read_bytes(path: Path, *, context: str) -> bytes:
    """Read one regular file through a single O_NOFOLLOW descriptor."""

    if path.is_symlink():
        raise ValueError(f"{context} must not be a symbolic link: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{context} must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _resolve_frozen_path(root: Path, relative_path: object) -> tuple[str, Path]:
    if not isinstance(relative_path, str) or not relative_path or relative_path.strip() != relative_path:
        raise ValueError("frozen resource path must be a trimmed relative path")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("frozen resource path must remain repository-relative")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"frozen resource path contains a symbolic link: {cursor}")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("frozen resource escapes the repository root") from exc
    return relative.as_posix(), resolved


def _read_verified_resource_bytes(resource: FrozenV29Resource) -> bytes:
    payload = _secure_read_bytes(
        resource.path,
        context=f"frozen v29 resource {resource.resource_id}",
    )
    if (
        hashlib.sha256(payload).hexdigest() != resource.sha256
        or len(payload) != resource.size_bytes
    ):
        raise ValueError(
            f"frozen v29 resource identity drifted: {resource.resource_id}"
        )
    return payload


def _validate_resource_descriptor(
    registry: FrozenV29ResourceRegistry,
    resource_id: str,
) -> FrozenV29Resource:
    if not isinstance(registry, FrozenV29ResourceRegistry):
        raise TypeError("registry must be FrozenV29ResourceRegistry")
    if not isinstance(registry.config_bytes, bytes):
        raise TypeError("registry config_bytes must be immutable bytes")
    config_sha256 = hashlib.sha256(registry.config_bytes).hexdigest()
    if (
        config_sha256 != registry.config_sha256
        or config_sha256 != _KNOWN_RESOURCE_CONFIG_SHA256
    ):
        raise ValueError("frozen v29 registry config identity drifted")
    resource = registry.require(resource_id)
    expected = _KNOWN_RESOURCES.get(resource_id)
    if expected is None:
        raise ValueError("resource is outside the code-frozen v29 bundle")
    expected_path = (registry.repository_root / str(expected["path"])).resolve(strict=True)
    actual = {
        "path": resource.relative_path,
        "sha256": resource.sha256,
        "size_bytes": resource.size_bytes,
        "format": resource.format,
        "tensor_key_profile": resource.tensor_key_profile,
    }
    if actual != dict(expected) or resource.path != expected_path:
        raise ValueError(f"frozen v29 resource descriptor drifted: {resource_id}")
    return resource


@contextmanager
def _private_safetensors_snapshot(payload: bytes) -> Iterator[Path]:
    """Expose verified bytes to safe_open without reopening the source path."""

    if not isinstance(payload, bytes):
        raise TypeError("safetensors snapshot payload must be bytes")
    with tempfile.TemporaryDirectory(prefix="evisoz-v29-safeopen-") as directory:
        snapshot = Path(directory) / "verified.safetensors"
        descriptor = os.open(
            snapshot,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("failed to write private safetensors snapshot")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        yield snapshot


def _validate_resource_config(payload: object, *, repository_root: Path) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "status",
        "method_id",
        "fold_count",
        "resources",
        "public_roster_policy",
        "route_policy",
        "access_policy",
    }:
        raise ValueError("frozen v29 resource config fields drifted")
    data = deepcopy(payload)
    if data["schema_version"] != RESOURCE_CONFIG_SCHEMA_VERSION:
        raise ValueError("frozen v29 resource config schema_version drifted")
    if data["status"] != "frozen" or data["method_id"] != METHOD_ID:
        raise ValueError("frozen v29 method/status drifted")
    if data["fold_count"] != N_FOLDS:
        raise ValueError("frozen v29 fold count drifted")

    rows = data["resources"]
    if not isinstance(rows, list) or len(rows) != len(_KNOWN_RESOURCES):
        raise ValueError("frozen v29 resource roster is incomplete")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "resource_id",
            "path",
            "sha256",
            "size_bytes",
            "format",
            "tensor_key_profile",
            "tensor_sha256",
        }:
            raise ValueError(f"resources[{index}] fields drifted")
        resource_id = row["resource_id"]
        if not isinstance(resource_id, str) or resource_id in seen:
            raise ValueError("frozen v29 resource IDs must be unique strings")
        seen.add(resource_id)
        expected = _KNOWN_RESOURCES.get(resource_id)
        if expected is None:
            raise ValueError(f"unrecognized frozen v29 resource: {resource_id!r}")
        for field in ("path", "sha256", "size_bytes", "format", "tensor_key_profile"):
            if row[field] != expected[field]:
                raise ValueError(f"frozen resource {resource_id!r} changed {field}")
        _sha256(row["sha256"], f"resources[{index}].sha256")
        relative, _ = _resolve_frozen_path(repository_root, row["path"])
        if relative != row["path"]:
            raise ValueError("frozen resource path is not canonical")
        tensor_hashes = row["tensor_sha256"]
        if type(tensor_hashes) is not dict:
            raise ValueError("tensor_sha256 must be an object")
        profile = row["tensor_key_profile"]
        if profile is None:
            if tensor_hashes:
                raise ValueError("JSON resources cannot declare tensor hashes")
        else:
            allowlist = _TENSOR_KEY_PROFILES.get(profile)
            if allowlist is None:
                raise ValueError("unknown frozen tensor-key profile")
            if not set(tensor_hashes).issubset(allowlist):
                raise ValueError("tensor receipt lies outside its resource allowlist")
            for key, digest in tensor_hashes.items():
                _sha256(digest, f"resources[{index}].tensor_sha256[{key!r}]")
    if seen != set(_KNOWN_RESOURCES):
        raise ValueError("frozen v29 resource roster does not match code freeze")

    roster = data["public_roster_policy"]
    if roster != {
        "expected_patient_count": 102,
        "h_manifest_resource_id": H_MANIFEST_RESOURCE,
        "union_manifest_resource_id": PUBLIC_UNION_RESOURCE,
        "fold_tensor_resource_id": PUBLIC_OOF_RESOURCE,
        "source_role_field": "legacy_model_split",
        "source_role_counts": {"source_train": 66, "source_dev": 15, "source_eval": 21},
        "source_eval_access_role": EVALUATOR_ONLY,
    }:
        raise ValueError("frozen public roster policy drifted")
    if data["route_policy"] != {
        "known_public_route": HISTORICAL_PUBLIC_ROUTE,
        "new_patient_route": FROZEN_FIVE_FOLD_ROUTE,
        "known_public_unit_kind": "patient",
        "new_patient_unit_kind": "event",
        "known_public_fold_policy": "frozen_held_out_fold_only",
        "new_patient_fold_policy": "equal_probability_mean_over_five_folds",
    }:
        raise ValueError("frozen v29 route policy drifted")
    if data["access_policy"] != {
        "reader": "safetensors.safe_open_explicit_key_allowlist",
        "forbidden_tensor_keys": ["target_mask", "targets"],
        "target_values_loaded": False,
        "official_source_eval_is_evaluator_only": True,
    }:
        raise ValueError("frozen v29 access policy drifted")
    return data


def load_frozen_v29_resource_registry(
    config_path: str | Path = DEFAULT_RESOURCE_CONFIG,
    *,
    repository_root: str | Path = ROOT,
) -> FrozenV29ResourceRegistry:
    """Load the code-frozen registry and verify every resource SHA/size."""

    root = Path(repository_root).resolve(strict=True)
    config = Path(config_path).absolute()
    config_bytes = _secure_read_bytes(
        config,
        context="frozen v29 resource config",
    )
    if hashlib.sha256(config_bytes).hexdigest() != _KNOWN_RESOURCE_CONFIG_SHA256:
        raise ValueError("frozen v29 resource config identity drifted")
    data = _validate_resource_config(
        _strict_json_bytes(config_bytes, "frozen v29 resource config"),
        repository_root=root,
    )
    resources: dict[str, FrozenV29Resource] = {}
    for row in data["resources"]:
        relative, path = _resolve_frozen_path(root, row["path"])
        resource = FrozenV29Resource(
            resource_id=row["resource_id"],
            path=path,
            relative_path=relative,
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            format=row["format"],
            tensor_key_profile=row["tensor_key_profile"],
            tensor_sha256=MappingProxyType(dict(row["tensor_sha256"])),
        )
        _read_verified_resource_bytes(resource)
        resources[resource.resource_id] = resource
    return FrozenV29ResourceRegistry(
        repository_root=root,
        config_path=config,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        config_bytes=config_bytes,
        resources=MappingProxyType(resources),
    )


def _resource_registry_projection(
    registry: FrozenV29ResourceRegistry,
) -> dict[str, object]:
    if not isinstance(registry, FrozenV29ResourceRegistry):
        raise TypeError("registry must be FrozenV29ResourceRegistry")
    return {
        "config_sha256": registry.config_sha256,
        "resources": [
            {
                "resource_id": resource_id,
                "relative_path": resource.relative_path,
                "sha256": resource.sha256,
                "size_bytes": resource.size_bytes,
                "format": resource.format,
                "tensor_key_profile": resource.tensor_key_profile,
                "tensor_sha256": dict(sorted(resource.tensor_sha256.items())),
            }
            for resource_id, resource in sorted(registry.resources.items())
        ],
    }


def validate_frozen_v29_resource_registry(
    registry: FrozenV29ResourceRegistry,
) -> FrozenV29ResourceRegistry:
    """Replay a registry from its config and current frozen resource bytes."""

    if not isinstance(registry, FrozenV29ResourceRegistry):
        raise TypeError("registry must be FrozenV29ResourceRegistry")
    if not isinstance(registry.config_bytes, bytes):
        raise TypeError("registry config_bytes must be immutable bytes")
    if (
        hashlib.sha256(registry.config_bytes).hexdigest() != registry.config_sha256
        or registry.config_sha256 != _KNOWN_RESOURCE_CONFIG_SHA256
    ):
        raise ValueError("registry config bytes/hash binding drifted")
    replayed = load_frozen_v29_resource_registry(
        registry.config_path,
        repository_root=registry.repository_root,
    )
    if (
        replayed.config_bytes != registry.config_bytes
        or replayed.config_sha256 != registry.config_sha256
        or _resource_registry_projection(replayed)
        != _resource_registry_projection(registry)
    ):
        raise ValueError("frozen v29 registry does not replay from its resource bundle")
    return replayed


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor receipt requires a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _is_forbidden_tensor_key(key: str) -> bool:
    return key in _FORBIDDEN_TENSOR_KEYS or key.rsplit(".", 1)[-1] in _FORBIDDEN_TENSOR_KEYS


def read_whitelisted_tensors(
    registry: FrozenV29ResourceRegistry,
    resource_id: str,
    tensor_keys: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Read only explicitly allowlisted tensors from one verified resource."""

    if not isinstance(registry, FrozenV29ResourceRegistry):
        raise TypeError("registry must be FrozenV29ResourceRegistry")
    if isinstance(tensor_keys, (str, bytes)) or not isinstance(tensor_keys, Sequence):
        raise TypeError("tensor_keys must be a sequence of exact tensor names")
    requested = tuple(tensor_keys)
    if not requested or any(not isinstance(key, str) or not key for key in requested):
        raise ValueError("tensor_keys must contain non-empty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("tensor_keys must not contain duplicates")
    if any(_is_forbidden_tensor_key(key) for key in requested):
        raise PermissionError("targets and target_mask are forbidden frozen-v29 reads")

    resource = _validate_resource_descriptor(registry, resource_id)
    if resource.format != "safetensors" or resource.tensor_key_profile is None:
        raise TypeError("requested frozen v29 resource is not a tensor container")
    allowlist = _TENSOR_KEY_PROFILES[resource.tensor_key_profile]
    if not set(requested).issubset(allowlist):
        unknown = sorted(set(requested).difference(allowlist))
        raise PermissionError(f"tensor request is outside the frozen allowlist: {unknown}")
    verified_bytes = _read_verified_resource_bytes(resource)

    tensors: dict[str, torch.Tensor] = {}
    with _private_safetensors_snapshot(verified_bytes) as snapshot:
        with safe_open(str(snapshot), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing = sorted(set(requested).difference(available))
            if missing:
                raise ValueError(f"frozen tensor resource is missing keys: {missing}")
            # Do not iterate over available keys and never call get_tensor for a
            # non-requested key.  In particular, legacy target tensors stay closed.
            for key in requested:
                tensor = handle.get_tensor(key)
                if not isinstance(tensor, torch.Tensor) or tensor.requires_grad:
                    raise TypeError(f"frozen tensor {key!r} is not a detached torch tensor")
                tensor = tensor.detach().cpu().contiguous().clone()
                expected_tensor_sha = resource.tensor_sha256.get(key)
                if expected_tensor_sha is not None and _tensor_sha256(tensor) != expected_tensor_sha:
                    raise ValueError(f"frozen tensor receipt drifted: {key}")
                tensors[key] = tensor
    return tensors


def read_frozen_json_resource(
    registry: FrozenV29ResourceRegistry,
    resource_id: str,
) -> dict[str, Any]:
    if not isinstance(registry, FrozenV29ResourceRegistry):
        raise TypeError("registry must be FrozenV29ResourceRegistry")
    resource = _validate_resource_descriptor(registry, resource_id)
    if resource.format != "json" or resource.tensor_key_profile is not None:
        raise TypeError("requested frozen v29 resource is not JSON")
    verified_bytes = _read_verified_resource_bytes(resource)
    return _strict_json_bytes(verified_bytes, resource_id)


def _validate_probability_row_tensor(value: torch.Tensor, name: str) -> None:
    if value.dtype != torch.float32 or tuple(value.shape) != (PUBLIC_PATIENT_COUNT, 19):
        raise ValueError(f"{name} must be float32 [102,19]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if not torch.allclose(
        value.sum(dim=1), torch.ones(PUBLIC_PATIENT_COUNT), atol=1e-6, rtol=0
    ):
        raise ValueError(f"{name} rows must sum to one")
    if not torch.equal(value[:, PZ_INDEX], torch.zeros(PUBLIC_PATIENT_COUNT)):
        raise ValueError(f"{name} assigns probability to excluded PZ")


def load_public_v29_oof_reference(
    registry: FrozenV29ResourceRegistry,
) -> dict[str, torch.Tensor]:
    """Read the three public OOF probabilities and target-free routing tensors."""

    tensors = read_whitelisted_tensors(
        registry, PUBLIC_OOF_RESOURCE, PUBLIC_OOF_TENSOR_KEYS
    )
    mask = tensors["candidate_mask"]
    folds = tensors["patient_folds"]
    if mask.dtype != torch.bool or tuple(mask.shape) != (19,) or tuple(mask.tolist()) != V29_CANDIDATE_MASK:
        raise ValueError("public v29 candidate mask drifted")
    if folds.dtype != torch.int64 or tuple(folds.shape) != (PUBLIC_PATIENT_COUNT,):
        raise ValueError("public v29 fold tensor must be int64 [102]")
    if any(int(value) not in range(N_FOLDS) for value in folds.tolist()):
        raise ValueError("public v29 fold tensor contains an invalid fold")
    h = tensors["oof.h_only_probability"]
    direct = tensors["oof.rank1_direct_probability"]
    equal = tensors["oof.portable_equal_ensemble_probability"]
    for name, value in (("pH", h), ("pD", direct), ("p0", equal)):
        _validate_probability_row_tensor(value, name)
    if not torch.equal(equal, 0.5 * h + 0.5 * direct):
        raise ValueError("public v29 p0 is not the frozen equal-probability H/D fusion")
    return tensors


def _require_shape_dtype(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    context: str,
) -> None:
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(f"{context} must have dtype={dtype} shape={shape}")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"{context} contains non-finite values")


def load_v29_inference_states(
    registry: FrozenV29ResourceRegistry,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Selectively load only the D/H fold-state tensors needed for inference."""

    direct = read_whitelisted_tensors(
        registry, DIRECT_STATES_RESOURCE, DIRECT_STATE_TENSOR_KEYS
    )
    h_states = read_whitelisted_tensors(registry, H_STATES_RESOURCE, H_STATE_TENSOR_KEYS)
    expected_mask = torch.tensor(V29_CANDIDATE_MASK, dtype=torch.bool)
    for fold in range(N_FOLDS):
        direct_prefix = f"outer_state.fold{fold}."
        _require_shape_dtype(
            direct[direct_prefix + "candidate_mask"],
            shape=(19,),
            dtype=torch.bool,
            context=f"direct fold {fold} candidate mask",
        )
        if not torch.equal(direct[direct_prefix + "candidate_mask"], expected_mask):
            raise ValueError("direct fold candidate mask drifted")
        for suffix, shape in (
            ("phase_weights", (5,)),
            ("prior_logits", (19,)),
            ("tile_scorer.bias", (1,)),
            ("tile_scorer.weight", (1, 200)),
        ):
            _require_shape_dtype(
                direct[direct_prefix + suffix],
                shape=shape,
                dtype=torch.float32,
                context=f"direct fold {fold} {suffix}",
            )

        h_arm = f"outer{fold}.frozen_labram_only."
        transform = f"outer{fold}.transform."
        _require_shape_dtype(
            h_states[h_arm + "candidate_mask"],
            shape=(19,),
            dtype=torch.bool,
            context=f"H fold {fold} candidate mask",
        )
        if not torch.equal(h_states[h_arm + "candidate_mask"], expected_mask):
            raise ValueError("H fold candidate mask drifted")
        for key, shape in (
            (h_arm + "h_weight", (16,)),
            (h_arm + "prior_logits", (19,)),
            (transform + "h_center", (600,)),
            (transform + "h_components", (600, 16)),
            (transform + "h_pca_mean", (600,)),
            (transform + "h_scale", (600,)),
        ):
            _require_shape_dtype(
                h_states[key],
                shape=shape,
                dtype=torch.float32,
                context=key,
            )
    return direct, h_states


def validate_public_v29_manifest_metric_receipt(
    public_manifest: object,
) -> dict[str, Any]:
    """Replay the frozen public metric *receipt* without reading targets.

    This pure mapping validator cannot establish that its input came from the
    SHA-frozen manifest, so its returned source is deliberately neutral.  Use
    :func:`replay_public_v29_manifest_metric_receipt` for a frozen-source claim.
    """

    if not isinstance(public_manifest, Mapping):
        raise TypeError("public v29 manifest must be a mapping")
    metrics_root = public_manifest.get("metrics")
    if not isinstance(metrics_root, Mapping):
        raise ValueError("public v29 manifest lacks metrics")
    metrics = metrics_root.get("portable_equal_ensemble")
    if not isinstance(metrics, Mapping):
        raise ValueError("public v29 manifest lacks portable ensemble metrics")
    top1 = metrics.get("top1")
    ranking = metrics.get("ranking")
    if not isinstance(top1, Mapping) or not isinstance(ranking, Mapping):
        raise ValueError("public v29 manifest metric sections drifted")
    denominator = _CANONICAL_PUBLIC_METRICS["denominator"]
    if top1.get("n_samples") != denominator or ranking.get("n_patients") != denominator:
        raise ValueError("public v29 metric denominator drifted from 102 patients")
    hit_at_k = ranking.get("hit_at_k")
    if not isinstance(hit_at_k, Mapping):
        raise ValueError("public v29 ranking lacks hit_at_k receipt")

    observed = {
        "strict_value": top1.get("strict_accuracy"),
        "n4_value": top1.get("relaxed_accuracy"),
        "hit_at_3_value": hit_at_k.get("3"),
        "hit_at_5_value": hit_at_k.get("5"),
        "mrr": ranking.get("mean_reciprocal_rank"),
    }
    for key, value in observed.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"public v29 metric {key} must be finite")
        if float(value) != _CANONICAL_PUBLIC_METRICS[key]:
            raise ValueError(f"public v29 canonical metric drifted: {key}")

    hit_pairs = (
        ("strict_value", "strict_hits"),
        ("n4_value", "n4_hits"),
        ("hit_at_3_value", "hit_at_3_hits"),
        ("hit_at_5_value", "hit_at_5_hits"),
    )
    for value_key, hits_key in hit_pairs:
        replayed_hits = round(float(observed[value_key]) * denominator)
        if replayed_hits != _CANONICAL_PUBLIC_METRICS[hits_key]:
            raise ValueError(f"public v29 canonical hit count drifted: {hits_key}")
    return {
        "receipt_kind": PUBLIC_METRIC_MAPPING_VALIDATION_KIND,
        "source": "caller_supplied_mapping_untrusted",
        "frozen_source_verified": False,
        "denominator_patients": denominator,
        "strict_hits": _CANONICAL_PUBLIC_METRICS["strict_hits"],
        "n4_hits": _CANONICAL_PUBLIC_METRICS["n4_hits"],
        "hit_at_3_hits": _CANONICAL_PUBLIC_METRICS["hit_at_3_hits"],
        "hit_at_5_hits": _CANONICAL_PUBLIC_METRICS["hit_at_5_hits"],
        "mean_reciprocal_rank": _CANONICAL_PUBLIC_METRICS["mrr"],
        "independently_recomputed_from_targets": False,
        "targets_or_target_mask_read": False,
    }


def replay_public_v29_manifest_metric_receipt(
    registry: FrozenV29ResourceRegistry,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse and validate one safely read, SHA-frozen public manifest.

    Returns ``(manifest, receipt)`` so callers use the exact bytes that were
    hashed, rather than reopening the source path after verification.
    """

    trusted_registry = validate_frozen_v29_resource_registry(registry)
    resource = _validate_resource_descriptor(
        trusted_registry, PUBLIC_MANIFEST_RESOURCE
    )
    verified_bytes = _read_verified_resource_bytes(resource)
    manifest = _strict_json_bytes(verified_bytes, PUBLIC_MANIFEST_RESOURCE)
    neutral = validate_public_v29_manifest_metric_receipt(manifest)
    manifest_ref = build_raw_artifact_ref(
        verified_bytes,
        artifact_kind="v29_public_manifest",
        media_type="application/json",
        payload_schema_version="soz_labram_portable_equal_ensemble_public_oof_v29",
    )
    receipt = {
        **{
            key: value
            for key, value in neutral.items()
            if key not in {"receipt_kind", "source", "frozen_source_verified"}
        },
        "receipt_kind": PUBLIC_METRIC_RECEIPT_KIND,
        "source": "sha_frozen_public_manifest",
        "frozen_source_verified": True,
        "public_manifest_ref": manifest_ref,
        "resource_registry_projection_sha256": _canonical_json_sha256(
            _resource_registry_projection(trusted_registry)
        ),
    }
    return manifest, receipt


def _public_index_projection(
    public_index: PublicV29RosterIndex,
) -> dict[str, object]:
    if not isinstance(public_index, PublicV29RosterIndex):
        raise TypeError("public_index must be PublicV29RosterIndex")
    if (
        len(public_index.patient_ids) != PUBLIC_PATIENT_COUNT
        or len(set(public_index.patient_ids)) != PUBLIC_PATIENT_COUNT
        or set(public_index.by_patient) != set(public_index.patient_ids)
    ):
        raise ValueError("public index patient roster/key set drifted")
    rows: list[dict[str, object]] = []
    for patient_index, patient in enumerate(public_index.patient_ids):
        binding = public_index.by_patient.get(patient)
        if not isinstance(binding, PublicV29PatientBinding):
            raise ValueError("public index contains a missing or invalid patient binding")
        rows.append(
            {
                "patient_id": binding.patient_id,
                "patient_index": binding.patient_index,
                "held_out_fold": binding.held_out_fold,
                "source_role": binding.source_role,
                "access_role": binding.access_role,
                "route": binding.route,
                "order_position": patient_index,
            }
        )
    return {
        "patient_ids": list(public_index.patient_ids),
        "rows": rows,
        "fold_counts": {
            str(key): int(value)
            for key, value in sorted(public_index.fold_counts.items())
        },
        "source_role_counts": dict(sorted(public_index.source_role_counts.items())),
        "resource_config_sha256": public_index.resource_config_sha256,
        "resource_registry_projection_sha256": (
            public_index.resource_registry_projection_sha256
        ),
        "authority_sha256": public_index.authority_sha256,
    }


def _build_public_v29_roster_index_from_trusted_registry(
    registry: FrozenV29ResourceRegistry,
) -> PublicV29RosterIndex:
    """Join the frozen 102-patient order/folds to historical source roles."""

    public_manifest, _metric_receipt = replay_public_v29_manifest_metric_receipt(
        registry
    )
    if (
        public_manifest.get("schema_version")
        != "soz_labram_portable_equal_ensemble_public_oof_v29"
        or public_manifest.get("go") is not True
        or public_manifest.get("decision") != "AUTHORIZE_ONE_TARGET_BLIND_PRIVATE_RUN"
    ):
        raise ValueError("historical public v29 freeze manifest drifted")
    ensemble = public_manifest.get("ensemble")
    if not isinstance(ensemble, Mapping) or ensemble.get("fold_count") != N_FOLDS:
        raise ValueError("historical public v29 ensemble fold count drifted")

    h_manifest = read_frozen_json_resource(registry, H_MANIFEST_RESOURCE)
    if h_manifest.get("schema_version") != "soz_labram_identity_recovery_closed_replay_v16":
        raise ValueError("historical H-fold manifest schema drifted")
    patient_ids_raw = h_manifest.get("patient_ids")
    patient_folds_raw = h_manifest.get("patient_folds")
    if not isinstance(patient_ids_raw, list) or not isinstance(patient_folds_raw, list):
        raise ValueError("historical H-fold manifest lacks patient roster/folds")
    patient_ids = tuple(_patient_id(value) for value in patient_ids_raw)
    if len(patient_ids) != PUBLIC_PATIENT_COUNT or len(set(patient_ids)) != len(patient_ids):
        raise ValueError("historical public v29 roster must contain 102 unique patients")
    if len(patient_folds_raw) != PUBLIC_PATIENT_COUNT or any(
        isinstance(value, bool) or not isinstance(value, int) or value not in range(N_FOLDS)
        for value in patient_folds_raw
    ):
        raise ValueError("historical public v29 fold roster drifted")
    folds = tuple(int(value) for value in patient_folds_raw)

    fold_tensor = read_whitelisted_tensors(
        registry, PUBLIC_OOF_RESOURCE, ("patient_folds",)
    )["patient_folds"]
    if fold_tensor.dtype != torch.int64 or tuple(fold_tensor.tolist()) != folds:
        raise ValueError("public OOF fold tensor disagrees with H-fold roster")

    union = read_frozen_json_resource(registry, PUBLIC_UNION_RESOURCE)
    if union.get("schema_version") != "soz_public_development_union_identity_v12":
        raise ValueError("public union manifest schema drifted")
    union_patients = union.get("patients")
    if not isinstance(union_patients, list):
        raise ValueError("public union manifest lacks patient rows")
    by_union_id: dict[str, Mapping[str, object]] = {}
    for row in union_patients:
        if not isinstance(row, Mapping):
            raise ValueError("public union patient row must be an object")
        patient = _patient_id(row.get("patient_id"))
        if patient in by_union_id:
            raise ValueError("public union manifest repeats a patient")
        by_union_id[patient] = row

    bindings: dict[str, PublicV29PatientBinding] = {}
    for patient_index, (patient, held_out_fold) in enumerate(zip(patient_ids, folds)):
        row = by_union_id.get(patient)
        if row is None:
            raise ValueError("historical public v29 patient is missing from union manifest")
        if row.get("outer_fold") != held_out_fold:
            raise ValueError("public union/H manifest held-out fold mismatch")
        source_role = row.get("legacy_model_split")
        if source_role not in _PUBLIC_SOURCE_ROLES:
            raise ValueError("public patient has an unsupported historical source role")
        access_role = EVALUATOR_ONLY if source_role == "source_eval" else DEVELOPMENT_ONLY
        bindings[patient] = PublicV29PatientBinding(
            patient_id=patient,
            patient_index=patient_index,
            held_out_fold=held_out_fold,
            source_role=source_role,
            access_role=access_role,
        )

    fold_counts = Counter(folds)
    source_counts = Counter(binding.source_role for binding in bindings.values())
    if dict(sorted(fold_counts.items())) != {0: 20, 1: 22, 2: 20, 3: 21, 4: 19}:
        raise ValueError("historical public v29 fold counts drifted")
    if dict(source_counts) != {"source_dev": 15, "source_train": 66, "source_eval": 21}:
        raise ValueError("historical public v29 source-role counts drifted")
    registry_projection_sha = _canonical_json_sha256(
        _resource_registry_projection(registry)
    )
    authority_source = {
        "domain": "evisoz_v29_public_roster_authority_v1",
        "patient_ids": list(patient_ids),
        "rows": [
            {
                "patient_id": patient,
                "patient_index": binding.patient_index,
                "held_out_fold": binding.held_out_fold,
                "source_role": binding.source_role,
                "access_role": binding.access_role,
            }
            for patient, binding in bindings.items()
        ],
        "fold_counts": {
            str(key): int(value) for key, value in sorted(fold_counts.items())
        },
        "source_role_counts": dict(sorted(source_counts.items())),
        "resource_config_sha256": registry.config_sha256,
        "resource_registry_projection_sha256": registry_projection_sha,
    }
    return PublicV29RosterIndex(
        patient_ids=patient_ids,
        by_patient=MappingProxyType(bindings),
        fold_counts=MappingProxyType(dict(sorted(fold_counts.items()))),
        source_role_counts=MappingProxyType(
            {role: source_counts[role] for role in ("source_train", "source_dev", "source_eval")}
        ),
        resource_config_sha256=registry.config_sha256,
        resource_registry_projection_sha256=registry_projection_sha,
        authority_sha256=_canonical_json_sha256(authority_source),
    )


def build_public_v29_roster_index(
    registry: FrozenV29ResourceRegistry,
) -> PublicV29RosterIndex:
    trusted_registry = validate_frozen_v29_resource_registry(registry)
    return _build_public_v29_roster_index_from_trusted_registry(trusted_registry)


def validate_public_v29_roster_index(
    public_index: PublicV29RosterIndex,
    registry: FrozenV29ResourceRegistry,
) -> PublicV29RosterIndex:
    """Rebuild the authoritative roster and reject forged or cross-bundle indexes."""

    if not isinstance(public_index, PublicV29RosterIndex):
        raise TypeError("public_index must be PublicV29RosterIndex")
    trusted_registry = validate_frozen_v29_resource_registry(registry)
    expected = _build_public_v29_roster_index_from_trusted_registry(trusted_registry)
    if _public_index_projection(public_index) != _public_index_projection(expected):
        raise ValueError("public v29 roster index does not bind the frozen registry")
    return expected


def build_frozen_public_member_relation(
    identity: V29PatientIdentity,
    public_index: PublicV29RosterIndex,
    registry: FrozenV29ResourceRegistry,
) -> V29PublicRosterRelation:
    if not isinstance(identity, V29PatientIdentity):
        raise TypeError("identity must be V29PatientIdentity")
    if _identity_namespace(identity.namespace) != FROZEN_PUBLIC_IDENTITY_NAMESPACE:
        raise ValueError("public member relation requires the frozen public namespace")
    trusted_index = validate_public_v29_roster_index(public_index, registry)
    trusted_index.require(_patient_id(identity.patient_id))
    identity_sha = v29_patient_identity_sha256(identity)
    proof_sha = trusted_index.authority_sha256
    relation = V29PublicRosterRelation(
        identity_sha256=identity_sha,
        state=FROZEN_MEMBER_RELATION,
        proof_kind=FROZEN_MEMBER_PROOF_KIND,
        proof_sha256=proof_sha,
        proof_ref=None,
        relation_sha256=_relation_sha256(
            identity_sha256=identity_sha,
            state=FROZEN_MEMBER_RELATION,
            proof_kind=FROZEN_MEMBER_PROOF_KIND,
            proof_sha256=proof_sha,
            proof_ref=None,
        ),
    )
    return _validate_public_roster_relation(relation, identity=identity)


def _resolve_v29_route_with_trusted_index(
    identity: V29PatientIdentity,
    public_roster_relation: V29PublicRosterRelation,
    public_index: PublicV29RosterIndex,
    *,
    requested_route: str | None,
    official_source_role: str | None,
) -> V29RouteDecision:
    if not isinstance(identity, V29PatientIdentity):
        raise TypeError("identity must be V29PatientIdentity")
    namespace = _identity_namespace(identity.namespace)
    patient = _patient_id(identity.patient_id)
    identity_sha = v29_patient_identity_sha256(identity)
    relation = _validate_public_roster_relation(
        public_roster_relation,
        identity=identity,
    )
    if requested_route is not None and requested_route not in {
        HISTORICAL_PUBLIC_ROUTE,
        FROZEN_FIVE_FOLD_ROUTE,
    }:
        raise ValueError("requested_route is not a frozen v29 route")
    if official_source_role is not None and official_source_role not in _PUBLIC_SOURCE_ROLES:
        raise ValueError("official_source_role must be source_train/source_dev/source_eval")

    if namespace == FROZEN_PUBLIC_IDENTITY_NAMESPACE:
        if relation.state != FROZEN_MEMBER_RELATION:
            raise PermissionError(
                "frozen public identities require an authoritative member relation"
            )
        binding = public_index.by_patient.get(patient)
        if binding is None:
            raise PermissionError("frozen public identity is not a roster member")
        if relation.proof_sha256 != public_index.authority_sha256:
            raise ValueError("public member relation is bound to another roster authority")
        if requested_route not in (None, HISTORICAL_PUBLIC_ROUTE):
            raise PermissionError(
                "historical public patients may only use their held-fold OOF route"
            )
        if official_source_role is not None and official_source_role != binding.source_role:
            raise PermissionError("caller source role disagrees with frozen public roster")
        return V29RouteDecision(
            patient_id=patient,
            identity_namespace=namespace,
            identity_sha256=identity_sha,
            public_roster_relation=relation.state,
            public_roster_relation_proof_kind=str(relation.proof_kind),
            public_roster_relation_proof_sha256=str(relation.proof_sha256),
            public_roster_relation_proof_ref=None,
            public_roster_relation_sha256=relation.relation_sha256,
            route=HISTORICAL_PUBLIC_ROUTE,
            unit_kind="patient",
            fold_indices=(binding.held_out_fold,),
            public_patient_index=binding.patient_index,
            historical_source_role=binding.source_role,
            access_role=binding.access_role,
            historical_development_eligible=not binding.evaluator_only,
            # A historical source role is descriptive provenance, not an
            # EviSOZ split/field authorization.  This layer never opens loss.
            route_layer_training_authorized=False,
        )

    if relation.state != PROVEN_ABSENT_RELATION:
        raise PermissionError(
            "non-public namespaces require a caller-proven absent relation"
        )
    if requested_route not in (None, FROZEN_FIVE_FOLD_ROUTE):
        raise PermissionError(
            "patients outside the historical public roster may only use five-fold event mean"
        )
    if official_source_role == "source_eval":
        access_role = EVALUATOR_ONLY
    elif official_source_role in ("source_train", "source_dev"):
        access_role = DEVELOPMENT_ONLY
    else:
        # An unregistered deployment patient may be inferred, but absent a
        # split receipt it is not silently authorized for training.
        access_role = INFERENCE_ONLY
    return V29RouteDecision(
        patient_id=patient,
        identity_namespace=namespace,
        identity_sha256=identity_sha,
        public_roster_relation=relation.state,
        public_roster_relation_proof_kind=str(relation.proof_kind),
        public_roster_relation_proof_sha256=str(relation.proof_sha256),
        public_roster_relation_proof_ref=deepcopy(dict(relation.proof_ref)),
        public_roster_relation_sha256=relation.relation_sha256,
        route=FROZEN_FIVE_FOLD_ROUTE,
        unit_kind="event",
        fold_indices=tuple(range(N_FOLDS)),
        public_patient_index=None,
        historical_source_role=official_source_role,
        access_role=access_role,
        historical_development_eligible=access_role == DEVELOPMENT_ONLY,
        # Final loss access requires the independently trusted EviSOZ split
        # assignment intersected with the field release.
        route_layer_training_authorized=False,
    )


def resolve_v29_route(
    identity: V29PatientIdentity,
    public_index: PublicV29RosterIndex,
    registry: FrozenV29ResourceRegistry,
    *,
    public_roster_relation: V29PublicRosterRelation,
    requested_route: str | None = None,
    official_source_role: str | None = None,
) -> V29RouteDecision:
    """Resolve only from a typed identity and an explicit roster relation."""

    trusted_index = validate_public_v29_roster_index(public_index, registry)
    return _resolve_v29_route_with_trusted_index(
        identity,
        public_roster_relation,
        trusted_index,
        requested_route=requested_route,
        official_source_role=official_source_role,
    )


def validate_v29_route_decision(
    decision: V29RouteDecision,
    public_index: PublicV29RosterIndex,
    registry: FrozenV29ResourceRegistry,
) -> V29RouteDecision:
    """Replay a decision against the same frozen registry/index authority."""

    if not isinstance(decision, V29RouteDecision):
        raise TypeError("decision must be V29RouteDecision")
    trusted_index = validate_public_v29_roster_index(public_index, registry)
    identity = build_v29_patient_identity(
        namespace=decision.identity_namespace,
        patient_id=decision.patient_id,
    )
    relation = V29PublicRosterRelation(
        identity_sha256=decision.identity_sha256,
        state=decision.public_roster_relation,
        proof_kind=decision.public_roster_relation_proof_kind,
        proof_sha256=decision.public_roster_relation_proof_sha256,
        proof_ref=decision.public_roster_relation_proof_ref,
        relation_sha256=decision.public_roster_relation_sha256,
    )
    expected = _resolve_v29_route_with_trusted_index(
        identity,
        relation,
        trusted_index,
        requested_route=decision.route,
        official_source_role=decision.historical_source_role,
    )
    if decision != expected:
        raise ValueError("v29 route decision does not replay from frozen authority")
    return expected


def replay_public_oof_rows(
    registry: FrozenV29ResourceRegistry,
    public_index: PublicV29RosterIndex,
    identities: Sequence[V29PatientIdentity],
    public_roster_relations: Sequence[V29PublicRosterRelation],
) -> dict[str, torch.Tensor]:
    """Select frozen public patient rows after enforcing held-fold routing."""

    if isinstance(identities, (str, bytes)) or not isinstance(identities, Sequence):
        raise TypeError("identities must be a sequence")
    if isinstance(public_roster_relations, (str, bytes)) or not isinstance(
        public_roster_relations, Sequence
    ):
        raise TypeError("public_roster_relations must be a sequence")
    if not identities or len(identities) != len(public_roster_relations):
        raise ValueError("identities and public roster relations must align")
    identity_hashes = tuple(v29_patient_identity_sha256(value) for value in identities)
    if len(set(identity_hashes)) != len(identity_hashes):
        raise ValueError("public replay identities must be unique")
    trusted_index = validate_public_v29_roster_index(public_index, registry)
    decisions = [
        _resolve_v29_route_with_trusted_index(
            identity,
            relation,
            trusted_index,
            requested_route=HISTORICAL_PUBLIC_ROUTE,
            official_source_role=None,
        )
        for identity, relation in zip(identities, public_roster_relations)
    ]
    rows = torch.tensor(
        [decision.public_patient_index for decision in decisions], dtype=torch.long
    )
    reference = load_public_v29_oof_reference(registry)
    return {
        "p_h_node": reference["oof.h_only_probability"].index_select(0, rows),
        "p_d_node": reference["oof.rank1_direct_probability"].index_select(0, rows),
        "p0_node": reference["oof.portable_equal_ensemble_probability"].index_select(0, rows),
        "candidate_mask_node": reference["candidate_mask"].clone(),
        "held_out_folds": reference["patient_folds"].index_select(0, rows),
    }


__all__ = [
    "RESOURCE_CONFIG_SCHEMA_VERSION",
    "METHOD_ID",
    "HISTORICAL_PUBLIC_ROUTE",
    "FROZEN_FIVE_FOLD_ROUTE",
    "EVALUATOR_ONLY",
    "DEVELOPMENT_ONLY",
    "INFERENCE_ONLY",
    "N_FOLDS",
    "PUBLIC_PATIENT_COUNT",
    "PUBLIC_METRIC_RECEIPT_KIND",
    "PUBLIC_METRIC_MAPPING_VALIDATION_KIND",
    "FROZEN_PUBLIC_IDENTITY_NAMESPACE",
    "FROZEN_MEMBER_RELATION",
    "PROVEN_ABSENT_RELATION",
    "UNKNOWN_RELATION",
    "FROZEN_MEMBER_PROOF_KIND",
    "CALLER_ABSENCE_PROOF_KIND",
    "P0_LINKAGE_PROOF_ARTIFACT_KIND",
    "P0_LINKAGE_PROOF_SCHEMA_VERSION",
    "STANDARD_19",
    "V29_CANDIDATE_MASK",
    "PUBLIC_OOF_TENSOR_KEYS",
    "DIRECT_STATE_TENSOR_KEYS",
    "H_STATE_TENSOR_KEYS",
    "FrozenV29Resource",
    "FrozenV29ResourceRegistry",
    "PublicV29PatientBinding",
    "PublicV29RosterIndex",
    "V29PatientIdentity",
    "V29PublicRosterRelation",
    "V29RouteDecision",
    "build_v29_patient_identity",
    "v29_patient_identity_sha256",
    "build_proven_absent_public_roster_relation",
    "build_unknown_public_roster_relation",
    "build_frozen_public_member_relation",
    "load_frozen_v29_resource_registry",
    "validate_frozen_v29_resource_registry",
    "read_whitelisted_tensors",
    "read_frozen_json_resource",
    "load_public_v29_oof_reference",
    "load_v29_inference_states",
    "validate_public_v29_manifest_metric_receipt",
    "replay_public_v29_manifest_metric_receipt",
    "build_public_v29_roster_index",
    "validate_public_v29_roster_index",
    "resolve_v29_route",
    "validate_v29_route_decision",
    "replay_public_oof_rows",
]
