"""Fail-closed compact-inventory gate for detector selection-fit references.

The legacy detector reference authority remains byte-compatible so the five
already sealed selection-fit receipts can still be replayed.  Formal new
selection-fit execution must enter through this additive gate.  The gate
requires the process-local, actual-outcome-replayed compact provider inventory
and validates its complete source-train by four-variant Cartesian binding
before it invokes any reference-reading function.

Serialized inventory JSON, a manifest, a receipt, or a bare digest is never an
input substitute.  A second entry point upgrades an already serialized legacy
selection-fit receipt by first validating the opaque inventory and then
replaying the original reference bytes without changing the legacy receipt.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

from . import detector_fold_reference_authority_v1 as _reference
from . import detector_provider_pre_reference_inventory_v1 as _inventory
from . import eventnet_cleanroom_registry_v1 as _eventnet
from . import seizuretransformer_cleanroom_registry_v1 as _st
from .tusz_detector_cleanroom_fold_plan_v1 import (
    validate_tusz_detector_cleanroom_fold_plan_v1,
)


GATE_REGISTRY_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_detector_pre_reference_inventory_selection_gate_registry_v1"
GATE_AUTHORITY_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_detector_pre_reference_inventory_selection_gate_authority_v1"
GATE_REGISTRY_ID: Final[
    str
] = "CLINICAL-EEG-DETECTOR-PRE-REFERENCE-INVENTORY-SELECTION-GATE-V1-20260824"
GATE_METHOD_ID: Final[
    str
] = "opaque_complete_cartesian_inventory_before_first_selection_reference_byte_v1"
GATE_MODULE_RELATIVE_PATH: Final[str] = (
    "src/clinical_eeg_long_recording/"
    "detector_pre_reference_inventory_phase_gate_v1.py"
)
INVENTORY_MODULE_RELATIVE_PATH: Final[str] = (
    "src/clinical_eeg_long_recording/" "detector_provider_pre_reference_inventory_v1.py"
)
REFERENCE_MODULE_RELATIVE_PATH: Final[
    str
] = "src/clinical_eeg_long_recording/detector_fold_reference_authority_v1.py"
DEFAULT_GATE_REGISTRY_RELATIVE_PATH: Final[
    str
] = "configs/clinical_eeg_detector_pre_reference_inventory_selection_gate_v1.json"
DEFAULT_FOLD_PLAN_RELATIVE_PATH: Final[str] = (
    "outputs/tusz_canonical_physical_signal_audit_v1_full_20260824r2/"
    "detector_cleanroom_fold_plan.json"
)
DEFAULT_REFERENCE_REGISTRY_RELATIVE_PATH: Final[
    str
] = "configs/clinical_eeg_detector_fold_reference_authority_registry_v1.json"

_CONTENT_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
_SHA256_CHARS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_GATE_AUTHORITY_SEAL = object()


class AuthorizedInventoryGatedDetectorSelectionFitPhaseV1:
    """Opaque inventory-plus-reference authority for one selection-fit phase."""

    __slots__ = (
        "__inventory_authority",
        "__phase_authority",
        "__receipt_json",
        "__issuer_seal",
    )

    def __init__(
        self,
        *,
        inventory_authority: (
            _inventory.AuthorizedDetectorProviderPreReferenceInventoryV1
        ),
        phase_authority: _reference.ValidatedDetectorFoldReferencePhaseAuthorityV1,
        receipt: Mapping[str, Any],
        _issuer_seal: object,
    ) -> None:
        if _issuer_seal is not _GATE_AUTHORITY_SEAL:
            raise PermissionError("inventory-gated phase has no valid issuer seal")
        self.__inventory_authority = inventory_authority
        self.__phase_authority = phase_authority
        self.__receipt_json = _canonical_json_bytes(receipt).decode("utf-8")
        self.__issuer_seal = _issuer_seal

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self.__receipt_json)

    def _authorities(
        self,
    ) -> tuple[
        _inventory.AuthorizedDetectorProviderPreReferenceInventoryV1,
        _reference.ValidatedDetectorFoldReferencePhaseAuthorityV1,
    ]:
        return self.__inventory_authority, self.__phase_authority

    def _has_valid_issuer_seal(self) -> bool:
        return self.__issuer_seal is _GATE_AUTHORITY_SEAL


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detector_pre_reference_inventory_phase_gate_source_sha256_v1() -> str:
    return _file_sha256(Path(__file__).resolve(strict=True))


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_CHARS)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a normalized non-empty string")
    return value


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    if result.get("receipt_sha256") != _CONTENT_PENDING:
        raise ValueError("content-addressed gate object must begin pending")
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _validate_content_address(
    value: object, *, required: set[str], context: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"{context} fields drifted")
    result = deepcopy(value)
    supplied = _require_sha256(result["receipt_sha256"], f"{context} receipt")
    result["receipt_sha256"] = _CONTENT_PENDING
    if supplied != _canonical_sha256(result):
        raise ValueError(f"{context} is not content-addressed")
    result["receipt_sha256"] = supplied
    return result


def _path_binding(path: str, file_sha256: str) -> dict[str, Any]:
    return {
        "path": _identifier(path, "bound path"),
        "file_sha256": _require_sha256(file_sha256, "bound file"),
    }


def _gate_contract() -> dict[str, Any]:
    return {
        "formal_new_selection_fit_entrypoint": (
            "materialize_inventory_gated_detector_selection_fit_phase_v1"
        ),
        "legacy_receipt_upgrade_entrypoint": (
            "authorize_inventory_gated_detector_selection_fit_phase_receipt_v1"
        ),
        "inventory_validation_logical_sequence": 1,
        "reference_opener_invocation_logical_sequence": 2,
        "opaque_actual_outcome_replayed_inventory_required": True,
        "serialized_inventory_manifest_JSONL_receipt_or_digest_accepted": False,
        "same_fold_plan_receipt_and_source_train_roster_required": True,
        "complete_source_train_by_four_variant_cartesian_required": True,
        "inventory_revalidated_before_every_formal_gate_consumption": True,
        "legacy_detector_reference_module_or_receipt_bytes_modified": False,
        "old_ungated_entrypoint_removed": False,
        "old_ungated_entrypoint_is_formal_new_execution_path": False,
    }


def build_detector_pre_reference_inventory_selection_gate_registry_v1(
    *,
    fold_plan: Mapping[str, Any],
    fold_plan_path: str,
    fold_plan_file_sha256: str,
    detector_reference_registry: Mapping[str, Any],
    detector_reference_registry_path: str,
    detector_reference_registry_file_sha256: str,
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic registry without opening any reference sidecar."""

    plan = validate_tusz_detector_cleanroom_fold_plan_v1(fold_plan)
    reference_registry = (
        _reference.validate_detector_fold_reference_authority_registry_v1(
            detector_reference_registry,
            fold_plan=plan,
            verify_bound_files=False,
        )
    )
    canonical_eventnet = _eventnet._require_canonical_eventnet_registry(
        eventnet_registry
    )
    canonical_st = _st._require_canonical_seizuretransformer_registry(
        seizuretransformer_registry
    )
    return _content_address(
        {
            "schema_version": GATE_REGISTRY_SCHEMA_VERSION,
            "registry_id": GATE_REGISTRY_ID,
            "status": "additive_formal_selection_fit_fail_closed_v1",
            "method_id": GATE_METHOD_ID,
            "gate_implementation_binding": {
                "path": GATE_MODULE_RELATIVE_PATH,
                "source_sha256": (
                    detector_pre_reference_inventory_phase_gate_source_sha256_v1()
                ),
            },
            "compact_inventory_binding": {
                "path": INVENTORY_MODULE_RELATIVE_PATH,
                "source_sha256": (
                    _inventory.detector_provider_pre_reference_inventory_source_sha256_v1()
                ),
                "inventory_schema_version": _inventory.INVENTORY_SCHEMA_VERSION,
                "authority_schema_version": (
                    _inventory.INVENTORY_AUTHORITY_SCHEMA_VERSION
                ),
                "compact_outcome_schema_version": (
                    _inventory.COMPACT_OUTCOME_SCHEMA_VERSION
                ),
                "materializer_id": _inventory.MATERIALIZER_ID,
                "variant_ids": list(_inventory.PROVIDER_VARIANTS_V1),
            },
            "fold_plan_binding": {
                **_path_binding(fold_plan_path, fold_plan_file_sha256),
                "plan_id": plan["plan_id"],
                "plan_receipt_sha256": plan["receipt_sha256"],
                "source_train_record_count": plan["source_split_rosters"][
                    "source_train"
                ]["recording_count"],
                "source_train_analysis_identity_roster_sha256": plan[
                    "source_split_rosters"
                ]["source_train"]["analysis_identity_roster_sha256"],
            },
            "detector_reference_authority_binding": {
                **_path_binding(
                    detector_reference_registry_path,
                    detector_reference_registry_file_sha256,
                ),
                "registry_id": reference_registry["registry_id"],
                "registry_receipt_sha256": reference_registry[
                    "registry_receipt_sha256"
                ],
                "module_path": REFERENCE_MODULE_RELATIVE_PATH,
                "module_source_sha256": (
                    _reference.detector_reference_authority_source_sha256_v1()
                ),
                "legacy_materializer_entrypoint": (
                    "materialize_detector_fold_reference_phase_v1"
                ),
                "legacy_receipt_replay_entrypoint": (
                    "authorize_detector_fold_reference_phase_receipt_v1"
                ),
            },
            "provider_registry_bindings": _inventory._registry_bindings(
                canonical_eventnet, canonical_st
            ),
            "gate_contract": _gate_contract(),
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def validate_detector_pre_reference_inventory_selection_gate_registry_v1(
    value: Mapping[str, Any],
    *,
    fold_plan: Mapping[str, Any],
    detector_reference_registry: Mapping[str, Any],
    eventnet_registry: Mapping[str, Any],
    seizuretransformer_registry: Mapping[str, Any],
) -> dict[str, Any]:
    plan = validate_tusz_detector_cleanroom_fold_plan_v1(fold_plan)
    reference_registry = (
        _reference.validate_detector_fold_reference_authority_registry_v1(
            detector_reference_registry,
            fold_plan=plan,
            verify_bound_files=False,
        )
    )
    canonical_eventnet = _eventnet._require_canonical_eventnet_registry(
        eventnet_registry
    )
    canonical_st = _st._require_canonical_seizuretransformer_registry(
        seizuretransformer_registry
    )
    required = {
        "schema_version",
        "registry_id",
        "status",
        "method_id",
        "gate_implementation_binding",
        "compact_inventory_binding",
        "fold_plan_binding",
        "detector_reference_authority_binding",
        "provider_registry_bindings",
        "gate_contract",
        "receipt_sha256",
    }
    registry = _validate_content_address(
        dict(value), required=required, context="inventory selection gate registry"
    )
    gate_binding = registry["gate_implementation_binding"]
    inventory_binding = registry["compact_inventory_binding"]
    plan_binding = registry["fold_plan_binding"]
    reference_binding = registry["detector_reference_authority_binding"]
    if (
        registry["schema_version"] != GATE_REGISTRY_SCHEMA_VERSION
        or registry["registry_id"] != GATE_REGISTRY_ID
        or registry["status"] != "additive_formal_selection_fit_fail_closed_v1"
        or registry["method_id"] != GATE_METHOD_ID
        or gate_binding
        != {
            "path": GATE_MODULE_RELATIVE_PATH,
            "source_sha256": (
                detector_pre_reference_inventory_phase_gate_source_sha256_v1()
            ),
        }
        or inventory_binding
        != {
            "path": INVENTORY_MODULE_RELATIVE_PATH,
            "source_sha256": (
                _inventory.detector_provider_pre_reference_inventory_source_sha256_v1()
            ),
            "inventory_schema_version": _inventory.INVENTORY_SCHEMA_VERSION,
            "authority_schema_version": _inventory.INVENTORY_AUTHORITY_SCHEMA_VERSION,
            "compact_outcome_schema_version": (
                _inventory.COMPACT_OUTCOME_SCHEMA_VERSION
            ),
            "materializer_id": _inventory.MATERIALIZER_ID,
            "variant_ids": list(_inventory.PROVIDER_VARIANTS_V1),
        }
        or plan_binding.get("plan_id") != plan["plan_id"]
        or plan_binding.get("plan_receipt_sha256") != plan["receipt_sha256"]
        or plan_binding.get("source_train_record_count")
        != plan["source_split_rosters"]["source_train"]["recording_count"]
        or plan_binding.get("source_train_analysis_identity_roster_sha256")
        != plan["source_split_rosters"]["source_train"][
            "analysis_identity_roster_sha256"
        ]
        or reference_binding.get("registry_id") != reference_registry["registry_id"]
        or reference_binding.get("registry_receipt_sha256")
        != reference_registry["registry_receipt_sha256"]
        or reference_binding.get("module_path") != REFERENCE_MODULE_RELATIVE_PATH
        or reference_binding.get("module_source_sha256")
        != _reference.detector_reference_authority_source_sha256_v1()
        or reference_binding.get("legacy_materializer_entrypoint")
        != "materialize_detector_fold_reference_phase_v1"
        or reference_binding.get("legacy_receipt_replay_entrypoint")
        != "authorize_detector_fold_reference_phase_receipt_v1"
        or registry["provider_registry_bindings"]
        != _inventory._registry_bindings(canonical_eventnet, canonical_st)
        or registry["gate_contract"] != _gate_contract()
    ):
        raise ValueError("inventory selection gate registry binding drifted")
    for binding, context in (
        (plan_binding, "gate fold-plan file"),
        (reference_binding, "gate reference-registry file"),
    ):
        if type(binding) is not dict:
            raise ValueError(f"{context} binding is malformed")
        _identifier(binding.get("path"), f"{context} path")
        _require_sha256(binding.get("file_sha256"), context)
    return registry


def _read_json_file(path: Path, context: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{context} is unavailable")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is invalid JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{context} must contain an object")
    return value, hashlib.sha256(payload).hexdigest()


def _canonical_provider_registries() -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    eventnet_registry = _eventnet.load_registry(
        project_root / _eventnet.CONFIG_RELATIVE_PATH
    )
    st_registry = _st.load_registry(project_root / _st.CONFIG_RELATIVE_PATH)
    return (
        _eventnet._require_canonical_eventnet_registry(eventnet_registry),
        _st._require_canonical_seizuretransformer_registry(st_registry),
    )


def build_default_detector_pre_reference_inventory_selection_gate_registry_v1() -> (
    dict[str, Any]
):
    """Rebuild the checked-in gate registry from its exact bound files."""

    project_root = Path(__file__).resolve().parents[2]
    plan_path = project_root / DEFAULT_FOLD_PLAN_RELATIVE_PATH
    reference_registry_path = project_root / DEFAULT_REFERENCE_REGISTRY_RELATIVE_PATH
    plan, plan_file_sha256 = _read_json_file(plan_path, "canonical fold plan")
    reference_registry, reference_registry_file_sha256 = _read_json_file(
        reference_registry_path, "canonical detector reference registry"
    )
    eventnet_registry, st_registry = _canonical_provider_registries()
    return build_detector_pre_reference_inventory_selection_gate_registry_v1(
        fold_plan=plan,
        fold_plan_path=DEFAULT_FOLD_PLAN_RELATIVE_PATH,
        fold_plan_file_sha256=plan_file_sha256,
        detector_reference_registry=reference_registry,
        detector_reference_registry_path=DEFAULT_REFERENCE_REGISTRY_RELATIVE_PATH,
        detector_reference_registry_file_sha256=(reference_registry_file_sha256),
        eventnet_registry=eventnet_registry,
        seizuretransformer_registry=st_registry,
    )


def load_default_detector_pre_reference_inventory_selection_gate_registry_v1() -> (
    dict[str, Any]
):
    """Require the checked-in registry to match all current source bytes."""

    project_root = Path(__file__).resolve().parents[2]
    path = project_root / DEFAULT_GATE_REGISTRY_RELATIVE_PATH
    observed, _file_hash = _read_json_file(path, "canonical inventory gate registry")
    expected = (
        build_default_detector_pre_reference_inventory_selection_gate_registry_v1()
    )
    if observed != expected:
        raise PermissionError(
            "checked-in inventory selection gate registry or source binding drifted"
        )
    return expected


def _admit_pre_reference_context(
    inventory_authority: object,
    *,
    fold_plan: Mapping[str, Any],
    detector_reference_registry: Mapping[str, Any],
) -> tuple[
    _inventory.AuthorizedDetectorProviderPreReferenceInventoryV1,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    # This must remain the first authority-consuming operation.  It also
    # replays the two compact bundle files before any reference-root Path is
    # constructed or any reference materializer is invoked.
    admitted_inventory = (
        _inventory.require_authorized_detector_provider_pre_reference_inventory_v1(
            inventory_authority
        )
    )
    manifest, rows, inventory_receipt = _inventory._require_inventory(
        admitted_inventory
    )
    plan = validate_tusz_detector_cleanroom_fold_plan_v1(fold_plan)
    reference_registry = (
        _reference.validate_detector_fold_reference_authority_registry_v1(
            detector_reference_registry,
            fold_plan=plan,
            verify_bound_files=False,
        )
    )
    eventnet_registry, st_registry = _canonical_provider_registries()
    gate_registry = (
        load_default_detector_pre_reference_inventory_selection_gate_registry_v1()
    )
    gate_registry = (
        validate_detector_pre_reference_inventory_selection_gate_registry_v1(
            gate_registry,
            fold_plan=plan,
            detector_reference_registry=reference_registry,
            eventnet_registry=eventnet_registry,
            seizuretransformer_registry=st_registry,
        )
    )
    expected_plan_binding = gate_registry["fold_plan_binding"]
    expected_inventory_binding = gate_registry["compact_inventory_binding"]
    if (
        manifest.get("schema_version")
        != expected_inventory_binding["inventory_schema_version"]
        or manifest.get("materializer_id")
        != expected_inventory_binding["materializer_id"]
        or manifest.get("materializer_source_sha256")
        != expected_inventory_binding["source_sha256"]
        or manifest.get("fold_plan_id") != expected_plan_binding["plan_id"]
        or manifest.get("fold_plan_receipt_sha256")
        != expected_plan_binding["plan_receipt_sha256"]
        or manifest.get("source_train_record_count")
        != expected_plan_binding["source_train_record_count"]
        or manifest.get("source_train_analysis_identity_roster_sha256")
        != expected_plan_binding["source_train_analysis_identity_roster_sha256"]
        or manifest.get("variant_ids") != expected_inventory_binding["variant_ids"]
        or manifest.get("cartesian_outcome_count")
        != expected_plan_binding["source_train_record_count"]
        * len(expected_inventory_binding["variant_ids"])
        or manifest.get("expected_cartesian_outcome_count")
        != manifest.get("cartesian_outcome_count")
        or len(rows) != manifest.get("cartesian_outcome_count")
        or manifest.get("complete_source_train_by_four_variant_cartesian") is not True
        or manifest.get("provider_registry_bindings")
        != gate_registry["provider_registry_bindings"]
        or inventory_receipt.get("actual_process_sealed_outcome_objects_replayed")
        is not True
        or inventory_receipt.get("manifest_receipt_sha256")
        != manifest.get("receipt_sha256")
    ):
        raise PermissionError(
            "selection reference gate requires the complete same-plan opaque inventory"
        )
    if (
        reference_registry["registry_receipt_sha256"]
        != gate_registry["detector_reference_authority_binding"][
            "registry_receipt_sha256"
        ]
    ):
        raise PermissionError(
            "selection reference authority registry differs from gate"
        )
    return (
        admitted_inventory,
        manifest,
        inventory_receipt,
        plan,
        reference_registry,
    )


def _gate_authority_receipt(
    *,
    mode: str,
    inventory_manifest: Mapping[str, Any],
    inventory_receipt: Mapping[str, Any],
    phase_receipt: Mapping[str, Any],
    legacy_receipt_replayed_unchanged: bool,
) -> dict[str, Any]:
    gate_registry = (
        load_default_detector_pre_reference_inventory_selection_gate_registry_v1()
    )
    return _content_address(
        {
            "schema_version": GATE_AUTHORITY_SCHEMA_VERSION,
            "registry_id": gate_registry["registry_id"],
            "registry_receipt_sha256": gate_registry["receipt_sha256"],
            "method_id": GATE_METHOD_ID,
            "gate_source_sha256": (
                detector_pre_reference_inventory_phase_gate_source_sha256_v1()
            ),
            "issuance_mode": mode,
            "inventory_authority_receipt_sha256": inventory_receipt["receipt_sha256"],
            "inventory_manifest_receipt_sha256": inventory_manifest["receipt_sha256"],
            "inventory_outcomes_file_sha256": inventory_manifest[
                "outcomes_file_sha256"
            ],
            "fold_plan_receipt_sha256": inventory_manifest["fold_plan_receipt_sha256"],
            "source_train_analysis_identity_roster_sha256": inventory_manifest[
                "source_train_analysis_identity_roster_sha256"
            ],
            "complete_cartesian_outcome_count": inventory_manifest[
                "cartesian_outcome_count"
            ],
            "detector_reference_registry_receipt_sha256": phase_receipt[
                "registry_receipt_sha256"
            ],
            "detector_phase_receipt_sha256": phase_receipt["receipt_sha256"],
            "outer_fold_id": phase_receipt["outer_fold_id"],
            "phase": phase_receipt["phase"],
            "inventory_validation_logical_sequence": 1,
            "reference_opener_invocation_logical_sequence": 2,
            "inventory_revalidated_before_reference_opener_invoked": True,
            "first_reference_byte_open_after_inventory_validation": True,
            "opaque_actual_outcome_replayed_inventory_consumed": True,
            "serialized_inventory_manifest_JSONL_receipt_or_digest_accepted": False,
            "legacy_phase_receipt_replayed_unchanged": (
                legacy_receipt_replayed_unchanged
            ),
            "underlying_detector_phase_authority_type": (
                "ValidatedDetectorFoldReferencePhaseAuthorityV1"
            ),
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def _issue_gate_authority(
    *,
    mode: str,
    inventory_authority: (_inventory.AuthorizedDetectorProviderPreReferenceInventoryV1),
    inventory_manifest: Mapping[str, Any],
    inventory_receipt: Mapping[str, Any],
    phase_authority: _reference.ValidatedDetectorFoldReferencePhaseAuthorityV1,
    legacy_receipt_replayed_unchanged: bool,
) -> AuthorizedInventoryGatedDetectorSelectionFitPhaseV1:
    phase = _reference.require_validated_detector_fold_reference_phase_authority_v1(
        phase_authority
    )
    phase_receipt = phase.to_receipt()
    if (
        phase_receipt.get("phase") != "selection_fit"
        or phase_receipt.get("fold_plan_binding", {}).get("plan_receipt_sha256")
        != inventory_manifest["fold_plan_receipt_sha256"]
        or phase_receipt.get("fold_plan_binding", {}).get(
            "source_train_analysis_identity_roster_sha256"
        )
        != inventory_manifest["source_train_analysis_identity_roster_sha256"]
    ):
        raise PermissionError("inventory gate received a nonmatching detector phase")
    return AuthorizedInventoryGatedDetectorSelectionFitPhaseV1(
        inventory_authority=inventory_authority,
        phase_authority=phase,
        receipt=_gate_authority_receipt(
            mode=mode,
            inventory_manifest=inventory_manifest,
            inventory_receipt=inventory_receipt,
            phase_receipt=phase_receipt,
            legacy_receipt_replayed_unchanged=(legacy_receipt_replayed_unchanged),
        ),
        _issuer_seal=_GATE_AUTHORITY_SEAL,
    )


def materialize_inventory_gated_detector_selection_fit_phase_v1(
    inventory_authority: object,
    *,
    fold_plan: Mapping[str, Any],
    detector_reference_registry: Mapping[str, Any],
    reference_root: str | Path,
    outer_fold_id: int,
) -> AuthorizedInventoryGatedDetectorSelectionFitPhaseV1:
    """Validate the opaque complete inventory before the first reference read."""

    (
        inventory,
        manifest,
        inventory_receipt,
        plan,
        reference_registry,
    ) = _admit_pre_reference_context(
        inventory_authority,
        fold_plan=fold_plan,
        detector_reference_registry=detector_reference_registry,
    )
    # Path conversion and the legacy reference opener deliberately occur only
    # after _admit_pre_reference_context has replayed the opaque inventory.
    phase = _reference.materialize_detector_fold_reference_phase_v1(
        fold_plan=plan,
        registry=reference_registry,
        reference_root=Path(reference_root),
        outer_fold_id=outer_fold_id,
        phase="selection_fit",
    )
    return _issue_gate_authority(
        mode="fresh_selection_fit_materialization_after_inventory",
        inventory_authority=inventory,
        inventory_manifest=manifest,
        inventory_receipt=inventory_receipt,
        phase_authority=phase,
        legacy_receipt_replayed_unchanged=False,
    )


def authorize_inventory_gated_detector_selection_fit_phase_receipt_v1(
    serialized_phase_receipt: Mapping[str, Any],
    inventory_authority: object,
    *,
    fold_plan: Mapping[str, Any],
    detector_reference_registry: Mapping[str, Any],
    replay_reference_root: str | Path,
) -> AuthorizedInventoryGatedDetectorSelectionFitPhaseV1:
    """Upgrade an old receipt only after inventory and exact reference replay."""

    (
        inventory,
        manifest,
        inventory_receipt,
        plan,
        reference_registry,
    ) = _admit_pre_reference_context(
        inventory_authority,
        fold_plan=fold_plan,
        detector_reference_registry=detector_reference_registry,
    )
    structural = _reference.validate_detector_fold_reference_phase_v1(
        serialized_phase_receipt,
        fold_plan=plan,
        registry=reference_registry,
        replay_reference_root=None,
    )
    if structural.get("phase") != "selection_fit":
        raise PermissionError("inventory gate upgrades selection-fit receipts only")
    phase = _reference.authorize_detector_fold_reference_phase_receipt_v1(
        structural,
        fold_plan=plan,
        registry=reference_registry,
        replay_reference_root=Path(replay_reference_root),
    )
    if phase.to_receipt() != structural:
        raise PermissionError(
            "legacy selection-fit receipt changed during gated replay"
        )
    return _issue_gate_authority(
        mode="legacy_selection_fit_receipt_actual_byte_replay_after_inventory",
        inventory_authority=inventory,
        inventory_manifest=manifest,
        inventory_receipt=inventory_receipt,
        phase_authority=phase,
        legacy_receipt_replayed_unchanged=True,
    )


def require_inventory_gated_detector_selection_fit_phase_v1(
    value: object,
) -> _reference.ValidatedDetectorFoldReferencePhaseAuthorityV1:
    """Revalidate both nested opaque authorities and release the detector phase."""

    if (
        not isinstance(value, AuthorizedInventoryGatedDetectorSelectionFitPhaseV1)
        or not value._has_valid_issuer_seal()
    ):
        raise TypeError(
            "formal new selection-fit consumption requires an opaque inventory-gated phase"
        )
    inventory_authority, phase_authority = value._authorities()
    manifest, _rows, inventory_receipt = _inventory._require_inventory(
        inventory_authority
    )
    phase = _reference.require_validated_detector_fold_reference_phase_authority_v1(
        phase_authority
    )
    phase_receipt = phase.to_receipt()
    gate_registry = (
        load_default_detector_pre_reference_inventory_selection_gate_registry_v1()
    )
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_id",
            "registry_receipt_sha256",
            "method_id",
            "gate_source_sha256",
            "issuance_mode",
            "inventory_authority_receipt_sha256",
            "inventory_manifest_receipt_sha256",
            "inventory_outcomes_file_sha256",
            "fold_plan_receipt_sha256",
            "source_train_analysis_identity_roster_sha256",
            "complete_cartesian_outcome_count",
            "detector_reference_registry_receipt_sha256",
            "detector_phase_receipt_sha256",
            "outer_fold_id",
            "phase",
            "inventory_validation_logical_sequence",
            "reference_opener_invocation_logical_sequence",
            "inventory_revalidated_before_reference_opener_invoked",
            "first_reference_byte_open_after_inventory_validation",
            "opaque_actual_outcome_replayed_inventory_consumed",
            "serialized_inventory_manifest_JSONL_receipt_or_digest_accepted",
            "legacy_phase_receipt_replayed_unchanged",
            "underlying_detector_phase_authority_type",
            "receipt_sha256",
        },
        context="opaque inventory-gated selection-fit phase",
    )
    if (
        receipt["schema_version"] != GATE_AUTHORITY_SCHEMA_VERSION
        or receipt["registry_id"] != gate_registry["registry_id"]
        or receipt["registry_receipt_sha256"] != gate_registry["receipt_sha256"]
        or receipt["method_id"] != GATE_METHOD_ID
        or receipt["gate_source_sha256"]
        != detector_pre_reference_inventory_phase_gate_source_sha256_v1()
        or receipt["issuance_mode"]
        not in {
            "fresh_selection_fit_materialization_after_inventory",
            "legacy_selection_fit_receipt_actual_byte_replay_after_inventory",
        }
        or receipt["inventory_authority_receipt_sha256"]
        != inventory_receipt["receipt_sha256"]
        or receipt["inventory_manifest_receipt_sha256"] != manifest["receipt_sha256"]
        or receipt["inventory_outcomes_file_sha256"] != manifest["outcomes_file_sha256"]
        or receipt["fold_plan_receipt_sha256"] != manifest["fold_plan_receipt_sha256"]
        or receipt["source_train_analysis_identity_roster_sha256"]
        != manifest["source_train_analysis_identity_roster_sha256"]
        or receipt["complete_cartesian_outcome_count"]
        != manifest["cartesian_outcome_count"]
        or receipt["detector_reference_registry_receipt_sha256"]
        != phase_receipt["registry_receipt_sha256"]
        or receipt["detector_phase_receipt_sha256"] != phase_receipt["receipt_sha256"]
        or receipt["outer_fold_id"] != phase_receipt["outer_fold_id"]
        or receipt["phase"] != "selection_fit"
        or phase_receipt["phase"] != "selection_fit"
        or receipt["inventory_validation_logical_sequence"] != 1
        or receipt["reference_opener_invocation_logical_sequence"] != 2
        or receipt["inventory_revalidated_before_reference_opener_invoked"] is not True
        or receipt["first_reference_byte_open_after_inventory_validation"] is not True
        or receipt["opaque_actual_outcome_replayed_inventory_consumed"] is not True
        or receipt["serialized_inventory_manifest_JSONL_receipt_or_digest_accepted"]
        is not False
        or receipt["legacy_phase_receipt_replayed_unchanged"]
        is not (
            receipt["issuance_mode"]
            == "legacy_selection_fit_receipt_actual_byte_replay_after_inventory"
        )
        or receipt["underlying_detector_phase_authority_type"]
        != "ValidatedDetectorFoldReferencePhaseAuthorityV1"
    ):
        raise ValueError("opaque inventory-gated selection-fit authority drifted")
    return phase


def authorize_eventnet_fold_phase_from_inventory_gate_v1(
    gated_phase_authority: AuthorizedInventoryGatedDetectorSelectionFitPhaseV1,
    *,
    registry: Mapping[str, Any],
) -> _eventnet.AuthorizedEventNetFoldPhase:
    return _eventnet.authorize_eventnet_fold_phase(
        require_inventory_gated_detector_selection_fit_phase_v1(gated_phase_authority),
        registry=registry,
    )


def authorize_seizuretransformer_fold_phase_from_inventory_gate_v1(
    gated_phase_authority: AuthorizedInventoryGatedDetectorSelectionFitPhaseV1,
    *,
    registry: Mapping[str, Any],
) -> _st.AuthorizedSeizureTransformerFoldPhase:
    return _st.authorize_seizuretransformer_fold_phase(
        require_inventory_gated_detector_selection_fit_phase_v1(gated_phase_authority),
        registry=registry,
    )


__all__ = [
    "AuthorizedInventoryGatedDetectorSelectionFitPhaseV1",
    "DEFAULT_GATE_REGISTRY_RELATIVE_PATH",
    "GATE_AUTHORITY_SCHEMA_VERSION",
    "GATE_METHOD_ID",
    "GATE_REGISTRY_ID",
    "GATE_REGISTRY_SCHEMA_VERSION",
    "authorize_eventnet_fold_phase_from_inventory_gate_v1",
    "authorize_inventory_gated_detector_selection_fit_phase_receipt_v1",
    "authorize_seizuretransformer_fold_phase_from_inventory_gate_v1",
    "build_default_detector_pre_reference_inventory_selection_gate_registry_v1",
    "build_detector_pre_reference_inventory_selection_gate_registry_v1",
    "detector_pre_reference_inventory_phase_gate_source_sha256_v1",
    "load_default_detector_pre_reference_inventory_selection_gate_registry_v1",
    "materialize_inventory_gated_detector_selection_fit_phase_v1",
    "require_inventory_gated_detector_selection_fit_phase_v1",
    "validate_detector_pre_reference_inventory_selection_gate_registry_v1",
]
