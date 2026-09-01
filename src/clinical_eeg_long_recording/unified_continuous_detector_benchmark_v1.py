"""Fail-closed unified benchmark contract for continuous long-EEG detectors.

This additive module binds three concerns which previously lived in separate
artifacts:

* provider/artifact maturity (source, checkpoint, preprocessing, licence,
  exposure and local execution);
* a provider-neutral, patient-isolated full-record benchmark protocol; and
* an explicit distinction between an alarm operating point and the ranked
  navigation operating point used to retrieve EEG for later Findings/SOZ
  analysis.

The validator performs no model loading and no inference.  It hashes only the
small receipts and source/config files named by the registry.  In particular,
it does not download a checkpoint, deserialize a PyTorch container, inspect an
EDF annotation, or read a spreadsheet/clinical report.

The current plan deliberately freezes ``accuracy_primary`` to ``None``.  A
future result can only be proposed after a complete, frozen, same-protocol
source-evaluation comparison.  Loading this contract cannot qualify a model,
authorize clinical use, or establish a SOTA claim.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, Iterable, Mapping, Sequence

from .continuous_detection_benchmark import (
    CONTINUOUS_BENCHMARK_METHOD_ID,
    CONTINUOUS_BENCHMARK_SCHEMA_VERSION,
)
from .detector_admission_addendum_v1_1 import (
    DETECTOR_ADMISSION_ADDENDUM_SCHEMA_VERSION,
    load_clinical_eeg_detector_admission_addendum_v1_1,
)
from .detector_dual_operating_point_v1 import (
    DETECTOR_DUAL_OP_DIAGNOSTIC_METHOD_ID,
    DETECTOR_DUAL_OP_DIAGNOSTIC_SCHEMA_VERSION,
)
from .detector_provider_contract import (
    PROVIDER_REGISTRY_SCHEMA_VERSION,
    validate_provider_registry,
)


UNIFIED_DETECTOR_REGISTRY_SCHEMA_VERSION: Final[str] = (
    "continuous_long_eeg_detector_benchmark_provider_registry_v1"
)
UNIFIED_DETECTOR_PLAN_SCHEMA_VERSION: Final[str] = (
    "continuous_long_eeg_detector_unified_benchmark_plan_v1"
)
UNIFIED_DETECTOR_READINESS_SCHEMA_VERSION: Final[str] = (
    "continuous_long_eeg_detector_unified_benchmark_readiness_v1"
)
UNIFIED_DETECTOR_SELECTION_SCHEMA_VERSION: Final[str] = (
    "continuous_long_eeg_detector_accuracy_primary_gate_v1"
)

UNIFIED_DETECTOR_REGISTRY_ID: Final[str] = (
    "CONTINUOUS-LONG-EEG-DETECTOR-PROVIDER-REGISTRY-V1-20260824"
)
UNIFIED_DETECTOR_PLAN_ID: Final[str] = (
    "CONTINUOUS-LONG-EEG-DETECTOR-UNIFIED-BENCHMARK-V1-20260824"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIFIED_DETECTOR_REGISTRY_PATH: Final[Path] = (
    _ROOT / "configs" / "continuous_detector_benchmark_provider_registry_v1.json"
)
DEFAULT_UNIFIED_DETECTOR_PLAN_PATH: Final[Path] = (
    _ROOT / "configs" / "continuous_detector_unified_benchmark_plan_v1.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MATURITY_DIMENSIONS = (
    "official_source",
    "official_checkpoint",
    "native_preprocessing",
    "license",
    "held_out_exposure",
    "local_runnable",
)
_MATURITY_EVIDENCE_LEVELS = frozenset(
    {"verified", "partial", "missing", "blocked"}
)
_LOCAL_EXECUTION_STATES = frozenset(
    {"runnable_research", "activation_blocked", "not_implemented"}
)
_BENCHMARK_LANES = frozenset({"alarm", "navigation", "efficiency"})
_PROVIDER_ROLES = frozenset(
    {
        "direct_event_engineering_control",
        "target_linked_navigation_secondary",
        "accuracy_challenger",
        "context_accuracy_challenger",
        "efficiency_control_only",
    }
)
_EXPECTED_PROVIDER_IDS = frozenset(
    {
        "eventnet_event_boundary_shadow_v1",
        "deepsoz_temporal_oof_candidate_v1",
        "seizuretransformer_timestep_shadow_v1",
        "lookaroundnet_context_shadow_v1",
        "rest_fft_shadow_v1",
    }
)

_REGISTRY_KEYS = {
    "schema_version",
    "registry_id",
    "status",
    "upstream_provider_inventory",
    "providers",
    "external_metadata_observations",
    "source_firewall",
    "scientific_permissions",
    "receipt_sha256",
}
_PROVIDER_KEYS = {
    "provider_id",
    "model_family",
    "benchmark_role",
    "benchmark_lanes",
    "local_execution_state",
    "execution_eligible_now",
    "accuracy_primary_eligible_now",
    "artifact_maturity",
    "artifact_bindings",
    "current_evidence",
    "blockers",
    "claim_limit",
}
_MATURITY_KEYS = {"status", "evidence_level", "detail"}
_ARTIFACT_BINDING_KEYS = {"path", "file_sha256", "semantic"}
_CURRENT_EVIDENCE_KEYS = {
    "same_protocol_scope",
    "complete_prediction_inventory",
    "alarm_operating_point_qualified",
    "navigation_operating_point_qualified",
    "warm_end_to_end_rtf",
    "performance_summary",
}
_EXTERNAL_OBSERVATION_KEYS = {
    "provider_id",
    "repository",
    "audited_commit",
    "audited_tree_sha",
    "observed_on",
    "official_tree_paths",
    "checkpoint_like_tree_paths",
    "release_count",
    "observation_method",
    "conclusion",
    "claim_limit",
}

_PLAN_KEYS = {
    "schema_version",
    "plan_id",
    "status",
    "contract_bindings",
    "execution_entrypoints",
    "benchmark_population",
    "provider_native_preprocessing",
    "operating_points",
    "prediction_inventory",
    "event_matching",
    "denominators",
    "required_metrics",
    "efficiency_protocol",
    "accuracy_primary_selection",
    "source_firewall",
    "scientific_permissions",
    "planned_outputs",
    "receipt_sha256",
}

_EXPECTED_SOURCE_FIREWALL: Mapping[str, bool] = {
    "eeg_samples_used": True,
    "edf_signal_header_used": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "video_or_behavior_used": False,
    "sleep_stage_labels_used": False,
    "activation_procedure_labels_used": False,
}
_EXPECTED_SCIENTIFIC_PERMISSIONS: Mapping[str, bool] = {
    "prediction_first_before_reference_join": True,
    "reference_labels_used_for_scoring_only": True,
    "upstream_reported_metrics_are_local_reproduction": False,
    "static_artifact_compatibility_is_accuracy_evidence": False,
    "navigation_op_is_clinical_alarm_op": False,
    "generic_receipt_can_promote_production": False,
    "sota_claim_authorized": False,
    "clinical_or_production_use_authorized": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _self_sha256(value: Mapping[str, object]) -> str:
    body = deepcopy(dict(value))
    body.pop("receipt_sha256", None)
    return _canonical_sha256(body)


def unified_detector_registry_self_sha256(value: Mapping[str, object]) -> str:
    """Return the canonical content hash of a provider maturity registry."""

    if not isinstance(value, Mapping):
        raise TypeError("unified detector registry must be an object")
    return _self_sha256(value)


def unified_detector_plan_self_sha256(value: Mapping[str, object]) -> str:
    """Return the canonical content hash of a unified benchmark plan."""

    if not isinstance(value, Mapping):
        raise TypeError("unified detector plan must be an object")
    return _self_sha256(value)


def _strict_object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    missing = keys - set(value)
    unknown = set(value) - keys
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")
    return deepcopy(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 2048 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite_optional(value: object, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{context} must be finite and non-negative")
    return result


def _unique_strings(value: object, context: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    result = [_identifier(item, context) for item in value]
    if not allow_empty and not result:
        raise ValueError(f"{context} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicates")
    return result


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_strict_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite token {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not valid UTF-8") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must contain an object")
    return value


def _safe_project_file(relative: object, context: str) -> Path:
    if not isinstance(relative, str):
        raise TypeError(f"{context} must be a relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{context} must be a canonical project-relative path")
    root = _ROOT.resolve(strict=True)
    unresolved = root.joinpath(*pure.parts)
    if unresolved.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    path = unresolved.resolve(strict=True)
    path.relative_to(root)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must resolve to a regular file")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file_binding(
    value: object,
    *,
    context: str,
    verify_file_bindings: bool,
) -> dict[str, Any]:
    row = _strict_object(value, _ARTIFACT_BINDING_KEYS, context)
    _identifier(row["semantic"], f"{context}.semantic")
    expected = _sha256(row["file_sha256"], f"{context}.file_sha256")
    if verify_file_bindings:
        path = _safe_project_file(row["path"], f"{context}.path")
        if _file_sha256(path) != expected:
            raise ValueError(f"{context} byte hash drifted")
    else:
        _identifier(row["path"], f"{context}.path")
    return row


def _validate_maturity(value: object, context: str) -> dict[str, Any]:
    row = _strict_object(value, _MATURITY_KEYS, context)
    _identifier(row["status"], f"{context}.status")
    if row["evidence_level"] not in _MATURITY_EVIDENCE_LEVELS:
        raise ValueError(f"{context}.evidence_level is invalid")
    _identifier(row["detail"], f"{context}.detail")
    return row


def _validate_current_evidence(value: object, context: str) -> dict[str, Any]:
    row = _strict_object(value, _CURRENT_EVIDENCE_KEYS, context)
    _identifier(row["same_protocol_scope"], f"{context}.same_protocol_scope")
    for field in (
        "complete_prediction_inventory",
        "alarm_operating_point_qualified",
        "navigation_operating_point_qualified",
    ):
        if type(row[field]) is not bool:
            raise TypeError(f"{context}.{field} must be boolean")
    row["warm_end_to_end_rtf"] = _finite_optional(
        row["warm_end_to_end_rtf"], f"{context}.warm_end_to_end_rtf"
    )
    _identifier(row["performance_summary"], f"{context}.performance_summary")
    return row


def _validate_external_observation(value: object, index: int) -> dict[str, Any]:
    context = f"external_metadata_observations[{index}]"
    row = _strict_object(value, _EXTERNAL_OBSERVATION_KEYS, context)
    for field in (
        "provider_id",
        "repository",
        "audited_commit",
        "audited_tree_sha",
        "observed_on",
        "observation_method",
        "conclusion",
        "claim_limit",
    ):
        _identifier(row[field], f"{context}.{field}")
    tree_paths = _unique_strings(
        row["official_tree_paths"], f"{context}.official_tree_paths", allow_empty=False
    )
    checkpoint_paths = _unique_strings(
        row["checkpoint_like_tree_paths"],
        f"{context}.checkpoint_like_tree_paths",
        allow_empty=True,
    )
    if tree_paths != sorted(tree_paths) or checkpoint_paths != sorted(checkpoint_paths):
        raise ValueError(f"{context} path inventories must be sorted")
    if any(path not in tree_paths for path in checkpoint_paths):
        raise ValueError(f"{context} checkpoint paths lie outside the tree")
    if isinstance(row["release_count"], bool) or not isinstance(
        row["release_count"], int
    ) or row["release_count"] < 0:
        raise TypeError(f"{context}.release_count must be non-negative integer")
    return row


def validate_unified_detector_provider_registry_v1(
    payload: object,
    *,
    verify_file_bindings: bool = True,
) -> dict[str, Any]:
    """Validate the additive provider maturity registry and all byte bindings."""

    data = _strict_object(payload, _REGISTRY_KEYS, "unified detector registry")
    if data["schema_version"] != UNIFIED_DETECTOR_REGISTRY_SCHEMA_VERSION:
        raise ValueError("unified detector registry schema drifted")
    if data["registry_id"] != UNIFIED_DETECTOR_REGISTRY_ID:
        raise ValueError("unified detector registry ID drifted")
    if data["status"] != "research_only_fail_closed_accuracy_primary_unselected":
        raise ValueError("unified detector registry status drifted")

    inventory = _strict_object(
        data["upstream_provider_inventory"],
        {"path", "file_sha256", "schema_version"},
        "upstream_provider_inventory",
    )
    if inventory["schema_version"] != PROVIDER_REGISTRY_SCHEMA_VERSION:
        raise ValueError("upstream provider inventory schema drifted")
    inventory_sha = _sha256(
        inventory["file_sha256"], "upstream_provider_inventory.file_sha256"
    )
    upstream_ids: set[str] = set()
    if verify_file_bindings:
        inventory_path = _safe_project_file(
            inventory["path"], "upstream_provider_inventory.path"
        )
        if _file_sha256(inventory_path) != inventory_sha:
            raise ValueError("upstream provider inventory byte hash drifted")
        upstream = validate_provider_registry(
            _load_strict_json(inventory_path, "upstream provider inventory")
        )
        upstream_ids = {
            str(row["execution_definition"]["provider_id"])
            for row in upstream["providers"]
        }
    else:
        _identifier(inventory["path"], "upstream_provider_inventory.path")

    if not isinstance(data["providers"], list) or not data["providers"]:
        raise TypeError("unified detector registry providers must be non-empty")
    providers: list[dict[str, Any]] = []
    provider_ids: list[str] = []
    for index, raw in enumerate(data["providers"]):
        context = f"providers[{index}]"
        row = _strict_object(raw, _PROVIDER_KEYS, context)
        provider_id = _identifier(row["provider_id"], f"{context}.provider_id")
        provider_ids.append(provider_id)
        _identifier(row["model_family"], f"{context}.model_family")
        if row["benchmark_role"] not in _PROVIDER_ROLES:
            raise ValueError(f"{context}.benchmark_role is invalid")
        lanes = _unique_strings(
            row["benchmark_lanes"], f"{context}.benchmark_lanes", allow_empty=False
        )
        if lanes != sorted(lanes) or any(lane not in _BENCHMARK_LANES for lane in lanes):
            raise ValueError(f"{context}.benchmark_lanes are invalid or unsorted")
        if row["local_execution_state"] not in _LOCAL_EXECUTION_STATES:
            raise ValueError(f"{context}.local_execution_state is invalid")
        for field in ("execution_eligible_now", "accuracy_primary_eligible_now"):
            if type(row[field]) is not bool:
                raise TypeError(f"{context}.{field} must be boolean")
        if row["execution_eligible_now"] is not (
            row["local_execution_state"] == "runnable_research"
        ):
            raise ValueError(f"{context} execution state/eligibility disagree")

        maturity = _strict_object(
            row["artifact_maturity"], set(_MATURITY_DIMENSIONS), f"{context}.artifact_maturity"
        )
        row["artifact_maturity"] = {
            dimension: _validate_maturity(
                maturity[dimension], f"{context}.artifact_maturity.{dimension}"
            )
            for dimension in _MATURITY_DIMENSIONS
        }
        if row["accuracy_primary_eligible_now"]:
            if "alarm" not in lanes or any(
                row["artifact_maturity"][dimension]["evidence_level"] != "verified"
                for dimension in _MATURITY_DIMENSIONS
            ):
                raise ValueError(
                    f"{context} accuracy eligibility lacks complete maturity"
                )

        if not isinstance(row["artifact_bindings"], list):
            raise TypeError(f"{context}.artifact_bindings must be an array")
        row["artifact_bindings"] = [
            _validate_file_binding(
                binding,
                context=f"{context}.artifact_bindings[{binding_index}]",
                verify_file_bindings=verify_file_bindings,
            )
            for binding_index, binding in enumerate(row["artifact_bindings"])
        ]
        binding_paths = [item["path"] for item in row["artifact_bindings"]]
        if len(set(binding_paths)) != len(binding_paths):
            raise ValueError(f"{context}.artifact_bindings contain duplicate paths")
        row["current_evidence"] = _validate_current_evidence(
            row["current_evidence"], f"{context}.current_evidence"
        )
        row["blockers"] = _unique_strings(
            row["blockers"], f"{context}.blockers", allow_empty=False
        )
        _identifier(row["claim_limit"], f"{context}.claim_limit")
        if row["accuracy_primary_eligible_now"] is not False:
            raise ValueError("v1 registry must keep every accuracy-primary gate closed")
        providers.append(row)

    if len(set(provider_ids)) != len(provider_ids):
        raise ValueError("unified detector provider IDs must be unique")
    if set(provider_ids) != _EXPECTED_PROVIDER_IDS:
        raise ValueError("unified detector provider roster drifted")
    if provider_ids != sorted(provider_ids):
        raise ValueError("unified detector providers must be sorted by provider_id")
    if verify_file_bindings and not set(provider_ids).issubset(upstream_ids):
        raise ValueError("unified registry names a provider absent upstream")

    by_id = {row["provider_id"]: row for row in providers}
    rest = by_id["rest_fft_shadow_v1"]
    if (
        rest["benchmark_role"] != "efficiency_control_only"
        or rest["benchmark_lanes"] != ["efficiency"]
        or rest["execution_eligible_now"] is not False
        or not {
            "missing_lightning_import_and_undefined_training_symbols",
            "random_cuda_bound_state_update_requires_deterministic_device_safe_repair",
            "clip_level_mse_objective_is_not_full_record_event_detection",
            "official_checkpoint_absent",
        }.issubset(rest["blockers"])
    ):
        raise ValueError("REST must remain a non-mature efficiency control")

    seizure_transformer = by_id["seizuretransformer_timestep_shadow_v1"]
    if (
        seizure_transformer["benchmark_role"] != "accuracy_challenger"
        or seizure_transformer["local_execution_state"] != "activation_blocked"
        or seizure_transformer["artifact_maturity"]["official_checkpoint"][
            "status"
        ]
        != "official_checkpoint_unavailable_third_party_safetensors_only"
        or seizure_transformer["artifact_maturity"]["native_preprocessing"][
            "evidence_level"
        ]
        == "verified"
        or seizure_transformer["artifact_maturity"]["held_out_exposure"][
            "evidence_level"
        ]
        == "verified"
    ):
        raise ValueError("SeizureTransformer activation blockers drifted")

    lookaround = by_id["lookaroundnet_context_shadow_v1"]
    if (
        lookaround["benchmark_role"] != "context_accuracy_challenger"
        or lookaround["local_execution_state"] != "activation_blocked"
        or lookaround["artifact_maturity"]["official_checkpoint"]["status"]
        != "no_checkpoint_in_audited_official_commit_or_releases_as_of_2026_08_24"
    ):
        raise ValueError("LookAroundNet official-checkpoint conclusion drifted")

    eventnet = by_id["eventnet_event_boundary_shadow_v1"]
    if (
        eventnet["benchmark_role"] != "direct_event_engineering_control"
        or eventnet["local_execution_state"] != "runnable_research"
        or eventnet["current_evidence"]["complete_prediction_inventory"] is not True
        or eventnet["current_evidence"]["alarm_operating_point_qualified"] is not False
    ):
        raise ValueError("EventNet current maturity/performance state drifted")

    deepsoz = by_id["deepsoz_temporal_oof_candidate_v1"]
    if (
        deepsoz["benchmark_role"] != "target_linked_navigation_secondary"
        or deepsoz["benchmark_lanes"] != ["navigation"]
        or deepsoz["artifact_maturity"]["held_out_exposure"]["evidence_level"]
        != "partial"
        or deepsoz["current_evidence"]["navigation_operating_point_qualified"]
        is not False
    ):
        raise ValueError("DeepSOZ must remain a provisional navigation secondary")

    if not isinstance(data["external_metadata_observations"], list):
        raise TypeError("external metadata observations must be an array")
    observations = [
        _validate_external_observation(value, index)
        for index, value in enumerate(data["external_metadata_observations"])
    ]
    observation_ids = [row["provider_id"] for row in observations]
    if observation_ids != sorted(observation_ids) or len(set(observation_ids)) != len(
        observation_ids
    ):
        raise ValueError("external metadata observations must be unique and sorted")
    observation_by_id = {row["provider_id"]: row for row in observations}
    lookaround_observation = observation_by_id.get(
        "lookaroundnet_context_shadow_v1"
    )
    if (
        lookaround_observation is None
        or lookaround_observation["audited_commit"]
        != "b9ba07f7913f663ae60d3e8fa6ca2ec0bcaff51b"
        or lookaround_observation["official_tree_paths"]
        != [
            "LICENSE",
            "README.md",
            "dataset.py",
            "models.py",
            "preprocess.py",
            "run.py",
        ]
        or lookaround_observation["checkpoint_like_tree_paths"] != []
        or lookaround_observation["release_count"] != 0
    ):
        raise ValueError("LookAroundNet official metadata observation drifted")
    seizure_transformer_observation = observation_by_id.get(
        "seizuretransformer_timestep_shadow_v1"
    )
    if (
        seizure_transformer_observation is None
        or seizure_transformer_observation["audited_commit"]
        != "cf83f5906a8aea88b60b56e4f962c5d6657c28f7"
        or seizure_transformer_observation["audited_tree_sha"]
        != "8d58f6a2cff852f4598295b2966ea849b3822a23"
        or seizure_transformer_observation["checkpoint_like_tree_paths"] != []
        or seizure_transformer_observation["release_count"] != 0
        or "cannot be independently reproduced"
        not in seizure_transformer_observation["claim_limit"]
    ):
        raise ValueError("SeizureTransformer official metadata observation drifted")

    if data["source_firewall"] != _EXPECTED_SOURCE_FIREWALL:
        raise ValueError("unified detector registry source firewall drifted")
    if data["scientific_permissions"] != _EXPECTED_SCIENTIFIC_PERMISSIONS:
        raise ValueError("unified detector registry scientific permissions drifted")
    receipt = _sha256(data["receipt_sha256"], "unified detector registry receipt")
    if receipt != unified_detector_registry_self_sha256(data):
        raise ValueError("unified detector registry canonical self-hash mismatch")
    data["providers"] = providers
    data["external_metadata_observations"] = observations
    return data


def load_unified_detector_provider_registry_v1(
    path: str | Path = DEFAULT_UNIFIED_DETECTOR_REGISTRY_PATH,
    *,
    verify_file_bindings: bool = True,
) -> dict[str, Any]:
    """Strictly load the default or an explicitly supplied registry."""

    candidate = Path(path)
    return validate_unified_detector_provider_registry_v1(
        _load_strict_json(candidate, "unified detector registry"),
        verify_file_bindings=verify_file_bindings,
    )


def _validate_contract_binding(
    value: object,
    *,
    context: str,
    expected_schema: str,
    verify_file_bindings: bool,
) -> dict[str, Any]:
    row = _strict_object(
        value,
        {"path", "file_sha256", "schema_version", "content_receipt_sha256"},
        context,
    )
    if row["schema_version"] != expected_schema:
        raise ValueError(f"{context}.schema_version drifted")
    expected_file_sha = _sha256(row["file_sha256"], f"{context}.file_sha256")
    if row["content_receipt_sha256"] is not None:
        _sha256(row["content_receipt_sha256"], f"{context}.content_receipt_sha256")
    if verify_file_bindings:
        path = _safe_project_file(row["path"], f"{context}.path")
        if _file_sha256(path) != expected_file_sha:
            raise ValueError(f"{context} byte hash drifted")
    else:
        _identifier(row["path"], f"{context}.path")
    return row


def _validate_exact_positive_grid(value: object, expected: Sequence[float], context: str) -> None:
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError(f"{context} drifted")
    normalized = [_finite_optional(item, context) for item in value]
    if normalized != [float(item) for item in expected]:
        raise ValueError(f"{context} drifted")


def validate_unified_continuous_detector_benchmark_plan_v1(
    payload: object,
    *,
    verify_file_bindings: bool = True,
) -> dict[str, Any]:
    """Validate the executable, provider-neutral long-record benchmark plan."""

    data = _strict_object(payload, _PLAN_KEYS, "unified detector benchmark plan")
    if data["schema_version"] != UNIFIED_DETECTOR_PLAN_SCHEMA_VERSION:
        raise ValueError("unified detector benchmark plan schema drifted")
    if data["plan_id"] != UNIFIED_DETECTOR_PLAN_ID:
        raise ValueError("unified detector benchmark plan ID drifted")
    if data["status"] != "protocol_executable_accuracy_primary_null":
        raise ValueError("unified detector benchmark plan status drifted")

    bindings = _strict_object(
        data["contract_bindings"],
        {
            "provider_maturity_registry",
            "detector_admission_addendum",
            "continuous_benchmark_scorer",
            "dual_operating_point_scorer",
        },
        "contract_bindings",
    )
    registry_binding = _validate_contract_binding(
        bindings["provider_maturity_registry"],
        context="contract_bindings.provider_maturity_registry",
        expected_schema=UNIFIED_DETECTOR_REGISTRY_SCHEMA_VERSION,
        verify_file_bindings=verify_file_bindings,
    )
    admission_binding = _validate_contract_binding(
        bindings["detector_admission_addendum"],
        context="contract_bindings.detector_admission_addendum",
        expected_schema=DETECTOR_ADMISSION_ADDENDUM_SCHEMA_VERSION,
        verify_file_bindings=verify_file_bindings,
    )
    benchmark_binding = _validate_contract_binding(
        bindings["continuous_benchmark_scorer"],
        context="contract_bindings.continuous_benchmark_scorer",
        expected_schema=CONTINUOUS_BENCHMARK_SCHEMA_VERSION,
        verify_file_bindings=verify_file_bindings,
    )
    dual_op_binding = _validate_contract_binding(
        bindings["dual_operating_point_scorer"],
        context="contract_bindings.dual_operating_point_scorer",
        expected_schema=DETECTOR_DUAL_OP_DIAGNOSTIC_SCHEMA_VERSION,
        verify_file_bindings=verify_file_bindings,
    )
    registry: dict[str, Any] | None = None
    if verify_file_bindings:
        registry_path = _safe_project_file(
            registry_binding["path"], "provider maturity registry path"
        )
        registry = load_unified_detector_provider_registry_v1(
            registry_path, verify_file_bindings=True
        )
        if registry["receipt_sha256"] != registry_binding["content_receipt_sha256"]:
            raise ValueError("provider maturity registry content receipt drifted")
        admission = load_clinical_eeg_detector_admission_addendum_v1_1(
            _safe_project_file(
                admission_binding["path"], "detector admission addendum path"
            ),
            verify_projection_binding=True,
        )
        if admission["receipt_sha256"] != admission_binding["content_receipt_sha256"]:
            raise ValueError("detector admission addendum content receipt drifted")
    if benchmark_binding["content_receipt_sha256"] is not None:
        raise ValueError("source module binding must not invent a content receipt")
    if dual_op_binding["content_receipt_sha256"] is not None:
        raise ValueError("source module binding must not invent a content receipt")

    entrypoints = _strict_object(
        data["execution_entrypoints"],
        {"plan_validator_cli", "dual_operating_point_cli"},
        "execution_entrypoints",
    )
    validated_entrypoints = {
        name: _validate_file_binding(
            entrypoints[name],
            context=f"execution_entrypoints.{name}",
            verify_file_bindings=verify_file_bindings,
        )
        for name in ("plan_validator_cli", "dual_operating_point_cli")
    }
    expected_entrypoints = {
        "plan_validator_cli": (
            "scripts/validate_continuous_detector_unified_benchmark_v1.py",
            "no_model_plan_registry_and_binding_validator",
        ),
        "dual_operating_point_cli": (
            "scripts/run_continuous_detector_dual_op_benchmark_v1.py",
            "prediction_first_provider_neutral_alarm_and_navigation_scorer",
        ),
    }
    for name, (path, semantic) in expected_entrypoints.items():
        if (
            validated_entrypoints[name]["path"] != path
            or validated_entrypoints[name]["semantic"] != semantic
        ):
            raise ValueError(f"execution_entrypoints.{name} drifted")

    population = _strict_object(
        data["benchmark_population"],
        {
            "calibration_split",
            "selection_split",
            "external_confirmation_split",
            "split_unit",
            "patient_overlap_allowed",
            "recording_unit",
            "native_recording_minimum_seconds",
            "cross_edf_concatenation_allowed",
            "all_edfs_for_patient_stay_in_one_split",
            "provider_training_exposure_overlap_disposition",
            "official_source_dev_denominator",
        },
        "benchmark_population",
    )
    expected_population = {
        "calibration_split": "source_dev",
        "selection_split": "source_eval",
        "external_confirmation_split": "external_eval",
        "split_unit": "patient",
        "patient_overlap_allowed": False,
        "recording_unit": "native_complete_long_recording",
        "native_recording_minimum_seconds": 1800,
        "cross_edf_concatenation_allowed": False,
        "all_edfs_for_patient_stay_in_one_split": True,
        "provider_training_exposure_overlap_disposition": (
            "exclude_from_untouched_selection_or_use_patient_held_out_checkpoint_"
            "with_complete_exposure_receipt"
        ),
    }
    for field, expected in expected_population.items():
        if population[field] != expected:
            raise ValueError(f"benchmark_population.{field} drifted")
    denominator = _strict_object(
        population["official_source_dev_denominator"],
        {
            "patient_count",
            "recording_count",
            "duration_hours",
            "reference_event_count",
            "seizure_bearing_recording_count",
            "seizure_free_recording_count",
            "identity_projection_receipt_sha256",
        },
        "benchmark_population.official_source_dev_denominator",
    )
    exact_counts = {
        "patient_count": 53,
        "recording_count": 1832,
        "reference_event_count": 1075,
        "seizure_bearing_recording_count": 325,
        "seizure_free_recording_count": 1507,
    }
    for field, expected in exact_counts.items():
        if denominator[field] != expected or isinstance(denominator[field], bool):
            raise ValueError(f"official source-dev {field} drifted")
    if float(denominator["duration_hours"]) != 435.548:
        raise ValueError("official source-dev duration drifted")
    _sha256(
        denominator["identity_projection_receipt_sha256"],
        "official source-dev identity projection receipt",
    )

    preprocessing = _strict_object(
        data["provider_native_preprocessing"],
        {
            "shared_canonical_root",
            "one_shared_model_transform_for_all_providers",
            "provider_owns_native_transform",
            "required_receipt_fields",
            "target_forbidden_inputs",
            "native_transform_comparison_rule",
        },
        "provider_native_preprocessing",
    )
    if (
        preprocessing["shared_canonical_root"]
        != "physical_eeg_samples_recording_clock_typed_units_qc_and_lineage"
        or preprocessing["one_shared_model_transform_for_all_providers"] is not False
        or preprocessing["provider_owns_native_transform"] is not True
        or preprocessing["native_transform_comparison_rule"]
        != "same_native_recording_roster_provider_native_preprocessing_included_in_end_to_end_cost"
    ):
        raise ValueError("provider-native preprocessing boundary drifted")
    expected_preprocessing_fields = [
        "channel_mapping_and_imputation",
        "filter_and_resample",
        "input_units_and_scaling",
        "montage_or_reference",
        "observed_support_padding_and_tiling",
        "provider_id_and_version",
        "raw_signal_sha256",
        "receipt_sha256",
    ]
    if preprocessing["required_receipt_fields"] != expected_preprocessing_fields:
        raise ValueError("provider-native preprocessing receipt fields drifted")
    if preprocessing["target_forbidden_inputs"] != [
        "clinical_text",
        "doctor_labels",
        "edf_annotations",
        "reference_seizure_intervals",
        "spreadsheet_fields",
    ]:
        raise ValueError("provider-native preprocessing firewall drifted")

    operating_points = _strict_object(
        data["operating_points"], {"alarm", "navigation"}, "operating_points"
    )
    alarm = _strict_object(
        operating_points["alarm"],
        {
            "lane_id",
            "purpose",
            "calibrated_on",
            "frozen_before",
            "event_match",
            "false_alarm_budget_metric",
            "hard_gates",
            "may_select_accuracy_primary",
            "may_drive_findings_retrieval",
        },
        "operating_points.alarm",
    )
    if (
        alarm["lane_id"] != "OP-ALARM"
        or alarm["purpose"] != "clinical_alarm_style_event_detection_benchmark"
        or alarm["calibrated_on"] != "source_dev"
        or alarm["frozen_before"] != "source_eval_reference_join"
        or alarm["event_match"] != CONTINUOUS_BENCHMARK_METHOD_ID
        or alarm["false_alarm_budget_metric"]
        != "all_unmatched_alarms_per_24_processed_evaluable_hours"
        or alarm["may_select_accuracy_primary"] is not True
        or alarm["may_drive_findings_retrieval"] is not False
    ):
        raise ValueError("Alarm OP semantics drifted")
    if alarm["hard_gates"] != {
        "all_unmatched_alarms_per_24h_maximum": 12.0,
        "patient_macro_event_sensitivity_minimum": 0.85,
        "pooled_event_sensitivity_minimum": 0.9,
        "warm_end_to_end_rtf_maximum": 0.05,
    }:
        raise ValueError("Alarm OP hard gates drifted")

    navigation = _strict_object(
        operating_points["navigation"],
        {
            "lane_id",
            "purpose",
            "calibrated_on",
            "frozen_before",
            "event_match",
            "candidate_budgets_per_hour",
            "queried_eeg_seconds_per_hour",
            "onset_tolerances_seconds",
            "ranking_scope",
            "may_select_accuracy_primary",
            "may_drive_findings_retrieval",
        },
        "operating_points.navigation",
    )
    if (
        navigation["lane_id"] != "OP-NAVIGATION"
        or navigation["purpose"]
        != "ranked_high_recall_candidate_retrieval_for_event_findings"
        or navigation["calibrated_on"] != "source_dev"
        or navigation["frozen_before"] != "source_eval_reference_join"
        or navigation["event_match"]
        != "ordered_one_to_one_onset_envelope_then_anchor_error"
        or navigation["ranking_scope"]
        != "within_record_ranked_candidates_under_explicit_query_budget"
        or navigation["may_select_accuracy_primary"] is not False
        or navigation["may_drive_findings_retrieval"] is not True
    ):
        raise ValueError("Navigation OP semantics drifted")
    _validate_exact_positive_grid(
        navigation["candidate_budgets_per_hour"],
        (1.0, 2.0, 4.0, 8.0, 16.0),
        "navigation candidate budgets",
    )
    _validate_exact_positive_grid(
        navigation["queried_eeg_seconds_per_hour"],
        (60.0, 120.0, 300.0, 600.0),
        "navigation query budgets",
    )
    _validate_exact_positive_grid(
        navigation["onset_tolerances_seconds"],
        (1.0, 3.0, 5.0, 10.0),
        "navigation onset tolerances",
    )

    inventory_contract = _strict_object(
        data["prediction_inventory"],
        {
            "prediction_first_reference_free",
            "complete_record_provider_policy_cross_product",
            "one_terminal_row_per_record_provider_policy",
            "terminal_outcomes",
            "zero_candidate_is_valid_completion",
            "technical_failure_is_zero_candidate",
            "partial_tail_is_complete_coverage",
            "candidate_force_minimum_allowed",
        },
        "prediction_inventory",
    )
    if inventory_contract != {
        "prediction_first_reference_free": True,
        "complete_record_provider_policy_cross_product": True,
        "one_terminal_row_per_record_provider_policy": True,
        "terminal_outcomes": [
            "completed_with_candidates",
            "completed_zero_candidate",
            "partial_coverage",
            "technical_failure",
        ],
        "zero_candidate_is_valid_completion": True,
        "technical_failure_is_zero_candidate": False,
        "partial_tail_is_complete_coverage": False,
        "candidate_force_minimum_allowed": False,
    }:
        raise ValueError("prediction inventory closure semantics drifted")

    matching = _strict_object(
        data["event_matching"],
        {
            "alarm_method_id",
            "alarm_objective_order",
            "navigation_method_id",
            "one_prediction_can_match_multiple_references",
            "one_reference_can_match_multiple_predictions",
            "unmatched_overlap_fragments_count_as_false_alarms",
        },
        "event_matching",
    )
    if matching != {
        "alarm_method_id": CONTINUOUS_BENCHMARK_METHOD_ID,
        "alarm_objective_order": [
            "maximize_match_count",
            "maximize_total_iou",
            "minimize_absolute_onset_error",
        ],
        "navigation_method_id": "ordered_one_to_one_onset_envelope_then_anchor_error",
        "one_prediction_can_match_multiple_references": False,
        "one_reference_can_match_multiple_predictions": False,
        "unmatched_overlap_fragments_count_as_false_alarms": True,
    }:
        raise ValueError("one-to-one event matching contract drifted")

    denominators = _strict_object(
        data["denominators"],
        {
            "reference_event_denominator",
            "false_alarm_opportunity_denominator",
            "zero_candidate_records",
            "technical_failures",
            "partial_coverage",
            "qualification_requires",
            "patient_aggregation",
        },
        "denominators",
    )
    if denominators != {
        "reference_event_denominator": (
            "all_joined_reference_events_including_events_in_technical_failure_"
            "records_and_unmodeled_partial_tails"
        ),
        "false_alarm_opportunity_denominator": (
            "processed_evaluable_modeled_eeg_hours_excluding_technical_failure_"
            "and_unmodeled_partial_tail"
        ),
        "zero_candidate_records": (
            "retained_in_complete_record_denominator_and_all_reference_events_are_misses"
        ),
        "technical_failures": (
            "retained_as_terminal_failures_never_relabelled_zero_candidate"
        ),
        "partial_coverage": (
            "modeled_support_scores_burden_but_unmodeled_reference_events_remain_misses"
        ),
        "qualification_requires": "zero_technical_failure_and_zero_partial_coverage",
        "patient_aggregation": "combine_records_within_patient_then_patient_macro",
    }:
        raise ValueError("benchmark denominator semantics drifted")

    required_metrics = _unique_strings(
        data["required_metrics"], "required_metrics", allow_empty=False
    )
    expected_metrics = [
        "all_unmatched_alarms_per_24h",
        "background_only_false_alarms_per_background_hour",
        "candidate_count_and_candidates_per_hour",
        "event_f1",
        "event_iou",
        "event_precision",
        "onset_absolute_error_and_reference_coverage",
        "onset_hit_rates_1_3_5_10_seconds",
        "patient_macro_event_sensitivity",
        "pooled_event_sensitivity",
        "queried_eeg_seconds_and_fraction",
        "reference_overlap_duplicate_or_fragment_alarm_count",
        "time_in_warning",
        "typed_onset_offset_boundary_f1",
    ]
    if required_metrics != expected_metrics:
        raise ValueError("unified detector required metrics drifted")

    efficiency = _strict_object(
        data["efficiency_protocol"],
        {
            "included_in_end_to_end_wall_time",
            "service_states",
            "rtf_definition",
            "required_per_record_fields",
            "hardware_context_fields",
            "warm_gate_requires_complete_receipt_coverage",
            "failures_may_dilute_rtf_denominator",
        },
        "efficiency_protocol",
    )
    if efficiency["included_in_end_to_end_wall_time"] != [
        "decoder_and_postprocessing",
        "edf_io",
        "inference",
        "provider_native_preprocessing",
    ] or efficiency["service_states"] != ["cold", "warm"]:
        raise ValueError("efficiency timing boundary drifted")
    if efficiency["rtf_definition"] != "total_wall_seconds_divided_by_eeg_seconds":
        raise ValueError("efficiency RTF definition drifted")
    if efficiency["required_per_record_fields"] != [
        "edf_io_seconds",
        "gpu_active_seconds",
        "inference_seconds",
        "peak_gpu_memory_bytes",
        "peak_host_memory_bytes",
        "postprocessing_seconds",
        "preprocessing_seconds",
        "total_wall_seconds",
    ]:
        raise ValueError("efficiency per-record fields drifted")
    if efficiency["hardware_context_fields"] != [
        "batch_size",
        "concurrency",
        "device_type",
        "precision",
        "software_environment_sha256",
    ]:
        raise ValueError("efficiency hardware context drifted")
    if (
        efficiency["warm_gate_requires_complete_receipt_coverage"] is not True
        or efficiency["failures_may_dilute_rtf_denominator"] is not False
    ):
        raise ValueError("efficiency coverage/failure semantics drifted")

    selection = _strict_object(
        data["accuracy_primary_selection"],
        {
            "current_value",
            "current_status",
            "eligible_lane",
            "minimum_same_protocol_provider_families",
            "required_absolute_gates",
            "required_evidence",
            "comparison_rule",
            "blended_score_allowed",
            "navigation_result_can_fill_accuracy_primary",
            "null_when_no_qualified_candidate",
            "current_null_reasons",
        },
        "accuracy_primary_selection",
    )
    if (
        selection["current_value"] is not None
        or selection["current_status"] != "no_qualified_operating_point"
        or selection["eligible_lane"] != "OP-ALARM"
        or selection["minimum_same_protocol_provider_families"] != 2
        or selection["required_absolute_gates"] != alarm["hard_gates"]
        or selection["comparison_rule"]
        != "absolute_gate_then_accuracy_efficiency_pareto_no_blended_score"
        or selection["blended_score_allowed"] is not False
        or selection["navigation_result_can_fill_accuracy_primary"] is not False
        or selection["null_when_no_qualified_candidate"] is not True
    ):
        raise ValueError("accuracy_primary null selection gate drifted")
    if selection["required_evidence"] != [
        "complete_source_eval_recording_inventory",
        "frozen_source_dev_alarm_policy",
        "one_to_one_event_matching",
        "patient_disjoint_source_dev_source_eval",
        "patient_level_bootstrap",
        "provider_native_preprocessing_receipts_for_every_record",
        "resource_receipts_for_every_record",
        "untouched_source_eval_reference_join_after_prediction_freeze",
        "zero_failure_and_zero_partial_coverage",
    ]:
        raise ValueError("accuracy_primary evidence gate drifted")
    expected_null_reasons = [
        "deepsoz_is_provisional_navigation_secondary_not_same_protocol_accuracy_primary",
        "eventnet_has_complete_dev_inventory_but_no_alarm_policy_meets_accuracy_gates",
        "lookaroundnet_official_checkpoint_not_published_in_audited_source_or_releases",
        "rest_is_efficiency_only_and_not_locally_replayable",
        "seizuretransformer_third_party_weights_preprocessing_license_and_exposure_unresolved",
        "source_eval_same_protocol_paired_comparison_not_complete",
    ]
    if selection["current_null_reasons"] != expected_null_reasons:
        raise ValueError("accuracy_primary current null reasons drifted")

    if data["source_firewall"] != _EXPECTED_SOURCE_FIREWALL:
        raise ValueError("unified benchmark source firewall drifted")
    if data["scientific_permissions"] != _EXPECTED_SCIENTIFIC_PERMISSIONS:
        raise ValueError("unified benchmark scientific permissions drifted")
    outputs = _unique_strings(data["planned_outputs"], "planned_outputs", allow_empty=False)
    if outputs != [
        "alarm_op_benchmark_receipt_per_provider",
        "complete_terminal_outcome_inventory",
        "navigation_op_budget_curve_per_provider",
        "paired_patient_comparison_receipt",
        "provider_native_preprocessing_receipt_roster",
        "resource_and_rtf_receipt_roster",
        "selection_gate_receipt_with_nullable_accuracy_primary",
    ]:
        raise ValueError("unified benchmark planned outputs drifted")

    if registry is not None:
        if not any(
            row["local_execution_state"] == "runnable_research"
            for row in registry["providers"]
        ):
            raise ValueError("unified benchmark registry has no runnable research lane")
        if any(row["accuracy_primary_eligible_now"] for row in registry["providers"]):
            raise ValueError("registry contradicts the frozen null accuracy primary")

    receipt = _sha256(data["receipt_sha256"], "unified detector plan receipt")
    if receipt != unified_detector_plan_self_sha256(data):
        raise ValueError("unified detector plan canonical self-hash mismatch")
    return data


def load_unified_continuous_detector_benchmark_plan_v1(
    path: str | Path = DEFAULT_UNIFIED_DETECTOR_PLAN_PATH,
    *,
    verify_file_bindings: bool = True,
) -> dict[str, Any]:
    """Strictly load the default or an explicitly supplied benchmark plan."""

    candidate = Path(path)
    return validate_unified_continuous_detector_benchmark_plan_v1(
        _load_strict_json(candidate, "unified detector benchmark plan"),
        verify_file_bindings=verify_file_bindings,
    )


def build_unified_detector_benchmark_readiness_v1(
    *,
    plan: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
    verify_file_bindings: bool = True,
) -> dict[str, Any]:
    """Build a content-bound no-inference readiness receipt."""

    validated_plan = (
        load_unified_continuous_detector_benchmark_plan_v1(
            verify_file_bindings=verify_file_bindings
        )
        if plan is None
        else validate_unified_continuous_detector_benchmark_plan_v1(
            dict(plan), verify_file_bindings=verify_file_bindings
        )
    )
    validated_registry = (
        load_unified_detector_provider_registry_v1(
            verify_file_bindings=verify_file_bindings
        )
        if registry is None
        else validate_unified_detector_provider_registry_v1(
            dict(registry), verify_file_bindings=verify_file_bindings
        )
    )
    runnable = sorted(
        row["provider_id"]
        for row in validated_registry["providers"]
        if row["execution_eligible_now"]
    )
    blocked = {
        row["provider_id"]: list(row["blockers"])
        for row in validated_registry["providers"]
        if not row["execution_eligible_now"]
    }
    body: dict[str, Any] = {
        "schema_version": UNIFIED_DETECTOR_READINESS_SCHEMA_VERSION,
        "readiness_id": "UNIFIED-DETECTOR-READINESS-PENDING",
        "plan_id": validated_plan["plan_id"],
        "plan_receipt_sha256": validated_plan["receipt_sha256"],
        "registry_id": validated_registry["registry_id"],
        "registry_receipt_sha256": validated_registry["receipt_sha256"],
        "provider_count": len(validated_registry["providers"]),
        "runnable_research_provider_ids": runnable,
        "blocked_provider_reasons": blocked,
        "alarm_operating_point_id": "OP-ALARM",
        "navigation_operating_point_id": "OP-NAVIGATION",
        "accuracy_primary": None,
        "accuracy_primary_status": "no_qualified_operating_point",
        "benchmark_contract_executable_without_model_load": True,
        "model_download_performed": False,
        "large_scale_inference_performed": False,
        "performance_or_sota_claim_authorized": False,
    }
    body["readiness_id"] = "UNIDETREADY-" + _canonical_sha256(body)[:24]
    return body


def evaluate_unified_accuracy_primary_gate_v1(
    *,
    plan: Mapping[str, Any] | None = None,
    eligible_alarm_provider_ids: Iterable[str] = (),
    complete_same_protocol_paired_comparison: bool = False,
) -> dict[str, Any]:
    """Evaluate the outer nullable gate without accepting self-reported metrics.

    This narrow function intentionally accepts only provider IDs that an upper
    layer has already proven eligible from native benchmark and paired-patient
    receipts.  It cannot inspect or bless metric values itself.  Consequently,
    no provider is selected when the paired comparison is incomplete, when
    fewer than two provider families are eligible, or when more than one
    provider remains (the Pareto/tie decision must be represented in a future
    content-bound comparison receipt).
    """

    validated_plan = (
        load_unified_continuous_detector_benchmark_plan_v1()
        if plan is None
        else validate_unified_continuous_detector_benchmark_plan_v1(
            dict(plan), verify_file_bindings=False
        )
    )
    ids = sorted({_identifier(value, "eligible alarm provider ID") for value in eligible_alarm_provider_ids})
    if type(complete_same_protocol_paired_comparison) is not bool:
        raise TypeError("paired comparison completion flag must be boolean")
    minimum = validated_plan["accuracy_primary_selection"][
        "minimum_same_protocol_provider_families"
    ]
    reasons: list[str] = []
    if not complete_same_protocol_paired_comparison:
        reasons.append("same_protocol_paired_comparison_incomplete")
    if len(ids) < minimum:
        reasons.append("fewer_than_two_eligible_alarm_provider_families")
    if len(ids) > 1:
        reasons.append("content_bound_pareto_winner_receipt_not_supplied")
    # A single ID can never satisfy the minimum-two comparison requirement;
    # multiple IDs need a future Pareto winner receipt.  This v1 outer gate is
    # therefore deliberately nullable and cannot be bypassed by caller metrics.
    body: dict[str, Any] = {
        "schema_version": UNIFIED_DETECTOR_SELECTION_SCHEMA_VERSION,
        "gate_id": "UNIFIED-DETECTOR-SELECTION-PENDING",
        "plan_receipt_sha256": validated_plan["receipt_sha256"],
        "eligible_alarm_provider_ids": ids,
        "complete_same_protocol_paired_comparison": (
            complete_same_protocol_paired_comparison
        ),
        "accuracy_primary": None,
        "selection_status": "no_content_bound_unique_qualified_winner",
        "null_reasons": reasons,
        "navigation_provider_selection_considered": False,
        "clinical_or_production_promotion_authorized": False,
    }
    body["gate_id"] = "UNIDETSELECT-" + _canonical_sha256(body)[:24]
    return body


__all__ = [
    "DEFAULT_UNIFIED_DETECTOR_PLAN_PATH",
    "DEFAULT_UNIFIED_DETECTOR_REGISTRY_PATH",
    "UNIFIED_DETECTOR_PLAN_SCHEMA_VERSION",
    "UNIFIED_DETECTOR_READINESS_SCHEMA_VERSION",
    "UNIFIED_DETECTOR_REGISTRY_SCHEMA_VERSION",
    "build_unified_detector_benchmark_readiness_v1",
    "evaluate_unified_accuracy_primary_gate_v1",
    "load_unified_continuous_detector_benchmark_plan_v1",
    "load_unified_detector_provider_registry_v1",
    "unified_detector_plan_self_sha256",
    "unified_detector_registry_self_sha256",
    "validate_unified_continuous_detector_benchmark_plan_v1",
    "validate_unified_detector_provider_registry_v1",
]
