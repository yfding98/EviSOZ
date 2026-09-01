"""Fail-closed validator for the BA-IEG onset-identity addendum v1.1.

The immutable v1 core contract is not edited.  This addendum binds a narrow
repair for the typed-unit training seam: whole-course causal rank pooling is
forbidden; identity evidence must overlap a target-free global causal onset
gate and the typed-unit boundary distribution.  Public event intervals may
train only an event-level boundary MIL objective, never per-unit identity.

Validation freezes method permissions and joint-checkpoint admission state.
It does not establish seizure-detection or SOZ-localization performance.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, Mapping, Sequence

from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID,
    BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID,
    BAIEGCausalTypedUnitTrace,
)
from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BA_IEG_CAUSAL_ONSET_CENTRAL_SUPPORT_MASS,
    BA_IEG_CAUSAL_ONSET_IDENTITY_POLICY_SHA256,
    BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID,
    BAIEGShallowCausalTypedUnitHeadOutput,
    BAIEGShallowCausalTypedUnitOnsetHead,
)
from .ba_ieg_shallow_causal_typed_unit_supervision_v1 import (
    BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_LOSS_CONTRACT_SHA256,
    BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_SUPERVISION_ID,
    build_ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v1,
    shallow_causal_typed_unit_mil_boundary_loss_v1,
)


BA_IEG_ONSET_IDENTITY_ADDENDUM_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_ba_ieg_onset_identity_addendum_v1_1"
)
BA_IEG_ONSET_IDENTITY_ADDENDUM_ID: Final[str] = (
    "CLINICAL-EEG-BA-IEG-ONSET-IDENTITY-ADDENDUM-V1.1-20260824"
)
TRUSTED_BA_IEG_ONSET_IDENTITY_ADDENDUM_RECEIPT_SHA256: Final[str] = (
    "9463f7ff25136b3b8fddb7ef8a394de2bc9b3e123d4c1a61e1923aecb378916b"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BA_IEG_ONSET_IDENTITY_ADDENDUM_PATH: Final[Path] = (
    _ROOT
    / "configs"
    / "clinical_eeg_ba_ieg_onset_identity_addendum_v1_1.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "addendum_id",
    "status",
    "base_core_contract",
    "implementation_bindings",
    "global_target_free_onset_gate",
    "event_level_boundary_mil",
    "onset_identity_association",
    "required_counterexample_contracts",
    "source_firewall",
    "joint_checkpoint_admission",
    "scientific_permissions",
    "receipt_sha256",
}

_EXPECTED_BASE: Mapping[str, object] = {
    "path": "configs/clinical_eeg_ba_ieg_v1_core_freeze.json",
    "contract_sha256": (
        "d02cb0044555195cb697b4e3e210f0dd6d55378d693c1e35f6a778420a09be91"
    ),
    "base_contract_mutation_required": False,
    "late_spread_clause_repaired_by_addendum": True,
}
_EXPECTED_BINDINGS: Mapping[str, object] = {
    "segmental_model_id": BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID,
    "causal_trace_id": BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID,
    "typed_unit_head_id": BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID,
    "typed_unit_supervision_id": (
        BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_SUPERVISION_ID
    ),
    "causal_axis_schema": "ba_ieg_causal_typed_unit_supervision_axis_v1",
    "onset_identity_policy_sha256": (
        BA_IEG_CAUSAL_ONSET_IDENTITY_POLICY_SHA256
    ),
    "boundary_mil_loss_contract_sha256": (
        BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_LOSS_CONTRACT_SHA256
    ),
}
_EXPECTED_GLOBAL_GATE: Mapping[str, object] = {
    "source_field": "global_onset_boundary_mass",
    "unresolved_status_fields": [
        "global_left_censor_state_mass",
        "global_no_onset_within_support_mass",
    ],
    "support_policy": "equal_tail_central_observed_onset_mass",
    "central_support_mass": 0.95,
    "observed_status_dominance_rule": (
        "observed_onset_mass_gt_left_censor_plus_no_onset_mass"
    ),
    "public_target_mask_used_in_forward": False,
    "patient_identity_gradient_to_global_gate": False,
    "left_or_no_onset_unresolved_is_evaluable": False,
}
_EXPECTED_BOUNDARY_MIL: Mapping[str, object] = {
    "optimization_split": "source_train",
    "target_authority": "public_seizure_interval",
    "target_fact": "event_level_observed_onset_interval_only",
    "objective": (
        "negative_log_noisy_or_at_least_one_eligible_typed_unit_boundary_"
        "in_projected_onset_support"
    ),
    "event_positive_copied_to_each_typed_unit": False,
    "supervised_head": "boundary_head_and_shared_fusion",
    "typed_unit_event_or_rank_logits_supervised": False,
    "patient_positive_set_or_channel_target_used": False,
    "offline_lane_or_spread_target_used": False,
    "not_evaluable_used_as_negative": False,
}
_EXPECTED_ASSOCIATION: Mapping[str, object] = {
    "formula": (
        "sum_g(detached_global_gate_g*typed_boundary_mass_gk*"
        "sigmoid(cell_rank_logit_gk))"
    ),
    "event_logit": "logit(clamped_joint_onset_identity_mass)",
    "whole_causal_course_rank_logmeanexp_used": False,
    "late_only_unit_without_global_support_overlap_is_evaluable": False,
    "left_unresolved_event_is_evaluable": False,
    "physical_projection_uses_onset_association_mask": True,
    "per_unit_identity_target_stage": (
        "event_to_record_to_complete_patient_positive_set_only"
    ),
}
_EXPECTED_COUNTEREXAMPLES = (
    "late_only_cell_rank_raises_legacy_lme_but_not_joint_onset_identity",
    "late_only_unit_without_global_support_overlap_is_not_evaluable",
    "left_unresolved_event_is_not_evaluable",
    "patient_identity_path_has_no_gradient_to_global_onset_mass",
    "one_unit_can_explain_event_boundary_without_unit_label_broadcast",
    "causal_axis_time_bounds_or_mask_tamper_fails_closed",
)
_EXPECTED_SOURCE_FIREWALL: Mapping[str, bool] = {
    "eeg_signal_and_public_event_interval_used": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "private_data_used": False,
    "clinical_text_or_report_used": False,
    "video_or_behavior_used": False,
    "sleep_activation_or_other_physiology_used": False,
}
_EXPECTED_JOINT_ADMISSION: Mapping[str, object] = {
    "status": "prohibited",
    "disk_training_runner_connected": False,
    "optimizer_loss_composition_registered": False,
    "immutable_training_receipt_available": False,
    "trained_checkpoint_available": False,
    "required_before_admission": [
        (
            "registered_source_train_runner_uses_segmental_onset_plus_boundary_"
            "mil_plus_complete_patient_positive_set_losses"
        ),
        "all_required_counterexample_contracts_pass",
        "immutable_training_configuration_and_checkpoint_receipts_replay",
        (
            "held_out_source_evaluation_completed_without_private_or_doctor_"
            "target_access"
        ),
    ],
}
_EXPECTED_SCIENTIFIC_PERMISSIONS: Mapping[str, bool] = {
    "component_contract_tests_are_model_performance": False,
    "soz_accuracy_claim_authorized": False,
    "sota_claim_authorized": False,
    "clinical_or_production_use_authorized": False,
    "report_route_connected": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def ba_ieg_onset_identity_addendum_self_sha256(
    value: Mapping[str, object],
) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("BA-IEG onset-identity addendum must be an object")
    body = deepcopy(dict(value))
    body.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


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
                ValueError(f"{context} contains non-finite JSON token {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not valid UTF-8") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must contain a JSON object")
    return value


def _strict_object(
    value: object, expected: Mapping[str, object], context: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    if set(value) != set(expected):
        missing = set(expected) - set(value)
        unknown = set(value) - set(expected)
        raise ValueError(
            f"{context} keys drifted; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if value != expected:
        raise ValueError(f"{context} policy drifted")
    return deepcopy(value)


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _safe_project_file(relative: object, context: str) -> Path:
    if not isinstance(relative, str):
        raise TypeError(f"{context} must be a relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{context} is not a canonical relative path")
    root = _ROOT.resolve(strict=True)
    unresolved = root.joinpath(*pure.parts)
    if unresolved.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    candidate = unresolved.resolve(strict=True)
    candidate.relative_to(root)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{context} must resolve to a regular non-symlink file")
    return candidate


def _validate_base_core(value: object) -> dict[str, Any]:
    row = _strict_object(value, _EXPECTED_BASE, "base_core_contract")
    core = _load_strict_json(
        _safe_project_file(row["path"], "base_core_contract.path"),
        "base BA-IEG core contract",
    )
    if core.get("contract_sha256") != row["contract_sha256"]:
        raise ValueError("base BA-IEG core contract binding drifted")
    body = deepcopy(core)
    body.pop("contract_sha256", None)
    if hashlib.sha256(_canonical_json_bytes(body)).hexdigest() != row[
        "contract_sha256"
    ]:
        raise ValueError("base BA-IEG core contract self-hash replay failed")
    return row


def _validate_code_bindings() -> None:
    if BA_IEG_CAUSAL_ONSET_CENTRAL_SUPPORT_MASS != 0.95:
        raise ValueError("causal onset central-support mass drifted from addendum")
    trace_fields = {item.name for item in fields(BAIEGCausalTypedUnitTrace)}
    required_trace = {
        "global_onset_boundary_mass",
        "global_left_censor_state_mass",
        "global_no_onset_within_support_mass",
    }
    if not required_trace.issubset(trace_fields):
        raise ValueError("causal trace lacks frozen global onset status masses")
    output_fields = {
        item.name for item in fields(BAIEGShallowCausalTypedUnitHeadOutput)
    }
    required_output = {
        "causal_global_onset_support_mask",
        "causal_global_onset_resolved_mask",
        "typed_unit_inventory_mask",
        "typed_unit_onset_association_mass",
        "typed_unit_onset_identity_mass",
    }
    if not required_output.issubset(output_fields):
        raise ValueError("typed-unit head lacks frozen onset-association outputs")
    if set(
        inspect.signature(BAIEGShallowCausalTypedUnitOnsetHead.forward).parameters
    ) != {"self", "trace"}:
        raise ValueError("typed-unit forward opened a target-conditioned input")
    if set(
        inspect.signature(
            build_ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v1
        ).parameters
    ) != {"trace", "targets", "projections"}:
        raise ValueError("typed-unit MIL target-builder API drifted")
    if set(
        inspect.signature(shallow_causal_typed_unit_mil_boundary_loss_v1).parameters
    ) != {"output", "target_bundle"}:
        raise ValueError("typed-unit MIL loss API drifted")


def validate_clinical_eeg_ba_ieg_onset_identity_addendum_v1_1(
    value: Mapping[str, object],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("BA-IEG onset-identity addendum must be a JSON object")
    candidate = deepcopy(value)
    if set(candidate) != _TOP_LEVEL_KEYS:
        missing = _TOP_LEVEL_KEYS - set(candidate)
        unknown = set(candidate) - _TOP_LEVEL_KEYS
        raise ValueError(
            f"onset-identity addendum keys drifted; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if candidate["schema_version"] != (
        BA_IEG_ONSET_IDENTITY_ADDENDUM_SCHEMA_VERSION
    ) or candidate["addendum_id"] != BA_IEG_ONSET_IDENTITY_ADDENDUM_ID:
        raise ValueError("onset-identity addendum identity drifted")
    if candidate["status"] != (
        "method_contract_frozen_joint_checkpoint_not_admitted_"
        "performance_not_established"
    ):
        raise ValueError("onset-identity addendum status was promoted")
    receipt = _sha256(candidate["receipt_sha256"], "receipt_sha256")
    replayed = ba_ieg_onset_identity_addendum_self_sha256(candidate)
    if receipt != replayed:
        raise ValueError("onset-identity addendum canonical self-hash mismatch")

    _validate_base_core(candidate["base_core_contract"])
    _strict_object(
        candidate["implementation_bindings"],
        _EXPECTED_BINDINGS,
        "implementation_bindings",
    )
    _strict_object(
        candidate["global_target_free_onset_gate"],
        _EXPECTED_GLOBAL_GATE,
        "global_target_free_onset_gate",
    )
    _strict_object(
        candidate["event_level_boundary_mil"],
        _EXPECTED_BOUNDARY_MIL,
        "event_level_boundary_mil",
    )
    _strict_object(
        candidate["onset_identity_association"],
        _EXPECTED_ASSOCIATION,
        "onset_identity_association",
    )
    if tuple(candidate["required_counterexample_contracts"]) != (
        _EXPECTED_COUNTEREXAMPLES
    ):
        raise ValueError("required onset-identity counterexamples drifted")
    _strict_object(
        candidate["source_firewall"],
        _EXPECTED_SOURCE_FIREWALL,
        "source_firewall",
    )
    _strict_object(
        candidate["joint_checkpoint_admission"],
        _EXPECTED_JOINT_ADMISSION,
        "joint_checkpoint_admission",
    )
    _strict_object(
        candidate["scientific_permissions"],
        _EXPECTED_SCIENTIFIC_PERMISSIONS,
        "scientific_permissions",
    )
    _validate_code_bindings()
    if receipt != TRUSTED_BA_IEG_ONSET_IDENTITY_ADDENDUM_RECEIPT_SHA256:
        raise ValueError("onset-identity addendum receipt is not trusted")
    return candidate


def load_clinical_eeg_ba_ieg_onset_identity_addendum_v1_1(
    path: str | Path = DEFAULT_BA_IEG_ONSET_IDENTITY_ADDENDUM_PATH,
) -> dict[str, Any]:
    candidate_path = Path(path)
    if not candidate_path.is_absolute():
        candidate_path = (_ROOT / candidate_path).resolve()
    return validate_clinical_eeg_ba_ieg_onset_identity_addendum_v1_1(
        _load_strict_json(candidate_path, "BA-IEG onset-identity addendum")
    )


__all__ = [
    "BA_IEG_ONSET_IDENTITY_ADDENDUM_ID",
    "BA_IEG_ONSET_IDENTITY_ADDENDUM_SCHEMA_VERSION",
    "DEFAULT_BA_IEG_ONSET_IDENTITY_ADDENDUM_PATH",
    "TRUSTED_BA_IEG_ONSET_IDENTITY_ADDENDUM_RECEIPT_SHA256",
    "ba_ieg_onset_identity_addendum_self_sha256",
    "load_clinical_eeg_ba_ieg_onset_identity_addendum_v1_1",
    "validate_clinical_eeg_ba_ieg_onset_identity_addendum_v1_1",
]
