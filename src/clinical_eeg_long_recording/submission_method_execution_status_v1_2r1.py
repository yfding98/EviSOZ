"""Additive machine-readable execution status for submission method v1.2r1."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .detector_channel_support_addendum_v1 import (
    validate_detector_channel_support_addendum_v1,
)
from .detector_selection_fit_phase_inventory_v1 import (
    validate_detector_selection_fit_phase_inventory_v1,
)
from .findings_onset_threshold_registry_v1 import (
    validate_findings_onset_threshold_preregistry_v1,
)
from .submission_method_profile_v1_2 import (
    validate_submission_method_profile_v1_2,
)
from . import eventnet_cleanroom_registry_v1 as _eventnet
from . import seizuretransformer_cleanroom_registry_v1 as _st


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = (
    ROOT / "configs" / "clinical_eeg_submission_method_execution_status_v1_2r1.json"
)
SCHEMA_VERSION = "clinical_eeg_submission_method_execution_status_v1_2r1"
STATUS_ID = "CLINICAL-EEG-SUBMISSION-METHOD-EXECUTION-STATUS-V1.2R1-20260824"
TRUSTED_RECEIPT_SHA256 = (
    "446683cfc817eb1fe1bde88bf9ad13191011b5bcc0b9e8602700a909d0a66a14"
)
_PENDING = "CONTENT-ADDRESS-PENDING"
_SHA_CHARS = frozenset("0123456789abcdef")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _strict_equal(actual: object, expected: object) -> bool:
    """Compare JSON-like values without Python's bool/int equality aliasing."""

    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        if set(actual) != set(expected):
            return False
        return all(_strict_equal(actual[key], expected[key]) for key in expected)
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _require_sha256(value: object, context: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA_CHARS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_file(binding: Mapping[str, Any], context: str) -> Path:
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise TypeError(f"{context} path must be a non-empty string")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{context} path escapes project root")
    path = (ROOT / candidate).resolve(strict=True)
    try:
        path.relative_to(ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{context} path escapes project root") from error
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{context} must be a regular non-symlink file")
    if _file_sha256(path) != _require_sha256(
        binding.get("file_sha256"), f"{context} file"
    ):
        raise ValueError(f"{context} bytes drifted")
    return path


def _json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not readable JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must contain an object")
    return value


_EXPECTED_SOFTWARE_STATUS = {
    "controller_reference_opaque_actual_byte_software_gate": True,
    "eventnet_phase_variant_target_loss_sampler_authority": True,
    "seizuretransformer_phase_variant_target_loss_sampler_authority": True,
    "five_real_selection_fit_phase_receipts": True,
    "synthetic_same_process_provider_epoch_exact_reload_conformance": True,
    "synthetic_hidden64_three_loss_exact_reload_conformance": True,
    "findings_threshold_persistence_trajectory_software_gate": True,
    "formal_complete_candidate_roster_and_kind_separated_projection": True,
}

_EXPECTED_REAL_ARTIFACT_COUNTS = {
    "selection_fit_phase_receipts": 5,
    "inner_validation_phase_receipts": 0,
    "final_refit_phase_receipts": 0,
    "eventnet_authorized_variant_rosters": 0,
    "seizuretransformer_authorized_variant_rosters": 0,
    "eventnet_real_checkpoints": 0,
    "seizuretransformer_real_checkpoints": 0,
    "detector_OOF_prediction_inventories": 0,
    "admitted_findings_threshold_registries": 0,
    "real_findings_stable_track_inventories": 0,
    "real_hidden64_BA_IEG_checkpoints": 0,
    "real_complete_ITA_reassemblies": 0,
}

_EXPECTED_EXECUTION_BOUNDARY = {
    "provider_executor_scope": (
        "synthetic_same_process_conformance_nonpromotable"
    ),
    "cross_process_formal_resume_admitted": False,
    "large_real_fold_streaming_scalability_admitted": False,
    "strict_ordered_architecture_execution_ledger_admitted": False,
    "findings_threshold_registry_status": (
        "unadmitted_no_real_crossfit_prediction_artifacts"
    ),
    "report_language_optimization_active": False,
    "website_or_viewer_active": False,
}

_EXPECTED_SOURCE_FIREWALL = {
    "model_forward_accepts_EEG_tensor": True,
    "allowlisted_acquisition_metadata_may_control_preprocessing": True,
    "public_source_train_TERM_seiz_used_as_training_target": True,
    "EDF_annotation_used_by_model_or_inference": False,
    "Excel_or_doctor_text_used_by_model_or_inference": False,
    "source_dev_or_eval_reference_used_for_training": False,
    "private_reference_or_labels_used_for_training": False,
    "video_behavior_sleep_activation_ECG_EMG_EOG_used": False,
}

_EXPECTED_PERMISSIONS = {
    "architecture_candidate_frozen": True,
    "real_detector_training_claim_authorized": False,
    "detector_performance_claim_authorized": False,
    "findings_performance_claim_authorized": False,
    "SOZ_performance_claim_authorized": False,
    "SOTA_claim_authorized": False,
    "clinical_report_or_diagnosis_authorized": False,
    "clinical_or_production_use_authorized": False,
}


def validate_submission_method_execution_status_v1_2r1(
    value: Mapping[str, Any], *, require_trusted_receipt: bool = True
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("execution status must be an object")
    row = deepcopy(dict(value))
    supplied = _require_sha256(row.get("receipt_sha256"), "execution status receipt")
    pending = deepcopy(row)
    pending["receipt_sha256"] = _PENDING
    if supplied != _canonical_sha256(pending):
        raise ValueError("execution status is not content-addressed")
    if require_trusted_receipt and supplied != TRUSTED_RECEIPT_SHA256:
        raise PermissionError("execution status receipt is not checked-in trusted state")
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("status_id") != STATUS_ID
        or row.get("status")
        != "software_controls_advanced_real_training_and_performance_incomplete"
    ):
        raise ValueError("execution status identity drifted")
    if not _strict_equal(row.get("software_status"), _EXPECTED_SOFTWARE_STATUS):
        raise ValueError("execution software status drifted")
    if not _strict_equal(
        row.get("real_artifact_counts"), _EXPECTED_REAL_ARTIFACT_COUNTS
    ):
        raise ValueError("execution artifact counts drifted")
    if not _strict_equal(row.get("execution_boundary"), _EXPECTED_EXECUTION_BOUNDARY):
        raise ValueError("execution boundary drifted")
    if not _strict_equal(row.get("source_firewall"), _EXPECTED_SOURCE_FIREWALL):
        raise PermissionError("execution source firewall drifted")
    if not _strict_equal(row.get("scientific_permissions"), _EXPECTED_PERMISSIONS):
        raise PermissionError("execution scientific permissions drifted")

    bindings = row.get("bindings")
    expected_binding_names = {
        "base_method_profile",
        "channel_support_addendum",
        "selection_fit_inventory",
        "provider_epoch_executor",
        "eventnet_registry",
        "seizuretransformer_registry",
        "findings_threshold_preregistry",
        "formal_complete_candidate_roster_implementation",
        "hidden64_conformance_test",
    }
    if type(bindings) is not dict or set(bindings) != expected_binding_names:
        raise ValueError("execution status binding roster drifted")
    paths = {
        name: _project_file(binding, name) for name, binding in bindings.items()
    }

    profile = validate_submission_method_profile_v1_2(
        _json(paths["base_method_profile"], "base method profile")
    )
    if profile["receipt_sha256"] != bindings["base_method_profile"]["semantic_sha256"]:
        raise ValueError("base method profile semantic receipt drifted")
    support = validate_detector_channel_support_addendum_v1(
        _json(paths["channel_support_addendum"], "channel support addendum")
    )
    if support["addendum_sha256"] != bindings["channel_support_addendum"]["semantic_sha256"]:
        raise ValueError("channel support addendum semantic receipt drifted")
    inventory = validate_detector_selection_fit_phase_inventory_v1(
        _json(paths["selection_fit_inventory"], "selection-fit inventory"),
        verify_bound_files=False,
    )
    if (
        inventory["receipt_sha256"]
        != bindings["selection_fit_inventory"]["semantic_sha256"]
        or inventory["aggregate"]["phase_receipt_count"] != 5
    ):
        raise ValueError("selection-fit inventory semantic receipt drifted")
    if (
        _eventnet.load_registry(paths["eventnet_registry"])["registry_sha256"]
        != bindings["eventnet_registry"]["semantic_sha256"]
    ):
        raise ValueError("EventNet registry semantic receipt drifted")
    if (
        _st.load_registry(paths["seizuretransformer_registry"])["registry_sha256"]
        != bindings["seizuretransformer_registry"]["semantic_sha256"]
    ):
        raise ValueError("SeizureTransformer registry semantic receipt drifted")
    findings = validate_findings_onset_threshold_preregistry_v1(
        _json(
            paths["findings_threshold_preregistry"],
            "Findings threshold preregistry",
        )
    )
    if (
        findings["registry_receipt_sha256"]
        != bindings["findings_threshold_preregistry"]["semantic_sha256"]
        or findings["status"]
        != "unadmitted_no_real_crossfit_prediction_artifacts"
    ):
        raise ValueError("Findings threshold preregistry state drifted")
    for name in (
        "provider_epoch_executor",
        "formal_complete_candidate_roster_implementation",
        "hidden64_conformance_test",
    ):
        if bindings[name].get("semantic_sha256") is not None:
            raise ValueError(f"{name} is source-byte evidence, not a semantic receipt")
    return deepcopy(value)


def load_submission_method_execution_status_v1_2r1(
    path: str | Path = DEFAULT_PATH,
) -> dict[str, Any]:
    return validate_submission_method_execution_status_v1_2r1(
        _json(Path(path).resolve(strict=True), "execution status")
    )


__all__ = [
    "DEFAULT_PATH",
    "SCHEMA_VERSION",
    "STATUS_ID",
    "TRUSTED_RECEIPT_SHA256",
    "load_submission_method_execution_status_v1_2r1",
    "validate_submission_method_execution_status_v1_2r1",
]
