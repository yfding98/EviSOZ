"""Validator for the target-blind detector channel-support addendum v1."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .detector_channel_support_router_v1 import (
    detector_channel_support_policy_receipt,
    validate_detector_channel_support_audit,
)
from .detector_signal_lineage_authority_v1 import (
    SCHEMA_VERSION as SIGNAL_LINEAGE_AUTHORITY_SCHEMA_VERSION,
)
from .tusz_missing_midline_source_audit_v1 import (
    validate_tusz_missing_midline_source_audit_v1,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    ROOT / "configs" / "clinical_eeg_detector_channel_support_routing_addendum_v1.json"
)
SCHEMA_VERSION = "clinical_eeg_detector_channel_support_routing_addendum_v1"
ADDENDUM_ID = "CLINICAL-EEG-DETECTOR-CHANNEL-SUPPORT-ROUTING-ADDENDUM-V1-20260824"
DEFAULT_ADDENDUM_SHA256 = (
    "3dc6a2a1f1fc6612332668b01df341f36420e026f04b3252dfeb72fe50a0e051"
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_detector_channel_support_addendum_v1(
    value: Mapping[str, Any],
    *,
    trusted_addendum_sha256: str = DEFAULT_ADDENDUM_SHA256,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("detector channel support addendum must be an object")
    config = deepcopy(dict(value))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("detector channel support addendum schema drifted")
    if config.get("addendum_id") != ADDENDUM_ID:
        raise ValueError("detector channel support addendum identifier drifted")
    supplied_hash = config.get("addendum_sha256")
    body = deepcopy(config)
    body.pop("addendum_sha256", None)
    if supplied_hash != _canonical_sha256(body) or supplied_hash != trusted_addendum_sha256:
        raise ValueError("detector channel support addendum does not replay exactly")

    parents = config.get("parent_bindings")
    if not isinstance(parents, list) or len(parents) != 2:
        raise ValueError("detector channel support addendum requires two parents")
    for row in parents:
        if not isinstance(row, Mapping):
            raise TypeError("detector channel support parent binding must be an object")
        path = ROOT / str(row.get("path", ""))
        if not path.is_file() or _file_sha256(path) != row.get("file_sha256"):
            raise ValueError(f"detector channel support parent binding drifted: {path}")

    implementation = config["policy_implementation"]
    implementation_path = ROOT / str(implementation["path"])
    if not implementation_path.is_file() or _file_sha256(implementation_path) != implementation[
        "file_sha256"
    ]:
        raise ValueError("detector channel support policy implementation drifted")
    policy = detector_channel_support_policy_receipt()
    if implementation["method_id"] != policy["method_id"]:
        raise ValueError("detector channel support method drifted")
    if implementation["policy_sha256"] != policy["policy_sha256"]:
        raise ValueError("detector channel support policy hash drifted")

    lineage = config["signal_lineage_authority_implementation"]
    lineage_path = ROOT / str(lineage["path"])
    if (
        not lineage_path.is_file()
        or _file_sha256(lineage_path) != lineage["file_sha256"]
        or lineage["schema_version"] != SIGNAL_LINEAGE_AUTHORITY_SCHEMA_VERSION
        or lineage["provider_authority_root"]
        != "CanonicalEEGRecord_object_plus_actual_payload_semantic_replay_v1"
        or lineage["policy_audit_authority_root"]
        != "validated_canonical_physical_audit_and_projection_artifact_membership_v1"
        or lineage["self_hash_only_accepted"] is not False
    ):
        raise ValueError("typed detector signal-lineage authority binding drifted")

    authority = config["routing_authority"]
    if authority["allowed"] != [
        "typed_canonical_physical_signal_authority",
        "typed_observed_physical_roster_authority",
        "typed_EEG_electrical_reference_system_authority",
        "typed_EEG_only_channel_QC_authority",
    ]:
        raise ValueError("detector channel support routing authority drifted")
    if (
        authority["bare_observed_or_usable_roster_allowed"] is not False
        or authority["bare_SHA256_lineage_allowed"] is not False
        or authority[
            "EEG_electrical_reference_provenance_is_legitimate_signal_control_plane"
        ]
        is not True
        or authority["seizure_target_or_reference_label_allowed"] is not False
    ):
        raise ValueError("detector channel routing trust/reference semantics drifted")
    if authority["one_route_per_record_before_neural_inference"] is not True:
        raise ValueError("detector route must be selected before inference")
    if authority["route_reselection_after_neural_inference_allowed"] is not False:
        raise ValueError("posterior-dependent detector rerouting is forbidden")
    required_forbidden = {
        "EDF_annotation",
        "reference_seizure_interval",
        "Excel_or_spreadsheet",
        "doctor_label_or_report",
        "clinical_text",
        "patient_or_subject_identity",
        "detector_posterior",
    }
    if set(authority["forbidden"]) != required_forbidden:
        raise ValueError("detector channel support forbidden authority drifted")

    provider = config["full_roster_provider_policy"]
    if provider["primary_candidate"] != "support_routed_complete19_then_lateral17":
        raise ValueError("detector full-roster primary candidate drifted")
    if provider["primary_missing_channel_zero_fill_or_interpolation_allowed"] is not False:
        raise ValueError("primary detector channel synthesis cannot be enabled")
    if provider["record_may_be_deleted_from_scoring_denominator"] is not False:
        raise ValueError("unsupported detector records cannot be deleted")
    if provider["policy_route_must_not_be_reported_as_provider_executable"] is not True:
        raise ValueError("policy route/provider execution distinction drifted")
    if provider["complete19"]["seizuretransformer_variant"] != policy["profiles"][
        "complete19"
    ]["seizuretransformer_variant"]:
        raise ValueError("ST18 route drifted")
    if provider["lateral17"]["seizuretransformer_variant"] != policy["profiles"][
        "lateral17"
    ]["seizuretransformer_variant"]:
        raise ValueError("ST16 route drifted")
    for profile_id in ("complete19", "lateral17"):
        if (
            provider[profile_id]["policy_route_available"] is not True
            or provider[profile_id]["full_stack_provider_executable"] is not False
        ):
            raise ValueError("policy route was conflated with provider execution")
    for profile_id in ("complete19", "lateral17"):
        if provider[profile_id]["eventnet_materialization"] != policy["profiles"][
            profile_id
        ]["provider_materialization"]["eventnet"]:
            raise ValueError("EventNet clean-room materialization ledger drifted")
    if provider["other_partial"][
        "future_challenger_may_provide_Findings_or_SOZ_spatial_evidence"
    ] is not False:
        raise ValueError("coverage fallback cannot provide spatial evidence")

    comparison = config["paired_policy_comparison"]
    if comparison["accuracy_primary_before_source_eval"] is not None:
        raise ValueError("accuracy primary cannot be selected before source-eval")
    if comparison["conditional_complete19_only_ablation_may_be_primary"] is not False:
        raise ValueError("complete-case conditional ablation cannot become primary")

    audit_binding = config["canonical_TUSZ_support_audit"]
    audit_path = ROOT / str(audit_binding["path"])
    if not audit_path.is_file() or _file_sha256(audit_path) != audit_binding["file_sha256"]:
        raise ValueError("detector channel support audit binding drifted")
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = validate_detector_channel_support_audit(json.load(handle))
    if audit["receipt_sha256"] != audit_binding["receipt_sha256"]:
        raise ValueError("detector channel support audit receipt drifted")
    if audit["canonical_weight_one_identity_count"] != 7349:
        raise ValueError("detector channel support audit denominator drifted")
    if audit["policy_route_available_identity_count"] != 7349:
        raise ValueError("canonical TUSZ policy route no longer covers every identity")
    if audit["provider_unmaterialized_identity_count"] != 7349:
        raise ValueError("canonical TUSZ provider-unmaterialized count drifted")
    if audit["full_stack_executable_identity_count"] != 0:
        raise ValueError("canonical TUSZ audit overclaims executable providers")
    if audit["terminal_support_policy_failure_identity_count"] != 0:
        raise ValueError("canonical TUSZ support policy has failures")
    for field in (
        "canonical_weight_one_identity_count",
        "policy_route_available_identity_count",
        "provider_unmaterialized_identity_count",
        "full_stack_executable_identity_count",
        "terminal_support_policy_failure_identity_count",
    ):
        if audit_binding[field] != audit[field]:
            raise ValueError("canonical TUSZ support audit summary drifted")

    source_resolution = config["source_missing_midline_resolution"]
    source_resolution_path = ROOT / str(source_resolution["path"])
    if (
        not source_resolution_path.is_file()
        or _file_sha256(source_resolution_path) != source_resolution["file_sha256"]
    ):
        raise ValueError("missing-midline source audit binding drifted")
    with source_resolution_path.open("r", encoding="utf-8") as handle:
        missing_midline = validate_tusz_missing_midline_source_audit_v1(
            json.load(handle)
        )
    if missing_midline["receipt_sha256"] != source_resolution["receipt_sha256"]:
        raise ValueError("missing-midline source receipt drifted")
    if missing_midline["missing_midline_canonical_record_count"] != 249:
        raise ValueError("missing-midline source denominator drifted")
    if missing_midline["source_classification_counts"] != {
        "FZ_and_PZ_nodes_genuinely_absent_from_raw_signal_labels": 249
    }:
        raise ValueError("missing-midline source classification drifted")
    if missing_midline["all_missing_records_ST18_midline_derivable"] is not False:
        raise ValueError("missing-midline ST18 derivability claim drifted")
    if (
        missing_midline[
            "all_missing_records_EventNet19_midline_referential_derivable"
        ]
        is not False
    ):
        raise ValueError("missing-midline EventNet19 derivability claim drifted")
    if source_resolution["neighbor_interpolation_or_zero_fill_promoted_to_primary"] is not False:
        raise ValueError("missing-midline source audit cannot authorize imputation")

    firewall = config["evidence_firewall"]
    if firewall != {
        "detector_carrier_is_a_Finding": False,
        "detector_route_or_missingness_is_SOZ_evidence": False,
        "Findings_remeasure_observed_native_EEG": True,
        "whole_bipolar_lead_endpoint_attribution_allowed": False,
    }:
        raise ValueError("detector support evidence firewall drifted")

    boundary = config["scientific_claim_boundary"]
    for key in (
        "channel_support_policy_is_executable",
        "typed_signal_lineage_authority_is_executable",
        "canonical_TUSZ_support_audit_is_policy_route_only",
        "ST18_and_ST16_transform_and_OLA_contracts_exist",
        "EventNet19_and_EventNet17_transform_architecture_target_loss_tiling_sampler_contracts_exist",
    ):
        if boundary[key] is not True:
            raise ValueError(f"materialized detector contract hidden: {key}")
    if boundary["channel_support_policy_is_executable"] is not True:
        raise ValueError("executable routing policy cannot be hidden")
    for key in (
        "ST18_or_ST16_checkpoint_exists",
        "EventNet17_full_stack_executable",
        "full_stack_provider_execution_coverage_established",
        "detector_accuracy_or_efficiency_gain_established",
        "private_dataset_full_roster_coverage_established",
        "clinical_or_production_use",
    ):
        if boundary[key] is not False:
            raise ValueError(f"unsupported detector channel claim opened: {key}")
    return deepcopy(config)


def load_and_validate_detector_channel_support_addendum_v1(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return validate_detector_channel_support_addendum_v1(json.load(handle))


__all__ = [
    "ADDENDUM_ID",
    "DEFAULT_ADDENDUM_SHA256",
    "DEFAULT_CONFIG_PATH",
    "SCHEMA_VERSION",
    "load_and_validate_detector_channel_support_addendum_v1",
    "validate_detector_channel_support_addendum_v1",
]
