"""Target-blind channel-support routing for clean-room long-EEG detectors.

The accuracy and efficiency detector arms do not share an input montage:
SeizureTransformer consumes 18 longitudinal bipolar units while EventNet
consumes 19 referential electrodes.  The canonical TUSZ physical roster also
contains records in which FZ and PZ are both absent.  Deleting those records
would change the prediction-first denominator; silently synthesising FZ/PZ
would create a detector carrier that could be mistaken for spatial evidence.

This module therefore implements a deterministic, typed-lineage-only route:

* all 19 standard electrodes usable -> ST18 and EventNet19 variants;
* the 17 lateral/CZ electrodes usable -> ST16 and EventNet17 variants;
* anything less -> a retained support-policy failure row until a separately
  qualified masked-coverage provider exists.

ST16/EventNet17 are specified as independent clean-room model variants.  They
are not tensors obtained by zero filling a 19-channel model.  The EventNet
EN19/EN17 direct transforms, architectures, and target/loss/tiling/sampling
contracts are materialized, but neither has an epoch executor, clean-room
checkpoint, or decoded full-record inference stack.  A policy route is
therefore not the same as an executable provider.  Detector routes are
navigation-only and are never an authority for Findings or SOZ spatial
evidence.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .detector_signal_lineage_authority_v1 import (
    ValidatedDetectorSignalLineageAuthority,
    authorize_detector_policy_lineage_from_canonical_audit,
    load_canonical_policy_audit_trust_anchor,
    require_validated_detector_signal_lineage_authority,
)


SCHEMA_VERSION = "clinical_eeg_detector_channel_support_route_v2"
METHOD_ID = (
    "typed_lineage_target_blind_support_policy_complete19_lateral17_v2"
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

MIDLINE_OPTIONAL = ("FZ", "PZ")
LATERAL_17 = tuple(channel for channel in STANDARD_19 if channel not in MIDLINE_OPTIONAL)

ST18_UNITS = (
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FZ-CZ",
    "CZ-PZ",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
)
ST16_UNITS = tuple(unit for unit in ST18_UNITS if unit not in {"FZ-CZ", "CZ-PZ"})

# Exact first-axis order used by the EventNet release and retained by each
# independently trained clean-room input-width variant.  This is intentionally
# not the same as the canonical ontology order above.
EVENTNET19_UNITS = (
    "FP1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T7",
    "P7",
    "FZ",
    "CZ",
    "PZ",
    "FP2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T8",
    "P8",
)
EVENTNET17_UNITS = tuple(
    channel for channel in EVENTNET19_UNITS if channel not in MIDLINE_OPTIONAL
)

_PROFILES: dict[str, dict[str, Any]] = {
    "complete19": {
        "required_electrodes": list(STANDARD_19),
        "seizuretransformer_variant": "seizuretransformer_st18_cleanroom_v1",
        "seizuretransformer_typed_units": list(ST18_UNITS),
        "eventnet_variant": "eventnet_en19_cleanroom_v1",
        "eventnet_typed_units": list(EVENTNET19_UNITS),
        "primary_channel_imputation": False,
        "navigation_only": True,
        "provider_materialization": {
            "seizuretransformer": {
                "transform_contract_materialized": True,
                "streaming_OLA_geometry_materialized": True,
                "trainer_contract_materialized": True,
                "checkpoint_count": 0,
                "full_stack_executable": False,
                "reason_code": "cleanroom_checkpoint_absent",
            },
            "eventnet": {
                "direct_referential_input_transform_materialized": True,
                "cleanroom_architecture_materialized": True,
                "target_loss_tiling_sampler_contract_materialized": True,
                "PyTorch_epoch_training_executor_materialized": False,
                "cleanroom_checkpoint_inference_decoder_materialized": False,
                "checkpoint_count": 0,
                "full_stack_executable": False,
                "reason_code": (
                    "epoch_executor_checkpoint_and_inference_decoder_absent"
                ),
            },
        },
    },
    "lateral17": {
        "required_electrodes": list(LATERAL_17),
        "seizuretransformer_variant": "seizuretransformer_st16_common_support_cleanroom_v1",
        "seizuretransformer_typed_units": list(ST16_UNITS),
        "eventnet_variant": "eventnet_en17_common_support_cleanroom_v1",
        "eventnet_typed_units": list(EVENTNET17_UNITS),
        "primary_channel_imputation": False,
        "navigation_only": True,
        "provider_materialization": {
            "seizuretransformer": {
                "transform_contract_materialized": True,
                "streaming_OLA_geometry_materialized": True,
                "trainer_contract_materialized": True,
                "checkpoint_count": 0,
                "full_stack_executable": False,
                "reason_code": "independent_ST16_checkpoint_absent",
            },
            "eventnet": {
                "direct_referential_input_transform_materialized": True,
                "cleanroom_architecture_materialized": True,
                "target_loss_tiling_sampler_contract_materialized": True,
                "PyTorch_epoch_training_executor_materialized": False,
                "cleanroom_checkpoint_inference_decoder_materialized": False,
                "checkpoint_count": 0,
                "full_stack_executable": False,
                "reason_code": (
                    "epoch_executor_checkpoint_and_inference_decoder_absent"
                ),
            },
        },
    },
}

_FORBIDDEN_ROUTING_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "label",
        "labels",
        "reference_event",
        "seizure_reference_label",
        "seizure_target",
        "excel",
        "doctor_text",
        "clinical_text",
        "patient_id",
        "subject_id",
        "identity",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def detector_channel_support_policy() -> dict[str, Any]:
    """Return the immutable routing policy without a self-referential hash."""

    return {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "routing_authority": [
            "typed_canonical_physical_signal_authority",
            "typed_observed_physical_roster_authority",
            "typed_EEG_electrical_reference_system_authority",
            "typed_EEG_only_channel_QC_authority",
        ],
        "forbidden_routing_authority": sorted(_FORBIDDEN_ROUTING_KEYS),
        "precedence": ["complete19", "lateral17", "unadmitted_partial"],
        "profiles": deepcopy(_PROFILES),
        "training_population": {
            "complete19_variants": (
                "fold_train_records_with_complete19_support_only"
            ),
            "lateral17_common_support_variants": (
                "all_fold_train_records_with_lateral17_support_including_complete19"
            ),
            "patient_disjoint_nested_cross_fit_required": True,
            "route_or_model_selection_from_targets_allowed": False,
        },
        "inference_policy": {
            "one_policy_route_per_record_before_neural_inference": True,
            "route_reselection_from_detector_posterior_allowed": False,
            "missing_channel_zero_fill_in_primary_allowed": False,
            "unadmitted_partial_policy": (
                "retain_prediction_first_terminal_support_policy_failure_row"
            ),
            "support_policy_failure_may_be_deleted_from_denominator": False,
            "policy_route_implies_provider_executable": False,
            "provider_checkpoint_required_for_executable_status": True,
        },
        "evidence_firewall": {
            "detector_carrier_is_finding_source": False,
            "detector_route_is_SOZ_spatial_evidence": False,
            "findings_must_remeasure_observed_native_EEG": True,
            "whole_bipolar_lead_may_be_split_to_endpoints": False,
        },
        "reference_semantics": {
            "EEG_electrical_reference_provenance_is_required_signal_fact": True,
            "seizure_target_or_reference_label_is_forbidden": True,
            "the_two_reference_meanings_must_not_share_a_permission_flag": True,
        },
        "evaluation": {
            "primary_candidate": "support_routed_complete19_then_lateral17",
            "required_comparator": "uniform_lateral17_on_all_supported_records",
            "conditional_complete19_ablation_is_primary": False,
            "all_record_prediction_first_denominator": True,
            "report_by_support_profile": True,
            "report_technical_failure_rate": True,
        },
    }


def detector_channel_support_policy_receipt() -> dict[str, Any]:
    policy = detector_channel_support_policy()
    return {**policy, "policy_sha256": _sha256(policy)}


def _canonical_roster(values: Iterable[str], *, context: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be a sequence of electrode identifiers")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            raise TypeError(f"{context} contains an invalid electrode identifier")
        channel = raw.strip().upper()
        if channel in seen:
            raise ValueError(f"{context} contains duplicate electrode {channel}")
        seen.add(channel)
        normalized.append(channel)
    return tuple(channel for channel in STANDARD_19 if channel in seen)


def route_detector_channel_support(
    *,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
) -> dict[str, Any]:
    """Choose a target-blind policy route from one externally replayed authority.

    The public API deliberately has no bare roster argument.  Observed and
    usable support are read from independently typed sub-authorities that bind
    the canonical physical signal, electrical reference system and EEG-only
    QC decision.
    """

    authority = require_validated_detector_signal_lineage_authority(
        signal_lineage_authority
    )
    observed = _canonical_roster(
        authority["observed_roster_authority"]["observed_standard_channel_ids"],
        context="typed observed roster",
    )
    usable = _canonical_roster(
        authority["EEG_only_channel_QC_authority"][
            "usable_standard_channel_ids"
        ],
        context="typed EEG-only QC usable roster",
    )
    if not set(usable).issubset(observed):
        raise ValueError("usable channels must be a subset of observed channels")

    usable_set = set(usable)
    if set(STANDARD_19).issubset(usable_set):
        profile_id = "complete19"
        support_policy_status = "policy_route_available"
    elif set(LATERAL_17).issubset(usable_set):
        profile_id = "lateral17"
        support_policy_status = "policy_route_available"
    else:
        profile_id = "unadmitted_partial"
        support_policy_status = "terminal_support_policy_failure"

    if profile_id in _PROFILES:
        materialization = deepcopy(_PROFILES[profile_id]["provider_materialization"])
        all_provider_stacks_executable = all(
            bool(row["full_stack_executable"])
            for row in materialization.values()
        )
        status = (
            "policy_route_available_provider_executable"
            if all_provider_stacks_executable
            and authority["provider_transform_authorized"] is True
            else "policy_route_available_provider_unmaterialized"
        )
    else:
        materialization = None
        all_provider_stacks_executable = False
        status = "terminal_support_policy_failure"

    route: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "policy_sha256": detector_channel_support_policy_receipt()["policy_sha256"],
        "status": status,
        "support_policy_status": support_policy_status,
        "profile_id": profile_id,
        "signal_lineage_authority_sha256": authority["receipt_sha256"],
        "canonical_physical_signal_binding": deepcopy(
            authority["canonical_physical_signal"]
        ),
        "observed_roster_authority_sha256": authority[
            "observed_roster_authority"
        ]["receipt_sha256"],
        "electrical_reference_system_authority_sha256": authority[
            "electrical_reference_system_authority"
        ]["receipt_sha256"],
        "EEG_only_channel_QC_authority_sha256": authority[
            "EEG_only_channel_QC_authority"
        ]["receipt_sha256"],
        "observed_standard_channel_ids": list(observed),
        "usable_standard_channel_ids": list(usable),
        "missing_standard_channel_ids": [
            channel for channel in STANDARD_19 if channel not in set(observed)
        ],
        "QC_unusable_observed_channel_ids": [
            channel for channel in observed if channel not in usable_set
        ],
        "routing_used_EEG_samples": authority["scope_receipt"][
            "EEG_samples_used"
        ],
        "routing_used_EEG_derived_QC": authority["authority_tier"]
        == "provider_transform_payload_replayed",
        "routing_bound_EEG_electrical_reference_authority": True,
        "EEG_electrical_reference_system_status": authority[
            "electrical_reference_system_authority"
        ]["system_status"],
        "routing_used_materialized_EEG_electrical_reference_provenance": authority[
            "scope_receipt"
        ].get("EEG_electrical_reference_provenance_used", False),
        "EEG_electrical_reference_provider_transform_authorized": authority[
            "provider_transform_authorized"
        ],
        "seizure_target_or_reference_label_used": False,
        "routing_used_annotations_or_targets": False,
        "primary_channel_imputation": False,
        "finding_or_SOZ_evidence_authority": False,
        "policy_route_is_provider_executable": bool(
            all_provider_stacks_executable
            and authority["provider_transform_authorized"] is True
        ),
    }
    if profile_id in _PROFILES:
        route["provider_variants"] = {
            "seizuretransformer": _PROFILES[profile_id][
                "seizuretransformer_variant"
            ],
            "eventnet": _PROFILES[profile_id]["eventnet_variant"],
        }
        route["provider_typed_units"] = {
            "seizuretransformer": deepcopy(
                _PROFILES[profile_id]["seizuretransformer_typed_units"]
            ),
            "eventnet": deepcopy(_PROFILES[profile_id]["eventnet_typed_units"]),
        }
        route["provider_materialization"] = materialization
        route["failure_reason"] = None
    else:
        route["provider_variants"] = None
        route["provider_typed_units"] = None
        route["provider_materialization"] = None
        route["failure_reason"] = (
            "insufficient_admitted_typed_observed_EEG_QC_usable_support"
        )

    route["route_sha256"] = _sha256(route)
    return route


def validate_detector_channel_support_route(
    value: Mapping[str, Any],
    *,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
) -> dict[str, Any]:
    """Replay a route against its external typed signal authority."""

    if not isinstance(value, Mapping):
        raise TypeError("detector channel support route must be an object")
    row = deepcopy(dict(value))
    forbidden_present = _FORBIDDEN_ROUTING_KEYS.intersection(row)
    if forbidden_present:
        raise ValueError(
            "detector support route contains forbidden target/text fields: "
            + ",".join(sorted(forbidden_present))
        )
    supplied_hash = row.pop("route_sha256", None)
    if not isinstance(supplied_hash, str) or supplied_hash != _sha256(row):
        raise ValueError("detector support route hash mismatch")
    expected = route_detector_channel_support(
        signal_lineage_authority=signal_lineage_authority,
    )
    if value != expected:
        raise ValueError("detector support route disagrees with frozen policy")
    return deepcopy(expected)


def audit_canonical_projection_support(
    *, audit_path: str | Path, projection_path: str | Path
) -> dict[str, Any]:
    """Audit every canonical identity as policy-route, not provider execution."""

    audit_bytes = Path(audit_path).read_bytes()
    projection_bytes = Path(projection_path).read_bytes()
    anchor = load_canonical_policy_audit_trust_anchor(
        audit_bytes=audit_bytes,
        projection_bytes=projection_bytes,
    )
    projection = json.loads(projection_bytes)
    records = projection["records"]

    counters: Counter[tuple[str, str]] = Counter()
    identity_count = 0
    policy_route_available_count = 0
    provider_unmaterialized_count = 0
    full_stack_executable_count = 0
    terminal_support_failure_count = 0
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or record.get("analysis_unit_weight") != 1:
            continue
        identity = record.get("analysis_identity_id")
        split = record.get("model_split")
        if not isinstance(identity, str) or not isinstance(split, str):
            raise ValueError("canonical projection row is malformed")
        if identity in seen:
            raise ValueError("weight-one canonical identity is duplicated")
        seen.add(identity)
        authority = authorize_detector_policy_lineage_from_canonical_audit(
            anchor, analysis_identity_id=identity
        )
        route = route_detector_channel_support(
            signal_lineage_authority=authority
        )
        counters[(split, route["profile_id"])] += 1
        identity_count += 1
        if route["support_policy_status"] == "policy_route_available":
            policy_route_available_count += 1
            if route["policy_route_is_provider_executable"]:
                full_stack_executable_count += 1
            else:
                provider_unmaterialized_count += 1
        else:
            terminal_support_failure_count += 1

    by_split_profile: dict[str, dict[str, int]] = {}
    for (split, profile), count in sorted(counters.items()):
        by_split_profile.setdefault(split, {})[profile] = count

    receipt = {
        "schema_version": "clinical_eeg_detector_channel_support_audit_v2",
        "method_id": METHOD_ID,
        "policy_sha256": detector_channel_support_policy_receipt()["policy_sha256"],
        "source_audit_file_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "source_projection_file_sha256": hashlib.sha256(projection_bytes).hexdigest(),
        "source_audit_receipt_sha256": anchor.audit_receipt_sha256,
        "source_projection_receipt_sha256": anchor.projection_receipt_sha256,
        "canonical_weight_one_identity_count": identity_count,
        "policy_route_available_identity_count": policy_route_available_count,
        "provider_unmaterialized_identity_count": provider_unmaterialized_count,
        "full_stack_executable_identity_count": full_stack_executable_count,
        "terminal_support_policy_failure_identity_count": (
            terminal_support_failure_count
        ),
        "by_split_profile": by_split_profile,
        "all_weight_one_identities_accounted_once": len(seen) == identity_count,
        "canonical_artifact_membership_and_outcomes_semantically_replayed": True,
        "policy_route_only_not_provider_execution_audit": True,
        "EN17_full_stack_executable_claimed": False,
        "seizure_target_or_reference_label_read": False,
        "EEG_electrical_reference_provenance_materialized_by_source_audit": False,
        "annotation_fields_read": False,
        "clinical_text_or_identity_used_for_routing": False,
        "primary_channel_imputation_used": False,
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    return receipt


def validate_detector_channel_support_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("detector channel support audit must be an object")
    row = deepcopy(dict(value))
    supplied_hash = row.pop("receipt_sha256", None)
    if not isinstance(supplied_hash, str) or supplied_hash != _sha256(row):
        raise ValueError("detector channel support audit hash mismatch")
    if row.get("schema_version") != "clinical_eeg_detector_channel_support_audit_v2":
        raise ValueError("unsupported detector channel support audit schema")
    if row.get("method_id") != METHOD_ID:
        raise ValueError("detector channel support audit method drifted")
    if row.get("policy_sha256") != detector_channel_support_policy_receipt()[
        "policy_sha256"
    ]:
        raise ValueError("detector channel support policy drifted")
    total = row.get("canonical_weight_one_identity_count")
    policy_route = row.get("policy_route_available_identity_count")
    unmaterialized = row.get("provider_unmaterialized_identity_count")
    executable = row.get("full_stack_executable_identity_count")
    failure = row.get("terminal_support_policy_failure_identity_count")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in (total, policy_route, unmaterialized, executable, failure)):
        raise TypeError("detector support audit counts must be integers")
    if total != policy_route + failure or policy_route != unmaterialized + executable:
        raise ValueError("detector support audit denominator does not close")
    if row.get("all_weight_one_identities_accounted_once") is not True:
        raise ValueError("detector support audit does not account identities once")
    if row.get("canonical_artifact_membership_and_outcomes_semantically_replayed") is not True:
        raise ValueError("detector support audit did not semantically replay sources")
    if row.get("policy_route_only_not_provider_execution_audit") is not True:
        raise ValueError("detector support audit overclaims provider execution")
    if row.get("EN17_full_stack_executable_claimed") is not False:
        raise ValueError("unimplemented EventNet17 was called executable")
    if row.get("seizure_target_or_reference_label_read") is not False:
        raise ValueError("detector support audit read seizure target/reference labels")
    if row.get("annotation_fields_read") is not False:
        raise ValueError("detector support audit read annotation fields")
    if row.get("EEG_electrical_reference_provenance_materialized_by_source_audit") is not False:
        raise ValueError("signal-only support audit invented EEG reference provenance")
    if row.get("clinical_text_or_identity_used_for_routing") is not False:
        raise ValueError("detector support audit used forbidden text/identity")
    if row.get("primary_channel_imputation_used") is not False:
        raise ValueError("detector support audit used primary channel imputation")
    return deepcopy(dict(value))


__all__ = [
    "EVENTNET17_UNITS",
    "EVENTNET19_UNITS",
    "LATERAL_17",
    "METHOD_ID",
    "MIDLINE_OPTIONAL",
    "SCHEMA_VERSION",
    "STANDARD_19",
    "ST16_UNITS",
    "ST18_UNITS",
    "audit_canonical_projection_support",
    "detector_channel_support_policy",
    "detector_channel_support_policy_receipt",
    "route_detector_channel_support",
    "validate_detector_channel_support_audit",
    "validate_detector_channel_support_route",
]
