"""Patient-disjoint calibration contract for EEG-native onset-trigger gates.

This module is deliberately a *control-plane* component.  It calibrates an
effect threshold and a minimum physical-time persistence for each frozen
``(operator, signed effect, scale, reference)`` stratum.  It does not read an
EDF, an EDF annotation, a spreadsheet, a report, an SOZ label, or an LLM.

The only seizure reference accepted by the admitted materializer is a
process-local, actual-byte-replayed
``detector_fold_reference_authority_v1`` opaque authority; a serialized
receipt mapping is evidence, not authority.  For every outer fold, the
selection-fit authority supplies background distributions and the disjoint
inner-validation authority selects a pre-registered grid point.  The outer
held-out fold is never opened here.  Every EEG prediction JSON is replayed
from its actual bytes before it contributes to a denominator.

The checked-in registry is intentionally unadmitted: it contains a protocol
and no fitted value or performance claim.  Consequently it cannot turn a
measurement atom into positive onset support or a rank contribution.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final

from .detector_fold_reference_authority_v1 import (
    require_validated_detector_fold_reference_phase_authority_v1,
    validate_detector_fold_reference_phase_v1,
)
from .detector_signal_lineage_authority_v1 import (
    ValidatedDetectorSignalLineageAuthority,
    require_validated_detector_signal_lineage_authority,
)


FINDINGS_ONSET_THRESHOLD_PREREGISTRY_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_onset_threshold_preregistry_v1"
)
FINDINGS_ONSET_THRESHOLD_ADMITTED_REGISTRY_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_onset_threshold_admitted_registry_v1"
)
FINDINGS_ONSET_THRESHOLD_PREDICTION_ARTIFACT_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_onset_threshold_prediction_artifact_v1"
)
FINDINGS_CANONICAL_EXACT_CLOCK_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_canonical_exact_clock_v1"
)
FINDINGS_ONSET_THRESHOLD_METHOD_ID: Final[str] = (
    "PATIENT-DISJOINT-CROSSFIT-NATIVE-ONSET-THRESHOLD-PERSISTENCE-V1"
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_PHASES: Final[tuple[str, str]] = ("selection_fit", "inner_validation")
_SIGNS = frozenset({"increase", "decrease", "absolute"})
_PRODUCER_STATES = frozenset({"present", "zero_atoms", "failed", "not_applicable"})
_QC_STATES = frozenset({"evaluable", "censored"})
_ADMITTED_REGISTRY_SEAL = object()


@dataclass(frozen=True)
class ValidatedFindingsOnsetThresholdRegistry:
    """Opaque authority issued only after real artifact/reference replay."""

    _receipt_json: str = field(repr=False)
    _validation_seal: object = field(repr=False, compare=False)

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)

    @property
    def receipt_sha256(self) -> str:
        return str(self.receipt["registry_receipt_sha256"])

_SOURCE_FIREWALL: Final[dict[str, bool]] = {
    "EEG_samples_used": True,
    "allowlisted_acquisition_metadata_used": True,
    "EEG_derived_QC_used": True,
    "typed_TUSZ_seizure_reference_control_plane_used": True,
    "EDF_annotations_read_directly": False,
    "spreadsheet_or_Excel_used": False,
    "doctor_labels_or_reports_used": False,
    "SOZ_or_channel_GT_used": False,
    "clinical_history_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_or_activation_used": False,
    "ECG_EMG_EOG_used": False,
    "LLM_used": False,
}

_FORWARD_FIREWALL: Final[dict[str, bool]] = {
    "EEG_samples_used": True,
    "allowlisted_acquisition_metadata_used": True,
    "EEG_derived_QC_used": True,
    "TUSZ_seizure_reference_used": False,
    "EDF_annotations_used": False,
    "spreadsheet_or_Excel_used": False,
    "doctor_labels_or_reports_used": False,
    "SOZ_or_channel_GT_used": False,
    "clinical_history_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_or_activation_used": False,
    "ECG_EMG_EOG_used": False,
    "LLM_used": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _sha(body)


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _fraction(value: object, context: str, *, positive: bool = False) -> Fraction:
    if (
        type(value) is not list
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
    ):
        raise ValueError(f"{context} must be a reduced [numerator, denominator]")
    result = Fraction(value[0], value[1])
    if [result.numerator, result.denominator] != value:
        raise ValueError(f"{context} must be reduced")
    if (positive and result <= 0) or (not positive and result < 0):
        raise ValueError(f"{context} has an invalid sign")
    return result


def _fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _exact_fields(value: object, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    row = deepcopy(dict(value))
    if set(row) != fields:
        raise ValueError(
            f"{context} fields drifted; missing={sorted(fields-set(row))}, "
            f"extra={sorted(set(row)-fields)}"
        )
    return row


def _validate_bool_map(value: object, expected: Mapping[str, bool], context: str) -> dict[str, bool]:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise PermissionError(f"{context} firewall drifted")
    return deepcopy(dict(expected))


def _validate_fraction_grid(
    value: object, context: str, *, lower: Fraction | None = None, upper: Fraction | None = None
) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be an array")
    fractions = [_fraction(item, context, positive=True) for item in value]
    if not fractions or fractions != sorted(set(fractions)):
        raise ValueError(f"{context} must be non-empty, unique, and sorted")
    if lower is not None and any(item < lower for item in fractions):
        raise ValueError(f"{context} is below its registered lower bound")
    if upper is not None and any(item > upper for item in fractions):
        raise ValueError(f"{context} exceeds its registered upper bound")
    return [_fraction_json(item) for item in fractions]


def build_findings_onset_threshold_preregistry_v1(
    *,
    registry_id: str = "CLINICAL-EEG-FINDINGS-ONSET-THRESHOLD-PREREGISTRY-V1-20260824",
    outer_fold_count: int = 5,
    background_quantile_grid: Sequence[Sequence[int]] = (
        (19, 20),
        (39, 40),
        (99, 100),
        (199, 200),
        (999, 1000),
    ),
    minimum_persistence_seconds_grid: Sequence[Sequence[int]] = (
        (1, 4),
        (1, 2),
        (1, 1),
        (2, 1),
        (4, 1),
    ),
    maximum_query_gap_seconds: Sequence[int] = (1, 2),
    onset_uncertainty_pre_seconds: Sequence[int] = (2, 1),
    onset_uncertainty_post_seconds: Sequence[int] = (2, 1),
    matched_background_guard_seconds: Sequence[int] = (30, 1),
    matched_background_stride_seconds: Sequence[int] = (1, 1),
    false_trigger_per_hour_limit: Sequence[int] = (1, 1),
    permutation_count: int = 999,
    permutation_alpha: Sequence[int] = (1, 20),
    minimum_inner_validation_patients: int = 5,
    minimum_inner_validation_onset_opportunities: int = 5,
    minimum_stable_query_count: int = 2,
) -> dict[str, Any]:
    """Build a target-free, content-addressed protocol registry.

    Numeric grid points are pre-registered method hyperparameters, never
    claimed fitted thresholds or observed performance.
    """

    _identifier(registry_id, "registry_id")
    _integer(outer_fold_count, "outer_fold_count", minimum=2)
    _integer(permutation_count, "permutation_count", minimum=19)
    _integer(minimum_inner_validation_patients, "minimum patients", minimum=1)
    _integer(
        minimum_inner_validation_onset_opportunities,
        "minimum onset opportunities",
        minimum=1,
    )
    _integer(minimum_stable_query_count, "minimum stable query count", minimum=2)
    quantiles = _validate_fraction_grid(
        [list(item) for item in background_quantile_grid],
        "background quantile grid",
        lower=Fraction(1, 2),
        upper=Fraction(1, 1),
    )
    persistence = _validate_fraction_grid(
        [list(item) for item in minimum_persistence_seconds_grid],
        "minimum persistence grid",
    )
    max_gap = _fraction(list(maximum_query_gap_seconds), "maximum query gap", positive=True)
    onset_pre = _fraction(list(onset_uncertainty_pre_seconds), "onset uncertainty pre")
    onset_post = _fraction(list(onset_uncertainty_post_seconds), "onset uncertainty post")
    guard = _fraction(list(matched_background_guard_seconds), "background guard")
    stride = _fraction(list(matched_background_stride_seconds), "background stride", positive=True)
    ftr_limit = _fraction(list(false_trigger_per_hour_limit), "false-trigger/hour limit", positive=True)
    alpha = _fraction(list(permutation_alpha), "permutation alpha", positive=True)
    if alpha > Fraction(1, 20):
        raise ValueError("permutation alpha must be <= 0.05")
    body: dict[str, Any] = {
        "schema_version": FINDINGS_ONSET_THRESHOLD_PREREGISTRY_SCHEMA_VERSION,
        "registry_id": registry_id,
        "method_id": FINDINGS_ONSET_THRESHOLD_METHOD_ID,
        "status": "unadmitted_no_real_crossfit_prediction_artifacts",
        "split_contract": {
            "outer_fold_count": outer_fold_count,
            "patient_disjoint": True,
            "selection_fit_role": "derive_signed_native_effect_quantile_thresholds",
            "inner_validation_role": "select_threshold_and_persistence_under_constraints",
            "outer_heldout_reference_opened": False,
            "source_dev_or_source_eval_used": False,
            "private_data_used": False,
        },
        "grid": {
            "threshold_parameterization": "selection_fit_same_record_background_higher_quantile",
            "background_quantile_grid": quantiles,
            "minimum_persistence_seconds_grid": persistence,
            "maximum_query_gap_seconds_fraction": _fraction_json(max_gap),
            "minimum_stable_query_count": minimum_stable_query_count,
            "effect_sign_policies": sorted(_SIGNS),
            "stratum_key_fields": [
                "operator_id",
                "operator_version",
                "measurement_name_id",
                "physical_unit",
                "effect_sign_policy",
                "scale_id",
                "reference_family",
            ],
        },
        "opportunity_protocol": {
            "onset_reference": "typed_fold_authority_TERM_seiz_start_control_plane_only",
            "onset_uncertainty_pre_seconds_fraction": _fraction_json(onset_pre),
            "onset_uncertainty_post_seconds_fraction": _fraction_json(onset_post),
            "matched_background_same_record_required": True,
            "matched_background_equal_duration_required": True,
            "matched_background_guard_seconds_fraction": _fraction_json(guard),
            "matched_background_stride_seconds_fraction": _fraction_json(stride),
            "matched_background_without_replacement": True,
            "ties_choose_earlier_window": True,
            "full_non_event_ledger": (
                "all_zero_onset_recording_time_plus_seizure_record_time_outside_"
                "uncertain_onset_and_seizure_guard"
            ),
            "full_non_event_query_coverage_required_for_admission": True,
            "paired_matched_background_is_null_control_not_FTR_denominator": True,
        },
        "selection_protocol": {
            "primary_objective": "inner_validation_patient_macro_onset_sensitivity",
            "false_trigger_per_hour_limit_fraction": _fraction_json(ftr_limit),
            "null_test": "deterministic_within_record_onset_background_pair_swap",
            "permutation_count": permutation_count,
            "permutation_alpha_fraction": _fraction_json(alpha),
            "multiplicity_control": (
                "Bonferroni_within_outer_fold_across_all_operator_grid_candidates"
            ),
            "minimum_evaluable_onset_opportunity_fraction": [9, 10],
            "minimum_inner_validation_patients": minimum_inner_validation_patients,
            "minimum_inner_validation_onset_opportunities": (
                minimum_inner_validation_onset_opportunities
            ),
            "tie_break_order": [
                "higher_patient_macro_onset_sensitivity",
                "lower_false_triggers_per_hour",
                "lower_background_quantile",
                "shorter_minimum_persistence",
                "lexicographic_candidate_id",
            ],
        },
        "trajectory_protocol": {
            "clock": "canonical_exact_clock_fraction_plus_integer_half_open_samples",
            "clock_authority": (
                "opaque_provider_transform_payload_replayed_signal_lineage_authority"
            ),
            "query_state_source": "multi_query_replay_never_atom_self_report",
            "native_track_identity": (
                "stable_receipt_from_record_occurrence_slot_typed_unit_directed_"
                "channels_operator_measurement_scale_reference_and_ordinal"
            ),
            "native_track_id_self_report_trusted": False,
            "change_interval_revision_tolerance_seconds_fraction": [1, 4],
            "pre_stabilization_atom_absence_semantics": (
                "invalidate_and_reset_consecutive_persistence_when_query_inventory_complete"
            ),
            "post_stabilization_gap_semantics": "changed_after_stabilization_latched",
            "post_stabilization_threshold_failure_semantics": (
                "changed_after_stabilization_latched"
            ),
            "post_stabilization_atom_absence_semantics": (
                "changed_after_stabilization_latched_when_query_inventory_complete"
            ),
            "restabilization_within_same_native_track_authorized": False,
        },
        "denominator_policy": {
            "all_authorized_records_required_in_each_artifact": True,
            "all_declared_strata_required_per_record": True,
            "producer_failure_onset_opportunities_count_as_misses": True,
            "zero_atom_onset_opportunities_count_as_misses": True,
            "censored_queries_never_become_masked_zeros": True,
            "unmatched_background_opportunities_reported": True,
            "admission_requires_complete_same_record_matching": True,
            "patient_with_zero_onset_opportunities_excluded_only_from_macro_mean_and_reported": True,
            "zero_onset_full_record_time_enters_non_event_FTR_denominator": True,
            "matched_background_duration_cannot_be_reported_as_full_record_FTR_time": True,
        },
        "artifact_policy": {
            "actual_prediction_file_bytes_replayed": True,
            "relative_path_size_and_sha256_bound": True,
            "opaque_fold_phase_reference_authority_required": True,
            "raw_fold_phase_reference_receipt_accepted": False,
            "phase_authority_must_replay_actual_reference_bytes": True,
            "nonselection_phase_controller_artifact_replay_required": True,
            "canonical_exact_clock_required": True,
            "integer_half_open_sample_indices_required": True,
            "raw_dependency_stop_must_not_exceed_locked_prefix_stop": True,
            "atom_reported_sample_rate_or_stability_trusted": False,
            "effective_bandwidth_must_not_exceed_exact_clock_nyquist": True,
            "opaque_provider_signal_lineage_authority_required": True,
            "policy_only_audit_authority_accepted": False,
            "producer_view_indices_must_map_back_to_canonical_source_samples": True,
        },
        "source_firewall": deepcopy(_SOURCE_FIREWALL),
        "forward_firewall": deepcopy(_FORWARD_FIREWALL),
        "fitted_operator_strata": [],
        "performance_claims": [],
        "authorization": {
            "registry_admitted": False,
            "positive_onset_trigger_authorized": False,
            "positive_rank_contribution_authorized": False,
            "clinical_term_authorized": False,
            "SOZ_EZ_or_surgical_target_claim_authorized": False,
            "report_text_authorized": False,
        },
        "registry_receipt_sha256": "",
    }
    body["registry_receipt_sha256"] = _self_hash(body, "registry_receipt_sha256")
    return body


def validate_findings_onset_threshold_preregistry_v1(value: object) -> dict[str, Any]:
    """Validate the pre-registered, target-free protocol exactly."""

    expected_fields = {
        "schema_version",
        "registry_id",
        "method_id",
        "status",
        "split_contract",
        "grid",
        "opportunity_protocol",
        "selection_protocol",
        "trajectory_protocol",
        "denominator_policy",
        "artifact_policy",
        "source_firewall",
        "forward_firewall",
        "fitted_operator_strata",
        "performance_claims",
        "authorization",
        "registry_receipt_sha256",
    }
    row = _exact_fields(value, expected_fields, "onset threshold preregistry")
    if row["schema_version"] != FINDINGS_ONSET_THRESHOLD_PREREGISTRY_SCHEMA_VERSION:
        raise ValueError("onset threshold preregistry schema drifted")
    if row["method_id"] != FINDINGS_ONSET_THRESHOLD_METHOD_ID:
        raise ValueError("onset threshold method drifted")
    _identifier(row["registry_id"], "registry_id")
    if row["status"] != "unadmitted_no_real_crossfit_prediction_artifacts":
        raise PermissionError("checked protocol must remain unadmitted")
    _validate_bool_map(row["source_firewall"], _SOURCE_FIREWALL, "source")
    _validate_bool_map(row["forward_firewall"], _FORWARD_FIREWALL, "forward")
    if row["fitted_operator_strata"] != [] or row["performance_claims"] != []:
        raise PermissionError("unadmitted registry contains fitted values or claims")
    if row["authorization"] != {
        "registry_admitted": False,
        "positive_onset_trigger_authorized": False,
        "positive_rank_contribution_authorized": False,
        "clinical_term_authorized": False,
        "SOZ_EZ_or_surgical_target_claim_authorized": False,
        "report_text_authorized": False,
    }:
        raise PermissionError("unadmitted authorization drifted")
    split = row["split_contract"]
    if not isinstance(split, Mapping) or split.get("patient_disjoint") is not True:
        raise ValueError("patient-disjoint split contract is absent")
    outer_count = _integer(split.get("outer_fold_count"), "outer fold count", minimum=2)
    if split.get("outer_heldout_reference_opened") is not False:
        raise PermissionError("outer-held-out reference was opened")
    grid = row["grid"]
    if not isinstance(grid, Mapping):
        raise TypeError("grid must be an object")
    _validate_fraction_grid(
        grid.get("background_quantile_grid"),
        "background quantile grid",
        lower=Fraction(1, 2),
        upper=Fraction(1, 1),
    )
    _validate_fraction_grid(
        grid.get("minimum_persistence_seconds_grid"),
        "minimum persistence grid",
    )
    _fraction(grid.get("maximum_query_gap_seconds_fraction"), "maximum query gap", positive=True)
    _integer(grid.get("minimum_stable_query_count"), "minimum stable queries", minimum=2)
    if grid.get("effect_sign_policies") != sorted(_SIGNS):
        raise ValueError("effect sign policy roster drifted")
    opportunity = row["opportunity_protocol"]
    selection = row["selection_protocol"]
    trajectory = row["trajectory_protocol"]
    if (
        not isinstance(opportunity, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(trajectory, Mapping)
    ):
        raise TypeError("protocol sections must be objects")
    for name in (
        "onset_uncertainty_pre_seconds_fraction",
        "onset_uncertainty_post_seconds_fraction",
        "matched_background_guard_seconds_fraction",
    ):
        _fraction(opportunity.get(name), name)
    if opportunity.get("full_non_event_query_coverage_required_for_admission") is not True:
        raise PermissionError("full non-event query coverage gate drifted")
    if opportunity.get("paired_matched_background_is_null_control_not_FTR_denominator") is not True:
        raise PermissionError("matched background was promoted to the FTR denominator")
    _fraction(opportunity.get("matched_background_stride_seconds_fraction"), "background stride", positive=True)
    _fraction(selection.get("false_trigger_per_hour_limit_fraction"), "FTR/hour limit", positive=True)
    alpha = _fraction(selection.get("permutation_alpha_fraction"), "permutation alpha", positive=True)
    if alpha > Fraction(1, 20):
        raise ValueError("permutation alpha exceeds 0.05")
    _integer(selection.get("permutation_count"), "permutation count", minimum=19)
    coverage = _fraction(
        selection.get("minimum_evaluable_onset_opportunity_fraction"),
        "minimum evaluable onset opportunity fraction",
        positive=True,
    )
    if coverage > 1:
        raise ValueError("minimum evaluable onset opportunity fraction exceeds one")
    if selection.get("multiplicity_control") != (
        "Bonferroni_within_outer_fold_across_all_operator_grid_candidates"
    ):
        raise PermissionError("selection multiplicity control drifted")
    _integer(selection.get("minimum_inner_validation_patients"), "minimum patients", minimum=1)
    _integer(
        selection.get("minimum_inner_validation_onset_opportunities"),
        "minimum onset opportunities",
        minimum=1,
    )
    _fraction(
        trajectory.get("change_interval_revision_tolerance_seconds_fraction"),
        "change interval revision tolerance",
    )
    if trajectory.get("query_state_source") != (
        "multi_query_replay_never_atom_self_report"
    ):
        raise PermissionError("trajectory state source drifted")
    if trajectory.get("clock_authority") != (
        "opaque_provider_transform_payload_replayed_signal_lineage_authority"
    ):
        raise PermissionError("trajectory clock authority drifted")
    if trajectory.get("native_track_identity") != (
        "stable_receipt_from_record_occurrence_slot_typed_unit_directed_"
        "channels_operator_measurement_scale_reference_and_ordinal"
    ):
        raise PermissionError("stable typed-unit/operator track identity drifted")
    if trajectory.get("native_track_id_self_report_trusted") is not False:
        raise PermissionError("self-reported native track identity was trusted")
    if trajectory.get("pre_stabilization_atom_absence_semantics") != (
        "invalidate_and_reset_consecutive_persistence_when_query_inventory_complete"
    ):
        raise PermissionError("pre-stabilization continuity reset drifted")
    if trajectory.get("restabilization_within_same_native_track_authorized") is not False:
        raise PermissionError("same-track restabilization was opened")
    artifact_policy = row["artifact_policy"]
    if not isinstance(artifact_policy, Mapping):
        raise TypeError("artifact policy must be an object")
    if artifact_policy.get("opaque_fold_phase_reference_authority_required") is not True:
        raise PermissionError("opaque fold-phase reference authority gate drifted")
    if artifact_policy.get("raw_fold_phase_reference_receipt_accepted") is not False:
        raise PermissionError("raw fold-phase reference receipts were opened")
    if artifact_policy.get("phase_authority_must_replay_actual_reference_bytes") is not True:
        raise PermissionError("actual reference-byte replay gate drifted")
    if artifact_policy.get("nonselection_phase_controller_artifact_replay_required") is not True:
        raise PermissionError("controller artifact replay gate drifted")
    receipt = _sha256(row["registry_receipt_sha256"], "registry receipt")
    if receipt != _self_hash(row, "registry_receipt_sha256"):
        raise ValueError("onset threshold preregistry receipt does not replay")
    # Force the count to be consumed; otherwise a bool could pass later code.
    if outer_count != split["outer_fold_count"]:
        raise AssertionError("unreachable outer-fold coercion")
    return row


def _provider_signal_lineage_receipt(
    authority: ValidatedDetectorSignalLineageAuthority,
) -> dict[str, Any]:
    receipt = require_validated_detector_signal_lineage_authority(authority)
    if receipt.get("authority_tier") != "provider_transform_payload_replayed":
        raise PermissionError(
            "Findings exact clock requires provider_transform_payload_replayed authority"
        )
    if receipt.get("provider_transform_authorized") is not True:
        raise PermissionError("signal-lineage authority is not provider-transform capable")
    return receipt


def build_findings_canonical_exact_clock_from_signal_authority_v1(
    *, authority: ValidatedDetectorSignalLineageAuthority, recording_id: str
) -> dict[str, Any]:
    """Project an opaque payload-replayed signal authority into an exact clock."""

    signal_authority = _provider_signal_lineage_receipt(authority)
    physical = signal_authority["canonical_physical_signal"]
    clock = signal_authority["common_sampling_clock_authority"]
    electrical = signal_authority["electrical_reference_system_authority"]
    qc = signal_authority["EEG_only_channel_QC_authority"]
    row: dict[str, Any] = {
        "schema_version": FINDINGS_CANONICAL_EXACT_CLOCK_SCHEMA_VERSION,
        "recording_id": _identifier(recording_id, "recording_id"),
        "sample_rate_hz_fraction": deepcopy(clock["sampling_rate_fraction_hz"]),
        "total_sample_count": clock["sample_count"],
        "detector_signal_lineage_authority_receipt_sha256": signal_authority[
            "receipt_sha256"
        ],
        "source_signal_sha256": physical["source_signal_sha256"],
        "canonical_source_tensor_sha256": physical["source_tensor_sha256"],
        "source_header_receipt_sha256": physical["source_header_receipt_sha256"],
        "canonical_signal_receipt_sha256": physical[
            "canonical_signal_receipt_sha256"
        ],
        "common_sampling_clock_authority_receipt_sha256": clock["receipt_sha256"],
        "electrical_reference_system_receipt_sha256": electrical["receipt_sha256"],
        "EEG_only_channel_QC_receipt_sha256": qc["receipt_sha256"],
        "clock_origin_sample_index": 0,
        "receipt_sha256": "",
    }
    row["receipt_sha256"] = _self_hash(row, "receipt_sha256")
    return row


def _validate_exact_clock(
    value: object,
    *,
    duration: Fraction,
    recording_id: str,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "recording_id",
        "sample_rate_hz_fraction",
        "total_sample_count",
        "detector_signal_lineage_authority_receipt_sha256",
        "source_signal_sha256",
        "canonical_source_tensor_sha256",
        "source_header_receipt_sha256",
        "canonical_signal_receipt_sha256",
        "common_sampling_clock_authority_receipt_sha256",
        "electrical_reference_system_receipt_sha256",
        "EEG_only_channel_QC_receipt_sha256",
        "clock_origin_sample_index",
        "receipt_sha256",
    }
    row = _exact_fields(value, fields, "canonical exact clock")
    if row["schema_version"] != FINDINGS_CANONICAL_EXACT_CLOCK_SCHEMA_VERSION:
        raise ValueError("canonical exact clock schema drifted")
    if row["recording_id"] != recording_id:
        raise ValueError("canonical exact clock recording crossed")
    fs = _fraction(row["sample_rate_hz_fraction"], "sample rate", positive=True)
    total = _integer(row["total_sample_count"], "total sample count", minimum=1)
    if row["clock_origin_sample_index"] != 0:
        raise ValueError("canonical exact clock must have zero sample origin")
    if Fraction(total, 1) != duration * fs:
        raise ValueError("exact clock sample count and authorized duration differ")
    for field in (
        "detector_signal_lineage_authority_receipt_sha256",
        "source_signal_sha256",
        "canonical_source_tensor_sha256",
        "source_header_receipt_sha256",
        "canonical_signal_receipt_sha256",
        "common_sampling_clock_authority_receipt_sha256",
        "electrical_reference_system_receipt_sha256",
        "EEG_only_channel_QC_receipt_sha256",
    ):
        _sha256(row[field], f"canonical exact clock {field}")
    receipt = _sha256(row["receipt_sha256"], "canonical exact clock receipt")
    if receipt != _self_hash(row, "receipt_sha256"):
        raise ValueError("canonical exact clock receipt does not replay")
    expected = build_findings_canonical_exact_clock_from_signal_authority_v1(
        authority=signal_lineage_authority, recording_id=recording_id
    )
    if row != expected:
        raise PermissionError(
            "canonical exact clock does not rebuild from opaque provider authority"
        )
    return row


def _validate_stratum(value: object) -> dict[str, Any]:
    fields = {
        "stratum_id",
        "operator_id",
        "operator_version",
        "measurement_name_id",
        "physical_unit",
        "effect_sign_policy",
        "scale_id",
        "reference_family",
        "effective_bandwidth_hz_fraction",
        "required_bandwidth_hz_fraction",
        "operator_parameter_receipt_sha256",
    }
    row = _exact_fields(value, fields, "operator stratum")
    for field in (
        "stratum_id",
        "operator_id",
        "operator_version",
        "measurement_name_id",
        "physical_unit",
        "scale_id",
        "reference_family",
    ):
        row[field] = _identifier(row[field], f"operator stratum {field}")
    if row["effect_sign_policy"] not in _SIGNS:
        raise ValueError("operator stratum effect sign is unsupported")
    for field in ("effective_bandwidth_hz_fraction", "required_bandwidth_hz_fraction"):
        band = row[field]
        if type(band) is not list or len(band) != 2:
            raise ValueError(f"{field} must be a two-fraction band")
        low = _fraction(band[0], f"{field} low")
        high = _fraction(band[1], f"{field} high", positive=True)
        if high <= low:
            raise ValueError(f"{field} is empty")
    effective = [_fraction(item, "effective bandwidth") for item in row["effective_bandwidth_hz_fraction"]]
    required = [_fraction(item, "required bandwidth") for item in row["required_bandwidth_hz_fraction"]]
    if required[0] < effective[0] or required[1] > effective[1]:
        raise ValueError("required bandwidth leaves effective bandwidth")
    _sha256(row["operator_parameter_receipt_sha256"], "operator parameter receipt")
    material = {key: row[key] for key in fields if key != "stratum_id"}
    expected_id = "STRATUM-" + _sha(material)[:24]
    if row["stratum_id"] != expected_id:
        raise ValueError("operator stratum ID does not replay")
    return row


def _score(value: float, sign: str) -> float:
    if sign == "increase":
        return value
    if sign == "decrease":
        return -value
    return abs(value)


def _validate_query(
    value: object,
    *,
    clock: Mapping[str, Any],
    stratum: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "query_id",
        "query_index",
        "anchor_sample_index",
        "measurement_start_sample_index",
        "measurement_stop_sample_index",
        "change_start_sample_index",
        "change_stop_sample_index",
        "locked_prefix_start_sample_index",
        "locked_prefix_stop_sample_index",
        "raw_dependency_half_open_sample_spans",
        "raw_dependency_sha256s",
        "canonical_exact_clock_receipt_sha256",
        "native_measurement_content_sha256",
        "producer_view_sample_rate_hz_fraction",
        "producer_view_measurement_half_open_sample_span",
        "producer_view_to_canonical_source_projection_receipt_sha256",
        "effect_value",
        "qc_state",
        "qc_receipt_sha256",
    }
    row = _exact_fields(value, fields, "prediction query")
    row["query_id"] = _identifier(row["query_id"], "query_id")
    row["query_index"] = _integer(row["query_index"], "query_index")
    total = clock["total_sample_count"]
    for field in (
        "anchor_sample_index",
        "measurement_start_sample_index",
        "measurement_stop_sample_index",
        "change_start_sample_index",
        "change_stop_sample_index",
        "locked_prefix_start_sample_index",
        "locked_prefix_stop_sample_index",
    ):
        row[field] = _integer(row[field], field)
    if not (
        0 <= row["locked_prefix_start_sample_index"]
        <= row["measurement_start_sample_index"]
        < row["measurement_stop_sample_index"]
        <= row["locked_prefix_stop_sample_index"]
        <= total
    ):
        raise PermissionError("measurement interval leaves the integer locked prefix")
    if not (
        row["measurement_start_sample_index"]
        <= row["change_start_sample_index"]
        < row["change_stop_sample_index"]
        <= row["measurement_stop_sample_index"]
    ):
        raise PermissionError("change interval leaves the integer measurement interval")
    if not (
        row["measurement_start_sample_index"]
        <= row["anchor_sample_index"]
        < row["measurement_stop_sample_index"]
    ):
        raise ValueError("query anchor leaves its measurement interval")
    spans = row["raw_dependency_half_open_sample_spans"]
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)) or not spans:
        raise ValueError("raw dependency sample spans must be non-empty")
    normalized: list[list[int]] = []
    for span in spans:
        if type(span) is not list or len(span) != 2:
            raise ValueError("raw dependency span must be a half-open integer pair")
        start = _integer(span[0], "raw dependency start")
        stop = _integer(span[1], "raw dependency stop", minimum=1)
        if not row["locked_prefix_start_sample_index"] <= start < stop <= row[
            "locked_prefix_stop_sample_index"
        ]:
            raise PermissionError("raw dependency leaves integer locked prefix")
        normalized.append([start, stop])
    if normalized != sorted(normalized) or any(
        normalized[index][0] < normalized[index - 1][1]
        for index in range(1, len(normalized))
    ):
        raise ValueError("raw dependency spans must be sorted and non-overlapping")
    row["raw_dependency_half_open_sample_spans"] = normalized
    hashes = row["raw_dependency_sha256s"]
    if not isinstance(hashes, Sequence) or isinstance(hashes, (str, bytes)):
        raise TypeError("raw dependency hashes must be an array")
    normalized_hashes = [_sha256(item, "raw dependency hash") for item in hashes]
    if not normalized_hashes or normalized_hashes != sorted(set(normalized_hashes)):
        raise ValueError("raw dependency hashes must be non-empty, unique, sorted")
    row["raw_dependency_sha256s"] = normalized_hashes
    if row["canonical_exact_clock_receipt_sha256"] != clock["receipt_sha256"]:
        raise PermissionError("query self-reported a different sample clock")
    _sha256(row["native_measurement_content_sha256"], "native measurement content")
    producer_view_fs = _fraction(
        row["producer_view_sample_rate_hz_fraction"],
        "producer view sample rate",
        positive=True,
    )
    producer_view_span = row["producer_view_measurement_half_open_sample_span"]
    if type(producer_view_span) is not list or len(producer_view_span) != 2:
        raise ValueError("producer view measurement span must be a sample-index pair")
    producer_view_span = [
        _integer(producer_view_span[0], "producer view start"),
        _integer(producer_view_span[1], "producer view stop", minimum=1),
    ]
    if producer_view_span[1] <= producer_view_span[0]:
        raise ValueError("producer view measurement span is empty")
    projection_material = {
        "domain": "threshold_prediction_view_to_canonical_source_projection_v1",
        "native_measurement_content_sha256": row[
            "native_measurement_content_sha256"
        ],
        "canonical_exact_clock_receipt_sha256": clock["receipt_sha256"],
        "producer_view_sample_rate_hz_fraction": _fraction_json(producer_view_fs),
        "producer_view_measurement_half_open_sample_span": producer_view_span,
        "canonical_source_measurement_half_open_sample_span": [
            row["measurement_start_sample_index"],
            row["measurement_stop_sample_index"],
        ],
        "canonical_source_raw_dependency_half_open_sample_spans": normalized,
    }
    projection_receipt = _sha256(
        row["producer_view_to_canonical_source_projection_receipt_sha256"],
        "prediction producer-view projection receipt",
    )
    if projection_receipt != _sha(projection_material):
        raise PermissionError(
            "prediction producer-view to canonical-source projection does not replay"
        )
    row["producer_view_sample_rate_hz_fraction"] = _fraction_json(producer_view_fs)
    row["producer_view_measurement_half_open_sample_span"] = producer_view_span
    row["effect_value"] = _finite(row["effect_value"], "effect value")
    if row["qc_state"] not in _QC_STATES:
        raise ValueError("query QC state is unsupported")
    _sha256(row["qc_receipt_sha256"], "query QC receipt")
    nyquist = _fraction(clock["sample_rate_hz_fraction"], "sample rate", positive=True) / 2
    effective_high = _fraction(stratum["effective_bandwidth_hz_fraction"][1], "effective high", positive=True)
    if effective_high > nyquist:
        raise PermissionError("effective bandwidth exceeds canonical exact-clock Nyquist")
    return row


def _validate_prediction_artifact(
    value: object,
    *,
    preregistry: Mapping[str, Any],
    authority: Mapping[str, Any],
    patient_by_identity: Mapping[str, str],
    signal_lineage_authority_by_identity: Mapping[
        str, ValidatedDetectorSignalLineageAuthority
    ],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "outer_fold_id",
        "phase",
        "preregistry_receipt_sha256",
        "fold_reference_authority_receipt_sha256",
        "producer_protocol_receipt_sha256",
        "operator_strata",
        "records",
        "source_firewall",
        "content_sha256",
    }
    row = _exact_fields(value, fields, "threshold prediction artifact")
    if row["schema_version"] != FINDINGS_ONSET_THRESHOLD_PREDICTION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("threshold prediction artifact schema drifted")
    if row["outer_fold_id"] != authority["outer_fold_id"] or row["phase"] != authority["phase"]:
        raise ValueError("prediction artifact crossed fold or phase")
    if row["preregistry_receipt_sha256"] != preregistry["registry_receipt_sha256"]:
        raise PermissionError("prediction artifact was not frozen to this preregistry")
    if row["fold_reference_authority_receipt_sha256"] != authority["receipt_sha256"]:
        raise PermissionError("prediction artifact lacks its typed fold authority binding")
    _sha256(row["producer_protocol_receipt_sha256"], "producer protocol receipt")
    _validate_bool_map(row["source_firewall"], _SOURCE_FIREWALL, "artifact source")
    strata = [_validate_stratum(item) for item in row["operator_strata"]]
    strata.sort(key=lambda item: item["stratum_id"])
    if not strata or [item["stratum_id"] for item in strata] != sorted(
        set(item["stratum_id"] for item in strata)
    ):
        raise ValueError("operator strata must be non-empty and unique")
    row["operator_strata"] = strata
    stratum_by_id = {item["stratum_id"]: item for item in strata}
    expected_records = {
        str(item["analysis_identity_id"]): item for item in authority["records"]
    }
    if (
        not isinstance(signal_lineage_authority_by_identity, Mapping)
        or set(signal_lineage_authority_by_identity) != set(expected_records)
    ):
        raise PermissionError(
            "prediction phase lacks an exact opaque signal-lineage authority roster"
        )
    records = row["records"]
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("prediction artifact records must be an array")
    normalized_records = []
    seen = set()
    for record_value in records:
        record = _exact_fields(
            record_value,
            {
                "analysis_identity_id",
                "local_patient_id",
                "canonical_exact_clock",
                "strata",
            },
            "prediction record",
        )
        identity = _identifier(record["analysis_identity_id"], "analysis identity")
        if identity in seen or identity not in expected_records:
            raise ValueError("prediction record roster differs from typed authority")
        seen.add(identity)
        expected_patient = patient_by_identity.get(identity)
        if expected_patient is None or record["local_patient_id"] != expected_patient:
            raise PermissionError("patient identity/fold mapping drifted")
        duration = _fraction(
            expected_records[identity]["recording_duration_seconds_fraction"],
            "authorized recording duration",
            positive=True,
        )
        clock = _validate_exact_clock(
            record["canonical_exact_clock"],
            duration=duration,
            recording_id=identity,
            signal_lineage_authority=signal_lineage_authority_by_identity.get(
                identity
            ),
        )
        stratum_rows = record["strata"]
        if not isinstance(stratum_rows, Sequence) or isinstance(stratum_rows, (str, bytes)):
            raise TypeError("record strata must be an array")
        normalized_strata = []
        stratum_seen = set()
        for state_value in stratum_rows:
            state = _exact_fields(
                state_value,
                {
                    "stratum_id",
                    "producer_state",
                    "failure_reason_code",
                    "full_record_scan_state",
                    "full_record_scan_receipt_sha256",
                    "queries",
                },
                "record stratum state",
            )
            stratum_id = _identifier(state["stratum_id"], "record stratum ID")
            if stratum_id in stratum_seen or stratum_id not in stratum_by_id:
                raise ValueError("record stratum roster differs from artifact roster")
            stratum_seen.add(stratum_id)
            if state["producer_state"] not in _PRODUCER_STATES:
                raise ValueError("producer state is unsupported")
            if state["producer_state"] == "failed":
                _identifier(state["failure_reason_code"], "failure reason code")
            elif state["failure_reason_code"] is not None:
                raise ValueError("non-failure row carries a failure reason")
            _sha256(
                state["full_record_scan_receipt_sha256"],
                "full-record scan receipt",
            )
            expected_scan_state = {
                "present": "completed",
                "zero_atoms": "completed_zero_atoms",
                "failed": "failed",
                "not_applicable": "not_applicable",
            }[state["producer_state"]]
            if state["full_record_scan_state"] != expected_scan_state:
                raise PermissionError("producer and full-record scan states disagree")
            queries = state["queries"]
            if not isinstance(queries, Sequence) or isinstance(queries, (str, bytes)):
                raise TypeError("queries must be an array")
            normalized_queries = [
                _validate_query(item, clock=clock, stratum=stratum_by_id[stratum_id])
                for item in queries
            ]
            normalized_queries.sort(
                key=lambda item: (item["anchor_sample_index"], item["query_index"], item["query_id"])
            )
            query_keys = [(item["query_index"], item["query_id"]) for item in normalized_queries]
            if len(query_keys) != len(set(query_keys)):
                raise ValueError("duplicate prediction query")
            if state["producer_state"] == "present" and not normalized_queries:
                raise ValueError("present producer state has no query rows")
            if state["producer_state"] != "present" and normalized_queries:
                raise ValueError("non-present producer state cannot carry query rows")
            state["queries"] = normalized_queries
            normalized_strata.append(state)
        if stratum_seen != set(stratum_by_id):
            raise ValueError("record omitted a declared operator stratum")
        record["canonical_exact_clock"] = clock
        record["strata"] = sorted(normalized_strata, key=lambda item: item["stratum_id"])
        normalized_records.append(record)
    if seen != set(expected_records):
        raise ValueError("prediction artifact omitted authorized records, including zero/failure rows")
    row["records"] = sorted(normalized_records, key=lambda item: item["analysis_identity_id"])
    content_hash = _sha256(row["content_sha256"], "prediction artifact content hash")
    if content_hash != _self_hash(row, "content_sha256"):
        raise ValueError("prediction artifact content hash does not replay")
    return row


def _safe_artifact_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("artifact relative path must be text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise PermissionError("artifact path must be safe and relative")
    root_resolved = root.resolve(strict=True)
    path = root_resolved.joinpath(*pure.parts).resolve(strict=True)
    if path != root_resolved and root_resolved not in path.parents:
        raise PermissionError("artifact path escaped its root")
    return path


def _read_bound_artifact(
    binding_value: object,
    *,
    root: Path,
    preregistry: Mapping[str, Any],
    authority: Mapping[str, Any],
    patient_by_identity: Mapping[str, str],
    signal_lineage_authority_by_identity: Mapping[
        str, ValidatedDetectorSignalLineageAuthority
    ],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _exact_fields(
        binding_value,
        {"relative_path", "file_bytes", "file_sha256", "content_sha256"},
        "prediction artifact binding",
    )
    path = _safe_artifact_path(root, binding["relative_path"])
    observed_sha, observed_size = _file_sha256(path)
    if observed_sha != _sha256(binding["file_sha256"], "artifact file hash"):
        raise PermissionError("prediction artifact file bytes changed")
    if observed_size != _integer(binding["file_bytes"], "artifact file bytes", minimum=1):
        raise PermissionError("prediction artifact byte count changed")
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prediction artifact is not canonical JSON") from error
    validated = _validate_prediction_artifact(
        payload,
        preregistry=preregistry,
        authority=authority,
        patient_by_identity=patient_by_identity,
        signal_lineage_authority_by_identity=signal_lineage_authority_by_identity,
    )
    if binding["content_sha256"] != validated["content_sha256"]:
        raise PermissionError("prediction artifact semantic content changed")
    return validated, binding


def _interval_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _opportunities(
    *,
    authority_record: Mapping[str, Any],
    clock: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    fs = _fraction(clock["sample_rate_hz_fraction"], "sample rate", positive=True)
    total = clock["total_sample_count"]
    pre = _fraction(protocol["onset_uncertainty_pre_seconds_fraction"], "onset pre")
    post = _fraction(protocol["onset_uncertainty_post_seconds_fraction"], "onset post")
    guard = _ceil(_fraction(protocol["matched_background_guard_seconds_fraction"], "guard") * fs)
    stride = max(1, _ceil(_fraction(protocol["matched_background_stride_seconds_fraction"], "stride", positive=True) * fs))
    events = []
    occupied = []
    for index, event in enumerate(authority_record["seizure_intervals"]):
        onset = Fraction(str(event["start_seconds"]))
        start = max(0, _floor((onset - pre) * fs))
        stop = min(total, _ceil((onset + post) * fs))
        if stop <= start:
            stop = min(total, start + 1)
        seizure_start = max(0, _floor(Fraction(str(event["start_seconds"])) * fs) - guard)
        seizure_stop = min(total, _ceil(Fraction(str(event["stop_seconds"])) * fs) + guard)
        occupied.append((seizure_start, seizure_stop))
        events.append(
            {
                "opportunity_id": f"ONSET-{authority_record['analysis_identity_id']}-{index:04d}",
                "event_index": index,
                "onset_interval_samples": [start, stop],
                "background_interval_samples": None,
                "matching_state": "unmatched",
            }
        )
    chosen: list[tuple[int, int]] = []
    for opportunity in events:
        onset_window = tuple(opportunity["onset_interval_samples"])
        length = onset_window[1] - onset_window[0]
        candidates = []
        for start in range(0, max(0, total - length) + 1, stride):
            stop = start + length
            candidate = (start, stop)
            if any(_interval_overlap(candidate, blocked) for blocked in occupied + chosen):
                continue
            distance = min(abs(start - onset_window[0]), abs(stop - onset_window[1]))
            candidates.append((distance, start, stop))
        if candidates:
            _, start, stop = min(candidates)
            opportunity["background_interval_samples"] = [start, stop]
            opportunity["matching_state"] = "matched_same_record_equal_duration"
            chosen.append((start, stop))
    return events, sum(item["matching_state"] == "unmatched" for item in events)


def _merge_sample_intervals(intervals: Sequence[Sequence[int]]) -> list[list[int]]:
    rows = sorted((int(item[0]), int(item[1])) for item in intervals)
    merged: list[list[int]] = []
    for start, stop in rows:
        if stop <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return merged


def _full_non_event_intervals(
    *,
    authority_record: Mapping[str, Any],
    clock: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[list[int]]:
    """Return the frozen full-record non-event opportunity ledger geometry."""

    fs = _fraction(clock["sample_rate_hz_fraction"], "sample rate", positive=True)
    total = clock["total_sample_count"]
    pre = _fraction(protocol["onset_uncertainty_pre_seconds_fraction"], "onset pre")
    post = _fraction(protocol["onset_uncertainty_post_seconds_fraction"], "onset post")
    guard = _fraction(
        protocol["matched_background_guard_seconds_fraction"], "seizure guard"
    )
    excluded = []
    for event in authority_record["seizure_intervals"]:
        start_seconds = Fraction(str(event["start_seconds"])) - pre - guard
        stop_seconds = Fraction(str(event["stop_seconds"])) + post + guard
        excluded.append(
            [
                max(0, _floor(start_seconds * fs)),
                min(total, _ceil(stop_seconds * fs)),
            ]
        )
    merged = _merge_sample_intervals(excluded)
    result = []
    cursor = 0
    for start, stop in merged:
        if cursor < start:
            result.append([cursor, start])
        cursor = max(cursor, stop)
    if cursor < total:
        result.append([cursor, total])
    return result


def _interval_has_complete_query_coverage(
    *,
    queries: Sequence[Mapping[str, Any]],
    interval: Sequence[int],
    maximum_gap_samples: int,
) -> bool:
    start, stop = int(interval[0]), int(interval[1])
    anchors = sorted(
        item["anchor_sample_index"]
        for item in queries
        if item["qc_state"] == "evaluable"
        and start <= item["anchor_sample_index"] < stop
    )
    if stop <= start:
        return True
    if not anchors:
        return False
    if anchors[0] - start > maximum_gap_samples:
        return False
    if (stop - 1) - anchors[-1] > maximum_gap_samples:
        return False
    return all(
        anchors[index] - anchors[index - 1] <= maximum_gap_samples
        for index in range(1, len(anchors))
    )


def _higher_quantile(values: Sequence[float], quantile: Fraction) -> float:
    if not values:
        raise ValueError("cannot derive a threshold from an empty background")
    ordered = sorted(values)
    rank = max(1, _ceil(quantile * len(ordered)))
    return float(ordered[rank - 1])


def _trigger_episodes(
    *,
    queries: Sequence[Mapping[str, Any]],
    interval: Sequence[int],
    threshold: float,
    sign: str,
    persistence_seconds: Fraction,
    maximum_query_gap_seconds: Fraction,
    minimum_query_count: int,
    fs: Fraction,
) -> list[int]:
    start, stop = int(interval[0]), int(interval[1])
    selected = [
        item
        for item in queries
        if start <= item["anchor_sample_index"] < stop and item["qc_state"] == "evaluable"
    ]
    selected.sort(key=lambda item: (item["anchor_sample_index"], item["query_index"]))
    max_gap = _ceil(maximum_query_gap_seconds * fs)
    min_persistence = _ceil(persistence_seconds * fs)
    episodes: list[int] = []
    run: list[Mapping[str, Any]] = []
    emitted = False
    for query in selected:
        passes = _score(query["effect_value"], sign) >= threshold
        if not passes:
            run = []
            emitted = False
            continue
        if run and query["anchor_sample_index"] - run[-1]["anchor_sample_index"] > max_gap:
            run = []
            emitted = False
        run.append(query)
        persistence = run[-1]["measurement_stop_sample_index"] - run[0][
            "measurement_start_sample_index"
        ]
        if not emitted and len(run) >= minimum_query_count and persistence >= min_persistence:
            episodes.append(query["anchor_sample_index"])
            emitted = True
    return episodes


def _patient_macro(indicators: Sequence[tuple[str, int]]) -> Fraction:
    grouped: dict[str, list[int]] = defaultdict(list)
    for patient, value in indicators:
        grouped[patient].append(value)
    if not grouped:
        return Fraction(0, 1)
    return sum(
        (Fraction(sum(values), len(values)) for values in grouped.values()),
        Fraction(0, 1),
    ) / len(grouped)


def _permutation_p_value(
    *,
    pairs: Sequence[tuple[str, int, int, str]],
    permutation_count: int,
    seed_material: str,
) -> tuple[Fraction, Fraction]:
    observed_onset = _patient_macro([(patient, onset) for patient, onset, _, _ in pairs])
    observed_background = _patient_macro(
        [(patient, background) for patient, _, background, _ in pairs]
    )
    observed = observed_onset - observed_background
    exceed = 0
    for permutation_index in range(permutation_count):
        permuted_onset = []
        permuted_background = []
        for patient, onset, background, pair_id in pairs:
            token = hashlib.sha256(
                f"{seed_material}|{permutation_index}|{pair_id}".encode("utf-8")
            ).digest()
            swap = bool(token[0] & 1)
            permuted_onset.append((patient, background if swap else onset))
            permuted_background.append((patient, onset if swap else background))
        statistic = _patient_macro(permuted_onset) - _patient_macro(permuted_background)
        if statistic >= observed:
            exceed += 1
    return observed, Fraction(exceed + 1, permutation_count + 1)


def _identity_patient_map(fold_plan: Mapping[str, Any]) -> dict[str, str]:
    rows = fold_plan.get("source_record_duration_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("fold plan lacks source record duration rows")
    result = {}
    for row in rows:
        identity = str(row["analysis_identity_id"])
        patient = str(row["local_patient_id"])
        if identity in result and result[identity] != patient:
            raise ValueError("fold plan maps one recording to multiple patients")
        result[identity] = patient
    return result


def _phase_evidence(
    *,
    artifact: Mapping[str, Any],
    authority: Mapping[str, Any],
    preregistry: Mapping[str, Any],
) -> dict[str, Any]:
    authority_by_identity = {
        item["analysis_identity_id"]: item for item in authority["records"]
    }
    strata = {item["stratum_id"]: item for item in artifact["operator_strata"]}
    evidence = {stratum_id: {"background_scores": [], "records": []} for stratum_id in strata}
    opportunity_protocol = preregistry["opportunity_protocol"]
    for record in artifact["records"]:
        identity = record["analysis_identity_id"]
        authority_record = authority_by_identity[identity]
        opportunities, unmatched = _opportunities(
            authority_record=authority_record,
            clock=record["canonical_exact_clock"],
            protocol=opportunity_protocol,
        )
        full_non_event_intervals = _full_non_event_intervals(
            authority_record=authority_record,
            clock=record["canonical_exact_clock"],
            protocol=opportunity_protocol,
        )
        state_by_id = {item["stratum_id"]: item for item in record["strata"]}
        for stratum_id, stratum in strata.items():
            state = state_by_id[stratum_id]
            queries = state["queries"]
            background_scores = []
            for opportunity in opportunities:
                background = opportunity["background_interval_samples"]
                if background is None:
                    continue
                background_scores.extend(
                    _score(query["effect_value"], stratum["effect_sign_policy"])
                    for query in queries
                    if background[0] <= query["anchor_sample_index"] < background[1]
                    and query["qc_state"] == "evaluable"
                )
            evidence[stratum_id]["background_scores"].extend(background_scores)
            evidence[stratum_id]["records"].append(
                {
                    "analysis_identity_id": identity,
                    "local_patient_id": record["local_patient_id"],
                    "clock": record["canonical_exact_clock"],
                    "producer_state": state["producer_state"],
                    "failure_reason_code": state["failure_reason_code"],
                    "full_record_scan_state": state["full_record_scan_state"],
                    "full_record_scan_receipt_sha256": state[
                        "full_record_scan_receipt_sha256"
                    ],
                    "queries": queries,
                    "opportunities": opportunities,
                    "full_non_event_intervals": full_non_event_intervals,
                    "unmatched_opportunity_count": unmatched,
                }
            )
    return {"strata": strata, "evidence": evidence}


def _evaluate_candidate(
    *,
    stratum: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
    threshold: float,
    threshold_quantile: Fraction,
    persistence: Fraction,
    preregistry: Mapping[str, Any],
    outer_fold_id: int,
) -> dict[str, Any]:
    grid = preregistry["grid"]
    selection = preregistry["selection_protocol"]
    onset_indicators: list[tuple[str, int]] = []
    pairs: list[tuple[str, int, int, str]] = []
    paired_matched_background_false_trigger_count = 0
    paired_matched_background_seconds = Fraction(0, 1)
    full_non_event_false_trigger_count = 0
    full_non_event_evaluable_seconds = Fraction(0, 1)
    full_non_event_available_seconds = Fraction(0, 1)
    incomplete_full_non_event_record_count = 0
    unmatched = 0
    producer_failure_records = 0
    zero_atom_records = 0
    not_applicable_records = 0
    zero_onset_records = 0
    zero_onset_patients: set[str] = set()
    all_patients: set[str] = set()
    evaluable_onset_opportunity_count = 0
    censored_query_count = 0
    total_query_count = 0
    for record in evidence_rows:
        fs = _fraction(record["clock"]["sample_rate_hz_fraction"], "sample rate", positive=True)
        queries = record["queries"]
        total_query_count += len(queries)
        censored_query_count += sum(item["qc_state"] == "censored" for item in queries)
        producer_failure_records += record["producer_state"] == "failed"
        zero_atom_records += record["producer_state"] == "zero_atoms"
        not_applicable_records += record["producer_state"] == "not_applicable"
        all_patients.add(record["local_patient_id"])
        if not record["opportunities"]:
            zero_onset_records += 1
            zero_onset_patients.add(record["local_patient_id"])
        unmatched += record["unmatched_opportunity_count"]
        maximum_gap_samples = _ceil(
            _fraction(
                grid["maximum_query_gap_seconds_fraction"],
                "maximum query gap",
                positive=True,
            )
            * fs
        )
        non_event_intervals = record["full_non_event_intervals"]
        non_event_seconds = sum(
            (
                Fraction(interval[1] - interval[0], 1) / fs
                for interval in non_event_intervals
            ),
            Fraction(0, 1),
        )
        full_non_event_available_seconds += non_event_seconds
        if record["producer_state"] == "zero_atoms":
            complete_non_event_coverage = (
                record["full_record_scan_state"] == "completed_zero_atoms"
            )
        elif record["producer_state"] == "present":
            complete_non_event_coverage = (
                record["full_record_scan_state"] == "completed"
                and all(
                    _interval_has_complete_query_coverage(
                        queries=queries,
                        interval=interval,
                        maximum_gap_samples=maximum_gap_samples,
                    )
                    for interval in non_event_intervals
                )
            )
        else:
            complete_non_event_coverage = not non_event_intervals
        if complete_non_event_coverage:
            full_non_event_evaluable_seconds += non_event_seconds
            for interval in non_event_intervals:
                full_non_event_false_trigger_count += len(
                    _trigger_episodes(
                        queries=queries,
                        interval=interval,
                        threshold=threshold,
                        sign=stratum["effect_sign_policy"],
                        persistence_seconds=persistence,
                        maximum_query_gap_seconds=_fraction(
                            grid["maximum_query_gap_seconds_fraction"],
                            "maximum query gap",
                            positive=True,
                        ),
                        minimum_query_count=grid["minimum_stable_query_count"],
                        fs=fs,
                    )
                )
        elif non_event_intervals:
            incomplete_full_non_event_record_count += 1
        for opportunity in record["opportunities"]:
            if record["producer_state"] in {"present", "zero_atoms"}:
                evaluable_onset_opportunity_count += 1
            onset_episodes = _trigger_episodes(
                queries=queries,
                interval=opportunity["onset_interval_samples"],
                threshold=threshold,
                sign=stratum["effect_sign_policy"],
                persistence_seconds=persistence,
                maximum_query_gap_seconds=_fraction(
                    grid["maximum_query_gap_seconds_fraction"], "maximum query gap", positive=True
                ),
                minimum_query_count=grid["minimum_stable_query_count"],
                fs=fs,
            )
            onset_hit = int(bool(onset_episodes))
            onset_indicators.append((record["local_patient_id"], onset_hit))
            background = opportunity["background_interval_samples"]
            if background is None:
                continue
            if record["producer_state"] in {"failed", "not_applicable"}:
                # A technical/non-applicable row is an onset miss and remains
                # in coverage denominators, but it is not a zero false
                # trigger over otherwise evaluable background time.
                continue
            background_episodes = _trigger_episodes(
                queries=queries,
                interval=background,
                threshold=threshold,
                sign=stratum["effect_sign_policy"],
                persistence_seconds=persistence,
                maximum_query_gap_seconds=_fraction(
                    grid["maximum_query_gap_seconds_fraction"], "maximum query gap", positive=True
                ),
                minimum_query_count=grid["minimum_stable_query_count"],
                fs=fs,
            )
            paired_matched_background_false_trigger_count += len(
                background_episodes
            )
            paired_matched_background_seconds += (
                Fraction(background[1] - background[0], 1) / fs
            )
            pairs.append(
                (
                    record["local_patient_id"],
                    onset_hit,
                    int(bool(background_episodes)),
                    opportunity["opportunity_id"],
                )
            )
    macro = _patient_macro(onset_indicators)
    full_non_event_ftr = (
        Fraction(full_non_event_false_trigger_count * 3600, 1)
        / full_non_event_evaluable_seconds
        if full_non_event_evaluable_seconds > 0
        else None
    )
    candidate_seed = {
        "outer_fold_id": outer_fold_id,
        "stratum_id": stratum["stratum_id"],
        "threshold": threshold,
        "quantile": _fraction_json(threshold_quantile),
        "persistence": _fraction_json(persistence),
        "preregistry": preregistry["registry_receipt_sha256"],
    }
    candidate_id = "CANDIDATE-" + _sha(candidate_seed)[:24]
    observed_contrast, p_value = _permutation_p_value(
        pairs=pairs,
        permutation_count=selection["permutation_count"],
        seed_material=candidate_id,
    ) if pairs else (Fraction(0, 1), Fraction(1, 1))
    patient_count = len({patient for patient, _ in onset_indicators})
    onset_count = len(onset_indicators)
    ftr_limit = _fraction(selection["false_trigger_per_hour_limit_fraction"], "FTR limit", positive=True)
    alpha = _fraction(selection["permutation_alpha_fraction"], "permutation alpha", positive=True)
    coverage = (
        Fraction(evaluable_onset_opportunity_count, onset_count)
        if onset_count
        else Fraction(0, 1)
    )
    required_coverage = _fraction(
        selection["minimum_evaluable_onset_opportunity_fraction"],
        "minimum evaluable onset coverage",
        positive=True,
    )
    gates = {
        "complete_same_record_matching": unmatched == 0,
        "minimum_patient_denominator": patient_count
        >= selection["minimum_inner_validation_patients"],
        "minimum_onset_opportunity_denominator": onset_count
        >= selection["minimum_inner_validation_onset_opportunities"],
        "nonzero_full_non_event_time": full_non_event_evaluable_seconds > 0,
        "complete_full_non_event_query_coverage": (
            incomplete_full_non_event_record_count == 0
            and full_non_event_evaluable_seconds == full_non_event_available_seconds
        ),
        "full_non_event_false_trigger_per_hour_constraint": (
            full_non_event_ftr is not None and full_non_event_ftr <= ftr_limit
        ),
        "permutation_null_gate": p_value <= alpha and observed_contrast > 0,
        "minimum_evaluable_onset_opportunity_coverage": coverage
        >= required_coverage,
    }
    return {
        "candidate_id": candidate_id,
        "stratum_id": stratum["stratum_id"],
        "effect_threshold_value": threshold,
        "effect_threshold_unit": stratum["physical_unit"],
        "effect_sign_policy": stratum["effect_sign_policy"],
        "selection_fit_background_quantile_fraction": _fraction_json(threshold_quantile),
        "minimum_persistence_seconds_fraction": _fraction_json(persistence),
        "maximum_query_gap_seconds_fraction": deepcopy(grid["maximum_query_gap_seconds_fraction"]),
        "minimum_stable_query_count": grid["minimum_stable_query_count"],
        "patient_macro_onset_sensitivity_fraction": _fraction_json(macro),
        "full_non_event_false_trigger_count": full_non_event_false_trigger_count,
        "full_non_event_available_duration_seconds_fraction": _fraction_json(
            full_non_event_available_seconds
        ),
        "full_non_event_evaluable_duration_seconds_fraction": _fraction_json(
            full_non_event_evaluable_seconds
        ),
        "full_non_event_false_triggers_per_hour_fraction": (
            _fraction_json(full_non_event_ftr)
            if full_non_event_ftr is not None
            else None
        ),
        "paired_matched_background_false_trigger_count": (
            paired_matched_background_false_trigger_count
        ),
        "paired_matched_background_duration_seconds_fraction": _fraction_json(
            paired_matched_background_seconds
        ),
        "paired_matched_background_endpoint_semantics": (
            "within_record_null_control_not_full_non_event_FTR_denominator"
        ),
        "permutation_observed_contrast_fraction": _fraction_json(observed_contrast),
        "permutation_p_value_fraction": _fraction_json(p_value),
        "Bonferroni_adjusted_permutation_p_value_fraction": None,
        "denominators": {
            "patient_count_with_onset_opportunity": patient_count,
            "authorized_record_count": len(evidence_rows),
            "authorized_patient_count": len(all_patients),
            "onset_opportunity_count": onset_count,
            "evaluable_onset_opportunity_count": evaluable_onset_opportunity_count,
            "evaluable_onset_opportunity_fraction": _fraction_json(coverage),
            "matched_onset_background_pair_count": len(pairs),
            "unmatched_background_opportunity_count": unmatched,
            "incomplete_full_non_event_record_count": (
                incomplete_full_non_event_record_count
            ),
            "producer_failure_record_count": producer_failure_records,
            "zero_atom_record_count": zero_atom_records,
            "not_applicable_record_count": not_applicable_records,
            "zero_onset_opportunity_record_count": zero_onset_records,
            "zero_onset_opportunity_patient_count": len(zero_onset_patients),
            "query_count": total_query_count,
            "censored_query_count": censored_query_count,
        },
        "admission_gates": gates,
        "candidate_admitted": all(gates.values()),
    }


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    sensitivity = _fraction(row["patient_macro_onset_sensitivity_fraction"], "sensitivity")
    ftr = _fraction(
        row["full_non_event_false_triggers_per_hour_fraction"],
        "full non-event FTR",
    )
    quantile = _fraction(row["selection_fit_background_quantile_fraction"], "quantile")
    persistence = _fraction(row["minimum_persistence_seconds_fraction"], "persistence")
    return (-sensitivity, ftr, quantile, persistence, row["candidate_id"])


def _materialize_findings_onset_threshold_candidate_registry_core_v1(
    *,
    preregistry: Mapping[str, Any],
    fold_plan: Mapping[str, Any],
    fold_reference_registry: Mapping[str, Any],
    fold_inputs: Sequence[Mapping[str, Any]],
    prediction_artifact_root: Path,
    reference_root: Path,
    _require_opaque_phase_authorities: bool = False,
) -> dict[str, Any]:
    """Replay cross-fit artifacts and materialize a candidate registry.

    The default raw-receipt branch exists only for deterministic synthetic
    contract tests.  The public materializer sets
    ``_require_opaque_phase_authorities=True`` and therefore accepts only
    process-local actual-byte/controller-replayed phase authorities.  A
    missing fold, artifact, event/background match, null gate, or denominator
    causes a fail-closed exception; it never manufactures a threshold.
    """

    protocol = validate_findings_onset_threshold_preregistry_v1(preregistry)
    patient_map = _identity_patient_map(fold_plan)
    outer_count = protocol["split_contract"]["outer_fold_count"]
    if not isinstance(fold_inputs, Sequence) or isinstance(fold_inputs, (str, bytes)):
        raise TypeError("fold inputs must be an array")
    by_fold = {}
    for item in fold_inputs:
        expected_fold_fields = {
            "outer_fold_id",
            "selection_fit_reference_authority",
            "inner_validation_reference_authority",
            "selection_fit_prediction_artifact",
            "inner_validation_prediction_artifact",
            "selection_fit_signal_lineage_authorities",
            "inner_validation_signal_lineage_authorities",
        }
        if not isinstance(item, Mapping) or set(item) != expected_fold_fields:
            raise ValueError("threshold fold input fields drifted")
        # Keep opaque signal-lineage authorities shallow.  Deep-copying their
        # private seal would correctly make them invalid downstream.
        fold = dict(item)
        outer_fold_id = _integer(fold["outer_fold_id"], "outer fold ID")
        if outer_fold_id in by_fold:
            raise ValueError("duplicate threshold outer fold")
        by_fold[outer_fold_id] = fold
    if set(by_fold) != set(range(outer_count)):
        raise PermissionError("cross-fit registry requires every pre-registered outer fold")

    fold_receipts = []
    expected_strata: list[dict[str, Any]] | None = None
    for outer_fold_id in range(outer_count):
        fold = by_fold[outer_fold_id]
        authorities = {}
        artifacts = {}
        bindings = {}
        for phase in _PHASES:
            if _require_opaque_phase_authorities:
                opaque_authority = (
                    require_validated_detector_fold_reference_phase_authority_v1(
                        fold[f"{phase}_reference_authority"]
                    )
                )
                authority = deepcopy(dict(opaque_authority))
            else:
                authority = validate_detector_fold_reference_phase_v1(
                    fold[f"{phase}_reference_authority"],
                    fold_plan=fold_plan,
                    registry=fold_reference_registry,
                    replay_reference_root=reference_root,
                )
            if authority["outer_fold_id"] != outer_fold_id or authority["phase"] != phase:
                raise PermissionError("typed fold authority crossed fold/phase")
            authorities[phase] = authority
            artifact, binding = _read_bound_artifact(
                fold[f"{phase}_prediction_artifact"],
                root=prediction_artifact_root,
                preregistry=protocol,
                authority=authority,
                patient_by_identity=patient_map,
                signal_lineage_authority_by_identity=fold[
                    f"{phase}_signal_lineage_authorities"
                ],
            )
            artifacts[phase] = artifact
            bindings[phase] = binding
        if artifacts["selection_fit"]["operator_strata"] != artifacts["inner_validation"][
            "operator_strata"
        ]:
            raise PermissionError("operator strata changed between calibration phases")
        if expected_strata is None:
            expected_strata = artifacts["selection_fit"]["operator_strata"]
        elif artifacts["selection_fit"]["operator_strata"] != expected_strata:
            raise PermissionError("operator strata changed across outer folds")

        fit = _phase_evidence(
            artifact=artifacts["selection_fit"],
            authority=authorities["selection_fit"],
            preregistry=protocol,
        )
        validation = _phase_evidence(
            artifact=artifacts["inner_validation"],
            authority=authorities["inner_validation"],
            preregistry=protocol,
        )
        selected_rows = []
        candidate_inventory = []
        candidate_by_stratum: dict[str, list[dict[str, Any]]] = {}
        for stratum in expected_strata:
            stratum_id = stratum["stratum_id"]
            background_scores = fit["evidence"][stratum_id]["background_scores"]
            if not background_scores:
                raise PermissionError(
                    f"outer fold {outer_fold_id} stratum {stratum_id} has no selection-fit background"
                )
            candidates = []
            seen_thresholds = set()
            for quantile_json in protocol["grid"]["background_quantile_grid"]:
                quantile = _fraction(quantile_json, "threshold quantile", positive=True)
                threshold = _higher_quantile(background_scores, quantile)
                threshold_key = float(threshold).hex()
                if threshold_key in seen_thresholds:
                    continue
                seen_thresholds.add(threshold_key)
                for persistence_json in protocol["grid"]["minimum_persistence_seconds_grid"]:
                    candidates.append(
                        _evaluate_candidate(
                            stratum=stratum,
                            evidence_rows=validation["evidence"][stratum_id]["records"],
                            threshold=threshold,
                            threshold_quantile=quantile,
                            persistence=_fraction(persistence_json, "persistence", positive=True),
                            preregistry=protocol,
                            outer_fold_id=outer_fold_id,
                        )
                    )
            candidate_inventory.extend(candidates)
            candidate_by_stratum[stratum_id] = candidates

        multiplicity = len(candidate_inventory)
        if multiplicity < 1:
            raise PermissionError("outer fold has an empty threshold candidate family")
        alpha = _fraction(
            protocol["selection_protocol"]["permutation_alpha_fraction"],
            "permutation alpha",
            positive=True,
        )
        for candidate in candidate_inventory:
            raw_p = _fraction(
                candidate["permutation_p_value_fraction"], "permutation p value"
            )
            adjusted = min(Fraction(1, 1), raw_p * multiplicity)
            candidate["Bonferroni_adjusted_permutation_p_value_fraction"] = (
                _fraction_json(adjusted)
            )
            candidate["admission_gates"]["permutation_null_gate"] = (
                adjusted <= alpha
                and _fraction(
                    candidate["permutation_observed_contrast_fraction"],
                    "permutation observed contrast",
                )
                > 0
            )
            candidate["candidate_admitted"] = all(
                candidate["admission_gates"].values()
            )

        for stratum in expected_strata:
            stratum_id = stratum["stratum_id"]
            admitted = sorted(
                [
                    item
                    for item in candidate_by_stratum[stratum_id]
                    if item["candidate_admitted"]
                ],
                key=_candidate_sort_key,
            )
            if not admitted:
                raise PermissionError(
                    f"outer fold {outer_fold_id} stratum {stratum_id} has no admitted grid point"
                )
            selected = deepcopy(admitted[0])
            selected["operator_stratum"] = deepcopy(stratum)
            selected_rows.append(selected)
        fold_body = {
            "outer_fold_id": outer_fold_id,
            "selection_fit_authority_receipt_sha256": authorities["selection_fit"]["receipt_sha256"],
            "inner_validation_authority_receipt_sha256": authorities["inner_validation"]["receipt_sha256"],
            "selection_fit_prediction_artifact_binding": deepcopy(bindings["selection_fit"]),
            "inner_validation_prediction_artifact_binding": deepcopy(bindings["inner_validation"]),
            "candidate_inventory_sha256": _sha(candidate_inventory),
            "candidate_count": len(candidate_inventory),
            "Bonferroni_candidate_family_size": multiplicity,
            "selected_operator_strata": sorted(selected_rows, key=lambda item: item["stratum_id"]),
        }
        fold_body["fold_receipt_sha256"] = _self_hash(fold_body, "fold_receipt_sha256")
        fold_receipts.append(fold_body)

    body: dict[str, Any] = {
        "schema_version": FINDINGS_ONSET_THRESHOLD_ADMITTED_REGISTRY_SCHEMA_VERSION,
        "registry_id": protocol["registry_id"] + "-ADMITTED",
        "method_id": FINDINGS_ONSET_THRESHOLD_METHOD_ID,
        "status": "admitted_real_patient_disjoint_crossfit_artifacts_replayed",
        "preregistry_receipt_sha256": protocol["registry_receipt_sha256"],
        "fold_plan_receipt_sha256": _sha256(
            fold_plan.get("receipt_sha256"), "fold plan receipt"
        ),
        "fold_reference_registry_receipt_sha256": _sha256(
            fold_reference_registry.get("registry_receipt_sha256"),
            "fold reference registry receipt",
        ),
        "operator_strata": deepcopy(expected_strata or []),
        "fold_receipts": fold_receipts,
        "source_firewall": deepcopy(_SOURCE_FIREWALL),
        "forward_firewall": deepcopy(_FORWARD_FIREWALL),
        "authorization": {
            "registry_admitted": True,
            "positive_onset_trigger_authorized_only_after_exact_trajectory_stabilization": True,
            "positive_rank_contribution_authorized": False,
            "clinical_term_authorized": False,
            "SOZ_EZ_or_surgical_target_claim_authorized": False,
            "report_text_authorized": False,
        },
        "registry_receipt_sha256": "",
    }
    body["registry_receipt_sha256"] = _self_hash(body, "registry_receipt_sha256")
    validated = validate_findings_onset_threshold_admitted_registry_v1(
        body, preregistry=protocol
    )
    return validated


def _materialize_findings_onset_threshold_candidate_registry_for_synthetic_tests_v1(
    *,
    preregistry: Mapping[str, Any],
    fold_plan: Mapping[str, Any],
    fold_reference_registry: Mapping[str, Any],
    fold_inputs: Sequence[Mapping[str, Any]],
    prediction_artifact_root: Path,
    reference_root: Path,
) -> dict[str, Any]:
    """Exercise the candidate logic with raw synthetic phase receipts only."""

    return _materialize_findings_onset_threshold_candidate_registry_core_v1(
        preregistry=preregistry,
        fold_plan=fold_plan,
        fold_reference_registry=fold_reference_registry,
        fold_inputs=fold_inputs,
        prediction_artifact_root=prediction_artifact_root,
        reference_root=reference_root,
        _require_opaque_phase_authorities=False,
    )


def materialize_findings_onset_threshold_admitted_registry_v1(
    *,
    preregistry: Mapping[str, Any],
    fold_plan: Mapping[str, Any],
    fold_reference_registry: Mapping[str, Any],
    fold_inputs: Sequence[Mapping[str, Any]],
    prediction_artifact_root: Path,
    reference_root: Path,
) -> ValidatedFindingsOnsetThresholdRegistry:
    """Materialize only from opaque, actual-byte-replayed phase authorities.

    Each fold input must contain process-local authorities issued by
    ``detector_fold_reference_authority_v1``.  Reloaded/self-hashed mappings
    are rejected before prediction artifacts are opened.  The phase issuer is
    responsible for exact reference-byte replay and, for non-selection
    phases, controller-signed checkpoint/prediction replay.  This materializer
    then replays the bound Findings prediction bytes and seals the admitted
    threshold registry as a second process-local authority.
    """

    admitted_receipt = (
        _materialize_findings_onset_threshold_candidate_registry_core_v1(
            preregistry=preregistry,
            fold_plan=fold_plan,
            fold_reference_registry=fold_reference_registry,
            fold_inputs=fold_inputs,
            prediction_artifact_root=prediction_artifact_root,
            reference_root=reference_root,
            _require_opaque_phase_authorities=True,
        )
    )
    return _seal_findings_onset_threshold_registry_v1(admitted_receipt)


def validate_findings_onset_threshold_admitted_registry_v1(
    value: object, *, preregistry: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate content structure of an admitted materializer output.

    Actual prediction/reference byte replay occurs in the materializer.  A
    caller that loads a stored admitted registry must additionally replay the
    materializer with its bound artifacts before using it.
    """

    protocol = validate_findings_onset_threshold_preregistry_v1(preregistry)
    fields = {
        "schema_version",
        "registry_id",
        "method_id",
        "status",
        "preregistry_receipt_sha256",
        "fold_plan_receipt_sha256",
        "fold_reference_registry_receipt_sha256",
        "operator_strata",
        "fold_receipts",
        "source_firewall",
        "forward_firewall",
        "authorization",
        "registry_receipt_sha256",
    }
    row = _exact_fields(value, fields, "admitted threshold registry")
    if row["schema_version"] != FINDINGS_ONSET_THRESHOLD_ADMITTED_REGISTRY_SCHEMA_VERSION:
        raise ValueError("admitted threshold registry schema drifted")
    if row["method_id"] != FINDINGS_ONSET_THRESHOLD_METHOD_ID:
        raise ValueError("admitted threshold registry method drifted")
    if row["status"] != "admitted_real_patient_disjoint_crossfit_artifacts_replayed":
        raise PermissionError("admitted threshold registry status drifted")
    if row["preregistry_receipt_sha256"] != protocol["registry_receipt_sha256"]:
        raise PermissionError("admitted registry crossed its preregistration")
    _sha256(row["fold_plan_receipt_sha256"], "fold plan receipt")
    _sha256(row["fold_reference_registry_receipt_sha256"], "fold reference registry receipt")
    strata = [_validate_stratum(item) for item in row["operator_strata"]]
    if not strata or [item["stratum_id"] for item in strata] != sorted(
        set(item["stratum_id"] for item in strata)
    ):
        raise ValueError("admitted operator stratum roster is invalid")
    row["operator_strata"] = strata
    folds = row["fold_receipts"]
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
        raise TypeError("admitted fold receipts must be an array")
    expected_folds = set(range(protocol["split_contract"]["outer_fold_count"]))
    observed_folds = set()
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise TypeError("fold receipt must be an object")
        outer_fold_id = _integer(fold.get("outer_fold_id"), "outer fold ID")
        if outer_fold_id in observed_folds:
            raise ValueError("duplicate admitted outer fold")
        observed_folds.add(outer_fold_id)
        for field in (
            "selection_fit_authority_receipt_sha256",
            "inner_validation_authority_receipt_sha256",
            "candidate_inventory_sha256",
            "fold_receipt_sha256",
        ):
            _sha256(fold.get(field), f"fold {field}")
        if fold["fold_receipt_sha256"] != _self_hash(fold, "fold_receipt_sha256"):
            raise ValueError("admitted fold receipt does not replay")
        selected = fold.get("selected_operator_strata")
        if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
            raise TypeError("selected operator strata must be an array")
        if [item.get("stratum_id") for item in selected] != [
            item["stratum_id"] for item in strata
        ]:
            raise PermissionError("admitted fold omitted or reordered an operator stratum")
        if not all(item.get("candidate_admitted") is True for item in selected):
            raise PermissionError("admitted fold contains a failed candidate")
    if observed_folds != expected_folds:
        raise PermissionError("admitted threshold registry lacks a complete cross-fit denominator")
    _validate_bool_map(row["source_firewall"], _SOURCE_FIREWALL, "admitted source")
    _validate_bool_map(row["forward_firewall"], _FORWARD_FIREWALL, "admitted forward")
    if row["authorization"] != {
        "registry_admitted": True,
        "positive_onset_trigger_authorized_only_after_exact_trajectory_stabilization": True,
        "positive_rank_contribution_authorized": False,
        "clinical_term_authorized": False,
        "SOZ_EZ_or_surgical_target_claim_authorized": False,
        "report_text_authorized": False,
    }:
        raise PermissionError("admitted threshold authorization drifted")
    receipt = _sha256(row["registry_receipt_sha256"], "admitted registry receipt")
    if receipt != _self_hash(row, "registry_receipt_sha256"):
        raise ValueError("admitted threshold registry receipt does not replay")
    return row


def _seal_findings_onset_threshold_registry_v1(
    value: Mapping[str, Any],
) -> ValidatedFindingsOnsetThresholdRegistry:
    """Seal a structurally validated receipt (materializer/test-internal only)."""

    return ValidatedFindingsOnsetThresholdRegistry(
        _receipt_json=_canonical_json_bytes(value).decode("utf-8"),
        _validation_seal=_ADMITTED_REGISTRY_SEAL,
    )


def require_validated_findings_onset_threshold_registry_v1(
    value: object, *, preregistry: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the opaque seal issued after real cross-fit byte replay."""

    if (
        not isinstance(value, ValidatedFindingsOnsetThresholdRegistry)
        or value._validation_seal is not _ADMITTED_REGISTRY_SEAL
    ):
        raise TypeError(
            "positive trajectory replay requires an opaque admitted threshold "
            "registry issued by the real artifact materializer"
        )
    return validate_findings_onset_threshold_admitted_registry_v1(
        value.receipt, preregistry=preregistry
    )


__all__ = [
    "FINDINGS_CANONICAL_EXACT_CLOCK_SCHEMA_VERSION",
    "FINDINGS_ONSET_THRESHOLD_ADMITTED_REGISTRY_SCHEMA_VERSION",
    "FINDINGS_ONSET_THRESHOLD_METHOD_ID",
    "FINDINGS_ONSET_THRESHOLD_PREDICTION_ARTIFACT_SCHEMA_VERSION",
    "FINDINGS_ONSET_THRESHOLD_PREREGISTRY_SCHEMA_VERSION",
    "ValidatedFindingsOnsetThresholdRegistry",
    "build_findings_canonical_exact_clock_from_signal_authority_v1",
    "build_findings_onset_threshold_preregistry_v1",
    "materialize_findings_onset_threshold_admitted_registry_v1",
    "require_validated_findings_onset_threshold_registry_v1",
    "validate_findings_onset_threshold_admitted_registry_v1",
    "validate_findings_onset_threshold_preregistry_v1",
]
