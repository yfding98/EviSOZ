"""Exact clean-room transform and trainer registry for ST18 and ST16.

The vendored SeizureTransformer source is an architecture source, not an
executable preprocessing or training specification.  This module freezes the
project profile needed to train two *independent* scratch models:

* ``ST18`` uses the explicit 18-lead longitudinal bipolar montage;
* ``ST16`` derives its own 16-lead lateral montage directly from referential
  volts.  It is never produced by deleting channels from an ST18 tensor.

Deterministic CPU transform/loss/sampling primitives are separated from the
formal authority path.  Formal targets, pools, class weights and epoch plans
accept only a shared process-sealed reference phase plus a complete opaque
variant roster derived from target-blind support/technical outcomes.  Importing
this module never opens an EDF, reference sidecar, spreadsheet, checkpoint,
GPU, or model service.  No real phase adapter/variant roster/checkpoint/OOF or
qualified operating point is materialized by this registry.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import scipy
from scipy.signal import resample_poly, sosfiltfilt

from .detector_signal_lineage_authority_v1 import (
    ValidatedDetectorSignalLineageAuthority,
    require_validated_detector_signal_lineage_authority,
    verify_provider_referential_payload,
)
from .detector_channel_support_router_v1 import (
    detector_channel_support_policy_receipt,
    route_detector_channel_support,
)
from .detector_fold_reference_authority_v1 import (
    ValidatedDetectorFoldReferencePhaseAuthorityV1,
    require_validated_detector_fold_reference_phase_authority_v1,
)


SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_seizuretransformer_cleanroom_transform_trainer_registry_v1"
)
REGISTRY_ID: Final[str] = (
    "CLINICAL-EEG-SEIZURETRANSFORMER-CLEANROOM-TRANSFORM-TRAINER-V1-20260824"
)
PROVIDER_ID: Final[str] = "seizuretransformer_cleanroom_retrained_v1"
ST18_VARIANT_ID: Final[str] = "seizuretransformer_st18_cleanroom_v1"
ST16_VARIANT_ID: Final[str] = (
    "seizuretransformer_st16_common_support_cleanroom_v1"
)
TARGET_FS_HZ: Final[int] = 256
TILE_SAMPLES: Final[int] = 15_360
TRAIN_HOP_SAMPLES: Final[int] = 3_840
ST16_SHORT_CONTEXT_POLICY_ID: Final[str] = (
    "ST16-short-reflect-context-valid-support-mask-v1"
)
CONFIG_RELATIVE_PATH: Final[str] = (
    "configs/clinical_eeg_seizuretransformer_cleanroom_transform_trainer_"
    "registry_v1.json"
)

_CONTENT_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
_SHA256_CHARS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TRANSFORM_RESULT_SEAL = object()
_FOLD_PHASE_AUTHORITY_SEAL = object()
_PRE_REFERENCE_ELIGIBILITY_SEAL = object()
_VARIANT_TRAINING_ROSTER_SEAL = object()
_TARGET_BUNDLE_SEAL = object()
_RECORD_POOL_SEAL = object()
_CLASS_WEIGHT_SEAL = object()

ST18_TYPED_UNITS: Final[tuple[str, ...]] = (
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
ST16_TYPED_UNITS: Final[tuple[str, ...]] = (
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
)

STANDARD_19_ELECTRODES: Final[tuple[str, ...]] = (
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

# Frozen little-endian float64 SOS generated once with SciPy 1.11.4:
# scipy.signal.butter(4, [0.5, 100], btype="bandpass", fs=256, output="sos")
# Hard-coding the coefficients prevents a future design-helper implementation
# change from silently changing the provider transform.
_BANDPASS_SOS = np.asarray(
    [
        [
            0.39187500036281836,
            0.7837500007256367,
            0.39187500036281836,
            1.0,
            0.9720075206139325,
            0.25990295021005744,
        ],
        [1.0, 2.0, 1.0, 1.0, 1.2443880046307907, 0.6103302356889025],
        [1.0, -2.0, 1.0, 1.0, -1.97736034580957, 0.9775103717835337],
        [1.0, -2.0, 1.0, 1.0, -1.9905305395630186, 0.9906806257270332],
    ],
    dtype="<f8",
)
_BANDPASS_SOS.setflags(write=False)
_BANDPASS_SOS_SHA256: Final[str] = hashlib.sha256(
    _BANDPASS_SOS.tobytes(order="C")
).hexdigest()


@dataclass(frozen=True)
class SeizureTransformerTransformResult:
    """One immutable provider-native full-record carrier and its receipt."""

    signal: np.ndarray
    receipt: dict[str, Any]
    _validation_seal: object


@dataclass(frozen=True)
class AuthorizedSeizureTransformerFoldPhase:
    """Opaque ST adapter over the shared actual-byte-replayed phase."""

    _phase_receipt_json: str
    _patient_by_identity_json: str
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class SeizureTransformerPreReferenceEligibilityOutcome:
    """Opaque target-blind support/technical outcome for one ST variant."""

    transform_result: SeizureTransformerTransformResult | None
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class AuthorizedSeizureTransformerVariantTrainingRoster:
    """Exact phase intersection with route and technical eligibility."""

    _roster_json: str
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class AuthorizedSeizureTransformerTargetBundle:
    """Dense binary target bound to one exact fully observed ST tile."""

    target: np.ndarray
    observed_mask: np.ndarray
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class AuthorizedSeizureTransformerRecordPool:
    """Opaque full-record tile pool bound to one phase/variant record."""

    _pool_json: str
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class AuthorizedSeizureTransformerClassWeight:
    """Opaque patient-equal class weight derived from the complete roster."""

    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _require_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return str(value)


def _strict_dict(
    value: object, required: Iterable[str], context: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    expected = set(required)
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise ValueError(f"{context} fields drifted; missing={missing}, extra={extra}")
    return deepcopy(value)


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(value))
    row["receipt_sha256"] = _CONTENT_PENDING
    row["receipt_sha256"] = _canonical_sha256(row)
    return row


def _validate_content_address(
    value: object, *, required: Iterable[str], context: str
) -> dict[str, Any]:
    row = _strict_dict(value, required, context)
    supplied = _require_sha256(row["receipt_sha256"], f"{context} receipt")
    row["receipt_sha256"] = _CONTENT_PENDING
    if supplied != _canonical_sha256(row):
        raise ValueError(f"{context} is not content-addressed")
    return deepcopy(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seizuretransformer_cleanroom_registry_code_sha256() -> str:
    """Bind runtime receipts to this exact implementation."""

    return _file_sha256(Path(__file__))


def _payload_receipt(value: np.ndarray, *, semantic: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    dtype_map = {
        np.dtype("float32"): ("<f4", "float32_little_endian"),
        np.dtype("float64"): ("<f8", "float64_little_endian"),
        np.dtype("uint8"): ("u1", "uint8"),
        np.dtype("int64"): ("<i8", "int64_little_endian"),
    }
    if array.dtype not in dtype_map:
        raise TypeError("unsupported receipt dtype")
    target_dtype, dtype_name = dtype_map[array.dtype]
    canonical = np.ascontiguousarray(array, dtype=target_dtype)
    result: dict[str, Any] = {
        "semantic": semantic,
        "dtype": dtype_name,
        "shape": [int(item) for item in canonical.shape],
        "payload_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }
    if canonical.size:
        result["minimum"] = float(np.min(canonical))
        result["maximum"] = float(np.max(canonical))
    else:
        result["minimum"] = None
        result["maximum"] = None
    return result


def _typed_unit_pairs(variant_id: str) -> tuple[tuple[str, str], ...]:
    if variant_id == ST18_VARIANT_ID:
        roster = ST18_TYPED_UNITS
    elif variant_id == ST16_VARIANT_ID:
        roster = ST16_TYPED_UNITS
    else:
        raise ValueError("unknown independent SeizureTransformer variant")
    return tuple(tuple(unit.split("-", 1)) for unit in roster)  # type: ignore[return-value]


def _variant_profile(variant_id: str) -> dict[str, Any]:
    pairs = _typed_unit_pairs(variant_id)
    typed_units = tuple(f"{left}-{right}" for left, right in pairs)
    required_electrodes = tuple(
        electrode
        for electrode in STANDARD_19_ELECTRODES
        if any(electrode in pair for pair in pairs)
    )
    expected_count = 18 if variant_id == ST18_VARIANT_ID else 16
    if len(typed_units) != expected_count:
        raise RuntimeError("internal typed-unit profile is inconsistent")
    return {
        "variant_id": variant_id,
        "architecture_input_channels": expected_count,
        "typed_unit_count": expected_count,
        "typed_units": list(typed_units),
        "bipolar_pairs": [list(pair) for pair in pairs],
        "polarity": "first_referential_electrode_minus_second_referential_electrode",
        "required_source_electrodes": list(required_electrodes),
        "direct_from_referential_volts": True,
        "may_accept_complete19_referential_carrier": True,
        "may_accept_lateral17_referential_carrier": variant_id == ST16_VARIANT_ID,
        "construct_ST18_then_delete_channels_allowed": False,
        "share_runtime_model_weights_with_other_variant_allowed": False,
        "independent_random_initialization_training_and_checkpoint_required": True,
    }


def _architecture_contract(variant_id: str) -> dict[str, Any]:
    return {
        "class": "third_party.SeizureTransformer.time_step_level.model.SeizureTransformer",
        "upstream_repository": "https://github.com/keruiwu/SeizureTransformer",
        "upstream_commit": "cf83f5906a8aea88b60b56e4f962c5d6657c28f7",
        "architecture_source_path": "third_party/SeizureTransformer/time_step_level/model.py",
        "architecture_source_sha256": (
            "0c3fd38a5350bb293e5337c26bb01c83945624b6eb8000da50e955e54174c7b2"
        ),
        "constructor": {
            "in_channels": 18 if variant_id == ST18_VARIANT_ID else 16,
            "in_samples": TILE_SAMPLES,
            "dim_feedforward": 2048,
            "num_layers": 8,
            "num_heads": 4,
            "drop_rate": 0.1,
        },
        "encoder_filters": [32, 64, 128, 256, 512],
        "encoder_kernel_sizes": [11, 9, 7, 7, 5],
        "encoder_activation": "ELU_inplace",
        "encoder_pool": "MaxPool1d_kernel_2_with_manual_right_negative_infinity_padding_for_odd_length",
        "residual_kernel_sizes": [3, 3, 3, 3, 2, 3, 2],
        "residual_normalization": "BatchNorm1d_epsilon_0_001",
        "residual_activation": "ReLU",
        "residual_spatial_dropout_probability": 0.1,
        "transformer_d_model": 512,
        "transformer_norm_first": False,
        "transformer_batch_first": False,
        "transformer_activation": "relu",
        "transformer_feedforward_dimension": 2048,
        "transformer_encoder_layers": 8,
        "transformer_attention_heads": 4,
        "transformer_and_positional_dropout_probability": 0.1,
        "positional_encoding_max_len": 6000,
        "attention_padding_mask_argument": None,
        "padded_short_record_forward_allowed": False,
        "decoder_filters": [512, 256, 128, 64, 32],
        "decoder_kernel_sizes": [3, 5, 5, 7, 7],
        "decoder_upsample": "nearest_factor_2_with_exact_output_crop_ledger",
        "final_convolution": {
            "input_channels": 32,
            "output_channels": 1,
            "kernel_size": 11,
            "padding": 5,
        },
        "dense_output": "sigmoid_probability_one_value_per_target_sample",
        "random_from_scratch_only": True,
        "public_or_third_party_checkpoint_initialization_allowed": False,
    }


def _common_transform_contract() -> dict[str, Any]:
    return {
        "profile_id": "st_cleanroom_full_record_robust_v1",
        "input": {
            "carrier": "canonical_referential_EEG_only",
            "unit": "volts",
            "dtype_at_boundary": "float32_or_float64_finite",
            "common_single_sampling_clock_required": True,
            "typed_detector_signal_lineage_authority_required": True,
            "bare_roster_or_SHA256_lineage_argument_allowed": False,
            "canonical_source_tensor_payload_replay_required": True,
            "common_EEG_reference_basis_provenance_required": True,
            "mixed_or_unknown_reference_basis_allowed": False,
            "referential_lineage_receipt_used_as_model_feature": False,
            "allowed_electrode_vocabulary": list(STANDARD_19_ELECTRODES),
            "non_EEG_auxiliary_channels_allowed": False,
        },
        "ordered_operations": [
            "validate_typed_canonical_signal_roster_reference_QC_authority_and_payload",
            "derive_variant_specific_bipolar_pairs_in_float64",
            "whole_record_rational_polyphase_resample_to_256Hz",
            "whole_record_zero_phase_fixed_SOS_bandpass_0_5_to_100Hz",
            "whole_record_per_typed_unit_median_MAD_normalization",
            "symmetric_clip_to_20_and_cast_little_endian_float32",
        ],
        "montage": {
            "performed_before_resample_filter_and_normalization": True,
            "arithmetic_dtype": "float64",
            "runtime_ST18_channel_deletion_for_ST16_allowed": False,
        },
        "resample": {
            "method": "scipy_signal_resample_poly_explicit_kaiser_sinc_taps_v1",
            "target_sampling_rate_hz": TARGET_FS_HZ,
            "FFT_resample_allowed": False,
            "ratio": "reduce_Fraction_256_divided_by_exact_input_rate",
            "tap_count": "20_times_max_up_down_plus_1",
            "ideal_lowpass_cutoff_normalized_to_Nyquist": "1_divided_by_max_up_down",
            "window": "numpy_kaiser_beta_5_0",
            "tap_normalization": "divide_by_float64_sum",
            "padtype": "line",
            "output_sample_count": "floor(input_samples_times_up_divided_by_down)",
            "discard_only_resample_poly_ceil_tail_beyond_last_supported_target_edge": True,
            "whole_record_execution": True,
        },
        "bandpass": {
            "method": "scipy_signal_sosfiltfilt_fixed_coefficients",
            "design": "fourth_order_Butterworth_bandpass_applied_forward_and_backward",
            "low_hz": 0.5,
            "high_hz": 100.0,
            "sampling_rate_hz": TARGET_FS_HZ,
            "sos_float64": _BANDPASS_SOS.tolist(),
            "sos_payload_sha256": _BANDPASS_SOS_SHA256,
            "padtype": "odd",
            "padlen_samples": 768,
            "minimum_target_sample_count": TILE_SAMPLES,
            "minimum_target_sample_reason": (
                "vendored_architecture_has_no_attention_padding_mask_so_every_"
                "model_tile_must_be_fully_observed"
            ),
            "edge_support_flag_samples_each_side": 768,
            "tile_filter_state_reset_allowed": False,
            "notch_filter": None,
            "one_hz_notch_allowed": False,
        },
        "normalization": {
            "primary": "whole_record_per_typed_unit_median_MAD",
            "center": "float64_median_over_all_target_samples",
            "scale": "1_4826_times_float64_median_absolute_deviation",
            "scale_multiplier": 1.4826,
            "division_safety_epsilon_volts": 1e-12,
            "division_safety_epsilon_role": (
                "numerical_degeneracy_guard_only_not_a_clinical_amplitude_or_"
                "EEG_channel_usability_threshold"
            ),
            "clinical_or_EEG_QC_amplitude_threshold_volts": None,
            "channel_usability_authority": (
                "upstream_EEG_derived_QC_and_support_router_never_inferred_"
                "from_this_normalization_epsilon"
            ),
            "scale_floor_substitution_allowed": False,
            "nonfinite_or_numerically_degenerate_scale_policy": (
                "terminal_technical_failure_without_flat_channel_clinical_claim"
            ),
            "clip_lower": -20.0,
            "clip_upper": 20.0,
            "output_dtype": "float32_little_endian_dimensionless",
            "fit_scope": "same_complete_record_no_cross_record_statistics",
        },
        "padding": {
            "provider_transform_creates_tile_padding": False,
            "model_input_padding_allowed": False,
            "records_shorter_than_15360_target_samples": (
                "ST16_only_deterministic_reflect_context_challenger_or_typed_"
                "exclusion_for_other_variants"
            ),
            "architecture_attention_padding_mask_supported": False,
            "short_context_policy_id": ST16_SHORT_CONTEXT_POLICY_ID,
            "short_context_is_observed_EEG": False,
            "short_context_may_receive_target_loss_or_metric_weight": False,
            "short_context_reflection_duplicates_endpoint": False,
            "short_context_valid_support_mask_payload_and_ledger_required": True,
            "short_context_boundary_prediction_influence_must_be_audited": True,
            "loss_mask_for_admitted_model_tiles": (
                "all_ones_for_native_tiles_or_original_support_ones_and_"
                "context_zeros_for_ST16_short_challenger"
            ),
        },
        "forbidden_shortcuts": [
            "per_tile_filter_or_normalization_fit",
            "per_tile_IIR_state_reset",
            "FFT_resample",
            "one_hz_notch",
            "whole_record_mean_standard_deviation_in_primary_profile",
            "annotation_or_spreadsheet_or_doctor_text",
            "detector_tensor_reuse_as_native_Findings_EEG",
        ],
    }


def _trainer_contract(
    *,
    fold_plan_file_sha256: str,
    fold_authority_registry_file_sha256: str,
    fold_authority_registry_receipt_sha256: str,
) -> dict[str, Any]:
    return {
        "trainer_id": "st_cleanroom_dense_patient_balanced_opaque_authority_trainer_v1",
        "outer_fold_plan": {
            "path": (
                "outputs/tusz_canonical_physical_signal_audit_v1_full_20260824r2/"
                "detector_cleanroom_fold_plan.json"
            ),
            "file_sha256": fold_plan_file_sha256,
            "plan_id": "TUSZDETCLEANFOLDV1-c4808802f6ab2626332782b9",
            "fold_count": 5,
            "selection_mappings": [
                {
                    "outer_heldout": 0,
                    "inner_validation": 1,
                    "selection_fit": [2, 3, 4],
                    "final_refit": [1, 2, 3, 4],
                },
                {
                    "outer_heldout": 1,
                    "inner_validation": 2,
                    "selection_fit": [0, 3, 4],
                    "final_refit": [0, 2, 3, 4],
                },
                {
                    "outer_heldout": 2,
                    "inner_validation": 3,
                    "selection_fit": [0, 1, 4],
                    "final_refit": [0, 1, 3, 4],
                },
                {
                    "outer_heldout": 3,
                    "inner_validation": 4,
                    "selection_fit": [0, 1, 2],
                    "final_refit": [0, 1, 2, 4],
                },
                {
                    "outer_heldout": 4,
                    "inner_validation": 0,
                    "selection_fit": [1, 2, 3],
                    "final_refit": [0, 1, 2, 3],
                },
            ],
        },
        "fold_reference_authority": {
            "path": "configs/clinical_eeg_detector_fold_reference_authority_registry_v1.json",
            "file_sha256": fold_authority_registry_file_sha256,
            "registry_receipt_sha256": fold_authority_registry_receipt_sha256,
            "shared_opaque_phase_type": (
                "ValidatedDetectorFoldReferencePhaseAuthorityV1"
            ),
            "raw_mapping_or_bare_SHA_allowed": False,
        },
        "formal_authority_wiring": {
            "fold_phase_type": "AuthorizedSeizureTransformerFoldPhase",
            "shared_controller_actual_byte_replayed_opaque_phase_consumed": True,
            "variant_training_roster_type": (
                "AuthorizedSeizureTransformerVariantTrainingRoster"
            ),
            "variant_training_roster_definition": (
                "authorized_phase_intersection_target_blind_provider_route_"
                "intersection_pre_reference_technical_eligibility"
            ),
            "pre_reference_eligibility_type": (
                "SeizureTransformerPreReferenceEligibilityOutcome"
            ),
            "target_bundle_type": "AuthorizedSeizureTransformerTargetBundle",
            "record_pool_type": "AuthorizedSeizureTransformerRecordPool",
            "class_weight_type": "AuthorizedSeizureTransformerClassWeight",
            "complete_variant_eligible_record_denominator_required": True,
            "typed_exclusions_retained_in_prediction_first_denominator": True,
            "raw_targets_masks_counts_patient_keys_or_rosters_allowed": False,
            "pre_reference_variant_roster_artifact_materialized": False,
            "fold_phase_adapter_artifact_materialized": False,
            "epoch_executor_implemented": False,
        },
        "stages_per_outer_fold": [
            {
                "stage": "selection",
                "gradient_roster": "three_frozen_selection_fit_patient_groups",
                "validation_roster": "one_frozen_inner_validation_patient_group",
                "class_weight_fit_roster": "selection_fit_only",
            },
            {
                "stage": "final_refit",
                "gradient_roster": "all_four_outer_train_patient_groups",
                "validation_roster": None,
                "class_weight_fit_roster": "all_four_outer_train_patient_groups",
                "epoch_count": "selected_by_the_corresponding_selection_stage",
                "reinitialize_from_scratch": True,
            },
        ],
        "target": {
            "type": "dense_binary_sample_target_on_256Hz_half_open_clock",
            "target_sample_rule": (
                "positive_when_target_sample_center_is_inside_an_authorized_"
                "TERM_seiz_half_open_interval"
            ),
            "right_context_target_value": 0,
            "right_context_mask_value": 0,
            "model_input_padding_allowed": False,
            "short_context_is_not_observed_EEG_or_target_support": True,
            "observed_loss_mask_for_every_admitted_tile": (
                "all_ones_native_or_one_on_original_support_zero_on_short_context"
            ),
            "outer_heldout_target_open_before_prediction_freeze_allowed": False,
            "formal_target_requires_opaque_phase_and_variant_roster": True,
            "raw_event_rows_are_numeric_primitive_only": True,
        },
        "loss": {
            "name": "masked_patient_macro_weighted_dense_binary_cross_entropy",
            "implementation": "binary_cross_entropy_on_architecture_sigmoid_probabilities_not_BCEWithLogits",
            "probability_epsilon": 1e-7,
            "fold_train_class_weight": {
                "negative_weight": 1.0,
                "positive_weight": (
                    "min_50_of_patient_equal_negative_mass_divided_by_"
                    "patient_equal_positive_mass"
                ),
                "positive_weight_cap": 50.0,
                "fit_population": "unique_observed_absolute_samples_not_overlapping_tile_copies",
                "zero_positive_or_zero_negative_mass": "fail_closed",
            },
            "tile_reduction": "weighted_mean_over_observed_mask_only",
            "patient_reduction": "mean_of_all_tile_numerators_divided_by_all_tile_denominators_per_patient",
            "batch_reduction": "unweighted_mean_over_distinct_patients",
            "formal_loss_accepts_raw_masks_counts_or_patient_keys": False,
        },
        "patient_balanced_sampling": {
            "method": "stateless_hash_patient_round_robin_v1",
            "tiles_per_patient_per_epoch": 8,
            "positive_tile_guarantee_for_patients_with_positive_tiles": 1,
            "remaining_tiles": "uniform_hash_cycle_over_complete_patient_tile_roster",
            "tile_positive_definition": "at_least_one_observed_positive_target_sample",
            "patients_per_batch_maximum": 16,
            "one_tile_per_patient_per_batch": True,
            "drop_last": False,
            "replacement_only_when_patient_pool_smaller_than_requested_quota": True,
            "identity_used_as_model_feature": False,
        },
        "optimizer": {
            "class": "torch.optim.RAdam",
            "learning_rate": 0.0001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.00002,
            "decoupled_weight_decay": False,
            "foreach": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "scheduler": None,
            "gradient_clip_global_L2_norm": 1.0,
        },
        "numeric_execution": {
            "training_autocast_dtype": "bfloat16",
            "master_parameter_and_optimizer_state_dtype": "float32",
            "validation_probability_and_loss_accumulation_dtype": "float64",
            "microbatch_patient_count_maximum": 16,
            "gradient_accumulation_steps": 1,
            "CUDA_TF32_allowed": False,
            "cudnn_benchmark": False,
            "torch_deterministic_algorithms": True,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "dataloader_workers": 0,
            "automatic_microbatch_backoff_allowed": False,
            "OOM_policy": "fail_preflight_do_not_silently_change_batchnorm_semantics",
        },
        "epoch_selection": {
            "maximum_selection_epochs": 100,
            "minimum_selection_epochs_before_patience": 20,
            "early_stop_patience_epochs": 15,
            "improvement_rule": "strictly_lower_patient_macro_dense_validation_loss",
            "tie_break": "earlier_epoch",
            "validation_reduction": (
                "sample_aligned_full_record_OLA_probability_then_one_BCE_per_"
                "unique_observed_absolute_sample_then_unweighted_patient_mean"
            ),
            "source_dev_used": False,
            "outer_heldout_used": False,
        },
        "seed": {
            "base_seed": 20260824,
            "derivation": "sha256_base_variant_outer_fold_stage_to_31bit_nonzero_v1",
            "best_seed_selection_allowed": False,
            "python_numpy_torch_cpu_and_all_cuda_seeded": True,
        },
        "checkpoint": {
            "model_payload": "tensor_only_safetensors",
            "optimizer_resume_payload": "separate_content_addressed_tensor_and_JSON_state",
            "pickle_checkpoint_allowed": False,
            "selection_checkpoint_may_initialize_final_refit": False,
            "final_refit_checkpoint_count_per_variant": 5,
            "save_boundary": "completed_epoch_only",
            "resume_boundary": "next_epoch_only",
            "mid_epoch_resume_allowed": False,
            "schema_only_validator_implemented": True,
            "formal_admission_status": (
                "fail_closed_pending_actual_artifact_byte_replay"
            ),
            "formal_admission_reads_required_artifact_bytes": False,
            "current_ST18_checkpoint_count": 0,
            "current_ST16_checkpoint_count": 0,
            "required_receipts": [
                "variant_and_architecture_source_hash",
                "registry_transform_and_trainer_hashes",
                "canonical_referential_carrier_lineage_roster_hash",
                "outer_fold_stage_and_all_patient_roster_hashes",
                "fold_scoped_reference_authority_receipt",
                "class_weight_receipt",
                "seed_and_RNG_state_receipt",
                "epoch_sampler_receipt",
                "optimizer_and_numeric_environment_receipt",
                "model_payload_sha256",
                "zero_out_of_scope_reference_open_receipt",
            ],
        },
    }


def _execution_contract(
    *,
    ola_adapter_source_sha256: str,
    channel_router_source_sha256: str,
    signal_lineage_authority_source_sha256: str,
    fold_plan_file_sha256: str,
    fold_authority_registry_file_sha256: str,
    fold_authority_source_sha256: str,
    phase_gate_source_sha256: str,
) -> dict[str, Any]:
    return {
        "static_source_bindings": [
            {
                "semantic": "vendored_architecture_source",
                "path": "third_party/SeizureTransformer/time_step_level/model.py",
                "file_sha256": "0c3fd38a5350bb293e5337c26bb01c83945624b6eb8000da50e955e54174c7b2",
            },
            {
                "semantic": "sample_aligned_streaming_OLA_adapter",
                "path": "src/clinical_eeg_long_recording/seizuretransformer_streaming_ola_adapter_v1.py",
                "file_sha256": ola_adapter_source_sha256,
            },
            {
                "semantic": "target_blind_channel_support_router",
                "path": "src/clinical_eeg_long_recording/detector_channel_support_router_v1.py",
                "file_sha256": channel_router_source_sha256,
            },
            {
                "semantic": "typed_detector_signal_lineage_authority",
                "path": "src/clinical_eeg_long_recording/detector_signal_lineage_authority_v1.py",
                "file_sha256": signal_lineage_authority_source_sha256,
            },
            {
                "semantic": "patient_disjoint_outer_fold_plan",
                "path": (
                    "outputs/tusz_canonical_physical_signal_audit_v1_full_20260824r2/"
                    "detector_cleanroom_fold_plan.json"
                ),
                "file_sha256": fold_plan_file_sha256,
            },
            {
                "semantic": "typed_fold_reference_authority_registry",
                "path": "configs/clinical_eeg_detector_fold_reference_authority_registry_v1.json",
                "file_sha256": fold_authority_registry_file_sha256,
            },
            {
                "semantic": "typed_fold_reference_authority_implementation",
                "path": "src/clinical_eeg_long_recording/detector_fold_reference_authority_v1.py",
                "file_sha256": fold_authority_source_sha256,
            },
            {
                "semantic": "typed_reference_phase_gate",
                "path": "src/clinical_eeg_long_recording/detector_reference_phase_gate_v1.py",
                "file_sha256": phase_gate_source_sha256,
            },
        ],
        "supported_runtime_versions": {
            "python": "3.11.x",
            "numpy": "1.26.4",
            "scipy": "1.11.4",
            "torch": "2.5.1",
        },
        "transform_execution": "whole_record_CPU_stream_or_bounded_array_then_provider_tile_reader",
        "CPU_array_reference_transform_implemented": True,
        "whole_corpus_array_materialization_allowed": False,
        "training_device": "single_CUDA_GPU_only_after_resource_preflight",
        "current_vLLM_service_may_be_stopped_by_this_registry": False,
        "GPU_long_training_started_by_this_registry": False,
        "pre_reference_variant_roster_artifact": None,
        "fold_phase_adapter_artifact": None,
        "checkpoint_admission_artifact": None,
        "inference": {
            "primary": "60_second_tiles_15_second_hop_sample_aligned_weighted_OLA",
            "efficiency_ablation": "60_second_nonoverlap_same_checkpoint",
            "tile_flatten_or_concatenate_allowed": False,
        },
        "streaming_adapter_status": {
            ST18_VARIANT_ID: "weighted_OLA_geometry_implemented_checkpoint_absent",
            ST16_VARIANT_ID: "independent_weighted_OLA_geometry_implemented_checkpoint_absent",
        },
        "ST16_runtime_policy": (
            "load_ST16_architecture_and_ST16_checkpoint_and_ST16_transform_"
            "receipt_never_slice_ST18_tensor_or_reuse_ST18_checkpoint"
        ),
        "forward_allowlist": [
            "provider_preprocessed_EEG_sample_tensor",
            "fixed_fully_observed_shape_control_plane",
        ],
        "forward_forbidden": [
            "EDF_annotation",
            "spreadsheet_or_doctor_text",
            "clinical_history",
            "video_or_behavior",
            "sleep_or_activation_labels",
            "ECG_EMG_EOG",
            "patient_identity_feature",
            "lineage_hash_feature",
        ],
    }


def build_registry(
    *,
    implementation_code_sha256: str,
    signal_lineage_authority_source_sha256: str | None = None,
    channel_router_source_sha256: str | None = None,
    fold_plan_file_sha256: str | None = None,
    fold_authority_registry_file_sha256: str | None = None,
    fold_authority_registry_receipt_sha256: str | None = None,
    fold_authority_source_sha256: str | None = None,
    phase_gate_source_sha256: str | None = None,
    ola_adapter_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the exact semantic registry before its self-address is assigned."""

    root = Path(__file__).resolve().parents[2]
    default_paths = {
        "signal_lineage_authority_source_sha256": root
        / "src/clinical_eeg_long_recording/detector_signal_lineage_authority_v1.py",
        "channel_router_source_sha256": root
        / "src/clinical_eeg_long_recording/detector_channel_support_router_v1.py",
        "fold_plan_file_sha256": root
        / "outputs/tusz_canonical_physical_signal_audit_v1_full_20260824r2/detector_cleanroom_fold_plan.json",
        "fold_authority_registry_file_sha256": root
        / "configs/clinical_eeg_detector_fold_reference_authority_registry_v1.json",
        "fold_authority_source_sha256": root
        / "src/clinical_eeg_long_recording/detector_fold_reference_authority_v1.py",
        "phase_gate_source_sha256": root
        / "src/clinical_eeg_long_recording/detector_reference_phase_gate_v1.py",
        "ola_adapter_source_sha256": root
        / "src/clinical_eeg_long_recording/seizuretransformer_streaming_ola_adapter_v1.py",
    }
    supplied: dict[str, str | None] = {
        "signal_lineage_authority_source_sha256": signal_lineage_authority_source_sha256,
        "channel_router_source_sha256": channel_router_source_sha256,
        "fold_plan_file_sha256": fold_plan_file_sha256,
        "fold_authority_registry_file_sha256": fold_authority_registry_file_sha256,
        "fold_authority_source_sha256": fold_authority_source_sha256,
        "phase_gate_source_sha256": phase_gate_source_sha256,
        "ola_adapter_source_sha256": ola_adapter_source_sha256,
    }
    resolved = {
        key: (_file_sha256(default_paths[key]) if value is None else value)
        for key, value in supplied.items()
    }
    if fold_authority_registry_receipt_sha256 is None:
        try:
            fold_authority_registry_receipt_sha256 = json.loads(
                default_paths["fold_authority_registry_file_sha256"].read_text(
                    encoding="utf-8"
                )
            )["registry_receipt_sha256"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                "canonical fold reference registry receipt is unavailable"
            ) from exc
    signal_lineage_authority_source_sha256 = resolved[
        "signal_lineage_authority_source_sha256"
    ]
    channel_router_source_sha256 = resolved["channel_router_source_sha256"]
    fold_plan_file_sha256 = resolved["fold_plan_file_sha256"]
    fold_authority_registry_file_sha256 = resolved[
        "fold_authority_registry_file_sha256"
    ]
    fold_authority_source_sha256 = resolved["fold_authority_source_sha256"]
    phase_gate_source_sha256 = resolved["phase_gate_source_sha256"]
    ola_adapter_source_sha256 = resolved["ola_adapter_source_sha256"]
    hashes = {
        "implementation_code_sha256": implementation_code_sha256,
        "signal_lineage_authority_source_sha256": signal_lineage_authority_source_sha256,
        "channel_router_source_sha256": channel_router_source_sha256,
        "fold_plan_file_sha256": fold_plan_file_sha256,
        "fold_authority_registry_file_sha256": fold_authority_registry_file_sha256,
        "fold_authority_registry_receipt_sha256": fold_authority_registry_receipt_sha256,
        "fold_authority_source_sha256": fold_authority_source_sha256,
        "phase_gate_source_sha256": phase_gate_source_sha256,
        "ola_adapter_source_sha256": ola_adapter_source_sha256,
    }
    for context, value in hashes.items():
        _require_sha256(value, context)
    code_hash = implementation_code_sha256
    registry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "registry_id": REGISTRY_ID,
        "status": (
            "transform_and_opaque_training_authority_primitives_executable_"
            "artifacts_and_models_not_materialized"
        ),
        "provider_id": PROVIDER_ID,
        "extends_without_modifying": [
            "configs/clinical_eeg_detector_cleanroom_execution_freeze_v1.json",
            "configs/clinical_eeg_detector_channel_support_routing_addendum_v1.json",
            "src/clinical_eeg_long_recording/seizuretransformer_streaming_ola_adapter_v1.py",
        ],
        "implementation": {
            "path": (
                "src/clinical_eeg_long_recording/"
                "seizuretransformer_cleanroom_registry_v1.py"
            ),
            "code_sha256": code_hash,
        },
        "variant_profiles": {
            ST18_VARIANT_ID: {
                "input_profile": _variant_profile(ST18_VARIANT_ID),
                "architecture": _architecture_contract(ST18_VARIANT_ID),
            },
            ST16_VARIANT_ID: {
                "input_profile": _variant_profile(ST16_VARIANT_ID),
                "architecture": _architecture_contract(ST16_VARIANT_ID),
            },
        },
        "transform": _common_transform_contract(),
        "trainer": _trainer_contract(
            fold_plan_file_sha256=fold_plan_file_sha256,
            fold_authority_registry_file_sha256=fold_authority_registry_file_sha256,
            fold_authority_registry_receipt_sha256=(
                fold_authority_registry_receipt_sha256
            ),
        ),
        "execution": _execution_contract(
            ola_adapter_source_sha256=ola_adapter_source_sha256,
            channel_router_source_sha256=channel_router_source_sha256,
            signal_lineage_authority_source_sha256=(
                signal_lineage_authority_source_sha256
            ),
            fold_plan_file_sha256=fold_plan_file_sha256,
            fold_authority_registry_file_sha256=(
                fold_authority_registry_file_sha256
            ),
            fold_authority_source_sha256=fold_authority_source_sha256,
            phase_gate_source_sha256=phase_gate_source_sha256,
        ),
        "finite_comparators": [
            {
                "comparator_id": "whole_record_mean_std_no_clip_v1",
                "difference_from_primary": (
                    "replace_median_MAD_and_clip_by_per_typed_unit_"
                    "whole_record_mean_population_standard_deviation_without_clip"
                ),
                "reason": "released_source_uses_whole_record_zscore_but_exact_checkpoint_profile_is_unverified",
                "role": "separately_trained_nonpromotion_ablation_only",
                "may_reuse_primary_checkpoint": False,
                "source_train_inner_may_select_for_primary": False,
                "source_dev_or_source_eval_may_select_for_primary": False,
            }
        ],
        "selection_authority": {
            "source_train_inner_may_select": [
                "epoch_by_minimum_patient_macro_dense_validation_loss"
            ],
            "source_train_inner_may_not_select": [
                "ST18_versus_ST16_route",
                "primary_transform_versus_normalization_comparator",
                "filter_or_resample_parameters",
                "architecture_width_depth_heads_or_dropout",
                "optimizer_or_loss",
                "random_seed",
                "outer_heldout_threshold",
            ],
            "source_dev_role": "post_OOF_provider_policy_and_operating_point_selection_only",
            "source_eval_role": "one_shot_after_freeze_only",
        },
        "scientific_claim_boundary": {
            "transform_registry_executable": True,
            "CPU_transform_loss_sampler_smoke_available": True,
            "opaque_shared_phase_adapter_API_implemented": True,
            "target_blind_variant_roster_API_implemented": True,
            "opaque_target_pool_class_weight_and_epoch_plan_APIs_implemented": True,
            "pre_reference_variant_roster_artifact_materialized": False,
            "fold_phase_adapter_artifact_materialized": False,
            "torch_training_loop_executor_implemented": False,
            "ST18_streaming_OLA_geometry_implemented": True,
            "ST16_streaming_OLA_geometry_implemented": True,
            "ST18_checkpoint_count": 0,
            "ST16_checkpoint_count": 0,
            "full_stack_OOF_materialized": False,
            "formal_checkpoint_admission_executable": False,
            "accuracy_primary": None,
            "performance_or_SOTA_claim_allowed": False,
            "clinical_or_production_use_allowed": False,
        },
        "registry_sha256": _CONTENT_PENDING,
    }
    pending = deepcopy(registry)
    pending["registry_sha256"] = _CONTENT_PENDING
    registry["registry_sha256"] = _canonical_sha256(pending)
    return registry


def validate_registry(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on any registry, transform, variant, or authority drift."""

    required = {
        "schema_version",
        "registry_id",
        "status",
        "provider_id",
        "extends_without_modifying",
        "implementation",
        "variant_profiles",
        "transform",
        "trainer",
        "execution",
        "finite_comparators",
        "selection_authority",
        "scientific_claim_boundary",
        "registry_sha256",
    }
    data = _strict_dict(dict(value), required, "SeizureTransformer registry")
    if data["schema_version"] != SCHEMA_VERSION or data["registry_id"] != REGISTRY_ID:
        raise ValueError("SeizureTransformer registry identity drifted")
    if data["provider_id"] != PROVIDER_ID:
        raise ValueError("SeizureTransformer provider identity drifted")
    implementation = _strict_dict(
        data["implementation"], {"path", "code_sha256"}, "implementation binding"
    )
    _require_sha256(implementation["code_sha256"], "implementation code hash")
    bindings = {
        row["semantic"]: row for row in data["execution"]["static_source_bindings"]
    }
    expected = build_registry(
        implementation_code_sha256=implementation["code_sha256"],
        signal_lineage_authority_source_sha256=bindings[
            "typed_detector_signal_lineage_authority"
        ]["file_sha256"],
        channel_router_source_sha256=bindings[
            "target_blind_channel_support_router"
        ]["file_sha256"],
        fold_plan_file_sha256=bindings[
            "patient_disjoint_outer_fold_plan"
        ]["file_sha256"],
        fold_authority_registry_file_sha256=bindings[
            "typed_fold_reference_authority_registry"
        ]["file_sha256"],
        fold_authority_registry_receipt_sha256=data["trainer"][
            "fold_reference_authority"
        ]["registry_receipt_sha256"],
        fold_authority_source_sha256=bindings[
            "typed_fold_reference_authority_implementation"
        ]["file_sha256"],
        phase_gate_source_sha256=bindings["typed_reference_phase_gate"][
            "file_sha256"
        ],
        ola_adapter_source_sha256=bindings[
            "sample_aligned_streaming_OLA_adapter"
        ]["file_sha256"],
    )
    if data != expected:
        raise ValueError("SeizureTransformer registry semantic content drifted")
    if data["registry_sha256"] != _canonical_sha256(
        {**data, "registry_sha256": _CONTENT_PENDING}
    ):
        raise ValueError("SeizureTransformer registry is not content-addressed")
    return data


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.is_file() or registry_path.is_symlink():
        raise ValueError("registry path must be a regular non-symlink file")
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("registry is not readable canonical JSON") from exc
    validated = validate_registry(payload)
    if validated["implementation"]["code_sha256"] != (
        seizuretransformer_cleanroom_registry_code_sha256()
    ):
        raise ValueError("registry implementation code binding is stale")
    return validated


def validate_static_execution_bindings(
    project_root: str | Path, *, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Hash-check architecture, router, OLA and fold plan before execution."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError("project_root must be a directory")
    validated = validate_registry(dict(registry))
    rows: list[dict[str, Any]] = []
    for binding in validated["execution"]["static_source_bindings"]:
        path = root / binding["path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"static binding is missing or not regular: {binding['path']}")
        observed = _file_sha256(path)
        if observed != binding["file_sha256"]:
            raise ValueError(f"static source binding drifted: {binding['semantic']}")
        rows.append(
            {
                "semantic": binding["semantic"],
                "path": binding["path"],
                "observed_file_sha256": observed,
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": "st_cleanroom_static_execution_binding_receipt_v1",
        "registry_sha256": validated["registry_sha256"],
        "binding_count": len(rows),
        "bindings": rows,
        "all_exact": True,
        "receipt_sha256": _CONTENT_PENDING,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def validate_runtime_environment(
    registry: Mapping[str, Any], *, training: bool = False
) -> dict[str, Any]:
    """Validate the frozen numerical runtime without opening a GPU service."""

    validated = validate_registry(dict(registry))
    expected = validated["execution"]["supported_runtime_versions"]
    observed: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("runtime Python is outside frozen 3.11.x support")
    for package in ("numpy", "scipy"):
        if observed[package] != expected[package]:
            raise RuntimeError(f"runtime {package} version drifted")
    if training:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment gate
            raise RuntimeError("training runtime has no PyTorch") from exc
        observed["torch"] = str(torch.__version__).split("+")[0]
        if observed["torch"] != expected["torch"]:
            raise RuntimeError("runtime torch version drifted")
    receipt: dict[str, Any] = {
        "schema_version": "st_cleanroom_numeric_runtime_receipt_v1",
        "registry_sha256": validated["registry_sha256"],
        "training_runtime_checked": bool(training),
        "expected_versions": deepcopy(expected),
        "observed_versions": observed,
        "GPU_initialized_or_queried": False,
        "receipt_sha256": _CONTENT_PENDING,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def _exact_rate(numerator: int, denominator: int) -> Fraction:
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator <= 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise ValueError("sampling rate must be a positive exact fraction")
    rate = Fraction(numerator, denominator)
    if rate < 128 or rate > 2048:
        raise ValueError("input EEG sampling rate is outside frozen 128--2048 Hz support")
    return rate


def _polyphase_taps(up: int, down: int) -> np.ndarray:
    rate = max(up, down)
    half_length = 10 * rate
    index = np.arange(-half_length, half_length + 1, dtype=np.float64)
    cutoff = 1.0 / float(rate)
    taps = cutoff * np.sinc(cutoff * index)
    taps *= np.kaiser(taps.size, 5.0)
    taps /= np.sum(taps, dtype=np.float64)
    result = np.ascontiguousarray(taps, dtype="<f8")
    result.setflags(write=False)
    return result


def _normalize_electrode_order(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("electrode_order must be a sequence")
    order = tuple(value)
    if not order or any(not isinstance(item, str) for item in order):
        raise ValueError("electrode_order must contain non-empty canonical strings")
    if any(item not in STANDARD_19_ELECTRODES for item in order):
        raise ValueError("non-allowlisted or noncanonical electrode entered transform")
    if len(set(order)) != len(order):
        raise ValueError("electrode_order contains duplicates")
    return order


def _transform_receipt_sha256(value: Mapping[str, Any]) -> str:
    pending = deepcopy(dict(value))
    pending["receipt_sha256"] = _CONTENT_PENDING
    return _canonical_sha256(pending)


def _right_context_extend(
    value: np.ndarray,
    *,
    target_sample_count: int,
    mode: str,
) -> np.ndarray:
    """Extend a verified carrier on the right for fixed-shape context only."""

    source = np.asarray(value)
    if (
        source.ndim != 2
        or source.shape[0] < 1
        or source.shape[1] < 2
        or not np.issubdtype(source.dtype, np.floating)
        or not np.isfinite(source).all()
    ):
        raise ValueError("short-context source must be finite [channels,time>=2]")
    if (
        isinstance(target_sample_count, bool)
        or not isinstance(target_sample_count, int)
        or target_sample_count < source.shape[1]
    ):
        raise ValueError("short-context target may not truncate observed support")
    if mode not in {"reflect", "edge", "wrap"}:
        raise ValueError("short-context mode must be reflect, edge or wrap")
    padding = target_sample_count - source.shape[1]
    canonical = np.ascontiguousarray(source, dtype="<f8")
    if not padding:
        return canonical
    result = np.pad(
        canonical,
        ((0, 0), (0, padding)),
        mode=mode,
    )
    return np.ascontiguousarray(result, dtype="<f8")


def _apply_full_record_transform_impl(
    referential_volts: object,
    *,
    variant_id: str,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
    short_context_mode: str,
    sensitivity_only: bool,
) -> SeizureTransformerTransformResult:
    """Execute the frozen primary transform on one complete EEG recording.

    The input must replay against a typed canonical physical-signal authority.
    EEG electrical-reference provenance is required signal control-plane
    information; seizure target/reference labels remain absent and forbidden.
    The function intentionally materializes the complete transformed recording;
    the downstream tile adapter provides bounded model reads and OLA.
    """

    validated_registry = validate_registry(dict(registry))
    runtime_receipt = validate_runtime_environment(
        validated_registry, training=False
    )
    lineage = require_validated_detector_signal_lineage_authority(
        signal_lineage_authority
    )
    canonical_input, verified_order, verified_rate = (
        verify_provider_referential_payload(
            signal_lineage_authority, referential_volts
        )
    )
    order = _normalize_electrode_order(verified_order)
    rate = _exact_rate(verified_rate[0], verified_rate[1])
    clock_hash = lineage["common_sampling_clock_authority"]["receipt_sha256"]
    carrier_lineage_hash = lineage["receipt_sha256"]
    input_receipt = _payload_receipt(
        canonical_input, semantic="canonical_referential_EEG_volts"
    )

    profile = _variant_profile(variant_id)
    usable = set(
        lineage["EEG_only_channel_QC_authority"]["usable_standard_channel_ids"]
    )
    missing = sorted(
        set(profile["required_source_electrodes"]).difference(usable)
    )
    if missing:
        raise ValueError(f"variant source support is incomplete: {missing}")
    channel_index = {electrode: index for index, electrode in enumerate(order)}
    # Each variant is derived directly from referential volts.  There is no
    # ST18 intermediate or channel-deletion path for ST16.
    bipolar = np.stack(
        [
            canonical_input[channel_index[left]]
            - canonical_input[channel_index[right]]
            for left, right in _typed_unit_pairs(variant_id)
        ],
        axis=0,
    )
    bipolar = np.ascontiguousarray(bipolar, dtype="<f8")

    ratio = Fraction(TARGET_FS_HZ, 1) / rate
    up, down = ratio.numerator, ratio.denominator
    if max(up, down) > 4096:
        raise ValueError("reduced resampling ratio exceeds frozen support")
    observed_target_count = (bipolar.shape[1] * up) // down
    short_context = observed_target_count < TILE_SAMPLES
    if short_context:
        if variant_id != ST16_VARIANT_ID:
            raise ValueError(
                "record shorter than 60 seconds is unadmitted for ST18 because "
                "the architecture has no attention padding mask; the short-record "
                "context challenger is ST16-only"
            )
        if bipolar.shape[1] < 2 or observed_target_count < 1:
            raise ValueError("ST16 short record lacks reflection support")
        required_source_count = math.ceil(Fraction(TILE_SAMPLES * down, up))
        transform_bipolar = _right_context_extend(
            bipolar,
            target_sample_count=required_source_count,
            mode=short_context_mode,
        )
        target_count = (transform_bipolar.shape[1] * up) // down
        if target_count < TILE_SAMPLES:
            raise RuntimeError("ST16 short-context extension missed model support")
    else:
        if sensitivity_only:
            raise PermissionError(
                "short-context sensitivity transform requires a short ST16 record"
            )
        transform_bipolar = bipolar
        target_count = observed_target_count
    if up == down == 1:
        taps: np.ndarray | None = None
        resampled = transform_bipolar.copy()
    else:
        taps = _polyphase_taps(up, down)
        resampled = resample_poly(
            transform_bipolar,
            up,
            down,
            axis=1,
            window=taps,
            padtype="line",
        )
        if resampled.shape[1] < target_count:
            raise RuntimeError("polyphase resampler returned insufficient support")
        resampled = resampled[:, :target_count]
    if short_context:
        resampled = resampled[:, :TILE_SAMPLES]
        target_count = TILE_SAMPLES
    resampled = np.ascontiguousarray(resampled, dtype="<f8")

    # SciPy 1.11's Cython SOS kernel requests a writable coefficient buffer
    # even though it does not semantically mutate the filter.  Keep the frozen
    # module constant immutable and provide an exact writable copy per call.
    filtered = sosfiltfilt(
        _BANDPASS_SOS.copy(),
        resampled,
        axis=1,
        padtype="odd",
        padlen=768,
    )
    filtered = np.ascontiguousarray(filtered, dtype="<f8")
    if not np.isfinite(filtered).all():
        raise ValueError("bandpass output contains nonfinite values")
    center = np.median(filtered, axis=1).astype("<f8", copy=False)
    mad = np.median(np.abs(filtered - center[:, None]), axis=1).astype(
        "<f8", copy=False
    )
    scale = np.ascontiguousarray(1.4826 * mad, dtype="<f8")
    if not np.isfinite(scale).all() or np.any(scale <= 1e-12):
        raise ValueError(
            "typed-unit robust scale is nonfinite or numerically degenerate at "
            "or below 1e-12 volts"
        )
    normalized = (filtered - center[:, None]) / scale[:, None]
    np.clip(normalized, -20.0, 20.0, out=normalized)
    output = np.ascontiguousarray(normalized, dtype="<f4")
    if not np.isfinite(output).all():
        raise ValueError("normalized provider carrier contains nonfinite values")
    output.setflags(write=False)

    tap_receipt = None if taps is None else _payload_receipt(
        taps, semantic="explicit_polyphase_kaiser_sinc_taps"
    )
    payload_semantic = (
        "SeizureTransformer_short_record_fixed_context_carrier"
        if short_context
        else "SeizureTransformer_provider_native_full_record"
    )
    short_context_ledger: dict[str, Any] | None = None
    if short_context:
        valid_mask = np.zeros(TILE_SAMPLES, dtype=np.uint8)
        valid_mask[:observed_target_count] = 1
        short_context_ledger = _content_address(
            {
                "schema_version": "st16_short_record_context_ledger_v1",
                "policy_id": ST16_SHORT_CONTEXT_POLICY_ID,
                "context_mode": short_context_mode,
                "formal_context_policy": (
                    short_context_mode == "reflect" and not sensitivity_only
                ),
                "sensitivity_only": sensitivity_only,
                "source_observed_sample_count": bipolar.shape[1],
                "source_context_sample_count": transform_bipolar.shape[1],
                "source_right_context_sample_range": [
                    bipolar.shape[1],
                    transform_bipolar.shape[1],
                ],
                "provider_observed_sample_count": observed_target_count,
                "provider_model_context_sample_count": TILE_SAMPLES,
                "provider_valid_support_sample_range": [0, observed_target_count],
                "provider_right_context_sample_range": [
                    observed_target_count,
                    TILE_SAMPLES,
                ],
                "reflection_without_endpoint_duplication": (
                    short_context_mode == "reflect"
                ),
                "context_is_observed_EEG": False,
                "context_may_receive_target_loss_or_metric_weight": False,
                "context_may_assert_Finding_or_clinical_fact": False,
                "normalization_and_model_boundary_predictions_may_depend_on_context": True,
                "observed_bipolar_payload_receipt": _payload_receipt(
                    bipolar, semantic="ST16_short_observed_bipolar_volts"
                ),
                "extended_bipolar_payload_receipt": _payload_receipt(
                    transform_bipolar,
                    semantic=f"ST16_short_{short_context_mode}_context_bipolar_volts",
                ),
                "valid_support_mask_payload_receipt": _payload_receipt(
                    valid_mask,
                    semantic="ST16_short_original_support_loss_metric_mask",
                ),
                "receipt_sha256": _CONTENT_PENDING,
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": (
            "seizuretransformer_short_context_transform_receipt_v1"
            if short_context
            else "seizuretransformer_full_record_transform_receipt_v1"
        ),
        "registry_id": validated_registry["registry_id"],
        "registry_sha256": validated_registry["registry_sha256"],
        "implementation_code_sha256": validated_registry["implementation"][
            "code_sha256"
        ],
        "provider_id": PROVIDER_ID,
        "variant_id": variant_id,
        "profile_id": validated_registry["transform"]["profile_id"],
        "input_clock_receipt_sha256": clock_hash,
        "detector_signal_lineage_authority_sha256": carrier_lineage_hash,
        "canonical_signal_receipt_sha256": lineage["canonical_physical_signal"][
            "canonical_signal_receipt_sha256"
        ],
        "canonical_source_header_receipt_sha256": lineage[
            "canonical_physical_signal"
        ]["source_header_receipt_sha256"],
        "canonical_source_tensor_sha256": lineage["canonical_physical_signal"][
            "source_tensor_sha256"
        ],
        "observed_roster_authority_sha256": lineage[
            "observed_roster_authority"
        ]["receipt_sha256"],
        "EEG_electrical_reference_system_authority_sha256": lineage[
            "electrical_reference_system_authority"
        ]["receipt_sha256"],
        "EEG_only_channel_QC_authority_sha256": lineage[
            "EEG_only_channel_QC_authority"
        ]["receipt_sha256"],
        "input_electrode_order": list(order),
        "input_sampling_rate_fraction_hz": [rate.numerator, rate.denominator],
        "input_payload_receipt": input_receipt,
        "direct_variant_bipolar_derivation": {
            "typed_units": profile["typed_units"],
            "bipolar_pairs": profile["bipolar_pairs"],
            "ST18_intermediate_used_for_ST16": False,
        },
        "resample": {
            "up": up,
            "down": down,
            "target_sample_count_floor_policy": target_count,
            "tap_payload_receipt": tap_receipt,
            "padtype": "line",
        },
        "bandpass": {
            "sos_payload_sha256": _BANDPASS_SOS_SHA256,
            "padtype": "odd",
            "padlen_samples": 768,
            "whole_record_single_execution": True,
        },
        "normalization": {
            "center_payload_receipt": _payload_receipt(
                np.ascontiguousarray(center, dtype="<f8"),
                semantic="per_typed_unit_whole_record_median_volts",
            ),
            "scale_payload_receipt": _payload_receipt(
                scale, semantic="per_typed_unit_whole_record_1_4826_MAD_volts"
            ),
            "clip_interval": [-20.0, 20.0],
        },
        "output": {
            "typed_units": profile["typed_units"],
            "sampling_rate_fraction_hz": [TARGET_FS_HZ, 1],
            "sample_count": target_count,
            "unit": "dimensionless_robust_score",
            "at_least_one_fully_observed_60_second_model_tile": not short_context,
            "attention_padding_mask_required_or_used": False,
            "edge_support_flag_sample_ranges": [
                [0, 768],
                [target_count - 768, target_count],
            ],
            "payload_receipt": _payload_receipt(
                output, semantic=payload_semantic
            ),
        },
        "scope_receipt": {
            "EEG_samples_used": True,
            "acquisition_clock_used": True,
            "EEG_electrical_reference_provenance_used_as_control_plane": True,
            "EEG_electrical_reference_provenance_used_as_model_feature": False,
            "seizure_target_or_reference_label_used": False,
            "EDF_annotation_used": False,
            "spreadsheet_or_doctor_text_used": False,
            "clinical_history_used": False,
            "auxiliary_non_EEG_channel_used": False,
        },
        "runtime_versions": runtime_receipt,
        "receipt_sha256": _CONTENT_PENDING,
    }
    if short_context_ledger is not None:
        receipt["resample"][
            "observed_target_sample_count_floor_policy"
        ] = observed_target_count
        receipt["short_record_context"] = short_context_ledger
    receipt["receipt_sha256"] = _transform_receipt_sha256(receipt)
    return SeizureTransformerTransformResult(
        signal=output,
        receipt=receipt,
        _validation_seal=_TRANSFORM_RESULT_SEAL,
    )


def apply_full_record_transform(
    referential_volts: object,
    *,
    variant_id: str,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> SeizureTransformerTransformResult:
    """Execute the native transform or the fixed ST16 short-record challenger."""

    return _apply_full_record_transform_impl(
        referential_volts,
        variant_id=variant_id,
        signal_lineage_authority=signal_lineage_authority,
        registry=registry,
        short_context_mode="reflect",
        sensitivity_only=False,
    )


def apply_short_record_context_sensitivity_transform(
    referential_volts: object,
    *,
    context_mode: str,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> SeizureTransformerTransformResult:
    """Audit-only short ST16 counterfactual; never formal training authority."""

    return _apply_full_record_transform_impl(
        referential_volts,
        variant_id=ST16_VARIANT_ID,
        signal_lineage_authority=signal_lineage_authority,
        registry=registry,
        short_context_mode=context_mode,
        sensitivity_only=True,
    )


def _transform_observed_sample_count(
    result: SeizureTransformerTransformResult,
) -> int:
    if (
        not isinstance(result, SeizureTransformerTransformResult)
        or result._validation_seal is not _TRANSFORM_RESULT_SEAL
    ):
        raise TypeError("observed support requires an opaque ST transform")
    context = result.receipt.get("short_record_context")
    if context is None:
        return int(result.signal.shape[1])
    observed = context.get("provider_observed_sample_count")
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or not 0 < observed < result.signal.shape[1]
    ):
        raise ValueError("short ST transform observed support is malformed")
    return observed


def seizuretransformer_transform_valid_support_mask(
    result: SeizureTransformerTransformResult,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the loss/metric support; context samples always have zero weight."""

    observed = _transform_observed_sample_count(result)
    mask = np.zeros(result.signal.shape[1], dtype=np.uint8)
    mask[:observed] = 1
    mask.setflags(write=False)
    receipt = _content_address(
        {
            "schema_version": "st_transform_valid_support_mask_receipt_v1",
            "transform_receipt_sha256": result.receipt["receipt_sha256"],
            "model_context_sample_count": result.signal.shape[1],
            "observed_support_sample_count": observed,
            "context_sample_count": result.signal.shape[1] - observed,
            "mask_payload_receipt": _payload_receipt(
                mask,
                semantic="SeizureTransformer_transform_valid_loss_metric_support",
            ),
            "context_may_receive_target_loss_or_metric_weight": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return mask, receipt


def validate_transform_result(
    result: SeizureTransformerTransformResult,
    *,
    registry: Mapping[str, Any],
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
) -> SeizureTransformerTransformResult:
    """Validate payload identity against an external typed lineage authority."""

    if (
        not isinstance(result, SeizureTransformerTransformResult)
        or result._validation_seal is not _TRANSFORM_RESULT_SEAL
    ):
        raise TypeError(
            "result must be an opaque materialized SeizureTransformer transform"
        )
    validated_registry = validate_registry(dict(registry))
    lineage = require_validated_detector_signal_lineage_authority(
        signal_lineage_authority
    )
    if lineage["provider_transform_authorized"] is not True:
        raise PermissionError("transform result lineage is not provider-authorized")
    receipt = deepcopy(result.receipt)
    required = {
        "schema_version",
        "registry_id",
        "registry_sha256",
        "implementation_code_sha256",
        "provider_id",
        "variant_id",
        "profile_id",
        "input_clock_receipt_sha256",
        "detector_signal_lineage_authority_sha256",
        "canonical_signal_receipt_sha256",
        "canonical_source_header_receipt_sha256",
        "canonical_source_tensor_sha256",
        "observed_roster_authority_sha256",
        "EEG_electrical_reference_system_authority_sha256",
        "EEG_only_channel_QC_authority_sha256",
        "input_electrode_order",
        "input_sampling_rate_fraction_hz",
        "input_payload_receipt",
        "direct_variant_bipolar_derivation",
        "resample",
        "bandpass",
        "normalization",
        "output",
        "scope_receipt",
        "runtime_versions",
        "receipt_sha256",
    }
    short_context = (
        receipt.get("schema_version")
        == "seizuretransformer_short_context_transform_receipt_v1"
    )
    if short_context:
        required.add("short_record_context")
    receipt = _strict_dict(receipt, required, "transform receipt")
    if receipt["schema_version"] not in {
        "seizuretransformer_full_record_transform_receipt_v1",
        "seizuretransformer_short_context_transform_receipt_v1",
    }:
        raise ValueError("transform receipt schema drifted")
    if receipt["registry_id"] != validated_registry["registry_id"] or receipt[
        "registry_sha256"
    ] != validated_registry["registry_sha256"]:
        raise ValueError("transform result registry binding drifted")
    expected_lineage_bindings = {
        "input_clock_receipt_sha256": lineage[
            "common_sampling_clock_authority"
        ]["receipt_sha256"],
        "detector_signal_lineage_authority_sha256": lineage["receipt_sha256"],
        "canonical_signal_receipt_sha256": lineage["canonical_physical_signal"][
            "canonical_signal_receipt_sha256"
        ],
        "canonical_source_header_receipt_sha256": lineage[
            "canonical_physical_signal"
        ]["source_header_receipt_sha256"],
        "canonical_source_tensor_sha256": lineage["canonical_physical_signal"][
            "source_tensor_sha256"
        ],
        "observed_roster_authority_sha256": lineage[
            "observed_roster_authority"
        ]["receipt_sha256"],
        "EEG_electrical_reference_system_authority_sha256": lineage[
            "electrical_reference_system_authority"
        ]["receipt_sha256"],
        "EEG_only_channel_QC_authority_sha256": lineage[
            "EEG_only_channel_QC_authority"
        ]["receipt_sha256"],
    }
    if any(receipt[key] != value for key, value in expected_lineage_bindings.items()):
        raise ValueError("transform result external signal-lineage binding drifted")
    if receipt["input_electrode_order"] != lineage[
        "observed_roster_authority"
    ]["observed_standard_channel_ids"]:
        raise ValueError("transform result observed-roster binding drifted")
    if receipt["input_sampling_rate_fraction_hz"] != lineage[
        "common_sampling_clock_authority"
    ]["sampling_rate_fraction_hz"]:
        raise ValueError("transform result common-clock binding drifted")
    profile = _variant_profile(receipt["variant_id"])
    output = np.asarray(result.signal)
    if output.dtype != np.dtype("float32") or output.ndim != 2:
        raise ValueError("transform payload dtype or rank drifted")
    if output.shape[0] != profile["typed_unit_count"]:
        raise ValueError("transform payload typed-unit count drifted")
    if not np.isfinite(output).all() or np.any(output < -20) or np.any(output > 20):
        raise ValueError("transform payload violates finite clipped support")
    expected_payload = _payload_receipt(
        output,
        semantic=(
            "SeizureTransformer_short_record_fixed_context_carrier"
            if short_context
            else "SeizureTransformer_provider_native_full_record"
        ),
    )
    if receipt["output"]["payload_receipt"] != expected_payload:
        raise ValueError("transform output payload receipt drifted")
    if receipt["output"]["typed_units"] != profile["typed_units"]:
        raise ValueError("transform output typed-unit roster drifted")
    if receipt["direct_variant_bipolar_derivation"][
        "ST18_intermediate_used_for_ST16"
    ] is not False:
        raise ValueError("runtime ST18 slicing for ST16 is forbidden")
    if short_context:
        ledger = _validate_content_address(
            receipt["short_record_context"],
            required={
                "schema_version",
                "policy_id",
                "context_mode",
                "formal_context_policy",
                "sensitivity_only",
                "source_observed_sample_count",
                "source_context_sample_count",
                "source_right_context_sample_range",
                "provider_observed_sample_count",
                "provider_model_context_sample_count",
                "provider_valid_support_sample_range",
                "provider_right_context_sample_range",
                "reflection_without_endpoint_duplication",
                "context_is_observed_EEG",
                "context_may_receive_target_loss_or_metric_weight",
                "context_may_assert_Finding_or_clinical_fact",
                "normalization_and_model_boundary_predictions_may_depend_on_context",
                "observed_bipolar_payload_receipt",
                "extended_bipolar_payload_receipt",
                "valid_support_mask_payload_receipt",
                "receipt_sha256",
            },
            context="ST16 short-record context ledger",
        )
        observed_count = ledger["provider_observed_sample_count"]
        mask = np.zeros(TILE_SAMPLES, dtype=np.uint8)
        if (
            receipt["variant_id"] != ST16_VARIANT_ID
            or ledger["schema_version"] != "st16_short_record_context_ledger_v1"
            or ledger["policy_id"] != ST16_SHORT_CONTEXT_POLICY_ID
            or ledger["context_mode"] != "reflect"
            or ledger["formal_context_policy"] is not True
            or ledger["sensitivity_only"] is not False
            or isinstance(observed_count, bool)
            or not isinstance(observed_count, int)
            or not 0 < observed_count < TILE_SAMPLES
            or ledger["provider_model_context_sample_count"] != TILE_SAMPLES
            or ledger["provider_valid_support_sample_range"] != [0, observed_count]
            or ledger["provider_right_context_sample_range"]
            != [observed_count, TILE_SAMPLES]
            or ledger["reflection_without_endpoint_duplication"] is not True
            or ledger["context_is_observed_EEG"] is not False
            or ledger["context_may_receive_target_loss_or_metric_weight"] is not False
            or ledger["context_may_assert_Finding_or_clinical_fact"] is not False
            or ledger[
                "normalization_and_model_boundary_predictions_may_depend_on_context"
            ]
            is not True
            or output.shape[1] != TILE_SAMPLES
            or receipt["output"]["sample_count"] != TILE_SAMPLES
            or receipt["output"][
                "at_least_one_fully_observed_60_second_model_tile"
            ]
            is not False
            or receipt["resample"].get(
                "observed_target_sample_count_floor_policy"
            )
            != observed_count
        ):
            raise ValueError("ST16 short-record context contract drifted")
        mask[:observed_count] = 1
        if ledger["valid_support_mask_payload_receipt"] != _payload_receipt(
            mask, semantic="ST16_short_original_support_loss_metric_mask"
        ):
            raise ValueError("ST16 short-record valid-support mask drifted")
    elif (
        receipt["schema_version"]
        != "seizuretransformer_full_record_transform_receipt_v1"
        or "short_record_context" in receipt
    ):
        raise ValueError("native transform unexpectedly carries short context")
    for field in (
        "input_clock_receipt_sha256",
        "detector_signal_lineage_authority_sha256",
        "canonical_signal_receipt_sha256",
        "canonical_source_header_receipt_sha256",
        "canonical_source_tensor_sha256",
        "observed_roster_authority_sha256",
        "EEG_electrical_reference_system_authority_sha256",
        "EEG_only_channel_QC_authority_sha256",
    ):
        _require_sha256(receipt[field], field)
    if receipt["scope_receipt"] != {
        "EEG_samples_used": True,
        "acquisition_clock_used": True,
        "EEG_electrical_reference_provenance_used_as_control_plane": True,
        "EEG_electrical_reference_provenance_used_as_model_feature": False,
        "seizure_target_or_reference_label_used": False,
        "EDF_annotation_used": False,
        "spreadsheet_or_doctor_text_used": False,
        "clinical_history_used": False,
        "auxiliary_non_EEG_channel_used": False,
    }:
        raise ValueError("transform evidence firewall drifted")
    if receipt["receipt_sha256"] != _transform_receipt_sha256(receipt):
        raise ValueError("transform receipt is not content-addressed")
    validated_output = np.ascontiguousarray(output, dtype="<f4")
    validated_output.setflags(write=False)
    return SeizureTransformerTransformResult(
        signal=validated_output,
        receipt=receipt,
        _validation_seal=_TRANSFORM_RESULT_SEAL,
    )


def _require_canonical_seizuretransformer_registry(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the repository-owned registry and replay every static byte."""

    validated = validate_registry(dict(registry))
    if (
        validated["implementation"]["code_sha256"]
        != seizuretransformer_cleanroom_registry_code_sha256()
    ):
        raise ValueError("SeizureTransformer formal registry binding is stale")
    project_root = Path(__file__).resolve().parents[2]
    registry_path = project_root / CONFIG_RELATIVE_PATH
    if not registry_path.is_file() or registry_path.is_symlink():
        raise ValueError("canonical SeizureTransformer registry is unavailable")
    try:
        on_disk = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical SeizureTransformer registry is unreadable") from exc
    if on_disk != validated:
        raise PermissionError(
            "caller-owned SeizureTransformer registry is not formal authority"
        )
    validate_static_execution_bindings(project_root, registry=validated)
    return validated


def authorize_seizuretransformer_fold_phase(
    detector_phase_authority: ValidatedDetectorFoldReferencePhaseAuthorityV1,
    *,
    registry: Mapping[str, Any],
) -> AuthorizedSeizureTransformerFoldPhase:
    """Adapt only the shared process-sealed reference-phase authority."""

    st_registry = _require_canonical_seizuretransformer_registry(registry)
    shared = require_validated_detector_fold_reference_phase_authority_v1(
        detector_phase_authority
    )
    phase = shared.to_receipt()
    expected_reference_registry = st_registry["trainer"][
        "fold_reference_authority"
    ]["registry_receipt_sha256"]
    if phase.get("registry_receipt_sha256") != expected_reference_registry:
        raise ValueError("SeizureTransformer binds a different reference authority")
    project_root = Path(__file__).resolve().parents[2]
    plan_binding = st_registry["trainer"]["outer_fold_plan"]
    plan_path = project_root / plan_binding["path"]
    if (
        not plan_path.is_file()
        or plan_path.is_symlink()
        or _file_sha256(plan_path) != plan_binding["file_sha256"]
    ):
        raise ValueError("SeizureTransformer canonical fold-plan bytes drifted")
    try:
        fold_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("SeizureTransformer canonical fold plan is unreadable") from exc
    phase_plan_binding = phase.get("fold_plan_binding")
    if (
        type(phase_plan_binding) is not dict
        or phase_plan_binding.get("file_sha256") != plan_binding["file_sha256"]
        or phase_plan_binding.get("plan_receipt_sha256")
        != fold_plan.get("receipt_sha256")
    ):
        raise ValueError("SeizureTransformer phase/fold-plan binding drifted")
    plan_rows = fold_plan.get("source_record_duration_rows")
    if type(plan_rows) is not list:
        raise ValueError("SeizureTransformer fold plan lacks its denominator")
    by_identity: dict[str, Mapping[str, Any]] = {}
    for row in plan_rows:
        if type(row) is not dict or not isinstance(
            row.get("analysis_identity_id"), str
        ):
            raise ValueError("SeizureTransformer fold-plan row is malformed")
        identity = str(row["analysis_identity_id"])
        if identity in by_identity:
            raise ValueError("SeizureTransformer fold plan repeats an identity")
        by_identity[identity] = row
    patient_by_identity: dict[str, str] = {}
    for phase_row in phase["records"]:
        identity = str(phase_row["analysis_identity_id"])
        plan_row = by_identity.get(identity)
        if plan_row is None:
            raise ValueError("SeizureTransformer phase identity is outside fold plan")
        patient = plan_row.get("local_patient_id")
        if (
            not isinstance(patient, str)
            or not patient
            or plan_row.get("local_edf_path")
            != phase_row["source_edf_relative_path"]
            or plan_row.get("recording_duration_seconds_fraction")
            != phase_row["recording_duration_seconds_fraction"]
        ):
            raise ValueError("SeizureTransformer fold-owned record binding drifted")
        patient_by_identity[identity] = patient
    identities = sorted(patient_by_identity)
    if len(identities) != phase["authorized_roster"]["recording_count"]:
        raise ValueError("SeizureTransformer phase record denominator drifted")
    receipt = _content_address(
        {
            "schema_version": "st_shared_opaque_fold_phase_adapter_v1",
            "registry_sha256": st_registry["registry_sha256"],
            "fold_reference_registry_receipt_sha256": expected_reference_registry,
            "detector_fold_phase_receipt_sha256": phase["receipt_sha256"],
            "outer_fold": phase["outer_fold_id"],
            "phase": phase["phase"],
            "authorized_record_count": len(identities),
            "authorized_patient_count": len(set(patient_by_identity.values())),
            "analysis_identity_roster_sha256": _canonical_sha256(identities),
            "fold_owned_patient_mapping_sha256": _canonical_sha256(
                [
                    {
                        "analysis_identity_id": identity,
                        "patient_key": patient_by_identity[identity],
                    }
                    for identity in identities
                ]
            ),
            "reference_event_inventory_sha256": phase[
                "reference_event_inventory_sha256"
            ],
            "shared_detector_phase_authority_type": (
                "ValidatedDetectorFoldReferencePhaseAuthorityV1"
            ),
            "raw_mapping_or_bare_hash_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AuthorizedSeizureTransformerFoldPhase(
        _phase_receipt_json=_canonical_json_bytes(phase).decode("utf-8"),
        _patient_by_identity_json=_canonical_json_bytes(patient_by_identity).decode(
            "utf-8"
        ),
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_FOLD_PHASE_AUTHORITY_SEAL,
    )


def _require_authorized_seizuretransformer_fold_phase(
    value: object,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    if (
        not isinstance(value, AuthorizedSeizureTransformerFoldPhase)
        or value._validation_seal is not _FOLD_PHASE_AUTHORITY_SEAL
    ):
        raise TypeError(
            "formal SeizureTransformer training requires an opaque fold-phase authority"
        )
    try:
        phase = json.loads(value._phase_receipt_json)
        patient_by_identity = json.loads(value._patient_by_identity_json)
    except json.JSONDecodeError as exc:
        raise ValueError("opaque SeizureTransformer phase payload is unreadable") from exc
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_sha256",
            "fold_reference_registry_receipt_sha256",
            "detector_fold_phase_receipt_sha256",
            "outer_fold",
            "phase",
            "authorized_record_count",
            "authorized_patient_count",
            "analysis_identity_roster_sha256",
            "fold_owned_patient_mapping_sha256",
            "reference_event_inventory_sha256",
            "shared_detector_phase_authority_type",
            "raw_mapping_or_bare_hash_accepted",
            "receipt_sha256",
        },
        context="authorized SeizureTransformer fold phase",
    )
    if (
        receipt["schema_version"] != "st_shared_opaque_fold_phase_adapter_v1"
        or receipt["shared_detector_phase_authority_type"]
        != "ValidatedDetectorFoldReferencePhaseAuthorityV1"
        or receipt["raw_mapping_or_bare_hash_accepted"] is not False
        or phase.get("receipt_sha256")
        != receipt["detector_fold_phase_receipt_sha256"]
        or phase.get("outer_fold_id") != receipt["outer_fold"]
        or phase.get("phase") != receipt["phase"]
        or phase.get("reference_event_inventory_sha256")
        != receipt["reference_event_inventory_sha256"]
        or type(patient_by_identity) is not dict
    ):
        raise ValueError("opaque SeizureTransformer phase semantics drifted")
    identities = sorted(str(row["analysis_identity_id"]) for row in phase["records"])
    if (
        set(patient_by_identity) != set(identities)
        or any(
            not isinstance(patient, str) or not patient
            for patient in patient_by_identity.values()
        )
        or receipt["authorized_record_count"] != len(identities)
        or receipt["authorized_patient_count"]
        != len(set(patient_by_identity.values()))
        or receipt["analysis_identity_roster_sha256"]
        != _canonical_sha256(identities)
        or receipt["fold_owned_patient_mapping_sha256"]
        != _canonical_sha256(
            [
                {
                    "analysis_identity_id": identity,
                    "patient_key": patient_by_identity[identity],
                }
                for identity in identities
            ]
        )
    ):
        raise ValueError("opaque SeizureTransformer fold-owned roster drifted")
    return phase, patient_by_identity, receipt


def _st_bind_provider_and_identity_authorities(
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    record_identity_authority: ValidatedDetectorSignalLineageAuthority,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    provider = require_validated_detector_signal_lineage_authority(
        signal_lineage_authority
    )
    identity = require_validated_detector_signal_lineage_authority(
        record_identity_authority
    )
    provider_signal = provider["canonical_physical_signal"]
    identity_signal = identity["canonical_physical_signal"]
    analysis_identity_id = identity_signal.get("analysis_identity_id")
    if (
        provider.get("authority_tier")
        != "provider_transform_payload_replayed"
        or identity.get("authority_tier")
        != "canonical_audit_policy_route_only"
        or identity.get("provider_transform_authorized") is not False
        or not isinstance(analysis_identity_id, str)
        or not analysis_identity_id
        or provider_signal.get("source_tensor_sha256")
        != identity_signal.get("source_tensor_sha256")
        or provider_signal.get("source_header_receipt_sha256")
        != identity_signal.get("source_header_receipt_sha256")
        or provider_signal.get("source_signal_sha256")
        != identity_signal.get("source_signal_sha256")
    ):
        raise PermissionError(
            "SeizureTransformer record identity is not externally bound to "
            "the exact provider EEG payload"
        )
    return provider, identity, analysis_identity_id


def _st_pre_reference_technical_policy(variant_id: str) -> dict[str, Any]:
    profile = _variant_profile(variant_id)
    return {
        "schema_version": "st_pre_reference_technical_eligibility_policy_v1",
        "variant_id": variant_id,
        "required_source_electrodes": profile["required_source_electrodes"],
        "accepted_support_profiles": (
            ["complete19"]
            if variant_id == ST18_VARIANT_ID
            else ["complete19", "lateral17"]
        ),
        "source_sampling_rate_hz_closed_interval": [128, 2048],
        "maximum_reduced_polyphase_factor": 4096,
        "minimum_provider_target_samples": (
            TILE_SAMPLES if variant_id == ST18_VARIANT_ID else 1
        ),
        "minimum_fully_observed_training_tile_count": 1,
        "ST16_short_context_policy_id": (
            ST16_SHORT_CONTEXT_POLICY_ID
            if variant_id == ST16_VARIANT_ID
            else None
        ),
        "ST16_short_context_requires_at_least_two_source_samples": True,
        "short_context_may_receive_target_loss_or_metric_weight": False,
        "provider_transform_payload_replay_required_for_eligible_status": True,
        "target_reference_annotation_or_clinical_input_allowed": False,
    }


def enumerate_seizuretransformer_training_tiles(
    sample_count: int,
) -> tuple[tuple[int, int], ...]:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("SeizureTransformer sample count must be nonnegative")
    if sample_count < TILE_SAMPLES:
        return ()
    return tuple(
        (start, TILE_SAMPLES)
        for start in range(0, sample_count - TILE_SAMPLES + 1, TRAIN_HOP_SAMPLES)
    )


def materialize_seizuretransformer_pre_reference_eligibility(
    referential_volts: object,
    *,
    variant_id: str,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    record_identity_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> SeizureTransformerPreReferenceEligibilityOutcome:
    """Replay support, clock, exact EEG and transform without target access."""

    st_registry = _require_canonical_seizuretransformer_registry(registry)
    provider, identity, analysis_identity_id = _st_bind_provider_and_identity_authorities(
        signal_lineage_authority, record_identity_authority
    )
    profile = _variant_profile(variant_id)
    route = route_detector_channel_support(
        signal_lineage_authority=signal_lineage_authority
    )
    policy = _st_pre_reference_technical_policy(variant_id)
    support_policy_sha256 = detector_channel_support_policy_receipt()[
        "policy_sha256"
    ]
    if route["policy_sha256"] != support_policy_sha256:
        raise ValueError("SeizureTransformer support-route policy drifted")
    reason_codes: list[str] = []
    if (
        route["support_policy_status"] != "policy_route_available"
        or route["profile_id"] not in set(policy["accepted_support_profiles"])
    ):
        reason_codes.append("variant_support_route_excluded")
    usable = set(route["usable_standard_channel_ids"])
    if not set(profile["required_source_electrodes"]).issubset(usable):
        reason_codes.append("variant_EEG_QC_usable_support_incomplete")
    if provider["provider_transform_authorized"] is not True:
        reason_codes.append("provider_transform_lineage_not_authorized")
    clock = provider["common_sampling_clock_authority"]
    try:
        source_rate = Fraction(*clock["sampling_rate_fraction_hz"])
        source_sample_count = int(clock["sample_count"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("SeizureTransformer typed source clock is malformed") from exc
    if source_rate < 128 or source_rate > 2048:
        reason_codes.append("source_sampling_rate_outside_frozen_support")
        up = down = 0
        target_sample_count = 0
    else:
        ratio = Fraction(TARGET_FS_HZ, 1) / source_rate
        up, down = ratio.numerator, ratio.denominator
        if max(up, down) > 4096:
            reason_codes.append("reduced_polyphase_ratio_outside_frozen_support")
        target_sample_count = (source_sample_count * up) // down
    native_tile_count = len(
        enumerate_seizuretransformer_training_tiles(target_sample_count)
    )
    short_context_tile_count = int(
        variant_id == ST16_VARIANT_ID
        and source_sample_count >= 2
        and 0 < target_sample_count < TILE_SAMPLES
    )
    admitted_tile_count = native_tile_count + short_context_tile_count
    if admitted_tile_count < 1:
        reason_codes.append("no_fully_observed_60_second_training_tile")
    transform: SeizureTransformerTransformResult | None = None
    if not reason_codes:
        verify_provider_referential_payload(
            signal_lineage_authority, referential_volts
        )
        try:
            transform = apply_full_record_transform(
                referential_volts,
                variant_id=variant_id,
                signal_lineage_authority=signal_lineage_authority,
                registry=st_registry,
            )
            transform = validate_transform_result(
                transform,
                registry=st_registry,
                signal_lineage_authority=signal_lineage_authority,
            )
        except (FloatingPointError, OverflowError, RuntimeError, ValueError):
            reason_codes.append("provider_transform_deterministic_technical_failure")
            transform = None
    eligible = not reason_codes and transform is not None
    if transform is not None and (
        transform.receipt["variant_id"] != variant_id
        or _transform_observed_sample_count(transform) != target_sample_count
        or transform.receipt["output"]["sample_count"]
        != (TILE_SAMPLES if short_context_tile_count else target_sample_count)
    ):
        raise ValueError("SeizureTransformer transform/clock binding drifted")
    receipt = _content_address(
        {
            "schema_version": "st_pre_reference_record_eligibility_outcome_v1",
            "registry_sha256": st_registry["registry_sha256"],
            "variant_id": variant_id,
            "analysis_identity_id": analysis_identity_id,
            "provider_signal_lineage_authority_sha256": provider["receipt_sha256"],
            "record_identity_authority_sha256": identity["receipt_sha256"],
            "canonical_source_tensor_sha256": provider["canonical_physical_signal"][
                "source_tensor_sha256"
            ],
            "support_route_policy_sha256": support_policy_sha256,
            "support_route_receipt_sha256": route["route_sha256"],
            "support_profile_id": route["profile_id"],
            "technical_eligibility_policy_sha256": _canonical_sha256(policy),
            "source_sampling_rate_fraction_hz": [
                source_rate.numerator,
                source_rate.denominator,
            ],
            "source_sample_count": source_sample_count,
            "provider_target_sample_count": target_sample_count,
            "fully_observed_training_tile_count": native_tile_count,
            "short_context_training_tile_count": short_context_tile_count,
            "admitted_training_tile_count": admitted_tile_count,
            "status": "eligible" if eligible else "typed_exclusion",
            "reason_codes": reason_codes,
            "transform_receipt_sha256": (
                None if transform is None else transform.receipt["receipt_sha256"]
            ),
            "phase_reference_event_annotation_or_clinical_input_consumed": False,
            "must_be_frozen_before_corresponding_reference_phase_open": True,
            "raw_caller_status_or_reason_code_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return SeizureTransformerPreReferenceEligibilityOutcome(
        transform_result=transform,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_PRE_REFERENCE_ELIGIBILITY_SEAL,
    )


def _require_st_pre_reference_eligibility(
    value: object,
) -> tuple[SeizureTransformerTransformResult | None, dict[str, Any]]:
    if (
        not isinstance(value, SeizureTransformerPreReferenceEligibilityOutcome)
        or value._validation_seal is not _PRE_REFERENCE_ELIGIBILITY_SEAL
    ):
        raise TypeError(
            "SeizureTransformer variant roster requires opaque pre-reference outcomes"
        )
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_sha256",
            "variant_id",
            "analysis_identity_id",
            "provider_signal_lineage_authority_sha256",
            "record_identity_authority_sha256",
            "canonical_source_tensor_sha256",
            "support_route_policy_sha256",
            "support_route_receipt_sha256",
            "support_profile_id",
            "technical_eligibility_policy_sha256",
            "source_sampling_rate_fraction_hz",
            "source_sample_count",
            "provider_target_sample_count",
            "fully_observed_training_tile_count",
            "short_context_training_tile_count",
            "admitted_training_tile_count",
            "status",
            "reason_codes",
            "transform_receipt_sha256",
            "phase_reference_event_annotation_or_clinical_input_consumed",
            "must_be_frozen_before_corresponding_reference_phase_open",
            "raw_caller_status_or_reason_code_accepted",
            "receipt_sha256",
        },
        context="SeizureTransformer pre-reference eligibility outcome",
    )
    transform = value.transform_result
    if (
        receipt["schema_version"]
        != "st_pre_reference_record_eligibility_outcome_v1"
        or receipt["status"] not in {"eligible", "typed_exclusion"}
        or type(receipt["reason_codes"]) is not list
        or receipt["phase_reference_event_annotation_or_clinical_input_consumed"]
        is not False
        or receipt["must_be_frozen_before_corresponding_reference_phase_open"]
        is not True
        or receipt["raw_caller_status_or_reason_code_accepted"] is not False
        or receipt["short_context_training_tile_count"] not in {0, 1}
        or receipt["admitted_training_tile_count"]
        != receipt["fully_observed_training_tile_count"]
        + receipt["short_context_training_tile_count"]
        or (receipt["status"] == "eligible") is not (transform is not None)
        or (receipt["status"] == "eligible") is not (receipt["reason_codes"] == [])
        or (receipt["status"] == "typed_exclusion")
        is not bool(receipt["reason_codes"])
        or receipt["transform_receipt_sha256"]
        != (None if transform is None else transform.receipt.get("receipt_sha256"))
    ):
        raise ValueError("SeizureTransformer pre-reference outcome drifted")
    if transform is not None:
        output = np.asarray(transform.signal)
        if (
            transform._validation_seal is not _TRANSFORM_RESULT_SEAL
            or transform.receipt.get("variant_id") != receipt["variant_id"]
            or transform.receipt.get("registry_sha256") != receipt["registry_sha256"]
            or transform.receipt.get("detector_signal_lineage_authority_sha256")
            != receipt["provider_signal_lineage_authority_sha256"]
            or transform.receipt.get("canonical_source_tensor_sha256")
            != receipt["canonical_source_tensor_sha256"]
            or _transform_observed_sample_count(transform)
            != receipt["provider_target_sample_count"]
            or transform.receipt.get("output", {}).get("payload_receipt")
            != _payload_receipt(
                output,
                semantic=(
                    "SeizureTransformer_short_record_fixed_context_carrier"
                    if receipt["short_context_training_tile_count"]
                    else "SeizureTransformer_provider_native_full_record"
                ),
            )
        ):
            raise ValueError("SeizureTransformer embedded transform drifted")
    return transform, receipt


def authorize_seizuretransformer_variant_training_roster(
    phase_authority: AuthorizedSeizureTransformerFoldPhase,
    pre_reference_outcomes: Sequence[
        SeizureTransformerPreReferenceEligibilityOutcome
    ],
    *,
    variant_id: str,
    registry: Mapping[str, Any],
) -> AuthorizedSeizureTransformerVariantTrainingRoster:
    """Build the exact phase ∩ support-route ∩ technical roster."""

    st_registry = _require_canonical_seizuretransformer_registry(registry)
    phase, patient_by_identity, phase_receipt = (
        _require_authorized_seizuretransformer_fold_phase(phase_authority)
    )
    _variant_profile(variant_id)
    outcomes: dict[str, dict[str, Any]] = {}
    for value in pre_reference_outcomes:
        _transform, receipt = _require_st_pre_reference_eligibility(value)
        identity = str(receipt["analysis_identity_id"])
        if identity in outcomes:
            raise ValueError("SeizureTransformer pre-reference roster repeats identity")
        if (
            receipt["registry_sha256"] != st_registry["registry_sha256"]
            or receipt["variant_id"] != variant_id
        ):
            raise ValueError("SeizureTransformer eligibility method binding drifted")
        outcomes[identity] = receipt
    expected = {str(row["analysis_identity_id"]) for row in phase["records"]}
    if set(outcomes) != expected:
        missing = sorted(expected.difference(outcomes))
        extra = sorted(set(outcomes).difference(expected))
        raise PermissionError(
            "SeizureTransformer complete phase-by-variant pre-reference Cartesian "
            f"set was not supplied; missing={missing}, extra={extra}"
        )
    support_policy_sha256 = detector_channel_support_policy_receipt()[
        "policy_sha256"
    ]
    technical_policy_sha256 = _canonical_sha256(
        _st_pre_reference_technical_policy(variant_id)
    )
    eligible_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for identity in sorted(expected):
        outcome = outcomes[identity]
        if (
            outcome["support_route_policy_sha256"] != support_policy_sha256
            or outcome["technical_eligibility_policy_sha256"]
            != technical_policy_sha256
        ):
            raise ValueError("SeizureTransformer eligibility policy drifted")
        common = {
            "analysis_identity_id": identity,
            "fold_owned_patient_key": patient_by_identity[identity],
            "pre_reference_eligibility_receipt_sha256": outcome["receipt_sha256"],
            "support_route_receipt_sha256": outcome[
                "support_route_receipt_sha256"
            ],
            "technical_eligibility_receipt_sha256": outcome["receipt_sha256"],
        }
        if outcome["status"] == "eligible":
            eligible_rows.append(
                {
                    **common,
                    "provider_signal_lineage_authority_sha256": outcome[
                        "provider_signal_lineage_authority_sha256"
                    ],
                    "record_identity_authority_sha256": outcome[
                        "record_identity_authority_sha256"
                    ],
                    "transform_receipt_sha256": outcome[
                        "transform_receipt_sha256"
                    ],
                    "provider_target_sample_count": outcome[
                        "provider_target_sample_count"
                    ],
                }
            )
        else:
            exclusion_rows.append(
                {
                    **common,
                    "terminal_status": "technical_or_support_exclusion",
                    "reason_codes": outcome["reason_codes"],
                    "retained_in_full_prediction_first_benchmark_denominator": True,
                }
            )
    roster = {"eligible_records": eligible_rows, "typed_exclusions": exclusion_rows}
    receipt = _content_address(
        {
            "schema_version": "st_target_blind_variant_training_roster_authority_v1",
            "registry_sha256": st_registry["registry_sha256"],
            "variant_id": variant_id,
            "outer_fold": phase_receipt["outer_fold"],
            "phase": phase_receipt["phase"],
            "detector_fold_phase_receipt_sha256": phase_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "support_route_policy_sha256": support_policy_sha256,
            "pre_reference_technical_eligibility_policy_sha256": technical_policy_sha256,
            "phase_record_count": len(expected),
            "eligible_record_count": len(eligible_rows),
            "eligible_patient_count": len(
                {row["fold_owned_patient_key"] for row in eligible_rows}
            ),
            "excluded_record_count": len(exclusion_rows),
            "eligible_analysis_identity_roster_sha256": _canonical_sha256(
                sorted(row["analysis_identity_id"] for row in eligible_rows)
            ),
            "typed_exclusion_ledger_sha256": _canonical_sha256(exclusion_rows),
            "all_phase_records_accounted_for": True,
            "prediction_first_denominator_preserved": True,
            "phase_reference_events_used_for_route_or_eligibility": False,
            "caller_owned_subset_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AuthorizedSeizureTransformerVariantTrainingRoster(
        _roster_json=_canonical_json_bytes(roster).decode("utf-8"),
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_VARIANT_TRAINING_ROSTER_SEAL,
    )


def _require_authorized_st_variant_training_roster(
    value: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(value, AuthorizedSeizureTransformerVariantTrainingRoster)
        or value._validation_seal is not _VARIANT_TRAINING_ROSTER_SEAL
    ):
        raise TypeError(
            "formal SeizureTransformer training requires an opaque variant roster"
        )
    try:
        roster = json.loads(value._roster_json)
    except json.JSONDecodeError as exc:
        raise ValueError("opaque SeizureTransformer roster is unreadable") from exc
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_sha256",
            "variant_id",
            "outer_fold",
            "phase",
            "detector_fold_phase_receipt_sha256",
            "support_route_policy_sha256",
            "pre_reference_technical_eligibility_policy_sha256",
            "phase_record_count",
            "eligible_record_count",
            "eligible_patient_count",
            "excluded_record_count",
            "eligible_analysis_identity_roster_sha256",
            "typed_exclusion_ledger_sha256",
            "all_phase_records_accounted_for",
            "prediction_first_denominator_preserved",
            "phase_reference_events_used_for_route_or_eligibility",
            "caller_owned_subset_accepted",
            "receipt_sha256",
        },
        context="authorized SeizureTransformer variant roster",
    )
    if (
        receipt["schema_version"]
        != "st_target_blind_variant_training_roster_authority_v1"
        or receipt["all_phase_records_accounted_for"] is not True
        or receipt["prediction_first_denominator_preserved"] is not True
        or receipt["phase_reference_events_used_for_route_or_eligibility"]
        is not False
        or receipt["caller_owned_subset_accepted"] is not False
        or type(roster) is not dict
        or set(roster) != {"eligible_records", "typed_exclusions"}
        or type(roster["eligible_records"]) is not list
        or type(roster["typed_exclusions"]) is not list
    ):
        raise ValueError("opaque SeizureTransformer variant roster drifted")
    eligible_fields = {
        "analysis_identity_id",
        "fold_owned_patient_key",
        "pre_reference_eligibility_receipt_sha256",
        "support_route_receipt_sha256",
        "technical_eligibility_receipt_sha256",
        "provider_signal_lineage_authority_sha256",
        "record_identity_authority_sha256",
        "transform_receipt_sha256",
        "provider_target_sample_count",
    }
    exclusion_fields = {
        "analysis_identity_id",
        "fold_owned_patient_key",
        "pre_reference_eligibility_receipt_sha256",
        "support_route_receipt_sha256",
        "technical_eligibility_receipt_sha256",
        "terminal_status",
        "reason_codes",
        "retained_in_full_prediction_first_benchmark_denominator",
    }
    if any(
        type(row) is not dict or set(row) != eligible_fields
        for row in roster["eligible_records"]
    ) or any(
        type(row) is not dict
        or set(row) != exclusion_fields
        or row["terminal_status"] != "technical_or_support_exclusion"
        or type(row["reason_codes"]) is not list
        or not row["reason_codes"]
        or row["retained_in_full_prediction_first_benchmark_denominator"]
        is not True
        for row in roster["typed_exclusions"]
    ):
        raise ValueError("opaque SeizureTransformer roster row drifted")
    eligible_ids = [row["analysis_identity_id"] for row in roster["eligible_records"]]
    excluded_ids = [row["analysis_identity_id"] for row in roster["typed_exclusions"]]
    patients = [row["fold_owned_patient_key"] for row in roster["eligible_records"]]
    if (
        len(set(eligible_ids)) != len(eligible_ids)
        or len(set(excluded_ids)) != len(excluded_ids)
        or set(eligible_ids).intersection(excluded_ids)
        or receipt["eligible_record_count"] != len(eligible_ids)
        or receipt["eligible_patient_count"] != len(set(patients))
        or receipt["excluded_record_count"] != len(excluded_ids)
        or receipt["phase_record_count"] != len(eligible_ids) + len(excluded_ids)
        or receipt["eligible_analysis_identity_roster_sha256"]
        != _canonical_sha256(sorted(eligible_ids))
        or receipt["typed_exclusion_ledger_sha256"]
        != _canonical_sha256(roster["typed_exclusions"])
    ):
        raise ValueError("opaque SeizureTransformer roster denominator drifted")
    return roster, receipt


def fit_patient_equal_class_weights_pure_primitive(
    counts_by_patient: Mapping[str, Sequence[int]],
    *,
    fit_roster_sha256: str,
) -> dict[str, Any]:
    """Fit the frozen positive BCE weight from unique absolute samples.

    ``counts_by_patient[p]`` is ``(positive_count, negative_count)`` after all
    non-observed samples have been excluded.  Each patient contributes total
    mass one, independent of recording duration.
    """

    roster_hash = _require_sha256(fit_roster_sha256, "fit_roster_sha256")
    if not counts_by_patient:
        raise ValueError("class-weight fit needs at least one patient")
    positive_mass = 0.0
    negative_mass = 0.0
    canonical_rows: list[dict[str, Any]] = []
    for patient in sorted(counts_by_patient):
        if not isinstance(patient, str) or not patient:
            raise ValueError("patient grouping key must be a non-empty string")
        counts = counts_by_patient[patient]
        if len(counts) != 2:
            raise ValueError("patient count row must contain positive and negative")
        positive, negative = counts
        if (
            isinstance(positive, bool)
            or not isinstance(positive, int)
            or positive < 0
            or isinstance(negative, bool)
            or not isinstance(negative, int)
            or negative < 0
            or positive + negative <= 0
        ):
            raise ValueError("patient sample counts are invalid")
        total = positive + negative
        positive_mass += positive / total
        negative_mass += negative / total
        canonical_rows.append(
            {"patient_key": patient, "positive": positive, "negative": negative}
        )
    patient_count = len(canonical_rows)
    positive_mass /= patient_count
    negative_mass /= patient_count
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        raise ValueError("both target classes must exist in the fit roster")
    uncapped = negative_mass / positive_mass
    positive_weight = min(50.0, uncapped)
    receipt: dict[str, Any] = {
        "schema_version": "st_fold_train_patient_equal_class_weight_v1",
        "fit_roster_sha256": roster_hash,
        "patient_count": patient_count,
        "count_rows_sha256": _canonical_sha256(canonical_rows),
        "patient_equal_positive_mass": positive_mass,
        "patient_equal_negative_mass": negative_mass,
        "negative_weight": 1.0,
        "positive_weight_uncapped": uncapped,
        "positive_weight_cap": 50.0,
        "positive_weight": positive_weight,
        "fit_from_unique_absolute_samples": True,
        "receipt_sha256": _CONTENT_PENDING,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def masked_patient_macro_bce_pure_primitive(
    probabilities: object,
    targets: object,
    observed_mask: object,
    patient_keys: Sequence[str],
    *,
    positive_weight: float,
) -> float:
    """CPU replay of the frozen masked patient-macro dense BCE reduction."""

    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(targets)
    mask = np.asarray(observed_mask)
    if probability.ndim != 2 or target.shape != probability.shape or mask.shape != probability.shape:
        raise ValueError("probability, target and mask must share shape [tiles,samples]")
    if len(patient_keys) != probability.shape[0]:
        raise ValueError("patient key count disagrees with tile count")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("probabilities must be finite in [0,1]")
    if not np.all(np.isin(target, [0, 1])) or not np.all(np.isin(mask, [0, 1])):
        raise ValueError("targets and observed mask must be binary")
    if not math.isfinite(positive_weight) or positive_weight <= 0 or positive_weight > 50:
        raise ValueError("positive_weight violates the frozen (0,50] support")
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    per_sample = -(
        positive_weight * target * np.log(clipped)
        + (1 - target) * np.log1p(-clipped)
    )
    numerator: dict[str, float] = {}
    denominator: dict[str, float] = {}
    for row, patient in enumerate(patient_keys):
        if not isinstance(patient, str) or not patient:
            raise ValueError("patient keys must be non-empty strings")
        valid = mask[row].astype(bool)
        weights = np.where(target[row] == 1, positive_weight, 1.0)
        row_denominator = float(np.sum(weights[valid], dtype=np.float64))
        if row_denominator <= 0:
            raise ValueError("each tile must contain at least one observed sample")
        numerator[patient] = numerator.get(patient, 0.0) + float(
            np.sum(per_sample[row, valid], dtype=np.float64)
        )
        denominator[patient] = denominator.get(patient, 0.0) + row_denominator
    return float(
        np.mean(
            [numerator[patient] / denominator[patient] for patient in sorted(numerator)],
            dtype=np.float64,
        )
    )


def masked_dense_binary_metric_counts_pure_primitive(
    probabilities: object,
    targets: object,
    observed_mask: object,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Replay dense confusion counts with zero contribution from context."""

    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(targets)
    mask = np.asarray(observed_mask)
    if (
        probability.ndim != 2
        or target.shape != probability.shape
        or mask.shape != probability.shape
    ):
        raise ValueError("metric probability, target and mask must share [tiles,samples]")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("metric probabilities must be finite in [0,1]")
    if not np.all(np.isin(target, [0, 1])) or not np.all(np.isin(mask, [0, 1])):
        raise ValueError("metric targets and masks must be binary")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("metric threshold must be numeric")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("metric threshold must be finite in [0,1]")
    valid = mask.astype(bool)
    if not np.all(np.any(valid, axis=1)):
        raise ValueError("every metric tile requires observed support")
    predicted = probability >= threshold
    truth = target.astype(bool)
    tp = int(np.count_nonzero(predicted & truth & valid))
    tn = int(np.count_nonzero(~predicted & ~truth & valid))
    fp = int(np.count_nonzero(predicted & ~truth & valid))
    fn = int(np.count_nonzero(~predicted & truth & valid))
    return _content_address(
        {
            "schema_version": "st_masked_dense_binary_metric_counts_v1",
            "threshold": threshold,
            "observed_support_sample_count": int(np.count_nonzero(valid)),
            "zero_weight_context_sample_count": int(valid.size - np.count_nonzero(valid)),
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "context_contributed_to_metric": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def derive_training_seed(
    *, variant_id: str, outer_fold: int, stage: str, base_seed: int = 20260824
) -> int:
    """Derive one non-selectable deterministic 31-bit seed."""

    _variant_profile(variant_id)
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, int) or not 0 <= outer_fold < 5:
        raise ValueError("outer_fold must be one of 0..4")
    if stage not in {"selection", "final_refit"}:
        raise ValueError("stage must be selection or final_refit")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed <= 0:
        raise ValueError("base_seed must be a positive integer")
    payload = f"{base_seed}|{variant_id}|{outer_fold}|{stage}|v1".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1) + 1


def build_patient_balanced_epoch_plan_pure_primitive(
    tile_pools_by_patient: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    variant_id: str,
    outer_fold: int,
    stage: str,
    epoch_index: int,
) -> dict[str, Any]:
    """Build a deterministic 8-tiles/patient, distinct-patient batch plan.

    Each pool object has exact fields ``all`` and ``positive``.  Positive must
    be a subset of all.  One positive tile is guaranteed where available; the
    other seven draws cycle through the full patient pool.  Target data affect
    training sampling only and are never model inputs.
    """

    if isinstance(epoch_index, bool) or not isinstance(epoch_index, int) or epoch_index < 0:
        raise ValueError("epoch_index must be a nonnegative integer")
    seed = derive_training_seed(
        variant_id=variant_id, outer_fold=outer_fold, stage=stage
    )
    patient_draws: dict[str, list[str]] = {}
    pool_rows: list[dict[str, Any]] = []
    for patient in sorted(tile_pools_by_patient):
        pools = _strict_dict(
            dict(tile_pools_by_patient[patient]), {"all", "positive"}, "tile pools"
        )
        all_tiles = tuple(pools["all"])
        positive_tiles = tuple(pools["positive"])
        if not all_tiles or any(not isinstance(item, str) or not item for item in all_tiles):
            raise ValueError("each patient needs non-empty canonical tile IDs")
        if len(set(all_tiles)) != len(all_tiles) or len(set(positive_tiles)) != len(positive_tiles):
            raise ValueError("tile pools may not contain duplicates")
        if not set(positive_tiles).issubset(all_tiles):
            raise ValueError("positive tile pool must be a subset of all tiles")

        def ordered(values: Sequence[str], domain: str) -> list[str]:
            return sorted(
                values,
                key=lambda tile: hashlib.sha256(
                    f"{seed}|{patient}|{domain}|{tile}".encode("utf-8")
                ).digest(),
            )

        all_order = ordered(all_tiles, "all")
        selected: list[str] = []
        if positive_tiles:
            pos_order = ordered(positive_tiles, "positive")
            selected.append(pos_order[epoch_index % len(pos_order)])
        cursor = epoch_index * 7 if positive_tiles else epoch_index * 8
        needed = 8 - len(selected)
        offset = 0
        while len(selected) < 8:
            candidate = all_order[(cursor + offset) % len(all_order)]
            offset += 1
            if candidate in selected and len(all_order) >= needed + len(selected):
                continue
            selected.append(candidate)
        patient_draws[patient] = selected
        pool_rows.append(
            {
                "patient_key": patient,
                "all_tiles": list(all_tiles),
                "positive_tiles": list(positive_tiles),
            }
        )
    if not patient_draws:
        raise ValueError("epoch plan needs at least one patient")

    batches: list[list[dict[str, str]]] = []
    for draw_index in range(8):
        patient_order = sorted(
            patient_draws,
            key=lambda patient: hashlib.sha256(
                f"{seed}|{epoch_index}|{draw_index}|{patient}".encode("utf-8")
            ).digest(),
        )
        for start in range(0, len(patient_order), 16):
            patients = patient_order[start : start + 16]
            batches.append(
                [
                    {
                        "patient_key": patient,
                        "tile_id": patient_draws[patient][draw_index],
                    }
                    for patient in patients
                ]
            )
    body: dict[str, Any] = {
        "schema_version": "st_patient_balanced_epoch_plan_v1",
        "variant_id": variant_id,
        "outer_fold": outer_fold,
        "stage": stage,
        "epoch_index": epoch_index,
        "derived_seed": seed,
        "patient_count": len(patient_draws),
        "draws_per_patient": 8,
        "batch_count": len(batches),
        "pool_roster_sha256": _canonical_sha256(pool_rows),
        "batches": batches,
        "receipt_sha256": _CONTENT_PENDING,
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _st_phase_record(
    phase: Mapping[str, Any], *, analysis_identity_id: str
) -> dict[str, Any]:
    rows = [
        deepcopy(row)
        for row in phase["records"]
        if row["analysis_identity_id"] == analysis_identity_id
    ]
    if len(rows) != 1:
        raise PermissionError(
            "record is absent or duplicated in the authorized SeizureTransformer phase"
        )
    return rows[0]


def _st_authorized_record_context(
    phase_authority: AuthorizedSeizureTransformerFoldPhase,
    variant_roster_authority: AuthorizedSeizureTransformerVariantTrainingRoster,
    transform_result: SeizureTransformerTransformResult,
    *,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    record_identity_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
    dict[str, Any],
    SeizureTransformerTransformResult,
    dict[str, Any],
    dict[str, Any],
]:
    st_registry = _require_canonical_seizuretransformer_registry(registry)
    phase, patient_by_identity, phase_receipt = (
        _require_authorized_seizuretransformer_fold_phase(phase_authority)
    )
    roster, roster_receipt = _require_authorized_st_variant_training_roster(
        variant_roster_authority
    )
    if (
        phase_receipt["registry_sha256"] != st_registry["registry_sha256"]
        or roster_receipt["registry_sha256"] != phase_receipt["registry_sha256"]
        or roster_receipt["outer_fold"] != phase_receipt["outer_fold"]
        or roster_receipt["phase"] != phase_receipt["phase"]
        or roster_receipt["detector_fold_phase_receipt_sha256"]
        != phase_receipt["detector_fold_phase_receipt_sha256"]
    ):
        raise ValueError("SeizureTransformer phase/variant authority binding drifted")
    provider, identity_lineage, identity = _st_bind_provider_and_identity_authorities(
        signal_lineage_authority, record_identity_authority
    )
    phase_record = _st_phase_record(phase, analysis_identity_id=identity)
    patient_key = patient_by_identity[identity]
    transform = validate_transform_result(
        transform_result,
        registry=st_registry,
        signal_lineage_authority=signal_lineage_authority,
    )
    eligible = [
        row for row in roster["eligible_records"] if row["analysis_identity_id"] == identity
    ]
    if (
        len(eligible) != 1
        or eligible[0]["fold_owned_patient_key"] != patient_key
        or eligible[0]["provider_signal_lineage_authority_sha256"]
        != provider["receipt_sha256"]
        or eligible[0]["record_identity_authority_sha256"]
        != identity_lineage["receipt_sha256"]
        or eligible[0]["transform_receipt_sha256"]
        != transform.receipt["receipt_sha256"]
        or eligible[0]["provider_target_sample_count"]
        != transform.receipt["output"]["sample_count"]
        or roster_receipt["variant_id"] != transform.receipt["variant_id"]
    ):
        raise ValueError("SeizureTransformer eligible record/transform binding drifted")
    clock = provider["common_sampling_clock_authority"]
    source_duration = Fraction(
        clock["sample_count"] * clock["sampling_rate_fraction_hz"][1],
        clock["sampling_rate_fraction_hz"][0],
    )
    if [source_duration.numerator, source_duration.denominator] != phase_record[
        "recording_duration_seconds_fraction"
    ]:
        raise ValueError("SeizureTransformer signal and fold-owned clocks differ")
    return (
        phase,
        phase_record,
        patient_key,
        phase_receipt,
        roster_receipt,
        transform,
        provider,
        identity_lineage,
    )


def _st_event_sample_spans(
    phase_record: Mapping[str, Any], *, provider_sample_count: int
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Project exact half-open events to sample-center-positive spans."""

    if provider_sample_count < 0:
        raise ValueError("SeizureTransformer provider sample count is invalid")
    provider_duration = Fraction(provider_sample_count, TARGET_FS_HZ)
    source_duration = Fraction(*phase_record["recording_duration_seconds_fraction"])
    if provider_duration > source_duration:
        raise ValueError("SeizureTransformer provider clock exceeds source duration")
    raw_spans: list[tuple[int, int]] = []
    unobservable: list[int] = []
    right_clamped: list[int] = []
    for event_index, event in enumerate(phase_record["seizure_intervals"]):
        start = Fraction(str(event["start_seconds"]))
        stop = Fraction(str(event["stop_seconds"]))
        if start < 0 or stop <= start:
            raise ValueError("SeizureTransformer authorized event interval is invalid")
        if start >= provider_duration:
            unobservable.append(event_index)
            continue
        projected_stop = min(stop, provider_duration)
        if projected_stop < stop:
            right_clamped.append(event_index)
        # start <= (sample+1/2)/fs < stop
        start_sample = max(
            0, math.ceil(start * TARGET_FS_HZ - Fraction(1, 2))
        )
        stop_sample = min(
            provider_sample_count,
            math.ceil(projected_stop * TARGET_FS_HZ - Fraction(1, 2)),
        )
        if stop_sample <= start_sample:
            unobservable.append(event_index)
            continue
        raw_spans.append((start_sample, stop_sample))
    merged: list[list[int]] = []
    for start, stop in sorted(raw_spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    spans = [(start, stop) for start, stop in merged]
    ledger = {
        "source_event_count": len(phase_record["seizure_intervals"]),
        "provider_clock_nonempty_event_span_count": len(raw_spans),
        "merged_positive_sample_span_count": len(spans),
        "right_boundary_clamped_event_indices": right_clamped,
        "entirely_unobservable_or_zero_sample_event_indices": unobservable,
        "sample_center_rule": "start_le_(n_plus_half)_div_256_lt_stop",
        "positive_unique_sample_count": sum(stop - start for start, stop in spans),
    }
    return spans, ledger


def build_seizuretransformer_dense_target_pure_primitive(
    positive_sample_spans: Sequence[Sequence[int]],
    *,
    target_start_sample: int,
    valid_support_sample_count: int = TILE_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pure dense-target math; serialized spans are not training authority."""

    if (
        isinstance(target_start_sample, bool)
        or not isinstance(target_start_sample, int)
        or target_start_sample < 0
    ):
        raise ValueError("SeizureTransformer target start must be nonnegative")
    if (
        isinstance(valid_support_sample_count, bool)
        or not isinstance(valid_support_sample_count, int)
        or not 0 < valid_support_sample_count <= TILE_SAMPLES
    ):
        raise ValueError("valid support must lie in (0,TILE_SAMPLES]")
    target = np.zeros(TILE_SAMPLES, dtype=np.uint8)
    target_stop = target_start_sample + TILE_SAMPLES
    canonical_spans: list[list[int]] = []
    previous_stop = -1
    for raw in positive_sample_spans:
        if len(raw) != 2:
            raise ValueError("SeizureTransformer positive span must have two bounds")
        start, stop = raw
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or start < 0
            or stop <= start
            or start < previous_stop
        ):
            raise ValueError("SeizureTransformer positive spans are not canonical")
        previous_stop = stop
        canonical_spans.append([start, stop])
        left = max(start, target_start_sample)
        right = min(stop, target_stop)
        if right > left:
            target[left - target_start_sample : right - target_start_sample] = 1
    mask = np.zeros(TILE_SAMPLES, dtype=np.uint8)
    mask[:valid_support_sample_count] = 1
    target[valid_support_sample_count:] = 0
    target.setflags(write=False)
    mask.setflags(write=False)
    receipt_body: dict[str, Any] = {
            "schema_version": "st_dense_target_pure_primitive_receipt_v1",
            "target_start_sample": target_start_sample,
            "target_stop_sample_exclusive": target_stop,
            "positive_sample_span_roster_sha256": _canonical_sha256(
                canonical_spans
            ),
            "positive_target_sample_count": int(np.count_nonzero(target)),
            "target_payload_receipt": _payload_receipt(
                target, semantic="SeizureTransformer_dense_binary_target"
            ),
            "observed_mask_payload_receipt": _payload_receipt(
                mask,
                semantic=(
                    "SeizureTransformer_fully_observed_loss_mask"
                    if valid_support_sample_count == TILE_SAMPLES
                    else "SeizureTransformer_short_original_support_loss_metric_mask"
                ),
            ),
            "receipt_sha256": _CONTENT_PENDING,
    }
    if valid_support_sample_count < TILE_SAMPLES:
        receipt_body.update(
            {
                "schema_version": "st_short_context_dense_target_pure_primitive_receipt_v1",
                "valid_support_sample_count": valid_support_sample_count,
                "context_sample_count": TILE_SAMPLES - valid_support_sample_count,
                "context_target_value": 0,
                "context_loss_and_metric_weight": 0,
            }
        )
    receipt = _content_address(receipt_body)
    return target, mask, receipt


def authorize_seizuretransformer_target_bundle(
    phase_authority: AuthorizedSeizureTransformerFoldPhase,
    variant_roster_authority: AuthorizedSeizureTransformerVariantTrainingRoster,
    transform_result: SeizureTransformerTransformResult,
    *,
    target_start_sample: int,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    record_identity_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> AuthorizedSeizureTransformerTargetBundle:
    """Create one formal dense target from opaque phase+variant authorities."""

    (
        _phase,
        phase_record,
        patient_key,
        phase_receipt,
        roster_receipt,
        transform,
        _provider,
        identity_lineage,
    ) = _st_authorized_record_context(
        phase_authority,
        variant_roster_authority,
        transform_result,
        signal_lineage_authority=signal_lineage_authority,
        record_identity_authority=record_identity_authority,
        registry=registry,
    )
    allowed_starts = {
        start
        for start, _length in enumerate_seizuretransformer_training_tiles(
            transform.signal.shape[1]
        )
    }
    if target_start_sample not in allowed_starts:
        raise PermissionError(
            "SeizureTransformer target tile is padded, short, tail or off-grid"
        )
    valid_support_sample_count = _transform_observed_sample_count(transform)
    short_context = valid_support_sample_count < transform.signal.shape[1]
    spans, event_projection = _st_event_sample_spans(
        phase_record, provider_sample_count=valid_support_sample_count
    )
    target, mask, primitive_receipt = (
        build_seizuretransformer_dense_target_pure_primitive(
            spans,
            target_start_sample=target_start_sample,
            valid_support_sample_count=valid_support_sample_count,
        )
    )
    receipt_body: dict[str, Any] = {
            "schema_version": (
                "st_authorized_short_context_dense_target_bundle_v1"
                if short_context
                else "st_authorized_dense_target_bundle_v1"
            ),
            "registry_sha256": phase_receipt["registry_sha256"],
            "variant_id": transform.receipt["variant_id"],
            "outer_fold": phase_receipt["outer_fold"],
            "phase": phase_receipt["phase"],
            "detector_fold_phase_receipt_sha256": phase_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": roster_receipt[
                "receipt_sha256"
            ],
            "analysis_identity_id": phase_record["analysis_identity_id"],
            "record_identity_authority_sha256": identity_lineage["receipt_sha256"],
            "fold_owned_patient_key": patient_key,
            "patient_key_used_as_model_feature": False,
            "record_event_inventory_sha256": phase_record[
                "event_inventory_sha256"
            ],
            "event_projection_ledger": event_projection,
            "transform_receipt_sha256": transform.receipt["receipt_sha256"],
            "transform_output_payload_sha256": transform.receipt["output"][
                "payload_receipt"
            ]["payload_sha256"],
            "target_start_sample": target_start_sample,
            "target_stop_sample_exclusive": target_start_sample + TILE_SAMPLES,
            "fully_observed_unpadded_training_tile": not short_context,
            "primitive_target_receipt_sha256": primitive_receipt[
                "receipt_sha256"
            ],
            "target_payload_receipt": primitive_receipt["target_payload_receipt"],
            "observed_mask_payload_receipt": primitive_receipt[
                "observed_mask_payload_receipt"
            ],
            "positive_target_sample_count": primitive_receipt[
                "positive_target_sample_count"
            ],
            "raw_caller_events_masks_or_patient_keys_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
    }
    if short_context:
        receipt_body.update(
            {
                "short_context_policy_id": ST16_SHORT_CONTEXT_POLICY_ID,
                "valid_support_sample_count": valid_support_sample_count,
                "context_sample_count": TILE_SAMPLES - valid_support_sample_count,
                "context_target_loss_and_metric_weight": 0,
                "short_context_ledger_receipt_sha256": transform.receipt[
                    "short_record_context"
                ]["receipt_sha256"],
            }
        )
    receipt = _content_address(receipt_body)
    return AuthorizedSeizureTransformerTargetBundle(
        target=target,
        observed_mask=mask,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_TARGET_BUNDLE_SEAL,
    )


def _require_authorized_st_target_bundle(
    value: object,
) -> AuthorizedSeizureTransformerTargetBundle:
    if (
        not isinstance(value, AuthorizedSeizureTransformerTargetBundle)
        or value._validation_seal is not _TARGET_BUNDLE_SEAL
    ):
        raise TypeError(
            "formal SeizureTransformer loss requires an opaque target bundle"
        )
    raw_receipt = value.receipt
    short_context = (
        raw_receipt.get("schema_version")
        == "st_authorized_short_context_dense_target_bundle_v1"
    )
    required = {
            "schema_version",
            "registry_sha256",
            "variant_id",
            "outer_fold",
            "phase",
            "detector_fold_phase_receipt_sha256",
            "variant_training_roster_receipt_sha256",
            "analysis_identity_id",
            "record_identity_authority_sha256",
            "fold_owned_patient_key",
            "patient_key_used_as_model_feature",
            "record_event_inventory_sha256",
            "event_projection_ledger",
            "transform_receipt_sha256",
            "transform_output_payload_sha256",
            "target_start_sample",
            "target_stop_sample_exclusive",
            "fully_observed_unpadded_training_tile",
            "primitive_target_receipt_sha256",
            "target_payload_receipt",
            "observed_mask_payload_receipt",
            "positive_target_sample_count",
            "raw_caller_events_masks_or_patient_keys_accepted",
            "receipt_sha256",
    }
    if short_context:
        required.update(
            {
                "short_context_policy_id",
                "valid_support_sample_count",
                "context_sample_count",
                "context_target_loss_and_metric_weight",
                "short_context_ledger_receipt_sha256",
            }
        )
    receipt = _validate_content_address(
        raw_receipt,
        required=required,
        context="authorized SeizureTransformer target bundle",
    )
    target = np.asarray(value.target)
    mask = np.asarray(value.observed_mask)
    valid_support_sample_count = (
        int(receipt["valid_support_sample_count"])
        if short_context
        else TILE_SAMPLES
    )
    expected_mask = np.zeros(TILE_SAMPLES, dtype=np.uint8)
    if 0 < valid_support_sample_count <= TILE_SAMPLES:
        expected_mask[:valid_support_sample_count] = 1
    mask_semantic = (
        "SeizureTransformer_short_original_support_loss_metric_mask"
        if short_context
        else "SeizureTransformer_fully_observed_loss_mask"
    )
    if (
        receipt["schema_version"]
        not in {
            "st_authorized_dense_target_bundle_v1",
            "st_authorized_short_context_dense_target_bundle_v1",
        }
        or receipt["patient_key_used_as_model_feature"] is not False
        or receipt["fully_observed_unpadded_training_tile"] is not (not short_context)
        or receipt["raw_caller_events_masks_or_patient_keys_accepted"] is not False
        or target.dtype != np.dtype("uint8")
        or mask.dtype != np.dtype("uint8")
        or target.shape != (TILE_SAMPLES,)
        or mask.shape != target.shape
        or not np.all(np.isin(target, [0, 1]))
        or not np.array_equal(mask, expected_mask)
        or np.any(target[valid_support_sample_count:] != 0)
        or receipt["target_payload_receipt"]
        != _payload_receipt(
            target, semantic="SeizureTransformer_dense_binary_target"
        )
        or receipt["observed_mask_payload_receipt"]
        != _payload_receipt(
            mask, semantic=mask_semantic
        )
        or receipt["positive_target_sample_count"]
        != int(np.count_nonzero(target))
    ):
        raise ValueError("authorized SeizureTransformer target payload drifted")
    if short_context and (
        receipt["short_context_policy_id"] != ST16_SHORT_CONTEXT_POLICY_ID
        or receipt["context_sample_count"]
        != TILE_SAMPLES - valid_support_sample_count
        or receipt["context_target_loss_and_metric_weight"] != 0
        or not _is_sha256(receipt["short_context_ledger_receipt_sha256"])
    ):
        raise ValueError("authorized ST16 short target support drifted")
    return value


def authorize_seizuretransformer_record_tile_pool(
    phase_authority: AuthorizedSeizureTransformerFoldPhase,
    variant_roster_authority: AuthorizedSeizureTransformerVariantTrainingRoster,
    transform_result: SeizureTransformerTransformResult,
    *,
    signal_lineage_authority: ValidatedDetectorSignalLineageAuthority,
    record_identity_authority: ValidatedDetectorSignalLineageAuthority,
    registry: Mapping[str, Any],
) -> AuthorizedSeizureTransformerRecordPool:
    """Build one complete record pool without caller-owned event rows."""

    (
        _phase,
        phase_record,
        patient_key,
        phase_receipt,
        roster_receipt,
        transform,
        _provider,
        identity_lineage,
    ) = _st_authorized_record_context(
        phase_authority,
        variant_roster_authority,
        transform_result,
        signal_lineage_authority=signal_lineage_authority,
        record_identity_authority=record_identity_authority,
        registry=registry,
    )
    spans, event_projection = _st_event_sample_spans(
        phase_record,
        provider_sample_count=_transform_observed_sample_count(transform),
    )
    all_tiles: list[str] = []
    positive_tiles: list[str] = []
    tile_rows: list[dict[str, Any]] = []
    identity = str(phase_record["analysis_identity_id"])
    for start, length in enumerate_seizuretransformer_training_tiles(
        transform.signal.shape[1]
    ):
        stop = start + length
        tile_id = f"{identity}:{start}:{stop}"
        positive = any(event_start < stop and event_stop > start for event_start, event_stop in spans)
        all_tiles.append(tile_id)
        if positive:
            positive_tiles.append(tile_id)
        tile_rows.append(
            {
                "tile_id": tile_id,
                "start_sample": start,
                "stop_sample_exclusive": stop,
                "positive": positive,
            }
        )
    pool = _content_address(
        {
            "schema_version": "st_complete_record_tile_pool_payload_v1",
            "record_key": identity,
            "all": all_tiles,
            "positive": positive_tiles,
            "tile_rows": tile_rows,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    receipt = _content_address(
        {
            "schema_version": "st_authorized_record_tile_pool_v1",
            "registry_sha256": phase_receipt["registry_sha256"],
            "variant_id": transform.receipt["variant_id"],
            "outer_fold": phase_receipt["outer_fold"],
            "phase": phase_receipt["phase"],
            "detector_fold_phase_receipt_sha256": phase_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": roster_receipt[
                "receipt_sha256"
            ],
            "analysis_identity_id": identity,
            "record_identity_authority_sha256": identity_lineage["receipt_sha256"],
            "fold_owned_patient_key": patient_key,
            "record_event_inventory_sha256": phase_record[
                "event_inventory_sha256"
            ],
            "event_projection_ledger": event_projection,
            "transform_receipt_sha256": transform.receipt["receipt_sha256"],
            "pool_receipt_sha256": pool["receipt_sha256"],
            "eligible_tile_count": len(all_tiles),
            "positive_tile_count": len(positive_tiles),
            "raw_caller_events_or_patient_roster_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AuthorizedSeizureTransformerRecordPool(
        _pool_json=_canonical_json_bytes(pool).decode("utf-8"),
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_RECORD_POOL_SEAL,
    )


def _require_authorized_st_record_pool(
    value: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(value, AuthorizedSeizureTransformerRecordPool)
        or value._validation_seal is not _RECORD_POOL_SEAL
    ):
        raise TypeError(
            "formal SeizureTransformer epoch plan requires opaque record pools"
        )
    try:
        pool = json.loads(value._pool_json)
    except json.JSONDecodeError as exc:
        raise ValueError("opaque SeizureTransformer record pool is unreadable") from exc
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_sha256",
            "variant_id",
            "outer_fold",
            "phase",
            "detector_fold_phase_receipt_sha256",
            "variant_training_roster_receipt_sha256",
            "analysis_identity_id",
            "record_identity_authority_sha256",
            "fold_owned_patient_key",
            "record_event_inventory_sha256",
            "event_projection_ledger",
            "transform_receipt_sha256",
            "pool_receipt_sha256",
            "eligible_tile_count",
            "positive_tile_count",
            "raw_caller_events_or_patient_roster_accepted",
            "receipt_sha256",
        },
        context="authorized SeizureTransformer record pool",
    )
    supplied_pool = pool.get("receipt_sha256")
    pending = deepcopy(pool)
    pending["receipt_sha256"] = _CONTENT_PENDING
    if (
        receipt["schema_version"] != "st_authorized_record_tile_pool_v1"
        or receipt["raw_caller_events_or_patient_roster_accepted"] is not False
        or supplied_pool != _canonical_sha256(pending)
        or receipt["pool_receipt_sha256"] != supplied_pool
        or pool.get("record_key") != receipt["analysis_identity_id"]
        or receipt["eligible_tile_count"] != len(pool.get("all", []))
        or receipt["positive_tile_count"] != len(pool.get("positive", []))
        or not set(pool.get("positive", [])).issubset(pool.get("all", []))
    ):
        raise ValueError("authorized SeizureTransformer record pool drifted")
    return pool, receipt


def build_authorized_seizuretransformer_epoch_plan(
    phase_authority: AuthorizedSeizureTransformerFoldPhase,
    variant_roster_authority: AuthorizedSeizureTransformerVariantTrainingRoster,
    record_pools: Sequence[AuthorizedSeizureTransformerRecordPool],
    *,
    variant_id: str,
    outer_fold: int,
    stage: str,
    epoch_index: int,
) -> dict[str, Any]:
    """Build an epoch only from the complete variant-eligible denominator."""

    _phase, patients, phase_receipt = _require_authorized_seizuretransformer_fold_phase(
        phase_authority
    )
    roster, roster_receipt = _require_authorized_st_variant_training_roster(
        variant_roster_authority
    )
    expected_phase = {"selection": "selection_fit", "final_refit": "final_refit"}.get(
        stage
    )
    if (
        expected_phase is None
        or phase_receipt["outer_fold"] != outer_fold
        or phase_receipt["phase"] != expected_phase
        or roster_receipt["variant_id"] != variant_id
        or roster_receipt["outer_fold"] != outer_fold
        or roster_receipt["phase"] != expected_phase
        or roster_receipt["registry_sha256"] != phase_receipt["registry_sha256"]
        or roster_receipt["detector_fold_phase_receipt_sha256"]
        != phase_receipt["detector_fold_phase_receipt_sha256"]
    ):
        raise PermissionError("SeizureTransformer epoch stage lacks matching authorities")
    pools_by_identity: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for value in record_pools:
        pool, receipt = _require_authorized_st_record_pool(value)
        identity = str(receipt["analysis_identity_id"])
        if identity in pools_by_identity:
            raise ValueError("SeizureTransformer epoch repeats a record pool")
        if (
            receipt["registry_sha256"] != phase_receipt["registry_sha256"]
            or receipt["variant_id"] != variant_id
            or receipt["outer_fold"] != outer_fold
            or receipt["phase"] != expected_phase
            or receipt["detector_fold_phase_receipt_sha256"]
            != phase_receipt["detector_fold_phase_receipt_sha256"]
            or receipt["variant_training_roster_receipt_sha256"]
            != roster_receipt["receipt_sha256"]
            or receipt["fold_owned_patient_key"] != patients.get(identity)
        ):
            raise ValueError("SeizureTransformer record pool authority drifted")
        pools_by_identity[identity] = (pool, receipt)
    expected_identities = {
        str(row["analysis_identity_id"]) for row in roster["eligible_records"]
    }
    if set(pools_by_identity) != expected_identities:
        missing = sorted(expected_identities.difference(pools_by_identity))
        extra = sorted(set(pools_by_identity).difference(expected_identities))
        raise PermissionError(
            "SeizureTransformer variant-eligible record denominator was deleted; "
            f"missing={missing}, extra={extra}"
        )
    patient_pools: dict[str, dict[str, list[str]]] = {}
    globally_seen: set[str] = set()
    for identity in sorted(expected_identities):
        pool, receipt = pools_by_identity[identity]
        patient = str(receipt["fold_owned_patient_key"])
        aggregate = patient_pools.setdefault(patient, {"all": [], "positive": []})
        for tile in pool["all"]:
            if tile in globally_seen:
                raise ValueError("SeizureTransformer tile identity collides")
            globally_seen.add(tile)
            aggregate["all"].append(tile)
        aggregate["positive"].extend(pool["positive"])
    expected_patients = {
        str(row["fold_owned_patient_key"]) for row in roster["eligible_records"]
    }
    if set(patient_pools) != expected_patients or any(
        not pools["all"] for pools in patient_pools.values()
    ):
        raise PermissionError(
            "SeizureTransformer complete eligible patient denominator drifted"
        )
    primitive = build_patient_balanced_epoch_plan_pure_primitive(
        patient_pools,
        variant_id=variant_id,
        outer_fold=outer_fold,
        stage=stage,
        epoch_index=epoch_index,
    )
    return _content_address(
        {
            "schema_version": "st_authorized_patient_balanced_epoch_plan_v1",
            "variant_id": variant_id,
            "outer_fold": outer_fold,
            "stage": stage,
            "epoch_index": epoch_index,
            "detector_fold_phase_receipt_sha256": phase_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": roster_receipt[
                "receipt_sha256"
            ],
            "complete_variant_eligible_record_count": len(expected_identities),
            "complete_variant_eligible_patient_count": len(patient_pools),
            "typed_excluded_phase_record_count": len(roster["typed_exclusions"]),
            "prediction_first_denominator_preserved": True,
            "authorized_record_pool_receipt_roster_sha256": _canonical_sha256(
                [
                    pools_by_identity[identity][1]["receipt_sha256"]
                    for identity in sorted(expected_identities)
                ]
            ),
            "eligible_record_or_patient_deletion_allowed": False,
            "primitive_plan": primitive,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def authorize_seizuretransformer_class_weight(
    phase_authority: AuthorizedSeizureTransformerFoldPhase,
    variant_roster_authority: AuthorizedSeizureTransformerVariantTrainingRoster,
) -> AuthorizedSeizureTransformerClassWeight:
    """Fit patient-equal class weight from unique absolute record samples."""

    phase, patients, phase_receipt = _require_authorized_seizuretransformer_fold_phase(
        phase_authority
    )
    roster, roster_receipt = _require_authorized_st_variant_training_roster(
        variant_roster_authority
    )
    if (
        roster_receipt["registry_sha256"] != phase_receipt["registry_sha256"]
        or roster_receipt["outer_fold"] != phase_receipt["outer_fold"]
        or roster_receipt["phase"] != phase_receipt["phase"]
        or roster_receipt["detector_fold_phase_receipt_sha256"]
        != phase_receipt["detector_fold_phase_receipt_sha256"]
    ):
        raise ValueError("SeizureTransformer class-weight authority binding drifted")
    phase_by_identity = {
        str(row["analysis_identity_id"]): row for row in phase["records"]
    }
    counts: dict[str, list[int]] = {}
    record_rows: list[dict[str, Any]] = []
    for row in roster["eligible_records"]:
        identity = str(row["analysis_identity_id"])
        sample_count = int(row["provider_target_sample_count"])
        spans, ledger = _st_event_sample_spans(
            phase_by_identity[identity], provider_sample_count=sample_count
        )
        positive = sum(stop - start for start, stop in spans)
        negative = sample_count - positive
        patient = patients[identity]
        aggregate = counts.setdefault(patient, [0, 0])
        aggregate[0] += positive
        aggregate[1] += negative
        record_rows.append(
            {
                "analysis_identity_id": identity,
                "patient_key": patient,
                "provider_target_sample_count": sample_count,
                "positive_unique_sample_count": positive,
                "negative_unique_sample_count": negative,
                "event_projection_ledger_sha256": _canonical_sha256(ledger),
            }
        )
    primitive = fit_patient_equal_class_weights_pure_primitive(
        {patient: tuple(value) for patient, value in counts.items()},
        fit_roster_sha256=roster_receipt["receipt_sha256"],
    )
    receipt = _content_address(
        {
            "schema_version": "st_authorized_patient_equal_class_weight_v1",
            "registry_sha256": phase_receipt["registry_sha256"],
            "variant_id": roster_receipt["variant_id"],
            "outer_fold": phase_receipt["outer_fold"],
            "phase": phase_receipt["phase"],
            "detector_fold_phase_receipt_sha256": phase_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": roster_receipt[
                "receipt_sha256"
            ],
            "unique_absolute_sample_count_rows_sha256": _canonical_sha256(
                record_rows
            ),
            "primitive_class_weight_receipt": primitive,
            "raw_caller_patient_counts_accepted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AuthorizedSeizureTransformerClassWeight(
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_CLASS_WEIGHT_SEAL,
    )


def _require_authorized_st_class_weight(
    value: object,
) -> dict[str, Any]:
    if (
        not isinstance(value, AuthorizedSeizureTransformerClassWeight)
        or value._validation_seal is not _CLASS_WEIGHT_SEAL
    ):
        raise TypeError(
            "formal SeizureTransformer loss requires an opaque class-weight authority"
        )
    receipt = _validate_content_address(
        value.receipt,
        required={
            "schema_version",
            "registry_sha256",
            "variant_id",
            "outer_fold",
            "phase",
            "detector_fold_phase_receipt_sha256",
            "variant_training_roster_receipt_sha256",
            "unique_absolute_sample_count_rows_sha256",
            "primitive_class_weight_receipt",
            "raw_caller_patient_counts_accepted",
            "receipt_sha256",
        },
        context="authorized SeizureTransformer class weight",
    )
    primitive = receipt["primitive_class_weight_receipt"]
    if (
        receipt["schema_version"] != "st_authorized_patient_equal_class_weight_v1"
        or receipt["raw_caller_patient_counts_accepted"] is not False
        or type(primitive) is not dict
        or primitive.get("fit_roster_sha256")
        != receipt["variant_training_roster_receipt_sha256"]
        or not isinstance(primitive.get("positive_weight"), (int, float))
        or not 0 < float(primitive["positive_weight"]) <= 50
    ):
        raise ValueError("authorized SeizureTransformer class weight drifted")
    pending = deepcopy(primitive)
    supplied = pending.get("receipt_sha256")
    pending["receipt_sha256"] = _CONTENT_PENDING
    if supplied != _canonical_sha256(pending):
        raise ValueError("SeizureTransformer primitive class weight drifted")
    return receipt


def seizuretransformer_authorized_patient_macro_bce(
    probabilities: object,
    *,
    target_bundles: Sequence[AuthorizedSeizureTransformerTargetBundle],
    class_weight_authority: AuthorizedSeizureTransformerClassWeight,
) -> float:
    """CPU replay loss with masks/patient keys only from opaque bundles."""

    if not target_bundles:
        raise ValueError("formal SeizureTransformer loss needs at least one tile")
    bundles = [_require_authorized_st_target_bundle(row) for row in target_bundles]
    class_weight = _require_authorized_st_class_weight(class_weight_authority)
    first = bundles[0].receipt
    if any(
        row.receipt["registry_sha256"] != first["registry_sha256"]
        or row.receipt["variant_id"] != first["variant_id"]
        or row.receipt["outer_fold"] != first["outer_fold"]
        or row.receipt["phase"] != first["phase"]
        or row.receipt["variant_training_roster_receipt_sha256"]
        != first["variant_training_roster_receipt_sha256"]
        for row in bundles
    ) or (
        class_weight["registry_sha256"] != first["registry_sha256"]
        or class_weight["variant_id"] != first["variant_id"]
        or class_weight["outer_fold"] != first["outer_fold"]
        or class_weight["phase"] != first["phase"]
        or class_weight["variant_training_roster_receipt_sha256"]
        != first["variant_training_roster_receipt_sha256"]
    ):
        raise ValueError("SeizureTransformer formal loss authority binding drifted")
    return masked_patient_macro_bce_pure_primitive(
        probabilities,
        np.stack([row.target for row in bundles]),
        np.stack([row.observed_mask for row in bundles]),
        [str(row.receipt["fold_owned_patient_key"]) for row in bundles],
        positive_weight=float(
            class_weight["primitive_class_weight_receipt"]["positive_weight"]
        ),
    )


def seizuretransformer_authorized_dense_metric_counts(
    probabilities: object,
    *,
    target_bundles: Sequence[AuthorizedSeizureTransformerTargetBundle],
    threshold: float,
) -> dict[str, Any]:
    """Formal dense metric path; bundle masks exclude all synthetic context."""

    if not target_bundles:
        raise ValueError("formal SeizureTransformer metric needs at least one tile")
    bundles = [_require_authorized_st_target_bundle(row) for row in target_bundles]
    first = bundles[0].receipt
    if any(
        row.receipt["registry_sha256"] != first["registry_sha256"]
        or row.receipt["variant_id"] != first["variant_id"]
        or row.receipt["outer_fold"] != first["outer_fold"]
        or row.receipt["phase"] != first["phase"]
        or row.receipt["variant_training_roster_receipt_sha256"]
        != first["variant_training_roster_receipt_sha256"]
        for row in bundles
    ):
        raise ValueError("SeizureTransformer formal metric authority binding drifted")
    result = masked_dense_binary_metric_counts_pure_primitive(
        probabilities,
        np.stack([row.target for row in bundles]),
        np.stack([row.observed_mask for row in bundles]),
        threshold=threshold,
    )
    return _content_address(
        {
            "schema_version": "st_authorized_dense_metric_counts_v1",
            "registry_sha256": first["registry_sha256"],
            "variant_id": first["variant_id"],
            "outer_fold": first["outer_fold"],
            "phase": first["phase"],
            "variant_training_roster_receipt_sha256": first[
                "variant_training_roster_receipt_sha256"
            ],
            "target_bundle_receipt_roster_sha256": _canonical_sha256(
                [row.receipt["receipt_sha256"] for row in bundles]
            ),
            "metric_counts": result,
            "context_contributed_to_metric": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def fit_patient_equal_class_weights(*_args: object, **_kwargs: object) -> None:
    """Legacy caller-owned count API; never formal training authority."""

    raise PermissionError(
        "raw patient counts are not SeizureTransformer training authority; use "
        "authorize_seizuretransformer_class_weight"
    )


def masked_patient_macro_bce(*_args: object, **_kwargs: object) -> None:
    """Legacy raw target/mask/patient API; permanently fail closed."""

    raise PermissionError(
        "raw targets, masks and patient keys are not SeizureTransformer formal loss inputs"
    )


def build_patient_balanced_epoch_plan(*_args: object, **_kwargs: object) -> None:
    """Legacy caller-owned pool API; permanently fail closed."""

    raise PermissionError(
        "caller-owned patient/tile pools are not SeizureTransformer epoch authority; "
        "use build_authorized_seizuretransformer_epoch_plan"
    )


def validate_checkpoint_receipt_schema_only(
    value: Mapping[str, Any], *, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate declarations only; this never admits a checkpoint artifact."""

    validated_registry = validate_registry(dict(registry))
    required = {
        "schema_version",
        "registry_sha256",
        "variant_id",
        "outer_fold",
        "stage",
        "epoch_completed",
        "derived_seed",
        "architecture_source_sha256",
        "transform_profile_id",
        "gradient_patient_roster_sha256",
        "validation_patient_roster_sha256",
        "reference_authority_receipt_sha256",
        "class_weight_receipt_sha256",
        "epoch_sampler_receipt_sha256",
        "optimizer_state_payload_sha256",
        "model_safetensors_sha256",
        "numeric_environment_receipt_sha256",
        "out_of_scope_reference_open_count",
        "resume_boundary",
        "receipt_sha256",
    }
    data = _strict_dict(dict(value), required, "checkpoint receipt")
    if data["schema_version"] != "st_cleanroom_epoch_checkpoint_receipt_v1":
        raise ValueError("checkpoint receipt schema drifted")
    if data["registry_sha256"] != validated_registry["registry_sha256"]:
        raise ValueError("checkpoint registry binding drifted")
    expected_seed = derive_training_seed(
        variant_id=data["variant_id"],
        outer_fold=data["outer_fold"],
        stage=data["stage"],
    )
    if data["derived_seed"] != expected_seed:
        raise ValueError("checkpoint seed binding drifted")
    if (
        isinstance(data["epoch_completed"], bool)
        or not isinstance(data["epoch_completed"], int)
        or data["epoch_completed"] <= 0
    ):
        raise ValueError("checkpoint epoch must be positive")
    for field in (
        "architecture_source_sha256",
        "gradient_patient_roster_sha256",
        "reference_authority_receipt_sha256",
        "class_weight_receipt_sha256",
        "epoch_sampler_receipt_sha256",
        "optimizer_state_payload_sha256",
        "model_safetensors_sha256",
        "numeric_environment_receipt_sha256",
    ):
        _require_sha256(data[field], field)
    if data["stage"] == "selection":
        _require_sha256(
            data["validation_patient_roster_sha256"],
            "validation_patient_roster_sha256",
        )
    elif data["validation_patient_roster_sha256"] is not None:
        raise ValueError("final refit may not bind a validation roster")
    if data["architecture_source_sha256"] != (
        "0c3fd38a5350bb293e5337c26bb01c83945624b6eb8000da50e955e54174c7b2"
    ):
        raise ValueError("checkpoint architecture source drifted")
    if data["transform_profile_id"] != "st_cleanroom_full_record_robust_v1":
        raise ValueError("checkpoint transform profile drifted")
    if data["out_of_scope_reference_open_count"] != 0:
        raise ValueError("checkpoint has forbidden reference exposure")
    if data["resume_boundary"] != "next_epoch_only":
        raise ValueError("only epoch-boundary resume is admitted")
    pending = deepcopy(data)
    pending["receipt_sha256"] = _CONTENT_PENDING
    if data["receipt_sha256"] != _canonical_sha256(pending):
        raise ValueError("checkpoint receipt is not content-addressed")
    return data


def admit_seizuretransformer_checkpoint(
    *_args: object, **_kwargs: object
) -> None:
    """Fail closed until model/optimizer/init/sampler bytes are replayed."""

    raise PermissionError(
        "SeizureTransformer checkpoint admission requires actual model, optimizer, "
        "initialization, epoch-plan, roster and phase artifact byte replay; no such "
        "materialized admission bundle is registered"
    )


def validate_checkpoint_receipt(*_args: object, **_kwargs: object) -> None:
    """Compatibility name for formal admission, never a schema-only bypass."""

    return admit_seizuretransformer_checkpoint(*_args, **_kwargs)


__all__ = [
    "AuthorizedSeizureTransformerClassWeight",
    "AuthorizedSeizureTransformerFoldPhase",
    "AuthorizedSeizureTransformerRecordPool",
    "AuthorizedSeizureTransformerTargetBundle",
    "AuthorizedSeizureTransformerVariantTrainingRoster",
    "CONFIG_RELATIVE_PATH",
    "PROVIDER_ID",
    "REGISTRY_ID",
    "SCHEMA_VERSION",
    "ST16_TYPED_UNITS",
    "ST16_SHORT_CONTEXT_POLICY_ID",
    "ST16_VARIANT_ID",
    "ST18_TYPED_UNITS",
    "ST18_VARIANT_ID",
    "SeizureTransformerTransformResult",
    "SeizureTransformerPreReferenceEligibilityOutcome",
    "admit_seizuretransformer_checkpoint",
    "apply_full_record_transform",
    "apply_short_record_context_sensitivity_transform",
    "authorize_seizuretransformer_class_weight",
    "authorize_seizuretransformer_fold_phase",
    "authorize_seizuretransformer_record_tile_pool",
    "authorize_seizuretransformer_target_bundle",
    "authorize_seizuretransformer_variant_training_roster",
    "build_authorized_seizuretransformer_epoch_plan",
    "build_patient_balanced_epoch_plan",
    "build_patient_balanced_epoch_plan_pure_primitive",
    "build_seizuretransformer_dense_target_pure_primitive",
    "build_registry",
    "derive_training_seed",
    "fit_patient_equal_class_weights",
    "fit_patient_equal_class_weights_pure_primitive",
    "load_registry",
    "masked_patient_macro_bce",
    "masked_patient_macro_bce_pure_primitive",
    "masked_dense_binary_metric_counts_pure_primitive",
    "materialize_seizuretransformer_pre_reference_eligibility",
    "enumerate_seizuretransformer_training_tiles",
    "seizuretransformer_authorized_patient_macro_bce",
    "seizuretransformer_authorized_dense_metric_counts",
    "seizuretransformer_transform_valid_support_mask",
    "seizuretransformer_cleanroom_registry_code_sha256",
    "validate_checkpoint_receipt",
    "validate_checkpoint_receipt_schema_only",
    "validate_registry",
    "validate_runtime_environment",
    "validate_static_execution_bindings",
    "validate_transform_result",
]
