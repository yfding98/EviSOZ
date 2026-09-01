"""Fail-closed promotion contract for the six formal TUSZ ictal producers.

The five out-of-fold heads and the final source-train head are not authorized
for evidence-cache materialization merely because checkpoints exist.  This
module binds every strictly loaded production run to four independent checks:

* explicit native positive/negative support and held-patient fidelity;
* time-only and mask-only shortcut controls on the identical observed cells;
* target-free cross-producer score-scale alignment on one shared dev probe;
* a patient-grouped probe showing that OOF evidence does not reveal fold ID.

No DeepSOZ SOZ target or private label is accepted by these receipts.  Even a
successful promotion keeps the output semantics at a conditional TUSZ ictal
involvement score; source-development evaluation never promotes it to a
probability calibrator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence, TypeVar

import numpy as np
import torch

from .concept_checkpoint import LoadedIctalConceptCheckpoint
from .concept_metrics import IctalConceptMetrics, patient_macro_ictal_metrics
from .concept_run import ICTAL_IDENTITY_SCALER_SHA256
from .concept_oof import IctalConceptOOFProtocolArtifact
from .evidence_authorization import NativeClassSupportReceipt, source_roster_sha256
from .ictal_production import (
    ICTAL_NATIVE_TARGET_SEMANTICS,
    ICTAL_PRODUCTION_RUN_SCHEMA,
    LoadedIctalProductionRun,
)
from .ictal_prediction_artifacts import (
    ICTAL_MASK_ONLY_CONTROL,
    ICTAL_TIME_ONLY_CONTROL,
    VerifiedIctalControlPredictionArtifact,
    VerifiedIctalFoldIdentityProbeArtifact,
    VerifiedIctalNativePredictionArtifact,
    VerifiedIctalScaleProbeArtifact,
    verified_shortcut_probe_from_artifacts,
)
from .ictal_gate_policy import (
    ICTAL_PROMOTION_GATE_POLICY_SCHEMA,
    ICTAL_SCALE_QUANTILE_LEVELS,
    IctalPromotionGatePolicy,
    VerifiedIctalPromotionGatePolicyArtifact,
)
from .ictal_support import audit_ictal_source_support


ICTAL_PROMOTION_SELECTIONS = (
    "fold0",
    "fold1",
    "fold2",
    "fold3",
    "fold4",
    "final",
)
ICTAL_PROMOTED_OUTPUT_SEMANTICS = "conditional_ictal_involvement_score"
ICTAL_NATIVE_SUPPORT_RECEIPT_SCHEMA = "soz_ictal_native_support_receipt_v1"
ICTAL_NATIVE_FIDELITY_RECEIPT_SCHEMA = "soz_ictal_native_fidelity_receipt_v1"
ICTAL_SHORTCUT_PROBE_RECEIPT_SCHEMA = "soz_ictal_shortcut_probe_receipt_v1"
ICTAL_SCALE_ALIGNMENT_RECEIPT_SCHEMA = "soz_ictal_scale_alignment_receipt_v1"
ICTAL_FOLD_IDENTITY_PROBE_RECEIPT_SCHEMA = (
    "soz_ictal_fold_identity_probe_receipt_v1"
)
ICTAL_PRODUCER_PROMOTION_SCHEMA = "soz_ictal_six_producer_promotion_v1"
ICTAL_PRODUCER_PROMOTION_ARTIFACT_SCHEMA = (
    "soz_ictal_six_producer_promotion_artifact_v1"
)
ICTAL_PRODUCER_PROMOTION_BUNDLE_RECEIPT_SCHEMA = (
    "soz_ictal_six_producer_promotion_bundle_receipt_v1"
)
ICTAL_PRODUCER_PROMOTION_ARTIFACT_FILENAME = "promotion.json"
ICTAL_PRODUCER_PROMOTION_BUNDLE_RECEIPT_FILENAME = "receipt.json"
ICTAL_FORMAL_PROMOTION_BLOCKERS: tuple[str, ...] = ()

ICTAL_SCALE_QUANTILE_ESTIMATOR = (
    "mean_of_within_patient_linear_quantiles_equal_patient_weight_v1"
)
ICTAL_FOLD_IDENTITY_FEATURE_POLICY = (
    "oof_ictal_four_second_mean_max_phase_masked_patient_summary_v1"
)
ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION = 52

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NATIVE_EVALUATION_MARKER = object()
_SHORTCUT_PROBE_MARKER = object()
_SCALE_ALIGNMENT_MARKER = object()
_FOLD_IDENTITY_MARKER = object()
_PROMOTION_MARKER = object()
_PROMOTION_ARTIFACT_MARKER = object()
_FOLD_IDENTITY_PROBE_SEED = 1729
_FOLD_IDENTITY_BOOTSTRAPS = 1000
_FOLD_IDENTITY_PERMUTATIONS = 500


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Ictal promotion receipt is not canonical JSON data") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = f"{name}|{tuple(value.shape)}|{value.dtype}".encode("ascii")
    digest.update(len(metadata).to_bytes(4, "little"))
    digest.update(metadata)
    raw = value.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _selection(value: object) -> str:
    if not isinstance(value, str) or value not in ICTAL_PROMOTION_SELECTIONS:
        raise ValueError(
            "Ictal producer selection must be one of fold0..fold4 or final"
        )
    return value


def _selection_order(value: str) -> int:
    return ICTAL_PROMOTION_SELECTIONS.index(_selection(value))


def _roster(
    values: Sequence[object], *, field: str, require_nonempty: bool = True
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    normalized = tuple(str(value).strip() for value in values)
    if (require_nonempty and not normalized) or any(not value for value in normalized):
        raise ValueError(f"{field} must be a non-empty trimmed roster")
    if normalized != tuple(sorted(normalized)) or len(set(normalized)) != len(
        normalized
    ):
        raise ValueError(f"{field} must be unique and canonically sorted")
    return normalized


def _finite(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or (
        minimum is not None and normalized < minimum
    ):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return normalized


@dataclass(frozen=True)
class IctalNativeSupportReceipt:
    selection: str
    production_run_manifest_sha256: str
    native_evaluation_manifest_sha256: str
    native_evaluation_corpus_index_sha256: str
    native_public_patient_ids: tuple[str, ...]
    native_public_patient_roster_sha256: str
    support_audit_input_sha256: str
    support: NativeClassSupportReceipt
    target_semantics: str = ICTAL_NATIVE_TARGET_SEMANTICS
    deepsoz_soz_labels_used: bool = False
    private_labels_used: bool = False
    missing_tusz_bins_imputed_as_negative: bool = False
    schema_version: str = ICTAL_NATIVE_SUPPORT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection", _selection(self.selection))
        for name in (
            "production_run_manifest_sha256",
            "native_evaluation_manifest_sha256",
            "native_evaluation_corpus_index_sha256",
            "native_public_patient_roster_sha256",
            "support_audit_input_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field=name)
            )
        patients = _roster(
            self.native_public_patient_ids, field="native_public_patient_ids"
        )
        object.__setattr__(self, "native_public_patient_ids", patients)
        if self.native_public_patient_roster_sha256 != source_roster_sha256(
            patients
        ):
            raise ValueError("Native support patient-roster SHA mismatch")
        if not isinstance(self.support, NativeClassSupportReceipt):
            raise TypeError("support must be a NativeClassSupportReceipt")
        if self.support.patient_count != len(patients):
            raise ValueError("Native support count disagrees with its patient roster")
        if self.target_semantics != ICTAL_NATIVE_TARGET_SEMANTICS:
            raise ValueError("Native support uses the wrong TUSZ target semantics")
        if (
            self.deepsoz_soz_labels_used is not False
            or self.private_labels_used is not False
            or self.missing_tusz_bins_imputed_as_negative is not False
        ):
            raise ValueError("Native support may use only explicit TUSZ labels")
        if self.schema_version != ICTAL_NATIVE_SUPPORT_RECEIPT_SCHEMA:
            raise ValueError("Unsupported ictal native-support schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class IctalNativeFidelityReceipt:
    selection: str
    production_run_manifest_sha256: str
    checkpoint_manifest_sha256: str
    native_evaluation_manifest_sha256: str
    native_evaluation_corpus_index_sha256: str
    native_public_patient_ids: tuple[str, ...]
    native_public_patient_roster_sha256: str
    native_support_receipt_sha256: str
    native_evaluation_role: str
    metrics: IctalConceptMetrics
    prevalence_baseline_metrics: IctalConceptMetrics
    training_explicit_positive_prevalence: float
    full_native_logits_sha256: str
    native_targets_sha256: str
    native_target_mask_sha256: str
    training_targets_sha256: str
    training_target_mask_sha256: str
    training_source_public_roster_sha256: str
    training_prevalence_input_sha256: str
    mean_patient_loss: float
    n_events: int
    target_semantics: str = ICTAL_NATIVE_TARGET_SEMANTICS
    output_semantics: str = ICTAL_PROMOTED_OUTPUT_SEMANTICS
    deepsoz_soz_labels_used: bool = False
    missing_tusz_bins_imputed_as_negative: bool = False
    schema_version: str = ICTAL_NATIVE_FIDELITY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        selection = _selection(self.selection)
        object.__setattr__(self, "selection", selection)
        for name in (
            "production_run_manifest_sha256",
            "checkpoint_manifest_sha256",
            "native_evaluation_manifest_sha256",
            "native_evaluation_corpus_index_sha256",
            "native_public_patient_roster_sha256",
            "native_support_receipt_sha256",
            "full_native_logits_sha256",
            "native_targets_sha256",
            "native_target_mask_sha256",
            "training_targets_sha256",
            "training_target_mask_sha256",
            "training_source_public_roster_sha256",
            "training_prevalence_input_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field=name)
            )
        patients = _roster(
            self.native_public_patient_ids, field="native_public_patient_ids"
        )
        object.__setattr__(self, "native_public_patient_ids", patients)
        if self.native_public_patient_roster_sha256 != source_roster_sha256(
            patients
        ):
            raise ValueError("Native fidelity patient-roster SHA mismatch")
        expected_role = (
            "source_dev_native_tusz"
            if selection == "final"
            else "source_train_oof_fold_heldout_native_tusz"
        )
        if self.native_evaluation_role != expected_role:
            raise ValueError("Native fidelity uses the wrong evaluation role")
        if not isinstance(self.metrics, IctalConceptMetrics):
            raise TypeError("metrics must be IctalConceptMetrics")
        if not isinstance(self.prevalence_baseline_metrics, IctalConceptMetrics):
            raise TypeError(
                "prevalence_baseline_metrics must be IctalConceptMetrics"
            )
        if self.metrics.n_patients != len(patients):
            raise ValueError("Native fidelity metrics disagree with patient roster")
        if (
            self.prevalence_baseline_metrics.n_patients != len(patients)
            or self.prevalence_baseline_metrics.n_observed_labels
            != self.metrics.n_observed_labels
            or self.prevalence_baseline_metrics.n_positive_labels
            != self.metrics.n_positive_labels
            or self.prevalence_baseline_metrics.n_negative_labels
            != self.metrics.n_negative_labels
        ):
            raise ValueError(
                "Prevalence baseline must use the identical held-patient observed cells"
            )
        prevalence = _finite(
            self.training_explicit_positive_prevalence,
            field="training_explicit_positive_prevalence",
            minimum=0.0,
        )
        if not 0.0 < prevalence < 1.0:
            raise ValueError("Training prevalence baseline requires both classes")
        object.__setattr__(
            self, "training_explicit_positive_prevalence", prevalence
        )
        object.__setattr__(
            self,
            "mean_patient_loss",
            _finite(
                self.mean_patient_loss,
                field="mean_patient_loss",
                minimum=0.0,
            ),
        )
        if isinstance(self.n_events, bool) or not isinstance(self.n_events, int) or self.n_events < 1:
            raise ValueError("n_events must be a positive integer")
        if self.target_semantics != ICTAL_NATIVE_TARGET_SEMANTICS:
            raise ValueError("Native fidelity uses the wrong target semantics")
        if self.output_semantics != ICTAL_PROMOTED_OUTPUT_SEMANTICS:
            raise ValueError("Ictal fidelity output cannot claim dense probability")
        if (
            self.deepsoz_soz_labels_used is not False
            or self.missing_tusz_bins_imputed_as_negative is not False
        ):
            raise ValueError("Native fidelity may use only explicit TUSZ labels")
        if self.schema_version != ICTAL_NATIVE_FIDELITY_RECEIPT_SCHEMA:
            raise ValueError("Unsupported ictal native-fidelity schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))

    @property
    def patient_macro_bce_improvement_over_prevalence(self) -> float:
        return (
            self.prevalence_baseline_metrics.patient_macro_bce
            - self.metrics.patient_macro_bce
        )

    @property
    def patient_macro_brier_improvement_over_prevalence(self) -> float:
        return (
            self.prevalence_baseline_metrics.patient_macro_brier
            - self.metrics.patient_macro_brier
        )

    @property
    def patient_macro_ap_lift_over_prevalence(self) -> float | None:
        full = self.metrics.patient_macro_average_precision
        baseline = self.prevalence_baseline_metrics.patient_macro_average_precision
        if full is None or baseline is None:
            return None
        return full - baseline


@dataclass(frozen=True)
class IctalShortcutProbeReceipt:
    selection: str
    production_run_manifest_sha256: str
    checkpoint_manifest_sha256: str
    native_evaluation_manifest_sha256: str
    native_evaluation_corpus_index_sha256: str
    native_public_patient_ids: tuple[str, ...]
    native_public_patient_roster_sha256: str
    probe_input_sha256: str
    time_only_control_run_sha256: str
    mask_only_control_run_sha256: str
    full_logits_sha256: str
    time_only_logits_sha256: str
    mask_only_logits_sha256: str
    native_targets_sha256: str
    native_target_mask_sha256: str
    evaluated_observed_label_count: int
    full_model_patient_macro_bce: float
    time_only_patient_macro_bce: float
    mask_only_patient_macro_bce: float
    identical_patient_and_observed_cell_roster: bool = True
    controls_fit_on_producer_training_roster_only: bool = True
    source_target_mask_used_as_model_input: bool = False
    deepsoz_soz_labels_used: bool = False
    private_labels_used: bool = False
    output_semantics: str = ICTAL_PROMOTED_OUTPUT_SEMANTICS
    schema_version: str = ICTAL_SHORTCUT_PROBE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection", _selection(self.selection))
        for name in (
            "production_run_manifest_sha256",
            "checkpoint_manifest_sha256",
            "native_evaluation_manifest_sha256",
            "native_evaluation_corpus_index_sha256",
            "native_public_patient_roster_sha256",
            "probe_input_sha256",
            "time_only_control_run_sha256",
            "mask_only_control_run_sha256",
            "full_logits_sha256",
            "time_only_logits_sha256",
            "mask_only_logits_sha256",
            "native_targets_sha256",
            "native_target_mask_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field=name)
            )
        patients = _roster(
            self.native_public_patient_ids, field="native_public_patient_ids"
        )
        object.__setattr__(self, "native_public_patient_ids", patients)
        if self.native_public_patient_roster_sha256 != source_roster_sha256(
            patients
        ):
            raise ValueError("Shortcut-probe patient-roster SHA mismatch")
        if (
            isinstance(self.evaluated_observed_label_count, bool)
            or not isinstance(self.evaluated_observed_label_count, int)
            or self.evaluated_observed_label_count < 1
        ):
            raise ValueError("Shortcut probe requires observed native labels")
        for name in (
            "full_model_patient_macro_bce",
            "time_only_patient_macro_bce",
            "mask_only_patient_macro_bce",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), field=name, minimum=0.0),
            )
        if (
            self.identical_patient_and_observed_cell_roster is not True
            or self.controls_fit_on_producer_training_roster_only is not True
            or self.source_target_mask_used_as_model_input is not False
            or self.deepsoz_soz_labels_used is not False
            or self.private_labels_used is not False
        ):
            raise ValueError("Shortcut controls violate held-native input isolation")
        if self.output_semantics != ICTAL_PROMOTED_OUTPUT_SEMANTICS:
            raise ValueError("Shortcut probe changed ictal output semantics")
        if self.schema_version != ICTAL_SHORTCUT_PROBE_RECEIPT_SCHEMA:
            raise ValueError("Unsupported ictal shortcut-probe schema")

    @property
    def minimum_control_bce_improvement(self) -> float:
        return min(
            self.time_only_patient_macro_bce - self.full_model_patient_macro_bce,
            self.mask_only_patient_macro_bce - self.full_model_patient_macro_bce,
        )

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class IctalScaleSummary:
    selection: str
    production_run_manifest_sha256: str
    checkpoint_manifest_sha256: str
    patient_count: int
    observed_score_count: int
    minimum_patient_observed_score_count: int
    maximum_patient_observed_score_count: int
    score_quantiles: tuple[float, ...]
    quantile_estimator: str = ICTAL_SCALE_QUANTILE_ESTIMATOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection", _selection(self.selection))
        for name in (
            "production_run_manifest_sha256",
            "checkpoint_manifest_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field=name)
            )
        integer_fields = (
            "patient_count",
            "observed_score_count",
            "minimum_patient_observed_score_count",
            "maximum_patient_observed_score_count",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Scale summary {field} must be positive")
        if (
            self.minimum_patient_observed_score_count
            > self.maximum_patient_observed_score_count
            or self.observed_score_count
            < self.patient_count * self.minimum_patient_observed_score_count
            or self.observed_score_count
            > self.patient_count * self.maximum_patient_observed_score_count
        ):
            raise ValueError("Scale summary patient/cell counts are inconsistent")
        values = tuple(
            _finite(value, field="score_quantiles", minimum=0.0)
            for value in self.score_quantiles
        )
        if len(values) != len(ICTAL_SCALE_QUANTILE_LEVELS) or any(
            value > 1.0 for value in values
        ):
            raise ValueError("Scale quantiles must contain five values in [0,1]")
        if values != tuple(sorted(values)):
            raise ValueError("Scale quantile values must be nondecreasing")
        object.__setattr__(self, "score_quantiles", values)
        if self.quantile_estimator != ICTAL_SCALE_QUANTILE_ESTIMATOR:
            raise ValueError("Scale quantile estimator cannot change")


@dataclass(frozen=True)
class IctalScaleAlignmentReceipt:
    summaries: tuple[IctalScaleSummary, ...]
    shared_probe_public_patient_ids: tuple[str, ...]
    shared_probe_public_patient_roster_sha256: str
    shared_probe_input_sha256: str
    shared_probe_role: str = "source_dev_target_free_shared_probe"
    all_producers_evaluated_on_identical_cells: bool = True
    native_or_soz_labels_used: bool = False
    private_labels_used: bool = False
    scaler_fitted_on_probe: bool = False
    fold_specific_transform_fitted: bool = False
    calibrator_fitted: bool = False
    score_transform: str = "identity_sigmoid_of_raw_head_logit"
    output_semantics: str = ICTAL_PROMOTED_OUTPUT_SEMANTICS
    quantile_levels: tuple[float, ...] = ICTAL_SCALE_QUANTILE_LEVELS
    schema_version: str = ICTAL_SCALE_ALIGNMENT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if not all(isinstance(item, IctalScaleSummary) for item in self.summaries):
            raise TypeError("summaries must contain IctalScaleSummary values")
        ordered = tuple(sorted(self.summaries, key=lambda item: _selection_order(item.selection)))
        if ordered != self.summaries or tuple(
            item.selection for item in ordered
        ) != ICTAL_PROMOTION_SELECTIONS:
            raise ValueError("Scale alignment requires exactly fold0..fold4 and final")
        counts = {item.observed_score_count for item in ordered}
        if len(counts) != 1:
            raise ValueError("Every producer must score the identical shared cells")
        patient_counts = {item.patient_count for item in ordered}
        patient_cell_ranges = {
            (
                item.minimum_patient_observed_score_count,
                item.maximum_patient_observed_score_count,
            )
            for item in ordered
        }
        if len(patient_counts) != 1 or len(patient_cell_ranges) != 1:
            raise ValueError(
                "Every producer must use identical patient-level mask support"
            )
        patients = _roster(
            self.shared_probe_public_patient_ids,
            field="shared_probe_public_patient_ids",
        )
        object.__setattr__(self, "shared_probe_public_patient_ids", patients)
        object.__setattr__(
            self,
            "shared_probe_public_patient_roster_sha256",
            _require_sha256(
                self.shared_probe_public_patient_roster_sha256,
                field="shared_probe_public_patient_roster_sha256",
            ),
        )
        if self.shared_probe_public_patient_roster_sha256 != source_roster_sha256(
            patients
        ):
            raise ValueError("Scale-alignment shared-probe roster SHA mismatch")
        object.__setattr__(
            self,
            "shared_probe_input_sha256",
            _require_sha256(
                self.shared_probe_input_sha256, field="shared_probe_input_sha256"
            ),
        )
        if self.shared_probe_role != "source_dev_target_free_shared_probe":
            raise ValueError("Scale alignment must use the target-free source-dev probe")
        if (
            self.all_producers_evaluated_on_identical_cells is not True
            or self.native_or_soz_labels_used is not False
            or self.private_labels_used is not False
            or self.scaler_fitted_on_probe is not False
            or self.fold_specific_transform_fitted is not False
            or self.calibrator_fitted is not False
        ):
            raise ValueError("Scale alignment may not fit to labels or probe outputs")
        if self.score_transform != "identity_sigmoid_of_raw_head_logit":
            raise ValueError("Scale audit must preserve raw-logit identity sigmoid scores")
        if self.output_semantics != ICTAL_PROMOTED_OUTPUT_SEMANTICS:
            raise ValueError("Scale alignment changed ictal output semantics")
        levels = tuple(float(value) for value in self.quantile_levels)
        if levels != ICTAL_SCALE_QUANTILE_LEVELS:
            raise ValueError("Scale-alignment quantiles cannot change")
        object.__setattr__(self, "quantile_levels", levels)
        if self.schema_version != ICTAL_SCALE_ALIGNMENT_RECEIPT_SCHEMA:
            raise ValueError("Unsupported ictal scale-alignment schema")
        if next(iter(patient_counts)) != len(patients):
            raise ValueError("Scale summaries disagree with the shared patient roster")

    @property
    def maximum_pairwise_quantile_gap(self) -> float:
        return max(
            abs(left - right)
            for left_index, left_summary in enumerate(self.summaries)
            for right_summary in self.summaries[left_index + 1 :]
            for left, right in zip(
                left_summary.score_quantiles, right_summary.score_quantiles
            )
        )

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class IctalFoldIdentityProbeReceipt:
    producer_bindings: tuple[tuple[str, str, str], ...]
    target_public_fold_assignments: tuple[tuple[str, str, int], ...]
    target_patient_roster_sha256: str
    public_patient_roster_sha256: str
    source_train_signal_attrition_target_ids: tuple[str, ...]
    source_train_signal_attrition_public_ids: tuple[str, ...]
    signal_attrition_receipt_sha256: str
    oof_protocol_artifact_sha256: str
    oof_protocol_receipt_sha256: str
    timeline_context_receipt_sha256: str
    signal_preflight_receipt_sha256: str
    event_registry_sha256: str
    probe_event_roster_sha256: str
    probe_input_sha256: str
    probe_patient_count: int
    probe_event_count: int
    probe_feature_dimension: int
    balanced_accuracy: float
    bootstrap_lower_95: float
    bootstrap_upper_95: float
    permutation_null_mean: float
    permutation_null_upper_95: float
    permutation_p_value: float
    bootstrap_count: int = _FOLD_IDENTITY_BOOTSTRAPS
    permutation_count: int = _FOLD_IDENTITY_PERMUTATIONS
    chance_balanced_accuracy: float = 0.2
    probe_algorithm: str = "fixed_l2_multinomial_ridge"
    probe_feature_policy: str = ICTAL_FOLD_IDENTITY_FEATURE_POLICY
    patient_aggregation_policy: str = (
        "one_row_per_patient_masked_over_complete_signal_eligible_events_v1"
    )
    deployment_mask_policy: str = (
        "full19_physical_edges_and_offset_aware_phase_mask_no_source_target_mask_v1"
    )
    cross_validation_policy: str = (
        "five_split_patient_disjoint_stratified_fixed_before_probe_fit"
    )
    interpretation_policy: str = (
        "non_detection_is_not_proof_gate_uses_bootstrap_effect_upper_bound"
    )
    source_target_mask_used_as_probe_feature: bool = False
    deepsoz_soz_labels_used: bool = False
    private_labels_used: bool = False
    final_producer_included: bool = False
    output_semantics: str = ICTAL_PROMOTED_OUTPUT_SEMANTICS
    schema_version: str = ICTAL_FOLD_IDENTITY_PROBE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        bindings: list[tuple[str, str, str]] = []
        for row in self.producer_bindings:
            if not isinstance(row, tuple) or len(row) != 3:
                raise TypeError("Fold-identity producer bindings must be triples")
            selection, run_sha, checkpoint_sha = row
            bindings.append(
                (
                    _selection(selection),
                    _require_sha256(run_sha, field="producer_run_manifest_sha256"),
                    _require_sha256(
                        checkpoint_sha, field="checkpoint_manifest_sha256"
                    ),
                )
            )
        normalized_bindings = tuple(
            sorted(bindings, key=lambda row: _selection_order(row[0]))
        )
        if tuple(row[0] for row in normalized_bindings) != ICTAL_PROMOTION_SELECTIONS[:5]:
            raise ValueError("Fold-identity probe requires exactly fold0..fold4")
        if tuple(bindings) != normalized_bindings:
            raise ValueError("Fold-identity producer bindings are not canonical")
        object.__setattr__(self, "producer_bindings", normalized_bindings)

        assignments: list[tuple[str, str, int]] = []
        for target_id, public_id, fold in self.target_public_fold_assignments:
            normalized_target = str(target_id).strip()
            normalized_public = str(public_id).strip()
            if not normalized_target or not normalized_public:
                raise ValueError("Fold-identity patient IDs cannot be blank")
            if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
                raise ValueError("Fold-identity assignments require folds in [0,4]")
            assignments.append((normalized_target, normalized_public, fold))
        normalized_assignments = tuple(sorted(assignments))
        if (
            not normalized_assignments
            or tuple(assignments) != normalized_assignments
            or len({target for target, _, _ in normalized_assignments})
            != len(normalized_assignments)
            or len({public for _, public, _ in normalized_assignments})
            != len(normalized_assignments)
            or {fold for _, _, fold in normalized_assignments} != set(range(5))
        ):
            raise ValueError(
                "Fold-identity target/public assignments must uniquely cover all folds"
            )
        object.__setattr__(
            self, "target_public_fold_assignments", normalized_assignments
        )
        target_roster = tuple(target for target, _, _ in normalized_assignments)
        public_roster = tuple(sorted(public for _, public, _ in normalized_assignments))
        for name in ("target_patient_roster_sha256", "public_patient_roster_sha256"):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field=name)
            )
        if self.target_patient_roster_sha256 != source_roster_sha256(target_roster):
            raise ValueError("Fold-identity target-patient roster SHA mismatch")
        if self.public_patient_roster_sha256 != source_roster_sha256(public_roster):
            raise ValueError("Fold-identity patient-roster SHA mismatch")

        attrition_targets = _roster(
            self.source_train_signal_attrition_target_ids,
            field="source_train_signal_attrition_target_ids",
            require_nonempty=False,
        )
        attrition_public = _roster(
            self.source_train_signal_attrition_public_ids,
            field="source_train_signal_attrition_public_ids",
            require_nonempty=False,
        )
        if len(attrition_targets) != len(attrition_public):
            raise ValueError("Signal attrition target/public rosters disagree")
        if set(attrition_targets) & set(target_roster) or set(attrition_public) & set(
            public_roster
        ):
            raise ValueError("Signal attrition cannot overlap the probe roster")
        object.__setattr__(
            self, "source_train_signal_attrition_target_ids", attrition_targets
        )
        object.__setattr__(
            self, "source_train_signal_attrition_public_ids", attrition_public
        )
        for name in (
            "signal_attrition_receipt_sha256",
            "oof_protocol_artifact_sha256",
            "oof_protocol_receipt_sha256",
            "timeline_context_receipt_sha256",
            "signal_preflight_receipt_sha256",
            "event_registry_sha256",
            "probe_event_roster_sha256",
            "probe_input_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field=name)
            )
        for name in (
            "probe_patient_count",
            "probe_event_count",
            "probe_feature_dimension",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.probe_patient_count != len(normalized_assignments):
            raise ValueError("Fold-identity probe patient count changed")
        if self.probe_feature_dimension != ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION:
            raise ValueError("Fold-identity feature dimension cannot change")
        object.__setattr__(
            self,
            "balanced_accuracy",
            _finite(self.balanced_accuracy, field="balanced_accuracy", minimum=0.0),
        )
        for name in (
            "bootstrap_lower_95",
            "bootstrap_upper_95",
            "permutation_null_mean",
            "permutation_null_upper_95",
            "permutation_p_value",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), field=name, minimum=0.0),
            )
        bounded = (
            self.balanced_accuracy,
            self.bootstrap_lower_95,
            self.bootstrap_upper_95,
            self.permutation_null_mean,
            self.permutation_null_upper_95,
            self.permutation_p_value,
        )
        if any(value > 1.0 for value in bounded) or self.chance_balanced_accuracy != 0.2:
            raise ValueError("Fold-identity balanced accuracy must lie in [0,1]")
        if self.bootstrap_lower_95 > self.bootstrap_upper_95:
            raise ValueError("Fold-identity bootstrap interval is reversed")
        if self.bootstrap_count != _FOLD_IDENTITY_BOOTSTRAPS or self.permutation_count != _FOLD_IDENTITY_PERMUTATIONS:
            raise ValueError("Fold-identity resampling counts cannot change")
        if self.probe_algorithm != "fixed_l2_multinomial_ridge":
            raise ValueError("Fold-identity probe algorithm cannot change")
        if self.probe_feature_policy != ICTAL_FOLD_IDENTITY_FEATURE_POLICY:
            raise ValueError("Fold-identity feature policy cannot change")
        if self.patient_aggregation_policy != (
            "one_row_per_patient_masked_over_complete_signal_eligible_events_v1"
        ):
            raise ValueError("Fold-identity patient aggregation cannot change")
        if self.deployment_mask_policy != (
            "full19_physical_edges_and_offset_aware_phase_mask_no_source_target_mask_v1"
        ):
            raise ValueError("Fold-identity mask policy cannot change")
        if self.cross_validation_policy != (
            "five_split_patient_disjoint_stratified_fixed_before_probe_fit"
        ):
            raise ValueError("Fold-identity probe must use patient-grouped OOF predictions")
        if self.interpretation_policy != (
            "non_detection_is_not_proof_gate_uses_bootstrap_effect_upper_bound"
        ):
            raise ValueError("Fold-identity non-detection cannot be claimed as proof")
        if (
            self.source_target_mask_used_as_probe_feature is not False
            or self.deepsoz_soz_labels_used is not False
            or self.private_labels_used is not False
            or self.final_producer_included is not False
        ):
            raise ValueError("Fold-identity probe contains forbidden labels or producer")
        if self.output_semantics != ICTAL_PROMOTED_OUTPUT_SEMANTICS:
            raise ValueError("Fold-identity probe changed ictal output semantics")
        if self.schema_version != ICTAL_FOLD_IDENTITY_PROBE_RECEIPT_SCHEMA:
            raise ValueError("Unsupported ictal fold-identity probe schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class IctalProducerPromotionRow:
    selection: str
    production_run_manifest_sha256: str
    checkpoint_manifest_sha256: str
    checkpoint_sha256: str
    training_manifest_sha256: str
    training_corpus_index_sha256: str
    native_evaluation_manifest_sha256: str
    native_evaluation_corpus_index_sha256: str
    native_support_receipt_sha256: str
    native_fidelity_receipt_sha256: str
    shortcut_probe_receipt_sha256: str
    native_class_sensitive_metrics_authorized: bool
    native_evaluation_role: str
    output_semantics: str = ICTAL_PROMOTED_OUTPUT_SEMANTICS


@dataclass(frozen=True)
class IctalProducerPromotionReceipt:
    gate_policy_receipt_sha256: str
    producers: tuple[IctalProducerPromotionRow, ...]
    scale_alignment_receipt_sha256: str
    fold_identity_probe_receipt_sha256: str
    output_semantics: str = ICTAL_PROMOTED_OUTPUT_SEMANTICS
    source_dev_probability_calibration_authorized: bool = False
    evidence_cache_materialization_authorized: bool = True
    all_required_gates_passed: bool = True
    schema_version: str = ICTAL_PRODUCER_PROMOTION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "gate_policy_receipt_sha256",
            "scale_alignment_receipt_sha256",
            "fold_identity_probe_receipt_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field=name)
            )
        if not all(isinstance(row, IctalProducerPromotionRow) for row in self.producers):
            raise TypeError("producers must contain promotion rows")
        if tuple(row.selection for row in self.producers) != ICTAL_PROMOTION_SELECTIONS:
            raise ValueError("Promotion receipt requires exactly six canonical producers")
        if any(row.output_semantics != ICTAL_PROMOTED_OUTPUT_SEMANTICS for row in self.producers):
            raise ValueError("A promoted producer changed output semantics")
        if self.output_semantics != ICTAL_PROMOTED_OUTPUT_SEMANTICS:
            raise ValueError("Promoted ictal output cannot claim dense probability")
        if self.source_dev_probability_calibration_authorized is not False:
            raise ValueError("Source-dev ictal probability calibration is forbidden")
        if (
            self.evidence_cache_materialization_authorized is not True
            or self.all_required_gates_passed is not True
        ):
            raise ValueError("Promotion receipt may only represent a passed contract")
        if self.schema_version != ICTAL_PRODUCER_PROMOTION_SCHEMA:
            raise ValueError("Unsupported ictal producer-promotion schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))

    def require_probability_calibration(self) -> None:
        raise ValueError(
            "TUSZ source-dev native support does not authorize ictal probability "
            "calibration; use conditional_ictal_involvement_score"
        )


@dataclass(frozen=True, init=False)
class VerifiedIctalNativeEvaluation:
    """Opaque support/fidelity capability computed from bound prediction tensors."""

    selection: str
    support: IctalNativeSupportReceipt
    fidelity: IctalNativeFidelityReceipt
    support_receipt_sha256: str
    fidelity_receipt_sha256: str

    def __init__(
        self,
        *,
        _verification_marker: object,
        support: IctalNativeSupportReceipt,
        fidelity: IctalNativeFidelityReceipt,
    ) -> None:
        if _verification_marker is not _NATIVE_EVALUATION_MARKER:
            raise TypeError(
                "VerifiedIctalNativeEvaluation can only be issued by tensor replay"
            )
        if not isinstance(support, IctalNativeSupportReceipt) or not isinstance(
            fidelity, IctalNativeFidelityReceipt
        ):
            raise TypeError("Verified native evaluation requires typed receipts")
        if support.selection != fidelity.selection:
            raise ValueError("Native support/fidelity selections disagree")
        if fidelity.native_support_receipt_sha256 != support.receipt_sha256:
            raise ValueError("Native fidelity is not bound to its support replay")
        object.__setattr__(self, "selection", support.selection)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "fidelity", fidelity)
        object.__setattr__(self, "support_receipt_sha256", support.receipt_sha256)
        object.__setattr__(self, "fidelity_receipt_sha256", fidelity.receipt_sha256)

    def assert_unchanged(self) -> None:
        if self.support.receipt_sha256 != self.support_receipt_sha256 or (
            self.fidelity.receipt_sha256 != self.fidelity_receipt_sha256
        ):
            raise ValueError("Verified native evaluation changed after issuance")


@dataclass(frozen=True, init=False)
class VerifiedIctalShortcutProbe:
    selection: str
    receipt: IctalShortcutProbeReceipt
    receipt_sha256: str

    def __init__(
        self,
        *,
        _verification_marker: object,
        receipt: IctalShortcutProbeReceipt,
    ) -> None:
        if _verification_marker is not _SHORTCUT_PROBE_MARKER:
            raise TypeError("Verified shortcut probe can only be issued by tensor replay")
        if not isinstance(receipt, IctalShortcutProbeReceipt):
            raise TypeError("receipt must be IctalShortcutProbeReceipt")
        object.__setattr__(self, "selection", receipt.selection)
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "receipt_sha256", receipt.receipt_sha256)

    def assert_unchanged(self) -> None:
        if self.receipt.receipt_sha256 != self.receipt_sha256:
            raise ValueError("Verified shortcut probe changed after issuance")


@dataclass(frozen=True, init=False)
class VerifiedIctalScaleAlignment:
    receipt: IctalScaleAlignmentReceipt
    receipt_sha256: str

    def __init__(
        self,
        *,
        _verification_marker: object,
        receipt: IctalScaleAlignmentReceipt,
    ) -> None:
        if _verification_marker is not _SCALE_ALIGNMENT_MARKER:
            raise TypeError("Verified scale alignment can only be issued by tensor replay")
        if not isinstance(receipt, IctalScaleAlignmentReceipt):
            raise TypeError("receipt must be IctalScaleAlignmentReceipt")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "receipt_sha256", receipt.receipt_sha256)

    def assert_unchanged(self) -> None:
        if self.receipt.receipt_sha256 != self.receipt_sha256:
            raise ValueError("Verified scale alignment changed after issuance")


@dataclass(frozen=True, init=False)
class VerifiedIctalFoldIdentityProbe:
    receipt: IctalFoldIdentityProbeReceipt
    receipt_sha256: str

    def __init__(
        self,
        *,
        _verification_marker: object,
        receipt: IctalFoldIdentityProbeReceipt,
    ) -> None:
        if _verification_marker is not _FOLD_IDENTITY_MARKER:
            raise TypeError("Verified fold-identity probe can only be issued by replay")
        if not isinstance(receipt, IctalFoldIdentityProbeReceipt):
            raise TypeError("receipt must be IctalFoldIdentityProbeReceipt")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "receipt_sha256", receipt.receipt_sha256)

    def assert_unchanged(self) -> None:
        if self.receipt.receipt_sha256 != self.receipt_sha256:
            raise ValueError("Verified fold-identity probe changed after issuance")


@dataclass(frozen=True, init=False)
class VerifiedIctalProducerPromotion:
    receipt: IctalProducerPromotionReceipt
    receipt_sha256: str

    def __init__(
        self,
        *,
        _verification_marker: object,
        receipt: IctalProducerPromotionReceipt,
    ) -> None:
        if _verification_marker is not _PROMOTION_MARKER:
            raise TypeError("Verified promotion can only be issued by the gate validator")
        if not isinstance(receipt, IctalProducerPromotionReceipt):
            raise TypeError("receipt must be IctalProducerPromotionReceipt")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "receipt_sha256", receipt.receipt_sha256)

    @property
    def producers(self) -> tuple[IctalProducerPromotionRow, ...]:
        return self.receipt.producers

    @property
    def output_semantics(self) -> str:
        return self.receipt.output_semantics

    @property
    def source_dev_probability_calibration_authorized(self) -> bool:
        return self.receipt.source_dev_probability_calibration_authorized

    def require_probability_calibration(self) -> None:
        self.receipt.require_probability_calibration()

    def assert_unchanged(self) -> None:
        if self.receipt.receipt_sha256 != self.receipt_sha256:
            raise ValueError("Verified promotion changed after issuance")


_ReceiptT = TypeVar("_ReceiptT")


def _index_receipts(
    values: Sequence[_ReceiptT], expected_type: type[_ReceiptT], *, label: str
) -> dict[str, _ReceiptT]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a receipt sequence")
    indexed: dict[str, _ReceiptT] = {}
    for value in values:
        if not isinstance(value, expected_type):
            raise TypeError(f"{label} contains an invalid receipt type")
        selection = _selection(getattr(value, "selection"))
        if selection in indexed:
            raise ValueError(f"{label} contains duplicate selection {selection}")
        indexed[selection] = value
    if tuple(sorted(indexed, key=_selection_order)) != ICTAL_PROMOTION_SELECTIONS:
        raise ValueError(f"{label} requires exactly fold0..fold4 and final")
    return indexed


def _production_runs(
    values: Sequence[LoadedIctalProductionRun],
) -> dict[str, LoadedIctalProductionRun]:
    indexed: dict[str, LoadedIctalProductionRun] = {}
    for run in values:
        if not isinstance(run, LoadedIctalProductionRun):
            raise TypeError("production_runs must be strictly loaded production runs")
        if not isinstance(run.manifest, Mapping):
            raise TypeError("Loaded production manifest must be a mapping")
        selection = _selection(run.manifest.get("selection"))
        if selection in indexed:
            raise ValueError(f"Duplicate loaded ictal producer: {selection}")
        if run.manifest.get("schema_version") != ICTAL_PRODUCTION_RUN_SCHEMA:
            raise ValueError("Loaded ictal producer uses the wrong run schema")
        _require_sha256(run.manifest_sha256, field="production_run_manifest_sha256")
        if not isinstance(run.checkpoint, LoadedIctalConceptCheckpoint):
            raise TypeError("Loaded production run lacks a strict checkpoint")
        if run.manifest.get("checkpoint_manifest_sha256") != run.checkpoint.manifest_sha256:
            raise ValueError("Production run/checkpoint manifest binding changed")
        if run.manifest.get("checkpoint_sha256") != run.checkpoint.checkpoint_sha256:
            raise ValueError("Production run/checkpoint bytes binding changed")
        if run.checkpoint.metadata.get("scaler_sha256") != (
            ICTAL_IDENTITY_SCALER_SHA256
        ):
            raise ValueError(
                "Formal ictal producer must retain the frozen identity scaler"
            )
        indexed[selection] = run
    if tuple(sorted(indexed, key=_selection_order)) != ICTAL_PROMOTION_SELECTIONS:
        raise ValueError("Promotion requires exactly six loaded production runs")
    return indexed


def _manifest_roster(manifest: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = manifest.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"Production {field} must be a JSON string array")
    return _roster(tuple(value), field=field)


def _run_metrics(manifest: Mapping[str, object]) -> tuple[IctalConceptMetrics, float, int]:
    payload = manifest.get("native_metrics")
    if not isinstance(payload, Mapping):
        raise TypeError("Production native_metrics must be a mapping")
    if payload.get("target_semantics") != ICTAL_NATIVE_TARGET_SEMANTICS:
        raise ValueError("Production native metrics changed target semantics")
    if (
        payload.get("deepsoz_soz_labels_used") is not False
        or payload.get("missing_tusz_bins_imputed_as_negative") is not False
    ):
        raise ValueError("Production native metrics contain forbidden labels")
    try:
        metrics = IctalConceptMetrics(
            **{
                name: payload[name]
                for name in IctalConceptMetrics.__dataclass_fields__
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Production native metrics are invalid") from exc
    mean_loss = _finite(
        payload.get("mean_patient_loss"),
        field="native_metrics.mean_patient_loss",
        minimum=0.0,
    )
    n_events = payload.get("n_events")
    if isinstance(n_events, bool) or not isinstance(n_events, int) or n_events < 1:
        raise ValueError("Production native event count must be positive")
    return metrics, mean_loss, n_events


def _event_patient_inputs(
    event_ids: Sequence[object],
    patient_ids: Sequence[object],
    *,
    event_count: int,
    label: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], torch.Tensor]:
    events = tuple(str(value).strip() for value in event_ids)
    patients_by_event = tuple(str(value).strip() for value in patient_ids)
    if len(events) != event_count or len(patients_by_event) != event_count:
        raise ValueError(f"{label} identities must align with event tensors")
    if any(not value for value in (*events, *patients_by_event)):
        raise ValueError(f"{label} identities cannot be blank")
    if len(set(events)) != len(events):
        raise ValueError(f"{label} event IDs must be unique")
    patient_roster = tuple(sorted(set(patients_by_event)))
    patient_index = {patient: index for index, patient in enumerate(patient_roster)}
    encoded = torch.tensor(
        [patient_index[patient] for patient in patients_by_event], dtype=torch.long
    )
    return events, patients_by_event, patient_roster, encoded


def _canonical_native_tensors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(logits, torch.Tensor) or not isinstance(
        targets, torch.Tensor
    ) or not isinstance(target_mask, torch.Tensor):
        raise TypeError(f"{label} logits, targets, and mask must be tensors")
    if logits.ndim != 4 or logits.shape[1] != 20 or logits.shape[-1] != 1:
        raise ValueError(f"{label} logits must have shape [E,20,T,1]")
    expected = tuple(logits.shape[:-1])
    if tuple(targets.shape) != expected or tuple(target_mask.shape) != expected:
        raise ValueError(f"{label} targets/mask must have shape [E,20,T]")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError(f"{label} logits and targets must be floating point")
    if target_mask.dtype != torch.bool:
        raise TypeError(f"{label} target mask must be bool")
    if logits.requires_grad or targets.requires_grad or target_mask.requires_grad:
        raise ValueError(f"{label} tensors must be detached")
    cloned_logits = logits.detach().cpu().contiguous().clone()
    cloned_targets = targets.detach().cpu().contiguous().clone()
    cloned_mask = target_mask.detach().cpu().contiguous().clone()
    if not torch.isfinite(cloned_logits).all():
        raise ValueError(f"{label} logits must be finite")
    observed = cloned_targets[cloned_mask]
    if not torch.isfinite(observed).all() or (
        observed.numel() and not torch.all((observed == 0) | (observed == 1))
    ):
        raise ValueError(f"{label} observed targets must be finite binary values")
    # Unknown cells are canonical zero-fill solely for hashing/storage.  The
    # bool mask remains the only authority for metric inclusion.
    cloned_targets[~cloned_mask] = 0
    return cloned_logits, cloned_targets, cloned_mask


def _metrics_close(left: IctalConceptMetrics, right: IctalConceptMetrics) -> bool:
    for name in IctalConceptMetrics.__dataclass_fields__:
        lhs = getattr(left, name)
        rhs = getattr(right, name)
        if lhs is None or rhs is None:
            if lhs is not rhs:
                return False
        elif isinstance(lhs, int):
            if lhs != rhs:
                return False
        elif not math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=1e-6):
            return False
    return True


def verify_ictal_native_evaluation_tensors(
    *,
    production_run: LoadedIctalProductionRun,
    full_native_logits: torch.Tensor,
    native_targets: torch.Tensor,
    native_target_mask: torch.Tensor,
    native_event_ids: Sequence[object],
    native_public_patient_ids: Sequence[object],
    training_targets: torch.Tensor,
    training_target_mask: torch.Tensor,
    training_event_ids: Sequence[object],
    training_public_patient_ids: Sequence[object],
) -> VerifiedIctalNativeEvaluation:
    """Recompute support, full fidelity, and a train-prevalence baseline.

    The prevalence constant is estimated only from the producer's fit roster
    and is evaluated on the exact held-patient observed cells.  No held target
    is used to fit that baseline.  This caller-tensor API is diagnostic-only
    and is never accepted directly by formal promotion.  The strict native
    artifact loader invokes it internally while independently regenerating
    the exact checkpoint/source prediction grid.
    """

    if not isinstance(production_run, LoadedIctalProductionRun):
        raise TypeError("production_run must be strictly loaded")
    manifest = production_run.manifest
    selection = _selection(manifest.get("selection"))
    native_logits, native_values, native_mask = _canonical_native_tensors(
        full_native_logits,
        native_targets,
        native_target_mask,
        label="native evaluation",
    )
    native_events, native_patients_by_event, native_roster, native_patient_index = (
        _event_patient_inputs(
            native_event_ids,
            native_public_patient_ids,
            event_count=native_logits.shape[0],
            label="native evaluation",
        )
    )
    if native_roster != _manifest_roster(
        manifest, "native_evaluation_public_patient_ids"
    ):
        raise ValueError("Native tensor patient roster differs from production run")

    if not isinstance(training_targets, torch.Tensor) or not isinstance(
        training_target_mask, torch.Tensor
    ):
        raise TypeError("Training prevalence inputs must be tensors")
    if training_targets.ndim != 3 or training_targets.shape[1] != 20 or tuple(
        training_target_mask.shape
    ) != tuple(training_targets.shape):
        raise ValueError("Training prevalence targets/mask must have shape [E,20,T]")
    dummy_logits = torch.zeros(
        (*training_targets.shape, 1), dtype=training_targets.dtype, device=training_targets.device
    )
    _, training_values, training_mask = _canonical_native_tensors(
        dummy_logits,
        training_targets,
        training_target_mask,
        label="training prevalence",
    )
    training_events, training_patients_by_event, training_roster, _ = (
        _event_patient_inputs(
            training_event_ids,
            training_public_patient_ids,
            event_count=training_values.shape[0],
            label="training prevalence",
        )
    )
    if training_roster != _manifest_roster(
        manifest, "training_source_public_patient_ids"
    ):
        raise ValueError("Training prevalence roster differs from producer fit roster")
    training_observed = training_values[training_mask]
    training_positive = int(training_observed.sum().item())
    training_total = int(training_observed.numel())
    if training_positive < 1 or training_positive >= training_total:
        raise ValueError("Training prevalence baseline requires explicit both-class support")
    prevalence = training_positive / training_total

    full_metrics = patient_macro_ictal_metrics(
        native_logits,
        native_values,
        native_mask,
        native_patient_index,
    )
    run_metrics, run_mean_loss, run_n_events = _run_metrics(manifest)
    if not _metrics_close(full_metrics, run_metrics):
        raise ValueError("Native prediction tensor does not replay production metrics")
    if run_n_events != len(native_events):
        raise ValueError("Native prediction event roster changed")

    prevalence_logit = math.log(prevalence / (1.0 - prevalence))
    baseline_logits = torch.full_like(native_logits, prevalence_logit)
    baseline_metrics = patient_macro_ictal_metrics(
        baseline_logits,
        native_values,
        native_mask,
        native_patient_index,
    )
    support_audit = audit_ictal_source_support(
        native_values,
        native_mask,
        event_ids=native_events,
        patient_ids=native_patients_by_event,
    )
    support = NativeClassSupportReceipt(
        event_count=len(support_audit.events),
        patient_count=len(support_audit.patients),
        positive_label_count=support_audit.positive_labels,
        negative_label_count=support_audit.explicit_negative_labels,
        unknown_label_count=int(native_mask.numel() - native_mask.sum().item()),
        positive_event_count=sum(row.positive_labels > 0 for row in support_audit.events),
        negative_event_count=sum(
            row.explicit_negative_labels > 0 for row in support_audit.events
        ),
        mixed_class_event_count=support_audit.events_with_both_classes,
        positive_patient_count=sum(
            row.positive_labels > 0 for row in support_audit.patients
        ),
        negative_patient_count=sum(
            row.explicit_negative_labels > 0 for row in support_audit.patients
        ),
        mixed_class_patient_count=support_audit.patients_with_both_classes,
    )
    native_targets_sha = _tensor_sha256("native_targets", native_values)
    native_mask_sha = _tensor_sha256("native_target_mask", native_mask)
    support_input_sha = _canonical_sha256(
        {
            "native_event_ids": native_events,
            "native_public_patient_ids_by_event": native_patients_by_event,
            "native_targets_sha256": native_targets_sha,
            "native_target_mask_sha256": native_mask_sha,
        }
    )
    support_receipt = IctalNativeSupportReceipt(
        selection=selection,
        production_run_manifest_sha256=production_run.manifest_sha256,
        native_evaluation_manifest_sha256=_require_sha256(
            manifest.get("native_evaluation_manifest_sha256"),
            field="native_evaluation_manifest_sha256",
        ),
        native_evaluation_corpus_index_sha256=_require_sha256(
            manifest.get("native_evaluation_corpus_index_sha256"),
            field="native_evaluation_corpus_index_sha256",
        ),
        native_public_patient_ids=native_roster,
        native_public_patient_roster_sha256=source_roster_sha256(native_roster),
        support_audit_input_sha256=support_input_sha,
        support=support,
    )
    full_logits_sha = _tensor_sha256("full_native_logits", native_logits)
    training_targets_sha = _tensor_sha256("training_targets", training_values)
    training_mask_sha = _tensor_sha256("training_target_mask", training_mask)
    training_input_sha = _canonical_sha256(
        {
            "training_event_ids": training_events,
            "training_public_patient_ids_by_event": training_patients_by_event,
            "training_targets_sha256": training_targets_sha,
            "training_target_mask_sha256": training_mask_sha,
        }
    )
    fidelity_receipt = IctalNativeFidelityReceipt(
        selection=selection,
        production_run_manifest_sha256=production_run.manifest_sha256,
        checkpoint_manifest_sha256=production_run.checkpoint.manifest_sha256,
        native_evaluation_manifest_sha256=support_receipt.native_evaluation_manifest_sha256,
        native_evaluation_corpus_index_sha256=support_receipt.native_evaluation_corpus_index_sha256,
        native_public_patient_ids=native_roster,
        native_public_patient_roster_sha256=source_roster_sha256(native_roster),
        native_support_receipt_sha256=support_receipt.receipt_sha256,
        native_evaluation_role=str(manifest.get("native_evaluation_role")),
        metrics=full_metrics,
        prevalence_baseline_metrics=baseline_metrics,
        training_explicit_positive_prevalence=prevalence,
        full_native_logits_sha256=full_logits_sha,
        native_targets_sha256=native_targets_sha,
        native_target_mask_sha256=native_mask_sha,
        training_targets_sha256=training_targets_sha,
        training_target_mask_sha256=training_mask_sha,
        training_source_public_roster_sha256=source_roster_sha256(training_roster),
        training_prevalence_input_sha256=training_input_sha,
        mean_patient_loss=run_mean_loss,
        n_events=run_n_events,
    )
    return VerifiedIctalNativeEvaluation(
        _verification_marker=_NATIVE_EVALUATION_MARKER,
        support=support_receipt,
        fidelity=fidelity_receipt,
    )


def verify_ictal_shortcut_prediction_tensors(
    *,
    production_run: LoadedIctalProductionRun,
    native_evaluation: VerifiedIctalNativeEvaluation,
    full_logits: torch.Tensor,
    time_only_logits: torch.Tensor,
    mask_only_logits: torch.Tensor,
    native_targets: torch.Tensor,
    native_target_mask: torch.Tensor,
    native_event_ids: Sequence[object],
    native_public_patient_ids: Sequence[object],
    time_only_control_run_sha256: str,
    mask_only_control_run_sha256: str,
) -> VerifiedIctalShortcutProbe:
    """Compute full/time-only/mask-only metrics on one identical native grid.

    This caller-tensor API remains diagnostic-only and is never accepted
    directly by formal promotion.  Strict native/time-only/mask-only artifact
    loaders invoke it internally after independently replaying their bundles.
    """

    if not isinstance(native_evaluation, VerifiedIctalNativeEvaluation):
        raise TypeError("native_evaluation must be an opaque verified capability")
    native_evaluation.assert_unchanged()
    if not isinstance(production_run, LoadedIctalProductionRun):
        raise TypeError("production_run must be strictly loaded")
    selection = _selection(production_run.manifest.get("selection"))
    if selection != native_evaluation.selection:
        raise ValueError("Shortcut probe uses another native evaluation")
    full, targets, mask = _canonical_native_tensors(
        full_logits, native_targets, native_target_mask, label="shortcut full"
    )
    time_logits, time_targets, time_mask = _canonical_native_tensors(
        time_only_logits, native_targets, native_target_mask, label="shortcut time-only"
    )
    mask_logits, mask_targets, mask_mask = _canonical_native_tensors(
        mask_only_logits, native_targets, native_target_mask, label="shortcut mask-only"
    )
    if not torch.equal(targets, time_targets) or not torch.equal(targets, mask_targets) or not torch.equal(mask, time_mask) or not torch.equal(mask, mask_mask):
        raise ValueError("Shortcut controls changed target/mask cells")
    events, patients_by_event, patient_roster, patient_index = _event_patient_inputs(
        native_event_ids,
        native_public_patient_ids,
        event_count=full.shape[0],
        label="shortcut probe",
    )
    fidelity = native_evaluation.fidelity
    full_sha = _tensor_sha256("full_native_logits", full)
    targets_sha = _tensor_sha256("native_targets", targets)
    mask_sha = _tensor_sha256("native_target_mask", mask)
    if (
        full_sha != fidelity.full_native_logits_sha256
        or targets_sha != fidelity.native_targets_sha256
        or mask_sha != fidelity.native_target_mask_sha256
        or patient_roster != fidelity.native_public_patient_ids
    ):
        raise ValueError("Shortcut tensor grid differs from verified native replay")
    full_metrics = patient_macro_ictal_metrics(full, targets, mask, patient_index)
    if not _metrics_close(full_metrics, fidelity.metrics):
        raise ValueError("Shortcut full prediction does not replay native fidelity")
    time_metrics = patient_macro_ictal_metrics(
        time_logits, targets, mask, patient_index
    )
    mask_metrics = patient_macro_ictal_metrics(
        mask_logits, targets, mask, patient_index
    )
    time_sha = _tensor_sha256("time_only_logits", time_logits)
    shortcut_mask_sha = _tensor_sha256("mask_only_logits", mask_logits)
    time_run_sha = _require_sha256(
        time_only_control_run_sha256, field="time_only_control_run_sha256"
    )
    mask_run_sha = _require_sha256(
        mask_only_control_run_sha256, field="mask_only_control_run_sha256"
    )
    probe_input_sha = _canonical_sha256(
        {
            "event_ids": events,
            "patient_ids_by_event": patients_by_event,
            "full_logits_sha256": full_sha,
            "time_only_logits_sha256": time_sha,
            "mask_only_logits_sha256": shortcut_mask_sha,
            "native_targets_sha256": targets_sha,
            "native_target_mask_sha256": mask_sha,
            "time_only_control_run_sha256": time_run_sha,
            "mask_only_control_run_sha256": mask_run_sha,
        }
    )
    receipt = IctalShortcutProbeReceipt(
        selection=selection,
        production_run_manifest_sha256=production_run.manifest_sha256,
        checkpoint_manifest_sha256=production_run.checkpoint.manifest_sha256,
        native_evaluation_manifest_sha256=fidelity.native_evaluation_manifest_sha256,
        native_evaluation_corpus_index_sha256=fidelity.native_evaluation_corpus_index_sha256,
        native_public_patient_ids=patient_roster,
        native_public_patient_roster_sha256=source_roster_sha256(patient_roster),
        probe_input_sha256=probe_input_sha,
        time_only_control_run_sha256=time_run_sha,
        mask_only_control_run_sha256=mask_run_sha,
        full_logits_sha256=full_sha,
        time_only_logits_sha256=time_sha,
        mask_only_logits_sha256=shortcut_mask_sha,
        native_targets_sha256=targets_sha,
        native_target_mask_sha256=mask_sha,
        evaluated_observed_label_count=full_metrics.n_observed_labels,
        full_model_patient_macro_bce=full_metrics.patient_macro_bce,
        time_only_patient_macro_bce=time_metrics.patient_macro_bce,
        mask_only_patient_macro_bce=mask_metrics.patient_macro_bce,
    )
    return VerifiedIctalShortcutProbe(
        _verification_marker=_SHORTCUT_PROBE_MARKER, receipt=receipt
    )


def _patient_macro_scale_summary(
    scores: torch.Tensor,
    mask: torch.Tensor,
    patient_ids_by_event: tuple[str, ...],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Equal-patient scale quantiles over target-free deployment cells."""

    patient_ids = tuple(sorted(set(patient_ids_by_event)))
    levels = torch.tensor(ICTAL_SCALE_QUANTILE_LEVELS, dtype=torch.float64)
    quantiles: list[torch.Tensor] = []
    counts: list[int] = []
    for patient_id in patient_ids:
        event_index = torch.tensor(
            [
                index
                for index, value in enumerate(patient_ids_by_event)
                if value == patient_id
            ],
            dtype=torch.long,
        )
        patient_values = scores.index_select(0, event_index)
        patient_mask = mask.index_select(0, event_index)
        observed = patient_values[patient_mask].to(torch.float64)
        if observed.numel() < 1:
            raise ValueError("Every shared-probe patient needs observed score cells")
        quantiles.append(torch.quantile(observed, levels))
        counts.append(int(observed.numel()))
    patient_macro = torch.stack(quantiles, dim=0).mean(dim=0)
    return tuple(float(value) for value in patient_macro), tuple(counts)


def verify_ictal_scale_alignment_tensors(
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    shared_source_dev_scores: Mapping[str, torch.Tensor],
    shared_deployment_mask: torch.Tensor,
    source_dev_event_ids: Sequence[object],
) -> VerifiedIctalScaleAlignment:
    """Compute six raw identity-sigmoid score summaries on one dev grid."""

    from .formal_reasoner_pipeline import VerifiedGlobalTimelineContext

    if not isinstance(oof_protocol, IctalConceptOOFProtocolArtifact):
        raise TypeError("oof_protocol must be strictly verified")
    if not isinstance(timeline_context, VerifiedGlobalTimelineContext):
        raise TypeError("timeline_context must be a verified formal capability")
    timeline_context.assert_unchanged()
    runs = _production_runs(production_runs)
    expected_records = tuple(
        record
        for record in timeline_context.event_registry
        if record.model_split == "source_dev"
    )
    expected_event_ids = tuple(record.event_id for record in expected_records)
    events = tuple(str(value).strip() for value in source_dev_event_ids)
    if events != expected_event_ids:
        raise ValueError("Scale probe must equal the complete signal-eligible dev events")
    if not isinstance(shared_deployment_mask, torch.Tensor) or shared_deployment_mask.dtype != torch.bool:
        raise TypeError("shared_deployment_mask must be bool tensor")
    mask = shared_deployment_mask.detach().cpu().contiguous().clone()
    if mask.ndim != 3 or mask.shape[0] != len(events) or mask.shape[1] != 20 or not mask.any():
        raise ValueError("Scale probe mask must have shape [E,20,T] with observed cells")
    if set(shared_source_dev_scores) != set(ICTAL_PROMOTION_SELECTIONS):
        raise ValueError("Scale probe requires exactly six score tensors")
    crosswalk = dict(oof_protocol.protocol.receipt.target_public_crosswalk)
    target_patients = tuple(sorted({record.patient_id for record in expected_records}))
    if any(patient not in crosswalk for patient in target_patients):
        raise ValueError("Scale probe target patient is absent from protocol crosswalk")
    public_patients = tuple(sorted(crosswalk[patient] for patient in target_patients))
    if public_patients != _manifest_roster(
        runs["final"].manifest, "native_evaluation_public_patient_ids"
    ):
        raise ValueError("Scale probe public roster differs from final native dev roster")
    target_by_event = tuple(record.patient_id for record in expected_records)
    public_by_event = tuple(crosswalk[patient] for patient in target_by_event)
    shared_input_sha = _canonical_sha256(
        {
            "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
            "event_registry_sha256": timeline_context.event_registry.manifest_sha256,
            "event_rows": tuple(
                zip(events, target_by_event, public_by_event, strict=True)
            ),
            "deployment_mask_sha256": _tensor_sha256("deployment_mask", mask),
            "source_target_mask_used": False,
        }
    )
    summaries: list[IctalScaleSummary] = []
    for selection in ICTAL_PROMOTION_SELECTIONS:
        scores = shared_source_dev_scores[selection]
        if not isinstance(scores, torch.Tensor) or not scores.is_floating_point() or scores.requires_grad:
            raise TypeError("Scale scores must be detached floating-point tensors")
        value = scores.detach().cpu().contiguous().clone()
        if tuple(value.shape) != tuple(mask.shape):
            raise ValueError("Every scale score tensor must match the shared mask")
        observed = value[mask]
        if (
            not torch.isfinite(value).all()
            or torch.any((value < 0) | (value > 1))
        ):
            raise ValueError("Scale audit scores must be finite identity-sigmoid values")
        quantiles, patient_counts = _patient_macro_scale_summary(
            value,
            mask,
            target_by_event,
        )
        run = runs[selection]
        summaries.append(
            IctalScaleSummary(
                selection=selection,
                production_run_manifest_sha256=run.manifest_sha256,
                checkpoint_manifest_sha256=run.checkpoint.manifest_sha256,
                patient_count=len(target_patients),
                observed_score_count=int(observed.numel()),
                minimum_patient_observed_score_count=min(patient_counts),
                maximum_patient_observed_score_count=max(patient_counts),
                score_quantiles=quantiles,
            )
        )
    receipt = IctalScaleAlignmentReceipt(
        summaries=tuple(summaries),
        shared_probe_public_patient_ids=public_patients,
        shared_probe_public_patient_roster_sha256=source_roster_sha256(public_patients),
        shared_probe_input_sha256=shared_input_sha,
    )
    return VerifiedIctalScaleAlignment(
        _verification_marker=_SCALE_ALIGNMENT_MARKER, receipt=receipt
    )


def _balanced_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    recalls = []
    for label in range(5):
        selected = true == label
        if not np.any(selected):
            raise ValueError("Fold-identity metric requires all five classes")
        recalls.append(float(np.mean(predicted[selected] == label)))
    return float(np.mean(recalls))


def _fixed_ridge_oof_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    probe_splits: np.ndarray,
) -> np.ndarray:
    predictions = np.empty(labels.shape[0], dtype=np.int64)
    for split in range(5):
        test = probe_splits == split
        train = ~test
        if not np.any(test) or int(train.sum()) < 5:
            raise ValueError("Fold-identity probe split lacks train/test patients")
        train_x = features[train]
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalized_train = (train_x - mean) / scale
        normalized_test = (features[test] - mean) / scale
        design_train = np.concatenate(
            [normalized_train, np.ones((normalized_train.shape[0], 1))], axis=1
        )
        design_test = np.concatenate(
            [normalized_test, np.ones((normalized_test.shape[0], 1))], axis=1
        )
        targets = np.eye(5, dtype=np.float64)[labels[train]]
        regularizer = np.eye(design_train.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        matrix = design_train.T @ design_train + regularizer
        weights = np.linalg.solve(matrix, design_train.T @ targets)
        predictions[test] = np.argmax(design_test @ weights, axis=1)
    return predictions


def _fold_identity_statistics(
    matrix: np.ndarray,
    label_array: np.ndarray,
    split_array: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    predictions = _fixed_ridge_oof_predictions(matrix, label_array, split_array)
    observed_ba = _balanced_accuracy(label_array, predictions)
    rng = np.random.default_rng(_FOLD_IDENTITY_PROBE_SEED)
    class_indices = [np.flatnonzero(label_array == fold) for fold in range(5)]
    bootstrap_values = [
        _balanced_accuracy(label_array[sampled], predictions[sampled])
        for sampled in (
            np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in class_indices
                ]
            )
            for _ in range(_FOLD_IDENTITY_BOOTSTRAPS)
        )
    ]
    permutation_values = []
    for _ in range(_FOLD_IDENTITY_PERMUTATIONS):
        permuted = rng.permutation(label_array)
        permuted_predictions = _fixed_ridge_oof_predictions(
            matrix, permuted, split_array
        )
        permutation_values.append(
            _balanced_accuracy(permuted, permuted_predictions)
        )
    bootstrap_array = np.asarray(bootstrap_values)
    permutation_array = np.asarray(permutation_values)
    permutation_p = float(
        (1 + int(np.sum(permutation_array >= observed_ba)))
        / (_FOLD_IDENTITY_PERMUTATIONS + 1)
    )
    return (
        observed_ba,
        float(np.quantile(bootstrap_array, 0.025)),
        float(np.quantile(bootstrap_array, 0.975)),
        float(permutation_array.mean()),
        float(np.quantile(permutation_array, 0.95)),
        permutation_p,
    )


def _masked_patient_fold_identity_features(
    scores: torch.Tensor,
    deployment_mask: torch.Tensor,
    phase_mask: torch.Tensor,
    patient_ids_by_event: tuple[str, ...],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Build the frozen 52-D patient rows from actual reasoner-visible I scores."""

    if (
        scores.dtype != torch.float32
        or scores.ndim != 3
        or tuple(scores.shape[1:]) != (20, 60)
        or not torch.isfinite(scores).all()
        or torch.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("Fold-ID scores must be finite float32 [E,20,60]")
    if (
        deployment_mask.dtype != torch.bool
        or tuple(deployment_mask.shape) != tuple(scores.shape)
        or phase_mask.dtype != torch.bool
        or tuple(phase_mask.shape) != (scores.shape[0], 15)
    ):
        raise ValueError("Fold-ID deployment/phase masks have invalid shape")
    if len(patient_ids_by_event) != scores.shape[0]:
        raise ValueError("Fold-ID event patients do not align with score rows")
    tile_values = scores.reshape(scores.shape[0], 20, 15, 4)
    evidence = torch.stack(
        (tile_values.mean(dim=-1), tile_values.amax(dim=-1)), dim=-1
    ).to(torch.float64)
    tile_available = deployment_mask.reshape(
        deployment_mask.shape[0], 20, 15, 4
    ).all(dim=-1)
    visible = tile_available & phase_mask[:, None, :]
    patients = tuple(sorted(set(patient_ids_by_event)))
    rows: list[torch.Tensor] = []
    quantile_levels = torch.tensor(
        ICTAL_SCALE_QUANTILE_LEVELS, dtype=torch.float64
    )
    for patient_id in patients:
        indices = torch.tensor(
            [
                index
                for index, value in enumerate(patient_ids_by_event)
                if value == patient_id
            ],
            dtype=torch.long,
        )
        patient_evidence = evidence.index_select(0, indices)
        patient_visible = visible.index_select(0, indices)
        denominator = patient_visible.sum(dim=(0, 2)).to(torch.float64)
        if torch.any(denominator < 1):
            raise ValueError(
                "Every fold-ID patient/edge needs reasoner-visible score support"
            )
        edge_summary = (
            (
                patient_evidence
                * patient_visible.unsqueeze(-1).to(torch.float64)
            ).sum(dim=(0, 2))
            / denominator.unsqueeze(-1)
        ).reshape(-1)
        distribution_summary: list[torch.Tensor] = []
        for feature_index in range(2):
            observed = patient_evidence[..., feature_index][patient_visible]
            if observed.numel() < 2:
                raise ValueError("Fold-ID patient has insufficient visible scores")
            distribution_summary.append(torch.quantile(observed, quantile_levels))
            distribution_summary.append(observed.std(unbiased=False).reshape(1))
        row = torch.cat((edge_summary, *distribution_summary), dim=0)
        if row.numel() != ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION:
            raise RuntimeError("Frozen fold-ID feature width changed")
        rows.append(row)
    matrix = torch.stack(rows, dim=0).numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("Fold-ID patient feature matrix is non-finite")
    return patients, matrix


def _verify_ictal_fold_identity_score_grid(
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    source_train_scores: torch.Tensor,
    deployment_mask: torch.Tensor,
    phase_mask: torch.Tensor,
    source_train_event_ids: Sequence[object],
) -> VerifiedIctalFoldIdentityProbe:
    """Issue the diagnostic receipt from a complete replayed OOF score grid.

    Formal promotion never accepts this capability directly.  The strict
    artifact loader in :mod:`ictal_prediction_artifacts` wraps it only after
    exact checkpoint/token replay and immutable bundle verification.
    """

    from .formal_reasoner_pipeline import VerifiedGlobalTimelineContext

    if not isinstance(oof_protocol, IctalConceptOOFProtocolArtifact):
        raise TypeError("oof_protocol must be strictly verified")
    if not isinstance(timeline_context, VerifiedGlobalTimelineContext):
        raise TypeError("timeline_context must be a verified formal capability")
    timeline_context.assert_unchanged()
    runs = _production_runs(production_runs)
    expected_records = tuple(
        record
        for record in timeline_context.event_registry
        if record.model_split == "source_train"
    )
    events = tuple(str(value).strip() for value in source_train_event_ids)
    if events != tuple(record.event_id for record in expected_records):
        raise ValueError("Fold-ID score grid must equal all signal-eligible train events")
    scores = source_train_scores.detach().cpu().to(torch.float32).contiguous().clone()
    mask = deployment_mask.detach().cpu().to(torch.bool).contiguous().clone()
    phases = phase_mask.detach().cpu().to(torch.bool).contiguous().clone()
    if tuple(scores.shape) != (len(events), 20, 60):
        raise ValueError("Fold-ID score grid must have shape [E,20,60]")
    if tuple(mask.shape) != tuple(scores.shape) or not mask.all():
        raise ValueError(
            "Signal-eligible full19 fold-ID deployment mask must cover every cell"
        )
    expected_phases = torch.stack(
        [timeline_context.phase_mask(event_id) for event_id in events], dim=0
    ).to(torch.bool)
    if not torch.equal(phases, expected_phases):
        raise ValueError("Fold-ID phase mask differs from verified timeline replay")

    protocol = oof_protocol.protocol
    if protocol.receipt.receipt_sha256 != next(
        iter(runs.values())
    ).manifest.get("oof_protocol_receipt_sha256") or any(
        run.manifest.get("oof_protocol_receipt_sha256")
        != protocol.receipt.receipt_sha256
        or run.manifest.get("oof_protocol_artifact_sha256")
        != oof_protocol.artifact_sha256
        for run in runs.values()
    ):
        raise ValueError("Fold-ID producers use another OOF protocol")
    crosswalk = dict(protocol.receipt.target_public_crosswalk)
    fold_by_target = {
        patient_id: fold
        for fold, plan in enumerate(protocol.fold_plans)
        for patient_id in plan.held_out_target_patient_ids
    }
    actual_targets = tuple(sorted({record.patient_id for record in expected_records}))
    if any(
        patient not in crosswalk or patient not in fold_by_target
        for patient in actual_targets
    ):
        raise ValueError("Signal-eligible source-train patient lacks protocol lineage")
    assignments = tuple(
        (patient, crosswalk[patient], fold_by_target[patient])
        for patient in actual_targets
    )
    for target, public, fold in assignments:
        held = set(
            _manifest_roster(
                runs[f"fold{fold}"].manifest,
                "held_out_exclusion_public_patient_ids",
            )
        )
        if public not in held:
            raise ValueError("Actual OOF patient is not excluded by its producer")
    complete_targets = tuple(protocol.receipt.source_train_patient_ids)
    if set(actual_targets) - set(complete_targets):
        raise ValueError("Timeline contains a source-train patient outside OOF protocol")
    attrition_targets = tuple(sorted(set(complete_targets) - set(actual_targets)))
    attrition_public = tuple(sorted(crosswalk[patient] for patient in attrition_targets))
    attrition_sha = _canonical_sha256(
        {
            "oof_protocol_receipt_sha256": protocol.receipt.receipt_sha256,
            "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
            "signal_preflight_receipt_sha256": timeline_context.signal_preflight_receipt_sha256,
            "source_train_signal_attrition_target_ids": attrition_targets,
            "source_train_signal_attrition_public_ids": attrition_public,
        }
    )

    event_patient = tuple(record.patient_id for record in expected_records)
    patients, matrix = _masked_patient_fold_identity_features(
        scores, mask, phases, event_patient
    )
    if patients != actual_targets:
        raise RuntimeError("Fold-ID patient feature order is not canonical")
    labels = np.asarray([fold_by_target[patient] for patient in patients], dtype=np.int64)
    split_by_patient: dict[str, int] = {}
    for fold in range(5):
        fold_patients = tuple(
            patient for patient in patients if fold_by_target[patient] == fold
        )
        if len(fold_patients) < 5:
            raise ValueError("Fold-ID probe requires at least five patients per producer")
        split_by_patient.update(
            {patient: index % 5 for index, patient in enumerate(fold_patients)}
        )
    probe_splits = np.asarray(
        [split_by_patient[patient] for patient in patients], dtype=np.int64
    )
    (
        observed_ba,
        bootstrap_lower,
        bootstrap_upper,
        permutation_mean,
        permutation_upper,
        permutation_p,
    ) = _fold_identity_statistics(matrix, labels, probe_splits)
    event_rows = tuple(
        (
            record.event_id,
            record.patient_id,
            crosswalk[record.patient_id],
            fold_by_target[record.patient_id],
        )
        for record in expected_records
    )
    probe_event_roster_sha = _canonical_sha256(event_rows)
    probe_input_sha = _canonical_sha256(
        {
            "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
            "event_rows": event_rows,
            "scores_sha256": _tensor_sha256("oof_ictal_scores", scores),
            "deployment_mask_sha256": _tensor_sha256("deployment_mask", mask),
            "phase_mask_sha256": _tensor_sha256("ictal_phase_mask", phases),
            "target_public_fold_assignments": assignments,
            "probe_split_assignments": tuple(
                (patient, split_by_patient[patient]) for patient in patients
            ),
            "feature_policy": ICTAL_FOLD_IDENTITY_FEATURE_POLICY,
            "feature_dimension": ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION,
            "source_target_mask_used": False,
        }
    )
    producer_bindings = tuple(
        (
            f"fold{fold}",
            runs[f"fold{fold}"].manifest_sha256,
            runs[f"fold{fold}"].checkpoint.manifest_sha256,
        )
        for fold in range(5)
    )
    receipt = IctalFoldIdentityProbeReceipt(
        producer_bindings=producer_bindings,
        target_public_fold_assignments=assignments,
        target_patient_roster_sha256=source_roster_sha256(actual_targets),
        public_patient_roster_sha256=source_roster_sha256(
            tuple(sorted(public for _, public, _ in assignments))
        ),
        source_train_signal_attrition_target_ids=attrition_targets,
        source_train_signal_attrition_public_ids=attrition_public,
        signal_attrition_receipt_sha256=attrition_sha,
        oof_protocol_artifact_sha256=oof_protocol.artifact_sha256,
        oof_protocol_receipt_sha256=protocol.receipt.receipt_sha256,
        timeline_context_receipt_sha256=timeline_context.receipt_sha256,
        signal_preflight_receipt_sha256=timeline_context.signal_preflight_receipt_sha256,
        event_registry_sha256=timeline_context.event_registry.manifest_sha256,
        probe_event_roster_sha256=probe_event_roster_sha,
        probe_input_sha256=probe_input_sha,
        probe_patient_count=len(actual_targets),
        probe_event_count=len(events),
        probe_feature_dimension=ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION,
        balanced_accuracy=observed_ba,
        bootstrap_lower_95=bootstrap_lower,
        bootstrap_upper_95=bootstrap_upper,
        permutation_null_mean=permutation_mean,
        permutation_null_upper_95=permutation_upper,
        permutation_p_value=permutation_p,
    )
    return VerifiedIctalFoldIdentityProbe(
        _verification_marker=_FOLD_IDENTITY_MARKER, receipt=receipt
    )


def verify_ictal_fold_identity_features(
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    source_train_event_features: torch.Tensor,
    source_train_event_ids: Sequence[object],
) -> VerifiedIctalFoldIdentityProbe:
    """Run a fixed patient-grouped fold-ID probe on actual eligible OOF events."""

    from .formal_reasoner_pipeline import VerifiedGlobalTimelineContext

    if not isinstance(oof_protocol, IctalConceptOOFProtocolArtifact):
        raise TypeError("oof_protocol must be strictly verified")
    if not isinstance(timeline_context, VerifiedGlobalTimelineContext):
        raise TypeError("timeline_context must be a verified formal capability")
    timeline_context.assert_unchanged()
    runs = _production_runs(production_runs)
    expected_records = tuple(
        record
        for record in timeline_context.event_registry
        if record.model_split == "source_train"
    )
    events = tuple(str(value).strip() for value in source_train_event_ids)
    expected_events = tuple(record.event_id for record in expected_records)
    if events != expected_events:
        raise ValueError("Fold-ID probe must equal complete signal-eligible train events")
    if not isinstance(source_train_event_features, torch.Tensor) or not source_train_event_features.is_floating_point() or source_train_event_features.requires_grad:
        raise TypeError("Fold-ID event features must be detached floating point")
    features = source_train_event_features.detach().cpu().to(torch.float64).contiguous()
    if (
        features.ndim != 2
        or features.shape[0] != len(events)
        or features.shape[1] != ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Diagnostic Fold-ID event features must have finite shape [E,52]")

    protocol = oof_protocol.protocol
    crosswalk = dict(protocol.receipt.target_public_crosswalk)
    fold_by_target = {
        patient_id: fold
        for fold, plan in enumerate(protocol.fold_plans)
        for patient_id in plan.held_out_target_patient_ids
    }
    actual_targets = tuple(sorted({record.patient_id for record in expected_records}))
    if any(patient not in crosswalk or patient not in fold_by_target for patient in actual_targets):
        raise ValueError("Signal-eligible source-train patient lacks protocol lineage")
    assignments = tuple(
        (patient, crosswalk[patient], fold_by_target[patient])
        for patient in actual_targets
    )
    for target, public, fold in assignments:
        held = set(
            _manifest_roster(
                runs[f"fold{fold}"].manifest,
                "held_out_exclusion_public_patient_ids",
            )
        )
        if public not in held:
            raise ValueError("Actual OOF patient is not excluded by its producer")
    complete_targets = tuple(protocol.receipt.source_train_patient_ids)
    attrition_targets = tuple(sorted(set(complete_targets) - set(actual_targets)))
    attrition_public = tuple(sorted(crosswalk[patient] for patient in attrition_targets))
    if set(actual_targets) - set(complete_targets):
        raise ValueError("Timeline contains a source-train patient outside OOF protocol")
    attrition_sha = _canonical_sha256(
        {
            "oof_protocol_receipt_sha256": protocol.receipt.receipt_sha256,
            "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
            "signal_preflight_receipt_sha256": timeline_context.signal_preflight_receipt_sha256,
            "source_train_signal_attrition_target_ids": attrition_targets,
            "source_train_signal_attrition_public_ids": attrition_public,
        }
    )

    event_patient = tuple(record.patient_id for record in expected_records)
    patient_features = []
    labels = []
    probe_splits = []
    for fold in range(5):
        fold_patients = tuple(
            patient for patient in actual_targets if fold_by_target[patient] == fold
        )
        if len(fold_patients) < 5:
            raise ValueError("Fold-ID probe requires at least five patients per producer")
        split_by_patient = {
            patient: index % 5 for index, patient in enumerate(fold_patients)
        }
        for patient in fold_patients:
            indices = [
                index for index, value in enumerate(event_patient) if value == patient
            ]
            if not indices:
                raise RuntimeError("Signal-eligible patient has no probe event")
            patient_features.append(features[indices].mean(dim=0).numpy())
            labels.append(fold)
            probe_splits.append(split_by_patient[patient])
    # Reorder features to canonical target-patient order rather than fold order.
    row_by_patient = {}
    cursor = 0
    for fold in range(5):
        for patient in tuple(
            value for value in actual_targets if fold_by_target[value] == fold
        ):
            row_by_patient[patient] = (
                patient_features[cursor], labels[cursor], probe_splits[cursor]
            )
            cursor += 1
    matrix = np.stack([row_by_patient[patient][0] for patient in actual_targets])
    label_array = np.asarray(
        [row_by_patient[patient][1] for patient in actual_targets], dtype=np.int64
    )
    split_array = np.asarray(
        [row_by_patient[patient][2] for patient in actual_targets], dtype=np.int64
    )
    predictions = _fixed_ridge_oof_predictions(matrix, label_array, split_array)
    observed_ba = _balanced_accuracy(label_array, predictions)
    rng = np.random.default_rng(_FOLD_IDENTITY_PROBE_SEED)
    bootstrap_values = []
    class_indices = [np.flatnonzero(label_array == fold) for fold in range(5)]
    for _ in range(_FOLD_IDENTITY_BOOTSTRAPS):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        bootstrap_values.append(
            _balanced_accuracy(label_array[sampled], predictions[sampled])
        )
    permutation_values = []
    for _ in range(_FOLD_IDENTITY_PERMUTATIONS):
        permuted = rng.permutation(label_array)
        permuted_predictions = _fixed_ridge_oof_predictions(
            matrix, permuted, split_array
        )
        permutation_values.append(
            _balanced_accuracy(permuted, permuted_predictions)
        )
    bootstrap_array = np.asarray(bootstrap_values)
    permutation_array = np.asarray(permutation_values)
    permutation_p = float(
        (1 + int(np.sum(permutation_array >= observed_ba)))
        / (_FOLD_IDENTITY_PERMUTATIONS + 1)
    )
    probe_event_roster_sha = _canonical_sha256(events)
    probe_input_sha = _canonical_sha256(
        {
            "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
            "event_ids": events,
            "event_features_sha256": _tensor_sha256("oof_event_features", features),
            "target_public_fold_assignments": assignments,
            "probe_split_assignments": tuple(
                (patient, int(row_by_patient[patient][2]))
                for patient in actual_targets
            ),
        }
    )
    producer_bindings = tuple(
        (
            f"fold{fold}",
            runs[f"fold{fold}"].manifest_sha256,
            runs[f"fold{fold}"].checkpoint.manifest_sha256,
        )
        for fold in range(5)
    )
    receipt = IctalFoldIdentityProbeReceipt(
        producer_bindings=producer_bindings,
        target_public_fold_assignments=assignments,
        target_patient_roster_sha256=source_roster_sha256(actual_targets),
        public_patient_roster_sha256=source_roster_sha256(
            tuple(sorted(public for _, public, _ in assignments))
        ),
        source_train_signal_attrition_target_ids=attrition_targets,
        source_train_signal_attrition_public_ids=attrition_public,
        signal_attrition_receipt_sha256=attrition_sha,
        oof_protocol_artifact_sha256=oof_protocol.artifact_sha256,
        oof_protocol_receipt_sha256=protocol.receipt.receipt_sha256,
        timeline_context_receipt_sha256=timeline_context.receipt_sha256,
        signal_preflight_receipt_sha256=timeline_context.signal_preflight_receipt_sha256,
        event_registry_sha256=timeline_context.event_registry.manifest_sha256,
        probe_event_roster_sha256=probe_event_roster_sha,
        probe_input_sha256=probe_input_sha,
        probe_patient_count=len(actual_targets),
        probe_event_count=len(events),
        probe_feature_dimension=ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION,
        balanced_accuracy=observed_ba,
        bootstrap_lower_95=float(np.quantile(bootstrap_array, 0.025)),
        bootstrap_upper_95=float(np.quantile(bootstrap_array, 0.975)),
        permutation_null_mean=float(permutation_array.mean()),
        permutation_null_upper_95=float(np.quantile(permutation_array, 0.95)),
        permutation_p_value=permutation_p,
    )
    return VerifiedIctalFoldIdentityProbe(
        _verification_marker=_FOLD_IDENTITY_MARKER, receipt=receipt
    )


def promote_ictal_production_runs(
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    verified_native_prediction_artifacts: Sequence[
        VerifiedIctalNativePredictionArtifact
    ],
    verified_time_only_control_artifacts: Sequence[
        VerifiedIctalControlPredictionArtifact
    ],
    verified_mask_only_control_artifacts: Sequence[
        VerifiedIctalControlPredictionArtifact
    ],
    verified_scale_alignment: VerifiedIctalScaleProbeArtifact,
    verified_fold_identity_probe: VerifiedIctalFoldIdentityProbeArtifact,
    gate_policy_artifact: VerifiedIctalPromotionGatePolicyArtifact,
    expected_gate_policy_artifact_sha256: str,
    expected_gate_policy_bundle_receipt_sha256: str,
) -> VerifiedIctalProducerPromotion:
    """Validate every gate and issue the only cache-authorizing ictal receipt."""

    if not isinstance(
        gate_policy_artifact, VerifiedIctalPromotionGatePolicyArtifact
    ):
        raise TypeError(
            "gate_policy_artifact must be issued by the strict policy loader"
        )
    gate_policy_artifact.assert_unchanged()
    if gate_policy_artifact.artifact_sha256 != _require_sha256(
        expected_gate_policy_artifact_sha256,
        field="expected_gate_policy_artifact_sha256",
    ):
        raise ValueError("Ictal promotion gate-policy artifact SHA mismatch")
    if gate_policy_artifact.receipt_sha256 != _require_sha256(
        expected_gate_policy_bundle_receipt_sha256,
        field="expected_gate_policy_bundle_receipt_sha256",
    ):
        raise ValueError("Ictal promotion gate-policy bundle receipt SHA mismatch")
    gate_policy = gate_policy_artifact.policy
    if not isinstance(verified_scale_alignment, VerifiedIctalScaleProbeArtifact):
        raise TypeError(
            "verified_scale_alignment must be a strict replayed artifact"
        )
    if not isinstance(
        verified_fold_identity_probe, VerifiedIctalFoldIdentityProbeArtifact
    ):
        raise TypeError(
            "verified_fold_identity_probe must be a strict replayed artifact"
        )
    verified_scale_alignment.assert_unchanged()
    verified_fold_identity_probe.assert_unchanged()
    scale_alignment_receipt = verified_scale_alignment.receipt
    fold_identity_probe_receipt = verified_fold_identity_probe.receipt

    runs = _production_runs(production_runs)
    for selection, run in runs.items():
        manifest = run.manifest
        bindings = {
            "promotion_gate_policy_artifact_sha256": (
                gate_policy_artifact.artifact_sha256
            ),
            "promotion_gate_policy_bundle_receipt_sha256": (
                gate_policy_artifact.receipt_sha256
            ),
            "promotion_gate_policy_receipt_sha256": (
                gate_policy_artifact.policy_receipt_sha256
            ),
            "promotion_gate_policy_document_sha256": (
                gate_policy_artifact.policy_document_sha256
            ),
        }
        for field, expected in bindings.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"{selection} production run is not bound to the frozen "
                    f"ictal gate policy: {field}"
                )
    native_artifact_by_selection = _index_receipts(
        verified_native_prediction_artifacts,
        VerifiedIctalNativePredictionArtifact,
        label="verified_native_prediction_artifacts",
    )
    time_control_by_selection = _index_receipts(
        verified_time_only_control_artifacts,
        VerifiedIctalControlPredictionArtifact,
        label="verified_time_only_control_artifacts",
    )
    mask_control_by_selection = _index_receipts(
        verified_mask_only_control_artifacts,
        VerifiedIctalControlPredictionArtifact,
        label="verified_mask_only_control_artifacts",
    )
    native_by_selection: dict[str, VerifiedIctalNativeEvaluation] = {}
    shortcut_capability_by_selection: dict[str, VerifiedIctalShortcutProbe] = {}
    for selection in ICTAL_PROMOTION_SELECTIONS:
        native_artifact = native_artifact_by_selection[selection]
        time_control = time_control_by_selection[selection]
        mask_control = mask_control_by_selection[selection]
        run = runs[selection]
        native_artifact.assert_unchanged()
        if native_artifact.production_run_manifest_sha256 != run.manifest_sha256:
            raise ValueError(
                f"{selection} native prediction uses another production run"
            )
        if time_control.control_type != ICTAL_TIME_ONLY_CONTROL:
            raise ValueError(f"{selection} time-only control type is swapped")
        if mask_control.control_type != ICTAL_MASK_ONLY_CONTROL:
            raise ValueError(f"{selection} mask-only control type is swapped")
        for control in (time_control, mask_control):
            control.assert_unchanged()
            if (
                control.native_prediction_artifact_sha256
                != native_artifact.artifact_sha256
                or control.native_prediction_bundle_receipt_sha256
                != native_artifact.receipt_sha256
                or control.native_grid_receipt_sha256
                != native_artifact.native_grid_receipt_sha256
            ):
                raise ValueError(
                    f"{selection} shortcut control uses another native grid"
                )
        native_evaluation = native_artifact.native_evaluation
        if not isinstance(native_evaluation, VerifiedIctalNativeEvaluation):
            raise TypeError(
                f"{selection} native artifact lacks a replayed evaluation capability"
            )
        native_evaluation.assert_unchanged()
        native_by_selection[selection] = native_evaluation
        shortcut_capability_by_selection[selection] = (
            verified_shortcut_probe_from_artifacts(
                native_prediction=native_artifact,
                time_only_control=time_control,
                mask_only_control=mask_control,
            )
        )
    support_by_selection = {
        selection: capability.support
        for selection, capability in native_by_selection.items()
    }
    fidelity_by_selection = {
        selection: capability.fidelity
        for selection, capability in native_by_selection.items()
    }
    shortcut_by_selection = {
        selection: capability.receipt
        for selection, capability in shortcut_capability_by_selection.items()
    }
    scale_by_selection = {
        summary.selection: summary for summary in scale_alignment_receipt.summaries
    }

    final_probe_roster = _manifest_roster(
        runs["final"].manifest, "native_evaluation_public_patient_ids"
    )
    if scale_alignment_receipt.shared_probe_public_patient_ids != final_probe_roster:
        raise ValueError(
            "Scale alignment must use the exact final source-dev public roster"
        )
    if (
        scale_alignment_receipt.maximum_pairwise_quantile_gap
        > gate_policy.maximum_pairwise_scale_quantile_gap
    ):
        raise ValueError("Ictal cross-producer scale-alignment gate failed")

    expected_fold_bindings: list[tuple[str, str, str]] = []
    for fold in range(5):
        selection = f"fold{fold}"
        run = runs[selection]
        expected_fold_bindings.append(
            (selection, run.manifest_sha256, run.checkpoint.manifest_sha256)
        )
    if tuple(expected_fold_bindings) != fold_identity_probe_receipt.producer_bindings:
        raise ValueError("Fold-identity probe uses the wrong producer bindings")
    for _, public_id, fold in fold_identity_probe_receipt.target_public_fold_assignments:
        held = set(
            _manifest_roster(
                runs[f"fold{fold}"].manifest,
                "held_out_exclusion_public_patient_ids",
            )
        )
        if public_id not in held:
            raise ValueError(
                "Fold-identity probe contains a patient not held out by its producer"
            )
    if fold_identity_probe_receipt.bootstrap_upper_95 > (
        gate_policy.maximum_fold_identity_bootstrap_upper_95
    ):
        raise ValueError("Ictal fold-identity effect-upper-bound gate failed")
    if fold_identity_probe_receipt.permutation_p_value < (
        gate_policy.minimum_fold_identity_permutation_p_value
    ):
        raise ValueError("Ictal fold-identity permutation gate failed")

    rows: list[IctalProducerPromotionRow] = []
    for selection in ICTAL_PROMOTION_SELECTIONS:
        run = runs[selection]
        manifest = run.manifest
        support_receipt = support_by_selection[selection]
        fidelity = fidelity_by_selection[selection]
        shortcut = shortcut_by_selection[selection]
        scale = scale_by_selection[selection]
        native_roster = _manifest_roster(
            manifest, "native_evaluation_public_patient_ids"
        )
        run_metrics, run_mean_loss, run_n_events = _run_metrics(manifest)
        run_sha_fields = (
            support_receipt.production_run_manifest_sha256,
            fidelity.production_run_manifest_sha256,
            shortcut.production_run_manifest_sha256,
            scale.production_run_manifest_sha256,
        )
        if any(value != run.manifest_sha256 for value in run_sha_fields):
            raise ValueError(f"{selection} gate receipt uses another production run")
        checkpoint_fields = (
            fidelity.checkpoint_manifest_sha256,
            shortcut.checkpoint_manifest_sha256,
            scale.checkpoint_manifest_sha256,
        )
        if any(value != run.checkpoint.manifest_sha256 for value in checkpoint_fields):
            raise ValueError(f"{selection} gate receipt uses another checkpoint")
        for receipt in (support_receipt, fidelity, shortcut):
            if receipt.native_evaluation_manifest_sha256 != manifest.get(
                "native_evaluation_manifest_sha256"
            ):
                raise ValueError(f"{selection} native-evaluation manifest binding changed")
            if receipt.native_evaluation_corpus_index_sha256 != manifest.get(
                "native_evaluation_corpus_index_sha256"
            ):
                raise ValueError(f"{selection} native-evaluation corpus binding changed")
            if receipt.native_public_patient_ids != native_roster:
                raise ValueError(f"{selection} native held-patient roster changed")
        if fidelity.native_support_receipt_sha256 != support_receipt.receipt_sha256:
            raise ValueError(f"{selection} fidelity/support receipt binding changed")
        support = support_receipt.support
        if (
            support.event_count != run_n_events
            or support.patient_count != run_metrics.n_patients
            or support.positive_label_count != run_metrics.n_positive_labels
            or support.negative_label_count != run_metrics.n_negative_labels
            or support.positive_label_count + support.negative_label_count
            != run_metrics.n_observed_labels
        ):
            raise ValueError(f"{selection} explicit native support disagrees with metrics")
        if not _metrics_close(fidelity.metrics, run_metrics) or not math.isclose(
            fidelity.mean_patient_loss,
            run_mean_loss,
            rel_tol=0.0,
            abs_tol=1e-7,
        ) or fidelity.n_events != run_n_events:
            raise ValueError(f"{selection} fidelity receipt disagrees with production metrics")
        if fidelity.training_source_public_roster_sha256 != manifest.get(
            "training_source_public_roster_sha256"
        ):
            raise ValueError(f"{selection} prevalence baseline uses another fit roster")
        if fidelity.native_evaluation_role != manifest.get("native_evaluation_role"):
            raise ValueError(f"{selection} native evaluation role changed")
        if shortcut.evaluated_observed_label_count != run_metrics.n_observed_labels:
            raise ValueError(f"{selection} shortcut probe changed observed native cells")
        if shortcut.full_model_patient_macro_bce != run_metrics.patient_macro_bce:
            raise ValueError(f"{selection} shortcut probe changed the full-model metric")
        if (
            shortcut.full_logits_sha256 != fidelity.full_native_logits_sha256
            or shortcut.native_targets_sha256 != fidelity.native_targets_sha256
            or shortcut.native_target_mask_sha256
            != fidelity.native_target_mask_sha256
        ):
            raise ValueError(f"{selection} shortcut probe changed the native tensor grid")
        if (
            shortcut.minimum_control_bce_improvement
            < gate_policy.minimum_shortcut_bce_improvement
        ):
            raise ValueError(f"{selection} time/mask-only shortcut gate failed")
        if run_metrics.patient_macro_bce > gate_policy.maximum_patient_macro_bce:
            raise ValueError(f"{selection} native BCE fidelity gate failed")
        if run_metrics.patient_macro_brier > gate_policy.maximum_patient_macro_brier:
            raise ValueError(f"{selection} native Brier fidelity gate failed")
        if fidelity.patient_macro_bce_improvement_over_prevalence < (
            gate_policy.minimum_patient_macro_bce_improvement_over_prevalence
        ):
            raise ValueError(
                f"{selection} native BCE lift over train-prevalence baseline failed"
            )
        if fidelity.patient_macro_brier_improvement_over_prevalence < (
            gate_policy.minimum_patient_macro_brier_improvement_over_prevalence
        ):
            raise ValueError(
                f"{selection} native Brier lift over train-prevalence baseline failed"
            )
        if selection != "final":
            if not support.class_sensitive_metrics_authorized:
                raise ValueError(
                    f"{selection} lacks explicit held-patient discrimination support"
                )
            ap_lift = fidelity.patient_macro_ap_lift_over_prevalence
            if ap_lift is None or ap_lift < (
                gate_policy.minimum_fold_patient_macro_ap_lift_over_prevalence
            ):
                raise ValueError(
                    f"{selection} native AP lift over prevalence baseline failed"
                )

        rows.append(
            IctalProducerPromotionRow(
                selection=selection,
                production_run_manifest_sha256=run.manifest_sha256,
                checkpoint_manifest_sha256=run.checkpoint.manifest_sha256,
                checkpoint_sha256=run.checkpoint.checkpoint_sha256,
                training_manifest_sha256=_require_sha256(
                    manifest.get("training_manifest_sha256"),
                    field="training_manifest_sha256",
                ),
                training_corpus_index_sha256=_require_sha256(
                    manifest.get("training_corpus_index_sha256"),
                    field="training_corpus_index_sha256",
                ),
                native_evaluation_manifest_sha256=(
                    support_receipt.native_evaluation_manifest_sha256
                ),
                native_evaluation_corpus_index_sha256=(
                    support_receipt.native_evaluation_corpus_index_sha256
                ),
                native_support_receipt_sha256=support_receipt.receipt_sha256,
                native_fidelity_receipt_sha256=fidelity.receipt_sha256,
                shortcut_probe_receipt_sha256=shortcut.receipt_sha256,
                native_class_sensitive_metrics_authorized=(
                    support.class_sensitive_metrics_authorized
                ),
                native_evaluation_role=fidelity.native_evaluation_role,
            )
        )

    receipt = IctalProducerPromotionReceipt(
        gate_policy_receipt_sha256=gate_policy.receipt_sha256,
        producers=tuple(rows),
        scale_alignment_receipt_sha256=scale_alignment_receipt.receipt_sha256,
        fold_identity_probe_receipt_sha256=(
            fold_identity_probe_receipt.receipt_sha256
        ),
    )
    return VerifiedIctalProducerPromotion(
        _verification_marker=_PROMOTION_MARKER,
        receipt=receipt,
    )


def _promotion_artifact_payload(
    *,
    promotion: VerifiedIctalProducerPromotion,
    production_runs: Sequence[LoadedIctalProductionRun],
    native_predictions: Sequence[VerifiedIctalNativePredictionArtifact],
    time_controls: Sequence[VerifiedIctalControlPredictionArtifact],
    mask_controls: Sequence[VerifiedIctalControlPredictionArtifact],
    scale_probe: VerifiedIctalScaleProbeArtifact,
    fold_identity_probe: VerifiedIctalFoldIdentityProbeArtifact,
    gate_policy: VerifiedIctalPromotionGatePolicyArtifact,
) -> dict[str, object]:
    if not isinstance(promotion, VerifiedIctalProducerPromotion):
        raise TypeError("promotion must be a verified six-producer capability")
    promotion.assert_unchanged()
    runs = _production_runs(production_runs)
    native = _index_receipts(
        native_predictions,
        VerifiedIctalNativePredictionArtifact,
        label="native_predictions",
    )
    time_only = _index_receipts(
        time_controls,
        VerifiedIctalControlPredictionArtifact,
        label="time_controls",
    )
    mask_only = _index_receipts(
        mask_controls,
        VerifiedIctalControlPredictionArtifact,
        label="mask_controls",
    )
    if not isinstance(scale_probe, VerifiedIctalScaleProbeArtifact) or not isinstance(
        fold_identity_probe, VerifiedIctalFoldIdentityProbeArtifact
    ):
        raise TypeError("Promotion probes must be strict replayed artifacts")
    if not isinstance(gate_policy, VerifiedIctalPromotionGatePolicyArtifact):
        raise TypeError("gate_policy must be a strict replayed artifact")
    scale_probe.assert_unchanged()
    fold_identity_probe.assert_unchanged()
    gate_policy.assert_unchanged()
    for selection in ICTAL_PROMOTION_SELECTIONS:
        if time_only[selection].control_type != ICTAL_TIME_ONLY_CONTROL:
            raise ValueError("Promotion time-only control type is swapped")
        if mask_only[selection].control_type != ICTAL_MASK_ONLY_CONTROL:
            raise ValueError("Promotion mask-only control type is swapped")
    return {
        "schema_version": ICTAL_PRODUCER_PROMOTION_ARTIFACT_SCHEMA,
        "promotion_receipt_sha256": promotion.receipt_sha256,
        "promotion_receipt": asdict(promotion.receipt),
        "producer_bindings": [
            [
                selection,
                runs[selection].manifest_sha256,
                runs[selection].checkpoint.manifest_sha256,
            ]
            for selection in ICTAL_PROMOTION_SELECTIONS
        ],
        "native_prediction_bindings": [
            [
                selection,
                native[selection].artifact_sha256,
                native[selection].receipt_sha256,
            ]
            for selection in ICTAL_PROMOTION_SELECTIONS
        ],
        "time_only_control_bindings": [
            [
                selection,
                time_only[selection].artifact_sha256,
                time_only[selection].receipt_sha256,
            ]
            for selection in ICTAL_PROMOTION_SELECTIONS
        ],
        "mask_only_control_bindings": [
            [
                selection,
                mask_only[selection].artifact_sha256,
                mask_only[selection].receipt_sha256,
            ]
            for selection in ICTAL_PROMOTION_SELECTIONS
        ],
        "scale_probe_artifact_sha256": scale_probe.artifact_sha256,
        "scale_probe_bundle_receipt_sha256": scale_probe.bundle_receipt_sha256,
        "fold_identity_probe_artifact_sha256": fold_identity_probe.artifact_sha256,
        "fold_identity_probe_bundle_receipt_sha256": (
            fold_identity_probe.bundle_receipt_sha256
        ),
        "gate_policy_artifact_sha256": gate_policy.artifact_sha256,
        "gate_policy_bundle_receipt_sha256": gate_policy.receipt_sha256,
        "output_semantics": ICTAL_PROMOTED_OUTPUT_SEMANTICS,
        "deepsoz_soz_labels_used": False,
        "private_labels_used": False,
        "source_dev_probability_calibration_authorized": False,
        "evidence_cache_materialization_authorized": True,
    }


def _promotion_bundle_receipt(
    artifact_payload: Mapping[str, object], artifact_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": ICTAL_PRODUCER_PROMOTION_BUNDLE_RECEIPT_SCHEMA,
        "artifact_sha256": _require_sha256(
            artifact_sha256, field="artifact_sha256"
        ),
        "promotion_receipt_sha256": _require_sha256(
            artifact_payload.get("promotion_receipt_sha256"),
            field="promotion_receipt_sha256",
        ),
        "gate_policy_artifact_sha256": _require_sha256(
            artifact_payload.get("gate_policy_artifact_sha256"),
            field="gate_policy_artifact_sha256",
        ),
        "scale_probe_artifact_sha256": _require_sha256(
            artifact_payload.get("scale_probe_artifact_sha256"),
            field="scale_probe_artifact_sha256",
        ),
        "fold_identity_probe_artifact_sha256": _require_sha256(
            artifact_payload.get("fold_identity_probe_artifact_sha256"),
            field="fold_identity_probe_artifact_sha256",
        ),
    }


def _strict_promotion_json(raw: bytes, *, label: str) -> dict[str, object]:
    if not 1 <= len(raw) <= 4 * 1024 * 1024:
        raise ValueError(f"{label} has an invalid size")

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains a non-finite constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _safe_promotion_output(path: str | Path) -> Path:
    target = Path(os.path.abspath(path))
    if target.name in {"", ".", ".."}:
        raise ValueError("Ictal promotion output requires a concrete directory")
    for component in (target.parent, *target.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Ictal promotion output cannot traverse symlinks")
    if not target.parent.is_dir():
        raise FileNotFoundError("Ictal promotion output parent does not exist")
    if os.path.lexists(target):
        raise FileExistsError(f"Ictal promotion output already exists: {target}")
    return target


def _fsync_promotion_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_promotion_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, init=False)
class VerifiedIctalProducerPromotionArtifact:
    """Persisted promotion capability reissued only after full gate replay."""

    path: Path
    artifact_sha256: str
    bundle_receipt_sha256: str
    promotion: VerifiedIctalProducerPromotion
    _production_runs: tuple[LoadedIctalProductionRun, ...]
    _native_predictions: tuple[VerifiedIctalNativePredictionArtifact, ...]
    _time_controls: tuple[VerifiedIctalControlPredictionArtifact, ...]
    _mask_controls: tuple[VerifiedIctalControlPredictionArtifact, ...]
    _scale_probe: VerifiedIctalScaleProbeArtifact
    _fold_identity_probe: VerifiedIctalFoldIdentityProbeArtifact
    _gate_policy: VerifiedIctalPromotionGatePolicyArtifact
    _expected_gate_policy_artifact_sha256: str
    _expected_gate_policy_bundle_receipt_sha256: str

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        artifact_sha256: str,
        bundle_receipt_sha256: str,
        promotion: VerifiedIctalProducerPromotion,
        production_runs: Sequence[LoadedIctalProductionRun],
        native_predictions: Sequence[VerifiedIctalNativePredictionArtifact],
        time_controls: Sequence[VerifiedIctalControlPredictionArtifact],
        mask_controls: Sequence[VerifiedIctalControlPredictionArtifact],
        scale_probe: VerifiedIctalScaleProbeArtifact,
        fold_identity_probe: VerifiedIctalFoldIdentityProbeArtifact,
        gate_policy: VerifiedIctalPromotionGatePolicyArtifact,
        expected_gate_policy_artifact_sha256: str,
        expected_gate_policy_bundle_receipt_sha256: str,
    ) -> None:
        if _verification_marker is not _PROMOTION_ARTIFACT_MARKER:
            raise TypeError(
                "VerifiedIctalProducerPromotionArtifact can only be issued "
                "by the strict full-gate loader"
            )
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("Verified promotion artifact path must be absolute")
        if not isinstance(promotion, VerifiedIctalProducerPromotion):
            raise TypeError("promotion must be a verified promotion capability")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "artifact_sha256",
            _require_sha256(artifact_sha256, field="artifact_sha256"),
        )
        object.__setattr__(
            self,
            "bundle_receipt_sha256",
            _require_sha256(
                bundle_receipt_sha256, field="bundle_receipt_sha256"
            ),
        )
        object.__setattr__(self, "promotion", promotion)
        object.__setattr__(self, "_production_runs", tuple(production_runs))
        object.__setattr__(self, "_native_predictions", tuple(native_predictions))
        object.__setattr__(self, "_time_controls", tuple(time_controls))
        object.__setattr__(self, "_mask_controls", tuple(mask_controls))
        object.__setattr__(self, "_scale_probe", scale_probe)
        object.__setattr__(self, "_fold_identity_probe", fold_identity_probe)
        object.__setattr__(self, "_gate_policy", gate_policy)
        object.__setattr__(
            self,
            "_expected_gate_policy_artifact_sha256",
            _require_sha256(
                expected_gate_policy_artifact_sha256,
                field="expected_gate_policy_artifact_sha256",
            ),
        )
        object.__setattr__(
            self,
            "_expected_gate_policy_bundle_receipt_sha256",
            _require_sha256(
                expected_gate_policy_bundle_receipt_sha256,
                field="expected_gate_policy_bundle_receipt_sha256",
            ),
        )

    @property
    def receipt(self) -> IctalProducerPromotionReceipt:
        return self.promotion.receipt

    @property
    def receipt_sha256(self) -> str:
        return self.promotion.receipt_sha256

    def assert_unchanged(self) -> None:
        replay = load_ictal_producer_promotion_artifact(
            self.path,
            production_runs=self._production_runs,
            verified_native_prediction_artifacts=self._native_predictions,
            verified_time_only_control_artifacts=self._time_controls,
            verified_mask_only_control_artifacts=self._mask_controls,
            verified_scale_alignment=self._scale_probe,
            verified_fold_identity_probe=self._fold_identity_probe,
            gate_policy_artifact=self._gate_policy,
            expected_gate_policy_artifact_sha256=(
                self._expected_gate_policy_artifact_sha256
            ),
            expected_gate_policy_bundle_receipt_sha256=(
                self._expected_gate_policy_bundle_receipt_sha256
            ),
            expected_artifact_sha256=self.artifact_sha256,
            expected_receipt_sha256=self.bundle_receipt_sha256,
        )
        if replay.receipt_sha256 != self.receipt_sha256:
            raise ValueError("Verified ictal promotion changed after loading")


def load_ictal_producer_promotion_artifact(
    path: str | Path,
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    verified_native_prediction_artifacts: Sequence[
        VerifiedIctalNativePredictionArtifact
    ],
    verified_time_only_control_artifacts: Sequence[
        VerifiedIctalControlPredictionArtifact
    ],
    verified_mask_only_control_artifacts: Sequence[
        VerifiedIctalControlPredictionArtifact
    ],
    verified_scale_alignment: VerifiedIctalScaleProbeArtifact,
    verified_fold_identity_probe: VerifiedIctalFoldIdentityProbeArtifact,
    gate_policy_artifact: VerifiedIctalPromotionGatePolicyArtifact,
    expected_gate_policy_artifact_sha256: str,
    expected_gate_policy_bundle_receipt_sha256: str,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedIctalProducerPromotionArtifact:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("Ictal promotion artifact must be a canonical directory")
    expected_names = {
        ICTAL_PRODUCER_PROMOTION_ARTIFACT_FILENAME,
        ICTAL_PRODUCER_PROMOTION_BUNDLE_RECEIPT_FILENAME,
    }
    if {entry.name for entry in source.iterdir()} != expected_names:
        raise ValueError("Ictal promotion artifact has missing or unknown files")
    artifact_path = source / ICTAL_PRODUCER_PROMOTION_ARTIFACT_FILENAME
    receipt_path = source / ICTAL_PRODUCER_PROMOTION_BUNDLE_RECEIPT_FILENAME
    if any(
        candidate.is_symlink() or not candidate.is_file()
        for candidate in (artifact_path, receipt_path)
    ):
        raise ValueError("Ictal promotion bundle members must be regular files")
    artifact_raw = artifact_path.read_bytes()
    receipt_raw = receipt_path.read_bytes()
    artifact_sha = hashlib.sha256(artifact_raw).hexdigest()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Ictal promotion artifact SHA mismatch")
    if receipt_sha != _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("Ictal promotion bundle receipt SHA mismatch")
    artifact_payload = _strict_promotion_json(
        artifact_raw, label="ictal promotion artifact"
    )
    receipt_payload = _strict_promotion_json(
        receipt_raw, label="ictal promotion bundle receipt"
    )
    promotion = promote_ictal_production_runs(
        production_runs=production_runs,
        verified_native_prediction_artifacts=(
            verified_native_prediction_artifacts
        ),
        verified_time_only_control_artifacts=(
            verified_time_only_control_artifacts
        ),
        verified_mask_only_control_artifacts=(
            verified_mask_only_control_artifacts
        ),
        verified_scale_alignment=verified_scale_alignment,
        verified_fold_identity_probe=verified_fold_identity_probe,
        gate_policy_artifact=gate_policy_artifact,
        expected_gate_policy_artifact_sha256=(
            expected_gate_policy_artifact_sha256
        ),
        expected_gate_policy_bundle_receipt_sha256=(
            expected_gate_policy_bundle_receipt_sha256
        ),
    )
    expected_payload = _promotion_artifact_payload(
        promotion=promotion,
        production_runs=production_runs,
        native_predictions=verified_native_prediction_artifacts,
        time_controls=verified_time_only_control_artifacts,
        mask_controls=verified_mask_only_control_artifacts,
        scale_probe=verified_scale_alignment,
        fold_identity_probe=verified_fold_identity_probe,
        gate_policy=gate_policy_artifact,
    )
    if artifact_raw != _canonical_json_bytes(expected_payload):
        raise ValueError("Persisted ictal promotion differs from full gate replay")
    if receipt_payload != _promotion_bundle_receipt(expected_payload, artifact_sha):
        raise ValueError("Ictal promotion bundle receipt does not bind its artifact")
    return VerifiedIctalProducerPromotionArtifact(
        _verification_marker=_PROMOTION_ARTIFACT_MARKER,
        path=source,
        artifact_sha256=artifact_sha,
        bundle_receipt_sha256=receipt_sha,
        promotion=promotion,
        production_runs=production_runs,
        native_predictions=verified_native_prediction_artifacts,
        time_controls=verified_time_only_control_artifacts,
        mask_controls=verified_mask_only_control_artifacts,
        scale_probe=verified_scale_alignment,
        fold_identity_probe=verified_fold_identity_probe,
        gate_policy=gate_policy_artifact,
        expected_gate_policy_artifact_sha256=(
            expected_gate_policy_artifact_sha256
        ),
        expected_gate_policy_bundle_receipt_sha256=(
            expected_gate_policy_bundle_receipt_sha256
        ),
    )


def materialize_ictal_producer_promotion_artifact(
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    verified_native_prediction_artifacts: Sequence[
        VerifiedIctalNativePredictionArtifact
    ],
    verified_time_only_control_artifacts: Sequence[
        VerifiedIctalControlPredictionArtifact
    ],
    verified_mask_only_control_artifacts: Sequence[
        VerifiedIctalControlPredictionArtifact
    ],
    verified_scale_alignment: VerifiedIctalScaleProbeArtifact,
    verified_fold_identity_probe: VerifiedIctalFoldIdentityProbeArtifact,
    gate_policy_artifact: VerifiedIctalPromotionGatePolicyArtifact,
    expected_gate_policy_artifact_sha256: str,
    expected_gate_policy_bundle_receipt_sha256: str,
    output_directory: str | Path,
) -> VerifiedIctalProducerPromotionArtifact:
    """Atomically publish a promotion only after replaying every frozen gate."""

    promotion = promote_ictal_production_runs(
        production_runs=production_runs,
        verified_native_prediction_artifacts=(
            verified_native_prediction_artifacts
        ),
        verified_time_only_control_artifacts=(
            verified_time_only_control_artifacts
        ),
        verified_mask_only_control_artifacts=(
            verified_mask_only_control_artifacts
        ),
        verified_scale_alignment=verified_scale_alignment,
        verified_fold_identity_probe=verified_fold_identity_probe,
        gate_policy_artifact=gate_policy_artifact,
        expected_gate_policy_artifact_sha256=(
            expected_gate_policy_artifact_sha256
        ),
        expected_gate_policy_bundle_receipt_sha256=(
            expected_gate_policy_bundle_receipt_sha256
        ),
    )
    artifact_payload = _promotion_artifact_payload(
        promotion=promotion,
        production_runs=production_runs,
        native_predictions=verified_native_prediction_artifacts,
        time_controls=verified_time_only_control_artifacts,
        mask_controls=verified_mask_only_control_artifacts,
        scale_probe=verified_scale_alignment,
        fold_identity_probe=verified_fold_identity_probe,
        gate_policy=gate_policy_artifact,
    )
    artifact_raw = _canonical_json_bytes(artifact_payload)
    artifact_sha = hashlib.sha256(artifact_raw).hexdigest()
    receipt_payload = _promotion_bundle_receipt(artifact_payload, artifact_sha)
    receipt_raw = _canonical_json_bytes(receipt_payload)
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    target = _safe_promotion_output(output_directory)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        artifact_path = temporary / ICTAL_PRODUCER_PROMOTION_ARTIFACT_FILENAME
        receipt_path = (
            temporary / ICTAL_PRODUCER_PROMOTION_BUNDLE_RECEIPT_FILENAME
        )
        artifact_path.write_bytes(artifact_raw)
        receipt_path.write_bytes(receipt_raw)
        _fsync_promotion_file(artifact_path)
        _fsync_promotion_file(receipt_path)
        _fsync_promotion_directory(temporary)
        load_ictal_producer_promotion_artifact(
            temporary,
            production_runs=production_runs,
            verified_native_prediction_artifacts=(
                verified_native_prediction_artifacts
            ),
            verified_time_only_control_artifacts=(
                verified_time_only_control_artifacts
            ),
            verified_mask_only_control_artifacts=(
                verified_mask_only_control_artifacts
            ),
            verified_scale_alignment=verified_scale_alignment,
            verified_fold_identity_probe=verified_fold_identity_probe,
            gate_policy_artifact=gate_policy_artifact,
            expected_gate_policy_artifact_sha256=(
                expected_gate_policy_artifact_sha256
            ),
            expected_gate_policy_bundle_receipt_sha256=(
                expected_gate_policy_bundle_receipt_sha256
            ),
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
        if os.path.lexists(target):
            raise FileExistsError(f"Ictal promotion output already exists: {target}")
        os.rename(temporary, target)
        published = True
        _fsync_promotion_directory(target.parent)
        return load_ictal_producer_promotion_artifact(
            target,
            production_runs=production_runs,
            verified_native_prediction_artifacts=(
                verified_native_prediction_artifacts
            ),
            verified_time_only_control_artifacts=(
                verified_time_only_control_artifacts
            ),
            verified_mask_only_control_artifacts=(
                verified_mask_only_control_artifacts
            ),
            verified_scale_alignment=verified_scale_alignment,
            verified_fold_identity_probe=verified_fold_identity_probe,
            gate_policy_artifact=gate_policy_artifact,
            expected_gate_policy_artifact_sha256=(
                expected_gate_policy_artifact_sha256
            ),
            expected_gate_policy_bundle_receipt_sha256=(
                expected_gate_policy_bundle_receipt_sha256
            ),
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "ICTAL_FORMAL_PROMOTION_BLOCKERS",
    "ICTAL_FOLD_IDENTITY_PROBE_RECEIPT_SCHEMA",
    "ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION",
    "ICTAL_FOLD_IDENTITY_FEATURE_POLICY",
    "ICTAL_NATIVE_FIDELITY_RECEIPT_SCHEMA",
    "ICTAL_NATIVE_SUPPORT_RECEIPT_SCHEMA",
    "ICTAL_PRODUCER_PROMOTION_ARTIFACT_FILENAME",
    "ICTAL_PRODUCER_PROMOTION_ARTIFACT_SCHEMA",
    "ICTAL_PRODUCER_PROMOTION_BUNDLE_RECEIPT_FILENAME",
    "ICTAL_PRODUCER_PROMOTION_BUNDLE_RECEIPT_SCHEMA",
    "ICTAL_PRODUCER_PROMOTION_SCHEMA",
    "ICTAL_PROMOTED_OUTPUT_SEMANTICS",
    "ICTAL_PROMOTION_GATE_POLICY_SCHEMA",
    "ICTAL_PROMOTION_SELECTIONS",
    "ICTAL_SCALE_ALIGNMENT_RECEIPT_SCHEMA",
    "ICTAL_SCALE_QUANTILE_ESTIMATOR",
    "ICTAL_SCALE_QUANTILE_LEVELS",
    "ICTAL_SHORTCUT_PROBE_RECEIPT_SCHEMA",
    "IctalFoldIdentityProbeReceipt",
    "IctalNativeFidelityReceipt",
    "IctalNativeSupportReceipt",
    "IctalProducerPromotionReceipt",
    "IctalProducerPromotionRow",
    "IctalPromotionGatePolicy",
    "IctalScaleAlignmentReceipt",
    "IctalScaleSummary",
    "IctalShortcutProbeReceipt",
    "VerifiedIctalFoldIdentityProbe",
    "VerifiedIctalNativeEvaluation",
    "VerifiedIctalPromotionGatePolicyArtifact",
    "VerifiedIctalProducerPromotion",
    "VerifiedIctalProducerPromotionArtifact",
    "VerifiedIctalScaleAlignment",
    "VerifiedIctalShortcutProbe",
    "load_ictal_producer_promotion_artifact",
    "materialize_ictal_producer_promotion_artifact",
    "promote_ictal_production_runs",
    "verify_ictal_fold_identity_features",
    "verify_ictal_native_evaluation_tensors",
    "verify_ictal_scale_alignment_tensors",
    "verify_ictal_shortcut_prediction_tensors",
]
