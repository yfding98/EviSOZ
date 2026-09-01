"""Target-free scale and producer/fold-ID probes for the v13 gate candidate.

The probes consume only a strict candidate-logit artifact and a separately
verified signal-derived deployment/phase-mask capability.  No target, native
annotation mask, DeepSOZ identity, private data, source-eval capability,
threshold fitting, calibration, or model selection input is accepted.

The current candidate artifact does not contain the required masks.  This
module therefore defines their closed contract but intentionally provides no
public issuer from arbitrary tensors.  Until a signal-preflight-backed loader
is implemented and authorized, the repository-wide ``V13_EXECUTION_HOLD``
remains in force.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from .ictal_gate_prediction_materialization_v13 import (
    EXPECTED_PRODUCER_ORDER,
    LoadedGatePredictionArtifactV13,
    V13_EXECUTION_HOLD,
)
from .ictal_inference_primitives_v13 import patient_roster_sha256
from .ictal_target_free_probe_primitives_v13 import (
    ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION,
    ICTAL_FOLD_IDENTITY_FEATURE_POLICY,
    ICTAL_SCALE_QUANTILE_ESTIMATOR,
    ICTAL_SCALE_QUANTILE_LEVELS,
    masked_patient_fold_identity_features as _masked_patient_fold_identity_features,
    patient_grouped_fold_identity_statistics,
    patient_macro_scale_summary as _patient_macro_scale_summary,
)


V13_TARGET_FREE_MASK_SCHEMA = "soz_labram_k31_i_gate_target_free_mask_v13"
V13_SCALE_PROBE_SCHEMA = "soz_labram_k31_i_gate_scale_probe_v13"
V13_FOLD_ID_PROBE_SCHEMA = "soz_labram_k31_i_gate_fold_id_probe_v13"
V13_TARGET_FREE_PROBE_RECEIPT_SCHEMA = (
    "soz_labram_k31_i_gate_target_free_probe_receipt_v13"
)

V13_SCALE_MAXIMUM_PAIRWISE_QUANTILE_GAP = 0.10
V13_FOLD_ID_MAXIMUM_BOOTSTRAP_UPPER_95 = 0.40
V13_FOLD_ID_MINIMUM_PERMUTATION_P_VALUE = 0.05
V13_FOLD_ID_SEED = 1729
V13_FOLD_ID_BOOTSTRAPS = 1000
V13_FOLD_ID_PERMUTATIONS = 500
V13_PATIENT_GROUPED_CV_POLICY = (
    "five_split_patient_grouped_all_five_producer_rows_colocated_v1"
)
V13_PATIENT_GROUPED_BOOTSTRAP_POLICY = (
    "resample_patients_with_replacement_keep_all_five_producer_rows_v1"
)
V13_PATIENT_GROUPED_PERMUTATION_POLICY = (
    "independently_permute_five_producer_labels_within_each_patient_v1"
)
V13_DEPLOYMENT_MASK_SEMANTICS = (
    "signal_derived_full19_physical_edge_second_availability_no_target_mask_v1"
)
V13_PHASE_MASK_SEMANTICS = (
    "signal_timeline_offset_aware_four_second_phase_visibility_no_target_mask_v1"
)

_SHA256_RE = __import__("re").compile(r"[0-9a-f]{64}")
_MASK_MARKER = object()
_PROBE_MARKER = object()


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
        raise ValueError("Probe receipt is not canonical JSON data") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


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


def _event_rows_from_candidate(
    candidate: LoadedGatePredictionArtifactV13,
) -> tuple[tuple[str, str], ...]:
    events = candidate.manifest["events"]
    return tuple((str(event["event_id"]), str(event["patient_id"])) for event in events)


@dataclass(frozen=True)
class V13TargetFreeSignalMaskReceipt:
    candidate_manifest_sha256: str
    event_rows: tuple[tuple[str, str], ...]
    event_roster_sha256: str
    patient_ids: tuple[str, ...]
    patient_roster_sha256: str
    event_count: int
    patient_count: int
    signal_preflight_artifact_sha256: str
    signal_preflight_receipt_sha256: str
    timeline_context_receipt_sha256: str
    mask_derivation_receipt_sha256: str
    deployment_mask_sha256: str
    phase_mask_sha256: str
    deployment_mask_shape: tuple[int, int, int]
    phase_mask_shape: tuple[int, int]
    deployment_mask_semantics: str = V13_DEPLOYMENT_MASK_SEMANTICS
    phase_mask_semantics: str = V13_PHASE_MASK_SEMANTICS
    mask_values_derived_from_signal_receipt: bool = True
    all_true_fallback_used: bool = False
    source_target_mask_used: bool = False
    tusz_target_values_loaded: bool = False
    tusz_target_masks_loaded: bool = False
    deepsoz_source_loaded: bool = False
    private_source_loaded: bool = False
    schema_version: str = V13_TARGET_FREE_MASK_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "candidate_manifest_sha256",
            "event_roster_sha256",
            "patient_roster_sha256",
            "signal_preflight_artifact_sha256",
            "signal_preflight_receipt_sha256",
            "timeline_context_receipt_sha256",
            "mask_derivation_receipt_sha256",
            "deployment_mask_sha256",
            "phase_mask_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        rows = tuple((str(event).strip(), str(patient).strip()) for event, patient in self.event_rows)
        if (
            not rows
            or any(not event or not patient for event, patient in rows)
            or len({event for event, _ in rows}) != len(rows)
        ):
            raise ValueError("Mask receipt event rows must be non-empty and unique")
        object.__setattr__(self, "event_rows", rows)
        if self.event_roster_sha256 != _canonical_sha256(rows):
            raise ValueError("Mask event-roster receipt mismatch")
        patients = tuple(self.patient_ids)
        if (
            not patients
            or patients != tuple(sorted(patients))
            or len(set(patients)) != len(patients)
            or set(patients) != {patient for _, patient in rows}
        ):
            raise ValueError("Mask patient roster is not canonical or complete")
        if self.patient_roster_sha256 != patient_roster_sha256(patients):
            raise ValueError("Mask patient-roster receipt mismatch")
        if self.event_count != len(rows) or self.patient_count != len(patients):
            raise ValueError("Mask receipt counts disagree with exact rosters")
        expected_deployment = (self.event_count, 20, 60)
        expected_phase = (self.event_count, 15)
        if self.deployment_mask_shape != expected_deployment:
            raise ValueError("Deployment mask must have shape [E,20,60]")
        if self.phase_mask_shape != expected_phase:
            raise ValueError("Phase mask must have shape [E,15]")
        if (
            self.deployment_mask_semantics != V13_DEPLOYMENT_MASK_SEMANTICS
            or self.phase_mask_semantics != V13_PHASE_MASK_SEMANTICS
            or self.mask_values_derived_from_signal_receipt is not True
            or self.all_true_fallback_used is not False
            or self.source_target_mask_used is not False
            or self.tusz_target_values_loaded is not False
            or self.tusz_target_masks_loaded is not False
            or self.deepsoz_source_loaded is not False
            or self.private_source_loaded is not False
        ):
            raise ValueError("Mask receipt violates the signal-only firewall")
        if self.schema_version != V13_TARGET_FREE_MASK_SCHEMA:
            raise ValueError("Unsupported v13 target-free mask schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, init=False)
class VerifiedV13TargetFreeSignalMaskArtifact:
    receipt: V13TargetFreeSignalMaskReceipt
    deployment_mask: torch.Tensor
    phase_mask: torch.Tensor

    def __init__(
        self,
        *,
        _verification_marker: object,
        receipt: V13TargetFreeSignalMaskReceipt,
        deployment_mask: torch.Tensor,
        phase_mask: torch.Tensor,
    ) -> None:
        if _verification_marker is not _MASK_MARKER:
            raise TypeError(
                "Target-free masks require a signal-preflight-backed strict loader"
            )
        if not isinstance(receipt, V13TargetFreeSignalMaskReceipt):
            raise TypeError("receipt must be V13TargetFreeSignalMaskReceipt")
        deployment = deployment_mask.detach().cpu().to(torch.bool).contiguous().clone()
        phase = phase_mask.detach().cpu().to(torch.bool).contiguous().clone()
        if tuple(deployment.shape) != receipt.deployment_mask_shape or not deployment.any():
            raise ValueError("Deployment mask tensor disagrees with its receipt")
        if tuple(phase.shape) != receipt.phase_mask_shape or not phase.any():
            raise ValueError("Phase mask tensor disagrees with its receipt")
        if _tensor_sha256("deployment_mask", deployment) != receipt.deployment_mask_sha256:
            raise ValueError("Deployment mask tensor hash mismatch")
        if _tensor_sha256("phase_mask", phase) != receipt.phase_mask_sha256:
            raise ValueError("Phase mask tensor hash mismatch")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "deployment_mask", deployment)
        object.__setattr__(self, "phase_mask", phase)

    def assert_unchanged(self) -> None:
        if (
            _tensor_sha256("deployment_mask", self.deployment_mask)
            != self.receipt.deployment_mask_sha256
            or _tensor_sha256("phase_mask", self.phase_mask)
            != self.receipt.phase_mask_sha256
        ):
            raise RuntimeError("Verified target-free mask mutated after issuance")


def _issue_v13_target_free_signal_mask_for_synthetic_test(
    *,
    candidate: LoadedGatePredictionArtifactV13,
    deployment_mask: torch.Tensor,
    phase_mask: torch.Tensor,
    signal_preflight_artifact_sha256: str,
    signal_preflight_receipt_sha256: str,
    timeline_context_receipt_sha256: str,
    mask_derivation_receipt_sha256: str,
) -> VerifiedV13TargetFreeSignalMaskArtifact:
    """Private synthetic-test seam; not a production mask issuer."""

    if not isinstance(candidate, LoadedGatePredictionArtifactV13):
        raise TypeError("candidate must be a strict v13 candidate artifact")
    rows = _event_rows_from_candidate(candidate)
    patients = tuple(sorted({patient for _, patient in rows}))
    deployment = deployment_mask.detach().cpu().to(torch.bool).contiguous().clone()
    phase = phase_mask.detach().cpu().to(torch.bool).contiguous().clone()
    receipt = V13TargetFreeSignalMaskReceipt(
        candidate_manifest_sha256=candidate.manifest_sha256,
        event_rows=rows,
        event_roster_sha256=_canonical_sha256(rows),
        patient_ids=patients,
        patient_roster_sha256=patient_roster_sha256(patients),
        event_count=len(rows),
        patient_count=len(patients),
        signal_preflight_artifact_sha256=signal_preflight_artifact_sha256,
        signal_preflight_receipt_sha256=signal_preflight_receipt_sha256,
        timeline_context_receipt_sha256=timeline_context_receipt_sha256,
        mask_derivation_receipt_sha256=mask_derivation_receipt_sha256,
        deployment_mask_sha256=_tensor_sha256("deployment_mask", deployment),
        phase_mask_sha256=_tensor_sha256("phase_mask", phase),
        deployment_mask_shape=tuple(deployment.shape),
        phase_mask_shape=tuple(phase.shape),
    )
    return VerifiedV13TargetFreeSignalMaskArtifact(
        _verification_marker=_MASK_MARKER,
        receipt=receipt,
        deployment_mask=deployment,
        phase_mask=phase,
    )


@dataclass(frozen=True)
class V13GateScaleProducerSummary:
    selection: str
    producer_manifest_sha256: str
    checkpoint_sha256: str
    patient_count: int
    observed_score_count: int
    minimum_patient_observed_score_count: int
    maximum_patient_observed_score_count: int
    score_quantiles: tuple[float, ...]
    quantile_estimator: str = ICTAL_SCALE_QUANTILE_ESTIMATOR

    def __post_init__(self) -> None:
        if self.selection not in EXPECTED_PRODUCER_ORDER:
            raise ValueError("Unknown v13 producer selection")
        _require_sha256(self.producer_manifest_sha256, field="producer_manifest_sha256")
        _require_sha256(self.checkpoint_sha256, field="checkpoint_sha256")
        for field in (
            "patient_count",
            "observed_score_count",
            "minimum_patient_observed_score_count",
            "maximum_patient_observed_score_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be positive")
        values = tuple(_finite(value, field="score_quantile") for value in self.score_quantiles)
        if (
            len(values) != 5
            or any(value > 1.0 for value in values)
            or values != tuple(sorted(values))
        ):
            raise ValueError("Scale summary quantiles are invalid")
        object.__setattr__(self, "score_quantiles", values)
        if self.quantile_estimator != ICTAL_SCALE_QUANTILE_ESTIMATOR:
            raise ValueError("Scale quantile estimator changed")


@dataclass(frozen=True)
class V13GateScaleProbeReceipt:
    candidate_manifest_sha256: str
    mask_receipt_sha256: str
    gate_patient_ids: tuple[str, ...]
    gate_patient_roster_sha256: str
    event_roster_sha256: str
    summaries: tuple[V13GateScaleProducerSummary, ...]
    maximum_pairwise_quantile_gap: float
    maximum_allowed_pairwise_quantile_gap: float
    passed: bool
    quantile_levels: tuple[float, ...] = ICTAL_SCALE_QUANTILE_LEVELS
    all_producers_identical_cells: bool = True
    score_transform: str = "identity_sigmoid_of_raw_head_logit"
    patient_weighting: str = ICTAL_SCALE_QUANTILE_ESTIMATOR
    target_or_private_values_loaded: bool = False
    schema_version: str = V13_SCALE_PROBE_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256(self.candidate_manifest_sha256, field="candidate_manifest_sha256")
        _require_sha256(self.mask_receipt_sha256, field="mask_receipt_sha256")
        _require_sha256(self.event_roster_sha256, field="event_roster_sha256")
        patients = tuple(self.gate_patient_ids)
        if self.gate_patient_roster_sha256 != patient_roster_sha256(patients):
            raise ValueError("Scale gate-patient roster receipt mismatch")
        if tuple(item.selection for item in self.summaries) != EXPECTED_PRODUCER_ORDER:
            raise ValueError("Scale probe requires fold0..fold4, final")
        if {item.patient_count for item in self.summaries} != {len(patients)}:
            raise ValueError("Scale summaries disagree with gate patient count")
        gap = _finite(self.maximum_pairwise_quantile_gap, field="maximum_pairwise_quantile_gap")
        threshold = _finite(
            self.maximum_allowed_pairwise_quantile_gap,
            field="maximum_allowed_pairwise_quantile_gap",
        )
        if threshold != V13_SCALE_MAXIMUM_PAIRWISE_QUANTILE_GAP:
            raise ValueError("Scale gap threshold changed")
        if self.passed is not (gap <= threshold):
            raise ValueError("Scale pass/fail disagrees with frozen threshold")
        if tuple(self.quantile_levels) != ICTAL_SCALE_QUANTILE_LEVELS:
            raise ValueError("Scale quantile levels changed")
        if (
            self.all_producers_identical_cells is not True
            or self.score_transform != "identity_sigmoid_of_raw_head_logit"
            or self.patient_weighting != ICTAL_SCALE_QUANTILE_ESTIMATOR
            or self.target_or_private_values_loaded is not False
            or self.schema_version != V13_SCALE_PROBE_SCHEMA
        ):
            raise ValueError("Scale probe scientific boundary changed")
        object.__setattr__(self, "maximum_pairwise_quantile_gap", gap)
        object.__setattr__(self, "maximum_allowed_pairwise_quantile_gap", threshold)

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class V13GateFoldIdentityProbeReceipt:
    candidate_manifest_sha256: str
    mask_receipt_sha256: str
    producer_bindings: tuple[tuple[str, str, str], ...]
    gate_patient_ids: tuple[str, ...]
    gate_patient_roster_sha256: str
    event_roster_sha256: str
    patient_feature_sha256: str
    probe_patient_count: int
    probe_row_count: int
    probe_feature_dimension: int
    balanced_accuracy: float
    bootstrap_lower_95: float
    bootstrap_upper_95: float
    permutation_null_mean: float
    permutation_null_upper_95: float
    permutation_p_value: float
    maximum_allowed_bootstrap_upper_95: float
    minimum_allowed_permutation_p_value: float
    passed: bool
    seed: int = V13_FOLD_ID_SEED
    bootstrap_count: int = V13_FOLD_ID_BOOTSTRAPS
    permutation_count: int = V13_FOLD_ID_PERMUTATIONS
    probe_algorithm: str = "fixed_l2_multinomial_ridge"
    probe_feature_policy: str = ICTAL_FOLD_IDENTITY_FEATURE_POLICY
    patient_grouped_cv_policy: str = V13_PATIENT_GROUPED_CV_POLICY
    patient_grouped_bootstrap_policy: str = V13_PATIENT_GROUPED_BOOTSTRAP_POLICY
    patient_grouped_permutation_policy: str = V13_PATIENT_GROUPED_PERMUTATION_POLICY
    label_semantics: str = "fixed_producer_axis_fold0_through_fold4"
    final_producer_included: bool = False
    target_or_private_values_loaded: bool = False
    schema_version: str = V13_FOLD_ID_PROBE_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "candidate_manifest_sha256",
            "mask_receipt_sha256",
            "event_roster_sha256",
            "patient_feature_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if tuple(row[0] for row in self.producer_bindings) != EXPECTED_PRODUCER_ORDER[:5]:
            raise ValueError("Fold-ID probe requires fold0..fold4 and excludes final")
        for _, manifest_sha, checkpoint_sha in self.producer_bindings:
            _require_sha256(manifest_sha, field="producer_manifest_sha256")
            _require_sha256(checkpoint_sha, field="checkpoint_sha256")
        patients = tuple(self.gate_patient_ids)
        if self.gate_patient_roster_sha256 != patient_roster_sha256(patients):
            raise ValueError("Fold-ID patient-roster receipt mismatch")
        if (
            self.probe_patient_count != len(patients)
            or self.probe_row_count != len(patients) * 5
            or self.probe_feature_dimension != ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION
        ):
            raise ValueError("Fold-ID probe count or feature dimension changed")
        metric_fields = (
            "balanced_accuracy",
            "bootstrap_lower_95",
            "bootstrap_upper_95",
            "permutation_null_mean",
            "permutation_null_upper_95",
            "permutation_p_value",
        )
        for field in metric_fields:
            value = _finite(getattr(self, field), field=field)
            if value > 1.0:
                raise ValueError(f"{field} must be in [0,1]")
            object.__setattr__(self, field, value)
        if self.bootstrap_lower_95 > self.bootstrap_upper_95:
            raise ValueError("Fold-ID bootstrap interval is reversed")
        if (
            self.maximum_allowed_bootstrap_upper_95
            != V13_FOLD_ID_MAXIMUM_BOOTSTRAP_UPPER_95
            or self.minimum_allowed_permutation_p_value
            != V13_FOLD_ID_MINIMUM_PERMUTATION_P_VALUE
        ):
            raise ValueError("Fold-ID thresholds changed")
        expected_pass = (
            self.bootstrap_upper_95 <= self.maximum_allowed_bootstrap_upper_95
            and self.permutation_p_value >= self.minimum_allowed_permutation_p_value
        )
        if self.passed is not expected_pass:
            raise ValueError("Fold-ID pass/fail disagrees with frozen thresholds")
        if (
            self.seed != V13_FOLD_ID_SEED
            or self.bootstrap_count != V13_FOLD_ID_BOOTSTRAPS
            or self.permutation_count != V13_FOLD_ID_PERMUTATIONS
            or self.probe_algorithm != "fixed_l2_multinomial_ridge"
            or self.probe_feature_policy != ICTAL_FOLD_IDENTITY_FEATURE_POLICY
            or self.patient_grouped_cv_policy != V13_PATIENT_GROUPED_CV_POLICY
            or self.patient_grouped_bootstrap_policy
            != V13_PATIENT_GROUPED_BOOTSTRAP_POLICY
            or self.patient_grouped_permutation_policy
            != V13_PATIENT_GROUPED_PERMUTATION_POLICY
            or self.label_semantics != "fixed_producer_axis_fold0_through_fold4"
            or self.final_producer_included is not False
            or self.target_or_private_values_loaded is not False
            or self.schema_version != V13_FOLD_ID_PROBE_SCHEMA
        ):
            raise ValueError("Fold-ID probe scientific boundary changed")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class V13TargetFreeGateProbeReceipt:
    candidate_manifest_sha256: str
    mask_receipt_sha256: str
    scale_probe: V13GateScaleProbeReceipt
    scale_probe_receipt_sha256: str
    fold_identity_probe: V13GateFoldIdentityProbeReceipt
    fold_identity_probe_receipt_sha256: str
    scale_passed: bool
    fold_identity_passed: bool
    all_target_free_probes_passed: bool
    target_open_authorized_after_failure: bool = False
    tusz_target_values_loaded: bool = False
    tusz_target_masks_loaded: bool = False
    deepsoz_source_loaded: bool = False
    private_source_loaded: bool = False
    source_eval_loaded: bool = False
    complete_stage_a_seal: bool = False
    v13_execution_hold: bool = True
    schema_version: str = V13_TARGET_FREE_PROBE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256(self.candidate_manifest_sha256, field="candidate_manifest_sha256")
        _require_sha256(self.mask_receipt_sha256, field="mask_receipt_sha256")
        if not isinstance(self.scale_probe, V13GateScaleProbeReceipt):
            raise TypeError("scale_probe must be a closed scale receipt")
        if not isinstance(self.fold_identity_probe, V13GateFoldIdentityProbeReceipt):
            raise TypeError("fold_identity_probe must be a closed fold-ID receipt")
        if self.scale_probe.receipt_sha256 != self.scale_probe_receipt_sha256:
            raise ValueError("Scale receipt SHA mismatch")
        if self.fold_identity_probe.receipt_sha256 != self.fold_identity_probe_receipt_sha256:
            raise ValueError("Fold-ID receipt SHA mismatch")
        if (
            self.scale_probe.candidate_manifest_sha256 != self.candidate_manifest_sha256
            or self.fold_identity_probe.candidate_manifest_sha256
            != self.candidate_manifest_sha256
            or self.scale_probe.mask_receipt_sha256 != self.mask_receipt_sha256
            or self.fold_identity_probe.mask_receipt_sha256 != self.mask_receipt_sha256
        ):
            raise ValueError("Probe bundle input identities disagree")
        if (
            self.scale_passed is not self.scale_probe.passed
            or self.fold_identity_passed is not self.fold_identity_probe.passed
            or self.all_target_free_probes_passed
            is not (self.scale_passed and self.fold_identity_passed)
        ):
            raise ValueError("Probe bundle pass/fail is inconsistent")
        if (
            self.target_open_authorized_after_failure is not False
            or self.tusz_target_values_loaded is not False
            or self.tusz_target_masks_loaded is not False
            or self.deepsoz_source_loaded is not False
            or self.private_source_loaded is not False
            or self.source_eval_loaded is not False
            or self.complete_stage_a_seal is not False
            or self.v13_execution_hold is not True
            or V13_EXECUTION_HOLD is not True
            or self.schema_version != V13_TARGET_FREE_PROBE_RECEIPT_SCHEMA
        ):
            raise ValueError("Probe bundle may not open targets or lift v13 HOLD")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, init=False)
class VerifiedV13TargetFreeGateProbes:
    receipt: V13TargetFreeGateProbeReceipt

    def __init__(
        self, *, _verification_marker: object, receipt: V13TargetFreeGateProbeReceipt
    ) -> None:
        if _verification_marker is not _PROBE_MARKER:
            raise TypeError("Verified probes can only be issued by the target-free runner")
        if not isinstance(receipt, V13TargetFreeGateProbeReceipt):
            raise TypeError("receipt must be V13TargetFreeGateProbeReceipt")
        object.__setattr__(self, "receipt", receipt)


def _validate_probe_inputs(
    candidate: LoadedGatePredictionArtifactV13,
    mask_artifact: VerifiedV13TargetFreeSignalMaskArtifact,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(candidate, LoadedGatePredictionArtifactV13):
        raise TypeError("candidate must be a strict v13 candidate artifact")
    manifest = candidate.manifest
    if (
        manifest.get("producer_order") != list(EXPECTED_PRODUCER_ORDER)
        or manifest.get("candidate_logits_present") is not True
        or manifest.get("complete_stage_a_seal") is not False
        or manifest.get("execution_authorized") is not False
        or manifest.get("target_free_scale_probe_passed") is not False
        or manifest.get("target_free_fold_identity_probe_passed") is not False
    ):
        raise ValueError("Input is not the closed candidate-only v13 component")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "tusz_target_values_loaded",
            "tusz_target_masks_loaded",
            "deepsoz_target_source_loaded",
            "deepsoz_soz_values_loaded",
            "private_eeg_loaded",
            "private_targets_loaded",
            "gate_outcomes_opened",
            "evaluation",
        )
    ):
        raise ValueError("Candidate artifact violates the target-free probe firewall")
    if not isinstance(mask_artifact, VerifiedV13TargetFreeSignalMaskArtifact):
        raise TypeError(
            "A verified signal-derived deployment/phase-mask artifact is required; "
            "target masks and all-one fallbacks are forbidden"
        )
    mask_artifact.assert_unchanged()
    if mask_artifact.receipt.candidate_manifest_sha256 != candidate.manifest_sha256:
        raise ValueError("Target-free mask is bound to another candidate artifact")
    rows = _event_rows_from_candidate(candidate)
    if rows != mask_artifact.receipt.event_rows:
        raise ValueError("Candidate event order differs from target-free mask")
    patients_by_event = tuple(patient for _, patient in rows)
    patients = tuple(sorted(set(patients_by_event)))
    if patients != mask_artifact.receipt.patient_ids:
        raise ValueError("Candidate patient roster differs from target-free mask")
    logits = candidate.logits
    if (
        logits.dtype != torch.float32
        or tuple(logits.shape) != (6, len(rows), 20, 60, 1)
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("Candidate logits changed from [6,E,20,60,1] float32")
    return patients_by_event, patients


def compute_v13_gate_scale_probe(
    *,
    candidate: LoadedGatePredictionArtifactV13,
    mask_artifact: VerifiedV13TargetFreeSignalMaskArtifact,
) -> V13GateScaleProbeReceipt:
    patients_by_event, patients = _validate_probe_inputs(candidate, mask_artifact)
    mask = mask_artifact.deployment_mask
    summaries: list[V13GateScaleProducerSummary] = []
    producer_rows = candidate.manifest["producers"]
    for index, selection in enumerate(EXPECTED_PRODUCER_ORDER):
        scores = torch.sigmoid(candidate.logits[index, ..., 0]).to(torch.float32)
        quantiles, counts = _patient_macro_scale_summary(
            scores, mask, patients_by_event
        )
        summaries.append(
            V13GateScaleProducerSummary(
                selection=selection,
                producer_manifest_sha256=str(producer_rows[index]["manifest_sha256"]),
                checkpoint_sha256=str(producer_rows[index]["checkpoint_sha256"]),
                patient_count=len(patients),
                observed_score_count=int(mask.sum().item()),
                minimum_patient_observed_score_count=min(counts),
                maximum_patient_observed_score_count=max(counts),
                score_quantiles=quantiles,
            )
        )
    maximum_gap = max(
        abs(left - right)
        for left_index, left_summary in enumerate(summaries)
        for right_summary in summaries[left_index + 1 :]
        for left, right in zip(
            left_summary.score_quantiles, right_summary.score_quantiles
        )
    )
    return V13GateScaleProbeReceipt(
        candidate_manifest_sha256=candidate.manifest_sha256,
        mask_receipt_sha256=mask_artifact.receipt.receipt_sha256,
        gate_patient_ids=patients,
        gate_patient_roster_sha256=patient_roster_sha256(patients),
        event_roster_sha256=mask_artifact.receipt.event_roster_sha256,
        summaries=tuple(summaries),
        maximum_pairwise_quantile_gap=maximum_gap,
        maximum_allowed_pairwise_quantile_gap=V13_SCALE_MAXIMUM_PAIRWISE_QUANTILE_GAP,
        passed=maximum_gap <= V13_SCALE_MAXIMUM_PAIRWISE_QUANTILE_GAP,
    )


def _patient_grouped_fold_identity_statistics(
    matrix: np.ndarray,
    labels: np.ndarray,
    patient_groups: np.ndarray,
    probe_splits: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    return patient_grouped_fold_identity_statistics(
        matrix,
        labels,
        patient_groups,
        probe_splits,
        seed=V13_FOLD_ID_SEED,
        bootstrap_count=V13_FOLD_ID_BOOTSTRAPS,
        permutation_count=V13_FOLD_ID_PERMUTATIONS,
    )


def compute_v13_gate_fold_identity_probe(
    *,
    candidate: LoadedGatePredictionArtifactV13,
    mask_artifact: VerifiedV13TargetFreeSignalMaskArtifact,
) -> V13GateFoldIdentityProbeReceipt:
    patients_by_event, patients = _validate_probe_inputs(candidate, mask_artifact)
    feature_by_producer: list[np.ndarray] = []
    for producer_index in range(5):
        scores = torch.sigmoid(candidate.logits[producer_index, ..., 0]).to(torch.float32)
        feature_patients, features = _masked_patient_fold_identity_features(
            scores,
            mask_artifact.deployment_mask,
            mask_artifact.phase_mask,
            patients_by_event,
        )
        if feature_patients != patients:
            raise RuntimeError("Fold-ID feature patient order changed")
        feature_by_producer.append(features)
    matrix = np.concatenate(feature_by_producer, axis=0)
    labels = np.repeat(np.arange(5, dtype=np.int64), len(patients))
    patient_groups = np.tile(np.arange(len(patients), dtype=np.int64), 5)
    patient_split = np.asarray(
        [index % 5 for index in range(len(patients))], dtype=np.int64
    )
    probe_splits = np.tile(patient_split, 5)
    statistics = _patient_grouped_fold_identity_statistics(
        matrix, labels, patient_groups, probe_splits
    )
    (
        observed,
        bootstrap_lower,
        bootstrap_upper,
        permutation_mean,
        permutation_upper,
        permutation_p,
    ) = statistics
    passed = (
        bootstrap_upper <= V13_FOLD_ID_MAXIMUM_BOOTSTRAP_UPPER_95
        and permutation_p >= V13_FOLD_ID_MINIMUM_PERMUTATION_P_VALUE
    )
    producer_rows = candidate.manifest["producers"]
    bindings = tuple(
        (
            EXPECTED_PRODUCER_ORDER[index],
            str(producer_rows[index]["manifest_sha256"]),
            str(producer_rows[index]["checkpoint_sha256"]),
        )
        for index in range(5)
    )
    feature_sha = _canonical_sha256(
        {
            "matrix_sha256": hashlib.sha256(
                np.ascontiguousarray(matrix, dtype=np.float64).tobytes()
            ).hexdigest(),
            "labels": tuple(int(value) for value in labels),
            "patient_groups": tuple(int(value) for value in patient_groups),
            "probe_splits": tuple(int(value) for value in probe_splits),
            "feature_policy": ICTAL_FOLD_IDENTITY_FEATURE_POLICY,
        }
    )
    return V13GateFoldIdentityProbeReceipt(
        candidate_manifest_sha256=candidate.manifest_sha256,
        mask_receipt_sha256=mask_artifact.receipt.receipt_sha256,
        producer_bindings=bindings,
        gate_patient_ids=patients,
        gate_patient_roster_sha256=patient_roster_sha256(patients),
        event_roster_sha256=mask_artifact.receipt.event_roster_sha256,
        patient_feature_sha256=feature_sha,
        probe_patient_count=len(patients),
        probe_row_count=matrix.shape[0],
        probe_feature_dimension=matrix.shape[1],
        balanced_accuracy=observed,
        bootstrap_lower_95=bootstrap_lower,
        bootstrap_upper_95=bootstrap_upper,
        permutation_null_mean=permutation_mean,
        permutation_null_upper_95=permutation_upper,
        permutation_p_value=permutation_p,
        maximum_allowed_bootstrap_upper_95=V13_FOLD_ID_MAXIMUM_BOOTSTRAP_UPPER_95,
        minimum_allowed_permutation_p_value=V13_FOLD_ID_MINIMUM_PERMUTATION_P_VALUE,
        passed=passed,
    )


def run_v13_target_free_gate_probes(
    *,
    candidate: LoadedGatePredictionArtifactV13,
    mask_artifact: VerifiedV13TargetFreeSignalMaskArtifact,
) -> VerifiedV13TargetFreeGateProbes:
    """Run both probes and return pass/fail without ever opening a target."""

    scale = compute_v13_gate_scale_probe(
        candidate=candidate, mask_artifact=mask_artifact
    )
    fold = compute_v13_gate_fold_identity_probe(
        candidate=candidate, mask_artifact=mask_artifact
    )
    receipt = V13TargetFreeGateProbeReceipt(
        candidate_manifest_sha256=candidate.manifest_sha256,
        mask_receipt_sha256=mask_artifact.receipt.receipt_sha256,
        scale_probe=scale,
        scale_probe_receipt_sha256=scale.receipt_sha256,
        fold_identity_probe=fold,
        fold_identity_probe_receipt_sha256=fold.receipt_sha256,
        scale_passed=scale.passed,
        fold_identity_passed=fold.passed,
        all_target_free_probes_passed=scale.passed and fold.passed,
    )
    return VerifiedV13TargetFreeGateProbes(
        _verification_marker=_PROBE_MARKER, receipt=receipt
    )


__all__ = (
    "V13_FOLD_ID_BOOTSTRAPS",
    "V13_FOLD_ID_PERMUTATIONS",
    "V13_FOLD_ID_SEED",
    "V13GateFoldIdentityProbeReceipt",
    "V13GateScaleProbeReceipt",
    "V13TargetFreeGateProbeReceipt",
    "V13TargetFreeSignalMaskReceipt",
    "VerifiedV13TargetFreeGateProbes",
    "VerifiedV13TargetFreeSignalMaskArtifact",
    "compute_v13_gate_fold_identity_probe",
    "compute_v13_gate_scale_probe",
    "run_v13_target_free_gate_probes",
)
