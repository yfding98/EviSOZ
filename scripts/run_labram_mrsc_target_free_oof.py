#!/usr/bin/env python3
"""Replay paired C-REF19 through frozen v11.1 folds and run MRSC.

This runner is deliberately outcome-inaccessible.  It opens only the
target-excluding C-CAR19 bridge, the target-free C-REF19 evidence cache, and
the already-frozen v11.1 outer-fold transform/reasoner states.  The fold id is
used solely to route a patient to the matching historical OOF checkpoint; it
is never passed to the MRSC core or used as a feature.

The output is descriptive.  It preserves the C-CAR19 scores bit-for-bit,
materializes paired C-REF19 patient/event scores, and reports uncalibrated
reference-agreement and MRSC uncertainty distributions.  It does not train,
select a model, define a selective threshold, or compute an SOZ outcome
metric.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES  # noqa: E402
from src.soz.mrsc import (  # noqa: E402
    MRSC_CANDIDATE_CHANNELS,
    MRSC_NONCONFORMITY_SEMANTICS,
    MRSC_REPORT_FACT_FIELDS,
    MRSC_SCHEMA,
    MRSC_USE_POLICY,
    assess_mrsc_score_preserving,
)
from src.soz.v11_reasoner import (  # noqa: E402
    FoldFeatureTransform,
    SharedPositiveSetReasoner,
    V11_CANDIDATE_INDICES,
    V11_CANDIDATE_MASK,
    extract_block9_phase_contrasts,
    robust_pool_complete_patient_bags,
)


DEFAULT_ANCHOR_BRIDGE = (
    ROOT / "outputs/labram_v11_1_anchor_target_excluding_20260812"
)
DEFAULT_REF_CACHE = ROOT / "outputs/labram_mrsc_ref19_cache_20260812"
DEFAULT_CAR_PREFIX_CACHE = (
    ROOT / "outputs/public_development_labram_prefix_v11_20260811"
)
DEFAULT_CAR_FINE_CACHE = (
    ROOT / "outputs/public_development_fine_evidence_v11_20260811"
)
DEFAULT_OUTER_STATES = (
    ROOT
    / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811_r2"
    / "outer_fold_states.safetensors"
)
DEFAULT_OUTPUT = ROOT / "outputs/labram_mrsc_target_free_oof_20260812"

SCHEMA = "soz_labram_mrsc_target_free_oof_descriptive_v1"
ANCHOR_BRIDGE_SCHEMA = "soz_labram_v11_1_target_excluding_anchor_bridge_v1"
REF_CACHE_SCHEMA = "soz_labram_mrsc_ref19_target_free_cache_v1"
CAR_PREFIX_SCHEMA = "soz_public_development_labram_block9_prefix_v11_full"
CAR_FINE_SCHEMA = "soz_public_development_fine_evidence_v11_full"
OUTER_FOLDS = (0, 1, 2, 3, 4)
PRIMARY_PATIENT_COUNT = 101
PRIMARY_EVENT_COUNT = 984

BRIDGE_TENSOR_KEYS_READ = (
    "candidate_indices",
    "car_event_scores",
    "car_patient_scores",
    "event_patient_index",
    "patient_event_counts",
    "patient_folds",
)
REF_CACHE_TENSOR_KEYS_READ = (
    "event_patient_index",
    "ref_fine_features",
    "ref_prefix_tokens",
)
CAR_CACHE_TENSOR_KEYS_READ = (
    "prefix_tokens",
    "features",
)
_TRANSFORM_STATE_SUFFIXES = (
    "transform.h_center",
    "transform.h_scale",
    "transform.h_pca_mean",
    "transform.h_components",
    "transform.fine_center",
    "transform.fine_scale",
)
_REASONER_STATE_SUFFIXES = (
    "full_frozen_labram_plus_fine.prior_logits",
    "full_frozen_labram_plus_fine.candidate_mask",
    "full_frozen_labram_plus_fine.h_weight",
    "full_frozen_labram_plus_fine.fine_weight",
)
_FORBIDDEN_STATE_NAME_TOKENS = (
    "target",
    "label",
    "private",
    "accuracy",
    "metric",
    "error",
)


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{name} must be a canonical regular file")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return payload


def _canonical_regular_file(path: Path, *, name: str) -> Path:
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{name} must be a canonical regular file")
    return source


def _read_exact_tensors(
    path: Path,
    *,
    keys: Sequence[str],
    name: str,
) -> dict[str, torch.Tensor]:
    source = _canonical_regular_file(path, name=name)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        expected = set(keys)
        if available != expected:
            raise ValueError(
                f"{name} tensor vocabulary changed: "
                f"missing={sorted(expected - available)}, "
                f"unexpected={sorted(available - expected)}"
            )
        return {key: handle.get_tensor(key).detach() for key in keys}


def _read_selected_tensor(
    path: Path,
    *,
    key: str,
    name: str,
) -> torch.Tensor:
    """Read one named tensor while rejecting outcome/private tensor ports."""

    source = _canonical_regular_file(path, name=name)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        suspicious = sorted(
            candidate
            for candidate in available
            if any(
                token in candidate.lower()
                for token in _FORBIDDEN_STATE_NAME_TOKENS
            )
        )
        if suspicious:
            raise ValueError(
                f"{name} exposes a forbidden outcome/private field: {suspicious}"
            )
        if key not in available:
            raise ValueError(f"{name} lacks required tensor {key!r}")
        return handle.get_tensor(key).detach()


def _required_outer_state_keys() -> tuple[str, ...]:
    return tuple(
        f"outer{fold}.{suffix}"
        for fold in OUTER_FOLDS
        for suffix in (*_TRANSFORM_STATE_SUFFIXES, *_REASONER_STATE_SUFFIXES)
    )


def _read_outer_states(path: Path) -> dict[str, torch.Tensor]:
    source = _canonical_regular_file(path, name="v11.1 outer-fold states")
    required = set(_required_outer_state_keys())
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        suspicious = sorted(
            key
            for key in available
            if any(token in key.lower() for token in _FORBIDDEN_STATE_NAME_TOKENS)
        )
        if suspicious:
            raise ValueError(
                "Outer-state container exposes a forbidden outcome/private field: "
                f"{suspicious}"
            )
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"Outer-state container lacks required fields: {missing}")
        # Other v11.1 model arms may coexist in the state file, but they are
        # never opened.  Only the frozen full-arm fields above cross this port.
        return {key: handle.get_tensor(key).detach() for key in sorted(required)}


def _require_finite_float(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if value.requires_grad or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be detached and finite")
    return value.detach().cpu().contiguous()


def _require_long(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long")
    if tuple(value.shape) != shape or value.requires_grad:
        raise ValueError(f"{name} must be detached with shape {list(shape)}")
    return value.detach().cpu().contiguous()


def _fold_transform(
    states: Mapping[str, torch.Tensor],
    *,
    fold: int,
    train_patient_indices: tuple[int, ...],
) -> FoldFeatureTransform:
    prefix = f"outer{fold}.transform."
    return FoldFeatureTransform(
        h_center=states[f"{prefix}h_center"],
        h_scale=states[f"{prefix}h_scale"],
        h_pca_mean=states[f"{prefix}h_pca_mean"],
        h_components=states[f"{prefix}h_components"],
        fine_center=states[f"{prefix}fine_center"],
        fine_scale=states[f"{prefix}fine_scale"],
        train_patient_indices=train_patient_indices,
    )


def _fold_reasoner(
    states: Mapping[str, torch.Tensor],
    *,
    fold: int,
) -> SharedPositiveSetReasoner:
    prefix = f"outer{fold}.full_frozen_labram_plus_fine."
    state = {
        "prior_logits": states[f"{prefix}prior_logits"],
        "candidate_mask": states[f"{prefix}candidate_mask"],
        "h_weight": states[f"{prefix}h_weight"],
        "fine_weight": states[f"{prefix}fine_weight"],
    }
    if not torch.equal(state["candidate_mask"].cpu(), V11_CANDIDATE_MASK):
        raise ValueError(f"Outer fold {fold} candidate mask changed")
    model = SharedPositiveSetReasoner(
        state["prior_logits"], use_h=True, use_fine=True
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@dataclass(frozen=True)
class FoldwiseReferenceScores:
    patient_scores_19: torch.Tensor
    event_scores_19: torch.Tensor
    patient_event_counts: torch.Tensor
    fold_held_patient_counts: tuple[int, ...]
    fold_held_event_counts: tuple[int, ...]


def compute_foldwise_reference_scores(
    ref_h_event: torch.Tensor,
    ref_fine_event: torch.Tensor,
    event_patient_index: torch.Tensor,
    patient_folds: torch.Tensor,
    expected_patient_event_counts: torch.Tensor,
    outer_states: Mapping[str, torch.Tensor],
) -> FoldwiseReferenceScores:
    """Apply each frozen outer-fold state to its paired C-REF19 held rows.

    This function has no outcome, identity, subgroup, or threshold port.  Fold
    membership is consumed only by checkpoint routing and is not forwarded to
    either the reasoner or MRSC.
    """

    if ref_h_event.ndim != 3 or tuple(ref_h_event.shape[1:]) != (19, 600):
        raise ValueError("ref_h_event must have shape [E,19,600]")
    events = int(ref_h_event.shape[0])
    if events < 1:
        raise ValueError("Reference evidence must contain at least one event")
    ref_h_event = _require_finite_float(
        ref_h_event, name="ref_h_event", shape=(events, 19, 600)
    )
    ref_fine_event = _require_finite_float(
        ref_fine_event, name="ref_fine_event", shape=(events, 19, 20)
    )
    event_patient_index = _require_long(
        event_patient_index, name="event_patient_index", shape=(events,)
    )
    if patient_folds.ndim != 1 or patient_folds.dtype != torch.long:
        raise TypeError("patient_folds must be long [P]")
    patients = int(patient_folds.numel())
    patient_folds = _require_long(
        patient_folds, name="patient_folds", shape=(patients,)
    )
    expected_patient_event_counts = _require_long(
        expected_patient_event_counts,
        name="expected_patient_event_counts",
        shape=(patients,),
    )
    if patients < len(OUTER_FOLDS) or events < patients:
        raise ValueError("Foldwise scoring requires a complete non-empty patient roster")
    if event_patient_index.min().item() != 0 or (
        event_patient_index.max().item() != patients - 1
    ):
        raise ValueError("event_patient_index is not a contiguous patient roster")
    if not set(patient_folds.tolist()).issubset(set(OUTER_FOLDS)) or (
        set(patient_folds.tolist()) != set(OUTER_FOLDS)
    ):
        raise ValueError("Every frozen outer fold must own held patients")

    artifact_index = FINE_TEMPORAL_FEATURE_NAMES.index("artifact_burden_0_12s")
    reference_reliability = (
        1.0 - ref_fine_event[:, :, artifact_index]
    ).clamp(0.0, 1.0)
    pooled_h = robust_pool_complete_patient_bags(
        ref_h_event, event_patient_index, patients, reference_reliability
    )
    pooled_fine = robust_pool_complete_patient_bags(
        ref_fine_event, event_patient_index, patients, reference_reliability
    )
    if not torch.equal(pooled_h.event_counts, pooled_fine.event_counts) or (
        not torch.equal(pooled_h.event_counts.cpu(), expected_patient_event_counts)
    ):
        raise ValueError("C-REF19 event bags differ from the target-excluding bridge")

    patient_scores = torch.full((patients, 19), torch.nan, dtype=torch.float32)
    event_scores = torch.full((events, 19), torch.nan, dtype=torch.float32)
    held_patient_counts: list[int] = []
    held_event_counts: list[int] = []
    with torch.no_grad():
        for fold in OUTER_FOLDS:
            held_patients = torch.nonzero(
                patient_folds == fold, as_tuple=False
            ).flatten()
            train_patients = torch.nonzero(
                patient_folds != fold, as_tuple=False
            ).flatten()
            if held_patients.numel() < 1 or train_patients.numel() < 1:
                raise ValueError(f"Outer fold {fold} lost its OOF partition")
            transform = _fold_transform(
                outer_states,
                fold=fold,
                train_patient_indices=tuple(train_patients.tolist()),
            )
            reasoner = _fold_reasoner(outer_states, fold=fold)
            held_evidence = transform.apply(
                pooled_h.features.index_select(0, held_patients),
                pooled_fine.features.index_select(0, held_patients),
            )
            held_patient_scores = reasoner(held_evidence).logits.detach().cpu()
            patient_scores.index_copy_(0, held_patients, held_patient_scores)

            held_event_mask = torch.isin(event_patient_index, held_patients)
            held_events = torch.nonzero(held_event_mask, as_tuple=False).flatten()
            if held_events.numel() < 1:
                raise ValueError(f"Outer fold {fold} has no paired reference events")
            event_evidence = transform.apply(
                ref_h_event.index_select(0, held_events),
                ref_fine_event.index_select(0, held_events),
            )
            held_event_scores = reasoner(event_evidence).logits.detach().cpu()
            event_scores.index_copy_(0, held_events, held_event_scores)
            held_patient_counts.append(int(held_patients.numel()))
            held_event_counts.append(int(held_events.numel()))

    if not torch.isfinite(patient_scores).all() or not torch.isfinite(event_scores).all():
        raise RuntimeError("Foldwise C-REF19 score matrices are incomplete")
    return FoldwiseReferenceScores(
        patient_scores_19=patient_scores.contiguous(),
        event_scores_19=event_scores.contiguous(),
        patient_event_counts=pooled_h.event_counts.cpu().contiguous(),
        fold_held_patient_counts=tuple(held_patient_counts),
        fold_held_event_counts=tuple(held_event_counts),
    )


def _rowwise_normalized_jsd(
    left_scores: torch.Tensor,
    right_scores: torch.Tensor,
) -> torch.Tensor:
    if tuple(left_scores.shape) != tuple(right_scores.shape) or (
        left_scores.ndim != 2
    ):
        raise ValueError("Paired score matrices must have the same rank-2 shape")
    for name, value in (("left", left_scores), ("right", right_scores)):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} scores must be finite floating point")
    left = torch.softmax(left_scores.detach().cpu().double(), dim=1)
    right = torch.softmax(right_scores.detach().cpu().double(), dim=1)
    midpoint = 0.5 * (left + right)
    value = 0.5 * torch.sum(left * (torch.log(left) - torch.log(midpoint)), dim=1)
    value += 0.5 * torch.sum(
        right * (torch.log(right) - torch.log(midpoint)), dim=1
    )
    return (value / math.log(2.0)).clamp(0.0, 1.0).contiguous()


def _stable_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    if scores.ndim != 2 or not scores.is_floating_point() or (
        not torch.isfinite(scores).all()
    ):
        raise ValueError("scores must be finite floating point [N,C]")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError("k is outside the candidate dimension")
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]


def _topk_jaccard(left: torch.Tensor, right: torch.Tensor, *, k: int) -> torch.Tensor:
    left_top = _stable_topk(left, k)
    right_top = _stable_topk(right, k)
    rows = []
    for left_row, right_row in zip(left_top.tolist(), right_top.tolist()):
        left_set = set(left_row)
        right_set = set(right_row)
        rows.append(len(left_set & right_set) / len(left_set | right_set))
    return torch.tensor(rows, dtype=torch.float64)


@dataclass(frozen=True)
class RosterMRSCAssessment:
    tensors: Mapping[str, torch.Tensor]
    review_reason_vocabulary: tuple[str, ...]
    abstention_reason_vocabulary: tuple[str, ...]


def assess_target_free_roster(
    car_patient_scores: torch.Tensor,
    ref_patient_scores: torch.Tensor,
    car_event_scores: torch.Tensor,
    event_patient_index: torch.Tensor,
    event_quality_valid_mask: torch.Tensor,
    report_fact_available_mask: Mapping[str, bool],
) -> RosterMRSCAssessment:
    """Run the outcome-free MRSC core once per complete patient bag."""

    if car_patient_scores.ndim != 2 or car_patient_scores.shape[1] != 18:
        raise ValueError("car_patient_scores must have shape [P,18]")
    patients = int(car_patient_scores.shape[0])
    events = int(car_event_scores.shape[0])
    for name, value, shape in (
        ("car_patient_scores", car_patient_scores, (patients, 18)),
        ("ref_patient_scores", ref_patient_scores, (patients, 18)),
        ("car_event_scores", car_event_scores, (events, 18)),
    ):
        _require_finite_float(value, name=name, shape=shape)
    event_patient_index = _require_long(
        event_patient_index, name="event_patient_index", shape=(events,)
    )
    if tuple(event_quality_valid_mask.shape) != (events, 18) or (
        event_quality_valid_mask.dtype != torch.bool
    ):
        raise TypeError("event_quality_valid_mask must be bool [E,18]")
    if patients < 1 or events < patients or event_patient_index.min().item() != 0 or (
        event_patient_index.max().item() != patients - 1
    ):
        raise ValueError("MRSC requires a contiguous complete patient/event roster")

    car_patient_before = car_patient_scores.detach().cpu().clone()
    car_event_before = car_event_scores.detach().cpu().clone()
    assessments = []
    for patient in range(patients):
        selected = torch.nonzero(
            event_patient_index == patient, as_tuple=False
        ).flatten()
        if selected.numel() < 1:
            raise ValueError("MRSC lost a patient event bag")
        assessment = assess_mrsc_score_preserving(
            car_patient_scores[patient],
            ref_patient_scores[patient],
            car_event_scores.index_select(0, selected),
            event_quality_valid_mask.index_select(0, selected),
            report_fact_available_mask,
        )
        if not torch.equal(assessment.anchor_scores, car_patient_before[patient]):
            raise RuntimeError("MRSC changed a C-CAR19 patient score")
        assessments.append(assessment)
    if not torch.equal(car_patient_scores.cpu(), car_patient_before) or (
        not torch.equal(car_event_scores.cpu(), car_event_before)
    ):
        raise RuntimeError("MRSC mutated its C-CAR19 input")

    review_vocabulary = tuple(
        sorted({code for row in assessments for code in row.review_reason_codes})
    )
    abstention_vocabulary = tuple(
        sorted({code for row in assessments for code in row.abstention_reason_codes})
    )
    review_flags = torch.tensor(
        [
            [code in row.review_reason_codes for code in review_vocabulary]
            for row in assessments
        ],
        dtype=torch.bool,
    )
    abstention_flags = torch.tensor(
        [
            [code in row.abstention_reason_codes for code in abstention_vocabulary]
            for row in assessments
        ],
        dtype=torch.bool,
    )
    dispersion_estimable = torch.tensor(
        [row.components.within_patient_event_dispersion is not None for row in assessments],
        dtype=torch.bool,
    )
    dispersion = torch.tensor(
        [
            0.0
            if row.components.within_patient_event_dispersion is None
            else row.components.within_patient_event_dispersion
            for row in assessments
        ],
        dtype=torch.float64,
    )
    tensors: dict[str, torch.Tensor] = {
        "car_patient_scores_preserved": torch.stack(
            [row.anchor_scores for row in assessments]
        ).contiguous(),
        "ref_patient_scores": torch.stack(
            [row.sensitivity_scores for row in assessments]
        ).contiguous(),
        "mrsc_car_top1_index": torch.tensor(
            [row.anchor_top1_index for row in assessments], dtype=torch.long
        ),
        "mrsc_ref_top1_index": torch.tensor(
            [row.sensitivity_top1_index for row in assessments], dtype=torch.long
        ),
        "mrsc_top1_reference_agreement": torch.tensor(
            [row.top1_reference_agreement for row in assessments], dtype=torch.bool
        ),
        "mrsc_top3_reference_jaccard": torch.tensor(
            [row.top3_reference_jaccard for row in assessments], dtype=torch.float64
        ),
        "mrsc_ranking_ambiguity": torch.tensor(
            [row.components.ranking_ambiguity for row in assessments],
            dtype=torch.float64,
        ),
        "mrsc_event_dispersion": dispersion,
        "mrsc_event_dispersion_estimable": dispersion_estimable,
        "mrsc_final_score_reference_disagreement": torch.tensor(
            [
                row.components.final_score_reference_disagreement
                for row in assessments
            ],
            dtype=torch.float64,
        ),
        "mrsc_signal_quality_uncertainty": torch.tensor(
            [row.components.signal_quality_uncertainty for row in assessments],
            dtype=torch.float64,
        ),
        "mrsc_report_fact_unavailability": torch.tensor(
            [row.components.report_fact_unavailability for row in assessments],
            dtype=torch.float64,
        ),
        "mrsc_raw_nonconformity": torch.tensor(
            [row.nonconformity for row in assessments], dtype=torch.float64
        ),
        "mrsc_abstain": torch.tensor(
            [row.abstain for row in assessments], dtype=torch.bool
        ),
        "mrsc_review_reason_flags": review_flags,
        "mrsc_abstention_reason_flags": abstention_flags,
    }
    return RosterMRSCAssessment(
        tensors=tensors,
        review_reason_vocabulary=review_vocabulary,
        abstention_reason_vocabulary=abstention_vocabulary,
    )


def _distribution(
    values: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    observed = values.detach().cpu().double().flatten()
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != tuple(values.shape):
            raise TypeError("valid_mask must be bool and match values")
        observed = observed[valid_mask.detach().cpu().flatten()]
    if observed.numel() < 1 or not torch.isfinite(observed).all():
        raise ValueError("A descriptive distribution needs finite observations")
    levels = torch.tensor((0.05, 0.25, 0.50, 0.75, 0.95), dtype=torch.float64)
    quantiles = torch.quantile(observed, levels)
    return {
        "count": int(observed.numel()),
        "mean": float(observed.mean()),
        "minimum": float(observed.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "maximum": float(observed.max()),
    }


def _validate_manifest_pair(
    bridge: Mapping[str, object],
    ref_cache: Mapping[str, object],
    *,
    required_patient_count: int,
    required_event_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if bridge.get("schema_version") != ANCHOR_BRIDGE_SCHEMA or (
        bridge.get("status") != "completed_target_excluding_anchor_bridge"
    ):
        raise ValueError("Unsupported target-excluding anchor bridge")
    if ref_cache.get("schema_version") != REF_CACHE_SCHEMA or (
        ref_cache.get("status") != "completed_target_free_ref19_evidence_cache"
    ):
        raise ValueError("Unsupported target-free C-REF19 cache")
    bridge_access = bridge.get("access_receipt")
    ref_access = ref_cache.get("access_receipt")
    if not isinstance(bridge_access, Mapping) or (
        bridge_access.get("target_tensor_values_loaded") is not False
    ):
        raise ValueError("Anchor bridge is not outcome-excluding")
    if not isinstance(ref_access, Mapping):
        raise TypeError("C-REF19 cache lacks an access receipt")
    for field in (
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "training_performed",
        "model_or_threshold_selection_performed",
    ):
        if ref_access.get(field) is not False:
            raise ValueError(f"C-REF19 cache violates the target-free contract: {field}")

    patient_ids_raw = bridge.get("patient_ids")
    event_ids_raw = bridge.get("event_ids")
    if not isinstance(patient_ids_raw, list) or not isinstance(event_ids_raw, list):
        raise TypeError("Anchor bridge lacks patient/event identities")
    patient_ids = tuple(str(value) for value in patient_ids_raw)
    event_ids = tuple(str(value) for value in event_ids_raw)
    if len(patient_ids) != required_patient_count or (
        len(set(patient_ids)) != required_patient_count
    ):
        raise ValueError("Patient roster size/uniqueness changed")
    if len(event_ids) != required_event_count or (
        len(set(event_ids)) != required_event_count
    ):
        raise ValueError("Event roster size/uniqueness changed")
    if tuple(str(value) for value in ref_cache.get("patient_ids", ())) != patient_ids or (
        tuple(str(value) for value in ref_cache.get("event_ids", ())) != event_ids
    ):
        raise ValueError("C-CAR19 and C-REF19 identity/order differ")
    if tuple(str(value) for value in bridge.get("candidate_channels", ())) != (
        MRSC_CANDIDATE_CHANNELS
    ):
        raise ValueError("Fixed-18 candidate order changed")
    if tuple(str(value) for value in ref_cache.get("fine_feature_names", ())) != (
        FINE_TEMPORAL_FEATURE_NAMES
    ):
        raise ValueError("C-REF19 fine feature vocabulary changed")
    return patient_ids, event_ids


@dataclass(frozen=True)
class TargetFreeCARReplayEvidence:
    selected_prefix: torch.Tensor
    selected_fine: torch.Tensor
    full_prefix: torch.Tensor
    full_fine: torch.Tensor
    full_event_patient_index: torch.Tensor
    full_patient_folds: torch.Tensor
    full_patient_event_counts: torch.Tensor
    selected_patient_indices: torch.Tensor


def _load_target_free_car_replay_evidence(
    *,
    prefix_directory: Path,
    fine_directory: Path,
    selected_event_ids: tuple[str, ...],
    selected_patient_ids: tuple[str, ...],
) -> TargetFreeCARReplayEvidence:
    """Load only CAR block-9/fine evidence needed for fold-state replay."""

    prefix_manifest = _load_json(
        prefix_directory / "manifest.json", name="target-free CAR prefix manifest"
    )
    fine_manifest = _load_json(
        fine_directory / "manifest.json", name="target-free CAR fine manifest"
    )
    if prefix_manifest.get("schema_version") != CAR_PREFIX_SCHEMA:
        raise ValueError("Unsupported target-free CAR prefix cache")
    if fine_manifest.get("schema_version") != CAR_FINE_SCHEMA:
        raise ValueError("Unsupported target-free CAR fine cache")
    for label, manifest in (
        ("CAR prefix", prefix_manifest),
        ("CAR fine", fine_manifest),
    ):
        access = manifest.get("access_receipt")
        if not isinstance(access, Mapping):
            raise TypeError(f"{label} cache lacks an access receipt")
        for field in (
            "deepsoz_target_values_loaded",
            "private_eeg_loaded",
            "private_target_values_loaded",
        ):
            if access.get(field) is not False:
                raise ValueError(f"{label} cache violates target-free contract: {field}")
    prefix_ids = tuple(str(value) for value in prefix_manifest.get("event_ids", ()))
    fine_ids = tuple(str(value) for value in fine_manifest.get("event_ids", ()))
    if not prefix_ids or prefix_ids != fine_ids or len(set(prefix_ids)) != len(prefix_ids):
        raise ValueError("Target-free CAR evidence caches have different event rosters")
    if tuple(str(value) for value in fine_manifest.get("feature_names", ())) != (
        FINE_TEMPORAL_FEATURE_NAMES
    ):
        raise ValueError("Target-free CAR fine vocabulary changed")
    position = {event_id: index for index, event_id in enumerate(prefix_ids)}
    if any(event_id not in position for event_id in selected_event_ids):
        raise ValueError("Target-excluding bridge is not contained in CAR caches")
    selected = torch.tensor(
        [position[event_id] for event_id in selected_event_ids], dtype=torch.long
    )
    prefix_all = _read_selected_tensor(
        prefix_directory / str(prefix_manifest.get("tensor_file")),
        key="prefix_tokens",
        name="target-free CAR prefix tensor",
    )
    fine_all = _read_selected_tensor(
        fine_directory / str(fine_manifest.get("tensor_file")),
        key="features",
        name="target-free CAR fine tensor",
    )
    if tuple(prefix_all.shape) != (len(prefix_ids), 15, 77, 200) or (
        not prefix_all.is_floating_point()
    ) or not torch.isfinite(prefix_all).all():
        raise ValueError("Target-free CAR prefix tensor is invalid")
    if tuple(fine_all.shape) != (len(prefix_ids), 19, 20) or (
        not fine_all.is_floating_point()
    ) or not torch.isfinite(fine_all).all():
        raise ValueError("Target-free CAR fine tensor is invalid")
    raw_events = prefix_manifest.get("events")
    if not isinstance(raw_events, list) or len(raw_events) != len(prefix_ids):
        raise ValueError("Target-free CAR prefix manifest lacks its event roster")
    full_patient_ids: list[str] = []
    full_patient_position: dict[str, int] = {}
    full_event_patient_index: list[int] = []
    fold_by_patient: dict[str, int] = {}
    for expected_event_id, row in zip(prefix_ids, raw_events):
        if not isinstance(row, Mapping) or str(row.get("event_id")) != expected_event_id:
            raise ValueError("Target-free CAR prefix event rows changed order")
        patient_id = str(row.get("patient_id", "")).strip()
        fold = int(row.get("outer_fold", -1))
        if not patient_id or fold not in OUTER_FOLDS:
            raise ValueError("Target-free CAR event has invalid patient/fold routing")
        if patient_id not in full_patient_position:
            full_patient_position[patient_id] = len(full_patient_ids)
            full_patient_ids.append(patient_id)
            fold_by_patient[patient_id] = fold
        elif fold_by_patient[patient_id] != fold:
            raise ValueError("One CAR patient appears in multiple outer folds")
        full_event_patient_index.append(full_patient_position[patient_id])
    if len(full_patient_ids) != int(prefix_manifest.get("patient_count", -1)) or (
        any(patient not in full_patient_position for patient in selected_patient_ids)
    ):
        raise ValueError("Target-free CAR patient roster changed")
    full_epi = torch.tensor(full_event_patient_index, dtype=torch.long)
    full_folds = torch.tensor(
        [fold_by_patient[patient] for patient in full_patient_ids], dtype=torch.long
    )
    return TargetFreeCARReplayEvidence(
        selected_prefix=prefix_all.index_select(0, selected).detach().cpu().contiguous(),
        selected_fine=fine_all.index_select(0, selected).detach().cpu().contiguous(),
        full_prefix=prefix_all.detach().cpu().contiguous(),
        full_fine=fine_all.detach().cpu().contiguous(),
        full_event_patient_index=full_epi,
        full_patient_folds=full_folds,
        full_patient_event_counts=torch.bincount(
            full_epi, minlength=len(full_patient_ids)
        ).long(),
        selected_patient_indices=torch.tensor(
            [full_patient_position[patient] for patient in selected_patient_ids],
            dtype=torch.long,
        ),
    )


def materialize_target_free_mrsc(
    *,
    anchor_bridge_directory: Path,
    ref_cache_directory: Path,
    car_prefix_cache_directory: Path,
    car_fine_cache_directory: Path,
    outer_fold_states_path: Path,
    output_directory: Path,
    required_patient_count: int = PRIMARY_PATIENT_COUNT,
    required_event_count: int = PRIMARY_EVENT_COUNT,
) -> dict[str, object]:
    bridge = _load_json(
        anchor_bridge_directory / "manifest.json", name="anchor bridge manifest"
    )
    ref_manifest = _load_json(
        ref_cache_directory / "manifest.json", name="C-REF19 cache manifest"
    )
    patient_ids, event_ids = _validate_manifest_pair(
        bridge,
        ref_manifest,
        required_patient_count=required_patient_count,
        required_event_count=required_event_count,
    )
    bridge_payload = _read_exact_tensors(
        anchor_bridge_directory / str(bridge.get("tensor_file")),
        keys=BRIDGE_TENSOR_KEYS_READ,
        name="target-excluding anchor tensor",
    )
    ref_payload = _read_exact_tensors(
        ref_cache_directory / str(ref_manifest.get("tensor_file")),
        keys=REF_CACHE_TENSOR_KEYS_READ,
        name="target-free C-REF19 tensor",
    )
    outer_states = _read_outer_states(outer_fold_states_path)

    patients = required_patient_count
    events = required_event_count
    candidate_indices = _require_long(
        bridge_payload["candidate_indices"],
        name="candidate_indices",
        shape=(18,),
    )
    if not torch.equal(
        candidate_indices, torch.tensor(V11_CANDIDATE_INDICES, dtype=torch.long)
    ):
        raise ValueError("Anchor bridge candidate indices changed")
    car_patient_scores = _require_finite_float(
        bridge_payload["car_patient_scores"],
        name="car_patient_scores",
        shape=(patients, 18),
    )
    car_event_scores = _require_finite_float(
        bridge_payload["car_event_scores"],
        name="car_event_scores",
        shape=(events, 18),
    )
    event_patient_index = _require_long(
        bridge_payload["event_patient_index"],
        name="anchor event_patient_index",
        shape=(events,),
    )
    ref_event_patient_index = _require_long(
        ref_payload["event_patient_index"],
        name="reference event_patient_index",
        shape=(events,),
    )
    if not torch.equal(event_patient_index, ref_event_patient_index):
        raise ValueError("C-CAR19 and C-REF19 event-to-patient routing differs")
    patient_event_counts = _require_long(
        bridge_payload["patient_event_counts"],
        name="patient_event_counts",
        shape=(patients,),
    )
    patient_folds = _require_long(
        bridge_payload["patient_folds"], name="patient_folds", shape=(patients,)
    )

    # The bridge was exported from the v11.1 r2 OOF artifact.  Replaying the
    # target-free CAR evidence through the proposed outer states is a hard
    # lineage gate: a directory name or manifest claim alone is insufficient.
    car_evidence = _load_target_free_car_replay_evidence(
        prefix_directory=car_prefix_cache_directory,
        fine_directory=car_fine_cache_directory,
        selected_event_ids=event_ids,
        selected_patient_ids=patient_ids,
    )
    car_fine = car_evidence.selected_fine
    car_h = extract_block9_phase_contrasts(car_evidence.selected_prefix)
    del car_evidence
    car_replay = compute_foldwise_reference_scores(
        car_h,
        car_fine,
        event_patient_index,
        patient_folds,
        patient_event_counts,
        outer_states,
    )
    car_patient_replay = car_replay.patient_scores_19.index_select(
        1, candidate_indices
    ).contiguous()
    car_event_replay = car_replay.event_scores_19.index_select(
        1, candidate_indices
    ).contiguous()
    patient_difference = float((car_patient_replay - car_patient_scores).abs().max())
    event_difference = float((car_event_replay - car_event_scores).abs().max())
    patient_top1_replay_count = int(
        (
            torch.argmax(car_patient_replay, dim=1)
            == torch.argmax(car_patient_scores, dim=1)
        ).sum()
    )
    event_top1_replay_count = int(
        (
            torch.argmax(car_event_replay, dim=1)
            == torch.argmax(car_event_scores, dim=1)
        ).sum()
    )
    replay_tolerance = 1e-6
    if patient_difference > replay_tolerance or (
        event_difference > replay_tolerance
    ) or patient_top1_replay_count != patients or event_top1_replay_count != events:
        raise RuntimeError(
            "v11.1 r2 outer states failed the frozen CAR replay gate: "
            f"patient_max_abs={patient_difference}, event_max_abs={event_difference}"
        )
    del car_h, car_fine, car_patient_replay, car_event_replay

    ref_prefix = _require_finite_float(
        ref_payload["ref_prefix_tokens"],
        name="ref_prefix_tokens",
        shape=(events, 15, 77, 200),
    )
    ref_fine = _require_finite_float(
        ref_payload["ref_fine_features"],
        name="ref_fine_features",
        shape=(events, 19, 20),
    )
    ref_h = extract_block9_phase_contrasts(ref_prefix)
    del ref_prefix, ref_payload

    foldwise = compute_foldwise_reference_scores(
        ref_h,
        ref_fine,
        event_patient_index,
        patient_folds,
        patient_event_counts,
        outer_states,
    )
    if foldwise.fold_held_patient_counts != car_replay.fold_held_patient_counts or (
        foldwise.fold_held_event_counts != car_replay.fold_held_event_counts
    ):
        raise RuntimeError("CAR/REF fold routing receipts differ")
    ref_patient_scores = foldwise.patient_scores_19.index_select(
        1, candidate_indices
    ).contiguous()
    ref_event_scores = foldwise.event_scores_19.index_select(
        1, candidate_indices
    ).contiguous()

    # The cache proves a finite, complete standard-19 computational carrier.
    # It does not qualify artifact thresholds or clinical channel quality.
    # Therefore this mask captures structural availability only and is never
    # presented as a validated artifact/QC decision.
    structural_quality_valid = torch.ones((events, 18), dtype=torch.bool)
    report_facts_unavailable = {
        field: False for field in MRSC_REPORT_FACT_FIELDS
    }
    roster = assess_target_free_roster(
        car_patient_scores,
        ref_patient_scores,
        car_event_scores,
        event_patient_index,
        structural_quality_valid,
        report_facts_unavailable,
    )
    output_tensors = dict(roster.tensors)
    output_tensors.update(
        {
            "candidate_indices": candidate_indices,
            "car_event_scores_preserved": car_event_scores.clone(),
            "ref_event_scores": ref_event_scores,
            "event_patient_index": event_patient_index,
            "patient_event_counts": patient_event_counts,
            "event_structural_quality_valid_mask": structural_quality_valid,
            "report_fact_available_mask": torch.tensor(
                [report_facts_unavailable[field] for field in MRSC_REPORT_FACT_FIELDS],
                dtype=torch.bool,
            ),
        }
    )
    if not torch.equal(
        output_tensors["car_patient_scores_preserved"], car_patient_scores
    ) or not torch.equal(
        output_tensors["car_event_scores_preserved"], car_event_scores
    ):
        raise RuntimeError("C-CAR19 score-parity gate failed")
    if int(output_tensors["mrsc_abstain"].sum()) != patients:
        raise RuntimeError("Undefined-threshold MRSC must fail closed for every patient")

    patient_top1_agreement = output_tensors["mrsc_top1_reference_agreement"]
    car_event_top1 = _stable_topk(car_event_scores, 1).flatten()
    ref_event_top1 = _stable_topk(ref_event_scores, 1).flatten()
    event_top1_agreement = car_event_top1 == ref_event_top1
    event_reference_jsd = _rowwise_normalized_jsd(
        car_event_scores, ref_event_scores
    )
    event_top3_jaccard = _topk_jaccard(
        car_event_scores, ref_event_scores, k=3
    )
    output_tensors.update(
        {
            "event_car_top1_index": car_event_top1,
            "event_ref_top1_index": ref_event_top1,
            "event_top1_reference_agreement": event_top1_agreement,
            "event_top3_reference_jaccard": event_top3_jaccard,
            "event_final_score_reference_disagreement": event_reference_jsd,
        }
    )

    descriptive = {
        "patient_reference_agreement": {
            "top1_agreement_count": int(patient_top1_agreement.sum()),
            "patient_count": patients,
            "top1_agreement_rate": float(patient_top1_agreement.double().mean()),
            "top3_jaccard": _distribution(
                output_tensors["mrsc_top3_reference_jaccard"]
            ),
            "final_score_normalized_jsd": _distribution(
                output_tensors["mrsc_final_score_reference_disagreement"]
            ),
        },
        "event_reference_agreement": {
            "top1_agreement_count": int(event_top1_agreement.sum()),
            "event_count": events,
            "top1_agreement_rate": float(event_top1_agreement.double().mean()),
            "top3_jaccard": _distribution(event_top3_jaccard),
            "final_score_normalized_jsd": _distribution(event_reference_jsd),
        },
        "mrsc_uncertainty": {
            "ranking_ambiguity": _distribution(
                output_tensors["mrsc_ranking_ambiguity"]
            ),
            "within_patient_event_dispersion": _distribution(
                output_tensors["mrsc_event_dispersion"],
                valid_mask=output_tensors["mrsc_event_dispersion_estimable"],
            ),
            "event_dispersion_not_estimable_count": int(
                (~output_tensors["mrsc_event_dispersion_estimable"]).sum()
            ),
            "final_score_reference_disagreement": _distribution(
                output_tensors["mrsc_final_score_reference_disagreement"]
            ),
            "structural_quality_uncertainty": _distribution(
                output_tensors["mrsc_signal_quality_uncertainty"]
            ),
            "report_fact_unavailability": _distribution(
                output_tensors["mrsc_report_fact_unavailability"]
            ),
            "raw_uncalibrated_nonconformity": _distribution(
                output_tensors["mrsc_raw_nonconformity"]
            ),
        },
        "reason_code_counts": {
            "review": {
                code: int(output_tensors["mrsc_review_reason_flags"][:, index].sum())
                for index, code in enumerate(roster.review_reason_vocabulary)
            },
            "abstention": {
                code: int(
                    output_tensors["mrsc_abstention_reason_flags"][:, index].sum()
                )
                for index, code in enumerate(roster.abstention_reason_vocabulary)
            },
        },
        "no_soz_correctness_stratification": True,
        "no_fold_stratification": True,
        "quantiles_are_descriptive_not_operating_thresholds": True,
    }

    target = Path(os.path.abspath(output_directory))
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / "mrsc_target_free.safetensors"
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in output_tensors.items()},
            str(tensor_path),
        )
        manifest: dict[str, object] = {
            "schema_version": SCHEMA,
            "status": "completed_target_free_descriptive_mrsc_threshold_undefined",
            "model_lineage": "frozen_v11_1_outer_fold_full_labram_plus_fine",
            "primary_reference": "C-CAR19_preserved",
            "sensitivity_reference": "C-REF19_same_event_same_frozen_fold_model",
            "patient_count": patients,
            "event_count": events,
            "patient_ids": list(patient_ids),
            "event_ids": list(event_ids),
            "candidate_channels": list(MRSC_CANDIDATE_CHANNELS),
            "tensor_file": tensor_path.name,
            "tensor_specs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(output_tensors.items())
            },
            "fold_checkpoint_routing": {
                "outer_folds": list(OUTER_FOLDS),
                "held_patient_counts": list(foldwise.fold_held_patient_counts),
                "held_event_counts": list(foldwise.fold_held_event_counts),
                "used_only_to_select_historical_oof_state": True,
                "fold_not_passed_to_reasoner_or_mrsc": True,
            },
            "score_parity": {
                "r2_outer_state_car_replay_maximum_allowed_absolute_difference": (
                    replay_tolerance
                ),
                "r2_outer_state_car_patient_replay_maximum_absolute_difference": (
                    patient_difference
                ),
                "r2_outer_state_car_event_replay_maximum_absolute_difference": (
                    event_difference
                ),
                "r2_outer_state_car_patient_top1_replay_count": (
                    patient_top1_replay_count
                ),
                "r2_outer_state_car_event_top1_replay_count": event_top1_replay_count,
                "r2_outer_state_car_replay_gate_passed": True,
                "car_patient_bitwise_equal_before_after_mrsc": True,
                "car_event_bitwise_equal_before_after_mrsc": True,
                "maximum_absolute_car_score_change": 0.0,
                "car_top1_preserved_count": patients,
                "car_top1_preserved_rate": 1.0,
            },
            "mrsc_contract": {
                "core_schema": MRSC_SCHEMA,
                "nonconformity_semantics": MRSC_NONCONFORMITY_SEMANTICS,
                "use_policy": MRSC_USE_POLICY,
                "selective_threshold_defined": False,
                "all_patients_fail_closed": True,
                "report_fact_fields": list(MRSC_REPORT_FACT_FIELDS),
                "report_fact_available_mask": report_facts_unavailable,
                "review_reason_vocabulary": list(roster.review_reason_vocabulary),
                "abstention_reason_vocabulary": list(
                    roster.abstention_reason_vocabulary
                ),
            },
            "quality_contract": {
                "mask_semantics": "finite_complete_structural_carrier_only",
                "artifact_quality_not_materialized": True,
                "patient_artifact_quality_not_repurposed_as_event_channel_qc": True,
                "fine_artifact_feature_used_only_by_frozen_reference_pooling": True,
                "quality_port_cannot_increase_or_rerank_car_scores": True,
            },
            "descriptive_results": descriptive,
            "access_receipt": {
                "target_excluding_anchor_bridge_only": True,
                "target_free_ref19_cache_only": True,
                "target_free_car_prefix_and_fine_cache_only": True,
                "historical_mixed_prediction_container_opened": False,
                "outer_fold_state_checkpoint_opened": True,
                "source_tensor_keys_read": {
                    "anchor_bridge": list(BRIDGE_TENSOR_KEYS_READ),
                    "ref19_cache": list(REF_CACHE_TENSOR_KEYS_READ),
                    "car_replay_caches": list(CAR_CACHE_TENSOR_KEYS_READ),
                    "outer_states": list(_required_outer_state_keys()),
                },
                "target_tensor_values_loaded": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "training_performed": False,
                "optimizer_parameters": 0,
                "model_selection_performed": False,
                "threshold_selection_or_calibration_performed": False,
                "soz_outcome_metrics_computed": False,
                "label_based_subgrouping_performed": False,
                "patient_identity_passed_to_mrsc": False,
                "fold_id_passed_to_mrsc": False,
            },
            "claim_boundary": {
                "developmental_oof_reference_sensitivity_only": True,
                "not_external_validation": True,
                "not_a_new_soz_localizer": True,
                "does_not_improve_or_change_full_cohort_top1": True,
                "reference_disagreement_is_not_error_probability": True,
                "raw_nonconformity_is_not_calibrated_risk": True,
                "cannot_claim_selective_80_or_85_percent_performance": True,
                "private_validation_allowed_from_this_artifact": False,
            },
        }
        with (staging / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(
                manifest,
                stream,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--anchor-bridge-directory", type=Path, default=DEFAULT_ANCHOR_BRIDGE
    )
    parser.add_argument("--ref-cache-directory", type=Path, default=DEFAULT_REF_CACHE)
    parser.add_argument(
        "--car-prefix-cache-directory", type=Path, default=DEFAULT_CAR_PREFIX_CACHE
    )
    parser.add_argument(
        "--car-fine-cache-directory", type=Path, default=DEFAULT_CAR_FINE_CACHE
    )
    parser.add_argument(
        "--outer-fold-states", type=Path, default=DEFAULT_OUTER_STATES
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # Match the frozen v11.1 runner's deterministic CPU reduction contract.
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = parse_args(argv)
    manifest = materialize_target_free_mrsc(
        anchor_bridge_directory=args.anchor_bridge_directory,
        ref_cache_directory=args.ref_cache_directory,
        car_prefix_cache_directory=args.car_prefix_cache_directory,
        car_fine_cache_directory=args.car_fine_cache_directory,
        outer_fold_states_path=args.outer_fold_states,
        output_directory=args.output_directory,
    )
    patient = manifest["descriptive_results"]["patient_reference_agreement"]
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output_directory),
                "patient_count": manifest["patient_count"],
                "event_count": manifest["event_count"],
                "patient_top1_reference_agreement_rate": patient[
                    "top1_agreement_rate"
                ],
                "car_score_change": 0.0,
                "selective_threshold_defined": False,
                "private_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
