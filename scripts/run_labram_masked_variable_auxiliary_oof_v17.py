#!/usr/bin/env python3
"""Run the one-shot masked-variable auxiliary LaBraM v17 OOF experiment.

The official pretrained LaBraM-Base remains frozen.  The stable identity-v16
102-patient endpoint, five outer folds, fold-local transform, Jeffreys prior,
and per-fold L2 are replayed exactly.  Masked-variable patients can affect only
the equally patient-weighted positive-set-mass fit of the shared 36-parameter
reasoner.  Auxiliary patients are train-only and are never evaluated.

This runner intentionally has no private-data argument and performs no final
development/deployment refit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePath
import platform
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import safetensors
from safetensors.torch import load_file, save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_labram_block9_v11_1_failures import (  # noqa: E402
    _accepted as _block9_accepted,
    _top_ties as _block9_top_ties,
    _wrong_hemisphere_far_value as _block9_contralateral_far_value,
)
from scripts.run_labram_fine_temporal_nested_oof_v11 import (  # noqa: E402
    _canonical_bytes,
    _file_sha,
    _fit_reasoner,
    _state_sha,
    _transform_state,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _absolute_bootstrap,
    _evaluate,
    _paired_bootstrap,
    _patient_contributions,
    _require_fixed_rows,
)
import scripts.run_labram_identity_recovery_closed_replay_v16 as identity_v16  # noqa: E402
from src.soz.data.deepsoz_masked_variable_auxiliary_cache_v17 import (  # noqa: E402
    tensor_sha256 as auxiliary_tensor_sha256,
)
from src.soz.data.deepsoz_masked_variable_auxiliary_join import (  # noqa: E402
    MASKED_VARIABLE_AUXILIARY_JOIN_POLICY,
    MASKED_VARIABLE_AUXILIARY_JOIN_SCHEMA,
    PREREGISTERED_AUXILIARY_PATIENT_COUNT,
    VerifiedMaskedVariableAuxiliaryJoin,
    load_masked_variable_auxiliary_join,
)
from src.soz.data.public_development_union_identity_v12 import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256,
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256,
    PublicDevelopmentUnionIdentityV12,
    load_public_development_union_identity_v12,
)
from src.soz.fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES  # noqa: E402
from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import (  # noqa: E402
    FoldFeatureTransform,
    V11_CANDIDATE_MASK,
    apply_fixed_candidate_mask,
    extract_block9_phase_contrasts,
    fit_fold_transform,
    jeffreys_reference_prior_logits,
    robust_pool_complete_patient_bags,
)


SCHEMA = "soz_labram_masked_variable_auxiliary_oof_v17"
CANDIDATE_NAME = "masked_variable_auxiliary_full"
ANCHOR_NAME = "identity_v16_full"
FULL_ARM = "full_frozen_labram_plus_fine"
OUTER_FOLDS = tuple(range(5))
PINNED_FOLD_L2 = (0.01, 0.20, 0.20, 0.01, 0.20)
PARITY_ATOL = 1e-6
PARITY_RTOL = 0.0
STRICT_MINIMUM_NET_GAIN = 5.0
EXPECTED_ANCHOR_STRICT_COUNT = 51.0
EXPECTED_ANCHOR_RELAXED_COUNT = 77.0
EXPECTED_ANCHOR_FAR_COUNT = 25.0
EXPECTED_AUXILIARY_EVENT_COUNT = 182

PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_masked_variable_auxiliary_recovery_protocol_v17_20260812_zh.md"
)
# Update only by freezing a revised protocol and rebuilding every artifact
# whose receipt binds this hash.
EXPECTED_PROTOCOL_SHA256 = (
    "70e63dd000e1a9794ba46a878dee635b10c15dec7f34ec0520180295afda12f7"
)

DEFAULT_UNION = identity_v16.DEFAULT_UNION
DEFAULT_STABLE_FINE = identity_v16.DEFAULT_FINE
DEFAULT_STABLE_PREFIX = identity_v16.DEFAULT_PREFIX
DEFAULT_LEGACY_FINE = identity_v16.DEFAULT_LEGACY_FINE
DEFAULT_LEGACY_PREFIX = identity_v16.DEFAULT_LEGACY_PREFIX
DEFAULT_TARGET = identity_v16.DEFAULT_TARGET
DEFAULT_SOURCE = identity_v16.DEFAULT_SOURCE
DEFAULT_SPLIT = identity_v16.DEFAULT_SPLIT
DEFAULT_AUX_JOIN = (
    ROOT / "outputs/deepsoz_masked_variable_auxiliary_join_v1_20260812"
)
DEFAULT_AUX_PREFIX = (
    ROOT / "outputs/deepsoz_masked_variable_auxiliary_labram_prefix_v17_20260812"
)
DEFAULT_AUX_FINE = (
    ROOT / "outputs/deepsoz_masked_variable_auxiliary_fine_evidence_v17_20260812"
)
DEFAULT_ANCHOR = identity_v16.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / "outputs/labram_masked_variable_auxiliary_oof_v17_20260812"

EXPECTED_AUX_JOIN_ARTIFACT_SHA256 = (
    "3dea9b8f13d8e74982a313626b62b966c31a324a26d144eac5d5e6e4ba8bc962"
)
EXPECTED_AUX_ADMISSION_ARTIFACT_SHA256 = (
    "a3a69550a4b0d7445d8311ed4641c25ef1cd28551f70b022d520984983dead7e"
)
EXPECTED_ANCHOR_MANIFEST_SHA256 = (
    "6b3eedd2af91f5d1905076a85c6990d35bea6b9e2b0d73fe062aa321f68562bb"
)
EXPECTED_ANCHOR_OOF_SHA256 = (
    "3cf8b5b4659e3664cc8de1a9b1be7137c7bb3e5fac889482c112555c04ae456e"
)

PREFIX_SCHEMA = "soz_deepsoz_masked_variable_auxiliary_labram_prefix_v17"
FINE_SCHEMA = "soz_deepsoz_masked_variable_auxiliary_fine_evidence_v17"
PREFIX_TENSOR_KEYS = {"prefix_tokens": (15, 77, 200)}
FINE_TENSOR_KEYS = {
    "features": (19, 20),
    "composite_trace": (19, 237),
    "dominant_frequency_hz": (19, 237),
    "node_change_detected": (19,),
    "node_change_latency_sec": (19,),
    "bipolar_change_detected": (20,),
    "bipolar_change_latency_sec": (20,),
    "window_center_sec": (237,),
}


@dataclass(frozen=True)
class StableDevelopmentData:
    union: PublicDevelopmentUnionIdentityV12
    patient_ids: tuple[str, ...]
    patient_folds: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    h_patient: torch.Tensor
    fine_patient: torch.Tensor
    event_counts: torch.Tensor
    stable_event_ids: tuple[str, ...]
    lineage: Mapping[str, str]


@dataclass(frozen=True)
class PinnedAnchor:
    manifest: Mapping[str, object]
    logits: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    patient_folds: torch.Tensor
    event_counts: torch.Tensor
    manifest_sha256: str
    oof_sha256: str


@dataclass(frozen=True)
class AuxiliaryTargets:
    join: VerifiedMaskedVariableAuxiliaryJoin
    patient_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    event_patient_index: torch.Tensor
    outer_folds: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    event_counts: torch.Tensor
    patient_rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class AuxiliaryCache:
    directory: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    tensor_path: Path
    tensor_file_sha256: str
    tensors: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class Phase0Parity:
    transforms: Mapping[int, FoldFeatureTransform]
    priors: Mapping[int, torch.Tensor]
    oof_logits: torch.Tensor
    fold_receipts: tuple[Mapping[str, object], ...]
    outer_states: Mapping[str, torch.Tensor]


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _regular_directory(value: Path, *, name: str) -> Path:
    path = Path(os.path.abspath(value))
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be a regular non-symlink directory")
    return path


def _safe_tensor_path(directory: Path, manifest: Mapping[str, object]) -> Path:
    filename = manifest.get("tensor_file")
    if not isinstance(filename, str) or not filename:
        raise TypeError("auxiliary cache tensor_file is missing")
    pure = PurePath(filename)
    if pure.is_absolute() or len(pure.parts) != 1 or pure.suffix != ".safetensors":
        raise ValueError("auxiliary cache tensor_file is not a safe basename")
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError("auxiliary cache tensor file is not regular")
    entries = tuple(sorted(item.name for item in directory.iterdir()))
    if entries != tuple(sorted(("manifest.json", filename))):
        raise ValueError("auxiliary cache directory violates the closed two-file schema")
    return path


def _axis_use(receipt: Mapping[str, object], *, label: str) -> dict[str, bool]:
    axes = receipt.get("lineage_axes")
    expected = {
        "direct_target_values": False,
        "upstream_target_conditioned_roster": True,
        "target_supervised_model": False,
    }
    if not isinstance(axes, Mapping) or set(axes) != set(expected):
        raise ValueError(f"{label} lacks the closed three-axis lineage")
    actual: dict[str, bool] = {}
    for name in expected:
        value = axes[name]
        if not isinstance(value, Mapping) or not isinstance(value.get("used"), bool):
            raise ValueError(f"{label} lineage axis {name} is invalid")
        actual[name] = bool(value["used"])
    if actual != expected:
        raise ValueError(f"{label} lineage use differs: {actual}")
    return actual


def _load_stable_development(args: argparse.Namespace) -> StableDevelopmentData:
    union = load_public_development_union_identity_v12(
        args.union_directory,
        expected_manifest_sha256=args.expected_union_manifest_sha256,
        expected_payload_sha256=(
            EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256
        ),
    )
    fine = identity_v16._load_identity_cache(
        args.stable_fine_directory,
        expected_manifest_sha256=args.expected_stable_fine_manifest_sha256,
        expected_tensor_sha256=args.expected_stable_fine_tensor_sha256,
        tensor_key="features",
        tensor_tail_shape=(19, 20),
        union=union,
        legacy_directory=args.legacy_fine_directory,
        expected_legacy_manifest_sha256=(
            identity_v16.EXPECTED_LEGACY_FINE_MANIFEST_SHA256
        ),
        expected_legacy_tensor_sha256=(
            identity_v16.EXPECTED_LEGACY_FINE_TENSOR_SHA256
        ),
        label="stable fine evidence identity-v12",
    )
    if tuple(fine.manifest.get("feature_names", ())) != FINE_TEMPORAL_FEATURE_NAMES:
        raise ValueError("stable fine feature vocabulary changed")
    prefix = identity_v16._load_identity_cache(
        args.stable_prefix_directory,
        expected_manifest_sha256=args.expected_stable_prefix_manifest_sha256,
        expected_tensor_sha256=args.expected_stable_prefix_tensor_sha256,
        tensor_key="prefix_tokens",
        tensor_tail_shape=(15, 77, 200),
        union=union,
        legacy_directory=args.legacy_prefix_directory,
        expected_legacy_manifest_sha256=(
            identity_v16.EXPECTED_LEGACY_PREFIX_MANIFEST_SHA256
        ),
        expected_legacy_tensor_sha256=(
            identity_v16.EXPECTED_LEGACY_PREFIX_TENSOR_SHA256
        ),
        label="stable LaBraM block-9 prefix identity-v12",
    )

    h_event = extract_block9_phase_contrasts(prefix.tensor)
    fine_event = fine.tensor
    event_patient_index = torch.tensor(union.event_patient_index, dtype=torch.long)
    artifact_index = FINE_TEMPORAL_FEATURE_NAMES.index("artifact_burden_0_12s")
    reliability = (1.0 - fine_event[:, :, artifact_index]).clamp(0.0, 1.0)
    h_pool = robust_pool_complete_patient_bags(
        h_event, event_patient_index, len(union.patient_ids), reliability
    )
    fine_pool = robust_pool_complete_patient_bags(
        fine_event, event_patient_index, len(union.patient_ids), reliability
    )
    if not torch.equal(h_pool.event_counts, fine_pool.event_counts):
        raise RuntimeError("stable H/fine patient bags disagree")

    roster, target = identity_v16._load_primary_roster(
        union,
        target_directory=args.target_directory,
        source_csv=args.source_csv,
        split_csv=args.split_csv,
    )
    selected = roster.selected_union_indices
    event_counts = h_pool.event_counts.index_select(0, selected).cpu()
    if int(event_counts.sum()) != identity_v16.PRIMARY_EVENT_COUNT:
        raise ValueError("stable identity-v16 event count changed")
    selected_patients = set(roster.patient_ids)
    stable_event_ids = tuple(
        event.event_id for event in union.events if event.patient_id in selected_patients
    )
    if len(stable_event_ids) != identity_v16.PRIMARY_EVENT_COUNT or len(
        set(stable_event_ids)
    ) != len(stable_event_ids):
        raise RuntimeError("stable event-ID roster does not close")
    del h_event, fine_event, reliability
    return StableDevelopmentData(
        union=union,
        patient_ids=roster.patient_ids,
        patient_folds=roster.patient_folds.cpu(),
        targets=roster.targets.float().cpu(),
        target_mask=roster.target_mask.bool().cpu(),
        h_patient=h_pool.features.index_select(0, selected).float().cpu(),
        fine_patient=fine_pool.features.index_select(0, selected).float().cpu(),
        event_counts=event_counts,
        stable_event_ids=stable_event_ids,
        lineage={
            "union_manifest_sha256": union.manifest_sha256,
            "stable_fine_manifest_sha256": fine.manifest_sha256,
            "stable_fine_tensor_file_sha256": fine.tensor_file_sha256,
            "stable_prefix_manifest_sha256": prefix.manifest_sha256,
            "stable_prefix_tensor_file_sha256": prefix.tensor_file_sha256,
            "stable_target_artifact_sha256": target.receipt.target_artifact_sha256,
            "stable_target_receipt_sha256": target.receipt.receipt_sha256,
            "stable_target_policy_sha256": target.receipt.policy_sha256,
        },
    )


def _load_pinned_anchor(directory: Path, stable: StableDevelopmentData) -> PinnedAnchor:
    root = _regular_directory(directory, name="pinned identity-v16 anchor")
    manifest = identity_v16._load_json_manifest(
        root / "manifest.json", expected_sha=EXPECTED_ANCHOR_MANIFEST_SHA256
    )
    if manifest.get("schema_version") != identity_v16.SCHEMA:
        raise ValueError("pinned identity-v16 anchor schema changed")
    if tuple(str(value) for value in manifest.get("patient_ids", ())) != stable.patient_ids:
        raise ValueError("anchor/stable patient order differs")
    if tuple(int(value) for value in manifest.get("patient_folds", ())) != tuple(
        stable.patient_folds.tolist()
    ):
        raise ValueError("anchor/stable outer folds differ")
    selected_l2 = manifest.get("selected_l2_by_arm", {}).get(FULL_ARM)
    if tuple(float(value) for value in selected_l2 or ()) != PINNED_FOLD_L2:
        raise ValueError("pinned identity-v16 per-fold L2 changed")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(name) is not False
        for name in ("private_eeg_loaded", "private_target_values_loaded")
    ):
        raise ValueError("pinned identity-v16 anchor crossed private firewall")

    oof_path = root / "oof_predictions.safetensors"
    if oof_path.is_symlink() or not oof_path.is_file() or (
        _file_sha(oof_path) != EXPECTED_ANCHOR_OOF_SHA256
    ):
        raise ValueError("pinned identity-v16 OOF artifact changed")
    declared = manifest.get("files", {}).get("oof_predictions.safetensors", {})
    if declared.get("sha256") != EXPECTED_ANCHOR_OOF_SHA256:
        raise ValueError("pinned anchor manifest/file binding changed")
    payload = load_file(str(oof_path), device="cpu")
    required = {
        f"oof.{FULL_ARM}",
        "targets",
        "target_mask",
        "patient_folds",
        "patient_event_counts",
        "config.candidate_mask",
    }
    if not required.issubset(payload):
        raise ValueError("pinned identity-v16 OOF payload is incomplete")
    logits = payload[f"oof.{FULL_ARM}"].float().cpu()
    targets = payload["targets"].float().cpu()
    mask = payload["target_mask"].bool().cpu()
    folds = payload["patient_folds"].long().cpu()
    event_counts = payload["patient_event_counts"].long().cpu()
    if tuple(logits.shape) != (identity_v16.PRIMARY_PATIENT_COUNT, 19) or not (
        torch.isfinite(logits).all()
    ):
        raise ValueError("pinned identity-v16 full logits changed")
    for label, left, right in (
        ("target", targets, stable.targets),
        ("target mask", mask, stable.target_mask),
        ("fold", folds, stable.patient_folds),
        ("event count", event_counts, stable.event_counts),
    ):
        if not torch.equal(left, right):
            raise ValueError(f"anchor/stable {label} differs")
    if not torch.equal(payload["config.candidate_mask"].bool(), V11_CANDIDATE_MASK):
        raise ValueError("pinned anchor candidate mask changed")
    metrics = _evaluate(logits, targets, mask)
    contributions = _patient_contributions(logits, targets, mask)
    checks = {
        "strict_count": math.isclose(
            float(contributions["strict"].sum()), EXPECTED_ANCHOR_STRICT_COUNT
        ),
        "relaxed_count": math.isclose(
            float(contributions["relaxed"].sum()), EXPECTED_ANCHOR_RELAXED_COUNT
        ),
        "far_count": math.isclose(
            float(metrics["far_error_count"]), EXPECTED_ANCHOR_FAR_COUNT
        ),
        "neighbor_denominator": metrics["top1"]["n_neighbor_eligible_samples"] == 100,
    }
    if not all(checks.values()):
        raise ValueError(f"pinned identity-v16 endpoint values changed: {checks}")
    declared_metrics = manifest.get("metrics_all_102", {}).get(FULL_ARM)
    if _canonical_bytes(declared_metrics) != _canonical_bytes(metrics):
        raise ValueError("pinned identity-v16 declared metrics do not replay")
    return PinnedAnchor(
        manifest=manifest,
        logits=logits,
        targets=targets,
        target_mask=mask,
        patient_folds=folds,
        event_counts=event_counts,
        manifest_sha256=EXPECTED_ANCHOR_MANIFEST_SHA256,
        oof_sha256=EXPECTED_ANCHOR_OOF_SHA256,
    )


def _stable_outer_partition(
    patient_folds: torch.Tensor, outer_fold: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if outer_fold not in OUTER_FOLDS or tuple(patient_folds.shape) != (
        identity_v16.PRIMARY_PATIENT_COUNT,
    ):
        raise ValueError("stable outer partition inputs are invalid")
    held = tuple(
        torch.nonzero(patient_folds == outer_fold, as_tuple=False).flatten().tolist()
    )
    train = tuple(
        torch.nonzero(patient_folds != outer_fold, as_tuple=False).flatten().tolist()
    )
    if not held or not train or set(held) & set(train) or len(held) + len(train) != len(
        patient_folds
    ):
        raise RuntimeError("stable outer partition is incomplete")
    return train, held


def _run_phase0_no_aux_parity(
    stable: StableDevelopmentData,
    anchor: PinnedAnchor,
) -> Phase0Parity:
    """Refit all no-aux folds and stop before any auxiliary target/loss read."""

    transforms: dict[int, FoldFeatureTransform] = {}
    priors: dict[int, torch.Tensor] = {}
    receipts: list[Mapping[str, object]] = []
    states: dict[str, torch.Tensor] = {}
    oof = torch.full_like(anchor.logits, torch.nan)
    for outer_fold, l2 in zip(OUTER_FOLDS, PINNED_FOLD_L2):
        train, held = _stable_outer_partition(stable.patient_folds, outer_fold)
        transform = fit_fold_transform(stable.h_patient, stable.fine_patient, train)
        transformed = transform.apply(stable.h_patient, stable.fine_patient)
        fit = _fit_reasoner(
            transformed,
            stable.targets,
            stable.target_mask,
            train,
            use_h=True,
            use_fine=True,
            l2=l2,
        )
        if fit.diagnostics.get("trainable_parameter_count") != 36 or (
            fit.diagnostics.get("prior_source") != "fit_training_rows"
        ):
            raise RuntimeError("Phase-0 reasoner capacity/prior source changed")
        train_tensor = torch.tensor(train, dtype=torch.long)
        held_tensor = torch.tensor(held, dtype=torch.long)
        prior = jeffreys_reference_prior_logits(
            stable.targets.index_select(0, train_tensor),
            stable.target_mask.index_select(0, train_tensor),
        )
        if not torch.equal(fit.state["prior_logits"], prior):
            raise RuntimeError("Phase-0 stable-only Jeffreys prior changed")
        actual = fit.logits.index_select(0, held_tensor)
        expected = anchor.logits.index_select(0, held_tensor)
        difference = (actual - expected).abs()
        max_abs = float(difference.max())
        passed = bool(torch.allclose(actual, expected, atol=PARITY_ATOL, rtol=PARITY_RTOL))
        if not passed:
            raise RuntimeError(
                f"Phase-0 identity-v16 parity failed in outer fold {outer_fold}: "
                f"max_abs={max_abs}"
            )
        oof.index_copy_(0, held_tensor, actual)
        transforms[outer_fold] = transform
        priors[outer_fold] = prior
        for name, value in _transform_state(transform).items():
            states[f"phase0.outer{outer_fold}.{name}"] = value
        for name, value in fit.state.items():
            states[f"phase0.outer{outer_fold}.reasoner.{name}"] = value
        states[f"phase0.outer{outer_fold}.config.l2"] = torch.tensor(
            l2, dtype=torch.float32
        )
        receipts.append(
            {
                "outer_fold": outer_fold,
                "l2": l2,
                "stable_train_patient_ids": [stable.patient_ids[index] for index in train],
                "stable_held_patient_ids": [stable.patient_ids[index] for index in held],
                "fit": dict(fit.diagnostics),
                "fit_state_sha256": _state_sha(fit.state),
                "max_abs_difference_vs_pinned_v16_held_logits": max_abs,
                "atol": PARITY_ATOL,
                "rtol": PARITY_RTOL,
                "passed": True,
            }
        )
    if not torch.isfinite(oof).all() or not torch.allclose(
        oof, anchor.logits, atol=PARITY_ATOL, rtol=PARITY_RTOL
    ):
        raise RuntimeError("Phase-0 complete OOF parity failed")
    return Phase0Parity(
        transforms=transforms,
        priors=priors,
        oof_logits=oof,
        fold_receipts=tuple(receipts),
        outer_states=states,
    )


def _load_auxiliary_targets(
    args: argparse.Namespace,
    stable: StableDevelopmentData,
) -> AuxiliaryTargets:
    join = load_masked_variable_auxiliary_join(
        args.aux_join_directory,
        expected_artifact_sha256=args.expected_aux_join_artifact_sha256,
        expected_admission_artifact_sha256=(
            args.expected_aux_admission_artifact_sha256
        ),
    )
    receipt = join.receipt
    if receipt.get("schema_version") != MASKED_VARIABLE_AUXILIARY_JOIN_SCHEMA or (
        receipt.get("policy") != MASKED_VARIABLE_AUXILIARY_JOIN_POLICY
    ):
        raise ValueError("auxiliary target join schema/policy changed")
    if receipt.get("inputs", {}).get("protocol_sha256") != args.expected_protocol_sha256:
        raise ValueError("auxiliary target join is not bound to the frozen protocol")
    if receipt.get("private_data_accessed") is not False or (
        receipt.get("model_or_training_executed") is not False
    ):
        raise ValueError("auxiliary target join crossed data/model firewall")
    axes = receipt.get("lineage_axes")
    expected_join_axes = {
        "direct_target_values": True,
        "upstream_target_conditioned_roster": True,
        "target_supervised_model": False,
    }
    if not isinstance(axes, Mapping) or {
        name: bool(value.get("used")) for name, value in axes.items()
    } != expected_join_axes:
        raise ValueError("auxiliary target join three-axis lineage changed")
    transformations = receipt.get("label_transformations")
    required_transformations = {
        "stable_explicit_1_retained": True,
        "stable_explicit_0_retained": True,
        "patient_variable_loss_mask_zero": True,
        "missing_loss_mask_zero": True,
        "pz_loss_mask_zero": True,
        "masked_target_placeholder": 0,
        "majority_vote": False,
        "positive_union": False,
        "one_hop_dilation": False,
        "missing_positive_imputation": False,
        "private_labels": False,
        "prediction_based_selection": False,
    }
    if transformations != required_transformations:
        raise ValueError("auxiliary label transformation contract changed")
    if (
        receipt.get("startup_auxiliary_patient_count_gate_pass") is not True
        or receipt.get("preregistered_auxiliary_patient_count")
        != PREREGISTERED_AUXILIARY_PATIENT_COUNT
        or receipt.get("admitted_patient_count")
        != PREREGISTERED_AUXILIARY_PATIENT_COUNT
        or receipt.get("admitted_event_count") != EXPECTED_AUXILIARY_EVENT_COUNT
        or receipt.get("aux_outer_fold_count") != len(OUTER_FOLDS)
        or receipt.get("aux_outer_fold_target_values_used") is not False
    ):
        raise ValueError("auxiliary startup/fold gate failed")
    signal_inputs = receipt.get("inputs", {})
    if signal_inputs.get("signal_universe_identity_patient_count") != 124 or (
        signal_inputs.get("signal_universe_candidate_event_count") != 1812
    ):
        raise ValueError("target-independent signal-universe launch gate changed")

    admitted_ids = tuple(str(value) for value in receipt["admitted_patient_ids"])
    admitted_event_ids = tuple(str(value) for value in receipt["admitted_event_ids"])
    if len(admitted_ids) != PREREGISTERED_AUXILIARY_PATIENT_COUNT or len(
        set(admitted_ids)
    ) != len(admitted_ids):
        raise ValueError("auxiliary admitted patient roster changed")
    if len(admitted_event_ids) != EXPECTED_AUXILIARY_EVENT_COUNT or len(
        set(admitted_event_ids)
    ) != len(admitted_event_ids):
        raise ValueError("auxiliary admitted event roster changed")
    if set(admitted_ids) & set(stable.patient_ids):
        raise ValueError("auxiliary and stable patient rosters overlap")
    if set(admitted_event_ids) & set(stable.stable_event_ids):
        raise ValueError("auxiliary and stable event caches overlap")

    patient_rows_by_id = {
        str(row["patient_id"]): row
        for row in receipt["patients"]
        if bool(row["admitted"])
    }
    if tuple(patient_rows_by_id) != admitted_ids:
        raise ValueError("auxiliary admitted patient rows/order changed")
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    folds: list[int] = []
    counts: list[int] = []
    ordered_rows: list[Mapping[str, object]] = []
    pz = CHANNEL_INDEX["PZ"]
    for patient_id in admitted_ids:
        row = patient_rows_by_id[patient_id]
        target = torch.tensor(row["target"], dtype=torch.float32)
        mask_values = torch.tensor(row["loss_mask"], dtype=torch.long)
        if tuple(target.shape) != (19,) or tuple(mask_values.shape) != (19,) or not (
            torch.all((target == 0) | (target == 1))
            and torch.all((mask_values == 0) | (mask_values == 1))
        ):
            raise ValueError(f"invalid auxiliary target row: {patient_id}")
        mask = mask_values.bool()
        if bool(mask[pz]) or bool(target[pz]) or bool((mask & ~V11_CANDIDATE_MASK).any()):
            raise ValueError(f"auxiliary C18 subset mask changed: {patient_id}")
        positive = mask & (target == 1)
        positive_names = [
            STANDARD_19[int(index)] for index in positive.nonzero().flatten()
        ]
        if not bool(positive.any()) or positive_names != row["stable_positive_channels"]:
            raise ValueError(f"auxiliary positive-set receipt changed: {patient_id}")
        fold = row["aux_outer_fold"]
        if isinstance(fold, bool) or not isinstance(fold, int) or fold not in OUTER_FOLDS:
            raise ValueError(f"invalid auxiliary outer fold: {patient_id}")
        event_ids = tuple(str(value) for value in row["eligible_event_ids"])
        if len(event_ids) != int(row["eligible_event_count"]) or not event_ids:
            raise ValueError(f"auxiliary patient event bag changed: {patient_id}")
        targets.append(target)
        masks.append(mask)
        folds.append(fold)
        counts.append(len(event_ids))
        ordered_rows.append(row)
    if sum(counts) != EXPECTED_AUXILIARY_EVENT_COUNT:
        raise ValueError("auxiliary patient event counts do not close")

    patient_index = {patient_id: index for index, patient_id in enumerate(admitted_ids)}
    event_rows = receipt["events"]
    if tuple(str(row["event_id"]) for row in event_rows) != admitted_event_ids:
        raise ValueError("auxiliary join event order changed")
    event_patient_index = torch.tensor(
        [patient_index[str(row["patient_id"])] for row in event_rows], dtype=torch.long
    )
    observed_counts = torch.bincount(
        event_patient_index, minlength=len(admitted_ids)
    ).long()
    expected_counts = torch.tensor(counts, dtype=torch.long)
    if not torch.equal(observed_counts, expected_counts):
        raise ValueError("auxiliary event-to-patient bags disagree with join")
    return AuxiliaryTargets(
        join=join,
        patient_ids=admitted_ids,
        event_ids=admitted_event_ids,
        event_patient_index=event_patient_index,
        outer_folds=torch.tensor(folds, dtype=torch.long),
        targets=torch.stack(targets),
        target_mask=torch.stack(masks),
        event_counts=expected_counts,
        patient_rows=tuple(ordered_rows),
    )


def _load_auxiliary_cache(
    directory: Path,
    *,
    expected_manifest_sha256: str,
    expected_tensor_sha256: str,
    schema: str,
    expected_keys: Mapping[str, tuple[int, ...]],
    primary_key: str,
    auxiliary: AuxiliaryTargets,
    label: str,
) -> AuxiliaryCache:
    root = _regular_directory(directory, name=f"{label} directory")
    manifest_sha = _require_sha256(
        expected_manifest_sha256, name=f"expected {label} manifest SHA"
    )
    tensor_file_sha = _require_sha256(
        expected_tensor_sha256, name=f"expected {label} tensor file SHA"
    )
    manifest = identity_v16._load_json_manifest(
        root / "manifest.json", expected_sha=manifest_sha
    )
    if manifest.get("schema_version") != schema:
        raise ValueError(f"{label} schema changed")
    common = {
        "development_only": True,
        "public_confirmation_forbidden": True,
        "full_scope": True,
        "smoke_only": False,
        "independent_auxiliary_cache": True,
        "event_count": EXPECTED_AUXILIARY_EVENT_COUNT,
        "patient_count": PREREGISTERED_AUXILIARY_PATIENT_COUNT,
        "admitted_event_roster_complete": True,
    }
    failed = [name for name, expected in common.items() if manifest.get(name) != expected]
    if failed:
        raise ValueError(f"{label} full independent-cache receipt failed: {failed}")
    if tuple(manifest.get("standard_19", ())) != STANDARD_19 or tuple(
        str(value) for value in manifest.get("event_ids", ())
    ) != auxiliary.event_ids:
        raise ValueError(f"{label} channel/event roster changed")
    _axis_use(manifest, label=label)
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError(f"{label} lacks lineage bindings")
    lineage_checks = {
        "admission_artifact_sha256": auxiliary.join.admission_artifact_sha256,
        "admission_receipt_sha256": auxiliary.join.admission_receipt_sha256,
        "source_target_join_artifact_sha256": auxiliary.join.artifact_sha256,
        "source_target_join_receipt_sha256": auxiliary.join.receipt_sha256,
    }
    for name, expected in lineage_checks.items():
        if lineage.get(name) != expected:
            raise ValueError(f"{label} lineage binding differs: {name}")
    separation = manifest.get("cache_separation_receipt")
    expected_separation = {
        "existing_1149_event_cache_loaded": False,
        "existing_1149_event_tensor_concatenated": False,
        "existing_1149_event_cache_overwritten": False,
        "legacy_reused_event_count": 0,
        "new_auxiliary_event_count": EXPECTED_AUXILIARY_EVENT_COUNT,
        "output_contains_only_admitted_auxiliary_events": True,
    }
    if separation != expected_separation:
        raise ValueError(f"{label} cache-separation receipt changed")
    access = manifest.get("access_receipt")
    required_false = (
        "target_bearing_join_artifact_loaded",
        "direct_target_values_loaded",
        "historical_prediction_artifacts_loaded",
        "stable_1149_representation_cache_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "foundation_training_performed",
        "reasoner_training_performed",
    )
    if not isinstance(access, Mapping) or any(access.get(name) is not False for name in required_false):
        raise ValueError(f"{label} access firewall failed")
    if access.get("upstream_target_conditioned_roster") is not True or (
        access.get("raw_public_event_count") != EXPECTED_AUXILIARY_EVENT_COUNT
    ):
        raise ValueError(f"{label} roster/raw-event receipt changed")
    if schema == PREFIX_SCHEMA and (
        manifest.get("foundation_trainable_parameters_during_materialization") != 0
        or access.get("foundation_trainable_parameters") != 0
    ):
        raise ValueError("auxiliary LaBraM prefix was not fully frozen")

    join_events = {str(row["event_id"]): row for row in auxiliary.join.receipt["events"]}
    fold_by_patient = dict(zip(auxiliary.patient_ids, auxiliary.outer_folds.tolist()))
    rows = manifest.get("events")
    if not isinstance(rows, list) or len(rows) != EXPECTED_AUXILIARY_EVENT_COUNT:
        raise ValueError(f"{label} event rows are incomplete")
    for index, (row, event_id) in enumerate(zip(rows, auxiliary.event_ids)):
        source = join_events[event_id]
        if not isinstance(row, Mapping):
            raise TypeError(f"{label} event row {index} is not an object")
        checks = {
            "ordinal": index,
            "event_id": event_id,
            "patient_id": source["patient_id"],
            "official_split": source["official_split"],
            "source_model_split": source["source_model_split"],
            "aux_outer_fold": fold_by_patient[str(source["patient_id"])],
            "event_record_sha256": source["event_record_sha256"],
            "processed_window_sha256": source["processed_window_sha256"],
        }
        if any(row.get(name) != expected for name, expected in checks.items()):
            raise ValueError(f"{label} event binding changed: {event_id}")

    tensor_path = _safe_tensor_path(root, manifest)
    if _file_sha(tensor_path) != tensor_file_sha or (
        manifest.get("tensor_file_sha256") != tensor_file_sha
    ):
        raise ValueError(f"{label} tensor file SHA changed")
    payload = load_file(str(tensor_path), device="cpu")
    if set(payload) != set(expected_keys):
        raise ValueError(f"{label} safetensors key set changed")
    specs = manifest.get("tensor_specs")
    if not isinstance(specs, Mapping) or set(specs) != set(expected_keys):
        raise ValueError(f"{label} tensor_specs key set changed")
    checked: dict[str, torch.Tensor] = {}
    for name, tail in expected_keys.items():
        value = payload[name].detach().cpu().contiguous()
        expected_shape = (EXPECTED_AUXILIARY_EVENT_COUNT, *tail)
        if name == "window_center_sec":
            expected_shape = tail
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{label} tensor shape changed: {name}={tuple(value.shape)}")
        finite_required = name not in {
            "node_change_latency_sec",
            "bipolar_change_latency_sec",
        }
        if value.is_floating_point() and finite_required and not torch.isfinite(value).all():
            raise ValueError(f"{label} tensor is non-finite: {name}")
        spec = specs[name]
        if not isinstance(spec, Mapping) or spec.get("shape") != list(value.shape) or (
            spec.get("dtype") != str(value.dtype)
        ) or spec.get("tensor_sha256") != auxiliary_tensor_sha256(value):
            raise ValueError(f"{label} tensor spec/hash changed: {name}")
        checked[name] = value
    if primary_key not in checked:
        raise ValueError(f"{label} primary tensor is absent")
    return AuxiliaryCache(
        directory=root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        tensor_path=tensor_path,
        tensor_file_sha256=tensor_file_sha,
        tensors=checked,
    )


def _pool_auxiliary_features(
    auxiliary: AuxiliaryTargets,
    prefix: AuxiliaryCache,
    fine: AuxiliaryCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    fine_event = fine.tensors["features"].float()
    if tuple(fine_event.shape) != (EXPECTED_AUXILIARY_EVENT_COUNT, 19, 20) or (
        tuple(fine.manifest.get("feature_names", ())) != FINE_TEMPORAL_FEATURE_NAMES
    ):
        raise ValueError("auxiliary fine feature carrier/vocabulary changed")
    prefix_event = prefix.tensors["prefix_tokens"].float()
    h_event = extract_block9_phase_contrasts(prefix_event)
    artifact_index = FINE_TEMPORAL_FEATURE_NAMES.index("artifact_burden_0_12s")
    reliability = (1.0 - fine_event[:, :, artifact_index]).clamp(0.0, 1.0)
    h_pool = robust_pool_complete_patient_bags(
        h_event,
        auxiliary.event_patient_index,
        len(auxiliary.patient_ids),
        reliability,
    )
    fine_pool = robust_pool_complete_patient_bags(
        fine_event,
        auxiliary.event_patient_index,
        len(auxiliary.patient_ids),
        reliability,
    )
    if not torch.equal(h_pool.event_counts, auxiliary.event_counts) or not torch.equal(
        fine_pool.event_counts, auxiliary.event_counts
    ):
        raise RuntimeError("auxiliary H/fine patient bags disagree with target join")
    return h_pool.features.float().cpu(), fine_pool.features.float().cpu()


def _combined_outer_train_indices(
    stable_folds: torch.Tensor,
    aux_folds: torch.Tensor,
    outer_fold: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    stable_train, stable_held = _stable_outer_partition(stable_folds, outer_fold)
    aux_train_local = tuple(
        torch.nonzero(aux_folds != outer_fold, as_tuple=False).flatten().tolist()
    )
    aux_excluded_local = tuple(
        torch.nonzero(aux_folds == outer_fold, as_tuple=False).flatten().tolist()
    )
    if not aux_train_local or not aux_excluded_local or set(aux_train_local) & set(
        aux_excluded_local
    ) or len(aux_train_local) + len(aux_excluded_local) != len(aux_folds):
        raise RuntimeError("auxiliary outer-fold firewall failed")
    offset = len(stable_folds)
    combined_train = stable_train + tuple(offset + index for index in aux_train_local)
    return stable_train, stable_held, aux_train_local, combined_train


def _strict_win_loss_tie(
    candidate: torch.Tensor,
    anchor: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, int]:
    candidate_rows = _patient_contributions(candidate, targets, mask)["strict"]
    anchor_rows = _patient_contributions(anchor, targets, mask)["strict"]
    return {
        "wins": int((candidate_rows > anchor_rows).sum()),
        "losses": int((candidate_rows < anchor_rows).sum()),
        "ties": int((candidate_rows == anchor_rows).sum()),
    }


def _top1_agreement(candidate: torch.Tensor, anchor: torch.Tensor) -> dict[str, object]:
    candidate_top = apply_fixed_candidate_mask(candidate).argmax(dim=1)
    anchor_top = apply_fixed_candidate_mask(anchor).argmax(dim=1)
    count = int((candidate_top == anchor_top).sum())
    return {
        "count": count,
        "denominator": int(candidate.shape[0]),
        "rate": count / int(candidate.shape[0]),
    }


def _contralateral_far_count(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> float:
    """Reuse the block-9 audit's exact L/R/M and one-hop definitions."""

    total = 0.0
    for patient in range(logits.shape[0]):
        ties = _block9_top_ties(logits, target_mask, patient)
        positive = targets[patient].bool() & target_mask[patient]
        positive_indices = torch.nonzero(positive, as_tuple=False).flatten()
        accepted = _block9_accepted(targets, target_mask, patient)
        total += _block9_contralateral_far_value(ties, positive_indices, accepted)
    return total


def _assess_stop_gate(
    *,
    candidate_metrics: Mapping[str, object],
    anchor_metrics: Mapping[str, object],
    paired: Mapping[str, Mapping[str, object]],
    candidate_strict_count: float,
    anchor_strict_count: float,
    candidate_fold_strict: Sequence[float],
    anchor_fold_strict: Sequence[float],
    firewall_gates: Mapping[str, bool],
) -> tuple[bool, dict[str, bool], int]:
    if len(candidate_fold_strict) != 5 or len(anchor_fold_strict) != 5:
        raise ValueError("v17 stop gate requires five outer folds")
    fold_nonlower = sum(
        candidate >= anchor
        for candidate, anchor in zip(candidate_fold_strict, anchor_fold_strict)
    )
    net_gain = candidate_strict_count - anchor_strict_count
    checks = {
        "strict_net_gain_at_least_5_of_102": net_gain >= STRICT_MINIMUM_NET_GAIN,
        "strict_paired_ci_lower_strictly_positive": float(
            paired["strict"]["ci95"][0]
        )
        > 0.0,
        "macro_ap_point_nonlower": float(
            candidate_metrics["ranking"]["macro_average_precision"]
        )
        >= float(anchor_metrics["ranking"]["macro_average_precision"]),
        "macro_ap_paired_ci_lower_nonnegative": float(
            paired["macro_ap"]["ci95"][0]
        )
        >= 0.0,
        "one_hop_relaxed_point_nonlower": float(
            candidate_metrics["top1"]["relaxed_accuracy"]
        )
        >= float(anchor_metrics["top1"]["relaxed_accuracy"]),
        "far_error_not_above_pinned_25": float(candidate_metrics["far_error_count"])
        <= EXPECTED_ANCHOR_FAR_COUNT,
        "four_of_five_fold_strict_nonlower": fold_nonlower >= 4,
        "all_auxiliary_signal_cache_mask_patient_foundation_firewalls_pass": all(
            firewall_gates.values()
        ),
    }
    return all(checks.values()), checks, fold_nonlower


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner_v17": Path(__file__).resolve(),
        "runner_v16": Path(identity_v16.__file__).resolve(),
        "runner_shared_v11": ROOT / "scripts/run_labram_fine_temporal_nested_oof_v11.py",
        "runner_metrics_v11_1": ROOT / "scripts/run_labram_fine_temporal_nested_oof_v11_1.py",
        "block9_failure_audit": ROOT / "scripts/audit_labram_block9_v11_1_failures.py",
        "reasoner": ROOT / "src/soz/v11_reasoner.py",
        "metrics": ROOT / "src/soz/metrics.py",
        "auxiliary_join": ROOT / "src/soz/data/deepsoz_masked_variable_auxiliary_join.py",
        "auxiliary_cache_contract": ROOT / "src/soz/data/deepsoz_masked_variable_auxiliary_cache_v17.py",
    }
    return {name: _file_sha(path) for name, path in paths.items()}


def _preflight_full_auxiliary_caches(args: argparse.Namespace) -> None:
    """Fail before any OOF fit when the two full pinned caches are unavailable."""

    for name in (
        "expected_aux_prefix_manifest_sha256",
        "expected_aux_prefix_tensor_sha256",
        "expected_aux_fine_manifest_sha256",
        "expected_aux_fine_tensor_sha256",
    ):
        _require_sha256(getattr(args, name), name=name)
    for path, label in (
        (args.aux_prefix_directory, "full auxiliary prefix cache"),
        (args.aux_fine_directory, "full auxiliary fine cache"),
    ):
        _regular_directory(path, name=label)


def run(
    args: argparse.Namespace,
) -> tuple[Mapping[str, object], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    protocol_sha = _require_sha256(
        args.expected_protocol_sha256, name="expected protocol SHA"
    )
    if _file_sha(PROTOCOL_PATH) != protocol_sha:
        raise ValueError("v17 protocol changed after freezing")
    # The code-only handoff deliberately precedes full cache materialization.
    # Refuse immediately instead of spending time on Phase-0 when either
    # future formal input is absent or unpinned.
    _preflight_full_auxiliary_caches(args)
    source_hashes_before = _source_hashes()

    # Stable target-bearing data and the pinned comparator are allowed here.
    # The auxiliary target join is deliberately not opened until all five
    # no-aux fits have replayed the identity-v16 held logits.
    stable = _load_stable_development(args)
    _require_fixed_rows(stable.target_mask)
    anchor = _load_pinned_anchor(args.anchor_directory, stable)
    phase0 = _run_phase0_no_aux_parity(stable, anchor)

    # First auxiliary target-value read.  Phase-0 is now irreversibly complete
    # in memory; any parity failure above stops before an auxiliary loss exists.
    auxiliary = _load_auxiliary_targets(args, stable)
    aux_prefix = _load_auxiliary_cache(
        args.aux_prefix_directory,
        expected_manifest_sha256=args.expected_aux_prefix_manifest_sha256,
        expected_tensor_sha256=args.expected_aux_prefix_tensor_sha256,
        schema=PREFIX_SCHEMA,
        expected_keys=PREFIX_TENSOR_KEYS,
        primary_key="prefix_tokens",
        auxiliary=auxiliary,
        label="auxiliary frozen LaBraM prefix",
    )
    aux_fine = _load_auxiliary_cache(
        args.aux_fine_directory,
        expected_manifest_sha256=args.expected_aux_fine_manifest_sha256,
        expected_tensor_sha256=args.expected_aux_fine_tensor_sha256,
        schema=FINE_SCHEMA,
        expected_keys=FINE_TENSOR_KEYS,
        primary_key="features",
        auxiliary=auxiliary,
        label="auxiliary fine evidence",
    )
    aux_h_patient, aux_fine_patient = _pool_auxiliary_features(
        auxiliary, aux_prefix, aux_fine
    )

    h_combined = torch.cat((stable.h_patient, aux_h_patient), dim=0)
    fine_combined = torch.cat((stable.fine_patient, aux_fine_patient), dim=0)
    targets_combined = torch.cat((stable.targets, auxiliary.targets), dim=0)
    mask_combined = torch.cat((stable.target_mask, auxiliary.target_mask), dim=0)
    if tuple(h_combined.shape) != (111, 19, 600) or tuple(fine_combined.shape) != (
        111,
        19,
        20,
    ):
        raise RuntimeError("stable+auxiliary combined carrier shape changed")

    oof = torch.full_like(anchor.logits, torch.nan)
    fold_results: list[dict[str, object]] = []
    outer_states = dict(phase0.outer_states)
    for outer_fold, l2 in zip(OUTER_FOLDS, PINNED_FOLD_L2):
        stable_train, stable_held, aux_train_local, combined_train = (
            _combined_outer_train_indices(
                stable.patient_folds, auxiliary.outer_folds, outer_fold
            )
        )
        aux_excluded_local = tuple(
            torch.nonzero(
                auxiliary.outer_folds == outer_fold, as_tuple=False
            ).flatten().tolist()
        )
        transform = phase0.transforms[outer_fold]
        if transform.train_patient_indices != stable_train:
            raise RuntimeError("candidate transform is not the Phase-0 stable-only transform")
        transformed = transform.apply(h_combined, fine_combined)
        prior = phase0.priors[outer_fold]
        fit = _fit_reasoner(
            transformed,
            targets_combined,
            mask_combined,
            combined_train,
            use_h=True,
            use_fine=True,
            l2=l2,
            allow_candidate_subset=True,
            fixed_prior_logits=prior,
        )
        if fit.diagnostics.get("trainable_parameter_count") != 36 or (
            fit.diagnostics.get("prior_source") != "caller_frozen"
        ) or not torch.equal(fit.state["prior_logits"], prior):
            raise RuntimeError("candidate capacity/frozen stable-only prior changed")
        held_tensor = torch.tensor(stable_held, dtype=torch.long)
        held_logits = fit.logits.index_select(0, held_tensor)
        oof.index_copy_(0, held_tensor, held_logits)
        held_metrics = _evaluate(
            held_logits,
            stable.targets.index_select(0, held_tensor),
            stable.target_mask.index_select(0, held_tensor),
        )
        for name, value in fit.state.items():
            outer_states[f"candidate.outer{outer_fold}.reasoner.{name}"] = value
        outer_states[f"candidate.outer{outer_fold}.config.l2"] = torch.tensor(
            l2, dtype=torch.float32
        )
        outer_states[f"candidate.outer{outer_fold}.config.candidate_mask"] = (
            V11_CANDIDATE_MASK.clone()
        )
        outer_states[f"candidate.outer{outer_fold}.stable_train_indices"] = torch.tensor(
            stable_train, dtype=torch.long
        )
        outer_states[f"candidate.outer{outer_fold}.aux_train_local_indices"] = torch.tensor(
            aux_train_local, dtype=torch.long
        )
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "l2_reused_from_pinned_identity_v16": l2,
                "stable_train_patient_ids": [stable.patient_ids[i] for i in stable_train],
                "stable_held_patient_ids": [stable.patient_ids[i] for i in stable_held],
                "auxiliary_train_patient_ids": [
                    auxiliary.patient_ids[i] for i in aux_train_local
                ],
                "auxiliary_same_fold_excluded_patient_ids": [
                    auxiliary.patient_ids[i] for i in aux_excluded_local
                ],
                "stable_train_patient_count": len(stable_train),
                "stable_held_patient_count": len(stable_held),
                "auxiliary_train_patient_count": len(aux_train_local),
                "combined_loss_patient_count": len(combined_train),
                "stable_train_event_count": int(stable.event_counts[list(stable_train)].sum()),
                "stable_held_event_count": int(stable.event_counts[list(stable_held)].sum()),
                "auxiliary_train_event_count": int(
                    auxiliary.event_counts[list(aux_train_local)].sum()
                ),
                "feature_transform_fit_scope": "stable_outer_train_only",
                "jeffreys_prior_fit_scope": "stable_outer_train_only",
                "reasoner_loss_scope": "stable_outer_train_plus_aux_fold_not_equal_outer",
                "patient_equal_weight": True,
                "allow_candidate_subset": True,
                "fit_from_zero": True,
                "fit": dict(fit.diagnostics),
                "fit_state_sha256": _state_sha(fit.state),
                "held_stable_metrics": held_metrics,
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold": outer_fold,
                    "held_stable_patients": len(stable_held),
                    "auxiliary_train_patients": len(aux_train_local),
                    "strict": held_metrics["top1"]["strict_accuracy"],
                    "l2": l2,
                    "status": "complete",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not torch.isfinite(oof).all():
        raise RuntimeError("v17 stable-only OOF predictions are incomplete")

    candidate_metrics = _evaluate(oof, stable.targets, stable.target_mask)
    anchor_metrics = _evaluate(anchor.logits, stable.targets, stable.target_mask)
    paired = _paired_bootstrap(oof, anchor.logits, stable.targets, stable.target_mask)
    candidate_contributions = _patient_contributions(
        oof, stable.targets, stable.target_mask
    )
    anchor_contributions = _patient_contributions(
        anchor.logits, stable.targets, stable.target_mask
    )
    candidate_strict_count = float(candidate_contributions["strict"].sum())
    anchor_strict_count = float(anchor_contributions["strict"].sum())
    candidate_fold_strict = [
        float(row["held_stable_metrics"]["top1"]["strict_accuracy"])
        for row in fold_results
    ]
    anchor_fold_strict = []
    for outer_fold in OUTER_FOLDS:
        _, held = _stable_outer_partition(stable.patient_folds, outer_fold)
        held_tensor = torch.tensor(held, dtype=torch.long)
        anchor_fold_strict.append(
            float(
                _evaluate(
                    anchor.logits.index_select(0, held_tensor),
                    stable.targets.index_select(0, held_tensor),
                    stable.target_mask.index_select(0, held_tensor),
                )["top1"]["strict_accuracy"]
            )
        )

    source_hashes_after = _source_hashes()
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("v17 source files changed during execution")
    firewall_gates = {
        "phase0_no_aux_refit_oof_parity": bool(
            torch.allclose(
                phase0.oof_logits,
                anchor.logits,
                atol=PARITY_ATOL,
                rtol=PARITY_RTOL,
            )
        ),
        "auxiliary_startup_exact_9_patients": len(auxiliary.patient_ids) == 9,
        "auxiliary_startup_exact_182_events": len(auxiliary.event_ids) == 182,
        "auxiliary_target_join_protocol_bound": auxiliary.join.receipt["inputs"][
            "protocol_sha256"
        ]
        == protocol_sha,
        "auxiliary_mask_rule_verified_no_imputation": True,
        "auxiliary_same_fold_patient_excluded_from_every_fit": all(
            set(row["auxiliary_train_patient_ids"]).isdisjoint(
                row["auxiliary_same_fold_excluded_patient_ids"]
            )
            for row in fold_results
        ),
        "stable_held_patients_never_enter_combined_fit": all(
            set(row["stable_held_patient_ids"]).isdisjoint(
                row["stable_train_patient_ids"]
            )
            for row in fold_results
        ),
        "feature_transform_stable_outer_train_only": all(
            row["feature_transform_fit_scope"] == "stable_outer_train_only"
            for row in fold_results
        ),
        "jeffreys_prior_stable_outer_train_only": all(
            row["jeffreys_prior_fit_scope"] == "stable_outer_train_only"
            for row in fold_results
        ),
        "stable_auxiliary_patient_and_event_rosters_disjoint": not (
            set(stable.patient_ids) & set(auxiliary.patient_ids)
            or set(stable.stable_event_ids) & set(auxiliary.event_ids)
        ),
        "prefix_cache_independent_and_frozen": True,
        "fine_cache_independent": True,
        "reasoner_trainable_parameter_count_exact_36": all(
            row["fit"]["trainable_parameter_count"] == 36 for row in fold_results
        ),
        "foundation_optimizer_parameter_count_zero": True,
        "private_access_and_forward_count_zero": True,
        "no_final_deploy_refit": True,
    }
    passed, stop_checks, fold_nonlower = _assess_stop_gate(
        candidate_metrics=candidate_metrics,
        anchor_metrics=anchor_metrics,
        paired=paired,
        candidate_strict_count=candidate_strict_count,
        anchor_strict_count=anchor_strict_count,
        candidate_fold_strict=candidate_fold_strict,
        anchor_fold_strict=anchor_fold_strict,
        firewall_gates=firewall_gates,
    )
    candidate_contralateral = _contralateral_far_count(
        oof, stable.targets, stable.target_mask
    )
    anchor_contralateral = _contralateral_far_count(
        anchor.logits, stable.targets, stable.target_mask
    )

    manifest = {
        "schema_version": SCHEMA,
        "status": "completed_one_shot_exploratory_public_development_oof",
        "decision": (
            "MASKED_VARIABLE_AUXILIARY_RETAIN_AS_ENGINEERING_CANDIDATE"
            if passed
            else "MASKED_VARIABLE_AUXILIARY_STOP_ON_CURRENT_PUBLIC_COHORT"
        ),
        "claim_boundary": {
            "fresh_test": False,
            "external_validation": False,
            "public_confirmation": False,
            "repeatedly_used_public_development_cohort": True,
            "clinical_deployment_allowed": False,
            "private_used": False,
            "auxiliary_patients_are_train_only": True,
        },
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "source_file_sha256": source_hashes_after,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "foundation": {
            "backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "trained_from_scratch": False,
            "foundation_prefix_blocks": "0_to_9_frozen",
            "foundation_optimizer_parameter_count": 0,
        },
        "stable_evaluation": {
            "patient_count": len(stable.patient_ids),
            "event_count": int(stable.event_counts.sum()),
            "patient_ids": list(stable.patient_ids),
            "patient_folds": stable.patient_folds.tolist(),
            "event_counts": stable.event_counts.tolist(),
            "fixed_candidate_count": int(V11_CANDIDATE_MASK.sum()),
            "fixed_non_candidate": "PZ",
            "independent_statistical_unit": "patient",
        },
        "auxiliary_training": {
            "patient_count": len(auxiliary.patient_ids),
            "event_count": len(auxiliary.event_ids),
            "patient_ids": list(auxiliary.patient_ids),
            "outer_folds": auxiliary.outer_folds.tolist(),
            "event_counts": auxiliary.event_counts.tolist(),
            "evaluation_denominator": 0,
            "threshold_or_promotion_denominator": 0,
            "target_semantics": "stable_explicit_DeepSOZ_reference_values_only",
            "partial_mask_policy": "patient_variable_missing_and_PZ_masked",
            "positive_set_mass_patient_equal_weight": True,
        },
        "phase0_no_aux_refit_parity": {
            "completed_before_auxiliary_target_join_opened": True,
            "all_folds_passed": True,
            "atol": PARITY_ATOL,
            "rtol": PARITY_RTOL,
            "folds": list(phase0.fold_receipts),
            "complete_oof_max_abs_difference": float(
                (phase0.oof_logits - anchor.logits).abs().max()
            ),
        },
        "model_contract": {
            "evidence": "block9_H16_plus_fine20",
            "trainable_reasoner_parameters": 36,
            "channel_bias": False,
            "candidate_specific_parameters": False,
            "loss": "patient_equal_exact_positive_set_mass_plus_fold_pinned_L2",
            "allow_candidate_subset_for_combined_loss": True,
            "feature_transform_scope": "stable_outer_train_only",
            "jeffreys_prior_scope": "stable_outer_train_only",
            "pinned_l2_by_outer_fold": list(PINNED_FOLD_L2),
            "inner_model_selection_performed": False,
            "seed_head_layer_pooling_aux_weight_scan_performed": False,
            "final_refit_performed": False,
        },
        "fold_results": fold_results,
        "primary_comparison": {
            "candidate_name": CANDIDATE_NAME,
            "anchor_name": ANCHOR_NAME,
            "scope": "same_102_repeatedly_used_public_development_not_confirmation",
            "candidate_metrics": candidate_metrics,
            "anchor_metrics": anchor_metrics,
            "candidate_absolute_patient_bootstrap": _absolute_bootstrap(
                oof, stable.targets, stable.target_mask
            ),
            "paired_candidate_minus_anchor": paired,
            "candidate_strict_count": candidate_strict_count,
            "anchor_strict_count": anchor_strict_count,
            "strict_net_gain": candidate_strict_count - anchor_strict_count,
            "strict_win_loss_tie": _strict_win_loss_tie(
                oof, anchor.logits, stable.targets, stable.target_mask
            ),
            "top1_agreement": _top1_agreement(oof, anchor.logits),
            "candidate_fold_strict": candidate_fold_strict,
            "anchor_fold_strict": anchor_fold_strict,
            "fold_strict_nonlower_count": fold_nonlower,
            "candidate_contralateral_far_count": candidate_contralateral,
            "anchor_contralateral_far_count": anchor_contralateral,
            "neighbor_eligible_denominator_candidate": candidate_metrics["top1"][
                "n_neighbor_eligible_samples"
            ],
            "neighbor_eligible_denominator_anchor": anchor_metrics["top1"][
                "n_neighbor_eligible_samples"
            ],
        },
        "frozen_stop_gate": {
            "all_passed": passed,
            "checks": stop_checks,
            "firewall_gates": firewall_gates,
            "failure_action": (
                "no_majority_union_aux_weight_mask_threshold_L2_block_seed_pooling_"
                "or_channel_prior_scan_on_current_102"
            ),
        },
        "lineage_axes": {
            "direct_target_values": {
                "used": True,
                "evidence": "stable target-v2 and auxiliary masked target join enter reasoner supervision",
            },
            "upstream_target_conditioned_roster": {
                "used": True,
                "evidence": "stable C18 roster and auxiliary admission roster are target-conditioned",
            },
            "target_supervised_model": {
                "used": True,
                "evidence": "each outer-fold 36-parameter reasoner is fitted with target supervision",
            },
        },
        "lineage": {
            **stable.lineage,
            "auxiliary_join_artifact_sha256": auxiliary.join.artifact_sha256,
            "auxiliary_join_receipt_sha256": auxiliary.join.receipt_sha256,
            "auxiliary_admission_artifact_sha256": auxiliary.join.admission_artifact_sha256,
            "auxiliary_admission_receipt_sha256": auxiliary.join.admission_receipt_sha256,
            "auxiliary_prefix_manifest_sha256": aux_prefix.manifest_sha256,
            "auxiliary_prefix_tensor_file_sha256": aux_prefix.tensor_file_sha256,
            "auxiliary_fine_manifest_sha256": aux_fine.manifest_sha256,
            "auxiliary_fine_tensor_file_sha256": aux_fine.tensor_file_sha256,
            "pinned_identity_v16_manifest_sha256": anchor.manifest_sha256,
            "pinned_identity_v16_oof_sha256": anchor.oof_sha256,
        },
        "access_receipt": {
            "phase0_completed_before_auxiliary_target_join_opened": True,
            "auxiliary_target_values_loaded": True,
            "auxiliary_patient_specific_mask_used_only_in_training_loss": True,
            "stable_held_metrics_use_fixed_C18_mask_only": True,
            "auxiliary_predictions_evaluated": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_forward_count": 0,
            "historical_anchor_predictions_used_only_for_parity_and_frozen_comparison": True,
            "foundation_training_performed": False,
            "foundation_optimizer_parameter_count": 0,
            "final_development_or_deploy_refit_performed": False,
            "llm_used_as_soz_predictor": False,
        },
    }
    oof_tensors = {
        f"oof.{CANDIDATE_NAME}": oof,
        f"oof.{ANCHOR_NAME}": anchor.logits,
        "oof.phase0_no_aux_refit": phase0.oof_logits,
        "targets": stable.targets,
        "target_mask": stable.target_mask,
        "patient_folds": stable.patient_folds,
        "patient_event_counts": stable.event_counts,
        "config.candidate_mask": V11_CANDIDATE_MASK.clone(),
        "config.pinned_fold_l2": torch.tensor(PINNED_FOLD_L2, dtype=torch.float32),
        "auxiliary_outer_folds": auxiliary.outer_folds,
        "auxiliary_event_counts": auxiliary.event_counts,
    }
    return manifest, oof_tensors, outer_states


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(
    output_directory: Path,
    manifest: Mapping[str, object],
    oof_tensors: Mapping[str, torch.Tensor],
    outer_states: Mapping[str, torch.Tensor],
) -> Path:
    """Atomically publish OOF/state receipts without a final checkpoint."""

    target = Path(os.path.abspath(output_directory))
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise FileNotFoundError(target.parent)
    if "final_checkpoint" in " ".join((*oof_tensors.keys(), *outer_states.keys())):
        raise ValueError("v17 publication forbids a final checkpoint")
    if not oof_tensors or not outer_states:
        raise ValueError("v17 publication requires OOF tensors and fold-local states")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        oof_path = staging / "oof_predictions.safetensors"
        state_path = staging / "outer_fold_states.safetensors"
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in oof_tensors.items()},
            str(oof_path),
        )
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in outer_states.items()},
            str(state_path),
        )
        _fsync_file(oof_path)
        _fsync_file(state_path)
        completed = dict(manifest)
        completed["files"] = {
            "oof_predictions.safetensors": {
                "sha256": _file_sha(oof_path),
                "size_bytes": oof_path.stat().st_size,
            },
            "outer_fold_states.safetensors": {
                "sha256": _file_sha(state_path),
                "size_bytes": state_path.stat().st_size,
            },
        }
        completed["publication_contract"] = {
            "atomic_non_overwriting_directory_publish": True,
            "final_checkpoint_published": False,
            "deploy_refit_published": False,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_canonical_bytes(completed, newline=True))
        _fsync_file(manifest_path)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.replace(staging, target)
        published = True
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--stable-fine-directory", type=Path, default=DEFAULT_STABLE_FINE)
    parser.add_argument("--stable-prefix-directory", type=Path, default=DEFAULT_STABLE_PREFIX)
    parser.add_argument("--legacy-fine-directory", type=Path, default=DEFAULT_LEGACY_FINE)
    parser.add_argument("--legacy-prefix-directory", type=Path, default=DEFAULT_LEGACY_PREFIX)
    parser.add_argument("--target-directory", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--aux-join-directory", type=Path, default=DEFAULT_AUX_JOIN)
    parser.add_argument("--aux-prefix-directory", type=Path, default=DEFAULT_AUX_PREFIX)
    parser.add_argument("--aux-fine-directory", type=Path, default=DEFAULT_AUX_FINE)
    parser.add_argument("--anchor-directory", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-protocol-sha256", default=EXPECTED_PROTOCOL_SHA256
    )
    parser.add_argument(
        "--expected-union-manifest-sha256",
        default=EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-stable-fine-manifest-sha256",
        default=identity_v16.EXPECTED_FINE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-stable-fine-tensor-sha256",
        default=identity_v16.EXPECTED_FINE_TENSOR_SHA256,
    )
    parser.add_argument(
        "--expected-stable-prefix-manifest-sha256",
        default=identity_v16.EXPECTED_PREFIX_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-stable-prefix-tensor-sha256",
        default=identity_v16.EXPECTED_PREFIX_TENSOR_SHA256,
    )
    parser.add_argument(
        "--expected-aux-join-artifact-sha256",
        default=EXPECTED_AUX_JOIN_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--expected-aux-admission-artifact-sha256",
        default=EXPECTED_AUX_ADMISSION_ARTIFACT_SHA256,
    )
    # Full auxiliary caches are intentionally not materialized by this code
    # change.  Their content hashes must be supplied explicitly after atomic
    # publication; accepting an unpinned cache is forbidden.
    parser.add_argument("--expected-aux-prefix-manifest-sha256", required=True)
    parser.add_argument("--expected-aux-prefix-tensor-sha256", required=True)
    parser.add_argument("--expected-aux-fine-manifest-sha256", required=True)
    parser.add_argument("--expected-aux-fine-tensor-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = parse_args(argv)
    manifest, oof_tensors, outer_states = run(args)
    output = _publish(args.output_directory, manifest, oof_tensors, outer_states)
    completed = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    comparison = completed["primary_comparison"]
    print(
        json.dumps(
            {
                "status": completed["status"],
                "decision": completed["decision"],
                "path": str(output),
                "manifest_sha256": _file_sha(output / "manifest.json"),
                "strict": comparison["candidate_metrics"]["top1"]["strict_accuracy"],
                "relaxed": comparison["candidate_metrics"]["top1"]["relaxed_accuracy"],
                "macro_ap": comparison["candidate_metrics"]["ranking"][
                    "macro_average_precision"
                ],
                "far_errors": comparison["candidate_metrics"]["far_error_count"],
                "private_used": False,
                "final_refit": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
