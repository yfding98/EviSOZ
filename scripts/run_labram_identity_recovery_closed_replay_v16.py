#!/usr/bin/env python3
"""Run the frozen LaBraM v11.1 identity-recovery closed replay (v16).

This is an internal public-development coverage audit.  It does not replace
LaBraM, train the foundation encoder, read private data, or create an
independent/confirmatory test result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePath
import platform
import sys
from typing import Mapping, Sequence

import safetensors
from safetensors.torch import load_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_fine_temporal_nested_oof_v11 import (  # noqa: E402
    ARMS,
    INNER_FOLDS,
    L2_CANDIDATES,
    OUTER_FOLDS,
    _InnerContext,
    _canonical_bytes,
    _file_sha,
    _fit_reasoner,
    _inner_assignments,
    _load_json_manifest,
    _select_l2,
    _state_sha,
    _transform_state,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _absolute_bootstrap,
    _artifact_selective_coverage,
    _complete_candidate_label_rows,
    _evaluate,
    _event_consistency,
    _event_count_strata,
    _load_reasoner_from_fit,
    _paired_bootstrap,
    _patient_contributions,
    _publish,
    _require_fixed_rows,
)
from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    TARGET_V2_POLICY_SHA256,
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.data.identity_v12_cache_extension import (  # noqa: E402
    LEGACY_EVENT_COUNT,
    tensor_sha256,
)
from src.soz.data.public_development_union_identity_v12 import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256,
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256,
    PublicDevelopmentUnionIdentityV12,
    load_public_development_union_identity_v12,
)
from src.soz.fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    apply_fixed_candidate_mask,
    extract_block9_phase_contrasts,
    fit_fold_transform,
    jeffreys_reference_prior_logits,
    robust_pool_complete_patient_bags,
)


SCHEMA = "soz_labram_identity_recovery_closed_replay_v16"
PROTOCOL_PATH = ROOT / (
    "docs/method/reference/labram_identity_recovery_closed_replay_protocol_20260812_zh.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "435662e616244979530fdf44774236517b32d1360565eb75ea203efb22b73a2a"
)

DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_FINE = (
    ROOT / "outputs/public_development_fine_evidence_identity_v12_20260812"
)
DEFAULT_PREFIX = (
    ROOT / "outputs/public_development_labram_prefix_identity_v12_20260812"
)
DEFAULT_LEGACY_FINE = ROOT / "outputs/public_development_fine_evidence_v11_20260811"
DEFAULT_LEGACY_PREFIX = ROOT / "outputs/public_development_labram_prefix_v11_20260811"
DEFAULT_TARGET = ROOT / "outputs/deepsoz_target_v2_identity_recovery_20260812"
DEFAULT_SOURCE = (
    ROOT
    / "outputs/deepsoz_tusz_adapted_manifest_20260803/source/"
    "TUH_manifest_final.csv"
)
DEFAULT_SPLIT = (
    ROOT
    / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/"
    "split_manifest.csv"
)
DEFAULT_FROZEN_V11_R2 = (
    ROOT / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811_r2"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_20260812"
)

EXPECTED_TARGET_ARTIFACT_SHA256 = (
    "40fd23e37d0a10d77944359d85418b340eaba9a1c1e0ccb68133b7a69def21b1"
)
EXPECTED_TARGET_SUMMARY_SHA256 = (
    "a0713e870a8d5d36fef8d566d679c12e0509a8df742f219a4358c7d825aecbbf"
)
EXPECTED_TARGET_README_SHA256 = (
    "0c14c86e3d3fabd4af0fd6c737013269d7b8c469309f7d9b0edbfbc36652a497"
)
EXPECTED_SOURCE_SHA256 = (
    "4d08552dbb94f1e8e8a3931249d2bd29538233e2282b8d21a39d0f5dd873fd5c"
)
EXPECTED_SPLIT_SHA256 = (
    "07d26ff3008469fab078f00c1ac651481bdaa7b918ba460a2395e51ef9ae69c4"
)
EXPECTED_TARGET_RECEIPT_SHA256 = (
    "fa6a224764438e21667ba7198e510c90b53063c93de449c46cd2c813d8b4594a"
)
EXPECTED_LEGACY_FINE_MANIFEST_SHA256 = (
    "60ce6c5af15dcff3a0c0dcbac1451f4d5cb3bb28e7b9c22180ab7adecfb417a2"
)
EXPECTED_LEGACY_FINE_TENSOR_SHA256 = (
    "24dc5da224c79446992cde08d800877ff1ea4349d217c225da95588c9e173bbb"
)
EXPECTED_LEGACY_PREFIX_MANIFEST_SHA256 = (
    "b3ce8913a33848b7a706f8b30ccedf09ad8b2f6ae27412b1ae56d187866ff71f"
)
EXPECTED_LEGACY_PREFIX_TENSOR_SHA256 = (
    "40396fabac11ead6ac870ee69f428951f0577445c291a45b58e37c8fc6bf12bc"
)
EXPECTED_FINE_MANIFEST_SHA256 = (
    "6368cd6ea7ec30217bb69f3a742e1ee697dcae109125948a58b4fc927c2a8839"
)
EXPECTED_FINE_TENSOR_SHA256 = (
    "8f5dc0ab75eeeeeffda1a70650218d6e02f35f20a316481b1bb4028ca851809a"
)
EXPECTED_PREFIX_MANIFEST_SHA256 = (
    "defb6e608051e2767b49d8a566b6d0f5ea768e0f22d5a1fb46b28929df2fbe64"
)
EXPECTED_PREFIX_TENSOR_SHA256 = (
    "727382c1d072b6b4a59a7bdec3f6ff8c7e771179cd3eb1fe6c2550840b58583d"
)
EXPECTED_FROZEN_V11_R2_MANIFEST_SHA256 = (
    "f399678e5756ae30cbe5f9f87d9d8bb5b220b16015e1b2a0417110f20e70195c"
)
EXPECTED_FROZEN_V11_R2_OOF_SHA256 = (
    "6443680b18b53b0c552b9634e7c9e2547284c9d08cccd5cd99c35b9e1a27ac08"
)

UNION_PATIENT_COUNT = 103
UNION_EVENT_COUNT = 1149
PRIMARY_PATIENT_COUNT = 102
PRIMARY_EVENT_COUNT = 1145
OLD_INTERSECTION_PATIENT_COUNT = 101
NEW_PATIENT_ID = "10489"
NEW_PATIENT_OUTER_FOLD = 1
NEW_PATIENT_EVENT_COUNT = 27
EXCLUDED_PARTIAL_REFERENCE_PATIENT = "258"
FULL_ARM = "full_frozen_labram_plus_fine"


@dataclass(frozen=True)
class IdentityCache:
    directory: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    tensor_path: Path
    tensor_file_sha256: str
    tensor: torch.Tensor


@dataclass(frozen=True)
class PrimaryRoster:
    patient_ids: tuple[str, ...]
    patient_folds: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    selected_union_indices: torch.Tensor


@dataclass(frozen=True)
class FrozenV11R2:
    patient_ids: tuple[str, ...]
    logits: Mapping[str, torch.Tensor]
    targets: torch.Tensor
    target_mask: torch.Tensor
    patient_folds: torch.Tensor
    manifest_sha256: str
    oof_sha256: str


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


def _cache_access_firewall(manifest: Mapping[str, object], *, label: str) -> None:
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping):
        raise TypeError(f"{label} lacks an access receipt")
    forbidden = (
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "historical_prediction_artifacts_loaded",
    )
    failed = tuple(name for name in forbidden if access.get(name) is not False)
    if failed:
        raise ValueError(f"{label} target/private/prediction firewall failed: {failed}")
    for training_key in (
        "training_performed",
        "foundation_training_performed",
        "reasoner_training_performed",
    ):
        if training_key in access and access[training_key] is not False:
            raise ValueError(f"{label} unexpectedly performed training")


def _safe_tensor_path(directory: Path, manifest: Mapping[str, object]) -> Path:
    filename = manifest.get("tensor_file")
    if not isinstance(filename, str) or not filename:
        raise TypeError("cache tensor_file is missing")
    pure = PurePath(filename)
    if pure.is_absolute() or len(pure.parts) != 1 or pure.suffix != ".safetensors":
        raise ValueError("cache tensor_file is not a safe basename")
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError("cache tensor file must be regular and non-symlinked")
    return path


def _validate_cache_rows(
    manifest: Mapping[str, object],
    union: PublicDevelopmentUnionIdentityV12,
    *,
    label: str,
) -> None:
    rows = manifest.get("events")
    if not isinstance(rows, list) or len(rows) != UNION_EVENT_COUNT:
        raise ValueError(f"{label} event rows do not cover 1149 events")
    for index, (raw, expected) in enumerate(zip(rows, union.events)):
        if not isinstance(raw, Mapping):
            raise TypeError(f"{label} events[{index}] is not an object")
        checks = {
            "event_id": expected.event_id,
            "patient_id": expected.patient_id,
            "outer_fold": expected.outer_fold,
            "processed_window_sha256": expected.processed_window_sha256,
        }
        for key, value in checks.items():
            if key in raw and raw.get(key) != value:
                raise ValueError(f"{label} {key} differs at event {expected.event_id}")


def _load_identity_cache(
    directory: Path,
    *,
    expected_manifest_sha256: str,
    expected_tensor_sha256: str,
    tensor_key: str,
    tensor_tail_shape: tuple[int, ...],
    union: PublicDevelopmentUnionIdentityV12,
    legacy_directory: Path,
    expected_legacy_manifest_sha256: str,
    expected_legacy_tensor_sha256: str,
    label: str,
) -> IdentityCache:
    root = _regular_directory(directory, name=f"{label} directory")
    expected_manifest = _require_sha256(
        expected_manifest_sha256, name=f"expected {label} manifest SHA"
    )
    expected_tensor = _require_sha256(
        expected_tensor_sha256, name=f"expected {label} tensor SHA"
    )
    manifest = _load_json_manifest(root / "manifest.json", expected_sha=expected_manifest)
    if manifest.get("event_count") != UNION_EVENT_COUNT or (
        manifest.get("patient_count") != UNION_PATIENT_COUNT
    ):
        raise ValueError(f"{label} does not cover the complete 103/1149 union")
    event_ids = tuple(str(value) for value in manifest.get("event_ids", ()))
    union_event_ids = tuple(event.event_id for event in union.events)
    if event_ids != union_event_ids:
        raise ValueError(f"{label} event order differs from union identity-v12")
    _cache_access_firewall(manifest, label=label)
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping) or (
        lineage.get("public_union_manifest_sha256") != union.manifest_sha256
    ):
        raise ValueError(f"{label} is not bound to the loaded union manifest")
    extension = manifest.get("cache_extension_receipt")
    if not isinstance(extension, Mapping):
        raise TypeError(f"{label} lacks append-only cache receipt")
    required_true = (
        "append_only",
        "legacy_event_rows_exact_prefix",
        "legacy_event_ids_exact_prefix",
        "legacy_tensor_prefix_exact",
    )
    if any(extension.get(key) is not True for key in required_true):
        raise ValueError(f"{label} append-only legacy parity failed")
    _validate_cache_rows(manifest, union, label=label)

    tensor_path = _safe_tensor_path(root, manifest)
    actual_tensor_file_sha = _file_sha(tensor_path)
    if actual_tensor_file_sha != expected_tensor or (
        manifest.get("tensor_file_sha256") != expected_tensor
    ):
        raise ValueError(f"{label} tensor file SHA mismatch")
    payload = load_file(str(tensor_path), device="cpu")
    if tensor_key not in payload:
        raise ValueError(f"{label} tensor key {tensor_key!r} is missing")
    tensor = payload[tensor_key].detach().cpu().contiguous()
    if tuple(tensor.shape) != (UNION_EVENT_COUNT, *tensor_tail_shape):
        raise ValueError(f"{label} tensor shape changed: {tuple(tensor.shape)}")
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError(f"{label} tensor is not finite floating point")
    specs = manifest.get("tensor_specs")
    if isinstance(specs, Mapping) and isinstance(specs.get(tensor_key), Mapping):
        spec = specs[tensor_key]
        if (
            spec.get("shape") != list(tensor.shape)
            or spec.get("dtype") != str(tensor.dtype)
            or spec.get("tensor_sha256") != tensor_sha256(tensor)
        ):
            raise ValueError(f"{label} tensor spec/hash does not replay")

    legacy_root = _regular_directory(
        legacy_directory, name=f"legacy {label} directory"
    )
    legacy_manifest = _load_json_manifest(
        legacy_root / "manifest.json",
        expected_sha=_require_sha256(
            expected_legacy_manifest_sha256,
            name=f"expected legacy {label} manifest SHA",
        ),
    )
    legacy_path = _safe_tensor_path(legacy_root, legacy_manifest)
    if _file_sha(legacy_path) != _require_sha256(
        expected_legacy_tensor_sha256,
        name=f"expected legacy {label} tensor SHA",
    ):
        raise ValueError(f"legacy {label} tensor SHA mismatch")
    legacy_payload = load_file(str(legacy_path), device="cpu")
    if tensor_key not in legacy_payload:
        raise ValueError(f"legacy {label} tensor key is missing")
    legacy_tensor = legacy_payload[tensor_key].detach().cpu()
    if tuple(legacy_tensor.shape) != (LEGACY_EVENT_COUNT, *tensor_tail_shape):
        raise ValueError(f"legacy {label} tensor shape changed")
    if not torch.equal(tensor[:LEGACY_EVENT_COUNT], legacy_tensor):
        raise ValueError(f"{label} legacy 988 tensor prefix changed")
    del payload, legacy_payload, legacy_tensor
    return IdentityCache(
        directory=root,
        manifest=manifest,
        manifest_sha256=expected_manifest,
        tensor_path=tensor_path,
        tensor_file_sha256=actual_tensor_file_sha,
        tensor=tensor,
    )


def _load_primary_roster(
    union: PublicDevelopmentUnionIdentityV12,
    *,
    target_directory: Path,
    source_csv: Path,
    split_csv: Path,
) -> tuple[PrimaryRoster, object]:
    target = load_verified_deepsoz_target_v2_artifact(
        target_directory,
        source_csv,
        split_csv,
        expected_target_artifact_sha256=EXPECTED_TARGET_ARTIFACT_SHA256,
        expected_summary_artifact_sha256=EXPECTED_TARGET_SUMMARY_SHA256,
        expected_readme_artifact_sha256=EXPECTED_TARGET_README_SHA256,
        expected_source_input_sha256=EXPECTED_SOURCE_SHA256,
        expected_split_input_sha256=EXPECTED_SPLIT_SHA256,
    )
    if target.receipt.receipt_sha256 != EXPECTED_TARGET_RECEIPT_SHA256 or (
        target.receipt.policy_sha256 != TARGET_V2_POLICY_SHA256
    ):
        raise ValueError("identity-recovery target-v2 receipt/policy changed")
    batch = target.registry.target_batch(union.patient_ids, require_eligible=True)
    targets_all = batch.values.detach().cpu()
    mask_all = batch.mask.detach().cpu()
    complete = _complete_candidate_label_rows(mask_all)
    excluded = tuple(
        union.patient_ids[index]
        for index in torch.nonzero(~complete, as_tuple=False).flatten().tolist()
    )
    if excluded != (EXCLUDED_PARTIAL_REFERENCE_PATIENT,):
        raise ValueError(f"unexpected incomplete C18 roster: {excluded}")
    selected = torch.nonzero(complete, as_tuple=False).flatten()
    if selected.numel() != PRIMARY_PATIENT_COUNT:
        raise ValueError("identity replay requires 102 C18-complete patients")
    patient_ids = tuple(union.patient_ids[index] for index in selected.tolist())
    if patient_ids[-1] != NEW_PATIENT_ID or patient_ids.count(NEW_PATIENT_ID) != 1:
        raise ValueError("identity replay must add exactly patient 10489")
    targets = targets_all.index_select(0, selected)
    target_mask = mask_all.index_select(0, selected)
    _require_fixed_rows(target_mask)
    if not bool(((targets == 1) & target_mask).any(dim=1).all()):
        raise ValueError("every primary patient requires a C18 reference positive")
    folds = torch.tensor(union.patient_folds, dtype=torch.long).index_select(0, selected)
    new_index = patient_ids.index(NEW_PATIENT_ID)
    if int(folds[new_index]) != NEW_PATIENT_OUTER_FOLD:
        raise ValueError("patient 10489 must remain in frozen outer fold 1")
    return (
        PrimaryRoster(
            patient_ids=patient_ids,
            patient_folds=folds,
            targets=targets,
            target_mask=target_mask,
            selected_union_indices=selected,
        ),
        target,
    )


def _load_frozen_v11_r2(directory: Path) -> FrozenV11R2:
    root = _regular_directory(directory, name="frozen v11.1 r2 directory")
    manifest = _load_json_manifest(
        root / "manifest.json", expected_sha=EXPECTED_FROZEN_V11_R2_MANIFEST_SHA256
    )
    if manifest.get("schema_version") != "soz_labram_fine_temporal_nested_oof_v11_1":
        raise ValueError("frozen v11.1 r2 schema changed")
    patient_ids = tuple(str(value) for value in manifest.get("patient_ids", ()))
    if len(patient_ids) != OLD_INTERSECTION_PATIENT_COUNT or len(set(patient_ids)) != len(
        patient_ids
    ):
        raise ValueError("frozen v11.1 r2 patient roster changed")
    path = root / "oof_predictions.safetensors"
    if path.is_symlink() or not path.is_file() or (
        _file_sha(path) != EXPECTED_FROZEN_V11_R2_OOF_SHA256
    ):
        raise ValueError("frozen v11.1 r2 OOF file changed")
    payload = load_file(str(path), device="cpu")
    logits = {
        name: payload[f"oof.{name}"].detach().cpu()
        for name in ("prevalence_only", *ARMS.keys())
    }
    for name, value in logits.items():
        if tuple(value.shape) != (OLD_INTERSECTION_PATIENT_COUNT, 19) or not torch.isfinite(
            value
        ).all():
            raise ValueError(f"frozen v11.1 r2 {name} logits changed")
    if not torch.equal(payload["config.candidate_mask"].cpu(), V11_CANDIDATE_MASK):
        raise ValueError("frozen v11.1 r2 candidate mask changed")
    return FrozenV11R2(
        patient_ids=patient_ids,
        logits=logits,
        targets=payload["targets"].detach().cpu(),
        target_mask=payload["target_mask"].detach().cpu(),
        patient_folds=payload["patient_folds"].detach().cpu(),
        manifest_sha256=EXPECTED_FROZEN_V11_R2_MANIFEST_SHA256,
        oof_sha256=EXPECTED_FROZEN_V11_R2_OOF_SHA256,
    )


def _assert_old101_alignment(
    roster: PrimaryRoster,
    frozen: FrozenV11R2,
) -> torch.Tensor:
    rows = torch.tensor(
        [index for index, patient in enumerate(roster.patient_ids) if patient != NEW_PATIENT_ID],
        dtype=torch.long,
    )
    ids = tuple(roster.patient_ids[index] for index in rows.tolist())
    if ids != frozen.patient_ids:
        raise ValueError("old101 roster/order differs from frozen v11.1 r2")
    if not torch.equal(roster.targets.index_select(0, rows), frozen.targets):
        raise ValueError("old101 target values differ from frozen v11.1 r2")
    if not torch.equal(roster.target_mask.index_select(0, rows), frozen.target_mask):
        raise ValueError("old101 target masks differ from frozen v11.1 r2")
    if not torch.equal(roster.patient_folds.index_select(0, rows), frozen.patient_folds):
        raise ValueError("old101 outer folds differ from frozen v11.1 r2")
    return rows


def _paired_patient_rows(
    patient_ids: Sequence[str],
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> list[dict[str, object]]:
    if len(patient_ids) != candidate.shape[0] or candidate.shape != baseline.shape:
        raise ValueError("paired patient roster/logits are not aligned")
    candidate_values = _patient_contributions(candidate, targets, target_mask)
    baseline_values = _patient_contributions(baseline, targets, target_mask)
    candidate_top = apply_fixed_candidate_mask(candidate).argmax(dim=1)
    baseline_top = apply_fixed_candidate_mask(baseline).argmax(dim=1)
    rows: list[dict[str, object]] = []
    for index, patient_id in enumerate(patient_ids):
        changes = {
            name: float(candidate_values[name][index] - baseline_values[name][index])
            for name in candidate_values
        }
        rows.append(
            {
                "patient_id": patient_id,
                "replay_top1": STANDARD_19[int(candidate_top[index])],
                "frozen_v11_1_r2_top1": STANDARD_19[int(baseline_top[index])],
                "top1_changed": bool(candidate_top[index] != baseline_top[index]),
                "metric_delta_replay_minus_frozen": changes,
            }
        )
    return rows


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner_v16": Path(__file__).resolve(),
        "runner_v11_1": ROOT / "scripts/run_labram_fine_temporal_nested_oof_v11_1.py",
        "runner_shared_v11": ROOT / "scripts/run_labram_fine_temporal_nested_oof_v11.py",
        "reasoner": ROOT / "src/soz/v11_reasoner.py",
        "metrics": ROOT / "src/soz/metrics.py",
        "target_loader": ROOT / "src/soz/data/deepsoz_target_v2.py",
        "union_loader": ROOT / "src/soz/data/public_development_union_identity_v12.py",
        "cache_loader": ROOT / "src/soz/data/identity_v12_cache_extension.py",
    }
    return {name: _file_sha(path) for name, path in paths.items()}


def run(
    args: argparse.Namespace,
) -> tuple[
    Mapping[str, object],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
]:
    if _file_sha(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("identity-recovery closed-replay protocol changed")
    source_hashes_before_target = _source_hashes()
    union = load_public_development_union_identity_v12(
        args.union_directory,
        expected_manifest_sha256=args.expected_union_manifest_sha256,
        expected_payload_sha256=EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_PAYLOAD_SHA256,
    )
    fine = _load_identity_cache(
        args.fine_directory,
        expected_manifest_sha256=args.expected_fine_manifest_sha256,
        expected_tensor_sha256=args.expected_fine_tensor_sha256,
        tensor_key="features",
        tensor_tail_shape=(19, 20),
        union=union,
        legacy_directory=args.legacy_fine_directory,
        expected_legacy_manifest_sha256=EXPECTED_LEGACY_FINE_MANIFEST_SHA256,
        expected_legacy_tensor_sha256=EXPECTED_LEGACY_FINE_TENSOR_SHA256,
        label="fine evidence identity-v12",
    )
    if tuple(fine.manifest.get("feature_names", ())) != FINE_TEMPORAL_FEATURE_NAMES:
        raise ValueError("fine temporal feature vocabulary changed")
    prefix = _load_identity_cache(
        args.prefix_directory,
        expected_manifest_sha256=args.expected_prefix_manifest_sha256,
        expected_tensor_sha256=args.expected_prefix_tensor_sha256,
        tensor_key="prefix_tokens",
        tensor_tail_shape=(15, 77, 200),
        union=union,
        legacy_directory=args.legacy_prefix_directory,
        expected_legacy_manifest_sha256=EXPECTED_LEGACY_PREFIX_MANIFEST_SHA256,
        expected_legacy_tensor_sha256=EXPECTED_LEGACY_PREFIX_TENSOR_SHA256,
        label="LaBraM block-9 prefix identity-v12",
    )

    fine_event_all = fine.tensor
    h_event_all = extract_block9_phase_contrasts(prefix.tensor)
    event_patient_index_all = torch.tensor(union.event_patient_index, dtype=torch.long)
    artifact_index = FINE_TEMPORAL_FEATURE_NAMES.index("artifact_burden_0_12s")
    reliability = (1.0 - fine_event_all[:, :, artifact_index]).clamp(0.0, 1.0)
    h_pool_all = robust_pool_complete_patient_bags(
        h_event_all, event_patient_index_all, len(union.patient_ids), reliability
    )
    fine_pool_all = robust_pool_complete_patient_bags(
        fine_event_all, event_patient_index_all, len(union.patient_ids), reliability
    )
    if not torch.equal(h_pool_all.event_counts, fine_pool_all.event_counts):
        raise RuntimeError("identity-v12 H/fine patient bags disagree")

    # First target-value read: union, folds, cache identity, feature definitions,
    # pooling, model arms and hyperparameter grid are already frozen above.
    roster, target = _load_primary_roster(
        union,
        target_directory=args.target_directory,
        source_csv=args.source_csv,
        split_csv=args.split_csv,
    )
    selected = roster.selected_union_indices
    patient_ids = roster.patient_ids
    patient_folds = roster.patient_folds
    targets = roster.targets
    target_mask = roster.target_mask
    h_patient = h_pool_all.features.index_select(0, selected).cpu()
    fine_patient = fine_pool_all.features.index_select(0, selected).cpu()
    event_counts = h_pool_all.event_counts.index_select(0, selected).cpu()
    if int(event_counts.sum()) != PRIMARY_EVENT_COUNT:
        raise ValueError("C18 primary cohort must contain 1145 events")
    new_patient_index = patient_ids.index(NEW_PATIENT_ID)
    if int(event_counts[new_patient_index]) != NEW_PATIENT_EVENT_COUNT:
        raise ValueError("patient 10489 must contain 27 eligible events")

    union_to_primary = torch.full((len(union.patient_ids),), -1, dtype=torch.long)
    union_to_primary[selected] = torch.arange(PRIMARY_PATIENT_COUNT)
    eligible_event = _complete_candidate_label_rows(
        target.registry.target_batch(union.patient_ids, require_eligible=True).mask.cpu()
    )[event_patient_index_all]
    event_patient_index = union_to_primary[event_patient_index_all[eligible_event]]
    h_event = h_event_all[eligible_event]
    fine_event = fine_event_all[eligible_event]
    if tuple(h_event.shape) != (PRIMARY_EVENT_COUNT, 19, 600) or tuple(
        fine_event.shape
    ) != (PRIMARY_EVENT_COUNT, 19, 20):
        raise RuntimeError("identity-replay eligible event carrier shape changed")
    del h_event_all, fine_event_all, reliability

    oof = {
        "prevalence_only": torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan),
        **{
            name: torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan)
            for name in ARMS
        },
    }
    event_oof_full = torch.full((PRIMARY_EVENT_COUNT, 19), torch.nan)
    fold_results: list[dict[str, object]] = []
    selected_l2_by_arm = {name: [] for name in ARMS}
    outer_states: dict[str, torch.Tensor] = {}

    for outer_fold in OUTER_FOLDS:
        held = tuple(
            torch.nonzero(patient_folds == outer_fold, as_tuple=False).flatten().tolist()
        )
        train = tuple(
            torch.nonzero(patient_folds != outer_fold, as_tuple=False).flatten().tolist()
        )
        if not held or not train:
            raise RuntimeError("identity replay lost an outer partition")
        transform = fit_fold_transform(h_patient, fine_patient, train)
        transformed = transform.apply(h_patient, fine_patient)
        for name, value in _transform_state(transform).items():
            outer_states[f"outer{outer_fold}.{name}"] = value
        train_tensor = torch.tensor(train, dtype=torch.long)
        held_tensor = torch.tensor(held, dtype=torch.long)
        prior = jeffreys_reference_prior_logits(
            targets.index_select(0, train_tensor),
            target_mask.index_select(0, train_tensor),
        )
        oof["prevalence_only"].index_copy_(0, held_tensor, prior.expand(len(held), -1))

        inner_assignment = _inner_assignments(
            train,
            patient_ids=patient_ids,
            event_counts=event_counts,
            outer_fold=outer_fold,
        )
        contexts: list[_InnerContext] = []
        inner_receipts: list[dict[str, object]] = []
        for inner_fold in INNER_FOLDS:
            inner_held = tuple(index for index in train if inner_assignment[index] == inner_fold)
            inner_train = tuple(index for index in train if inner_assignment[index] != inner_fold)
            inner_transform = fit_fold_transform(h_patient, fine_patient, inner_train)
            contexts.append(
                _InnerContext(
                    fold=inner_fold,
                    train_indices=inner_train,
                    held_indices=inner_held,
                    transformed=inner_transform.apply(h_patient, fine_patient),
                )
            )
            inner_receipts.append(
                {
                    "inner_fold": inner_fold,
                    "train_patient_ids": [patient_ids[index] for index in inner_train],
                    "held_patient_ids": [patient_ids[index] for index in inner_held],
                }
            )

        arm_rows: dict[str, object] = {}
        full_fit = None
        for arm, (use_h, use_fine) in ARMS.items():
            selected_l2, selection = _select_l2(
                contexts,
                targets,
                target_mask,
                use_h=use_h,
                use_fine=use_fine,
            )
            selected_l2_by_arm[arm].append(selected_l2)
            fitted = _fit_reasoner(
                transformed,
                targets,
                target_mask,
                train,
                use_h=use_h,
                use_fine=use_fine,
                l2=selected_l2,
            )
            oof[arm].index_copy_(0, held_tensor, fitted.logits.index_select(0, held_tensor))
            held_metrics = _evaluate(
                fitted.logits.index_select(0, held_tensor),
                targets.index_select(0, held_tensor),
                target_mask.index_select(0, held_tensor),
            )
            for name, value in fitted.state.items():
                outer_states[f"outer{outer_fold}.{arm}.{name}"] = value
            arm_rows[arm] = {
                "selected_l2": selected_l2,
                "inner_selection": selection,
                "fit": dict(fitted.diagnostics),
                "held_metrics": held_metrics,
            }
            if arm == FULL_ARM:
                full_fit = fitted
        if full_fit is None:
            raise RuntimeError("identity replay full arm was not fitted")
        event_transformed = transform.apply(h_event, fine_event)
        held_event_indices = torch.nonzero(
            torch.isin(event_patient_index, held_tensor), as_tuple=False
        ).flatten()
        full_model = _load_reasoner_from_fit(full_fit.state)
        with torch.no_grad():
            held_event_logits = full_model(
                event_transformed.index_select(held_event_indices)
            ).logits.cpu()
        event_oof_full.index_copy_(0, held_event_indices, held_event_logits)
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "train_patient_count": len(train),
                "held_patient_count": len(held),
                "train_event_count": int(event_counts[train_tensor].sum()),
                "held_event_count": int(event_counts[held_tensor].sum()),
                "train_patient_ids": [patient_ids[index] for index in train],
                "held_patient_ids": [patient_ids[index] for index in held],
                "inner_folds": inner_receipts,
                "arms": arm_rows,
                "prevalence_held_metrics": _evaluate(
                    oof["prevalence_only"].index_select(0, held_tensor),
                    targets.index_select(0, held_tensor),
                    target_mask.index_select(0, held_tensor),
                ),
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold": outer_fold,
                    "held_patients": len(held),
                    "full_strict": arm_rows[FULL_ARM]["held_metrics"]["top1"][
                        "strict_accuracy"
                    ],
                    "full_l2": arm_rows[FULL_ARM]["selected_l2"],
                    "status": "complete",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if any(not torch.isfinite(value).all() for value in oof.values()) or not torch.isfinite(
        event_oof_full
    ).all():
        raise RuntimeError("identity replay OOF predictions are incomplete")
    metrics = {name: _evaluate(value, targets, target_mask) for name, value in oof.items()}
    absolute_bootstrap = {
        name: _absolute_bootstrap(value, targets, target_mask)
        for name, value in oof.items()
    }
    paired_baselines = {
        name: _paired_bootstrap(oof[FULL_ARM], oof[name], targets, target_mask)
        for name in ("fine_change_only", "frozen_labram_only", "prevalence_only")
    }

    l2_counts = Counter(selected_l2_by_arm[FULL_ARM])
    final_l2 = max(
        L2_CANDIDATES,
        key=lambda value: (l2_counts[value], -abs(math.log(value / 0.05))),
    )
    all_indices = tuple(range(PRIMARY_PATIENT_COUNT))
    final_transform = fit_fold_transform(h_patient, fine_patient, all_indices)
    final_fit = _fit_reasoner(
        final_transform.apply(h_patient, fine_patient),
        targets,
        target_mask,
        all_indices,
        use_h=True,
        use_fine=True,
        l2=final_l2,
    )
    final_state = {
        **_transform_state(final_transform),
        **{f"reasoner.{name}": value for name, value in final_fit.state.items()},
        "config.l2": torch.tensor(final_l2, dtype=torch.float32),
        "config.candidate_mask": V11_CANDIDATE_MASK.clone(),
    }

    # Frozen historical predictions are opened only after every new OOF score
    # and final refit has been computed; they are comparison-only inputs.
    frozen = _load_frozen_v11_r2(args.frozen_v11_r2_directory)
    old_rows = _assert_old101_alignment(roster, frozen)
    old_targets = targets.index_select(0, old_rows)
    old_mask = target_mask.index_select(0, old_rows)
    old_ids = tuple(patient_ids[index] for index in old_rows.tolist())
    old_comparison: dict[str, object] = {}
    for name in ("prevalence_only", *ARMS.keys()):
        replay_logits = oof[name].index_select(0, old_rows)
        old_comparison[name] = {
            "replay_metrics": _evaluate(replay_logits, old_targets, old_mask),
            "frozen_v11_1_r2_metrics": _evaluate(
                frozen.logits[name], old_targets, old_mask
            ),
            "paired_replay_minus_frozen": _paired_bootstrap(
                replay_logits, frozen.logits[name], old_targets, old_mask
            ),
        }
    old_comparison[FULL_ARM]["per_patient"] = _paired_patient_rows(
        old_ids,
        oof[FULL_ARM].index_select(0, old_rows),
        frozen.logits[FULL_ARM],
        old_targets,
        old_mask,
    )

    new_row = torch.tensor([new_patient_index], dtype=torch.long)
    new_positive = torch.nonzero(
        (targets[new_patient_index] == 1) & target_mask[new_patient_index],
        as_tuple=False,
    ).flatten()
    new_top = int(apply_fixed_candidate_mask(oof[FULL_ARM])[new_patient_index].argmax())
    new_patient_result = {
        "patient_id": NEW_PATIENT_ID,
        "scope": "identity_recovery_extension_not_fresh_test",
        "outer_fold": int(patient_folds[new_patient_index]),
        "event_count": int(event_counts[new_patient_index]),
        "reference_positive_set": [STANDARD_19[index] for index in new_positive.tolist()],
        "top1_channel": STANDARD_19[new_top],
        "metrics": _evaluate(
            oof[FULL_ARM].index_select(0, new_row),
            targets.index_select(0, new_row),
            target_mask.index_select(0, new_row),
        ),
    }

    quality = 1.0 - fine_patient[:, :, artifact_index].mean(dim=1)
    source_hashes_after = _source_hashes()
    if source_hashes_after != source_hashes_before_target:
        raise RuntimeError("closed-replay source files changed during execution")
    full_metrics = metrics[FULL_ARM]
    manifest = {
        "schema_version": SCHEMA,
        "status": "completed_internal_developmental_identity_recovery_closed_replay",
        "decision": "coverage_recovery_audit_complete_no_confirmatory_claim",
        "claim_boundary": {
            "public_confirmation": False,
            "external_validation": False,
            "fresh_test": False,
            "all_102_patients_are_developmental": True,
            "private_used": False,
            "clinical_deployment_allowed": False,
        },
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
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
            "foundation_trainable_parameters": 0,
            "foundation_prefix_blocks": "0_to_9_frozen",
        },
        "union_patient_count": UNION_PATIENT_COUNT,
        "union_event_count": UNION_EVENT_COUNT,
        "primary_patient_count": PRIMARY_PATIENT_COUNT,
        "primary_event_count": PRIMARY_EVENT_COUNT,
        "independent_statistical_unit": "patient",
        "excluded_partial_reference_patients": [EXCLUDED_PARTIAL_REFERENCE_PATIENT],
        "patient_ids": list(patient_ids),
        "event_counts": event_counts.tolist(),
        "patient_folds": patient_folds.tolist(),
        "outer_folds": list(OUTER_FOLDS),
        "inner_fold_count": len(INNER_FOLDS),
        "arms": {name: list(config) for name, config in ARMS.items()},
        "l2_candidates": list(L2_CANDIDATES),
        "selected_l2_by_arm": selected_l2_by_arm,
        "fold_results": fold_results,
        "metrics_all_102": metrics,
        "absolute_patient_bootstrap_all_102": absolute_bootstrap,
        "paired_full_minus_baselines_all_102": paired_baselines,
        "old101_intersection_vs_frozen_v11_1_r2": {
            "patient_count": OLD_INTERSECTION_PATIENT_COUNT,
            "patient_ids": list(old_ids),
            "targets_elementwise_equal": True,
            "target_masks_elementwise_equal": True,
            "outer_folds_elementwise_equal": True,
            "comparison": old_comparison,
        },
        "patient_10489": new_patient_result,
        "event_count_strata": _event_count_strata(oof, targets, target_mask, event_counts),
        "event_to_patient_consistency": _event_consistency(
            oof[FULL_ARM], event_oof_full, event_patient_index, patient_ids
        ),
        "artifact_quality_selective_coverage": _artifact_selective_coverage(
            oof[FULL_ARM], targets, target_mask, quality
        ),
        "goal_thresholds_descriptive_only": {
            "strict_top1_ge_0_80": full_metrics["top1"]["strict_accuracy"] >= 0.80,
            "relaxed_top1_ge_0_85": full_metrics["top1"]["relaxed_accuracy"] >= 0.85,
            "not_used_for_model_selection": True,
        },
        "development_refit_non_deployable": {
            "selected_l2_by_outer_mode": final_l2,
            "outer_selected_l2_counts": {
                str(key): value for key, value in l2_counts.items()
            },
            "fit": dict(final_fit.diagnostics),
            "state_sha256": _state_sha(final_state),
            "foundation_weights_serialized": False,
            "clinical_deployment_allowed": False,
        },
        "lineage": {
            "union_manifest_sha256": union.manifest_sha256,
            "fine_manifest_sha256": fine.manifest_sha256,
            "fine_tensor_file_sha256": fine.tensor_file_sha256,
            "prefix_manifest_sha256": prefix.manifest_sha256,
            "prefix_tensor_file_sha256": prefix.tensor_file_sha256,
            "target_artifact_sha256": target.receipt.target_artifact_sha256,
            "target_receipt_sha256": target.receipt.receipt_sha256,
            "target_policy_sha256": target.receipt.policy_sha256,
            "frozen_v11_1_r2_manifest_sha256": frozen.manifest_sha256,
            "frozen_v11_1_r2_oof_sha256": frozen.oof_sha256,
            "legacy_988_fine_tensor_prefix_elementwise_equal": True,
            "legacy_988_prefix_tensor_prefix_elementwise_equal": True,
        },
        "access_receipt": {
            "target_values_loaded_only_after_union_folds_features_pooling_and_code_hashes_frozen": True,
            "frozen_v11_predictions_loaded_only_after_new_oof_complete": True,
            "patient_specific_target_mask_used_for_prediction": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_forward_count": 0,
            "foundation_training_performed": False,
            "llm_used_as_soz_predictor": False,
        },
    }
    oof_tensors = {
        **{f"oof.{name}": value for name, value in oof.items()},
        "oof.event_full": event_oof_full,
        "targets": targets,
        "target_mask": target_mask,
        "config.candidate_mask": V11_CANDIDATE_MASK.clone(),
        "patient_folds": patient_folds,
        "patient_event_counts": event_counts,
        "patient_artifact_quality": quality,
        "old101_primary_indices": old_rows,
        "new10489_primary_index": torch.tensor(new_patient_index, dtype=torch.long),
    }
    return manifest, oof_tensors, final_state, outer_states


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--fine-directory", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--prefix-directory", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--legacy-fine-directory", type=Path, default=DEFAULT_LEGACY_FINE)
    parser.add_argument(
        "--legacy-prefix-directory", type=Path, default=DEFAULT_LEGACY_PREFIX
    )
    parser.add_argument("--target-directory", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--frozen-v11-r2-directory", type=Path, default=DEFAULT_FROZEN_V11_R2
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-union-manifest-sha256",
        default=EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-fine-manifest-sha256", default=EXPECTED_FINE_MANIFEST_SHA256
    )
    parser.add_argument(
        "--expected-fine-tensor-sha256", default=EXPECTED_FINE_TENSOR_SHA256
    )
    parser.add_argument(
        "--expected-prefix-manifest-sha256", default=EXPECTED_PREFIX_MANIFEST_SHA256
    )
    parser.add_argument(
        "--expected-prefix-tensor-sha256", default=EXPECTED_PREFIX_TENSOR_SHA256
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = parse_args(argv)
    manifest, oof, final_state, outer_states = run(args)
    path = _publish(args.output_directory, manifest, oof, final_state, outer_states)
    completed = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    full = completed["metrics_all_102"][FULL_ARM]
    print(
        json.dumps(
            {
                "status": completed["status"],
                "path": str(path),
                "manifest_sha256": _file_sha(path / "manifest.json"),
                "patients": completed["primary_patient_count"],
                "events": completed["primary_event_count"],
                "strict_top1": full["top1"]["strict_accuracy"],
                "relaxed_top1": full["top1"]["relaxed_accuracy"],
                "macro_ap": full["ranking"]["macro_average_precision"],
                "private_used": False,
                "fresh_test": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
