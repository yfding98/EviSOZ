#!/usr/bin/env python3
"""Run the single preregistered endpoint-aligned LaBraM PEFT v8 OOF test.

The formal path is deliberately closed to 65 source-train patients, 582
events, the existing five patient folds and 20 epochs.  ``--limit`` and a
non-20 ``--epochs`` value exist only for execution smoke tests and can never
produce a promotion decision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Callable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # The cache publisher is intentionally an independently audited module.
    from src.soz.labram_peft_prefix_cache import (  # type: ignore
        load_labram_peft_prefix_cache,
    )
except ImportError as _PREFIX_CACHE_IMPORT_ERROR:  # pragma: no cover - transient
    load_labram_peft_prefix_cache = None
else:
    _PREFIX_CACHE_IMPORT_ERROR = None

from scripts.run_labram_temporal_mil_nested_oof_v1 import (  # noqa: E402
    _file_sha256,
    _indices_for_folds,
    _metrics,
    _tensor_state_sha256,
)
from scripts.run_labram_v_directed_endpoint_oof_v5 import (  # noqa: E402
    _direction_payload,
    _paired_patient_bootstrap,
    _transition_diagnostic,
)
from src.soz.aggregation import aggregate_patient_logits  # noqa: E402
from src.soz.development_reasoner_training_v1_1 import (  # noqa: E402
    FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
)
from src.soz.frozen_h_recovery import FrozenHStandardization  # noqa: E402
from src.soz.geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS  # noqa: E402
from src.soz.labram_peft_recovery import (  # noqa: E402
    LABRAM_PEFT_EVENT_TILES,
    LABRAM_PEFT_HEAD_TRAINABLE_PARAMETERS,
    DifferentiableFullPhaseHVHead,
    seeded_differentiable_full_phase_head,
    suffix_node_tokens,
)
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
)
from src.soz.models.labram_peft import (  # noqa: E402
    LABRAM_PEFT_BLOCKS,
    LABRAM_PEFT_PREFIX_TOKENS,
    LABRAM_PEFT_TOKEN_DIM,
    LABRAM_PEFT_TRAINABLE_PARAMETERS,
    OfficialLaBraMMinimalPEFTSuffix,
)
from src.soz.safe_anchor_h_recovery import (  # noqa: E402
    within_tcp_edge_direction_metrics,
)
from src.soz.source_train_iv_capability import (  # noqa: E402
    EXPECTED_SOURCE_TRAIN_EVENT_COUNT,
    EXPECTED_SOURCE_TRAIN_PATIENT_COUNT,
    load_and_join_source_train_iv_target_scope,
)
from src.soz.temporal_mil_recovery import (  # noqa: E402
    TemporalMILPatientBatch,
    exact_positive_set_mass_loss,
    jeffreys_channel_prior_logits,
    subset_patient_batch,
)


SCHEMA_VERSION = "soz_labram_endpoint_aligned_peft_oof_v8"
PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_endpoint_aligned_peft_recovery_protocol_v8_20260811_zh.md"
)
PEFT_MODULE_PATH = ROOT / "src/soz/models/labram_peft.py"
RECOVERY_MODULE_PATH = ROOT / "src/soz/labram_peft_recovery.py"
ACCESS_AUDIT_PATH = ROOT / "outputs/labram_peft_v8_access_audit_20260811"
ACCESS_AUDIT_MANIFEST_SHA256 = (
    "48ddda4ada53ae3bc3019120beda6504859b3b2d114e0cabb88069018dfd5543"
)
ACCESS_AUDIT_JSON_SHA256 = (
    "1bfceb44e3e1f4936193197beaf06c0b43dbf508e0846c4ba3a7bc03cf80134e"
)

DEFAULT_SOURCE_TRAIN_IV = (
    ROOT / "outputs/labram_iv_source_train_only_capability_v1_20260811"
)
DEFAULT_SOURCE_TRAIN_IV_MANIFEST_SHA256 = (
    "ccd238b17e1da0aa24f2542a314c770900eeed71cbc31282a4acb76dcf957821"
)
DEFAULT_TARGET_SCOPE = (
    ROOT / "outputs/development_target_scope_v1_1_final_20260810/train"
)
DEFAULT_PREFIX_CACHE = ROOT / "outputs/labram_peft_prefix_cache_v8_20260811"
DEFAULT_PREFIX_CACHE_MANIFEST_SHA256 = (
    "82679da220ecbd3c09c01b8badc6a2d610b42bc16cf717ce73ec6ab443c97ff4"
)
DEFAULT_LABRAM_MODELING = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py"
)
DEFAULT_LABRAM_CHECKPOINT = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth"
)

V7_COMPARATOR_PATH = ROOT / "outputs/labram_native_8s_context_oof_v7_20260811"
V7_MANIFEST_SHA256 = (
    "80bd997ed61ed624ff9f7321df593b14bcdddbf63392a603833bed3d250a43b9"
)
V7_PREDICTION_SHA256 = (
    "b4fa350fb1c0922e83c5330fe8f0a63cf0eed473e91fd39ef171e7d2a9b9cad5"
)

MATCHED_FROZEN = "matched_frozen_qkv_off"
PEFT_QKV_R4 = "peft_qkv_r4"
V7_FROZEN_4S = "labram4_nonoverlap_full_phase_h_v"
TEMPORAL_ANCHOR = "temporal_mil_exact_anchor"
OUTER_FOLDS = tuple(range(5))
FORMAL_EPOCHS = 20
BASE_SEED = 20260811
HEAD_LR = 3e-3
LORA_LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0
BOOTSTRAP_REPLICATES = 2000
DEFAULT_EVENT_MICROBATCH = 4
ANCHOR_MACRO_AP_GATE = 0.6328
FORMAL_STRICT_HITS_GATE = 46
FORMAL_RELAXED_HITS_GATE = 56
FORMAL_FAR_ERROR_GATE = 10
# Top-1 metrics assign fractional credit to exact score ties.  Keep a small
# tolerance only for float32 mean-rounding (for example 46/65 is represented
# about 1.1e-6 above 46 expected hits); never round fractional tie credit to a
# whole patient when applying a promotion gate.
EXPECTED_HIT_COUNT_ATOL = 1e-5


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _scope_sha256(values: Sequence[object]) -> str:
    return hashlib.sha256(_canonical_bytes(list(values))).hexdigest()


def _load_access_audit() -> dict[str, object]:
    manifest_path = ACCESS_AUDIT_PATH / "manifest.json"
    audit_path = ACCESS_AUDIT_PATH / "access_audit.json"
    for path in (manifest_path, audit_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"v8 access-audit file is unavailable: {path}")
    if _file_sha256(manifest_path) != ACCESS_AUDIT_MANIFEST_SHA256:
        raise ValueError("v8 access-audit manifest changed")
    if _file_sha256(audit_path) != ACCESS_AUDIT_JSON_SHA256:
        raise ValueError("v8 access-audit payload changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("training_authorized_only_if_status_pass") is not True
        or manifest.get("files", {})
        .get("access_audit.json", {})
        .get("sha256")
        != ACCESS_AUDIT_JSON_SHA256
        or audit.get("status") != "PASS"
    ):
        raise ValueError("v8 access audit does not authorize training")
    assertions = audit.get("boundary_assertions", {})
    required_true = (
        "adaptation_access_boundary_pass",
        "eligible_split_edf_sha256_sets_pairwise_disjoint",
        "eligible_split_patient_rosters_pairwise_disjoint",
        "eligible_split_relative_path_sets_pairwise_disjoint",
        "planned_cache_is_exactly_582_events_65_patients",
        "planned_raw_paths_beneath_tusz_root",
        "planned_reads_exclude_source_dev_and_source_eval",
    )
    if any(assertions.get(name) is not True for name in required_true) or (
        assertions.get("foundation_pretraining_clean") is not False
    ):
        raise ValueError("v8 access-audit boundary assertions changed")
    exposure = audit.get("foundation_pretraining_exposure", {})
    if (
        exposure.get("official_pretraining_contains_tusz") is not True
        or exposure.get("pretraining_clean") is not False
    ):
        raise ValueError("v8 foundation pretraining-exposure disclosure changed")
    return {
        "path": str(ACCESS_AUDIT_PATH.relative_to(ROOT)),
        "manifest_sha256": ACCESS_AUDIT_MANIFEST_SHA256,
        "access_audit_json_sha256": ACCESS_AUDIT_JSON_SHA256,
        "audit_receipt_sha256": manifest.get("audit_receipt_sha256"),
        "status": "PASS",
        "foundation_pretraining_clean": False,
        "adaptation_access_boundary_pass": True,
    }


def _chunks(count: int, size: int) -> tuple[torch.Tensor, ...]:
    if count < 1 or size < 1:
        raise ValueError("chunk count and size must be positive")
    return tuple(
        torch.arange(start, min(start + size, count), dtype=torch.long)
        for start in range(0, count, size)
    )


def _strict_pz_mask_audit(batch: TemporalMILPatientBatch) -> None:
    pz = CHANNEL_INDEX["PZ"]
    if pz != 14:
        raise RuntimeError("standard-19 PZ index changed")
    expected = torch.ones_like(batch.target_mask)
    expected[:, pz] = False
    if not torch.equal(batch.target_mask, expected):
        raise ValueError("v8 requires the frozen 18-electrode mask with PZ only excluded")
    if torch.any(batch.targets[:, pz] != 0):
        raise ValueError("masked PZ target carrier must contain no positive value")


@dataclass(frozen=True)
class RunScope:
    patient_indices: tuple[int, ...]
    active_folds: tuple[int, ...]
    epochs: int
    smoke_only: bool


def _smoke_patient_indices(
    patient_folds: Sequence[int], limit: int
) -> tuple[int, ...]:
    """Choose a deterministic fold-covering patient subset for smoke only."""

    if limit < 2 or limit > len(patient_folds):
        raise ValueError(
            "--limit must be in [2, patient_count] so every held bag has a train bag"
        )
    selected: list[int] = []
    for fold in OUTER_FOLDS:
        match = next(
            (index for index, value in enumerate(patient_folds) if value == fold),
            None,
        )
        if match is not None and len(selected) < limit:
            selected.append(match)
    selected.extend(
        index
        for index in range(len(patient_folds))
        if index not in selected and len(selected) < limit
    )
    return tuple(sorted(selected))


def _resolve_scope(
    patient_folds: Sequence[int], *, limit: int | None, epochs: int
) -> RunScope:
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    if limit is None:
        if epochs != FORMAL_EPOCHS:
            raise ValueError("a non-20 epoch run requires --limit and is smoke-only")
        if len(patient_folds) != EXPECTED_SOURCE_TRAIN_PATIENT_COUNT:
            raise ValueError("formal v8 requires exactly 65 patients")
        if set(patient_folds) != set(OUTER_FOLDS):
            raise ValueError("formal v8 requires all five frozen folds")
        indices = tuple(range(len(patient_folds)))
        return RunScope(indices, OUTER_FOLDS, epochs, False)
    indices = _smoke_patient_indices(patient_folds, int(limit))
    folds = tuple(sorted({int(patient_folds[index]) for index in indices}))
    return RunScope(indices, folds, epochs, True)


def _to_temporal_batch(joined: object) -> tuple[TemporalMILPatientBatch, tuple[str, ...]]:
    patient = joined.batch
    batch = TemporalMILPatientBatch(
        evidence=patient.evidence,
        event_patient_index=patient.event_patient_index,
        patient_ids=patient.patient_ids,
        targets=patient.targets,
        target_mask=patient.target_mask,
    )
    _strict_pz_mask_audit(batch)
    return batch, tuple(patient.event_ids)


def _load_inputs(
    *,
    prefix_cache_path: Path,
    expected_prefix_manifest_sha256: str,
    source_train_iv_path: Path,
    expected_source_train_iv_manifest_sha256: str,
    target_scope_path: Path,
    expected_target_receipt_sha256: str,
    require_full_scope: bool,
) -> tuple[object, TemporalMILPatientBatch, tuple[int, ...], tuple[str, ...], dict[str, object]]:
    if load_labram_peft_prefix_cache is None:
        raise RuntimeError(
            "src.soz.labram_peft_prefix_cache is required; publish the audited "
            "source-train-only prefix cache module before running v8"
        ) from _PREFIX_CACHE_IMPORT_ERROR
    cache = load_labram_peft_prefix_cache(
        prefix_cache_path,
        expected_manifest_sha256=expected_prefix_manifest_sha256,
        require_full_scope=require_full_scope,
    )
    joined = load_and_join_source_train_iv_target_scope(
        source_train_iv_path,
        target_scope_path,
        expected_capability_manifest_sha256=(
            expected_source_train_iv_manifest_sha256
        ),
        expected_target_receipt_file_sha256=expected_target_receipt_sha256,
    )
    full, event_ids = _to_temporal_batch(joined)
    patient_folds = tuple(int(value) for value in joined.patient_folds)
    patient_by_event = tuple(
        full.patient_ids[int(index)] for index in full.event_patient_index.tolist()
    )
    folds_by_event = tuple(
        patient_folds[int(index)] for index in full.event_patient_index.tolist()
    )
    checks = {
        "event ids": tuple(cache.event_ids) == event_ids,
        "event patient ids": tuple(cache.patient_ids_by_event) == patient_by_event,
        "event folds": tuple(int(value) for value in cache.oof_folds)
        == folds_by_event,
        "patient roster": tuple(cache.patient_ids) == full.patient_ids,
        "token shape": tuple(cache.tokens.shape)
        == (
            full.evidence.batch_size,
            LABRAM_PEFT_EVENT_TILES,
            LABRAM_PEFT_PREFIX_TOKENS,
            LABRAM_PEFT_TOKEN_DIM,
        ),
        "token dtype": cache.tokens.dtype == torch.float32,
        "tokens detached": not cache.tokens.requires_grad,
        "source-train manifest lineage": cache.manifest.get("lineage", {}).get(
            "source_train_iv_manifest_sha256"
        )
        == expected_source_train_iv_manifest_sha256,
        "DeepSOZ targets absent": cache.manifest.get(
            "deepsoz_target_values_loaded"
        )
        is False,
        "source-train targets absent": cache.manifest.get(
            "source_train_target_values_loaded"
        )
        is False,
        "source dev absent": cache.manifest.get("source_dev_used") is False,
        "source eval absent": cache.manifest.get("source_eval_used") is False,
        "private absent": cache.manifest.get("private_used") is False,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"prefix/evidence/target strict join failed: {failed}")
    if require_full_scope:
        formal_checks = {
            "full scope": cache.full_scope is True,
            "formal input authorization": cache.manifest.get(
                "formal_training_input_authorized"
            )
            is True,
            "event count": full.evidence.batch_size
            == EXPECTED_SOURCE_TRAIN_EVENT_COUNT,
            "patient count": len(full.patient_ids)
            == EXPECTED_SOURCE_TRAIN_PATIENT_COUNT,
            "zero adapter official equivalence verified": cache.manifest.get(
                "zero_adapter_official_equivalence_verified"
            )
            is True,
            "zero adapter official equivalence at most 1e-6": float(
                cache.manifest.get(
                    "zero_adapter_official_equivalence_max_abs_error",
                    float("inf"),
                )
            )
            <= 1e-6,
        }
        formal_failed = tuple(
            name for name, passed in formal_checks.items() if not passed
        )
        if formal_failed:
            raise ValueError(f"formal v8 input boundary failed: {formal_failed}")
    lineage = {
        "prefix_cache_manifest_sha256": cache.manifest_sha256,
        "prefix_cache_event_order_sha256": cache.manifest.get(
            "event_order_sha256"
        ),
        "source_train_iv_manifest_sha256": joined.evidence_manifest_sha256,
        "source_train_iv_receipt_sha256": joined.evidence_receipt_sha256,
        "source_train_target_receipt_sha256": joined.target_receipt_file_sha256,
    }
    return cache, full, patient_folds, event_ids, lineage


def _load_fixed_comparators(
    full: TemporalMILPatientBatch,
    patient_folds: Sequence[int],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    manifest_path = V7_COMPARATOR_PATH / "manifest.json"
    prediction_path = V7_COMPARATOR_PATH / "oof_predictions.safetensors"
    if _file_sha256(manifest_path) != V7_MANIFEST_SHA256:
        raise ValueError("fixed v7 comparator manifest changed")
    if _file_sha256(prediction_path) != V7_PREDICTION_SHA256:
        raise ValueError("fixed v7 comparator predictions changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary = {
        "patient ids": tuple(manifest.get("patient_ids", ())) == full.patient_ids,
        "patient folds": tuple(manifest.get("patient_folds", ()))
        == tuple(patient_folds),
        "source dev forward": manifest.get("source_dev_forward_count") == 0,
        "source eval forward": manifest.get("source_eval_forward_count") == 0,
        "private forward": manifest.get("private_forward_count") == 0,
    }
    failed = tuple(name for name, passed in boundary.items() if not passed)
    if failed:
        raise ValueError(f"fixed v7 comparator boundary changed: {failed}")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    tensors = load_file(str(prediction_path), device="cpu")
    required = {
        V7_FROZEN_4S,
        TEMPORAL_ANCHOR,
        "targets",
        "target_mask",
        "patient_folds",
    }
    if not required <= set(tensors):
        raise ValueError("fixed v7 comparator tensor schema changed")
    if not torch.equal(tensors["targets"], full.targets.cpu()) or not torch.equal(
        tensors["target_mask"], full.target_mask.cpu()
    ):
        raise ValueError("fixed comparator target carrier changed")
    expected_folds = torch.tensor(patient_folds, dtype=torch.int64)
    if not torch.equal(tensors["patient_folds"], expected_folds):
        raise ValueError("fixed comparator fold carrier changed")
    result = {
        V7_FROZEN_4S: tensors[V7_FROZEN_4S].float().contiguous(),
        TEMPORAL_ANCHOR: tensors[TEMPORAL_ANCHOR].float().contiguous(),
    }
    anchor_metrics = _metrics(
        result[TEMPORAL_ANCHOR], full.targets, full.target_mask
    )
    strict_hits = _expected_hits(anchor_metrics, "strict_accuracy")
    relaxed_hits = _expected_hits(anchor_metrics, "relaxed_accuracy")
    if (
        abs(strict_hits - 42.0) > EXPECTED_HIT_COUNT_ATOL
        or abs(relaxed_hits - 55.0) > EXPECTED_HIT_COUNT_ATOL
    ):
        raise ValueError("fixed temporal anchor endpoint counts changed")
    return result, {
        "artifact_path": str(V7_COMPARATOR_PATH.relative_to(ROOT)),
        "manifest_sha256": V7_MANIFEST_SHA256,
        "prediction_sha256": V7_PREDICTION_SHA256,
        "status": manifest.get("status"),
    }


def _subset_scope(
    full: TemporalMILPatientBatch,
    patient_indices: Sequence[int],
) -> tuple[TemporalMILPatientBatch, torch.Tensor]:
    selected = tuple(int(value) for value in patient_indices)
    base = subset_patient_batch(
        full.evidence,
        full.event_patient_index,
        full.patient_ids,
        full.targets,
        full.target_mask,
        selected,
    )
    patient_mask = torch.zeros(len(full.patient_ids), dtype=torch.bool)
    patient_mask[torch.tensor(selected, dtype=torch.long)] = True
    global_events = torch.nonzero(
        patient_mask[full.event_patient_index], as_tuple=False
    ).flatten()
    if global_events.numel() != base.evidence.batch_size:
        raise RuntimeError("patient subset lost a complete event bag")
    return base, global_events


def _seeded_suffix(
    *,
    modeling_path: Path,
    checkpoint_path: Path,
    seed: int,
    device: torch.device,
) -> OfficialLaBraMMinimalPEFTSuffix:
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        suffix = OfficialLaBraMMinimalPEFTSuffix(
            modeling_path=modeling_path,
            checkpoint_path=checkpoint_path,
            expected_sha256=AUDITED_LABRAM_BASE_SHA256,
            expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
        )
    return suffix.to(device)


@dataclass(frozen=True)
class FrozenZeroFeatures:
    h_full: torch.Tensor
    moment_sum: torch.Tensor
    moment_square_sum: torch.Tensor
    moment_count: torch.Tensor
    qkv_original_sha256: str


def _qkv_original_state(
    suffix: OfficialLaBraMMinimalPEFTSuffix,
) -> dict[str, torch.Tensor]:
    return {
        f"block{block}.qkv_original": suffix.backbone.blocks[
            block
        ].attn.qkv.parametrizations.weight.original.detach().cpu().clone()
        for block in LABRAM_PEFT_BLOCKS
    }


def _precompute_zero_features(
    cache_tokens: torch.Tensor,
    full: TemporalMILPatientBatch,
    *,
    modeling_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    event_microbatch: int,
    included_event_indices: torch.Tensor | None = None,
) -> FrozenZeroFeatures:
    """Run the zero-LoRA suffix once and retain only sufficient H statistics."""

    suffix = _seeded_suffix(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        seed=BASE_SEED,
        device=device,
    )
    suffix.eval()
    original = _qkv_original_state(suffix)
    events = full.evidence.batch_size
    selected_events = (
        torch.arange(events, dtype=torch.long)
        if included_event_indices is None
        else included_event_indices.detach().cpu().long().contiguous()
    )
    if selected_events.ndim != 1 or selected_events.numel() < 1 or (
        torch.unique(selected_events).numel() != selected_events.numel()
    ):
        raise ValueError("zero-feature event scope must be a non-empty unique vector")
    if int(selected_events.min()) < 0 or int(selected_events.max()) >= events:
        raise ValueError("zero-feature event scope is out of range")
    h_full = torch.zeros(events, N_STANDARD_CHANNELS, LABRAM_PEFT_TOKEN_DIM)
    moment_sum = torch.zeros(events, LABRAM_PEFT_TOKEN_DIM, dtype=torch.float64)
    moment_square_sum = torch.zeros_like(moment_sum)
    moment_count = torch.zeros(events, dtype=torch.int64)
    with torch.no_grad():
        for selected_local in _chunks(len(selected_events), event_microbatch):
            local = selected_events.index_select(0, selected_local)
            prefix = cache_tokens.index_select(0, local).to(device=device)
            node = suffix_node_tokens(suffix, prefix)
            phase = full.evidence.phase_mask.index_select(0, local).to(device)
            matched = phase[:, :3].all(dim=1) & phase[:, 3:6].all(dim=1)
            valid = (phase & matched.unsqueeze(1)).unsqueeze(1).expand(
                -1, N_STANDARD_CHANNELS, -1
            )
            tile = node.mean(dim=3)
            denominator = valid.sum(dim=2).clamp_min(1).to(tile.dtype).unsqueeze(-1)
            event_h = torch.where(
                valid.unsqueeze(-1), tile, torch.zeros_like(tile)
            ).sum(dim=2) / denominator
            event_h = torch.where(
                matched[:, None, None], event_h, torch.zeros_like(event_h)
            )
            h_full.index_copy_(0, local, event_h.detach().cpu())
            for offset, event_index in enumerate(local.tolist()):
                rows = tile[offset][valid[offset]].double()
                if rows.numel():
                    moment_sum[event_index] = rows.sum(dim=0).cpu()
                    moment_square_sum[event_index] = rows.square().sum(dim=0).cpu()
                    moment_count[event_index] = rows.shape[0]
            del prefix, node, tile, event_h
    unchanged = _qkv_original_state(suffix)
    if any(not torch.equal(original[name], unchanged[name]) for name in original):
        raise RuntimeError("zero-feature pass modified an original qkv weight")
    if any(parameter.grad is not None for parameter in suffix.parameters()):
        raise RuntimeError("zero-feature pass unexpectedly created gradients")
    del suffix
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if torch.any(moment_count < 0) or not torch.isfinite(h_full).all():
        raise RuntimeError("zero-LoRA sufficient features are invalid")
    return FrozenZeroFeatures(
        h_full=h_full.contiguous(),
        moment_sum=moment_sum,
        moment_square_sum=moment_square_sum,
        moment_count=moment_count,
        qkv_original_sha256=_tensor_state_sha256(original),
    )


def _fold_standardization(
    features: FrozenZeroFeatures, global_event_indices: torch.Tensor
) -> FrozenHStandardization:
    total_count = features.moment_count.index_select(
        0, global_event_indices
    ).sum()
    if int(total_count) < 2:
        raise ValueError("fold-local H standardization has fewer than two rows")
    total = features.moment_sum.index_select(0, global_event_indices).sum(dim=0)
    total_square = features.moment_square_sum.index_select(
        0, global_event_indices
    ).sum(dim=0)
    mean64 = total / total_count.double()
    variance64 = (total_square / total_count.double() - mean64.square()).clamp_min(0)
    return FrozenHStandardization(
        mean=mean64.float(),
        scale=variance64.sqrt().float().clamp_min(1e-5),
    )


EventForward = Callable[[torch.Tensor], torch.Tensor]


def _collect_event_logits(
    event_count: int,
    forward_events: EventForward,
    *,
    event_microbatch: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for indices in _chunks(event_count, event_microbatch):
            logits = forward_events(indices)
            if tuple(logits.shape) != (len(indices), N_STANDARD_CHANNELS):
                raise RuntimeError("event forward returned the wrong logit shape")
            rows.append(logits.detach().cpu())
    result = torch.cat(rows, dim=0)
    if tuple(result.shape) != (event_count, N_STANDARD_CHANNELS) or not torch.isfinite(
        result
    ).all():
        raise RuntimeError("event-logit collection is incomplete")
    return result


def _patient_loss_and_event_upstream(
    event_logits: torch.Tensor,
    batch: TemporalMILPatientBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Form the complete patient bag before deriving exact event gradients."""

    aggregation = aggregate_patient_logits(event_logits, batch.event_patient_index)
    expected = torch.arange(len(batch.patient_ids), dtype=torch.long)
    if not torch.equal(aggregation.patient_ids.cpu(), expected):
        raise RuntimeError("complete patient aggregation carrier changed")
    expected_counts = torch.bincount(
        batch.event_patient_index, minlength=len(batch.patient_ids)
    )
    if not torch.equal(aggregation.event_counts.cpu(), expected_counts.cpu()):
        raise RuntimeError("complete patient bag was reweighted or truncated")
    leaf = aggregation.logits.detach().requires_grad_(True)
    loss = exact_positive_set_mass_loss(leaf, batch.targets, batch.target_mask)
    patient_gradient = torch.autograd.grad(loss, leaf, create_graph=False)[0]
    event_upstream = patient_gradient.index_select(
        0, batch.event_patient_index
    ) / expected_counts.index_select(0, batch.event_patient_index).to(
        patient_gradient.dtype
    ).unsqueeze(1)
    return loss.detach(), aggregation.logits.detach(), event_upstream.detach()


def _gradient_payload(
    suffix: OfficialLaBraMMinimalPEFTSuffix | None,
) -> dict[str, object]:
    if suffix is None:
        return {"lora_active": False}
    result: dict[str, object] = {"lora_active": True}
    for block in LABRAM_PEFT_BLOCKS:
        adapter = suffix._lora(block)
        for factor in ("A", "B"):
            gradient = getattr(adapter, f"lora_{factor}").grad
            key = f"block{block}_lora_{factor}"
            result[f"{key}_finite"] = bool(
                gradient is not None and torch.isfinite(gradient).all()
            )
            result[f"{key}_nonzero"] = bool(
                gradient is not None and torch.count_nonzero(gradient).item() > 0
            )
    return result


def _train_candidate(
    batch: TemporalMILPatientBatch,
    global_event_indices: torch.Tensor,
    cache_tokens: torch.Tensor,
    frozen_h_full: torch.Tensor,
    standardization: FrozenHStandardization,
    *,
    candidate: str,
    modeling_path: Path,
    checkpoint_path: Path,
    seed: int,
    epochs: int,
    device: torch.device,
    event_microbatch: int,
) -> tuple[
    DifferentiableFullPhaseHVHead,
    OfficialLaBraMMinimalPEFTSuffix | None,
    dict[str, object],
]:
    if candidate not in (MATCHED_FROZEN, PEFT_QKV_R4):
        raise ValueError("unknown v8 candidate")
    prior = jeffreys_channel_prior_logits(batch).detach().cpu()
    head = seeded_differentiable_full_phase_head(
        prior, standardization, seed=seed, device=device
    )
    suffix: OfficialLaBraMMinimalPEFTSuffix | None = None
    original_qkv: dict[str, torch.Tensor] = {}
    if candidate == PEFT_QKV_R4:
        suffix = _seeded_suffix(
            modeling_path=modeling_path,
            checkpoint_path=checkpoint_path,
            seed=seed,
            device=device,
        )
        original_qkv = _qkv_original_state(suffix)
        parameter_groups = [
            {
                "params": tuple(head.parameters()),
                "lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": tuple(
                    parameter
                    for parameter in suffix.parameters()
                    if parameter.requires_grad
                ),
                "lr": LORA_LR,
                "weight_decay": WEIGHT_DECAY,
            },
        ]
        if suffix.n_trainable_parameters != LABRAM_PEFT_TRAINABLE_PARAMETERS:
            raise RuntimeError("LaBraM suffix must expose exactly 6,400 parameters")
    else:
        parameter_groups = [
            {
                "params": tuple(head.parameters()),
                "lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
            }
        ]
    total_trainable = sum(
        parameter.numel()
        for group in parameter_groups
        for parameter in group["params"]
    )
    expected_trainable = LABRAM_PEFT_HEAD_TRAINABLE_PARAMETERS + (
        LABRAM_PEFT_TRAINABLE_PARAMETERS if suffix is not None else 0
    )
    if total_trainable != expected_trainable:
        raise RuntimeError("v8 trainable parameter contract changed")
    optimizer = torch.optim.AdamW(parameter_groups)

    local_prefix = cache_tokens
    local_h = frozen_h_full
    evidence = batch.evidence

    def forward_events(indices: torch.Tensor) -> torch.Tensor:
        moved_evidence = evidence.index_select(indices).to(device)
        if suffix is None:
            h = local_h.index_select(0, indices).to(device)
            node = h[:, :, None, None, :].expand(
                -1,
                N_STANDARD_CHANNELS,
                LABRAM_PEFT_EVENT_TILES,
                4,
                LABRAM_PEFT_TOKEN_DIM,
            )
        else:
            prefix = local_prefix.index_select(0, indices).to(device)
            node = suffix_node_tokens(suffix, prefix)
        return head(node, moved_evidence).event_logits

    first_head_state = {
        name: value.detach().cpu().clone() for name, value in head.state_dict().items()
    }
    curve: list[dict[str, object]] = []
    maximum_two_pass_replay_error = 0.0
    first_backward: dict[str, object] | None = None
    post_zero_backward: dict[str, object] | None = None
    for epoch in range(epochs):
        head.train()
        if suffix is not None:
            suffix.train()
        first_pass = _collect_event_logits(
            batch.evidence.batch_size,
            forward_events,
            event_microbatch=event_microbatch,
        )
        loss, patient_logits, event_upstream = _patient_loss_and_event_upstream(
            first_pass, batch
        )
        optimizer.zero_grad(set_to_none=True)
        second_rows: list[torch.Tensor] = []
        for indices in _chunks(batch.evidence.batch_size, event_microbatch):
            logits = forward_events(indices)
            second_rows.append(logits.detach().cpu())
            logits.backward(event_upstream.index_select(0, indices).to(device))
        second_pass = torch.cat(second_rows, dim=0)
        replay_error = float((second_pass - first_pass).abs().max())
        maximum_two_pass_replay_error = max(
            maximum_two_pass_replay_error, replay_error
        )
        if replay_error > 1e-6:
            raise RuntimeError("two-pass patient objective replay became stochastic")
        gradient = _gradient_payload(suffix)
        if epoch == 0:
            first_backward = gradient
        elif epoch == 1:
            post_zero_backward = gradient
        parameters = [
            parameter for group in parameter_groups for parameter in group["params"]
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("v8 gradient norm is non-finite")
        if suffix is not None:
            frozen_with_grad = tuple(
                name
                for name, parameter in suffix.named_parameters()
                if not parameter.requires_grad and parameter.grad is not None
            )
            if frozen_with_grad:
                raise RuntimeError("a frozen LaBraM parameter received a gradient")
        optimizer.step()
        curve.append(
            {
                "epoch": epoch + 1,
                "exact_positive_set_mass": float(loss),
                "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                "patient_logit_abs_mean": float(patient_logits.abs().mean()),
                "two_pass_replay_max_abs_error": replay_error,
            }
        )
    optimizer.zero_grad(set_to_none=True)

    implementation_checks: dict[str, bool] = {
        "head_parameter_count_314": head.n_trainable_parameters
        == LABRAM_PEFT_HEAD_TRAINABLE_PARAMETERS,
        "total_trainable_parameter_count": total_trainable == expected_trainable,
        "two_pass_replay_le_1e_6": maximum_two_pass_replay_error <= 1e-6,
        "pure_set_loss_only": True,
        "complete_patient_bag_before_loss": True,
        "no_amp": True,
    }
    qkv_post_sha256: str | None = None
    if suffix is not None:
        current_qkv = _qkv_original_state(suffix)
        qkv_post_sha256 = _tensor_state_sha256(current_qkv)
        implementation_checks.update(
            {
                "backbone_trainable_6400": suffix.n_trainable_parameters
                == LABRAM_PEFT_TRAINABLE_PARAMETERS,
                "total_trainable_6714": total_trainable == 6714,
                "original_qkv_unchanged": all(
                    torch.equal(original_qkv[name], current_qkv[name])
                    for name in original_qkv
                ),
                "first_backward_B_finite_nonzero": bool(
                    first_backward
                    and all(
                        first_backward[f"block{block}_lora_B_finite"]
                        and first_backward[f"block{block}_lora_B_nonzero"]
                        for block in LABRAM_PEFT_BLOCKS
                    )
                ),
                "post_zero_backward_A_B_finite_nonzero": bool(
                    epochs < 2
                    or (
                        post_zero_backward
                        and all(
                            post_zero_backward[
                                f"block{block}_lora_{factor}_finite"
                            ]
                            and post_zero_backward[
                                f"block{block}_lora_{factor}_nonzero"
                            ]
                            for block in LABRAM_PEFT_BLOCKS
                            for factor in ("A", "B")
                        )
                    )
                ),
            }
        )
        adapter_before = suffix.lora_state_dict()
        adapter_before_sha256 = _tensor_state_sha256(adapter_before)
        suffix.load_lora_state_dict(adapter_before)
        adapter_after_sha256 = _tensor_state_sha256(suffix.lora_state_dict())
        implementation_checks["adapter_only_state_self_replay_exact"] = (
            adapter_before_sha256 == adapter_after_sha256
        )
    if not all(implementation_checks.values()):
        failed = tuple(
            name for name, passed in implementation_checks.items() if not passed
        )
        raise RuntimeError(f"v8 implementation validity gate failed: {failed}")
    fit = {
        "candidate": candidate,
        "seed": seed,
        "epochs": epochs,
        "train_patient_count": len(batch.patient_ids),
        "train_event_count": batch.evidence.batch_size,
        "train_patient_roster_sha256": _scope_sha256(batch.patient_ids),
        "trainable_parameter_count": total_trainable,
        "head_learning_rate": HEAD_LR,
        "lora_learning_rate": LORA_LR if suffix is not None else None,
        "weight_decay": WEIGHT_DECAY,
        "max_gradient_norm": MAX_GRAD_NORM,
        "objective": "exact_positive_set_mass_after_equal_event_logit_mean",
        "curve": curve,
        "first_backward": first_backward,
        "post_zero_backward": post_zero_backward,
        "implementation_checks": implementation_checks,
        "initial_head_state_sha256": _tensor_state_sha256(first_head_state),
        "final_head_state_sha256": _tensor_state_sha256(
            {
                name: value.detach().cpu() for name, value in head.state_dict().items()
            }
        ),
        "original_qkv_pre_sha256": (
            _tensor_state_sha256(original_qkv) if original_qkv else None
        ),
        "original_qkv_post_sha256": qkv_post_sha256,
    }
    return head, suffix, fit


def _predict_candidate(
    batch: TemporalMILPatientBatch,
    cache_tokens: torch.Tensor,
    frozen_h_full: torch.Tensor,
    head: DifferentiableFullPhaseHVHead,
    suffix: OfficialLaBraMMinimalPEFTSuffix | None,
    *,
    device: torch.device,
    event_microbatch: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    head.eval()
    if suffix is not None:
        suffix.eval()

    def forward_events(indices: torch.Tensor) -> torch.Tensor:
        moved_evidence = batch.evidence.index_select(indices).to(device)
        if suffix is None:
            h = frozen_h_full.index_select(0, indices).to(device)
            node = h[:, :, None, None, :].expand(
                -1,
                N_STANDARD_CHANNELS,
                LABRAM_PEFT_EVENT_TILES,
                4,
                LABRAM_PEFT_TOKEN_DIM,
            )
        else:
            prefix = cache_tokens.index_select(0, indices).to(device)
            node = suffix_node_tokens(suffix, prefix)
        return head(node, moved_evidence).event_logits

    event_logits = _collect_event_logits(
        batch.evidence.batch_size,
        forward_events,
        event_microbatch=event_microbatch,
    )
    aggregation = aggregate_patient_logits(event_logits, batch.event_patient_index)
    expected_counts = torch.bincount(
        batch.event_patient_index, minlength=len(batch.patient_ids)
    )
    if not torch.equal(aggregation.event_counts, expected_counts):
        raise RuntimeError("prediction did not preserve complete patient bags")
    return aggregation.logits.contiguous(), {
        "event_count": batch.evidence.batch_size,
        "patient_count": len(batch.patient_ids),
        "event_count_min": int(expected_counts.min()),
        "event_count_max": int(expected_counts.max()),
        "event_pooling": "arithmetic_mean_of_all_event_logits",
    }


def _checkpoint_state(
    *,
    fold: int,
    candidate: str,
    head: DifferentiableFullPhaseHVHead,
    suffix: OfficialLaBraMMinimalPEFTSuffix | None,
) -> dict[str, torch.Tensor]:
    state = {
        f"fold{fold}.{candidate}.head.{name}": value.detach().cpu().contiguous()
        for name, value in head.state_dict().items()
    }
    if suffix is not None:
        state.update(
            {
                f"fold{fold}.{candidate}.adapter.{name}": value.contiguous()
                for name, value in suffix.lora_state_dict().items()
            }
        )
    forbidden = ("parametrizations.weight.original", "checkpoint", "backbone")
    if any(any(token in name for token in forbidden) for name in state):
        raise RuntimeError("adapter-only checkpoint contains a foundation weight")
    return state


def _expected_hits(metrics: Mapping[str, object], name: str) -> float:
    """Return tie-aware expected hits without integer rounding.

    ``deepsoz_style_top1_metrics`` evaluates an exact top-score tie by uniform
    random tie breaking.  Its aggregate accuracy can therefore correspond to
    a fractional expected patient count.  Rounding here would allow, e.g.,
    45.5 expected strict hits to satisfy the frozen 46/65 gate.
    """

    top1 = metrics["top1"]
    return float(top1[name]) * int(top1["n_samples"])


def _expected_hits_at_least(
    metrics: Mapping[str, object], name: str, threshold: int
) -> bool:
    return (
        _expected_hits(metrics, name) + EXPECTED_HIT_COUNT_ATOL
        >= float(threshold)
    )


def _run_oof(
    cache: object,
    full: TemporalMILPatientBatch,
    patient_folds: tuple[int, ...],
    comparators: Mapping[str, torch.Tensor],
    zero: FrozenZeroFeatures,
    scope: RunScope,
    *,
    modeling_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    event_microbatch: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    scoped, scoped_global_events = _subset_scope(full, scope.patient_indices)
    scoped_folds = tuple(patient_folds[index] for index in scope.patient_indices)
    scoped_comparators = {
        name: value.index_select(0, torch.tensor(scope.patient_indices))
        for name, value in comparators.items()
    }
    scoped_prefix = cache.tokens.index_select(0, scoped_global_events)
    scoped_h = zero.h_full.index_select(0, scoped_global_events)
    patient_count = len(scoped.patient_ids)
    predictions = {
        MATCHED_FROZEN: torch.full((patient_count, N_STANDARD_CHANNELS), torch.nan),
        PEFT_QKV_R4: torch.full((patient_count, N_STANDARD_CHANNELS), torch.nan),
    }
    checkpoint: dict[str, torch.Tensor] = {}
    fold_rows: list[dict[str, object]] = []
    fold_nonlower_count = 0
    fold_net_losses: list[float] = []

    for fold in scope.active_folds:
        held_indices = _indices_for_folds(scoped_folds, (fold,))
        train_indices = tuple(
            index for index in range(patient_count) if index not in held_indices
        )
        if not held_indices or not train_indices:
            continue
        train, train_events = _subset_scope(scoped, train_indices)
        held, held_events = _subset_scope(scoped, held_indices)
        train_global = scoped_global_events.index_select(0, train_events)
        stats = _fold_standardization(zero, train_global)
        train_prefix = scoped_prefix.index_select(0, train_events)
        held_prefix = scoped_prefix.index_select(0, held_events)
        train_h = scoped_h.index_select(0, train_events)
        held_h = scoped_h.index_select(0, held_events)
        seed = BASE_SEED + fold * 1000
        candidate_rows: dict[str, object] = {}
        initial_head_hashes: dict[str, str] = {}
        for candidate in (MATCHED_FROZEN, PEFT_QKV_R4):
            head, suffix, fit = _train_candidate(
                train,
                train_global,
                train_prefix,
                train_h,
                stats,
                candidate=candidate,
                modeling_path=modeling_path,
                checkpoint_path=checkpoint_path,
                seed=seed,
                epochs=scope.epochs,
                device=device,
                event_microbatch=event_microbatch,
            )
            initial_head_hashes[candidate] = str(
                fit["initial_head_state_sha256"]
            )
            held_logits, diagnostics = _predict_candidate(
                held,
                held_prefix,
                held_h,
                head,
                suffix,
                device=device,
                event_microbatch=event_microbatch,
            )
            predictions[candidate][list(held_indices)] = held_logits
            metrics = _metrics(held_logits, held.targets, held.target_mask)
            direction = _direction_payload(
                within_tcp_edge_direction_metrics(
                    held_logits, held.targets, held.target_mask
                )
            )
            candidate_rows[candidate] = {
                "fit": fit,
                "held_metrics": metrics,
                "held_within_tcp_direction": direction,
                "prediction_diagnostics": diagnostics,
            }
            checkpoint.update(
                _checkpoint_state(
                    fold=fold,
                    candidate=candidate,
                    head=head,
                    suffix=suffix,
                )
            )
            del head, suffix
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if len(set(initial_head_hashes.values())) != 1:
            raise RuntimeError("matched candidates did not share head initialization")
        held_tensor = torch.tensor(held_indices, dtype=torch.long)
        anchor_held = scoped_comparators[TEMPORAL_ANCHOR].index_select(
            0, held_tensor
        )
        anchor_metrics = _metrics(anchor_held, held.targets, held.target_mask)
        peft_hits = _expected_hits(
            candidate_rows[PEFT_QKV_R4]["held_metrics"], "strict_accuracy"
        )
        anchor_hits = _expected_hits(anchor_metrics, "strict_accuracy")
        net = peft_hits - anchor_hits
        fold_net_losses.append(net)
        fold_nonlower_count += int(net >= -EXPECTED_HIT_COUNT_ATOL)
        fold_rows.append(
            {
                "outer_fold": fold,
                "seed": seed,
                "train_patient_count": len(train.patient_ids),
                "train_event_count": train.evidence.batch_size,
                "held_patient_count": len(held.patient_ids),
                "held_event_count": held.evidence.batch_size,
                "candidate_rows": candidate_rows,
                "shared_initial_head_state_sha256": next(
                    iter(initial_head_hashes.values())
                ),
                "anchor_metrics": anchor_metrics,
                "peft_minus_anchor_strict_hits": net,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "outer_complete",
                    "fold": fold,
                    "strict_peft_expected_hits": peft_hits,
                    "strict_anchor_expected_hits": anchor_hits,
                    "smoke_only": scope.smoke_only,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if any(not torch.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("v8 OOF left a scoped patient prediction unfilled")

    scores = {**predictions, **scoped_comparators}
    metrics = {
        name: _metrics(value, scoped.targets, scoped.target_mask)
        for name, value in scores.items()
    }
    directions = {
        name: _direction_payload(
            within_tcp_edge_direction_metrics(
                value, scoped.targets, scoped.target_mask
            )
        )
        for name, value in scores.items()
    }
    transitions = {
        f"{PEFT_QKV_R4}_vs_{reference}": _transition_diagnostic(
            scores[PEFT_QKV_R4],
            scores[reference],
            scoped.targets,
            scoped.target_mask,
        )
        for reference in (MATCHED_FROZEN, V7_FROZEN_4S, TEMPORAL_ANCHOR)
    }
    bootstrap = {
        f"{PEFT_QKV_R4}_vs_{reference}": _paired_patient_bootstrap(
            scores[PEFT_QKV_R4],
            scores[reference],
            scoped.targets,
            scoped.target_mask,
        )
        for reference in (MATCHED_FROZEN, V7_FROZEN_4S, TEMPORAL_ANCHOR)
    }
    peft_metrics = metrics[PEFT_QKV_R4]
    matched_metrics = metrics[MATCHED_FROZEN]
    anchor_transition = transitions[f"{PEFT_QKV_R4}_vs_{TEMPORAL_ANCHOR}"]
    if scope.smoke_only:
        gate_checks: dict[str, bool] = {}
        decision = "smoke_only_no_scientific_gate"
        gate_go = False
    else:
        reporting_complete = (
            len(fold_rows) == 5
            and len(fold_net_losses) == 5
            and all(len(value) == 6 for value in bootstrap.values())
        )
        gate_checks = {
            "strict_top1_at_least_46_of_65": _expected_hits_at_least(
                peft_metrics, "strict_accuracy", FORMAL_STRICT_HITS_GATE
            ),
            "relaxed_top1_at_least_56_of_65": _expected_hits_at_least(
                peft_metrics, "relaxed_accuracy", FORMAL_RELAXED_HITS_GATE
            ),
            "macro_ap_at_least_0_6328": float(
                peft_metrics["ranking"]["macro_average_precision"]
            )
            >= ANCHOR_MACRO_AP_GATE,
            "strict_strictly_above_matched_frozen": float(
                peft_metrics["top1"]["strict_accuracy"]
            )
            > float(matched_metrics["top1"]["strict_accuracy"]),
            "macro_ap_strictly_above_matched_frozen": float(
                peft_metrics["ranking"]["macro_average_precision"]
            )
            > float(matched_metrics["ranking"]["macro_average_precision"]),
            "far_errors_at_most_10": int(anchor_transition["candidate_far_count"])
            <= FORMAL_FAR_ERROR_GATE,
            "strict_nonlower_in_at_least_4_of_5_folds": fold_nonlower_count
            >= 4,
            "no_fold_loses_more_than_one_strict_patient": min(fold_net_losses)
            >= -1.0 - EXPECTED_HIT_COUNT_ATOL,
            "all_folds_bootstrap_and_failures_reported": reporting_complete,
        }
        gate_go = all(gate_checks.values())
        decision = (
            "go_freeze_then_allow_one_source_eval_exploratory_evaluation"
            if gate_go
            else "no_go_keep_temporal_mil_exact"
        )
    result = {
        "screen_kind": (
            "post_hoc_source_train_patient_oof_endpoint_aligned_peft"
        ),
        "smoke_only": scope.smoke_only,
        "metrics": metrics,
        "within_tcp_direction": directions,
        "paired_patient_bootstrap": bootstrap,
        "far_error_and_top1_transitions": transitions,
        "outer_folds": fold_rows,
        "fold_strict_nonlower_count_vs_anchor": fold_nonlower_count,
        "fold_peft_minus_anchor_strict_hits": fold_net_losses,
        "promotion_gate_checks": gate_checks,
        "promotion_go": gate_go,
        "decision": decision,
    }
    tensors = {
        **{name: value.contiguous() for name, value in scores.items()},
        "targets": scoped.targets.cpu().contiguous(),
        "target_mask": scoped.target_mask.cpu().contiguous(),
        "patient_folds": torch.tensor(scoped_folds, dtype=torch.int64),
        "source_patient_indices": torch.tensor(
            scope.patient_indices, dtype=torch.int64
        ),
    }
    return result, tensors, checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--prefix-cache", type=Path, default=DEFAULT_PREFIX_CACHE)
    parser.add_argument(
        "--expected-prefix-cache-manifest-sha256",
        default=DEFAULT_PREFIX_CACHE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--source-train-iv-capability",
        type=Path,
        default=DEFAULT_SOURCE_TRAIN_IV,
    )
    parser.add_argument(
        "--expected-source-train-iv-manifest-sha256",
        default=DEFAULT_SOURCE_TRAIN_IV_MANIFEST_SHA256,
    )
    parser.add_argument("--target-scope", type=Path, default=DEFAULT_TARGET_SCOPE)
    parser.add_argument(
        "--expected-target-receipt-sha256",
        default=FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
    )
    parser.add_argument(
        "--labram-modeling-path", type=Path, default=DEFAULT_LABRAM_MODELING
    )
    parser.add_argument(
        "--labram-checkpoint-path", type=Path, default=DEFAULT_LABRAM_CHECKPOINT
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--event-microbatch", type=int, default=DEFAULT_EVENT_MICROBATCH)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument(
        "--limit",
        type=int,
        help="smoke-only patient limit; all events of selected patients are retained",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.event_microbatch < 1:
        raise ValueError("--event-microbatch must be positive")
    if not args.preflight_only and args.output_directory is None:
        raise ValueError("training requires --output-directory")
    pinned_cli = {
        "prefix cache manifest": (
            args.expected_prefix_cache_manifest_sha256.strip().lower(),
            DEFAULT_PREFIX_CACHE_MANIFEST_SHA256,
        ),
        "source-train I/V manifest": (
            args.expected_source_train_iv_manifest_sha256.strip().lower(),
            DEFAULT_SOURCE_TRAIN_IV_MANIFEST_SHA256,
        ),
        "source-train target receipt": (
            args.expected_target_receipt_sha256.strip().lower(),
            FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
        ),
    }
    changed_trust_anchors = tuple(
        name for name, (actual, expected) in pinned_cli.items() if actual != expected
    )
    if changed_trust_anchors:
        raise ValueError(
            f"v8 CLI cannot override frozen trust anchors: {changed_trust_anchors}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    access_audit_receipt = _load_access_audit()
    cache, full, patient_folds, event_ids, lineage = _load_inputs(
        prefix_cache_path=args.prefix_cache,
        expected_prefix_manifest_sha256=(
            args.expected_prefix_cache_manifest_sha256.strip().lower()
        ),
        source_train_iv_path=args.source_train_iv_capability,
        expected_source_train_iv_manifest_sha256=(
            args.expected_source_train_iv_manifest_sha256.strip().lower()
        ),
        target_scope_path=args.target_scope,
        expected_target_receipt_sha256=(
            args.expected_target_receipt_sha256.strip().lower()
        ),
        # Smoke limits patients after the strict full-cache join.  A prefix of
        # events could truncate patient bags and is therefore never accepted.
        require_full_scope=True,
    )
    scope = _resolve_scope(patient_folds, limit=args.limit, epochs=args.epochs)
    comparators, comparator_receipt = _load_fixed_comparators(full, patient_folds)
    preflight = {
        "status": "ready_endpoint_aligned_peft_oof_v8",
        "schema_version": SCHEMA_VERSION,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "peft_module_sha256": _file_sha256(PEFT_MODULE_PATH),
        "recovery_module_sha256": _file_sha256(RECOVERY_MODULE_PATH),
        "device": str(device),
        "formal_run": not scope.smoke_only,
        "smoke_only": scope.smoke_only,
        "patient_count": len(scope.patient_indices),
        "event_count": (
            full.evidence.batch_size
            if not scope.smoke_only
            else int(
                torch.isin(
                    full.event_patient_index,
                    torch.tensor(scope.patient_indices),
                ).sum()
            )
        ),
        "full_source_patient_count": len(full.patient_ids),
        "full_source_event_count": full.evidence.batch_size,
        "patient_ids": [full.patient_ids[index] for index in scope.patient_indices],
        "patient_folds": [patient_folds[index] for index in scope.patient_indices],
        "event_order_sha256": _scope_sha256(event_ids),
        "lineage": {**lineage, "access_audit": access_audit_receipt},
        "comparator_receipt": comparator_receipt,
        "config": {
            "candidates": [MATCHED_FROZEN, PEFT_QKV_R4],
            "outer_folds": list(scope.active_folds),
            "epochs": scope.epochs,
            "head_learning_rate": HEAD_LR,
            "lora_learning_rate": LORA_LR,
            "weight_decay": WEIGHT_DECAY,
            "max_gradient_norm": MAX_GRAD_NORM,
            "base_seed": BASE_SEED,
            "event_microbatch": args.event_microbatch,
            "patient_event_pooling": "arithmetic_mean_complete_event_bag",
            "objective": "pure_exact_positive_set_mass",
            "amp": False,
            "early_stopping": False,
            "candidate_scan": False,
            "pz_masked": True,
            "foundation_backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "adapted_blocks": list(LABRAM_PEFT_BLOCKS),
            "adapted_weight": "combined_attention_qkv_only",
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_dropout": 0,
            "lora_trainable_parameters": LABRAM_PEFT_TRAINABLE_PARAMETERS,
            "head_trainable_parameters": LABRAM_PEFT_HEAD_TRAINABLE_PARAMETERS,
            "total_peft_candidate_trainable_parameters": 6714,
        },
        "access_counters": {
            "source_train_event_count": full.evidence.batch_size,
            "source_dev_waveform_evidence_target_forward_count": 0,
            "source_eval_waveform_evidence_target_forward_count": 0,
            "private_waveform_evidence_target_forward_count": 0,
        },
        "scientific_boundary": {
            "development_only": True,
            "source_train_previously_observed": True,
            "not_independent_confirmation": True,
            "source_eval_used": False,
            "private_used": False,
            "deepsoz_target": "clinician_derived_scalp_electrode_soz_overlay",
            "not_seeg_confirmed_cortical_soz": True,
        },
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0

    output = Path(os.path.abspath(args.output_directory))
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise FileExistsError("v8 output exists or is invalid")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FileNotFoundError(output.parent)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    execution_started = time.perf_counter()
    zero = _precompute_zero_features(
        cache.tokens,
        full,
        modeling_path=args.labram_modeling_path,
        checkpoint_path=args.labram_checkpoint_path,
        device=device,
        event_microbatch=args.event_microbatch,
        included_event_indices=_subset_scope(full, scope.patient_indices)[1],
    )
    result, tensors, checkpoint = _run_oof(
        cache,
        full,
        patient_folds,
        comparators,
        zero,
        scope,
        modeling_path=args.labram_modeling_path,
        checkpoint_path=args.labram_checkpoint_path,
        device=device,
        event_microbatch=args.event_microbatch,
    )
    execution_audit = {
        "zero_feature_plus_oof_wall_time_sec": time.perf_counter()
        - execution_started,
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        ),
        "device": str(device),
    }
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        prediction_path = staging / "oof_predictions.safetensors"
        checkpoint_path = staging / "adapter_only_checkpoints.safetensors"
        save_file(tensors, str(prediction_path))
        save_file(checkpoint, str(checkpoint_path))
        from safetensors.torch import load_file

        checkpoint_replay = load_file(str(checkpoint_path), device="cpu")
        if _tensor_state_sha256(checkpoint_replay) != _tensor_state_sha256(
            checkpoint
        ):
            raise RuntimeError("serialized adapter-only checkpoint did not replay")
        manifest = {
            **preflight,
            "status": (
                "completed_smoke_only"
                if scope.smoke_only
                else "completed_development_only"
            ),
            "zero_suffix_audit": {
                "qkv_original_sha256": zero.qkv_original_sha256,
                "feature_shape": list(zero.h_full.shape),
                "fold_moments_target_free": True,
            },
            "execution_audit": execution_audit,
            "result": result,
            "files": {
                prediction_path.name: {
                    "sha256": _file_sha256(prediction_path),
                    "size_bytes": prediction_path.stat().st_size,
                },
                checkpoint_path.name: {
                    "sha256": _file_sha256(checkpoint_path),
                    "size_bytes": checkpoint_path.stat().st_size,
                    "state_sha256": _tensor_state_sha256(checkpoint),
                    "foundation_weights_serialized": False,
                },
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_canonical_bytes(manifest))
        manifest_sha256 = _file_sha256(manifest_path)
        os.rename(staging, output)
        published = True
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "path": str(output),
                    "manifest_sha256": manifest_sha256,
                    "decision": result["decision"],
                    "promotion_go": result["promotion_go"],
                    "metrics": result["metrics"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
