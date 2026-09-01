#!/usr/bin/env python3
"""Run the single locked static DAPT-v2 versus zero final-suffix OOF trial."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import safetensors
from safetensors.torch import load_file, save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_fine_temporal_nested_oof_v11 import (  # noqa: E402
    DEFAULT_FINE,
    DEFAULT_PREFIX,
    DEFAULT_SOURCE,
    DEFAULT_SPLIT,
    DEFAULT_TARGET,
    DEFAULT_UNION,
    OUTER_FOLDS,
    _canonical_bytes,
    _file_sha,
    _state_sha,
    _transform_state,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _patient_contributions,
)
from scripts.run_labram_fine_temporal_peft_oof_v11_b import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_MODELING,
    EVENT_MICROBATCH,
    FORMAL_EPOCHS,
    PredictionBatch,
    TrainBatch,
    V11BInputs,
    _collect_suffix_h,
    _fit_fold_initialization,
    _load_inputs as _load_v11b_inputs,
    _masked_oof_scores_for_publish,
    _prediction_batch,
    _qkv_original_state,
    _subset_events,
    _train_batch,
    _train_matched,
)
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
)
from src.soz.models.labram_peft import (  # noqa: E402
    LABRAM_PEFT_BLOCKS,
    LABRAM_PEFT_TRAINABLE_PARAMETERS,
)
from src.soz.models.labram_static_suffix import (  # noqa: E402
    OfficialLaBraMStaticAdapterSuffix,
)
from src.soz.v11_reasoner import (  # noqa: E402
    FoldFeatureTransform,
    SharedPositiveSetReasoner,
    V11_CANDIDATE_MASK,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/labram_dapt_v2_locked_downstream_protocol_20260812_zh.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "5e8bbc28ab389b576b7e036fa060efed10b16ce66b6b937779bb5c66d049cdca"
)
DEFAULT_DAPT_RUN = ROOT / "outputs/labram_tuep_dapt_v2_20260811"
DEFAULT_RECEIPT = DEFAULT_DAPT_RUN / "run_receipt.json"
DEFAULT_ADAPTER = DEFAULT_DAPT_RUN / "selected_lora.pt"
DEFAULT_QUALIFICATION = (
    ROOT
    / "outputs/labram_tuep_dapt_v2_qualification_v1_20260812/qualification.json"
)
DEFAULT_V11B_R3 = ROOT / "outputs/labram_fine_temporal_peft_oof_v11_b_20260811_r3"
DEFAULT_V11_1 = ROOT / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811_r2"
DEFAULT_OUTPUT = ROOT / "outputs/labram_dapt_v2_locked_downstream_oof_20260812"

EXPECTED_RECEIPT_SHA256 = (
    "605332172062f547e3e09780fb732a52d8c73019bc9400d8bb2958e752b70d33"
)
EXPECTED_ADAPTER_SHA256 = (
    "62aa1a8f3673c22bfbb3ffe1f4c2fb8402c3c85dd67bb2b553d9f3e7a18ee9d8"
)
EXPECTED_QUALIFICATION_SHA256 = (
    "3222d7d53efb5e0a9f2b4f6037b7d841b535848492eb7ea4384d99e43556fc59"
)
EXPECTED_V11B_R3_MANIFEST_SHA256 = (
    "1efc1564d7355596063b72a83af851a260d54d0377687e7efe458799e1f8a685"
)
EXPECTED_V11B_R3_OOF_SHA256 = (
    "ee46592f81bc91814ed491fe607031572a617f484fa8675e00d7e66e7f4d9743"
)
EXPECTED_V11_1_MANIFEST_SHA256 = (
    "f399678e5756ae30cbe5f9f87d9d8bb5b220b16015e1b2a0417110f20e70195c"
)
EXPECTED_V11_1_OOF_SHA256 = (
    "6443680b18b53b0c552b9634e7c9e2547284c9d08cccd5cd99c35b9e1a27ac08"
)
EXPECTED_UNION_MANIFEST_SHA256 = (
    "89a9ca456c724c2dee4d14a2c0da5a1190e58f97ad602060f6dda5f619b97232"
)
EXPECTED_TARGET_RECEIPT_SHA256 = (
    "80f2b71cfdf23d604849b2d1a52cc36f0b01c593906e3cef74e79d425cc442d3"
)
EXPECTED_TARGET_ARTIFACT_SHA256 = (
    "5c01591c20328fb60817099cac669032bd743e36f47df77ac390842e9a2c67ed"
)
EXPECTED_SOURCE_CSV_SHA256 = (
    "4d08552dbb94f1e8e8a3931249d2bd29538233e2282b8d21a39d0f5dd873fd5c"
)
EXPECTED_SPLIT_CSV_SHA256 = (
    "5062e894ec139ffaf7abc1b8f45b326f50a118cfcb8907bb25ff81dbbaa91d57"
)
EXPECTED_PREFIX_MANIFEST_SHA256 = (
    "b3ce8913a33848b7a706f8b30ccedf09ad8b2f6ae27412b1ae56d187866ff71f"
)
EXPECTED_PREFIX_TENSOR_SHA256 = (
    "40396fabac11ead6ac870ee69f428951f0577445c291a45b58e37c8fc6bf12bc"
)
EXPECTED_FINE_MANIFEST_SHA256 = (
    "60ce6c5af15dcff3a0c0dcbac1451f4d5cb3bb28e7b9c22180ab7adecfb417a2"
)
EXPECTED_FINE_TENSOR_SHA256 = (
    "24dc5da224c79446992cde08d800877ff1ea4349d217c225da95588c9e173bbb"
)
EXPECTED_PINNED_SOURCE_SHA256 = {
    "deepsoz_target_v2": "36bd343baa7cc43c60cd793244a5e30a991b21511f9a958ed83b7124a7a85397",
    "deepsoz_alias": "ea5955af6990b614957cc32ede067db1fa2402786e2c62f21e0cebda57bf0fd7",
    "geometry": "73ef8a89d75ac257a6767da8f4fe9dfe3cde9dee901774e2f9aa7ba8f806cf85",
    "metrics": "860a7a767ea897187245f3ed8da62dd03906f44947fb1dc5b4b961282a9db856",
    "development_union": "58a1c002197de91a9fffc708f3e8d9792919747c05ab5c1c227bad673a577cd3",
    "v11_reasoner": "75a3b16d5c216b9b749b2732dbe8be84f0421056d3803bcdda1659a52ed7637f",
    "v11_1_evaluator": "c3b4703e874d72ebd7742bfdf61f863a726c01fd7b4126fb3d451ea9acd62f42",
    "v11b_input_runner": "07eadd89727550d84d4ece89b7905186ff26f7d071867c00c5c22df286bddd2e",
}

SCHEMA = "soz_labram_dapt_v2_locked_downstream_oof_v1"
ZERO = "exact_zero_lora_final_suffix"
DAPT = "qualified_static_dapt_v2_final_suffix"
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_REPLICATES = 2_000
NONINFERIORITY_MARGIN = 0.05
ZERO_PARITY_TOLERANCE = 1e-6
BASE_SEED = 20260812


@dataclass(frozen=True)
class AuthorizedDAPT:
    adapter_state: Mapping[str, torch.Tensor]
    receipt: Mapping[str, object]
    qualification: Mapping[str, object]
    adapter_state_sha256: str


def _load_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return value


def _adapter_state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return _state_sha({name: value.detach().cpu() for name, value in state.items()})


def _validate_adapter_state(state: object) -> Mapping[str, torch.Tensor]:
    expected = {
        f"blocks.{block}.attn.qkv.lora_{factor}"
        for block in LABRAM_PEFT_BLOCKS
        for factor in ("A", "B")
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("Qualified DAPT adapter keys changed")
    result: dict[str, torch.Tensor] = {}
    for key in sorted(expected):
        value = state[key]
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"Qualified DAPT adapter tensor invalid: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Qualified DAPT adapter tensor non-finite: {key}")
        expected_shape = (4, 200) if key.endswith("lora_A") else (600, 4)
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"Qualified DAPT adapter tensor shape changed: {key}")
        result[key] = value.detach().cpu().float().contiguous()
    if sum(value.numel() for value in result.values()) != LABRAM_PEFT_TRAINABLE_PARAMETERS:
        raise ValueError("Qualified DAPT adapter must contain exactly 6,400 values")
    if any(
        torch.count_nonzero(value).item() == 0
        for key, value in result.items()
        if key.endswith("lora_B")
    ):
        raise ValueError("Qualified DAPT adapter must have non-zero LoRA-B factors")
    return result


def _authorize_dapt_candidate(args: argparse.Namespace) -> AuthorizedDAPT:
    """Validate every target-free qualification artifact before DeepSOZ loads."""

    pinned = (
        (PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256, "protocol"),
        (args.receipt_path, EXPECTED_RECEIPT_SHA256, "formal receipt"),
        (args.adapter_path, EXPECTED_ADAPTER_SHA256, "selected adapter"),
        (args.qualification_path, EXPECTED_QUALIFICATION_SHA256, "qualification"),
    )
    for path, expected, label in pinned:
        if _file_sha(path) != expected:
            raise ValueError(f"Locked DAPT-v2 {label} SHA256 changed")
    receipt = _load_json(args.receipt_path)
    exact_receipt = {
        "training_completed": True,
        "selection_fallback_to_zero_lora": False,
        "private_data_loaded": False,
        "target_values_loaded": False,
        "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
        "selected_adapter_sha256": EXPECTED_ADAPTER_SHA256,
    }
    for key, expected in exact_receipt.items():
        if receipt.get(key) != expected or type(receipt.get(key)) is not type(expected):
            raise ValueError(f"Locked DAPT-v2 receipt field changed: {key}")
    if type(receipt.get("best_epoch_by_frozen_eligibility_then_ce")) is not int or (
        receipt["best_epoch_by_frozen_eligibility_then_ce"] != 0
    ):
        raise ValueError("Locked DAPT-v2 receipt must select epoch 0")
    if Path(str(receipt.get("selected_adapter_path", ""))).resolve() != (
        args.adapter_path.resolve()
    ):
        raise ValueError("Locked DAPT-v2 receipt points to a different adapter")

    qualification = _load_json(args.qualification_path)
    exact_qualification = {
        "representation_qualified": True,
        "eligible_for_locked_downstream_comparison": True,
        "all_representation_gates_pass": True,
        "external_validation": False,
        "private_data_loaded": False,
        "target_values_loaded": False,
        "candidate_promotable": False,
        "soz_promotion": False,
        "qualification_scope": (
            "TUH-internal, target-excluded, likely pretraining-exposed"
        ),
    }
    for key, expected in exact_qualification.items():
        if qualification.get(key) != expected or type(qualification.get(key)) is not type(
            expected
        ):
            raise ValueError(f"Locked DAPT-v2 qualification field changed: {key}")
    paired = qualification.get("paired_metrics")
    if not isinstance(paired, Mapping):
        raise TypeError("Locked DAPT-v2 qualification lacks paired metrics")
    q_gates = (
        "q1_ce_zero_minus_v2",
        "q2_accuracy_v2_minus_zero",
        "q3_hard_log_perplexity_v2_minus_zero",
        "q4_source_reference_car_jsd_v2_minus_zero",
    )
    if any(
        not isinstance(paired.get(name), Mapping)
        or paired[name].get("passed") is not True
        for name in q_gates
    ):
        raise ValueError("Locked DAPT-v2 qualification Q1--Q4 did not all pass")
    lineage = qualification.get("source_run_lineage")
    if not isinstance(lineage, Mapping) or (
        lineage.get("source_run_receipt_sha256") != EXPECTED_RECEIPT_SHA256
        or lineage.get("selected_adapter_sha256") != EXPECTED_ADAPTER_SHA256
        or lineage.get("selected_epoch") != 0
        or lineage.get("selected_epoch_dev_eligible") is not True
    ):
        raise ValueError("Locked DAPT-v2 qualification lineage changed")
    state = _validate_adapter_state(
        torch.load(args.adapter_path, map_location="cpu", weights_only=True)
    )
    return AuthorizedDAPT(
        adapter_state=state,
        receipt=receipt,
        qualification=qualification,
        adapter_state_sha256=_adapter_state_sha(state),
    )


def _authorize_then_load_inputs(
    args: argparse.Namespace,
) -> tuple[AuthorizedDAPT, V11BInputs]:
    authorized = _authorize_dapt_candidate(args)
    # This call is the first operation allowed to load DeepSOZ targets.
    inputs = _load_v11b_inputs(args)
    return authorized, inputs


def _static_suffix(
    args: argparse.Namespace,
    *,
    adapter_state: Mapping[str, torch.Tensor] | None,
    seed: int,
    device: torch.device,
) -> OfficialLaBraMStaticAdapterSuffix:
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        suffix = OfficialLaBraMStaticAdapterSuffix(
            modeling_path=args.modeling_path,
            checkpoint_path=args.checkpoint_path,
            adapter_state=adapter_state,
            expected_sha256=AUDITED_LABRAM_BASE_SHA256,
            expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
        )
    suffix = suffix.to(device).eval()
    if suffix.n_trainable_parameters != 0 or any(
        parameter.requires_grad for parameter in suffix.parameters()
    ):
        raise RuntimeError("Locked downstream suffix must be strictly static")
    reference = next(suffix.parameters())
    if reference.dtype != torch.float32 or suffix.training or suffix.backbone.training:
        raise RuntimeError("Locked downstream suffix must be FP32 and eval-only")
    return suffix


def _validate_static_h(value: torch.Tensor, *, events: int, label: str) -> None:
    if tuple(value.shape) != (events, 19, 600):
        raise RuntimeError(f"{label} H shape changed")
    if value.dtype != torch.float32 or value.requires_grad:
        raise RuntimeError(f"{label} H must be detached FP32")
    if not torch.isfinite(value).all() or torch.count_nonzero(value).item() == 0:
        raise RuntimeError(f"{label} H must be finite and non-zero")


def _zero_effective_qkv_exact(
    suffix: OfficialLaBraMStaticAdapterSuffix,
) -> bool:
    for block in LABRAM_PEFT_BLOCKS:
        qkv = suffix.backbone.blocks[block].attn.qkv
        original = qkv.parametrizations.weight.original
        if not torch.equal(qkv.weight.detach(), original.detach()):
            return False
    return True


def _dapt_effective_qkv_changed(
    suffix: OfficialLaBraMStaticAdapterSuffix,
) -> bool:
    changed = False
    for block in LABRAM_PEFT_BLOCKS:
        qkv = suffix.backbone.blocks[block].attn.qkv
        effective = qkv.weight.detach()
        original = qkv.parametrizations.weight.original.detach()
        if not torch.isfinite(effective).all():
            return False
        changed |= not torch.equal(effective, original)
    return changed


def _reasoner_from_state(
    state: Mapping[str, torch.Tensor], *, device: torch.device
) -> SharedPositiveSetReasoner:
    model = SharedPositiveSetReasoner(state["prior_logits"], use_h=True, use_fine=True)
    model.load_state_dict(
        {name: value.detach().cpu().clone() for name, value in state.items()},
        strict=True,
    )
    model = model.to(device)
    if model.n_trainable_parameters != 36:
        raise RuntimeError("Locked downstream reasoner must expose exactly 36 parameters")
    return model


def _paired_train_batch(reference: TrainBatch, event_h: torch.Tensor) -> TrainBatch:
    if tuple(event_h.shape) != tuple(reference.zero_h.shape):
        raise ValueError("Paired final-suffix event H shapes differ")
    return TrainBatch(
        prefix=reference.prefix,
        zero_h=event_h,
        reliability=reference.reliability,
        event_patient_index=reference.event_patient_index,
        fine_patient=reference.fine_patient,
        targets=reference.targets,
        target_mask=reference.target_mask,
        patient_ids=reference.patient_ids,
    )


def _paired_prediction_batch(
    reference: PredictionBatch, event_h: torch.Tensor
) -> PredictionBatch:
    if tuple(event_h.shape) != tuple(reference.zero_h.shape):
        raise ValueError("Paired held final-suffix event H shapes differ")
    return PredictionBatch(
        prefix=reference.prefix,
        zero_h=event_h,
        reliability=reference.reliability,
        event_patient_index=reference.event_patient_index,
        fine_patient=reference.fine_patient,
        patient_ids=reference.patient_ids,
    )


def _fit_arm(
    batch: TrainBatch, *, device: torch.device, epochs: int
) -> tuple[FoldFeatureTransform, SharedPositiveSetReasoner, Mapping[str, object]]:
    transform, initial_state, warm = _fit_fold_initialization(batch)
    warm_h_norm = float(torch.linalg.vector_norm(initial_state["h_weight"]))
    if not warm_h_norm > 0:
        raise RuntimeError("Locked downstream warm-start H weight must be non-zero")
    reasoner = _reasoner_from_state(initial_state, device=device)
    reasoner, fit = _train_matched(
        batch, transform, reasoner, epochs=epochs, device=device
    )
    if reasoner.h_weight is None:
        raise RuntimeError("Locked downstream reasoner lost its H weight")
    final_h_norm = float(torch.linalg.vector_norm(reasoner.h_weight.detach()).cpu())
    if not final_h_norm > 0:
        raise RuntimeError("Locked downstream final H weight must be non-zero")
    return transform, reasoner, {
        "warm_start": dict(warm),
        "warm_start_h_weight_l2_norm": warm_h_norm,
        "final_h_weight_l2_norm": final_h_norm,
        "head_fit": fit,
    }


def _transform_pair_gates(
    zero_transform: FoldFeatureTransform,
    dapt_transform: FoldFeatureTransform,
    zero_head: SharedPositiveSetReasoner,
    dapt_head: SharedPositiveSetReasoner,
) -> dict[str, bool]:
    return {
        "independent_h_transform_storage": (
            zero_transform.h_components.data_ptr()
            != dapt_transform.h_components.data_ptr()
        ),
        "fine_center_exact": torch.equal(
            zero_transform.fine_center, dapt_transform.fine_center
        ),
        "fine_scale_exact": torch.equal(
            zero_transform.fine_scale, dapt_transform.fine_scale
        ),
        "prior_exact": torch.equal(
            zero_head.prior_logits.detach().cpu(), dapt_head.prior_logits.detach().cpu()
        ),
        "candidate_mask_exact": torch.equal(
            zero_head.candidate_mask.detach().cpu(),
            dapt_head.candidate_mask.detach().cpu(),
        ),
        "both_heads_36_parameters": (
            zero_head.n_trainable_parameters == 36
            and dapt_head.n_trainable_parameters == 36
        ),
    }


def _predict_from_static_h(
    batch: PredictionBatch,
    transform: FoldFeatureTransform,
    reasoner: SharedPositiveSetReasoner,
    *,
    device: torch.device,
) -> torch.Tensor:
    # Reuse the audited v11-B held predictor with precomputed static H.
    from scripts.run_labram_fine_temporal_peft_oof_v11_b import (  # noqa: PLC0415
        _predict_outer_held_no_grad,
    )

    return _predict_outer_held_no_grad(
        batch,
        transform,
        reasoner,
        None,
        device=device,
        event_microbatch=EVENT_MICROBATCH,
    )


def _checkpoint_arm(
    transform: FoldFeatureTransform,
    head: SharedPositiveSetReasoner,
    *,
    outer_fold: int,
    arm: str,
) -> dict[str, torch.Tensor]:
    prefix = f"outer{outer_fold}.{arm}"
    state = {
        f"{prefix}.transform.{name}": value.detach().cpu().clone()
        for name, value in _transform_state(transform).items()
    }
    state.update(
        {
            f"{prefix}.head.{name}": value.detach().cpu().clone()
            for name, value in head.state_dict().items()
        }
    )
    return state


def _bootstrap_indices(n_patients: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    return torch.randint(
        0,
        n_patients,
        (BOOTSTRAP_REPLICATES, n_patients),
        generator=generator,
    )


def _locked_patient_contributions(
    logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    rows = dict(_patient_contributions(logits, targets, target_mask))
    brier: list[float] = []
    nll: list[float] = []
    for patient in range(logits.shape[0]):
        result = _evaluate(
            logits[patient : patient + 1],
            targets[patient : patient + 1],
            target_mask[patient : patient + 1],
        )
        brier.append(result["ranking"]["reference_membership_brier"])
        nll.append(result["ranking"]["reference_membership_nll"])
    rows["reference_membership_brier"] = torch.tensor(brier, dtype=torch.float64)
    rows["reference_membership_nll"] = torch.tensor(nll, dtype=torch.float64)
    return rows


def _absolute_bootstrap_locked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    indices: torch.Tensor | None = None,
) -> dict[str, object]:
    rows = _locked_patient_contributions(logits, targets, target_mask)
    if indices is None:
        indices = _bootstrap_indices(logits.shape[0])
    if tuple(indices.shape) != (BOOTSTRAP_REPLICATES, logits.shape[0]):
        raise ValueError("Locked bootstrap draw tensor has the wrong shape")
    result: dict[str, object] = {}
    for name, contribution in rows.items():
        samples = contribution[indices].mean(dim=1)
        result[name] = {
            "estimate": float(contribution.mean()),
            "ci95": [
                float(torch.quantile(samples, 0.025)),
                float(torch.quantile(samples, 0.975)),
            ],
        }
    return result


def _paired_bootstrap_locked(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    indices: torch.Tensor | None = None,
) -> dict[str, object]:
    left = _locked_patient_contributions(candidate, targets, target_mask)
    right = _locked_patient_contributions(baseline, targets, target_mask)
    if indices is None:
        indices = _bootstrap_indices(candidate.shape[0])
    if tuple(indices.shape) != (BOOTSTRAP_REPLICATES, candidate.shape[0]):
        raise ValueError("Locked paired-bootstrap draw tensor has the wrong shape")
    result: dict[str, object] = {}
    for name in left:
        difference = left[name] - right[name]
        samples = difference[indices].mean(dim=1)
        result[name] = {
            "delta": float(difference.mean()),
            "ci95": [
                float(torch.quantile(samples, 0.025)),
                float(torch.quantile(samples, 0.975)),
            ],
        }
    return result


def _win_loss_tie(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, dict[str, int]]:
    left = _patient_contributions(candidate, targets, target_mask)
    right = _patient_contributions(baseline, targets, target_mask)
    result: dict[str, dict[str, int]] = {}
    for name in ("strict", "relaxed", "macro_ap"):
        delta = left[name] - right[name]
        result[name] = {
            "wins": int((delta > 0).sum()),
            "losses": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
        }
    return result


def _assess_decision(
    metrics: Mapping[str, Mapping[str, object]],
    paired: Mapping[str, Mapping[str, object]],
    fold_strict: Mapping[str, Sequence[float]],
    implementation_valid: bool,
) -> tuple[str, Mapping[str, bool]]:
    dapt = metrics[DAPT]
    zero = metrics[ZERO]
    four_folds = sum(
        left >= right
        for left, right in zip(fold_strict[DAPT], fold_strict[ZERO])
    ) >= 4
    shared = {
        "relaxed_point_nonlower": (
            dapt["top1"]["relaxed_accuracy"]
            >= zero["top1"]["relaxed_accuracy"]
        ),
        "far_error_nonincreasing": dapt["far_error_count"] <= zero["far_error_count"],
        "four_of_five_fold_strict_nonlower": four_folds,
        "implementation_valid": implementation_valid,
    }
    increment = {
        **shared,
        "strict_ci_lower_positive": paired["strict"]["ci95"][0] > 0,
        "macro_ap_ci_lower_positive": paired["macro_ap"]["ci95"][0] > 0,
    }
    noninferior = {
        "strict_ci_lower_ge_minus_0_05": (
            paired["strict"]["ci95"][0] >= -NONINFERIORITY_MARGIN
        ),
        "relaxed_ci_lower_ge_minus_0_05": (
            paired["relaxed"]["ci95"][0] >= -NONINFERIORITY_MARGIN
        ),
        "macro_ap_point_nonlower": (
            dapt["ranking"]["macro_average_precision"]
            >= zero["ranking"]["macro_average_precision"]
        ),
        "far_error_nonincreasing": shared["far_error_nonincreasing"],
        "four_of_five_fold_strict_nonlower": four_folds,
        "implementation_valid": implementation_valid,
    }
    checks = {
        **{f"increment.{name}": bool(value) for name, value in increment.items()},
        **{f"noninferior.{name}": bool(value) for name, value in noninferior.items()},
    }
    if all(increment.values()):
        return "DEVELOPMENTAL_SCALP_SOZ_INCREMENT_SUPPORTED", checks
    if all(noninferior.values()):
        return "DOWNSTREAM_NONINFERIOR_ONLY", checks
    return "DAPT_V2_DOWNSTREAM_NOT_SUPPORTED_STOP", checks


def _historical_payload(
    directory: Path,
    *,
    manifest_sha: str,
    oof_sha: str,
) -> tuple[Mapping[str, object], Mapping[str, torch.Tensor]]:
    manifest_path = directory / "manifest.json"
    oof_path = directory / "oof_predictions.safetensors"
    if _file_sha(manifest_path) != manifest_sha or _file_sha(oof_path) != oof_sha:
        raise ValueError("Pinned historical OOF artifact changed")
    return _load_json(manifest_path), load_file(str(oof_path), device="cpu")


def _formal_zero_parity(
    inputs: V11BInputs, zero_oof: torch.Tensor, directory: Path
) -> Mapping[str, object]:
    manifest, payload = _historical_payload(
        directory,
        manifest_sha=EXPECTED_V11B_R3_MANIFEST_SHA256,
        oof_sha=EXPECTED_V11B_R3_OOF_SHA256,
    )
    if tuple(str(value) for value in manifest.get("patient_ids", ())) != inputs.patient_ids:
        raise ValueError("v11-B r3 zero-parity patient order changed")
    for key, current in (("targets", inputs.targets), ("target_mask", inputs.target_mask)):
        if key not in payload or not torch.equal(payload[key], current):
            raise ValueError(f"v11-B r3 zero-parity {key} changed")
    reference = payload["oof.matched_frozen_final_suffix"]
    candidate = _masked_oof_scores_for_publish(zero_oof)
    error = float((candidate - reference).abs().max())
    if error > ZERO_PARITY_TOLERANCE:
        raise RuntimeError(
            f"Locked zero final-suffix parity failed: max_abs_error={error}"
        )
    return {
        "passed": True,
        "maximum_absolute_error": error,
        "tolerance": ZERO_PARITY_TOLERANCE,
        "manifest_sha256": EXPECTED_V11B_R3_MANIFEST_SHA256,
        "oof_sha256": EXPECTED_V11B_R3_OOF_SHA256,
    }


def _anchor_comparison(
    inputs: V11BInputs,
    dapt_oof: torch.Tensor,
    directory: Path,
    bootstrap_indices: torch.Tensor,
) -> Mapping[str, object]:
    manifest, payload = _historical_payload(
        directory,
        manifest_sha=EXPECTED_V11_1_MANIFEST_SHA256,
        oof_sha=EXPECTED_V11_1_OOF_SHA256,
    )
    patient_ids = tuple(str(value) for value in manifest.get("patient_ids", ()))
    if patient_ids != inputs.patient_ids:
        raise ValueError("v11.1 anchor patient order changed")
    if not torch.equal(payload["targets"], inputs.targets) or not torch.equal(
        payload["target_mask"], inputs.target_mask
    ):
        raise ValueError("v11.1 anchor target carrier changed")
    anchor = payload["oof.full_frozen_labram_plus_fine"]
    paired = _paired_bootstrap_locked(
        dapt_oof,
        anchor,
        inputs.targets,
        inputs.target_mask,
        bootstrap_indices,
    )
    eligible = (
        paired["strict"]["ci95"][0] >= -NONINFERIORITY_MARGIN
        and paired["macro_ap"]["ci95"][0] >= -NONINFERIORITY_MARGIN
    )
    return {
        "not_capacity_matched": True,
        "metrics": _evaluate(anchor, inputs.targets, inputs.target_mask),
        "paired_dapt_minus_anchor": paired,
        "future_mainline_review_noninferiority": bool(eligible),
    }


def _source_hashes() -> Mapping[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "static_suffix": ROOT / "src/soz/models/labram_static_suffix.py",
        "deepsoz_target_v2": ROOT / "src/soz/data/deepsoz_target_v2.py",
        "deepsoz_alias": ROOT / "src/soz/data/deepsoz.py",
        "geometry": ROOT / "src/soz/geometry.py",
        "metrics": ROOT / "src/soz/metrics.py",
        "development_union": ROOT / "src/soz/v11_development_union.py",
        "v11_1_evaluator": ROOT
        / "scripts/run_labram_fine_temporal_nested_oof_v11_1.py",
        "v11b_input_runner": ROOT
        / "scripts/run_labram_fine_temporal_peft_oof_v11_b.py",
        "v11b_bridge": ROOT / "src/soz/v11b_peft.py",
        "v11_reasoner": ROOT / "src/soz/v11_reasoner.py",
        "peft_suffix": ROOT / "src/soz/models/labram_peft.py",
    }
    return {name: _file_sha(path) for name, path in paths.items()}


def _assert_pinned_source_hashes(source_hashes: Mapping[str, str]) -> None:
    for name, expected in EXPECTED_PINNED_SOURCE_SHA256.items():
        if source_hashes.get(name) != expected:
            raise ValueError(f"Locked downstream source SHA256 changed: {name}")


def run(
    args: argparse.Namespace,
) -> tuple[Mapping[str, object], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    started = time.monotonic()
    source_hashes = _source_hashes()
    _assert_pinned_source_hashes(source_hashes)
    authorized, inputs = _authorize_then_load_inputs(args)
    device = torch.device(args.device)
    formal = inputs.formal_scope and args.epochs == FORMAL_EPOCHS and (
        args.event_microbatch == EVENT_MICROBATCH
    )
    if not formal or args.smoke:
        raise RuntimeError(
            "The locked two-phase comparison has no DeepSOZ smoke path; use unit tests"
        )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal locked DAPT-v2 downstream OOF requires CUDA")
    if (
        inputs.target_free_union_manifest_sha256 != EXPECTED_UNION_MANIFEST_SHA256
        or inputs.target_receipt_sha256 != EXPECTED_TARGET_RECEIPT_SHA256
        or inputs.target_artifact_sha256 != EXPECTED_TARGET_ARTIFACT_SHA256
        or _file_sha(args.source_csv) != EXPECTED_SOURCE_CSV_SHA256
        or _file_sha(args.split_csv) != EXPECTED_SPLIT_CSV_SHA256
    ):
        raise ValueError("Locked union/DeepSOZ target lineage changed")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    oof = {
        ZERO: torch.full((len(inputs.patient_ids), 19), torch.nan),
        DAPT: torch.full((len(inputs.patient_ids), 19), torch.nan),
    }
    fold_strict = {ZERO: [], DAPT: []}
    fold_results: dict[int, dict[str, object]] = {}
    zero_fold_reference: dict[
        int,
        tuple[
            FoldFeatureTransform,
            SharedPositiveSetReasoner,
            torch.Tensor,
            torch.Tensor,
        ],
    ] = {}
    checkpoint: dict[str, torch.Tensor] = {
        "config.candidate_mask": V11_CANDIDATE_MASK.clone()
    }
    original_qkv_sha: str | None = None
    implementation_valid_all = True

    # ------------------------------------------------------------------
    # Phase A: the complete five-fold exact-zero baseline.  No DAPT suffix
    # may be constructed or forwarded anywhere above the parity gate below.
    # ------------------------------------------------------------------
    for outer_fold in OUTER_FOLDS:
        torch.manual_seed(BASE_SEED + outer_fold)
        torch.cuda.manual_seed_all(BASE_SEED + outer_fold)
        train_indices = torch.nonzero(
            inputs.patient_folds != outer_fold, as_tuple=False
        ).flatten()
        held_indices = torch.nonzero(
            inputs.patient_folds == outer_fold, as_tuple=False
        ).flatten()
        train_event_indices, _ = _subset_events(inputs.event_patient_index, train_indices)
        train_prefix = inputs.prefix.index_select(0, train_event_indices)
        seed = BASE_SEED + outer_fold
        zero_suffix = _static_suffix(
            args, adapter_state=None, seed=seed, device=device
        )
        zero_state_before = zero_suffix.lora_state_dict()
        if any(
            torch.count_nonzero(value).item() != 0
            for key, value in zero_state_before.items()
            if key.endswith("lora_B")
        ):
            raise RuntimeError("Exact-zero suffix has a non-zero LoRA-B factor")
        zero_qkv = _state_sha(_qkv_original_state(zero_suffix))
        if original_qkv_sha is None:
            original_qkv_sha = zero_qkv
        elif original_qkv_sha != zero_qkv:
            raise RuntimeError("Original qkv state changed across outer folds")

        zero_h = _collect_suffix_h(
            zero_suffix,
            train_prefix,
            device=device,
            event_microbatch=args.event_microbatch,
        )
        _validate_static_h(zero_h, events=len(train_event_indices), label="zero train")
        del train_prefix
        zero_train = _train_batch(inputs, train_indices, zero_h)
        zero_transform, zero_head, zero_fit = _fit_arm(
            zero_train, device=device, epochs=args.epochs
        )

        # The zero held prefix enters the suffix only after this fold's zero
        # head optimizer has completed all 20 outer-train steps.
        held_event_indices, _ = _subset_events(inputs.event_patient_index, held_indices)
        held_prefix = inputs.prefix.index_select(0, held_event_indices)
        zero_held_h = _collect_suffix_h(
            zero_suffix,
            held_prefix,
            device=device,
            event_microbatch=args.event_microbatch,
        )
        _validate_static_h(
            zero_held_h, events=len(held_event_indices), label="zero held"
        )
        del held_prefix
        zero_held = _prediction_batch(inputs, held_indices, zero_held_h)
        zero_logits = _predict_from_static_h(
            zero_held, zero_transform, zero_head, device=device
        )
        zero_gates = {
            "zero_effective_qkv_exactly_official": _zero_effective_qkv_exact(
                zero_suffix
            ),
            "zero_adapter_immutable": all(
                torch.equal(zero_state_before[key], zero_suffix.lora_state_dict()[key])
                for key in zero_state_before
            ),
            "zero_foundation_trainable_parameters_0": (
                zero_suffix.n_trainable_parameters == 0
            ),
            "foundation_gradients_absent": all(
                parameter.grad is None
                for parameter in zero_suffix.parameters()
            ),
            "original_qkv_immutable": (
                _state_sha(_qkv_original_state(zero_suffix)) == original_qkv_sha
            ),
            "held_forward_after_zero_head_fit": True,
            "foundation_optimizer_parameter_count_0": True,
            "patient_pooling_before_train_only_scaler_pca": True,
            "warm_and_final_h_weight_nonzero": (
                zero_fit["warm_start_h_weight_l2_norm"] > 0
                and zero_fit["final_h_weight_l2_norm"] > 0
            ),
        }
        if not all(zero_gates.values()):
            failed = [name for name, value in zero_gates.items() if not value]
            raise RuntimeError(f"Locked zero-phase implementation gate failed: {failed}")
        implementation_valid_all &= all(zero_gates.values())
        oof[ZERO].index_copy_(0, held_indices, zero_logits)
        held_targets = inputs.targets.index_select(0, held_indices)
        held_mask = inputs.target_mask.index_select(0, held_indices)
        held_metrics = _evaluate(zero_logits, held_targets, held_mask)
        fold_strict[ZERO].append(held_metrics["top1"]["strict_accuracy"])
        checkpoint.update(
            _checkpoint_arm(
                zero_transform, zero_head, outer_fold=outer_fold, arm=ZERO
            )
        )
        zero_head = zero_head.to("cpu")
        zero_fold_reference[outer_fold] = (
            zero_transform,
            zero_head,
            train_event_indices.clone(),
            held_event_indices.clone(),
        )
        fold_results[outer_fold] = {
            "outer_fold": outer_fold,
            "train_patient_count": len(zero_train.patient_ids),
            "train_event_count": int(zero_train.prefix.shape[0]),
            "held_patient_count": len(zero_held.patient_ids),
            "held_event_count": int(zero_held.prefix.shape[0]),
            "phase_a_zero_fit": zero_fit,
            "phase_a_implementation_gates": zero_gates,
            "zero_adapter_state_sha256_before_after": _adapter_state_sha(
                zero_state_before
            ),
            "zero_held_metrics": held_metrics,
        }
        print(
            json.dumps(
                {
                    "phase": "A_zero_only",
                    "outer_fold": outer_fold,
                    "zero_strict": held_metrics["top1"]["strict_accuracy"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del zero_suffix
        torch.cuda.empty_cache()

    if not torch.isfinite(oof[ZERO]).all():
        raise RuntimeError("Phase A zero OOF scores are incomplete")
    zero_parity = _formal_zero_parity(
        inputs, oof[ZERO], args.v11b_r3_directory
    )

    # ------------------------------------------------------------------
    # Phase B: this is intentionally the first construction and first
    # DeepSOZ-prefix forward of the qualified DAPT suffix.  It is unreachable
    # unless the complete Phase-A historical tensor parity passed above.
    # ------------------------------------------------------------------
    for outer_fold in OUTER_FOLDS:
        if zero_parity.get("passed") is not True:
            raise RuntimeError("DAPT phase is forbidden before zero parity passes")
        torch.manual_seed(BASE_SEED + outer_fold)
        torch.cuda.manual_seed_all(BASE_SEED + outer_fold)
        train_indices = torch.nonzero(
            inputs.patient_folds != outer_fold, as_tuple=False
        ).flatten()
        held_indices = torch.nonzero(
            inputs.patient_folds == outer_fold, as_tuple=False
        ).flatten()
        train_event_indices, _ = _subset_events(inputs.event_patient_index, train_indices)
        train_prefix = inputs.prefix.index_select(0, train_event_indices)
        dapt_suffix = _static_suffix(
            args,
            adapter_state=authorized.adapter_state,
            seed=BASE_SEED + outer_fold,
            device=device,
        )
        dapt_state_before = dapt_suffix.lora_state_dict()
        if any(
            not torch.equal(dapt_state_before[key], authorized.adapter_state[key])
            for key in authorized.adapter_state
        ):
            raise RuntimeError("Static DAPT suffix did not exactly restore the adapter")
        if _state_sha(_qkv_original_state(dapt_suffix)) != original_qkv_sha:
            raise RuntimeError("DAPT original qkv differs from the zero phase")
        dapt_h = _collect_suffix_h(
            dapt_suffix,
            train_prefix,
            device=device,
            event_microbatch=args.event_microbatch,
        )
        _validate_static_h(dapt_h, events=len(train_event_indices), label="DAPT train")
        del train_prefix
        # Build the same complete-patient bag carrier as Phase A; this helper
        # pools patient events before train-only scaler/PCA fitting.
        dapt_train = _train_batch(inputs, train_indices, dapt_h)
        dapt_transform, dapt_head, dapt_fit = _fit_arm(
            dapt_train, device=device, epochs=args.epochs
        )
        (
            zero_transform,
            zero_head_reference,
            zero_train_event_order,
            zero_held_event_order,
        ) = zero_fold_reference[outer_fold]
        transform_gates = _transform_pair_gates(
            zero_transform, dapt_transform, zero_head_reference, dapt_head
        )

        # DAPT held forward occurs only after its 20-epoch head fit.
        held_event_indices, _ = _subset_events(inputs.event_patient_index, held_indices)
        held_prefix = inputs.prefix.index_select(0, held_event_indices)
        dapt_held_h = _collect_suffix_h(
            dapt_suffix,
            held_prefix,
            device=device,
            event_microbatch=args.event_microbatch,
        )
        _validate_static_h(
            dapt_held_h, events=len(held_event_indices), label="DAPT held"
        )
        del held_prefix
        # Reuse Phase-A batch metadata without another feature-family change.
        zero_placeholder = _prediction_batch(inputs, held_indices, dapt_held_h)
        dapt_logits = _predict_from_static_h(
            zero_placeholder, dapt_transform, dapt_head, device=device
        )
        dapt_gates = {
            **transform_gates,
            "dapt_effective_qkv_differs_from_official": _dapt_effective_qkv_changed(
                dapt_suffix
            ),
            "same_train_event_order_as_zero": torch.equal(
                train_event_indices, zero_train_event_order
            ),
            "same_held_event_order_as_zero": torch.equal(
                held_event_indices, zero_held_event_order
            ),
            "dapt_adapter_immutable": all(
                torch.equal(dapt_state_before[key], dapt_suffix.lora_state_dict()[key])
                for key in dapt_state_before
            ),
            "dapt_foundation_trainable_parameters_0": (
                dapt_suffix.n_trainable_parameters == 0
            ),
            "foundation_gradients_absent": all(
                parameter.grad is None for parameter in dapt_suffix.parameters()
            ),
            "original_qkv_matches_zero_and_is_immutable": (
                _state_sha(_qkv_original_state(dapt_suffix)) == original_qkv_sha
            ),
            "held_forward_after_dapt_head_fit": True,
            "foundation_optimizer_parameter_count_0": True,
            "patient_pooling_before_train_only_scaler_pca": True,
            "warm_and_final_h_weight_nonzero": (
                dapt_fit["warm_start_h_weight_l2_norm"] > 0
                and dapt_fit["final_h_weight_l2_norm"] > 0
            ),
            "zero_parity_passed_before_dapt_construction": True,
        }
        if not all(dapt_gates.values()):
            failed = [name for name, value in dapt_gates.items() if not value]
            raise RuntimeError(f"Locked DAPT-phase implementation gate failed: {failed}")
        implementation_valid_all &= all(dapt_gates.values())
        oof[DAPT].index_copy_(0, held_indices, dapt_logits)
        held_targets = inputs.targets.index_select(0, held_indices)
        held_mask = inputs.target_mask.index_select(0, held_indices)
        held_metrics = _evaluate(dapt_logits, held_targets, held_mask)
        fold_strict[DAPT].append(held_metrics["top1"]["strict_accuracy"])
        checkpoint.update(
            _checkpoint_arm(
                dapt_transform, dapt_head, outer_fold=outer_fold, arm=DAPT
            )
        )
        fold_results[outer_fold].update(
            {
                "phase_b_dapt_fit": dapt_fit,
                "phase_b_implementation_gates": dapt_gates,
                "dapt_adapter_state_sha256_before_after": _adapter_state_sha(
                    dapt_state_before
                ),
                "dapt_held_metrics": held_metrics,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "B_dapt_after_zero_parity",
                    "outer_fold": outer_fold,
                    "dapt_strict": held_metrics["top1"]["strict_accuracy"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del dapt_suffix, dapt_head
        torch.cuda.empty_cache()

    if any(not torch.isfinite(value).all() for value in oof.values()):
        raise RuntimeError("Locked two-phase OOF scores are incomplete")
    metrics = {
        arm: _evaluate(value, inputs.targets, inputs.target_mask)
        for arm, value in oof.items()
    }
    bootstrap_indices = _bootstrap_indices(len(inputs.patient_ids))
    bootstrap_bytes = bootstrap_indices.contiguous().numpy().tobytes(order="C")
    bootstrap_sha256 = hashlib.sha256(bootstrap_bytes).hexdigest()
    absolute = {
        arm: _absolute_bootstrap_locked(
            value, inputs.targets, inputs.target_mask, bootstrap_indices
        )
        for arm, value in oof.items()
    }
    paired = _paired_bootstrap_locked(
        oof[DAPT],
        oof[ZERO],
        inputs.targets,
        inputs.target_mask,
        bootstrap_indices,
    )
    win_loss_tie = _win_loss_tie(
        oof[DAPT], oof[ZERO], inputs.targets, inputs.target_mask
    )
    decision, decision_checks = _assess_decision(
        metrics,
        paired,
        fold_strict,
        implementation_valid_all and bool(zero_parity.get("passed")),
    )
    anchor = _anchor_comparison(
        inputs,
        oof[DAPT],
        args.v11_1_directory,
        bootstrap_indices,
    )
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    ending_source_hashes = _source_hashes()
    _assert_pinned_source_hashes(ending_source_hashes)
    if ending_source_hashes != source_hashes:
        raise RuntimeError("Locked downstream source files changed during execution")
    if any(
        token in key.lower()
        for key in checkpoint
        for token in ("backbone", "original", "adapter", "optimizer", "raw", "target")
    ):
        raise RuntimeError("Locked downstream checkpoint contains forbidden state")

    manifest = {
        "schema_version": SCHEMA,
        "status": (
            "completed_formal_public_development_oof"
        ),
        "decision": decision,
        "formal_scope": formal,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "claim_boundary": {
            "public_developmental_paired_oof": True,
            "pretraining_exposed": True,
            "external_validation": False,
            "clinical_deployment_allowed": False,
            "private_used": False,
        },
        "foundation": {
            "backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "original_qkv_sha256": original_qkv_sha,
            "foundation_trainable_parameters_both_arms": 0,
            "foundation_optimizer_parameters_both_arms": 0,
            "official_weights_serialized": False,
        },
        "adapter_lineage": {
            "receipt_sha256": EXPECTED_RECEIPT_SHA256,
            "adapter_file_sha256": EXPECTED_ADAPTER_SHA256,
            "adapter_state_sha256": authorized.adapter_state_sha256,
            "qualification_sha256": EXPECTED_QUALIFICATION_SHA256,
            "selected_epoch": 0,
            "representation_qualified": True,
            "qualification_scope": (
                "TUH-internal, target-excluded, likely pretraining-exposed"
            ),
            "adapter_serialized_in_downstream_state": False,
        },
        "training": {
            "outer_folds": list(OUTER_FOLDS),
            "epochs": args.epochs,
            "head_trainable_parameters_per_arm": 36,
            "warm_start_l2": 0.20,
            "head_optimizer": "AdamW",
            "head_learning_rate": 3e-3,
            "weight_decay": 1e-2,
            "gradient_clip": 1.0,
            "early_stopping": False,
            "amp": False,
            "loss": "patient_equal_positive_set_mass",
            "arm_specific_train_only_h_scaler_pca": True,
        },
        "bootstrap": {
            "unit": "patient",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "ci": [0.025, 0.975],
            "method": "uncorrected_percentile_not_BCa",
            "conditional_on_fixed_folds_and_fitted_models": True,
            "patient_index_draws_shape": list(bootstrap_indices.shape),
            "patient_index_draws_encoding": "numpy_dtype_<i8_C_order_raw_bytes_no_header",
            "patient_index_draws_sha256": bootstrap_sha256,
        },
        "patient_count": len(inputs.patient_ids),
        "event_count": int(inputs.prefix.shape[0]),
        "patient_ids": list(inputs.patient_ids),
        "patient_folds": inputs.patient_folds.tolist(),
        "event_counts": inputs.event_counts.tolist(),
        "fixed_candidate_mask": V11_CANDIDATE_MASK.tolist(),
        "metrics": metrics,
        "absolute_patient_bootstrap": absolute,
        "paired_dapt_minus_zero": paired,
        "patient_win_loss_tie": win_loss_tie,
        "fold_strict": fold_strict,
        "fold_results": [fold_results[fold] for fold in OUTER_FOLDS],
        "decision_checks": decision_checks,
        "zero_arm_v11b_r3_parity": zero_parity,
        "v11_1_block9_anchor_comparison": anchor,
        "implementation_valid_all_folds": implementation_valid_all,
        "lineage": {
            "union_manifest_sha256": inputs.target_free_union_manifest_sha256,
            "target_receipt_sha256": inputs.target_receipt_sha256,
            "target_artifact_sha256": inputs.target_artifact_sha256,
            "source_csv_sha256": EXPECTED_SOURCE_CSV_SHA256,
            "split_csv_sha256": EXPECTED_SPLIT_CSV_SHA256,
            "v11b_r3_manifest_sha256": EXPECTED_V11B_R3_MANIFEST_SHA256,
            "v11b_r3_oof_sha256": EXPECTED_V11B_R3_OOF_SHA256,
            "v11_1_manifest_sha256": EXPECTED_V11_1_MANIFEST_SHA256,
            "v11_1_oof_sha256": EXPECTED_V11_1_OOF_SHA256,
            "prefix_manifest_sha256": EXPECTED_PREFIX_MANIFEST_SHA256,
            "prefix_tensor_sha256": EXPECTED_PREFIX_TENSOR_SHA256,
            "fine_manifest_sha256": EXPECTED_FINE_MANIFEST_SHA256,
            "fine_tensor_sha256": EXPECTED_FINE_TENSOR_SHA256,
        },
        "source_file_sha256": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "resource_usage": {
            "wall_time_seconds": time.monotonic() - started,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
        },
        "access_receipt": {
            "qualification_validated_before_deepsoz_load": True,
            "phase_a_complete_zero_oof_before_any_dapt_suffix_construction": True,
            "zero_v11b_r3_parity_passed_before_any_dapt_suffix_construction": True,
            "patient_258_excluded_before_all_fit": True,
            "held_prefix_forward_only_after_corresponding_arm_head_fit": True,
            "held_targets_not_passed_to_prediction": True,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_forward_count": 0,
            "llm_used_as_soz_predictor": False,
        },
    }
    tensors = {
        f"oof.{arm}": _masked_oof_scores_for_publish(value)
        for arm, value in oof.items()
    }
    tensors.update(
        {
            "targets": inputs.targets,
            "target_mask": inputs.target_mask,
            "patient_folds": inputs.patient_folds,
            "patient_event_counts": inputs.event_counts,
            "config.candidate_mask": V11_CANDIDATE_MASK.clone(),
        }
    )
    return manifest, tensors, checkpoint


def _publish(
    output_directory: Path,
    manifest: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    checkpoint: Mapping[str, torch.Tensor],
) -> Path:
    target = Path(os.path.abspath(output_directory))
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        predictions = staging / "oof_predictions.safetensors"
        states = staging / "fold_downstream_states.safetensors"
        save_file(dict(tensors), str(predictions))
        save_file(dict(checkpoint), str(states))
        completed = dict(manifest)
        completed["files"] = {
            predictions.name: {
                "sha256": _file_sha(predictions),
                "size_bytes": predictions.stat().st_size,
            },
            states.name: {
                "sha256": _file_sha(states),
                "size_bytes": states.stat().st_size,
            },
        }
        (staging / "manifest.json").write_bytes(_canonical_bytes(completed, newline=True))
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--fine-directory", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--prefix-directory", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--target-directory", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--modeling-path", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--receipt-path", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--qualification-path", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--v11b-r3-directory", type=Path, default=DEFAULT_V11B_R3)
    parser.add_argument("--v11-1-directory", type=Path, default=DEFAULT_V11_1)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--event-microbatch", type=int, default=EVENT_MICROBATCH)
    # The reused v11-B input loader requires this attribute.  This protocol
    # intentionally exposes no DeepSOZ smoke path because Phase B is forbidden
    # until complete formal zero parity has passed.
    parser.set_defaults(smoke=False)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.event_microbatch < 1:
        parser.error("epochs and event-microbatch must be positive")
    if not args.smoke and (
        args.epochs != FORMAL_EPOCHS or args.event_microbatch != EVENT_MICROBATCH
    ):
        parser.error("formal run locks epochs=20 and event-microbatch=4")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    manifest, tensors, checkpoint = run(args)
    output = args.output_directory
    path = _publish(output, manifest, tensors, checkpoint)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "decision": manifest["decision"],
                "path": str(path),
                "private_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
