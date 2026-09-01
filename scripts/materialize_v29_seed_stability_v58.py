#!/usr/bin/env python3
"""Materialize frozen-v29 D-head seed stability on public/private v58.

Only the initialization seed of the 206-parameter D head is varied.  The
official frozen LaBraM features, H carrier, patient folds, auxiliary roster,
patient-equal seizure bags, set-mass loss, optimizer, epochs, prior, and fixed
0.5 H / 0.5 D fusion remain unchanged.  The formal v29 artifacts are retained
as seed family zero; four additional seed families are replayed for audit only.

All 88 private predictions are materialized without loading the historically
available significant/spread reference.  No seed may replace or be ensembled
with the frozen v29 based on this experiment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_labram_masked_variable_auxiliary_oof_v17 as v17  # noqa: E402
import scripts.run_labram_rank1_direct_token_oof_v28 as v28  # noqa: E402
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import _evaluate  # noqa: E402
from scripts.materialize_v29_direct_objective_sensitivity_v53 import (  # noqa: E402
    _load_private_phase,
    _probability,
)
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    jeffreys_reference_prior_logits,
)


SCHEMA = "trustworthy_soz_v29_D_head_seed_stability_v58"
DEFAULT_PUBLIC_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE_PHASE = ROOT / "outputs/private_target_blind_rank1_phase_v29_20260815"
DEFAULT_PRIVATE_V29 = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_seed_stability_v58_20260816"
SEED_OFFSETS = (0, 100_000, 200_000, 300_000, 400_000)
SEED_NAMES = tuple(f"seed_offset_{value}" for value in SEED_OFFSETS)


def _topk_jaccard(left: torch.Tensor, right: torch.Tensor, k: int = 3) -> float:
    if left.shape != right.shape or left.ndim != 2 or not 1 <= k <= left.shape[1]:
        raise ValueError("top-k inputs must be aligned probability matrices")
    lhs = left.topk(k, dim=1).indices
    rhs = right.topk(k, dim=1).indices
    values = []
    for row in range(len(lhs)):
        a = set(int(value) for value in lhs[row].tolist())
        b = set(int(value) for value in rhs[row].tolist())
        values.append(len(a & b) / len(a | b))
    return float(sum(values) / len(values))


def _rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("rank inputs must be aligned matrices")
    lhs = left.argsort(dim=1).argsort(dim=1).double()
    rhs = right.argsort(dim=1).argsort(dim=1).double()
    lhs = lhs - lhs.mean(dim=1, keepdim=True)
    rhs = rhs - rhs.mean(dim=1, keepdim=True)
    denominator = lhs.square().sum(dim=1).sqrt() * rhs.square().sum(dim=1).sqrt()
    value = (lhs * rhs).sum(dim=1) / denominator.clamp_min(1e-12)
    return float(value.mean())


def _stability_summary(probabilities: Mapping[str, torch.Tensor]) -> dict[str, object]:
    if tuple(probabilities) != SEED_NAMES:
        raise ValueError("seed probability mapping order changed")
    formal = probabilities[SEED_NAMES[0]]
    formal_top = formal.argmax(dim=1)
    versus_formal: dict[str, object] = {}
    pairwise_top1: list[float] = []
    pairwise_top3: list[float] = []
    pairwise_rank: list[float] = []
    names = list(SEED_NAMES)
    for index, name in enumerate(names):
        value = probabilities[name]
        if value.shape != formal.shape or not torch.isfinite(value).all():
            raise ValueError("seed probability shape/value changed")
        versus_formal[name] = {
            "top1_retention": float((value.argmax(dim=1) == formal_top).float().mean()),
            "top3_jaccard": _topk_jaccard(value, formal, k=3),
            "mean_rank_correlation": _rank_correlation(value, formal),
            "mean_probability_L1": float((value - formal).abs().sum(dim=1).mean()),
        }
        for other in names[index + 1 :]:
            rhs = probabilities[other]
            pairwise_top1.append(
                float((value.argmax(dim=1) == rhs.argmax(dim=1)).float().mean())
            )
            pairwise_top3.append(_topk_jaccard(value, rhs, k=3))
            pairwise_rank.append(_rank_correlation(value, rhs))
    return {
        "versus_formal_v29": versus_formal,
        "all_pairwise": {
            "pair_count": len(pairwise_top1),
            "top1_agreement_mean": float(sum(pairwise_top1) / len(pairwise_top1)),
            "top1_agreement_range": [min(pairwise_top1), max(pairwise_top1)],
            "top3_jaccard_mean": float(sum(pairwise_top3) / len(pairwise_top3)),
            "top3_jaccard_range": [min(pairwise_top3), max(pairwise_top3)],
            "rank_correlation_mean": float(sum(pairwise_rank) / len(pairwise_rank)),
            "rank_correlation_range": [min(pairwise_rank), max(pairwise_rank)],
        },
    }


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    started = time.monotonic()
    stable = v17._load_stable_development(args)
    auxiliary = v17._load_auxiliary_targets(args, stable)
    auxiliary_prefix_cache = v17._load_auxiliary_cache(
        args.aux_prefix_directory,
        expected_manifest_sha256=args.expected_aux_prefix_manifest_sha256,
        expected_tensor_sha256=args.expected_aux_prefix_tensor_sha256,
        schema=v17.PREFIX_SCHEMA,
        expected_keys=v17.PREFIX_TENSOR_KEYS,
        primary_key="prefix_tokens",
        auxiliary=auxiliary,
        label="auxiliary LaBraM prefix for v58 seed stability",
    )
    stable_prefix, stable_event_patient_index = v28._load_stable_prefix(args, stable)
    stable_features = v28.extract_rank1_phase_features(stable_prefix)
    del stable_prefix
    auxiliary_features = v28.extract_rank1_phase_features(
        auxiliary_prefix_cache.tensors["prefix_tokens"].float()
    )
    stable_count = len(stable.patient_ids)
    combined = v28.PatientBag(
        phase_features=torch.cat((stable_features, auxiliary_features), dim=0),
        event_patient_index=torch.cat(
            (
                stable_event_patient_index,
                auxiliary.event_patient_index + stable_count,
            ),
            dim=0,
        ),
        targets=torch.cat((stable.targets, auxiliary.targets), dim=0),
        target_mask=torch.cat((stable.target_mask, auxiliary.target_mask), dim=0),
        patient_ids=stable.patient_ids + auxiliary.patient_ids,
    )

    public_v29 = load_file(
        str((args.public_v29_directory / "oof_predictions.safetensors").resolve(strict=True))
    )
    if not torch.equal(stable.targets, public_v29["targets"]) or not torch.equal(
        stable.target_mask, public_v29["target_mask"].bool()
    ):
        raise ValueError("public stable target differs from frozen v29")
    h_public = public_v29["oof.h_only_probability"].float()
    formal_public_d = public_v29["oof.rank1_direct_probability"].float()
    formal_public = public_v29["oof.portable_equal_ensemble_probability"].float()

    private_events, private_phase = _load_private_phase(args.private_phase_directory)
    private_v29 = load_file(
        str((args.private_v29_directory / "predictions.safetensors").resolve(strict=True))
    )
    private_h_fold = private_v29["private_h_only_fold_probability"].float()
    formal_private_d_fold = private_v29["private_rank1_direct_fold_probability"].float()
    formal_private = private_v29["private_portable_equal_probability"].float()
    if tuple(private_h_fold.shape) != (88, 5, 19):
        raise ValueError("private frozen H fold predictions changed")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    public_probability: dict[str, torch.Tensor] = {
        SEED_NAMES[0]: formal_public
    }
    private_probability: dict[str, torch.Tensor] = {
        SEED_NAMES[0]: formal_private
    }
    public_d_probability: dict[str, torch.Tensor] = {
        SEED_NAMES[0]: formal_public_d
    }
    private_d_fold_probability: dict[str, torch.Tensor] = {
        SEED_NAMES[0]: formal_private_d_fold
    }
    fit_receipts: dict[str, list[dict[str, object]]] = {
        SEED_NAMES[0]: [
            {
                "source": "formal_frozen_v29_v28_outer_states",
                "seed_offset": 0,
                "training_replayed": False,
            }
        ]
    }

    for offset, name in zip(SEED_OFFSETS[1:], SEED_NAMES[1:]):
        oof_logits = torch.full((stable_count, 19), torch.nan)
        private_fold_rows: list[torch.Tensor] = []
        receipts: list[dict[str, object]] = []
        for fold in v28.OUTER_FOLDS:
            stable_train = tuple(
                torch.nonzero(stable.patient_folds != fold, as_tuple=False).flatten().tolist()
            )
            stable_held = tuple(
                torch.nonzero(stable.patient_folds == fold, as_tuple=False).flatten().tolist()
            )
            auxiliary_train = tuple(
                stable_count + index
                for index in torch.nonzero(
                    auxiliary.outer_folds != fold, as_tuple=False
                ).flatten().tolist()
            )
            train_bag = v28._subset_bag(combined, stable_train + auxiliary_train)
            held_bag = v28._subset_bag(combined, stable_held)
            train_tensor = torch.tensor(stable_train, dtype=torch.long)
            prior = jeffreys_reference_prior_logits(
                stable.targets.index_select(0, train_tensor),
                stable.target_mask.index_select(0, train_tensor),
            )
            seed = v28.BASE_SEED + offset + 1_000 * fold
            model, receipt = v28._fit(train_bag, prior, seed=seed, device=device)
            held_logits = v28._predict(model, held_bag, device)
            oof_logits[list(stable_held)] = held_logits
            with torch.inference_mode():
                private_logits = model(private_phase.to(device)).detach().cpu()
            private_fold_rows.append(_probability(private_logits))
            receipts.append(
                {
                    **receipt,
                    "seed_offset": offset,
                    "fold": fold,
                    "stable_train_patients": len(stable_train),
                    "auxiliary_train_patients": len(auxiliary_train),
                    "held_stable_patients": len(stable_held),
                }
            )
            print(
                json.dumps(
                    {
                        "seed_family": name,
                        "fold": fold,
                        "elapsed_sec": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not torch.isfinite(oof_logits).all():
            raise RuntimeError(f"{name} public OOF predictions are incomplete")
        d_public = _probability(oof_logits)
        d_private_fold = torch.stack(private_fold_rows, dim=1).contiguous()
        ensemble_public = (0.5 * h_public + 0.5 * d_public).contiguous()
        ensemble_private = (
            0.5 * private_h_fold + 0.5 * d_private_fold
        ).mean(dim=1).contiguous()
        public_d_probability[name] = d_public
        private_d_fold_probability[name] = d_private_fold
        public_probability[name] = ensemble_public
        private_probability[name] = ensemble_private
        fit_receipts[name] = receipts

    tensors: dict[str, torch.Tensor] = {
        "public.targets": stable.targets,
        "public.target_mask": stable.target_mask,
        "public.patient_folds": stable.patient_folds,
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
    }
    public_metrics: dict[str, object] = {}
    for name in SEED_NAMES:
        tensors[f"public.{name}.D_probability"] = public_d_probability[name]
        tensors[f"public.{name}.ensemble_probability"] = public_probability[name]
        tensors[f"private.{name}.D_fold_probability"] = private_d_fold_probability[name]
        tensors[f"private.{name}.ensemble_probability"] = private_probability[name]
        public_metrics[name] = _evaluate(
            torch.log(public_probability[name].clamp_min(1e-12)),
            stable.targets,
            stable.target_mask,
        )

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_public_OOF_and_target_blind_private_D_seed_stability",
        "seed_offsets": list(SEED_OFFSETS),
        "seed_names": list(SEED_NAMES),
        "formal_v29_seed_family": SEED_NAMES[0],
        "fixed_components": {
            "foundation": "official pretrained LaBraM block-9 frozen",
            "foundation_trainable_parameters": 0,
            "H_carrier_and_heads": "frozen formal v29",
            "D_head_parameters": 206,
            "patient_folds": 5,
            "patient_pooling": "equal seizure mean",
            "auxiliary_roster": "frozen 9 patients / 182 events",
            "loss": "patient_equal_positive_set_probability_mass_nll",
            "optimizer": "AdamW",
            "epochs": v28.EPOCHS,
            "learning_rate": v28.LEARNING_RATE,
            "weight_decay": v28.WEIGHT_DECAY,
            "fusion": "fixed 0.5 H probability + 0.5 D probability",
        },
        "public": {
            "patients": stable_count,
            "events": len(stable_features),
            "metrics_by_seed": public_metrics,
            "prediction_stability": _stability_summary(public_probability),
        },
        "private": {
            "target_blind_events": len(private_events),
            "patients": len({str(row["patient_id"]) for row in private_events}),
            "events": private_events,
            "prediction_stability": _stability_summary(private_probability),
        },
        "fit_receipts": fit_receipts,
        "tensor_file": "seed_predictions.safetensors",
        "access_receipt": {
            "public_targets_loaded_for_fixed_OOF_training_and_evaluation": True,
            "private_target_blind_phase_and_H_loaded": True,
            "private_significant_or_spread_reference_loaded": False,
            "private_used_for_seed_model_or_ensemble_selection": False,
            "foundation_training_performed": False,
            "backbone_loss_window_anchor_fusion_or_threshold_changed": False,
        },
        "interpretation_boundary": {
            "audit_varies_only_D_head_initialization_seed": True,
            "formal_v29_may_be_replaced_or_ensembled_from_audit": False,
            "H_branch_training_stochasticity_audited": False,
            "foundation_pretraining_stochasticity_audited": False,
            "private_is_fresh_external_validation": False,
            "allowed_claim": (
                "optimizer-seed sensitivity of the frozen v29 D adaptation head "
                "is measured without changing the released model"
            ),
        },
        "elapsed_sec": time.monotonic() - started,
    }
    return result, tensors


def publish(
    *, output: Path, result: Mapping[str, object], tensors: Mapping[str, torch.Tensor]
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        save_file(dict(tensors), str(staging / "seed_predictions.safetensors"))
        (staging / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = v28.build_parser()
    parser.description = __doc__
    parser.set_defaults(output_directory=DEFAULT_OUTPUT, device="cpu")
    parser.add_argument("--public-v29-directory", type=Path, default=DEFAULT_PUBLIC_V29)
    parser.add_argument("--private-phase-directory", type=Path, default=DEFAULT_PRIVATE_PHASE)
    parser.add_argument("--private-v29-directory", type=Path, default=DEFAULT_PRIVATE_V29)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = build_parser().parse_args(argv)
    result, tensors = run(args)
    output = publish(output=args.output_directory, result=result, tensors=tensors)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "seed_families": len(SEED_NAMES),
                "private_reference_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
