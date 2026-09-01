#!/usr/bin/env python3
"""Materialize public/private v29 direct-token objective sensitivity v53.

The frozen v29 set-mass D branch is compared with two prespecified alternative
objectives while keeping the official frozen LaBraM features, public patient
folds, patient-equal seizure bags, 206-parameter head, fold-local Jeffreys
prior, optimizer, and fixed 0.5 H / 0.5 D probability ensemble unchanged.

The alternatives are uniform-positive soft cross-entropy and patient-balanced
binary cross-entropy over the benchmark-complement view.  All public OOF and
88 private target-blind predictions are materialized before a separate audit
opens the historically available private significant/spread reference.  This
experiment cannot replace v29 or select an objective from private results.
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
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_labram_masked_variable_auxiliary_oof_v17 as v17  # noqa: E402
import scripts.run_labram_rank1_direct_token_oof_v28 as v28  # noqa: E402
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import _evaluate  # noqa: E402
from scripts.predict_private_labram_portable_equal_v29 import _direct_probability  # noqa: E402
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    jeffreys_reference_prior_logits,
)


SCHEMA = "trustworthy_soz_v29_direct_objective_sensitivity_v53"
DEFAULT_PUBLIC_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_V28 = ROOT / "outputs/labram_rank1_direct_token_oof_v28_20260815"
DEFAULT_PRIVATE_PHASE = ROOT / "outputs/private_target_blind_rank1_phase_v29_20260815"
DEFAULT_PRIVATE_V29 = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_direct_objective_sensitivity_v53_20260816"
OBJECTIVES = (
    "set_mass_frozen_v28",
    "uniform_positive_soft_ce",
    "patient_balanced_bce",
)


def _objective_loss(
    name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if name not in OBJECTIVES[1:]:
        raise ValueError(f"unsupported trainable objective: {name}")
    if logits.ndim != 2 or tuple(targets.shape) != tuple(logits.shape) or tuple(
        target_mask.shape
    ) != tuple(logits.shape):
        raise ValueError("objective inputs must have aligned [P,19] shapes")
    if target_mask.dtype != torch.bool or not targets.is_floating_point():
        raise TypeError("objective targets/mask dtype is invalid")
    rows: list[torch.Tensor] = []
    for patient in range(len(logits)):
        observed = target_mask[patient]
        positive = observed & (targets[patient] == 1)
        negative = observed & (targets[patient] == 0)
        if not bool(positive.any()) or not bool(negative.any()):
            raise ValueError("each objective row requires positive and complement channels")
        if name == "uniform_positive_soft_ce":
            log_probability = F.log_softmax(logits[patient, observed], dim=0)
            observed_positive = targets[patient, observed] == 1
            rows.append(-log_probability[observed_positive].mean())
        else:
            positive_loss = F.binary_cross_entropy_with_logits(
                logits[patient, positive], torch.ones_like(logits[patient, positive])
            )
            negative_loss = F.binary_cross_entropy_with_logits(
                logits[patient, negative], torch.zeros_like(logits[patient, negative])
            )
            rows.append(0.5 * positive_loss + 0.5 * negative_loss)
    result = torch.stack(rows).mean()
    if not torch.isfinite(result):
        raise RuntimeError("objective loss became non-finite")
    return result


def _fit_alternative(
    bag: v28.PatientBag,
    prior: torch.Tensor,
    *,
    objective: str,
    seed: int,
    device: torch.device,
) -> tuple[v28.RankOneDirectTokenHead, dict[str, object]]:
    moved = bag.to(device)
    model = v28._seeded_model(prior, seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=v28.LEARNING_RATE, weight_decay=v28.WEIGHT_DECAY
    )
    first = None
    final = None
    for _ in range(v28.EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        event_logits = model(moved.phase_features)
        patient_logits = v28._aggregate_equal(
            event_logits, moved.event_patient_index, len(moved.patient_ids)
        )
        loss = _objective_loss(
            objective, patient_logits, moved.targets, moved.target_mask
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), v28.MAX_GRAD_NORM)
        optimizer.step()
        value = float(loss.detach().cpu())
        first = value if first is None else first
        final = value
    model.eval().requires_grad_(False)
    return model, {
        "objective": objective,
        "seed": seed,
        "epochs": v28.EPOCHS,
        "first_loss": first,
        "final_loss": final,
        "trainable_parameters": model.n_trainable_parameters,
    }


def _probability(logits: torch.Tensor) -> torch.Tensor:
    result = torch.softmax(logits.masked_fill(~V11_CANDIDATE_MASK, -torch.inf), dim=1)
    if not torch.isfinite(result).all() or not torch.allclose(
        result.sum(dim=1), torch.ones(len(result)), atol=1e-6, rtol=0
    ):
        raise RuntimeError("candidate probability contract failed")
    return result.contiguous()


def _load_private_phase(directory: Path) -> tuple[list[dict[str, object]], torch.Tensor]:
    manifest = json.loads((directory / "manifest.json").resolve(strict=True).read_text())
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != 88:
        raise ValueError("private phase roster changed")
    access = manifest.get("access_receipt", {})
    if access.get("private_target_values_loaded") not in (False, None):
        raise ValueError("private phase artifact is not target blind")
    payload = load_file(str((directory / str(manifest["tensor_file"])).resolve(strict=True)))
    phase = payload["phase_features"].float()
    if tuple(phase.shape) != (88, 19, 5, 200):
        raise ValueError("private phase feature shape changed")
    return events, phase


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
        label="auxiliary LaBraM prefix for v53 objective sensitivity",
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
    frozen_v28 = load_file(
        str((args.v28_directory / "model_and_oof.safetensors").resolve(strict=True))
    )
    for name in ("targets", "target_mask", "patient_folds"):
        if not torch.equal(public_v29[name], frozen_v28[name]):
            raise ValueError(f"public v29/v28 carrier differs: {name}")
    if not torch.equal(stable.targets, public_v29["targets"]) or not torch.equal(
        stable.target_mask, public_v29["target_mask"].bool()
    ):
        raise ValueError("loaded public stable target differs from frozen v29")
    h_public = public_v29["oof.h_only_probability"].float()
    current_public_d = public_v29["oof.rank1_direct_probability"].float()
    current_public_ensemble = public_v29[
        "oof.portable_equal_ensemble_probability"
    ].float()

    private_events, private_phase = _load_private_phase(args.private_phase_directory)
    private_v29 = load_file(
        str((args.private_v29_directory / "predictions.safetensors").resolve(strict=True))
    )
    current_private_d_fold = private_v29["private_rank1_direct_fold_probability"].float()
    private_h_fold = private_v29["private_h_only_fold_probability"].float()
    current_private_ensemble = private_v29[
        "private_portable_equal_probability"
    ].float()
    if tuple(private_h_fold.shape) != (88, 5, 19):
        raise ValueError("private H fold carrier changed")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    public_d: dict[str, torch.Tensor] = {
        "set_mass_frozen_v28": current_public_d
    }
    private_d_fold: dict[str, torch.Tensor] = {
        "set_mass_frozen_v28": current_private_d_fold
    }
    fit_receipts: dict[str, list[dict[str, object]]] = {
        "set_mass_frozen_v28": [
            {
                "source": "frozen_v28_outer_states",
                "training_replayed": False,
                "objective": "positive_set_probability_mass_nll",
            }
        ]
    }
    state_tensors: dict[str, torch.Tensor] = {}
    for objective in OBJECTIVES[1:]:
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
                for index in torch.nonzero(auxiliary.outer_folds != fold, as_tuple=False)
                .flatten()
                .tolist()
            )
            train_bag = v28._subset_bag(combined, stable_train + auxiliary_train)
            held_bag = v28._subset_bag(combined, stable_held)
            train_tensor = torch.tensor(stable_train, dtype=torch.long)
            prior = jeffreys_reference_prior_logits(
                stable.targets.index_select(0, train_tensor),
                stable.target_mask.index_select(0, train_tensor),
            )
            model, receipt = _fit_alternative(
                train_bag,
                prior,
                objective=objective,
                seed=v28.BASE_SEED + 1000 * fold,
                device=device,
            )
            held_logits = v28._predict(model, held_bag, device)
            oof_logits[list(stable_held)] = held_logits
            with torch.inference_mode():
                private_logits = model(private_phase.to(device)).detach().cpu()
            private_fold_rows.append(_probability(private_logits))
            for name, value in model.state_dict().items():
                state_tensors[f"{objective}.fold{fold}.{name}"] = (
                    value.detach().cpu().contiguous()
                )
            receipts.append(
                {
                    **receipt,
                    "fold": fold,
                    "stable_train_patients": len(stable_train),
                    "auxiliary_train_patients": len(auxiliary_train),
                    "held_stable_patients": len(stable_held),
                }
            )
            print(
                json.dumps(
                    {
                        "objective": objective,
                        "fold": fold,
                        "stage": "complete",
                        "elapsed_sec": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not torch.isfinite(oof_logits).all():
            raise RuntimeError(f"{objective} public OOF is incomplete")
        public_d[objective] = _probability(oof_logits)
        private_d_fold[objective] = torch.stack(private_fold_rows, dim=1).contiguous()
        fit_receipts[objective] = receipts

    tensors: dict[str, torch.Tensor] = {
        "public.targets": stable.targets,
        "public.target_mask": stable.target_mask,
        "public.patient_folds": stable.patient_folds,
        "private.candidate_mask": V11_CANDIDATE_MASK.clone(),
        **state_tensors,
    }
    public_metrics: dict[str, object] = {}
    for objective in OBJECTIVES:
        d_probability = public_d[objective]
        ensemble = 0.5 * h_public + 0.5 * d_probability
        private_fold = 0.5 * private_h_fold + 0.5 * private_d_fold[objective]
        private_probability = private_fold.mean(dim=1).contiguous()
        tensors[f"public.{objective}.D_probability"] = d_probability.contiguous()
        tensors[f"public.{objective}.ensemble_probability"] = ensemble.contiguous()
        tensors[f"private.{objective}.D_fold_probability"] = private_d_fold[
            objective
        ].contiguous()
        tensors[f"private.{objective}.ensemble_probability"] = private_probability
        public_metrics[objective] = {
            "D_only": _evaluate(
                torch.log(d_probability.clamp_min(1e-12)),
                stable.targets,
                stable.target_mask,
            ),
            "H_D_equal": _evaluate(
                torch.log(ensemble.clamp_min(1e-12)),
                stable.targets,
                stable.target_mask,
            ),
        }
    public_parity = float(
        (
            tensors["public.set_mass_frozen_v28.ensemble_probability"]
            - current_public_ensemble
        )
        .abs()
        .max()
    )
    private_parity = float(
        (
            tensors["private.set_mass_frozen_v28.ensemble_probability"]
            - current_private_ensemble
        )
        .abs()
        .max()
    )
    if public_parity != 0.0 or private_parity > 1e-7:
        raise RuntimeError("frozen v29 set-mass parity failed")

    manifest: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_public_oof_and_target_blind_private_objective_sensitivity",
        "objectives": list(OBJECTIVES),
        "fixed_components": {
            "foundation": "official pretrained LaBraM block-9 frozen",
            "foundation_trainable_parameters": 0,
            "direct_head_parameters": 206,
            "outer_patient_folds": 5,
            "patient_pooling": "equal seizure mean",
            "prior": "fold-local Jeffreys reference prior",
            "optimizer_epochs_lr_weight_decay": [
                "AdamW",
                v28.EPOCHS,
                v28.LEARNING_RATE,
                v28.WEIGHT_DECAY,
            ],
            "ensemble": "0.5 H probability + 0.5 D probability",
        },
        "objective_semantics": {
            "set_mass_frozen_v28": "negative log probability mass assigned to any observed positive",
            "uniform_positive_soft_ce": "uniform soft target over every observed positive within the candidate simplex",
            "patient_balanced_bce": "equal per-patient weight to positive and benchmark-complement channel BCE",
        },
        "public": {
            "patients": stable_count,
            "events": len(stable_features),
            "auxiliary_patients": len(auxiliary.patient_ids),
            "auxiliary_events": len(auxiliary_features),
            "metrics": public_metrics,
        },
        "private": {
            "target_blind_events": len(private_events),
            "patients": len({str(event["patient_id"]) for event in private_events}),
            "events": private_events,
        },
        "fit_receipts": fit_receipts,
        "parity": {
            "public_set_mass_v29_max_abs": public_parity,
            "private_set_mass_v29_max_abs": private_parity,
        },
        "tensor_file": "objective_predictions.safetensors",
        "access_receipt": {
            "public_targets_loaded_for_oof_training_and_evaluation": True,
            "private_phase_and_frozen_H_loaded": True,
            "private_significant_or_spread_reference_loaded": False,
            "private_used_for_objective_model_threshold_or_report_selection": False,
            "foundation_training_performed": False,
        },
        "interpretation_boundary": {
            "direct_D_branch_objective_sensitivity_not_full_H_objective_ablation": True,
            "set_mass_v29_remains_frozen_primary": True,
            "private_is_fresh_validation": False,
            "alternative_may_replace_v29_from_this_audit": False,
            "balanced_bce_complement_is_clinician_confirmed_negative": False,
        },
        "elapsed_sec": time.monotonic() - started,
    }
    return manifest, tensors


def publish(
    output: Path, manifest: Mapping[str, object], tensors: Mapping[str, torch.Tensor]
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        save_file(dict(tensors), str(staging / "objective_predictions.safetensors"))
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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
    parser.add_argument("--v28-directory", type=Path, default=DEFAULT_V28)
    parser.add_argument("--private-phase-directory", type=Path, default=DEFAULT_PRIVATE_PHASE)
    parser.add_argument("--private-v29-directory", type=Path, default=DEFAULT_PRIVATE_V29)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = build_parser().parse_args(argv)
    manifest, tensors = run(args)
    output = publish(args.output_directory, manifest, tensors)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": manifest["status"],
                "objectives": manifest["objectives"],
                "private_reference_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
