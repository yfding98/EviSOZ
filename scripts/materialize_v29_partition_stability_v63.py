#!/usr/bin/env python3
"""Materialize full H/D patient-partition stability without private reference.

Five label-independent balanced patient five-fold assignments replace only the
formal public outer partition.  H nested-L2 selection, D training protocol,
foundation carriers, auxiliary roster, set-mass objective, C18 mask and fixed
0.5/0.5 fusion are replayed.  Every partition is retained for audit; none may
replace or be ensembled with formal v29.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
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

import scripts.run_labram_fine_temporal_nested_oof_v11 as shared_v11  # noqa: E402
import scripts.run_labram_masked_variable_auxiliary_oof_v17 as v17  # noqa: E402
import scripts.run_labram_rank1_direct_token_oof_v28 as v28  # noqa: E402
from scripts.materialize_v29_direct_objective_sensitivity_v53 import (  # noqa: E402
    _load_private_phase,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _load_reasoner_from_fit,
)
from scripts.materialize_v29_seed_stability_v58 import (  # noqa: E402
    _rank_correlation,
    _topk_jaccard,
)
from src.soz.metrics import deepsoz_style_top1_metrics  # noqa: E402
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    fit_fold_transform,
    jeffreys_reference_prior_logits,
)


SCHEMA = "trustworthy_soz_v29_H_D_patient_partition_stability_v63"
PROTOCOL = ROOT / "research/02_method/post_open_fixed_audit_extensions_v60_20260816_zh.md"
DEFAULT_PUBLIC_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE_V29 = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_PRIVATE_PHASE = ROOT / "outputs/private_target_blind_rank1_phase_v29_20260815"
DEFAULT_PRIVATE_H = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_partition_stability_v63_20260816"
PARTITIONS = 5
BASE_SEED = 20260863


def _balanced_assignment(
    rows: Sequence[int],
    event_counts: torch.Tensor,
    *,
    groups: int,
    seed: int,
) -> dict[int, int]:
    selected = tuple(int(value) for value in rows)
    if len(selected) < groups or len(set(selected)) != len(selected):
        raise ValueError("balanced assignment rows must be unique and cover every group")
    rng = random.Random(seed)
    jitter = {row: rng.random() for row in selected}
    ordered = sorted(
        selected, key=lambda row: (-int(event_counts[row]), jitter[row])
    )
    group_events = [0] * groups
    group_patients = [0] * groups
    assignment: dict[int, int] = {}
    for row in ordered:
        tie = [rng.random() for _ in range(groups)]
        group = min(
            range(groups),
            key=lambda value: (
                group_events[value],
                group_patients[value],
                tie[value],
            ),
        )
        assignment[row] = group
        group_events[group] += int(event_counts[row])
        group_patients[group] += 1
    if set(assignment.values()) != set(range(groups)):
        raise RuntimeError("balanced assignment left an empty group")
    return assignment


def _probability(logits: torch.Tensor) -> torch.Tensor:
    result = torch.softmax(
        logits.masked_fill(~V11_CANDIDATE_MASK.view(1, -1), -torch.inf), dim=1
    )
    if not torch.isfinite(result).all():
        raise RuntimeError("partition audit probability is non-finite")
    return result


def _metric_row(
    probability: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    logits = torch.log(probability.clamp_min(1e-12))
    metrics = _evaluate(logits, targets, mask)
    n2 = asdict(
        deepsoz_style_top1_metrics(
            logits, targets, mask, max_positive_for_neighbor=2
        )
    )
    return {
        "strict": float(metrics["top1"]["strict_accuracy"]),
        "official_N2": float(n2["relaxed_accuracy"]),
        "official_N4": float(metrics["top1"]["relaxed_accuracy"]),
        "macro_average_precision": float(
            metrics["ranking"]["macro_average_precision"]
        ),
        "mean_reciprocal_rank": float(metrics["ranking"]["mean_reciprocal_rank"]),
    }


def _stability(
    candidate: torch.Tensor, formal: torch.Tensor
) -> dict[str, float]:
    return {
        "top1_agreement": float(
            (candidate.argmax(dim=1) == formal.argmax(dim=1)).float().mean()
        ),
        "top3_jaccard": _topk_jaccard(candidate, formal, k=3),
        "mean_rank_correlation": _rank_correlation(candidate, formal),
        "mean_probability_L1": float((candidate - formal).abs().sum(dim=1).mean()),
    }


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    started = time.monotonic()
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    stable = v17._load_stable_development(args)
    auxiliary = v17._load_auxiliary_targets(args, stable)
    auxiliary_prefix = v17._load_auxiliary_cache(
        args.aux_prefix_directory,
        expected_manifest_sha256=args.expected_aux_prefix_manifest_sha256,
        expected_tensor_sha256=args.expected_aux_prefix_tensor_sha256,
        schema=v17.PREFIX_SCHEMA,
        expected_keys=v17.PREFIX_TENSOR_KEYS,
        primary_key="prefix_tokens",
        auxiliary=auxiliary,
        label="auxiliary LaBraM prefix for v63 partition stability",
    )
    stable_prefix, stable_event_patient_index = v28._load_stable_prefix(args, stable)
    stable_phase = v28.extract_rank1_phase_features(stable_prefix)
    del stable_prefix
    auxiliary_phase = v28.extract_rank1_phase_features(
        auxiliary_prefix.tensors["prefix_tokens"].float()
    )
    stable_count = len(stable.patient_ids)
    combined = v28.PatientBag(
        phase_features=torch.cat((stable_phase, auxiliary_phase), dim=0),
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

    private_evidence = load_file(
        str((args.private_h_directory / "evidence.safetensors").resolve(strict=True))
    )
    private_h = private_evidence["h_event"].float()
    private_fine = private_evidence["fine_event"].float()
    if tuple(private_h.shape) != (88, 19, 600) or tuple(private_fine.shape) != (
        88,
        19,
        20,
    ):
        raise ValueError("private H/fine carrier shape changed")
    private_events, private_phase = _load_private_phase(args.private_phase_directory)
    if len(private_events) != 88:
        raise ValueError("private D phase roster changed")

    public_v29 = load_file(
        str((args.public_v29_directory / "oof_predictions.safetensors").resolve(strict=True))
    )
    formal_public = public_v29["oof.portable_equal_ensemble_probability"].float()
    private_v29 = load_file(
        str((args.private_v29_directory / "predictions.safetensors").resolve(strict=True))
    )
    formal_private = private_v29["private_portable_equal_probability"].float()
    if not torch.equal(public_v29["targets"], stable.targets) or not torch.equal(
        public_v29["target_mask"].bool(), stable.target_mask
    ):
        raise ValueError("stable target differs from frozen v29")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    public_rows: list[torch.Tensor] = []
    private_rows: list[torch.Tensor] = []
    assignment_rows: list[torch.Tensor] = []
    public_metrics: list[dict[str, float]] = []
    receipts: list[dict[str, object]] = []
    for partition in range(PARTITIONS):
        assignment = _balanced_assignment(
            range(stable_count),
            stable.event_counts,
            groups=5,
            seed=BASE_SEED + 10_000 * partition,
        )
        fold_tensor = torch.tensor(
            [assignment[index] for index in range(stable_count)], dtype=torch.long
        )
        oof_h = torch.full((stable_count, 19), torch.nan)
        oof_d = torch.full((stable_count, 19), torch.nan)
        private_fold_probability: list[torch.Tensor] = []
        fold_receipts: list[dict[str, object]] = []
        for fold in range(5):
            train = torch.nonzero(fold_tensor != fold, as_tuple=False).flatten()
            held = torch.nonzero(fold_tensor == fold, as_tuple=False).flatten()
            transform = fit_fold_transform(
                stable.h_patient, stable.fine_patient, train.tolist()
            )
            transformed = transform.apply(stable.h_patient, stable.fine_patient)
            inner_assignment = _balanced_assignment(
                train.tolist(),
                stable.event_counts,
                groups=4,
                seed=BASE_SEED + 10_000 * partition + 100 * fold + 7,
            )
            contexts: list[shared_v11._InnerContext] = []
            for inner in range(4):
                inner_held = tuple(
                    index for index in train.tolist() if inner_assignment[index] == inner
                )
                inner_train = tuple(
                    index for index in train.tolist() if inner_assignment[index] != inner
                )
                inner_transform = fit_fold_transform(
                    stable.h_patient, stable.fine_patient, inner_train
                )
                contexts.append(
                    shared_v11._InnerContext(
                        fold=inner,
                        train_indices=inner_train,
                        held_indices=inner_held,
                        transformed=inner_transform.apply(
                            stable.h_patient, stable.fine_patient
                        ),
                    )
                )
            h_l2, h_selection = shared_v11._select_l2(
                contexts,
                stable.targets,
                stable.target_mask,
                use_h=True,
                use_fine=False,
            )
            h_fit = shared_v11._fit_reasoner(
                transformed,
                stable.targets,
                stable.target_mask,
                train.tolist(),
                use_h=True,
                use_fine=False,
                l2=h_l2,
            )
            oof_h.index_copy_(
                0, held, _probability(h_fit.logits.index_select(0, held))
            )
            h_model = _load_reasoner_from_fit(h_fit.state)
            with torch.no_grad():
                private_h_probability = _probability(
                    h_model(transform.apply(private_h, private_fine)).logits
                )

            auxiliary_train = tuple(
                stable_count + index
                for index in torch.nonzero(
                    auxiliary.outer_folds != fold, as_tuple=False
                ).flatten()
                .tolist()
            )
            train_bag = v28._subset_bag(
                combined, tuple(train.tolist()) + auxiliary_train
            )
            prior = jeffreys_reference_prior_logits(
                stable.targets.index_select(0, train),
                stable.target_mask.index_select(0, train),
            )
            d_model, d_fit = v28._fit(
                train_bag,
                prior,
                seed=v28.BASE_SEED + 1_000 * fold,
                device=device,
            )
            held_bag = v28._subset_bag(combined, held.tolist())
            oof_d.index_copy_(
                0, held, _probability(v28._predict(d_model, held_bag, device))
            )
            with torch.no_grad():
                private_d_probability = _probability(
                    d_model(private_phase.to(device)).detach().cpu()
                )
            private_fold_probability.append(
                0.5 * private_h_probability + 0.5 * private_d_probability
            )
            fold_receipts.append(
                {
                    "fold": fold,
                    "train_patients": len(train),
                    "held_patients": len(held),
                    "train_events": int(stable.event_counts.index_select(0, train).sum()),
                    "held_events": int(stable.event_counts.index_select(0, held).sum()),
                    "H_selected_l2": h_l2,
                    "H_inner_selection": h_selection,
                    "D_fit": d_fit,
                }
            )
        if not torch.isfinite(oof_h).all() or not torch.isfinite(oof_d).all():
            raise RuntimeError("partition public OOF prediction is incomplete")
        public_probability = (0.5 * oof_h + 0.5 * oof_d).contiguous()
        private_probability = torch.stack(private_fold_probability).mean(dim=0).contiguous()
        metric = _metric_row(public_probability, stable.targets, stable.target_mask)
        public_rows.append(public_probability)
        private_rows.append(private_probability)
        assignment_rows.append(fold_tensor)
        public_metrics.append(metric)
        receipts.append(
            {
                "partition": partition,
                "fold_patient_counts": [int((fold_tensor == fold).sum()) for fold in range(5)],
                "fold_event_counts": [
                    int(stable.event_counts[fold_tensor == fold].sum()) for fold in range(5)
                ],
                "folds": fold_receipts,
                "public_metrics": metric,
                "public_vs_formal_prediction_stability": _stability(
                    public_probability, formal_public
                ),
                "private_all88_vs_formal_prediction_stability": _stability(
                    private_probability, formal_private
                ),
            }
        )
        print(
            json.dumps(
                {
                    "partition": partition,
                    **metric,
                    "elapsed_sec": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_public_OOF_private_reference_isolated_H_D_partition_stability",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "partitions": PARTITIONS,
        "formal_v29_public_metrics": _metric_row(
            formal_public, stable.targets, stable.target_mask
        ),
        "alternative_public_metrics": public_metrics,
        "partition_receipts": receipts,
        "tensor_file": "partition_predictions.safetensors",
        "fixed_components": {
            "foundation": "official pretrained LaBraM block-9 frozen",
            "H_nested_l2_candidates": list(shared_v11.L2_CANDIDATES),
            "D_auxiliary_roster": "fixed 9 patients / 182 events",
            "D_optimizer_epochs_and_seed_policy": "formal v28",
            "loss": "patient-equal positive-set probability-mass NLL",
            "fusion": "fixed 0.5 H + 0.5 D probability",
            "candidate_space": "C18 with PZ masked",
        },
        "access_receipt": {
            "public_targets_loaded_for_nested_training_and_OOF_evaluation": True,
            "private_H_D_carriers_loaded_for_reference_isolated_inference": True,
            "private_significant_or_spread_reference_loaded": False,
            "formal_v29_changed_selected_or_ensembled": False,
            "backbone_loss_window_anchor_fusion_or_threshold_changed": False,
        },
        "interpretation_boundary": {
            "partition_assignment_uses_SOZ_labels": False,
            "foundation_pretraining_stochasticity_audited": False,
            "private_is_fresh_or_target_blind_validation": False,
            "best_partition_may_replace_or_join_v29": False,
            "allowed_claim": "H/D adaptation sensitivity to label-independent patient partitions",
        },
        "elapsed_sec": time.monotonic() - started,
    }
    tensors = {
        "public.alternative_partition_probability": torch.stack(public_rows),
        "private.alternative_partition_probability": torch.stack(private_rows),
        "public.alternative_patient_folds": torch.stack(assignment_rows),
        "public.formal_v29_probability": formal_public,
        "private.formal_v29_probability": formal_private,
        "public.targets": stable.targets,
        "public.target_mask": stable.target_mask,
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
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
        save_file(dict(tensors), str(staging / "partition_predictions.safetensors"))
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
    parser.add_argument("--private-v29-directory", type=Path, default=DEFAULT_PRIVATE_V29)
    parser.add_argument("--private-phase-directory", type=Path, default=DEFAULT_PRIVATE_PHASE)
    parser.add_argument("--private-h-directory", type=Path, default=DEFAULT_PRIVATE_H)
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
                "partitions": result["partitions"],
                "private_reference_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
