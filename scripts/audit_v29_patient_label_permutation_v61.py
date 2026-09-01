#!/usr/bin/env python3
"""Patient-level label-permutation falsification audit for frozen-v29 H/D heads.

Only labels inside each public outer-training fold are permuted.  Held-out
patient references never enter the corresponding fit.  The frozen LaBraM
representations, folds, H regularization, D optimizer, auxiliary roster,
candidate mask, set-mass loss, and 0.5/0.5 fusion remain fixed.  No private EEG
or private reference is loaded.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_labram_fine_temporal_nested_oof_v11 as shared_v11  # noqa: E402
import scripts.run_labram_masked_variable_auxiliary_oof_v17 as v17  # noqa: E402
import scripts.run_labram_rank1_direct_token_oof_v28 as v28  # noqa: E402
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import _evaluate  # noqa: E402
from src.soz.metrics import deepsoz_style_top1_metrics  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    fit_fold_transform,
    jeffreys_reference_prior_logits,
)


SCHEMA = "trustworthy_soz_v29_patient_label_permutation_v61"
PROTOCOL = ROOT / "docs/method/reference/post_open_fixed_audit_extensions_v60_20260816_zh.md"
DEFAULT_PUBLIC_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_H_REPLAY = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_patient_label_permutation_v61_20260816"
REPETITIONS = 20
BASE_SEED = 20260861


def _reference_stratum(targets: torch.Tensor, mask: torch.Tensor) -> tuple[str, str]:
    positives = torch.nonzero((targets == 1) & mask, as_tuple=False).flatten().tolist()
    if not positives:
        raise ValueError("permutation stratum requires a positive reference")
    size = "1-2" if len(positives) <= 2 else "3-4" if len(positives) <= 4 else ">=5"
    sides: set[str] = set()
    for index in positives:
        channel = STANDARD_19[int(index)]
        if channel in ("FZ", "CZ", "PZ"):
            sides.add("midline")
        elif channel[-1] in ("1", "3", "5", "7"):
            sides.add("left")
        elif channel[-1] in ("2", "4", "6", "8"):
            sides.add("right")
        else:
            raise ValueError(f"cannot assign reference laterality: {channel}")
    laterality = next(iter(sides)) if len(sides) == 1 else "mixed"
    return size, laterality


def _permutation_source(
    targets: torch.Tensor,
    mask: torch.Tensor,
    rows: torch.Tensor,
    *,
    generator: torch.Generator,
    mode: str,
) -> torch.Tensor:
    if mode == "unconditional":
        return rows.index_select(0, torch.randperm(len(rows), generator=generator))
    if mode != "cardinality_laterality_stratified":
        raise ValueError(f"unknown permutation mode: {mode}")
    source = rows.clone()
    groups: dict[tuple[str, str], list[int]] = {}
    for position, row in enumerate(rows.tolist()):
        groups.setdefault(_reference_stratum(targets[row], mask[row]), []).append(position)
    for positions in groups.values():
        position_tensor = torch.tensor(positions, dtype=torch.long)
        shuffled = position_tensor.index_select(
            0, torch.randperm(len(position_tensor), generator=generator)
        )
        source.index_copy_(
            0,
            position_tensor,
            rows.index_select(0, shuffled),
        )
    return source


def _probability(logits: torch.Tensor) -> torch.Tensor:
    result = torch.softmax(
        logits.masked_fill(~V11_CANDIDATE_MASK.view(1, -1), -torch.inf), dim=1
    )
    if not torch.isfinite(result).all():
        raise RuntimeError("permutation probability is non-finite")
    return result


def _permuted_rows(
    targets: torch.Tensor,
    mask: torch.Tensor,
    rows: torch.Tensor,
    *,
    generator: torch.Generator,
    mode: str = "unconditional",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = _permutation_source(
        targets, mask, rows, generator=generator, mode=mode
    )
    permuted_targets = targets.clone()
    permuted_mask = mask.clone()
    permuted_targets.index_copy_(0, rows, targets.index_select(0, source))
    permuted_mask.index_copy_(0, rows, mask.index_select(0, source))
    return permuted_targets, permuted_mask, source


def _permuted_bag(
    bag: v28.PatientBag,
    *,
    generator: torch.Generator,
    mode: str = "unconditional",
) -> tuple[v28.PatientBag, torch.Tensor]:
    rows = torch.arange(len(bag.patient_ids), dtype=torch.long)
    source = _permutation_source(
        bag.targets,
        bag.target_mask,
        rows,
        generator=generator,
        mode=mode,
    )
    return (
        v28.PatientBag(
            phase_features=bag.phase_features,
            event_patient_index=bag.event_patient_index,
            targets=bag.targets.index_select(0, source),
            target_mask=bag.target_mask.index_select(0, source),
            patient_ids=bag.patient_ids,
        ),
        source,
    )


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


def _null_summary(rows: Sequence[Mapping[str, float]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in rows[0]:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "range": [float(values.min()), float(values.max())],
            "quantile_05_50_95": [
                float(value) for value in np.quantile(values, (0.05, 0.5, 0.95))
            ],
        }
    return result


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
        label="auxiliary LaBraM prefix for v61 permutation audit",
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

    public_v29 = load_file(
        str((args.public_v29_directory / "oof_predictions.safetensors").resolve(strict=True))
    )
    formal_probability = public_v29["oof.portable_equal_ensemble_probability"].float()
    if not torch.equal(public_v29["targets"], stable.targets) or not torch.equal(
        public_v29["target_mask"].bool(), stable.target_mask
    ):
        raise ValueError("stable target differs from frozen v29")
    h_manifest = json.loads(
        (args.h_replay_directory / "manifest.json").resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    h_l2 = {
        int(row["outer_fold"]): float(
            row["arms"]["frozen_labram_only"]["selected_l2"]
        )
        for row in h_manifest["fold_results"]
    }
    if set(h_l2) != set(range(5)):
        raise ValueError("formal H fold regularization is incomplete")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    repetitions = int(args.repetitions)
    if repetitions < 20:
        raise ValueError("formal permutation audit requires at least 20 repetitions")
    probabilities: list[torch.Tensor] = []
    metrics: list[dict[str, float]] = []
    receipts: list[dict[str, object]] = []
    for repetition in range(repetitions):
        oof_h = torch.full((stable_count, 19), torch.nan)
        oof_d = torch.full((stable_count, 19), torch.nan)
        fold_receipts: list[dict[str, object]] = []
        for fold in range(5):
            stable_train = torch.nonzero(
                stable.patient_folds != fold, as_tuple=False
            ).flatten()
            stable_held = torch.nonzero(
                stable.patient_folds == fold, as_tuple=False
            ).flatten()
            seed = BASE_SEED + 100_000 * repetition + 1_000 * fold
            h_generator = torch.Generator().manual_seed(seed + 11)
            perm_targets, perm_mask, h_source = _permuted_rows(
                stable.targets,
                stable.target_mask,
                stable_train,
                generator=h_generator,
                mode=args.permutation_mode,
            )
            transform = fit_fold_transform(
                stable.h_patient, stable.fine_patient, stable_train.tolist()
            )
            transformed = transform.apply(stable.h_patient, stable.fine_patient)
            h_fit = shared_v11._fit_reasoner(
                transformed,
                perm_targets,
                perm_mask,
                stable_train.tolist(),
                use_h=True,
                use_fine=False,
                l2=h_l2[fold],
            )
            oof_h.index_copy_(
                0,
                stable_held,
                _probability(h_fit.logits.index_select(0, stable_held)),
            )

            auxiliary_train = tuple(
                stable_count + index
                for index in torch.nonzero(
                    auxiliary.outer_folds != fold, as_tuple=False
                ).flatten()
                .tolist()
            )
            combined_train_indices = tuple(stable_train.tolist()) + auxiliary_train
            train_bag = v28._subset_bag(combined, combined_train_indices)
            d_generator = torch.Generator().manual_seed(seed + 29)
            permuted_train_bag, d_order = _permuted_bag(
                train_bag,
                generator=d_generator,
                mode=args.permutation_mode,
            )
            prior = jeffreys_reference_prior_logits(
                perm_targets.index_select(0, stable_train),
                perm_mask.index_select(0, stable_train),
            )
            d_model, d_receipt = v28._fit(
                permuted_train_bag,
                prior,
                seed=v28.BASE_SEED + 100_000 * repetition + 1_000 * fold,
                device=device,
            )
            held_bag = v28._subset_bag(combined, stable_held.tolist())
            oof_d.index_copy_(
                0, stable_held, _probability(v28._predict(d_model, held_bag, device))
            )
            fold_receipts.append(
                {
                    "fold": fold,
                    "stable_train_patients": len(stable_train),
                    "stable_held_patients": len(stable_held),
                    "auxiliary_train_patients": len(auxiliary_train),
                    "H_l2": h_l2[fold],
                    "H_permutation_is_bijective": len(set(h_source.tolist()))
                    == len(stable_train),
                    "D_permutation_is_bijective": len(set(d_order.tolist()))
                    == len(train_bag.patient_ids),
                    "D_fit": d_receipt,
                }
            )
        if not torch.isfinite(oof_h).all() or not torch.isfinite(oof_d).all():
            raise RuntimeError("permutation OOF prediction is incomplete")
        probability = (0.5 * oof_h + 0.5 * oof_d).contiguous()
        probabilities.append(probability)
        metric = _metric_row(probability, stable.targets, stable.target_mask)
        metrics.append(metric)
        receipts.append({"repetition": repetition, "folds": fold_receipts})
        print(
            json.dumps(
                {
                    "repetition": repetition,
                    **metric,
                    "elapsed_sec": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    formal_metrics = _metric_row(
        formal_probability, stable.targets, stable.target_mask
    )
    prevalence_metrics = _metric_row(
        _probability(
            load_file(
                str(
                    (args.h_replay_directory / "oof_predictions.safetensors").resolve(
                        strict=True
                    )
                )
            )["oof.prevalence_only"].float()
        ),
        stable.targets,
        stable.target_mask,
    )
    empirical = {
        key: (1 + sum(float(row[key]) >= float(formal_metrics[key]) for row in metrics))
        / (1 + len(metrics))
        for key in formal_metrics
    }
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_public_patient_label_permutation_falsification",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "repetitions": repetitions,
        "permutation_unit": (
            "complete_patient_positive_set_within_outer_training_fold"
            if args.permutation_mode == "unconditional"
            else "complete_patient_positive_set_within_outer_training_fold_and_cardinality_laterality_stratum"
        ),
        "permutation_mode": args.permutation_mode,
        "formal_v29_metrics": formal_metrics,
        "prevalence_only_metrics": prevalence_metrics,
        "permutation_null_summary": _null_summary(metrics),
        "permutation_metrics": metrics,
        "descriptive_empirical_tail_probability_null_ge_formal": empirical,
        "fit_receipts": receipts,
        "tensor_file": "permutation_predictions.safetensors",
        "access_receipt": {
            "public_targets_loaded_for_within_training_fold_permutation": True,
            "held_reference_used_in_corresponding_training_fold": False,
            "private_EEG_loaded": False,
            "private_significant_or_spread_reference_loaded": False,
            "foundation_training_or_forward_performed": False,
            "formal_v29_changed_selected_or_ensembled": False,
        },
        "interpretation_boundary": {
            "public_is_consumed_adaptive_development": True,
            "tail_probability_is_confirmatory_p_value": False,
            "permutation_proves_foundation_pretraining_clean": False,
            "permutation_can_detect_all_forms_of_leakage": False,
            "allowed_claim": (
                "patient-label correspondence is required for the audited H/D "
                "head pipeline to retain the formal public OOF performance"
            ),
        },
        "elapsed_sec": time.monotonic() - started,
    }
    tensors = {
        "permutation.ensemble_probability": torch.stack(probabilities),
        "formal_v29_probability": formal_probability,
        "targets": stable.targets,
        "target_mask": stable.target_mask,
        "patient_folds": stable.patient_folds,
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
        save_file(dict(tensors), str(staging / "permutation_predictions.safetensors"))
        (staging / "result.json").write_text(
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
    parser.add_argument("--h-replay-directory", type=Path, default=DEFAULT_H_REPLAY)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument(
        "--permutation-mode",
        choices=("unconditional", "cardinality_laterality_stratified"),
        default="unconditional",
    )
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
                "repetitions": result["repetitions"],
                "private_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
