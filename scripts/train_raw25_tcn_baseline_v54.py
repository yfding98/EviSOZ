#!/usr/bin/env python3
"""Train the fixed public-OOF raw25 TCN and predict private target blindly."""

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

import scripts.run_labram_identity_recovery_closed_replay_v16 as v16  # noqa: E402
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import _evaluate  # noqa: E402
from src.soz.baseline.raw25_tcn import Raw25ChannelTCN  # noqa: E402
from src.soz.data.public_development_union_identity_v12 import (  # noqa: E402
    load_public_development_union_identity_v12,
)
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    jeffreys_reference_prior_logits,
    positive_set_mass_loss,
)


SCHEMA = "trustworthy_soz_raw25_tcn_baseline_v54"
DEFAULT_WAVEFORM = ROOT / "outputs/trustworthy_soz_public_private_raw25_waveforms_v54_20260816"
DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_raw25_tcn_baseline_v54_20260816"
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_PATIENTS = 16
BASE_SEED = 20260854


def _seeded_model(prior: torch.Tensor, seed: int) -> Raw25ChannelTCN:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = Raw25ChannelTCN(prior)
    if model.n_trainable_parameters != 1_425:
        raise RuntimeError("raw25 TCN parameter count drifted")
    return model


def _fit_fold(
    train_waveform: torch.Tensor,
    train_targets: torch.Tensor,
    train_mask: torch.Tensor,
    prior: torch.Tensor,
    *,
    seed: int,
) -> tuple[Raw25ChannelTCN, float, float, float]:
    scale = float(train_waveform.square().mean().sqrt())
    if not (scale > 0 and torch.isfinite(torch.tensor(scale))):
        raise RuntimeError("raw25 training scale is invalid")
    waveform = (train_waveform / scale).contiguous()
    model = _seeded_model(prior, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(seed + 1)
    first = None
    final = None
    for _ in range(EPOCHS):
        permutation = torch.randperm(len(waveform), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(waveform), BATCH_PATIENTS):
            rows = permutation[start : start + BATCH_PATIENTS]
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveform.index_select(0, rows))
            loss = positive_set_mass_loss(
                logits,
                train_targets.index_select(0, rows),
                train_mask.index_select(0, rows),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(rows)
        epoch_loss /= len(waveform)
        first = epoch_loss if first is None else first
        final = epoch_loss
    assert first is not None and final is not None
    model.eval().requires_grad_(False)
    return model, scale, first, final


@torch.inference_mode()
def _predict(model: Raw25ChannelTCN, waveform: torch.Tensor, scale: float) -> torch.Tensor:
    rows = []
    for start in range(0, len(waveform), 16):
        rows.append(model((waveform[start : start + 16] / scale).contiguous()).cpu())
    result = torch.cat(rows).contiguous()
    if not torch.isfinite(result).all():
        raise RuntimeError("raw25 TCN prediction is non-finite")
    return result


def train(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    started = time.monotonic()
    waveform_manifest = json.loads(
        (args.waveform / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    if waveform_manifest.get("status") != "completed_target_free_public_private_raw25_waveforms":
        raise ValueError("formal target-free raw25 waveform materialization is missing")
    access = waveform_manifest.get("access_receipt", {})
    if access.get("public_SOZ_target_loaded") is not False or access.get(
        "private_significant_or_spread_reference_loaded"
    ) is not False:
        raise ValueError("raw25 waveform target firewall failed")
    waveforms = load_file(
        str((args.waveform / "raw25_waveforms.safetensors").resolve(strict=True))
    )
    public_all = waveforms["public.patient_waveform"].float()
    private = waveforms["private.event_waveform"].float()
    union = load_public_development_union_identity_v12(args.union)
    roster, _ = v16._load_primary_roster(
        union,
        target_directory=args.target_directory,
        source_csv=args.source_csv,
        split_csv=args.split_csv,
    )
    public = public_all.index_select(0, roster.selected_union_indices)
    if tuple(public.shape) != (102, 19, 1_500) or tuple(private.shape) != (88, 19, 1_500):
        raise ValueError("raw25 formal waveform shape changed")
    oof = torch.full((102, 19), torch.nan)
    private_fold_rows: list[torch.Tensor] = []
    states: dict[str, torch.Tensor] = {}
    folds: list[dict[str, object]] = []
    for fold in range(5):
        train_rows = torch.nonzero(roster.patient_folds != fold, as_tuple=False).flatten()
        held_rows = torch.nonzero(roster.patient_folds == fold, as_tuple=False).flatten()
        prior = jeffreys_reference_prior_logits(
            roster.targets.index_select(0, train_rows),
            roster.target_mask.index_select(0, train_rows),
        )
        model, scale, first, final = _fit_fold(
            public.index_select(0, train_rows),
            roster.targets.index_select(0, train_rows),
            roster.target_mask.index_select(0, train_rows),
            prior,
            seed=BASE_SEED + 1000 * fold,
        )
        held_logits = _predict(model, public.index_select(0, held_rows), scale)
        oof.index_copy_(0, held_rows, held_logits)
        private_logits = _predict(model, private, scale)
        private_fold_rows.append(
            torch.softmax(
                private_logits.masked_fill(~V11_CANDIDATE_MASK, -torch.inf), dim=1
            )
        )
        for name, value in model.state_dict().items():
            states[f"fold{fold}.{name}"] = value.detach().cpu().contiguous()
        states[f"fold{fold}.input_rms_scale"] = torch.tensor(scale)
        fold_metrics = _evaluate(
            held_logits,
            roster.targets.index_select(0, held_rows),
            roster.target_mask.index_select(0, held_rows),
        )
        folds.append(
            {
                "fold": fold,
                "train_patients": len(train_rows),
                "held_patients": len(held_rows),
                "input_rms_scale_volts": scale,
                "first_epoch_loss": first,
                "final_epoch_loss": final,
                "held_metrics": fold_metrics,
            }
        )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "strict": fold_metrics["top1"]["strict_accuracy"],
                    "n4": fold_metrics["top1"]["relaxed_accuracy"],
                    "elapsed_sec": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not torch.isfinite(oof).all():
        raise RuntimeError("raw25 public OOF is incomplete")
    private_fold = torch.stack(private_fold_rows, dim=1).contiguous()
    private_probability = private_fold.mean(dim=1).contiguous()
    metrics = _evaluate(oof, roster.targets, roster.target_mask)
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_fixed_public_oof_and_target_blind_private_raw25_tcn",
        "architecture": {
            "input": "CAR19 60s fixed-polyphase-resampled 25Hz [19,1500]",
            "public_patient_pooling": "equal raw-waveform mean before TCN",
            "shared_channel_local_TCN": [
                "Conv1d 1->8 kernel25 stride5 GELU",
                "Conv1d 8->16 kernel9 stride5 GELU",
                "pre/early/late mean pool",
                "Linear48->1 plus fold-local Jeffreys prior",
            ],
            "temporal_grid_after_TCN": "60 one-second positions",
            "trainable_parameters": 1_425,
            "foundation_or_pretrained_parameters": 0,
        },
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_patients": BATCH_PATIENTS,
            "seed": BASE_SEED,
            "hyperparameter_candidates": 1,
            "loss": "same patient-equal positive-set probability-mass NLL",
        },
        "public": {
            "patients": 102,
            "events_before_patient_waveform_pooling": int(
                waveforms["public.patient_event_count"]
                .index_select(0, roster.selected_union_indices)
                .sum()
            ),
            "metrics": metrics,
            "folds": folds,
        },
        "private": {
            "target_blind_events": 88,
            "patients": len(
                {
                    str(event["patient_id"])
                    for event in waveform_manifest["private"]["events"]
                }
            ),
            "events": waveform_manifest["private"]["events"],
        },
        "tensor_file": "raw25_tcn_predictions.safetensors",
        "access_receipt": {
            "public_targets_loaded_for_training_and_oof_evaluation": True,
            "private_waveforms_loaded_for_target_blind_inference": True,
            "private_significant_or_spread_reference_loaded": False,
            "private_used_for_training_model_threshold_or_report_selection": False,
            "foundation_training_or_inference_performed": False,
        },
        "interpretation_boundary": {
            "canonical_EEGNet_or_ShallowConvNet_exact_reproduction": False,
            "low_capacity_raw_waveform_neural_baseline": True,
            "25Hz_input_excludes_higher_frequency_information": True,
            "public_raw_waveform_patient_averaging_may_smear_events": True,
            "baseline_may_replace_v29_from_private_results": False,
            "private_is_fresh_validation": False,
        },
        "elapsed_sec": time.monotonic() - started,
    }
    tensors = {
        "public.oof_logits": oof,
        "public.targets": roster.targets,
        "public.target_mask": roster.target_mask,
        "public.patient_folds": roster.patient_folds,
        "private.fold_probability": private_fold,
        "private.probability": private_probability,
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
        **states,
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
        save_file(dict(tensors), str(staging / "raw25_tcn_predictions.safetensors"))
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
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--waveform", type=Path, default=DEFAULT_WAVEFORM)
    parser.add_argument("--union", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--target-directory", type=Path, default=v16.DEFAULT_TARGET)
    parser.add_argument("--source-csv", type=Path, default=v16.DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=v16.DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = build_parser().parse_args(argv)
    result, tensors = train(args)
    output = publish(output=args.output, result=result, tensors=tensors)
    metrics = result["public"]["metrics"]
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "public_strict": metrics["top1"]["strict_accuracy"],
                "public_n4": metrics["top1"]["relaxed_accuracy"],
                "private_reference_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
