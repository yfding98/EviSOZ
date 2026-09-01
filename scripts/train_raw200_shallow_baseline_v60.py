#!/usr/bin/env python3
"""Train the fixed public-OOF Raw200-Shallow comparator and infer private EEG."""

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
from src.soz.baseline.raw200_shallow import Raw200ChannelShallowNet  # noqa: E402
from src.soz.data.public_development_union_identity_v12 import (  # noqa: E402
    load_public_development_union_identity_v12,
)
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    jeffreys_reference_prior_logits,
    positive_set_mass_loss,
)


SCHEMA = "trustworthy_soz_raw200_shallow_baseline_v60"
PROTOCOL = ROOT / "research/02_method/post_open_fixed_audit_extensions_v60_20260816_zh.md"
DEFAULT_WAVEFORM = ROOT / "outputs/trustworthy_soz_raw200_events_v60_20260816"
DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_raw200_shallow_baseline_v60_20260816"
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_PATIENTS = 8
BASE_SEED = 20260860


def _seeded_model(prior: torch.Tensor, seed: int) -> Raw200ChannelShallowNet:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = Raw200ChannelShallowNet(prior)
    if model.n_trainable_parameters != 3_425:
        raise RuntimeError("raw200 shallow parameter count drifted")
    return model


def _event_rms_square(waveform: torch.Tensor, *, chunk: int = 32) -> torch.Tensor:
    values = []
    for start in range(0, len(waveform), chunk):
        value = waveform[start : start + chunk].double()
        values.append(value.square().mean(dim=(1, 2)).cpu())
    result = torch.cat(values).float().contiguous()
    if tuple(result.shape) != (len(waveform),) or not torch.isfinite(result).all():
        raise RuntimeError("raw200 event RMS computation failed")
    return result


def _patient_event_rows(
    event_patient: torch.Tensor, patient_count: int
) -> tuple[torch.Tensor, ...]:
    rows = tuple(
        torch.nonzero(event_patient == patient, as_tuple=False).flatten()
        for patient in range(patient_count)
    )
    if any(len(value) == 0 for value in rows):
        raise ValueError("raw200 patient bag lost all events")
    return rows


def _fit_fold(
    waveform: torch.Tensor,
    patient_event_rows: Sequence[torch.Tensor],
    train_patients: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    prior: torch.Tensor,
    event_rms_square: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> tuple[Raw200ChannelShallowNet, float, list[float]]:
    train_event_rows = torch.cat(
        [patient_event_rows[int(patient)] for patient in train_patients.tolist()]
    )
    scale = float(event_rms_square.index_select(0, train_event_rows).mean().sqrt())
    if not (scale > 0.0 and torch.isfinite(torch.tensor(scale))):
        raise RuntimeError("raw200 fold input scale is invalid")
    model = _seeded_model(prior, seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(seed + 1)
    losses: list[float] = []
    patient_list = train_patients.tolist()
    for epoch in range(EPOCHS):
        sampled_events = []
        for patient in patient_list:
            candidates = patient_event_rows[int(patient)]
            choice = int(torch.randint(len(candidates), (1,), generator=generator))
            sampled_events.append(int(candidates[choice]))
        order = torch.randperm(len(patient_list), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(order), BATCH_PATIENTS):
            positions = order[start : start + BATCH_PATIENTS]
            patient_batch = train_patients.index_select(0, positions)
            event_batch = torch.tensor(
                [sampled_events[int(value)] for value in positions.tolist()],
                dtype=torch.long,
            )
            signal = waveform.index_select(0, event_batch).to(device)
            signal = signal.div(scale).contiguous()
            optimizer.zero_grad(set_to_none=True)
            logits = model(signal)
            loss = positive_set_mass_loss(
                logits,
                targets.index_select(0, patient_batch).to(device),
                target_mask.index_select(0, patient_batch).to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(positions)
        losses.append(epoch_loss / len(patient_list))
        if epoch in {0, 9, 19, 29, 39}:
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "epoch": epoch + 1,
                        "loss": losses[-1],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    model.eval().requires_grad_(False)
    return model, scale, losses


@torch.inference_mode()
def _predict_events(
    model: Raw200ChannelShallowNet,
    waveform: torch.Tensor,
    rows: torch.Tensor,
    scale: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    output = []
    for start in range(0, len(rows), BATCH_PATIENTS):
        selected = rows[start : start + BATCH_PATIENTS]
        signal = waveform.index_select(0, selected).to(device).div(scale).contiguous()
        logits = model(signal).cpu()
        output.append(
            torch.softmax(
                logits.masked_fill(~V11_CANDIDATE_MASK.view(1, -1), -torch.inf),
                dim=1,
            )
        )
    result = torch.cat(output).contiguous()
    if tuple(result.shape) != (len(rows), 19) or not torch.isfinite(result).all():
        raise RuntimeError("raw200 event prediction failed")
    return result


def _pool_patients(
    event_probability: torch.Tensor,
    event_rows: Sequence[torch.Tensor],
    patients: torch.Tensor,
) -> torch.Tensor:
    result = torch.stack(
        [event_probability.index_select(0, event_rows[int(patient)]).mean(dim=0) for patient in patients]
    ).contiguous()
    if not torch.isfinite(result).all():
        raise RuntimeError("raw200 patient probability pooling failed")
    return result


def train(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    started = time.monotonic()
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    manifest = json.loads(
        (args.waveform / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    if manifest.get("status") != "completed_reference_isolated_raw200_event_waveforms":
        raise ValueError("formal raw200 event materialization is missing")
    access = manifest.get("access_receipt", {})
    if access.get("public_SOZ_target_values_loaded") is not False or access.get(
        "private_significant_or_spread_reference_loaded"
    ) is not False:
        raise ValueError("raw200 waveform reference firewall failed")
    payload = load_file(
        str((args.waveform / "raw200_events.safetensors").resolve(strict=True)),
        device="cpu",
    )
    public_all = payload["public.event_waveform_microvolts"].float()
    public_union_patient = payload["public.event_patient_union_index"].long()
    private = payload["private.event_waveform_microvolts"].float()
    if tuple(public_all.shape) != (1_149, 19, 12_000) or tuple(private.shape) != (
        88,
        19,
        12_000,
    ):
        raise ValueError("raw200 formal waveform roster changed")

    union = load_public_development_union_identity_v12(args.union)
    roster, _ = v16._load_primary_roster(
        union,
        target_directory=args.target_directory,
        source_csv=args.source_csv,
        split_csv=args.split_csv,
    )
    union_to_primary = torch.full((len(union.patient_ids),), -1, dtype=torch.long)
    union_to_primary[roster.selected_union_indices] = torch.arange(102)
    public_primary_patient = union_to_primary.index_select(0, public_union_patient)
    selected_events = torch.nonzero(public_primary_patient >= 0, as_tuple=False).flatten()
    public = public_all.index_select(0, selected_events).contiguous()
    event_patient = public_primary_patient.index_select(0, selected_events).contiguous()
    del public_all, payload
    if tuple(public.shape) != (1_145, 19, 12_000):
        raise ValueError("raw200 primary public event count changed")
    patient_event_rows = _patient_event_rows(event_patient, 102)
    event_rms = _event_rms_square(public)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    oof_probability = torch.full((102, 19), torch.nan)
    private_fold_rows: list[torch.Tensor] = []
    states: dict[str, torch.Tensor] = {}
    folds: list[dict[str, object]] = []
    all_private_rows = torch.arange(88)
    for fold in range(5):
        train_patients = torch.nonzero(
            roster.patient_folds != fold, as_tuple=False
        ).flatten()
        held_patients = torch.nonzero(
            roster.patient_folds == fold, as_tuple=False
        ).flatten()
        prior = jeffreys_reference_prior_logits(
            roster.targets.index_select(0, train_patients),
            roster.target_mask.index_select(0, train_patients),
        )
        model, scale, losses = _fit_fold(
            public,
            patient_event_rows,
            train_patients,
            roster.targets,
            roster.target_mask,
            prior,
            event_rms,
            seed=BASE_SEED + 1_000 * fold,
            device=device,
        )
        held_event_rows = torch.cat(
            [patient_event_rows[int(patient)] for patient in held_patients.tolist()]
        )
        held_event_probability = _predict_events(
            model, public, held_event_rows, scale, device=device
        )
        # Restore the local held-event predictions to the public event index so
        # every held patient can be pooled over its complete event bag.
        held_lookup = torch.full((len(public), 19), torch.nan)
        held_lookup.index_copy_(0, held_event_rows, held_event_probability)
        held_patient_probability = _pool_patients(
            held_lookup, patient_event_rows, held_patients
        )
        oof_probability.index_copy_(0, held_patients, held_patient_probability)
        private_probability = _predict_events(
            model, private, all_private_rows, scale, device=device
        )
        private_fold_rows.append(private_probability)
        for name, value in model.state_dict().items():
            states[f"fold{fold}.{name}"] = value.detach().cpu().contiguous()
        states[f"fold{fold}.input_rms_scale_microvolts"] = torch.tensor(scale)
        held_metrics = _evaluate(
            torch.log(held_patient_probability.clamp_min(1e-12)),
            roster.targets.index_select(0, held_patients),
            roster.target_mask.index_select(0, held_patients),
        )
        folds.append(
            {
                "fold": fold,
                "train_patients": len(train_patients),
                "held_patients": len(held_patients),
                "train_events": int(
                    sum(len(patient_event_rows[int(value)]) for value in train_patients)
                ),
                "held_events": len(held_event_rows),
                "input_rms_scale_microvolts": scale,
                "first_epoch_loss": losses[0],
                "final_epoch_loss": losses[-1],
                "held_metrics": held_metrics,
            }
        )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "strict": held_metrics["top1"]["strict_accuracy"],
                    "n4": held_metrics["top1"]["relaxed_accuracy"],
                    "elapsed_sec": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not torch.isfinite(oof_probability).all():
        raise RuntimeError("raw200 public OOF prediction is incomplete")
    private_fold = torch.stack(private_fold_rows, dim=1).contiguous()
    private_probability = private_fold.mean(dim=1).contiguous()
    metrics = _evaluate(
        torch.log(oof_probability.clamp_min(1e-12)),
        roster.targets,
        roster.target_mask,
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_post_open_fixed_raw200_public_OOF_private_comparator",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "architecture": {
            "input": "CAR19 200Hz 60s [19,12000] event waveform",
            "public_training": "one uniformly sampled event per patient per epoch",
            "public_evaluation": "equal probability mean over complete patient event bag",
            "shared_channel_local_temporal_power_scorer": [
                "Conv1d 1->32 kernel101 stride4 no bias",
                "square then AvgPool kernel50 stride12 then log",
                "pre/early/late mean and population-SD",
                "Linear192->1 plus fold-local Jeffreys prior",
            ],
            "trainable_parameters": 3_425,
            "foundation_or_pretrained_parameters": 0,
            "canonical_EEGNet_or_original_ShallowConvNet": False,
        },
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_patients": BATCH_PATIENTS,
            "seed": BASE_SEED,
            "hyperparameter_candidates": 1,
            "loss": "patient-equal positive-set probability-mass NLL",
        },
        "public": {
            "patients": 102,
            "events": 1_145,
            "metrics": metrics,
            "folds": folds,
        },
        "private": {
            "events": manifest["private"]["events"],
            "event_count": 88,
            "patients": len(
                {str(event["patient_id"]) for event in manifest["private"]["events"]}
            ),
        },
        "tensor_file": "raw200_shallow_predictions.safetensors",
        "access_receipt": {
            "public_targets_loaded_for_fixed_training_and_OOF_evaluation": True,
            "private_EEG_loaded_for_reference_isolated_inference": True,
            "private_significant_or_spread_reference_loaded": False,
            "private_used_for_model_hyperparameter_or_seed_selection": False,
            "foundation_training_or_inference_performed": False,
        },
        "interpretation_boundary": {
            "experiment_began_after_private_reference_opening": True,
            "fresh_or_target_blind_private_validation": False,
            "comparator_may_replace_v29_from_private_results": False,
            "one_task_adapted_raw_network_covers_all_raw_EEG_models": False,
            "N4_is_strict_accuracy": False,
        },
        "elapsed_sec": time.monotonic() - started,
    }
    tensors = {
        "public.oof_probability": oof_probability,
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
        save_file(dict(tensors), str(staging / "raw200_shallow_predictions.safetensors"))
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
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 16)))
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
