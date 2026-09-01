#!/usr/bin/env python3
"""Run a controlled Raw200 head-only versus CPBF refinement benchmark."""

from __future__ import annotations

import argparse
import copy
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
from src.soz.baseline.raw200_cpbf import Raw200CPBFRefinement  # noqa: E402
from src.soz.baseline.raw200_shallow import Raw200ChannelShallowNet  # noqa: E402
from src.soz.data.public_development_union_identity_v12 import (  # noqa: E402
    load_public_development_union_identity_v12,
)
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    positive_set_mass_loss,
)


SCHEMA = "raw200_cpbf_unified_c18_benchmark_v80"
PROTOCOL = ROOT / "research/02_method/raw200_cpbf_unified_benchmark_protocol_20260817_zh.md"
DEFAULT_WAVEFORM = ROOT / "outputs/trustworthy_soz_raw200_events_v60_20260816"
DEFAULT_BASELINE = ROOT / "outputs/trustworthy_soz_raw200_shallow_baseline_v60_20260816"
DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_OUTPUT = ROOT / "outputs/raw200_cpbf_unified_c18_benchmark_v80r1_20260817"
VARIANTS = ("head_only", "cpbf")
EPOCHS = 10
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-3
BATCH_PATIENTS = 8
TOKEN_BATCH = 16
BASE_SEED = 20260817


def _patient_event_rows(
    event_patient: torch.Tensor, patient_count: int
) -> tuple[torch.Tensor, ...]:
    rows = tuple(
        torch.nonzero(event_patient == patient, as_tuple=False).flatten()
        for patient in range(patient_count)
    )
    if any(len(value) == 0 for value in rows):
        raise ValueError("Raw200 CPBF patient bag lost all events")
    return rows


def _base_model(
    baseline: Mapping[str, torch.Tensor], fold: int
) -> Raw200ChannelShallowNet:
    prefix = f"fold{fold}."
    prior = baseline[f"{prefix}prior_logits"].float()
    model = Raw200ChannelShallowNet(prior)
    state = {
        "temporal.weight": baseline[f"{prefix}temporal.weight"].float(),
        "channel_scorer.weight": baseline[f"{prefix}channel_scorer.weight"].float(),
        "channel_scorer.bias": baseline[f"{prefix}channel_scorer.bias"].float(),
        "prior_logits": prior,
    }
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def _extract_tokens(
    base: Raw200ChannelShallowNet,
    waveform: torch.Tensor,
    scale: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    extractor = Raw200CPBFRefinement(
        copy.deepcopy(base), variant="head_only"
    ).eval().requires_grad_(False).to(device)
    output = []
    for start in range(0, len(waveform), TOKEN_BATCH):
        signal = waveform[start : start + TOKEN_BATCH].to(device)
        output.append(extractor.extract_phase_tokens(signal.div(scale)).cpu())
    result = torch.cat(output).float().contiguous()
    expected = (len(waveform), 19, 6, 32)
    if tuple(result.shape) != expected or not torch.isfinite(result).all():
        raise RuntimeError(f"Raw200 CPBF token cache changed: {tuple(result.shape)}")
    return result


def _seeded_refinement(
    base: Raw200ChannelShallowNet, *, variant: str, seed: int
) -> Raw200CPBFRefinement:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return Raw200CPBFRefinement(copy.deepcopy(base), variant=variant)


def _fit_refinement(
    base: Raw200ChannelShallowNet,
    tokens: torch.Tensor,
    patient_event_rows: Sequence[torch.Tensor],
    train_patients: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    variant: str,
    seed: int,
    device: torch.device,
) -> tuple[Raw200CPBFRefinement, list[float], int]:
    torch.manual_seed(seed)
    model = _seeded_refinement(base, variant=variant, seed=seed).to(device)
    trainable = [value for value in model.parameters() if value.requires_grad]
    if not trainable:
        raise RuntimeError("Raw200 CPBF refinement has no trainable parameters")
    trainable_parameter_count = sum(value.numel() for value in trainable)
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(seed + 1)
    patient_list = train_patients.tolist()
    losses: list[float] = []
    for epoch in range(EPOCHS):
        sampled_events = []
        for patient in patient_list:
            candidates = patient_event_rows[int(patient)]
            choice = int(torch.randint(len(candidates), (1,), generator=generator))
            sampled_events.append(int(candidates[choice]))
        order = torch.randperm(len(patient_list), generator=generator)
        model.train()
        epoch_loss = 0.0
        for start in range(0, len(order), BATCH_PATIENTS):
            positions = order[start : start + BATCH_PATIENTS]
            patient_batch = train_patients.index_select(0, positions)
            event_batch = torch.tensor(
                [sampled_events[int(value)] for value in positions.tolist()],
                dtype=torch.long,
            )
            feature = tokens.index_select(0, event_batch).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model.forward_tokens(feature)
            loss = positive_set_mass_loss(
                logits,
                targets.index_select(0, patient_batch).to(device),
                target_mask.index_select(0, patient_batch).to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(positions)
        losses.append(epoch_loss / len(patient_list))
    model.eval().requires_grad_(False)
    return model, losses, trainable_parameter_count


@torch.inference_mode()
def _predict_events(
    model: Raw200CPBFRefinement,
    tokens: torch.Tensor,
    rows: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    output = []
    for start in range(0, len(rows), BATCH_PATIENTS):
        selected = rows[start : start + BATCH_PATIENTS]
        logits, _ = model.forward_tokens(tokens.index_select(0, selected).to(device))
        output.append(
            torch.softmax(
                logits.masked_fill(
                    ~V11_CANDIDATE_MASK.to(device).view(1, -1), -torch.inf
                ),
                dim=1,
            ).cpu()
        )
    result = torch.cat(output).float().contiguous()
    if tuple(result.shape) != (len(rows), 19) or not torch.isfinite(result).all():
        raise RuntimeError("Raw200 CPBF event prediction failed")
    return result


def _pool_patients(
    event_probability: torch.Tensor,
    event_rows: Sequence[torch.Tensor],
    patients: torch.Tensor,
) -> torch.Tensor:
    result = torch.stack(
        [
            event_probability.index_select(0, event_rows[int(patient)]).mean(dim=0)
            for patient in patients
        ]
    ).contiguous()
    if not torch.isfinite(result).all():
        raise RuntimeError("Raw200 CPBF patient pooling failed")
    return result


def train(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    started = time.monotonic()
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    waveform_manifest = json.loads(
        (args.waveform / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    if waveform_manifest.get("status") != "completed_reference_isolated_raw200_event_waveforms":
        raise ValueError("Formal Raw200 waveform materialization is missing")
    access = waveform_manifest.get("access_receipt", {})
    if access.get("private_significant_or_spread_reference_loaded") is not False:
        raise ValueError("Raw200 CPBF waveform producer opened private reference")
    waveforms = load_file(
        str((args.waveform / "raw200_events.safetensors").resolve(strict=True)),
        device="cpu",
    )
    public_all = waveforms["public.event_waveform_microvolts"].float()
    public_union_patient = waveforms["public.event_patient_union_index"].long()
    private = waveforms["private.event_waveform_microvolts"].float()
    if tuple(public_all.shape) != (1_149, 19, 12_000) or tuple(private.shape) != (
        88,
        19,
        12_000,
    ):
        raise ValueError("Raw200 CPBF waveform roster changed")

    baseline_manifest = json.loads(
        (args.baseline / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    if baseline_manifest.get("status") != "completed_post_open_fixed_raw200_public_OOF_private_comparator":
        raise ValueError("Frozen Raw200 fold checkpoints are missing")
    baseline = load_file(
        str((args.baseline / "raw200_shallow_predictions.safetensors").resolve(strict=True)),
        device="cpu",
    )

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
    selected_events = torch.nonzero(
        public_primary_patient >= 0, as_tuple=False
    ).flatten()
    public = public_all.index_select(0, selected_events).contiguous()
    event_patient = public_primary_patient.index_select(
        0, selected_events
    ).contiguous()
    del public_all, waveforms
    if tuple(public.shape) != (1_145, 19, 12_000):
        raise ValueError("Raw200 CPBF primary public event roster changed")
    patient_event_rows = _patient_event_rows(event_patient, 102)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    oof = {
        variant: torch.full((102, 19), torch.nan) for variant in VARIANTS
    }
    private_folds: dict[str, list[torch.Tensor]] = {
        variant: [] for variant in VARIANTS
    }
    states: dict[str, torch.Tensor] = {}
    folds: list[dict[str, object]] = []
    all_private_rows = torch.arange(len(private))
    for fold in range(5):
        fold_started = time.monotonic()
        scale = float(baseline[f"fold{fold}.input_rms_scale_microvolts"])
        base = _base_model(baseline, fold)
        public_tokens = _extract_tokens(base, public, scale, device=device)
        private_tokens = _extract_tokens(base, private, scale, device=device)
        train_patients = torch.nonzero(
            roster.patient_folds != fold, as_tuple=False
        ).flatten()
        held_patients = torch.nonzero(
            roster.patient_folds == fold, as_tuple=False
        ).flatten()
        held_event_rows = torch.cat(
            [patient_event_rows[int(patient)] for patient in held_patients.tolist()]
        )

        # Both refinements start from the exact same frozen base.  CPBF is an
        # exact identity before optimization because its residual scale is 0.
        init_head = _seeded_refinement(
            base, variant="head_only", seed=BASE_SEED + 10_000 * fold
        ).eval()
        init_cpbf = _seeded_refinement(
            base, variant="cpbf", seed=BASE_SEED + 10_000 * fold
        ).eval()
        with torch.inference_mode():
            identity_head, _ = init_head.forward_tokens(public_tokens[:2])
            identity_cpbf, _ = init_cpbf.forward_tokens(public_tokens[:2])
        identity_error = float((identity_head - identity_cpbf).abs().max())
        if identity_error != 0.0:
            raise RuntimeError("CPBF zero-residual initialization lost exact identity")

        fold_result: dict[str, object] = {
            "fold": fold,
            "train_patients": len(train_patients),
            "held_patients": len(held_patients),
            "train_events": int(
                sum(len(patient_event_rows[int(value)]) for value in train_patients)
            ),
            "held_events": len(held_event_rows),
            "input_rms_scale_microvolts": scale,
            "initial_cpbf_vs_head_max_abs_logit": identity_error,
            "variants": {},
        }
        for variant in VARIANTS:
            seed = BASE_SEED + 10_000 * fold
            model, losses, trainable_parameter_count = _fit_refinement(
                base,
                public_tokens,
                patient_event_rows,
                train_patients,
                roster.targets,
                roster.target_mask,
                variant=variant,
                seed=seed,
                device=device,
            )
            held_probability = _predict_events(
                model, public_tokens, held_event_rows, device=device
            )
            held_lookup = torch.full((len(public), 19), torch.nan)
            held_lookup.index_copy_(0, held_event_rows, held_probability)
            held_patient_probability = _pool_patients(
                held_lookup, patient_event_rows, held_patients
            )
            oof[variant].index_copy_(0, held_patients, held_patient_probability)
            private_probability = _predict_events(
                model, private_tokens, all_private_rows, device=device
            )
            private_folds[variant].append(private_probability)
            for name, value in model.state_dict().items():
                states[f"{variant}.fold{fold}.{name}"] = (
                    value.detach().cpu().contiguous()
                )
            held_metrics = _evaluate(
                torch.log(held_patient_probability.clamp_min(1e-12)),
                roster.targets.index_select(0, held_patients),
                roster.target_mask.index_select(0, held_patients),
            )
            residual_scale = None
            if variant == "cpbf":
                residual_scale = float(
                    torch.tanh(model.cpbf_graph.residual_scale).detach().cpu()
                )
            fold_result["variants"][variant] = {
                "trainable_parameters": trainable_parameter_count,
                "first_epoch_loss": losses[0],
                "final_epoch_loss": losses[-1],
                "cpbf_residual_scale": residual_scale,
                "held_metrics": held_metrics,
            }
        fold_result["elapsed_sec"] = time.monotonic() - fold_started
        folds.append(fold_result)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "head_strict": fold_result["variants"]["head_only"]["held_metrics"]["top1"]["strict_accuracy"],
                    "cpbf_strict": fold_result["variants"]["cpbf"]["held_metrics"]["top1"]["strict_accuracy"],
                    "elapsed_sec": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del public_tokens, private_tokens

    tensors: dict[str, torch.Tensor] = {
        "public.targets": roster.targets,
        "public.target_mask": roster.target_mask,
        "public.patient_folds": roster.patient_folds,
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
        "baseline.public.oof_probability": baseline["public.oof_probability"],
        "baseline.private.probability": baseline["private.probability"],
        **states,
    }
    public_metrics: dict[str, object] = {
        "baseline": baseline_manifest["public"]["metrics"]
    }
    for variant in VARIANTS:
        if not torch.isfinite(oof[variant]).all():
            raise RuntimeError(f"Raw200 {variant} OOF prediction is incomplete")
        stacked_private = torch.stack(private_folds[variant], dim=1).contiguous()
        mean_private = stacked_private.mean(dim=1).contiguous()
        tensors[f"{variant}.public.oof_probability"] = oof[variant]
        tensors[f"{variant}.private.fold_probability"] = stacked_private
        tensors[f"{variant}.private.probability"] = mean_private
        public_metrics[variant] = _evaluate(
            torch.log(oof[variant].clamp_min(1e-12)),
            roster.targets,
            roster.target_mask,
        )

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_posthoc_public_oof_private_reference_isolated_inference",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "analysis_role": "controlled_model_layer_CPBF_graph_adapter_audit",
        "architecture": {
            "input": "CAR19 200Hz 60s [19,12000]",
            "frozen_base": "Raw200-Shallow v60 fold-specific temporal convolution and prior",
            "phase_tokens": "[event,19,6,32] pre/early/late mean and SD",
            "head_only": "continued Linear(192,1) refinement",
            "cpbf": (
                "historical CPBFSparseGraphBlock; context_graph; standard19 physical "
                "support; dynamic topk6; zero-initialized signed residual"
            ),
            "complete_historical_TFM_CPBF_retraining": False,
        },
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_patients": BATCH_PATIENTS,
            "seed": BASE_SEED,
            "loss": "patient-equal positive-set probability-mass NLL",
            "private_reference_used": False,
        },
        "public": {
            "patients": 102,
            "events": 1_145,
            "metrics": public_metrics,
            "folds": folds,
        },
        "private": {
            "events": waveform_manifest["private"]["events"],
            "event_count": 88,
            "reference_loaded": False,
        },
        "tensor_file": "predictions.safetensors",
        "access_receipt": {
            "public_targets_loaded_for_fixed_training_and_OOF_evaluation": True,
            "private_EEG_loaded_for_reference_isolated_inference": True,
            "private_significant_or_spread_reference_loaded": False,
            "private_used_for_model_seed_graph_epoch_or_threshold_selection": False,
        },
        "interpretation_boundary": {
            "public_and_private_have_been_historically_opened": True,
            "confirmatory_or_SOTA_evidence": False,
            "adapter_is_complete_historical_TFM_CPBF": False,
            "N4_is_strict_accuracy": False,
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
        save_file(dict(tensors), str(staging / "predictions.safetensors"))
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
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--union", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-directory",
        type=Path,
        default=ROOT / "outputs/deepsoz_target_v2_identity_recovery_20260812",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=ROOT / "outputs/deepsoz_tusz_adapted_manifest_20260803/source/TUH_manifest_final.csv",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=ROOT / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/split_manifest.csv",
    )
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, tensors = train(args)
    output = publish(output=args.output, result=result, tensors=tensors)
    print(
        json.dumps(
            {
                "output": str(output),
                "public_metrics": result["public"]["metrics"],
                "elapsed_sec": result["elapsed_sec"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
