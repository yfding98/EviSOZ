#!/usr/bin/env python3
"""Run the single preregistered v28 public patient-OOF direct-token trial.

The official pretrained LaBraM-Base remains fully frozen.  This runner has no
private input and trains one 206-parameter shared rank-1 token head with an
equally patient-weighted positive-set objective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_fine_temporal_nested_oof_v11 import (  # noqa: E402
    _state_sha,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _paired_bootstrap,
)
import scripts.run_labram_masked_variable_auxiliary_oof_v17 as v17  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    jeffreys_reference_prior_logits,
    positive_set_mass_loss,
)


SCHEMA = "soz_labram_rank1_direct_token_oof_v28"
PROTOCOL = ROOT / "research/02_method/labram_rank1_direct_token_protocol_v28_20260815_zh.md"
DEFAULT_ANCHOR = ROOT / "outputs/labram_masked_variable_auxiliary_oof_v17_replay_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/labram_rank1_direct_token_oof_v28_20260815"
EXPECTED_AUX_PREFIX_MANIFEST_SHA256 = (
    "32ae76e5151f997cad70cee71070646dad4c6febe70ced50b0e20574fc5e4ed9"
)
EXPECTED_AUX_PREFIX_TENSOR_SHA256 = (
    "23e9726f5456da2a79c4c17a1b697428ccbd5d5eb1d46d866337b20bc901ffc6"
)
OUTER_FOLDS = tuple(range(5))
EPOCHS = 100
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-2
MAX_GRAD_NORM = 1.0
BASE_SEED = 20260828
FINAL_SEED = 20265828
N_PHASES = 5
TOKEN_DIM = 200


@dataclass(frozen=True)
class PatientBag:
    phase_features: torch.Tensor
    event_patient_index: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    patient_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.phase_features.ndim != 4 or tuple(self.phase_features.shape[1:]) != (
            19,
            N_PHASES,
            TOKEN_DIM,
        ):
            raise ValueError("phase_features must be [E,19,5,200]")
        events = len(self.phase_features)
        patients = len(self.patient_ids)
        if tuple(self.event_patient_index.shape) != (events,) or (
            self.event_patient_index.dtype != torch.long
        ):
            raise TypeError("event_patient_index must be long [E]")
        if tuple(self.targets.shape) != (patients, 19) or tuple(
            self.target_mask.shape
        ) != (patients, 19):
            raise ValueError("patient targets/mask must be [P,19]")
        if self.target_mask.dtype != torch.bool:
            raise TypeError("patient target mask must be bool")
        if events < patients or int(self.event_patient_index.min()) != 0 or int(
            self.event_patient_index.max()
        ) != patients - 1:
            raise ValueError("patient event index is incomplete")
        if torch.unique(self.event_patient_index).numel() != patients:
            raise ValueError("every patient must retain at least one event")
        if not torch.isfinite(self.phase_features).all():
            raise ValueError("phase features must be finite")

    def to(self, device: torch.device) -> "PatientBag":
        return PatientBag(
            phase_features=self.phase_features.to(device),
            event_patient_index=self.event_patient_index.to(device),
            targets=self.targets.to(device),
            target_mask=self.target_mask.to(device),
            patient_ids=self.patient_ids,
        )


class RankOneDirectTokenHead(nn.Module):
    """Shared rank-1 phase-by-token scorer with a fixed channel prior."""

    def __init__(self, prior_logits: torch.Tensor) -> None:
        super().__init__()
        if tuple(prior_logits.shape) != (19,) or not torch.isfinite(prior_logits).all():
            raise ValueError("prior_logits must be finite [19]")
        self.tile_scorer = nn.Linear(TOKEN_DIM, 1)
        self.phase_weights = nn.Parameter(torch.full((N_PHASES,), 0.2))
        self.register_buffer("prior_logits", prior_logits.detach().float().contiguous())
        self.register_buffer("candidate_mask", V11_CANDIDATE_MASK.clone())

    @property
    def n_trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if phase_features.ndim != 4 or tuple(phase_features.shape[1:]) != (
            19,
            N_PHASES,
            TOKEN_DIM,
        ):
            raise ValueError("rank-1 head expects [E,19,5,200]")
        score = self.tile_scorer(phase_features).squeeze(-1)
        # The linear scorer's bias cancels in early-pre and late-early.
        score = score.clone()
        score[:, :, 3:] -= self.tile_scorer.bias.view(1, 1, 1)
        contribution = (score * self.phase_weights.view(1, 1, -1)).sum(dim=-1)
        return contribution + self.prior_logits.view(1, -1)


def extract_rank1_phase_features(
    prefix: torch.Tensor, *, chunk_size: int = 64
) -> torch.Tensor:
    """Compress [E,15,77,200] without retaining the 1-GB physical view."""

    if prefix.ndim != 4 or tuple(prefix.shape[1:]) != (15, 77, 200):
        raise ValueError("LaBraM prefix must be [E,15,77,200]")
    if prefix.requires_grad or not torch.isfinite(prefix).all():
        raise ValueError("LaBraM prefix must be detached and finite")
    rows: list[torch.Tensor] = []
    for start in range(0, len(prefix), chunk_size):
        value = prefix[start : start + chunk_size]
        events = len(value)
        tiles = (
            value[:, :, 1:, :]
            .reshape(events, 15, 19, 4, TOKEN_DIM)
            .mean(dim=3)
        )
        pre = tiles[:, 0:3].mean(dim=1)
        early = tiles[:, 3:6].mean(dim=1)
        late = tiles[:, 6:15].mean(dim=1)
        rows.append(
            torch.stack((pre, early, late, early - pre, late - early), dim=2)
            .float()
            .cpu()
            .contiguous()
        )
    result = torch.cat(rows, dim=0).contiguous()
    if tuple(result.shape) != (len(prefix), 19, N_PHASES, TOKEN_DIM):
        raise RuntimeError("rank-1 phase feature shape drifted")
    return result


def _aggregate_equal(
    event_logits: torch.Tensor, event_patient_index: torch.Tensor, patients: int
) -> torch.Tensor:
    if tuple(event_logits.shape) != (len(event_patient_index), 19):
        raise ValueError("event logits/index mismatch")
    result = event_logits.new_zeros((patients, 19))
    result.index_add_(0, event_patient_index, event_logits)
    counts = torch.bincount(event_patient_index, minlength=patients).clamp_min(1)
    return result / counts.to(event_logits.dtype).unsqueeze(1)


def _subset_bag(full: PatientBag, patient_indices: Sequence[int]) -> PatientBag:
    selected = tuple(int(value) for value in patient_indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("patient subset must be non-empty and unique")
    old_to_new = torch.full((len(full.patient_ids),), -1, dtype=torch.long)
    old_to_new[torch.tensor(selected, dtype=torch.long)] = torch.arange(len(selected))
    event_selector = old_to_new.index_select(0, full.event_patient_index) >= 0
    event_rows = torch.nonzero(event_selector, as_tuple=False).flatten()
    new_index = old_to_new.index_select(
        0, full.event_patient_index.index_select(0, event_rows)
    )
    patient_rows = torch.tensor(selected, dtype=torch.long)
    return PatientBag(
        phase_features=full.phase_features.index_select(0, event_rows),
        event_patient_index=new_index,
        targets=full.targets.index_select(0, patient_rows),
        target_mask=full.target_mask.index_select(0, patient_rows),
        patient_ids=tuple(full.patient_ids[index] for index in selected),
    )


def _seeded_model(
    prior: torch.Tensor, seed: int, device: torch.device
) -> RankOneDirectTokenHead:
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = RankOneDirectTokenHead(prior)
    if model.n_trainable_parameters != 206:
        raise RuntimeError("v28 trainable parameter count drifted")
    return model.to(device)


def _fit(
    bag: PatientBag,
    prior: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> tuple[RankOneDirectTokenHead, dict[str, object]]:
    moved = bag.to(device)
    model = _seeded_model(prior, seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    first = None
    final = None
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        event_logits = model(moved.phase_features)
        patient_logits = _aggregate_equal(
            event_logits, moved.event_patient_index, len(moved.patient_ids)
        )
        loss = positive_set_mass_loss(
            patient_logits,
            moved.targets,
            moved.target_mask,
            allow_candidate_subset=True,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("v28 loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        value = float(loss.detach().cpu())
        first = value if first is None else first
        final = value
    model.eval().requires_grad_(False)
    return model, {
        "seed": seed,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "first_positive_set_mass": first,
        "final_positive_set_mass": final,
        "trainable_parameter_count": model.n_trainable_parameters,
    }


@torch.no_grad()
def _predict(model: RankOneDirectTokenHead, bag: PatientBag, device: torch.device) -> torch.Tensor:
    moved = bag.to(device)
    return _aggregate_equal(
        model(moved.phase_features), moved.event_patient_index, len(moved.patient_ids)
    ).cpu().contiguous()


def _load_stable_prefix(
    args: argparse.Namespace, stable: v17.StableDevelopmentData
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = v17.identity_v16._load_identity_cache(
        args.stable_prefix_directory,
        expected_manifest_sha256=args.expected_stable_prefix_manifest_sha256,
        expected_tensor_sha256=args.expected_stable_prefix_tensor_sha256,
        tensor_key="prefix_tokens",
        tensor_tail_shape=(15, 77, 200),
        union=stable.union,
        legacy_directory=args.legacy_prefix_directory,
        expected_legacy_manifest_sha256=(
            v17.identity_v16.EXPECTED_LEGACY_PREFIX_MANIFEST_SHA256
        ),
        expected_legacy_tensor_sha256=(
            v17.identity_v16.EXPECTED_LEGACY_PREFIX_TENSOR_SHA256
        ),
        label="stable LaBraM block-9 prefix identity-v12 for v28",
    )
    patient_index = {value: index for index, value in enumerate(stable.patient_ids)}
    selected_rows: list[int] = []
    selected_patients: list[int] = []
    selected_event_ids: list[str] = []
    for row, event in enumerate(stable.union.events):
        if event.patient_id not in patient_index:
            continue
        selected_rows.append(row)
        selected_patients.append(patient_index[event.patient_id])
        selected_event_ids.append(event.event_id)
    if tuple(selected_event_ids) != stable.stable_event_ids or len(selected_rows) != 1145:
        raise RuntimeError("v28 stable event identity/order drifted")
    row_tensor = torch.tensor(selected_rows, dtype=torch.long)
    return (
        cache.tensor.index_select(0, row_tensor).detach().contiguous(),
        torch.tensor(selected_patients, dtype=torch.long),
    )


def _load_anchor(
    directory: Path, stable: v17.StableDevelopmentData
) -> tuple[dict[str, object], torch.Tensor]:
    manifest = json.loads((directory / "manifest.json").resolve(strict=True).read_text())
    payload = load_file(str((directory / "oof_predictions.safetensors").resolve(strict=True)))
    required = {
        "oof.masked_variable_auxiliary_full",
        "targets",
        "target_mask",
        "patient_folds",
    }
    if not required.issubset(payload):
        raise ValueError("v28 anchor prediction payload is incomplete")
    if not torch.equal(payload["targets"], stable.targets) or not torch.equal(
        payload["target_mask"].bool(), stable.target_mask
    ) or not torch.equal(payload["patient_folds"].long(), stable.patient_folds):
        raise ValueError("v28 anchor/stable target or fold carrier differs")
    logits = payload["oof.masked_variable_auxiliary_full"].float().contiguous()
    declared = manifest["primary_comparison"]["candidate_metrics"]
    replay = _evaluate(logits, stable.targets, stable.target_mask)
    if json.dumps(declared, sort_keys=True) != json.dumps(replay, sort_keys=True):
        raise ValueError("v28 anchor metrics do not replay")
    return manifest, logits


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
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
        label="auxiliary LaBraM prefix for v28",
    )
    stable_prefix, stable_event_patient_index = _load_stable_prefix(args, stable)
    stable_features = extract_rank1_phase_features(stable_prefix)
    del stable_prefix
    auxiliary_features = extract_rank1_phase_features(
        auxiliary_prefix_cache.tensors["prefix_tokens"].float()
    )
    stable_count = len(stable.patient_ids)
    auxiliary_count = len(auxiliary.patient_ids)
    combined = PatientBag(
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
    anchor_manifest, anchor_logits = _load_anchor(args.anchor_directory, stable)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    oof = torch.full((stable_count, 19), torch.nan)
    states: dict[str, torch.Tensor] = {}
    fold_rows: list[dict[str, object]] = []
    candidate_fold_strict: list[float] = []
    anchor_fold_strict: list[float] = []
    for fold in OUTER_FOLDS:
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
        train_indices = stable_train + auxiliary_train
        train_bag = _subset_bag(combined, train_indices)
        held_bag = _subset_bag(combined, stable_held)
        stable_train_tensor = torch.tensor(stable_train, dtype=torch.long)
        prior = jeffreys_reference_prior_logits(
            stable.targets.index_select(0, stable_train_tensor),
            stable.target_mask.index_select(0, stable_train_tensor),
        )
        model, fit = _fit(
            train_bag, prior, seed=BASE_SEED + 1000 * fold, device=device
        )
        held_logits = _predict(model, held_bag, device)
        oof[list(stable_held)] = held_logits
        for name, value in model.state_dict().items():
            states[f"fold{fold}.{name}"] = value.detach().cpu().contiguous()
        held_targets = stable.targets[list(stable_held)]
        held_mask = stable.target_mask[list(stable_held)]
        candidate_metrics = _evaluate(held_logits, held_targets, held_mask)
        anchor_metrics = _evaluate(
            anchor_logits[list(stable_held)], held_targets, held_mask
        )
        candidate_fold_strict.append(candidate_metrics["top1"]["strict_accuracy"])
        anchor_fold_strict.append(anchor_metrics["top1"]["strict_accuracy"])
        fold_rows.append(
            {
                "fold": fold,
                "stable_train_patients": len(stable_train),
                "auxiliary_train_patients": len(auxiliary_train),
                "held_stable_patients": len(stable_held),
                "train_events": len(train_bag.phase_features),
                "held_events": len(held_bag.phase_features),
                "fit": fit,
                "candidate_metrics": candidate_metrics,
                "anchor_metrics": anchor_metrics,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "outer_fold_complete",
                    "fold": fold,
                    "strict": candidate_metrics["top1"]["strict_accuracy"],
                    "relaxed": candidate_metrics["top1"]["relaxed_accuracy"],
                    "elapsed_sec": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not torch.isfinite(oof).all():
        raise RuntimeError("v28 OOF predictions are incomplete")

    candidate_metrics = _evaluate(oof, stable.targets, stable.target_mask)
    anchor_metrics = _evaluate(anchor_logits, stable.targets, stable.target_mask)
    fold_nonlower = sum(
        candidate + 1e-12 >= anchor
        for candidate, anchor in zip(candidate_fold_strict, anchor_fold_strict)
    )
    go_checks = {
        "strict_not_lower_than_v17": candidate_metrics["top1"]["strict_accuracy"]
        >= anchor_metrics["top1"]["strict_accuracy"],
        "relaxed_at_least_80_of_102_and_strictly_higher_than_v17": (
            candidate_metrics["top1"]["relaxed_accuracy"] >= 80 / 102
            and candidate_metrics["top1"]["relaxed_accuracy"]
            > anchor_metrics["top1"]["relaxed_accuracy"]
        ),
        "macro_ap_strictly_higher_than_v17": candidate_metrics["ranking"][
            "macro_average_precision"
        ]
        > anchor_metrics["ranking"]["macro_average_precision"],
        "far_errors_not_higher_than_v17": candidate_metrics["far_error_count"]
        <= anchor_metrics["far_error_count"],
        "at_least_four_of_five_folds_strict_nonlower": fold_nonlower >= 4,
        "finite_complete_and_pz_masked": bool(torch.isfinite(oof).all())
        and not bool(V11_CANDIDATE_MASK[STANDARD_19.index("PZ")]),
    }
    go = all(go_checks.values())

    final_state: dict[str, torch.Tensor] = {}
    final_fit: dict[str, object] | None = None
    if go:
        all_stable = tuple(range(stable_count))
        prior = jeffreys_reference_prior_logits(stable.targets, stable.target_mask)
        final_model, final_fit = _fit(
            combined, prior, seed=FINAL_SEED, device=device
        )
        final_state = {
            f"reasoner.{name}": value.detach().cpu().contiguous()
            for name, value in final_model.state_dict().items()
        }
        final_state.update(
            {
                "config.candidate_mask": V11_CANDIDATE_MASK.clone(),
                "config.foundation_trainable_parameters": torch.tensor(0),
                "config.reasoner_trainable_parameters": torch.tensor(206),
                "config.epochs": torch.tensor(EPOCHS),
                "config.learning_rate": torch.tensor(LEARNING_RATE),
                "config.weight_decay": torch.tensor(WEIGHT_DECAY),
                "config.stable_patient_count": torch.tensor(len(all_stable)),
                "config.auxiliary_patient_count": torch.tensor(auxiliary_count),
            }
        )

    manifest: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_public_patient_oof",
        "decision": "PUBLIC_GO_FULL_REFIT_FROZEN" if go else "PUBLIC_NO_GO_STOP_BEFORE_PRIVATE",
        "protocol": str(PROTOCOL),
        "architecture": {
            "foundation": "official pretrained LaBraM-Base block-9 frozen",
            "event_prefix_shape": [15, 77, 200],
            "phase_feature_shape": [19, 5, 200],
            "reasoner": "shared rank-1 phase-by-token scorer plus fold-local Jeffreys prior",
            "reasoner_trainable_parameters": 206,
            "patient_pooling": "equal mean over complete seizure bag",
            "loss": "equally patient-weighted positive-set probability-mass NLL",
        },
        "data": {
            "stable_patients": stable_count,
            "stable_events": len(stable_features),
            "auxiliary_patients": auxiliary_count,
            "auxiliary_events": len(auxiliary_features),
            "patient_overlap": 0,
            "event_overlap": 0,
        },
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "base_seed": BASE_SEED,
            "candidate_count": 1,
            "foundation_trainable_parameters": 0,
            "final_refit_performed": go,
            "final_fit": final_fit,
        },
        "fold_results": fold_rows,
        "primary_comparison": {
            "candidate_metrics": candidate_metrics,
            "v17_anchor_metrics": anchor_metrics,
            "candidate_fold_strict": candidate_fold_strict,
            "v17_anchor_fold_strict": anchor_fold_strict,
            "fold_strict_nonlower_count": fold_nonlower,
            "paired_candidate_minus_v17": _paired_bootstrap(
                oof, anchor_logits, stable.targets, stable.target_mask
            ),
            "go_checks": go_checks,
            "go": go,
        },
        "access_receipt": {
            "private_path_argument_exposed": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_prediction_or_metric_loaded": False,
            "tusz_involvement_values_loaded": False,
            "clinical_report_text_loaded": False,
            "foundation_training_performed": False,
            "foundation_trainable_parameters": 0,
            "public_targets_used_for_patient_oof_training_and_evaluation": True,
        },
        "claim_boundary": {
            "public_102_is_fresh_confirmation": False,
            "neighborhood4_is_strict_accuracy": False,
            "output_is_cortical_soz_or_surgical_target": False,
            "private_inference_authorized_only_if_public_go": True,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    tensors = {
        "oof.rank1_direct_token": oof,
        "oof.v17_anchor": anchor_logits,
        "targets": stable.targets,
        "target_mask": stable.target_mask,
        "patient_folds": stable.patient_folds,
        "config.candidate_mask": V11_CANDIDATE_MASK.clone(),
        **{f"outer_state.{name}": value for name, value in states.items()},
        **{f"final.{name}": value for name, value in final_state.items()},
    }
    if final_state:
        manifest["final_state_sha256"] = _state_sha(final_state)
    return manifest, tensors


def publish(
    output_directory: Path,
    manifest: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
) -> Path:
    target = output_directory.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        save_file(dict(tensors), str(staging / "model_and_oof.safetensors"))
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
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=v17.DEFAULT_UNION)
    parser.add_argument("--stable-fine-directory", type=Path, default=v17.DEFAULT_STABLE_FINE)
    parser.add_argument("--stable-prefix-directory", type=Path, default=v17.DEFAULT_STABLE_PREFIX)
    parser.add_argument("--legacy-fine-directory", type=Path, default=v17.DEFAULT_LEGACY_FINE)
    parser.add_argument("--legacy-prefix-directory", type=Path, default=v17.DEFAULT_LEGACY_PREFIX)
    parser.add_argument("--target-directory", type=Path, default=v17.DEFAULT_TARGET)
    parser.add_argument("--source-csv", type=Path, default=v17.DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=v17.DEFAULT_SPLIT)
    parser.add_argument("--aux-join-directory", type=Path, default=v17.DEFAULT_AUX_JOIN)
    parser.add_argument("--aux-prefix-directory", type=Path, default=v17.DEFAULT_AUX_PREFIX)
    parser.add_argument("--anchor-directory", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-protocol-sha256", default=v17.EXPECTED_PROTOCOL_SHA256)
    parser.add_argument(
        "--expected-union-manifest-sha256",
        default=v17.EXPECTED_PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-stable-fine-manifest-sha256",
        default=v17.identity_v16.EXPECTED_FINE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-stable-fine-tensor-sha256",
        default=v17.identity_v16.EXPECTED_FINE_TENSOR_SHA256,
    )
    parser.add_argument(
        "--expected-stable-prefix-manifest-sha256",
        default=v17.identity_v16.EXPECTED_PREFIX_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-stable-prefix-tensor-sha256",
        default=v17.identity_v16.EXPECTED_PREFIX_TENSOR_SHA256,
    )
    parser.add_argument(
        "--expected-aux-join-artifact-sha256",
        default=v17.EXPECTED_AUX_JOIN_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--expected-aux-admission-artifact-sha256",
        default=v17.EXPECTED_AUX_ADMISSION_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--expected-aux-prefix-manifest-sha256",
        default=EXPECTED_AUX_PREFIX_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-aux-prefix-tensor-sha256",
        default=EXPECTED_AUX_PREFIX_TENSOR_SHA256,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = build_parser().parse_args(argv)
    manifest, tensors = run(args)
    output = publish(args.output_directory, manifest, tensors)
    metrics = manifest["primary_comparison"]["candidate_metrics"]
    print(
        json.dumps(
            {
                "decision": manifest["decision"],
                "output": str(output),
                "strict": metrics["top1"]["strict_accuracy"],
                "relaxed": metrics["top1"]["relaxed_accuracy"],
                "macro_ap": metrics["ranking"]["macro_average_precision"],
                "private_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
