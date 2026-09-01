#!/usr/bin/env python3
"""Frozen token-level channel/phase stress tests for the v29 D carrier.

The script replays saved outer-fold D heads on the already materialized
LaBraM prefix cache.  No head is fitted and no perturbation is selected by its
performance.  The H carrier remains frozen and unperturbed when D stress
predictions are combined with H; consequently this is a D-token reliance
audit, not a whole-model raw-EEG causal test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Callable, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_labram_rank1_direct_token_oof_v28 as v28  # noqa: E402
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _paired_bootstrap,
)
from src.soz.geometry import STANDARD_19  # noqa: E402


SCHEMA = "trustworthy_soz_labram_v29_token_stress_v38"
DEFAULT_V28 = ROOT / "outputs/labram_rank1_direct_token_oof_v28_20260815"
DEFAULT_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_labram_v29_token_stress_v38_20260816"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probability(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
    if not torch.isfinite(value).all():
        raise ValueError("non-finite probability")
    return value


def _probability_logits(probability: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    normalized = probability.masked_fill(~mask, 0.0)
    normalized = normalized / normalized.sum(dim=1, keepdim=True)
    return torch.log(normalized.clamp_min(1e-12))


def _identity(features: torch.Tensor) -> torch.Tensor:
    return features


def _zero(features: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(features)


def _channel_mean(features: torch.Tensor) -> torch.Tensor:
    return features.mean(dim=1, keepdim=True).expand_as(features).contiguous()


def _left_right_swap(features: torch.Tensor) -> torch.Tensor:
    pairs = {
        "FP1": "FP2",
        "FP2": "FP1",
        "F7": "F8",
        "F8": "F7",
        "F3": "F4",
        "F4": "F3",
        "T7": "T8",
        "T8": "T7",
        "C3": "C4",
        "C4": "C3",
        "P7": "P8",
        "P8": "P7",
        "P3": "P4",
        "P4": "P3",
        "O1": "O2",
        "O2": "O1",
        "FZ": "FZ",
        "CZ": "CZ",
        "PZ": "PZ",
    }
    index = torch.tensor(
        [STANDARD_19.index(pairs[channel]) for channel in STANDARD_19],
        dtype=torch.long,
    )
    return features.index_select(1, index)


def _reverse_pre_late(features: torch.Tensor) -> torch.Tensor:
    pre = features[:, :, 0]
    early = features[:, :, 1]
    late = features[:, :, 2]
    return torch.stack(
        (late, early, pre, early - late, pre - early), dim=2
    ).contiguous()


def _remove_difference_phases(features: torch.Tensor) -> torch.Tensor:
    result = features.clone()
    result[:, :, 3:] = 0.0
    return result


PERTURBATIONS: tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor], str], ...] = (
    ("identity_replay", _identity, "saved feature semantics unchanged"),
    (
        "zero_token_features",
        _zero,
        "remove all patient/event token content while retaining saved head priors and bias",
    ),
    (
        "channel_mean_locality_removed",
        _channel_mean,
        "replace every channel token by the within-event across-channel mean",
    ),
    (
        "left_right_channel_swap",
        _left_right_swap,
        "swap homologous left/right token identities while retaining the saved channel prior",
    ),
    (
        "reverse_pre_late",
        _reverse_pre_late,
        "exchange pre and late aggregate phases and recompute both difference phases",
    ),
    (
        "remove_difference_phases",
        _remove_difference_phases,
        "zero early-minus-pre and late-minus-early phases while retaining absolute phases",
    ),
)


def _state_for_fold(payload: Mapping[str, torch.Tensor], fold: int) -> dict[str, torch.Tensor]:
    prefix = f"outer_state.fold{fold}."
    state = {
        name[len(prefix) :]: value
        for name, value in payload.items()
        if name.startswith(prefix)
    }
    expected = {
        "phase_weights",
        "prior_logits",
        "tile_scorer.bias",
        "tile_scorer.weight",
        "candidate_mask",
    }
    if set(state) != expected:
        raise ValueError(f"fold {fold} state is incomplete: {sorted(state)}")
    return state


def _topk(probability: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(probability.masked_fill(~mask, -torch.inf), k=k, dim=1).indices


def _jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    rows = []
    for lhs, rhs in zip(left.tolist(), right.tolist()):
        lhs_set = set(lhs)
        rhs_set = set(rhs)
        rows.append(len(lhs_set & rhs_set) / len(lhs_set | rhs_set))
    return float(sum(rows) / len(rows))


def _stability(
    original: torch.Tensor, stressed: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    original_top1 = _topk(original, mask, 1).squeeze(1)
    stressed_top1 = _topk(stressed, mask, 1).squeeze(1)
    return {
        "top1_retention": float((original_top1 == stressed_top1).float().mean()),
        "top3_jaccard": _jaccard(
            _topk(original, mask, 3), _topk(stressed, mask, 3)
        ),
        "mean_absolute_probability_shift": float((original - stressed).abs()[mask].mean()),
    }


def _summary_row(
    perturbation: str,
    scope: str,
    metrics: Mapping[str, object],
    stability: Mapping[str, float],
) -> dict[str, object]:
    top1 = metrics["top1"]
    ranking = metrics["ranking"]
    assert isinstance(top1, Mapping) and isinstance(ranking, Mapping)
    return {
        "perturbation": perturbation,
        "scope": scope,
        "strict_top1": float(top1["strict_accuracy"]),
        "neighborhood4_top1": float(top1["relaxed_accuracy"]),
        "macro_average_precision": float(ranking["macro_average_precision"]),
        "hit_at_3": float(ranking["hit_at_k"][3]),
        "hit_at_5": float(ranking["hit_at_k"][5]),
        "far_error_count": float(metrics["far_error_count"]),
        **stability,
    }


def run(
    v28_directory: Path,
    v29_directory: Path,
) -> tuple[dict[str, object], dict[str, torch.Tensor], list[dict[str, object]]]:
    v28_manifest_path = (v28_directory / "manifest.json").resolve(strict=True)
    v28_tensor_path = (v28_directory / "model_and_oof.safetensors").resolve(strict=True)
    v29_manifest_path = (v29_directory / "manifest.json").resolve(strict=True)
    v29_tensor_path = (v29_directory / "oof_predictions.safetensors").resolve(strict=True)
    v28_payload = load_file(str(v28_tensor_path), device="cpu")
    v29_payload = load_file(str(v29_tensor_path), device="cpu")

    loader_args = v28.build_parser().parse_args(["--device", "cpu"])
    stable = v28.v17._load_stable_development(loader_args)
    prefix, event_patient_index = v28._load_stable_prefix(loader_args, stable)
    features = v28.extract_rank1_phase_features(prefix)
    del prefix
    bag = v28.PatientBag(
        phase_features=features,
        event_patient_index=event_patient_index,
        targets=stable.targets,
        target_mask=stable.target_mask,
        patient_ids=stable.patient_ids,
    )
    if tuple(stable.patient_ids) != tuple(
        json.loads(
            (
                ROOT
                / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/manifest.json"
            ).read_text(encoding="utf-8")
        )["patient_ids"]
    ):
        raise ValueError("stable public patient order differs from the v29 carrier")
    if not torch.equal(stable.targets, v29_payload["targets"]) or not torch.equal(
        stable.target_mask, v29_payload["target_mask"].bool()
    ):
        raise ValueError("stable public target carrier differs from v29")

    oof_logits = {
        name: torch.full((len(stable.patient_ids), 19), torch.nan)
        for name, _, _ in PERTURBATIONS
    }
    with torch.inference_mode():
        for fold in range(5):
            held_indices = tuple(
                torch.nonzero(stable.patient_folds == fold, as_tuple=False)
                .flatten()
                .tolist()
            )
            held = v28._subset_bag(bag, held_indices)
            state = _state_for_fold(v28_payload, fold)
            model = v28.RankOneDirectTokenHead(state["prior_logits"])
            model.load_state_dict(state, strict=True)
            model.eval()
            for name, transform, _ in PERTURBATIONS:
                transformed = transform(held.phase_features)
                event_logits = model(transformed)
                patient_logits = v28._aggregate_equal(
                    event_logits, held.event_patient_index, len(held.patient_ids)
                )
                oof_logits[name][list(held_indices)] = patient_logits.cpu()

    if any(not torch.isfinite(value).all() for value in oof_logits.values()):
        raise RuntimeError("a token stress OOF carrier is incomplete")
    original_saved = v28_payload["oof.rank1_direct_token"].float()
    replay_difference = float((oof_logits["identity_replay"] - original_saved).abs().max())
    if replay_difference > 1e-5:
        raise ValueError(f"saved D head replay drifted by {replay_difference}")

    mask = stable.target_mask
    targets = stable.targets
    h_probability = v29_payload["oof.h_only_probability"].float()
    original_d_probability = v29_payload["oof.rank1_direct_probability"].float()
    original_ensemble_probability = v29_payload[
        "oof.portable_equal_ensemble_probability"
    ].float()
    tensors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    metrics: dict[str, object] = {}
    paired: dict[str, object] = {}
    stability: dict[str, object] = {}
    original_ensemble_logits = _probability_logits(original_ensemble_probability, mask)
    for name, _, semantics in PERTURBATIONS:
        d_probability = _probability(oof_logits[name], mask)
        ensemble_probability = 0.5 * h_probability + 0.5 * d_probability
        d_metrics = _evaluate(_probability_logits(d_probability, mask), targets, mask)
        ensemble_metrics = _evaluate(
            _probability_logits(ensemble_probability, mask), targets, mask
        )
        d_stability = _stability(original_d_probability, d_probability, mask)
        ensemble_stability = _stability(
            original_ensemble_probability, ensemble_probability, mask
        )
        metrics[name] = {
            "semantics": semantics,
            "D_only": d_metrics,
            "H_plus_stressed_D_equal": ensemble_metrics,
        }
        stability[name] = {
            "D_only": d_stability,
            "H_plus_stressed_D_equal": ensemble_stability,
        }
        paired[name] = _paired_bootstrap(
            _probability_logits(ensemble_probability, mask),
            original_ensemble_logits,
            targets,
            mask,
        )
        rows.append(_summary_row(name, "D_only", d_metrics, d_stability))
        rows.append(
            _summary_row(
                name,
                "H_plus_stressed_D_equal",
                ensemble_metrics,
                ensemble_stability,
            )
        )
        tensors[f"oof.D.{name}"] = d_probability.contiguous()
        tensors[f"oof.H_plus_D.{name}"] = ensemble_probability.contiguous()

    identity_metrics = metrics["identity_replay"]["H_plus_stressed_D_equal"]
    declared_metrics = json.loads(v29_manifest_path.read_text(encoding="utf-8"))[
        "metrics"
    ]["portable_equal_ensemble"]
    if abs(
        identity_metrics["top1"]["strict_accuracy"]
        - declared_metrics["top1"]["strict_accuracy"]
    ) > 1e-7 or abs(
        identity_metrics["top1"]["relaxed_accuracy"]
        - declared_metrics["top1"]["relaxed_accuracy"]
    ) > 1e-7:
        raise ValueError("identity stress replay does not recover v29 metrics")

    prefix_manifest_path = (
        loader_args.stable_prefix_directory / "manifest.json"
    ).resolve(strict=True)
    prefix_tensor_path = (
        loader_args.stable_prefix_directory / "prefix.safetensors"
    ).resolve(strict=True)
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_D_token_reliance_stress",
        "analysis_role": "posthoc_consumed_public_development_audit",
        "patient_count": len(stable.patient_ids),
        "event_count": len(features),
        "metrics": metrics,
        "stability": stability,
        "paired_stressed_minus_original_v29": paired,
        "identity_replay_max_absolute_logit_difference": replay_difference,
        "source_files": {
            "v28_manifest": str(v28_manifest_path.relative_to(ROOT)),
            "v28_manifest_sha256": _sha256(v28_manifest_path),
            "v28_tensor": str(v28_tensor_path.relative_to(ROOT)),
            "v28_tensor_sha256": _sha256(v28_tensor_path),
            "v29_manifest": str(v29_manifest_path.relative_to(ROOT)),
            "v29_manifest_sha256": _sha256(v29_manifest_path),
            "v29_tensor": str(v29_tensor_path.relative_to(ROOT)),
            "v29_tensor_sha256": _sha256(v29_tensor_path),
            "prefix_manifest": str(prefix_manifest_path.relative_to(ROOT)),
            "prefix_manifest_sha256": _sha256(prefix_manifest_path),
            "prefix_tensor": str(prefix_tensor_path.relative_to(ROOT)),
            "prefix_tensor_sha256": _sha256(prefix_tensor_path),
        },
        "access_receipt": {
            "raw_eeg_loaded": False,
            "cached_foundation_tokens_loaded": True,
            "foundation_forward_performed": False,
            "model_training_performed": False,
            "model_or_perturbation_selected": False,
            "threshold_or_report_selected": False,
            "private_eeg_or_target_loaded": False,
            "existing_public_targets_loaded_for_frozen_evaluation": True,
        },
        "interpretation_boundary": {
            "stress_level": "cached_LaBraM_token_D_carrier_only",
            "H_carrier_perturbed": False,
            "raw_EEG_causal_intervention": False,
            "channel_time_clinical_explanation_validated": False,
            "all_public_patients_consumed_development": True,
            "confirmatory_inference": False,
            "allowed_claim": (
                "frozen D-head output and the H/D ensemble are sensitive to "
                "prespecified channel-locality and phase-token perturbations"
            ),
        },
    }
    tensors["targets"] = targets.contiguous()
    tensors["target_mask"] = mask.contiguous()
    tensors["patient_folds"] = stable.patient_folds.contiguous()
    return result, tensors, rows


def publish(
    output: Path,
    result: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    rows: list[dict[str, object]],
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        save_file(dict(tensors), str(staging / "oof_stress_predictions.safetensors"))
        with (staging / "stress_table.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--v28", type=Path, default=DEFAULT_V28)
    parser.add_argument("--v29", type=Path, default=DEFAULT_V29)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, tensors, rows = run(args.v28, args.v29)
    output = publish(args.output, result, tensors, rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "patient_count": result["patient_count"],
                "event_count": result["event_count"],
                "training_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
