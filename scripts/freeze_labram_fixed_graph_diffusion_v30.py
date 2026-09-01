#!/usr/bin/env python3
"""Freeze the deterministic one-step C18 scalp-graph diffusion v30."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_fine_temporal_nested_oof_v11_1 import _evaluate  # noqa: E402
from src.soz.geometry import CHANNEL_INDEX  # noqa: E402
from src.soz.metrics import DEEPSOZ_STANDARD19_NEIGHBORS  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "soz_labram_fixed_graph_diffusion_public_oof_v30"
PROTOCOL = ROOT / "research/02_method/labram_fixed_graph_diffusion_protocol_v30_20260815_zh.md"
DEFAULT_INPUT = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/labram_fixed_graph_diffusion_public_oof_v30_20260815"


def fixed_graph_diffusion(probability: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    if probability.ndim != 2 or probability.shape[1] != 19 or tuple(candidate_mask.shape) != (19,):
        raise ValueError("graph diffusion expects probability [N,19] and mask [19]")
    if candidate_mask.dtype != torch.bool or not torch.equal(candidate_mask, V11_CANDIDATE_MASK):
        raise ValueError("graph diffusion requires the frozen C18 mask")
    if not torch.isfinite(probability).all() or torch.any(probability < 0):
        raise ValueError("graph diffusion requires finite nonnegative input")
    neighborhood = []
    for channel in range(19):
        indices = [
            int(value)
            for value in DEEPSOZ_STANDARD19_NEIGHBORS[channel]
            if bool(candidate_mask[int(value)])
        ]
        neighborhood.append(
            probability[:, indices].mean(dim=1) if indices else probability[:, channel]
        )
    result = 0.5 * probability + 0.5 * torch.stack(neighborhood, dim=1)
    result[:, ~candidate_mask] = 0.0
    result /= result.sum(dim=1, keepdim=True).clamp_min(1e-12)
    if not torch.isfinite(result).all() or not torch.allclose(
        result.sum(dim=1), torch.ones(len(result)), atol=1e-6, rtol=0
    ):
        raise RuntimeError("graph diffusion probability contract failed")
    return result.contiguous()


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    source_manifest = json.loads((args.input / "manifest.json").resolve(strict=True).read_text())
    source = load_file(str((args.input / "oof_predictions.safetensors").resolve(strict=True)))
    if source_manifest.get("decision") != "AUTHORIZE_ONE_TARGET_BLIND_PRIVATE_RUN":
        raise ValueError("v29 public gate was not passed")
    probability = source["oof.portable_equal_ensemble_probability"].float()
    mask = source["candidate_mask"].bool()
    target_mask = source["target_mask"].bool()
    targets = source["targets"].float()
    folds = source["patient_folds"].long()
    diffused = fixed_graph_diffusion(probability, mask)
    logits = torch.log(diffused.clamp_min(1e-12))
    baseline_logits = torch.log(probability.clamp_min(1e-12))
    metrics = _evaluate(logits, targets, target_mask)
    baseline = _evaluate(baseline_logits, targets, target_mask)
    fold_rows = []
    fold_nonlower = 0
    for fold in range(5):
        held = folds == fold
        candidate_metrics = _evaluate(logits[held], targets[held], target_mask[held])
        baseline_metrics = _evaluate(
            baseline_logits[held], targets[held], target_mask[held]
        )
        nonlower = candidate_metrics["top1"]["strict_accuracy"] >= baseline_metrics["top1"]["strict_accuracy"]
        fold_nonlower += int(nonlower)
        fold_rows.append({
            "fold": fold,
            "candidate_metrics": candidate_metrics,
            "v29_metrics": baseline_metrics,
            "strict_nonlower": nonlower,
        })
    checks = {
        "strict_not_lower_than_v29": metrics["top1"]["strict_accuracy"] >= baseline["top1"]["strict_accuracy"],
        # `_evaluate` returns float32 rates.  Convert the discrete rate back to
        # its hit count so 81/102 cannot fail solely from float32 rounding.
        "relaxed_at_least_81_of_102": round(
            metrics["top1"]["relaxed_accuracy"] * metrics["top1"]["n_samples"]
        )
        >= 81,
        "macro_ap_higher_than_v29": metrics["ranking"]["macro_average_precision"] > baseline["ranking"]["macro_average_precision"],
        "far_error_not_higher_than_21": metrics["far_error_count"] <= 21,
        "at_least_four_folds_strict_nonlower": fold_nonlower >= 4,
        "finite_c18": bool(torch.isfinite(diffused).all())
        and not bool(mask[CHANNEL_INDEX["PZ"]]),
    }
    go = all(checks.values())
    manifest: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_public_adaptive_fixed_graph_oof_freeze",
        "decision": "AUTHORIZE_ONE_TARGET_BLIND_PRIVATE_TRANSFORM" if go else "PUBLIC_NO_GO",
        "protocol": str(PROTOCOL),
        "operator": {
            "self_weight": 0.5,
            "one_hop_neighbor_mean_weight": 0.5,
            "steps": 1,
            "trainable_parameters": 0,
            "adjacency": "frozen_DEEPSOZ_STANDARD19_NEIGHBORS",
        },
        "metrics": {"v30": metrics, "v29": baseline},
        "fold_results": fold_rows,
        "go_checks": checks,
        "go": go,
        "access_receipt": {
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_prediction_or_metric_loaded": False,
            "training_or_parameter_fitting_performed": False,
        },
        "claim_boundary": {
            "formed_after_viewing_v29_public_oof": True,
            "evaluation_adjacency_shared_with_model_prior": True,
            "strict_and_undiffused_metrics_must_be_reported": True,
            "fresh_confirmation": False,
        },
    }
    tensors = {
        "oof.v30_probability": diffused,
        "oof.v30_logits": logits,
        "oof.v29_probability": probability,
        "targets": targets,
        "target_mask": target_mask,
        "patient_folds": folds,
        "candidate_mask": mask,
    }
    return manifest, tensors


def publish(output: Path, manifest: Mapping[str, object], tensors: Mapping[str, torch.Tensor]) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        save_file(dict(tensors), str(staging / "oof_predictions.safetensors"))
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, tensors = run(args)
    output = publish(args.output, manifest, tensors)
    metrics = manifest["metrics"]["v30"]
    print(json.dumps({
        "decision": manifest["decision"],
        "output": str(output),
        "strict": metrics["top1"]["strict_accuracy"],
        "relaxed": metrics["top1"]["relaxed_accuracy"],
        "macro_ap": metrics["ranking"]["macro_average_precision"],
        "private_used": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
