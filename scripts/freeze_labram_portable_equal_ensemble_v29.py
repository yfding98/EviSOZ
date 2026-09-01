#!/usr/bin/env python3
"""Freeze the parameter-free H-only/direct-token public OOF ensemble v29."""

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
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "soz_labram_portable_equal_ensemble_public_oof_v29"
PROTOCOL = ROOT / "research/02_method/labram_portable_equal_ensemble_protocol_v29_20260815_zh.md"
DEFAULT_V28 = ROOT / "outputs/labram_rank1_direct_token_oof_v28_20260815"
DEFAULT_V16 = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
PAPER_POINT_ESTIMATE = 0.744


def _probability(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    result = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
    if not torch.isfinite(result).all():
        raise RuntimeError("v29 probability is non-finite")
    return result


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    v28_manifest = json.loads((args.v28 / "manifest.json").resolve(strict=True).read_text())
    v28 = load_file(str((args.v28 / "model_and_oof.safetensors").resolve(strict=True)))
    v16_manifest = json.loads((args.v16 / "manifest.json").resolve(strict=True).read_text())
    v16 = load_file(str((args.v16 / "oof_predictions.safetensors").resolve(strict=True)))
    mask = v28["target_mask"].bool()
    candidate_mask = v28["config.candidate_mask"].bool()
    if not torch.equal(candidate_mask, V11_CANDIDATE_MASK):
        raise ValueError("v29 candidate mask drifted")
    for name in ("targets", "target_mask", "patient_folds"):
        left = v28[name]
        right = v16[name]
        if not torch.equal(left, right):
            raise ValueError(f"v28/v16 carrier mismatch: {name}")
    direct = _probability(v28["oof.rank1_direct_token"].float(), mask)
    h_only = _probability(v16["oof.frozen_labram_only"].float(), mask)
    v17 = _probability(v28["oof.v17_anchor"].float(), mask)
    equal_probability = 0.5 * h_only + 0.5 * direct
    equal_logits = torch.log(equal_probability.clamp_min(1e-12))
    targets = v28["targets"].float()
    folds = v28["patient_folds"].long()
    metrics = _evaluate(equal_logits, targets, mask)
    h_metrics = _evaluate(v16["oof.frozen_labram_only"].float(), targets, mask)
    v17_metrics = _evaluate(v28["oof.v17_anchor"].float(), targets, mask)
    fold_rows = []
    fold_nonlower = 0
    for fold in range(5):
        held = folds == fold
        candidate = _evaluate(equal_logits[held], targets[held], mask[held])
        anchor = _evaluate(
            v16["oof.frozen_labram_only"].float()[held], targets[held], mask[held]
        )
        nonlower = (
            candidate["top1"]["strict_accuracy"]
            >= anchor["top1"]["strict_accuracy"]
        )
        fold_nonlower += int(nonlower)
        fold_rows.append(
            {
                "fold": fold,
                "candidate_metrics": candidate,
                "h_only_metrics": anchor,
                "strict_nonlower": nonlower,
            }
        )
    go_checks = {
        "strict_not_lower_than_v17": metrics["top1"]["strict_accuracy"]
        >= v17_metrics["top1"]["strict_accuracy"],
        "relaxed_strictly_above_deepsoz_paper_point": metrics["top1"][
            "relaxed_accuracy"
        ]
        > PAPER_POINT_ESTIMATE,
        "macro_ap_strictly_higher_than_v17": metrics["ranking"][
            "macro_average_precision"
        ]
        > v17_metrics["ranking"]["macro_average_precision"],
        "far_errors_not_higher_than_v17": metrics["far_error_count"]
        <= v17_metrics["far_error_count"],
        "at_least_four_folds_strict_nonlower_than_h_only": fold_nonlower >= 4,
        "finite_c18_pz_masked": bool(torch.isfinite(equal_probability).all())
        and not bool(candidate_mask[STANDARD_19.index("PZ")]),
    }
    go = all(go_checks.values())
    manifest: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_public_adaptive_exploratory_oof_freeze",
        "decision": "AUTHORIZE_ONE_TARGET_BLIND_PRIVATE_RUN" if go else "PUBLIC_NO_GO",
        "protocol": str(PROTOCOL),
        "ensemble": {
            "h_only_weight": 0.5,
            "rank1_direct_token_weight": 0.5,
            "combination_space": "candidate_masked_probability",
            "weight_trainable": False,
            "confidence_gate": False,
            "fine_feature_family_used": False,
            "fold_count": 5,
        },
        "metrics": {
            "portable_equal_ensemble": metrics,
            "h_only": h_metrics,
            "v17_h_plus_fine": v17_metrics,
        },
        "fold_results": fold_rows,
        "go_checks": go_checks,
        "go": go,
        "access_receipt": {
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_prediction_or_metric_loaded": False,
            "training_performed": False,
            "ensemble_parameter_fitted": False,
        },
        "claim_boundary": {
            "public_candidate_was_formed_after_v28_public_oof": True,
            "fresh_public_confirmation": False,
            "private_is_fresh_external_validation": False,
            "neighborhood4_is_strict_accuracy": False,
        },
        "source_status": {
            "v28_decision": v28_manifest["decision"],
            "v16_status": v16_manifest["status"],
        },
    }
    tensors = {
        "oof.portable_equal_ensemble_probability": equal_probability.contiguous(),
        "oof.portable_equal_ensemble_logits": equal_logits.contiguous(),
        "oof.h_only_probability": h_only.contiguous(),
        "oof.rank1_direct_probability": direct.contiguous(),
        "oof.v17_probability": v17.contiguous(),
        "targets": targets,
        "target_mask": mask,
        "patient_folds": folds,
        "candidate_mask": candidate_mask,
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
    parser.add_argument("--v28", type=Path, default=DEFAULT_V28)
    parser.add_argument("--v16", type=Path, default=DEFAULT_V16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, tensors = run(args)
    output = publish(args.output, manifest, tensors)
    metrics = manifest["metrics"]["portable_equal_ensemble"]
    print(json.dumps({
        "decision": manifest["decision"],
        "output": str(output),
        "strict": metrics["top1"]["strict_accuracy"],
        "relaxed": metrics["top1"]["relaxed_accuracy"],
        "macro_ap": metrics["ranking"]["macro_average_precision"],
        "private_used": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
