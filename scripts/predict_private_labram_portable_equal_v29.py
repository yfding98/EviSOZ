#!/usr/bin/env python3
"""Apply the frozen five-fold portable v29 ensemble without target access."""

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

from scripts.audit_trustworthy_soz_candidate_v21 import _fold_h_only_probability  # noqa: E402
from scripts.run_labram_rank1_direct_token_oof_v28 import RankOneDirectTokenHead  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "soz_private_target_blind_labram_portable_equal_v29"
DEFAULT_PUBLIC = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_V28 = ROOT / "outputs/labram_rank1_direct_token_oof_v28_20260815"
DEFAULT_H_STATES = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815/outer_fold_states.safetensors"
DEFAULT_PHASE = ROOT / "outputs/private_target_blind_rank1_phase_v29_20260815"
DEFAULT_H_EVIDENCE = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
DEFAULT_OUTPUT = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _direct_probability(
    phase: torch.Tensor, states: Mapping[str, torch.Tensor], fold: int
) -> torch.Tensor:
    prefix = f"outer_state.fold{fold}."
    names = (
        "tile_scorer.weight",
        "tile_scorer.bias",
        "phase_weights",
        "prior_logits",
        "candidate_mask",
    )
    state = {name: states[prefix + name] for name in names}
    model = RankOneDirectTokenHead(state["prior_logits"])
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        logits = model(phase)
    probability = torch.softmax(logits.masked_fill(~V11_CANDIDATE_MASK, -torch.inf), dim=1)
    if not torch.isfinite(probability).all():
        raise RuntimeError("v29 direct probability is non-finite")
    return probability


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    public = _json(args.public_directory / "manifest.json")
    if public.get("decision") != "AUTHORIZE_ONE_TARGET_BLIND_PRIVATE_RUN" or public.get("go") is not True:
        raise ValueError("v29 public gate did not authorize private inference")
    phase_manifest = _json(args.phase_directory / "manifest.json")
    h_manifest = _json(args.h_evidence_directory / "manifest.json")
    for manifest, label in ((phase_manifest, "phase"), (h_manifest, "H")):
        access = manifest.get("access_receipt")
        if not isinstance(access, Mapping):
            raise ValueError(f"private {label} manifest lacks access receipt")
        if access.get("private_target_values_loaded") not in (False, None):
            raise ValueError(f"private {label} evidence is not target blind")
    phase_events = phase_manifest.get("events")
    h_events = h_manifest.get("events")
    if not isinstance(phase_events, list) or not isinstance(h_events, list) or [
        value["event_id"] for value in phase_events
    ] != [value["event_id"] for value in h_events]:
        raise ValueError("private phase/H event rosters differ")
    phase = load_file(str((args.phase_directory / phase_manifest["tensor_file"]).resolve(strict=True)))["phase_features"].float()
    h = load_file(str((args.h_evidence_directory / h_manifest["tensor_file"]).resolve(strict=True)))["h_event"].float()
    if tuple(phase.shape) != (88, 19, 5, 200) or tuple(h.shape) != (88, 19, 600):
        raise ValueError("private v29 evidence shape drifted")
    direct_states = load_file(str((args.v28_directory / "model_and_oof.safetensors").resolve(strict=True)))
    h_states = load_file(str(args.h_states.resolve(strict=True)))
    direct_fold = torch.stack(
        [_direct_probability(phase, direct_states, fold) for fold in range(5)], dim=1
    ).contiguous()
    h_fold = torch.stack(
        [_fold_h_only_probability(h, h_states, fold) for fold in range(5)], dim=1
    ).contiguous()
    equal_fold = (0.5 * direct_fold + 0.5 * h_fold).contiguous()
    probability = equal_fold.mean(dim=1).contiguous()
    if not torch.isfinite(probability).all() or not torch.allclose(
        probability.sum(dim=1), torch.ones(88), atol=1e-6, rtol=0
    ):
        raise RuntimeError("v29 ensemble probability contract failed")
    tensors = {
        "private_portable_equal_probability": probability,
        "private_portable_equal_fold_probability": equal_fold,
        "private_rank1_direct_fold_probability": direct_fold,
        "private_h_only_fold_probability": h_fold,
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
    }
    manifest: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_target_blind_private_inference",
        "event_count": 88,
        "patient_count": len({str(value["patient_id"]) for value in phase_events}),
        "events": phase_events,
        "channels": list(STANDARD_19),
        "tensor_file": "predictions.safetensors",
        "ensemble": public["ensemble"],
        "access_receipt": {
            "private_target_blind_phase_loaded": True,
            "private_target_blind_h_loaded": True,
            "private_target_ledger_path_argument_exposed": False,
            "private_target_values_loaded": False,
            "training_calibration_or_model_selection_performed": False,
            "llm_used_for_prediction_or_ranking": False,
        },
        "claim_boundary": {
            "private_is_fresh_external_validation": False,
            "scores_are_calibrated_error_probabilities": False,
            "output_is_cortical_soz_or_surgical_target": False,
        },
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
        save_file(dict(tensors), str(staging / "predictions.safetensors"))
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
    parser.add_argument("--public-directory", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--v28-directory", type=Path, default=DEFAULT_V28)
    parser.add_argument("--h-states", type=Path, default=DEFAULT_H_STATES)
    parser.add_argument("--phase-directory", type=Path, default=DEFAULT_PHASE)
    parser.add_argument("--h-evidence-directory", type=Path, default=DEFAULT_H_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, tensors = run(args)
    output = publish(args.output, manifest, tensors)
    print(json.dumps({"output": str(output), "events": 88, "private_targets_loaded": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
