#!/usr/bin/env python3
"""Run frozen v27 H-only SOZ-candidate inference without opening targets."""

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

from scripts.run_labram_fine_temporal_nested_oof_v11 import _file_sha  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


TRAIN_SCHEMA = "soz_labram_h_only_full_source_refit_v27"
OUTPUT_SCHEMA = "soz_private_target_blind_labram_h_only_prediction_v27"
DEFAULT_CHECKPOINT = ROOT / "outputs/labram_h_only_full_source_refit_v27_20260815"
DEFAULT_EVIDENCE = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
DEFAULT_OUTPUT = ROOT / "outputs/labram_h_only_private_target_blind_v27_20260815"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _required_checkpoint(state: Mapping[str, torch.Tensor]) -> None:
    shapes = {
        "transform.h_center": (600,),
        "transform.h_scale": (600,),
        "transform.h_pca_mean": (600,),
        "transform.h_components": (600, 16),
        "reasoner.h_weight": (16,),
        "reasoner.prior_logits": (19,),
        "reasoner.candidate_mask": (19,),
        "config.candidate_mask": (19,),
        "config.foundation_trainable_parameters": (),
        "config.reasoner_trainable_parameters": (),
    }
    for name, shape in shapes.items():
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"missing or malformed v27 checkpoint tensor: {name}")
    if not torch.equal(state["reasoner.candidate_mask"].bool(), V11_CANDIDATE_MASK):
        raise ValueError("reasoner candidate mask drifted")
    if not torch.equal(state["config.candidate_mask"].bool(), V11_CANDIDATE_MASK):
        raise ValueError("config candidate mask drifted")
    if int(state["config.foundation_trainable_parameters"]) != 0 or int(
        state["config.reasoner_trainable_parameters"]
    ) != 16:
        raise ValueError("v27 model-capacity receipt drifted")


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    train_manifest = _json(args.checkpoint_directory / "manifest.json")
    if train_manifest.get("schema_version") != TRAIN_SCHEMA or train_manifest.get(
        "status"
    ) != "completed_public_only_full_refit_frozen_for_deployment":
        raise ValueError("v27 public refit is not a frozen deployment artifact")
    train_access = train_manifest.get("access_receipt")
    if not isinstance(train_access, Mapping) or any(
        train_access.get(name) is not False
        for name in (
            "private_raw_eeg_loaded",
            "private_cached_evidence_loaded",
            "private_target_values_loaded",
            "private_used_for_transform_prior_loss_or_model_selection",
        )
    ):
        raise ValueError("v27 public refit crossed the private firewall")
    checkpoint_path = args.checkpoint_directory / "checkpoint.safetensors"
    declared = train_manifest.get("files", {}).get("checkpoint.safetensors", {})
    if declared.get("sha256") != _file_sha(checkpoint_path):
        raise ValueError("v27 checkpoint is not bound to its manifest")
    state = load_file(str(checkpoint_path.resolve(strict=True)), device="cpu")
    _required_checkpoint(state)

    evidence_manifest = _json(args.evidence_directory / "manifest.json")
    access = evidence_manifest.get("access_receipt")
    if not isinstance(access, Mapping) or access.get("target_ledger_opened") is not False:
        raise ValueError("private evidence is not target blind")
    evidence_path = args.evidence_directory / str(evidence_manifest["tensor_file"])
    evidence = load_file(str(evidence_path.resolve(strict=True)), device="cpu")
    h_event = evidence.get("h_event")
    events = evidence_manifest.get("events")
    if (
        not isinstance(h_event, torch.Tensor)
        or tuple(h_event.shape[1:]) != (19, 600)
        or not torch.isfinite(h_event).all()
        or not isinstance(events, list)
        or len(events) != len(h_event)
    ):
        raise ValueError("private target-blind H/event roster is malformed")
    if len(events) != 88:
        raise ValueError("private v27 inference roster changed")

    h = h_event.float().contiguous()
    transformed_h = torch.matmul(
        (h - state["transform.h_center"]) / state["transform.h_scale"]
        - state["transform.h_pca_mean"],
        state["transform.h_components"],
    ).contiguous()
    prior = state["reasoner.prior_logits"].expand(len(h), -1).clone()
    h_contribution = torch.einsum(
        "ecd,d->ec", transformed_h, state["reasoner.h_weight"]
    ).contiguous()
    logits_unmasked = (prior + h_contribution).contiguous()
    logits = logits_unmasked.masked_fill(~V11_CANDIDATE_MASK, -torch.inf)
    probability = torch.softmax(logits, dim=1).contiguous()
    if not torch.isfinite(probability).all() or not torch.allclose(
        probability.sum(dim=1), torch.ones(len(h)), atol=1e-6, rtol=0
    ):
        raise RuntimeError("v27 private probability contract failed")

    output = {
        "private_h_only_probability": probability,
        "private_h_only_logits_unmasked": logits_unmasked,
        "private_h_only_h_contribution": h_contribution,
        "private_h_only_prior": prior,
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
    }
    manifest: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "completed_frozen_target_blind_private_inference",
        "event_count": len(events),
        "patient_count": len({str(event["patient_id"]) for event in events}),
        "events": events,
        "channels": list(STANDARD_19),
        "tensor_file": "predictions.safetensors",
        "tensor_shapes": {name: list(value.shape) for name, value in output.items()},
        "model": {
            "arm": "public_only_full_refit_frozen_labram_h_only_v27",
            "checkpoint_directory": str(args.checkpoint_directory.resolve()),
            "checkpoint_file_sha256": declared["sha256"],
            "candidate_space": "C18_physical_scalp_electrodes_PZ_masked",
        },
        "access_receipt": {
            "public_model_loaded": True,
            "private_target_blind_cached_evidence_loaded": True,
            "private_target_ledger_path_argument_exposed": False,
            "private_target_values_loaded": False,
            "training_performed": False,
            "calibration_or_threshold_selection_performed": False,
            "foundation_training_performed": False,
            "llm_used_for_prediction_or_ranking": False,
        },
        "claim_boundary": {
            "scores_are_calibrated_error_probabilities": False,
            "output_is_cortical_soz": False,
            "output_is_surgical_target": False,
            "clinician_review_required": True,
        },
    }
    return manifest, output


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
        tensor_path = staging / "predictions.safetensors"
        save_file(dict(tensors), str(tensor_path))
        completed = dict(manifest)
        completed["prediction_file_sha256"] = _file_sha(tensor_path)
        (staging / "manifest.json").write_text(
            json.dumps(completed, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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
    parser.add_argument("--checkpoint-directory", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--evidence-directory", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, tensors = run(args)
    output = publish(args.output_directory, manifest, tensors)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(output),
                "events": manifest["event_count"],
                "private_targets_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
