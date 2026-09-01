#!/usr/bin/env python3
"""Apply frozen v30 graph diffusion to frozen target-blind v29 predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_labram_fixed_graph_diffusion_v30 import fixed_graph_diffusion  # noqa: E402


SCHEMA = "soz_private_target_blind_labram_fixed_graph_v30"
DEFAULT_PUBLIC = ROOT / "outputs/labram_fixed_graph_diffusion_public_oof_v30_20260815"
DEFAULT_INPUT = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/labram_fixed_graph_private_target_blind_v30_20260815"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    public = _json(args.public / "manifest.json")
    source_manifest = _json(args.input / "manifest.json")
    if public.get("decision") != "AUTHORIZE_ONE_TARGET_BLIND_PRIVATE_TRANSFORM":
        raise ValueError("v30 public gate did not authorize private transform")
    access = source_manifest.get("access_receipt")
    if not isinstance(access, Mapping) or access.get("private_target_values_loaded") is not False:
        raise ValueError("v29 source prediction is not target blind")
    source = load_file(str((args.input / source_manifest["tensor_file"]).resolve(strict=True)))
    probability = source["private_portable_equal_probability"].float()
    mask = source["candidate_mask"].bool()
    diffused = fixed_graph_diffusion(probability, mask)
    tensors = {
        "private_fixed_graph_probability": diffused,
        "private_v29_probability": probability,
        "candidate_mask": mask,
    }
    manifest: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_target_blind_private_transform",
        "events": source_manifest["events"],
        "event_count": source_manifest["event_count"],
        "patient_count": source_manifest["patient_count"],
        "tensor_file": "predictions.safetensors",
        "operator": public["operator"],
        "access_receipt": {
            "frozen_target_blind_v29_prediction_loaded": True,
            "private_target_ledger_path_argument_exposed": False,
            "private_target_values_loaded": False,
            "training_or_parameter_fitting_performed": False,
        },
        "claim_boundary": public["claim_boundary"],
    }
    return manifest, tensors


def publish(output: Path, manifest: Mapping[str, object], tensors: Mapping[str, object]) -> Path:
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
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, tensors = run(args)
    output = publish(args.output, manifest, tensors)
    print(json.dumps({"output": str(output), "events": manifest["event_count"], "private_targets_loaded": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
