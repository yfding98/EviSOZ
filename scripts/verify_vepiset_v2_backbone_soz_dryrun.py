#!/usr/bin/env python3
"""Dry-run VEPiSet v2 backbone initialization on the clinical SOZ interface.

This is not a clinical SOZ evaluation.  It verifies that the exported
VEPiSet-trained shared backbone can instantiate `integration_model_v2.py`,
load without unexpected keys, run a SOZ-style forward pass, and compute the
multi-task SOZ loss used by the existing trainer interface.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch


DEFAULT_INIT = Path(
    "outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_noamp20"
    "/vepiset_v2_backbone_init.pt"
)


def _add_project_code_to_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def _config_kwargs(config: Dict[str, Any], integration_config_cls: Any) -> Dict[str, Any]:
    valid = {field.name for field in dataclasses.fields(integration_config_cls)}
    kwargs = {key: value for key, value in config.items() if key in valid}
    if "brain_network_features" in kwargs:
        kwargs["brain_network_features"] = tuple(kwargs["brain_network_features"])
    kwargs["task_mode"] = "soz"
    kwargs["task_training_mode"] = "multitask"
    return kwargs


def _build_targets(batch_size: int, n_channels: int, n_regions: int) -> Dict[str, torch.Tensor]:
    soz = torch.zeros(batch_size, n_channels, dtype=torch.float32)
    for idx in range(batch_size):
        primary = (idx * 3) % n_channels
        secondary = min(primary + 1, n_channels - 1)
        soz[idx, primary] = 1.0
        if idx % 2 == 0:
            soz[idx, secondary] = 1.0

    region = torch.zeros(batch_size, n_regions, dtype=torch.float32)
    region[:, 0] = 1.0
    if n_regions > 3:
        region[1::2, 3] = 1.0

    hemisphere = torch.tensor([0 if idx % 3 == 0 else 1 if idx % 3 == 1 else 2 for idx in range(batch_size)])
    return {
        "soz": soz,
        "region": region,
        "hemisphere": hemisphere.long(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default=str(DEFAULT_INIT))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    _add_project_code_to_path()
    from models.integration_model_v2 import (  # pylint: disable=import-outside-toplevel
        IntegrationConfig,
        Lightweight_Transformer_BrainNetwork_Integration,
    )
    from models.dynamic_network_evolution import (  # pylint: disable=import-outside-toplevel
        DynamicNetworkEvolutionModel,
    )

    init_path = Path(args.init)
    if not init_path.exists():
        raise FileNotFoundError(init_path)
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise KeyError("Init checkpoint does not contain model_state")

    cfg = IntegrationConfig(**_config_kwargs(checkpoint.get("config", {}), IntegrationConfig))
    device = torch.device(args.device)
    model = Lightweight_Transformer_BrainNetwork_Integration(cfg).to(device)
    load_info = model.load_state_dict(state, strict=False)
    model.train()

    batch_size = int(args.batch_size)
    total_samples = int(cfg.patch_len) * (int(cfg.n_pre_patches) + int(cfg.n_post_patches))
    x = torch.randn(batch_size, int(cfg.n_channels), total_samples, device=device) * 1e-4
    onset = torch.full((batch_size,), 2.0, dtype=torch.float32, device=device)
    start = torch.zeros(batch_size, dtype=torch.float32, device=device)

    outputs = model(x, seizure_onset_sec=onset, window_start_sec=start)
    targets = _build_targets(batch_size, int(cfg.n_monopolar), int(cfg.n_regions))
    soz_targets = targets["soz"].to(device)
    region_targets = targets["region"].to(device)
    hemisphere_targets = targets["hemisphere"].to(device)

    valid_mask = DynamicNetworkEvolutionModel._build_valid_mask(
        outputs["valid_patch_counts"],
        outputs["transition_probs"].size(1),
    )
    aux = DynamicNetworkEvolutionModel.compute_auxiliary_targets(
        outputs["seizure_relative_time"],
        valid_mask,
    )
    loss, losses = model.compute_loss(
        outputs,
        soz_targets,
        region_targets=region_targets,
        hemisphere_targets=hemisphere_targets,
        transition_targets=aux["transition_targets"].to(device),
        pattern_targets=aux["pattern_targets"].to(device),
    )

    shape_checks = {
        "soz_probs": list(outputs["soz_probs"].shape),
        "region_probs": list(outputs["region_probs"].shape),
        "hemisphere_logits": list(outputs["hemisphere_logits"].shape),
        "transition_probs": list(outputs["transition_probs"].shape),
        "pattern_logits": list(outputs["pattern_logits"].shape),
    }
    expected_shapes = {
        "soz_probs": [batch_size, int(cfg.n_monopolar)],
        "region_probs": [batch_size, int(cfg.n_regions)],
        "hemisphere_logits": [batch_size, int(cfg.n_hemisphere_classes)],
        "transition_probs": [batch_size, int(cfg.n_pre_patches) + int(cfg.n_post_patches)],
        "pattern_logits": [batch_size, 3],
    }
    failures = []
    for name, expected in expected_shapes.items():
        if shape_checks.get(name) != expected:
            failures.append(f"{name}: expected {expected}, got {shape_checks.get(name)}")
    if not torch.isfinite(loss).item():
        failures.append(f"loss is not finite: {float(loss.detach().cpu())}")
    if len(state) <= 0:
        failures.append("no weights loaded from init checkpoint")
    if load_info.unexpected_keys:
        failures.append(f"unexpected keys while loading init: {load_info.unexpected_keys[:10]}")

    report = {
        "init": str(init_path),
        "dryrun_passed": not failures,
        "clinical_soz_claim_supported": False,
        "loaded_keys": len(state),
        "missing_keys": sorted(load_info.missing_keys),
        "unexpected_keys": sorted(load_info.unexpected_keys),
        "output_shapes": shape_checks,
        "loss": float(loss.detach().cpu()),
        "losses": {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if torch.is_tensor(value) and math.isfinite(float(value.detach().cpu()))
        },
        "failures": failures,
        "note": "Interface dry-run only; requires clinical SOZ labels for SOTA claims.",
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
