#!/usr/bin/env python3
"""Export a VEPiSet-trained integration_model_v2 checkpoint for SOZ finetuning.

The strict VEPiSet result is an IED spatial-distribution proxy, not clinical
SOZ supervision.  This exporter therefore defaults to a backbone-only package:
shared temporal, TimeFilter, brain-network evolution, and fusion weights are
kept, while VEPiSet task heads are removed.  The resulting checkpoint is meant
as an initialization asset for a later clinical SOZ-labeled experiment.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


DEFAULT_SOURCE = Path(
    "outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_noamp20"
    "/best_model.pt"
)
DEFAULT_SUMMARY = Path(
    "outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_main_patientprior_conservative_macro_valacc87"
    "/strict_main_summary.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_noamp20"
    "/vepiset_v2_backbone_init.pt"
)

SHARED_PREFIXES: Tuple[str, ...] = (
    "patching.",
    "backbone.",
    "timefilter.",
    "brain_timefilter.",
    "net_evolution.",
    "fusion.",
)
WEAK_SOZ_PREFIXES: Tuple[str, ...] = (
    "soz_head.",
)
TASK_PREFIXES: Tuple[str, ...] = (
    "ied_",
    "ied_head.",
    "ied_binary_head.",
    "ied_spatial_head.",
    "raw_ied_",
    "morph_ied_",
    "state_ied_",
    "stage_head.",
)


def _add_project_code_to_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def _clean_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {"repr": repr(config)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    return value


def _starts_with_any(key: str, prefixes: Iterable[str]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def _select_state(
    state: Dict[str, torch.Tensor],
    include_weak_soz_head: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    keep_prefixes = SHARED_PREFIXES + (WEAK_SOZ_PREFIXES if include_weak_soz_head else ())
    selected: Dict[str, torch.Tensor] = {}
    excluded_by_prefix: Dict[str, int] = {}
    excluded_weak_soz = 0
    skipped_unmatched = 0

    for raw_key, value in state.items():
        key = _clean_key(raw_key)
        if _starts_with_any(key, TASK_PREFIXES):
            prefix = next(prefix for prefix in TASK_PREFIXES if key.startswith(prefix))
            excluded_by_prefix[prefix] = excluded_by_prefix.get(prefix, 0) + 1
            continue
        if not include_weak_soz_head and _starts_with_any(key, WEAK_SOZ_PREFIXES):
            excluded_weak_soz += 1
            continue
        if _starts_with_any(key, keep_prefixes):
            selected[key] = value.detach().cpu() if torch.is_tensor(value) else value
            continue
        skipped_unmatched += 1

    stats = {
        "mode": "shared_plus_weak_soz_head" if include_weak_soz_head else "backbone_only",
        "source_keys": len(state),
        "exported_keys": len(selected),
        "skipped_unmatched_keys": skipped_unmatched,
        "excluded_weak_soz_head_keys": excluded_weak_soz,
        "excluded_task_keys": excluded_by_prefix,
        "kept_prefixes": list(keep_prefixes),
        "excluded_prefixes": list(TASK_PREFIXES),
    }
    return selected, stats


def _verify_load(export_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    _add_project_code_to_path()
    from models.integration_model_v2 import (  # pylint: disable=import-outside-toplevel
        IntegrationConfig,
        Lightweight_Transformer_BrainNetwork_Integration,
    )

    valid_fields = {field.name for field in dataclasses.fields(IntegrationConfig)}
    cfg_kwargs = {key: value for key, value in config.items() if key in valid_fields}
    if "brain_network_features" in cfg_kwargs:
        cfg_kwargs["brain_network_features"] = tuple(cfg_kwargs["brain_network_features"])
    model = Lightweight_Transformer_BrainNetwork_Integration(IntegrationConfig(**cfg_kwargs))
    checkpoint = torch.load(export_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    info = model.load_state_dict(state, strict=False)
    return {
        "loaded_keys": len(state),
        "missing_keys": sorted(info.missing_keys),
        "unexpected_keys": sorted(info.unexpected_keys),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--include-weak-soz-head", action="store_true")
    parser.add_argument("--verify-load", action="store_true")
    args = parser.parse_args()

    _add_project_code_to_path()

    source_path = Path(args.source)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise KeyError("Source checkpoint does not contain model_state/state_dict")

    selected_state, stats = _select_state(
        state,
        include_weak_soz_head=bool(args.include_weak_soz_head),
    )
    if not selected_state:
        raise RuntimeError("No weights selected for export")

    config = _config_to_dict(checkpoint.get("config", {}))
    strict_summary: Dict[str, Any] = {}
    if summary_path.exists():
        strict_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    payload = {
        "model_state": selected_state,
        "config": config,
        "source_checkpoint": str(source_path),
        "source_strict_summary": str(summary_path) if summary_path.exists() else "",
        "export_stats": stats,
        "clinical_soz_claim_supported": False,
        "intended_use": (
            "Initialization for later patient-disjoint clinical SOZ finetuning. "
            "This checkpoint alone is not clinical SOZ validation."
        ),
        "vepiset_proxy_metrics": strict_summary.get("test_window_metrics", {}),
        "vepiset_claim_boundary": strict_summary.get("requirement_checks", {}),
        "source_args": checkpoint.get("args", {}),
        "source_val_metrics": checkpoint.get("val_metrics", {}),
        "class_names": checkpoint.get("class_names", []),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    report = {
        "output": str(output_path),
        "source": str(source_path),
        "export_stats": stats,
        "config": _jsonable(config),
        "clinical_soz_claim_supported": False,
    }
    if args.verify_load:
        report["verify_load"] = _verify_load(output_path, config)

    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    sidecar.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
