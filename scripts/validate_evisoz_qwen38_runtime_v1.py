#!/usr/bin/env python3
"""Validate the local Qwen3.8-27B artifact without loading weights for inference.

This is an inventory/configuration probe, not a training or clinical-generation
entry point.  It verifies the immutable local model files, the advertised base
model, the language interface expected by the EviSOZ connector, and (when run
inside a vLLM environment) vLLM's model-config resolution.  A missing GPU is
recorded explicitly; CPU config resolution must never be reported as an
end-to-end generation check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/Qwen3.8-27B-FP8"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_qwen38_runtime_probe_v1_20260901"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _advertised_base_models(readme: Path) -> list[str]:
    text = readme.read_text(encoding="utf-8")
    # The model card uses a YAML front matter list.  Keep this deliberately
    # narrow so arbitrary prose cannot be mistaken for an authority field.
    match = re.search(r"(?ms)^base_model:\s*\n((?:^- .+\n?)+)", text)
    if match is None:
        return []
    return [line[2:].strip() for line in match.group(1).splitlines() if line.startswith("- ")]


def _indexed_shards(model: Path, index: dict[str, Any]) -> tuple[list[str], list[str]]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no non-empty weight_map")
    names = sorted({str(value) for value in weight_map.values()})
    missing = [name for name in names if not (model / name).is_file()]
    return names, missing


def _vllm_probe(model: Path, cache_root: Path) -> dict[str, Any]:
    """Resolve vLLM's config only; never instantiate an engine or read weights."""
    result: dict[str, Any] = {"available": False, "status": "not_attempted"}
    os.environ.setdefault("VLLM_CACHE_ROOT", str(cache_root / "vllm_cache"))
    try:
        import vllm  # type: ignore
        from vllm.config import ModelConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional env
        result.update({"status": "unavailable", "error_type": type(exc).__name__, "error": str(exc)})
        return result
    result["available"] = True
    result["vllm_version"] = getattr(vllm, "__version__", "unknown")
    try:
        config = ModelConfig(
            model=str(model),
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=2048,
        )
        hf_config = getattr(config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None)
        result.update(
            {
                "status": "config_resolved",
                "resolved_architecture": (
                    getattr(hf_config, "architectures", None) or [None]
                )[0],
                "resolved_model_type": getattr(hf_config, "model_type", None),
                "resolved_text_hidden_size": getattr(text_config, "hidden_size", None),
                "resolved_text_num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
                "resolved_text_max_position_embeddings": getattr(
                    text_config, "max_position_embeddings", None
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - depends on optional env
        result.update({"status": "config_resolution_failed", "error_type": type(exc).__name__, "error": str(exc)})
    return result


def build_receipt(model: Path, output: Path, *, probe_vllm: bool) -> dict[str, Any]:
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    readme_path = model / "README.md"
    required = [config_path, index_path, readme_path, model / "tokenizer.json", model / "tokenizer_config.json"]
    missing_required = [str(path.relative_to(model)) for path in required if not path.is_file()]
    if missing_required:
        raise FileNotFoundError(f"missing required model files: {missing_required}")

    config = _read_json(config_path)
    index = _read_json(index_path)
    shards, missing_shards = _indexed_shards(model, index)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("config.json has no text_config object")

    advertised = _advertised_base_models(readme_path)
    expected = {
        "model_family": "Qwen3.8-27B",
        "base_model_advertised": "Qwen/Qwen3.8-27B" in advertised,
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "language_hidden_size": text_config.get("hidden_size"),
        "language_layers": text_config.get("num_hidden_layers"),
        "language_context_length": text_config.get("max_position_embeddings"),
        "language_vocab_size": text_config.get("vocab_size"),
        "vision_out_hidden_size": (config.get("vision_config") or {}).get("out_hidden_size"),
    }
    contract_ok = (
        expected["base_model_advertised"]
        and expected["architecture"] == ["Qwen3_5ForConditionalGeneration"]
        and expected["model_type"] == "qwen3_5"
        and expected["language_hidden_size"] == 5120
        and expected["language_layers"] == 64
        and expected["language_context_length"] == 262144
        and expected["vision_out_hidden_size"] == 5120
        and not missing_shards
    )

    output.mkdir(parents=True, exist_ok=False)
    vllm = _vllm_probe(model, output) if probe_vllm else {"available": False, "status": "not_requested"}
    gpu = {"cuda_available": False, "end_to_end_generation_checked": False}
    try:
        import torch  # type: ignore
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["torch_version"] = getattr(torch, "__version__", "unknown")
        if gpu["cuda_available"]:
            gpu["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - optional dependency
        gpu.update({"torch_error_type": type(exc).__name__, "torch_error": str(exc)})

    receipt: dict[str, Any] = {
        "schema_version": "evisoz_qwen38_runtime_probe_v1",
        "probe_id": "EVISOZ-QWEN38-PROBE-" + _sha256(config_path)[:20],
        "status": "CONFIG_GO_GPU_UNVERIFIED" if contract_ok else "NO_GO",
        "model": {
            "path": str(model),
            "readme_base_models": advertised,
            "config_sha256": _sha256(config_path),
            "index_sha256": _sha256(index_path),
            "indexed_shard_count": len(shards),
            "indexed_shards": [
                {"name": name, "size_bytes": (model / name).stat().st_size}
                for name in shards
                if (model / name).is_file()
            ],
            "missing_shards": missing_shards,
        },
        "interface_contract": expected,
        "vllm_probe": vllm,
        "hardware_probe": gpu,
        "safety": {
            "weights_loaded_for_inference": False,
            "generation_executed": False,
            "training_authorized": False,
            "stage0_gate_changed": False,
            "qwen_sft_authorized": False,
            "eeg_to_qwen_alignment_authorized": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-vllm", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_receipt(args.model.resolve(), args.output.resolve(), probe_vllm=args.probe_vllm)
    print(json.dumps({"status": receipt["status"], "probe_id": receipt["probe_id"], "output": str(args.output.resolve()), "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
