#!/usr/bin/env python3
"""Run a synthetic, CPU-only forward probe for the pinned public ELM encoder.

This is not a teacher-candidate or training entry point.  It loads only the
public ELM encoder checkpoints from an external artifact root and feeds them
zero tensors.  No EviSOZ patient/cache/report path is opened.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/tmp/evisoz_elm_source")
DEFAULT_ARTIFACTS = Path(
    "/mnt/hd1/dyf/workspace/laptop/EviSOZ_artifacts/"
    "elm_public_artifacts_v1_20260901_r1"
)
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_elm_runtime_probe_v1_20260901_r1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The repository has a real ``code`` package alongside Python's stdlib module
# of the same name.  Initialize the repository package before importing the
# EviSOZ package, whose validators may reference ``code.soz_pre``.
if "code" not in sys.modules or not hasattr(sys.modules["code"], "__path__"):
    code_init = ROOT / "code" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "code", code_init, submodule_search_locations=[str(ROOT / "code")]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot initialize repository code package")
    code_module = importlib.util.module_from_spec(spec)
    sys.modules["code"] = code_module
    spec.loader.exec_module(code_module)

from src.evisoz.data.artifact_ref import (  # noqa: E402
    build_raw_artifact_ref,
    canonical_json_bytes,
    sha256_bytes,
)
from src.evisoz.forge.elm_runtime_probe import (  # noqa: E402
    build_elm_runtime_probe_receipt,
)


PINNED_COMMIT = "fcd929a57ce3dc9a409be37a71f4ee80ee59979d"


def _git_head(source: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _ref(path: Path, *, kind: str, schema: str, media_type: str) -> dict[str, Any]:
    return build_raw_artifact_ref(
        path.read_bytes(),
        artifact_kind=kind,
        media_type=media_type,
        payload_schema_version=schema,
    )


def _build_encoder(source: Path, config: dict[str, Any]):
    # Import the exact pinned ELM implementation, not a reimplementation in
    # this repository.  The caller has already verified the source commit.
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from models.models import EEG_ResNet  # type: ignore

    model_cfg = config["model"]
    return EEG_ResNet(
        in_channels=model_cfg["in_channels"],
        conv1_params=model_cfg["encoder_conv1_params"],
        n_blocks=model_cfg["encoder_blocks"],
        res_params=model_cfg["encoder_res_params"],
        res_pool_size=model_cfg["encoder_pool_size"],
        dropout_p=model_cfg["encoder_dropout_p"],
        res_dropout_p=model_cfg["res_dropout_p"],
        proj_size=model_cfg["ELM"]["eeg_proj_size"],
    )


def build_receipt(source: Path, artifacts: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    artifacts = artifacts.resolve(strict=True)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("ELM source must be a regular directory")
    if artifacts.is_symlink() or not artifacts.is_dir():
        raise ValueError("ELM artifact root must be a regular directory")
    head = _git_head(source)
    if head != PINNED_COMMIT:
        raise ValueError(f"ELM source commit mismatch: {head}")
    requirements = source / "requirements.txt"
    if not requirements.is_file() or requirements.is_symlink():
        raise ValueError("ELM source requirements.txt is missing")

    model_artifacts: dict[str, Any] = {}
    variant_rows: list[dict[str, Any]] = []
    torch.set_num_threads(1)
    for variant, samples in (("5s", 500), ("60s", 6000)):
        config_path = artifacts / f"config_{variant}.yaml"
        checkpoint_path = artifacts / f"model_0_checkpoint_{variant}.pt"
        if not config_path.is_file() or config_path.is_symlink():
            raise FileNotFoundError(config_path)
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            raise FileNotFoundError(checkpoint_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if type(config) is not dict:
            raise ValueError(f"ELM config {variant} is not an object")
        model_cfg = config.get("model")
        if not isinstance(model_cfg, dict) or model_cfg.get("in_channels") != 20:
            raise ValueError(f"ELM {variant} input channel contract drifted")
        if model_cfg.get("n_time_samples") != samples:
            raise ValueError(f"ELM {variant} time-sample contract drifted")
        encoder = _build_encoder(source, config)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state, strict=True)
        encoder.eval()
        x = torch.zeros((1, 20, samples), dtype=torch.float32)
        with torch.no_grad():
            raw_a, projected_a = encoder(x)
            raw_b, projected_b = encoder(x)
        if not torch.isfinite(raw_a).all() or not torch.isfinite(projected_a).all():
            raise ValueError(f"ELM {variant} produced non-finite output")
        variant_rows.append(
            {
                "variant": variant,
                "input_shape": [1, 20, samples],
                "raw_embedding_shape": list(raw_a.shape),
                "projected_embedding_shape": list(projected_a.shape),
                "finite": True,
                "repeat_exact": bool(torch.equal(raw_a, raw_b) and torch.equal(projected_a, projected_b)),
            }
        )
        model_artifacts[variant] = {
            "config_ref": _ref(
                config_path,
                kind="elm_teacher_config",
                schema="elm_config_xy_v1",
                media_type="application/x-yaml",
            ),
            "checkpoint_ref": _ref(
                checkpoint_path,
                kind="elm_teacher_checkpoint",
                schema="elm_pytorch_checkpoint_v1",
                media_type="application/vnd.pytorch+zip",
            ),
        }

    source_payload = {
        "repository": "https://github.com/SamGijsen/ELM",
        "commit": head,
        "source_commit_verified": True,
        "source_root": str(source),
        "model_artifacts": model_artifacts,
        "software": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "source_requirements_sha256": sha256_bytes(requirements.read_bytes()),
        },
    }
    return build_elm_runtime_probe_receipt(
        source=source_payload,
        variants=variant_rows,
        probe={
            "input_kind": "synthetic_zeros",
            "batch_size": 1,
            "patient_data_opened": False,
            "forward_count": 4,
        },
        safety={
            "weights_loaded_for_synthetic_forward": True,
            "candidate_cache_materialized": False,
            "large_scale_teacher_inference": False,
            "training": False,
            "optimizer": False,
            "patient_data": False,
            "physician_report_text": False,
            "qwen_generation": False,
            "training_authorized": False,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    receipt = build_receipt(args.source, args.artifacts)
    output.mkdir(parents=True)
    (output / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "probe_id": receipt["probe_id"],
                "receipt_sha256": receipt["receipt_sha256"],
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
