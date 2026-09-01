#!/usr/bin/env python3
"""Run one or more SOZ events through a deployed local Qwen3.6 vLLM server."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.auto_annotate.llm_soz_annotator import main as annotator_main  # noqa: E402


def _option_value(argv: Sequence[str], option: str) -> str | None:
    try:
        index = list(argv).index(option)
    except ValueError:
        return None
    return str(argv[index + 1]) if index + 1 < len(argv) else None


def main(argv: Sequence[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    dataset = _option_value(user_args, "--dataset")
    if dataset not in {"private", "tusz", "generic"}:
        raise SystemExit("--dataset private|tusz|generic is required")
    base_url = os.environ.get(
        "QWEN36_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"
    ).strip()
    served_model = os.environ.get(
        "QWEN36_VLLM_MODEL_NAME", "qwen36-soz"
    ).strip()
    if "--output-dir" not in user_args:
        user_args.extend(
            ["--output-dir", str(ROOT / "outputs" / f"qwen36_vllm_{dataset}")]
        )
    defaults = [
        "--inference-backend",
        "vllm",
        "--api-base-url",
        base_url,
        "--api-mode",
        "chat",
        "--model",
        served_model,
        "--require-local-model-release",
        "qwen3.6",
        "--local-two-stage",
        "--local-constrained-json",
        "--local-core-max-tokens",
        "1300",
        "--local-narrative-max-tokens",
        "2400",
        "--local-validation-retries",
        "2",
        "--api-timeout-s",
        "1800",
        "--api-retries",
        "2",
        "--api-image-max-edge",
        "1280",
        "--api-image-jpeg-quality",
        "82",
        "--knowledge-top-k",
        "12",
        "--max-review-rounds",
        "2",
    ]
    return annotator_main(defaults + user_args)


if __name__ == "__main__":
    raise SystemExit(main())
