#!/usr/bin/env python3
"""Run label-neutral EEG screening plus offline Qwen3.5 SOZ review.

This entry point is pinned to the local GPTQ-Int4 build whose base model is
``Qwen/Qwen3.5-35B-A3B``.  Manifest/TUSZ onset times are used only as temporal
navigation hints.  Source channel/region labels and AutoLabel conclusions are
not exposed to Qwen as answers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.auto_annotate.llm_soz_annotator import main as annotator_main  # noqa: E402


DEFAULT_MODEL = ROOT / "models" / "Qwen3.5-35B-A3B-GPTQ-Int4"


def _option_value(argv: Sequence[str], option: str) -> str | None:
    try:
        index = list(argv).index(option)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return str(argv[index + 1])


def _validate_complete_model(model_path: Path) -> None:
    required = (
        "README.md",
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (model_path / name).is_file()]
    # Hugging Face can leave stale resume markers under its private cache even
    # after every indexed final shard is present.  The weight index and final
    # files are authoritative; an incomplete file outside .cache still fails
    # closed because it may be an actual model artifact.
    incomplete = [
        path
        for path in model_path.rglob("*.incomplete")
        if ".cache" not in path.relative_to(model_path).parts
    ]
    referenced_shards: set[str] = set()
    index_error = ""
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            referenced_shards = {
                str(name) for name in payload.get("weight_map", {}).values()
            }
        except (OSError, TypeError, ValueError) as exc:
            index_error = str(exc)
    missing_shards = sorted(
        name for name in referenced_shards if not (model_path / name).is_file()
    )
    readme = ""
    try:
        readme = (model_path / "README.md").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        pass
    correct_base_model = "Qwen/Qwen3.5-35B-A3B" in readme
    if (
        missing
        or index_error
        or not referenced_shards
        or missing_shards
        or incomplete
        or not correct_base_model
    ):
        raise SystemExit(
            "Qwen3.5 model validation failed: "
            f"path={model_path}, missing={missing}, index_error={index_error!r}, "
            f"referenced_shards={len(referenced_shards)}, "
            f"missing_shards={missing_shards}, incomplete_files={len(incomplete)}, "
            f"correct_base_model={correct_base_model}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    dataset = _option_value(user_args, "--dataset")
    if dataset not in {"private", "tusz", "generic"}:
        raise SystemExit("--dataset private|tusz|generic is required")
    configured_model = os.environ.get("QWEN35_MODEL_PATH", "").strip()
    model_path = (
        Path(configured_model).expanduser() if configured_model else DEFAULT_MODEL
    ).resolve()
    _validate_complete_model(model_path)
    if "--output-dir" not in user_args:
        user_args.extend(
            ["--output-dir", str(ROOT / "outputs" / f"qwen35_soz_{dataset}")]
        )
    defaults = [
        "--inference-backend",
        "local_transformers",
        "--model",
        str(model_path),
        "--require-local-model-release",
        "qwen3.5",
        "--local-gptq-backend",
        "marlin",
        "--local-max-pixels",
        "501760",
        "--local-two-stage",
        "--local-constrained-json",
        "--local-core-max-tokens",
        "1300",
        "--local-narrative-max-tokens",
        "2400",
        "--local-validation-retries",
        "1",
        "--knowledge-top-k",
        "12",
        "--max-review-rounds",
        "2",
    ]
    return annotator_main(defaults + user_args)


if __name__ == "__main__":
    raise SystemExit(main())
