#!/usr/bin/env python3
"""Preflight or materialize locked target-free source-eval ictal evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.locked_source_eval_ictal import (  # noqa: E402
    materialize_locked_source_eval_ictal,
    preflight_locked_source_eval_ictal,
)


DEFAULT_HEAD = (
    ROOT / "outputs/tusz_ictal_concept_formal_v4_20260809/final/checkpoint"
)
DEFAULT_LABRAM_MODELING = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py"
)
DEFAULT_LABRAM_CHECKPOINT = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth"
)


def _sha256(value: str) -> str:
    text = str(value).strip()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--locked-roster", type=Path, required=True)
    parser.add_argument(
        "--expected-locked-roster-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, default=DEFAULT_HEAD)
    parser.add_argument(
        "--labram-modeling-path", type=Path, default=DEFAULT_LABRAM_MODELING
    )
    parser.add_argument(
        "--labram-checkpoint-path", type=Path, default=DEFAULT_LABRAM_CHECKPOINT
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "roster_directory": args.locked_roster,
        "expected_roster_artifact_sha256": (
            args.expected_locked_roster_artifact_sha256
        ),
        "head_checkpoint_directory": args.head_checkpoint,
        "tusz_root": args.tusz_root,
        "labram_modeling_path": args.labram_modeling_path,
        "labram_checkpoint_path": args.labram_checkpoint_path,
    }
    if args.preflight_only:
        if args.output_directory is not None:
            raise ValueError("--preflight-only does not accept --output-directory")
        result = preflight_locked_source_eval_ictal(**common, device=args.device)
    else:
        if args.output_directory is None:
            raise ValueError("Formal forward requires --output-directory")
        artifact = materialize_locked_source_eval_ictal(
            **common,
            output_directory=args.output_directory,
            device=args.device,
        )
        result = {
            "status": "published_locked_target_free_source_eval_ictal_evidence",
            "path": str(artifact.path),
            "manifest_sha256": artifact.manifest_sha256,
            "event_count": len(artifact.event_ids),
            "patient_count": len(artifact.patient_ids),
            "scores_shape": list(artifact.scores.shape),
            "pooled_scores_shape": list(artifact.pooled_scores.shape),
            "contains_tusz_channel_targets_or_masks": False,
            "contains_deepsoz_targets": False,
            "source_eval_label_release_used": False,
        }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
