#!/usr/bin/env python3
"""Evaluate clinical EEG report language without leaking evaluation labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_report.language_quality import evaluate_language_quality


def _duplicate_safe_pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid constant {value!r}")


def _load_json(path: Path, context: str) -> object:
    if path.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{context} must be a regular file")
    return json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=_duplicate_safe_pairs,
        parse_constant=_invalid_constant,
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Auditable EEG-report language evaluation. BLEU/ROUGE are computed "
            "only for same-recording complete physician references."
        )
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--references",
        type=Path,
        help="Optional complete, paired physician-report reference manifest",
    )
    parser.add_argument(
        "--postfreeze-doctor-label-bundle",
        type=Path,
        help=(
            "Optional already-frozen structured doctor-label sidecar; summarized "
            "separately and never treated as report text"
        ),
    )
    parser.add_argument(
        "--meteor",
        action="store_true",
        help="Use locally installed NLTK METEOR; never downloads resources",
    )
    parser.add_argument(
        "--bertscore-local-model",
        type=Path,
        help="Optional local-only BERTScore model directory",
    )
    parser.add_argument(
        "--bertscore-num-layers",
        type=int,
        help="Required with --bertscore-local-model",
    )
    parser.add_argument(
        "--bertscore-model-domain",
        choices=("general", "medical_eeg"),
        help="Required operator declaration with --bertscore-local-model",
    )
    parser.add_argument(
        "--bertscore-device",
        default="cpu",
        help="Local inference device; defaults to cpu",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bertscore_requested = args.bertscore_local_model is not None
    if bertscore_requested != (args.bertscore_num_layers is not None):
        raise ValueError(
            "--bertscore-local-model and --bertscore-num-layers must be provided together"
        )
    if bertscore_requested != (args.bertscore_model_domain is not None):
        raise ValueError(
            "--bertscore-local-model and --bertscore-model-domain must be provided together"
        )
    candidates = _load_json(args.candidates, "candidate manifest")
    references = (
        _load_json(args.references, "reference manifest")
        if args.references is not None
        else None
    )
    doctor_bundle = (
        _load_json(args.postfreeze_doctor_label_bundle, "post-freeze doctor-label bundle")
        if args.postfreeze_doctor_label_bundle is not None
        else None
    )
    bertscore_config = None
    if bertscore_requested:
        bertscore_config = {
            "model_path": args.bertscore_local_model,
            "num_layers": args.bertscore_num_layers,
            "model_domain": args.bertscore_model_domain,
            "device": args.bertscore_device,
        }
    result = evaluate_language_quality(
        candidates,
        references=references,
        doctor_label_bundle=doctor_bundle,
        meteor_requested=args.meteor,
        bertscore_config=bertscore_config,
    )
    _atomic_json(args.output, result)
    summary = {
        "status": result["status"],
        "evaluation_id": result["evaluation_id"],
        "record_count": result["reference_free_cohort_summary"]["record_count"],
        "paired_reference_status": result["paired_complete_reference_metrics"]["status"],
        "paired_record_count": result["paired_complete_reference_metrics"][
            "paired_record_count"
        ],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
