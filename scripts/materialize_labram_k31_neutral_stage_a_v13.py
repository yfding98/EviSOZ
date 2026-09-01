#!/usr/bin/env python3
"""Preflight the target-neutral v13 LaBraM Stage-A runtime.

Formal execution remains held.  This command accepts only the minimal k31
projection, physical gate-only token view, parameter-only control projection,
and a hash-pinned split file.  It has no label, target, evaluation, clinical,
threshold, calibration, or source-broker input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v13_neutral.control_parameters import load_control_parameter_projection
from v13_neutral.control_head_projection import load_control_head_projection
from v13_neutral.projection import load_k31_projection
from v13_neutral.stage_a import V13_EXECUTION_HOLD, materialize_stage_a, prepare_stage_a
from v13_neutral.token_view import load_gate_token_view


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return text


def _hash_without_parsing(path: Path, expected: str) -> str:
    source = path.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError("split must be a regular file")
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("split changed while hash-pinned")
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError("split SHA mismatch")
    return actual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--expected-v5-split-sha256", type=_sha, required=True)
    parser.add_argument("--k31-inference-projection", type=Path, required=True)
    parser.add_argument("--expected-k31-inference-projection-manifest-sha256", type=_sha, required=True)
    parser.add_argument("--physical-gate-token-view", type=Path, required=True)
    parser.add_argument("--expected-physical-gate-token-view-manifest-sha256", type=_sha, required=True)
    parser.add_argument("--control-parameter-projection", type=Path, required=True)
    parser.add_argument("--expected-control-parameter-projection-manifest-sha256", type=_sha, required=True)
    parser.add_argument("--control-head-projection", type=Path, required=True)
    parser.add_argument("--expected-control-head-projection-manifest-sha256", type=_sha, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split_sha = _hash_without_parsing(args.v5_split, args.expected_v5_split_sha256)
    projection = load_k31_projection(
        args.k31_inference_projection,
        expected_manifest_sha256=args.expected_k31_inference_projection_manifest_sha256,
    )
    token_view = load_gate_token_view(
        args.physical_gate_token_view,
        expected_manifest_sha256=args.expected_physical_gate_token_view_manifest_sha256,
    )
    controls = load_control_parameter_projection(
        args.control_parameter_projection,
        expected_manifest_sha256=args.expected_control_parameter_projection_manifest_sha256,
    )
    control_heads = load_control_head_projection(
        args.control_head_projection,
        expected_manifest_sha256=args.expected_control_head_projection_manifest_sha256,
    )
    if split_sha != projection.v5_split_sha256:
        raise ValueError("Hash-pinned split differs from inference projection")
    prepared = prepare_stage_a(
        projection=projection,
        token_view=token_view,
        control_heads=control_heads,
        control_parameters=controls,
    )
    payload = {
        "schema_version": "soz_labram_k31_neutral_stage_a_preflight_v13_1",
        "preparation_sha256": prepared.preparation_sha256,
        "projection_manifest_sha256": projection.manifest_sha256,
        "token_view_manifest_sha256": token_view.manifest_sha256,
        "control_parameter_manifest_sha256": controls.manifest_sha256,
        "control_head_manifest_sha256": control_heads.manifest_sha256,
        "split_bytes_hashed_only": True,
        "split_json_parsed": False,
        "target_values_loaded": False,
        "target_masks_loaded": False,
        "clinical_data_loaded": False,
        "prediction_forward_executed": False,
        "output_published": False,
        "v13_execution_hold": V13_EXECUTION_HOLD,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if args.preflight_only:
        return 0
    if args.output_directory is None:
        raise ValueError("--output-directory is required outside preflight")
    materialize_stage_a(prepared=prepared, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
