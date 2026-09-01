#!/usr/bin/env python3
"""Preflight or materialize the sealed LaBraM k31 I-gate logits.

The command has no target, mask, DeepSOZ, private-data, outcome, threshold,
metric, calibrator, or training argument.  While the v13 execution hold is in
force, ``--preflight-only`` reads only the minimal k31 inference projection,
preprocessing receipts, and master index.  It never imports or parses a legacy
recovery manifest, enumerate event directories, open token bundles, execute a
forward, or create output.  Non-preflight execution fails closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.ictal_gate_prediction_materialization_v13 import (  # noqa: E402
    EXPECTED_PRODUCER_ORDER,
    V13_EXECUTION_HOLD,
    V13_EXECUTION_HOLD_BLOCKERS,
    hash_gate_split_without_parsing,
    inspect_master_gate_index_metadata_v13,
)
from src.soz.ictal_k31_inference_projection_v13 import (  # noqa: E402
    load_k31_inference_projection_v13,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--expected-v5-split-sha256", type=_sha256, required=True)
    parser.add_argument("--preprocessing-selection-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-preprocessing-selection-artifact-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument(
        "--expected-preprocessing-protocol-receipt-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--master-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-master-token-corpus-index-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-master-manifest-bundle-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-master-manifest-source-sha256", type=_sha256, required=True
    )
    parser.add_argument("--k31-inference-projection", type=Path, required=True)
    parser.add_argument(
        "--expected-k31-inference-projection-manifest-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _progress(stage: str) -> None:
    print(
        json.dumps(
            {
                "stage": stage,
                "tusz_target_values_loaded": False,
                "tusz_target_masks_loaded": False,
                "deepsoz_identity_source_loaded": False,
                "deepsoz_target_source_loaded": False,
                "minimal_k31_inference_projection_loaded": (
                    stage != "hash_gate_split_without_json_parse"
                ),
                "legacy_recovery_bundle_loaded": False,
                "legacy_recovery_manifest_metadata_loaded": False,
                "legacy_native_evaluation_roster_metadata_loaded": False,
                "legacy_training_run_metrics_loaded": False,
                "private_eeg_loaded": False,
                "gate_outcomes_opened": False,
                "non_gate_token_bundles_opened": False,
                "gate_token_values_loaded": False,
                "evaluation": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _load_projection(args: argparse.Namespace):
    projection = load_k31_inference_projection_v13(
        args.k31_inference_projection,
        expected_manifest_sha256=(
            args.expected_k31_inference_projection_manifest_sha256
        ),
    )
    if tuple(item.selection for item in projection.producers) != EXPECTED_PRODUCER_ORDER:
        raise ValueError("Producer order changed")
    if projection.v5_split_sha256 != args.expected_v5_split_sha256:
        raise ValueError("Projection uses another frozen v5 split")
    return projection, projection.gate_patient_ids


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _progress("hash_gate_split_without_json_parse")
    split_sha = hash_gate_split_without_parsing(
        args.v5_split, expected_sha256=args.expected_v5_split_sha256
    )
    _progress("strict_load_minimal_k31_inference_projection")
    projection, gate_patients = _load_projection(args)
    _progress("strict_load_preprocessing_selection")
    preprocessing = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        expected_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
    )
    _progress("metadata_only_validate_master_index_without_bundle_enumeration")
    metadata = inspect_master_gate_index_metadata_v13(
        args.master_token_corpus,
        expected_index_sha256=args.expected_master_token_corpus_index_sha256,
        expected_master_bundle_manifest_sha256=(
            args.expected_master_manifest_bundle_sha256
        ),
        expected_master_source_manifest_sha256=(
            args.expected_master_manifest_source_sha256
        ),
        expected_preprocessing_selection_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        expected_preprocessing_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
        gate_patient_ids=gate_patients,
    )
    del preprocessing, projection
    if metadata["index_sha256"] != args.expected_master_token_corpus_index_sha256:
        raise RuntimeError("Metadata preflight index identity drifted")
    if split_sha != args.expected_v5_split_sha256:
        raise RuntimeError("Metadata preflight split identity drifted")
    if args.preflight_only:
        _progress("metadata_preflight_complete_no_token_no_forward_no_output")
        print(json.dumps(metadata, sort_keys=True), flush=True)
        return 0

    if V13_EXECUTION_HOLD:
        raise RuntimeError(
            "V13_EXECUTION_HOLD: candidate logits are only one incomplete Stage-A "
            "component; blockers=" + ",".join(V13_EXECUTION_HOLD_BLOCKERS)
        )
    raise RuntimeError(
        "Execution authorization path is intentionally absent until the v13 "
        "controls and probes are sealed"
    )


if __name__ == "__main__":
    raise SystemExit(main())
