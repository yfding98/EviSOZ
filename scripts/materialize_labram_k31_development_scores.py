#!/usr/bin/env python3
"""Materialize target-free LaBraM k31 development score grids.

The command intentionally exposes no DeepSOZ target table/source CSV and no
caller-provided scores, labels, masks, event rows, or fold assignments.
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

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
    load_ictal_native_eval_manifest,
    load_ictal_native_eval_token_corpus,
)
from src.soz.ictal_recovery_evidence import (  # noqa: E402
    build_target_free_signal_timeline_view,
    load_target_free_ictal_oof_protocol,
    materialize_labram_k31_development_scores,
)
from src.soz.ictal_recovery_oof import (  # noqa: E402
    load_labram_k31_oof_recovery_run,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)


_SELECTIONS = ("fold0", "fold1", "fold2", "fold3", "fold4", "final")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def _run_spec(value: str) -> tuple[str, Path, str]:
    fields = value.split("=", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(
            "recovery run must use SELECTION=PATH=MANIFEST_SHA256"
        )
    selection, raw_path, raw_sha = fields
    if selection not in _SELECTIONS or not raw_path:
        raise argparse.ArgumentTypeError("selection must be fold0..fold4 or final")
    return selection, Path(raw_path), _sha256(raw_sha)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovery-run",
        action="append",
        type=_run_spec,
        required=True,
        help="repeat exactly once for fold0..fold4 and final",
    )
    parser.add_argument("--oof-protocol", type=Path, required=True)
    parser.add_argument(
        "--expected-oof-protocol-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-oof-protocol-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-signal-preflight-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-signal-preflight-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-target-v2-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-target-v2-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-target-v2-policy-sha256", type=_sha256, required=True
    )
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
    parser.add_argument("--source-train-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-source-train-token-index-sha256", type=_sha256, required=True
    )
    parser.add_argument("--source-dev-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-source-dev-manifest-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-source-dev-manifest-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--source-dev-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-source-dev-token-index-sha256", type=_sha256, required=True
    )
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = tuple(args.recovery_run)
    if len(specs) != 6:
        raise ValueError("Exactly six recovery-run specifications are required")
    by_selection = {selection: (path, digest) for selection, path, digest in specs}
    if set(by_selection) != set(_SELECTIONS) or len(by_selection) != 6:
        raise ValueError("Recovery runs must contain fold0..fold4 and final once")
    runs = tuple(
        load_labram_k31_oof_recovery_run(
            by_selection[selection][0],
            expected_manifest_sha256=by_selection[selection][1],
        )
        for selection in _SELECTIONS
    )

    protocol = load_target_free_ictal_oof_protocol(
        args.oof_protocol,
        expected_artifact_sha256=args.expected_oof_protocol_artifact_sha256,
        expected_protocol_receipt_sha256=args.expected_oof_protocol_receipt_sha256,
    )
    signal = load_bound_deepsoz_signal_preflight_artifact(
        args.signal_preflight_bundle,
        expected_artifact_sha256=args.expected_signal_preflight_artifact_sha256,
        expected_receipt_sha256=args.expected_signal_preflight_receipt_sha256,
    )
    timeline = build_target_free_signal_timeline_view(
        signal,
        protocol,
        expected_target_v2_artifact_sha256=(
            args.expected_target_v2_artifact_sha256
        ),
        expected_target_v2_receipt_sha256=(
            args.expected_target_v2_receipt_sha256
        ),
        expected_target_v2_policy_sha256=args.expected_target_v2_policy_sha256,
    )
    preprocessing = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        expected_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
    )
    source_train = load_formal_token_corpus(
        args.source_train_token_corpus,
        expected_index_sha256=args.expected_source_train_token_index_sha256,
        preprocessing_selection=preprocessing,
    )
    source_dev_manifest = load_ictal_native_eval_manifest(
        args.source_dev_manifest_bundle,
        signal,
        args.edf_root,
        expected_artifact_sha256=args.expected_source_dev_manifest_artifact_sha256,
        expected_receipt_sha256=args.expected_source_dev_manifest_receipt_sha256,
        expected_signal_artifact_sha256=(
            args.expected_signal_preflight_artifact_sha256
        ),
        expected_signal_receipt_sha256=args.expected_signal_preflight_receipt_sha256,
    )
    source_dev = load_ictal_native_eval_token_corpus(
        args.source_dev_token_corpus,
        source_dev_manifest,
        expected_index_sha256=args.expected_source_dev_token_index_sha256,
        expected_manifest_artifact_sha256=(
            args.expected_source_dev_manifest_artifact_sha256
        ),
        expected_manifest_receipt_sha256=(
            args.expected_source_dev_manifest_receipt_sha256
        ),
        expected_signal_artifact_sha256=(
            args.expected_signal_preflight_artifact_sha256
        ),
        expected_signal_receipt_sha256=args.expected_signal_preflight_receipt_sha256,
    )
    artifact = materialize_labram_k31_development_scores(
        recovery_runs=runs,
        protocol=protocol,
        timeline=timeline,
        source_train_corpus=source_train,
        source_dev_corpus=source_dev,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "artifact_sha256": artifact.artifact_sha256,
                "receipt_sha256": artifact.receipt_sha256,
                "source_train_shape": list(artifact.source_train_scores.shape),
                "source_dev_shape": list(artifact.source_dev_scores.shape),
                "development_only": True,
                "formal_promotion": False,
                "authorized_for_formal_evidence_or_reasoner": False,
                "target_vectors_loaded": False,
                "source_eval_signals_or_events_used": False,
                "private_data_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
