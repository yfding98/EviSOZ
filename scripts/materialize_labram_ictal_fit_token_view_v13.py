#!/usr/bin/env python3
"""Broker one physical fit-only token view for v13 control training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus_fit_subset,
)
from src.soz.ictal_fit_only_consumer_v13 import (  # noqa: E402
    load_fit_only_target_artifact_v13,
)
from src.soz.ictal_fit_token_view_v13 import (  # noqa: E402
    materialize_fit_token_view_v13,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        required=True,
        choices=(*[f"fold{index}" for index in range(5)], "final"),
    )
    parser.add_argument("--source-training-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-source-training-token-corpus-index-sha256",
        type=_sha256,
        required=True,
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
    parser.add_argument("--fit-only-target-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-fit-only-target-manifest-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-fit-only-target-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(os.path.abspath(args.output_directory))
    if output.name in {"", ".", ".."} or not output.parent.is_dir():
        raise ValueError("fit-token output needs a concrete existing parent")
    if os.path.lexists(output):
        raise FileExistsError(f"fit-token output already exists: {output}")
    targets = load_fit_only_target_artifact_v13(
        args.fit_only_target_bundle,
        expected_manifest_sha256=args.expected_fit_only_target_manifest_sha256,
        expected_receipt_sha256=args.expected_fit_only_target_receipt_sha256,
    )
    if targets.manifest["selection"] != args.selection:
        raise ValueError("Fit target selection differs from CLI")
    if (
        args.expected_source_training_token_corpus_index_sha256
        != targets.manifest["training_corpus_index_sha256"]
    ):
        raise ValueError("Source corpus index differs from fit target authority")
    preprocessing = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=args.expected_preprocessing_selection_artifact_sha256,
        expected_protocol_receipt_sha256=args.expected_preprocessing_protocol_receipt_sha256,
    )
    fit = tuple(targets.manifest["fit_patient_ids"])
    gate = tuple(targets.manifest["i_gate_patient_ids_excluded_unopened"])
    corpus = load_formal_token_corpus_fit_subset(
        args.source_training_token_corpus,
        expected_index_sha256=args.expected_source_training_token_corpus_index_sha256,
        preprocessing_selection=preprocessing,
        patient_ids=fit,
        forbidden_patient_ids=gate,
        expected_selected_event_count=int(targets.manifest["fit_event_count"]),
        expected_training_bundle_manifest_sha256=str(
            targets.manifest["training_manifest_bundle_sha256"]
        ),
        expected_training_source_manifest_sha256=str(
            targets.manifest["training_manifest_sha256"]
        ),
    )
    preflight = {
        "schema_version": "soz_ictal_fit_token_view_preflight_v13",
        "selection": args.selection,
        "preflight_passed": True,
        "materialization_started": False,
        "fit_patient_count": len(fit),
        "fit_event_count": corpus.selected_event_count,
        "i_gate_patient_count": len(gate),
        "source_full_corpus_index_metadata_loaded_by_broker": True,
        "source_full_corpus_root_accessible_to_broker": True,
        "source_unselected_bundle_contents_opened_by_broker": False,
        "i_gate_signal_or_tokens_opened": False,
        "training_started": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0
    saved = materialize_fit_token_view_v13(
        output,
        source=corpus,
        fit_targets=targets,
    )
    print(
        json.dumps(
            {
                **preflight,
                "materialization_started": True,
                "physical_view_contains_fit_bundles_only": True,
                "path": str(saved.path),
                "manifest_sha256": saved.manifest_sha256,
                "receipt_sha256": saved.receipt_sha256,
                "training_started": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
