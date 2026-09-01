#!/usr/bin/env python3
"""Seal one k31 selection's fit-only TUSZ involvement targets.

Preflight reads only pinned JSON metadata and NumPy headers.  Materialization
uses explicit C-order ``os.pread`` row ranges and never invokes ``np.load`` on
the legacy monolithic source arrays.
"""

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

from src.soz.data.tusz_training import load_tusz_ictal_training_manifest  # noqa: E402
from src.soz.ictal_fit_only_targets_v13 import (  # noqa: E402
    load_source_target_index_v13,
    materialize_fit_only_target_artifact_v13,
    selected_fit_source_rows,
)
from src.soz.ictal_recovery_oof_v1_2 import (  # noqa: E402
    load_labram_k31_oof_recovery_run_v1_2,
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
    parser.add_argument("--k31-v1-2-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-k31-v1-2-manifest-sha256", type=_sha256, required=True
    )
    parser.add_argument("--training-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-training-manifest-bundle-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-training-manifest-source-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-training-token-corpus-index-sha256", type=_sha256, required=True
    )
    parser.add_argument("--source-formal-v4-target-snapshot", type=Path, required=True)
    parser.add_argument(
        "--expected-source-target-snapshot-manifest-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument(
        "--expected-source-target-snapshot-receipt-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(os.path.abspath(args.output_directory))
    if output.name in {"", ".", ".."} or not output.parent.is_dir():
        raise ValueError("fit-only target output needs a concrete existing parent")
    if os.path.lexists(output):
        raise FileExistsError(f"fit-only target output already exists: {output}")
    k31 = load_labram_k31_oof_recovery_run_v1_2(
        args.k31_v1_2_bundle,
        expected_manifest_sha256=args.expected_k31_v1_2_manifest_sha256,
    )
    if k31.manifest["selection"] != args.selection:
        raise ValueError("CLI selection differs from strict-loaded k31")
    manifest = load_tusz_ictal_training_manifest(
        args.training_manifest_bundle,
        expected_bundle_manifest_sha256=args.expected_training_manifest_bundle_sha256,
        expected_source_manifest_sha256=args.expected_training_manifest_source_sha256,
    )
    if manifest.manifest_sha256 != k31.manifest["training_manifest_sha256"]:
        raise ValueError("Training manifest differs from k31")
    if (
        args.expected_training_token_corpus_index_sha256
        != k31.manifest["training_corpus_index_sha256"]
    ):
        raise ValueError("Training token index differs from k31")
    source = load_source_target_index_v13(
        args.source_formal_v4_target_snapshot,
        expected_manifest_sha256=args.expected_source_target_snapshot_manifest_sha256,
        expected_receipt_sha256=args.expected_source_target_snapshot_receipt_sha256,
    )
    if (
        source.manifest_sha256 != k31.manifest["target_snapshot_manifest_sha256"]
        or source.receipt_sha256 != k31.manifest["target_snapshot_receipt_sha256"]
    ):
        raise ValueError("Source target identity differs from k31")
    selected = selected_fit_source_rows(
        source,
        manifest,
        fit_patient_ids=k31.manifest["training_public_patient_ids"],
        forbidden_i_gate_patient_ids=k31.manifest[
            "i_gate_patient_ids_excluded_unopened"
        ],
    )
    preflight = {
        "schema_version": "soz_ictal_fit_only_target_preflight_v13",
        "selection": args.selection,
        "preflight_passed": True,
        "materialization_started": False,
        "fit_patient_count": len(k31.manifest["training_public_patient_ids"]),
        "fit_event_count": len(selected),
        "i_gate_patient_count": len(
            k31.manifest["i_gate_patient_ids_excluded_unopened"]
        ),
        "source_json_identity_loaded": True,
        "source_npy_headers_loaded": True,
        "source_full_tensor_hashes_computed": False,
        "source_full_arrays_loaded": False,
        "source_full_arrays_mapped": False,
        "source_data_rows_read": 0,
        "broker_full_training_manifest_metadata_loaded": True,
        "broker_gate_target_derived_hashes_counts_loaded": True,
        "broker_legacy_k31_full_manifest_loaded": True,
        "broker_legacy_k31_native_roster_metrics_metadata_loaded": True,
        "broker_legacy_k31_checkpoint_weights_loaded": True,
        "broker_gate_target_values_read": False,
        "broker_gate_target_masks_read": False,
        "i_gate_target_row_byte_ranges_read": False,
        "i_gate_target_values_materialized": False,
        "deepsoz_target_source_loaded": False,
        "private_labels_used": False,
        "training_started": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0
    saved = materialize_fit_only_target_artifact_v13(
        output,
        source=source,
        training_manifest=manifest,
        training_bundle_manifest_sha256=args.expected_training_manifest_bundle_sha256,
        training_corpus_index_sha256=args.expected_training_token_corpus_index_sha256,
        k31_manifest=k31.manifest,
        k31_manifest_sha256=k31.manifest_sha256,
    )
    print(
        json.dumps(
            {
                **preflight,
                "materialization_started": True,
                "source_data_rows_read": len(selected),
                "fit_rows_only_pread": True,
                "path": str(saved.path),
                "manifest_sha256": saved.manifest_sha256,
                "receipt_sha256": saved.receipt_sha256,
                "i_gate_target_row_byte_ranges_read": False,
                "i_gate_target_values_materialized": False,
                "training_started": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
