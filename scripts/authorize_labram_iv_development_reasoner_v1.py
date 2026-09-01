#!/usr/bin/env python3
"""Strictly issue the target-free LaBraM I+V candidate capability.

This CLI has target-v2 identity SHA arguments but no target table, label,
private-data, source-eval, raw-logit, tensor, event-row, or fold-assignment
argument.  All evidence is reproduced by strict artifact loaders before the
candidate-only capability is atomically published.
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
from src.soz.development_reasoner import (  # noqa: E402
    issue_development_iv_evidence_capability,
    publish_development_iv_evidence_capability,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
    load_ictal_native_eval_manifest,
    load_ictal_native_eval_token_corpus,
)
from src.soz.ictal_recovery_evidence_v1_2 import (  # noqa: E402
    build_target_free_signal_timeline_view,
    load_labram_k31_development_score_artifact_v1_2,
    load_target_free_ictal_oof_protocol,
)
from src.soz.ictal_recovery_oof_v1_2 import (  # noqa: E402
    load_labram_k31_oof_recovery_run_v1_2,
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
            "v1.2 recovery run must use SELECTION=PATH=MANIFEST_SHA256"
        )
    selection, raw_path, raw_sha = fields
    if selection not in _SELECTIONS or not raw_path:
        raise argparse.ArgumentTypeError("selection must be fold0..fold4 or final")
    return selection, Path(raw_path), _sha256(raw_sha)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-run", action="append", type=_run_spec, required=True)
    parser.add_argument("--oof-protocol", type=Path, required=True)
    parser.add_argument("--expected-oof-protocol-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-oof-protocol-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument("--expected-signal-preflight-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-signal-preflight-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-target-v2-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-target-v2-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-target-v2-policy-sha256", type=_sha256, required=True)
    parser.add_argument("--preprocessing-selection-bundle", type=Path, required=True)
    parser.add_argument("--expected-preprocessing-selection-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-preprocessing-protocol-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--source-train-token-corpus", type=Path, required=True)
    parser.add_argument("--expected-source-train-token-index-sha256", type=_sha256, required=True)
    parser.add_argument("--source-dev-manifest-bundle", type=Path, required=True)
    parser.add_argument("--expected-source-dev-manifest-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-source-dev-manifest-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--source-dev-token-corpus", type=Path, required=True)
    parser.add_argument("--expected-source-dev-token-index-sha256", type=_sha256, required=True)
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--ictal-score-bundle", type=Path, required=True)
    parser.add_argument("--expected-ictal-score-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-ictal-score-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--source-train-vaq-bundle", type=Path, required=True)
    parser.add_argument("--expected-source-train-vaq-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--source-dev-vaq-bundle", type=Path, required=True)
    parser.add_argument("--expected-source-dev-vaq-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = tuple(args.recovery_run)
    if len(specs) != 6:
        raise ValueError("Exactly six v1.2 recovery-run specifications are required")
    by_selection = {selection: (path, digest) for selection, path, digest in specs}
    if set(by_selection) != set(_SELECTIONS) or len(by_selection) != 6:
        raise ValueError("v1.2 recovery runs must contain fold0..fold4 and final once")
    runs = tuple(
        load_labram_k31_oof_recovery_run_v1_2(
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
        expected_target_v2_artifact_sha256=args.expected_target_v2_artifact_sha256,
        expected_target_v2_receipt_sha256=args.expected_target_v2_receipt_sha256,
        expected_target_v2_policy_sha256=args.expected_target_v2_policy_sha256,
    )
    preprocessing = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=args.expected_preprocessing_selection_artifact_sha256,
        expected_protocol_receipt_sha256=args.expected_preprocessing_protocol_receipt_sha256,
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
        expected_signal_artifact_sha256=args.expected_signal_preflight_artifact_sha256,
        expected_signal_receipt_sha256=args.expected_signal_preflight_receipt_sha256,
    )
    source_dev = load_ictal_native_eval_token_corpus(
        args.source_dev_token_corpus,
        source_dev_manifest,
        expected_index_sha256=args.expected_source_dev_token_index_sha256,
        expected_manifest_artifact_sha256=args.expected_source_dev_manifest_artifact_sha256,
        expected_manifest_receipt_sha256=args.expected_source_dev_manifest_receipt_sha256,
        expected_signal_artifact_sha256=args.expected_signal_preflight_artifact_sha256,
        expected_signal_receipt_sha256=args.expected_signal_preflight_receipt_sha256,
    )
    ictal = load_labram_k31_development_score_artifact_v1_2(
        args.ictal_score_bundle,
        recovery_runs=runs,
        protocol=protocol,
        timeline=timeline,
        source_train_corpus=source_train,
        source_dev_corpus=source_dev,
        expected_artifact_sha256=args.expected_ictal_score_artifact_sha256,
        expected_receipt_sha256=args.expected_ictal_score_receipt_sha256,
    )
    capability = issue_development_iv_evidence_capability(
        ictal_artifact=ictal,
        source_train_vaq_bundle=args.source_train_vaq_bundle,
        expected_source_train_vaq_manifest_sha256=(
            args.expected_source_train_vaq_manifest_sha256
        ),
        source_dev_vaq_bundle=args.source_dev_vaq_bundle,
        expected_source_dev_vaq_manifest_sha256=(
            args.expected_source_dev_vaq_manifest_sha256
        ),
    )
    published = publish_development_iv_evidence_capability(
        capability, args.output_directory
    )
    print(
        json.dumps(
            {
                "path": str(published.path),
                "manifest_sha256": published.manifest_sha256,
                "authorization_receipt_sha256": (
                    published.authorization_receipt_sha256
                ),
                "source_train_events": capability.source_train.evidence.batch_size,
                "source_train_patients": len(capability.source_train.patient_ids),
                "source_dev_events": capability.source_dev.evidence.batch_size,
                "source_dev_patients": len(capability.source_dev.patient_ids),
                "target_values_loaded": False,
                "candidate_reasoner_input_authorized": True,
                "formal_reasoner_authorized": False,
                "formal_promotion": False,
                "source_eval_used": False,
                "private_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
