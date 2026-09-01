#!/usr/bin/env python3
"""Train one fold-bound formal TUSZ ictal-involvement concept producer.

The frozen DeepSOZ registry is read only to reconstruct and verify its OOF
patient rosters.  SOZ channel vectors never enter the ictal head, its loss, or
its native-task metrics.  The command exposes no optimizer, epoch, seed, or
model-selection knobs.  All artifact hashes, including both external formal
token-corpus index hashes, are mandatory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Sequence


# This literal must execute before any project/torch import.  It is checked
# against the typed training-policy constant immediately after local imports.
_CLI_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _bootstrap_cublas_workspace_config() -> None:
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CLI_CUBLAS_WORKSPACE_CONFIG
        return
    if observed != _CLI_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Refusing conflicting CUBLAS_WORKSPACE_CONFIG: required exact "
            f"{_CLI_CUBLAS_WORKSPACE_CONFIG!r}, observed {observed!r}"
        )


_bootstrap_cublas_workspace_config()


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.derive_tusz_ictal_oof_fold_manifests import (  # noqa: E402
    load_bound_deepsoz_registry,
)
from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from src.soz.concept_oof import load_ictal_concept_oof_protocol  # noqa: E402
from src.soz.concept_run import ICTAL_CUBLAS_WORKSPACE_CONFIG  # noqa: E402
from src.soz.data.public_ledger_builder import (  # noqa: E402
    load_tusz_deepsoz_public_ledger_build,
)
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
)
from src.soz.ictal_gate_policy import (  # noqa: E402
    load_ictal_promotion_gate_policy,
)
from src.soz.ictal_production import (  # noqa: E402
    load_ictal_production_run,
    train_formal_ictal_production_run,
    validate_ictal_production_selection,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
    load_ictal_native_eval_manifest,
    load_ictal_native_eval_token_corpus,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)


if _CLI_CUBLAS_WORKSPACE_CONFIG != ICTAL_CUBLAS_WORKSPACE_CONFIG:
    raise RuntimeError("CLI and typed ictal CUBLAS policies disagree")


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_arg(value: str) -> str:
    normalized = str(value).strip()
    if not _SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256 digest")
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one fixed-policy formal ictal-involvement head and report "
            "held-out native TUSZ edge-time metrics"
        )
    )
    parser.add_argument(
        "--selection",
        required=True,
        choices=("fold0", "fold1", "fold2", "fold3", "fold4", "final"),
    )
    parser.add_argument(
        "--promotion-gate-policy-bundle", type=Path, required=True
    )
    parser.add_argument(
        "--expected-promotion-gate-policy-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-promotion-gate-policy-bundle-receipt-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--oof-protocol", type=Path, required=True)
    parser.add_argument(
        "--expected-oof-protocol-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-oof-protocol-receipt-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--public-ledger", type=Path, required=True)
    parser.add_argument(
        "--expected-public-ledger-bundle-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-public-ledger-build-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--deepsoz-source-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-deepsoz-source-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument("--deepsoz-split-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-split-manifest-sha256", type=_sha256_arg, required=True
    )

    parser.add_argument("--training-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-training-manifest-bundle-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-training-manifest-source-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--training-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-training-token-corpus-index-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--preprocessing-selection-bundle", type=Path, required=True
    )
    parser.add_argument(
        "--expected-preprocessing-selection-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-preprocessing-protocol-receipt-sha256",
        type=_sha256_arg,
        required=True,
    )

    parser.add_argument(
        "--native-evaluation-manifest-bundle", type=Path, required=True
    )
    parser.add_argument(
        "--expected-native-evaluation-manifest-bundle-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-native-evaluation-manifest-source-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--native-evaluation-token-corpus", type=Path, required=True
    )
    parser.add_argument(
        "--expected-native-evaluation-token-corpus-index-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--native-evaluation-signal-preflight-bundle", type=Path
    )
    parser.add_argument(
        "--expected-native-evaluation-signal-preflight-artifact-sha256",
        type=_sha256_arg,
    )
    parser.add_argument(
        "--expected-native-evaluation-signal-preflight-receipt-sha256",
        type=_sha256_arg,
    )
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "strictly load and validate every fold/policy/corpus binding, then "
            "exit before model construction or optimization"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    preprocessing_selection = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        expected_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
    )
    promotion_gate_policy = load_ictal_promotion_gate_policy(
        args.promotion_gate_policy_bundle,
        expected_artifact_sha256=(
            args.expected_promotion_gate_policy_artifact_sha256
        ),
        expected_receipt_sha256=(
            args.expected_promotion_gate_policy_bundle_receipt_sha256
        ),
    )
    registry, _ = load_bound_deepsoz_registry(
        args.deepsoz_source_csv,
        args.deepsoz_split_csv,
        expected_source_sha256=args.expected_deepsoz_source_sha256,
        expected_split_sha256=args.expected_split_manifest_sha256,
    )
    public_artifact = load_tusz_deepsoz_public_ledger_build(
        args.public_ledger,
        expected_bundle_sha256=args.expected_public_ledger_bundle_sha256,
        expected_build_sha256=args.expected_public_ledger_build_sha256,
    )
    protocol_artifact = load_ictal_concept_oof_protocol(
        args.oof_protocol,
        registry,
        public_artifact,
        expected_artifact_sha256=(
            args.expected_oof_protocol_artifact_sha256
        ),
        expected_protocol_sha256=(
            args.expected_oof_protocol_receipt_sha256
        ),
    )
    training_manifest = load_tusz_ictal_training_manifest(
        args.training_manifest_bundle,
        expected_bundle_manifest_sha256=(
            args.expected_training_manifest_bundle_sha256
        ),
        expected_source_manifest_sha256=(
            args.expected_training_manifest_source_sha256
        ),
    )
    training_corpus = load_formal_token_corpus(
        args.training_token_corpus,
        expected_index_sha256=(
            args.expected_training_token_corpus_index_sha256
        ),
        preprocessing_selection=preprocessing_selection,
    )
    if args.selection == "final":
        final_signal_args = {
            "native_evaluation_signal_preflight_bundle": (
                args.native_evaluation_signal_preflight_bundle
            ),
            "expected_native_evaluation_signal_preflight_artifact_sha256": (
                args.expected_native_evaluation_signal_preflight_artifact_sha256
            ),
            "expected_native_evaluation_signal_preflight_receipt_sha256": (
                args.expected_native_evaluation_signal_preflight_receipt_sha256
            ),
        }
        missing = tuple(
            name for name, value in final_signal_args.items() if value is None
        )
        if missing:
            raise ValueError(
                "Final selection requires externally pinned source-dev signal "
                f"preflight arguments: {missing}"
            )
        signal_bundle = load_bound_deepsoz_signal_preflight_artifact(
            args.native_evaluation_signal_preflight_bundle,
            expected_artifact_sha256=(
                args.expected_native_evaluation_signal_preflight_artifact_sha256
            ),
            expected_receipt_sha256=(
                args.expected_native_evaluation_signal_preflight_receipt_sha256
            ),
        )
        evaluation_manifest = load_ictal_native_eval_manifest(
            args.native_evaluation_manifest_bundle,
            signal_bundle,
            args.edf_root,
            expected_artifact_sha256=(
                args.expected_native_evaluation_manifest_bundle_sha256
            ),
            expected_receipt_sha256=(
                args.expected_native_evaluation_manifest_source_sha256
            ),
            expected_signal_artifact_sha256=(
                args.expected_native_evaluation_signal_preflight_artifact_sha256
            ),
            expected_signal_receipt_sha256=(
                args.expected_native_evaluation_signal_preflight_receipt_sha256
            ),
        )
        evaluation_corpus = load_ictal_native_eval_token_corpus(
            args.native_evaluation_token_corpus,
            evaluation_manifest,
            expected_index_sha256=(
                args.expected_native_evaluation_token_corpus_index_sha256
            ),
            expected_manifest_artifact_sha256=(
                args.expected_native_evaluation_manifest_bundle_sha256
            ),
            expected_manifest_receipt_sha256=(
                args.expected_native_evaluation_manifest_source_sha256
            ),
            expected_signal_artifact_sha256=(
                args.expected_native_evaluation_signal_preflight_artifact_sha256
            ),
            expected_signal_receipt_sha256=(
                args.expected_native_evaluation_signal_preflight_receipt_sha256
            ),
        )
    else:
        evaluation_manifest = load_tusz_ictal_training_manifest(
            args.native_evaluation_manifest_bundle,
            expected_bundle_manifest_sha256=(
                args.expected_native_evaluation_manifest_bundle_sha256
            ),
            expected_source_manifest_sha256=(
                args.expected_native_evaluation_manifest_source_sha256
            ),
        )
        evaluation_corpus = load_formal_token_corpus(
            args.native_evaluation_token_corpus,
            expected_index_sha256=(
                args.expected_native_evaluation_token_corpus_index_sha256
            ),
            preprocessing_selection=preprocessing_selection,
        )
    selection_kwargs = {
        "promotion_gate_policy_artifact": promotion_gate_policy,
        "expected_promotion_gate_policy_artifact_sha256": (
            args.expected_promotion_gate_policy_artifact_sha256
        ),
        "expected_promotion_gate_policy_bundle_receipt_sha256": (
            args.expected_promotion_gate_policy_bundle_receipt_sha256
        ),
        "protocol_artifact": protocol_artifact,
        "expected_protocol_artifact_sha256": (
            args.expected_oof_protocol_artifact_sha256
        ),
        "expected_protocol_receipt_sha256": (
            args.expected_oof_protocol_receipt_sha256
        ),
        "expected_split_manifest_sha256": args.expected_split_manifest_sha256,
        "selection": args.selection,
        "training_manifest": training_manifest,
        "training_corpus": training_corpus,
        "expected_training_corpus_index_sha256": (
            args.expected_training_token_corpus_index_sha256
        ),
        "native_evaluation_manifest": evaluation_manifest,
        "native_evaluation_corpus": evaluation_corpus,
        "expected_native_evaluation_corpus_index_sha256": (
            args.expected_native_evaluation_token_corpus_index_sha256
        ),
    }
    if args.preflight_only:
        validated = validate_ictal_production_selection(**selection_kwargs)
        print(
            json.dumps(
                {
                    "selection": validated.selection,
                    "oof_fold": validated.oof_fold,
                    "promotion_gate_policy_artifact_sha256": (
                        validated.promotion_gate_policy_artifact_sha256
                    ),
                    "promotion_gate_policy_bundle_receipt_sha256": (
                        validated.promotion_gate_policy_bundle_receipt_sha256
                    ),
                    "promotion_gate_policy_receipt_sha256": (
                        validated.promotion_gate_policy_receipt_sha256
                    ),
                    "held_out_exclusion_patient_count": len(
                        validated.held_out_exclusion_public_patient_ids
                    ),
                    "native_evaluable_patient_count": len(
                        validated.native_evaluation_public_patient_ids
                    ),
                    "native_unevaluable_patient_count": len(
                        validated.native_unevaluable_public_patient_ids
                    ),
                    "training_started": False,
                    "preflight_passed": True,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
        return 0
    artifact = train_formal_ictal_production_run(
        **selection_kwargs,
        edf_root=args.edf_root,
        output_directory=args.output_directory,
        device=args.device,
    )
    loaded = load_ictal_production_run(
        artifact.path,
        expected_manifest_sha256=artifact.manifest_sha256,
    )
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "production_manifest_sha256": artifact.manifest_sha256,
                "checkpoint_manifest_sha256": (
                    artifact.checkpoint.manifest_sha256
                ),
                "selection": loaded.manifest["selection"],
                "native_evaluation_role": loaded.manifest[
                    "native_evaluation_role"
                ],
                "native_metrics": loaded.manifest["native_metrics"],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
