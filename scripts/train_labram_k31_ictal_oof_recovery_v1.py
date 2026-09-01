#!/usr/bin/env python3
"""Train one patient-excluded LaBraM k31 recovery producer.

This is a post-formal-v5 development protocol.  It reuses the verified
formal-v4 LaBraM token corpora and OOF exclusions, but writes a distinct
non-promoted artifact because the k31 architecture was selected after I-dev
was opened.  The frozen 12-patient I-gate is removed from fitting and from
native-label evaluation for every selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
from typing import Sequence


_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
_observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _observed_workspace is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _REQUIRED_CUBLAS_WORKSPACE
elif _observed_workspace != _REQUIRED_CUBLAS_WORKSPACE:
    raise RuntimeError("LaBraM k31 OOF recovery requires CUBLAS_WORKSPACE_CONFIG=':4096:8'")

import torch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.derive_tusz_ictal_oof_fold_manifests import (  # noqa: E402
    load_bound_deepsoz_registry,
)
from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from scripts.run_ictal_v5_dev import _load_split, _train_head  # noqa: E402
from scripts.run_labram_ictal_long_context_recovery_v1 import (  # noqa: E402
    _memoized_subset,
)
from src.soz.concept_oof import load_ictal_concept_oof_protocol  # noqa: E402
from src.soz.concept_run import IctalTrainingConfig  # noqa: E402
from src.soz.data.public_ledger_builder import (  # noqa: E402
    load_tusz_deepsoz_public_ledger_build,
)
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
)
from src.soz.ictal_gate_policy import (  # noqa: E402
    load_ictal_promotion_gate_policy,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    VerifiedIctalNativeEvalManifestArtifact,
    build_ictal_native_eval_token_bag_dataset,
    load_bound_deepsoz_signal_preflight_artifact,
    load_ictal_native_eval_manifest,
    load_ictal_native_eval_token_corpus,
)
from src.soz.ictal_production import (  # noqa: E402
    validate_ictal_production_selection,
)
from src.soz.ictal_recovery_oof import (  # noqa: E402
    save_labram_k31_oof_recovery_run,
)
from src.soz.ictal_target_snapshot import (  # noqa: E402
    build_tusz_ictal_token_bag_dataset_from_target_snapshot,
    load_verified_ictal_target_snapshot,
)
from src.soz.models.concept_heads import (  # noqa: E402
    LongContextTemporalResidualIctalInvolvementHead,
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
        choices=("fold0", "fold1", "fold2", "fold3", "fold4", "final"),
    )
    parser.add_argument("--promotion-gate-policy-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-promotion-gate-policy-artifact-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument(
        "--expected-promotion-gate-policy-bundle-receipt-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--oof-protocol", type=Path, required=True)
    parser.add_argument(
        "--expected-oof-protocol-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-oof-protocol-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--public-ledger", type=Path, required=True)
    parser.add_argument(
        "--expected-public-ledger-bundle-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-public-ledger-build-sha256", type=_sha256, required=True
    )
    parser.add_argument("--deepsoz-source-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-deepsoz-source-sha256", type=_sha256, required=True
    )
    parser.add_argument("--deepsoz-split-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-split-manifest-sha256", type=_sha256, required=True
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
    parser.add_argument("--master-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-master-manifest-bundle-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-master-manifest-source-sha256", type=_sha256, required=True
    )
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--formal-v4-target-snapshot", type=Path, required=True)
    parser.add_argument(
        "--expected-formal-v4-target-snapshot-manifest-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument(
        "--expected-formal-v4-target-snapshot-receipt-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--training-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-training-manifest-bundle-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-training-manifest-source-sha256", type=_sha256, required=True
    )
    parser.add_argument("--training-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-training-token-corpus-index-sha256", type=_sha256, required=True
    )
    parser.add_argument("--native-evaluation-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-native-evaluation-manifest-bundle-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument(
        "--expected-native-evaluation-manifest-source-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--native-evaluation-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-native-evaluation-token-corpus-index-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--native-evaluation-signal-preflight-bundle", type=Path)
    parser.add_argument(
        "--expected-native-evaluation-signal-preflight-artifact-sha256",
        type=_sha256,
    )
    parser.add_argument(
        "--expected-native-evaluation-signal-preflight-receipt-sha256",
        type=_sha256,
    )
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _progress(selection: str, stage: str) -> None:
    print(
        json.dumps(
            {"selection": selection, "stage": stage, "i_gate_outcomes_opened": False},
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    if not args.preflight_only and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    _progress(args.selection, "load_preprocessing_and_oof_authorities")
    preprocessing = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=args.expected_preprocessing_selection_artifact_sha256,
        expected_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
    )
    gate_policy = load_ictal_promotion_gate_policy(
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
    public_ledger = load_tusz_deepsoz_public_ledger_build(
        args.public_ledger,
        expected_bundle_sha256=args.expected_public_ledger_bundle_sha256,
        expected_build_sha256=args.expected_public_ledger_build_sha256,
    )
    protocol = load_ictal_concept_oof_protocol(
        args.oof_protocol,
        registry,
        public_ledger,
        expected_artifact_sha256=args.expected_oof_protocol_artifact_sha256,
        expected_protocol_sha256=args.expected_oof_protocol_receipt_sha256,
    )
    _progress(args.selection, "load_master_and_i_gate_roster")
    master_manifest = load_tusz_ictal_training_manifest(
        args.master_manifest_bundle,
        expected_bundle_manifest_sha256=args.expected_master_manifest_bundle_sha256,
        expected_source_manifest_sha256=args.expected_master_manifest_source_sha256,
    )
    _, gate_patients = _load_split(args.v5_split, master_manifest.patient_ids)
    _progress(args.selection, "load_training_manifest_and_token_corpus")
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
        expected_index_sha256=args.expected_training_token_corpus_index_sha256,
        preprocessing_selection=preprocessing,
    )

    _progress(args.selection, "load_native_evaluation_manifest_and_token_corpus")
    if args.selection == "final":
        required_signal = (
            args.native_evaluation_signal_preflight_bundle,
            args.expected_native_evaluation_signal_preflight_artifact_sha256,
            args.expected_native_evaluation_signal_preflight_receipt_sha256,
        )
        if any(value is None for value in required_signal):
            raise ValueError("Final selection requires the pinned source-dev signal bundle")
        signal = load_bound_deepsoz_signal_preflight_artifact(
            args.native_evaluation_signal_preflight_bundle,
            expected_artifact_sha256=(
                args.expected_native_evaluation_signal_preflight_artifact_sha256
            ),
            expected_receipt_sha256=(
                args.expected_native_evaluation_signal_preflight_receipt_sha256
            ),
        )
        native_manifest = load_ictal_native_eval_manifest(
            args.native_evaluation_manifest_bundle,
            signal,
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
        native_corpus = load_ictal_native_eval_token_corpus(
            args.native_evaluation_token_corpus,
            native_manifest,
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
        if any(
            value is not None
            for value in (
                args.native_evaluation_signal_preflight_bundle,
                args.expected_native_evaluation_signal_preflight_artifact_sha256,
                args.expected_native_evaluation_signal_preflight_receipt_sha256,
            )
        ):
            raise ValueError("Fold selections cannot accept a source-dev signal bundle")
        native_manifest = load_tusz_ictal_training_manifest(
            args.native_evaluation_manifest_bundle,
            expected_bundle_manifest_sha256=(
                args.expected_native_evaluation_manifest_bundle_sha256
            ),
            expected_source_manifest_sha256=(
                args.expected_native_evaluation_manifest_source_sha256
            ),
        )
        native_corpus = load_formal_token_corpus(
            args.native_evaluation_token_corpus,
            expected_index_sha256=(
                args.expected_native_evaluation_token_corpus_index_sha256
            ),
            preprocessing_selection=preprocessing,
        )

    _progress(args.selection, "validate_patient_exclusion_and_lineage")
    validated = validate_ictal_production_selection(
        promotion_gate_policy_artifact=gate_policy,
        expected_promotion_gate_policy_artifact_sha256=(
            args.expected_promotion_gate_policy_artifact_sha256
        ),
        expected_promotion_gate_policy_bundle_receipt_sha256=(
            args.expected_promotion_gate_policy_bundle_receipt_sha256
        ),
        protocol_artifact=protocol,
        expected_protocol_artifact_sha256=args.expected_oof_protocol_artifact_sha256,
        expected_protocol_receipt_sha256=args.expected_oof_protocol_receipt_sha256,
        expected_split_manifest_sha256=args.expected_split_manifest_sha256,
        selection=args.selection,
        training_manifest=training_manifest,
        training_corpus=training_corpus,
        expected_training_corpus_index_sha256=(
            args.expected_training_token_corpus_index_sha256
        ),
        native_evaluation_manifest=native_manifest,
        native_evaluation_corpus=native_corpus,
        expected_native_evaluation_corpus_index_sha256=(
            args.expected_native_evaluation_token_corpus_index_sha256
        ),
    )

    fit_patients = tuple(
        sorted(set(training_manifest.patient_ids) - set(gate_patients))
    )
    native_patients = tuple(
        sorted(
            set(validated.native_evaluation_public_patient_ids)
            - set(gate_patients)
        )
    )
    if not fit_patients or not native_patients:
        raise ValueError("I-gate exclusion left an empty recovery cohort")
    if set(fit_patients) & set(validated.held_out_exclusion_public_patient_ids):
        raise ValueError("Validated OOF exclusion was lost after I-gate filtering")
    if set(fit_patients) & set(native_patients):
        raise ValueError("Recovery fitting and native evaluation overlap")
    if set(gate_patients) & (set(fit_patients) | set(native_patients)):
        raise ValueError("I-gate patients entered recovery fitting or evaluation")

    preflight = {
        "schema_version": "soz_labram_k31_oof_recovery_preflight_v1",
        "selection": args.selection,
        "oof_fold": validated.oof_fold,
        "training_patient_count_after_i_gate_exclusion": len(fit_patients),
        "native_evaluation_patient_count_after_i_gate_exclusion": len(native_patients),
        "held_out_exclusion_patient_count": len(
            validated.held_out_exclusion_public_patient_ids
        ),
        "i_gate_patient_count_excluded_unopened": len(gate_patients),
        "i_gate_outcomes_opened": False,
        "deepsoz_soz_labels_used": False,
        "private_labels_used": False,
        "training_started": not args.preflight_only,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0

    _progress(args.selection, "load_hash_pinned_formal_v4_target_snapshot")
    target_snapshot = load_verified_ictal_target_snapshot(
        args.formal_v4_target_snapshot,
        expected_manifest_sha256=(
            args.expected_formal_v4_target_snapshot_manifest_sha256
        ),
        expected_receipt_sha256=(
            args.expected_formal_v4_target_snapshot_receipt_sha256
        ),
    )
    _progress(args.selection, "build_gate_filtered_datasets")
    training_dataset = build_tusz_ictal_token_bag_dataset_from_target_snapshot(
        training_manifest,
        training_corpus,
        target_snapshot,
        patient_ids=fit_patients,
    )
    if isinstance(native_manifest, VerifiedIctalNativeEvalManifestArtifact):
        evaluation_dataset = build_ictal_native_eval_token_bag_dataset(
            native_manifest, args.edf_root, native_corpus
        )
        native_manifest_sha256 = native_manifest.receipt_sha256
    else:
        evaluation_dataset = build_tusz_ictal_token_bag_dataset_from_target_snapshot(
            native_manifest,
            native_corpus,
            target_snapshot,
            patient_ids=native_patients,
        )
        native_manifest_sha256 = native_manifest.manifest_sha256
    if training_dataset.foundation_feature_receipt_sha256 != (
        evaluation_dataset.foundation_feature_receipt_sha256
    ):
        raise ValueError("Training and evaluation use different LaBraM feature receipts")
    if tuple(training_dataset.patient_ids) != fit_patients:
        raise ValueError("Training snapshot view changed its gate-filtered roster")
    if tuple(evaluation_dataset.patient_ids) != native_patients:
        raise ValueError("Native snapshot view changed its evaluation roster")

    _progress(args.selection, "memoize_gate_filtered_token_bags")
    fit_dataset = _memoized_subset(training_dataset, fit_patients)
    native_dataset = _memoized_subset(evaluation_dataset, native_patients)
    config = IctalTrainingConfig()
    _progress(args.selection, "optimize_k31_head")
    head, training_run = _train_head(
        name=f"{args.selection}_labram_temporal_residual_k31",
        factory=LongContextTemporalResidualIctalInvolvementHead,
        fit_dataset=fit_dataset,
        evaluation_dataset=native_dataset,
        evaluation_patient_ids=native_patients,
        config=config,
        device=device,
    )
    loaded = save_labram_k31_oof_recovery_run(
        args.output_directory,
        selection=args.selection,
        head=head,
        split_manifest_sha256=args.expected_split_manifest_sha256,
        oof_protocol_artifact_sha256=protocol.artifact_sha256,
        oof_protocol_receipt_sha256=protocol.protocol.receipt.receipt_sha256,
        oof_plan_receipt_sha256=validated.plan.receipt.receipt_sha256,
        training_manifest_sha256=training_manifest.manifest_sha256,
        training_corpus_index_sha256=training_corpus.index_sha256,
        target_snapshot_manifest_sha256=target_snapshot.manifest_sha256,
        target_snapshot_receipt_sha256=target_snapshot.receipt_sha256,
        native_evaluation_manifest_sha256=native_manifest_sha256,
        native_evaluation_corpus_index_sha256=native_corpus.index_sha256,
        training_public_patient_ids=fit_patients,
        held_out_exclusion_public_patient_ids=(
            validated.held_out_exclusion_public_patient_ids
        ),
        native_evaluation_public_patient_ids=native_patients,
        i_gate_patient_ids_excluded_unopened=gate_patients,
        training_config=asdict(config),
        training_run=training_run,
    )
    print(
        json.dumps(
            {
                **preflight,
                "path": str(loaded.path),
                "manifest_sha256": loaded.manifest_sha256,
                "native_metrics": training_run["metrics"],
                "formal_promotion": False,
                "checkpoint_authorized_for_formal_evidence_or_reasoner": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
