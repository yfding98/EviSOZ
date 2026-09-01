#!/usr/bin/env python3
"""Train one target-free, patient-excluded LaBraM k31 v1.2 producer.

Unlike v1.1 this entry point has no DeepSOZ source/target-table argument.  It
loads only the hash-pinned target-free OOF protocol identities and TUSZ native
ictal-involvement targets.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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
    raise RuntimeError("LaBraM k31 v1.2 requires CUBLAS_WORKSPACE_CONFIG=':4096:8'")

import torch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tusz_ictal_token_cache import load_formal_token_corpus  # noqa: E402
from scripts.run_ictal_v5_dev import _load_split, _train_head  # noqa: E402
from scripts.run_labram_ictal_long_context_recovery_v1 import _memoized_subset  # noqa: E402
from src.soz.concept_run import IctalTrainingConfig  # noqa: E402
from src.soz.data.tusz_training import load_tusz_ictal_training_manifest  # noqa: E402
from src.soz.ictal_gate_policy import load_ictal_promotion_gate_policy  # noqa: E402
from src.soz.ictal_native_eval import (  # noqa: E402
    VerifiedIctalNativeEvalManifestArtifact,
    build_ictal_native_eval_token_bag_dataset,
    load_bound_deepsoz_signal_preflight_artifact,
    load_ictal_native_eval_manifest,
    load_ictal_native_eval_token_corpus,
)
from src.soz.ictal_recovery_evidence import load_target_free_ictal_oof_protocol  # noqa: E402
from src.soz.ictal_recovery_oof_v1_2 import (  # noqa: E402
    LABRAM_K31_EXECUTION_RECEIPT_SCHEMA,
    save_labram_k31_oof_recovery_run_v1_2,
)
from src.soz.ictal_recovery_target_free_validation import (  # noqa: E402
    validate_target_free_ictal_recovery_selection,
)
from src.soz.ictal_target_snapshot import (  # noqa: E402
    build_tusz_ictal_token_bag_dataset_from_target_snapshot,
    load_verified_ictal_target_snapshot,
)
from src.soz.models.concept_heads import (  # noqa: E402
    LongContextTemporalResidualIctalInvolvementHead,
)
from src.soz.preprocessing_parity import load_preprocessing_selection_capability  # noqa: E402


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, choices=(*[f"fold{i}" for i in range(5)], "final"))
    for name in ("promotion-gate-policy-bundle", "oof-protocol", "preprocessing-selection-bundle", "master-manifest-bundle", "v5-split", "formal-v4-target-snapshot", "training-manifest-bundle", "training-token-corpus", "native-evaluation-manifest-bundle", "native-evaluation-token-corpus", "edf-root", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "expected-promotion-gate-policy-artifact-sha256",
        "expected-promotion-gate-policy-bundle-receipt-sha256",
        "expected-oof-protocol-artifact-sha256",
        "expected-oof-protocol-receipt-sha256",
        "expected-split-manifest-sha256",
        "expected-preprocessing-selection-artifact-sha256",
        "expected-preprocessing-protocol-receipt-sha256",
        "expected-master-manifest-bundle-sha256",
        "expected-master-manifest-source-sha256",
        "expected-v5-split-sha256",
        "expected-formal-v4-target-snapshot-manifest-sha256",
        "expected-formal-v4-target-snapshot-receipt-sha256",
        "expected-training-manifest-bundle-sha256",
        "expected-training-manifest-source-sha256",
        "expected-training-token-corpus-index-sha256",
        "expected-native-evaluation-manifest-bundle-sha256",
        "expected-native-evaluation-manifest-source-sha256",
        "expected-native-evaluation-token-corpus-index-sha256",
    ):
        parser.add_argument(f"--{name}", type=_sha256, required=True)
    parser.add_argument("--native-evaluation-signal-preflight-bundle", type=Path)
    parser.add_argument("--expected-native-evaluation-signal-preflight-artifact-sha256", type=_sha256)
    parser.add_argument("--expected-native-evaluation-signal-preflight-receipt-sha256", type=_sha256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _progress(selection: str, stage: str) -> None:
    print(json.dumps({"selection": selection, "stage": stage, "deepsoz_target_values_reachable": False, "i_gate_outcomes_opened": False}, sort_keys=True), flush=True)


def _load_hash_pinned_v5_split(
    path: Path, *, expected_sha256: str, master_patients: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError("v5 split must be a regular absolute file")
    before = source.stat()
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("v5 split SHA mismatch")
    dev, gate = _load_split(source, master_patients)
    after_raw = source.read_bytes()
    after = source.stat()
    fingerprints = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if fingerprints(before) != fingerprints(after) or after_raw != raw:
        raise RuntimeError("v5 split changed during strict loading")
    return dev, gate, digest


def _execution_receipt(config: IctalTrainingConfig, device: torch.device) -> dict[str, object]:
    probe = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW(
        [probe], lr=float(config.learning_rate), weight_decay=float(config.weight_decay)
    )
    group = optimizer.param_groups[0]
    capability = None
    device_name = "cpu"
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
    config_payload = asdict(config)
    config_sha = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": LABRAM_K31_EXECUTION_RECEIPT_SCHEMA,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": device.type,
        "device_name": device_name,
        "compute_capability": capability,
        "optimizer_class": "torch.optim.AdamW",
        "optimizer_effective_hyperparameters": {
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "betas": [float(value) for value in group["betas"]],
            "eps": float(group["eps"]),
            "amsgrad": bool(group["amsgrad"]),
            "maximize": bool(group["maximize"]),
            "foreach": group.get("foreach"),
            "capturable": bool(group["capturable"]),
            "differentiable": bool(group["differentiable"]),
            "fused": group.get("fused"),
        },
        "training_config_sha256": config_sha,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    if not args.preflight_only and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    _progress(args.selection, "load_target_free_authorities")
    preprocessing = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=args.expected_preprocessing_selection_artifact_sha256,
        expected_protocol_receipt_sha256=args.expected_preprocessing_protocol_receipt_sha256,
    )
    gate_policy = load_ictal_promotion_gate_policy(
        args.promotion_gate_policy_bundle,
        expected_artifact_sha256=args.expected_promotion_gate_policy_artifact_sha256,
        expected_receipt_sha256=args.expected_promotion_gate_policy_bundle_receipt_sha256,
    )
    protocol = load_target_free_ictal_oof_protocol(
        args.oof_protocol,
        expected_artifact_sha256=args.expected_oof_protocol_artifact_sha256,
        expected_protocol_receipt_sha256=args.expected_oof_protocol_receipt_sha256,
    )
    if protocol.receipt.split_manifest_sha256 != args.expected_split_manifest_sha256:
        raise ValueError("Target-free protocol uses another DeepSOZ identity split")
    master = load_tusz_ictal_training_manifest(
        args.master_manifest_bundle,
        expected_bundle_manifest_sha256=args.expected_master_manifest_bundle_sha256,
        expected_source_manifest_sha256=args.expected_master_manifest_source_sha256,
    )
    _, gate_patients, v5_split_sha = _load_hash_pinned_v5_split(
        args.v5_split,
        expected_sha256=args.expected_v5_split_sha256,
        master_patients=master.patient_ids,
    )
    training_manifest = load_tusz_ictal_training_manifest(
        args.training_manifest_bundle,
        expected_bundle_manifest_sha256=args.expected_training_manifest_bundle_sha256,
        expected_source_manifest_sha256=args.expected_training_manifest_source_sha256,
    )
    training_corpus = load_formal_token_corpus(
        args.training_token_corpus,
        expected_index_sha256=args.expected_training_token_corpus_index_sha256,
        preprocessing_selection=preprocessing,
    )
    if args.selection == "final":
        required = (
            args.native_evaluation_signal_preflight_bundle,
            args.expected_native_evaluation_signal_preflight_artifact_sha256,
            args.expected_native_evaluation_signal_preflight_receipt_sha256,
        )
        if any(value is None for value in required):
            raise ValueError("Final requires the pinned source-dev signal bundle")
        signal = load_bound_deepsoz_signal_preflight_artifact(
            args.native_evaluation_signal_preflight_bundle,
            expected_artifact_sha256=args.expected_native_evaluation_signal_preflight_artifact_sha256,
            expected_receipt_sha256=args.expected_native_evaluation_signal_preflight_receipt_sha256,
        )
        native_manifest = load_ictal_native_eval_manifest(
            args.native_evaluation_manifest_bundle,
            signal,
            args.edf_root,
            expected_artifact_sha256=args.expected_native_evaluation_manifest_bundle_sha256,
            expected_receipt_sha256=args.expected_native_evaluation_manifest_source_sha256,
            expected_signal_artifact_sha256=args.expected_native_evaluation_signal_preflight_artifact_sha256,
            expected_signal_receipt_sha256=args.expected_native_evaluation_signal_preflight_receipt_sha256,
        )
        native_corpus = load_ictal_native_eval_token_corpus(
            args.native_evaluation_token_corpus,
            native_manifest,
            expected_index_sha256=args.expected_native_evaluation_token_corpus_index_sha256,
            expected_manifest_artifact_sha256=args.expected_native_evaluation_manifest_bundle_sha256,
            expected_manifest_receipt_sha256=args.expected_native_evaluation_manifest_source_sha256,
            expected_signal_artifact_sha256=args.expected_native_evaluation_signal_preflight_artifact_sha256,
            expected_signal_receipt_sha256=args.expected_native_evaluation_signal_preflight_receipt_sha256,
        )
    else:
        if any(value is not None for value in (
            args.native_evaluation_signal_preflight_bundle,
            args.expected_native_evaluation_signal_preflight_artifact_sha256,
            args.expected_native_evaluation_signal_preflight_receipt_sha256,
        )):
            raise ValueError("Fold selections cannot accept a source-dev signal bundle")
        native_manifest = load_tusz_ictal_training_manifest(
            args.native_evaluation_manifest_bundle,
            expected_bundle_manifest_sha256=args.expected_native_evaluation_manifest_bundle_sha256,
            expected_source_manifest_sha256=args.expected_native_evaluation_manifest_source_sha256,
        )
        native_corpus = load_formal_token_corpus(
            args.native_evaluation_token_corpus,
            expected_index_sha256=args.expected_native_evaluation_token_corpus_index_sha256,
            preprocessing_selection=preprocessing,
        )
    validated = validate_target_free_ictal_recovery_selection(
        promotion_gate_policy_artifact=gate_policy,
        expected_promotion_gate_policy_artifact_sha256=args.expected_promotion_gate_policy_artifact_sha256,
        expected_promotion_gate_policy_bundle_receipt_sha256=args.expected_promotion_gate_policy_bundle_receipt_sha256,
        protocol=protocol,
        expected_protocol_artifact_sha256=args.expected_oof_protocol_artifact_sha256,
        expected_protocol_receipt_sha256=args.expected_oof_protocol_receipt_sha256,
        expected_split_manifest_sha256=args.expected_split_manifest_sha256,
        selection=args.selection,
        training_manifest=training_manifest,
        training_corpus=training_corpus,
        expected_training_corpus_index_sha256=args.expected_training_token_corpus_index_sha256,
        native_evaluation_manifest=native_manifest,
        native_evaluation_corpus=native_corpus,
        expected_native_evaluation_corpus_index_sha256=args.expected_native_evaluation_token_corpus_index_sha256,
    )
    fit_patients = tuple(sorted(set(training_manifest.patient_ids) - set(gate_patients)))
    native_patients = tuple(sorted(set(validated.native_evaluation_public_patient_ids) - set(gate_patients)))
    if not fit_patients or not native_patients:
        raise ValueError("I-gate exclusion left an empty recovery cohort")
    if set(fit_patients) & (set(validated.held_out_exclusion_public_patient_ids) | set(native_patients) | set(gate_patients)):
        raise ValueError("Patient firewall failed after I-gate exclusion")
    if set(native_patients) & set(gate_patients):
        raise ValueError("I-gate outcome entered native evaluation")
    preflight = {
        "schema_version": "soz_labram_k31_oof_recovery_preflight_v1_2",
        "selection": args.selection,
        "oof_fold": validated.oof_fold,
        "training_patient_count_after_i_gate_exclusion": len(fit_patients),
        "native_evaluation_patient_count_after_i_gate_exclusion": len(native_patients),
        "held_out_exclusion_patient_count": len(validated.held_out_exclusion_public_patient_ids),
        "i_gate_patient_count_excluded_unopened": len(gate_patients),
        "v5_split_sha256": v5_split_sha,
        "deepsoz_target_source_loaded": False,
        "deepsoz_target_values_reachable": False,
        "tusz_ictal_involvement_targets_loaded": not args.preflight_only,
        "i_gate_outcomes_opened": False,
        "private_labels_used": False,
        "training_started": not args.preflight_only,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0
    target_snapshot = load_verified_ictal_target_snapshot(
        args.formal_v4_target_snapshot,
        expected_manifest_sha256=args.expected_formal_v4_target_snapshot_manifest_sha256,
        expected_receipt_sha256=args.expected_formal_v4_target_snapshot_receipt_sha256,
    )
    training_dataset = build_tusz_ictal_token_bag_dataset_from_target_snapshot(
        training_manifest, training_corpus, target_snapshot, patient_ids=fit_patients
    )
    if isinstance(native_manifest, VerifiedIctalNativeEvalManifestArtifact):
        evaluation_dataset = build_ictal_native_eval_token_bag_dataset(
            native_manifest, args.edf_root, native_corpus
        )
        native_manifest_sha = native_manifest.receipt_sha256
    else:
        evaluation_dataset = build_tusz_ictal_token_bag_dataset_from_target_snapshot(
            native_manifest, native_corpus, target_snapshot, patient_ids=native_patients
        )
        native_manifest_sha = native_manifest.manifest_sha256
    if training_dataset.foundation_feature_receipt_sha256 != evaluation_dataset.foundation_feature_receipt_sha256:
        raise ValueError("Training/evaluation LaBraM receipts differ")
    if tuple(training_dataset.patient_ids) != fit_patients or tuple(evaluation_dataset.patient_ids) != native_patients:
        raise ValueError("Target snapshot changed a gate-filtered roster")
    fit_dataset = _memoized_subset(training_dataset, fit_patients)
    native_dataset = _memoized_subset(evaluation_dataset, native_patients)
    config = IctalTrainingConfig()
    execution = _execution_receipt(config, device)
    head, training_run = _train_head(
        name=f"{args.selection}_labram_temporal_residual_k31_v1_2",
        factory=LongContextTemporalResidualIctalInvolvementHead,
        fit_dataset=fit_dataset,
        evaluation_dataset=native_dataset,
        evaluation_patient_ids=native_patients,
        config=config,
        device=device,
    )
    loaded = save_labram_k31_oof_recovery_run_v1_2(
        args.output_directory,
        v5_split_sha256=v5_split_sha,
        execution_receipt=execution,
        selection=args.selection,
        head=head,
        split_manifest_sha256=args.expected_split_manifest_sha256,
        oof_protocol_artifact_sha256=protocol.artifact_sha256,
        oof_protocol_receipt_sha256=protocol.receipt_sha256,
        oof_plan_receipt_sha256=validated.plan_receipt.receipt_sha256,
        training_manifest_sha256=training_manifest.manifest_sha256,
        training_corpus_index_sha256=training_corpus.index_sha256,
        target_snapshot_manifest_sha256=target_snapshot.manifest_sha256,
        target_snapshot_receipt_sha256=target_snapshot.receipt_sha256,
        native_evaluation_manifest_sha256=native_manifest_sha,
        native_evaluation_corpus_index_sha256=native_corpus.index_sha256,
        training_public_patient_ids=fit_patients,
        held_out_exclusion_public_patient_ids=validated.held_out_exclusion_public_patient_ids,
        native_evaluation_public_patient_ids=native_patients,
        i_gate_patient_ids_excluded_unopened=gate_patients,
        training_config=asdict(config),
        training_run=training_run,
    )
    print(json.dumps({**preflight, "path": str(loaded.path), "manifest_sha256": loaded.manifest_sha256, "native_metrics": training_run["metrics"], "formal_promotion": False, "checkpoint_authorized_for_formal_evidence_or_reasoner": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
