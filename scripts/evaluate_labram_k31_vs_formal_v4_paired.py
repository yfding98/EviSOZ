#!/usr/bin/env python3
"""Five-fold paired replay of LaBraM k31 v1.2 vs formal-v4.

The formal-v4 comparator is the independent-second (context=1) producer.  No
five-fold k5 production artifact exists, so this command cannot accept or
describe one.  It exposes no source-dev, source-eval, private, DeepSOZ SOZ, or
final-producer input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
)
from src.soz.ictal_production import load_ictal_production_run  # noqa: E402
from src.soz.ictal_recovery_oof_v1_2 import (  # noqa: E402
    load_labram_k31_oof_recovery_run_v1_2,
)
from src.soz.ictal_recovery_paired_evaluation import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EVENT_MICROBATCH_SIZE,
    SELECTIONS,
    build_paired_evaluation_payload,
    replay_paired_fold,
    save_paired_evaluation,
    validate_fold_pair_lineage,
    verify_replay_against_run_metrics,
)
from src.soz.ictal_target_snapshot import (  # noqa: E402
    build_tusz_ictal_token_bag_dataset_from_target_snapshot,
    load_verified_ictal_target_snapshot,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def _fold_sha(value: str) -> tuple[str, str]:
    try:
        selection, digest = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected foldN=SHA256") from exc
    if selection not in SELECTIONS:
        raise argparse.ArgumentTypeError("selection must be fold0..fold4")
    return selection, _sha256(digest)


def _closed_fold_sha_map(
    rows: Sequence[tuple[str, str]], *, field: str
) -> dict[str, str]:
    mapping = dict(rows)
    if len(rows) != len(mapping) or tuple(sorted(mapping)) != SELECTIONS:
        raise ValueError(f"{field} requires each of fold0..fold4 exactly once")
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-v4-root", type=Path, required=True)
    parser.add_argument("--k31-v1-2-root", type=Path, required=True)
    parser.add_argument(
        "--expected-formal-v4-manifest-sha256",
        type=_fold_sha,
        action="append",
        required=True,
        metavar="foldN=SHA256",
    )
    parser.add_argument(
        "--expected-k31-v1-2-manifest-sha256",
        type=_fold_sha,
        action="append",
        required=True,
        metavar="foldN=SHA256",
    )
    parser.add_argument("--master-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-master-manifest-bundle-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-master-manifest-source-sha256", type=_sha256, required=True
    )
    parser.add_argument("--master-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-master-token-corpus-index-sha256", type=_sha256, required=True
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
    parser.add_argument("--target-snapshot", type=Path, required=True)
    parser.add_argument(
        "--expected-target-snapshot-manifest-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-target-snapshot-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--event-microbatch-size",
        type=int,
        default=DEFAULT_EVENT_MICROBATCH_SIZE,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def _absolute_regular_directory(value: Path, *, field: str) -> Path:
    path = Path(os.path.abspath(value))
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ValueError(f"{field} must be a regular absolute directory")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if args.event_microbatch_size < 1:
        raise ValueError("event_microbatch_size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    formal_hashes = _closed_fold_sha_map(
        args.expected_formal_v4_manifest_sha256,
        field="expected formal-v4 manifests",
    )
    candidate_hashes = _closed_fold_sha_map(
        args.expected_k31_v1_2_manifest_sha256,
        field="expected k31-v1.2 manifests",
    )
    formal_root = _absolute_regular_directory(
        args.formal_v4_root, field="formal-v4 root"
    )
    candidate_root = _absolute_regular_directory(
        args.k31_v1_2_root, field="k31-v1.2 root"
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
    master_manifest = load_tusz_ictal_training_manifest(
        args.master_manifest_bundle,
        expected_bundle_manifest_sha256=(
            args.expected_master_manifest_bundle_sha256
        ),
        expected_source_manifest_sha256=(
            args.expected_master_manifest_source_sha256
        ),
    )
    master_corpus = load_formal_token_corpus(
        args.master_token_corpus,
        expected_index_sha256=args.expected_master_token_corpus_index_sha256,
        preprocessing_selection=preprocessing,
    )
    target_snapshot = load_verified_ictal_target_snapshot(
        args.target_snapshot,
        expected_manifest_sha256=args.expected_target_snapshot_manifest_sha256,
        expected_receipt_sha256=args.expected_target_snapshot_receipt_sha256,
    )
    if target_snapshot.training_manifest_sha256 != master_manifest.manifest_sha256:
        raise ValueError("Target snapshot binds another master TUSZ manifest")
    if target_snapshot.training_corpus_index_sha256 != master_corpus.index_sha256:
        raise ValueError("Target snapshot binds another master token corpus")

    formal_runs = {}
    candidate_runs = {}
    fold_lineage = []
    all_patients: list[str] = []
    for selection in SELECTIONS:
        formal = load_ictal_production_run(
            formal_root / selection,
            expected_manifest_sha256=formal_hashes[selection],
        )
        candidate = load_labram_k31_oof_recovery_run_v1_2(
            candidate_root / selection,
            expected_manifest_sha256=candidate_hashes[selection],
        )
        lineage = validate_fold_pair_lineage(
            selection=selection,
            formal_v4=formal,
            k31_v1_2=candidate,
            target_snapshot=target_snapshot,
        )
        formal_runs[selection] = formal
        candidate_runs[selection] = candidate
        fold_lineage.append(lineage)
        all_patients.extend(lineage["native_evaluation_public_patient_ids"])
    if len(all_patients) != len(set(all_patients)):
        raise ValueError("A patient appears in more than one OOF evaluation fold")
    sorted_patients = tuple(sorted(all_patients))
    dataset = build_tusz_ictal_token_bag_dataset_from_target_snapshot(
        master_manifest,
        master_corpus,
        target_snapshot,
        patient_ids=sorted_patients,
    )
    foundation_receipts = {
        lineage["formal_v4_independent_second"][
            "foundation_feature_receipt_sha256"
        ]
        for lineage in fold_lineage
    }
    foundation_receipts.add(dataset.foundation_feature_receipt_sha256)
    if len(foundation_receipts) != 1:
        raise ValueError("Paired folds do not use one frozen LaBraM token receipt")

    patient_rows = []
    replay_checks = []
    for selection, lineage in zip(SELECTIONS, fold_lineage, strict=True):
        comparator_head = formal_runs[selection].checkpoint.head.to(device)
        candidate_head = candidate_runs[selection].head.to(device)
        rows = replay_paired_fold(
            selection=selection,
            dataset=dataset,
            patient_ids=lineage["native_evaluation_public_patient_ids"],
            comparator_head=comparator_head,
            candidate_head=candidate_head,
            event_microbatch_size=args.event_microbatch_size,
        )
        comparator_head.to("cpu")
        candidate_head.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        replay_checks.append(
            {
                "selection": selection,
                **verify_replay_against_run_metrics(
                    rows=rows,
                    formal_v4=formal_runs[selection],
                    k31_v1_2=candidate_runs[selection],
                ),
            }
        )
        patient_rows.extend(rows)

    payload = build_paired_evaluation_payload(
        patient_rows=patient_rows,
        fold_lineage=fold_lineage,
        replay_checks=replay_checks,
        target_snapshot=target_snapshot,
        master_bundle_manifest_sha256=(
            args.expected_master_manifest_bundle_sha256
        ),
        master_source_manifest_sha256=master_manifest.manifest_sha256,
        master_corpus_index_sha256=master_corpus.index_sha256,
        foundation_feature_receipt_sha256=dataset.foundation_feature_receipt_sha256,
        preprocessing_selection_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        preprocessing_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
        execution_device=device.type,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_path, artifact_sha, receipt_sha = save_paired_evaluation(
        args.output_directory, payload
    )
    print(
        json.dumps(
            {
                "path": str(output_path),
                "artifact_sha256": artifact_sha,
                "receipt_sha256": receipt_sha,
                "patient_count": len(patient_rows),
                "development_only": True,
                "formal_promotion": False,
                "comparator": "formal_v4_independent_second_context_1_not_k5",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
