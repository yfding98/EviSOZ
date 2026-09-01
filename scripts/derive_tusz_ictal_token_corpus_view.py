#!/usr/bin/env python3
"""Derive one fold corpus from a verified formal-v4 master token corpus.

The operation never reads EDF, annotation, DeepSOZ, or SOZ targets and never
runs the foundation encoder.  It strictly reloads the externally pinned master
corpus, proves that the requested training manifest is an exact master subset,
then reserializes only those detached target-free tensors under the fold
manifest lineage.  The output uses the same closed formal-v4 schema and is
atomically published only after a complete strict reload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import materialize_tusz_ictal_token_cache as materializer  # noqa: E402
from src.soz.concept_token_io import (  # noqa: E402
    load_labram_concept_tokens,
    save_labram_concept_tokens,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    VerifiedPreprocessingSelectionCapability,
    load_preprocessing_selection_capability,
)


def derive_formal_tusz_ictal_token_corpus_view(
    *,
    master_manifest_bundle: str | Path,
    expected_master_bundle_manifest_sha256: str,
    expected_master_source_manifest_sha256: str,
    training_manifest_bundle: str | Path,
    expected_training_bundle_manifest_sha256: str,
    expected_training_source_manifest_sha256: str,
    master_token_corpus: str | Path,
    expected_master_token_corpus_index_sha256: str,
    preprocessing_selection: VerifiedPreprocessingSelectionCapability,
    output_directory: str | Path,
) -> materializer.VerifiedFormalTokenCorpusArtifact:
    """Publish one fold-bound tensor view without foundation recomputation."""

    target = materializer._reject_symlink_components(
        Path(output_directory), field="derived formal token corpus output"
    )
    if target.name in {"", ".", ".."}:
        raise ValueError("Derived corpus output requires a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(f"Derived corpus output already exists: {target}")

    master, master_binding = materializer._load_bound_manifest(
        master_manifest_bundle,
        expected_bundle_manifest_sha256=expected_master_bundle_manifest_sha256,
        expected_source_manifest_sha256=expected_master_source_manifest_sha256,
        field="master_manifest",
    )
    training, training_binding = materializer._load_bound_manifest(
        training_manifest_bundle,
        expected_bundle_manifest_sha256=expected_training_bundle_manifest_sha256,
        expected_source_manifest_sha256=expected_training_source_manifest_sha256,
        field="training_manifest",
    )
    training_role = materializer._validate_master_training_relation(master, training)
    if training_role != "derived_fold":
        raise ValueError("Token-corpus view derivation requires a derived fold manifest")
    authorization = materializer._authorize_preprocessing_selection(
        preprocessing_selection,
        preprocess_config=training.preprocess_config,
    )
    master_corpus = materializer.load_formal_token_corpus(
        master_token_corpus,
        expected_index_sha256=expected_master_token_corpus_index_sha256,
        preprocessing_selection=preprocessing_selection,
    )
    corpus_lineage_checks = {
        "master_bundle": (
            master_corpus.master_bundle_manifest_sha256
            == master_binding["bundle_manifest_sha256"]
        ),
        "master_source": (
            master_corpus.master_source_manifest_sha256 == master.manifest_sha256
        ),
        "source_training_bundle_is_master": (
            master_corpus.training_bundle_manifest_sha256
            == master_binding["bundle_manifest_sha256"]
        ),
        "source_training_manifest_is_master": (
            master_corpus.training_source_manifest_sha256 == master.manifest_sha256
        ),
        "selection_artifact": (
            master_corpus.preprocessing_selection_artifact_sha256
            == preprocessing_selection.selection_artifact_sha256
        ),
        "selection_bundle": (
            master_corpus.preprocessing_selection_bundle_receipt_sha256
            == preprocessing_selection.selection_bundle_receipt_sha256
        ),
        "selected_arm_result": (
            master_corpus.preprocessing_selected_arm_result_receipt_sha256
            == preprocessing_selection.selected_arm_result_receipt_sha256
        ),
    }
    failed_lineage = tuple(
        field for field, passed in corpus_lineage_checks.items() if not passed
    )
    if failed_lineage:
        raise ValueError(
            f"Master token corpus failed derivation lineage {failed_lineage}"
        )

    source_index_raw, _ = materializer._read_stable_regular_file(
        master_corpus.path / materializer.INDEX_FILENAME,
        field="master formal token corpus index",
        max_bytes=materializer.MAX_INDEX_BYTES,
    )
    source_index = materializer._validate_index_payload(
        materializer._parse_canonical_json(
            source_index_raw, field="master formal token corpus index"
        ),
        preprocessing_authorization=authorization,
    )
    source_event_by_id = {event.event_id: event for event in master_corpus.events}
    expected_training_ids = tuple(event.event_id for event in training)
    if not set(expected_training_ids) <= set(source_event_by_id):
        raise ValueError("Fold event roster is not a subset of the master corpus")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        event_root = staging / materializer.EVENTS_DIRECTORY
        event_root.mkdir()
        event_rows: list[dict[str, object]] = []
        foundation_receipt = None
        foundation_receipt_sha256 = None
        for event in training:
            binding = source_event_by_id[event.event_id]
            source_token = load_labram_concept_tokens(
                binding.bundle_path,
                expected_manifest_sha256=binding.bundle_manifest_sha256,
            )
            checks = {
                "source_manifest": (
                    source_token.source_concept_manifest_sha256
                    == master.manifest_sha256
                ),
                "event_record": (
                    source_token.event_record_sha256 == event.event_record_sha256
                ),
                "preprocess": (
                    source_token.preprocess_receipt_sha256
                    == event.signal_preflight_receipt_sha256
                ),
                "tensor": source_token.tensor_sha256 == binding.tensor_sha256,
            }
            failed = tuple(name for name, passed in checks.items() if not passed)
            if failed:
                raise ValueError(
                    f"Master token {event.event_id} failed fold derivation {failed}"
                )
            if foundation_receipt is None:
                foundation_receipt = source_token.foundation_feature_receipt
                foundation_receipt_sha256 = (
                    source_token.foundation_feature_receipt_sha256
                )
            elif (
                source_token.foundation_feature_receipt != foundation_receipt
                or source_token.foundation_feature_receipt_sha256
                != foundation_receipt_sha256
            ):
                raise ValueError("Master corpus contains mixed foundation receipts")
            artifact = save_labram_concept_tokens(
                event_root / event.event_id,
                source_token.tokens,
                event_id=event.event_id,
                source_concept_manifest_sha256=training.manifest_sha256,
                event_record_sha256=event.event_record_sha256,
                preprocess_receipt_sha256=event.signal_preflight_receipt_sha256,
                foundation_feature_receipt=source_token.foundation_feature_receipt,
            )
            event_rows.append(
                materializer._validate_generated_token(
                    artifact,
                    event,
                    training_manifest_sha256=training.manifest_sha256,
                    foundation_feature_receipt_sha256=(
                        source_token.foundation_feature_receipt_sha256
                    ),
                )
            )
            if event_rows[-1]["tensor_sha256"] != binding.tensor_sha256:
                raise RuntimeError("Fold derivation changed a foundation tensor")
        if foundation_receipt is None or foundation_receipt_sha256 is None:
            raise ValueError("Fold derivation cannot publish an empty corpus")

        index = materializer._validate_index_payload(
            materializer._build_index_payload(
                master_binding=master_binding,
                training_binding=training_binding,
                training_role=training_role,
                training_manifest=training,
                preprocessing_authorization=authorization,
                foundation_receipt=foundation_receipt,
                foundation_receipt_sha256=foundation_receipt_sha256,
                materialization_device=torch.device(
                    str(source_index["foundation"]["materialization_device"])
                ),
                events=event_rows,
            ),
            preprocessing_authorization=authorization,
        )
        index_bytes = materializer._canonical_json_bytes(index)
        index_path = staging / materializer.INDEX_FILENAME
        index_path.write_bytes(index_bytes)
        materializer._fsync_file(index_path)
        materializer._fsync_directory(event_root)
        materializer._fsync_directory(staging)
        index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        materializer.load_formal_token_corpus(
            staging,
            expected_index_sha256=index_sha256,
            preprocessing_selection=preprocessing_selection,
        )
        if os.path.lexists(target):
            raise FileExistsError(f"Derived corpus output already exists: {target}")
        os.rename(staging, target)
        published = True
        materializer._fsync_directory(target.parent)
        return materializer.load_formal_token_corpus(
            target,
            expected_index_sha256=index_sha256,
            preprocessing_selection=preprocessing_selection,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive one formal-v4 fold corpus from a verified master corpus"
    )
    parser.add_argument("--master-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-master-bundle-manifest-sha256",
        type=materializer._sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-master-source-manifest-sha256",
        type=materializer._sha256_arg,
        required=True,
    )
    parser.add_argument("--training-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-training-bundle-manifest-sha256",
        type=materializer._sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-training-source-manifest-sha256",
        type=materializer._sha256_arg,
        required=True,
    )
    parser.add_argument("--master-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-master-token-corpus-index-sha256",
        type=materializer._sha256_arg,
        required=True,
    )
    parser.add_argument("--preprocessing-selection-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-preprocessing-selection-artifact-sha256",
        type=materializer._sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-preprocessing-protocol-receipt-sha256",
        type=materializer._sha256_arg,
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    capability = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        expected_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
    )
    artifact = derive_formal_tusz_ictal_token_corpus_view(
        master_manifest_bundle=args.master_manifest_bundle,
        expected_master_bundle_manifest_sha256=(
            args.expected_master_bundle_manifest_sha256
        ),
        expected_master_source_manifest_sha256=(
            args.expected_master_source_manifest_sha256
        ),
        training_manifest_bundle=args.training_manifest_bundle,
        expected_training_bundle_manifest_sha256=(
            args.expected_training_bundle_manifest_sha256
        ),
        expected_training_source_manifest_sha256=(
            args.expected_training_source_manifest_sha256
        ),
        master_token_corpus=args.master_token_corpus,
        expected_master_token_corpus_index_sha256=(
            args.expected_master_token_corpus_index_sha256
        ),
        preprocessing_selection=capability,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "index_sha256": artifact.index_sha256,
                "training_source_manifest_sha256": (
                    artifact.training_source_manifest_sha256
                ),
                "event_count": artifact.event_count,
                "patient_count": artifact.patient_count,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
