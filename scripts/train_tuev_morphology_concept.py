#!/usr/bin/env python3
"""Train and atomically publish the formal TUEV CE6 morphology head.

The command reloads the first-party EDF-to-LaBraM master corpus under the
same strict cohort/preprocessing authorization used at materialization.  The
fit/held roles are derived from that opaque authorization; no label tensor,
token tensor, split roster, morphology class weight, head hyperparameter, or
preprocessing arm is accepted from the caller.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


_REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_observed_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _observed_cublas_workspace is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _REQUIRED_CUBLAS_WORKSPACE_CONFIG
elif _observed_cublas_workspace != _REQUIRED_CUBLAS_WORKSPACE_CONFIG:
    raise RuntimeError(
        "Morphology training requires CUBLAS_WORKSPACE_CONFIG=':4096:8'; "
        f"observed {_observed_cublas_workspace!r}"
    )

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tuev_morphology_master_corpus import (  # noqa: E402
    _sha256_arg,
    build_arg_parser as build_corpus_arg_parser,
    load_authorized_morphology_inputs,
)
from src.soz.data.tuev_morphology import (  # noqa: E402
    derive_tuev_morphology_fold_manifest,
)
from src.soz.models.concept_heads import MorphologyEvidenceHead  # noqa: E402
from src.soz.morphology_checkpoint import (  # noqa: E402
    load_morphology_checkpoint,
    save_morphology_checkpoint,
)
from src.soz.morphology_token_io import (  # noqa: E402
    load_morphology_training_group_tokens,
)
from src.soz.morphology_training import (  # noqa: E402
    MorphologyTrainingConfig,
    build_morphology_bag_dataset,
    evaluate_morphology_groups,
    train_fixed_epoch_morphology_head,
)
from src.soz.tuev_morphology_producer import (  # noqa: E402
    load_tuev_morphology_master_corpus,
)


_FROZEN_CONFIG = MorphologyTrainingConfig()
_FROZEN_TOKEN_DIM = 200
_FROZEN_HIDDEN_DIM = 128


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_corpus_arg_parser()
    parser.description = (
        "Train the fixed formal TUEV CE6 morphology head from a strictly "
        "replayed first-party master corpus"
    )
    parser.add_argument("--master-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-master-corpus-bundle-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-master-corpus-producer-receipt-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-master-corpus-token-index-sha256",
        type=_sha256_arg,
        required=True,
    )
    return parser


def _safe_new_output(value: str | Path) -> Path:
    target = Path(os.path.abspath(value))
    if target.name in {"", ".", ".."}:
        raise ValueError("Output requires a concrete directory")
    for component in (target.parent, *target.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Output path cannot traverse symlinks")
    if not target.parent.is_dir():
        raise FileNotFoundError("Output parent does not exist")
    if os.path.lexists(target):
        raise FileExistsError(f"Morphology checkpoint already exists: {target}")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    target = _safe_new_output(args.output_directory)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    (
        _verified_target,
        _public_protocol,
        preflight,
        authorization,
        holding_manifest,
        preprocessing_selection,
    ) = load_authorized_morphology_inputs(args, replay_live_source=False)
    master = load_tuev_morphology_master_corpus(
        args.master_corpus,
        edf_root=args.edf_root,
        holding_manifest=holding_manifest,
        preflight=preflight,
        cohort_authorization=authorization,
        preprocessing_selection=preprocessing_selection,
        labram_modeling_path=args.labram_modeling_path,
        labram_checkpoint_path=args.labram_checkpoint_path,
        expected_bundle_manifest_sha256=(
            args.expected_master_corpus_bundle_manifest_sha256
        ),
        expected_producer_receipt_sha256=(
            args.expected_master_corpus_producer_receipt_sha256
        ),
        expected_token_index_sha256=(
            args.expected_master_corpus_token_index_sha256
        ),
        replay_live_signal=False,
    )
    fold_manifest = derive_tuev_morphology_fold_manifest(
        holding_manifest, authorization
    )
    fit_dataset = build_morphology_bag_dataset(
        fold_manifest,
        master.token_corpus,
        role="fit",
        preload_tokens=True,
    )
    held_dataset = build_morphology_bag_dataset(
        fold_manifest,
        master.token_corpus,
        role="held",
        preload_tokens=True,
    )
    first_binding = master.bindings[0]
    first_tokens = load_morphology_training_group_tokens(
        first_binding.bundle_path,
        expected_manifest_sha256=first_binding.bundle_manifest_sha256,
    )
    if (
        first_tokens.foundation_feature_receipt_sha256
        != master.token_corpus.foundation_feature_receipt_sha256
    ):
        raise ValueError("Morphology master corpus changed its foundation receipt")

    head = MorphologyEvidenceHead(
        token_dim=_FROZEN_TOKEN_DIM,
        hidden_dim=_FROZEN_HIDDEN_DIM,
    )
    training_run = train_fixed_epoch_morphology_head(
        head,
        fit_dataset,
        fold_manifest,
        config=_FROZEN_CONFIG,
        device=device,
    )
    evaluation = evaluate_morphology_groups(
        head,
        held_dataset,
        checkpoint_or_run_sha256=training_run.receipt_sha256,
        device=device,
        crop_microbatch_size=_FROZEN_CONFIG.crop_microbatch_size,
        ece_bins=15,
    )
    checkpoint = save_morphology_checkpoint(
        target,
        head,
        training_run=training_run,
        evaluation=evaluation,
        fold_manifest=fold_manifest,
        foundation_feature_receipt=first_tokens.foundation_feature_receipt,
    )
    loaded = load_morphology_checkpoint(
        checkpoint.path,
        fold_manifest,
        expected_manifest_sha256=checkpoint.manifest_sha256,
        expected_master_manifest_sha256=holding_manifest.manifest_sha256,
        expected_master_token_corpus_index_sha256=master.index_sha256,
    )
    if loaded.training_run_sha256 != training_run.receipt_sha256:
        raise RuntimeError("Published morphology checkpoint changed its training run")
    print(
        json.dumps(
            {
                "checkpoint_manifest_sha256": loaded.manifest_sha256,
                "dataset_role": evaluation.dataset_role,
                "authorized_fit_role_group_count": len(fold_manifest.fit_group_ids),
                "authorized_held_role_group_count": len(fold_manifest.held_group_ids),
                "fit_target_bearing_group_count": len(training_run.fit_group_ids),
                "held_target_bearing_group_count": len(evaluation.group_ids),
                "native_group_macro_balanced_accuracy": (
                    evaluation.group_macro_balanced_accuracy
                ),
                "native_target_count": evaluation.target_count,
                "output_semantics": "tuev_native_ce6_morphology_not_soz",
                "path": str(loaded.path),
                "selected_arm_id": master.selected_arm_id,
                "training_epochs": _FROZEN_CONFIG.fixed_epochs,
                "weighted_brier": evaluation.weighted_brier,
                "weighted_ece": evaluation.weighted_ece,
                "weighted_nll": evaluation.weighted_nll,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
