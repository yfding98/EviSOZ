#!/usr/bin/env python3
"""Atomically materialize one native/time-only/mask-only artifact triplet.

The command strictly reloads an existing formal ictal production run and its
bound token/annotation artifacts.  It exposes no prediction tensors, target
tensors, masks, control-fit parameters, or model-training knobs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
    load_ictal_native_eval_manifest,
    load_ictal_native_eval_token_corpus,
)
from src.soz.ictal_prediction_artifacts import (  # noqa: E402
    ICTAL_MASK_ONLY_CONTROL,
    ICTAL_TIME_ONLY_CONTROL,
    load_ictal_control_prediction_artifact,
    materialize_ictal_mask_only_control_artifact,
    materialize_ictal_native_prediction_artifact,
    materialize_ictal_time_only_control_artifact,
    relocate_verified_ictal_native_prediction_artifact,
)
from src.soz.ictal_production import (  # noqa: E402
    load_ictal_production_run,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NATIVE_DIRECTORY = "native"
_TIME_ONLY_DIRECTORY = "time_only"
_MASK_ONLY_DIRECTORY = "mask_only"


def _sha256_arg(value: str) -> str:
    normalized = str(value).strip()
    if not _SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256 digest")
    return normalized


def _output_directory(value: str | Path) -> Path:
    target = Path(os.path.abspath(value))
    if target.name in {"", ".", ".."}:
        raise ValueError("Output requires a concrete directory")
    for component in (target.parent, *target.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Output path cannot traverse symlinks")
    if not target.parent.is_dir():
        raise FileNotFoundError("Output parent does not exist")
    if os.path.lexists(target):
        raise FileExistsError(f"Output already exists: {target}")
    return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one formal ictal checkpoint and atomically emit its native, "
            "fixed time-only, and fixed mask-only prediction artifacts"
        )
    )
    parser.add_argument("--production-run", type=Path, required=True)
    parser.add_argument(
        "--expected-production-run-manifest-sha256",
        type=_sha256_arg,
        required=True,
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
    parser.add_argument("--native-evaluation-signal-preflight-bundle", type=Path)
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
    return parser


def _load_inputs(args: argparse.Namespace):
    preprocessing_selection = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        expected_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
    )
    production_run = load_ictal_production_run(
        args.production_run,
        expected_manifest_sha256=args.expected_production_run_manifest_sha256,
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
        expected_index_sha256=args.expected_training_token_corpus_index_sha256,
        preprocessing_selection=preprocessing_selection,
    )
    if production_run.manifest["selection"] == "final":
        required_signal = {
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
            name for name, value in required_signal.items() if value is None
        )
        if missing:
            raise ValueError(
                "Final selection requires externally pinned source-dev signal "
                f"preflight arguments: {missing}"
            )
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
            raise ValueError(
                "Fold selections cannot accept source-dev signal-preflight inputs"
            )
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
            preprocessing_selection=preprocessing_selection,
        )
    return production_run, training_manifest, training_corpus, native_manifest, native_corpus


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    (
        production_run,
        training_manifest,
        training_corpus,
        native_manifest,
        native_corpus,
    ) = _load_inputs(args)
    target = _output_directory(args.output_directory)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        native = materialize_ictal_native_prediction_artifact(
            production_run=production_run,
            training_manifest=training_manifest,
            training_corpus=training_corpus,
            native_evaluation_manifest=native_manifest,
            native_evaluation_corpus=native_corpus,
            edf_root=args.edf_root,
            output_directory=staging / _NATIVE_DIRECTORY,
        )
        time_only = materialize_ictal_time_only_control_artifact(
            native_prediction=native,
            expected_native_prediction_artifact_sha256=native.artifact_sha256,
            expected_native_prediction_receipt_sha256=native.receipt_sha256,
            output_directory=staging / _TIME_ONLY_DIRECTORY,
        )
        mask_only = materialize_ictal_mask_only_control_artifact(
            native_prediction=native,
            expected_native_prediction_artifact_sha256=native.artifact_sha256,
            expected_native_prediction_receipt_sha256=native.receipt_sha256,
            output_directory=staging / _MASK_ONLY_DIRECTORY,
        )
        hashes = {
            "native_artifact_sha256": native.artifact_sha256,
            "native_receipt_sha256": native.receipt_sha256,
            "time_only_artifact_sha256": time_only.artifact_sha256,
            "time_only_receipt_sha256": time_only.receipt_sha256,
            "mask_only_artifact_sha256": mask_only.artifact_sha256,
            "mask_only_receipt_sha256": mask_only.receipt_sha256,
        }
        if os.path.lexists(target):
            raise FileExistsError(f"Output already exists: {target}")
        os.rename(staging, target)
        published = True
        _fsync_directory(target.parent)

        native = relocate_verified_ictal_native_prediction_artifact(
            native,
            target / _NATIVE_DIRECTORY,
        )
        load_ictal_control_prediction_artifact(
            target / _TIME_ONLY_DIRECTORY,
            native_prediction=native,
            expected_control_type=ICTAL_TIME_ONLY_CONTROL,
            expected_artifact_sha256=hashes["time_only_artifact_sha256"],
            expected_receipt_sha256=hashes["time_only_receipt_sha256"],
        )
        load_ictal_control_prediction_artifact(
            target / _MASK_ONLY_DIRECTORY,
            native_prediction=native,
            expected_control_type=ICTAL_MASK_ONLY_CONTROL,
            expected_artifact_sha256=hashes["mask_only_artifact_sha256"],
            expected_receipt_sha256=hashes["mask_only_receipt_sha256"],
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    print(
        json.dumps(
            {
                "path": str(target),
                "selection": production_run.manifest["selection"],
                **hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
