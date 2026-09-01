#!/usr/bin/env python3
"""Materialize the fail-closed published-external DeepSOZ exposure receipt.

This command opens only published split/checkpoint files and the already
reference-free posterior bundle.  It never accepts a seizure-reference,
annotation, spreadsheet, doctor-label, clinical-text, or raw-EEG argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.deepsoz_posterior_batch_validation import (  # noqa: E402
    DEEPSOZ_MATERIALIZER_CODE_SHA256,
    validate_deepsoz_posterior_batch_without_references,
)
from src.clinical_eeg_long_recording.deepsoz_reference_free_batch_validation_artifact import (  # noqa: E402
    load_deepsoz_identity_roster_binding,
)
from src.clinical_eeg_long_recording.deepsoz_published_external_exposure_attestation_v1 import (  # noqa: E402
    DEEPSOZ_UPSTREAM_COMMIT,
    DEEPSOZ_UPSTREAM_REPOSITORY_URL,
    PUBLISHED_DEEPSOZ_SELECTED_CHECKPOINT_BASENAME,
    PUBLISHED_DEEPSOZ_TRAIN_FOLD_NPY_SHA256,
    build_deepsoz_published_external_exposure_attestation_v1,
    canonical_sha256,
    validate_deepsoz_published_external_exposure_attestation_v1,
)
from src.clinical_eeg_long_recording.deepsoz_temporal_adapter import (  # noqa: E402
    PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256,
    PUBLISHED_DEEPSOZ_TEST_FOLD_NPY_SHA256,
    PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256,
)


DEFAULT_UPSTREAM = Path("/mnt/hd1/dyf/workspace/DeepSOZ")
DEFAULT_WEIGHTS = ROOT / "models/deepsoz_official_weights"
DEFAULT_POSTERIOR_BATCH = (
    ROOT / "outputs/deepsoz_stagea_source_train_oof_v1_physical_binding_20260823"
)
DEFAULT_SPLIT_ROSTER_RECEIPT = (
    ROOT / "outputs/continuous_detector_split_roster_full_v1_20260822/roster_receipt.json"
)
OUTPUT_FILENAME = "exposure_attestation.json"
TRAINING_CODE_PATHS = (
    "code/preprocess/utils_preprocess.py",
    "code/train/dataloader.py",
    "code/train/lopofn.py",
    "code/train/lopofn_finetune.py",
    "code/train/run_lopo_finetune.py",
    "code/train/run_lopo_main.py",
    "code/train/txlstm_szpool.py",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file: {path}")
    return path


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _tracked_and_clean(repository: Path, relative_path: str) -> None:
    observed = _git(repository, "ls-files", "--error-unmatch", "--", relative_path)
    if observed != relative_path:
        raise ValueError(f"upstream bound path is not uniquely tracked: {relative_path}")
    status = _git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        relative_path,
    )
    if status:
        raise ValueError(f"upstream bound path has tracked worktree drift: {relative_path}")


def _patient_roster(path: Path, *, expected_count: int) -> list[str]:
    payload = _regular_file(path, "published patient fold array").read_bytes()
    values = np.load(path, allow_pickle=False)
    if (
        values.ndim != 1
        or values.size != expected_count
        or values.dtype.kind not in "iu"
    ):
        raise ValueError(f"published patient fold array schema drifted: {path}")
    patient_ids = sorted({str(int(value)) for value in values.tolist()}, key=int)
    if len(patient_ids) != expected_count:
        raise ValueError(f"published patient fold array contains duplicates: {path}")
    if hashlib.sha256(payload).hexdigest() != _file_sha256(path):
        raise RuntimeError("published patient fold array changed during read")
    return patient_ids


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(_regular_file(path, "JSON artifact").read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _safe_posterior_path(root: Path, relative_value: object) -> Path:
    relative = PurePosixPath(str(relative_value))
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "posteriors"
        or relative.suffix != ".json"
        or ".." in relative.parts
    ):
        raise ValueError("posterior index contains an unsafe relative path")
    path = root.joinpath(*relative.parts)
    _regular_file(path, "aggregate posterior artifact")
    path.resolve(strict=True).relative_to(root)
    return path


def _fold_exposures(
    *, upstream: Path, weights: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    training_code: list[dict[str, Any]] = []
    for relative in sorted(TRAINING_CODE_PATHS):
        _tracked_and_clean(upstream, relative)
        path = _regular_file(upstream / relative, "upstream training code")
        training_code.append(
            {
                "relative_path": relative,
                "file_sha256": _file_sha256(path),
                "git_tracked": True,
                "worktree_clean": True,
            }
        )
    for fold in range(15):
        suffix = fold % 5
        train_relative = f"final_models/fold{fold}/pts_train{suffix}.npy"
        test_relative = f"final_models/fold{fold}/pts_test{suffix}.npy"
        checkpoint_relative = (
            f"final_models/fold{fold}/"
            f"{PUBLISHED_DEEPSOZ_SELECTED_CHECKPOINT_BASENAME[fold]}"
        )
        for relative in (train_relative, test_relative, checkpoint_relative):
            _tracked_and_clean(upstream, relative)
        train_path = _regular_file(upstream / train_relative, "published train array")
        test_path = _regular_file(upstream / test_relative, "published test array")
        upstream_checkpoint = _regular_file(
            upstream / checkpoint_relative, "published upstream checkpoint"
        )
        local_checkpoint = _regular_file(
            weights / f"fold{fold}.pth.tar", "normalized local checkpoint"
        )
        train_sha256 = _file_sha256(train_path)
        test_sha256 = _file_sha256(test_path)
        upstream_checkpoint_sha256 = _file_sha256(upstream_checkpoint)
        local_checkpoint_sha256 = _file_sha256(local_checkpoint)
        if train_sha256 != PUBLISHED_DEEPSOZ_TRAIN_FOLD_NPY_SHA256[fold]:
            raise ValueError(f"published fold {fold} train-array SHA-256 drifted")
        if test_sha256 != PUBLISHED_DEEPSOZ_TEST_FOLD_NPY_SHA256[fold]:
            raise ValueError(f"published fold {fold} test-array SHA-256 drifted")
        expected_checkpoint = PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256[fold]
        if (
            upstream_checkpoint_sha256 != expected_checkpoint
            or local_checkpoint_sha256 != expected_checkpoint
        ):
            raise ValueError(f"published fold {fold} checkpoint byte identity drifted")
        train = _patient_roster(train_path, expected_count=100)
        test = _patient_roster(test_path, expected_count=24)
        universe = sorted(set(train).union(test), key=int)
        if set(train).intersection(test) or len(universe) != 124:
            raise ValueError(f"published fold {fold} train/test split is not disjoint")
        rows.append(
            {
                "fold_index": fold,
                "train_relative_path": train_relative,
                "train_file_sha256": train_sha256,
                "train_patient_ids": train,
                "train_patient_roster_sha256": canonical_sha256(train),
                "test_relative_path": test_relative,
                "test_file_sha256": test_sha256,
                "test_patient_ids": test,
                "test_patient_roster_sha256": canonical_sha256(test),
                "fold_universe_patient_roster_sha256": canonical_sha256(universe),
                "train_patient_count": len(train),
                "test_patient_count": len(test),
                "fold_universe_patient_count": len(universe),
                "train_test_disjoint": True,
                "test_is_exact_complement_of_train": True,
                "checkpoint_upstream_relative_path": checkpoint_relative,
                "checkpoint_local_filename": local_checkpoint.name,
                "checkpoint_sha256": expected_checkpoint,
                "checkpoint_upstream_git_tracked": True,
                "checkpoint_upstream_worktree_clean": True,
                "checkpoint_local_upstream_byte_identical": True,
                "published_directory_colocation_attested": True,
                "training_run_receipt_available": False,
                "complete_checkpoint_training_exposure_verified": False,
                "clean_room_verified": False,
            }
        )
    return rows, training_code


def _posterior_usage(
    *,
    batch_root: Path,
    split_roster_receipt: Path,
    fold_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binding = load_deepsoz_identity_roster_binding(
        split_roster_receipt,
        selected_split="source_train",
    )
    sealed = validate_deepsoz_posterior_batch_without_references(
        batch_root,
        expected_split="source_train",
        expected_manifest_sha256=binding.source_manifest_file_sha256,
        expected_recording_ids=binding.recording_ids,
        expected_patient_ids=binding.patient_ids,
        expected_materializer_code_sha256=DEEPSOZ_MATERIALIZER_CODE_SHA256,
        require_complete_inventory=True,
    )
    validation = sealed.validation_receipt()
    batch_path = batch_root / "batch_receipt.json"
    index_path = batch_root / "posterior_index.jsonl"
    batch = _read_json(batch_path)
    index_bytes = _regular_file(index_path, "posterior index").read_bytes()
    lines = [line for line in index_bytes.splitlines() if line.strip()]
    fold_lookup = {int(row["fold_index"]): row for row in fold_rows}
    usage: list[dict[str, Any]] = []
    recording_ids: set[str] = set()
    patient_ids: set[str] = set()
    for expected_ordinal, line in enumerate(lines, start=1):
        index = json.loads(line)
        if type(index) is not dict or index.get("ordinal") != expected_ordinal:
            raise ValueError("posterior index ordinal/schema drifted")
        recording_id = str(index["recording_id"])
        patient_id = str(int(str(index["deepsoz_patient_id"])))
        artifact_path = _safe_posterior_path(
            batch_root, index["posterior_relative_path"]
        )
        artifact_bytes = artifact_path.read_bytes()
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if artifact_sha256 != index["posterior_file_sha256"]:
            raise ValueError("aggregate posterior file hash drifted after validation")
        artifact = json.loads(artifact_bytes)
        folds = list(index["held_out_fold_indices"])
        if (
            artifact.get("deepsoz_patient_id") != patient_id
            or artifact.get("recording_id") != recording_id
            or artifact.get("held_out_fold_indices") != folds
            or len(artifact.get("fold_posterior_artifact_ids", [])) != len(folds)
        ):
            raise ValueError("aggregate posterior fold/patient binding drifted")
        runtime_folds = [
            int(row["fold_index"])
            for row in artifact["posterior_runtime_receipt"][
                "held_out_fold_wall_seconds"
            ]
        ]
        if runtime_folds != folds:
            raise ValueError("aggregate posterior runtime fold set drifted")
        recording_ids.add(recording_id)
        patient_ids.add(patient_id)
        for fold, fold_artifact_id in zip(
            folds, artifact["fold_posterior_artifact_ids"]
        ):
            exposure = fold_lookup.get(int(fold))
            if exposure is None:
                raise ValueError("posterior used a fold outside the published inventory")
            if (
                patient_id not in exposure["test_patient_ids"]
                or patient_id in exposure["train_patient_ids"]
            ):
                raise ValueError(
                    "inference patient is not held out from the published fold train roster"
                )
            usage.append(
                {
                    "record_fold_ordinal": len(usage) + 1,
                    "patient_id": patient_id,
                    "recording_id": recording_id,
                    "fold_index": int(fold),
                    "fold_receipt_sha256": "PENDING-FOLD-RECEIPT",
                    "checkpoint_sha256": exposure["checkpoint_sha256"],
                    "aggregate_posterior_artifact_id": artifact[
                        "posterior_artifact_id"
                    ],
                    "aggregate_posterior_file_sha256": artifact_sha256,
                    "fold_posterior_artifact_id": str(fold_artifact_id),
                    "patient_in_published_test_roster": True,
                    "patient_absent_from_published_train_roster": True,
                    "aggregate_declares_fold": True,
                    "original_runtime_declares_fold_inference": True,
                    "legacy_fold_posterior_content_id_present": True,
                    "native_fold_posterior_payload_available": False,
                    "native_fold_checkpoint_hash_replayed": False,
                    "historical_checkpoint_load_receipt_available": False,
                    "actual_posterior_usage_verified": False,
                    "usage_verification_status": (
                        "declared_fold_content_id_and_runtime_only_pending_native_fold_payload"
                    ),
                }
            )
    if (
        len(recording_ids) != int(validation["recording_count"])
        or len(patient_ids) != int(validation["patient_count"])
    ):
        raise ValueError("posterior usage inventory differs from sealed validation")
    posterior_batch = {
        "batch_root_name": batch_root.name,
        "batch_receipt_id": str(batch["receipt_id"]),
        "batch_receipt_file_sha256": _file_sha256(batch_path),
        "posterior_index_file_sha256": _file_sha256(index_path),
        "reference_free_validation_receipt_sha256": validation["receipt_sha256"],
        "selected_split": "source_train",
        "recording_count": len(recording_ids),
        "patient_count": len(patient_ids),
        "record_fold_usage_count": len(usage),
        "materializer_code_sha256": validation["materializer_code_sha256"],
        "adapter_code_sha256": validation["adapter_code_sha256"],
        "weights_manifest_sha256": PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256,
        "all_aggregate_artifact_content_ids_verified": True,
        "all_original_runtime_fold_indices_verified": True,
        "native_per_fold_posterior_payloads_persisted": False,
        "actual_per_fold_checkpoint_usage_replayable": False,
    }
    return posterior_batch, usage


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    upstream = args.upstream_repository.resolve(strict=True)
    weights = args.weights_directory.resolve(strict=True)
    batch_root = args.posterior_batch.resolve(strict=True)
    split_roster_receipt = _regular_file(
        args.split_roster_receipt,
        "identity/split-only roster receipt",
    )
    if upstream.is_symlink() or weights.is_symlink() or batch_root.is_symlink():
        raise ValueError("attestation input directories must not be symlinks")
    observed_head = _git(upstream, "rev-parse", "HEAD")
    upstream_url = _git(upstream, "remote", "get-url", "upstream")
    if observed_head != DEEPSOZ_UPSTREAM_COMMIT:
        raise ValueError("DeepSOZ upstream HEAD differs from the pinned commit")
    if upstream_url != DEEPSOZ_UPSTREAM_REPOSITORY_URL:
        raise ValueError("DeepSOZ upstream remote URL differs from the pinned source")
    fold_rows, training_code = _fold_exposures(
        upstream=upstream, weights=weights
    )
    posterior_batch, usage = _posterior_usage(
        batch_root=batch_root,
        split_roster_receipt=split_roster_receipt,
        fold_rows=fold_rows,
    )

    # Build once to obtain the fold subreceipt hashes, then rebuild the usage
    # ledger with those exact hashes.  The first pass uses a syntactically valid
    # placeholder only and is not emitted.
    placeholder = "0" * 64
    for row in usage:
        row["fold_receipt_sha256"] = placeholder
    # Fold hashes are deterministic and independent of the usage ledger.  Use a
    # minimal no-op derivation through the public builder is impossible because
    # usage is mandatory, so mirror its one-field subreceipt rule here.
    sealed_fold_hashes = {
        int(row["fold_index"]): canonical_sha256(
            {**row, "fold_receipt_sha256": "CONTENT-ADDRESS-PENDING"}
        )
        for row in fold_rows
    }
    for row in usage:
        row["fold_receipt_sha256"] = sealed_fold_hashes[int(row["fold_index"])]

    upstream_repository = {
        "repository_url": DEEPSOZ_UPSTREAM_REPOSITORY_URL,
        "upstream_remote_url": upstream_url,
        "pinned_commit": DEEPSOZ_UPSTREAM_COMMIT,
        "observed_head_commit": observed_head,
        "head_matches_pinned_commit": True,
        "bound_paths_git_tracked": True,
        "bound_paths_worktree_clean": True,
        "selected_checkpoint_bytes_match_normalized_local": True,
        "training_code_artifacts": training_code,
        "training_run_receipts_available": False,
        "exact_environment_receipt_available": False,
        "original_preprocessing_execution_receipt_available": False,
        "checkpoint_training_patient_roster_declared_complete": False,
        "model_license_verified": False,
        "commit_signature_verified": False,
    }
    attestation = build_deepsoz_published_external_exposure_attestation_v1(
        upstream_repository=upstream_repository,
        posterior_batch=posterior_batch,
        fold_exposures=fold_rows,
        prediction_fold_usage=usage,
    )
    output = args.output_directory / OUTPUT_FILENAME
    text = json.dumps(attestation, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.verify_existing:
        existing = _read_json(output)
        validate_deepsoz_published_external_exposure_attestation_v1(existing)
        existing_text = output.read_text(encoding="utf-8")
        if existing_text != text:
            raise ValueError("existing published-external exposure receipt drifted")
    else:
        if output.exists():
            raise FileExistsError(
                "output receipt already exists; use --verify-existing to replay it"
            )
        _atomic_write(output, text)
    return attestation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a fail-closed published-external DeepSOZ exposure attestation"
        )
    )
    parser.add_argument("--upstream-repository", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--weights-directory", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--posterior-batch", type=Path, default=DEFAULT_POSTERIOR_BATCH)
    parser.add_argument(
        "--split-roster-receipt",
        type=Path,
        default=DEFAULT_SPLIT_ROSTER_RECEIPT,
        help=(
            "identity/split-only roster receipt used to fail-closed validate the "
            "complete source-train posterior inventory"
        ),
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    receipt = materialize(_parse_args())
    print(
        json.dumps(
            {
                "attestation_id": receipt["attestation_id"],
                "receipt_sha256": receipt["receipt_sha256"],
                "verification_status": receipt["verification_status"],
                "counts": receipt["counts"],
                "strict_g0a_verified": receipt["evidence_gates"][
                    "strict_g0a_verified"
                ],
                "model_training_authorized": receipt["evidence_gates"][
                    "model_training_authorized"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
