#!/usr/bin/env python3
"""Derive five TUSZ ictal OOF-fold manifests from frozen artifacts only.

This command deliberately has no EDF-root argument.  It loads a persisted
master event manifest, rebuilds the frozen OOF protocol against the exact
DeepSOZ registry and public ledger, and then uses
``derive_tusz_ictal_training_manifest``.  Consequently it cannot discover or
re-read EDF files or annotation sidecars.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.concept_oof import (  # noqa: E402
    IctalConceptOOFProtocolArtifact,
    load_ictal_concept_oof_protocol,
)
from src.soz.data.deepsoz import (  # noqa: E402
    DeepSOZReferenceRegistry,
    build_deepsoz_reference_registry,
)
from src.soz.data.public_ledger_builder import (  # noqa: E402
    TUSZDeepSOZPublicLedgerArtifact,
    load_tusz_deepsoz_public_ledger_build,
)
from src.soz.data.tusz_training import (  # noqa: E402
    TUSZIctalTrainingManifest,
    derive_tusz_ictal_training_manifest,
    load_tusz_ictal_training_manifest,
    save_tusz_ictal_training_manifest,
)


SUMMARY_SCHEMA = "tusz_ictal_oof_fold_manifest_summary_v1"
SUMMARY_FILENAME = "summary.json"
FOLD_DIRECTORY_TEMPLATE = "fold_{fold}"
EXPECTED_FOLDS = tuple(range(5))
MAX_REGISTRY_CSV_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return text


def _optional_sha256(value: str) -> str:
    try:
        return _require_sha256(value, field="expected SHA256")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Summary contains non-canonical JSON data") from exc
    return (encoded + "\n").encode("utf-8")


def _reject_symlink_components(path: Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot contain symlink components")
    return absolute


def _read_stable_regular_file(
    path: str | Path,
    *,
    field: str,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    source = _reject_symlink_components(Path(path), field=field)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{field} must be a regular file: {source}")
    before = source.stat()
    if max_bytes is not None and (before.st_size < 1 or before.st_size > max_bytes):
        raise ValueError(f"{field} has an invalid size")
    payload = source.read_bytes()
    after = source.stat()
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_fingerprint != after_fingerprint:
        raise RuntimeError(f"{field} changed while it was read")
    return payload, hashlib.sha256(payload).hexdigest()


def _check_expected_sha(
    actual: str,
    expected: str | None,
    *,
    field: str,
) -> None:
    if expected is None:
        return
    if actual != _require_sha256(expected, field=field):
        raise ValueError(f"{field} does not match the persisted input")


def _parse_registry_csv(
    payload: bytes,
    *,
    field: str,
    allow_deepsoz_pz_pair: bool = False,
) -> pd.DataFrame:
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} must be strict UTF-8 CSV") from exc
    try:
        header = next(csv.reader(io.StringIO(text, newline="")))
    except (csv.Error, StopIteration) as exc:
        raise ValueError(f"{field} has no valid CSV header") from exc
    if not header or any(not name.strip() for name in header):
        raise ValueError(f"{field} contains an empty CSV column name")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    allowed_pz_pair = (
        allow_deepsoz_pz_pair
        and duplicates == ["pz"]
        and header.count("pz") == 2
    )
    if duplicates and not allowed_pz_pair:
        raise ValueError(f"{field} contains duplicate CSV columns: {duplicates}")
    try:
        frame = pd.read_csv(io.StringIO(text, newline=""))
    except (pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"{field} is not a valid CSV table") from exc
    if frame.columns.duplicated().any():
        raise ValueError(f"{field} contains duplicate parsed columns")
    if allowed_pz_pair and not {"pz", "pz.1"} <= set(frame.columns):
        raise ValueError(
            f"{field} did not preserve the two upstream PZ columns as pz/pz.1"
        )
    return frame


def load_bound_deepsoz_registry(
    source_csv: str | Path,
    split_csv: str | Path,
    *,
    expected_source_sha256: str | None = None,
    expected_split_sha256: str | None = None,
) -> tuple[DeepSOZReferenceRegistry, dict[str, str]]:
    """Load both registry tables once and bind their exact persisted bytes."""

    source_bytes, source_sha = _read_stable_regular_file(
        source_csv,
        field="DeepSOZ source CSV",
        max_bytes=MAX_REGISTRY_CSV_BYTES,
    )
    split_bytes, split_sha = _read_stable_regular_file(
        split_csv,
        field="DeepSOZ split CSV",
        max_bytes=MAX_REGISTRY_CSV_BYTES,
    )
    _check_expected_sha(
        source_sha,
        expected_source_sha256,
        field="expected_deepsoz_source_sha256",
    )
    _check_expected_sha(
        split_sha,
        expected_split_sha256,
        field="expected_deepsoz_split_sha256",
    )
    registry = build_deepsoz_reference_registry(
        _parse_registry_csv(
            source_bytes,
            field="DeepSOZ source CSV",
            allow_deepsoz_pz_pair=True,
        ),
        _parse_registry_csv(split_bytes, field="DeepSOZ split CSV"),
    )
    return registry, {
        "source_csv_sha256": source_sha,
        "split_csv_sha256": split_sha,
    }


def load_bound_master_manifest(
    bundle_directory: str | Path,
    *,
    expected_bundle_sha256: str | None = None,
    expected_source_sha256: str | None = None,
) -> tuple[TUSZIctalTrainingManifest, dict[str, str]]:
    """Strictly load a manifest against hashes of its current exact bytes."""

    bundle = _reject_symlink_components(
        Path(bundle_directory), field="master manifest bundle"
    )
    manifest_bytes, bundle_sha = _read_stable_regular_file(
        bundle / "manifest.json", field="master manifest.json"
    )
    receipt_bytes, receipt_sha = _read_stable_regular_file(
        bundle / "receipt.json", field="master receipt.json"
    )
    del manifest_bytes, receipt_bytes
    _check_expected_sha(
        bundle_sha,
        expected_bundle_sha256,
        field="expected_master_bundle_sha256",
    )
    _check_expected_sha(
        receipt_sha,
        expected_source_sha256,
        field="expected_master_source_sha256",
    )
    manifest = load_tusz_ictal_training_manifest(
        bundle,
        expected_bundle_manifest_sha256=bundle_sha,
        expected_source_manifest_sha256=receipt_sha,
    )
    return manifest, {
        "bundle_manifest_sha256": bundle_sha,
        "source_manifest_sha256": manifest.manifest_sha256,
        "receipt_sha256": receipt_sha,
        "cohort_receipt_sha256": manifest.cohort_receipt.receipt_sha256,
    }


def _fold_counts(manifest: TUSZIctalTrainingManifest) -> dict[str, object]:
    seizure_types = Counter(event.seizure_type for event in manifest)
    omission_reasons = Counter(
        reason for omission in manifest.omissions for reason in omission.reasons
    )
    return {
        "event_count": len(manifest),
        "patient_count": len(manifest.patient_ids),
        "omission_count": len(manifest.omissions),
        "authorized_source_count": len(
            manifest.authorized_source_record_sha256s
        ),
        "excluded_source_count": len(manifest.excluded_source_record_sha256s),
        "duplicate_edf_alias_count": len(manifest.duplicate_edf_aliases),
        "seizure_type_counts": dict(sorted(seizure_types.items())),
        "omission_reason_counts": dict(sorted(omission_reasons.items())),
    }


def _validate_cross_artifact_lineage(
    master: TUSZIctalTrainingManifest,
    registry_hashes: Mapping[str, str],
    public_artifact: TUSZDeepSOZPublicLedgerArtifact,
    protocol_artifact: IctalConceptOOFProtocolArtifact,
) -> None:
    protocol = protocol_artifact.protocol
    if registry_hashes["split_csv_sha256"] != protocol.receipt.split_manifest_sha256:
        raise ValueError(
            "DeepSOZ split CSV bytes do not match the split SHA bound by "
            "the OOF protocol"
        )
    if protocol.receipt.public_ledger_build_sha256 != public_artifact.build_sha256:
        raise ValueError("OOF protocol and public-ledger build SHA disagree")
    if master.cohort_receipt.ledger_sha256 != protocol.receipt.ledger_sha256:
        raise ValueError("Master manifest uses a different public overlap ledger")
    if (
        master.cohort_receipt.ledger_receipt_sha256
        != protocol.receipt.ledger_receipt_sha256
    ):
        raise ValueError(
            "Master manifest overlap-ledger receipt disagrees with protocol"
        )
    final_cohort = protocol.final_plan.training_cohort
    if master.cohort_receipt.receipt_sha256 != final_cohort.receipt.receipt_sha256:
        raise ValueError(
            "Master manifest must bind the exact final-plan training cohort"
        )
    if tuple(plan.oof_fold for plan in protocol.fold_plans) != EXPECTED_FOLDS:
        raise ValueError("OOF protocol does not contain canonical folds 0 through 4")


def _build_summary(
    *,
    master: TUSZIctalTrainingManifest,
    master_hashes: Mapping[str, str],
    registry: DeepSOZReferenceRegistry,
    registry_hashes: Mapping[str, str],
    public_artifact: TUSZDeepSOZPublicLedgerArtifact,
    protocol_artifact: IctalConceptOOFProtocolArtifact,
    fold_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    protocol = protocol_artifact.protocol
    eligible_counts = {
        split: len(registry.for_split(split))
        for split in ("source_train", "source_dev", "source_eval")
    }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "serialization": "canonical_json_utf8_newline_no_pickle",
        "preflight_performed": master.preflight_performed,
        "inputs": {
            "master_manifest": dict(master_hashes),
            "deepsoz_registry": {
                **dict(registry_hashes),
                "eligible_patient_counts": eligible_counts,
            },
            "public_ledger": {
                "bundle_sha256": public_artifact.bundle_sha256,
                "build_sha256": public_artifact.build_sha256,
                "ledger_sha256": public_artifact.ledger.receipt.ledger_sha256,
                "ledger_receipt_sha256": (
                    public_artifact.ledger.receipt.receipt_sha256
                ),
            },
            "oof_protocol": {
                "artifact_sha256": protocol_artifact.artifact_sha256,
                "protocol_sha256": protocol_artifact.protocol_sha256,
                "split_manifest_sha256": protocol.receipt.split_manifest_sha256,
                "public_ledger_build_sha256": (
                    protocol.receipt.public_ledger_build_sha256
                ),
                "final_plan_receipt_sha256": (
                    protocol.final_plan.receipt.receipt_sha256
                ),
            },
        },
        "master_counts": _fold_counts(master),
        "folds": list(fold_rows),
    }


def derive_and_publish_fold_manifests(
    *,
    output_directory: str | Path,
    master: TUSZIctalTrainingManifest,
    master_hashes: Mapping[str, str],
    registry: DeepSOZReferenceRegistry,
    registry_hashes: Mapping[str, str],
    public_artifact: TUSZDeepSOZPublicLedgerArtifact,
    protocol_artifact: IctalConceptOOFProtocolArtifact,
) -> tuple[dict[str, object], str]:
    """Derive, validate, and atomically publish one closed five-fold bundle."""

    _validate_cross_artifact_lineage(
        master, registry_hashes, public_artifact, protocol_artifact
    )
    output = _reject_symlink_components(
        Path(output_directory), field="fold-manifest output"
    )
    if output.name in {"", ".", ".."}:
        raise ValueError("fold-manifest output requires a concrete directory name")
    if os.path.lexists(output):
        raise FileExistsError(f"Fold-manifest output already exists: {output}")
    parent = _reject_symlink_components(output.parent, field="output parent")
    if not parent.is_dir():
        raise FileNotFoundError("Fold-manifest output parent does not exist")

    protocol = protocol_artifact.protocol
    derived_manifests: list[tuple[int, TUSZIctalTrainingManifest]] = []
    for plan in protocol.fold_plans:
        fold = int(plan.oof_fold)
        derived = derive_tusz_ictal_training_manifest(
            master, plan.training_cohort
        )
        if derived.cohort_receipt.receipt_sha256 != (
            plan.training_cohort.receipt.receipt_sha256
        ):
            raise RuntimeError("Derived manifest lost its fold cohort binding")
        if derived.derived_from_manifest_sha256 != master.manifest_sha256:
            raise RuntimeError("Derived manifest lost its master lineage")
        if derived.preflight_performed != master.preflight_performed:
            raise RuntimeError("Derived manifest changed the signal-preflight policy")
        derived_manifests.append((fold, derived))

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    published = False
    try:
        fold_rows: list[dict[str, object]] = []
        for fold, derived in derived_manifests:
            plan = protocol.for_fold(fold)
            relative_bundle = FOLD_DIRECTORY_TEMPLATE.format(fold=fold)
            artifact = save_tusz_ictal_training_manifest(
                staging / relative_bundle, derived
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "bundle_directory": relative_bundle,
                    "preflight_performed": derived.preflight_performed,
                    "derived_from_master_manifest_sha256": (
                        derived.derived_from_manifest_sha256
                    ),
                    "fold_plan_receipt_sha256": plan.receipt.receipt_sha256,
                    "cohort_receipt_sha256": (
                        derived.cohort_receipt.receipt_sha256
                    ),
                    "training_target_roster_sha256": (
                        plan.receipt.training_target_roster_sha256
                    ),
                    "held_out_target_roster_sha256": (
                        plan.receipt.held_out_target_roster_sha256
                    ),
                    "held_out_public_roster_sha256": (
                        plan.receipt.held_out_public_roster_sha256
                    ),
                    "bundle_manifest_sha256": artifact.bundle_manifest_sha256,
                    "source_manifest_sha256": artifact.source_manifest_sha256,
                    "receipt_sha256": artifact.receipt_sha256,
                    "counts": _fold_counts(derived),
                }
            )
        summary = _build_summary(
            master=master,
            master_hashes=master_hashes,
            registry=registry,
            registry_hashes=registry_hashes,
            public_artifact=public_artifact,
            protocol_artifact=protocol_artifact,
            fold_rows=fold_rows,
        )
        summary_bytes = _canonical_json_bytes(summary)
        summary_sha = hashlib.sha256(summary_bytes).hexdigest()
        summary_path = staging / SUMMARY_FILENAME
        with summary_path.open("xb") as handle:
            handle.write(summary_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if os.path.lexists(output):
            raise FileExistsError(f"Fold-manifest output already exists: {output}")
        os.rename(staging, output)
        published = True
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return summary, summary_sha


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive five overlap-safe TUSZ ictal OOF fold manifests without "
            "reading EDF or annotation files"
        )
    )
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--oof-protocol", type=Path, required=True)
    parser.add_argument("--public-ledger", type=Path, required=True)
    parser.add_argument("--deepsoz-source-csv", type=Path, required=True)
    parser.add_argument("--deepsoz-split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-master-bundle-sha256", type=_optional_sha256)
    parser.add_argument("--expected-master-source-sha256", type=_optional_sha256)
    parser.add_argument("--expected-oof-artifact-sha256", type=_optional_sha256)
    parser.add_argument("--expected-oof-protocol-sha256", type=_optional_sha256)
    parser.add_argument("--expected-public-ledger-bundle-sha256", type=_optional_sha256)
    parser.add_argument("--expected-public-ledger-build-sha256", type=_optional_sha256)
    parser.add_argument("--expected-deepsoz-source-sha256", type=_optional_sha256)
    parser.add_argument("--expected-deepsoz-split-sha256", type=_optional_sha256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry, registry_hashes = load_bound_deepsoz_registry(
        args.deepsoz_source_csv,
        args.deepsoz_split_csv,
        expected_source_sha256=args.expected_deepsoz_source_sha256,
        expected_split_sha256=args.expected_deepsoz_split_sha256,
    )

    public_file = Path(args.public_ledger) / "public_ledger_build.json"
    _, public_bundle_sha = _read_stable_regular_file(
        public_file, field="public-ledger artifact"
    )
    _check_expected_sha(
        public_bundle_sha,
        args.expected_public_ledger_bundle_sha256,
        field="expected_public_ledger_bundle_sha256",
    )
    public_artifact = load_tusz_deepsoz_public_ledger_build(
        args.public_ledger,
        expected_bundle_sha256=public_bundle_sha,
        expected_build_sha256=args.expected_public_ledger_build_sha256,
    )

    protocol_file = Path(args.oof_protocol) / "ictal_concept_oof_protocol.json"
    _, protocol_artifact_sha = _read_stable_regular_file(
        protocol_file, field="OOF protocol artifact"
    )
    _check_expected_sha(
        protocol_artifact_sha,
        args.expected_oof_artifact_sha256,
        field="expected_oof_artifact_sha256",
    )
    protocol_artifact = load_ictal_concept_oof_protocol(
        args.oof_protocol,
        registry,
        public_artifact,
        expected_artifact_sha256=protocol_artifact_sha,
        expected_protocol_sha256=args.expected_oof_protocol_sha256,
    )

    master, master_hashes = load_bound_master_manifest(
        args.master_manifest,
        expected_bundle_sha256=args.expected_master_bundle_sha256,
        expected_source_sha256=args.expected_master_source_sha256,
    )
    _, summary_sha = derive_and_publish_fold_manifests(
        output_directory=args.output_dir,
        master=master,
        master_hashes=master_hashes,
        registry=registry,
        registry_hashes=registry_hashes,
        public_artifact=public_artifact,
        protocol_artifact=protocol_artifact,
    )
    print(
        json.dumps(
            {
                "fold_count": len(EXPECTED_FOLDS),
                "output_directory": str(args.output_dir),
                "summary_file": SUMMARY_FILENAME,
                "summary_sha256": summary_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
