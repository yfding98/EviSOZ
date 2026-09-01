#!/usr/bin/env python3
"""Replay and refresh every signal receipt in a frozen TUSZ ictal manifest.

This is the required bridge when an older manifest has intact source/event
lineage but its signal-preflight hashes were issued under an obsolete EDF
receipt schema.  Only current target-free EDF preprocessing is replayed.  The
event/source/annotation/target/split rosters are immutable; any ineligible or
changed source aborts the whole atomic publication.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.edf import load_standard19_edf_event  # noqa: E402
from src.soz.data.tusz_training import (  # noqa: E402
    TUSZIctalTrainingManifest,
    load_tusz_ictal_training_manifest,
    parse_tusz_official_train_path,
    save_tusz_ictal_training_manifest,
    tusz_signal_preflight_receipt_sha256,
)


def _sha256_arg(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return normalized


def _non_preflight_event_payload(event: object) -> dict[str, object]:
    payload = dict(event.canonical_payload)
    payload.pop("signal_preflight_receipt_sha256")
    return payload


def replay_tusz_ictal_manifest_signal_preflight(
    *,
    source_manifest_bundle: str | Path,
    expected_source_bundle_manifest_sha256: str,
    expected_source_manifest_sha256: str,
    edf_root: str | Path,
    output_manifest_bundle: str | Path,
    reader_factory: Callable[[str], object] | None = None,
    progress_every: int = 50,
) -> tuple[TUSZIctalTrainingManifest, dict[str, object]]:
    """Refresh only current signal receipts and publish a strict new bundle."""

    source = load_tusz_ictal_training_manifest(
        source_manifest_bundle,
        expected_bundle_manifest_sha256=expected_source_bundle_manifest_sha256,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    if not source.preflight_performed:
        raise ValueError("Signal receipt replay requires a preflighted source manifest")
    if isinstance(progress_every, bool) or not isinstance(progress_every, int):
        raise TypeError("progress_every must be an integer")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    root = Path(edf_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("edf_root must be a regular directory")

    refreshed_events = []
    changed_count = 0
    for index, event in enumerate(source, start=1):
        source_record = parse_tusz_official_train_path(root, event.relative_edf_path)
        identity_checks = {
            "patient": source_record.patient_id == event.patient_id,
            "session": source_record.session_id == event.session_id,
            "montage": source_record.montage == event.montage,
            "record": source_record.record_id == event.record_id,
            "relative_path": (
                source_record.relative_edf_path == event.relative_edf_path
            ),
        }
        failed_identity = tuple(
            field for field, passed in identity_checks.items() if not passed
        )
        if failed_identity:
            raise ValueError(
                f"Frozen event {event.event_id} changed identity {failed_identity}"
            )
        loaded = load_standard19_edf_event(
            source_record.edf_path,
            event.event_t0_sec,
            config=source.preprocess_config,
            reader_factory=reader_factory,
        )
        if loaded.edf_receipt.edf_sha256 != event.edf_sha256:
            raise ValueError(f"Frozen EDF changed for event {event.event_id}")
        replay_sha256 = tusz_signal_preflight_receipt_sha256(loaded)
        refreshed = replace(
            event,
            signal_preflight_receipt_sha256=replay_sha256,
        )
        if _non_preflight_event_payload(refreshed) != _non_preflight_event_payload(
            event
        ):
            raise RuntimeError("Signal replay changed non-preflight event lineage")
        if replay_sha256 != event.signal_preflight_receipt_sha256:
            changed_count += 1
        refreshed_events.append(refreshed)
        if index % progress_every == 0 or index == len(source):
            print(
                json.dumps(
                    {
                        "stage": "signal_preflight_replay",
                        "completed": index,
                        "total": len(source),
                        "changed_receipt_count": changed_count,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    refreshed_manifest = replace(source, events=tuple(refreshed_events))
    invariant_checks = {
        "cohort": refreshed_manifest.cohort_receipt == source.cohort_receipt,
        "config": refreshed_manifest.preprocess_config == source.preprocess_config,
        "discovery": (
            refreshed_manifest.discovered_source_count
            == source.discovered_source_count
        ),
        "aliases": refreshed_manifest.duplicate_edf_aliases == source.duplicate_edf_aliases,
        "authorized": (
            refreshed_manifest.authorized_source_record_sha256s
            == source.authorized_source_record_sha256s
        ),
        "excluded": (
            refreshed_manifest.excluded_source_record_sha256s
            == source.excluded_source_record_sha256s
        ),
        "omissions": refreshed_manifest.omissions == source.omissions,
        "event_roster": tuple(event.event_id for event in refreshed_manifest)
        == tuple(event.event_id for event in source),
    }
    failed_invariants = tuple(
        field for field, passed in invariant_checks.items() if not passed
    )
    if failed_invariants:
        raise RuntimeError(
            f"Signal preflight replay changed frozen lineage {failed_invariants}"
        )
    artifact = save_tusz_ictal_training_manifest(
        output_manifest_bundle,
        refreshed_manifest,
    )
    verified = load_tusz_ictal_training_manifest(
        artifact.path,
        expected_bundle_manifest_sha256=artifact.bundle_manifest_sha256,
        expected_source_manifest_sha256=artifact.source_manifest_sha256,
    )
    if verified != refreshed_manifest:
        raise RuntimeError("Re-preflighted manifest changed after strict reload")
    receipt = {
        "schema_version": "tusz_ictal_signal_repreflight_receipt_v1",
        "source_bundle_manifest_sha256": expected_source_bundle_manifest_sha256,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "output_bundle_manifest_sha256": artifact.bundle_manifest_sha256,
        "output_source_manifest_sha256": artifact.source_manifest_sha256,
        "event_count": len(verified),
        "patient_count": len(verified.patient_ids),
        "changed_receipt_count": changed_count,
        "event_roster_sha256": hashlib.sha256(
            json.dumps(
                tuple(event.event_id for event in verified),
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
        "non_preflight_lineage_preserved": True,
        "targets_or_annotations_read": False,
    }
    return verified, receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-source-bundle-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--output-manifest-bundle", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _, receipt = replay_tusz_ictal_manifest_signal_preflight(
        source_manifest_bundle=args.source_manifest_bundle,
        expected_source_bundle_manifest_sha256=(
            args.expected_source_bundle_manifest_sha256
        ),
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        edf_root=args.edf_root,
        output_manifest_bundle=args.output_manifest_bundle,
    )
    print(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
