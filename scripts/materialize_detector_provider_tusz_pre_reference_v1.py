#!/usr/bin/env python3
"""Run the real EEG-only TUSZ pre-reference provider outcome factory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording import eventnet_cleanroom_registry_v1 as eventnet
from src.clinical_eeg_long_recording import (
    seizuretransformer_cleanroom_registry_v1 as st,
)
from src.clinical_eeg_long_recording.detector_provider_pre_reference_inventory_v1 import (
    materialize_detector_provider_pre_reference_inventory_v1,
)
from src.clinical_eeg_long_recording.detector_provider_tusz_outcome_factory_v1 import (
    TUSZTargetFreeProviderOutcomeFactoryV1,
    materialize_tusz_target_free_provider_record_smoke_v1,
    select_source_train_provider_smoke_record_v1,
)


DEFAULT_AUDIT_ROOT = (
    ROOT / "outputs" / "tusz_canonical_physical_signal_audit_v1_full_20260824r2"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_EVENTNET_REGISTRY = ROOT / eventnet.CONFIG_RELATIVE_PATH
DEFAULT_ST_REGISTRY = ROOT / st.CONFIG_RELATIVE_PATH
DEFAULT_SMOKE_OUTPUT = (
    ROOT
    / "outputs"
    / "clinical_eeg_tusz_provider_record_smoke_v1_20260824r4"
    / "receipt.json"
)
DEFAULT_FULL_OUTPUT = (
    ROOT
    / "outputs"
    / "clinical_eeg_detector_provider_pre_reference_inventory_v1_full_source_train_20260824r1"
)


def _read_bytes(path: Path, context: str) -> bytes:
    candidate = path.resolve(strict=True)
    if path.is_symlink() or not candidate.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return candidate.read_bytes()


def _read_json(path: Path, context: str) -> dict[str, Any]:
    raw = _read_bytes(path, context)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not readable JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must contain a JSON object")
    return value


def _write_new_json(path: Path, value: object) -> None:
    destination = path.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument(
        "--canonical-audit", type=Path, default=DEFAULT_AUDIT_ROOT / "audit.json"
    )
    parser.add_argument(
        "--physical-projection",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "physical_analysis_projection.json",
    )
    parser.add_argument(
        "--fold-plan",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "detector_cleanroom_fold_plan.json",
    )
    parser.add_argument(
        "--eventnet-registry", type=Path, default=DEFAULT_EVENTNET_REGISTRY
    )
    parser.add_argument("--st-registry", type=Path, default=DEFAULT_ST_REGISTRY)
    parser.add_argument("--target-duration-seconds", type=int, default=121)
    parser.add_argument(
        "--smoke-support",
        choices=("complete19", "lateral17"),
        default="complete19",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="smoke JSON path or full inventory directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit_bytes = _read_bytes(args.canonical_audit, "canonical physical audit")
    projection_bytes = _read_bytes(
        args.physical_projection, "canonical physical projection"
    )
    audit = json.loads(audit_bytes)
    fold_plan = _read_json(args.fold_plan, "detector clean-room fold plan")
    eventnet_registry = eventnet.load_registry(args.eventnet_registry)
    st_registry = st.load_registry(args.st_registry)
    output = args.output

    factory = TUSZTargetFreeProviderOutcomeFactoryV1(
        tusz_root=args.tusz_root,
        canonical_audit_bytes=audit_bytes,
        physical_projection_bytes=projection_bytes,
        eventnet_registry=eventnet_registry,
        seizuretransformer_registry=st_registry,
    )
    try:
        if args.mode == "smoke":
            destination = DEFAULT_SMOKE_OUTPUT if output is None else output
            source_record = select_source_train_provider_smoke_record_v1(
                fold_plan=fold_plan,
                canonical_audit=audit,
                target_duration_seconds=args.target_duration_seconds,
                support_profile=args.smoke_support,
                require_eventnet_training_tile=True,
            )
            receipt = materialize_tusz_target_free_provider_record_smoke_v1(
                factory=factory,
                source_record=source_record,
            )
            _write_new_json(destination, receipt)
            print(
                json.dumps(
                    {
                        "mode": "smoke",
                        "output": str(destination.resolve()),
                        "receipt_sha256": receipt["receipt_sha256"],
                        "recording_duration_seconds_fraction": receipt["source_record"][
                            "recording_duration_seconds_fraction"
                        ],
                        "status_by_variant": receipt["status_by_variant"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            destination = DEFAULT_FULL_OUTPUT if output is None else output
            authority = materialize_detector_provider_pre_reference_inventory_v1(
                destination,
                fold_plan=fold_plan,
                eventnet_registry=eventnet_registry,
                seizuretransformer_registry=st_registry,
                outcome_factory=factory,
            )
            factory.assert_idle()
            lifecycle = factory.lifecycle_receipt()
            print(
                json.dumps(
                    {
                        "mode": "full",
                        "output": str(destination.resolve()),
                        "manifest_receipt_sha256": authority.manifest["receipt_sha256"],
                        "process_local_authority_receipt_sha256": authority.receipt[
                            "receipt_sha256"
                        ],
                        "factory_lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
                        "serialized_bundle_alone_is_formal_authority": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    finally:
        factory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
