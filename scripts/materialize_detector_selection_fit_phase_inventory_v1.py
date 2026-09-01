#!/usr/bin/env python3
"""Actual-byte replay five selection-fit receipts and write one inventory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from src.clinical_eeg_long_recording.detector_selection_fit_phase_inventory_v1 import (
    build_detector_selection_fit_phase_inventory_v1,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE_DIR = (
    ROOT / "outputs" / "clinical_eeg_detector_selection_fit_phase_v1_20260824"
)
DEFAULT_FOLD_PLAN = (
    ROOT
    / "outputs"
    / "tusz_canonical_physical_signal_audit_v1_full_20260824r2"
    / "detector_cleanroom_fold_plan.json"
)
DEFAULT_REGISTRY = (
    ROOT / "configs" / "clinical_eeg_detector_fold_reference_authority_registry_v1.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")


def _write_new_json(path: Path, value: object) -> None:
    target = path.resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite inventory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, default=DEFAULT_PHASE_DIR)
    parser.add_argument("--fold-plan", type=Path, default=DEFAULT_FOLD_PLAN)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_PHASE_DIR / "inventory.json"
    )
    args = parser.parse_args(argv)
    paths = [args.phase_dir / f"fold-{fold}.json" for fold in range(5)]
    inventory = build_detector_selection_fit_phase_inventory_v1(
        phase_receipt_paths=paths,
        fold_plan_path=args.fold_plan,
        registry_path=args.registry,
        replay_reference_root=args.reference_root,
    )
    _write_new_json(args.output, inventory)
    print(
        json.dumps(
            {
                "inventory_id": inventory["inventory_id"],
                "receipt_sha256": inventory["receipt_sha256"],
                **inventory["aggregate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
