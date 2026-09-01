#!/usr/bin/env python3
"""Build or source-validate the target-blind TUSZ detector five-fold plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.tusz_detector_cleanroom_fold_plan_v1 import (  # noqa: E402
    build_tusz_detector_cleanroom_fold_plan_v1,
    validate_tusz_detector_cleanroom_fold_plan_v1,
)


def _load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a regular non-symlink JSON file: {path}")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"JSON artifact is empty: {path}")
    return json.loads(raw.decode("utf-8"))


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"append-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise ValueError(f"append-only output appeared during write: {path}")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--analysis-projection", required=True, type=Path)
    parser.add_argument("--canonical-physical-audit", required=True, type=Path)
    parser.add_argument("--canonical-physical-projection", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate-plan",
        type=Path,
        help="validate an existing plan against all sources instead of building",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    roster = _load_json(arguments.roster)
    projection = _load_json(arguments.analysis_projection)
    audit = _load_json(arguments.canonical_physical_audit)
    physical = _load_json(arguments.canonical_physical_projection)
    if arguments.validate_plan is not None:
        if arguments.output is not None:
            raise ValueError("--output is not accepted with --validate-plan")
        plan = validate_tusz_detector_cleanroom_fold_plan_v1(
            _load_json(arguments.validate_plan),
            source_roster=roster,
            source_analysis_projection=projection,
            source_canonical_physical_audit=audit,
            source_canonical_physical_projection=physical,
        )
        status = "valid"
    else:
        if arguments.output is None:
            raise ValueError("--output is required when building a plan")
        plan = build_tusz_detector_cleanroom_fold_plan_v1(
            source_roster=roster,
            source_analysis_projection=projection,
            source_canonical_physical_audit=audit,
            source_canonical_physical_projection=physical,
        )
        _write_new_json(arguments.output, plan)
        status = "built"
    print(
        json.dumps(
            {
                "status": status,
                "plan_id": plan["plan_id"],
                "receipt_sha256": plan["receipt_sha256"],
                "fold_count": plan["fold_count"],
                "canonical_physical_audit_id": plan["source_binding"][
                    "source_canonical_physical_audit_id"
                ],
                "canonical_physical_audit_shard_count": plan["source_binding"][
                    "source_canonical_physical_audit_shard_count"
                ],
                "source_train_patient_count": plan["source_split_rosters"][
                    "source_train"
                ]["patient_count"],
                "source_train_recording_count": plan["source_split_rosters"][
                    "source_train"
                ]["recording_count"],
                "source_eval_execution_authorized": plan["role_permissions"][
                    "source_eval"
                ]["execution_authorized_by_this_plan"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
