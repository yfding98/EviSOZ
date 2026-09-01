#!/usr/bin/env python3
"""Materialize the TUSZ v2 audit roster and deduplicated analysis identities."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.tusz_complete_detector_roster_v1 import (  # noqa: E402
    TUSZ_V203_EXPECTED_INVENTORY,
)
from src.clinical_eeg_long_recording.tusz_complete_detector_roster_v2 import (  # noqa: E402
    build_tusz_analysis_identity_projection_v2,
    build_tusz_complete_detector_roster_v2,
)


DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")


def _load_expected_inventory(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return TUSZ_V203_EXPECTED_INVENTORY
    raw = path.read_bytes()
    if not raw:
        raise ValueError("expected inventory JSON is empty")
    parsed = json.loads(raw.decode("utf-8"))
    if type(parsed) is not dict:
        raise ValueError("expected inventory JSON must be an object")
    return parsed


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise ValueError("output already exists; roster v2 artifacts are append-only")
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
        if path.exists():
            raise ValueError("output appeared during roster v2 materialization")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retain every official TUSZ path for audit while publishing a "
            "separate exact-container-deduplicated analysis identity roster"
        )
    )
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--expected-inventory", type=Path)
    parser.add_argument("--roster-output", type=Path)
    parser.add_argument("--analysis-projection-output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    outputs_supplied = (
        arguments.roster_output is not None
        and arguments.analysis_projection_output is not None
    )
    if (arguments.roster_output is None) != (
        arguments.analysis_projection_output is None
    ):
        raise SystemExit("roster and analysis projection outputs must be paired")
    if arguments.preflight_only == outputs_supplied:
        raise SystemExit(
            "choose exactly one of --preflight-only or the paired output paths"
        )
    if outputs_supplied:
        assert arguments.roster_output is not None
        assert arguments.analysis_projection_output is not None
        if (
            arguments.roster_output.absolute()
            == arguments.analysis_projection_output.absolute()
        ):
            raise SystemExit("roster and analysis projection outputs must differ")
        if (
            arguments.roster_output.exists()
            or arguments.analysis_projection_output.exists()
        ):
            raise ValueError("paired roster v2 outputs are append-only")

    roster = build_tusz_complete_detector_roster_v2(
        tusz_root=arguments.tusz_root,
        expected_inventory=_load_expected_inventory(arguments.expected_inventory),
    )
    projection = build_tusz_analysis_identity_projection_v2(roster)
    summary = {
        "roster_id": roster["roster_id"],
        "roster_receipt_sha256": roster["receipt_sha256"],
        "audit_recording_count": roster["observed_inventory"][
            "total_recording_count"
        ],
        "equivalence_inventory": roster[
            "exact_container_equivalence_inventory"
        ],
        "analysis_projection_id": projection["projection_id"],
        "analysis_projection_receipt_sha256": projection["receipt_sha256"],
        "analysis_identity_count": len(projection["records"]),
        "reference_files_opened": projection["reference_access_receipt"][
            "reference_files_opened"
        ],
    }
    if arguments.preflight_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        assert arguments.roster_output is not None
        assert arguments.analysis_projection_output is not None
        _write_new_json(arguments.roster_output, roster)
        _write_new_json(arguments.analysis_projection_output, projection)
        summary["roster_output"] = str(arguments.roster_output)
        summary["analysis_projection_output"] = str(
            arguments.analysis_projection_output
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
