#!/usr/bin/env python3
"""Materialize an already-produced CerebraGloss or ELM candidate cache.

This importer does not run a teacher model and does not open EEG or private
reports.  The input is a canonical JSON envelope produced by a separately
audited inference job.  It writes only development-fold candidate caches and
keeps the Stage-0 calibration/training gate closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.artifact_ref import canonical_json_bytes  # noqa: E402
from src.evisoz.forge.teacher_candidates import (  # noqa: E402
    TEACHER_IDS,
    build_teacher_candidate_cache,
    build_teacher_candidate_materialization,
    validate_teacher_candidate_materialization,
)


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("teacher input must be a regular JSON file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("teacher input must be a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = _json(args.input.resolve(strict=True))
    required = {
        "teacher_id",
        "source_split_roster_ref",
        "teacher_model_ref",
        "events",
    }
    if set(source) != required:
        raise ValueError("teacher importer input fields drifted")
    teacher_id = source["teacher_id"]
    if teacher_id not in TEACHER_IDS:
        raise ValueError("teacher importer teacher ID is invalid")
    events = source["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("teacher importer event list is empty")
    caches = []
    for event in events:
        if type(event) is not dict or set(event) != {
            "event_id",
            "linkage_group_id",
            "outer_holdout_fold",
            "event_identity_ref",
            "source_dual_montage_cache_ref",
            "input_view",
            "input_sampling_rate_hz",
            "input_window_seconds",
            "candidate_rows",
        }:
            raise ValueError("teacher importer event fields drifted")
        caches.append(
            build_teacher_candidate_cache(
                teacher_id=teacher_id,
                event_id=event["event_id"],
                linkage_group_id=event["linkage_group_id"],
                evisoz_role="development_cv",
                outer_holdout_fold=event["outer_holdout_fold"],
                event_identity_ref=event["event_identity_ref"],
                source_dual_montage_cache_ref=event["source_dual_montage_cache_ref"],
                teacher_model_ref=source["teacher_model_ref"],
                input_view=event["input_view"],
                input_sampling_rate_hz=event["input_sampling_rate_hz"],
                input_window_seconds=event["input_window_seconds"],
                candidate_rows=event["candidate_rows"],
            )
        )
    manifest = build_teacher_candidate_materialization(
        teacher_id=teacher_id,
        source_split_roster_ref=source["source_split_roster_ref"],
        teacher_model_ref=source["teacher_model_ref"],
        caches=caches,
    )
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for cache in caches:
        event_root = output / "events" / str(cache["event_id"])
        event_root.mkdir(parents=True)
        (event_root / "candidate_cache.json").write_bytes(canonical_json_bytes(cache))
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    validate_teacher_candidate_materialization(manifest, output_root=str(output))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "teacher_id": teacher_id,
                "event_count": manifest["counts"]["event_count"],
                "candidate_count": manifest["counts"]["candidate_count"],
                "materialization_id": manifest["materialization_id"],
                "training_authorized": manifest["permissions"]["training_authorized"],
                "output": str(output),
                "receipt_sha256": manifest["receipt_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
