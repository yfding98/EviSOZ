#!/usr/bin/env python3
"""Materialize the privacy-safe public v29↔TUSZ patient crosswalk."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.public_v29_tusz_crosswalk import (  # noqa: E402
    build_public_v29_tusz_crosswalk,
    build_raw_source_refs,
)


DEFAULT_V29_ROSTER = ROOT / "outputs/evisoz_v29_public_held_fold_cache_v2_20260831/sidecars/patient_identity_roster.json"
DEFAULT_IDENTITY_CROSSWALK = ROOT / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/record_crosswalk.csv"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_public_v29_tusz_crosswalk_v1_20260831"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--v29-roster", type=Path, default=DEFAULT_V29_ROSTER)
    parser.add_argument("--identity-crosswalk", type=Path, default=DEFAULT_IDENTITY_CROSSWALK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.v29_roster.is_symlink() or not args.v29_roster.is_file():
        raise ValueError("v29 roster must be a regular file")
    if args.identity_crosswalk.is_symlink() or not args.identity_crosswalk.is_file():
        raise ValueError("identity crosswalk must be a regular file")
    roster = json.loads(args.v29_roster.read_text(encoding="utf-8"))
    with args.identity_crosswalk.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("identity crosswalk is empty")
    roster_ref, crosswalk_ref = build_raw_source_refs(
        v29_roster_payload=roster,
        identity_crosswalk_bytes=args.identity_crosswalk.read_bytes(),
    )
    crosswalk = build_public_v29_tusz_crosswalk(
        v29_roster=roster,
        identity_rows=rows,
        v29_roster_ref=roster_ref,
        identity_crosswalk_ref=crosswalk_ref,
    )
    args.output.mkdir(parents=True)
    (args.output / "crosswalk.json").write_text(
        json.dumps(crosswalk, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": crosswalk["status"],
        "crosswalk_id": crosswalk["crosswalk_id"],
        "counts": crosswalk["counts"],
        "training_authorized": crosswalk["permissions"]["training_authorized"],
        "receipt_sha256": crosswalk["receipt_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
