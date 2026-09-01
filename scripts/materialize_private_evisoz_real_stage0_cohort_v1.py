#!/usr/bin/env python3
"""Materialize the patient-split-aware real private EviSOZ Stage-0 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.private_stage0_cohort_materializer import (  # noqa: E402
    materialize_private_stage0_cohort,
)


DEFAULT_BUNDLE = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814"
DEFAULT_SPLIT = ROOT / "outputs/evisoz_stage0_private_split_v1_20260831"
DEFAULT_AUTHORITY = ROOT / "outputs/evisoz_stage0_private_opaque_reference_authority_v1_20260831/authority.json"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    return parser


def _progress(value: Mapping[str, object]) -> None:
    print(json.dumps(dict(value), sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_manifest = json.loads(
        (args.bundle / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    if bundle_manifest.get("schema_version") != "soz_private_zero_adaptation_bundle_v18":
        raise ValueError("private bundle schema mismatch")
    result = materialize_private_stage0_cohort(
        signal_roster_path=args.bundle / "signal_roster.csv",
        eeg_root=Path(str(bundle_manifest["eeg_root"])),
        split_roster_path=args.split / "split_roster.json",
        split_manifest_path=args.split / "manifest.json",
        reference_authority_path=args.authority,
        output=args.output,
        limit=args.limit,
        raise_on_event_error=args.strict,
        progress=_progress,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "candidate_event_count",
                    "materialized_event_count",
                    "preexcluded_event_count",
                    "runtime_excluded_event_count",
                    "materialized_role_event_counts",
                    "exclusion_reason_counts",
                    "receipt_sha256",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
