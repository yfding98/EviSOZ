#!/usr/bin/env python3
"""Assemble fail-closed grounded reports from sealed public artifacts only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.public_report_assembler import (  # noqa: E402
    assemble_public_grounded_reports,
    write_public_report_batch,
)


DEFAULT_PHENOTYPE = ROOT / "outputs/event_phenotype_source_only_n64_20260811.json"
DEFAULT_RANKING = (
    ROOT / "outputs/labram_temporal_mil_exact_full_source_train_refit_v1_20260811"
)
DEFAULT_RANKING_ROSTER = (
    ROOT / "outputs/labram_iv_source_train_only_capability_v1_20260811"
)
DEFAULT_PUBLIC_UNION = (
    ROOT / "outputs/public_development_union_v11_20260811/manifest.json"
)
DEFAULT_REFERENCE_AUDIT = (
    ROOT / "outputs/labram_reference_robustness_public_development_all988_20260811.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=__doc__,
    )
    parser.add_argument("--phenotype-artifact", type=Path, default=DEFAULT_PHENOTYPE)
    parser.add_argument("--ranking-directory", type=Path, default=DEFAULT_RANKING)
    parser.add_argument(
        "--ranking-roster-directory", type=Path, default=DEFAULT_RANKING_ROSTER
    )
    parser.add_argument(
        "--public-union-manifest", type=Path, default=DEFAULT_PUBLIC_UNION
    )
    parser.add_argument(
        "--reference-audit-artifact", type=Path, default=DEFAULT_REFERENCE_AUDIT
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch = assemble_public_grounded_reports(
        phenotype_artifact=args.phenotype_artifact,
        ranking_directory=args.ranking_directory,
        ranking_roster_directory=args.ranking_roster_directory,
        public_union_manifest=args.public_union_manifest,
        reference_audit_artifact=args.reference_audit_artifact,
    )
    output_sha = write_public_report_batch(batch, args.output)
    print(
        json.dumps(
            {
                "schema_version": batch.schema_version,
                "status": batch.status,
                "assembled": batch.assembled_count,
                "blocked": batch.blocked_count,
                "output": os.path.abspath(args.output),
                "output_sha256": output_sha,
                "deepsoz_target_values_loaded": False,
                "private_data_loaded": False,
                "evaluation_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
