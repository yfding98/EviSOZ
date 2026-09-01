#!/usr/bin/env python3
"""Materialize real private field releases and EviSOZ training envelopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.forge.private_stage0_examples import (  # noqa: E402
    materialize_private_stage0_examples,
)


DEFAULT_REAL_COHORT = (
    ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831"
)
DEFAULT_SPLIT = ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json"
DEFAULT_SIGNAL = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/signal_roster.csv"
)
DEFAULT_TARGET = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
)
DEFAULT_SOURCE = ROOT / "outputs/soz_pre/private_edf_soz_manifest.csv"
DEFAULT_OUTPUT = (
    ROOT / "outputs/evisoz_stage0_private_real_examples_v1_20260831"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--real-cohort", type=Path, default=DEFAULT_REAL_COHORT)
    parser.add_argument("--split-roster", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--signal-roster", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--target-ledger", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--private-training-authorization",
        type=Path,
        help=(
            "optional external data-controller authorization receipt; when absent, "
            "all private fields remain evaluator-only"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_private_stage0_examples(
        real_cohort_root=args.real_cohort,
        split_roster_path=args.split_roster,
        signal_roster_path=args.signal_roster,
        target_ledger_path=args.target_ledger,
        source_manifest_path=args.source_manifest,
        output=args.output,
        limit=args.limit,
        private_training_authorization_path=args.private_training_authorization,
    )
    (args.output / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "event_count": result["counts"]["event_count"],
                "role_event_counts": result["counts"]["role_event_counts"],
                "enabled_loss_port_event_counts": result["counts"][
                    "enabled_loss_port_event_counts"
                ],
                "loss_enabled_field_event_counts": result["counts"][
                    "loss_enabled_field_event_counts"
                ],
                "physician_report_text_training_count": result["counts"][
                    "physician_report_text_training_count"
                ],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
