#!/usr/bin/env python3
"""Validate and persist a DeepSOZ posterior batch without references.

The command reads expected manifest/recording/patient identities exclusively
from a validated identity/split roster receipt.  It performs no calibration,
source-evaluation scoring or reference join and writes only to a new output
directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.deepsoz_reference_free_batch_validation_artifact import (  # noqa: E402
    validate_and_write_deepsoz_batch_reference_free,
)


_FORBIDDEN_OPTION_FRAGMENTS = (
    "reference",
    "annotation",
    "excel",
    "clinical",
    "source-eval",
    "source_eval",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete DeepSOZ posterior batch against an identity-only "
            "split roster and write append-only reference-free receipts"
        )
    )
    parser.add_argument("--posterior-batch-root", type=Path, required=True)
    parser.add_argument("--split-roster-receipt", type=Path, required=True)
    parser.add_argument(
        "--selected-split",
        choices=("source_train", "source_dev"),
        required=True,
    )
    parser.add_argument("--provider-registry", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def _reject_forbidden_options(argv: Sequence[str]) -> None:
    for raw in argv:
        if not raw.startswith("-"):
            continue
        option = raw.split("=", 1)[0].lower()
        if any(fragment in option for fragment in _FORBIDDEN_OPTION_FRAGMENTS):
            raise ValueError(
                "reference, annotation, Excel, clinical and source-eval arguments "
                "are forbidden for this reference-free validation CLI"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    try:
        _reject_forbidden_options(raw_args)
        args = parser.parse_args(raw_args)
        (
            validation,
            write_receipt,
            write_summary,
        ) = validate_and_write_deepsoz_batch_reference_free(
            posterior_batch_root=args.posterior_batch_root,
            split_roster_receipt=args.split_roster_receipt,
            selected_split=args.selected_split,
            provider_registry_path=args.provider_registry,
            output_directory=args.output_directory,
        )
        print(
            json.dumps(
                {
                    "validation_receipt": validation,
                    "write_receipt": write_receipt,
                    "write_summary": write_summary,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2  # pragma: no cover - argparse.error exits


if __name__ == "__main__":
    raise SystemExit(main())
