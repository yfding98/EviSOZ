#!/usr/bin/env python3
"""Statically inspect one detector checkpoint without unpickling it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.detector_provider_contract import (  # noqa: E402
    audit_checkpoint_container,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-executing detector checkpoint ZIP/pickle audit"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--artifact-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = audit_checkpoint_container(
        args.checkpoint,
        expected_sha256=args.expected_sha256,
        artifact_id=args.artifact_id,
    )
    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
