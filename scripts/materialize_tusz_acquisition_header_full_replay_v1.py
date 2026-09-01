#!/usr/bin/env python3
"""Materialize the identity-free aggregate TUSZ acquisition-header replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clinical_eeg_long_recording.tusz_acquisition_header_full_replay_v1 import (
    audit_tusz_acquisition_headers_v1,
    validate_tusz_acquisition_header_full_replay_v1,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-audit", type=Path, required=True)
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit_tusz_acquisition_headers_v1(
        canonical_audit_path=args.canonical_audit,
        tusz_root=args.tusz_root,
    )
    validate_tusz_acquisition_header_full_replay_v1(receipt)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(receipt["receipt_sha256"])


if __name__ == "__main__":
    main()
