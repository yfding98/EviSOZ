#!/usr/bin/env python3
"""Validate the frozen common-17 continuous-detector benchmark v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.clinical_eeg_long_recording.common17_continuous_detector_benchmark_v3 import (
    DEFAULT_CONFIG_PATH,
    load_common17_continuous_detector_benchmark_v3,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--show-expected-receipt",
        action="store_true",
        help="validate semantics and bindings but do not require the embedded self receipt",
    )
    args = parser.parse_args()
    _, readiness = load_common17_continuous_detector_benchmark_v3(
        args.config,
        verify_receipt=not args.show_expected_receipt,
    )
    print(json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

