#!/usr/bin/env python3
"""Compare frozen detector benchmark receipts on one identical inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_detection_comparison import (  # noqa: E402
    compare_continuous_detection_benchmark_receipts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an accuracy/efficiency Pareto receipt for detectors"
    )
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--onset-tolerance-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"benchmark receipt {path} must be an object")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    args = _parser().parse_args()
    result = compare_continuous_detection_benchmark_receipts(
        [_read(path) for path in args.receipts],
        onset_tolerance_seconds=args.onset_tolerance_seconds,
    )
    _write(args.output, result)
    print(result["comparison_receipt_id"])


if __name__ == "__main__":
    main()
