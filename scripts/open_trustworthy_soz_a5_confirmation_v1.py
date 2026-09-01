#!/usr/bin/env python3
"""Open and evaluate one A5 confirmation after predictions and reports were sealed."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.soz.label_fresh_confirmation import (
    load_json_object,
    open_a5_confirmation,
    utc_now_iso,
    write_new_canonical_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-seal", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--s1c-receipt", type=Path, required=True)
    parser.add_argument("--opened-analysis-at", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else workspace / path

    result = open_a5_confirmation(
        prediction_seal=load_json_object(absolute(args.prediction_seal)),
        reference_bundle=load_json_object(absolute(args.reference_bundle)),
        s1c_receipt=load_json_object(absolute(args.s1c_receipt)),
        opened_analysis_at=args.opened_analysis_at or utc_now_iso(),
    )
    output = absolute(args.output)
    write_new_canonical_json(output, result)
    primary = result["result_payload"]["metrics"]["full_coverage"]["strict_top1"]
    print(f"status={result['status']}")
    print(f"evidence_class={result['evidence_class']}")
    print(f"strict_top1={primary['successes']}/{primary['total']}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
