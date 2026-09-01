#!/usr/bin/env python3
"""Run one S1-C policy calibration from sealed scores and an opened reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.soz.label_fresh_confirmation import (
    calibrate_s1c_from_sealed_predictions,
    canonical_sha256,
    load_json_object,
    utc_now_iso,
    write_new_canonical_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-seal", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--policy-contract", type=Path, required=True)
    parser.add_argument("--calibrated-at", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else workspace / path

    policy = load_json_object(absolute(args.policy_contract))
    result = calibrate_s1c_from_sealed_predictions(
        prediction_seal=load_json_object(absolute(args.prediction_seal)),
        reference_bundle=load_json_object(absolute(args.reference_bundle)),
        policy_contract=policy,
        policy_contract_sha256=canonical_sha256(policy),
        calibrated_at=args.calibrated_at or utc_now_iso(),
    )
    output = absolute(args.output)
    write_new_canonical_json(output, result)
    print(f"status={result['status']}")
    print(f"evidence_class={result['evidence_class']}")
    print(f"output={output}")
    return 0 if result["status"] == "QUALIFIED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
