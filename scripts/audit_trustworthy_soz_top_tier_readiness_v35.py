#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.soz.top_tier_readiness import audit_top_tier_readiness, write_readiness_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit frozen NeuroSOZ top-tier publication readiness")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/trustworthy_soz_top_tier_confirmation_v35.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trustworthy_soz_top_tier_readiness_v35_20260816/result.json"),
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return a non-zero exit code when confirmatory or reader evidence is still missing.",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    contract_path = args.contract if args.contract.is_absolute() else workspace / args.contract
    output_path = args.output if args.output.is_absolute() else workspace / args.output
    result = audit_top_tier_readiness(workspace=workspace, contract_path=contract_path)
    write_readiness_result(result, output_path)
    print(f"submission_status={result['submission_status']}")
    print(f"development_evidence_complete={str(result['development_evidence_complete']).lower()}")
    print("blocking_gates=" + ",".join(result["blocking_gates"]))
    print(f"result={output_path}")
    return 2 if args.require_ready and not result["top_tier_submission_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
