#!/usr/bin/env python3
"""Seal target-free S1-C or A5 patient predictions without reading references."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.soz.label_fresh_confirmation import (
    canonical_sha256,
    file_sha256,
    load_json_object,
    load_jsonl_objects,
    load_lineage_firewall,
    seal_target_free_predictions,
    utc_now_iso,
    write_new_canonical_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("S1_C", "A5"), required=True)
    parser.add_argument(
        "--evidence-class",
        choices=("synthetic_rehearsal", "real_label_fresh_confirmation"),
        required=True,
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--policy-contract", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--preprocessing-artifact", type=Path, required=True)
    parser.add_argument("--comparator-artifact", type=Path)
    parser.add_argument("--s1c-receipt", type=Path)
    parser.add_argument(
        "--firewall-config",
        type=Path,
        default=Path("configs/trustworthy_soz_label_fresh_lineage_firewall_v1.json"),
    )
    parser.add_argument("--sealed-at", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else workspace / path

    policy = load_json_object(absolute(args.policy_contract))
    s1c = None if args.s1c_receipt is None else load_json_object(absolute(args.s1c_receipt))
    seal = seal_target_free_predictions(
        rows=load_jsonl_objects(absolute(args.predictions)),
        cohort_role=args.role,
        evidence_class=args.evidence_class,
        firewall=load_lineage_firewall(
            workspace=workspace, config_path=absolute(args.firewall_config)
        ),
        policy_contract=policy,
        policy_contract_sha256=canonical_sha256(policy),
        model_artifact_sha256=file_sha256(absolute(args.model_artifact)),
        preprocessing_artifact_sha256=file_sha256(absolute(args.preprocessing_artifact)),
        comparator_artifact_sha256=(
            None
            if args.comparator_artifact is None
            else file_sha256(absolute(args.comparator_artifact))
        ),
        sealed_at=args.sealed_at or utc_now_iso(),
        s1c_receipt=s1c,
    )
    output = absolute(args.output)
    write_new_canonical_json(output, seal)
    print(f"status={seal['status']}")
    print(f"role={args.role}")
    print(f"patients={seal['sealed_payload']['patient_count']}")
    print(f"seal_payload_sha256={seal['seal_payload_sha256']}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
