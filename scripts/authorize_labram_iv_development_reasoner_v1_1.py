#!/usr/bin/env python3
"""Issue the target-free 69-to-65 amendment and v1.1 I+V capability."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.development_reasoner import (  # noqa: E402
    load_development_iv_evidence_capability,
)
from src.soz.development_reasoner_v1_1 import (  # noqa: E402
    FROZEN_BASE_V1_AUTHORIZATION_RECEIPT_SHA256,
    FROZEN_BASE_V1_MANIFEST_SHA256,
    FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
    FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
    FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
    FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
    FROZEN_TARGET_V2_ARTIFACT_SHA256,
    FROZEN_TARGET_V2_POLICY_SHA256,
    FROZEN_TARGET_V2_RECEIPT_SHA256,
    build_signal_evidence_eligibility_amendment,
    issue_development_iv_evidence_capability_v1_1,
    load_development_iv_evidence_capability_v1_1,
    load_signal_evidence_eligibility_amendment,
    publish_development_iv_evidence_capability_v1_1,
    publish_signal_evidence_eligibility_amendment,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)
from src.soz.ictal_recovery_evidence_v1_2 import (  # noqa: E402
    load_target_free_ictal_oof_protocol,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-v1-capability", type=Path, required=True)
    parser.add_argument("--expected-base-v1-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--oof-protocol", type=Path, required=True)
    parser.add_argument("--expected-oof-protocol-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-oof-protocol-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument("--expected-signal-preflight-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-signal-preflight-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-target-v2-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-target-v2-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-target-v2-policy-sha256", type=_sha256, required=True)
    parser.add_argument("--amendment-output-directory", type=Path, required=True)
    parser.add_argument("--capability-output-directory", type=Path, required=True)
    return parser


def _guard_path_topology(args: argparse.Namespace) -> None:
    inputs = {
        "base-v1": args.base_v1_capability,
        "OOF": args.oof_protocol,
        "signal": args.signal_preflight_bundle,
    }
    normalized_inputs = {}
    for name, raw in inputs.items():
        path = Path(os.path.abspath(raw)).resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"{name} input must be a directory")
        normalized_inputs[name] = path
    outputs = {
        "amendment-output": args.amendment_output_directory,
        "capability-output": args.capability_output_directory,
    }
    normalized_outputs = {}
    for name, raw in outputs.items():
        lexical = Path(os.path.abspath(raw))
        if lexical.exists() or lexical.is_symlink():
            raise FileExistsError(f"{name} already exists")
        normalized_outputs[name] = lexical.resolve(strict=False)

    def overlaps(left: Path, right: Path) -> bool:
        return left == right or left in right.parents or right in left.parents

    named = {**normalized_inputs, **normalized_outputs}
    keys = tuple(named)
    collisions = []
    for index, left_name in enumerate(keys):
        for right_name in keys[index + 1 :]:
            if overlaps(named[left_name], named[right_name]):
                collisions.append((left_name, right_name))
    if collisions:
        raise ValueError(f"Authorization path topology overlaps: {collisions}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _guard_path_topology(args)
    pinned_cli = {
        "base-v1 manifest": (
            args.expected_base_v1_manifest_sha256,
            FROZEN_BASE_V1_MANIFEST_SHA256,
        ),
        "OOF artifact": (
            args.expected_oof_protocol_artifact_sha256,
            FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
        ),
        "OOF receipt": (
            args.expected_oof_protocol_receipt_sha256,
            FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
        ),
        "signal artifact": (
            args.expected_signal_preflight_artifact_sha256,
            FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
        ),
        "signal receipt": (
            args.expected_signal_preflight_receipt_sha256,
            FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
        ),
        "target artifact": (
            args.expected_target_v2_artifact_sha256,
            FROZEN_TARGET_V2_ARTIFACT_SHA256,
        ),
        "target receipt": (
            args.expected_target_v2_receipt_sha256,
            FROZEN_TARGET_V2_RECEIPT_SHA256,
        ),
        "target policy": (
            args.expected_target_v2_policy_sha256,
            FROZEN_TARGET_V2_POLICY_SHA256,
        ),
    }
    changed = tuple(
        name for name, (actual, expected) in pinned_cli.items() if actual != expected
    )
    if changed:
        raise ValueError(f"CLI trust anchors differ from frozen v1.1: {changed}")
    protocol = load_target_free_ictal_oof_protocol(
        args.oof_protocol,
        expected_artifact_sha256=args.expected_oof_protocol_artifact_sha256,
        expected_protocol_receipt_sha256=args.expected_oof_protocol_receipt_sha256,
    )
    signal = load_bound_deepsoz_signal_preflight_artifact(
        args.signal_preflight_bundle,
        expected_artifact_sha256=args.expected_signal_preflight_artifact_sha256,
        expected_receipt_sha256=args.expected_signal_preflight_receipt_sha256,
    )
    base = load_development_iv_evidence_capability(
        args.base_v1_capability,
        expected_manifest_sha256=args.expected_base_v1_manifest_sha256,
    )
    amendment_receipt = build_signal_evidence_eligibility_amendment(
        signal,
        protocol,
        expected_target_v2_artifact_sha256=args.expected_target_v2_artifact_sha256,
        expected_target_v2_receipt_sha256=args.expected_target_v2_receipt_sha256,
        expected_target_v2_policy_sha256=args.expected_target_v2_policy_sha256,
    )
    if base.authorization_receipt_sha256 != (
        FROZEN_BASE_V1_AUTHORIZATION_RECEIPT_SHA256
    ):
        raise ValueError("Base-v1 authorization receipt differs from frozen v1.1")
    base_receipt = base.capability.receipt
    prepublish_checks = {
        "train evidence roster": base.capability.source_train.patient_ids
        == amendment_receipt.signal_evidence_source_train_patient_ids,
        "dev evidence roster": base.capability.source_dev.patient_ids
        == amendment_receipt.signal_evidence_source_dev_patient_ids,
        "target artifact": base_receipt.verified_target_v2_artifact_sha256
        == amendment_receipt.verified_target_v2_artifact_sha256,
        "target receipt": base_receipt.verified_target_v2_receipt_sha256
        == amendment_receipt.verified_target_v2_receipt_sha256,
        "target policy": base_receipt.verified_target_v2_policy_sha256
        == amendment_receipt.verified_target_v2_policy_sha256,
    }
    failed = tuple(name for name, passed in prepublish_checks.items() if not passed)
    if failed:
        raise ValueError(f"Pre-publication v1/amendment check failed: {failed}")
    amendment_published = publish_signal_evidence_eligibility_amendment(
        amendment_receipt, args.amendment_output_directory
    )
    amendment = load_signal_evidence_eligibility_amendment(
        amendment_published.path,
        signal,
        protocol,
        expected_artifact_sha256=amendment_published.artifact_sha256,
        expected_receipt_sha256=amendment_published.receipt_sha256,
    )
    capability = issue_development_iv_evidence_capability_v1_1(base, amendment)
    published = publish_development_iv_evidence_capability_v1_1(
        capability, args.capability_output_directory
    )
    strict_reload = load_development_iv_evidence_capability_v1_1(
        published.path,
        signal,
        protocol,
        expected_manifest_sha256=published.manifest_sha256,
    )
    receipt = strict_reload.capability.receipt
    print(
        json.dumps(
            {
                "status": "authorized_development_candidate_only",
                "amendment_path": str(amendment.path),
                "amendment_artifact_sha256": amendment.artifact_sha256,
                "amendment_receipt_sha256": amendment.receipt_sha256,
                "capability_path": str(strict_reload.path),
                "capability_manifest_sha256": strict_reload.manifest_sha256,
                "authorization_receipt_sha256": receipt.receipt_sha256,
                "target_header_source_train_patients": 69,
                "signal_evidence_source_train_patients": len(
                    receipt.source_train_patient_ids
                ),
                "source_dev_patients": len(receipt.source_dev_patient_ids),
                "excluded_source_train_patient_ids": [
                    row.patient_id
                    for row in amendment.receipt.excluded_source_train_patients
                ],
                "target_values_loaded": False,
                "candidate_reasoner_input_authorized": True,
                "formal_reasoner_authorized": False,
                "formal_promotion": False,
                "source_eval_used": False,
                "private_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
