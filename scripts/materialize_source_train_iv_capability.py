#!/usr/bin/env python3
"""Verify frozen v1.1 and publish a physically source-train-only I/V bundle."""

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

from src.soz import development_reasoner_v1_1 as v11  # noqa: E402
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)
from src.soz.ictal_recovery_evidence_v1_2 import (  # noqa: E402
    load_target_free_ictal_oof_protocol,
)
from src.soz.source_train_iv_capability import (  # noqa: E402
    load_source_train_iv_capability,
    publish_source_train_iv_capability_from_v1_1,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    result = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(result):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--v1-1-capability", type=Path, required=True)
    parser.add_argument(
        "--expected-v1-1-manifest-sha256", type=_sha256, required=True
    )
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-signal-preflight-artifact-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument(
        "--expected-signal-preflight-receipt-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--oof-protocol", type=Path, required=True)
    parser.add_argument(
        "--expected-oof-protocol-artifact-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument(
        "--expected-oof-protocol-receipt-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def _guard_topology(args: argparse.Namespace) -> None:
    sources = tuple(
        Path(os.path.abspath(value)).resolve(strict=True)
        for value in (
            args.v1_1_capability,
            args.signal_preflight_bundle,
            args.oof_protocol,
        )
    )
    output = Path(os.path.abspath(args.output_directory)).resolve(strict=False)
    if os.path.lexists(output):
        raise FileExistsError(output)
    for source in sources:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output topology overlaps an immutable input")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _guard_topology(args)
    pinned = {
        "parent manifest": (
            args.expected_v1_1_manifest_sha256,
            v11.FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
        ),
        "signal artifact": (
            args.expected_signal_preflight_artifact_sha256,
            v11.FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
        ),
        "signal receipt": (
            args.expected_signal_preflight_receipt_sha256,
            v11.FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
        ),
        "OOF artifact": (
            args.expected_oof_protocol_artifact_sha256,
            v11.FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
        ),
        "OOF receipt": (
            args.expected_oof_protocol_receipt_sha256,
            v11.FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
        ),
    }
    changed = tuple(
        name for name, (actual, expected) in pinned.items() if actual != expected
    )
    if changed:
        raise ValueError(f"CLI trust anchors differ from frozen v1.1: {changed}")
    signal = load_bound_deepsoz_signal_preflight_artifact(
        args.signal_preflight_bundle,
        expected_artifact_sha256=args.expected_signal_preflight_artifact_sha256,
        expected_receipt_sha256=args.expected_signal_preflight_receipt_sha256,
    )
    protocol = load_target_free_ictal_oof_protocol(
        args.oof_protocol,
        expected_artifact_sha256=args.expected_oof_protocol_artifact_sha256,
        expected_protocol_receipt_sha256=(
            args.expected_oof_protocol_receipt_sha256
        ),
    )
    parent = v11.load_development_iv_evidence_capability_v1_1(
        args.v1_1_capability,
        signal,
        protocol,
        expected_manifest_sha256=args.expected_v1_1_manifest_sha256,
    )
    published = publish_source_train_iv_capability_from_v1_1(
        parent, args.output_directory
    )
    strict_reload = load_source_train_iv_capability(
        published.path,
        expected_manifest_sha256=published.manifest_sha256,
    )
    print(
        json.dumps(
            {
                "status": "published_source_train_only_target_free_iv",
                "path": str(strict_reload.path),
                "manifest_sha256": strict_reload.manifest_sha256,
                "receipt_sha256": strict_reload.receipt.receipt_sha256,
                "patient_count": len(strict_reload.patient_ids),
                "event_count": len(strict_reload.event_ids),
                "tensor_keys": [
                    "evolution",
                    "ictal",
                    "evolution_mask",
                    "ictal_mask",
                    "phase_mask",
                    "reliability",
                    "event_abstain",
                ],
                "target_values_loaded": False,
                "source_eval_used": False,
                "private_used": False,
                "formal_promotion": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
