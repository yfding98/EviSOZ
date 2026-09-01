#!/usr/bin/env python3
"""Run one append-only source-dev diagnostic of the frozen v1.1 reasoner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.development_reasoner_training_v1_1 import (  # noqa: E402
    FROZEN_SOURCE_DEV_TARGET_SCOPE_RECEIPT_SHA256,
    diagnose_development_iv_reasoner_v1_1,
    join_development_iv_split_targets_v1_1,
    load_development_reasoner_fit_v1_1,
    publish_development_reasoner_dev_diagnostic_v1_1,
)
from src.soz.development_reasoner_v1_1 import (  # noqa: E402
    FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
    FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
    FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
    FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
    FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
    load_development_iv_evidence_capability_v1_1,
)
from src.soz.development_target_scope_v1_1 import (  # noqa: E402
    load_development_target_scope_v1_1,
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
    parser.add_argument("--capability-bundle", type=Path, required=True)
    parser.add_argument("--expected-capability-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--oof-protocol", type=Path, required=True)
    parser.add_argument("--expected-oof-protocol-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-oof-protocol-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument("--expected-signal-preflight-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-signal-preflight-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--dev-target-bundle", type=Path, required=True)
    parser.add_argument("--expected-dev-target-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--fit-artifact", type=Path, required=True)
    parser.add_argument("--expected-fit-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def _guard_paths(args: argparse.Namespace) -> None:
    inputs = tuple(
        Path(os.path.abspath(value)).resolve(strict=True)
        for value in (
            args.capability_bundle,
            args.oof_protocol,
            args.signal_preflight_bundle,
            args.dev_target_bundle,
            args.fit_artifact,
        )
    )
    output = Path(os.path.abspath(args.output_directory)).resolve(strict=False)
    if os.path.lexists(output):
        raise FileExistsError(f"diagnostic output already exists: {output}")
    if any(output == source or output in source.parents or source in output.parents for source in inputs):
        raise ValueError("diagnostic output/input path topology overlaps")


def _check_frozen_cli(args: argparse.Namespace) -> None:
    expected = {
        "capability manifest": (
            args.expected_capability_manifest_sha256,
            FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
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
        "dev target receipt": (
            args.expected_dev_target_receipt_sha256,
            FROZEN_SOURCE_DEV_TARGET_SCOPE_RECEIPT_SHA256,
        ),
    }
    changed = tuple(name for name, values in expected.items() if values[0] != values[1])
    if changed:
        raise ValueError(f"diagnostic CLI trust anchors changed: {changed}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _guard_paths(args)
    _check_frozen_cli(args)
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
    capability = load_development_iv_evidence_capability_v1_1(
        args.capability_bundle,
        signal,
        protocol,
        expected_manifest_sha256=args.expected_capability_manifest_sha256,
    )
    target = load_development_target_scope_v1_1(
        args.dev_target_bundle,
        expected_model_split="source_dev",
        expected_receipt_file_sha256=args.expected_dev_target_receipt_sha256,
    )
    data = join_development_iv_split_targets_v1_1(capability, target)
    fit = load_development_reasoner_fit_v1_1(
        args.fit_artifact,
        expected_manifest_sha256=args.expected_fit_manifest_sha256,
    )
    run = diagnose_development_iv_reasoner_v1_1(fit, data, device=args.device)
    artifact = publish_development_reasoner_dev_diagnostic_v1_1(
        run, args.output_directory
    )
    print(
        json.dumps(
            {
                "status": "completed_append_only_source_dev_diagnostic",
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "tensor_file_sha256": artifact.tensor_file_sha256,
                "diagnostic_receipt_sha256": artifact.diagnostic_receipt_sha256,
                "fit_manifest_sha256": run.receipt.fit_manifest_sha256,
                "checkpoint_state_before_sha256": run.receipt.checkpoint_state_before_sha256,
                "checkpoint_state_after_sha256": run.receipt.checkpoint_state_after_sha256,
                "optimizer_instantiated": False,
                "all_parameters_require_grad_false": True,
                "full_bag_forward_count": 1,
                "numeric_explanation_reconstructs_logits": True,
                "source_dev_summary": asdict(run.receipt.summary),
                "threshold_selected": False,
                "calibrator_fitted": False,
                "formal_promotion": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
