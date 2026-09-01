#!/usr/bin/env python3
"""Train the frozen 20-epoch LaBraM I+V development candidate reasoner.

The CLI accepts only a previously published target-free candidate capability
and a strictly verified target-v2 artifact.  Epochs, optimizer, learning rate,
weight decay, clipping, seed, loss weights, checkpoint selection, calibration,
and thresholding are not command-line surfaces.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.development_reasoner import (  # noqa: E402
    join_development_iv_targets,
    load_development_iv_evidence_capability,
)
from src.soz.development_reasoner_training import (  # noqa: E402
    publish_development_reasoner_training_run,
    train_development_iv_reasoner,
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
    parser.add_argument("--target-v2-artifact-directory", type=Path, required=True)
    parser.add_argument("--deepsoz-source-csv", type=Path, required=True)
    parser.add_argument("--split-manifest-csv", type=Path, required=True)
    parser.add_argument("--expected-target-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-target-summary-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-target-readme-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-source-input-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-split-input-sha256", type=_sha256, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight_only and args.output_directory is None:
        raise ValueError("Training requires --output-directory")
    published_capability = load_development_iv_evidence_capability(
        args.capability_bundle,
        expected_manifest_sha256=args.expected_capability_manifest_sha256,
    )
    target = load_verified_deepsoz_target_v2_artifact(
        args.target_v2_artifact_directory,
        args.deepsoz_source_csv,
        args.split_manifest_csv,
        expected_target_artifact_sha256=args.expected_target_artifact_sha256,
        expected_summary_artifact_sha256=args.expected_target_summary_sha256,
        expected_readme_artifact_sha256=args.expected_target_readme_sha256,
        expected_source_input_sha256=args.expected_source_input_sha256,
        expected_split_input_sha256=args.expected_split_input_sha256,
    )
    data = join_development_iv_targets(published_capability.capability, target)
    preflight = {
        "status": "ready",
        "capability_manifest_sha256": published_capability.manifest_sha256,
        "evidence_authorization_sha256": data.evidence_authorization_sha256,
        "verified_target_v2_receipt_sha256": data.verified_target_v2_receipt_sha256,
        "source_train_patients": len(data.source_train.patient_ids),
        "source_train_events": data.source_train.full_batch().evidence.batch_size,
        "source_dev_patients": len(data.source_dev.patient_ids),
        "source_dev_events": data.source_dev.full_batch().evidence.batch_size,
        "epochs": 20,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "seed": 20260808,
        "ranking_weight": 0.25,
        "checkpoint_selection": "final_epoch_20_only",
        "source_dev_policy": "one_post_freeze_diagnostic_forward_only",
        "threshold_selected": False,
        "formal_promotion": False,
        "source_eval_used": False,
        "private_used": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0
    run = train_development_iv_reasoner(data, device=args.device)
    artifact = publish_development_reasoner_training_run(
        run, args.output_directory
    )
    receipt = run.receipt
    print(
        json.dumps(
            {
                **preflight,
                "status": "completed",
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "training_receipt_sha256": artifact.training_receipt_sha256,
                "final_state_sha256": receipt.final_state_sha256,
                "source_train_diagnostic": asdict(
                    receipt.final_source_train_diagnostic
                ),
                "source_dev_diagnostic": asdict(
                    receipt.final_source_dev_diagnostic
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
