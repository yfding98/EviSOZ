#!/usr/bin/env python3
"""Publish the source-train-only bridge to frozen formal-v4 LaBraM H tokens.

This command has no DeepSOZ target, source-dev, source-eval, private-data, or
trainable-head input.  It replays the 582 authorized source-train EEG windows,
binds them to formal-v4 tokens by source identity, and publishes only a lazy
ordered crosswalk receipt.
"""

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

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
)
from src.soz.development_reasoner_v1_1 import (  # noqa: E402
    load_development_iv_evidence_capability_v1_1,
)
from src.soz.frozen_h_crosswalk import (  # noqa: E402
    materialize_frozen_h_source_train_crosswalk,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)
from src.soz.ictal_recovery_evidence_v1_2 import (  # noqa: E402
    load_target_free_ictal_oof_protocol,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument(
        "--expected-capability-manifest-sha256", type=_sha256, required=True
    )
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-signal-preflight-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-signal-preflight-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--oof-protocol", type=Path, required=True)
    parser.add_argument(
        "--expected-oof-protocol-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-oof-protocol-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--master-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-master-bundle-manifest-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-master-source-manifest-sha256", type=_sha256, required=True
    )
    parser.add_argument("--preprocessing-selection-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-preprocessing-selection-artifact-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument(
        "--expected-preprocessing-protocol-receipt-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--formal-token-corpus", type=Path, required=True)
    parser.add_argument(
        "--expected-formal-token-index-sha256", type=_sha256, required=True
    )
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--verify-complete-tensor",
        action="store_true",
        help="also allocate and verify the complete [E,19,15,4,200] CPU tensor",
    )
    return parser


def _guard_paths(args: argparse.Namespace) -> None:
    input_paths = {
        "capability": args.capability,
        "signal": args.signal_preflight_bundle,
        "OOF protocol": args.oof_protocol,
        "master manifest": args.master_manifest_bundle,
        "preprocessing selection": args.preprocessing_selection_bundle,
        "formal token corpus": args.formal_token_corpus,
        "TUSZ root": args.tusz_root,
    }
    normalized_inputs: dict[str, Path] = {}
    for name, raw in input_paths.items():
        path = Path(os.path.abspath(raw))
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{name} must be an existing non-symlink directory")
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise ValueError(f"{name} path may not traverse symlinks")
        normalized_inputs[name] = resolved
    output = Path(os.path.abspath(args.output_directory)).resolve(strict=False)
    if os.path.lexists(output):
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)

    def overlaps(left: Path, right: Path) -> bool:
        return left == right or left in right.parents or right in left.parents

    collisions = tuple(
        name
        for name, source in normalized_inputs.items()
        if overlaps(output, source)
    )
    if collisions:
        raise ValueError(f"output topology overlaps immutable inputs: {collisions}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _guard_paths(args)
    signal = load_bound_deepsoz_signal_preflight_artifact(
        args.signal_preflight_bundle,
        expected_artifact_sha256=args.expected_signal_preflight_artifact_sha256,
        expected_receipt_sha256=args.expected_signal_preflight_receipt_sha256,
    )
    protocol = load_target_free_ictal_oof_protocol(
        args.oof_protocol,
        expected_artifact_sha256=args.expected_oof_protocol_artifact_sha256,
        expected_protocol_receipt_sha256=args.expected_oof_protocol_receipt_sha256,
    )
    capability = load_development_iv_evidence_capability_v1_1(
        args.capability,
        signal,
        protocol,
        expected_manifest_sha256=args.expected_capability_manifest_sha256,
    )
    master = load_tusz_ictal_training_manifest(
        args.master_manifest_bundle,
        expected_bundle_manifest_sha256=args.expected_master_bundle_manifest_sha256,
        expected_source_manifest_sha256=args.expected_master_source_manifest_sha256,
    )
    preprocessing = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        expected_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
    )
    token_corpus = load_formal_token_corpus(
        args.formal_token_corpus,
        expected_index_sha256=args.expected_formal_token_index_sha256,
        preprocessing_selection=preprocessing,
    )
    artifact = materialize_frozen_h_source_train_crosswalk(
        capability=capability,
        signal=signal,
        protocol=protocol,
        master_manifest=master,
        token_corpus=token_corpus,
        tusz_root=args.tusz_root,
        output_directory=args.output_directory,
    )
    artifact.assert_unchanged()
    first_shape = list(artifact.load_event_tokens(0).shape)
    last_shape = list(artifact.load_event_tokens(len(artifact.events) - 1).shape)
    complete_shape = None
    if args.verify_complete_tensor:
        complete = artifact.materialize_tokens()
        complete_shape = list(complete.shape)
        del complete
    print(
        json.dumps(
            {
                "status": "STRICT_SOURCE_TRAIN_FROZEN_H_CROSSWALK_PASS",
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "receipt_sha256": artifact.receipt_sha256,
                "event_count": len(artifact.events),
                "patient_count": len(artifact.patient_ids),
                "event_order_sha256": artifact.receipt["event_order_sha256"],
                "token_binding_roster_sha256": artifact.receipt[
                    "token_binding_roster_sha256"
                ],
                "first_event_shape": first_shape,
                "last_event_shape": last_shape,
                "complete_tensor_shape": complete_shape,
                "raw_replay_verified": True,
                "deepsoz_target_values_loaded": False,
                "source_dev_used": False,
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
