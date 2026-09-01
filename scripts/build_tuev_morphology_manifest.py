#!/usr/bin/env python3
"""Build and replay the authorized TUEV morphology producer manifest.

The command strict-loads the DeepSOZ target registry, public overlap ledger,
complete OOF protocol, and TUEV signal preflight.  It then replays the live
TUEV roster, derives one shared auxiliary-source authorization, builds the
holding audit in memory, and publishes its authorization-projected producer
manifest.  No ledger digest or fit/held/excluded group roster is accepted as a
self-attested substitute.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.tuev_morphology import (  # noqa: E402
    HOLDING_COUNT_SEMANTICS,
    TUEV_MORPHOLOGY_HOLDING_TARGET_UPPER_BOUND,
    authorize_tuev_morphology_cohort,
    build_tuev_morphology_manifest,
    derive_tuev_morphology_fold_manifest,
    discover_tuev_morphology_sources,
    load_authorized_tuev_morphology_manifest,
    load_tuev_morphology_preflight,
    load_tuev_morphology_public_protocol,
    replay_tuev_morphology_source_bindings,
    save_tuev_morphology_manifest,
)
from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.data.tuev_morphology_signal_preflight import (  # noqa: E402
    require_first_party_tuev_morphology_bindings,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_arg(value: str) -> str:
    normalized = str(value).strip()
    if not _SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return normalized


def _holding_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed != TUEV_MORPHOLOGY_HOLDING_TARGET_UPPER_BOUND:
        raise argparse.ArgumentTypeError(
            "the only frozen reference count is 58,722; omit this option "
            "unless the real artifact is explicitly tied to that audit"
        )
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a replay-verified authorized TUEV morphology manifest",
        allow_abbrev=False,
    )
    target = parser.add_argument_group("verified DeepSOZ target registry")
    target.add_argument("--target-v2-directory", type=Path, required=True)
    target.add_argument("--deepsoz-source-csv", type=Path, required=True)
    target.add_argument("--deepsoz-split-csv", type=Path, required=True)
    target.add_argument(
        "--expected-target-v2-target-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    target.add_argument(
        "--expected-target-v2-summary-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    target.add_argument(
        "--expected-target-v2-readme-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    target.add_argument(
        "--expected-deepsoz-source-sha256", type=_sha256_arg, required=True
    )
    target.add_argument(
        "--expected-deepsoz-split-sha256", type=_sha256_arg, required=True
    )

    protocol = parser.add_argument_group("strict public cohort protocol")
    protocol.add_argument("--public-ledger-bundle", type=Path, required=True)
    protocol.add_argument("--oof-protocol-bundle", type=Path, required=True)
    protocol.add_argument(
        "--expected-public-ledger-bundle-sha256",
        type=_sha256_arg,
        required=True,
    )
    protocol.add_argument(
        "--expected-public-ledger-build-sha256",
        type=_sha256_arg,
        required=True,
    )
    protocol.add_argument(
        "--expected-oof-protocol-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    protocol.add_argument(
        "--expected-oof-protocol-sha256",
        type=_sha256_arg,
        required=True,
    )

    preflight = parser.add_argument_group("verified TUEV signal preflight")
    preflight.add_argument("--edf-root", type=Path, required=True)
    preflight.add_argument("--preflight-bundle", type=Path, required=True)
    preflight.add_argument("--external-metadata-json", type=Path, required=True)
    preflight.add_argument(
        "--expected-preflight-bundle-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    preflight.add_argument(
        "--expected-preflight-receipt-sha256", type=_sha256_arg, required=True
    )
    preflight.add_argument(
        "--expected-external-metadata-sha256", type=_sha256_arg, required=True
    )
    preflight.add_argument(
        "--expected-producer-source-sha256", type=_sha256_arg, required=True
    )
    preflight.add_argument(
        "--expected-preprocessing-policy-sha256", type=_sha256_arg, required=True
    )
    preflight.add_argument(
        "--expected-standard19-mapping-policy-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--holding-reference-target-count", type=_holding_count
    )
    parser.add_argument(
        "--holding-output-directory",
        type=Path,
        required=True,
        help=(
            "Fresh bundle for the authorization-bound holding/master manifest; "
            "this is the only manifest allowed to own the shared token corpus"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Fresh bundle for the authorization-projected fit/held manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    verified_target = load_verified_deepsoz_target_v2_artifact(
        args.target_v2_directory,
        args.deepsoz_source_csv,
        args.deepsoz_split_csv,
        expected_target_artifact_sha256=(
            args.expected_target_v2_target_artifact_sha256
        ),
        expected_summary_artifact_sha256=(
            args.expected_target_v2_summary_artifact_sha256
        ),
        expected_readme_artifact_sha256=(
            args.expected_target_v2_readme_artifact_sha256
        ),
        expected_source_input_sha256=args.expected_deepsoz_source_sha256,
        expected_split_input_sha256=args.expected_deepsoz_split_sha256,
    )
    public_protocol = load_tuev_morphology_public_protocol(
        args.public_ledger_bundle,
        args.oof_protocol_bundle,
        verified_target.registry,
        expected_public_ledger_bundle_sha256=(
            args.expected_public_ledger_bundle_sha256
        ),
        expected_public_ledger_build_sha256=(
            args.expected_public_ledger_build_sha256
        ),
        expected_oof_protocol_artifact_sha256=(
            args.expected_oof_protocol_artifact_sha256
        ),
        expected_oof_protocol_sha256=args.expected_oof_protocol_sha256,
    )
    require_first_party_tuev_morphology_bindings(
        producer_source_sha256=args.expected_producer_source_sha256,
        preprocessing_policy_sha256=args.expected_preprocessing_policy_sha256,
        standard19_mapping_policy_sha256=(
            args.expected_standard19_mapping_policy_sha256
        ),
    )
    preflight = load_tuev_morphology_preflight(
        args.preflight_bundle,
        edf_root=args.edf_root,
        external_metadata_path=args.external_metadata_json,
        expected_bundle_manifest_sha256=(
            args.expected_preflight_bundle_manifest_sha256
        ),
        expected_preflight_receipt_sha256=(
            args.expected_preflight_receipt_sha256
        ),
        expected_external_metadata_sha256=(
            args.expected_external_metadata_sha256
        ),
        expected_producer_source_sha256=args.expected_producer_source_sha256,
        expected_preprocessing_policy_sha256=(
            args.expected_preprocessing_policy_sha256
        ),
        expected_standard19_mapping_policy_sha256=(
            args.expected_standard19_mapping_policy_sha256
        ),
    )
    sources = discover_tuev_morphology_sources(args.edf_root)
    authorization = authorize_tuev_morphology_cohort(
        sources,
        public_protocol,
    )
    holding_manifest = build_tuev_morphology_manifest(
        sources,
        preflight,
        authorization,
        preprocessing_policy_sha256=(
            args.expected_preprocessing_policy_sha256
        ),
        holding_reference_target_count=args.holding_reference_target_count,
    )
    holding_artifact = save_tuev_morphology_manifest(
        args.holding_output_directory,
        holding_manifest,
    )
    loaded_holding = load_authorized_tuev_morphology_manifest(
        holding_artifact.path,
        authorization,
        expected_bundle_manifest_sha256=(
            holding_artifact.bundle_manifest_sha256
        ),
        expected_source_manifest_sha256=(
            holding_artifact.source_manifest_sha256
        ),
        expected_count_semantics=HOLDING_COUNT_SEMANTICS,
    )
    replay_tuev_morphology_source_bindings(loaded_holding, args.edf_root)
    manifest = derive_tuev_morphology_fold_manifest(
        loaded_holding,
        authorization,
    )
    artifact = save_tuev_morphology_manifest(args.output_directory, manifest)
    loaded = load_authorized_tuev_morphology_manifest(
        artifact.path,
        authorization,
        expected_bundle_manifest_sha256=artifact.bundle_manifest_sha256,
        expected_source_manifest_sha256=artifact.source_manifest_sha256,
    )
    replay_tuev_morphology_source_bindings(loaded, args.edf_root)
    print(
        json.dumps(
            {
                "bundle_manifest_sha256": artifact.bundle_manifest_sha256,
                "cohort_authorization_sha256": (
                    authorization.receipt_sha256
                ),
                "count_semantics": loaded.count_semantics,
                "excluded_group_count": len(loaded.excluded_group_ids),
                "fit_group_count": len(loaded.fit_group_ids),
                "held_group_count": len(loaded.held_group_ids),
                "holding_bundle_manifest_sha256": (
                    holding_artifact.bundle_manifest_sha256
                ),
                "holding_manifest_path": str(holding_artifact.path),
                "holding_manifest_sha256": loaded_holding.manifest_sha256,
                "interval_group_count": len(loaded.interval_groups),
                "duplicate_ledger_sha256": loaded.duplicate_ledger_sha256,
                "exact_duplicate_class_count": len(
                    loaded.duplicate_ledger.duplicate_classes
                ),
                "omission_counts": dict(
                    sorted(Counter(item.reason_code for item in loaded.omissions).items())
                ),
                "path": str(artifact.path),
                "record_count": len(loaded.records),
                "source_manifest_sha256": artifact.source_manifest_sha256,
                "target_count": loaded.target_count,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
