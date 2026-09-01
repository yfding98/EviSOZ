#!/usr/bin/env python3
"""Materialize and strictly replay the formal TUEV morphology token corpus.

The command reconstructs the public-cohort authorization from the verified
DeepSOZ target, public overlap ledger, OOF protocol, live TUEV source roster,
and signal preflight.  It then loads the already published holding manifest
under that opaque authorization and runs the first-party
EDF -> selected preprocessing arm -> audited LaBraM producer.

No token tensor, target tensor, interval roster, fit/held roster, model
hyperparameter, or alternate preprocessing arm is accepted from the caller.
"""

from __future__ import annotations

import argparse
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
from src.soz.data.tuev_morphology import (  # noqa: E402
    HOLDING_COUNT_SEMANTICS,
    TUEVMorphologySourceRecord,
    authorize_tuev_morphology_cohort,
    discover_tuev_morphology_sources,
    load_authorized_tuev_morphology_manifest,
    load_tuev_morphology_manifest,
    load_tuev_morphology_preflight,
    load_tuev_morphology_public_protocol,
    replay_tuev_morphology_source_bindings,
)
from src.soz.data.tuev_morphology_signal_preflight import (  # noqa: E402
    require_first_party_tuev_morphology_bindings,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)
from src.soz.tuev_morphology_producer import (  # noqa: E402
    load_tuev_morphology_master_corpus,
    materialize_tuev_morphology_master_corpus,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FROZEN_MICROBATCH_SIZE = 16


def _sha256_arg(value: str) -> str:
    normalized = str(value).strip()
    if not _SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the authorization-bound TUEV morphology master corpus",
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
        "--expected-oof-protocol-sha256", type=_sha256_arg, required=True
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

    holding = parser.add_argument_group("authorization-bound holding manifest")
    holding.add_argument("--holding-manifest-bundle", type=Path, required=True)
    holding.add_argument(
        "--expected-holding-bundle-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    holding.add_argument(
        "--expected-holding-source-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )

    selection = parser.add_argument_group("formal preprocessing selection")
    selection.add_argument(
        "--preprocessing-selection-bundle", type=Path, required=True
    )
    selection.add_argument(
        "--expected-preprocessing-selection-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    selection.add_argument(
        "--expected-preprocessing-protocol-receipt-sha256",
        type=_sha256_arg,
        required=True,
    )

    foundation = parser.add_argument_group("audited LaBraM implementation")
    foundation.add_argument("--labram-modeling-path", type=Path, required=True)
    foundation.add_argument("--labram-checkpoint-path", type=Path, required=True)
    foundation.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def _sources_from_holding_manifest(holding_manifest, edf_root: Path):
    """Recreate cohort identities from the already published manifest."""

    root = Path(edf_root).absolute()
    return tuple(
        TUEVMorphologySourceRecord(
            edf_root=root,
            edf_path=root / record.relative_edf_path,
            rec_path=root / record.relative_rec_path,
            relative_edf_path=record.relative_edf_path,
            relative_rec_path=record.relative_rec_path,
            official_split=record.official_split,
            group_id=record.parent_group_id,
            group_kind=record.group_kind,
            source_subject_id=record.source_subject_id,
            record_id=record.record_id,
            edf_sha256=record.edf_sha256,
            rec_sha256=record.rec_sha256,
            derivative_files=record.derivative_files,
            parent_group_files=record.parent_group_files,
            group_file_roster_sha256=record.group_file_roster_sha256,
        )
        for record in holding_manifest.records
    )


def load_authorized_morphology_inputs(
    args: argparse.Namespace,
    *,
    replay_live_source: bool = True,
):
    """Strictly rebuild the shared morphology cohort and preprocessing inputs.

    The helper is intentionally argument-namespace based so the formal corpus
    and formal head-training CLIs execute one identical authorization path.
    It returns opaque verified objects rather than caller-controlled rosters.
    """

    if not isinstance(replay_live_source, bool):
        raise TypeError("replay_live_source must be bool")
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
    if replay_live_source:
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
        expected_external_metadata_sha256=args.expected_external_metadata_sha256,
        expected_producer_source_sha256=args.expected_producer_source_sha256,
        expected_preprocessing_policy_sha256=(
            args.expected_preprocessing_policy_sha256
        ),
        expected_standard19_mapping_policy_sha256=(
            args.expected_standard19_mapping_policy_sha256
        ),
        replay_live_source=replay_live_source,
    )
    if replay_live_source:
        sources = discover_tuev_morphology_sources(args.edf_root)
    else:
        holding_for_roster = load_tuev_morphology_manifest(
            args.holding_manifest_bundle,
            expected_bundle_manifest_sha256=(
                args.expected_holding_bundle_manifest_sha256
            ),
            expected_source_manifest_sha256=(
                args.expected_holding_source_manifest_sha256
            ),
        )
        sources = _sources_from_holding_manifest(
            holding_for_roster,
            args.edf_root,
        )
    authorization = authorize_tuev_morphology_cohort(
        sources,
        public_protocol,
        replay_live_source=replay_live_source,
    )
    holding_manifest = load_authorized_tuev_morphology_manifest(
        args.holding_manifest_bundle,
        authorization,
        expected_bundle_manifest_sha256=(
            args.expected_holding_bundle_manifest_sha256
        ),
        expected_source_manifest_sha256=(
            args.expected_holding_source_manifest_sha256
        ),
        expected_count_semantics=HOLDING_COUNT_SEMANTICS,
    )
    if replay_live_source:
        replay_tuev_morphology_source_bindings(holding_manifest, args.edf_root)
    preprocessing_selection = load_preprocessing_selection_capability(
        args.preprocessing_selection_bundle,
        expected_artifact_sha256=(
            args.expected_preprocessing_selection_artifact_sha256
        ),
        expected_protocol_receipt_sha256=(
            args.expected_preprocessing_protocol_receipt_sha256
        ),
    )
    return (
        verified_target,
        public_protocol,
        preflight,
        authorization,
        holding_manifest,
        preprocessing_selection,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    (
        _verified_target,
        _public_protocol,
        preflight,
        authorization,
        holding_manifest,
        preprocessing_selection,
    ) = load_authorized_morphology_inputs(args)
    artifact = materialize_tuev_morphology_master_corpus(
        args.output_directory,
        edf_root=args.edf_root,
        holding_manifest=holding_manifest,
        preflight=preflight,
        cohort_authorization=authorization,
        preprocessing_selection=preprocessing_selection,
        labram_modeling_path=args.labram_modeling_path,
        labram_checkpoint_path=args.labram_checkpoint_path,
        device=args.device,
        microbatch_size=_FROZEN_MICROBATCH_SIZE,
    )
    verified = load_tuev_morphology_master_corpus(
        artifact.path,
        edf_root=args.edf_root,
        holding_manifest=holding_manifest,
        preflight=preflight,
        cohort_authorization=authorization,
        preprocessing_selection=preprocessing_selection,
        labram_modeling_path=args.labram_modeling_path,
        labram_checkpoint_path=args.labram_checkpoint_path,
        expected_bundle_manifest_sha256=artifact.bundle_manifest_sha256,
        expected_producer_receipt_sha256=artifact.producer_receipt_sha256,
        expected_token_index_sha256=artifact.token_index_sha256,
    )
    verified.assert_unchanged()
    print(
        json.dumps(
            {
                "bundle_manifest_sha256": artifact.bundle_manifest_sha256,
                "cohort_authorization_sha256": authorization.receipt_sha256,
                "crop_count": verified.crop_count,
                "holding_manifest_sha256": holding_manifest.manifest_sha256,
                "microbatch_size": _FROZEN_MICROBATCH_SIZE,
                "path": str(artifact.path),
                "producer_receipt_sha256": artifact.producer_receipt_sha256,
                "selected_arm_id": verified.selected_arm_id,
                "token_index_sha256": verified.index_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
