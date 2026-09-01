#!/usr/bin/env python3
"""Build or source-replay the additive full-stack nested exposure registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.full_stack_nested_exposure_graph_v1 import (  # noqa: E402
    DEFAULT_PLAN_PATH,
    DEFAULT_REGISTRY_PATH,
    build_full_stack_nested_exposure_registry_from_paths_v1,
    materialize_full_stack_nested_exposure_registry_v1,
    replay_full_stack_nested_exposure_registry_from_paths_v1,
)


DEFAULT_TUSZ_AUDIT_ROOT = (
    ROOT / "outputs" / "tusz_canonical_physical_signal_audit_v1_full_20260824r2"
)
DEFAULT_DEEPSOZ_IDENTITY = (
    ROOT
    / "outputs"
    / "deepsoz_tusz_source_train_identity_binding_v1_20260823"
    / "identity_binding.json"
)
DEFAULT_DEEPSOZ_UNION = (
    ROOT / "outputs" / "public_development_union_identity_v12_20260812" / "manifest.json"
)
DEFAULT_DEEPSOZ_EXTERNAL = (
    ROOT
    / "outputs"
    / "deepsoz_published_external_exposure_attestation_v1_20260824r1"
    / "exposure_attestation.json"
)
DEFAULT_TUEV_ROOT = Path("/mnt/hd1/dyf/dataset/tuh_eeg_events/v2.0.1/edf")
DEFAULT_TUEV_README = Path(
    "/mnt/hd1/dyf/dataset/tuh_eeg_events/v2.0.1/AAREADME.txt"
)
DEFAULT_TUAR_ROOT = Path("/mnt/hd1/dyf/dataset/TUAR/v3.0.1_metadata_only")
DEFAULT_TUAR_AUDIT = DEFAULT_TUAR_ROOT / "AUDIT_RECEIPT.md"
DEFAULT_SZCORE_CANDIDATES = (
    Path("/mnt/hd1/dyf/dataset/SzCORE"),
    Path("/mnt/hd1/dyf/dataset/szcore"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument(
        "--tusz-fold-plan",
        type=Path,
        default=DEFAULT_TUSZ_AUDIT_ROOT / "detector_cleanroom_fold_plan.json",
    )
    parser.add_argument(
        "--tusz-physical-projection",
        type=Path,
        default=DEFAULT_TUSZ_AUDIT_ROOT / "physical_analysis_projection.json",
    )
    parser.add_argument(
        "--deepsoz-identity-binding", type=Path, default=DEFAULT_DEEPSOZ_IDENTITY
    )
    parser.add_argument(
        "--deepsoz-public-union", type=Path, default=DEFAULT_DEEPSOZ_UNION
    )
    parser.add_argument(
        "--deepsoz-external-attestation", type=Path, default=DEFAULT_DEEPSOZ_EXTERNAL
    )
    parser.add_argument("--tuev-edf-root", type=Path, default=DEFAULT_TUEV_ROOT)
    parser.add_argument("--tuev-readme", type=Path, default=DEFAULT_TUEV_README)
    parser.add_argument("--tuar-metadata-root", type=Path, default=DEFAULT_TUAR_ROOT)
    parser.add_argument("--tuar-audit-receipt", type=Path, default=DEFAULT_TUAR_AUDIT)
    parser.add_argument(
        "--szcore-candidate-root",
        action="append",
        type=Path,
        dest="szcore_candidate_roots",
        help="repeatable; defaults to the two locally audited candidate roots",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--validate-registry",
        type=Path,
        help="rebuild from sources and compare with an existing registry",
    )
    return parser


def _source_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        "plan_path": arguments.plan,
        "tusz_fold_plan_path": arguments.tusz_fold_plan,
        "tusz_physical_projection_path": arguments.tusz_physical_projection,
        "deepsoz_identity_binding_path": arguments.deepsoz_identity_binding,
        "deepsoz_public_union_manifest_path": arguments.deepsoz_public_union,
        "deepsoz_external_attestation_path": arguments.deepsoz_external_attestation,
        "tuev_edf_root": arguments.tuev_edf_root,
        "tuev_readme_path": arguments.tuev_readme,
        "tuar_metadata_root": arguments.tuar_metadata_root,
        "tuar_audit_receipt_path": arguments.tuar_audit_receipt,
        "szcore_candidate_roots": (
            arguments.szcore_candidate_roots or list(DEFAULT_SZCORE_CANDIDATES)
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    source_arguments = _source_arguments(arguments)
    if arguments.validate_registry is not None:
        payload = json.loads(
            arguments.validate_registry.read_text(encoding="utf-8")
        )
        registry = replay_full_stack_nested_exposure_registry_from_paths_v1(
            payload, **source_arguments
        )
        status = "source_replay_valid"
        output = arguments.validate_registry
    else:
        registry = build_full_stack_nested_exposure_registry_from_paths_v1(
            **source_arguments
        )
        output = materialize_full_stack_nested_exposure_registry_v1(
            registry, arguments.output
        )
        status = "materialized"
    by_dataset = {
        row["dataset_id"]: row for row in registry["dataset_registry"]
    }
    tuev = by_dataset["TUEV"]["identity_snapshot"]
    print(
        json.dumps(
            {
                "status": status,
                "output": str(output),
                "registry_id": registry["registry_id"],
                "receipt_sha256": registry["receipt_sha256"],
                "outer_fold_count": len(registry["outer_folds"]),
                "TUSZ_source_train_patients": by_dataset["TUSZ"]["patient_count"],
                "TUSZ_source_train_records": by_dataset["TUSZ"]["record_count"],
                "DeepSOZ_bound_source_train_patients": by_dataset["DeepSOZ"][
                    "source_train_patient_count"
                ],
                "DeepSOZ_bound_source_train_records": by_dataset["DeepSOZ"][
                    "source_train_record_count"
                ],
                "TUEV_train_visible_patients": tuev["train_patient_count"],
                "TUEV_TUSZ_source_train_visible_patient_overlap": len(
                    tuev["tusz_visible_patient_overlap_by_split"]["source_train"]
                ),
                "TUEV_TUSZ_exact_container_overlap": sum(
                    len(value)
                    for value in tuev["tusz_exact_container_overlap_by_split"].values()
                ),
                "legal_full_stack_OOF_inventory_exists": registry[
                    "artifact_inventory"
                ]["legal_full_stack_OOF_inventory_exists"],
                "source_eval_reference_opened": registry["data_access_receipt"][
                    "TUSZ_source_eval_reference_opened"
                ],
                "model_training_executed": registry["data_access_receipt"][
                    "model_training_executed"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
