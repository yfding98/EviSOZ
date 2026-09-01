#!/usr/bin/env python3
"""Validate EventNet Stage-P and freeze a reference-free native decoder grid."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.eventnet_native_decoder_grid import (  # noqa: E402
    EVENTNET_TUSZ_ANALYSIS_PROJECTION_BINDING_SCHEMA_VERSION,
    materialize_eventnet_native_decoder_grid,
    validate_eventnet_native_decoder_grid,
)
from src.clinical_eeg_long_recording.eventnet_stage_p_raw_prediction_bridge import (  # noqa: E402
    validate_eventnet_stage_p_raw_prediction_bundle_without_references,
)
from src.clinical_eeg_long_recording.tusz_complete_detector_roster_v2 import (  # noqa: E402
    validate_tusz_analysis_identity_projection_v2,
)
from src.clinical_eeg_long_recording.tusz_complete_detector_roster_v1 import (  # noqa: E402
    TUSZ_V203_EXPECTED_INVENTORY,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_json(path: Path, context: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"{context} is empty")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must contain one JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_complete_roster_projection(path: Path) -> dict[str, Any]:
    return validate_tusz_analysis_identity_projection_v2(
        _read_json(path, "complete TUSZ analysis-identity projection")
    )


def _source_dev_projection_rows(
    projection: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = [
        row for row in projection["records"] if row["model_split"] == "source_dev"
    ]
    recording_ids = [row["local_edf_path"] for row in rows]
    if not rows or recording_ids != sorted(recording_ids):
        raise ValueError("complete TUSZ projection source-dev roster is not canonical")
    summary = projection["split_summaries"]["source_dev"]
    if summary["analysis_identity_count"] != len(rows):
        raise ValueError("complete TUSZ projection source-dev count drifted")
    return rows, recording_ids


def _source_dev_projection_binding(
    projection: dict[str, Any],
    *,
    projection_file_sha256: str,
    raw_bundle_receipt: dict[str, Any],
) -> dict[str, Any]:
    projection = validate_tusz_analysis_identity_projection_v2(projection)
    rows, recording_ids = _source_dev_projection_rows(projection)
    recording_roster_sha256 = _canonical_sha256(recording_ids)
    stage_lineage = raw_bundle_receipt.get("stage_p_lineage")
    if not isinstance(stage_lineage, dict):
        raise ValueError("raw bundle lacks Stage-P lineage")
    projection_matches_stage_p = (
        stage_lineage.get("upstream_complete_projection_receipt_sha256")
        == projection["receipt_sha256"]
    )
    if not projection_matches_stage_p:
        raise ValueError("TUSZ projection receipt disagrees with Stage-P input")
    if (
        raw_bundle_receipt["expected_recording_roster_sha256"]
        != recording_roster_sha256
    ):
        raise ValueError("TUSZ projection roster disagrees with Stage-P denominator")
    summary = projection["split_summaries"]["source_dev"]
    source = projection["source_roster_binding"]
    expected_dev = TUSZ_V203_EXPECTED_INVENTORY["split_expectations"]["dev"]
    completeness = {
        "analysis_projection_schema_validated": True,
        "projection_receipt_matches_stage_p_input": projection_matches_stage_p,
        "source_roster_schema_v2_bound": (
            source["source_schema_version"] == "tusz_complete_detector_roster_v2"
        ),
        "source_release_matches_tusz_v203_expected_inventory": (
            source["source_release_id"]
            == TUSZ_V203_EXPECTED_INVENTORY["release_id"]
        ),
        "source_global_audit_recording_count_matches_tusz_v203": (
            source["source_audit_recording_count"]
            == TUSZ_V203_EXPECTED_INVENTORY["total_recording_count"]
        ),
        "source_dev_audit_recording_count_matches_tusz_v203": (
            summary["audit_official_path_count"]
            == expected_dev["recording_count"]
        ),
        "source_dev_patient_alias_count_matches_tusz_v203": (
            summary["analysis_patient_alias_count"]
            == expected_dev["patient_count"]
        ),
        "source_dev_analysis_identity_count_matches_tusz_v203": (
            summary["analysis_identity_count"]
            == expected_dev["recording_count"]
        ),
        "source_dev_has_no_projection_exclusions_or_quarantine": (
            summary["same_patient_alias_excluded_path_count"] == 0
            and summary["quarantined_path_count"] == 0
        ),
        "source_dev_path_count_closure_verified": (
            summary["path_count_closure_verified"] is True
        ),
        "source_dev_recording_count_matches_analysis_identity_count": (
            len(rows) == summary["analysis_identity_count"]
        ),
    }
    return {
        "binding_schema_version": (
            EVENTNET_TUSZ_ANALYSIS_PROJECTION_BINDING_SCHEMA_VERSION
        ),
        "analysis_projection_schema_version": projection["schema_version"],
        "analysis_projection_id": projection["projection_id"],
        "analysis_projection_receipt_sha256": projection["receipt_sha256"],
        "analysis_projection_file_sha256": projection_file_sha256,
        "source_roster_binding": deepcopy(source),
        "source_dev_split_summary": deepcopy(summary),
        "source_dev_recording_count": len(rows),
        "source_dev_recording_roster_sha256": recording_roster_sha256,
        "inventory_scope": (
            "complete_tusz_v2_source_dev_analysis_identity_projection"
        ),
        "completeness_receipt": completeness,
        "complete_split_inventory_verified": all(completeness.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete EventNet source-dev Stage-P committed-attempt "
            "inventory and materialize a prediction-only native decoder grid"
        )
    )
    parser.add_argument("--stage-p-root", type=Path, required=True)
    parser.add_argument(
        "--complete-roster-projection",
        type=Path,
        required=True,
        help=(
            "validated tusz_analysis_identity_projection_v2 identity denominator; "
            "not a seizure-reference file"
        ),
    )
    parser.add_argument("--expected-run-contract-sha256")
    parser.add_argument("--decoder-grid", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    projection_file_sha256 = _file_sha256(arguments.complete_roster_projection)
    projection = _load_complete_roster_projection(arguments.complete_roster_projection)
    if _file_sha256(arguments.complete_roster_projection) != projection_file_sha256:
        raise ValueError("complete TUSZ projection changed during validation")
    _source_dev_rows, source_dev_recording_ids = _source_dev_projection_rows(
        projection
    )
    raw_bundle = (
        validate_eventnet_stage_p_raw_prediction_bundle_without_references(
            arguments.stage_p_root,
            expected_recording_ids=source_dev_recording_ids,
            expected_run_contract_sha256=arguments.expected_run_contract_sha256,
        )
    )
    grid = validate_eventnet_native_decoder_grid(
        _read_json(arguments.decoder_grid, "EventNet decoder grid")
    )
    binding = _source_dev_projection_binding(
        projection,
        projection_file_sha256=projection_file_sha256,
        raw_bundle_receipt=raw_bundle.validation_receipt(),
    )
    if binding["complete_split_inventory_verified"] is not True:
        raise ValueError(
            "TUSZ projection is not the complete official v2.0.3 source-dev "
            "analysis denominator"
        )
    receipt = materialize_eventnet_native_decoder_grid(
        raw_bundle,
        grid_definition=grid,
        output_directory=arguments.output_directory,
        source_dev_roster_binding=binding,
    )
    print(
        json.dumps(
            {
                "bundle_id": receipt["bundle_id"],
                "receipt_sha256": receipt["receipt_sha256"],
                "raw_bundle_validation_id": receipt[
                    "raw_bundle_validation_id"
                ],
                "analysis_projection_id": projection["projection_id"],
                "record_count": receipt["record_count"],
                "policy_count": receipt["policy_count"],
                "prediction_row_count": receipt["prediction_row_count"],
                "raw_proposal_count": receipt["raw_proposal_count"],
                "merged_alarm_count": receipt["merged_alarm_count"],
                "reference_files_opened": receipt["reference_access"][
                    "reference_files_opened"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
