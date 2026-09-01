#!/usr/bin/env python3
"""Create a de-identified recovery overlay for private EEG report batches.

This command is intentionally a manifest-only operation.  It reads one frozen
inventory, the primary and recovery coverage manifests, optional remediation
coverage/release-audit receipts, and only the selected per-record
``state.json`` and report ``manifest.json`` files.  It never opens an EDF,
annotation, spreadsheet, report body, or waveform artifact.

A recovery artifact is effective only when a completed EEG report replaces a
primary ``completed_technical_unassessable`` shell.  A recovery technical
shell, pending row, or second EEG report never displaces the primary artifact.
An EEG-to-EEG renderer remediation is a separate, optional path: the primary
cohort release audit must explicitly authorize the record and a fresh
remediation-scoped release audit must pass that exact replacement set.
The source trees are not copied, deleted, or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import materialize_private_long_recording_reports_v1 as batch  # noqa: E402
from src.clinical_eeg_long_recording.pipeline import (  # noqa: E402
    FILTERED_MATERIALIZATION_SCHEMA,
)


SCHEMA_VERSION = "private_long_recording_report_combined_coverage_v1"
EEG_REPORT_SCHEMA_VERSION = "trustworthy_long_term_clinical_eeg_materialization_v1"
EEG_REPORT_SIGNAL_PARTITION_SCHEMA_VERSION = FILTERED_MATERIALIZATION_SCHEMA
RELEASE_AUDIT_SCHEMA_VERSION = "private_long_recording_report_release_audit_v1"
REPLACEMENT_AUTHORIZATION_SCHEMA_VERSION = (
    "private_long_recording_report_replacement_authorization_v1"
)
REMEDIATION_SCOPE_SCHEMA_VERSION = (
    "private_long_recording_report_remediation_release_scope_v1"
)
COHORT_AUDIT_MODE = "cohort_release"
REMEDIATION_AUDIT_MODE = "remediation_subset_release"
RELEASE_AUDIT_PASS_STATUS = "release_audit_passed"
RELEASE_AUDIT_FAIL_STATUS = "release_audit_failed"
AUTHORIZATION_POLICY_ID = (
    "replace_only_audit_failed_primary_eeg_with_freshly_audited_eeg_v1"
)
TECHNICAL_STATUS = "completed_technical_unassessable"
EEG_STATUSES = frozenset(batch.COMPLETED_DIAGNOSTIC_STATUSES)
ALL_COMPLETED_STATUSES = frozenset({*EEG_STATUSES, TECHNICAL_STATUS})
RENDERER_ONLY_FAILURE_CODES = frozenset(
    {
        "language_projection_invalid",
        "document_artifact_invalid",
        "unauthorized_positive_clinical_language",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TECHNICAL_DIR_RE = re.compile(r"^technical_reports/attempt_[0-9]{4,}$")
_COVERAGE_KEYS = {
    "schema_version",
    "inventory_id",
    "recording_unit_policy",
    "mode",
    "expected_record_count",
    "expected_subject_count",
    "inventory_rejection_count",
    "completed_report_count",
    "completed_report_artifact_count",
    "completed_eeg_report_count",
    "technical_unassessable_report_count",
    "technical_failure_count",
    "pending_or_not_run_count",
    "dataset_coverage_complete",
    "dataset_artifact_coverage_complete",
    "dataset_eeg_coverage_complete",
    "diagnostic_status_counts",
    "records",
    "subjects",
    "scope_receipt",
}
_COVERAGE_ROW_KEYS = {
    "recording_id",
    "patient_pseudonym",
    "inventory_validation_status",
    "run_status",
    "diagnostic_status",
    "event_count",
    "failure_stage",
    "existing_success_reused",
    "technical_artifact_relative_dir",
}
_SOURCE_SUBJECT_KEYS = {
    "patient_pseudonym",
    "expected_record_count",
    "completed_report_artifact_count",
    "completed_eeg_report_count",
    "technical_unassessable_report_count",
    "technical_failure_count",
    "has_at_least_one_report_artifact",
    "has_at_least_one_completed_eeg_report",
    "coverage_complete",
    "eeg_coverage_complete",
}
_COVERAGE_SCOPE_KEYS = {
    "one_report_unit_per_inventory_recording_unit",
    "recording_unit_policy",
    "source_event_rows_deduplicated_before_inference",
    "generation_uses_eeg_signal_only",
    "edf_annotations_loaded",
    "excel_or_workbook_loaded",
    "onset_or_label_fields_forwarded",
    "ground_truth_forwarded",
    "qwen_optional",
    "qwen_requested",
    "qwen_failure_blocks_report",
    "zero_candidates_still_materialize_report",
    "findings_insufficiency_is_completed_abstention",
    "technical_failure_is_not_eeg_insufficiency",
    "technical_failure_gets_non_diagnostic_report_shell",
    "raw_edf_paths_or_patient_names_in_coverage",
}
_STATE_KEYS = {
    "schema_version",
    "recording_id",
    "patient_pseudonym",
    "status",
    "last_completed_stage",
    "diagnostic_status",
    "event_count",
    "attempt",
    "scope_receipt",
}
_STATE_SCOPE = {
    "eeg_signal_only": True,
    "edf_annotations_loaded": False,
    "excel_loaded": False,
    "onset_or_ground_truth_loaded": False,
}
_EEG_REPORT_KEYS = {
    "schema_version",
    "status",
    "bundle_id",
    "recording_id",
    "patient_pseudonym",
    "event_count",
    "diagnostic_status",
    "diagnostic_outcome",
    "artifacts",
    "language_service_receipt",
    "source_receipts",
    "scope_receipt",
}
_EEG_REPORT_SIGNAL_PARTITION_KEYS = _EEG_REPORT_KEYS | {
    "detector_selected_candidate_count",
    "analysis_analyzable_candidate_count",
    "analysis_rejected_candidate_count",
}
_TECHNICAL_REPORT_KEYS = {
    "schema_version",
    "status",
    "recording_id",
    "patient_pseudonym",
    "diagnostic_status",
    "failure_stage",
    "event_count",
    "artifacts",
    "technical_failure_receipt_fingerprint",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid constant {value!r}")


def _snapshot(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"manifest input must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"manifest input must be a regular file: {path}")
    raw = resolved.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_invalid_constant,
    )
    if not isinstance(value, Mapping):
        raise TypeError(f"manifest input must be a JSON object: {path}")
    return {
        "path": resolved,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "value": dict(value),
    }


def _assert_snapshots_unchanged(snapshots: Sequence[Mapping[str, Any]]) -> None:
    seen: set[Path] = set()
    for snapshot in snapshots:
        path = Path(snapshot["path"])
        if path in seen:
            continue
        seen.add(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("a source manifest changed during overlay construction")
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if current != snapshot["sha256"]:
            raise ValueError("a source manifest changed during overlay construction")


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _safe_relative(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be non-empty relative text")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{context} is not a safe relative path")
    return relative


def _source_subjects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["patient_pseudonym"]), []).append(row)
    result: list[dict[str, Any]] = []
    for patient, patient_rows in sorted(grouped.items()):
        eeg = sum(row["diagnostic_status"] in EEG_STATUSES for row in patient_rows)
        technical = sum(
            row["diagnostic_status"] == TECHNICAL_STATUS for row in patient_rows
        )
        failures = sum(row["failure_stage"] is not None for row in patient_rows)
        artifacts = eeg + technical
        result.append(
            {
                "patient_pseudonym": patient,
                "expected_record_count": len(patient_rows),
                "completed_report_artifact_count": artifacts,
                "completed_eeg_report_count": eeg,
                "technical_unassessable_report_count": technical,
                "technical_failure_count": failures,
                "has_at_least_one_report_artifact": artifacts > 0,
                "has_at_least_one_completed_eeg_report": eeg > 0,
                "coverage_complete": artifacts == len(patient_rows),
                "eeg_coverage_complete": eeg == len(patient_rows),
            }
        )
    return result


def _validate_coverage(
    value: object,
    *,
    inventory: Mapping[str, Any],
    source_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _COVERAGE_KEYS:
        raise ValueError(f"{source_name} coverage has missing or unknown keys")
    if value["schema_version"] != batch.COVERAGE_SCHEMA_VERSION:
        raise ValueError(f"{source_name} coverage schema drifted")
    if value["inventory_id"] != inventory["inventory_id"]:
        raise ValueError(f"{source_name} coverage inventory binding drifted")
    if value["recording_unit_policy"] != inventory["recording_unit_policy"]:
        raise ValueError(f"{source_name} coverage recording policy drifted")
    if value["mode"] != "execution":
        raise ValueError(f"{source_name} coverage must be an execution manifest")
    if value["expected_record_count"] != inventory["record_count"]:
        raise ValueError(f"{source_name} coverage record count drifted")
    if value["expected_subject_count"] != inventory["subject_count"]:
        raise ValueError(f"{source_name} coverage subject count drifted")
    if value["inventory_rejection_count"] != len(inventory["source_rejections"]):
        raise ValueError(f"{source_name} coverage rejection count drifted")

    raw_rows = value["records"]
    if not isinstance(raw_rows, list):
        raise TypeError(f"{source_name} coverage records must be an array")
    inventory_records = list(inventory["records"])
    if len(raw_rows) != len(inventory_records):
        raise ValueError(f"{source_name} coverage does not span the inventory")
    rows: list[dict[str, Any]] = []
    for index, (raw, inventory_record) in enumerate(
        zip(raw_rows, inventory_records, strict=True)
    ):
        if not isinstance(raw, Mapping) or set(raw) != _COVERAGE_ROW_KEYS:
            raise ValueError(
                f"{source_name} coverage row {index} has missing or unknown keys"
            )
        for key in (
            "recording_id",
            "patient_pseudonym",
            "inventory_validation_status",
        ):
            if raw[key] != inventory_record[key]:
                raise ValueError(f"{source_name} coverage row identity drifted")
        if not isinstance(raw["run_status"], str) or not raw["run_status"]:
            raise ValueError(f"{source_name} coverage run_status is invalid")
        diagnostic = raw["diagnostic_status"]
        if diagnostic is not None and diagnostic not in ALL_COMPLETED_STATUSES:
            raise ValueError(f"{source_name} coverage diagnostic status is invalid")
        failure_stage = raw["failure_stage"]
        if failure_stage is not None and (
            not isinstance(failure_stage, str) or not failure_stage
        ):
            raise ValueError(f"{source_name} coverage failure stage is invalid")
        if not isinstance(raw["existing_success_reused"], bool):
            raise TypeError(f"{source_name} coverage reused flag must be boolean")
        technical_dir = raw["technical_artifact_relative_dir"]
        if technical_dir is not None and (
            not isinstance(technical_dir, str)
            or _TECHNICAL_DIR_RE.fullmatch(technical_dir) is None
        ):
            raise ValueError(f"{source_name} coverage technical path is unsafe")
        event_count = raw["event_count"]
        if diagnostic in EEG_STATUSES:
            _nonnegative_int(event_count, f"{source_name} EEG event_count")
            if failure_stage is not None or technical_dir is not None:
                raise ValueError(f"{source_name} EEG row carries technical fields")
        elif diagnostic == TECHNICAL_STATUS:
            if event_count != 0 or failure_stage is None or technical_dir is None:
                raise ValueError(f"{source_name} technical row is incomplete")
        elif event_count is not None or technical_dir is not None:
            raise ValueError(f"{source_name} pending row carries an artifact")
        rows.append(dict(raw))

    completed_eeg = sum(row["diagnostic_status"] in EEG_STATUSES for row in rows)
    technical = sum(row["diagnostic_status"] == TECHNICAL_STATUS for row in rows)
    failures = sum(row["failure_stage"] is not None for row in rows)
    artifacts = completed_eeg + technical
    expected_counts = {
        "completed_report_count": artifacts,
        "completed_report_artifact_count": artifacts,
        "completed_eeg_report_count": completed_eeg,
        "technical_unassessable_report_count": technical,
        "technical_failure_count": failures,
        "pending_or_not_run_count": len(rows) - artifacts,
        "dataset_coverage_complete": (
            artifacts == inventory["record_count"]
            and not inventory["source_rejections"]
        ),
        "dataset_artifact_coverage_complete": (
            artifacts == inventory["record_count"]
            and not inventory["source_rejections"]
        ),
        "dataset_eeg_coverage_complete": (
            completed_eeg == inventory["record_count"]
            and not inventory["source_rejections"]
        ),
    }
    for key, expected in expected_counts.items():
        if value[key] != expected:
            raise ValueError(f"{source_name} coverage {key} is inconsistent")
    expected_diagnostics = {
        status: sum(row["diagnostic_status"] == status for row in rows)
        for status in sorted(ALL_COMPLETED_STATUSES)
    }
    if value["diagnostic_status_counts"] != expected_diagnostics:
        raise ValueError(f"{source_name} coverage diagnostic counts drifted")
    raw_subjects = value["subjects"]
    if not isinstance(raw_subjects, list) or any(
        not isinstance(subject, Mapping) or set(subject) != _SOURCE_SUBJECT_KEYS
        for subject in raw_subjects
    ):
        raise ValueError(f"{source_name} coverage subject rows are invalid")
    if raw_subjects != _source_subjects(rows):
        raise ValueError(f"{source_name} coverage subject summaries drifted")
    scope = value["scope_receipt"]
    if not isinstance(scope, Mapping) or set(scope) != _COVERAGE_SCOPE_KEYS:
        raise ValueError(f"{source_name} coverage scope receipt drifted")
    required_scope = {
        "one_report_unit_per_inventory_recording_unit": True,
        "recording_unit_policy": inventory["recording_unit_policy"],
        "source_event_rows_deduplicated_before_inference": True,
        "generation_uses_eeg_signal_only": True,
        "edf_annotations_loaded": False,
        "excel_or_workbook_loaded": False,
        "onset_or_label_fields_forwarded": False,
        "ground_truth_forwarded": False,
        "qwen_optional": True,
        "qwen_failure_blocks_report": False,
        "zero_candidates_still_materialize_report": True,
        "findings_insufficiency_is_completed_abstention": True,
        "technical_failure_is_not_eeg_insufficiency": True,
        "technical_failure_gets_non_diagnostic_report_shell": True,
        "raw_edf_paths_or_patient_names_in_coverage": False,
    }
    if any(scope.get(key) != expected for key, expected in required_scope.items()):
        raise ValueError(f"{source_name} coverage violates the EEG-only boundary")
    if not isinstance(scope["qwen_requested"], bool):
        raise TypeError(f"{source_name} qwen_requested must be boolean")
    return {**dict(value), "records": rows}


def _validate_release_audit_identity(
    value: object,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    audit = dict(value)
    if audit.get("schema_version") != RELEASE_AUDIT_SCHEMA_VERSION:
        raise ValueError(f"{context} schema drifted")
    audit_id = audit.get("audit_id")
    body = {key: item for key, item in audit.items() if key != "audit_id"}
    if audit_id != "PLRAUD-" + _canonical_sha256(body)[:24]:
        raise ValueError(f"{context} ID does not bind its content")
    return audit


def _validate_primary_release_authorization(
    value: object,
    *,
    inventory: Mapping[str, Any],
    inventory_manifest_sha256: str,
    primary_coverage_manifest_sha256: str,
    primary_coverage: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    audit = _validate_release_audit_identity(
        value, context="primary release audit"
    )
    if audit.get("audit_mode") != COHORT_AUDIT_MODE:
        raise ValueError("primary release audit is not an explicit cohort audit")
    if audit.get("status") != RELEASE_AUDIT_FAIL_STATUS or audit.get(
        "release_ready"
    ) is not False:
        raise ValueError("an audit-passed primary cannot authorize replacement")
    if audit.get("coverage_kind") != "full":
        raise ValueError("primary release audit must bind full primary coverage")
    if audit.get("inventory_id") != inventory["inventory_id"] or audit.get(
        "recording_unit_policy"
    ) != inventory["recording_unit_policy"]:
        raise ValueError("primary release audit inventory binding drifted")
    sources = audit.get("source_receipts")
    if not isinstance(sources, Mapping) or sources.get(
        "inventory_manifest_sha256"
    ) != inventory_manifest_sha256 or sources.get(
        "coverage_manifest_sha256"
    ) != primary_coverage_manifest_sha256:
        raise ValueError("primary release audit source snapshot binding drifted")
    scope = audit.get("scope_receipt")
    required_scope = {
        "edf_signal_files_read": False,
        "edf_annotations_read": False,
        "excel_or_workbook_read": False,
        "onset_label_or_ground_truth_read": False,
        "inventory_source_locator_resolved_or_opened": False,
        "report_artifacts_modified": False,
    }
    if not isinstance(scope, Mapping) or any(
        scope.get(key) != expected for key, expected in required_scope.items()
    ):
        raise ValueError("primary release audit violates the source boundary")
    counts = audit.get("cohort_counts")
    completed_eeg = int(primary_coverage["completed_eeg_report_count"])
    if (
        not isinstance(counts, Mapping)
        or counts.get("expected_record_count") != inventory["record_count"]
        or counts.get("expected_subject_count") != inventory["subject_count"]
        or counts.get("completed_eeg_report_count") != completed_eeg
        or counts.get("completed_eeg_reports_audited") != completed_eeg
    ):
        raise ValueError("primary release audit cohort counts drifted")
    passed = _nonnegative_int(
        counts.get("completed_eeg_reports_passed"),
        "primary release audit passed count",
    )
    failed = _nonnegative_int(
        counts.get("completed_eeg_reports_failed"),
        "primary release audit failed count",
    )
    if passed + failed != completed_eeg:
        raise ValueError("primary release audit report counts do not close")
    checks = audit.get("checks")
    if not isinstance(checks, Mapping) or any(
        checks.get(key) is not True
        for key in (
            "inventory_schema_and_binding_validated",
            "coverage_schema_and_binding_validated",
            "every_completed_eeg_row_audited",
            "source_artifact_snapshots_unchanged",
        )
    ):
        raise ValueError("primary release audit did not complete required checks")

    failed_reports = audit.get("failed_reports")
    if not isinstance(failed_reports, list) or len(failed_reports) != failed:
        raise ValueError("primary release audit failure roster drifted")
    receipt = audit.get("replacement_authorization_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("primary replacement authorization is absent")
    authorization_id = receipt.get("authorization_id")
    receipt_body = {
        key: item for key, item in receipt.items() if key != "authorization_id"
    }
    if authorization_id != "PLRAUTH-" + _canonical_sha256(receipt_body)[:24]:
        raise ValueError("primary replacement authorization ID drifted")
    if (
        receipt.get("schema_version")
        != REPLACEMENT_AUTHORIZATION_SCHEMA_VERSION
        or receipt.get("authorization_policy_id") != AUTHORIZATION_POLICY_ID
        or receipt.get("eligible_for_primary_replacement") is not True
    ):
        raise ValueError("primary replacement authorization policy drifted")
    raw_authorizations = receipt.get("authorizations")
    if (
        not isinstance(raw_authorizations, list)
        or receipt.get("authorized_failed_primary_eeg_count")
        != len(raw_authorizations)
        or raw_authorizations != failed_reports
    ):
        raise ValueError("primary replacement authorization set drifted")
    constraints = receipt.get("constraints")
    required_constraints = {
        "only_listed_recording_ids_may_replace_primary_eeg": True,
        "source_report_manifest_sha256_must_match_when_present": True,
        "audit_passed_primary_eeg_may_be_replaced": False,
        "replacement_technical_or_pending_artifact_allowed": False,
        "replacement_must_be_completed_eeg_report": True,
        "replacement_must_pass_fresh_release_audit": True,
        "replacement_may_change_inventory_or_recording_unit": False,
    }
    if not isinstance(constraints, Mapping) or any(
        constraints.get(key) != expected
        for key, expected in required_constraints.items()
    ):
        raise ValueError("primary replacement authorization constraints drifted")

    inventory_ids = {str(record["recording_id"]) for record in inventory["records"]}
    all_authorizations: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_authorizations:
        if not isinstance(raw, Mapping):
            raise TypeError("primary replacement authorization row is invalid")
        item = dict(raw)
        recording_id = item.get("recording_id")
        source_sha = item.get("source_report_manifest_sha256")
        reasons = item.get("failure_reason_codes")
        renderer_only = (
            item.get("failure_layer") == "language_or_renderer_projection"
            and item.get("minimum_remediation") == "rerender_report_only"
        )
        if (
            not isinstance(recording_id, str)
            or recording_id not in inventory_ids
            or recording_id in seen
            or item.get("selected_artifact_source") != "primary"
            or (
                source_sha is not None
                and (
                    not isinstance(source_sha, str)
                    or _SHA256_RE.fullmatch(source_sha) is None
                )
            )
            or not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise ValueError("primary replacement authorization row drifted")
        seen.add(recording_id)
        all_authorizations.append(item)
        if renderer_only:
            if (
                not isinstance(source_sha, str)
                or _SHA256_RE.fullmatch(source_sha) is None
                or not set(reasons).issubset(RENDERER_ONLY_FAILURE_CODES)
            ):
                raise ValueError(
                    "primary renderer authorization is not manifest-bound"
                )
            eligible.append(item)
    if not eligible:
        raise ValueError("primary audit authorizes no renderer-only replacement")
    return (
        audit,
        sorted(all_authorizations, key=lambda item: str(item["recording_id"])),
        sorted(eligible, key=lambda item: str(item["recording_id"])),
    )


def _validate_remediation_release_audit(
    value: object,
    *,
    inventory: Mapping[str, Any],
    inventory_manifest_sha256: str,
    remediation_coverage_manifest_sha256: str,
    primary_release_audit: Mapping[str, Any],
    primary_release_audit_sha256: str,
    all_authorizations: Sequence[Mapping[str, Any]],
    eligible_authorizations: Sequence[Mapping[str, Any]],
    primary_rows: Mapping[str, Mapping[str, Any]],
    remediation_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    audit = _validate_release_audit_identity(
        value, context="remediation release audit"
    )
    if audit.get("audit_mode") != REMEDIATION_AUDIT_MODE:
        raise ValueError("fresh release audit is not remediation-scoped")
    if (
        audit.get("status") != RELEASE_AUDIT_PASS_STATUS
        or audit.get("release_ready") is not True
        or audit.get("remediation_release_ready") is not True
    ):
        raise ValueError("fresh remediation release audit did not pass")
    if audit.get("inventory_id") != inventory["inventory_id"] or audit.get(
        "recording_unit_policy"
    ) != inventory["recording_unit_policy"]:
        raise ValueError("remediation release audit inventory binding drifted")
    audit_scope = audit.get("scope_receipt")
    required_audit_scope = {
        "edf_signal_files_read": False,
        "edf_annotations_read": False,
        "excel_or_workbook_read": False,
        "onset_label_or_ground_truth_read": False,
        "inventory_source_locator_resolved_or_opened": False,
        "report_artifacts_modified": False,
    }
    if not isinstance(audit_scope, Mapping) or any(
        audit_scope.get(key) != expected
        for key, expected in required_audit_scope.items()
    ):
        raise ValueError("remediation release audit violates the source boundary")
    sources = audit.get("source_receipts")
    if (
        not isinstance(sources, Mapping)
        or sources.get("inventory_manifest_sha256") != inventory_manifest_sha256
        or sources.get("coverage_manifest_sha256")
        != remediation_coverage_manifest_sha256
        or sources.get("primary_release_audit_sha256")
        != primary_release_audit_sha256
    ):
        raise ValueError("remediation release audit source snapshot binding drifted")
    primary_receipt = primary_release_audit["replacement_authorization_receipt"]
    eligible_ids = [str(item["recording_id"]) for item in eligible_authorizations]
    selected_ids = sorted(
        recording_id
        for recording_id, row in remediation_rows.items()
        if row["diagnostic_status"] in EEG_STATUSES
    )
    if any(
        row["diagnostic_status"] == TECHNICAL_STATUS
        for row in remediation_rows.values()
    ):
        raise ValueError("technical remediation artifacts cannot replace primary EEG")
    if selected_ids != eligible_ids:
        raise ValueError(
            "remediation coverage completed EEG set differs from authorization"
        )
    eligible_id_set = set(eligible_ids)
    for recording_id, row in remediation_rows.items():
        if recording_id in eligible_id_set:
            continue
        if (
            row["run_status"] != "not_selected_in_this_run"
            or row["diagnostic_status"] is not None
            or row["event_count"] is not None
            or row["failure_stage"] is not None
            or row["technical_artifact_relative_dir"] is not None
        ):
            raise ValueError("remediation coverage selected an unauthorized record")

    scope = audit.get("remediation_scope")
    expected_scope_keys = {
        "schema_version",
        "source_primary_release_audit_id",
        "source_primary_release_audit_sha256",
        "source_replacement_authorization_id",
        "source_authorization_policy_id",
        "authorization_set_sha256",
        "expected_authorized_recording_count",
        "expected_authorized_recording_ids",
        "selected_coverage_recording_count",
        "selected_coverage_recording_ids",
        "selected_set_exactly_matches_authorization",
        "no_extra_or_missing_selected_reports",
        "only_language_or_renderer_rerender_authorized",
    }
    if not isinstance(scope, Mapping) or set(scope) != expected_scope_keys:
        raise ValueError("remediation release scope schema drifted")
    expected_scope = {
        "schema_version": REMEDIATION_SCOPE_SCHEMA_VERSION,
        "source_primary_release_audit_id": primary_release_audit["audit_id"],
        "source_primary_release_audit_sha256": primary_release_audit_sha256,
        "source_replacement_authorization_id": primary_receipt["authorization_id"],
        "source_authorization_policy_id": AUTHORIZATION_POLICY_ID,
        "authorization_set_sha256": _canonical_sha256(
            [dict(item) for item in all_authorizations]
        ),
        "expected_authorized_recording_count": len(eligible_ids),
        "expected_authorized_recording_ids": eligible_ids,
        "selected_coverage_recording_count": len(selected_ids),
        "selected_coverage_recording_ids": selected_ids,
        "selected_set_exactly_matches_authorization": True,
        "no_extra_or_missing_selected_reports": True,
        "only_language_or_renderer_rerender_authorized": True,
    }
    if dict(scope) != expected_scope:
        raise ValueError("remediation release scope authorization binding drifted")

    raw_results = audit.get("selected_report_results")
    result_keys = {
        "recording_id",
        "audit_status",
        "diagnostic_status",
        "event_count",
        "remediated_report_manifest_sha256",
        "source_primary_report_manifest_sha256",
        "failure_reason_codes",
        "failure_layer",
        "minimum_remediation",
    }
    if not isinstance(raw_results, list) or len(raw_results) != len(eligible_ids):
        raise ValueError("remediation selected-report results do not close")
    authorization_by_id = {
        str(item["recording_id"]): item for item in eligible_authorizations
    }
    results: dict[str, dict[str, Any]] = {}
    for raw in raw_results:
        if not isinstance(raw, Mapping) or set(raw) != result_keys:
            raise ValueError("remediation selected-report result schema drifted")
        item = dict(raw)
        recording_id = item["recording_id"]
        if recording_id not in authorization_by_id or recording_id in results:
            raise ValueError("remediation selected-report result set drifted")
        authorization = authorization_by_id[recording_id]
        primary_row = primary_rows[recording_id]
        remediation_row = remediation_rows[recording_id]
        fresh_sha = item["remediated_report_manifest_sha256"]
        source_sha = item["source_primary_report_manifest_sha256"]
        if (
            item["audit_status"] != "passed"
            or item["failure_layer"] != "language_or_renderer_projection"
            or item["minimum_remediation"] != "rerender_report_only"
            or item["failure_reason_codes"]
            != authorization["failure_reason_codes"]
            or source_sha != authorization["source_report_manifest_sha256"]
            or not isinstance(fresh_sha, str)
            or _SHA256_RE.fullmatch(fresh_sha) is None
            or fresh_sha == source_sha
            or item["diagnostic_status"] != primary_row["diagnostic_status"]
            or item["event_count"] != primary_row["event_count"]
            or remediation_row["diagnostic_status"]
            != primary_row["diagnostic_status"]
            or remediation_row["event_count"] != primary_row["event_count"]
            or remediation_row["existing_success_reused"] is not False
            or remediation_row["run_status"] != "completed"
        ):
            raise ValueError("remediation selected-report pass binding drifted")
        results[recording_id] = item
    if sorted(results) != eligible_ids:
        raise ValueError("remediation selected-report results omit authorization")
    return results


def _manifest_snapshot(
    root: Path,
    relative: PurePosixPath,
) -> dict[str, Any]:
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("source manifest path contains a symlink")
    resolved = cursor.resolve(strict=True)
    resolved.relative_to(root)
    return _snapshot(resolved)


def _validate_artifact_hashes(value: object, context: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{context} artifacts must be a non-empty object")
    for relative_text, digest in value.items():
        _safe_relative(relative_text, f"{context} artifact path")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"{context} artifact SHA-256 is invalid")


def _validate_state(
    value: object,
    *,
    inventory_record: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping) or set(value) != _STATE_KEYS:
        raise ValueError("selected state manifest has missing or unknown keys")
    if value["schema_version"] != batch.STATE_SCHEMA_VERSION:
        raise ValueError("selected state manifest schema drifted")
    for key in ("recording_id", "patient_pseudonym"):
        if value[key] != inventory_record[key]:
            raise ValueError("selected state manifest identity drifted")
    if value["diagnostic_status"] != row["diagnostic_status"]:
        raise ValueError("selected state diagnostic status drifted")
    if value["event_count"] != row["event_count"]:
        raise ValueError("selected state event count drifted")
    if (
        isinstance(value["attempt"], bool)
        or not isinstance(value["attempt"], int)
        or value["attempt"] < 1
    ):
        raise ValueError("selected state attempt is invalid")
    if value["scope_receipt"] != _STATE_SCOPE:
        raise ValueError("selected state violates the EEG-only boundary")
    if row["diagnostic_status"] in EEG_STATUSES:
        if value["status"] != "completed" or value["last_completed_stage"] != (
            "report_materialization"
        ):
            raise ValueError("selected EEG state is not completed")
    elif row["diagnostic_status"] == TECHNICAL_STATUS:
        if value["status"] != TECHNICAL_STATUS or value[
            "last_completed_stage"
        ] != "technical_report_materialization":
            raise ValueError("selected technical state is not completed")
    else:
        raise ValueError("a non-completed row cannot select an artifact")


def _validate_eeg_report(
    value: object,
    *,
    inventory_record: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("selected EEG report manifest must be an object")
    schema = value.get("schema_version")
    if schema == EEG_REPORT_SCHEMA_VERSION:
        expected_keys = _EEG_REPORT_KEYS
    elif schema == EEG_REPORT_SIGNAL_PARTITION_SCHEMA_VERSION:
        expected_keys = _EEG_REPORT_SIGNAL_PARTITION_KEYS
    else:
        raise ValueError("selected EEG report schema drifted")
    if set(value) != expected_keys:
        raise ValueError("selected EEG report manifest has missing or unknown keys")
    if value["status"] != "completed_unsigned_ai_draft":
        raise ValueError("selected EEG report is not completed")
    for key in ("recording_id", "patient_pseudonym"):
        if value[key] != inventory_record[key]:
            raise ValueError("selected EEG report identity drifted")
    if value["diagnostic_status"] != row["diagnostic_status"]:
        raise ValueError("selected EEG report diagnostic status drifted")
    if value["event_count"] != row["event_count"]:
        raise ValueError("selected EEG report event count drifted")
    outcome = value["diagnostic_outcome"]
    if not isinstance(outcome, Mapping) or outcome.get("report_status") != row[
        "diagnostic_status"
    ] or outcome.get("event_count") != row["event_count"]:
        raise ValueError("selected EEG diagnostic outcome drifted")
    _validate_artifact_hashes(value["artifacts"], "selected EEG report")
    scope = value["scope_receipt"]
    if not isinstance(scope, Mapping):
        raise TypeError("selected EEG report scope receipt must be an object")
    required = {
        "entire_record_detection_manifest_validated": True,
        "eeg_signal_only_generation": True,
        "eeg_facts_and_automatic_impression_signal_only": True,
        "external_edf_annotations_loaded": False,
        "excel_observations_loaded": False,
        "source_context_joined_post_freeze": False,
        "source_context_sent_to_qwen": False,
        "research_soz_used_in_clinical_facts_or_llm": False,
        "sleep_activation_ecg_emg_or_demographics_generated": False,
        "physician_signed": False,
    }
    if any(scope.get(key) != expected for key, expected in required.items()):
        raise ValueError("selected EEG report violates the EEG-only boundary")
    if schema == EEG_REPORT_SIGNAL_PARTITION_SCHEMA_VERSION:
        detector_selected = _nonnegative_int(
            value["detector_selected_candidate_count"],
            "selected EEG detector-selected count",
        )
        analyzable = _nonnegative_int(
            value["analysis_analyzable_candidate_count"],
            "selected EEG analyzable count",
        )
        rejected = _nonnegative_int(
            value["analysis_rejected_candidate_count"],
            "selected EEG rejected count",
        )
        if detector_selected != analyzable + rejected or analyzable != value[
            "event_count"
        ]:
            raise ValueError("selected EEG signal-eligibility partition does not close")
        sources = value["source_receipts"]
        artifacts = value["artifacts"]
        selection_sha = (
            sources.get("analysis_selection_sha256")
            if isinstance(sources, Mapping)
            else None
        )
        if (
            not isinstance(selection_sha, str)
            or _SHA256_RE.fullmatch(selection_sha) is None
            or artifacts.get("analysis_selection_manifest.json") != selection_sha
            or scope.get("signal_eligibility_partition_validated") is not True
            or scope.get("detector_selected_candidates_exactly_partitioned")
            is not True
            or scope.get("rejected_candidate_is_not_no_seizure") is not True
        ):
            raise ValueError("selected EEG signal-eligibility receipt drifted")


def _validate_technical_report(
    value: object,
    *,
    inventory_record: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping) or set(value) != _TECHNICAL_REPORT_KEYS:
        raise ValueError("selected technical report has missing or unknown keys")
    if value["schema_version"] != batch.TECHNICAL_REPORT_SCHEMA_VERSION:
        raise ValueError("selected technical report schema drifted")
    for key in ("status", "diagnostic_status"):
        if value[key] != TECHNICAL_STATUS:
            raise ValueError("selected technical report is not completed")
    for key in ("recording_id", "patient_pseudonym"):
        if value[key] != inventory_record[key]:
            raise ValueError("selected technical report identity drifted")
    if value["event_count"] != 0 or value["failure_stage"] != row["failure_stage"]:
        raise ValueError("selected technical report status drifted")
    fingerprint = value["technical_failure_receipt_fingerprint"]
    if not isinstance(fingerprint, str) or _SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError("selected technical report fingerprint is invalid")
    _validate_artifact_hashes(value["artifacts"], "selected technical report")


def _verify_artifact(
    *,
    source_root: Path,
    source_name: str,
    inventory_record: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recording_id = str(inventory_record["recording_id"])
    state_relative = PurePosixPath("records") / recording_id / "state.json"
    state = _manifest_snapshot(source_root, state_relative)
    _validate_state(state["value"], inventory_record=inventory_record, row=row)
    diagnostic = row["diagnostic_status"]
    if diagnostic in EEG_STATUSES:
        kind = "eeg_report"
        report_relative = (
            PurePosixPath("records") / recording_id / "report" / "manifest.json"
        )
        report = _manifest_snapshot(source_root, report_relative)
        _validate_eeg_report(
            report["value"], inventory_record=inventory_record, row=row
        )
    elif diagnostic == TECHNICAL_STATUS:
        kind = "technical_unassessable_report"
        technical_dir = _safe_relative(
            row["technical_artifact_relative_dir"], "technical artifact directory"
        )
        report_relative = (
            PurePosixPath("records")
            / recording_id
            / technical_dir
            / "manifest.json"
        )
        report = _manifest_snapshot(source_root, report_relative)
        _validate_technical_report(
            report["value"], inventory_record=inventory_record, row=row
        )
    else:
        raise ValueError("cannot verify an incomplete artifact")
    receipt = {
        "artifact_source": source_name,
        "effective_report_kind": kind,
        "state_manifest_relative_path": state_relative.as_posix(),
        "state_manifest_sha256": state["sha256"],
        "report_manifest_relative_path": report_relative.as_posix(),
        "report_manifest_sha256": report["sha256"],
    }
    return receipt, [state, report]


def _combined_subjects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["patient_pseudonym"]), []).append(row)
    subjects: list[dict[str, Any]] = []
    for patient, patient_rows in sorted(grouped.items()):
        eeg = sum(row["diagnostic_status"] in EEG_STATUSES for row in patient_rows)
        technical = sum(
            row["diagnostic_status"] == TECHNICAL_STATUS for row in patient_rows
        )
        artifacts = eeg + technical
        subjects.append(
            {
                "patient_pseudonym": patient,
                "expected_record_count": len(patient_rows),
                "completed_report_artifact_count": artifacts,
                "completed_eeg_report_count": eeg,
                "technical_unassessable_report_count": technical,
                "technical_failure_count": sum(
                    row["failure_stage"] is not None for row in patient_rows
                ),
                "recovery_overlay_count": sum(
                    bool(row["recovery_overlay_applied"]) for row in patient_rows
                ),
                "has_at_least_one_report_artifact": artifacts > 0,
                "has_at_least_one_completed_eeg_report": eeg > 0,
                "coverage_complete": artifacts == len(patient_rows),
                "eeg_coverage_complete": eeg == len(patient_rows),
            }
        )
    return subjects


def build_recovery_overlay(
    *,
    inventory_path: Path,
    primary_coverage_path: Path,
    recovery_coverage_path: Path,
    output_root: Path,
    primary_release_audit_path: Path | None = None,
    remediation_coverage_path: Path | None = None,
    remediation_release_audit_path: Path | None = None,
    expected_record_count: int | None = None,
    expected_subject_count: int | None = None,
) -> dict[str, Any]:
    """Verify source batches and publish one immutable combined manifest."""

    inventory_snapshot = _snapshot(inventory_path)
    inventory = batch.validate_inventory(inventory_snapshot["value"])
    if inventory["recording_unit_policy"] != "unique_signal_sha256_v1":
        raise ValueError("recovery overlay requires unique_signal_sha256_v1")
    if inventory["source_rejections"]:
        raise ValueError("recovery overlay requires a rejection-free inventory")
    if any(
        record["inventory_validation_status"] != batch.READY
        for record in inventory["records"]
    ):
        raise ValueError("recovery overlay requires every inventory record to be ready")
    signal_hashes = [record["source_signal_sha256"] for record in inventory["records"]]
    if any(not isinstance(digest, str) for digest in signal_hashes) or len(
        set(signal_hashes)
    ) != len(signal_hashes):
        raise ValueError("inventory does not contain one unit per unique EEG signal")
    if expected_record_count is not None and inventory["record_count"] != (
        expected_record_count
    ):
        raise ValueError("inventory record count differs from the required cohort size")
    if expected_subject_count is not None and inventory["subject_count"] != (
        expected_subject_count
    ):
        raise ValueError("inventory subject count differs from the required cohort size")

    primary_snapshot = _snapshot(primary_coverage_path)
    recovery_snapshot = _snapshot(recovery_coverage_path)
    primary_root = Path(primary_snapshot["path"]).parent.resolve(strict=True)
    recovery_root = Path(recovery_snapshot["path"]).parent.resolve(strict=True)
    if primary_root == recovery_root:
        raise ValueError("primary and recovery coverage must come from separate roots")
    remediation_arguments = (
        primary_release_audit_path,
        remediation_coverage_path,
        remediation_release_audit_path,
    )
    if any(value is not None for value in remediation_arguments) and not all(
        value is not None for value in remediation_arguments
    ):
        raise ValueError(
            "primary audit, remediation coverage, and remediation audit must be "
            "provided together"
        )
    remediation_enabled = all(value is not None for value in remediation_arguments)
    primary_audit_snapshot: dict[str, Any] | None = None
    remediation_snapshot: dict[str, Any] | None = None
    remediation_audit_snapshot: dict[str, Any] | None = None
    remediation_root: Path | None = None
    if remediation_enabled:
        assert primary_release_audit_path is not None
        assert remediation_coverage_path is not None
        assert remediation_release_audit_path is not None
        primary_audit_snapshot = _snapshot(primary_release_audit_path)
        remediation_snapshot = _snapshot(remediation_coverage_path)
        remediation_audit_snapshot = _snapshot(remediation_release_audit_path)
        remediation_root = Path(remediation_snapshot["path"]).parent.resolve(
            strict=True
        )
        if remediation_root in {primary_root, recovery_root}:
            raise ValueError("remediation coverage must have an independent root")
    if output_root.is_symlink():
        raise ValueError("combined output root must not be a symlink")
    output = output_root.resolve()
    source_roots = [primary_root, recovery_root]
    if remediation_root is not None:
        source_roots.append(remediation_root)
    for source_root in source_roots:
        if output == source_root or output.is_relative_to(source_root) or (
            source_root.is_relative_to(output)
        ):
            raise ValueError("combined output root must be independent of source roots")
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir() or output.is_symlink():
        raise ValueError("combined output root must be a regular directory")
    os.chmod(output, 0o700)
    output_path = output / "combined_coverage_manifest.json"
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)

    primary = _validate_coverage(
        primary_snapshot["value"], inventory=inventory, source_name="primary"
    )
    recovery = _validate_coverage(
        recovery_snapshot["value"], inventory=inventory, source_name="recovery"
    )
    primary_rows = {row["recording_id"]: row for row in primary["records"]}
    recovery_rows = {row["recording_id"]: row for row in recovery["records"]}
    remediation: dict[str, Any] | None = None
    remediation_rows: dict[str, dict[str, Any]] = {}
    eligible_authorizations: list[dict[str, Any]] = []
    authorization_by_id: dict[str, dict[str, Any]] = {}
    remediation_audit_results: dict[str, dict[str, Any]] = {}
    primary_release_audit: dict[str, Any] | None = None
    all_authorizations: list[dict[str, Any]] = []
    if remediation_enabled:
        assert primary_audit_snapshot is not None
        assert remediation_snapshot is not None
        assert remediation_audit_snapshot is not None
        primary_release_audit, all_authorizations, eligible_authorizations = (
            _validate_primary_release_authorization(
                primary_audit_snapshot["value"],
                inventory=inventory,
                inventory_manifest_sha256=inventory_snapshot["sha256"],
                primary_coverage_manifest_sha256=primary_snapshot["sha256"],
                primary_coverage=primary,
            )
        )
        authorization_by_id = {
            str(item["recording_id"]): item for item in eligible_authorizations
        }
        for recording_id in authorization_by_id:
            if primary_rows[recording_id]["diagnostic_status"] not in EEG_STATUSES:
                raise ValueError(
                    "primary audit authorized a non-EEG artifact for rerendering"
                )
        remediation = _validate_coverage(
            remediation_snapshot["value"],
            inventory=inventory,
            source_name="remediation",
        )
        remediation_rows = {
            row["recording_id"]: row for row in remediation["records"]
        }
        remediation_audit_results = _validate_remediation_release_audit(
            remediation_audit_snapshot["value"],
            inventory=inventory,
            inventory_manifest_sha256=inventory_snapshot["sha256"],
            remediation_coverage_manifest_sha256=remediation_snapshot["sha256"],
            primary_release_audit=primary_release_audit,
            primary_release_audit_sha256=primary_audit_snapshot["sha256"],
            all_authorizations=all_authorizations,
            eligible_authorizations=eligible_authorizations,
            primary_rows=primary_rows,
            remediation_rows=remediation_rows,
        )
    artifact_snapshots: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    technical_recovery_count = 0
    renderer_remediation_count = 0
    successful_recovery_eeg_count = sum(
        row["diagnostic_status"] in EEG_STATUSES for row in recovery["records"]
    )

    for inventory_record, primary_row in zip(
        inventory["records"], primary["records"], strict=True
    ):
        recording_id = str(inventory_record["recording_id"])
        recovery_row = recovery_rows[recording_id]
        technical_overlay = (
            primary_row["diagnostic_status"] == TECHNICAL_STATUS
            and recovery_row["diagnostic_status"] in EEG_STATUSES
        )
        renderer_overlay = recording_id in authorization_by_id
        if technical_overlay and renderer_overlay:
            raise ValueError("one primary artifact received conflicting replacement roles")
        overlay_applied = technical_overlay or renderer_overlay
        superseded_receipt: dict[str, Any] | None = None
        if overlay_applied:
            superseded_receipt, snapshots = _verify_artifact(
                source_root=primary_root,
                source_name="primary",
                inventory_record=inventory_record,
                row=primary_row,
            )
            artifact_snapshots.extend(snapshots)
        if technical_overlay:
            selected_row = recovery_row
            selected_source = "recovery"
            selected_root = recovery_root
            decision = "recovery_eeg_replaced_primary_technical"
            technical_recovery_count += 1
        elif renderer_overlay:
            assert remediation_root is not None
            selected_row = remediation_rows[recording_id]
            selected_source = "remediation"
            selected_root = remediation_root
            decision = "authorized_remediation_eeg_replaced_audit_failed_primary_eeg"
            authorization = authorization_by_id[recording_id]
            if (
                superseded_receipt is None
                or superseded_receipt["effective_report_kind"] != "eeg_report"
                or superseded_receipt["report_manifest_sha256"]
                != authorization["source_report_manifest_sha256"]
            ):
                raise ValueError(
                    "primary report manifest differs from replacement authorization"
                )
            renderer_remediation_count += 1
        else:
            selected_row = primary_row
            selected_source = "primary"
            selected_root = primary_root
            if primary_row["diagnostic_status"] in EEG_STATUSES:
                decision = "primary_eeg_retained"
            elif primary_row["diagnostic_status"] == TECHNICAL_STATUS:
                decision = "primary_technical_retained_recovery_not_successful_eeg"
            else:
                decision = "primary_incomplete_retained"

        if selected_row["diagnostic_status"] in ALL_COMPLETED_STATUSES:
            artifact_receipt, snapshots = _verify_artifact(
                source_root=selected_root,
                source_name=selected_source,
                inventory_record=inventory_record,
                row=selected_row,
            )
            artifact_snapshots.extend(snapshots)
            if renderer_overlay:
                audit_result = remediation_audit_results[recording_id]
                if artifact_receipt["report_manifest_sha256"] != audit_result[
                    "remediated_report_manifest_sha256"
                ]:
                    raise ValueError(
                        "fresh report manifest differs from remediation release audit"
                    )
        else:
            artifact_receipt = {
                "artifact_source": None,
                "effective_report_kind": "none",
                "state_manifest_relative_path": None,
                "state_manifest_sha256": None,
                "report_manifest_relative_path": None,
                "report_manifest_sha256": None,
            }
        combined_rows.append(
            {
                "recording_id": inventory_record["recording_id"],
                "patient_pseudonym": inventory_record["patient_pseudonym"],
                "inventory_validation_status": inventory_record[
                    "inventory_validation_status"
                ],
                "run_status": selected_row["run_status"],
                "diagnostic_status": selected_row["diagnostic_status"],
                "event_count": selected_row["event_count"],
                "failure_stage": selected_row["failure_stage"],
                "existing_success_reused": selected_row[
                    "existing_success_reused"
                ],
                "technical_artifact_relative_dir": selected_row[
                    "technical_artifact_relative_dir"
                ],
                **artifact_receipt,
                "source_coverage_row_sha256": _canonical_sha256(selected_row),
                "recovery_overlay_applied": overlay_applied,
                "overlay_decision": decision,
                "superseded_primary_artifact_receipt": superseded_receipt,
            }
        )

    completed_eeg = sum(
        row["diagnostic_status"] in EEG_STATUSES for row in combined_rows
    )
    technical = sum(
        row["diagnostic_status"] == TECHNICAL_STATUS for row in combined_rows
    )
    failures = sum(row["failure_stage"] is not None for row in combined_rows)
    artifacts = completed_eeg + technical
    if artifacts != inventory["record_count"]:
        raise ValueError(
            "combined overlay does not cover every unique inventory signal with a report"
        )
    subjects = _combined_subjects(combined_rows)
    if len(subjects) != inventory["subject_count"] or not all(
        subject["coverage_complete"] for subject in subjects
    ):
        raise ValueError("combined overlay does not cover every inventory subject")

    body = {
        "schema_version": SCHEMA_VERSION,
        "inventory_id": inventory["inventory_id"],
        "recording_unit_policy": inventory["recording_unit_policy"],
        "mode": "recovery_overlay",
        "expected_record_count": inventory["record_count"],
        "expected_subject_count": inventory["subject_count"],
        "inventory_rejection_count": 0,
        "unique_signal_count": len(signal_hashes),
        "completed_report_count": artifacts,
        "completed_report_artifact_count": artifacts,
        "completed_eeg_report_count": completed_eeg,
        "technical_unassessable_report_count": technical,
        "technical_failure_count": failures,
        "pending_or_not_run_count": inventory["record_count"] - artifacts,
        "dataset_coverage_complete": True,
        "dataset_artifact_coverage_complete": True,
        "dataset_eeg_coverage_complete": completed_eeg == inventory["record_count"],
        "diagnostic_status_counts": {
            status: sum(row["diagnostic_status"] == status for row in combined_rows)
            for status in sorted(ALL_COMPLETED_STATUSES)
        },
        "overlay_counts": {
            "primary_technical_shell_count": primary[
                "technical_unassessable_report_count"
            ],
            "successful_recovery_eeg_report_count": successful_recovery_eeg_count,
            "recovery_eeg_replaced_primary_technical_count": technical_recovery_count,
            "successful_recovery_eeg_not_applied_count": (
                successful_recovery_eeg_count - technical_recovery_count
            ),
            "authorized_primary_eeg_renderer_remediation_count": len(
                eligible_authorizations
            ),
            "audited_remediation_eeg_replaced_primary_eeg_count": (
                renderer_remediation_count
            ),
            "authorized_primary_eeg_not_remediated_count": (
                len(eligible_authorizations) - renderer_remediation_count
            ),
            "total_effective_overlay_count": (
                technical_recovery_count + renderer_remediation_count
            ),
            "primary_technical_shell_remaining_count": technical,
        },
        "records": combined_rows,
        "subjects": subjects,
        "source_manifest_receipts": {
            "inventory_manifest_sha256": inventory_snapshot["sha256"],
            "primary_coverage_manifest_sha256": primary_snapshot["sha256"],
            "recovery_coverage_manifest_sha256": recovery_snapshot["sha256"],
            "primary_release_audit_sha256": (
                None
                if primary_audit_snapshot is None
                else primary_audit_snapshot["sha256"]
            ),
            "primary_replacement_authorization_id": (
                None
                if primary_release_audit is None
                else primary_release_audit["replacement_authorization_receipt"][
                    "authorization_id"
                ]
            ),
            "remediation_coverage_manifest_sha256": (
                None if remediation_snapshot is None else remediation_snapshot["sha256"]
            ),
            "remediation_release_audit_sha256": (
                None
                if remediation_audit_snapshot is None
                else remediation_audit_snapshot["sha256"]
            ),
            "source_paths_persisted": False,
            "selected_state_and_report_manifests_hash_bound": True,
        },
        "overlay_policy_receipt": {
            "policy_id": (
                "technical_recovery_plus_explicit_audited_renderer_remediation_v2"
            ),
            "recovery_technical_replaces_primary": False,
            "recovery_pending_replaces_primary": False,
            "recovery_eeg_replaces_primary_eeg": False,
            "successful_recovery_eeg_replaces_primary_technical": True,
            "unaudited_remediation_eeg_replaces_primary_eeg": False,
            "audit_passed_primary_eeg_may_be_replaced": False,
            "authorized_audit_failed_primary_eeg_may_be_replaced": True,
            "renderer_remediation_may_change_diagnostic_status_or_event_count": False,
            "remediation_technical_or_pending_replaces_primary_eeg": False,
            "fresh_remediation_release_audit_pass_required": True,
            "source_artifacts_copied": False,
            "primary_artifacts_deleted_or_overwritten": False,
            "recovery_artifacts_deleted_or_overwritten": False,
            "remediation_artifacts_deleted_or_overwritten": False,
        },
        "scope_receipt": {
            "inventory_manifest_read": True,
            "primary_and_recovery_coverage_manifests_read": True,
            "primary_and_remediation_release_audits_read": remediation_enabled,
            "remediation_coverage_manifest_read": remediation_enabled,
            "selected_state_and_report_manifests_read": True,
            "edf_signal_files_read": False,
            "edf_annotations_read": False,
            "excel_or_workbook_read": False,
            "onset_label_or_ground_truth_read": False,
            "report_bodies_read": False,
            "waveform_artifacts_read": False,
            "raw_edf_paths_persisted": False,
            "inventory_edf_locator_values_used_for_artifact_selection": False,
            "source_files_resolved_from_inventory_edf_paths": False,
            "source_output_trees_modified": False,
            "combined_manifest_only": True,
            "report_body_artifact_hashes_reverified": False,
            "source_manifest_snapshot_consistency_verified": True,
        },
    }
    combined = {
        **body,
        "combined_coverage_id": "PLCOMB-" + _canonical_sha256(body)[:24],
    }
    source_snapshots = [inventory_snapshot, primary_snapshot, recovery_snapshot]
    for optional_snapshot in (
        primary_audit_snapshot,
        remediation_snapshot,
        remediation_audit_snapshot,
    ):
        if optional_snapshot is not None:
            source_snapshots.append(optional_snapshot)
    _assert_snapshots_unchanged([*source_snapshots, *artifact_snapshots])
    batch._atomic_json(output_path, combined, replace=False)
    return combined


def _positive_or_none(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected count must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--primary-coverage", type=Path, required=True)
    parser.add_argument("--recovery-coverage", type=Path, required=True)
    parser.add_argument("--primary-release-audit", type=Path)
    parser.add_argument("--remediation-coverage", type=Path)
    parser.add_argument("--remediation-release-audit", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expect-records", type=_positive_or_none)
    parser.add_argument("--expect-subjects", type=_positive_or_none)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    combined = build_recovery_overlay(
        inventory_path=args.inventory,
        primary_coverage_path=args.primary_coverage,
        recovery_coverage_path=args.recovery_coverage,
        primary_release_audit_path=args.primary_release_audit,
        remediation_coverage_path=args.remediation_coverage,
        remediation_release_audit_path=args.remediation_release_audit,
        output_root=args.output_root,
        expected_record_count=args.expect_records,
        expected_subject_count=args.expect_subjects,
    )
    print(
        json.dumps(
            {
                "combined_coverage_id": combined["combined_coverage_id"],
                "expected_record_count": combined["expected_record_count"],
                "expected_subject_count": combined["expected_subject_count"],
                "completed_eeg_report_count": combined[
                    "completed_eeg_report_count"
                ],
                "technical_unassessable_report_count": combined[
                    "technical_unassessable_report_count"
                ],
                "recovery_overlay_count": combined["overlay_counts"][
                    "recovery_eeg_replaced_primary_technical_count"
                ],
                "renderer_remediation_overlay_count": combined["overlay_counts"][
                    "audited_remediation_eeg_replaced_primary_eeg_count"
                ],
                "edf_annotation_excel_onset_label_or_gt_read": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "build_recovery_overlay", "main"]
