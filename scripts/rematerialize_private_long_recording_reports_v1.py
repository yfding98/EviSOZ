#!/usr/bin/env python3
"""Rematerialize only release-audit-authorized private EEG reports.

This command is a report-layer remediation, not an EEG inference rerun.  It
accepts only ``rerender_report_only`` authorizations from a canonical release
audit, revalidates the primary report's signal-only bundle, detection manifest,
event receipts and waveform hashes, and invokes the current long-recording
report materializer with a fresh Qwen request.  It never opens EDF files,
annotations, spreadsheets, physician labels, or the primary language/HTML/
DOCX artifacts.

The primary tree is read-only.  A successful run atomically publishes an
independent, self-contained batch root with per-record report/state artifacts,
a full-inventory recovery coverage manifest, and a de-identified remediation
manifest.  Non-authorized and audit-passed primary EEG reports are represented
as pending rows and are never copied or overwritten.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_private_long_recording_report_release_v1 as release  # noqa: E402
from scripts import combine_private_long_recording_report_recovery_v1 as overlay  # noqa: E402
from scripts import materialize_private_long_recording_reports_v1 as batch  # noqa: E402
from src.clinical_eeg_long_recording.aggregation import (  # noqa: E402
    validate_trustworthy_long_term_clinical_eeg_bundle,
)
from src.clinical_eeg_long_recording.pipeline import (  # noqa: E402
    materialize_long_term_clinical_eeg_report,
)
from src.clinical_eeg_long_recording.report_outcome import (  # noqa: E402
    classify_recording_eeg_outcome,
)
from src.clinical_eeg_long_recording.schema import (  # noqa: E402
    validate_long_term_event_segment_receipt,
)


SCHEMA_VERSION = "private_long_recording_report_rematerialization_v1"
STATUS = "completed_authorized_report_rematerialization"
_STATE_SCOPE = {
    "eeg_signal_only": True,
    "edf_annotations_loaded": False,
    "excel_loaded": False,
    "onset_or_ground_truth_loaded": False,
}
_REPORT_SCOPE_REQUIRED = {
    "entire_record_detection_manifest_validated": True,
    "three_timebase_closure_verified": True,
    "eeg_signal_only_generation": True,
    "eeg_facts_and_automatic_impression_signal_only": True,
    "external_edf_annotations_loaded": False,
    "excel_observations_loaded": False,
    "source_context_joined_post_freeze": False,
    "source_context_sent_to_qwen": False,
    "research_soz_used_in_clinical_facts_or_llm": False,
    "sleep_activation_ecg_emg_or_demographics_generated": False,
    "all_waveforms_hash_verified": True,
    "physician_signed": False,
}
_SOURCE_ARTIFACTS = frozenset(
    {
        "bundle.json",
        "detection_manifest.json",
        "event_segment_receipts.json",
    }
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _strict_snapshot(path: Path) -> tuple[dict[str, Any], tuple[Path, str]]:
    value, snapshot = release._read_snapshot(path)  # noqa: SLF001
    if not isinstance(value, Mapping):
        raise TypeError("manifest input must be a JSON object")
    return dict(value), snapshot


def _atomic_json(path: Path, value: object) -> None:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, target)
        os.chmod(target, 0o600)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _validate_audit(
    value: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
    coverage_sha256: str,
    expected_authorizations: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    if value.get("schema_version") != release.SCHEMA_VERSION:
        raise ValueError("release audit schema drifted")
    audit_id = value.get("audit_id")
    body = {key: item for key, item in value.items() if key != "audit_id"}
    if audit_id != release.AUDIT_ID_PREFIX + _canonical_sha256(body)[:24]:
        raise ValueError("release audit ID does not bind its content")
    if value.get("coverage_kind") != "full":
        raise ValueError("rematerialization requires a full primary audit")
    if value.get("audit_mode") != release.COHORT_AUDIT_MODE:
        raise ValueError("rematerialization requires a primary cohort audit")
    if value.get("status") != release.FAIL_STATUS or value.get("release_ready") is not False:
        raise ValueError("an audit-passed primary cannot authorize rematerialization")
    if value.get("inventory_id") != inventory["inventory_id"]:
        raise ValueError("release audit inventory binding drifted")
    sources = value.get("source_receipts")
    if not isinstance(sources, Mapping) or sources.get(
        "inventory_manifest_sha256"
    ) != inventory_sha256 or sources.get("coverage_manifest_sha256") != coverage_sha256:
        raise ValueError("release audit source snapshot binding drifted")
    receipt = value.get("replacement_authorization_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != (
        release.REPLACEMENT_AUTHORIZATION_SCHEMA
    ):
        raise ValueError("replacement authorization receipt drifted")
    authorization_id = receipt.get("authorization_id")
    authorization_body = {
        key: item for key, item in receipt.items() if key != "authorization_id"
    }
    if authorization_id != "PLRAUTH-" + _canonical_sha256(authorization_body)[:24]:
        raise ValueError("replacement authorization ID drifted")
    if receipt.get("authorization_policy_id") != release.AUTHORIZATION_POLICY_ID:
        raise ValueError("replacement authorization policy drifted")
    if receipt.get("eligible_for_primary_replacement") is not True:
        raise ValueError("release audit does not authorize primary replacement")
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
        constraints.get(key) != expected for key, expected in required_constraints.items()
    ):
        raise ValueError("replacement authorization constraints drifted")
    raw = receipt.get("authorizations")
    if not isinstance(raw, list) or receipt.get(
        "authorized_failed_primary_eeg_count"
    ) != len(raw):
        raise ValueError("replacement authorization count drifted")
    failed_reports = value.get("failed_reports")
    if raw != failed_reports:
        raise ValueError("replacement authorization does not equal failed reports")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    inventory_ids = {str(item["recording_id"]) for item in inventory["records"]}
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("replacement authorization must be an object")
        recording_id = item.get("recording_id")
        manifest_sha = item.get("source_report_manifest_sha256")
        reasons = item.get("failure_reason_codes")
        if (
            not isinstance(recording_id, str)
            or recording_id not in inventory_ids
            or recording_id in seen
            or item.get("selected_artifact_source") != "primary"
            or not isinstance(manifest_sha, str)
            or release._SHA256_RE.fullmatch(manifest_sha) is None  # noqa: SLF001
            or not isinstance(reasons, list)
            or not reasons
        ):
            raise ValueError("replacement authorization row drifted")
        seen.add(recording_id)
        if (
            item.get("failure_layer") == "language_or_renderer_projection"
            and item.get("minimum_remediation") == "rerender_report_only"
            and all(
                reason
                in {
                    "language_projection_invalid",
                    "document_artifact_invalid",
                    "unauthorized_positive_clinical_language",
                }
                for reason in reasons
            )
        ):
            result.append(dict(item))
    if expected_authorizations is not None and len(result) != expected_authorizations:
        raise ValueError("eligible rematerialization count differs from expectation")
    return str(audit_id), sorted(result, key=lambda item: item["recording_id"])


def _report_manifest(
    *,
    primary_root: Path,
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    authorization: Mapping[str, Any],
    snapshots: list[tuple[Path, str]],
) -> tuple[Path, dict[str, Any]]:
    relative = (
        PurePosixPath("records")
        / str(record["recording_id"])
        / "report"
        / "manifest.json"
    )
    path = release._resolve_regular(primary_root, relative)  # noqa: SLF001
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    snapshots.append((path, digest))
    if digest != authorization["source_report_manifest_sha256"]:
        raise ValueError("authorized source report manifest changed")
    manifest = release._strict_json_bytes(raw)  # noqa: SLF001
    overlay._validate_eeg_report(  # noqa: SLF001
        manifest,
        inventory_record=record,
        row=row,
    )
    assert isinstance(manifest, Mapping)
    scope = manifest.get("scope_receipt")
    if not isinstance(scope, Mapping) or any(
        scope.get(key) != expected for key, expected in _REPORT_SCOPE_REQUIRED.items()
    ):
        raise ValueError("authorized source report violates the EEG-only boundary")
    return path.parent, dict(manifest)


def _source_artifact(
    *,
    report_root: Path,
    manifest: Mapping[str, Any],
    relative_text: str,
    snapshots: list[tuple[Path, str]],
) -> tuple[Path, bytes]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or relative_text not in artifacts:
        raise ValueError("authorized source report lacks a required artifact")
    path = release._resolve_regular(  # noqa: SLF001
        report_root,
        release._safe_relative(relative_text, "source report artifact"),  # noqa: SLF001
    )
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    snapshots.append((path, digest))
    if digest != artifacts[relative_text]:
        raise ValueError("authorized source artifact hash drifted")
    return path, raw


def _validated_source_inputs(
    *,
    primary_root: Path,
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    authorization: Mapping[str, Any],
    snapshots: list[tuple[Path, str]],
) -> dict[str, Any]:
    report_root, manifest = _report_manifest(
        primary_root=primary_root,
        record=record,
        row=row,
        authorization=authorization,
        snapshots=snapshots,
    )
    raw_by_name: dict[str, bytes] = {}
    source_artifacts = set(_SOURCE_ARTIFACTS)
    if "analysis_selection_manifest.json" in manifest.get("artifacts", {}):
        source_artifacts.add("analysis_selection_manifest.json")
    for name in source_artifacts:
        _, raw_by_name[name] = _source_artifact(
            report_root=report_root,
            manifest=manifest,
            relative_text=name,
            snapshots=snapshots,
        )
    bundle_value = release._strict_json_bytes(raw_by_name["bundle.json"])  # noqa: SLF001
    bundle = validate_trustworthy_long_term_clinical_eeg_bundle(bundle_value)
    release._current_bundle_semantics(bundle)  # noqa: SLF001
    if (
        bundle["recording_id"] != record["recording_id"]
        or bundle["patient_pseudonym"] != record["patient_pseudonym"]
        or bundle["event_count"] != row["event_count"]
        or classify_recording_eeg_outcome(bundle) != manifest["diagnostic_outcome"]
    ):
        raise ValueError("authorized source bundle identity or outcome drifted")
    detection = release._strict_json_bytes(  # noqa: SLF001
        raw_by_name["detection_manifest.json"]
    )
    if detection != bundle["detection_manifest"]:
        raise ValueError("authorized detection artifact differs from bundle")
    analysis_selection: dict[str, Any] | None = None
    if "analysis_selection_manifest.json" in raw_by_name:
        selection_value = release._strict_json_bytes(  # noqa: SLF001
            raw_by_name["analysis_selection_manifest.json"]
        )
        if selection_value != bundle.get("analysis_selection"):
            raise ValueError("authorized analysis selection differs from bundle")
        analysis_selection = dict(selection_value)
    segments_value = release._strict_json_bytes(  # noqa: SLF001
        raw_by_name["event_segment_receipts.json"]
    )
    if not isinstance(segments_value, list) or len(segments_value) != bundle[
        "event_count"
    ]:
        raise ValueError("authorized event receipt count drifted")
    segments = [
        validate_long_term_event_segment_receipt(item) for item in segments_value
    ]
    if [item["eeg_event_id"] for item in segments] != [
        item["eeg_event_id"] for item in bundle["events"]
    ]:
        raise ValueError("authorized event receipt order drifted")
    waveform_sources: list[tuple[str, bytes]] = []
    for index, event in enumerate(bundle["events"], start=1):
        relative = f"waveforms/eeg_waveform_{index:02d}.png"
        path, raw = _source_artifact(
            report_root=report_root,
            manifest=manifest,
            relative_text=relative,
            snapshots=snapshots,
        )
        if (
            event["waveform_attachment"]["figure_file"] != relative
            or _sha256_bytes(raw) != event["waveform_attachment"]["figure_sha256"]
        ):
            raise ValueError("authorized waveform binding drifted")
        release._verify_png(raw)  # noqa: SLF001
        path.relative_to(report_root)
        segment_relative = segments[index - 1]["waveform_attachment"]["figure_file"]
        release._safe_relative(segment_relative, "source segment waveform")  # noqa: SLF001
        waveform_sources.append((str(segment_relative), raw))
    return {
        "bundle_id": bundle["bundle_id"],
        "detection": detection,
        "segments": segments,
        "analysis_selection": analysis_selection,
        "waveform_sources": waveform_sources,
    }


def _state(
    record: Mapping[str, Any],
    *,
    diagnostic_status: str,
    event_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": batch.STATE_SCHEMA_VERSION,
        "recording_id": record["recording_id"],
        "patient_pseudonym": record["patient_pseudonym"],
        "status": "completed",
        "last_completed_stage": "report_materialization",
        "diagnostic_status": diagnostic_status,
        "event_count": event_count,
        "attempt": 1,
        "scope_receipt": dict(_STATE_SCOPE),
    }


def _coverage_row(
    record: Mapping[str, Any],
    *,
    diagnostic_status: str | None = None,
    event_count: int | None = None,
) -> dict[str, Any]:
    completed = diagnostic_status is not None
    return {
        "recording_id": record["recording_id"],
        "patient_pseudonym": record["patient_pseudonym"],
        "inventory_validation_status": record["inventory_validation_status"],
        "run_status": "completed" if completed else "not_selected_in_this_run",
        "diagnostic_status": diagnostic_status,
        "event_count": event_count,
        "failure_stage": None,
        "existing_success_reused": False,
        "technical_artifact_relative_dir": None,
    }


def _coverage_subjects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["patient_pseudonym"]), []).append(row)
    result: list[dict[str, Any]] = []
    for patient, patient_rows in sorted(grouped.items()):
        completed = sum(
            row["diagnostic_status"] in batch.COMPLETED_DIAGNOSTIC_STATUSES
            for row in patient_rows
        )
        result.append(
            {
                "patient_pseudonym": patient,
                "expected_record_count": len(patient_rows),
                "completed_report_artifact_count": completed,
                "completed_eeg_report_count": completed,
                "technical_unassessable_report_count": 0,
                "technical_failure_count": 0,
                "has_at_least_one_report_artifact": completed > 0,
                "has_at_least_one_completed_eeg_report": completed > 0,
                "coverage_complete": completed == len(patient_rows),
                "eeg_coverage_complete": completed == len(patient_rows),
            }
        )
    return result


def _coverage(
    *,
    inventory: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    qwen_requested: bool,
) -> dict[str, Any]:
    completed = sum(
        row["diagnostic_status"] in batch.COMPLETED_DIAGNOSTIC_STATUSES
        for row in rows
    )
    return {
        "schema_version": batch.COVERAGE_SCHEMA_VERSION,
        "inventory_id": inventory["inventory_id"],
        "recording_unit_policy": inventory["recording_unit_policy"],
        "mode": "execution",
        "expected_record_count": inventory["record_count"],
        "expected_subject_count": inventory["subject_count"],
        "inventory_rejection_count": len(inventory["source_rejections"]),
        "completed_report_count": completed,
        "completed_report_artifact_count": completed,
        "completed_eeg_report_count": completed,
        "technical_unassessable_report_count": 0,
        "technical_failure_count": 0,
        "pending_or_not_run_count": len(rows) - completed,
        "dataset_coverage_complete": (
            completed == inventory["record_count"]
            and not inventory["source_rejections"]
        ),
        "dataset_artifact_coverage_complete": (
            completed == inventory["record_count"]
            and not inventory["source_rejections"]
        ),
        "dataset_eeg_coverage_complete": (
            completed == inventory["record_count"]
            and not inventory["source_rejections"]
        ),
        "diagnostic_status_counts": {
            status: sum(row["diagnostic_status"] == status for row in rows)
            for status in sorted(
                {
                    *batch.COMPLETED_DIAGNOSTIC_STATUSES,
                    "completed_technical_unassessable",
                }
            )
        },
        "records": [dict(row) for row in rows],
        "subjects": _coverage_subjects(rows),
        "scope_receipt": {
            "one_report_unit_per_inventory_recording_unit": True,
            "recording_unit_policy": inventory["recording_unit_policy"],
            "source_event_rows_deduplicated_before_inference": True,
            "generation_uses_eeg_signal_only": True,
            "edf_annotations_loaded": False,
            "excel_or_workbook_loaded": False,
            "onset_or_label_fields_forwarded": False,
            "ground_truth_forwarded": False,
            "qwen_optional": True,
            "qwen_requested": qwen_requested,
            "qwen_failure_blocks_report": False,
            "zero_candidates_still_materialize_report": True,
            "findings_insufficiency_is_completed_abstention": True,
            "technical_failure_is_not_eeg_insufficiency": True,
            "technical_failure_gets_non_diagnostic_report_shell": True,
            "raw_edf_paths_or_patient_names_in_coverage": False,
        },
    }


def rematerialize_authorized_reports(
    *,
    inventory_path: Path,
    primary_coverage_path: Path,
    primary_root: Path,
    release_audit_path: Path,
    output_root: Path,
    policy_path: Path,
    style_path: Path,
    base_url: str = "http://127.0.0.1:8000/v1",
    use_qwen: bool = True,
    expected_authorizations: int | None = None,
) -> dict[str, Any]:
    """Atomically publish a recovery batch for authorized report-only failures."""

    inventory_value, inventory_snapshot = _strict_snapshot(inventory_path)
    inventory = batch.validate_inventory(inventory_value)
    coverage_value, coverage_snapshot = _strict_snapshot(primary_coverage_path)
    primary_coverage = overlay._validate_coverage(  # noqa: SLF001
        coverage_value,
        inventory=inventory,
        source_name="primary",
    )
    root = release._regular_root(primary_root, "primary report root")  # noqa: SLF001
    if coverage_snapshot[0].parent != root:
        raise ValueError("primary coverage must be rooted in primary_root")
    audit_value, audit_snapshot = _strict_snapshot(release_audit_path)
    audit_id, authorizations = _validate_audit(
        audit_value,
        inventory=inventory,
        inventory_sha256=inventory_snapshot[1],
        coverage_sha256=coverage_snapshot[1],
        expected_authorizations=expected_authorizations,
    )
    if not authorizations:
        raise ValueError("release audit authorizes no report rematerialization")

    target = output_root.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if target == root or target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("remediation output root must be independent of primary root")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    os.chmod(staging, 0o700)
    published = False
    snapshots: list[tuple[Path, str]] = [
        inventory_snapshot,
        coverage_snapshot,
        audit_snapshot,
    ]
    inventory_by_id = {
        str(record["recording_id"]): record for record in inventory["records"]
    }
    primary_rows = {
        str(row["recording_id"]): row for row in primary_coverage["records"]
    }
    completed_rows: dict[str, dict[str, Any]] = {}
    report_receipts: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="source-json-", dir=staging) as temp_name:
            temp_source = Path(temp_name)
            for authorization in authorizations:
                recording_id = str(authorization["recording_id"])
                record = inventory_by_id[recording_id]
                row = primary_rows[recording_id]
                if row["diagnostic_status"] not in batch.COMPLETED_DIAGNOSTIC_STATUSES:
                    raise ValueError("authorized primary row is not a completed EEG report")
                source = _validated_source_inputs(
                    primary_root=root,
                    record=record,
                    row=row,
                    authorization=authorization,
                    snapshots=snapshots,
                )
                case_source = temp_source / recording_id
                detection_path = case_source / "detection_manifest.json"
                _write_json(detection_path, source["detection"])
                segment_paths: list[Path] = []
                for index, segment in enumerate(source["segments"], start=1):
                    path = case_source / f"event_segment_{index:02d}.json"
                    _write_json(path, segment)
                    segment_paths.append(path)
                analysis_selection_path: Path | None = None
                if source["analysis_selection"] is not None:
                    analysis_selection_path = case_source / "analysis_selection.json"
                    _write_json(analysis_selection_path, source["analysis_selection"])
                waveform_root = case_source / "waveform_sources"
                waveform_root.mkdir(parents=True, exist_ok=True)
                for relative, raw in source["waveform_sources"]:
                    waveform_path = waveform_root / relative
                    waveform_path.parent.mkdir(parents=True, exist_ok=True)
                    waveform_path.write_bytes(raw)
                    os.chmod(waveform_path, 0o600)
                case_root = staging / "records" / recording_id
                manifest = materialize_long_term_clinical_eeg_report(
                    detection_manifest_path=detection_path,
                    segment_receipt_paths=segment_paths,
                    waveform_root=waveform_root,
                    output_dir=case_root / "report",
                    bundle_id=str(source["bundle_id"]),
                    analysis_selection_path=analysis_selection_path,
                    policy_path=policy_path.resolve(strict=True),
                    style_path=style_path.resolve(strict=True),
                    source_context_path=None,
                    base_url=base_url,
                    use_qwen=use_qwen,
                )
                if manifest["diagnostic_status"] != row["diagnostic_status"]:
                    raise ValueError("report-only remediation changed diagnostic status")
                state = _state(
                    record,
                    diagnostic_status=str(manifest["diagnostic_status"]),
                    event_count=int(manifest["event_count"]),
                )
                _write_json(case_root / "state.json", state)
                completed_rows[recording_id] = _coverage_row(
                    record,
                    diagnostic_status=str(manifest["diagnostic_status"]),
                    event_count=int(manifest["event_count"]),
                )
                manifest_path = case_root / "report" / "manifest.json"
                report_receipts.append(
                    {
                        "recording_id": recording_id,
                        "source_report_manifest_sha256": authorization[
                            "source_report_manifest_sha256"
                        ],
                        "remediated_report_manifest_sha256": _sha256_file(
                            manifest_path
                        ),
                        "diagnostic_status": manifest["diagnostic_status"],
                        "event_count": manifest["event_count"],
                        "validated_qwen_wording_count": manifest[
                            "language_service_receipt"
                        ]["validated_qwen_wording_count"],
                        "deterministic_fallback_count": manifest[
                            "language_service_receipt"
                        ]["deterministic_fallback_count"],
                    }
                )

        rows = [
            completed_rows.get(str(record["recording_id"]), _coverage_row(record))
            for record in inventory["records"]
        ]
        coverage = _coverage(
            inventory=inventory,
            rows=rows,
            qwen_requested=use_qwen,
        )
        overlay._validate_coverage(  # noqa: SLF001
            coverage,
            inventory=inventory,
            source_name="remediation",
        )
        _write_json(staging / "coverage_manifest.json", coverage)
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "inventory_id": inventory["inventory_id"],
            "source_release_audit_id": audit_id,
            "authorized_record_count": len(authorizations),
            "rematerialized_record_count": len(report_receipts),
            "report_receipts": sorted(
                report_receipts, key=lambda item: item["recording_id"]
            ),
            "source_manifest_receipts": {
                "inventory_manifest_sha256": inventory_snapshot[1],
                "primary_coverage_manifest_sha256": coverage_snapshot[1],
                "release_audit_manifest_sha256": audit_snapshot[1],
                "source_paths_persisted": False,
            },
            "scope_receipt": {
                "edf_signal_files_read": False,
                "edf_annotations_read": False,
                "excel_or_workbook_read": False,
                "onset_label_or_ground_truth_read": False,
                "primary_bundle_detection_segments_and_waveforms_read": True,
                "primary_language_records_read": False,
                "primary_html_or_docx_read": False,
                "old_language_layer_copied": False,
                "qwen_requested_fresh": use_qwen,
                "primary_tree_modified": False,
                "output_is_independent_self_contained_batch": True,
            },
        }
        remediation = {
            **body,
            "rematerialization_id": "PLREMAT-" + _canonical_sha256(body)[:24],
        }
        _write_json(staging / "rematerialization_manifest.json", remediation)
        release._assert_unchanged(snapshots)  # noqa: SLF001
        for path in staging.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        os.replace(staging, target)
        os.chmod(target, 0o700)
        published = True
        return deepcopy(remediation)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _positive_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected authorization count must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--primary-coverage", type=Path, required=True)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--release-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs/clinical_eeg_report_v1.json",
    )
    parser.add_argument(
        "--style",
        type=Path,
        default=ROOT / "configs/clinical_eeg_report_style_zh_v1.json",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--expect-authorizations", type=_positive_count)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = rematerialize_authorized_reports(
        inventory_path=args.inventory,
        primary_coverage_path=args.primary_coverage,
        primary_root=args.primary_root,
        release_audit_path=args.release_audit,
        output_root=args.output_root,
        policy_path=args.policy,
        style_path=args.style,
        base_url=args.base_url,
        use_qwen=True,
        expected_authorizations=args.expect_authorizations,
    )
    print(
        json.dumps(
            {
                "rematerialization_id": result["rematerialization_id"],
                "status": result["status"],
                "authorized_record_count": result["authorized_record_count"],
                "rematerialized_record_count": result["rematerialized_record_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "rematerialize_authorized_reports", "main"]
