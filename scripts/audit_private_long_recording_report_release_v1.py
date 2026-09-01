#!/usr/bin/env python3
"""Read-only release audit for private long-recording EEG reports.

The audit consumes a frozen private inventory and either one execution
coverage manifest or one recovery-overlay combined coverage manifest.  It
opens only the selected, already materialized report artifacts.  EDF signals,
EDF annotations, spreadsheets, physician labels and source dataset locators
are never resolved or opened.

Every completed EEG report is revalidated at the current publication
boundary: report manifest and artifact hashes, the strict long-recording
bundle, the independently recomputed diagnostic outcome, the facts-locked
language projection, HTML/DOCX structure, and waveform attachment closure.
The current neutral long-recording producer contract is enforced in addition
to the historical generic report schema.  Report-level failures are emitted
only as aggregate error-code counts; record IDs, patient pseudonyms and source
paths are deliberately absent from the audit JSON.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import combine_private_long_recording_report_recovery_v1 as overlay  # noqa: E402
from scripts import materialize_private_long_recording_reports_v1 as batch  # noqa: E402
from src.clinical_eeg_long_recording.aggregation import (  # noqa: E402
    validate_trustworthy_long_term_clinical_eeg_bundle,
)
from src.clinical_eeg_long_recording.event_materialization import (  # noqa: E402
    _CURRENT_SIGNAL_FINDING_REQUIRED_VALUE_KEYS,
    _CURRENT_SIGNAL_FINDING_VALUE_KEYS,
)
from src.clinical_eeg_long_recording.pipeline import (  # noqa: E402
    FILTERED_MATERIALIZATION_SCHEMA,
    MATERIALIZATION_SCHEMA,
)
from src.clinical_eeg_long_recording.schema import canonical_payload_sha256  # noqa: E402
from src.clinical_eeg_long_recording.render import (  # noqa: E402
    _fact_locked_event_language,
    render_long_term_html,
)
from src.clinical_eeg_long_recording.report_outcome import (  # noqa: E402
    classify_recording_eeg_outcome,
)
from src.clinical_eeg_long_recording.schema import (  # noqa: E402
    validate_long_term_event_segment_receipt,
)
from src.clinical_eeg_long_recording.signal_findings import (  # noqa: E402
    DEFAULT_SIGNAL_FINDING_POLICY,
    SIGNAL_FINDINGS_PRODUCER_ID,
)
from src.clinical_eeg_report.generation import (  # noqa: E402
    PIPELINE_RECORD_SCHEMA,
    validate_narrative_payload,
)


SCHEMA_VERSION = "private_long_recording_report_release_audit_v1"
AUDIT_ID_PREFIX = "PLRAUD-"
PASS_STATUS = "release_audit_passed"
FAIL_STATUS = "release_audit_failed"
REPLACEMENT_AUTHORIZATION_SCHEMA = (
    "private_long_recording_report_replacement_authorization_v1"
)
REMEDIATION_SCOPE_SCHEMA = (
    "private_long_recording_report_remediation_release_scope_v1"
)
COHORT_AUDIT_MODE = "cohort_release"
REMEDIATION_AUDIT_MODE = "remediation_subset_release"
AUTHORIZATION_POLICY_ID = (
    "replace_only_audit_failed_primary_eeg_with_freshly_audited_eeg_v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CURRENT_FACT_TYPES = frozenset(
    {
        "recording_duration",
        "electrode_setup",
        "acquisition_settings",
        "electrographic_event_occurrence",
        "algorithmic_sustained_eeg_change",
    }
)
_BASE_ARTIFACTS = frozenset(
    {
        "bundle.json",
        "detection_manifest.json",
        "event_segment_receipts.json",
        "report.html",
        "report.docx",
    }
)
_LANGUAGE_LAYER_KEYS = frozenset(
    {
        "schema_version",
        "role",
        "served_model_name",
        "qwen_requested",
        "event_records",
        "scope_receipt",
    }
)
_LANGUAGE_EVENT_RECORD_KEYS = frozenset(
    {
        "eeg_event_id",
        "recording_event_start_offset_seconds",
        "language_record",
        "request_audit",
    }
)
_LANGUAGE_RECEIPT_KEYS = frozenset(
    {
        "configured",
        "qwen_requested",
        "event_count",
        "validated_qwen_wording_count",
        "deterministic_fallback_count",
        "language_failure_blocks_report_publication",
    }
)
_PIPELINE_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "report_id",
        "patient_pseudonym",
        "source_schema",
        "source_sha256",
        "scope_receipt",
        "style_receipt",
        "narrative",
        "generation",
        "release",
        "access_receipt",
    }
)
_FORBIDDEN_ACCESS_TRUE = frozenset(
    {
        "raw_eeg_loaded_by_narrator",
        "patient_identity_sent_to_llm",
        "signature_sent_to_llm",
        "non_eeg_context_sent_to_llm",
        "sleep_eeg_sent_to_llm",
        "activation_experiment_sent_to_llm",
        "event_occurrence_sent_to_llm",
        "treatment_generated",
    }
)
_POSITIVE_CLINICAL_TERM_RE = re.compile(
    r"(?:\b(?:spikes?|sharp(?:\s+waves?)?|IEDs?|ESz|LVFA|electrodecrement|"
    r"electrographic\s+seizures?|ictal\s+(?:onset|evolution|spread|termination)|"
    r"diffuse|generalized|bilateral(?:ly)?\s+synchronous|SOZ)\b|"
    r"棘波|尖波|癫痫样放电|电图发作|脑电发作|低电压快活动|"
    r"(?:电压|电极)递减|病理性\s*[δθ]|局灶(?:性)?慢化|弥漫(?:性)?慢化|"
    r"起始|演变|传播|扩散|终止|弥漫|广泛性|双侧同步|SOZ|致痫区|致痫灶|"
    r"手术靶点)",
    re.IGNORECASE,
)
_STRUCTURAL_PREFIXES = (
    "起始/候选脑电：",
    "发作起始形态：",
    "演变/传播：",
    "演变、后续头皮变化、终止及事件后：",
    "终止/事件后：",
    "四、SOZ 定位结论：",
    "五、不确定性与结论边界：",
    "三、SOZ 定位结论：",
    "四、不确定性与结论边界：",
    "头皮分布与定位推理：",
    "SOZ 定位结论：",
)
_FIXED_NEGATIVE_LANGUAGE_BLOCKS = frozenset(
    {
        (
            "普通量化持续变化只表示双极导联级信号改变，不等同于经确认的"
            "发作起始、临床演变、传播、皮层 SOZ、致痫区或治疗靶点。"
        ),
        (
            "该区间仅为检测器待复核候选支持范围，不表示已确认的"
            "脑电发作起始或终止。"
        ),
    }
)

_COMBINED_KEYS = frozenset(
    {
        "schema_version",
        "inventory_id",
        "recording_unit_policy",
        "mode",
        "expected_record_count",
        "expected_subject_count",
        "inventory_rejection_count",
        "unique_signal_count",
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
        "overlay_counts",
        "records",
        "subjects",
        "source_manifest_receipts",
        "overlay_policy_receipt",
        "scope_receipt",
        "combined_coverage_id",
    }
)
_COMBINED_ROW_KEYS = frozenset(
    {
        "recording_id",
        "patient_pseudonym",
        "inventory_validation_status",
        "run_status",
        "diagnostic_status",
        "event_count",
        "failure_stage",
        "existing_success_reused",
        "technical_artifact_relative_dir",
        "artifact_source",
        "effective_report_kind",
        "state_manifest_relative_path",
        "state_manifest_sha256",
        "report_manifest_relative_path",
        "report_manifest_sha256",
        "source_coverage_row_sha256",
        "recovery_overlay_applied",
        "overlay_decision",
        "superseded_primary_artifact_receipt",
    }
)
_FAILURE_CLASSIFICATION = {
    "report_manifest_binding_failed": (
        "report_binding_or_publication",
        "rerender_report_only",
    ),
    "artifact_incomplete_or_hash_mismatch": (
        "report_artifact_publication",
        "rerender_report_only",
    ),
    "bundle_validation_failed": (
        "event_fact_ledger",
        "redo_event_materialization_then_rerender",
    ),
    "current_neutral_producer_contract_failed": (
        "event_fact_ledger",
        "redo_event_materialization_then_rerender",
    ),
    "json_artifact_closure_failed": (
        "event_fact_ledger",
        "redo_event_materialization_then_rerender",
    ),
    "diagnostic_outcome_mismatch": (
        "report_outcome_projection",
        "rerender_report_only",
    ),
    "language_projection_invalid": (
        "language_or_renderer_projection",
        "rerender_report_only",
    ),
    "waveform_attachment_invalid": (
        "report_artifact_publication",
        "rerender_report_only",
    ),
    "document_artifact_invalid": (
        "language_or_renderer_projection",
        "rerender_report_only",
    ),
    "unauthorized_positive_clinical_language": (
        "language_or_renderer_projection",
        "rerender_report_only",
    ),
    "unexpected_report_validation_failure": (
        "unclassified_pipeline_failure",
        "rerun_full_recording_pipeline",
    ),
    "remediation_report_not_fresh": (
        "report_artifact_publication",
        "rerender_report_only",
    ),
    "technical_artifact_invalid": (
        "technical_report_artifact",
        "rerun_full_recording_pipeline",
    ),
}


class AuditDefect(ValueError):
    """A de-identified report-level release defect."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _replacement_authorization_receipt(
    *,
    authorizations: Sequence[Mapping[str, Any]],
    eligible: bool,
) -> dict[str, Any]:
    normalized = sorted(
        (dict(item) for item in authorizations),
        key=lambda item: str(item["recording_id"]),
    )
    body = {
        "schema_version": REPLACEMENT_AUTHORIZATION_SCHEMA,
        "authorization_policy_id": AUTHORIZATION_POLICY_ID,
        "eligible_for_primary_replacement": eligible,
        "authorized_failed_primary_eeg_count": len(normalized) if eligible else 0,
        "authorizations": normalized if eligible else [],
        "constraints": {
            "only_listed_recording_ids_may_replace_primary_eeg": True,
            "source_report_manifest_sha256_must_match_when_present": True,
            "audit_passed_primary_eeg_may_be_replaced": False,
            "replacement_technical_or_pending_artifact_allowed": False,
            "replacement_must_be_completed_eeg_report": True,
            "replacement_must_pass_fresh_release_audit": True,
            "replacement_may_change_inventory_or_recording_unit": False,
            "overlay_script_enforcement_implemented_in_this_change": False,
        },
    }
    return {
        "authorization_id": "PLRAUTH-" + _canonical_sha256(body)[:24],
        **body,
    }


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("JSON contains duplicate keys")
        result[key] = value
    return result


def _invalid_constant(_: str) -> None:
    raise ValueError("JSON contains a non-finite constant")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_snapshot(path: Path) -> tuple[object, tuple[Path, str]]:
    if path.is_symlink():
        raise ValueError("JSON input must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("JSON input must be a regular file")
    raw = resolved.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_invalid_constant,
    )
    return value, (resolved, _sha256_bytes(raw))


def _strict_json_bytes(raw: bytes) -> object:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_invalid_constant,
    )


def _regular_root(path: Path, context: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError(f"{context} must be a regular directory")
    return resolved


def _safe_relative(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context} is not a safe relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{context} is not a safe relative path")
    return relative


def _resolve_regular(root: Path, relative: PurePosixPath) -> Path:
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise AuditDefect("artifact_incomplete_or_hash_mismatch")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise AuditDefect("artifact_incomplete_or_hash_mismatch") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise AuditDefect("artifact_incomplete_or_hash_mismatch")
    return resolved


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _validate_combined_coverage(
    value: object,
    *,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _COMBINED_KEYS:
        raise ValueError("combined coverage has missing or unknown keys")
    if value["schema_version"] != overlay.SCHEMA_VERSION:
        raise ValueError("combined coverage schema drifted")
    body = {key: item for key, item in value.items() if key != "combined_coverage_id"}
    expected_id = "PLCOMB-" + _canonical_sha256(body)[:24]
    if value["combined_coverage_id"] != expected_id:
        raise ValueError("combined coverage ID does not bind its content")
    if value["inventory_id"] != inventory["inventory_id"]:
        raise ValueError("combined coverage inventory binding drifted")
    if value["recording_unit_policy"] != inventory["recording_unit_policy"]:
        raise ValueError("combined coverage recording policy drifted")
    if value["mode"] != "recovery_overlay":
        raise ValueError("combined coverage mode drifted")
    if value["expected_record_count"] != inventory["record_count"] or value[
        "expected_subject_count"
    ] != inventory["subject_count"]:
        raise ValueError("combined coverage cohort counts drifted")
    if value["inventory_rejection_count"] != len(inventory["source_rejections"]):
        raise ValueError("combined coverage inventory rejection count drifted")
    source_receipts = value["source_manifest_receipts"]
    if not isinstance(source_receipts, Mapping) or source_receipts.get(
        "inventory_manifest_sha256"
    ) != inventory_sha256:
        raise ValueError("combined coverage inventory snapshot binding drifted")
    scope = value["scope_receipt"]
    required_scope = {
        "edf_signal_files_read": False,
        "edf_annotations_read": False,
        "excel_or_workbook_read": False,
        "onset_label_or_ground_truth_read": False,
        "raw_edf_paths_persisted": False,
        "source_output_trees_modified": False,
        "combined_manifest_only": True,
    }
    if not isinstance(scope, Mapping) or any(
        scope.get(key) != expected for key, expected in required_scope.items()
    ):
        raise ValueError("combined coverage violates the source boundary")

    raw_rows = value["records"]
    if not isinstance(raw_rows, list) or len(raw_rows) != inventory["record_count"]:
        raise ValueError("combined coverage does not span the inventory")
    rows: list[dict[str, Any]] = []
    for raw, inventory_record in zip(
        raw_rows, inventory["records"], strict=True
    ):
        if not isinstance(raw, Mapping) or set(raw) != _COMBINED_ROW_KEYS:
            raise ValueError("combined coverage row schema drifted")
        for key in (
            "recording_id",
            "patient_pseudonym",
            "inventory_validation_status",
        ):
            if raw[key] != inventory_record[key]:
                raise ValueError("combined coverage row identity drifted")
        diagnostic = raw["diagnostic_status"]
        if diagnostic not in overlay.ALL_COMPLETED_STATUSES:
            raise ValueError("combined coverage contains an incomplete selection")
        _positive_int(raw["event_count"], "combined event count")
        for key in (
            "state_manifest_sha256",
            "report_manifest_sha256",
            "source_coverage_row_sha256",
        ):
            if not isinstance(raw[key], str) or _SHA256_RE.fullmatch(raw[key]) is None:
                raise ValueError("combined coverage carries an invalid SHA-256")
        state_relative = _safe_relative(
            raw["state_manifest_relative_path"], "combined state manifest"
        )
        report_relative = _safe_relative(
            raw["report_manifest_relative_path"], "combined report manifest"
        )
        recording_id = str(raw["recording_id"])
        if state_relative != PurePosixPath("records") / recording_id / "state.json":
            raise ValueError("combined state locator drifted")
        if diagnostic in overlay.EEG_STATUSES:
            if raw["effective_report_kind"] != "eeg_report":
                raise ValueError("combined EEG report kind drifted")
            if report_relative != (
                PurePosixPath("records") / recording_id / "report" / "manifest.json"
            ):
                raise ValueError("combined EEG report locator drifted")
        elif raw["effective_report_kind"] != "technical_unassessable_report":
            raise ValueError("combined technical report kind drifted")
        if raw["artifact_source"] not in {"primary", "recovery", "remediation"}:
            raise ValueError("combined artifact source drifted")
        rows.append(dict(raw))

    completed_eeg = sum(
        row["diagnostic_status"] in overlay.EEG_STATUSES for row in rows
    )
    technical = sum(
        row["diagnostic_status"] == overlay.TECHNICAL_STATUS for row in rows
    )
    if value["completed_eeg_report_count"] != completed_eeg or value[
        "technical_unassessable_report_count"
    ] != technical:
        raise ValueError("combined report counts drifted")
    if value["completed_report_artifact_count"] != completed_eeg + technical:
        raise ValueError("combined artifact count drifted")
    expected_diagnostics = {
        status: sum(row["diagnostic_status"] == status for row in rows)
        for status in sorted(overlay.ALL_COMPLETED_STATUSES)
    }
    if value["diagnostic_status_counts"] != expected_diagnostics:
        raise ValueError("combined diagnostic counts drifted")
    return {**dict(value), "records": rows}


def _validate_coverage_input(
    value: object,
    *,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("coverage must be a JSON object")
    schema = value.get("schema_version")
    if schema == batch.COVERAGE_SCHEMA_VERSION:
        return "full", overlay._validate_coverage(  # noqa: SLF001
            value, inventory=inventory, source_name="full"
        )
    if schema == overlay.SCHEMA_VERSION:
        return "combined", _validate_combined_coverage(
            value,
            inventory=inventory,
            inventory_sha256=inventory_sha256,
        )
    raise ValueError("coverage schema is unsupported")


def _remediation_authorization(
    path: Path,
    *,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
) -> tuple[
    dict[str, Any],
    tuple[Path, str],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    value, snapshot = _read_snapshot(path)
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("primary release audit schema drifted")
    audit = dict(value)
    audit_id = audit.get("audit_id")
    audit_body = {key: item for key, item in audit.items() if key != "audit_id"}
    if audit_id != AUDIT_ID_PREFIX + _canonical_sha256(audit_body)[:24]:
        raise ValueError("primary release audit ID does not bind its content")
    if audit.get("audit_mode") != COHORT_AUDIT_MODE:
        raise ValueError("remediation must bind a primary cohort audit")
    if audit.get("status") != FAIL_STATUS or audit.get("release_ready") is not False:
        raise ValueError("an audit-passed primary cannot authorize remediation")
    if audit.get("coverage_kind") != "full" or audit.get("inventory_id") != inventory[
        "inventory_id"
    ]:
        raise ValueError("primary release audit inventory or coverage kind drifted")
    sources = audit.get("source_receipts")
    if not isinstance(sources, Mapping) or sources.get(
        "inventory_manifest_sha256"
    ) != inventory_sha256:
        raise ValueError("primary release audit inventory snapshot drifted")
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
        receipt.get("schema_version") != REPLACEMENT_AUTHORIZATION_SCHEMA
        or receipt.get("authorization_policy_id") != AUTHORIZATION_POLICY_ID
        or receipt.get("eligible_for_primary_replacement") is not True
    ):
        raise ValueError("primary audit is not replacement eligible")
    raw = receipt.get("authorizations")
    if not isinstance(raw, list) or receipt.get(
        "authorized_failed_primary_eeg_count"
    ) != len(raw) or raw != audit.get("failed_reports"):
        raise ValueError("primary replacement authorization set drifted")
    all_authorizations: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("primary replacement authorization is invalid")
        recording_id = item.get("recording_id")
        if (
            not isinstance(recording_id, str)
            or recording_id in seen
            or item.get("selected_artifact_source") != "primary"
            or not isinstance(item.get("failure_reason_codes"), list)
            or not item["failure_reason_codes"]
            or not isinstance(item.get("source_report_manifest_sha256"), str)
            or _SHA256_RE.fullmatch(item["source_report_manifest_sha256"]) is None
        ):
            raise ValueError("primary authorization row drifted")
        seen.add(recording_id)
        normalized = dict(item)
        all_authorizations.append(normalized)
        if (
            item.get("failure_layer") == "language_or_renderer_projection"
            and item.get("minimum_remediation") == "rerender_report_only"
        ):
            eligible.append(normalized)
    if not eligible:
        raise ValueError("primary audit authorizes no renderer-only remediation")
    return (
        audit,
        snapshot,
        sorted(all_authorizations, key=lambda item: item["recording_id"]),
        sorted(eligible, key=lambda item: item["recording_id"]),
    )


class _VisibleHTML(HTMLParser):
    _BREAK_TAGS = frozenset(
        {
            "br",
            "p",
            "div",
            "td",
            "th",
            "tr",
            "h1",
            "h2",
            "h3",
            "li",
            "table",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._parts: list[str] = []
        self._ignored_depth = 0

    def _flush(self) -> None:
        value = "".join(self._parts).strip()
        if value:
            self.blocks.append(value)
        self._parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        if tag in self._BREAK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BREAK_TAGS:
            self._flush()
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def finish(self) -> list[str]:
        self._flush()
        return self.blocks


def _html_blocks(raw: bytes) -> list[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditDefect("document_artifact_invalid") from exc
    parser = _VisibleHTML()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:
        raise AuditDefect("document_artifact_invalid") from exc
    if "<html" not in text.lower() or "</html>" not in text.lower():
        raise AuditDefect("document_artifact_invalid")
    return parser.finish()


def _docx_blocks_and_media(raw: bytes) -> tuple[list[str], dict[str, bytes]]:
    try:
        with tempfile.TemporaryFile() as stream:
            stream.write(raw)
            stream.seek(0)
            with ZipFile(stream) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or any(
                    PurePosixPath(name).is_absolute()
                    or any(part in {"", ".", ".."} for part in PurePosixPath(name).parts)
                    for name in names
                ):
                    raise AuditDefect("document_artifact_invalid")
                required = {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
                if not required.issubset(names) or archive.testzip() is not None:
                    raise AuditDefect("document_artifact_invalid")
                document = archive.read("word/document.xml")
                media = {
                    name: archive.read(name)
                    for name in names
                    if name.startswith("word/media/") and not name.endswith("/")
                }
    except (BadZipFile, KeyError, OSError) as exc:
        raise AuditDefect("document_artifact_invalid") from exc
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        raise AuditDefect("document_artifact_invalid") from exc
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    blocks: list[str] = []
    for paragraph in root.iter(namespace + "p"):
        pieces: list[str] = []
        for node in paragraph.iter():
            if node.tag == namespace + "t":
                pieces.append(node.text or "")
            elif node.tag == namespace + "br":
                # A Word line break is a visible semantic boundary.  Preserve
                # it during audit extraction so two independently rendered
                # report lines cannot be concatenated into a different token
                # stream than the deterministic HTML reference surface.
                pieces.append(" ")
        value = "".join(pieces).strip()
        if value:
            blocks.append(value)
    if not blocks:
        raise AuditDefect("document_artifact_invalid")
    return blocks, media


def _normalize_block(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_structural_prefixes(value: str) -> str:
    result = value
    for prefix in _STRUCTURAL_PREFIXES:
        result = result.replace(prefix, " ")
    return result


def _forbidden_blocks(
    blocks: Iterable[str],
    *,
    allowed_boundary_blocks: set[str],
) -> list[str]:
    result: list[str] = []
    for raw in blocks:
        value = _normalize_block(raw)
        stripped = _normalize_block(_strip_structural_prefixes(value))
        if _POSITIVE_CLINICAL_TERM_RE.search(stripped) is None:
            continue
        if value in allowed_boundary_blocks or stripped in allowed_boundary_blocks:
            continue
        residual = stripped
        for allowed in sorted(allowed_boundary_blocks, key=len, reverse=True):
            normalized_allowed = _normalize_block(
                _strip_structural_prefixes(allowed)
            )
            if normalized_allowed:
                residual = residual.replace(normalized_allowed, " ")
        if _POSITIVE_CLINICAL_TERM_RE.search(_normalize_block(residual)) is None:
            continue
        result.append("unauthorized_positive_clinical_language")
    return result


def _current_bundle_semantics(bundle: Mapping[str, Any]) -> None:
    expected_qualification = {
        "producer_id": SIGNAL_FINDINGS_PRODUCER_ID,
        "policy_sha256": DEFAULT_SIGNAL_FINDING_POLICY.sha256,
        "artifact_gate_passed": True,
        "sustained_change_gate_passed": True,
        "reproducibility_gate_passed": True,
        "source_signal_only": True,
        "external_context_used": False,
        "research_ranking_used": False,
        "morphology_terms_qualified": False,
        "spatial_spread_terms_qualified": False,
    }
    for event in bundle["events"]:
        report = event["event_report_payload"]
        for fact in report["facts"]:
            fact_type = fact["fact_type"]
            if fact_type not in _CURRENT_FACT_TYPES:
                raise AuditDefect("current_neutral_producer_contract_failed")
            if fact_type == "electrographic_event_occurrence":
                value = fact["value"]
                if value.get("event_class") != "uncertain_electrographic_pattern":
                    raise AuditDefect("current_neutral_producer_contract_failed")
                if value.get("text_zh") != (
                    "该区间仅为检测器待复核候选支持范围，不表示已确认的"
                    "脑电发作起始或终止。"
                ):
                    raise AuditDefect("current_neutral_producer_contract_failed")
            elif fact_type == "algorithmic_sustained_eeg_change":
                value = fact["value"]
                if not isinstance(value, Mapping):
                    raise AuditDefect("current_neutral_producer_contract_failed")
                if not _CURRENT_SIGNAL_FINDING_REQUIRED_VALUE_KEYS.issubset(value):
                    raise AuditDefect("current_neutral_producer_contract_failed")
                if set(value).difference(_CURRENT_SIGNAL_FINDING_VALUE_KEYS):
                    raise AuditDefect("current_neutral_producer_contract_failed")
                if value.get("qualification") != expected_qualification:
                    raise AuditDefect("current_neutral_producer_contract_failed")
                text = value.get("text_zh")
                if not isinstance(text, str) or _POSITIVE_CLINICAL_TERM_RE.search(text):
                    raise AuditDefect("current_neutral_producer_contract_failed")


def _validate_language(
    *,
    bundle: Mapping[str, Any],
    language: object | None,
    receipt: object,
) -> tuple[int, int, int, Mapping[str, Any] | None]:
    if not isinstance(receipt, Mapping) or set(receipt) != _LANGUAGE_RECEIPT_KEYS:
        raise AuditDefect("language_projection_invalid")
    if receipt.get("language_failure_blocks_report_publication") is not False:
        raise AuditDefect("language_projection_invalid")
    event_count = int(bundle["event_count"])
    configured = receipt.get("configured") is True
    if not configured:
        if language is not None or receipt != {
            "configured": False,
            "qwen_requested": False,
            "event_count": 0,
            "validated_qwen_wording_count": 0,
            "deterministic_fallback_count": 0,
            "language_failure_blocks_report_publication": False,
        }:
            raise AuditDefect("language_projection_invalid")
        return 0, 0, event_count, None

    if not isinstance(language, Mapping) or set(language) != _LANGUAGE_LAYER_KEYS:
        raise AuditDefect("language_projection_invalid")
    if language.get("schema_version") != "long_term_event_language_layer_v1":
        raise AuditDefect("language_projection_invalid")
    raw_records = language.get("event_records")
    if not isinstance(raw_records, list) or len(raw_records) != event_count:
        raise AuditDefect("language_projection_invalid")
    if receipt.get("event_count") != event_count or receipt.get(
        "qwen_requested"
    ) != language.get("qwen_requested"):
        raise AuditDefect("language_projection_invalid")

    events_by_id = {str(event["eeg_event_id"]): event for event in bundle["events"]}
    seen: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping) or set(raw) != _LANGUAGE_EVENT_RECORD_KEYS:
            raise AuditDefect("language_projection_invalid")
        event_id = raw.get("eeg_event_id")
        if not isinstance(event_id, str) or event_id not in events_by_id or event_id in seen:
            raise AuditDefect("language_projection_invalid")
        seen.add(event_id)
        record = raw.get("language_record")
        if not isinstance(record, Mapping) or set(record) != _PIPELINE_RECORD_KEYS:
            raise AuditDefect("language_projection_invalid")
        event_report = events_by_id[event_id]["event_report_payload"]
        if (
            record.get("schema_version") != PIPELINE_RECORD_SCHEMA
            or record.get("report_id") != event_report["report_id"]
            or record.get("patient_pseudonym") != event_report["patient_pseudonym"]
            or record.get("source_schema") != event_report["schema_version"]
            or record.get("source_sha256") != _canonical_sha256(event_report)
        ):
            raise AuditDefect("language_projection_invalid")
        access = record.get("access_receipt")
        if not isinstance(access, Mapping) or any(
            access.get(key) is not False for key in _FORBIDDEN_ACCESS_TRUE
        ):
            raise AuditDefect("language_projection_invalid")
        try:
            validate_narrative_payload(record["narrative"], event_report)
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditDefect("language_projection_invalid") from exc

    projection = _fact_locked_event_language(bundle, language)
    qwen_count = len(projection)
    fallback_count = event_count - qwen_count
    if receipt.get("validated_qwen_wording_count") != qwen_count or receipt.get(
        "deterministic_fallback_count"
    ) != fallback_count:
        raise AuditDefect("language_projection_invalid")
    scope = language.get("scope_receipt")
    expected_scope = {
        "clinical_eeg_fact_ledgers_sent": bool(
            language.get("qwen_requested") is True and raw_records
        ),
        "source_context_sent": False,
        "edf_annotation_sent": False,
        "excel_observation_sent": False,
        "waveform_image_or_path_sent": False,
        "research_soz_ranking_sent": False,
        "may_change_event_count": False,
        "may_change_event_coordinates": False,
        "may_change_recording_impression": False,
        "used_by_deterministic_renderer": qwen_count > 0,
        "bounded_event_wording_projection_eligible_count": qwen_count,
        "projection_generator_must_equal": "qwen3.6_facts_locked_draft",
        "projection_excludes_findings_and_impression": True,
        "prompt_or_schema_content_persisted": False,
        "request_audit_hashes_only": True,
        "prompt_firewall_fail_closed": True,
    }
    if scope != expected_scope:
        raise AuditDefect("language_projection_invalid")
    projected_blocks = [
        text
        for event_text in projection.values()
        for text in event_text.values()
    ]
    return qwen_count, fallback_count, 0, {
        "layer": language,
        "projected_blocks": projected_blocks,
    }


def _verify_png(raw: bytes) -> None:
    if len(raw) < 33 or not raw.startswith(_PNG_SIGNATURE) or raw[12:16] != b"IHDR":
        raise AuditDefect("waveform_attachment_invalid")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width < 1 or height < 1:
        raise AuditDefect("waveform_attachment_invalid")


def _artifact_bytes(
    report_root: Path,
    artifacts: Mapping[str, Any],
    snapshots: list[tuple[Path, str]],
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for relative_text, declared_sha in artifacts.items():
        if not isinstance(declared_sha, str) or _SHA256_RE.fullmatch(declared_sha) is None:
            raise AuditDefect("artifact_incomplete_or_hash_mismatch")
        try:
            relative = _safe_relative(relative_text, "report artifact")
        except ValueError as exc:
            raise AuditDefect("artifact_incomplete_or_hash_mismatch") from exc
        path = _resolve_regular(report_root, relative)
        raw = path.read_bytes()
        actual_sha = _sha256_bytes(raw)
        snapshots.append((path, actual_sha))
        if actual_sha != declared_sha:
            raise AuditDefect("artifact_incomplete_or_hash_mismatch")
        result[relative.as_posix()] = raw
    declared_files = {report_root / relative for relative in map(Path, result)}
    declared_files.add(report_root / "manifest.json")
    for path in report_root.rglob("*"):
        if path.is_symlink():
            raise AuditDefect("artifact_incomplete_or_hash_mismatch")
        if path.is_file() and path not in declared_files:
            raise AuditDefect("artifact_incomplete_or_hash_mismatch")
    return result


def _audit_one_report(
    *,
    root: Path,
    row: Mapping[str, Any],
    inventory_record: Mapping[str, Any],
    combined: bool,
    snapshots: list[tuple[Path, str]],
) -> dict[str, int]:
    recording_id = str(inventory_record["recording_id"])
    state_relative = (
        _safe_relative(row["state_manifest_relative_path"], "state manifest")
        if combined
        else PurePosixPath("records") / recording_id / "state.json"
    )
    state_path = _resolve_regular(root, state_relative)
    state_raw = state_path.read_bytes()
    state_sha = _sha256_bytes(state_raw)
    snapshots.append((state_path, state_sha))
    if combined and state_sha != row["state_manifest_sha256"]:
        raise AuditDefect("report_manifest_binding_failed")
    try:
        state = _strict_json_bytes(state_raw)
        overlay._validate_state(  # noqa: SLF001
            state,
            inventory_record=inventory_record,
            row=row,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditDefect("report_manifest_binding_failed") from exc
    if combined:
        relative = _safe_relative(row["report_manifest_relative_path"], "report manifest")
    else:
        relative = PurePosixPath("records") / recording_id / "report" / "manifest.json"
    manifest_path = _resolve_regular(root, relative)
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = _sha256_bytes(manifest_raw)
    snapshots.append((manifest_path, manifest_sha))
    if combined and manifest_sha != row["report_manifest_sha256"]:
        raise AuditDefect("report_manifest_binding_failed")
    try:
        manifest = _strict_json_bytes(manifest_raw)
        overlay._validate_eeg_report(  # noqa: SLF001
            manifest,
            inventory_record=inventory_record,
            row=row,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditDefect("report_manifest_binding_failed") from exc
    assert isinstance(manifest, Mapping)
    filtered = manifest.get("schema_version") == FILTERED_MATERIALIZATION_SCHEMA
    if manifest.get("schema_version") not in {
        MATERIALIZATION_SCHEMA,
        FILTERED_MATERIALIZATION_SCHEMA,
    }:
        raise AuditDefect("report_manifest_binding_failed")
    scope = manifest.get("scope_receipt")
    if not isinstance(scope, Mapping) or scope.get("all_waveforms_hash_verified") is not True:
        raise AuditDefect("report_manifest_binding_failed")
    source_receipts = manifest.get("source_receipts")
    if (
        not isinstance(source_receipts, Mapping)
        or source_receipts.get("context_sha256") is not None
        or not isinstance(source_receipts.get("segment_receipts"), list)
        or len(source_receipts["segment_receipts"]) != manifest["event_count"]
    ):
        raise AuditDefect("report_manifest_binding_failed")
    if filtered:
        if (
            not isinstance(source_receipts.get("analysis_selection_sha256"), str)
            or _SHA256_RE.fullmatch(source_receipts["analysis_selection_sha256"])
            is None
            or scope.get("signal_eligibility_partition_validated") is not True
            or scope.get("detector_selected_candidates_exactly_partitioned")
            is not True
            or scope.get("rejected_candidate_is_not_no_seizure") is not True
        ):
            raise AuditDefect("report_manifest_binding_failed")
    elif "analysis_selection_sha256" in source_receipts:
        raise AuditDefect("report_manifest_binding_failed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AuditDefect("artifact_incomplete_or_hash_mismatch")
    report_root = manifest_path.parent
    raw_artifacts = _artifact_bytes(report_root, artifacts, snapshots)
    if not _BASE_ARTIFACTS.issubset(raw_artifacts):
        raise AuditDefect("artifact_incomplete_or_hash_mismatch")

    try:
        bundle_raw = _strict_json_bytes(raw_artifacts["bundle.json"])
        bundle = validate_trustworthy_long_term_clinical_eeg_bundle(bundle_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditDefect("bundle_validation_failed") from exc
    if (
        bundle["recording_id"] != manifest["recording_id"]
        or bundle["patient_pseudonym"] != manifest["patient_pseudonym"]
        or bundle["bundle_id"] != manifest["bundle_id"]
        or bundle["event_count"] != manifest["event_count"]
    ):
        raise AuditDefect("bundle_validation_failed")
    _current_bundle_semantics(bundle)

    if filtered:
        try:
            selection_raw = _strict_json_bytes(
                raw_artifacts["analysis_selection_manifest.json"]
            )
        except (KeyError, TypeError, UnicodeDecodeError, ValueError) as exc:
            raise AuditDefect("json_artifact_closure_failed") from exc
        if (
            selection_raw != bundle.get("analysis_selection")
            or canonical_payload_sha256(selection_raw)
            != bundle.get("analysis_selection_sha256")
            or manifest.get("detector_selected_candidate_count")
            != bundle.get("detector_selected_candidate_count")
            or manifest.get("analysis_analyzable_candidate_count")
            != bundle.get("analysis_analyzable_candidate_count")
            or manifest.get("analysis_rejected_candidate_count")
            != bundle.get("analysis_rejected_candidate_count")
            or manifest.get("event_count")
            != bundle.get("analysis_analyzable_candidate_count")
        ):
            raise AuditDefect("json_artifact_closure_failed")
    elif "analysis_selection_manifest.json" in raw_artifacts:
        raise AuditDefect("artifact_incomplete_or_hash_mismatch")

    try:
        detection = _strict_json_bytes(raw_artifacts["detection_manifest.json"])
        segments = _strict_json_bytes(raw_artifacts["event_segment_receipts.json"])
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise AuditDefect("json_artifact_closure_failed") from exc
    if detection != bundle["detection_manifest"]:
        raise AuditDefect("json_artifact_closure_failed")
    if not isinstance(segments, list) or len(segments) != bundle["event_count"]:
        raise AuditDefect("json_artifact_closure_failed")
    try:
        validated_segments = [
            validate_long_term_event_segment_receipt(item) for item in segments
        ]
    except (TypeError, ValueError) as exc:
        raise AuditDefect("json_artifact_closure_failed") from exc
    if [item["eeg_event_id"] for item in validated_segments] != [
        item["eeg_event_id"] for item in bundle["events"]
    ]:
        raise AuditDefect("json_artifact_closure_failed")

    expected_outcome = classify_recording_eeg_outcome(bundle)
    if (
        manifest["diagnostic_outcome"] != expected_outcome
        or manifest["diagnostic_status"] != expected_outcome["report_status"]
        or row["diagnostic_status"] != expected_outcome["report_status"]
    ):
        raise AuditDefect("diagnostic_outcome_mismatch")

    receipt = manifest.get("language_service_receipt")
    configured = isinstance(receipt, Mapping) and receipt.get("configured") is True
    language_raw = raw_artifacts.get("language_records.json")
    if configured != (language_raw is not None):
        raise AuditDefect("language_projection_invalid")
    language: object | None = None
    if language_raw is not None:
        try:
            language = _strict_json_bytes(language_raw)
        except (UnicodeDecodeError, TypeError, ValueError) as exc:
            raise AuditDefect("language_projection_invalid") from exc
    qwen, fallback, unconfigured, language_projection = _validate_language(
        bundle=bundle,
        language=language,
        receipt=receipt,
    )

    expected_waveforms: dict[str, str] = {}
    hrefs: dict[str, str] = {}
    for index, event in enumerate(bundle["events"], start=1):
        relative = f"waveforms/eeg_waveform_{index:02d}.png"
        attachment = event["waveform_attachment"]
        if attachment["figure_file"] != relative:
            raise AuditDefect("waveform_attachment_invalid")
        expected_waveforms[relative] = str(attachment["figure_sha256"])
        hrefs[str(event["eeg_event_id"])] = relative
        raw = raw_artifacts.get(relative)
        if raw is None or _sha256_bytes(raw) != attachment["figure_sha256"]:
            raise AuditDefect("waveform_attachment_invalid")
        _verify_png(raw)
    expected_artifact_names = set(_BASE_ARTIFACTS) | set(expected_waveforms)
    if filtered:
        expected_artifact_names.add("analysis_selection_manifest.json")
    if configured:
        expected_artifact_names.add("language_records.json")
    if set(raw_artifacts) != expected_artifact_names:
        raise AuditDefect("artifact_incomplete_or_hash_mismatch")

    html_blocks = _html_blocks(raw_artifacts["report.html"])
    docx_blocks, docx_media = _docx_blocks_and_media(raw_artifacts["report.docx"])
    expected_media = {
        f"word/media/eeg_waveform_{index:02d}.png": raw_artifacts[relative]
        for index, relative in enumerate(expected_waveforms, start=1)
    }
    if docx_media != expected_media:
        raise AuditDefect("waveform_attachment_invalid")
    if not any("长程头皮脑电分析报告" in block for block in html_blocks) or not any(
        "长程头皮脑电分析报告" in block for block in docx_blocks
    ):
        raise AuditDefect("document_artifact_invalid")
    html_visible = "\n".join(html_blocks)
    docx_visible = "\n".join(docx_blocks)
    if bundle["recording_id"] not in html_visible or bundle[
        "recording_id"
    ] not in docx_visible:
        raise AuditDefect("document_artifact_invalid")

    deterministic_html = render_long_term_html(
        bundle,
        waveform_hrefs=hrefs,
        language_layer=None,
    ).encode("utf-8")
    allowed_blocks = {
        _normalize_block(block)
        for block in _html_blocks(deterministic_html)
        if _POSITIVE_CLINICAL_TERM_RE.search(_normalize_block(block))
    }
    allowed_blocks.update(map(_normalize_block, _FIXED_NEGATIVE_LANGUAGE_BLOCKS))
    scan_blocks = [*html_blocks, *docx_blocks]
    if language_projection is not None:
        scan_blocks.extend(language_projection["projected_blocks"])
    if _forbidden_blocks(
        scan_blocks,
        allowed_boundary_blocks=allowed_blocks,
    ):
        raise AuditDefect("unauthorized_positive_clinical_language")

    return {
        "event_count": int(bundle["event_count"]),
        "validated_qwen_wording_count": qwen,
        "deterministic_language_fallback_count": fallback,
        "deterministic_without_language_count": unconfigured,
        "waveform_count": len(expected_waveforms),
        "artifact_count": len(raw_artifacts),
    }


def _assert_unchanged(snapshots: Iterable[tuple[Path, str]]) -> None:
    seen: set[Path] = set()
    for path, expected_sha in snapshots:
        if path in seen:
            continue
        seen.add(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("an audited source artifact changed during the audit")
        if _sha256_bytes(path.read_bytes()) != expected_sha:
            raise ValueError("an audited source artifact changed during the audit")


def _audit_one_technical_report(
    *,
    root: Path,
    row: Mapping[str, Any],
    inventory_record: Mapping[str, Any],
    combined: bool,
    snapshots: list[tuple[Path, str]],
) -> int:
    """Validate a non-diagnostic technical shell and its failure receipt."""

    recording_id = str(row["recording_id"])
    if combined:
        state_relative = _safe_relative(
            row["state_manifest_relative_path"], "technical state manifest"
        )
        report_relative = _safe_relative(
            row["report_manifest_relative_path"], "technical report manifest"
        )
    else:
        state_relative = PurePosixPath("records") / recording_id / "state.json"
        technical_dir = _safe_relative(
            row["technical_artifact_relative_dir"], "technical artifact directory"
        )
        report_relative = (
            PurePosixPath("records")
            / recording_id
            / technical_dir
            / "manifest.json"
        )
    state_path = _resolve_regular(root, state_relative)
    report_path = _resolve_regular(root, report_relative)
    state_raw = state_path.read_bytes()
    report_raw = report_path.read_bytes()
    state_sha = _sha256_bytes(state_raw)
    report_sha = _sha256_bytes(report_raw)
    snapshots.extend(((state_path, state_sha), (report_path, report_sha)))
    if combined and (
        state_sha != row["state_manifest_sha256"]
        or report_sha != row["report_manifest_sha256"]
    ):
        raise AuditDefect("technical_artifact_invalid")
    try:
        state = _strict_json_bytes(state_raw)
        report_manifest = _strict_json_bytes(report_raw)
        overlay._validate_state(  # noqa: SLF001
            state, inventory_record=inventory_record, row=row
        )
        overlay._validate_technical_report(  # noqa: SLF001
            report_manifest, inventory_record=inventory_record, row=row
        )
        assert isinstance(state, Mapping) and isinstance(report_manifest, Mapping)
        report_dir = report_path.parent
        artifacts = report_manifest["artifacts"]
        if set(artifacts) != {"report.json", "report.html"}:
            raise ValueError("technical report artifact roster drifted")
        artifact_raw: dict[str, bytes] = {}
        for name, expected_sha in artifacts.items():
            path = _resolve_regular(report_dir, _safe_relative(name, "technical artifact"))
            raw = path.read_bytes()
            digest = _sha256_bytes(raw)
            snapshots.append((path, digest))
            if digest != expected_sha:
                raise ValueError("technical report artifact hash drifted")
            artifact_raw[str(name)] = raw
        report_json = _strict_json_bytes(artifact_raw["report.json"])
        if not isinstance(report_json, Mapping) or set(report_json) != {
            "schema_version",
            "status",
            "recording_id",
            "patient_pseudonym",
            "failure_stage",
            "technical_failure_receipt_fingerprint",
            "eeg_analysis_completed",
            "diagnostic_status",
            "soz_conclusion_code",
            "conclusion_zh",
            "claim_boundary",
            "scope_receipt",
        }:
            raise ValueError("technical report JSON schema drifted")
        if (
            report_json["schema_version"] != batch.TECHNICAL_REPORT_SCHEMA_VERSION
            or report_json["status"] != overlay.TECHNICAL_STATUS
            or report_json["diagnostic_status"] != overlay.TECHNICAL_STATUS
            or report_json["recording_id"] != inventory_record["recording_id"]
            or report_json["patient_pseudonym"]
            != inventory_record["patient_pseudonym"]
            or report_json["failure_stage"] != row["failure_stage"]
            or report_json["eeg_analysis_completed"] is not False
            or report_json["soz_conclusion_code"]
            != "soz_unassessable_technical_failure"
            or report_json["technical_failure_receipt_fingerprint"]
            != report_manifest["technical_failure_receipt_fingerprint"]
            or report_json["claim_boundary"]
            != {
                "normal_or_negative_eeg_claimed": False,
                "diffuse_or_bilateral_onset_claimed": False,
                "focal_soz_claimed": False,
                "clinical_diagnosis_generated": False,
                "physician_signed": False,
            }
            or report_json["scope_receipt"]
            != {
                "edf_annotations_loaded": False,
                "excel_or_workbook_loaded": False,
                "onset_label_or_ground_truth_loaded": False,
                "exception_message_or_raw_path_persisted": False,
            }
        ):
            raise ValueError("technical report JSON binding drifted")
        technical_html = artifact_raw["report.html"].decode("utf-8")
        if (
            "<html" not in technical_html.lower()
            or "</html>" not in technical_html.lower()
            or str(report_json["conclusion_zh"]) not in technical_html
            or str(inventory_record["recording_id"]) not in technical_html
            or str(inventory_record["patient_pseudonym"]) not in technical_html
        ):
            raise ValueError("technical HTML projection drifted")

        receipt_relative = (
            PurePosixPath("records") / recording_id / "technical_failure_receipt.json"
        )
        receipt_path = _resolve_regular(root, receipt_relative)
        receipt_raw = receipt_path.read_bytes()
        receipt_sha = _sha256_bytes(receipt_raw)
        snapshots.append((receipt_path, receipt_sha))
        receipt = _strict_json_bytes(receipt_raw)
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "schema_version",
            "status",
            "recording_id",
            "patient_pseudonym",
            "failed_stage",
            "error_code",
            "exception_class",
            "error_fingerprint",
            "attempt",
            "retryable",
            "report_generated",
            "subprocess_receipt",
            "privacy_receipt",
            "technical_report_relative_dir",
        }:
            raise ValueError("technical failure receipt schema drifted")
        fingerprint_source: dict[str, Any] = {
            "stage": receipt["failed_stage"],
            "error_code": receipt["error_code"],
            "exception_class": receipt["exception_class"],
        }
        if receipt["subprocess_receipt"] is not None:
            subprocess_receipt = receipt["subprocess_receipt"]
            if (
                not isinstance(subprocess_receipt, Mapping)
                or set(subprocess_receipt)
                != {
                    "returncode",
                    "stdout_sha256",
                    "stderr_sha256",
                    "stdout_or_stderr_persisted",
                }
                or isinstance(subprocess_receipt["returncode"], bool)
                or not isinstance(subprocess_receipt["returncode"], int)
                or not isinstance(subprocess_receipt["stdout_sha256"], str)
                or _SHA256_RE.fullmatch(subprocess_receipt["stdout_sha256"])
                is None
                or not isinstance(subprocess_receipt["stderr_sha256"], str)
                or _SHA256_RE.fullmatch(subprocess_receipt["stderr_sha256"])
                is None
                or subprocess_receipt["stdout_or_stderr_persisted"] is not False
            ):
                raise ValueError("technical subprocess receipt drifted")
            fingerprint_source["subprocess"] = receipt["subprocess_receipt"]
        expected_dir = report_relative.parent.relative_to(
            PurePosixPath("records") / recording_id
        ).as_posix()
        if (
            receipt["schema_version"] != batch.FAILURE_SCHEMA_VERSION
            or receipt["status"] != "technical_failure_receipt"
            or receipt["recording_id"] != inventory_record["recording_id"]
            or receipt["patient_pseudonym"] != inventory_record["patient_pseudonym"]
            or receipt["failed_stage"] != row["failure_stage"]
            or not isinstance(receipt["error_code"], str)
            or not receipt["error_code"]
            or not isinstance(receipt["exception_class"], str)
            or not receipt["exception_class"]
            or isinstance(receipt["attempt"], bool)
            or not isinstance(receipt["attempt"], int)
            or receipt["attempt"] < 1
            or receipt["attempt"] != state["attempt"]
            or receipt["retryable"] is not True
            or receipt["report_generated"] is not True
            or receipt["technical_report_relative_dir"] != expected_dir
            or receipt["error_fingerprint"] != _canonical_sha256(fingerprint_source)
            or receipt["error_fingerprint"]
            != report_manifest["technical_failure_receipt_fingerprint"]
            or receipt["privacy_receipt"]
            != {
                "exception_message_persisted": False,
                "raw_edf_path_persisted": False,
                "raw_patient_identity_persisted": False,
                "annotation_excel_onset_or_gt_persisted": False,
            }
        ):
            raise ValueError("technical failure receipt binding drifted")
    except AuditDefect:
        raise
    except (KeyError, OSError, TypeError, UnicodeError, ValueError) as exc:
        raise AuditDefect("technical_artifact_invalid") from exc
    return 5


def _selected_report_manifest_sha256(
    *,
    root: Path,
    row: Mapping[str, Any],
    recording_id: str,
    combined: bool,
) -> str | None:
    """Best-effort immutable binding for a failed-report replacement receipt."""

    try:
        relative = (
            _safe_relative(row["report_manifest_relative_path"], "report manifest")
            if combined
            else PurePosixPath("records")
            / recording_id
            / "report"
            / "manifest.json"
        )
        path = _resolve_regular(root, relative)
        return _sha256_bytes(path.read_bytes())
    except (AuditDefect, KeyError, OSError, TypeError, ValueError):
        return None


def audit_release(
    *,
    inventory_path: Path,
    coverage_path: Path,
    output_path: Path,
    full_root: Path | None = None,
    primary_root: Path | None = None,
    recovery_root: Path | None = None,
    remediation_root: Path | None = None,
    primary_release_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Audit selected completed EEG reports and atomically write one aggregate."""

    inventory_value, inventory_snapshot = _read_snapshot(inventory_path)
    inventory = batch.validate_inventory(inventory_value)
    coverage_value, coverage_snapshot = _read_snapshot(coverage_path)
    kind, coverage = _validate_coverage_input(
        coverage_value,
        inventory=inventory,
        inventory_sha256=inventory_snapshot[1],
    )
    snapshots: list[tuple[Path, str]] = [inventory_snapshot, coverage_snapshot]
    audit_mode = (
        REMEDIATION_AUDIT_MODE
        if primary_release_audit_path is not None
        else COHORT_AUDIT_MODE
    )
    primary_audit: dict[str, Any] | None = None
    primary_audit_snapshot: tuple[Path, str] | None = None
    all_primary_authorizations: list[dict[str, Any]] = []
    remediation_authorizations: list[dict[str, Any]] = []
    if primary_release_audit_path is not None:
        if kind != "full":
            raise ValueError("remediation subset audit requires full-schema coverage")
        (
            primary_audit,
            primary_audit_snapshot,
            all_primary_authorizations,
            remediation_authorizations,
        ) = _remediation_authorization(
            primary_release_audit_path,
            inventory=inventory,
            inventory_sha256=inventory_snapshot[1],
        )
        snapshots.append(primary_audit_snapshot)

    if kind == "full":
        if any(
            root is not None
            for root in (primary_root, recovery_root, remediation_root)
        ):
            raise ValueError(
                "primary/recovery/remediation roots apply only to combined coverage"
            )
        roots = {
            "full": _regular_root(
                full_root if full_root is not None else coverage_snapshot[0].parent,
                "full report root",
            )
        }
    else:
        if full_root is not None:
            raise ValueError("full root applies only to full coverage")
        selected_sources = {
            str(row["artifact_source"])
            for row in coverage["records"]
            if row["diagnostic_status"] in overlay.ALL_COMPLETED_STATUSES
        }
        supplied = {
            "primary": primary_root,
            "recovery": recovery_root,
            "remediation": remediation_root,
        }
        if any(supplied.get(source) is None for source in selected_sources):
            raise ValueError("combined coverage requires every selected artifact root")
        roots = {
            source: _regular_root(supplied[source], f"{source} report root")
            for source in selected_sources
            if supplied[source] is not None
        }

    output = output_path.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for root in roots.values():
        records_root = root / "records"
        if output == records_root or output.is_relative_to(records_root):
            raise ValueError("release audit output must not modify a report tree")

    inventory_by_id = {
        str(record["recording_id"]): record for record in inventory["records"]
    }
    remediation_by_id = {
        str(item["recording_id"]): item for item in remediation_authorizations
    }
    selected_ids = sorted(
        str(row["recording_id"])
        for row in coverage["records"]
        if row["diagnostic_status"] in overlay.EEG_STATUSES
    )
    expected_remediation_ids = sorted(remediation_by_id)
    selected_set_matches = selected_ids == expected_remediation_ids
    if audit_mode == REMEDIATION_AUDIT_MODE:
        if not selected_set_matches:
            raise ValueError(
                "remediation coverage selected set differs from primary authorization"
            )
        if coverage["technical_unassessable_report_count"] != 0:
            raise ValueError("remediation coverage cannot contain technical artifacts")
    issue_counts: Counter[str] = Counter()
    failure_layer_counts: Counter[str] = Counter()
    remediation_counts: Counter[str] = Counter()
    failed_reports: list[dict[str, Any]] = []
    selected_report_results: list[dict[str, Any]] = []
    audited = 0
    passed = 0
    technical_audited = 0
    technical_passed = 0
    totals: Counter[str] = Counter()
    for row in coverage["records"]:
        if row["diagnostic_status"] == overlay.TECHNICAL_STATUS:
            technical_audited += 1
            source = "full" if kind == "full" else str(row["artifact_source"])
            try:
                artifact_count = _audit_one_technical_report(
                    root=roots[source],
                    row=row,
                    inventory_record=inventory_by_id[str(row["recording_id"])],
                    combined=kind == "combined",
                    snapshots=snapshots,
                )
            except (AuditDefect, KeyError, OSError, TypeError, ValueError):
                issue_counts["technical_artifact_invalid"] += 1
                failure_layer, remediation = _FAILURE_CLASSIFICATION[
                    "technical_artifact_invalid"
                ]
                failure_layer_counts[failure_layer] += 1
                remediation_counts[remediation] += 1
            else:
                technical_passed += 1
                totals["technical_artifact_count"] += artifact_count
            continue
        if row["diagnostic_status"] not in overlay.EEG_STATUSES:
            continue
        audited += 1
        source = "full" if kind == "full" else str(row["artifact_source"])
        code: str | None = None
        try:
            result = _audit_one_report(
                root=roots[source],
                row=row,
                inventory_record=inventory_by_id[str(row["recording_id"])],
                combined=kind == "combined",
                snapshots=snapshots,
            )
        except AuditDefect as exc:
            code = exc.code
        except (KeyError, OSError, TypeError, ValueError) as exc:
            del exc
            code = "unexpected_report_validation_failure"
        else:
            remediated_manifest_sha = _selected_report_manifest_sha256(
                root=roots[source],
                row=row,
                recording_id=str(row["recording_id"]),
                combined=kind == "combined",
            )
            if audit_mode == REMEDIATION_AUDIT_MODE:
                authorization = remediation_by_id[str(row["recording_id"])]
                source_manifest_sha = authorization[
                    "source_report_manifest_sha256"
                ]
                if (
                    remediated_manifest_sha is None
                    or remediated_manifest_sha == source_manifest_sha
                ):
                    code = "remediation_report_not_fresh"
                else:
                    selected_report_results.append(
                        {
                            "recording_id": str(row["recording_id"]),
                            "audit_status": "passed",
                            "diagnostic_status": row["diagnostic_status"],
                            "event_count": row["event_count"],
                            "remediated_report_manifest_sha256": remediated_manifest_sha,
                            "source_primary_report_manifest_sha256": source_manifest_sha,
                            "failure_reason_codes": authorization[
                                "failure_reason_codes"
                            ],
                            "failure_layer": authorization["failure_layer"],
                            "minimum_remediation": authorization[
                                "minimum_remediation"
                            ],
                        }
                    )
            if code == "remediation_report_not_fresh":
                pass
            else:
                passed += 1
                totals.update(result)
                continue
        assert code is not None
        issue_counts[code] += 1
        failure_layer, remediation = _FAILURE_CLASSIFICATION[code]
        failure_layer_counts[failure_layer] += 1
        remediation_counts[remediation] += 1
        recording_id = str(row["recording_id"])
        failed_reports.append(
            {
                "recording_id": recording_id,
                "selected_artifact_source": (
                    "primary" if kind == "full" else str(row["artifact_source"])
                ),
                "source_report_manifest_sha256": _selected_report_manifest_sha256(
                    root=roots[source],
                    row=row,
                    recording_id=recording_id,
                    combined=kind == "combined",
                ),
                "failure_reason_codes": [code],
                "failure_layer": failure_layer,
                "minimum_remediation": remediation,
            }
        )
        if audit_mode == REMEDIATION_AUDIT_MODE:
            authorization = remediation_by_id.get(recording_id)
            selected_report_results.append(
                {
                    "recording_id": recording_id,
                    "audit_status": "failed",
                    "diagnostic_status": row["diagnostic_status"],
                    "event_count": row["event_count"],
                    "remediated_report_manifest_sha256": (
                        _selected_report_manifest_sha256(
                            root=roots[source],
                            row=row,
                            recording_id=recording_id,
                            combined=kind == "combined",
                        )
                    ),
                    "source_primary_report_manifest_sha256": (
                        authorization.get("source_report_manifest_sha256")
                        if authorization is not None
                        else None
                    ),
                    "failure_reason_codes": [code],
                    "failure_layer": failure_layer,
                    "minimum_remediation": remediation,
                }
            )

    _assert_unchanged(snapshots)
    completed_eeg = int(coverage["completed_eeg_report_count"])
    completed_technical = int(coverage["technical_unassessable_report_count"])
    artifact_complete = bool(coverage["dataset_artifact_coverage_complete"])
    all_completed_valid = audited == completed_eeg and passed == audited
    all_technical_valid = (
        technical_audited == completed_technical
        and technical_passed == technical_audited
    )
    remediation_release_ready = (
        audit_mode == REMEDIATION_AUDIT_MODE
        and selected_set_matches
        and audited == len(remediation_authorizations)
        and passed == audited
        and completed_technical == 0
    )
    release_ready = (
        remediation_release_ready
        if audit_mode == REMEDIATION_AUDIT_MODE
        else artifact_complete and all_completed_valid and all_technical_valid
    )
    source_receipts = {
        "inventory_manifest_sha256": inventory_snapshot[1],
        "coverage_manifest_sha256": coverage_snapshot[1],
        "source_paths_persisted": False,
        "patient_pseudonyms_persisted": False,
        "failed_recording_ids_persisted_for_replacement_authorization": bool(
            failed_reports
        ),
    }
    remediation_scope: dict[str, Any] | None = None
    if audit_mode == REMEDIATION_AUDIT_MODE:
        assert primary_audit is not None and primary_audit_snapshot is not None
        primary_receipt = primary_audit["replacement_authorization_receipt"]
        source_receipts["primary_release_audit_sha256"] = primary_audit_snapshot[1]
        remediation_scope = {
            "schema_version": REMEDIATION_SCOPE_SCHEMA,
            "source_primary_release_audit_id": primary_audit["audit_id"],
            "source_primary_release_audit_sha256": primary_audit_snapshot[1],
            "source_replacement_authorization_id": primary_receipt[
                "authorization_id"
            ],
            "source_authorization_policy_id": AUTHORIZATION_POLICY_ID,
            "authorization_set_sha256": _canonical_sha256(
                all_primary_authorizations
            ),
            "expected_authorized_recording_count": len(expected_remediation_ids),
            "expected_authorized_recording_ids": expected_remediation_ids,
            "selected_coverage_recording_count": len(selected_ids),
            "selected_coverage_recording_ids": selected_ids,
            "selected_set_exactly_matches_authorization": selected_set_matches,
            "no_extra_or_missing_selected_reports": selected_set_matches,
            "only_language_or_renderer_rerender_authorized": all(
                item["failure_layer"] == "language_or_renderer_projection"
                and item["minimum_remediation"] == "rerender_report_only"
                for item in remediation_authorizations
            ),
        }
    body = {
        "schema_version": SCHEMA_VERSION,
        "audit_mode": audit_mode,
        "status": PASS_STATUS if release_ready else FAIL_STATUS,
        "release_ready": release_ready,
        "remediation_release_ready": remediation_release_ready,
        "coverage_kind": kind,
        "inventory_id": inventory["inventory_id"],
        "recording_unit_policy": inventory["recording_unit_policy"],
        "cohort_counts": {
            "expected_record_count": inventory["record_count"],
            "expected_subject_count": inventory["subject_count"],
            "completed_eeg_report_count": completed_eeg,
            "technical_unassessable_report_count": int(
                coverage["technical_unassessable_report_count"]
            ),
            "pending_or_not_run_count": int(coverage["pending_or_not_run_count"]),
            "completed_eeg_reports_audited": audited,
            "completed_eeg_reports_passed": passed,
            "completed_eeg_reports_failed": audited - passed,
            "technical_reports_audited": technical_audited,
            "technical_reports_passed": technical_passed,
            "technical_reports_failed": technical_audited - technical_passed,
        },
        "diagnostic_status_counts": dict(coverage["diagnostic_status_counts"]),
        "language_totals": {
            "event_count_in_passed_reports": totals["event_count"],
            "validated_qwen_wording_count": totals[
                "validated_qwen_wording_count"
            ],
            "deterministic_language_fallback_count": totals[
                "deterministic_language_fallback_count"
            ],
            "deterministic_without_language_count": totals[
                "deterministic_without_language_count"
            ],
            "deterministically_rendered_event_count": (
                totals["deterministic_language_fallback_count"]
                + totals["deterministic_without_language_count"]
            ),
        },
        "artifact_totals": {
            "verified_artifact_count_in_passed_reports": totals["artifact_count"],
            "verified_waveform_count_in_passed_reports": totals["waveform_count"],
            "verified_artifact_count_in_passed_technical_reports": totals[
                "technical_artifact_count"
            ],
        },
        "failure_code_counts": dict(sorted(issue_counts.items())),
        "failure_layer_counts": dict(sorted(failure_layer_counts.items())),
        "minimum_remediation_counts": dict(sorted(remediation_counts.items())),
        "failed_reports": sorted(
            failed_reports,
            key=lambda item: str(item["recording_id"]),
        ),
        "checks": {
            "inventory_schema_and_binding_validated": True,
            "coverage_schema_and_binding_validated": True,
            "every_completed_eeg_row_audited": audited == completed_eeg,
            "all_completed_eeg_reports_valid": all_completed_valid,
            "every_technical_report_row_audited": (
                technical_audited == completed_technical
            ),
            "all_technical_reports_valid": all_technical_valid,
            "dataset_artifact_coverage_complete": artifact_complete,
            "dataset_eeg_coverage_complete": bool(
                coverage["dataset_eeg_coverage_complete"]
            ),
            "current_neutral_producer_contract_enforced": True,
            "facts_locked_language_projection_revalidated": True,
            "html_docx_json_and_waveforms_revalidated": True,
            "source_artifact_snapshots_unchanged": True,
        },
        "source_receipts": source_receipts,
        "remediation_scope": remediation_scope,
        "selected_report_results": (
            sorted(selected_report_results, key=lambda item: item["recording_id"])
            if audit_mode == REMEDIATION_AUDIT_MODE
            else []
        ),
        "scope_receipt": {
            "edf_signal_files_read": False,
            "edf_annotations_read": False,
            "excel_or_workbook_read": False,
            "onset_label_or_ground_truth_read": False,
            "inventory_source_locator_resolved_or_opened": False,
            "inventory_source_locator_projected_into_audit": False,
            "report_artifacts_read_only": True,
            "report_artifacts_modified": False,
            "aggregate_plus_pseudonymous_failure_authorization_only": True,
        },
        "replacement_authorization_receipt": _replacement_authorization_receipt(
            authorizations=failed_reports,
            eligible=audit_mode == COHORT_AUDIT_MODE and kind == "full",
        ),
    }
    audit = {
        **body,
        "audit_id": AUDIT_ID_PREFIX + _canonical_sha256(body)[:24],
    }
    batch._atomic_json(output, audit, replace=False)  # noqa: SLF001
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-root", type=Path)
    parser.add_argument("--primary-root", type=Path)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--remediation-root", type=Path)
    parser.add_argument("--primary-release-audit", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = audit_release(
        inventory_path=args.inventory,
        coverage_path=args.coverage,
        output_path=args.output,
        full_root=args.full_root,
        primary_root=args.primary_root,
        recovery_root=args.recovery_root,
        remediation_root=args.remediation_root,
        primary_release_audit_path=args.primary_release_audit,
    )
    print(
        json.dumps(
            {
                "audit_id": result["audit_id"],
                "status": result["status"],
                "release_ready": result["release_ready"],
                **result["cohort_counts"],
                "failure_code_counts": result["failure_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["release_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "audit_release", "main"]
