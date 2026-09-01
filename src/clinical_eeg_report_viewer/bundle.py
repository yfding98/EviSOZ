"""Build an immutable, de-identified web bundle from frozen EEG reports.

The builder accepts only a release-audited report coverage manifest and
optional *post-freeze* typed evaluation artifacts.  It never opens an EDF,
EDF annotation, workbook, raw clinical note, or source inventory locator.
Only hash-bound report HTML and PNG waveform artifacts are copied.  Physician
labels are projected from closed-vocabulary evaluation fields into separate
viewer JSON after the report bytes have been verified as frozen.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tempfile
from typing import Any, Mapping, Sequence
import zlib

from src.clinical_eeg_long_recording.research_soz_evidence import (
    DESCRIPTIVE_EVIDENCE_LEVELS,
    RESEARCH_SOZ_EVIDENCE_POLICY_ID,
    validate_research_soz_descriptive_strength,
)
from src.clinical_eeg_long_recording.research_soz_prediction import (
    C18_ELECTRODES,
    RESEARCH_SOZ_PREDICTION_METHOD_ID,
    validate_research_soz_prediction_artifact,
)


LEGACY_RELEASE_BUNDLE_SCHEMA_VERSION = "clinical_eeg_report_viewer_release_bundle_v1"
RELEASE_BUNDLE_SCHEMA_VERSION = "clinical_eeg_report_viewer_release_bundle_v1_1"
INDEX_SCHEMA_VERSION = "clinical_eeg_report_viewer_index_v1_1"
DETAIL_SCHEMA_VERSION = "clinical_eeg_report_viewer_record_detail_v1_1"
RELEASE_AUDIT_SCHEMA_VERSION = "private_long_recording_report_release_audit_v1"
DOCTOR_LABEL_RELEASE_SCHEMA_VERSION = (
    "private_postfreeze_doctor_label_release_bundle_v1"
)
RESEARCH_SOZ_BATCH_SCHEMA_VERSION = (
    "private_long_recording_research_soz_sidecar_batch_v1_1"
)
RESEARCH_SOZ_BATCH_STATUS = "completed_research_soz_sidecar_batch"
RESEARCH_SOZ_PROJECTION_SCHEMA_VERSION = (
    "clinical_eeg_viewer_research_scalp_onset_projection_v1"
)

DIRECT_COVERAGE_SCHEMA_VERSION = "private_long_recording_report_coverage_v1"
COMBINED_COVERAGE_SCHEMA_VERSION = (
    "private_long_recording_report_combined_coverage_v1"
)
SUPPORTED_COVERAGE_SCHEMAS = frozenset(
    {DIRECT_COVERAGE_SCHEMA_VERSION, COMBINED_COVERAGE_SCHEMA_VERSION}
)

TECHNICAL_STATUS = "completed_technical_unassessable"
EEG_STATUSES = frozenset(
    {
        "completed_localizable",
        "completed_nonlocalizable",
        "completed_insufficient_evidence",
    }
)
COMPLETED_STATUSES = EEG_STATUSES | {TECHNICAL_STATUS}
COMPARISON_STATUSES = frozenset(
    {"match", "partial_match", "mismatch", "not_available"}
)
_EVALUATION_DISPOSITIONS = frozenset(
    {
        "exact_or_compatible",
        "mismatch",
        "generated_abstention",
        "label_missing",
        "technical_unassessable",
        "source_conflict",
        "ambiguous_mapping",
    }
)
_DOCTOR_REFERENCE_STATUSES = frozenset(
    {"doctor_clear", "doctor_uncertain", "label_missing"}
)
_GENERATED_REPORT_STATUSES = frozenset(
    {
        "generated_localization",
        "generated_nonfocal_conclusion",
        "generated_abstention",
        "technical_unassessable",
    }
)
_ALIGNMENT_CODES = frozenset(
    {
        "exact_spatial_match",
        "compatible_spatial_overlap",
        "spatial_mismatch",
        "uncertainty_aligned",
        "doctor_uncertain_but_generated_localization",
        "generated_abstention_not_scored_as_conflict",
        "label_not_comparable",
        "technical_unassessable",
        "source_conflict_not_scored",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/]")
_ABSOLUTE_PATH_MARKERS = ("file://", "/mnt/", "/home/", "/data/")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_LATERALITIES = frozenset(
    {"left", "right", "bilateral", "midline", "none", "indeterminate"}
)
_REGIONS = frozenset(
    {
        "frontal",
        "temporal",
        "central",
        "parietal",
        "occipital",
        "frontotemporal",
        "centrotemporal",
        "temporoparietal",
        "posterior",
        "diffuse",
        "midline",
        "unknown",
    }
)
_ONSET_UNCERTAINTY = frozenset(
    {"clear", "uncertain_or_unclear", "indeterminate"}
)
_ELECTRODES = frozenset(
    {
        "FP1",
        "FP2",
        "F7",
        "F8",
        "F3",
        "F4",
        "FZ",
        "C3",
        "C4",
        "CZ",
        "T7",
        "T8",
        "P7",
        "P8",
        "P3",
        "P4",
        "PZ",
        "O1",
        "O2",
        "M1",
        "M2",
    }
)


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid constant {value!r}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a de-identified identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _safe_relative(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{context} is not a safe relative path")
    return relative


def _regular_file(root: Path, relative: PurePosixPath, context: str) -> Path:
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{context} path contains a symlink")
    resolved = cursor.resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return resolved


def _snapshot_json(path: str | Path, context: str) -> dict[str, Any]:
    raw_path = Path(path)
    if raw_path.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    resolved = raw_path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{context} must be a regular file")
    raw = resolved.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_invalid_constant,
    )
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must contain a JSON object")
    return {
        "path": resolved,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "value": dict(value),
    }


def _assert_unchanged(snapshots: Sequence[Mapping[str, Any]]) -> None:
    for snapshot in snapshots:
        path = Path(snapshot["path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError("a frozen source changed while the viewer bundle was built")
        if _file_sha256(path) != snapshot["sha256"]:
            raise ValueError("a frozen source changed while the viewer bundle was built")


class _ReportHTMLGate(HTMLParser):
    """Reject active content and non-local resources in a frozen report."""

    _BLOCKED_TAGS = frozenset(
        {
            "script",
            "iframe",
            "object",
            "embed",
            "link",
            "base",
            "form",
            "input",
            "button",
            "textarea",
            "select",
            "audio",
            "video",
            "svg",
            "math",
        }
    )

    def __init__(self, allowed_images: frozenset[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.allowed_images = allowed_images

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        lowered = tag.lower()
        if lowered in self._BLOCKED_TAGS:
            raise ValueError(f"report HTML contains blocked tag <{lowered}>")
        attribute_map = {name.lower(): value or "" for name, value in attrs}
        if lowered == "meta" and "http-equiv" in attribute_map:
            raise ValueError("report HTML contains an active meta directive")
        for raw_name, raw_value in attrs:
            name = raw_name.lower()
            value = raw_value or ""
            if name.startswith("on") or name in {
                "srcdoc",
                "srcset",
                "integrity",
                "background",
            }:
                raise ValueError("report HTML contains an active-content attribute")
            if name == "style" and "url(" in value.lower():
                raise ValueError("report HTML inline style references an external URL")
            if name in {"src", "href", "action", "poster"}:
                if lowered == "img" and name == "src":
                    if value not in self.allowed_images:
                        raise ValueError("report HTML references an unverified image")
                elif value:
                    raise ValueError("report HTML contains a non-image resource link")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)


def _validate_report_html(raw: bytes, allowed_images: frozenset[str]) -> None:
    text = raw.decode("utf-8")
    lowered = text.lower()
    if any(marker in lowered for marker in _ABSOLUTE_PATH_MARKERS):
        raise ValueError("report HTML contains an absolute filesystem locator")
    if _WINDOWS_PATH_RE.search(text):
        raise ValueError("report HTML contains a Windows filesystem locator")
    parser = _ReportHTMLGate(allowed_images)
    parser.feed(text)
    parser.close()


def _sanitize_png(raw: bytes) -> bytes:
    """Validate a PNG and drop all non-display ancillary metadata chunks."""

    if not raw.startswith(_PNG_SIGNATURE):
        raise ValueError("waveform artifact is not a PNG")
    cursor = len(_PNG_SIGNATURE)
    chunks: list[bytes] = []
    first = True
    seen_idat = False
    seen_iend = False
    keep_ancillary = {b"tRNS", b"gAMA", b"cHRM", b"sRGB", b"pHYs"}
    while cursor < len(raw):
        if len(raw) - cursor < 12:
            raise ValueError("PNG has a truncated chunk")
        length = struct.unpack(">I", raw[cursor : cursor + 4])[0]
        end = cursor + 12 + length
        if end > len(raw):
            raise ValueError("PNG chunk extends beyond the file")
        chunk_type = raw[cursor + 4 : cursor + 8]
        data = raw[cursor + 8 : cursor + 8 + length]
        expected_crc = struct.unpack(">I", raw[cursor + 8 + length : end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("PNG chunk checksum mismatch")
        if first:
            if chunk_type != b"IHDR" or length != 13:
                raise ValueError("PNG must start with one IHDR chunk")
            width, height = struct.unpack(">II", data[:8])
            if width < 1 or height < 1 or width > 20000 or height > 20000:
                raise ValueError("PNG dimensions are outside the viewer limit")
            if width * height > 100_000_000:
                raise ValueError("PNG pixel count is outside the viewer limit")
            first = False
        elif chunk_type == b"IHDR":
            raise ValueError("PNG contains repeated IHDR chunks")

        critical = (chunk_type[0] & 0x20) == 0
        if chunk_type == b"IDAT":
            seen_idat = True
        if chunk_type == b"IEND":
            if length != 0 or seen_iend:
                raise ValueError("PNG contains an invalid IEND chunk")
            seen_iend = True
        if critical and chunk_type not in {b"IHDR", b"PLTE", b"IDAT", b"IEND"}:
            raise ValueError("PNG contains an unknown critical chunk")
        if critical or chunk_type in keep_ancillary:
            chunks.append(raw[cursor:end])
        cursor = end
        if seen_iend:
            break
    if not seen_idat or not seen_iend or cursor != len(raw):
        raise ValueError("PNG is incomplete or has trailing bytes")
    return _PNG_SIGNATURE + b"".join(chunks)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    os.chmod(path, 0o600)


def _write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    os.chmod(path, 0o600)


def _validate_coverage(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("coverage manifest must be an object")
    data = dict(value)
    if data.get("schema_version") not in SUPPORTED_COVERAGE_SCHEMAS:
        raise ValueError("coverage manifest schema is unsupported")
    if data.get("dataset_artifact_coverage_complete") is not True:
        raise ValueError("viewer requires complete report artifact coverage")
    if data.get("pending_or_not_run_count") != 0:
        raise ValueError("viewer cannot publish a cohort with pending records")
    records = data.get("records")
    expected = _nonnegative_int(data.get("expected_record_count"), "expected count")
    if not isinstance(records, list) or len(records) != expected:
        raise ValueError("coverage record set does not close")
    seen: set[str] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise TypeError(f"coverage records[{index}] must be an object")
        recording_id = _identifier(raw.get("recording_id"), "recording_id")
        _identifier(raw.get("patient_pseudonym"), "patient pseudonym")
        if recording_id in seen:
            raise ValueError("coverage contains a duplicate recording_id")
        seen.add(recording_id)
        if raw.get("diagnostic_status") not in COMPLETED_STATUSES:
            raise ValueError("coverage contains a non-completed report row")
        _nonnegative_int(raw.get("event_count"), "event_count")
    return data


def _validate_release_audit(
    value: object,
    *,
    coverage: Mapping[str, Any],
    coverage_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("release audit must be an object")
    audit = dict(value)
    if audit.get("schema_version") != RELEASE_AUDIT_SCHEMA_VERSION:
        raise ValueError("release audit schema is unsupported")
    if audit.get("status") != "release_audit_passed" or audit.get(
        "release_ready"
    ) is not True:
        raise ValueError("viewer requires a passing cohort release audit")
    if audit.get("inventory_id") != coverage.get("inventory_id"):
        raise ValueError("release audit inventory binding differs from coverage")
    if audit.get("recording_unit_policy") != coverage.get("recording_unit_policy"):
        raise ValueError("release audit recording policy differs from coverage")
    receipts = audit.get("source_receipts")
    if not isinstance(receipts, Mapping) or receipts.get(
        "coverage_manifest_sha256"
    ) != coverage_sha256:
        raise ValueError("release audit is not bound to this coverage manifest")
    counts = audit.get("cohort_counts")
    if not isinstance(counts, Mapping) or counts.get(
        "expected_record_count"
    ) != coverage.get("expected_record_count"):
        raise ValueError("release audit cohort count differs from coverage")
    scope = audit.get("scope_receipt")
    required_false = (
        "edf_signal_files_read",
        "edf_annotations_read",
        "excel_or_workbook_read",
        "onset_label_or_ground_truth_read",
        "inventory_source_locator_resolved_or_opened",
        "report_artifacts_modified",
    )
    if not isinstance(scope, Mapping) or any(scope.get(key) is not False for key in required_false):
        raise ValueError("release audit violates the signal/report source boundary")
    return audit


def _root_map(values: Mapping[str, str | Path]) -> dict[str, Path]:
    if not values:
        raise ValueError("at least one artifact root is required")
    roots: dict[str, Path] = {}
    for raw_name, raw_path in values.items():
        name = _identifier(raw_name, "artifact root name")
        path = Path(raw_path)
        if path.is_symlink():
            raise ValueError("artifact roots must not be symlinks")
        resolved = path.resolve(strict=True)
        if not resolved.is_dir() or resolved.is_symlink():
            raise ValueError("artifact roots must be regular directories")
        roots[name] = resolved
    return roots


def _source_manifest_location(
    coverage: Mapping[str, Any],
    row: Mapping[str, Any],
    roots: Mapping[str, Path],
) -> tuple[str, Path, PurePosixPath]:
    recording_id = str(row["recording_id"])
    if coverage["schema_version"] == COMBINED_COVERAGE_SCHEMA_VERSION:
        source = _identifier(row.get("artifact_source"), "artifact_source")
        relative = _safe_relative(
            row.get("report_manifest_relative_path"),
            "combined report manifest path",
        )
    else:
        if "default" in roots:
            source = "default"
        elif "full" in roots and len(roots) == 1:
            source = "full"
        elif len(roots) == 1:
            source = next(iter(roots))
        else:
            raise ValueError("direct coverage requires a default artifact root")
        if row["diagnostic_status"] in EEG_STATUSES:
            relative = PurePosixPath("records") / recording_id / "report" / "manifest.json"
        else:
            technical = _safe_relative(
                row.get("technical_artifact_relative_dir"),
                "technical artifact directory",
            )
            relative = PurePosixPath("records") / recording_id / technical / "manifest.json"
    if source not in roots:
        raise ValueError(f"no artifact root was supplied for source {source!r}")
    expected_prefix = ("records", recording_id)
    if relative.parts[:2] != expected_prefix or relative.name != "manifest.json":
        raise ValueError("report manifest path escapes its recording directory")
    return source, roots[source], relative


def _validate_report_manifest(
    manifest: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    recording_id = str(row["recording_id"])
    subject_id = str(row["patient_pseudonym"])
    if manifest.get("recording_id") != recording_id or manifest.get(
        "patient_pseudonym"
    ) != subject_id:
        raise ValueError("report manifest pseudonymous identity drifted")
    if manifest.get("diagnostic_status") != row["diagnostic_status"]:
        raise ValueError("report manifest diagnostic status drifted")
    if manifest.get("event_count") != row["event_count"]:
        raise ValueError("report manifest event count drifted")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("report manifest artifacts must be an object")
    normalized: dict[str, str] = {}
    for raw_relative, raw_digest in artifacts.items():
        relative = _safe_relative(raw_relative, "report artifact path")
        normalized[relative.as_posix()] = _sha256(raw_digest, "artifact digest")
    if "report.html" not in normalized:
        raise ValueError("report manifest does not contain report.html")
    diagnostic = row["diagnostic_status"]
    if diagnostic in EEG_STATUSES:
        if manifest.get("status") != "completed_unsigned_ai_draft":
            raise ValueError("EEG report manifest is not completed")
        scope = manifest.get("scope_receipt")
        required = {
            "eeg_signal_only_generation": True,
            "external_edf_annotations_loaded": False,
            "excel_observations_loaded": False,
            "source_context_joined_post_freeze": False,
            "source_context_sent_to_qwen": False,
            "research_soz_used_in_clinical_facts_or_llm": False,
            "physician_signed": False,
        }
        if not isinstance(scope, Mapping) or any(
            scope.get(key) is not expected for key, expected in required.items()
        ):
            raise ValueError("EEG report manifest violates the EEG-only boundary")
        return "eeg_report", normalized
    if manifest.get("status") != TECHNICAL_STATUS:
        raise ValueError("technical report manifest is not completed")
    return "technical_report", normalized


def _controlled_list(
    value: object,
    *,
    allowed: frozenset[str],
    context: str,
) -> list[str]:
    if value in (None, "not_available"):
        return []
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a controlled list or not_available")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in allowed:
            raise ValueError(f"{context} contains an unsupported controlled value")
        if item in result:
            raise ValueError(f"{context} contains duplicate values")
        result.append(item)
    return result


def _overall_consistency(statuses: Sequence[str]) -> str:
    available = [status for status in statuses if status != "not_available"]
    if not available:
        return "not_available"
    if "mismatch" in available:
        return "mismatch"
    if "partial_match" in available:
        return "partial_match"
    return "match"


def _empty_evaluation(event_count: int) -> dict[str, Any]:
    return {
        "status": "not_available",
        "events": [],
        "unmatched_reference_count": 0,
        "withheld_conflicting_label_count": 0,
        "record_consistency_disposition": "label_missing",
        "location_consistency": "not_available",
        "onset_certainty_consistency": "not_available",
        "expected_report_event_count": event_count,
        "claim_boundary": {
            "postfreeze_only": True,
            "typed_deidentified_values_only": True,
            "raw_excel_text_included": False,
            "edf_annotations_included": False,
            "source_paths_included": False,
            "used_for_report_generation": False,
        },
    }


def _label_spatial_comparison(
    value: object,
    *,
    field: str,
    allowed: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("field") != field:
        raise ValueError(f"doctor-label {field} consistency shape drifted")
    status = value.get("status")
    if status not in COMPARISON_STATUSES:
        raise ValueError(f"doctor-label {field} consistency status is unsupported")
    return {
        "field": field,
        "status": status,
        "report_values": _controlled_list(
            value.get("report_values"),
            allowed=allowed,
            context=f"doctor-label {field} report values",
        ),
        "doctor_values": _controlled_list(
            value.get("doctor_values"),
            allowed=allowed,
            context=f"doctor-label {field} doctor values",
        ),
    }


def _project_doctor_label_record(
    value: object,
    *,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("doctor-label record must be an object")
    recording_id = str(prepared["recording_id"])
    if value.get("recording_id") != recording_id or value.get(
        "patient_pseudonym"
    ) != prepared["subject_id"]:
        raise ValueError("doctor-label record pseudonymous identity drifted")
    report = value.get("report_receipt")
    if not isinstance(report, Mapping):
        raise TypeError("doctor-label report receipt must be an object")
    expected_report_kind = (
        "eeg_report"
        if prepared["report_kind"] == "eeg_report"
        else "technical_unassessable_report"
    )
    expected_report = {
        "artifact_source": prepared["source_name"],
        "report_kind": expected_report_kind,
        "diagnostic_status": prepared["row"]["diagnostic_status"],
        "event_count": prepared["row"]["event_count"],
        "report_manifest_relative_path": prepared["manifest_relative"],
        "report_manifest_sha256": prepared["manifest_sha256"],
        "report_html_relative_path": prepared["source_report_relative"],
        "report_html_sha256": prepared["report_html_sha256"],
    }
    if any(report.get(key) != expected for key, expected in expected_report.items()):
        raise ValueError("doctor-label release is bound to another report artifact")
    label_status = value.get("doctor_label_status")
    if label_status not in {
        "available",
        "not_available",
        "source_conflict",
        "ambiguous_mapping",
    }:
        raise ValueError("doctor-label status is unsupported")
    raw_labels = value.get("doctor_labels")
    if not isinstance(raw_labels, list) or value.get("doctor_label_count") != len(
        raw_labels
    ):
        raise ValueError("doctor-label count does not close")
    if label_status in {"not_available", "ambiguous_mapping"} and raw_labels:
        raise ValueError("closed doctor-label record publishes labels")
    if label_status == "available" and not raw_labels:
        raise ValueError("available doctor-label record has no labels")
    if label_status == "source_conflict" and not raw_labels:
        raise ValueError("source-conflict doctor-label record has no variants")

    projected_labels: list[dict[str, Any]] = []
    location_statuses: list[str] = []
    certainty_statuses: list[str] = []
    seen_slots: set[str] = set()
    seen_label_ids: set[str] = set()
    for raw in raw_labels:
        if not isinstance(raw, Mapping):
            raise TypeError("doctor label must be an object")
        label_id = _identifier(raw.get("label_id"), "doctor label ID")
        slot = _identifier(raw.get("source_event_slot"), "doctor source event slot")
        if (
            not re.fullmatch(r"SZ[1-9][0-9]*", slot)
            or label_id in seen_label_ids
            or (slot in seen_slots and label_status != "source_conflict")
        ):
            raise ValueError("doctor source slot/label identity is invalid or repeated")
        seen_slots.add(slot)
        seen_label_ids.add(label_id)
        conflict_variant = raw.get("source_conflict_variant", False)
        evaluation_eligible = raw.get("evaluation_eligible", True)
        if not isinstance(conflict_variant, bool) or not isinstance(
            evaluation_eligible, bool
        ) or evaluation_eligible is conflict_variant:
            raise ValueError("doctor label conflict/evaluation flags are invalid")
        if label_status == "source_conflict" and not conflict_variant:
            raise ValueError("source-conflict record contains an eligible label")
        if label_status == "available" and conflict_variant:
            raise ValueError("available record contains a conflict variant")
        onset = raw.get("onset")
        if not isinstance(onset, Mapping):
            raise TypeError("doctor onset projection must be an object")
        onset_status = onset.get("status")
        if onset_status not in {"available", "not_available"}:
            raise ValueError("doctor onset status is unsupported")
        if onset.get("raw_text_included") is not False:
            raise ValueError("doctor onset projection includes raw free text")
        laterality = onset.get("laterality")
        uncertainty = onset.get("onset_uncertainty")
        if laterality not in _LATERALITIES or uncertainty not in _ONSET_UNCERTAINTY:
            raise ValueError("doctor onset projection contains unsupported codes")
        regions = _controlled_list(
            onset.get("regions"),
            allowed=_REGIONS,
            context="doctor onset regions",
        )
        channels = raw.get("physician_channel_reference")
        if not isinstance(channels, Mapping):
            raise TypeError("physician channel reference must be an object")
        if channels.get("evaluation_only") is not True or channels.get(
            "eligible_for_report_body"
        ) is not False or channels.get("eligible_for_llm") is not False:
            raise ValueError("physician channels are not confined to evaluation")
        significant = _controlled_list(
            channels.get("significant_electrodes"),
            allowed=_ELECTRODES,
            context="physician significant electrodes",
        )
        spread = _controlled_list(
            channels.get("spread_electrodes"),
            allowed=_ELECTRODES,
            context="physician spread electrodes",
        )
        consistency = raw.get("fact_consistency")
        if not isinstance(consistency, Mapping) or consistency.get(
            "policy_id"
        ) != "selective_recording_level_fact_consistency_v1":
            raise ValueError("doctor-label fact consistency policy drifted")
        overall = consistency.get("overall_status")
        if overall not in COMPARISON_STATUSES:
            raise ValueError("doctor-label overall consistency is unsupported")
        evaluation_disposition = consistency.get("evaluation_disposition")
        doctor_reference_status = consistency.get("doctor_reference_status")
        generated_report_status = consistency.get("generated_report_status")
        alignment_code = consistency.get("alignment_code")
        if evaluation_disposition not in _EVALUATION_DISPOSITIONS:
            raise ValueError("doctor-label evaluation disposition is unsupported")
        if doctor_reference_status not in _DOCTOR_REFERENCE_STATUSES:
            raise ValueError("doctor-label doctor-reference status is unsupported")
        if generated_report_status not in _GENERATED_REPORT_STATUSES:
            raise ValueError("doctor-label generated-report status is unsupported")
        if alignment_code not in _ALIGNMENT_CODES:
            raise ValueError("doctor-label alignment code is unsupported")
        if conflict_variant and (
            evaluation_disposition != "source_conflict"
            or alignment_code != "source_conflict_not_scored"
        ):
            raise ValueError("doctor-label conflict variant is not closed to scoring")
        uncertainty_consistency = consistency.get("onset_uncertainty")
        if not isinstance(uncertainty_consistency, Mapping) or uncertainty_consistency.get(
            "status"
        ) not in {"match", "mismatch", "not_available"}:
            raise ValueError("doctor-label onset consistency is unsupported")
        if uncertainty_consistency.get("doctor_value") != uncertainty:
            raise ValueError("doctor-label uncertainty comparison value drifted")
        spatial = consistency.get("spatial_fields")
        if not isinstance(spatial, list):
            raise TypeError("doctor-label spatial consistency must be a list")
        spatial_by_field = {
            str(item.get("field")): item
            for item in spatial
            if isinstance(item, Mapping)
        }
        projected_comparisons = {
            "laterality": _label_spatial_comparison(
                spatial_by_field.get("laterality"),
                field="laterality",
                allowed=_LATERALITIES,
            ),
            "regions": _label_spatial_comparison(
                spatial_by_field.get("regions"),
                field="regions",
                allowed=_REGIONS,
            ),
            "onset_uncertainty": {
                "field": "onset_uncertainty",
                "status": uncertainty_consistency["status"],
                "report_values": [],
                "doctor_values": (
                    [uncertainty] if onset_status == "available" else []
                ),
            },
        }
        if consistency.get("missing_prediction_is_not_scored_as_mismatch") is not True:
            raise ValueError("doctor-label missing prediction semantics drifted")
        if consistency.get(
            "doctor_unclear_and_report_abstention_is_compatible"
        ) is not True:
            raise ValueError("doctor-label abstention compatibility semantics drifted")
        location_status = _overall_consistency(
            [
                projected_comparisons["laterality"]["status"],
                projected_comparisons["regions"]["status"],
            ]
        )
        location_statuses.append(location_status)
        certainty_statuses.append(str(uncertainty_consistency["status"]))
        projected_labels.append(
            {
                "label_id": label_id,
                "source_event_slot": slot,
                "association_scope": "recording_level_not_detector_candidate",
                "label_status": "available",
                "doctor_onset": {
                    "laterality": (
                        [str(laterality)]
                        if onset_status == "available" and laterality != "indeterminate"
                        else []
                    ),
                    "regions": regions if onset_status == "available" else [],
                    "onset_uncertainty": (
                        [str(uncertainty)] if onset_status == "available" else []
                    ),
                },
                "physician_channels": {
                    "significant": significant,
                    "spread_soft_label": spread,
                    "diffuse_spread_present": channels.get(
                        "diffuse_spread_present"
                    ) is True,
                    "reference_completeness": (
                        "positive_only_unknown_complement"
                        if significant or spread
                        else "not_available"
                    ),
                },
                "report_fact_consistency": projected_comparisons,
                "location_consistency": location_status,
                "onset_certainty_consistency": uncertainty_consistency["status"],
                "overall_fact_consistency": overall,
                "evaluation_disposition": evaluation_disposition,
                "doctor_reference_status": doctor_reference_status,
                "generated_report_status": generated_report_status,
                "alignment_code": alignment_code,
                "research_ranking_metrics": {},
            }
        )
    if label_status == "source_conflict":
        # Conflict variants remain auditable in the sealed doctor-label
        # bundle, but the viewer fails closed and publishes no competing value.
        withheld_conflicting_label_count = len(projected_labels)
        projected_labels = []
        location_statuses = []
        certainty_statuses = []
    else:
        withheld_conflicting_label_count = 0
    record_disposition = value.get("record_consistency_disposition")
    if record_disposition not in _EVALUATION_DISPOSITIONS:
        raise ValueError("record consistency disposition is unsupported")
    if label_status == "ambiguous_mapping" and record_disposition not in {
        "ambiguous_mapping",
        "technical_unassessable",
    }:
        raise ValueError("ambiguous doctor-label mapping disposition drifted")
    if label_status == "source_conflict" and record_disposition not in {
        "source_conflict",
        "technical_unassessable",
    }:
        raise ValueError("source-conflict record disposition drifted")
    return {
        "status": label_status,
        "events": projected_labels,
        "unmatched_reference_count": 0,
        "withheld_conflicting_label_count": withheld_conflicting_label_count,
        "ambiguous_mapping_closed": label_status == "ambiguous_mapping",
        "record_consistency_disposition": record_disposition,
        "location_consistency": _overall_consistency(location_statuses),
        "onset_certainty_consistency": _overall_consistency(certainty_statuses),
        "label_unit": "source_recording_sz_slot",
        "claim_boundary": {
            "postfreeze_only": True,
            "typed_deidentified_values_only": True,
            "raw_excel_text_included": False,
            "edf_annotations_included": False,
            "source_paths_included": False,
            "used_for_report_generation": False,
        },
    }


def _project_doctor_label_release(
    value: object,
    *,
    coverage: Mapping[str, Any],
    coverage_sha256: str,
    release_audit: Mapping[str, Any],
    release_audit_sha256: str,
    prepared_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    if not isinstance(value, Mapping):
        raise TypeError("doctor-label release bundle must be an object")
    if value.get("schema_version") != DOCTOR_LABEL_RELEASE_SCHEMA_VERSION or value.get(
        "status"
    ) != "completed_postfreeze_doctor_label_release":
        raise ValueError("doctor-label release bundle schema/status is unsupported")
    if value.get("inventory_id") != coverage["inventory_id"] or value.get(
        "recording_unit_policy"
    ) != coverage["recording_unit_policy"]:
        raise ValueError("doctor-label release inventory binding drifted")
    if value.get("record_count") != coverage["expected_record_count"] or value.get(
        "subject_count"
    ) != coverage["expected_subject_count"]:
        raise ValueError("doctor-label release cohort count drifted")
    expected_kind = (
        "combined"
        if coverage["schema_version"] == COMBINED_COVERAGE_SCHEMA_VERSION
        else "full"
    )
    if value.get("coverage_kind") != expected_kind:
        raise ValueError("doctor-label release coverage kind drifted")
    receipts = value.get("source_receipts")
    if not isinstance(receipts, Mapping) or receipts.get(
        "coverage_manifest_sha256"
    ) != coverage_sha256 or receipts.get(
        "release_audit_sha256"
    ) != release_audit_sha256 or receipts.get(
        "release_audit_id"
    ) != release_audit.get("audit_id") or receipts.get(
        "source_paths_persisted"
    ) is not False:
        raise ValueError("doctor-label release source receipt binding drifted")
    boundary = value.get("claim_boundary")
    required_boundary = {
        "reports_and_hashes_frozen_before_workbook_open": True,
        "final_release_audit_pass_required": True,
        "report_artifacts_modified": False,
        "generation_pipeline_read_workbook": False,
        "generation_pipeline_read_edf_annotation": False,
        "doctor_label_used_for_detection": False,
        "doctor_label_used_for_candidate_selection": False,
        "doctor_label_used_for_soz_ranking": False,
        "doctor_label_used_for_findings": False,
        "doctor_label_used_for_impression": False,
        "doctor_label_used_for_renderer": False,
        "doctor_label_used_for_llm": False,
        "raw_onset_free_text_included": False,
        "raw_significant_or_spread_text_included": False,
        "raw_patient_identity_included": False,
        "private_edf_path_included": False,
        "workbook_path_sheet_or_row_included": False,
        "edf_annotation_loaded": False,
        "structured_labels_postfreeze_sidecar_only": True,
    }
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) is not expected
        for key, expected in required_boundary.items()
    ):
        raise ValueError("doctor-label release violates the post-freeze boundary")
    leakage = value.get("leakage_gate")
    if not isinstance(leakage, Mapping) or leakage.get("status") != "passed" or any(
        leakage.get(key) is not True
        for key in (
            "forbidden_output_keys_checked",
            "raw_identity_and_private_path_exact_values_checked",
            "absolute_private_path_pattern_checked",
            "closed_vocabulary_onset_projection_only",
            "channel_reference_confined_to_evaluation_only_sidecar",
        )
    ):
        raise ValueError("doctor-label release leakage gate did not pass")
    label_release_id = _identifier(value.get("label_release_id"), "label release ID")
    label_body = {key: deepcopy(item) for key, item in value.items() if key != "label_release_id"}
    if label_release_id != "DRREL-" + _canonical_sha256(label_body)[:24]:
        raise ValueError("doctor-label release ID does not bind its payload")
    raw_records = value.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(prepared_records):
        raise ValueError("doctor-label release record set does not close")
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise TypeError("doctor-label release record must be an object")
        recording_id = _identifier(raw.get("recording_id"), "recording_id")
        if recording_id in raw_by_id:
            raise ValueError("doctor-label release repeats a recording")
        raw_by_id[recording_id] = raw
    association_summary = value.get("association_summary")
    if not isinstance(association_summary, Mapping):
        raise TypeError("doctor-label association summary must be an object")
    expected_status_counts = {
        "record_count": len(raw_records),
        "record_with_doctor_label_count": sum(
            raw.get("doctor_label_status") == "available" for raw in raw_records
        ),
        "record_without_doctor_label_count": sum(
            raw.get("doctor_label_status") == "not_available" for raw in raw_records
        ),
        "record_with_source_conflict_count": sum(
            raw.get("doctor_label_status") == "source_conflict" for raw in raw_records
        ),
        "record_with_ambiguous_mapping_count": sum(
            raw.get("doctor_label_status") == "ambiguous_mapping"
            for raw in raw_records
        ),
    }
    if any(
        association_summary.get(key) != expected
        for key, expected in expected_status_counts.items()
    ):
        raise ValueError("doctor-label association status counts do not close")
    result: dict[str, dict[str, Any]] = {}
    for prepared in prepared_records:
        recording_id = str(prepared["recording_id"])
        if recording_id not in raw_by_id:
            raise ValueError("doctor-label release omits a report recording")
        result[recording_id] = _project_doctor_label_record(
            raw_by_id[recording_id], prepared=prepared
        )
    return result, label_release_id


def _research_claim_boundary() -> dict[str, bool]:
    return {
        "research_scalp_eeg_ranked_hypothesis_only": True,
        "uncalibrated_descriptive_strength_only": True,
        "probability_or_confidence_claim_permitted": False,
        "cortical_soz_or_epileptogenic_zone_claim_permitted": False,
        "diagnosis_or_treatment_target_claim_permitted": False,
        "qualified_eeg_impression_modified": False,
        "edf_annotations_used": False,
        "excel_fields_used": False,
        "doctor_labels_used": False,
        "free_text_used": False,
        "llm_invoked": False,
    }


def _empty_research_projection(
    *,
    status: str,
    input_event_count: int,
    reason_code: str,
) -> dict[str, Any]:
    if status not in {
        "not_published",
        "technical_unassessable",
        "no_valid_event_rankings",
    }:
        raise ValueError("unsupported empty research projection status")
    return {
        "schema_version": RESEARCH_SOZ_PROJECTION_SCHEMA_VERSION,
        "status": status,
        "input_event_count": input_event_count,
        "top1_candidate": None,
        "ranked_candidates": [],
        "evidence_level": None,
        "reason_codes": [reason_code],
        "event_support": None,
        "event_mode_clusters": [],
        "claim_boundary": _research_claim_boundary(),
    }


def _default_research_projection(prepared: Mapping[str, Any]) -> dict[str, Any]:
    row = prepared["row"]
    event_count = int(row["event_count"])
    if prepared["report_kind"] == "technical_report":
        return _empty_research_projection(
            status="technical_unassessable",
            input_event_count=0,
            reason_code="technical_record_has_no_eeg_candidate_projection",
        )
    if event_count == 0:
        return _empty_research_projection(
            status="no_valid_event_rankings",
            input_event_count=0,
            reason_code="no_valid_eeg_event_rankings",
        )
    return _empty_research_projection(
        status="not_published",
        input_event_count=event_count,
        reason_code="research_sidecar_not_attached_to_viewer_release",
    )


def _research_sidecar_directory(value: str | Path) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise ValueError("research SOZ sidecar root must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError("research SOZ sidecar root must be a regular directory")
    return resolved


def _count_map(
    value: object,
    *,
    keys: Sequence[str],
    context: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{context} must contain the complete candidate space")
    return {
        key: _nonnegative_int(value.get(key), f"{context}.{key}") for key in keys
    }


def _project_research_prediction(
    prediction: Mapping[str, Any],
    strength: Mapping[str, Any],
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for row in prediction["ranked_hypotheses"]:
        ranked.append(
            {
                "rank": int(row["rank"]),
                "electrode": str(row["electrode"]),
                "rank1_support_event_count": int(row["top1_support_count"]),
                "rank1_support_rate": float(row["top1_support_rate"]),
                "top3_support_event_count": int(row["top3_support_count"]),
                "top3_support_rate": float(row["top3_support_rate"]),
                "mean_event_rank": float(row["mean_rank"]),
                "display_tier": str(row["display_tier"]),
            }
        )
    top = ranked[0]
    clusters: list[dict[str, Any]] = []
    for index, cluster in enumerate(prediction["event_mode_clusters"], start=1):
        cluster_ranked = cluster["ranked_hypotheses"]
        clusters.append(
            {
                "mode_number": index,
                "event_count": int(cluster["event_count"]),
                "leading_candidates": [
                    str(item["electrode"]) for item in cluster_ranked[:3]
                ],
            }
        )
    consistency = prediction["cross_event_consistency"]
    event_count = int(prediction["input_event_count"])
    return {
        "schema_version": RESEARCH_SOZ_PROJECTION_SCHEMA_VERSION,
        "status": "available",
        "input_event_count": event_count,
        "top1_candidate": deepcopy(top),
        "ranked_candidates": ranked,
        "evidence_level": str(strength["evidence_level"]),
        "reason_codes": list(strength["reason_codes"]),
        "event_support": {
            "rank1_support_event_count": top["rank1_support_event_count"],
            "top3_support_event_count": top["top3_support_event_count"],
            "mode_cluster_count": int(consistency["mode_cluster_count"]),
            "multimodal": bool(consistency["multimodal"]),
        },
        "event_mode_clusters": clusters,
        "claim_boundary": _research_claim_boundary(),
    }


def _project_research_soz_sidecar(
    value: str | Path,
    *,
    coverage: Mapping[str, Any],
    prepared_records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    str,
    str,
    str,
    list[dict[str, Any]],
]:
    """Validate one immutable EEG-only sidecar and return display projections."""

    root = _research_sidecar_directory(value)
    summary_path = _regular_file(
        root, PurePosixPath("cohort_summary.json"), "research SOZ cohort summary"
    )
    summary_snapshot = _snapshot_json(
        summary_path, "research SOZ cohort summary"
    )
    summary = summary_snapshot["value"]
    required_summary_keys = {
        "schema_version",
        "status",
        "source_batch_root_name",
        "input_record_count",
        "bundle_count",
        "generated_prediction_count",
        "top_k_covered_record_count",
        "deterministic_research_conclusion_count",
        "llm_input_eligible_record_count",
        "llm_invoked_record_count",
        "skipped_record_count",
        "input_event_ranking_count",
        "explicit_evidence_weight_event_count",
        "default_unit_weight_event_count",
        "top_k",
        "js_threshold",
        "prediction_method_id",
        "descriptive_evidence_policy_id",
        "evidence_level_counts",
        "skip_reason_counts",
        "candidate_channel_distribution",
        "records",
        "calibration_receipt",
        "scope_receipt",
        "content_sha256",
    }
    if set(summary) != required_summary_keys:
        raise ValueError("research SOZ cohort summary schema drifted")
    if (
        summary.get("schema_version") != RESEARCH_SOZ_BATCH_SCHEMA_VERSION
        or summary.get("status") != RESEARCH_SOZ_BATCH_STATUS
    ):
        raise ValueError("research SOZ cohort summary schema/status is unsupported")
    summary_content_sha256 = _sha256(
        summary.get("content_sha256"), "research SOZ cohort content hash"
    )
    hashable_summary = dict(summary)
    hashable_summary.pop("content_sha256")
    if hashlib.sha256(
        _canonical_bytes(hashable_summary) + b"\n"
    ).hexdigest() != summary_content_sha256:
        raise ValueError("research SOZ cohort summary content hash mismatch")
    if summary.get("prediction_method_id") != RESEARCH_SOZ_PREDICTION_METHOD_ID:
        raise ValueError("research SOZ cohort prediction method drifted")
    if summary.get("descriptive_evidence_policy_id") != RESEARCH_SOZ_EVIDENCE_POLICY_ID:
        raise ValueError("research SOZ cohort evidence policy drifted")
    expected_record_count = int(coverage["expected_record_count"])
    if summary.get("input_record_count") != expected_record_count:
        raise ValueError("research SOZ cohort count differs from report coverage")
    top_k = summary.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(
        C18_ELECTRODES
    ):
        raise ValueError("research SOZ cohort Top-k is invalid")
    js_threshold = summary.get("js_threshold")
    if (
        isinstance(js_threshold, bool)
        or not isinstance(js_threshold, (int, float))
        or not 0.0 <= float(js_threshold) <= 1.0
    ):
        raise ValueError("research SOZ cohort JS threshold is invalid")
    calibration = summary.get("calibration_receipt")
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("status") != "not_attached"
        or calibration.get("receipt") is not None
        or calibration.get("private_cohort_used_to_tune_descriptive_cutpoints")
        is not False
        or calibration.get("clinical_probability_interpretation_permitted")
        is not False
    ):
        raise ValueError("research SOZ cohort calibration boundary drifted")
    scope = summary.get("scope_receipt")
    required_scope_false = (
        "raw_eeg_used",
        "edf_annotations_used",
        "excel_fields_used",
        "doctor_labels_used",
        "postfreeze_evaluation_used",
        "free_text_used_for_prediction",
        "qwen_service_called",
        "llm_may_add_facts",
        "cortical_soz_or_epileptogenic_zone_claim_permitted",
    )
    if (
        not isinstance(scope, Mapping)
        or any(scope.get(key) is not False for key in required_scope_false)
        or scope.get("top_k_is_research_scalp_eeg_ranked_hypothesis") is not True
    ):
        raise ValueError("research SOZ cohort violates its EEG-only claim boundary")

    raw_records = summary.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != expected_record_count:
        raise ValueError("research SOZ cohort record set does not close")
    prepared_by_id = {
        str(prepared["recording_id"]): prepared for prepared in prepared_records
    }
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise TypeError("research SOZ cohort record must be an object")
        recording_id = _identifier(raw.get("recording_id"), "recording_id")
        if recording_id in rows_by_id:
            raise ValueError("research SOZ cohort repeats a recording")
        rows_by_id[recording_id] = raw
    if set(rows_by_id) != set(prepared_by_id):
        raise ValueError("research SOZ cohort recording set differs from report coverage")

    snapshots: list[dict[str, Any]] = [summary_snapshot]
    projections: dict[str, dict[str, Any]] = {}
    completed_count = 0
    skipped_count = 0
    input_event_count = 0
    explicit_weight_count = 0
    default_weight_count = 0
    evidence_counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()
    top1_counts: Counter[str] = Counter()
    top_k_counts: Counter[str] = Counter()
    by_rank_counts: dict[str, Counter[str]] = {
        str(rank): Counter() for rank in range(1, top_k + 1)
    }
    bundle_count = sum(
        prepared["report_kind"] == "eeg_report" for prepared in prepared_records
    )

    for recording_id, prepared in prepared_by_id.items():
        raw = rows_by_id[recording_id]
        status = raw.get("status")
        report_kind = str(prepared["report_kind"])
        report_event_count = int(prepared["row"]["event_count"])
        expected_bundle_relative = (
            PurePosixPath("records") / recording_id / "report" / "bundle.json"
        ).as_posix()
        bundle_relative = raw.get("source_bundle_relative_path")
        bundle_sha256 = raw.get("source_bundle_file_sha256")
        if bundle_relative is not None and bundle_relative != expected_bundle_relative:
            raise ValueError("research SOZ source bundle path binding drifted")
        expected_bundle_sha256 = prepared.get("bundle_sha256")
        if bundle_sha256 is not None:
            _sha256(bundle_sha256, "research SOZ source bundle hash")
            if bundle_sha256 != expected_bundle_sha256:
                raise ValueError("research SOZ source bundle hash binding drifted")

        if status == "skipped":
            skipped_count += 1
            reason = raw.get("skip_reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError("research SOZ skipped record has no reason")
            skip_counts[reason] += 1
            if report_kind == "technical_report":
                if reason != "technical_unassessable_bundle_absent":
                    raise ValueError("technical record research skip reason drifted")
                projections[recording_id] = _empty_research_projection(
                    status="technical_unassessable",
                    input_event_count=0,
                    reason_code=reason,
                )
            else:
                if report_event_count == 0 and reason != "no_event_rankings":
                    raise ValueError("zero-event research skip reason drifted")
                projections[recording_id] = _empty_research_projection(
                    status="no_valid_event_rankings",
                    input_event_count=report_event_count,
                    reason_code=reason,
                )
            continue
        if status != "completed":
            raise ValueError("research SOZ cohort record status is unsupported")
        if report_kind != "eeg_report" or report_event_count < 1:
            raise ValueError("research SOZ prediction was attached to an ineligible report")
        if bundle_relative != expected_bundle_relative or bundle_sha256 is None:
            raise ValueError("completed research SOZ prediction lacks source bundle binding")
        if raw.get("input_event_ranking_count") != report_event_count:
            raise ValueError("research SOZ event count differs from frozen report")
        explicit = _nonnegative_int(
            raw.get("explicit_evidence_weight_event_count"),
            "explicit evidence-weight event count",
        )
        default = _nonnegative_int(
            raw.get("default_unit_weight_event_count"),
            "default evidence-weight event count",
        )
        if explicit + default != report_event_count:
            raise ValueError("research SOZ event-weight accounting does not close")

        prediction_relative = _safe_relative(
            raw.get("prediction_artifact_relative_path"),
            "research SOZ prediction artifact path",
        )
        strength_relative = _safe_relative(
            raw.get("descriptive_strength_relative_path"),
            "research SOZ descriptive-strength path",
        )
        expected_prefix = ("records", recording_id)
        if (
            prediction_relative.parts[:2] != expected_prefix
            or prediction_relative.name != "research_soz_prediction.json"
            or strength_relative.parts[:2] != expected_prefix
            or strength_relative.name != "research_soz_descriptive_strength.json"
        ):
            raise ValueError("research SOZ artifact path escapes its recording")
        prediction_path = _regular_file(
            root, prediction_relative, "research SOZ prediction artifact"
        )
        strength_path = _regular_file(
            root, strength_relative, "research SOZ descriptive-strength artifact"
        )
        prediction_snapshot = _snapshot_json(
            prediction_path, "research SOZ prediction artifact"
        )
        strength_snapshot = _snapshot_json(
            strength_path, "research SOZ descriptive-strength artifact"
        )
        snapshots.extend((prediction_snapshot, strength_snapshot))
        if prediction_snapshot["sha256"] != _sha256(
            raw.get("prediction_file_sha256"), "research SOZ prediction file hash"
        ) or strength_snapshot["sha256"] != _sha256(
            raw.get("descriptive_strength_file_sha256"),
            "research SOZ descriptive-strength file hash",
        ):
            raise ValueError("research SOZ artifact file hash binding drifted")
        prediction = validate_research_soz_prediction_artifact(
            prediction_snapshot["value"]
        )
        strength = validate_research_soz_descriptive_strength(
            strength_snapshot["value"]
        )
        if (
            prediction["artifact_id"] != raw.get("prediction_artifact_id")
            or prediction["content_sha256"] != raw.get("prediction_content_sha256")
            or prediction["content_sha256"] != strength["prediction_content_sha256"]
            or prediction["artifact_id"] != strength["prediction_artifact_id"]
            or strength["content_sha256"]
            != raw.get("descriptive_strength_content_sha256")
            or strength["recording_id"] != recording_id
            or prediction["input_event_count"] != report_event_count
            or prediction["top_k"] != top_k
            or prediction["js_threshold"] != float(js_threshold)
        ):
            raise ValueError("research SOZ prediction/strength binding drifted")
        ranked_electrodes = [
            str(item["electrode"]) for item in prediction["ranked_hypotheses"]
        ]
        if (
            raw.get("top_k_covered") is not True
            or raw.get("top1_electrode") != ranked_electrodes[0]
            or raw.get("ranked_electrodes") != ranked_electrodes
            or raw.get("evidence_level") != strength["evidence_level"]
            or raw.get("deterministic_research_conclusion")
            != strength["deterministic_research_conclusion"]["text"]
            or raw.get("llm_input_eligible") is not True
            or raw.get("llm_invoked") is not False
            or raw.get("llm_may_add_facts") is not False
        ):
            raise ValueError("research SOZ cohort row projection drifted")

        completed_count += 1
        input_event_count += report_event_count
        explicit_weight_count += explicit
        default_weight_count += default
        evidence_counts[str(strength["evidence_level"])] += 1
        top1_counts[ranked_electrodes[0]] += 1
        for rank, electrode in enumerate(ranked_electrodes, start=1):
            top_k_counts[electrode] += 1
            by_rank_counts[str(rank)][electrode] += 1
        projections[recording_id] = _project_research_prediction(
            prediction, strength
        )

    scalar_expectations = {
        "bundle_count": bundle_count,
        "generated_prediction_count": completed_count,
        "top_k_covered_record_count": completed_count,
        "deterministic_research_conclusion_count": completed_count,
        "llm_input_eligible_record_count": completed_count,
        "llm_invoked_record_count": 0,
        "skipped_record_count": skipped_count,
        "input_event_ranking_count": input_event_count,
        "explicit_evidence_weight_event_count": explicit_weight_count,
        "default_unit_weight_event_count": default_weight_count,
    }
    if any(summary.get(key) != expected for key, expected in scalar_expectations.items()):
        raise ValueError("research SOZ cohort scalar accounting does not close")
    if summary.get("skip_reason_counts") != dict(sorted(skip_counts.items())):
        raise ValueError("research SOZ cohort skip accounting does not close")
    expected_evidence = {
        level: evidence_counts[level] for level in DESCRIPTIVE_EVIDENCE_LEVELS
    }
    if summary.get("evidence_level_counts") != expected_evidence:
        raise ValueError("research SOZ cohort evidence accounting does not close")
    distribution = summary.get("candidate_channel_distribution")
    if not isinstance(distribution, Mapping) or distribution.get(
        "candidate_space"
    ) != list(C18_ELECTRODES):
        raise ValueError("research SOZ candidate distribution space drifted")
    candidate_keys = list(C18_ELECTRODES)
    if _count_map(
        distribution.get("top1_record_counts"),
        keys=candidate_keys,
        context="research SOZ Top-1 counts",
    ) != {key: top1_counts[key] for key in candidate_keys}:
        raise ValueError("research SOZ Top-1 distribution does not close")
    if _count_map(
        distribution.get("top_k_occurrence_counts"),
        keys=candidate_keys,
        context="research SOZ Top-k counts",
    ) != {key: top_k_counts[key] for key in candidate_keys}:
        raise ValueError("research SOZ Top-k distribution does not close")
    raw_by_rank = distribution.get("rank_position_record_counts")
    if not isinstance(raw_by_rank, Mapping) or set(raw_by_rank) != set(by_rank_counts):
        raise ValueError("research SOZ rank-position distribution drifted")
    for rank, expected_counter in by_rank_counts.items():
        if _count_map(
            raw_by_rank[rank],
            keys=candidate_keys,
            context=f"research SOZ rank {rank} counts",
        ) != {key: expected_counter[key] for key in candidate_keys}:
            raise ValueError("research SOZ rank-position distribution does not close")

    sidecar_id = "RSOZVIEW-" + summary_content_sha256[:24]
    return (
        projections,
        sidecar_id,
        summary_snapshot["sha256"],
        summary_content_sha256,
        snapshots,
    )


def _load_static_assets() -> dict[str, bytes]:
    root = Path(__file__).with_name("static")
    result: dict[str, bytes] = {}
    for name in ("index.html", "app.js", "styles.css"):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"viewer static asset is unavailable: {name}")
        result[name] = path.read_bytes()
    return result


def build_release_bundle(
    *,
    coverage_manifest_path: str | Path,
    release_audit_path: str | Path,
    artifact_roots: Mapping[str, str | Path],
    output_root: str | Path,
    doctor_label_bundle_path: str | Path | None = None,
    research_soz_sidecar_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build one immutable directory that is safe for the read-only server."""

    coverage_snapshot = _snapshot_json(coverage_manifest_path, "coverage manifest")
    audit_snapshot = _snapshot_json(release_audit_path, "release audit")
    coverage = _validate_coverage(coverage_snapshot["value"])
    audit = _validate_release_audit(
        audit_snapshot["value"],
        coverage=coverage,
        coverage_sha256=coverage_snapshot["sha256"],
    )
    roots = _root_map(artifact_roots)
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output_parent = output.parent.resolve(strict=True)
    destination = (output_parent / output.name).resolve()
    for source_root in roots.values():
        if destination == source_root or destination.is_relative_to(source_root):
            raise ValueError("viewer release bundle must be outside report source trees")
    if research_soz_sidecar_root is not None:
        sidecar_source = _research_sidecar_directory(research_soz_sidecar_root)
        if destination == sidecar_source or destination.is_relative_to(sidecar_source):
            raise ValueError("viewer release bundle must be outside research sidecars")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output_parent))
    os.chmod(temporary, 0o700)
    snapshots: list[dict[str, Any]] = [coverage_snapshot, audit_snapshot]
    file_receipts: dict[str, dict[str, Any]] = {}
    index_records: list[dict[str, Any]] = []
    prepared_records: list[dict[str, Any]] = []
    doctor_label_release_id: str | None = None
    doctor_label_bundle_sha256: str | None = None
    research_sidecar_id: str | None = None
    research_sidecar_summary_sha256: str | None = None
    research_sidecar_content_sha256: str | None = None
    waveform_source_count = 0
    waveform_published_count = 0
    try:
        for static_name, raw in _load_static_assets().items():
            _write_bytes(temporary / static_name, raw)

        for row in coverage["records"]:
            recording_id = str(row["recording_id"])
            subject_id = str(row["patient_pseudonym"])
            source_name, source_root, manifest_relative = _source_manifest_location(
                coverage, row, roots
            )
            manifest_path = _regular_file(
                source_root, manifest_relative, "report manifest"
            )
            manifest_snapshot = _snapshot_json(manifest_path, "report manifest")
            snapshots.append(manifest_snapshot)
            if (
                coverage["schema_version"] == COMBINED_COVERAGE_SCHEMA_VERSION
                and row.get("report_manifest_sha256")
                != manifest_snapshot["sha256"]
            ):
                raise ValueError(
                    "combined coverage report manifest hash differs from source"
                )
            manifest = manifest_snapshot["value"]
            report_kind, artifacts = _validate_report_manifest(manifest, row=row)
            report_dir = manifest_path.parent
            source_report_path = _regular_file(
                report_dir, PurePosixPath("report.html"), "report HTML"
            )
            report_html_sha256 = _file_sha256(source_report_path)
            if report_html_sha256 != artifacts["report.html"]:
                raise ValueError("report HTML hash differs from its manifest")
            report_raw = source_report_path.read_bytes()

            waveform_entries: list[dict[str, Any]] = []
            allowed_source_images: set[str] = set()
            if report_kind == "eeg_report":
                for relative_text, expected_hash in sorted(artifacts.items()):
                    if not relative_text.startswith("waveforms/"):
                        continue
                    relative = _safe_relative(relative_text, "waveform artifact")
                    if len(relative.parts) != 2 or relative.suffix.lower() != ".png":
                        raise ValueError("waveform artifact path is unsupported")
                    waveform_source_count += 1
                    source_waveform = _regular_file(
                        report_dir, relative, "waveform artifact"
                    )
                    if _file_sha256(source_waveform) != expected_hash:
                        raise ValueError("waveform hash differs from its manifest")
                    sanitized = _sanitize_png(source_waveform.read_bytes())
                    target_relative = (
                        PurePosixPath("reports")
                        / recording_id
                        / "waveforms"
                        / relative.name
                    )
                    _write_bytes(temporary / target_relative, sanitized)
                    waveform_entries.append(
                        {
                            "name": relative.name,
                            "url": target_relative.as_posix(),
                            "source_sha256": expected_hash,
                            "published_sha256": hashlib.sha256(sanitized).hexdigest(),
                            "metadata_stripped": True,
                        }
                    )
                    allowed_source_images.add(relative.as_posix())
                    waveform_published_count += 1
            _validate_report_html(report_raw, frozenset(allowed_source_images))
            target_report_relative = (
                PurePosixPath("reports") / recording_id / "report.html"
            )
            _write_bytes(temporary / target_report_relative, report_raw)
            prepared_records.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "row": dict(row),
                    "source_name": source_name,
                    "manifest_relative": manifest_relative.as_posix(),
                    "source_report_relative": (
                        manifest_relative.parent / "report.html"
                    ).as_posix(),
                    "manifest_sha256": manifest_snapshot["sha256"],
                    "report_html_sha256": report_html_sha256,
                    "bundle_sha256": artifacts.get("bundle.json"),
                    "report_kind": report_kind,
                    "target_report_relative": target_report_relative.as_posix(),
                    "waveforms": waveform_entries,
                }
            )

        # All selected report manifests, bodies and waveforms are verified
        # before either independent cohort sidecar is opened.
        research_by_id = {
            str(prepared["recording_id"]): _default_research_projection(prepared)
            for prepared in prepared_records
        }
        if research_soz_sidecar_root is not None:
            (
                research_by_id,
                research_sidecar_id,
                research_sidecar_summary_sha256,
                research_sidecar_content_sha256,
                research_snapshots,
            ) = _project_research_soz_sidecar(
                research_soz_sidecar_root,
                coverage=coverage,
                prepared_records=prepared_records,
            )
            snapshots.extend(research_snapshots)

        doctor_label_by_id: dict[str, dict[str, Any]] = {}
        if doctor_label_bundle_path is not None:
            doctor_snapshot = _snapshot_json(
                doctor_label_bundle_path, "post-freeze doctor-label release bundle"
            )
            snapshots.append(doctor_snapshot)
            doctor_label_by_id, doctor_label_release_id = _project_doctor_label_release(
                doctor_snapshot["value"],
                coverage=coverage,
                coverage_sha256=coverage_snapshot["sha256"],
                release_audit=audit,
                release_audit_sha256=audit_snapshot["sha256"],
                prepared_records=prepared_records,
            )
            doctor_label_bundle_sha256 = doctor_snapshot["sha256"]

        for prepared in prepared_records:
            recording_id = str(prepared["recording_id"])
            subject_id = str(prepared["subject_id"])
            row = prepared["row"]
            report_kind = str(prepared["report_kind"])
            waveform_entries = prepared["waveforms"]
            evaluation = doctor_label_by_id.get(
                recording_id, _empty_evaluation(int(row["event_count"]))
            )
            research_projection = research_by_id[recording_id]
            detail = {
                "schema_version": DETAIL_SCHEMA_VERSION,
                "recording_id": recording_id,
                "subject_id": subject_id,
                "report_kind": report_kind,
                "diagnostic_status": row["diagnostic_status"],
                "event_count": row["event_count"],
                "failure_stage": row.get("failure_stage"),
                "report_url": prepared["target_report_relative"],
                "waveforms": waveform_entries,
                "research_scalp_onset_candidates": research_projection,
                "physician_labels_and_evaluation": evaluation,
                "provenance": {
                    "report_manifest_sha256": prepared["manifest_sha256"],
                    "report_html_sha256": prepared["report_html_sha256"],
                    "doctor_label_release_id": doctor_label_release_id,
                    "doctor_label_bundle_sha256": doctor_label_bundle_sha256,
                    "research_sidecar_id": research_sidecar_id,
                    "research_sidecar_summary_sha256": (
                        research_sidecar_summary_sha256
                    ),
                    "report_verified_before_labels_loaded": True,
                    "report_verified_before_research_sidecar_loaded": True,
                },
                "claim_boundary": {
                    "report_generated_from_eeg_signal_only": report_kind == "eeg_report",
                    "physician_labels_joined_postfreeze_only": True,
                    "research_candidates_are_independent_display_only": True,
                    "research_candidates_modify_qualified_impression": False,
                    "edf_annotations_included": False,
                    "raw_excel_text_included": False,
                    "source_paths_included": False,
                    "direct_identity_included": False,
                    "unsigned_ai_draft": True,
                },
            }
            detail_relative = (
                PurePosixPath("data") / "records" / f"{recording_id}.json"
            )
            _atomic_json(temporary / detail_relative, detail)
            index_records.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "report_kind": report_kind,
                    "diagnostic_status": row["diagnostic_status"],
                    "event_count": row["event_count"],
                    "has_physician_label": evaluation["status"] == "available",
                    "label_status": evaluation["status"],
                    "record_consistency_disposition": evaluation[
                        "record_consistency_disposition"
                    ],
                    "location_consistency": evaluation["location_consistency"],
                    "onset_certainty_consistency": evaluation[
                        "onset_certainty_consistency"
                    ],
                    "waveform_count": len(waveform_entries),
                    "research_soz_status": research_projection["status"],
                    "research_soz_top1": (
                        research_projection["top1_candidate"]["electrode"]
                        if research_projection["top1_candidate"] is not None
                        else None
                    ),
                    "research_soz_evidence_level": research_projection[
                        "evidence_level"
                    ],
                    "detail_url": detail_relative.as_posix(),
                }
            )

        index_records.sort(key=lambda item: item["recording_id"])
        index = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "bundle_title": "私有长程 EEG 报告（去标识化只读发布）",
            "counts": {
                "record_count": len(index_records),
                "subject_count": len({row["subject_id"] for row in index_records}),
                "eeg_report_count": sum(
                    row["report_kind"] == "eeg_report" for row in index_records
                ),
                "technical_report_count": sum(
                    row["report_kind"] == "technical_report"
                    for row in index_records
                ),
                "physician_label_available_count": sum(
                    row["has_physician_label"] for row in index_records
                ),
                "physician_label_source_conflict_count": sum(
                    row["label_status"] == "source_conflict"
                    for row in index_records
                ),
                "physician_label_ambiguous_mapping_count": sum(
                    row["label_status"] == "ambiguous_mapping"
                    for row in index_records
                ),
                "waveform_count": waveform_published_count,
                "research_soz_candidate_count": sum(
                    row["research_soz_status"] == "available"
                    for row in index_records
                ),
            },
            "records": index_records,
            "notice": {
                "unsigned_ai_draft": True,
                "labels_are_postfreeze_reference_only": True,
                "missing_label_is_not_a_negative_label": True,
                "technical_unassessable_is_not_eeg_evidence_insufficiency": True,
                "research_candidates_are_uncalibrated_scalp_hypotheses_only": True,
                "research_candidates_do_not_replace_qualified_impression": True,
            },
        }
        _atomic_json(temporary / "data" / "index.json", index)

        for path in sorted(temporary.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(temporary).as_posix()
            file_receipts[relative] = {
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        bundle_body = {
            "schema_version": RELEASE_BUNDLE_SCHEMA_VERSION,
            "bundle_id": "EEGVIEW-" + _canonical_sha256(
                {
                    "coverage": coverage_snapshot["sha256"],
                    "release_audit": audit_snapshot["sha256"],
                    "doctor_label_bundle": doctor_label_bundle_sha256,
                    "research_soz_sidecar": research_sidecar_summary_sha256,
                    "files": file_receipts,
                }
            )[:24],
            "inventory_id": coverage["inventory_id"],
            "expected_record_count": coverage["expected_record_count"],
            "expected_subject_count": coverage["expected_subject_count"],
            "release_receipt": {
                "release_audit_id": audit.get("audit_id"),
                "coverage_manifest_sha256": coverage_snapshot["sha256"],
                "release_audit_sha256": audit_snapshot["sha256"],
                "release_ready": True,
            },
            "doctor_label_release_receipt": {
                "label_release_id": doctor_label_release_id,
                "doctor_label_bundle_sha256": doctor_label_bundle_sha256,
                "loaded_after_all_reports_verified": True,
            },
            "research_soz_release_receipt": {
                "sidecar_id": research_sidecar_id,
                "cohort_summary_sha256": research_sidecar_summary_sha256,
                "cohort_content_sha256": research_sidecar_content_sha256,
                "loaded_after_all_reports_verified": True,
                "used_for_qualified_impression": False,
            },
            "counts": index["counts"],
            "waveform_receipt": {
                "source_waveform_count": waveform_source_count,
                "published_waveform_count": waveform_published_count,
                "source_hashes_verified": True,
                "ancillary_metadata_stripped": True,
            },
            "files": file_receipts,
            "claim_boundary": {
                "explicit_release_bundle_only": True,
                "read_only_service": True,
                "reports_verified_before_labels_loaded": True,
                "generation_uses_eeg_signal_only": True,
                "physician_labels_used_for_generation": False,
                "physician_labels_used_for_renderer": False,
                "research_soz_sidecar_is_eeg_only": True,
                "research_soz_used_for_qualified_impression": False,
                "research_soz_used_as_cortical_or_treatment_target": False,
                "raw_excel_text_included": False,
                "edf_annotations_included": False,
                "source_paths_included": False,
                "direct_identity_included": False,
            },
        }
        _assert_unchanged(snapshots)
        _atomic_json(temporary / "release_bundle.json", bundle_body)
        os.replace(temporary, destination)
        os.chmod(destination, 0o700)
        return deepcopy(bundle_body)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _release_file_kind(relative: PurePosixPath) -> str:
    """Return the one supported route class for a manifest allowlist path."""

    parts = relative.parts
    if parts in {("index.html",), ("app.js",), ("styles.css",)}:
        return "static"
    if parts == ("data", "index.json"):
        return "index"
    if (
        len(parts) == 3
        and parts[:2] == ("data", "records")
        and parts[2].endswith(".json")
        and _IDENTIFIER_RE.fullmatch(parts[2][:-5]) is not None
    ):
        return "record"
    if (
        len(parts) == 3
        and parts[0] == "reports"
        and _IDENTIFIER_RE.fullmatch(parts[1]) is not None
        and parts[2] == "report.html"
    ):
        return "report"
    if (
        len(parts) == 4
        and parts[0] == "reports"
        and _IDENTIFIER_RE.fullmatch(parts[1]) is not None
        and parts[2] == "waveforms"
        and parts[3].lower().endswith(".png")
    ):
        return "waveform"
    raise ValueError(f"release bundle file path is unsupported: {relative}")


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{context} schema drifted")


def verify_release_bundle(
    bundle_root: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify every allowlisted web file and return the immutable manifest."""

    raw_root = Path(bundle_root)
    if raw_root.is_symlink():
        raise ValueError("release bundle root must not be a symlink")
    root = raw_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("release bundle root must be a regular directory")
    manifest_path = root / "release_bundle.json"
    snapshot = _snapshot_json(manifest_path, "release bundle manifest")
    manifest = snapshot["value"]
    if expected_manifest_sha256 is not None:
        expected_pin = _sha256(
            expected_manifest_sha256, "expected release bundle manifest hash"
        )
        if snapshot["sha256"] != expected_pin:
            raise ValueError("release bundle manifest differs from the external SHA-256 pin")
    schema_version = manifest.get("schema_version")
    if schema_version not in {
        LEGACY_RELEASE_BUNDLE_SCHEMA_VERSION,
        RELEASE_BUNDLE_SCHEMA_VERSION,
    }:
        raise ValueError("release bundle schema is unsupported")
    manifest_keys = {
        "schema_version",
        "bundle_id",
        "inventory_id",
        "expected_record_count",
        "expected_subject_count",
        "release_receipt",
        "doctor_label_release_receipt",
        "counts",
        "waveform_receipt",
        "files",
        "claim_boundary",
    }
    if schema_version == RELEASE_BUNDLE_SCHEMA_VERSION:
        manifest_keys.add("research_soz_release_receipt")
    _exact_keys(manifest, frozenset(manifest_keys), "release bundle manifest")
    bundle_id = _identifier(manifest.get("bundle_id"), "release bundle ID")
    _identifier(manifest.get("inventory_id"), "release bundle inventory ID")
    expected_record_count = _nonnegative_int(
        manifest.get("expected_record_count"), "release bundle expected record count"
    )
    expected_subject_count = _nonnegative_int(
        manifest.get("expected_subject_count"), "release bundle expected subject count"
    )
    release_receipt = manifest.get("release_receipt")
    if not isinstance(release_receipt, Mapping):
        raise TypeError("release receipt must be an object")
    _exact_keys(
        release_receipt,
        frozenset(
            {
                "release_audit_id",
                "coverage_manifest_sha256",
                "release_audit_sha256",
                "release_ready",
            }
        ),
        "release receipt",
    )
    _identifier(release_receipt.get("release_audit_id"), "release audit ID")
    coverage_sha256 = _sha256(
        release_receipt.get("coverage_manifest_sha256"), "coverage manifest hash"
    )
    release_audit_sha256 = _sha256(
        release_receipt.get("release_audit_sha256"), "release audit hash"
    )
    if release_receipt.get("release_ready") is not True:
        raise ValueError("release receipt is not release-ready")
    doctor_receipt = manifest.get("doctor_label_release_receipt")
    if not isinstance(doctor_receipt, Mapping):
        raise TypeError("doctor-label release receipt must be an object")
    _exact_keys(
        doctor_receipt,
        frozenset(
            {
                "label_release_id",
                "doctor_label_bundle_sha256",
                "loaded_after_all_reports_verified",
            }
        ),
        "doctor-label release receipt",
    )
    label_release_id = doctor_receipt.get("label_release_id")
    doctor_label_sha256 = doctor_receipt.get("doctor_label_bundle_sha256")
    if (label_release_id is None) is not (doctor_label_sha256 is None):
        raise ValueError("doctor-label release receipt is incomplete")
    if label_release_id is not None:
        _identifier(label_release_id, "doctor-label release ID")
        doctor_label_sha256 = _sha256(
            doctor_label_sha256, "doctor-label release bundle hash"
        )
    if doctor_receipt.get("loaded_after_all_reports_verified") is not True:
        raise ValueError("doctor-label release load-order receipt is invalid")
    research_sidecar_sha256: str | None = None
    if schema_version == RELEASE_BUNDLE_SCHEMA_VERSION:
        research_receipt = manifest.get("research_soz_release_receipt")
        if not isinstance(research_receipt, Mapping):
            raise TypeError("research SOZ release receipt must be an object")
        _exact_keys(
            research_receipt,
            frozenset(
                {
                    "sidecar_id",
                    "cohort_summary_sha256",
                    "cohort_content_sha256",
                    "loaded_after_all_reports_verified",
                    "used_for_qualified_impression",
                }
            ),
            "research SOZ release receipt",
        )
        sidecar_id = research_receipt.get("sidecar_id")
        sidecar_summary_sha256 = research_receipt.get("cohort_summary_sha256")
        sidecar_content_sha256 = research_receipt.get("cohort_content_sha256")
        if not (
            (sidecar_id is None)
            == (sidecar_summary_sha256 is None)
            == (sidecar_content_sha256 is None)
        ):
            raise ValueError("research SOZ release receipt is incomplete")
        if sidecar_id is not None:
            _identifier(sidecar_id, "research SOZ sidecar ID")
            research_sidecar_sha256 = _sha256(
                sidecar_summary_sha256, "research SOZ cohort summary hash"
            )
            _sha256(sidecar_content_sha256, "research SOZ cohort content hash")
        if (
            research_receipt.get("loaded_after_all_reports_verified") is not True
            or research_receipt.get("used_for_qualified_impression") is not False
        ):
            raise ValueError("research SOZ release receipt violates display-only use")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise TypeError("release bundle counts must be an object")
    count_keys = frozenset(
        {
            "record_count",
            "subject_count",
            "eeg_report_count",
            "technical_report_count",
            "physician_label_available_count",
            "physician_label_source_conflict_count",
            "physician_label_ambiguous_mapping_count",
            "waveform_count",
            *(
                {"research_soz_candidate_count"}
                if schema_version == RELEASE_BUNDLE_SCHEMA_VERSION
                else set()
            ),
        }
    )
    _exact_keys(counts, count_keys, "release bundle counts")
    normalized_counts = {
        key: _nonnegative_int(counts.get(key), f"release bundle count {key}")
        for key in count_keys
    }
    if (
        normalized_counts["record_count"] != expected_record_count
        or normalized_counts["subject_count"] != expected_subject_count
        or normalized_counts["eeg_report_count"]
        + normalized_counts["technical_report_count"]
        != expected_record_count
        or normalized_counts["physician_label_available_count"]
        + normalized_counts["physician_label_source_conflict_count"]
        + normalized_counts["physician_label_ambiguous_mapping_count"]
        > expected_record_count
        or (
            schema_version == RELEASE_BUNDLE_SCHEMA_VERSION
            and normalized_counts["research_soz_candidate_count"]
            > normalized_counts["eeg_report_count"]
        )
    ):
        raise ValueError("release bundle cohort counts do not close")
    waveform_receipt = manifest.get("waveform_receipt")
    if not isinstance(waveform_receipt, Mapping):
        raise TypeError("waveform receipt must be an object")
    _exact_keys(
        waveform_receipt,
        frozenset(
            {
                "source_waveform_count",
                "published_waveform_count",
                "source_hashes_verified",
                "ancillary_metadata_stripped",
            }
        ),
        "waveform receipt",
    )
    source_waveform_count = _nonnegative_int(
        waveform_receipt.get("source_waveform_count"), "source waveform count"
    )
    published_waveform_count = _nonnegative_int(
        waveform_receipt.get("published_waveform_count"),
        "published waveform count",
    )
    if (
        source_waveform_count != published_waveform_count
        or published_waveform_count != normalized_counts["waveform_count"]
        or waveform_receipt.get("source_hashes_verified") is not True
        or waveform_receipt.get("ancillary_metadata_stripped") is not True
    ):
        raise ValueError("waveform receipt does not close")
    boundary = manifest.get("claim_boundary")
    required = {
        "explicit_release_bundle_only": True,
        "read_only_service": True,
        "reports_verified_before_labels_loaded": True,
        "generation_uses_eeg_signal_only": True,
        "physician_labels_used_for_generation": False,
        "physician_labels_used_for_renderer": False,
        "raw_excel_text_included": False,
        "edf_annotations_included": False,
        "source_paths_included": False,
        "direct_identity_included": False,
    }
    if schema_version == RELEASE_BUNDLE_SCHEMA_VERSION:
        required.update(
            {
                "research_soz_sidecar_is_eeg_only": True,
                "research_soz_used_for_qualified_impression": False,
                "research_soz_used_as_cortical_or_treatment_target": False,
            }
        )
    if not isinstance(boundary, Mapping) or frozenset(boundary) != frozenset(required) or any(
        boundary.get(key) is not expected for key, expected in required.items()
    ):
        raise ValueError("release bundle claim boundary is invalid")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("release bundle file allowlist is invalid")
    required_files = {"index.html", "app.js", "styles.css", "data/index.json"}
    if not required_files.issubset(files):
        raise ValueError("release bundle file allowlist is incomplete")
    file_kinds: dict[str, int] = {
        "static": 0,
        "index": 0,
        "record": 0,
        "report": 0,
        "waveform": 0,
    }
    record_json_ids: set[str] = set()
    report_html_ids: set[str] = set()
    for raw_relative, raw_receipt in files.items():
        relative = _safe_relative(raw_relative, "release bundle file")
        kind = _release_file_kind(relative)
        file_kinds[kind] += 1
        if kind == "record":
            record_json_ids.add(relative.name[:-5])
        elif kind == "report":
            report_html_ids.add(relative.parts[1])
        if not isinstance(raw_receipt, Mapping):
            raise TypeError("release bundle file receipt must be an object")
        _exact_keys(
            raw_receipt,
            frozenset({"sha256", "size_bytes"}),
            "release bundle file receipt",
        )
        expected_hash = _sha256(raw_receipt.get("sha256"), "bundle file hash")
        expected_size = _nonnegative_int(
            raw_receipt.get("size_bytes"), "bundle file size"
        )
        path = _regular_file(root, relative, "release bundle file")
        if path.stat().st_size != expected_size or _file_sha256(path) != expected_hash:
            raise ValueError(f"release bundle file failed verification: {relative}")
    if (
        record_json_ids != report_html_ids
        or len(record_json_ids) != expected_record_count
        or file_kinds["waveform"] != normalized_counts["waveform_count"]
    ):
        raise ValueError("release bundle record/report allowlists do not close")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("release bundle contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != set(files) | {"release_bundle.json"}:
        raise ValueError("release bundle contains files outside its allowlist")
    bundle_binding = {
        "coverage": coverage_sha256,
        "release_audit": release_audit_sha256,
        "doctor_label_bundle": doctor_label_sha256,
        "files": files,
    }
    if schema_version == RELEASE_BUNDLE_SCHEMA_VERSION:
        bundle_binding["research_soz_sidecar"] = research_sidecar_sha256
    expected_bundle_id = "EEGVIEW-" + _canonical_sha256(bundle_binding)[:24]
    if bundle_id != expected_bundle_id:
        raise ValueError("release bundle ID does not bind its receipts and files")
    return deepcopy(manifest)


__all__ = [
    "DETAIL_SCHEMA_VERSION",
    "INDEX_SCHEMA_VERSION",
    "RELEASE_BUNDLE_SCHEMA_VERSION",
    "build_release_bundle",
    "verify_release_bundle",
]
