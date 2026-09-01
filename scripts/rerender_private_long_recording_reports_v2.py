#!/usr/bin/env python3
"""Safely rerender a complete private long-recording EEG report cohort.

This is a presentation-only migration over an already frozen combined cohort.
It accepts the strict inventory, combined coverage manifest and the three
possible artifact roots selected by that manifest.  For an EEG report it reads
only the hash-bound ``bundle.json``, optional ``language_records.json`` and the
portable waveform PNGs (plus the other already-published report artifacts that
must be copied).  The current deterministic HTML and DOCX renderers are then
called without EEG inference or a language-model request.  Technical-
unassessable report shells are copied byte-for-byte.

EDF signals, EDF annotations, CSV annotation ledgers, spreadsheets, physician
labels and post-freeze evaluation files have no input parameter or resolution
route in this command.  Source roots are never modified and the destination is
published atomically as a new, independent full batch root.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_private_long_recording_report_release_v1 as release  # noqa: E402
from scripts import materialize_private_long_recording_reports_v1 as batch  # noqa: E402
from src.clinical_eeg_long_recording import render as render_module  # noqa: E402
from src.clinical_eeg_long_recording.report_outcome import (  # noqa: E402
    classify_recording_eeg_outcome,
)
from src.clinical_eeg_report.language_quality import (  # noqa: E402
    CANDIDATE_SCHEMA,
    TERM_PATTERNS,
    validate_candidate_manifest,
)


SCHEMA_VERSION = "private_long_recording_report_layer_rerender_v2"
STATUS = "completed_full_report_layer_rerender"
LANGUAGE_QUALITY_FILENAME = "language_quality_candidates.json"
RERENDER_RECEIPT_FILENAME = "rerender_manifest.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EEG_FIXED_ARTIFACTS = frozenset(
    {
        "bundle.json",
        "detection_manifest.json",
        "event_segment_receipts.json",
        "analysis_selection_manifest.json",
        "language_records.json",
        "report.html",
        "report.docx",
    }
)
_TECHNICAL_ARTIFACTS = frozenset({"report.json", "report.html"})
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
_TECHNICAL_FAILURE_RECEIPT_KEYS = frozenset(
    {
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
    }
)
_FORBIDDEN_SOURCE_SUFFIXES = frozenset(
    {".edf", ".bdf", ".csv", ".tsv", ".xls", ".xlsx", ".xlsm", ".ods"}
)
_FORBIDDEN_SOURCE_NAMES = frozenset(
    {
        "edf_annotations.csv",
        "annotations.csv",
        "doctor_labels.json",
        "postfreeze_doctor_labels.json",
    }
)
_SECTION_NAMES = {
    "记录信息": "recording_information",
    "脑电图表现": "eeg_findings",
    "相关 EEG 波形证据": "waveform_evidence",
    "脑电图印象": "eeg_impression",
    "研究性附录": "research_appendix",
    "范围说明": "scope_boundary",
}
_FACT_TERM_CODES = {
    "ictal_onset_pattern": frozenset(
        {
            "ictal_onset",
            "electrographic_seizure",
            "low_voltage_fast_activity",
            "electrodecrement",
        }
    ),
    "ictal_evolution": frozenset({"ictal_evolution"}),
    "ictal_spread": frozenset({"ictal_spread"}),
    "postictal_pattern": frozenset({"postictal"}),
}
_MORPHOLOGY_TERM_CODES = {
    "spike": frozenset({"spike"}),
    "repetitive_spikes": frozenset({"spike"}),
    "polyspike": frozenset({"spike"}),
    "spike_and_slow_wave": frozenset({"spike", "epileptiform_discharge"}),
    "sharp_wave": frozenset({"sharp_wave"}),
    "sharp_and_slow_wave": frozenset(
        {"sharp_wave", "epileptiform_discharge"}
    ),
    "low_voltage_fast_activity": frozenset({"low_voltage_fast_activity"}),
    "electrodecrement": frozenset({"electrodecrement"}),
}


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid constant {value!r}")


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
    return _sha256_bytes(raw)


def _strict_json_file(path: Path, context: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{context} must be a regular file")
    raw = resolved.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_invalid_constant,
    )
    if type(value) is not dict:
        raise TypeError(f"{context} must be a JSON object")
    return value, _sha256_bytes(raw)


def _strict_json_artifact(path: Path, context: str) -> dict[str, Any]:
    value, _ = _strict_json_file(path, context)
    return value


def _safe_relative(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context} must be a safe relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{context} must be a safe relative path")
    return relative


def _regular_root(path: Path, context: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    root = path.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{context} must be a regular directory")
    return root


def _resolve_regular(root: Path, relative: PurePosixPath, context: str) -> Path:
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{context} traverses a symlink")
    resolved = cursor.resolve(strict=True)
    resolved.relative_to(root)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{context} must resolve to a regular file")
    return resolved


def _assert_allowed_artifact(relative: PurePosixPath, *, technical: bool) -> None:
    name = relative.name.lower()
    suffix = relative.suffix.lower()
    if name in _FORBIDDEN_SOURCE_NAMES or suffix in _FORBIDDEN_SOURCE_SUFFIXES:
        raise ValueError("forbidden source artifact entered report-layer rerender")
    if technical:
        if len(relative.parts) != 1 or relative.as_posix() not in _TECHNICAL_ARTIFACTS:
            raise ValueError("technical report contains a non-canonical artifact")
        return
    text = relative.as_posix()
    if text in _EEG_FIXED_ARTIFACTS:
        return
    if (
        len(relative.parts) == 2
        and relative.parts[0] == "waveforms"
        and suffix == ".png"
        and relative.name.startswith("eeg_waveform_")
    ):
        return
    raise ValueError("EEG report contains an artifact outside the frozen whitelist")


def _artifact_files(directory: Path) -> set[PurePosixPath]:
    result: set[PurePosixPath] = set()
    for current, directory_names, filenames in os.walk(directory, followlinks=False):
        current_path = Path(current)
        for directory_name in directory_names:
            if (current_path / directory_name).is_symlink():
                raise ValueError("source report tree contains a directory symlink")
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("source report tree contains a non-regular file")
            result.add(PurePosixPath(path.relative_to(directory).as_posix()))
    return result


def _validated_source_report(
    *,
    root: Path,
    row: Mapping[str, Any],
    recording: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[PurePosixPath, Path], Path]:
    recording_id = str(recording["recording_id"])
    state_relative = _safe_relative(
        row.get("state_manifest_relative_path"), "combined state manifest path"
    )
    report_relative = _safe_relative(
        row.get("report_manifest_relative_path"), "combined report manifest path"
    )
    if state_relative != PurePosixPath("records") / recording_id / "state.json":
        raise ValueError("combined state path does not bind the recording")
    state_path = _resolve_regular(root, state_relative, "source state manifest")
    if _sha256_file(state_path) != row.get("state_manifest_sha256"):
        raise ValueError("source state manifest hash differs from combined coverage")
    state = _strict_json_artifact(state_path, "source state manifest")
    if (
        state.get("recording_id") != recording_id
        or state.get("patient_pseudonym") != recording["patient_pseudonym"]
        or state.get("diagnostic_status") != row.get("diagnostic_status")
    ):
        raise ValueError("source state identity or diagnostic status drifted")

    manifest_path = _resolve_regular(root, report_relative, "source report manifest")
    if _sha256_file(manifest_path) != row.get("report_manifest_sha256"):
        raise ValueError("source report manifest hash differs from combined coverage")
    manifest = _strict_json_artifact(manifest_path, "source report manifest")
    if (
        manifest.get("recording_id") != recording_id
        or manifest.get("patient_pseudonym") != recording["patient_pseudonym"]
        or manifest.get("diagnostic_status") != row.get("diagnostic_status")
        or manifest.get("event_count") != row.get("event_count")
    ):
        raise ValueError("source report manifest identity or outcome drifted")
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict or not artifacts:
        raise ValueError("source report manifest has no artifact ledger")
    technical = row.get("effective_report_kind") == "technical_unassessable_report"
    if technical != (
        row.get("diagnostic_status") == "completed_technical_unassessable"
    ):
        raise ValueError("combined report kind and diagnostic status disagree")
    report_directory = manifest_path.parent
    resolved_artifacts: dict[PurePosixPath, Path] = {}
    for raw_relative, expected_sha in artifacts.items():
        relative = _safe_relative(raw_relative, "report artifact path")
        _assert_allowed_artifact(relative, technical=technical)
        if not isinstance(expected_sha, str) or _SHA256_RE.fullmatch(expected_sha) is None:
            raise ValueError("report artifact SHA-256 is malformed")
        artifact = _resolve_regular(report_directory, relative, "report artifact")
        if _sha256_file(artifact) != expected_sha:
            raise ValueError("report artifact hash differs from its manifest")
        resolved_artifacts[relative] = artifact
    expected_files = set(resolved_artifacts) | {PurePosixPath("manifest.json")}
    if _artifact_files(report_directory) != expected_files:
        raise ValueError("source report directory has undeclared or missing files")
    if technical:
        if set(resolved_artifacts) != {
            PurePosixPath("report.json"),
            PurePosixPath("report.html"),
        }:
            raise ValueError("technical artifact ledger is not canonical")
    else:
        required = {
            PurePosixPath("bundle.json"),
            PurePosixPath("report.html"),
            PurePosixPath("report.docx"),
        }
        if not required.issubset(resolved_artifacts):
            raise ValueError("EEG report is missing a frozen renderer input/output")
        language_receipt = manifest.get("language_service_receipt")
        configured = isinstance(language_receipt, Mapping) and (
            language_receipt.get("configured") is True
        )
        if configured != (PurePosixPath("language_records.json") in resolved_artifacts):
            raise ValueError("language-record artifact and service receipt disagree")
    return report_directory, manifest, resolved_artifacts, state_path


def _copy_verified(source: Path, destination: Path, expected_sha: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256_file(destination) != expected_sha:
        raise ValueError("copied report artifact failed SHA-256 verification")


def _validated_technical_failure_receipt(
    *,
    root: Path,
    row: Mapping[str, Any],
    recording: Mapping[str, Any],
    report_directory: Path,
    report_manifest: Mapping[str, Any],
    state_path: Path,
) -> tuple[Path, dict[str, Any], str]:
    """Validate the case-level receipt that authorizes a technical shell."""

    recording_id = str(recording["recording_id"])
    case_relative = PurePosixPath("records") / recording_id
    case_root = root.joinpath(*case_relative.parts).resolve(strict=True)
    case_root.relative_to(root)
    if case_root.is_symlink() or not case_root.is_dir():
        raise ValueError("technical case root must be a regular directory")
    receipt_path = _resolve_regular(
        root,
        case_relative / "technical_failure_receipt.json",
        "technical failure receipt",
    )
    receipt, receipt_sha = _strict_json_file(
        receipt_path, "technical failure receipt"
    )
    if set(receipt) != _TECHNICAL_FAILURE_RECEIPT_KEYS:
        raise ValueError("technical failure receipt schema drifted")
    if (
        receipt.get("schema_version") != batch.FAILURE_SCHEMA_VERSION
        or receipt.get("status") != "technical_failure_receipt"
        or receipt.get("recording_id") != recording_id
        or receipt.get("patient_pseudonym") != recording["patient_pseudonym"]
        or receipt.get("failed_stage") != row.get("failure_stage")
        or receipt.get("failed_stage") != report_manifest.get("failure_stage")
        or receipt.get("report_generated") is not True
        or receipt.get("retryable") is not True
    ):
        raise ValueError("technical failure receipt identity or status drifted")
    fingerprint = receipt.get("error_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or _SHA256_RE.fullmatch(fingerprint) is None
        or fingerprint
        != report_manifest.get("technical_failure_receipt_fingerprint")
    ):
        raise ValueError("technical failure receipt fingerprint drifted")
    for field in ("error_code", "exception_class"):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise ValueError(f"technical failure receipt {field} is invalid")
    attempt = receipt.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("technical failure receipt attempt is invalid")
    state = _strict_json_artifact(state_path, "technical report state")
    if state.get("attempt") != attempt:
        raise ValueError("technical failure receipt attempt differs from state")
    expected_relative = report_directory.relative_to(case_root).as_posix()
    if (
        receipt.get("technical_report_relative_dir") != expected_relative
        or report_directory.name != f"attempt_{attempt:04d}"
    ):
        raise ValueError("technical failure receipt report locator drifted")
    privacy = receipt.get("privacy_receipt")
    if privacy != {
        "exception_message_persisted": False,
        "raw_edf_path_persisted": False,
        "raw_patient_identity_persisted": False,
        "annotation_excel_onset_or_gt_persisted": False,
    }:
        raise ValueError("technical failure receipt privacy boundary drifted")
    subprocess = receipt.get("subprocess_receipt")
    if subprocess is not None:
        if not isinstance(subprocess, Mapping) or set(subprocess) != {
            "returncode",
            "stdout_sha256",
            "stderr_sha256",
            "stdout_or_stderr_persisted",
        }:
            raise ValueError("technical subprocess receipt schema drifted")
        if (
            isinstance(subprocess.get("returncode"), bool)
            or not isinstance(subprocess.get("returncode"), int)
            or subprocess.get("stdout_or_stderr_persisted") is not False
            or any(
                not isinstance(subprocess.get(field), str)
                or _SHA256_RE.fullmatch(str(subprocess.get(field))) is None
                for field in ("stdout_sha256", "stderr_sha256")
            )
        ):
            raise ValueError("technical subprocess receipt values drifted")
    return receipt_path, receipt, receipt_sha


def _write_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _normal_text(value: str) -> str:
    lines = []
    for raw in value.replace("\xa0", " ").splitlines():
        line = " ".join(raw.split())
        if line:
            lines.append(line)
    return "\n".join(lines)


class _ReportSurfaceParser(HTMLParser):
    """Extract visible body text and deterministic H2 sections."""

    _BLOCKS = frozenset(
        {"p", "div", "tr", "table", "figure", "figcaption", "h1", "h2", "h3", "br"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_body = False
        self.skip_depth = 0
        self.heading_parts: list[str] | None = None
        self.current_section: str | None = None
        self.surface_parts: list[str] = []
        self.section_parts: dict[str, list[str]] = {}
        self._unknown_section_index = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag == "body":
            self.in_body = True
            return
        if not self.in_body:
            return
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "h2":
            self.heading_parts = []
        if tag == "br":
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if not self.in_body or self.skip_depth:
            return
        if tag == "h2" and self.heading_parts is not None:
            heading = _normal_text("".join(self.heading_parts))
            section = _SECTION_NAMES.get(heading)
            if section is None:
                self._unknown_section_index += 1
                section = f"section_{self._unknown_section_index}"
            self.current_section = section
            self.section_parts.setdefault(section, [])
            self.heading_parts = None
        if tag in self._BLOCKS:
            self._append("\n")
        if tag == "body":
            self.in_body = False

    def handle_data(self, data: str) -> None:
        if not self.in_body or self.skip_depth or not data:
            return
        self.surface_parts.append(data)
        if self.heading_parts is not None:
            self.heading_parts.append(data)
        elif self.current_section is not None:
            self.section_parts.setdefault(self.current_section, []).append(data)

    def _append(self, text: str) -> None:
        self.surface_parts.append(text)
        if self.heading_parts is None and self.current_section is not None:
            self.section_parts.setdefault(self.current_section, []).append(text)

    def result(self) -> tuple[str, dict[str, str]]:
        surface = _normal_text("".join(self.surface_parts))
        sections = {
            key: text
            for key, parts in self.section_parts.items()
            if (text := _normal_text("".join(parts)))
        }
        if not surface:
            raise ValueError("rendered report has an empty visible surface")
        return surface, sections


def _surface(html_text: str) -> tuple[str, dict[str, str]]:
    parser = _ReportSurfaceParser()
    parser.feed(html_text)
    parser.close()
    return parser.result()


def _all_fact_ids(bundle: Mapping[str, Any]) -> tuple[list[str], dict[str, set[str]]]:
    """Return only facts that the current renderer may put on the surface.

    A bundle can retain rejected or historical facts for audit closure.  Calling
    every retained ID ``reportable`` would make fact coverage spuriously equal
    to one when a summary cites the whole ledger.  Reuse the renderer's final
    qualification gate here, so the evaluation denominator is the same set of
    patient-level facts that can actually contribute wording.
    """

    result: list[str] = []
    term_facts: dict[str, set[str]] = {code: set() for code in TERM_PATTERNS}
    events = bundle.get("events")
    if not isinstance(events, list):
        raise TypeError("bundle.events must be an array")
    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            raise TypeError("bundle event must be an object")
        report = raw_event.get("event_report_payload")
        if not isinstance(report, Mapping) or not isinstance(report.get("facts"), list):
            raise TypeError("bundle event report has no fact ledger")
        for raw_fact in report["facts"]:
            if not isinstance(raw_fact, Mapping):
                raise TypeError("event fact must be an object")
            fact_id = raw_fact.get("fact_id")
            if not isinstance(fact_id, str) or _IDENTIFIER_RE.fullmatch(fact_id) is None:
                raise ValueError("event fact ID is not a safe identifier")
            try:
                authorized = render_module._event_fact_language_authorized(  # noqa: SLF001
                    raw_event, raw_fact
                )
            except (KeyError, TypeError, ValueError):
                authorized = False
            if not authorized:
                continue
            if fact_id not in result:
                result.append(fact_id)
            fact_type = str(raw_fact.get("fact_type"))
            codes = set(_FACT_TERM_CODES.get(fact_type, frozenset()))
            value = raw_fact.get("value")
            if fact_type == "ictal_onset_pattern" and isinstance(value, Mapping):
                morphology = value.get("morphology")
                if isinstance(morphology, str):
                    codes.update(_MORPHOLOGY_TERM_CODES.get(morphology, frozenset()))
            for code in codes:
                term_facts[code].add(fact_id)
    return result, term_facts


def _term_codes(text: str) -> set[str]:
    return {code for code, pattern in TERM_PATTERNS.items() if pattern.search(text)}


def _eeg_candidate(
    *,
    recording_id: str,
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any],
    html_text: str,
) -> dict[str, Any]:
    report_text, sections = _surface(html_text)
    required_sections = [
        "recording_information",
        "eeg_findings",
        "waveform_evidence",
        "eeg_impression",
    ]
    if any(section not in sections for section in required_sections):
        raise ValueError("EEG report is missing a required rendered section")

    outcome = classify_recording_eeg_outcome(bundle)
    if (
        manifest.get("diagnostic_outcome") != outcome
        or manifest.get("diagnostic_status") != outcome["report_status"]
    ):
        raise ValueError(
            "frozen manifest diagnostic outcome no longer classifies identically: "
            + recording_id
        )
    outcome_id = "OUTCOME-" + _canonical_sha256(outcome)[:24].upper()
    fact_ids, fact_term_ids = _all_fact_ids(bundle)
    reportable = [*fact_ids, outcome_id]
    outcome_text = str(outcome["conclusion_zh"])
    if outcome_text not in report_text:
        raise ValueError("frozen diagnostic conclusion is absent from rendered surface")

    impression = render_module._automatic_eeg_impression(bundle)  # noqa: SLF001
    claims: list[dict[str, Any]] = []
    authorization: dict[str, set[str]] = {code: set() for code in TERM_PATTERNS}
    outcome_codes = _term_codes(outcome_text)
    for code in outcome_codes:
        authorization[code].add(outcome_id)
    claims.append(
        {
            "claim_id": "diagnostic-outcome",
            "text_zh": outcome_text,
            "fact_ids": [outcome_id],
            "term_codes": sorted(outcome_codes),
        }
    )

    for field, claim_id in (
        ("findings", "record-findings"),
        ("localization", "localization-reasoning"),
        ("uncertainty", "uncertainty-boundary"),
    ):
        text = str(impression[field])
        if not text or text == outcome_text or text not in report_text:
            continue
        cited = [*fact_ids, outcome_id]
        codes: list[str] = []
        for code in sorted(_term_codes(text)):
            supported = set(fact_term_ids.get(code, set()))
            if code in outcome_codes:
                supported.add(outcome_id)
            if supported:
                authorization[code].update(supported)
                codes.append(code)
        claims.append(
            {
                "claim_id": claim_id,
                "text_zh": text,
                "fact_ids": cited,
                "term_codes": codes,
            }
        )

    authorized_terms = [
        {"term_code": code, "fact_ids": sorted(ids)}
        for code, ids in sorted(authorization.items())
        if ids
    ]
    return {
        "recording_id": recording_id,
        "report_kind": "eeg_report",
        "report_text_zh": report_text,
        "sections": sections,
        "required_sections": required_sections,
        "event_count": int(manifest["event_count"]),
        "claims": claims,
        "evidence": {
            "source_scope": "frozen_eeg_fact_ledger_only",
            "reportable_fact_ids": reportable,
            "authorized_terms": authorized_terms,
            "doctor_labels_used": False,
            "edf_annotations_used": False,
            "excel_fields_used": False,
        },
    }


def _technical_candidate(
    *, recording_id: str, manifest: Mapping[str, Any], html_text: str
) -> dict[str, Any]:
    report_text, sections = _surface(html_text)
    if "eeg_impression" not in sections:
        raise ValueError("technical report has no EEG-impression section")
    return {
        "recording_id": recording_id,
        "report_kind": "technical_unassessable_report",
        "report_text_zh": report_text,
        "sections": sections,
        "required_sections": ["eeg_impression"],
        "event_count": int(manifest["event_count"]),
        "claims": [],
        "evidence": {
            "source_scope": "frozen_eeg_fact_ledger_only",
            "reportable_fact_ids": [],
            "authorized_terms": [],
            "doctor_labels_used": False,
            "edf_annotations_used": False,
            "excel_fields_used": False,
        },
    }


def _normalized_language_projection(
    bundle: Mapping[str, Any],
    language: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Rebind language scope and manifest counts to the current renderer gate."""

    if language is None:
        return None, {
            "configured": False,
            "qwen_requested": False,
            "event_count": 0,
            "validated_qwen_wording_count": 0,
            "deterministic_fallback_count": 0,
            "language_failure_blocks_report_publication": False,
        }
    if set(language) != _LANGUAGE_LAYER_KEYS:
        raise ValueError("frozen language layer schema drifted")
    if language.get("schema_version") != render_module.LANGUAGE_LAYER_SCHEMA:
        raise ValueError("frozen language layer version is unsupported")
    if not isinstance(language.get("qwen_requested"), bool):
        raise TypeError("frozen language qwen_requested must be boolean")
    raw_records = language.get("event_records")
    event_count = bundle.get("event_count")
    if (
        not isinstance(raw_records, list)
        or isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or len(raw_records) != event_count
    ):
        raise ValueError("frozen language records do not cover the bundle events")
    layer = deepcopy(dict(language))
    projection = render_module._fact_locked_event_language(  # noqa: SLF001
        bundle, layer
    )
    projected_event_count = len(projection)
    layer["scope_receipt"] = {
        "clinical_eeg_fact_ledgers_sent": bool(
            layer["qwen_requested"] is True and raw_records
        ),
        "source_context_sent": False,
        "edf_annotation_sent": False,
        "excel_observation_sent": False,
        "waveform_image_or_path_sent": False,
        "research_soz_ranking_sent": False,
        "may_change_event_count": False,
        "may_change_event_coordinates": False,
        "may_change_recording_impression": False,
        "used_by_deterministic_renderer": projected_event_count > 0,
        "bounded_event_wording_projection_eligible_count": projected_event_count,
        "projection_generator_must_equal": "qwen3.6_facts_locked_draft",
        "projection_excludes_findings_and_impression": True,
        "prompt_or_schema_content_persisted": False,
        "request_audit_hashes_only": True,
        "prompt_firewall_fail_closed": True,
    }
    rebound = render_module._fact_locked_event_language(  # noqa: SLF001
        bundle, layer
    )
    if rebound != projection:
        raise ValueError("language scope normalization changed renderer projection")
    return layer, {
        "configured": True,
        "qwen_requested": bool(layer["qwen_requested"]),
        "event_count": event_count,
        "validated_qwen_wording_count": projected_event_count,
        "deterministic_fallback_count": event_count - projected_event_count,
        "language_failure_blocks_report_publication": False,
    }


def _rerender_eeg_report(
    *,
    source_manifest: Mapping[str, Any],
    source_artifacts: Mapping[PurePosixPath, Path],
    target_directory: Path,
    recording_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_directory.mkdir(parents=True, exist_ok=False)
    source_hashes = source_manifest["artifacts"]
    for relative, source in source_artifacts.items():
        if relative in {
            PurePosixPath("report.html"),
            PurePosixPath("report.docx"),
            PurePosixPath("language_records.json"),
        }:
            continue
        _copy_verified(source, target_directory.joinpath(*relative.parts), source_hashes[relative.as_posix()])

    bundle_path = target_directory / "bundle.json"
    bundle = _strict_json_artifact(bundle_path, "frozen portable EEG bundle")
    if bundle.get("recording_id") != recording_id:
        raise ValueError("portable EEG bundle identity drifted")
    current_outcome = classify_recording_eeg_outcome(bundle)
    source_outcome = source_manifest.get("diagnostic_outcome")
    if not isinstance(source_outcome, Mapping):
        raise ValueError("source report has no diagnostic outcome")
    # The classifier schema subsequently added an explicit nullable spatial
    # consensus field.  Permit only this non-semantic normalization; status,
    # conclusion, evidence counts and every other field must remain identical.
    normalized_source_outcome = dict(source_outcome)
    normalized_source_outcome.setdefault("spatial_consensus", None)
    if normalized_source_outcome != current_outcome:
        raise ValueError(
            "frozen diagnostic decision changed during renderer migration: "
            + recording_id
        )
    source_language_path = source_artifacts.get(
        PurePosixPath("language_records.json")
    )
    source_language = (
        _strict_json_artifact(source_language_path, "frozen event-language records")
        if source_language_path is not None
        else None
    )
    language, language_receipt = _normalized_language_projection(
        bundle, source_language
    )
    language_path = target_directory / "language_records.json"
    if language is not None:
        _write_json(language_path, language)

    waveform_hrefs: dict[str, str] = {}
    waveform_paths: dict[str, Path] = {}
    raw_events = bundle.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("portable EEG bundle events must be an array")
    declared_waveforms: set[PurePosixPath] = set()
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            raise TypeError("portable EEG bundle event must be an object")
        event_id = raw_event.get("eeg_event_id")
        attachment = raw_event.get("waveform_attachment")
        if not isinstance(event_id, str) or not isinstance(attachment, Mapping):
            raise ValueError("portable event waveform binding is malformed")
        relative = _safe_relative(attachment.get("figure_file"), "waveform figure path")
        if len(relative.parts) != 2 or relative.parts[0] != "waveforms" or relative.suffix.lower() != ".png":
            raise ValueError("portable bundle waveform path is outside the whitelist")
        expected_sha = attachment.get("figure_sha256")
        if source_hashes.get(relative.as_posix()) != expected_sha:
            raise ValueError("waveform bundle and report-manifest hashes disagree")
        target_waveform = target_directory.joinpath(*relative.parts)
        if _sha256_file(target_waveform) != expected_sha:
            raise ValueError("copied waveform differs from the portable bundle")
        declared_waveforms.add(relative)
        waveform_hrefs[event_id] = relative.as_posix()
        waveform_paths[event_id] = target_waveform
    ledger_waveforms = {
        relative for relative in source_artifacts if relative.parts[0] == "waveforms"
    }
    if declared_waveforms != ledger_waveforms:
        raise ValueError("waveform artifact set differs from bundle attachments")

    html_text = render_module.render_long_term_html(
        bundle,
        waveform_hrefs=waveform_hrefs,
        language_layer=language,
    )
    html_path = target_directory / "report.html"
    html_path.write_text(html_text, encoding="utf-8")
    os.chmod(html_path, 0o600)
    docx_path = target_directory / "report.docx"
    render_module.render_long_term_docx(
        docx_path,
        bundle,
        waveform_paths=waveform_paths,
        language_layer=language,
    )
    os.chmod(docx_path, 0o600)

    manifest = deepcopy(dict(source_manifest))
    manifest["diagnostic_outcome"] = current_outcome
    manifest["diagnostic_status"] = current_outcome["report_status"]
    artifacts = deepcopy(dict(source_hashes))
    if language is not None:
        artifacts["language_records.json"] = _sha256_file(language_path)
    artifacts["report.html"] = _sha256_file(html_path)
    artifacts["report.docx"] = _sha256_file(docx_path)
    manifest["artifacts"] = artifacts
    manifest["language_service_receipt"] = language_receipt
    manifest_path = target_directory / "manifest.json"
    _write_json(manifest_path, manifest)
    candidate = _eeg_candidate(
        recording_id=recording_id,
        bundle=bundle,
        manifest=manifest,
        html_text=html_text,
    )
    return manifest, candidate


def _copy_technical_report(
    *,
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
    source_artifacts: Mapping[PurePosixPath, Path],
    target_directory: Path,
    recording_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_directory.mkdir(parents=True, exist_ok=False)
    hashes = source_manifest["artifacts"]
    for relative, source in source_artifacts.items():
        _copy_verified(source, target_directory.joinpath(*relative.parts), hashes[relative.as_posix()])
    manifest_hash = _sha256_file(source_manifest_path)
    _copy_verified(source_manifest_path, target_directory / "manifest.json", manifest_hash)
    html_text = (target_directory / "report.html").read_text(encoding="utf-8")
    candidate = _technical_candidate(
        recording_id=recording_id,
        manifest=source_manifest,
        html_text=html_text,
    )
    return dict(source_manifest), candidate


def rerender_full_report_layer(
    *,
    inventory_path: Path,
    combined_coverage_path: Path,
    primary_root: Path,
    recovery_root: Path,
    remediation_root: Path,
    output_root: Path,
    expected_record_count: int | None = None,
    expected_subject_count: int | None = None,
) -> dict[str, Any]:
    """Publish a full renderer-v2 tree without opening any EEG/source labels."""

    inventory_raw, inventory_sha = _strict_json_file(inventory_path, "inventory")
    inventory = batch.validate_inventory(inventory_raw)
    combined_raw, combined_sha = _strict_json_file(
        combined_coverage_path, "combined coverage"
    )
    combined = release._validate_combined_coverage(  # noqa: SLF001
        combined_raw,
        inventory=inventory,
        inventory_sha256=inventory_sha,
    )
    if expected_record_count is not None and inventory["record_count"] != expected_record_count:
        raise ValueError("inventory record count differs from expectation")
    if expected_subject_count is not None and inventory["subject_count"] != expected_subject_count:
        raise ValueError("inventory subject count differs from expectation")

    roots = {
        "primary": _regular_root(primary_root, "primary root"),
        "recovery": _regular_root(recovery_root, "recovery root"),
        "remediation": _regular_root(remediation_root, "remediation root"),
    }
    if output_root.is_symlink():
        raise ValueError("output root must not be a symlink")
    output = output_root.resolve()
    for source_root in roots.values():
        if (
            output == source_root
            or output.is_relative_to(source_root)
            or source_root.is_relative_to(output)
        ):
            raise ValueError("output root must be independent of every source root")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        coverage_rows: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        source_manifest_snapshots: list[tuple[Path, str]] = []
        for recording, row in zip(
            inventory["records"], combined["records"], strict=True
        ):
            recording_id = str(recording["recording_id"])
            source_name = str(row["artifact_source"])
            source_root = roots[source_name]
            source_directory, source_manifest, artifacts, state_path = (
                _validated_source_report(
                    root=source_root,
                    row=row,
                    recording=recording,
                )
            )
            source_manifest_path = source_directory / "manifest.json"
            source_manifest_snapshots.extend(
                [
                    (source_manifest_path, str(row["report_manifest_sha256"])),
                    (state_path, str(row["state_manifest_sha256"])),
                ]
            )
            case_target = staging / "records" / recording_id
            case_target.mkdir(parents=True, exist_ok=False)
            _copy_verified(
                state_path,
                case_target / "state.json",
                str(row["state_manifest_sha256"]),
            )
            technical_failure_receipt_sha: str | None = None
            if row["effective_report_kind"] == "eeg_report":
                target_directory = case_target / "report"
                target_manifest, candidate = _rerender_eeg_report(
                    source_manifest=source_manifest,
                    source_artifacts=artifacts,
                    target_directory=target_directory,
                    recording_id=recording_id,
                )
                technical_relative = None
                run_status = "completed"
            else:
                source_relative = _safe_relative(
                    row["report_manifest_relative_path"], "technical report path"
                )
                expected_prefix = PurePosixPath("records") / recording_id
                local_parts = source_relative.parts[len(expected_prefix.parts) : -1]
                if len(local_parts) != 2 or local_parts[0] != "technical_reports":
                    raise ValueError("technical report locator is non-canonical")
                technical_relative = PurePosixPath(*local_parts).as_posix()
                (
                    technical_failure_receipt_path,
                    _,
                    technical_failure_receipt_sha,
                ) = _validated_technical_failure_receipt(
                    root=source_root,
                    row=row,
                    recording=recording,
                    report_directory=source_directory,
                    report_manifest=source_manifest,
                    state_path=state_path,
                )
                source_manifest_snapshots.append(
                    (
                        technical_failure_receipt_path,
                        technical_failure_receipt_sha,
                    )
                )
                _copy_verified(
                    technical_failure_receipt_path,
                    case_target / "technical_failure_receipt.json",
                    technical_failure_receipt_sha,
                )
                target_directory = case_target.joinpath(*local_parts)
                target_manifest, candidate = _copy_technical_report(
                    source_manifest_path=source_manifest_path,
                    source_manifest=source_manifest,
                    source_artifacts=artifacts,
                    target_directory=target_directory,
                    recording_id=recording_id,
                )
                run_status = "completed_technical_unassessable"
            candidates.append(candidate)
            coverage_rows.append(
                batch._case_result(  # noqa: SLF001
                    recording,
                    run_status=run_status,
                    diagnostic_status=str(row["diagnostic_status"]),
                    event_count=int(row["event_count"]),
                    failure_stage=row["failure_stage"],
                    technical_artifact_relative_dir=technical_relative,
                )
            )
            target_manifest_path = target_directory / "manifest.json"
            receipts.append(
                {
                    "recording_id": recording_id,
                    "report_kind": row["effective_report_kind"],
                    "artifact_source": source_name,
                    "source_report_manifest_sha256": row["report_manifest_sha256"],
                    "target_report_manifest_sha256": _sha256_file(target_manifest_path),
                    "technical_failure_receipt_sha256": (
                        technical_failure_receipt_sha
                    ),
                    "diagnostic_status_unchanged": (
                        target_manifest["diagnostic_status"] == row["diagnostic_status"]
                    ),
                    "event_count_unchanged": (
                        target_manifest["event_count"] == row["event_count"]
                    ),
                }
            )

        coverage_path = staging / "coverage_manifest.json"
        coverage = batch._write_coverage(  # noqa: SLF001
            output=coverage_path,
            inventory=inventory,
            rows=coverage_rows,
            mode="execution",
            qwen_requested=False,
        )
        if not coverage["dataset_coverage_complete"]:
            raise ValueError("rerendered cohort did not retain full artifact coverage")

        cohort_id = "RERENDER-V2-" + _canonical_sha256(
            {
                "inventory_id": inventory["inventory_id"],
                "combined_coverage_sha256": combined_sha,
            }
        )[:24].upper()
        candidate_manifest = validate_candidate_manifest(
            {
                "schema_version": CANDIDATE_SCHEMA,
                "cohort_id": cohort_id,
                "records": candidates,
            }
        )
        candidate_path = staging / LANGUAGE_QUALITY_FILENAME
        _write_json(candidate_path, candidate_manifest)

        receipt_body = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "inventory_id": inventory["inventory_id"],
            "combined_coverage_id": combined["combined_coverage_id"],
            "record_count": inventory["record_count"],
            "subject_count": inventory["subject_count"],
            "eeg_report_count": coverage["completed_eeg_report_count"],
            "technical_unassessable_report_count": coverage[
                "technical_unassessable_report_count"
            ],
            "source_receipts": {
                "inventory_manifest_sha256": inventory_sha,
                "combined_coverage_manifest_sha256": combined_sha,
                "source_paths_persisted": False,
            },
            "output_receipts": {
                "coverage_manifest_sha256": _sha256_file(coverage_path),
                "language_quality_candidates_sha256": _sha256_file(candidate_path),
            },
            "records": receipts,
            "scope_receipt": {
                "report_layer_only": True,
                "frozen_bundle_read": True,
                "frozen_language_records_read_when_declared": True,
                "frozen_waveform_png_read": True,
                "current_html_and_docx_renderer_called": True,
                "qwen_or_other_llm_called": False,
                "edf_signal_read": False,
                "edf_annotation_read": False,
                "annotation_csv_read": False,
                "excel_or_workbook_read": False,
                "doctor_label_or_onset_field_read": False,
                "source_output_trees_modified": False,
                "technical_reports_copied_without_body_changes": True,
                "technical_failure_receipts_validated_and_copied": True,
                "language_projection_scope_rebound_to_current_renderer": True,
                "diagnostic_decision_or_event_counts_changed": False,
                "diagnostic_outcome_schema_normalized": True,
            },
        }
        receipt = {
            **receipt_body,
            "rerender_id": "PLRRV2-" + _canonical_sha256(receipt_body)[:24],
        }
        _write_json(staging / RERENDER_RECEIPT_FILENAME, receipt)

        if _sha256_file(inventory_path.resolve(strict=True)) != inventory_sha:
            raise ValueError("inventory changed during report-layer rerender")
        if _sha256_file(combined_coverage_path.resolve(strict=True)) != combined_sha:
            raise ValueError("combined coverage changed during report-layer rerender")
        for source_path, expected_sha in source_manifest_snapshots:
            if _sha256_file(source_path) != expected_sha:
                raise ValueError("a selected source manifest changed during rerender")
        for path in staging.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        os.chmod(staging, 0o700)
        os.replace(staging, output)
        os.chmod(output, 0o700)
        published = True
        return receipt
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected count must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--combined-coverage", type=Path, required=True)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--remediation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expect-records", type=_positive)
    parser.add_argument("--expect-subjects", type=_positive)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt = rerender_full_report_layer(
        inventory_path=args.inventory,
        combined_coverage_path=args.combined_coverage,
        primary_root=args.primary_root,
        recovery_root=args.recovery_root,
        remediation_root=args.remediation_root,
        output_root=args.output_root,
        expected_record_count=args.expect_records,
        expected_subject_count=args.expect_subjects,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "rerender_id": receipt["rerender_id"],
                "record_count": receipt["record_count"],
                "eeg_report_count": receipt["eeg_report_count"],
                "technical_unassessable_report_count": receipt[
                    "technical_unassessable_report_count"
                ],
                "output_root": str(args.output_root.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
