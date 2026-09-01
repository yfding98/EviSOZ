"""Conservative de-identification candidates for private physician reports.

The emitted texts are review candidates, not training targets.  A separate
human-reviewed release must explicitly promote development candidates for
Qwen supervision or locked candidates for evaluator-only use.
"""

from __future__ import annotations

from collections import Counter
import csv
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    build_raw_artifact_ref,
    canonical_json_sha256,
    sha256_bytes,
)
from src.evisoz.data.private_physician_reports import (
    DOCX_MEDIA_TYPE,
    PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
    validate_private_physician_report_inventory,
)


PRIVATE_REPORT_DEID_CANDIDATES_SCHEMA_VERSION = (
    "evisoz_private_physician_report_deidentification_candidates_v1"
)
DEIDENTIFIED_REPORT_TEXT_SCHEMA_VERSION = "evisoz_deidentified_physician_report_text_v1"
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_ID_PREFIX = "EVISOZ-DEID-"
_BUNDLE_ID_PREFIX = "EVISOZ-DEIDSET-"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE = re.compile(
    r"(?:(?:19|20)\d{2})\s*(?:年|[-/.])\s*\d{1,2}\s*(?:月|[-/.])\s*\d{1,2}\s*(?:日)?"
)
_LONG_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d{4,}(?![A-Za-z0-9])")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_FORBIDDEN_LABELS = (
    "姓名",
    "性别",
    "年龄",
    "出生日期",
    "住院号",
    "门诊号",
    "病历号",
    "病案号",
    "床号",
    "科室",
    "报告医师",
    "审核医师",
    "检查医师",
    "报告日期",
    "审核日期",
    "联系电话",
    "电话",
    "地址",
)


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = _hash_source(value)
    result["bundle_id"] = _PENDING_ID
    return result


def _candidate_id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["candidate_id"] = _PENDING_ID
    result["relative_text_path"] = "CONTENT-ADDRESS-PENDING"
    return result


def _docx_paragraphs(raw: bytes) -> list[str]:
    with ZipFile(BytesIO(raw)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{{{_WORD_NS}}}p"):
        fragments: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{{{_WORD_NS}}}t" and node.text:
                fragments.append(node.text)
            elif node.tag == f"{{{_WORD_NS}}}tab":
                fragments.append("\t")
            elif node.tag in {f"{{{_WORD_NS}}}br", f"{{{_WORD_NS}}}cr"}:
                fragments.append("\n")
        text = "".join(fragments).strip()
        if text:
            paragraphs.append(text)
    if not paragraphs:
        raise ValueError("physician report has no extractable paragraphs")
    return paragraphs


def _source_patient_names(path: Path) -> tuple[list[str], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("private report de-identification source manifest is invalid")
    raw = path.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    names = sorted({str(row.get("base_patient_id", "")).strip() for row in rows})
    if not names or "" in names:
        raise ValueError("private report de-identification patient-name authority is invalid")
    return names, raw


def _safe_candidate_text(
    paragraphs: Sequence[str],
    *,
    patient_names: Sequence[str],
    source_stems: Sequence[str],
) -> tuple[str, dict[str, object], dict[str, object]]:
    ictal_indices = [index for index, text in enumerate(paragraphs) if "发作期" in text]
    impression_indices = [index for index, text in enumerate(paragraphs) if "印象" in text]
    if ictal_indices:
        start = ictal_indices[0]
        route = "first_ictal_section"
    elif impression_indices:
        start = impression_indices[0]
        route = "impression_only_fallback"
    else:
        raise ValueError("physician report lacks an ictal or impression section")
    signature_indices = [
        index
        for index, text in enumerate(paragraphs)
        if index > start and "报告医师" in text
    ]
    if not signature_indices:
        raise ValueError("physician report lacks a report-physician truncation boundary")
    stop = signature_indices[0]
    if stop <= start:
        raise ValueError("physician report clinical extraction interval is empty")
    selected = list(paragraphs[start:stop])
    selected = [
        text
        for text in selected
        if not any(label in text for label in _FORBIDDEN_LABELS)
    ]
    text = "\n".join(selected).strip()
    if not text:
        raise ValueError("physician report de-identification removed all clinical text")
    literal_redaction_count = 0
    sensitive_literals = sorted(
        {item for item in [*patient_names, *source_stems] if item},
        key=len,
        reverse=True,
    )
    for literal in sensitive_literals:
        hits = text.count(literal)
        if hits:
            text = text.replace(literal, "<PERSON>")
            literal_redaction_count += hits
    regex_redaction_counts = {
        "email": len(_EMAIL_RE.findall(text)),
        "phone": len(_PHONE_RE.findall(text)),
        "date": len(_DATE_RE.findall(text)),
        "long_number": len(_LONG_NUMBER_RE.findall(text)),
    }
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    text = _DATE_RE.sub("<DATE>", text)
    text = _LONG_NUMBER_RE.sub("<ID>", text)
    residual_name_hits = sum(literal in text for literal in patient_names if literal)
    residual_stem_hits = sum(literal in text for literal in source_stems if literal)
    residual_label_hits = sum(label in text for label in _FORBIDDEN_LABELS)
    residual_regex_hits = {
        "email": len(_EMAIL_RE.findall(text)),
        "phone": len(_PHONE_RE.findall(text)),
        "date": len(_DATE_RE.findall(text)),
        "long_number": len(_LONG_NUMBER_RE.findall(text)),
    }
    if (
        residual_name_hits
        or residual_stem_hits
        or residual_label_hits
        or any(residual_regex_hits.values())
    ):
        raise ValueError("physician report de-identification residual PHI scan failed")
    extraction = {
        "route": route,
        "source_nonempty_paragraph_count": len(paragraphs),
        "selected_start_paragraph_index": start,
        "selected_stop_paragraph_index_exclusive": stop,
        "released_nonempty_paragraph_count": len(
            [item for item in text.splitlines() if item.strip()]
        ),
        "signature_boundary_found": True,
    }
    scan = {
        "automated_scan_status": "pass",
        "literal_redaction_count": literal_redaction_count,
        "regex_redaction_counts": regex_redaction_counts,
        "residual_patient_name_hits": residual_name_hits,
        "residual_source_stem_hits": residual_stem_hits,
        "residual_forbidden_label_hits": residual_label_hits,
        "residual_regex_hits": residual_regex_hits,
    }
    return text, extraction, scan


def _safe_relative_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("de-identification candidate path must be a string")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError("de-identification candidate path is unsafe")
    path = root.joinpath(*rel.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError("de-identification candidate text must be a regular file")
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    return resolved


def build_private_report_deidentification_candidates(
    *,
    report_paths: Sequence[Path],
    report_inventory: Mapping[str, object],
    source_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Extract privacy-scanned review candidates without releasing any loss."""

    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inventory = validate_private_physician_report_inventory(report_inventory)
    patient_names, source_manifest_raw = _source_patient_names(source_manifest_path)
    source_stems = [path.stem for path in report_paths]
    path_by_digest: dict[str, Path] = {}
    for path in report_paths:
        if path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".docx":
            raise ValueError("private report de-identification accepts regular DOCX only")
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        if digest in path_by_digest:
            raise ValueError("private report de-identification source bytes are duplicated")
        path_by_digest[digest] = path
    expected_digests = {
        row["document_ref"]["content_hash"]["sha256"]
        for row in inventory["reports"]
    }
    if set(path_by_digest) != expected_digests:
        raise ValueError("private report files do not reproduce the frozen report inventory")
    output.mkdir(parents=True)
    candidates: list[dict[str, object]] = []
    role_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    for report in inventory["reports"]:
        digest = report["document_ref"]["content_hash"]["sha256"]
        raw = path_by_digest[digest].read_bytes()
        text, extraction, scan = _safe_candidate_text(
            _docx_paragraphs(raw),
            patient_names=patient_names,
            source_stems=source_stems,
        )
        text_bytes = (text + "\n").encode("utf-8")
        text_ref = build_raw_artifact_ref(
            text_bytes,
            artifact_kind="deidentified_physician_report_candidate",
            media_type="text/plain; charset=utf-8",
            payload_schema_version=DEIDENTIFIED_REPORT_TEXT_SCHEMA_VERSION,
        )
        row: dict[str, Any] = {
            "candidate_id": _PENDING_ID,
            "report_id": report["report_id"],
            "association": deepcopy(report["association"]),
            "text_ref": text_ref,
            "relative_text_path": "CONTENT-ADDRESS-PENDING",
            "extraction": extraction,
            "automated_phi_scan": scan,
            "review_release": {
                "manual_review_status": "pending",
                "development_qwen_training_released": False,
                "locked_language_evaluation_released": False,
                "report_text_can_supervise_localization": False,
            },
        }
        row["candidate_id"] = _ID_PREFIX + canonical_json_sha256(
            _candidate_id_source(row)
        )[:24]
        row["relative_text_path"] = f"candidates/{row['candidate_id']}.txt"
        target = output.joinpath(*PurePosixPath(row["relative_text_path"]).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text_bytes)
        candidates.append(row)
        route_counts[extraction["route"]] += 1
        assignment = report["association"]["split_assignment"]
        role_counts[
            assignment["evisoz_role"] if assignment is not None else "unresolved"
        ] += 1
    candidates.sort(key=lambda row: str(row["candidate_id"]))
    inventory_ref = build_json_artifact_ref(
        inventory,
        artifact_kind="physician_report_inventory",
        payload_schema_version=PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
    )
    body: dict[str, Any] = {
        "schema_version": PRIVATE_REPORT_DEID_CANDIDATES_SCHEMA_VERSION,
        "bundle_id": _PENDING_ID,
        "source_bindings": {
            "physician_report_inventory_ref": inventory_ref,
            "patient_name_authority_ref": build_raw_artifact_ref(
                source_manifest_raw,
                artifact_kind="private_label_authority_manifest",
                media_type="text/csv",
            ),
        },
        "candidates": candidates,
        "counts": {
            "candidate_count": len(candidates),
            "automated_phi_scan_pass_count": len(candidates),
            "split_role_candidate_counts": dict(sorted(role_counts.items())),
            "extraction_route_counts": dict(sorted(route_counts.items())),
            "manual_review_pass_count": 0,
            "development_qwen_training_release_count": 0,
            "locked_language_evaluation_release_count": 0,
        },
        "deidentification_policy": {
            "policy_id": "ictal_or_impression_to_pre_signature_conservative_v1",
            "header_demographics_excluded": True,
            "report_physician_signature_and_tail_excluded": True,
            "known_patient_names_redacted": True,
            "dates_contacts_and_long_identifiers_redacted": True,
            "automated_scan_is_manual_approval": False,
            "unreviewed_candidate_can_train_qwen": False,
            "unreviewed_candidate_can_enter_language_evaluation": False,
            "locked_candidate_can_enter_training": False,
            "report_text_can_supervise_localization": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["bundle_id"] = _BUNDLE_ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_private_report_deidentification_candidates(
        body,
        output_root=output,
    )


def validate_private_report_deidentification_candidates(
    value: object,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Reopen every candidate and verify hashes plus the fail-closed release gate."""

    if type(value) is not dict or set(value) != {
        "schema_version",
        "bundle_id",
        "source_bindings",
        "candidates",
        "counts",
        "deidentification_policy",
        "receipt_sha256",
    }:
        raise ValueError("private report de-identification candidate fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PRIVATE_REPORT_DEID_CANDIDATES_SCHEMA_VERSION:
        raise ValueError("private report de-identification candidate schema drifted")
    candidates = data["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("private report de-identification candidate bundle is empty")
    if candidates != sorted(candidates, key=lambda row: row["candidate_id"]) or len(
        {row["candidate_id"] for row in candidates}
    ) != len(candidates):
        raise ValueError("private report de-identification candidates are not uniquely sorted")
    role_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    for row in candidates:
        if type(row) is not dict or set(row) != {
            "candidate_id",
            "report_id",
            "association",
            "text_ref",
            "relative_text_path",
            "extraction",
            "automated_phi_scan",
            "review_release",
        }:
            raise ValueError("private report de-identification candidate row drifted")
        text_path = _safe_relative_file(output_root, row["relative_text_path"])
        raw = text_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("de-identification candidate is not UTF-8") from exc
        expected_ref = build_raw_artifact_ref(
            raw,
            artifact_kind="deidentified_physician_report_candidate",
            media_type="text/plain; charset=utf-8",
            payload_schema_version=DEIDENTIFIED_REPORT_TEXT_SCHEMA_VERSION,
        )
        if row["text_ref"] != expected_ref or not text.strip():
            raise ValueError("de-identification candidate text reference drifted")
        if any(label in text for label in _FORBIDDEN_LABELS) or any(
            pattern.search(text)
            for pattern in (_DATE_RE, _PHONE_RE, _EMAIL_RE, _LONG_NUMBER_RE)
        ):
            raise ValueError("de-identification candidate failed generic PHI replay scan")
        scan = row["automated_phi_scan"]
        if (
            scan.get("automated_scan_status") != "pass"
            or scan.get("residual_patient_name_hits") != 0
            or scan.get("residual_source_stem_hits") != 0
            or scan.get("residual_forbidden_label_hits") != 0
            or any(scan.get("residual_regex_hits", {}).values())
        ):
            raise ValueError("de-identification candidate recorded a residual PHI hit")
        if row["review_release"] != {
            "manual_review_status": "pending",
            "development_qwen_training_released": False,
            "locked_language_evaluation_released": False,
            "report_text_can_supervise_localization": False,
        }:
            raise ValueError("unreviewed de-identification candidate was released")
        expected_id = _ID_PREFIX + canonical_json_sha256(
            _candidate_id_source(row)
        )[:24]
        if row["candidate_id"] != expected_id:
            raise ValueError("de-identification candidate ID drifted")
        assignment = row["association"]["split_assignment"]
        role_counts[
            assignment["evisoz_role"] if assignment is not None else "unresolved"
        ] += 1
        route_counts[row["extraction"]["route"]] += 1
    expected_counts = {
        "candidate_count": len(candidates),
        "automated_phi_scan_pass_count": len(candidates),
        "split_role_candidate_counts": dict(sorted(role_counts.items())),
        "extraction_route_counts": dict(sorted(route_counts.items())),
        "manual_review_pass_count": 0,
        "development_qwen_training_release_count": 0,
        "locked_language_evaluation_release_count": 0,
    }
    if data["counts"] != expected_counts:
        raise ValueError("private report de-identification candidate counts drifted")
    if data["deidentification_policy"] != {
        "policy_id": "ictal_or_impression_to_pre_signature_conservative_v1",
        "header_demographics_excluded": True,
        "report_physician_signature_and_tail_excluded": True,
        "known_patient_names_redacted": True,
        "dates_contacts_and_long_identifiers_redacted": True,
        "automated_scan_is_manual_approval": False,
        "unreviewed_candidate_can_train_qwen": False,
        "unreviewed_candidate_can_enter_language_evaluation": False,
        "locked_candidate_can_enter_training": False,
        "report_text_can_supervise_localization": False,
    }:
        raise ValueError("private report de-identification policy drifted")
    expected_bundle_id = _BUNDLE_ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["bundle_id"] != expected_bundle_id:
        raise ValueError("private report de-identification bundle ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("private report de-identification bundle hash drifted")
    return data


__all__ = [
    "PRIVATE_REPORT_DEID_CANDIDATES_SCHEMA_VERSION",
    "DEIDENTIFIED_REPORT_TEXT_SCHEMA_VERSION",
    "build_private_report_deidentification_candidates",
    "validate_private_report_deidentification_candidates",
]
