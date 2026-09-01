"""Privacy-safe inventory and split binding for private physician reports.

The source reports contain protected health information.  This module may read
that information in-memory to establish an exact association, but the emitted
receipt intentionally contains no source path, filename, patient name, or
report text.  Raw reports are represented only by content-addressed byte
references and are never copied into the EviSOZ output directory.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
from io import BytesIO
import re
from pathlib import Path
import unicodedata
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

from .artifact_ref import (
    build_json_artifact_ref,
    build_raw_artifact_ref,
    canonical_json_sha256,
    sha256_bytes,
    validate_artifact_ref,
)
from .private_stage0_split import build_private_patient_linkage_group
from .split_ledger import (
    SPLIT_ROSTER_SCHEMA_VERSION,
    validate_split_roster,
)


PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION = (
    "evisoz_private_physician_report_inventory_v1"
)
PHYSICIAN_REPORT_DOCUMENT_SCHEMA_VERSION = "private_physician_report_docx_v1"
DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64
_REPORT_ID_PREFIX = "EVISOZ-PRPT-"
_INVENTORY_ID_PREFIX = "EVISOZ-REPORTS-"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PATIENT_RE = re.compile(r"^PRIV-P[0-9]{3}$")
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_ALLOWED_ASSOCIATION_STATUS = {"linked_high_confidence", "unresolved"}
_ALLOWED_ASSOCIATION_BASIS = {
    "source_filename_unique_patient_substring",
    "physician_report_name_field_unique_patient_substring",
    "none",
}


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = _hash_source(value)
    result["inventory_id"] = _PENDING_ID
    return result


def _normalized_match_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _read_csv(path: Path) -> tuple[list[dict[str, str]], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("private physician-report authority CSV must be a regular file")
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("private physician-report authority CSV is empty")
    return rows, raw


def _zip_member_is_symlink(info: object) -> bool:
    mode = (int(getattr(info, "external_attr")) >> 16) & 0o170000
    return mode == 0o120000


def _docx_text_and_audit(raw: bytes) -> tuple[str, dict[str, object]]:
    try:
        with ZipFile(BytesIO(raw)) as archive:
            if archive.testzip() is not None:
                raise ValueError("physician report DOCX has a corrupt ZIP member")
            members = archive.infolist()
            names = [item.filename for item in members]
            if any(
                name.startswith("/")
                or ".." in Path(name).parts
                or _zip_member_is_symlink(item)
                for name, item in zip(names, members)
            ):
                raise ValueError("physician report DOCX contains an unsafe member")
            required = {"[Content_Types].xml", "word/document.xml"}
            if not required.issubset(names):
                raise ValueError("physician report DOCX lacks required OOXML members")
            if any(name.lower().endswith("vbaproject.bin") for name in names):
                raise ValueError("macro-enabled physician reports are not admitted")
            document_xml = archive.read("word/document.xml")
            try:
                root = ET.fromstring(document_xml)
            except ET.ParseError as exc:
                raise ValueError("physician report document.xml is invalid") from exc
            paragraphs: list[str] = []
            for paragraph in root.iter(f"{{{_WORD_NS}}}p"):
                fragments: list[str] = []
                for node in paragraph.iter():
                    if node.tag == f"{{{_WORD_NS}}}t" and node.text:
                        fragments.append(node.text)
                    elif node.tag == f"{{{_WORD_NS}}}tab":
                        fragments.append("\t")
                    elif node.tag in {
                        f"{{{_WORD_NS}}}br",
                        f"{{{_WORD_NS}}}cr",
                    }:
                        fragments.append("\n")
                text = "".join(fragments).strip()
                if text:
                    paragraphs.append(text)
            report_text = "\n".join(paragraphs).strip()
            if not report_text:
                raise ValueError("physician report DOCX has no extractable body text")
            external_relationship_count = 0
            relationship_namespaces = {_REL_NS, _OFFICE_REL_NS}
            for relationship_path in (
                "_rels/.rels",
                "word/_rels/document.xml.rels",
            ):
                if relationship_path not in names:
                    continue
                try:
                    relationship_root = ET.fromstring(
                        archive.read(relationship_path)
                    )
                except ET.ParseError as exc:
                    raise ValueError("physician report relationship XML is invalid") from exc
                for node in relationship_root.iter():
                    if node.tag.rsplit("}", 1)[0].lstrip("{") not in relationship_namespaces:
                        continue
                    if node.attrib.get("TargetMode") == "External":
                        external_relationship_count += 1
            audit = {
                "container": "docx",
                "zip_member_count": len(members),
                "uncompressed_size_bytes": sum(item.file_size for item in members),
                "document_xml_sha256": sha256_bytes(document_xml),
                "nonempty_paragraph_count": len(paragraphs),
                "body_character_count": len(report_text),
                "external_relationship_count": external_relationship_count,
                "macros_present": False,
                "parse_status": "valid",
            }
            return report_text, audit
    except BadZipFile as exc:
        raise ValueError("physician report is not a valid DOCX ZIP container") from exc


def _patient_authority(
    *,
    source_rows: Sequence[Mapping[str, str]],
    signal_rows: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    if any("base_patient_id" not in row for row in source_rows):
        raise ValueError("private source manifest lacks base_patient_id")
    raw_patients = sorted(
        {str(row["base_patient_id"]).strip() for row in source_rows}
    )
    if not raw_patients or "" in raw_patients:
        raise ValueError("private source manifest contains an empty patient identity")
    if len({_normalized_match_text(name) for name in raw_patients}) != len(raw_patients):
        raise ValueError("private patient names are not unique after match normalization")
    authority = {
        name: f"PRIV-P{index + 1:03d}"
        for index, name in enumerate(raw_patients)
    }
    if any("patient_id" not in row for row in signal_rows):
        raise ValueError("private signal roster lacks patient_id")
    signal_patients = {str(row["patient_id"]).strip() for row in signal_rows}
    if signal_patients != set(authority.values()):
        raise ValueError(
            "private source-manifest identity order does not reproduce the frozen signal roster"
        )
    return authority


def _trusted_split_context(
    split_roster: Mapping[str, object],
    patient_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if any(_SAFE_PATIENT_RE.fullmatch(value) is None for value in patient_ids):
        raise ValueError("private patient pseudonym authority drifted")
    groups = {
        patient_id: build_private_patient_linkage_group(patient_id)
        for patient_id in patient_ids
    }
    trusted_by_group = {
        group["linkage_group_id"]: group for group in groups.values()
    }
    roster = validate_split_roster(
        split_roster,
        trusted_linkage_groups=trusted_by_group,
    )
    assignments = {
        row["linkage_group_id"]: row for row in roster["assignments"]
    }
    return groups, assignments


def _association_candidate(
    *,
    source_filename: str,
    report_text: str,
    raw_patient_names: Sequence[str],
) -> tuple[str | None, str, str]:
    filename_text = _normalized_match_text(Path(source_filename).stem)
    filename_matches = [
        name
        for name in raw_patient_names
        if _normalized_match_text(name) in filename_text
    ]
    if len(filename_matches) == 1:
        return (
            filename_matches[0],
            "linked_high_confidence",
            "source_filename_unique_patient_substring",
        )
    if len(filename_matches) > 1:
        return None, "unresolved", "none"
    normalized_body = _normalized_match_text(report_text)
    name_field_matches = [
        name
        for name in raw_patient_names
        if "姓名" + _normalized_match_text(name) in normalized_body
    ]
    if len(name_field_matches) == 1:
        return (
            name_field_matches[0],
            "linked_high_confidence",
            "physician_report_name_field_unique_patient_substring",
        )
    return None, "unresolved", "none"


def _source_ref(raw: bytes, *, payload_schema_version: str | None = None) -> dict[str, Any]:
    return build_raw_artifact_ref(
        raw,
        artifact_kind="private_authority_source",
        media_type="text/csv",
        payload_schema_version=payload_schema_version,
    )


def build_private_physician_report_inventory(
    *,
    report_paths: Sequence[Path],
    source_manifest_path: Path,
    signal_roster_path: Path,
    split_roster: Mapping[str, object],
) -> dict[str, Any]:
    """Build a closed, privacy-safe physician-report inventory receipt."""

    if not report_paths:
        raise ValueError("private physician report inventory is empty")
    source_rows, source_raw = _read_csv(source_manifest_path)
    signal_rows, signal_raw = _read_csv(signal_roster_path)
    patient_authority = _patient_authority(
        source_rows=source_rows,
        signal_rows=signal_rows,
    )
    patient_ids = sorted(patient_authority.values())
    groups, assignments = _trusted_split_context(
        split_roster,
        patient_ids,
    )
    raw_patient_names = sorted(patient_authority)
    reports: list[dict[str, object]] = []
    linked_group_ids: set[str] = set()
    seen_document_digests: set[str] = set()
    for path in sorted(report_paths, key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError("private physician report must be a regular file")
        if path.suffix.casefold() != ".docx":
            raise ValueError("private physician report inventory accepts DOCX only")
        raw = path.read_bytes()
        document_ref = build_raw_artifact_ref(
            raw,
            artifact_kind="physician_authored_report_source",
            media_type=DOCX_MEDIA_TYPE,
            payload_schema_version=PHYSICIAN_REPORT_DOCUMENT_SCHEMA_VERSION,
        )
        document_digest = document_ref["content_hash"]["sha256"]
        if document_digest in seen_document_digests:
            raise ValueError("duplicate private physician report document bytes")
        seen_document_digests.add(document_digest)
        report_text, container_audit = _docx_text_and_audit(raw)
        raw_patient, status, basis = _association_candidate(
            source_filename=path.name,
            report_text=report_text,
            raw_patient_names=raw_patient_names,
        )
        report_id = _REPORT_ID_PREFIX + document_digest[:24]
        if raw_patient is None:
            association: dict[str, object] = {
                "status": status,
                "basis": basis,
                "linkage_group_id": None,
                "source_patient_sha256": None,
                "split_assignment": None,
            }
        else:
            patient_id = patient_authority[raw_patient]
            group = groups[patient_id]
            group_id = group["linkage_group_id"]
            if group_id in linked_group_ids:
                raise ValueError("multiple private physician reports map to one patient")
            linked_group_ids.add(group_id)
            assignment = assignments[group_id]
            association = {
                "status": status,
                "basis": basis,
                "linkage_group_id": group_id,
                "source_patient_sha256": group["members"][0][
                    "source_patient_sha256"
                ],
                "split_assignment": {
                    "evisoz_role": assignment["evisoz_role"],
                    "outer_holdout_fold": assignment["outer_holdout_fold"],
                    "locked": assignment["locked"],
                },
            }
        reports.append(
            {
                "report_id": report_id,
                "document_ref": document_ref,
                "authorship": "physician_authored",
                "container_audit": container_audit,
                "association": association,
                "deidentification": {
                    "source_contains_phi": True,
                    "deidentified_text_released": False,
                    "status": "pending_manual_review",
                    "eligible_for_qwen_training": False,
                    "eligible_for_language_evaluation": False,
                },
            }
        )
    reports.sort(key=lambda row: str(row["report_id"]))
    association_counts = Counter(
        str(row["association"]["status"]) for row in reports
    )
    association_basis_counts = Counter(
        str(row["association"]["basis"]) for row in reports
    )
    split_role_counts = Counter(
        str(row["association"]["split_assignment"]["evisoz_role"])
        for row in reports
        if row["association"]["split_assignment"] is not None
    )
    split_ref = build_json_artifact_ref(
        split_roster,
        artifact_kind="split_roster",
        payload_schema_version=SPLIT_ROSTER_SCHEMA_VERSION,
    )
    body: dict[str, Any] = {
        "schema_version": PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
        "inventory_id": _PENDING_ID,
        "source_bindings": {
            "private_source_manifest_ref": _source_ref(source_raw),
            "frozen_signal_roster_ref": _source_ref(signal_raw),
            "split_roster_ref": split_ref,
            "patient_count": len(patient_ids),
            "report_document_set_sha256": canonical_json_sha256(
                [row["document_ref"] for row in reports]
            ),
        },
        "association_policy": {
            "policy_id": "private_report_exact_identity_two_route_v1",
            "filename_route": "unique_normalized_source_patient_substring",
            "body_fallback_route": "unique_normalized_name_field_substring",
            "fuzzy_matching_allowed": False,
            "manual_mapping_required_when_unresolved": True,
        },
        "reports": reports,
        "counts": {
            "report_count": len(reports),
            "valid_docx_count": sum(
                row["container_audit"]["parse_status"] == "valid"
                for row in reports
            ),
            "association_status_counts": dict(sorted(association_counts.items())),
            "association_basis_counts": dict(sorted(association_basis_counts.items())),
            "linked_split_role_counts": dict(sorted(split_role_counts.items())),
            "deidentified_text_release_count": 0,
            "qwen_training_eligible_count": 0,
            "language_evaluation_eligible_count": 0,
        },
        "manual_mapping_required_report_ids": sorted(
            row["report_id"]
            for row in reports
            if row["association"]["status"] == "unresolved"
        ),
        "privacy_and_use_contract": {
            "source_paths_or_filenames_persisted": False,
            "patient_names_persisted": False,
            "raw_report_text_persisted": False,
            "raw_report_bytes_copied": False,
            "source_documents_are_physician_authored": True,
            "source_documents_are_generated_text": False,
            "association_uses_phi_in_memory_only": True,
            "unreviewed_text_can_enter_qwen_training": False,
            "locked_test_text_can_enter_training": False,
            "report_text_can_supervise_localization": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["inventory_id"] = _INVENTORY_ID_PREFIX + canonical_json_sha256(
        _id_source(body)
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_private_physician_report_inventory(body)


def validate_private_physician_report_inventory(value: object) -> dict[str, Any]:
    """Validate the closed privacy and split contract of one inventory."""

    top_keys = {
        "schema_version",
        "inventory_id",
        "source_bindings",
        "association_policy",
        "reports",
        "counts",
        "manual_mapping_required_report_ids",
        "privacy_and_use_contract",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != top_keys:
        raise ValueError("private physician report inventory fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION:
        raise ValueError("private physician report inventory schema drifted")
    bindings = data["source_bindings"]
    if type(bindings) is not dict or set(bindings) != {
        "private_source_manifest_ref",
        "frozen_signal_roster_ref",
        "split_roster_ref",
        "patient_count",
        "report_document_set_sha256",
    }:
        raise ValueError("private physician report source bindings drifted")
    for key in ("private_source_manifest_ref", "frozen_signal_roster_ref"):
        ref = validate_artifact_ref(bindings[key])
        if (
            ref["artifact_kind"] != "private_authority_source"
            or ref["content_hash"]["domain"] != "raw_bytes_v1"
            or ref["media_type"] != "text/csv"
        ):
            raise ValueError("private physician report authority source type drifted")
    split_ref = validate_artifact_ref(bindings["split_roster_ref"])
    if (
        split_ref["artifact_kind"] != "split_roster"
        or split_ref["payload_schema_version"] != SPLIT_ROSTER_SCHEMA_VERSION
        or split_ref["content_hash"]["domain"] != "canonical_json_v1"
    ):
        raise ValueError("private physician report split binding type drifted")
    patient_count = bindings["patient_count"]
    if isinstance(patient_count, bool) or not isinstance(patient_count, int) or patient_count < 1:
        raise ValueError("private physician report patient_count is invalid")
    if not isinstance(bindings["report_document_set_sha256"], str) or _SHA256_RE.fullmatch(
        bindings["report_document_set_sha256"]
    ) is None:
        raise ValueError("private physician report document-set digest is invalid")
    if data["association_policy"] != {
        "policy_id": "private_report_exact_identity_two_route_v1",
        "filename_route": "unique_normalized_source_patient_substring",
        "body_fallback_route": "unique_normalized_name_field_substring",
        "fuzzy_matching_allowed": False,
        "manual_mapping_required_when_unresolved": True,
    }:
        raise ValueError("private physician report association policy drifted")
    reports = data["reports"]
    if not isinstance(reports, list) or not reports:
        raise ValueError("private physician report rows must be non-empty")
    normalized_reports: list[dict[str, Any]] = []
    linked_groups: set[str] = set()
    for index, row in enumerate(reports):
        if type(row) is not dict or set(row) != {
            "report_id",
            "document_ref",
            "authorship",
            "container_audit",
            "association",
            "deidentification",
        }:
            raise ValueError(f"reports[{index}] fields drifted")
        ref = validate_artifact_ref(row["document_ref"])
        if (
            ref["artifact_kind"] != "physician_authored_report_source"
            or ref["media_type"] != DOCX_MEDIA_TYPE
            or ref["payload_schema_version"] != PHYSICIAN_REPORT_DOCUMENT_SCHEMA_VERSION
            or ref["content_hash"]["domain"] != "raw_bytes_v1"
        ):
            raise ValueError("physician report document reference type drifted")
        expected_report_id = _REPORT_ID_PREFIX + ref["content_hash"]["sha256"][:24]
        if row["report_id"] != expected_report_id:
            raise ValueError("physician report ID does not bind document bytes")
        if row["authorship"] != "physician_authored":
            raise ValueError("physician report authorship drifted")
        audit = row["container_audit"]
        if type(audit) is not dict or set(audit) != {
            "container",
            "zip_member_count",
            "uncompressed_size_bytes",
            "document_xml_sha256",
            "nonempty_paragraph_count",
            "body_character_count",
            "external_relationship_count",
            "macros_present",
            "parse_status",
        }:
            raise ValueError("physician report container audit fields drifted")
        if audit["container"] != "docx" or audit["parse_status"] != "valid":
            raise ValueError("physician report container is not valid DOCX")
        for key in (
            "zip_member_count",
            "uncompressed_size_bytes",
            "nonempty_paragraph_count",
            "body_character_count",
            "external_relationship_count",
        ):
            item = audit[key]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError("physician report container count is invalid")
        if (
            audit["zip_member_count"] < 2
            or audit["uncompressed_size_bytes"] < 1
            or audit["nonempty_paragraph_count"] < 1
            or audit["body_character_count"] < 1
            or audit["macros_present"] is not False
            or not isinstance(audit["document_xml_sha256"], str)
            or _SHA256_RE.fullmatch(audit["document_xml_sha256"]) is None
        ):
            raise ValueError("physician report container audit is invalid")
        association = row["association"]
        if type(association) is not dict or set(association) != {
            "status",
            "basis",
            "linkage_group_id",
            "source_patient_sha256",
            "split_assignment",
        }:
            raise ValueError("physician report association fields drifted")
        if association["status"] not in _ALLOWED_ASSOCIATION_STATUS or association[
            "basis"
        ] not in _ALLOWED_ASSOCIATION_BASIS:
            raise ValueError("physician report association status/basis drifted")
        if association["status"] == "unresolved":
            if association != {
                "status": "unresolved",
                "basis": "none",
                "linkage_group_id": None,
                "source_patient_sha256": None,
                "split_assignment": None,
            }:
                raise ValueError("unresolved physician report leaked a guessed association")
        else:
            if association["basis"] == "none":
                raise ValueError("linked physician report lacks an exact basis")
            group_id = association["linkage_group_id"]
            patient_sha = association["source_patient_sha256"]
            if (
                not isinstance(group_id, str)
                or not group_id.startswith("EVISOZ-PAT-")
                or not isinstance(patient_sha, str)
                or _SHA256_RE.fullmatch(patient_sha) is None
                or group_id in linked_groups
            ):
                raise ValueError("linked physician report identity is invalid or duplicated")
            linked_groups.add(group_id)
            split = association["split_assignment"]
            if type(split) is not dict or set(split) != {
                "evisoz_role",
                "outer_holdout_fold",
                "locked",
            }:
                raise ValueError("physician report split assignment fields drifted")
            if split["evisoz_role"] == "development_cv":
                if (
                    isinstance(split["outer_holdout_fold"], bool)
                    or not isinstance(split["outer_holdout_fold"], int)
                    or split["outer_holdout_fold"] < 0
                    or split["locked"] is not False
                ):
                    raise ValueError("development physician report split is invalid")
            elif split["evisoz_role"] == "locked_test":
                if split["outer_holdout_fold"] is not None or split["locked"] is not True:
                    raise ValueError("locked physician report split is invalid")
            else:
                raise ValueError("physician report has an unsupported split role")
        if row["deidentification"] != {
            "source_contains_phi": True,
            "deidentified_text_released": False,
            "status": "pending_manual_review",
            "eligible_for_qwen_training": False,
            "eligible_for_language_evaluation": False,
        }:
            raise ValueError("unreviewed physician report text was released")
        normalized = deepcopy(row)
        normalized["document_ref"] = ref
        normalized_reports.append(normalized)
    if normalized_reports != sorted(
        normalized_reports, key=lambda row: row["report_id"]
    ) or len({row["report_id"] for row in normalized_reports}) != len(
        normalized_reports
    ):
        raise ValueError("physician report rows are not uniquely canonically sorted")
    expected_set_sha = canonical_json_sha256(
        [row["document_ref"] for row in normalized_reports]
    )
    if bindings["report_document_set_sha256"] != expected_set_sha:
        raise ValueError("physician report document-set digest drifted")
    status_counts = Counter(
        row["association"]["status"] for row in normalized_reports
    )
    basis_counts = Counter(row["association"]["basis"] for row in normalized_reports)
    role_counts = Counter(
        row["association"]["split_assignment"]["evisoz_role"]
        for row in normalized_reports
        if row["association"]["split_assignment"] is not None
    )
    expected_counts = {
        "report_count": len(normalized_reports),
        "valid_docx_count": len(normalized_reports),
        "association_status_counts": dict(sorted(status_counts.items())),
        "association_basis_counts": dict(sorted(basis_counts.items())),
        "linked_split_role_counts": dict(sorted(role_counts.items())),
        "deidentified_text_release_count": 0,
        "qwen_training_eligible_count": 0,
        "language_evaluation_eligible_count": 0,
    }
    if data["counts"] != expected_counts:
        raise ValueError("private physician report inventory counts drifted")
    expected_pending = sorted(
        row["report_id"]
        for row in normalized_reports
        if row["association"]["status"] == "unresolved"
    )
    if data["manual_mapping_required_report_ids"] != expected_pending:
        raise ValueError("private physician report unresolved roster drifted")
    if data["privacy_and_use_contract"] != {
        "source_paths_or_filenames_persisted": False,
        "patient_names_persisted": False,
        "raw_report_text_persisted": False,
        "raw_report_bytes_copied": False,
        "source_documents_are_physician_authored": True,
        "source_documents_are_generated_text": False,
        "association_uses_phi_in_memory_only": True,
        "unreviewed_text_can_enter_qwen_training": False,
        "locked_test_text_can_enter_training": False,
        "report_text_can_supervise_localization": False,
    }:
        raise ValueError("private physician report privacy/use contract drifted")
    expected_id = _INVENTORY_ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["inventory_id"] != expected_id:
        raise ValueError("private physician report inventory ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("private physician report inventory receipt hash drifted")
    data["reports"] = normalized_reports
    return data


__all__ = [
    "PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION",
    "PHYSICIAN_REPORT_DOCUMENT_SCHEMA_VERSION",
    "build_private_physician_report_inventory",
    "validate_private_physician_report_inventory",
]
