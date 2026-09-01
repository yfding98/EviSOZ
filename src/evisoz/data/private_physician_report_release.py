"""Independent release contract for de-identified physician-authored reports.

The inventory and de-identification candidate bundle intentionally remain
immutable and non-trainable.  This module adds the missing promotion boundary:
an independently issued authorization plus a manual-review receipt must
explicitly release a candidate for either development Qwen text training or
locked language evaluation.  The release contains references to text bytes,
never the source DOCX or raw patient identifiers, and can never supervise SOZ
localization.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence

from .artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)
from ..forge.private_report_deidentification import (
    DEIDENTIFIED_REPORT_TEXT_SCHEMA_VERSION,
    PRIVATE_REPORT_DEID_CANDIDATES_SCHEMA_VERSION,
    validate_private_report_deidentification_candidates,
)


PHYSICIAN_REPORT_RELEASE_SCHEMA_VERSION = (
    "evisoz_private_physician_report_release_v1"
)
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_ID_PREFIX = "EVISOZ-PRREL-"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPORT_ID_RE = re.compile(r"^EVISOZ-PRPT-[0-9a-f]{24}$")
_CANDIDATE_ID_RE = re.compile(r"^EVISOZ-DEID-[0-9a-f]{24}$")
_GROUP_ID_RE = re.compile(r"^EVISOZ-PAT-[A-Za-z0-9._:-]+$")

_PERMISSIONS = {
    "physician_authored_text_released": True,
    "qwen_text_training_allowed": True,
    "locked_language_evaluation_allowed": True,
    "report_text_can_supervise_localization": False,
    "generated_text_is_not_physician_authored": True,
    "raw_patient_identifiers_stored": False,
}


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["release_id"] = _PENDING_ID
    return body


def _safe_candidate_text(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("released physician text path must be a string")
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or not relative.startswith("candidates/")
    ):
        raise ValueError("released physician text path is unsafe")
    candidate = root.joinpath(*parsed.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("released physician text must be a regular file")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    return resolved


def _validate_authorization(value: object) -> dict[str, Any]:
    required = {
        "authorization_ref",
        "patient_roster_sha256",
        "allowed_evisoz_roles",
        "approved_purposes",
        "manual_review_required",
        "report_text_release_authorized",
        "training_authorized",
        "raw_patient_identifiers_stored",
        "report_text_can_supervise_localization",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("physician report release authorization fields drifted")
    data = deepcopy(value)
    ref = validate_artifact_ref(data["authorization_ref"])
    if (
        ref["artifact_kind"] != "private_report_release_authorization"
        or ref["payload_schema_version"]
        != "evisoz_private_report_release_authorization_v1"
        or ref["content_hash"]["domain"] != "raw_bytes_v1"
    ):
        raise ValueError("physician report release authorization ref drifted")
    _sha256(data["patient_roster_sha256"], "authorization.patient_roster_sha256")
    roles = data["allowed_evisoz_roles"]
    if (
        not isinstance(roles, list)
        or not roles
        or any(role not in {"development_cv", "locked_test"} for role in roles)
        or roles != sorted(set(roles))
    ):
        raise ValueError("authorization allowed_evisoz_roles must be sorted/unique")
    purposes = data["approved_purposes"]
    if (
        not isinstance(purposes, list)
        or not purposes
        or any(purpose not in {"qwen_text_training", "language_evaluation"} for purpose in purposes)
        or purposes != sorted(set(purposes))
    ):
        raise ValueError("authorization approved_purposes must be sorted/unique")
    if data["manual_review_required"] is not True:
        raise ValueError("manual review is mandatory for physician report release")
    if data["report_text_release_authorized"] is not True:
        raise ValueError("physician report text release is not authorized")
    if type(data["training_authorized"]) is not bool:
        raise TypeError("authorization.training_authorized must be boolean")
    if data["training_authorized"] and not (
        "development_cv" in roles and "qwen_text_training" in purposes
    ):
        raise ValueError("training authorization must include development Qwen purpose")
    if data["raw_patient_identifiers_stored"] is not False:
        raise ValueError("physician report release cannot store raw patient identifiers")
    if data["report_text_can_supervise_localization"] is not False:
        raise ValueError("physician report text cannot supervise SOZ localization")
    return data


def _validate_manual_review(value: object) -> dict[str, Any]:
    required = {
        "status",
        "phi_scan_replayed",
        "clinical_scope_confirmed",
        "release_decision",
        "review_receipt_ref",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("physician report manual review fields drifted")
    data = deepcopy(value)
    if data["status"] != "passed":
        raise ValueError("physician report manual review has not passed")
    if data["phi_scan_replayed"] is not True:
        raise ValueError("physician report PHI scan was not replayed")
    if data["clinical_scope_confirmed"] is not True:
        raise ValueError("physician report clinical scope was not confirmed")
    if data["release_decision"] != "approved":
        raise ValueError("physician report manual review decision is not approved")
    ref = validate_artifact_ref(data["review_receipt_ref"])
    if (
        ref["artifact_kind"] != "private_report_manual_review_receipt"
        or ref["payload_schema_version"]
        != "evisoz_private_report_manual_review_receipt_v1"
        or ref["content_hash"]["domain"] != "raw_bytes_v1"
    ):
        raise ValueError("physician report manual review receipt ref drifted")
    return data


def _validate_row(
    value: object,
    *,
    candidate_by_id: Mapping[str, Mapping[str, object]],
    candidate_output_root: Path,
    authorization: Mapping[str, object],
) -> dict[str, Any]:
    required = {
        "candidate_id",
        "report_id",
        "linkage_group_id",
        "evisoz_role",
        "outer_holdout_fold",
        "purpose",
        "text_ref",
        "relative_text_path",
        "manual_review",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("physician report release row fields drifted")
    row = deepcopy(value)
    candidate_id = row["candidate_id"]
    report_id = row["report_id"]
    group_id = row["linkage_group_id"]
    if not isinstance(candidate_id, str) or _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        raise ValueError("physician report release candidate ID drifted")
    if not isinstance(report_id, str) or _REPORT_ID_RE.fullmatch(report_id) is None:
        raise ValueError("physician report release report ID drifted")
    if not isinstance(group_id, str) or _GROUP_ID_RE.fullmatch(group_id) is None:
        raise ValueError("physician report release linkage group drifted")
    candidate = candidate_by_id.get(candidate_id)
    if candidate is None:
        raise ValueError("physician report release references an unknown candidate")
    if candidate["report_id"] != report_id:
        raise ValueError("physician report release report/candidate binding drifted")
    association = candidate["association"]
    if association["status"] != "linked_high_confidence":
        raise ValueError("unresolved or low-confidence report cannot be released")
    if association["linkage_group_id"] != group_id:
        raise ValueError("physician report release patient binding drifted")
    split = association["split_assignment"]
    if row["evisoz_role"] != split["evisoz_role"]:
        raise ValueError("physician report release split role drifted")
    if row["outer_holdout_fold"] != split["outer_holdout_fold"]:
        raise ValueError("physician report release fold drifted")
    role = row["evisoz_role"]
    purpose = row["purpose"]
    if role not in authorization["allowed_evisoz_roles"]:
        raise ValueError("physician report release role is outside authorization")
    if purpose not in authorization["approved_purposes"]:
        raise ValueError("physician report release purpose is outside authorization")
    if purpose == "qwen_text_training" and role != "development_cv":
        raise ValueError("locked-test physician text cannot enter Qwen training")
    if purpose == "language_evaluation" and role != "locked_test":
        raise ValueError("development physician text cannot be released as locked evaluation")
    text_ref = validate_artifact_ref(row["text_ref"])
    if (
        text_ref["artifact_kind"] != "deidentified_physician_report_candidate"
        or text_ref["payload_schema_version"] != DEIDENTIFIED_REPORT_TEXT_SCHEMA_VERSION
        or text_ref["content_hash"]["domain"] != "raw_bytes_v1"
    ):
        raise ValueError("physician report release text reference drifted")
    if text_ref != candidate["text_ref"]:
        raise ValueError("released text differs from the de-identification candidate")
    relative = row["relative_text_path"]
    if relative != candidate["relative_text_path"]:
        raise ValueError("released text path differs from the candidate path")
    text_path = _safe_candidate_text(candidate_output_root, relative)
    verify_artifact_content(text_ref, text_path.read_bytes())
    manual_review = _validate_manual_review(row["manual_review"])
    return {
        "candidate_id": candidate_id,
        "report_id": report_id,
        "linkage_group_id": group_id,
        "evisoz_role": role,
        "outer_holdout_fold": row["outer_holdout_fold"],
        "purpose": purpose,
        "text_ref": text_ref,
        "relative_text_path": relative,
        "manual_review": manual_review,
    }


def build_private_physician_report_release(
    *,
    candidate_bundle: Mapping[str, object],
    candidate_output_root: Path,
    authorization: Mapping[str, object],
    reviewed_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Build a released report manifest from explicit external approvals.

    The function cannot promote a pending candidate by itself: every row must
    carry a manual-review receipt and the authorization must be an external,
    raw-byte artifact reference.  No report text is copied into the result.
    """

    root = Path(candidate_output_root).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("physician report candidate root must be a regular directory")
    candidates = validate_private_report_deidentification_candidates(
        candidate_bundle,
        output_root=root,
    )
    auth = _validate_authorization(authorization)
    if isinstance(reviewed_rows, (str, bytes)) or not isinstance(reviewed_rows, Sequence):
        raise TypeError("reviewed_rows must be an array")
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates["candidates"]}
    rows = [
        _validate_row(
            row,
            candidate_by_id=candidate_by_id,
            candidate_output_root=root,
            authorization=auth,
        )
        for row in reviewed_rows
    ]
    if not rows:
        raise ValueError("physician report release must contain at least one reviewed row")
    if rows != sorted(rows, key=lambda item: str(item["candidate_id"])):
        raise ValueError("physician report release rows must be sorted")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("physician report release candidates must be unique")
    purposes = Counter(str(row["purpose"]) for row in rows)
    if purposes.get("qwen_text_training", 0) and auth["training_authorized"] is not True:
        raise ValueError("Qwen text-training rows require training authorization")
    body: dict[str, Any] = {
        "schema_version": PHYSICIAN_REPORT_RELEASE_SCHEMA_VERSION,
        "release_id": _PENDING_ID,
        "candidate_bundle_ref": build_json_artifact_ref(
            candidates,
            artifact_kind="physician_report_deidentification_candidates",
            payload_schema_version=PRIVATE_REPORT_DEID_CANDIDATES_SCHEMA_VERSION,
        ),
        "authorization": auth,
        "rows": rows,
        "counts": {
            "released_row_count": len(rows),
            "development_qwen_training_count": sum(
                row["purpose"] == "qwen_text_training" for row in rows
            ),
            "development_language_evaluation_count": sum(
                row["purpose"] == "language_evaluation" and row["evisoz_role"] == "development_cv"
                for row in rows
            ),
            "locked_language_evaluation_count": sum(
                row["purpose"] == "language_evaluation" and row["evisoz_role"] == "locked_test"
                for row in rows
            ),
        },
        "permissions": deepcopy(_PERMISSIONS),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["release_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_private_physician_report_release(
        body,
        candidate_bundle=candidates,
        candidate_output_root=root,
    )


def validate_private_physician_report_release(
    value: object,
    *,
    candidate_bundle: Mapping[str, object],
    candidate_output_root: Path,
) -> dict[str, Any]:
    """Replay a release against the immutable candidate bundle and text bytes."""

    required = {
        "schema_version",
        "release_id",
        "candidate_bundle_ref",
        "authorization",
        "rows",
        "counts",
        "permissions",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("physician report release fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PHYSICIAN_REPORT_RELEASE_SCHEMA_VERSION:
        raise ValueError("physician report release schema_version drifted")
    root = Path(candidate_output_root).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("physician report candidate root must be a regular directory")
    candidates = validate_private_report_deidentification_candidates(
        candidate_bundle,
        output_root=root,
    )
    candidate_ref = validate_artifact_ref(data["candidate_bundle_ref"])
    expected_candidate_ref = build_json_artifact_ref(
        candidates,
        artifact_kind="physician_report_deidentification_candidates",
        payload_schema_version=PRIVATE_REPORT_DEID_CANDIDATES_SCHEMA_VERSION,
    )
    if candidate_ref != expected_candidate_ref:
        raise ValueError("physician report release candidate bundle reference drifted")
    auth = _validate_authorization(data["authorization"])
    rows = data["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("physician report release rows must be non-empty")
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates["candidates"]}
    normalized_rows = [
        _validate_row(
            row,
            candidate_by_id=candidate_by_id,
            candidate_output_root=root,
            authorization=auth,
        )
        for row in rows
    ]
    if normalized_rows != sorted(normalized_rows, key=lambda item: str(item["candidate_id"])):
        raise ValueError("physician report release rows must be sorted")
    if len({row["candidate_id"] for row in normalized_rows}) != len(normalized_rows):
        raise ValueError("physician report release rows must be unique")
    purposes = Counter(str(row["purpose"]) for row in normalized_rows)
    if purposes.get("qwen_text_training", 0) and auth["training_authorized"] is not True:
        raise ValueError("Qwen text-training rows require training authorization")
    expected_counts = {
        "released_row_count": len(normalized_rows),
        "development_qwen_training_count": sum(
            row["purpose"] == "qwen_text_training" for row in normalized_rows
        ),
        "development_language_evaluation_count": sum(
            row["purpose"] == "language_evaluation" and row["evisoz_role"] == "development_cv"
            for row in normalized_rows
        ),
        "locked_language_evaluation_count": sum(
            row["purpose"] == "language_evaluation" and row["evisoz_role"] == "locked_test"
            for row in normalized_rows
        ),
    }
    if data["counts"] != expected_counts:
        raise ValueError("physician report release counts drifted")
    if data["permissions"] != _PERMISSIONS:
        raise ValueError("physician report release permissions drifted")
    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["release_id"] != expected_id:
        raise ValueError("physician report release ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("physician report release receipt drifted")
    data["candidate_bundle_ref"] = candidate_ref
    data["authorization"] = auth
    data["rows"] = normalized_rows
    return data


def materialize_private_physician_report_release(
    *,
    candidate_bundle: Mapping[str, object],
    candidate_output_root: Path,
    authorization: Mapping[str, object],
    reviewed_rows: Sequence[Mapping[str, object]],
    output: Path,
) -> dict[str, Any]:
    """Build and publish one release manifest without copying report text."""

    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    release = build_private_physician_report_release(
        candidate_bundle=candidate_bundle,
        candidate_output_root=candidate_output_root,
        authorization=authorization,
        reviewed_rows=reviewed_rows,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temp:
        staging = Path(temp) / destination.name
        staging.mkdir()
        (staging / "release.json").write_text(
            json.dumps(release, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    return validate_private_physician_report_release(
        release,
        candidate_bundle=candidate_bundle,
        candidate_output_root=candidate_output_root,
    )


__all__ = [
    "PHYSICIAN_REPORT_RELEASE_SCHEMA_VERSION",
    "build_private_physician_report_release",
    "validate_private_physician_report_release",
    "materialize_private_physician_report_release",
]
