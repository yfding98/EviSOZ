"""Read-only discovery of offline EviSOZ teacher artifacts.

Discovery is deliberately weaker than admission.  A matching filename or
directory is only an inventory candidate; it never creates a teacher cache,
calibration receipt, training permission, or patient fact.  The resulting
receipt is useful when a controlled operator later supplies an audited model
manifest to ``materialize_evisoz_teacher_candidates_v1.py``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.evisoz.data.artifact_ref import canonical_json_sha256


TEACHER_ARTIFACT_DISCOVERY_SCHEMA_VERSION = "evisoz_teacher_artifact_discovery_v1"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-TEACHER-DISC-"
_TEACHER_HINTS = {
    "cerebragloss": ("cerebragloss", "cerebra-gloss", "gloss"),
    # Keep ELM matching conservative: generic ``language``/``eeg`` report
    # artifacts are common in this repository and must not be misclassified.
    "elm": ("elm", "eeglm", "eeg-lm"),
}
_MODEL_SUFFIXES = {
    ".bin", ".ckpt", ".h5", ".json", ".pkl", ".pt", ".pth", ".safetensors",
    ".yaml", ".yml",
}
_SKIP_DIR_NAMES = {
    ".git", ".cache", "__pycache__", "node_modules", "edf", "reports", "outputs",
}


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["discovery_id"] = "CONTENT-ADDRESS-PENDING"
    return body


def _safe_root(value: str | Path) -> Path:
    root = Path(value)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("teacher discovery root must be a regular directory")
    return root.resolve(strict=True)


def _path_has_hint(path: Path, root: Path, teacher_id: str) -> bool:
    hints = _TEACHER_HINTS[teacher_id]
    # Ignore temporary-parent names (which may contain a test name such as
    # ``missing_elm``).  Only the scan root and its descendants are evidence.
    relative_parts = (root.name,) + path.relative_to(root).parts
    tokens = [part.casefold().replace("_", "-") for part in relative_parts]
    if teacher_id == "elm":
        return any(
            token == "elm"
            or token.startswith("elm-")
            or token.startswith("eeglm")
            or token.startswith("eeg-lm")
            for token in tokens
        )
    return any(any(hint in token for hint in hints) for token in tokens)


def _iter_candidate_files(root: Path, teacher_id: str, max_depth: int) -> Iterable[Path]:
    """Yield model/manifest-looking files without opening arbitrary payloads."""

    root_depth = len(root.parts)
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirnames[:] = sorted(
            name for name in dirnames
            if name.casefold() not in _SKIP_DIR_NAMES
            and not (current_path / name).is_symlink()
        )
        if len(current_path.parts) - root_depth > max_depth:
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.casefold() not in _MODEL_SUFFIXES:
                continue
            # A generic checkpoint name is admissible only when its path is
            # explicitly teacher-hinted.  This avoids treating arbitrary EEG
            # checkpoints in a shared dataset root as ELM/CerebraGloss.
            if _path_has_hint(path, root, teacher_id):
                yield path


def _candidate_row(path: Path, root: Path, *, hash_files: bool) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    size = path.stat().st_size
    digest = None
    hash_status = "not_hashed"
    if hash_files:
        digest_obj = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest_obj.update(block)
        digest = digest_obj.hexdigest()
        hash_status = "sha256_full"
    return {
        "root": str(root),
        "relative_path": relative,
        "suffix": path.suffix.casefold(),
        "size_bytes": int(size),
        "sha256": digest,
        "hash_status": hash_status,
    }


def build_teacher_artifact_discovery(
    *,
    teacher_id: str,
    roots: Sequence[str | Path],
    max_depth: int = 6,
    max_candidates: int = 128,
    hash_files: bool = False,
) -> dict[str, Any]:
    """Inventory possible teacher files while keeping admission fail-closed."""

    if teacher_id not in _TEACHER_HINTS:
        raise ValueError("unknown teacher ID")
    if not roots:
        raise ValueError("teacher discovery requires at least one root")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    safe_roots: list[Path] = []
    for value in roots:
        root = _safe_root(value)
        if root not in safe_roots:
            safe_roots.append(root)
    root_rows = [
        {
            "root": str(root),
            "exists": True,
            "regular_directory": True,
        }
        for root in sorted(safe_roots, key=str)
    ]
    candidates: list[dict[str, Any]] = []
    truncated = False
    for root in sorted(safe_roots, key=str):
        for path in _iter_candidate_files(root, teacher_id, max_depth):
            candidates.append(_candidate_row(path, root, hash_files=hash_files))
            if len(candidates) >= max_candidates:
                truncated = True
                break
        if truncated:
            break
    candidates.sort(key=lambda row: (row["root"], row["relative_path"]))
    status = "found_unvalidated" if candidates else "missing"
    missing = (
        ["teacher_artifact_missing"]
        if not candidates
        else ["teacher_artifact_provenance_and_manifest_missing"]
    )
    body: dict[str, Any] = {
        "schema_version": TEACHER_ARTIFACT_DISCOVERY_SCHEMA_VERSION,
        "discovery_id": "CONTENT-ADDRESS-PENDING",
        "teacher_id": teacher_id,
        "status": status,
        "scan": {
            "roots": root_rows,
            "max_depth": max_depth,
            "candidate_limit": max_candidates,
            "truncated": truncated,
            "hash_files": bool(hash_files),
        },
        "candidates": candidates,
        "counts": {"root_count": len(root_rows), "candidate_count": len(candidates)},
        "permissions": {
            "training_authorized": False,
            "calibration_authorized": False,
            "teacher_cache_materialization_authorized": False,
            "patient_fact_creation_allowed": False,
        },
        "missing_closure_codes": missing,
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["discovery_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_teacher_artifact_discovery(body)


def validate_teacher_artifact_discovery(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "discovery_id", "teacher_id", "status", "scan",
        "candidates", "counts", "permissions", "missing_closure_codes",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("teacher discovery fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != TEACHER_ARTIFACT_DISCOVERY_SCHEMA_VERSION:
        raise ValueError("teacher discovery schema drifted")
    if data["teacher_id"] not in _TEACHER_HINTS:
        raise ValueError("teacher discovery teacher ID drifted")
    if data["status"] not in {"missing", "found_unvalidated"}:
        raise ValueError("teacher discovery status is invalid")
    scan = data["scan"]
    if type(scan) is not dict or set(scan) != {
        "roots", "max_depth", "candidate_limit", "truncated", "hash_files"
    }:
        raise ValueError("teacher discovery scan fields drifted")
    roots = scan["roots"]
    if not isinstance(roots, list) or not roots:
        raise ValueError("teacher discovery roots are empty")
    root_names: list[str] = []
    for row in roots:
        if type(row) is not dict or set(row) != {"root", "exists", "regular_directory"}:
            raise ValueError("teacher discovery root row drifted")
        if not isinstance(row["root"], str) or not row["root"]:
            raise ValueError("teacher discovery root is invalid")
        if row["exists"] is not True or row["regular_directory"] is not True:
            raise ValueError("teacher discovery root is not available")
        root_names.append(row["root"])
    if root_names != sorted(set(root_names)):
        raise ValueError("teacher discovery roots are not sorted/unique")
    if (
        isinstance(scan["max_depth"], bool)
        or not isinstance(scan["max_depth"], int)
        or scan["max_depth"] < 0
        or isinstance(scan["candidate_limit"], bool)
        or not isinstance(scan["candidate_limit"], int)
        or scan["candidate_limit"] <= 0
        or not isinstance(scan["truncated"], bool)
        or not isinstance(scan["hash_files"], bool)
    ):
        raise ValueError("teacher discovery scan limits are invalid")
    candidates = data["candidates"]
    if not isinstance(candidates, list) or candidates != sorted(
        candidates, key=lambda row: (row["root"], row["relative_path"])
    ):
        raise ValueError("teacher discovery candidates are not sorted")
    for row in candidates:
        if type(row) is not dict or set(row) != {
            "root", "relative_path", "suffix", "size_bytes", "sha256", "hash_status"
        }:
            raise ValueError("teacher discovery candidate fields drifted")
        if row["root"] not in root_names:
            raise ValueError("teacher discovery candidate root is unknown")
        if (
            not isinstance(row["relative_path"], str)
            or not row["relative_path"]
            or row["relative_path"].startswith("/")
            or ".." in row["relative_path"].split("/")
            or row["suffix"] not in _MODEL_SUFFIXES
            or isinstance(row["size_bytes"], bool)
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 0
        ):
            raise ValueError("teacher discovery candidate identity is invalid")
        if row["hash_status"] == "not_hashed":
            if row["sha256"] is not None:
                raise ValueError("unhashed teacher candidate contains a digest")
        elif row["hash_status"] == "sha256_full":
            if (
                not isinstance(row["sha256"], str)
                or len(row["sha256"]) != 64
                or set(row["sha256"]) - set("0123456789abcdef")
            ):
                raise ValueError("teacher candidate digest is invalid")
        else:
            raise ValueError("teacher candidate hash status is invalid")
    counts = data["counts"]
    if counts != {"root_count": len(roots), "candidate_count": len(candidates)}:
        raise ValueError("teacher discovery counts drifted")
    expected_status = "found_unvalidated" if candidates else "missing"
    if data["status"] != expected_status:
        raise ValueError("teacher discovery status/count mismatch")
    expected_missing = (
        ["teacher_artifact_missing"]
        if not candidates
        else ["teacher_artifact_provenance_and_manifest_missing"]
    )
    if data["missing_closure_codes"] != expected_missing:
        raise ValueError("teacher discovery closure codes drifted")
    if data["permissions"] != {
        "training_authorized": False,
        "calibration_authorized": False,
        "teacher_cache_materialization_authorized": False,
        "patient_fact_creation_allowed": False,
    }:
        raise ValueError("teacher discovery permissions drifted")
    if not isinstance(data["discovery_id"], str) or not data["discovery_id"].startswith(_ID_PREFIX):
        raise ValueError("teacher discovery ID is invalid")
    if data["discovery_id"] != _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]:
        raise ValueError("teacher discovery ID does not bind content")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("teacher discovery receipt drifted")
    return data


__all__ = [
    "TEACHER_ARTIFACT_DISCOVERY_SCHEMA_VERSION",
    "build_teacher_artifact_discovery",
    "validate_teacher_artifact_discovery",
]
