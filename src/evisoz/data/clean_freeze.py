"""Fail-closed clean-snapshot audit for the EviSOZ Stage-0 route.

The audit is intentionally narrower than a training authorization.  It records
whether the repository has a reproducible Git snapshot and whether the frozen
EviSOZ contract files are present and regular.  A successful audit never opens
the Stage-0 gate; it is only one required input to a later controller decision.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Mapping, Sequence

from .artifact_ref import canonical_json_sha256


CLEAN_FREEZE_AUDIT_SCHEMA_VERSION = "evisoz_holdout_freeze_audit_v1"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-FREEZE-"
_SHA256_RE = set("0123456789abcdef")

DEFAULT_CONTRACT_PATHS = (
    "configs/evisoz_schema_registry_v1.json",
    "configs/evisoz_structured_evidence_pipeline_v1.json",
    "schemas/evisoz_artifact_ref_v1.schema.json",
    "schemas/evisoz_training_example_v1.schema.json",
    "knowledge/eeg/manifest.json",
    "knowledge/eeg/reasoning/grounding_rules.json",
    "knowledge/eeg/reasoning/inference_rules.json",
    "knowledge/eeg/reporting/claim_policy.json",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["audit_id"] = "CONTENT-ADDRESS-PENDING"
    return body


def _repository_root(value: str | Path) -> Path:
    root = Path(value)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("clean-freeze repository root must be a regular directory")
    return root.resolve(strict=True)


def _relative_path(value: str | Path) -> str:
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("clean-freeze contract paths must be repository-relative")
    return path.as_posix()


def _git(root: Path, *args: str) -> tuple[int, str, str]:
    process = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return process.returncode, process.stdout, process.stderr


def _git_snapshot(root: Path, *, excluded_status_paths: Sequence[str]) -> dict[str, Any]:
    head_code, head_out, _ = _git(root, "rev-parse", "--verify", "HEAD")
    branch_code, branch_out, _ = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    status_code, status_out, status_err = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    all_rows = [line for line in status_out.splitlines() if line.strip()]
    excluded = set(excluded_status_paths)
    rows = []
    for line in all_rows:
        # Porcelain v1 uses three prefix characters before the path.  A
        # rename has two paths; exclude the row if either endpoint is the
        # explicitly requested audit output path.
        path_text = line[3:] if len(line) >= 3 else ""
        path_parts = [part.strip() for part in path_text.split(" -> ")]
        if any(part in excluded for part in path_parts):
            continue
        rows.append(line)
    status_counts = {"tracked_modified": 0, "untracked": 0, "other": 0}
    for row in rows:
        if row.startswith("??"):
            status_counts["untracked"] += 1
        elif len(row) >= 2 and row[:2].strip():
            status_counts["tracked_modified"] += 1
        else:
            status_counts["other"] += 1
    # Keep file names out of the receipt.  The digest still detects any change
    # while avoiding accidental disclosure of local report/checkpoint names.
    status_rows_sha256 = canonical_json_sha256(sorted(rows))
    return {
        "head_sha": head_out.strip() if head_code == 0 else None,
        "branch": branch_out.strip() if branch_code == 0 else None,
        "git_status_exit_code": int(status_code),
        "git_status_error_present": bool(status_err.strip()),
        "status_counts": status_counts,
        "status_rows_sha256": status_rows_sha256,
        "excluded_status_paths": sorted(excluded),
        "clean": bool(head_code == 0 and status_code == 0 and not rows),
    }


def _contract_rows(root: Path, paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    normalized = sorted({_relative_path(path) for path in paths})
    rows: list[dict[str, Any]] = []
    for relative in normalized:
        candidate = root / relative
        regular = candidate.is_file() and not candidate.is_symlink()
        row: dict[str, Any] = {
            "path": relative,
            "present": bool(regular),
            "regular": bool(regular),
            "sha256": None,
            "size_bytes": None,
        }
        if regular:
            payload = candidate.read_bytes()
            row["sha256"] = _sha256_bytes(payload)
            row["size_bytes"] = len(payload)
        rows.append(row)
    return rows


def build_clean_freeze_audit(
    *,
    repository_root: str | Path,
    contract_paths: Sequence[str | Path] = DEFAULT_CONTRACT_PATHS,
    stage0_gate_path: str | Path | None = None,
    excluded_status_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Build a content-addressed, non-authorizing clean-freeze audit."""

    root = _repository_root(repository_root)
    contracts = _contract_rows(root, contract_paths)
    excluded = sorted({_relative_path(path) for path in excluded_status_paths})
    git_snapshot = _git_snapshot(root, excluded_status_paths=excluded)
    missing_contracts = [row["path"] for row in contracts if not row["regular"]]
    checks = [
        {
            "check_id": "git_snapshot_clean",
            "status": "GO" if git_snapshot["clean"] else "NO_GO",
            "blocker_codes": [] if git_snapshot["clean"] else ["repository_worktree_not_clean"],
            "facts": {
                "head_available": git_snapshot["head_sha"] is not None,
                "status_entry_count": sum(git_snapshot["status_counts"].values()),
            },
        },
        {
            "check_id": "required_contracts_present",
            "status": "GO" if not missing_contracts else "NO_GO",
            "blocker_codes": [] if not missing_contracts else ["required_contract_missing"],
            "facts": {
                "contract_count": len(contracts),
                "missing_count": len(missing_contracts),
                "missing_paths_sha256": canonical_json_sha256(missing_contracts),
            },
        },
    ]
    status = "GO" if all(row["status"] == "GO" for row in checks) else "NO_GO"
    body: dict[str, Any] = {
        "schema_version": CLEAN_FREEZE_AUDIT_SCHEMA_VERSION,
        "audit_id": "CONTENT-ADDRESS-PENDING",
        "status": status,
        "git_snapshot": git_snapshot,
        "required_contracts": contracts,
        "stage0_gate_path_present": bool(stage0_gate_path is not None and (root / str(stage0_gate_path)).is_file()),
        "checks": checks,
        # This audit can never grant training permission.  Stage-0 must be
        # replayed after this receipt and all independent blockers are closed.
        "training_authorized": False,
        "non_authorizing": True,
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["audit_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_clean_freeze_audit(body)


def validate_clean_freeze_audit(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "audit_id", "status", "git_snapshot",
        "required_contracts", "stage0_gate_path_present", "checks",
        "training_authorized", "non_authorizing", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("clean-freeze audit fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != CLEAN_FREEZE_AUDIT_SCHEMA_VERSION:
        raise ValueError("clean-freeze audit schema drifted")
    if data["status"] not in {"GO", "NO_GO"}:
        raise ValueError("clean-freeze audit status is invalid")
    if data["training_authorized"] is not False or data["non_authorizing"] is not True:
        raise ValueError("clean-freeze audit authorization invariant violated")
    snapshot = data["git_snapshot"]
    if type(snapshot) is not dict or set(snapshot) != {
        "head_sha", "branch", "git_status_exit_code", "git_status_error_present",
        "status_counts", "status_rows_sha256", "excluded_status_paths", "clean",
    }:
        raise ValueError("clean-freeze Git snapshot fields drifted")
    if snapshot["head_sha"] is not None and (
        not isinstance(snapshot["head_sha"], str)
        or len(snapshot["head_sha"]) != 40
        or set(snapshot["head_sha"]) - _SHA256_RE
    ):
        raise ValueError("clean-freeze Git head is invalid")
    if snapshot["branch"] is not None and not isinstance(snapshot["branch"], str):
        raise ValueError("clean-freeze Git branch is invalid")
    if not isinstance(snapshot["status_counts"], dict) or set(snapshot["status_counts"]) != {
        "tracked_modified", "untracked", "other"
    } or any(type(v) is not int or v < 0 for v in snapshot["status_counts"].values()):
        raise ValueError("clean-freeze status counts are invalid")
    if (
        not isinstance(snapshot["excluded_status_paths"], list)
        or snapshot["excluded_status_paths"] != sorted(set(snapshot["excluded_status_paths"]))
        or any(not isinstance(path, str) for path in snapshot["excluded_status_paths"])
    ):
        raise ValueError("clean-freeze excluded status paths are invalid")
    contracts = data["required_contracts"]
    if not isinstance(contracts, list) or not contracts:
        raise ValueError("clean-freeze contract roster is empty")
    for row in contracts:
        if type(row) is not dict or set(row) != {"path", "present", "regular", "sha256", "size_bytes"}:
            raise ValueError("clean-freeze contract row fields drifted")
        _relative_path(row["path"])
        if row["present"] != row["regular"]:
            raise ValueError("clean-freeze contract presence drifted")
        if row["regular"]:
            if not isinstance(row["sha256"], str) or len(row["sha256"]) != 64 or set(row["sha256"]) - _SHA256_RE:
                raise ValueError("clean-freeze contract hash is invalid")
            if type(row["size_bytes"]) is not int or row["size_bytes"] < 0:
                raise ValueError("clean-freeze contract size is invalid")
        elif row["sha256"] is not None or row["size_bytes"] is not None:
            raise ValueError("missing clean-freeze contract contains content")
    checks = data["checks"]
    if not isinstance(checks, list) or not checks:
        raise ValueError("clean-freeze checks are empty")
    expected_status = "GO" if all(row.get("status") == "GO" for row in checks) else "NO_GO"
    if data["status"] != expected_status:
        raise ValueError("clean-freeze aggregate status drifted")
    if data["audit_id"] != _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]:
        raise ValueError("clean-freeze audit ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("clean-freeze audit receipt drifted")
    return data


__all__ = [
    "CLEAN_FREEZE_AUDIT_SCHEMA_VERSION",
    "DEFAULT_CONTRACT_PATHS",
    "build_clean_freeze_audit",
    "validate_clean_freeze_audit",
]
