"""Aggregate audit for the five real TUSZ selection-fit phase receipts.

The inventory is evidence, not a transferable training authority.  Formal
provider code must still replay each reference sidecar and receive the opaque
process-local authority issued by ``authorize_detector_fold_reference_phase_receipt_v1``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .detector_fold_reference_authority_v1 import (
    authorize_detector_fold_reference_phase_receipt_v1,
    validate_detector_fold_reference_authority_registry_v1,
    validate_detector_fold_reference_phase_v1,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "clinical_eeg_detector_selection_fit_phase_inventory_v1"
INVENTORY_ID = "CLINICAL-EEG-DETECTOR-SELECTION-FIT-PHASE-INVENTORY-V1-20260824"
_PENDING = "CONTENT-ADDRESS-PENDING"
_SHA_CHARS = frozenset("0123456789abcdef")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, context: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA_CHARS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _safe_project_file(relative_path: object, context: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise TypeError(f"{context} path must be a non-empty string")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} path escapes the project root")
    resolved = (ROOT / relative).resolve(strict=True)
    try:
        resolved.relative_to(ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{context} path escapes the project root") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return resolved


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not readable JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must contain an object")
    return value


def build_detector_selection_fit_phase_inventory_v1(
    *,
    phase_receipt_paths: Sequence[str | Path],
    fold_plan_path: str | Path,
    registry_path: str | Path,
    replay_reference_root: str | Path,
) -> dict[str, Any]:
    """Actual-byte reauthorize all five receipts and build a compact inventory."""

    plan_file = Path(fold_plan_path).resolve(strict=True)
    registry_file = Path(registry_path).resolve(strict=True)
    reference_root = Path(replay_reference_root).resolve(strict=True)
    plan = _read_json(plan_file, "fold plan")
    registry = validate_detector_fold_reference_authority_registry_v1(
        _read_json(registry_file, "fold authority registry"),
        fold_plan=plan,
        verify_bound_files=True,
    )
    if len(phase_receipt_paths) != 5:
        raise ValueError("selection-fit inventory requires exactly five receipts")

    rows: list[dict[str, Any]] = []
    seen_folds: set[int] = set()
    for supplied_path in phase_receipt_paths:
        receipt_file = Path(supplied_path).resolve(strict=True)
        try:
            relative = receipt_file.relative_to(ROOT.resolve(strict=True))
        except ValueError as error:
            raise ValueError("phase receipt must be beneath the project root") from error
        if not receipt_file.is_file() or receipt_file.is_symlink():
            raise ValueError("phase receipt must be a regular non-symlink file")
        serialized = _read_json(receipt_file, "selection-fit phase receipt")
        authority = authorize_detector_fold_reference_phase_receipt_v1(
            serialized,
            fold_plan=plan,
            registry=registry,
            replay_reference_root=reference_root,
        )
        replayed = authority.to_receipt()
        if replayed != serialized or replayed.get("phase") != "selection_fit":
            raise PermissionError("selection-fit receipt did not exact-byte reauthorize")
        fold = replayed.get("outer_fold_id")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold in seen_folds:
            raise ValueError("selection-fit outer fold is invalid or duplicated")
        seen_folds.add(fold)
        roster = replayed["authorized_roster"]
        opened = replayed["reference_open_log"]
        forbidden = {
            key: opened[key]
            for key in (
                "outer_heldout_reference_files_opened",
                "source_dev_reference_files_opened",
                "source_eval_reference_files_opened",
                "private_reference_files_opened",
            )
        }
        if any(value != 0 for value in forbidden.values()):
            raise PermissionError("selection-fit receipt opened a forbidden reference scope")
        rows.append(
            {
                "outer_fold_id": fold,
                "phase": "selection_fit",
                "authorized_fold_ids": list(replayed["authorized_fold_ids"]),
                "patient_count": roster["patient_count"],
                "recording_count": roster["recording_count"],
                "duration_seconds_fraction": list(roster["duration_seconds_fraction"]),
                "analysis_identity_roster_sha256": roster[
                    "analysis_identity_roster_sha256"
                ],
                "receipt_path": relative.as_posix(),
                "file_size_bytes": receipt_file.stat().st_size,
                "file_sha256": _file_sha256(receipt_file),
                "phase_receipt_sha256": replayed["receipt_sha256"],
                "reference_files_opened": opened["reference_files_opened"],
                "reference_bytes_read": opened["reference_bytes_read"],
                **forbidden,
            }
        )
    if seen_folds != set(range(5)):
        raise ValueError("selection-fit inventory does not cover outer folds 0..4")
    rows.sort(key=lambda row: row["outer_fold_id"])

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inventory_id": INVENTORY_ID,
        "status": "five_real_selection_fit_receipts_actual_byte_reauthorized",
        "fold_reference_registry_binding": {
            "path": registry_file.relative_to(ROOT.resolve(strict=True)).as_posix(),
            "file_sha256": _file_sha256(registry_file),
            "registry_receipt_sha256": registry["registry_receipt_sha256"],
        },
        "fold_plan_binding": {
            "path": plan_file.relative_to(ROOT.resolve(strict=True)).as_posix(),
            "file_sha256": _file_sha256(plan_file),
            "plan_receipt_sha256": plan["receipt_sha256"],
        },
        "selection_fit_phase_receipts": rows,
        "aggregate": {
            "outer_fold_count": 5,
            "phase_receipt_count": len(rows),
            "reference_open_operations": sum(
                row["reference_files_opened"] for row in rows
            ),
            "reference_bytes_read": sum(row["reference_bytes_read"] for row in rows),
            "all_serialized_receipts_exact_byte_reauthorized": True,
            "all_forbidden_reference_open_counts_zero": True,
        },
        "source_firewall": {
            "source_train_global_TERM_seiz_reference_used": True,
            "outer_heldout_reference_used": False,
            "source_dev_or_eval_reference_used": False,
            "private_reference_used": False,
            "EDF_annotation_used": False,
            "Excel_doctor_report_or_clinical_text_used": False,
        },
        "scientific_claim_boundary": {
            "serialized_inventory_is_transferable_training_authority": False,
            "process_local_reauthorization_still_required": True,
            "inner_validation_phase_receipt_count": 0,
            "final_refit_phase_receipt_count": 0,
            "provider_checkpoint_count": 0,
            "OOF_prediction_count": 0,
            "detector_performance_claim_authorized": False,
            "clinical_or_production_use_authorized": False,
        },
        "receipt_sha256": _PENDING,
    }
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def validate_detector_selection_fit_phase_inventory_v1(
    value: Mapping[str, Any], *, verify_bound_files: bool = True
) -> dict[str, Any]:
    """Validate inventory semantics and optionally replay every bound file byte."""

    if type(value) is not dict:
        raise TypeError("selection-fit inventory must be an object")
    row = deepcopy(dict(value))
    supplied = _require_sha256(row.get("receipt_sha256"), "inventory receipt")
    row["receipt_sha256"] = _PENDING
    if supplied != _canonical_sha256(row):
        raise ValueError("selection-fit inventory is not content-addressed")
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("inventory_id") != INVENTORY_ID
        or row.get("status")
        != "five_real_selection_fit_receipts_actual_byte_reauthorized"
    ):
        raise ValueError("selection-fit inventory identity/status drifted")
    receipts = row.get("selection_fit_phase_receipts")
    if not isinstance(receipts, list) or len(receipts) != 5:
        raise ValueError("selection-fit inventory must contain five fold receipts")
    if [item.get("outer_fold_id") for item in receipts] != list(range(5)):
        raise ValueError("selection-fit fold Cartesian drifted")
    aggregate = row.get("aggregate")
    if type(aggregate) is not dict or aggregate != {
        "outer_fold_count": 5,
        "phase_receipt_count": 5,
        "reference_open_operations": sum(
            item["reference_files_opened"] for item in receipts
        ),
        "reference_bytes_read": sum(item["reference_bytes_read"] for item in receipts),
        "all_serialized_receipts_exact_byte_reauthorized": True,
        "all_forbidden_reference_open_counts_zero": True,
    }:
        raise ValueError("selection-fit aggregate drifted")
    for item in receipts:
        if item.get("phase") != "selection_fit":
            raise ValueError("selection-fit inventory contains another phase")
        for key in (
            "outer_heldout_reference_files_opened",
            "source_dev_reference_files_opened",
            "source_eval_reference_files_opened",
            "private_reference_files_opened",
        ):
            if item.get(key) != 0:
                raise PermissionError("selection-fit inventory opened forbidden reference")
        _require_sha256(item.get("file_sha256"), "phase receipt file")
        _require_sha256(item.get("phase_receipt_sha256"), "phase receipt semantic")
        _require_sha256(
            item.get("analysis_identity_roster_sha256"), "phase identity roster"
        )

    firewall = row.get("source_firewall")
    if firewall != {
        "source_train_global_TERM_seiz_reference_used": True,
        "outer_heldout_reference_used": False,
        "source_dev_or_eval_reference_used": False,
        "private_reference_used": False,
        "EDF_annotation_used": False,
        "Excel_doctor_report_or_clinical_text_used": False,
    }:
        raise PermissionError("selection-fit source firewall drifted")
    boundary = row.get("scientific_claim_boundary")
    if boundary != {
        "serialized_inventory_is_transferable_training_authority": False,
        "process_local_reauthorization_still_required": True,
        "inner_validation_phase_receipt_count": 0,
        "final_refit_phase_receipt_count": 0,
        "provider_checkpoint_count": 0,
        "OOF_prediction_count": 0,
        "detector_performance_claim_authorized": False,
        "clinical_or_production_use_authorized": False,
    }:
        raise PermissionError("selection-fit scientific boundary drifted")

    if verify_bound_files:
        plan_binding = row["fold_plan_binding"]
        registry_binding = row["fold_reference_registry_binding"]
        plan_path = _safe_project_file(plan_binding["path"], "fold plan")
        registry_path = _safe_project_file(
            registry_binding["path"], "fold authority registry"
        )
        if _file_sha256(plan_path) != plan_binding["file_sha256"]:
            raise ValueError("selection-fit fold plan bytes drifted")
        if _file_sha256(registry_path) != registry_binding["file_sha256"]:
            raise ValueError("selection-fit registry bytes drifted")
        plan = _read_json(plan_path, "fold plan")
        registry = validate_detector_fold_reference_authority_registry_v1(
            _read_json(registry_path, "fold authority registry"),
            fold_plan=plan,
            verify_bound_files=True,
        )
        if (
            plan.get("receipt_sha256") != plan_binding["plan_receipt_sha256"]
            or registry.get("registry_receipt_sha256")
            != registry_binding["registry_receipt_sha256"]
        ):
            raise ValueError("selection-fit plan/registry semantic receipt drifted")
        for item in receipts:
            path = _safe_project_file(item["receipt_path"], "phase receipt")
            if (
                path.stat().st_size != item["file_size_bytes"]
                or _file_sha256(path) != item["file_sha256"]
            ):
                raise ValueError("selection-fit phase receipt bytes drifted")
            serialized = _read_json(path, "phase receipt")
            validated = validate_detector_fold_reference_phase_v1(
                serialized, fold_plan=plan, registry=registry
            )
            if (
                validated["receipt_sha256"] != item["phase_receipt_sha256"]
                or validated["outer_fold_id"] != item["outer_fold_id"]
                or validated["authorized_roster"]["recording_count"]
                != item["recording_count"]
            ):
                raise ValueError("selection-fit phase receipt semantic drifted")
    return deepcopy(value)


__all__ = [
    "INVENTORY_ID",
    "SCHEMA_VERSION",
    "build_detector_selection_fit_phase_inventory_v1",
    "validate_detector_selection_fit_phase_inventory_v1",
]
