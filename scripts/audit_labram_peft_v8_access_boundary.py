#!/usr/bin/env python3
"""Audit the metadata-only access boundary for the frozen LaBraM PEFT v8 run.

The audit reads only three already-published metadata artifacts and filesystem
metadata for the planned source-train EDF paths.  It never opens an EDF,
evidence tensor, SOZ target artifact, or private-data file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)


AUDIT_SCHEMA = "soz_labram_peft_v8_access_boundary_audit_v1"
ARTIFACT_SCHEMA = "soz_labram_peft_v8_access_boundary_artifact_v1"
JSON_SCHEMA_ID = "soz_labram_peft_v8_access_boundary_audit_v1.schema.json"

DEFAULT_SIGNAL_BUNDLE = (
    ROOT / "outputs/deepsoz_signal_preflight_v2_20260809_current"
)
DEFAULT_CAPABILITY_BUNDLE = (
    ROOT / "outputs/labram_iv_source_train_only_capability_v1_20260811"
)
DEFAULT_CROSSWALK_BUNDLE = (
    ROOT / "outputs/labram_frozen_h_source_train_crosswalk_v1_20260810"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = ROOT / "outputs/labram_peft_v8_access_audit_20260811"

FROZEN_SIGNAL_ARTIFACT_SHA256 = (
    "a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66"
)
FROZEN_SIGNAL_RECEIPT_SHA256 = (
    "10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446"
)
FROZEN_CAPABILITY_MANIFEST_SHA256 = (
    "ccd238b17e1da0aa24f2542a314c770900eeed71cbc31282a4acb76dcf957821"
)
FROZEN_CROSSWALK_MANIFEST_SHA256 = (
    "f5a0b40e7d9ecc48ffb2f10a76128da4e110b791db47ac09ace54495bd2d797b"
)
FROZEN_CROSSWALK_RECEIPT_SHA256 = (
    "4eec735065d93f761c1e17753977fe1f0e633d1fdbb6c6888f0af4eb78f6bbee"
)

SPLITS = ("source_train", "source_dev", "source_eval")
EXPECTED_SPLIT_COUNTS = {
    "source_train": {
        "event_count": 582,
        "target_patient_count": 65,
        "public_patient_count": 65,
        "unique_edf_count": 243,
        "unique_path_count": 243,
    },
    "source_dev": {
        "event_count": 221,
        "target_patient_count": 16,
        "public_patient_count": 16,
        "unique_edf_count": 98,
        "unique_path_count": 98,
    },
    "source_eval": {
        "event_count": 185,
        "target_patient_count": 21,
        "public_patient_count": 21,
        "unique_edf_count": 102,
        "unique_path_count": 102,
    },
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _strict_json(raw: bytes, *, field: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field} repeats JSON field {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"{field} contains non-finite value {value}")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must be strict UTF-8 JSON") from exc


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return result


def _stable_file(
    path: str | Path,
    *,
    field: str,
    maximum_bytes: int,
    expected_sha256: str | None = None,
) -> tuple[bytes, str]:
    source = _absolute_no_symlink(path, field=field)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field} must be a regular non-symlinked file")
    before = source.stat()
    if not 1 <= before.st_size <= maximum_bytes:
        raise ValueError(f"{field} has an invalid size")
    raw = source.read_bytes()
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{field} changed while read")
    digest = _bytes_sha256(raw)
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256, field=f"expected {field} SHA256"
    ):
        raise ValueError(f"{field} SHA256 mismatch")
    return raw, digest


def _closed_directory(path: str | Path, *, entries: set[str], field: str) -> Path:
    source = _absolute_no_symlink(path, field=field)
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"{field} must be an existing non-symlink directory")
    actual = {entry.name for entry in source.iterdir()}
    if actual != entries:
        raise ValueError(
            f"{field} violates its closed schema; expected={sorted(entries)}, "
            f"actual={sorted(actual)}"
        )
    return source


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _array(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return value


def _load_capability_metadata(
    directory: str | Path, *, expected_manifest_sha256: str
) -> tuple[Mapping[str, object], Mapping[str, object], dict[str, str]]:
    source = _closed_directory(
        directory,
        entries={"manifest.json", "events.json", "evidence.safetensors"},
        field="source-train-only capability",
    )
    manifest_raw, manifest_sha = _stable_file(
        source / "manifest.json",
        field="source-train capability manifest",
        maximum_bytes=_MAX_MANIFEST_BYTES,
        expected_sha256=expected_manifest_sha256,
    )
    manifest = _mapping(
        _strict_json(manifest_raw, field="source-train capability manifest"),
        field="source-train capability manifest",
    )
    fixed = {
        "schema_version": "soz_source_train_only_iv_capability_v1",
        "model_split": "source_train",
        "source_train_only": True,
        "development_only": True,
        "target_values_loaded": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
        "event_count": 582,
        "patient_count": 65,
    }
    changed = tuple(name for name, expected in fixed.items() if manifest.get(name) != expected)
    if changed:
        raise ValueError(f"source-train capability boundary changed: {changed}")
    files = _mapping(manifest.get("files"), field="capability files")
    event_file = _mapping(files.get("events.json"), field="capability events file")
    expected_events_sha = _require_sha256(
        event_file.get("sha256"), field="capability events SHA256"
    )
    events_raw, events_sha = _stable_file(
        source / "events.json",
        field="source-train capability events",
        maximum_bytes=_MAX_MANIFEST_BYTES,
        expected_sha256=expected_events_sha,
    )
    events = _mapping(
        _strict_json(events_raw, field="source-train capability events"),
        field="source-train capability events",
    )
    if events.get("schema_version") != "soz_source_train_only_iv_event_roster_v1" or (
        events.get("model_split") != "source_train"
    ):
        raise ValueError("source-train capability event document changed")
    return manifest, events, {
        "manifest_sha256": manifest_sha,
        "events_sha256": events_sha,
    }


def _load_crosswalk_metadata(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> tuple[Mapping[str, object], Mapping[str, object], dict[str, str]]:
    source = _closed_directory(
        directory,
        entries={"manifest.json", "receipt.json"},
        field="source-train frozen-H crosswalk",
    )
    manifest_raw, manifest_sha = _stable_file(
        source / "manifest.json",
        field="crosswalk manifest",
        maximum_bytes=_MAX_MANIFEST_BYTES,
        expected_sha256=expected_manifest_sha256,
    )
    manifest = _mapping(
        _strict_json(manifest_raw, field="crosswalk manifest"),
        field="crosswalk manifest",
    )
    fixed_manifest = {
        "schema_version": "soz_labram_frozen_h_source_train_crosswalk_artifact_v1",
        "model_split": "source_train",
        "development_only": True,
        "deepsoz_target_values_loaded": False,
        "source_dev_used": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_promotion": False,
        "receipt_file": "receipt.json",
    }
    changed = tuple(
        name for name, expected in fixed_manifest.items() if manifest.get(name) != expected
    )
    if changed:
        raise ValueError(f"crosswalk manifest boundary changed: {changed}")
    receipt_raw, receipt_sha = _stable_file(
        source / "receipt.json",
        field="crosswalk receipt",
        maximum_bytes=_MAX_RECEIPT_BYTES,
        expected_sha256=expected_receipt_sha256,
    )
    if manifest.get("receipt_sha256") != receipt_sha or manifest.get(
        "receipt_size_bytes"
    ) != len(receipt_raw):
        raise ValueError("crosswalk manifest does not bind the receipt")
    receipt = _mapping(
        _strict_json(receipt_raw, field="crosswalk receipt"),
        field="crosswalk receipt",
    )
    fixed_receipt = {
        "schema_version": "soz_labram_frozen_h_source_train_crosswalk_v1",
        "model_split": "source_train",
        "development_only": True,
        "candidate_input_authorized": True,
        "deepsoz_target_values_loaded": False,
        "source_dev_signal_loaded": False,
        "source_dev_target_loaded": False,
        "source_dev_token_loaded": False,
        "source_eval_used": False,
        "private_used": False,
        "raw_eeg_serialized": False,
        "foundation_token_values_serialized": False,
        "event_count": 582,
        "patient_count": 65,
    }
    changed = tuple(
        name for name, expected in fixed_receipt.items() if receipt.get(name) != expected
    )
    if changed:
        raise ValueError(f"crosswalk receipt boundary changed: {changed}")
    return manifest, receipt, {
        "manifest_sha256": manifest_sha,
        "receipt_sha256": receipt_sha,
    }


def _roster_digest(values: Sequence[str]) -> str:
    return _canonical_sha256(tuple(sorted(set(values))))


def _ordered_digest(values: Sequence[str]) -> str:
    return _canonical_sha256(tuple(values))


def _split_inventory(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    event_ids = [str(row["event_id"]) for row in rows]
    target_patients = [str(row["patient_id"]) for row in rows]
    public_patients = [str(row["local_patient_id"]) for row in rows]
    edfs = [str(row["edf_sha256"]) for row in rows]
    paths = [str(row["relative_edf_path"]) for row in rows]
    return {
        "event_count": len(event_ids),
        "target_patient_count": len(set(target_patients)),
        "public_patient_count": len(set(public_patients)),
        "unique_edf_count": len(set(edfs)),
        "unique_path_count": len(set(paths)),
        "event_roster_sha256": _ordered_digest(event_ids),
        "target_patient_roster_sha256": _roster_digest(target_patients),
        "public_patient_roster_sha256": _roster_digest(public_patients),
        "edf_sha256_roster_sha256": _roster_digest(edfs),
        "relative_path_roster_sha256": _roster_digest(paths),
    }


def _pairwise_receipt(
    left_rows: Sequence[Mapping[str, object]],
    right_rows: Sequence[Mapping[str, object]],
) -> dict[str, int | bool]:
    def values(rows: Sequence[Mapping[str, object]], field: str) -> set[str]:
        return {str(row[field]) for row in rows}

    counts = {
        "target_patient_intersection_count": len(
            values(left_rows, "patient_id") & values(right_rows, "patient_id")
        ),
        "public_patient_intersection_count": len(
            values(left_rows, "local_patient_id")
            & values(right_rows, "local_patient_id")
        ),
        "edf_sha256_intersection_count": len(
            values(left_rows, "edf_sha256") & values(right_rows, "edf_sha256")
        ),
        "relative_path_intersection_count": len(
            values(left_rows, "relative_edf_path")
            & values(right_rows, "relative_edf_path")
        ),
    }
    return {**counts, "all_intersections_empty": not any(counts.values())}


def _validate_raw_paths(
    relative_paths: Sequence[str], tusz_root: str | Path
) -> dict[str, object]:
    root = _absolute_no_symlink(tusz_root, field="TUSZ root")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("TUSZ root must be an existing non-symlink directory")
    resolved_root = root.resolve(strict=True)
    unique_paths = sorted(set(relative_paths))
    for value in unique_paths:
        relative = PurePosixPath(value)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"unsafe relative EDF path: {value!r}")
        candidate = root.joinpath(*relative.parts)
        candidate_absolute = _absolute_no_symlink(candidate, field="planned EDF path")
        resolved = candidate_absolute.resolve(strict=True)
        if resolved == resolved_root or resolved_root not in resolved.parents:
            raise ValueError("planned EDF path escapes the TUSZ root")
        if candidate_absolute.is_symlink() or not candidate_absolute.is_file():
            raise ValueError("planned EDF path is not a regular non-symlinked file")
    return {
        "tusz_root": str(resolved_root),
        "event_path_binding_count": len(relative_paths),
        "unique_path_stat_count": len(unique_paths),
        "all_paths_relative": True,
        "all_paths_resolve_beneath_tusz_root": True,
        "all_paths_regular_non_symlink_files": True,
        "edf_file_bytes_opened": False,
        "edf_content_rehashed": False,
    }


def evaluate_access_boundary(
    *,
    signal_receipt: Mapping[str, object],
    capability_manifest: Mapping[str, object],
    capability_events_document: Mapping[str, object],
    crosswalk_receipt: Mapping[str, object],
    tusz_root: str | Path,
    expected_split_counts: Mapping[str, Mapping[str, int]] = EXPECTED_SPLIT_COUNTS,
) -> dict[str, object]:
    """Build the audit receipt from already-verified metadata objects."""

    signal_rows_raw = _array(signal_receipt.get("events"), field="signal events")
    signal_rows = [
        _mapping(row, field=f"signal events[{index}]")
        for index, row in enumerate(signal_rows_raw)
    ]
    by_split: dict[str, list[Mapping[str, object]]] = {split: [] for split in SPLITS}
    signal_by_event: dict[str, Mapping[str, object]] = {}
    required_signal_fields = {
        "event_id",
        "patient_id",
        "local_patient_id",
        "model_split",
        "edf_sha256",
        "relative_edf_path",
    }
    for row in signal_rows:
        if not required_signal_fields.issubset(row):
            raise ValueError("signal event lacks an access-audit identity field")
        split = str(row["model_split"])
        if split not in by_split:
            raise ValueError(f"unexpected signal model_split {split!r}")
        event_id = str(row["event_id"])
        if event_id in signal_by_event:
            raise ValueError("signal event roster contains duplicate IDs")
        _require_sha256(row["edf_sha256"], field=f"EDF SHA for {event_id}")
        signal_by_event[event_id] = row
        by_split[split].append(row)

    split_inventory = {
        split: _split_inventory(by_split[split]) for split in SPLITS
    }
    for split in SPLITS:
        expected = dict(expected_split_counts[split])
        actual = {
            name: split_inventory[split][name]
            for name in expected
        }
        if actual != expected:
            raise ValueError(
                f"{split} frozen count changed: expected={expected}, actual={actual}"
            )

    pairs = (
        ("source_train", "source_dev"),
        ("source_train", "source_eval"),
        ("source_dev", "source_eval"),
    )
    pairwise = {
        f"{left}__{right}": _pairwise_receipt(by_split[left], by_split[right])
        for left, right in pairs
    }
    if not all(bool(value["all_intersections_empty"]) for value in pairwise.values()):
        raise ValueError("train/dev/eval patient, EDF, or path rosters overlap")

    capability_rows_raw = _array(
        capability_events_document.get("events"), field="capability events"
    )
    capability_rows = [
        _mapping(row, field=f"capability events[{index}]")
        for index, row in enumerate(capability_rows_raw)
    ]
    crosswalk_rows_raw = _array(crosswalk_receipt.get("events"), field="crosswalk events")
    crosswalk_rows = [
        _mapping(row, field=f"crosswalk events[{index}]")
        for index, row in enumerate(crosswalk_rows_raw)
    ]
    if len(capability_rows) != 582 or len(crosswalk_rows) != 582:
        raise ValueError("planned source-train cache must bind exactly 582 events")
    capability_ids = tuple(str(row.get("event_id")) for row in capability_rows)
    crosswalk_ids = tuple(str(row.get("evidence_event_id")) for row in crosswalk_rows)
    if capability_ids != crosswalk_ids or len(set(capability_ids)) != 582:
        raise ValueError("capability and crosswalk event order differ")

    planned_paths: list[str] = []
    planned_edfs: list[str] = []
    planned_target_patients: list[str] = []
    planned_public_patients: list[str] = []
    for index, (capability, crosswalk) in enumerate(zip(capability_rows, crosswalk_rows)):
        event_id = capability_ids[index]
        signal = signal_by_event.get(event_id)
        if signal is None or signal["model_split"] != "source_train":
            raise ValueError("planned cache event is absent from source_train signal metadata")
        bindings = {
            "target patient": str(capability.get("patient_id"))
            == str(crosswalk.get("target_patient_id"))
            == str(signal["patient_id"]),
            "public patient": str(crosswalk.get("public_patient_id"))
            == str(signal["local_patient_id"]),
            "EDF SHA": str(crosswalk.get("edf_sha256"))
            == str(signal["edf_sha256"]),
            "relative EDF path": str(crosswalk.get("relative_edf_path"))
            == str(signal["relative_edf_path"]),
            "OOF fold": crosswalk.get("oof_fold") == capability.get("oof_fold"),
        }
        failed = tuple(name for name, passed in bindings.items() if not passed)
        if failed:
            raise ValueError(f"event {event_id} binding changed: {failed}")
        planned_paths.append(str(signal["relative_edf_path"]))
        planned_edfs.append(str(signal["edf_sha256"]))
        planned_target_patients.append(str(signal["patient_id"]))
        planned_public_patients.append(str(signal["local_patient_id"]))

    if len(set(planned_target_patients)) != 65 or len(set(planned_public_patients)) != 65:
        raise ValueError("planned source-train cache must bind exactly 65 patients")
    capability_receipt = _mapping(
        capability_manifest.get("receipt"), field="capability receipt"
    )
    checks = {
        "event order": _ordered_digest(capability_ids)
        == capability_receipt.get("source_train_event_roster_sha256")
        == capability_receipt.get("event_order_sha256")
        == crosswalk_receipt.get("event_order_sha256"),
        "event set": _roster_digest(capability_ids)
        == capability_receipt.get("source_train_event_set_sha256"),
        "target patient roster": _roster_digest(planned_target_patients)
        == capability_receipt.get("source_train_patient_roster_sha256")
        == crosswalk_receipt.get("patient_roster_sha256"),
        "signal artifact lineage": crosswalk_receipt.get(
            "signal_preflight_artifact_sha256"
        )
        == FROZEN_SIGNAL_ARTIFACT_SHA256,
        "signal receipt lineage": crosswalk_receipt.get(
            "signal_preflight_receipt_sha256"
        )
        == FROZEN_SIGNAL_RECEIPT_SHA256,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"frozen metadata lineage changed: {failed}")

    dev_eval_rows = [*by_split["source_dev"], *by_split["source_eval"]]
    dev_eval_target = {str(row["patient_id"]) for row in dev_eval_rows}
    dev_eval_public = {str(row["local_patient_id"]) for row in dev_eval_rows}
    dev_eval_edfs = {str(row["edf_sha256"]) for row in dev_eval_rows}
    dev_eval_paths = {str(row["relative_edf_path"]) for row in dev_eval_rows}
    exclusions = {
        "target_patient_intersection_count": len(
            set(planned_target_patients) & dev_eval_target
        ),
        "public_patient_intersection_count": len(
            set(planned_public_patients) & dev_eval_public
        ),
        "edf_sha256_intersection_count": len(set(planned_edfs) & dev_eval_edfs),
        "relative_path_intersection_count": len(set(planned_paths) & dev_eval_paths),
    }
    if any(exclusions.values()):
        raise ValueError("planned source-train reads intersect source-dev/eval metadata")
    path_receipt = _validate_raw_paths(planned_paths, tusz_root)

    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS",
        "audit_scope": {
            "protocol": "LaBraM endpoint-aligned PEFT recovery v8",
            "cohort": "DeepSOZ/TUSZ signal-preflight eligible events only",
            "metadata_only": True,
            "split_names": list(SPLITS),
            "soz_target_values_read": False,
            "waveform_bytes_read": False,
            "evidence_tensor_files_opened": False,
            "private_data_paths_received_or_read": False,
            "source_dev_or_source_eval_forward": False,
        },
        "split_inventory": split_inventory,
        "pairwise_disjointness": pairwise,
        "source_train_cache_read_plan": {
            "event_access_count": len(capability_ids),
            "target_patient_count": len(set(planned_target_patients)),
            "public_patient_count": len(set(planned_public_patients)),
            "unique_edf_count": len(set(planned_edfs)),
            "unique_path_count": len(set(planned_paths)),
            "capability_crosswalk_event_order_exact": True,
            "crosswalk_signal_identity_exact": True,
            "planned_vs_source_dev_eval_intersections": exclusions,
            "all_planned_events_are_source_train": True,
            "path_validation": path_receipt,
        },
        "foundation_pretraining_exposure": {
            "backbone": "official LaBraM-Base",
            "official_pretraining_contains_tusz": True,
            "exposure_eliminable_by_current_patient_split": False,
            "exact_checkpoint_patient_content_overlap_auditable_here": False,
            "pretraining_clean": False,
            "adaptation_stage_patient_disjoint": True,
            "required_claim": "pretraining-exposed, patient-held-out adaptation",
            "prohibited_claim": "pretraining-clean held-out evaluation",
            "interpretation": (
                "This audit proves the v8 adaptation-time read boundary only; "
                "it cannot undo or quantify LaBraM's historical TUSZ exposure."
            ),
        },
        "boundary_assertions": {
            "eligible_split_patient_rosters_pairwise_disjoint": True,
            "eligible_split_edf_sha256_sets_pairwise_disjoint": True,
            "eligible_split_relative_path_sets_pairwise_disjoint": True,
            "planned_cache_is_exactly_582_events_65_patients": True,
            "planned_raw_paths_beneath_tusz_root": True,
            "planned_reads_exclude_source_dev_and_source_eval": True,
            "adaptation_access_boundary_pass": True,
            "foundation_pretraining_clean": False,
        },
    }


AUDIT_JSON_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": JSON_SCHEMA_ID,
    "title": "LaBraM PEFT v8 metadata-only access-boundary audit",
    "type": "object",
    "required": [
        "schema_version",
        "status",
        "audit_scope",
        "frozen_inputs",
        "split_inventory",
        "pairwise_disjointness",
        "source_train_cache_read_plan",
        "foundation_pretraining_exposure",
        "boundary_assertions",
    ],
    "properties": {
        "schema_version": {"const": AUDIT_SCHEMA},
        "status": {"const": "PASS"},
        "audit_scope": {"type": "object"},
        "frozen_inputs": {"type": "object"},
        "split_inventory": {
            "type": "object",
            "required": list(SPLITS),
        },
        "pairwise_disjointness": {"type": "object"},
        "source_train_cache_read_plan": {"type": "object"},
        "foundation_pretraining_exposure": {
            "type": "object",
            "required": [
                "official_pretraining_contains_tusz",
                "pretraining_clean",
                "required_claim",
                "prohibited_claim",
            ],
            "properties": {
                "official_pretraining_contains_tusz": {"const": True},
                "pretraining_clean": {"const": False},
                "required_claim": {"type": "string"},
                "prohibited_claim": {"type": "string"},
            },
        },
        "boundary_assertions": {"type": "object"},
    },
    "additionalProperties": False,
}


def _readme(audit: Mapping[str, object]) -> str:
    plan = _mapping(audit["source_train_cache_read_plan"], field="read plan")
    return (
        "# LaBraM PEFT v8 access-boundary audit\n\n"
        "Status: **PASS**. This artifact is a metadata-only pre-training-run "
        "gate, not a model result.\n\n"
        f"- Planned source-train access: {plan['event_access_count']} events, "
        f"{plan['target_patient_count']} target patients, "
        f"{plan['unique_edf_count']} unique EDF identities.\n"
        "- DeepSOZ/TUSZ eligible source-train, source-dev, and source-eval "
        "patient rosters, EDF SHA sets, and relative paths are pairwise disjoint.\n"
        "- Every planned path was checked by filesystem metadata to be a regular "
        "non-symlink file beneath the frozen TUSZ root. EDF bytes were not opened "
        "or rehashed.\n"
        "- No SOZ target value, waveform, evidence tensor, or private-data path "
        "was read.\n\n"
        "Important limitation: official LaBraM pretraining contains TUSZ. The "
        "permitted claim is therefore **pretraining-exposed, patient-held-out "
        "adaptation**, never pretraining-clean evaluation.\n"
    )


def _publish(output_directory: str | Path, audit: Mapping[str, object]) -> Path:
    target = _absolute_no_symlink(output_directory, field="audit output")
    if target.name in {"", ".", ".."} or os.path.lexists(target):
        raise FileExistsError("audit output must be a new concrete directory")
    parent = _absolute_no_symlink(target.parent, field="audit output parent")
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    published = False
    try:
        payloads = {
            "access_audit.json": _canonical_json_bytes(audit) + b"\n",
            "access_audit.schema.json": _canonical_json_bytes(AUDIT_JSON_SCHEMA) + b"\n",
            "README.md": _readme(audit).encode("utf-8"),
        }
        for name, raw in payloads.items():
            (staging / name).write_bytes(raw)
        manifest = {
            "schema_version": ARTIFACT_SCHEMA,
            "serialization": "canonical_json_utf8_newline_plus_markdown_no_pickle",
            "status": "PASS",
            "audit_receipt_sha256": _canonical_sha256(audit),
            "files": {
                name: {"sha256": _bytes_sha256(raw), "size_bytes": len(raw)}
                for name, raw in sorted(payloads.items())
            },
            "scientific_result": False,
            "training_authorized_only_if_status_pass": True,
        }
        (staging / "manifest.json").write_bytes(
            _canonical_json_bytes(manifest) + b"\n"
        )
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(staging, target)
        published = True
        return target
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--signal-preflight-bundle", type=Path, default=DEFAULT_SIGNAL_BUNDLE)
    parser.add_argument(
        "--expected-signal-artifact-sha256", default=FROZEN_SIGNAL_ARTIFACT_SHA256
    )
    parser.add_argument(
        "--expected-signal-receipt-sha256", default=FROZEN_SIGNAL_RECEIPT_SHA256
    )
    parser.add_argument("--source-train-capability", type=Path, default=DEFAULT_CAPABILITY_BUNDLE)
    parser.add_argument(
        "--expected-capability-manifest-sha256",
        default=FROZEN_CAPABILITY_MANIFEST_SHA256,
    )
    parser.add_argument("--source-train-crosswalk", type=Path, default=DEFAULT_CROSSWALK_BUNDLE)
    parser.add_argument(
        "--expected-crosswalk-manifest-sha256",
        default=FROZEN_CROSSWALK_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-crosswalk-receipt-sha256",
        default=FROZEN_CROSSWALK_RECEIPT_SHA256,
    )
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    signal = load_bound_deepsoz_signal_preflight_artifact(
        args.signal_preflight_bundle,
        expected_artifact_sha256=args.expected_signal_artifact_sha256,
        expected_receipt_sha256=args.expected_signal_receipt_sha256,
    )
    capability_manifest, capability_events, capability_hashes = (
        _load_capability_metadata(
            args.source_train_capability,
            expected_manifest_sha256=args.expected_capability_manifest_sha256,
        )
    )
    _, crosswalk_receipt, crosswalk_hashes = _load_crosswalk_metadata(
        args.source_train_crosswalk,
        expected_manifest_sha256=args.expected_crosswalk_manifest_sha256,
        expected_receipt_sha256=args.expected_crosswalk_receipt_sha256,
    )
    audit = evaluate_access_boundary(
        signal_receipt=signal.receipt,
        capability_manifest=capability_manifest,
        capability_events_document=capability_events,
        crosswalk_receipt=crosswalk_receipt,
        tusz_root=args.tusz_root,
    )
    audit["frozen_inputs"] = {
        "signal_preflight": {
            "artifact_sha256": signal.artifact_sha256,
            "receipt_sha256": signal.receipt_sha256,
            "metadata_file_opened": True,
        },
        "source_train_only_capability": {
            **capability_hashes,
            "manifest_and_event_roster_opened": True,
            "evidence_safetensors_opened": False,
        },
        "source_train_frozen_h_crosswalk": {
            **crosswalk_hashes,
            "metadata_files_opened": True,
            "token_tensors_opened": False,
        },
    }
    output = _publish(args.output_directory, audit)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "event_count": 582,
                "patient_count": 65,
                "source_dev_eval_intersections": 0,
                "waveform_bytes_read": False,
                "soz_target_values_read": False,
                "private_data_read": False,
                "pretraining_clean": False,
                "required_claim": "pretraining-exposed, patient-held-out adaptation",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
