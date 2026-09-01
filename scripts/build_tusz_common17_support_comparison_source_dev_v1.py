#!/usr/bin/env python3
"""Freeze all source-dev events for the common-17 support-policy ablation.

The extraction manifest and post-freeze references are emitted as separate
content-addressed files.  The former contains only navigation onsets.  The
latter contains global TERM onset/offset intervals and must never be opened by
the extraction command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.adaptive_native_evidence_common17 import (  # noqa: E402
    COMMON17_CHANNELS,
)
from src.clinical_eeg_long_recording.tusz_real_edf_support_comparison_v1 import (  # noqa: E402
    TUSZ_REAL_EDF_SUPPORT_COMPARISON_MANIFEST_SCHEMA,
    validate_tusz_real_edf_support_comparison_manifest_v1,
)


DEFAULT_PHASE_MANIFEST = (
    ROOT
    / "outputs/clinical_eeg_common17_car17_labram_phase_v1_20260824/manifest.json"
)
DEFAULT_ROSTER_PROJECTION = (
    ROOT / "outputs/tusz_complete_detector_roster_v2_20260823/analysis_projection.json"
)
DEFAULT_EVENT_INPUTS = (
    ROOT / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/event_inputs.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tusz_common17_support_comparison_source_dev259_manifest_v1_20260825"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _event_input_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        event_id = str(row.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("event input table contains an empty event_id")
        if event_id in result:
            raise ValueError(f"event input table duplicates {event_id}")
        result[event_id] = row
    return result


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    phase_path = arguments.phase_manifest.resolve(strict=True)
    roster_path = arguments.roster_projection.resolve(strict=True)
    inputs_path = arguments.event_inputs.resolve(strict=True)
    phase = _read_json(phase_path)
    roster = _read_json(roster_path)
    inputs = _event_input_rows(inputs_path)

    event_roster = phase.get("scope", {}).get("event_roster")
    if not isinstance(event_roster, list) or not event_roster:
        raise ValueError("common17 phase manifest lacks its frozen event roster")
    split_patients: dict[str, set[str]] = {"train": set(), "dev": set(), "eval": set()}
    for row in event_roster:
        relative = PurePosixPath(str(row["relative_edf_path"]))
        if not relative.parts or relative.parts[0] not in split_patients:
            raise ValueError("common17 phase roster has an unknown official split")
        split_patients[relative.parts[0]].add(str(row["patient_id"]))
    if split_patients["dev"] & (split_patients["train"] | split_patients["eval"]):
        raise ValueError("common17 phase roster is not patient-disjoint by official split")

    projection_rows = roster.get("records")
    if not isinstance(projection_rows, list):
        raise ValueError("complete roster projection lacks records")
    sha_by_path: dict[str, str] = {}
    for row in projection_rows:
        path = str(row["local_edf_path"])
        if path in sha_by_path:
            raise ValueError(f"complete roster projection duplicates {path}")
        sha_by_path[path] = str(row["source_edf_container_sha256"])

    selected = [
        row
        for row in event_roster
        if PurePosixPath(str(row["relative_edf_path"])).parts[0] == "dev"
    ]
    selected.sort(key=lambda row: int(row["ordinal"]))
    entries: list[dict[str, object]] = []
    references: dict[str, dict[str, float]] = {}
    recording_ids: set[str] = set()
    patient_ids: set[str] = set()
    for ordinal, row in enumerate(selected):
        event_id = str(row["event_id"])
        relative = PurePosixPath(str(row["relative_edf_path"]))
        source = inputs.get(event_id)
        if source is None:
            raise KeyError(f"event input table lacks frozen event {event_id}")
        if str(source["local_edf_path"]).strip() != relative.as_posix():
            raise ValueError(f"event path mismatch for {event_id}")
        patient = str(row["patient_id"])
        if str(source["deepsoz_patient_id"]).strip() != patient:
            raise ValueError(f"patient group mismatch for {event_id}")
        onset = float(row["global_t0_sec"])
        input_onset = float(source["t0_sec"])
        offset = float(source["seizure_end_sec"])
        if (
            not math.isfinite(onset)
            or not math.isfinite(input_onset)
            or not math.isfinite(offset)
            or abs(onset - input_onset) > 1.0e-6
            or offset <= onset
        ):
            raise ValueError(f"reference interval mismatch for {event_id}")
        recording = relative.stem
        sha = sha_by_path.get(relative.as_posix())
        if sha is None:
            raise KeyError(f"complete roster lacks {relative.as_posix()}")
        entries.append(
            {
                "ordinal": ordinal,
                "rollout_id": f"SOURCE-DEV-SUPPORT-{ordinal:04d}",
                "event_id": event_id,
                "recording_id": recording,
                "patient_group_id": f"DEEPSOZ-{patient}",
                "official_split": "dev",
                "relative_edf_path": relative.as_posix(),
                "edf_sha256": sha,
                "navigation_anchor_recording_seconds": onset,
            }
        )
        references[event_id] = {
            "onset_seconds": onset,
            "offset_seconds": offset,
        }
        recording_ids.add(recording)
        patient_ids.add(patient)

    if len(entries) != 259 or len(recording_ids) != 100 or len(patient_ids) != 15:
        raise ValueError(
            "frozen common17 source-dev denominator drifted from 259/100/15"
        )
    bindings = {
        "common17_phase_manifest_sha256": _file_sha256(phase_path),
        "complete_roster_projection_sha256": _file_sha256(roster_path),
        "event_input_table_sha256": _file_sha256(inputs_path),
    }
    manifest: dict[str, Any] = {
        "schema_version": TUSZ_REAL_EDF_SUPPORT_COMPARISON_MANIFEST_SCHEMA,
        "cohort_id": "TUSZ-COMMON17-SOURCE-DEV-ALL-259-SUPPORT-COMPARISON-V1",
        "common17_channel_order": list(COMMON17_CHANNELS),
        "selection_contract": {
            "official_split": "source_dev_only",
            "population": "all_events_in_frozen_common17_oracle_roster_source_dev",
            "patient_disjoint_from_source_train_and_source_eval": True,
            "navigation_anchor_source": "frozen_global_TERM_seiz_onset_navigation_only",
            "seizure_offset_present_in_extraction_manifest": False,
            "runtime_TERM_or_annotation_sidecar_access": False,
            "window_or_stopping_selection_uses_reference": False,
            "channel_or_SOZ_target_used": False,
            "clinical_text_used": False,
            "source_eval_opened": False,
        },
        "source_bindings": bindings,
        "cohort_statistics": {
            "event_count": len(entries),
            "recording_count": len(recording_ids),
            "patient_group_count": len(patient_ids),
            "one_official_split": True,
            "patient_overlap_with_source_train": 0,
            "patient_overlap_with_source_eval": 0,
        },
        "entries": entries,
    }
    validate_tusz_real_edf_support_comparison_manifest_v1(manifest)
    output = arguments.output.resolve()
    manifest_path = output / "extraction_manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest_sha = _file_sha256(manifest_path)

    postfreeze: dict[str, Any] = {
        "schema_version": "clinical_eeg_tusz_support_comparison_postfreeze_references_v1",
        "cohort_id": manifest["cohort_id"],
        "extraction_manifest_sha256": manifest_sha,
        "reference_source": "global_TERM_seiz_intervals_postfreeze_audit_only",
        "event_count": len(references),
        "reference_intervals_by_event_id": references,
        "firewall": {
            "file_is_input_to_extraction": False,
            "file_may_be_opened_only_after_event_receipts_are_frozen": True,
            "contains_channel_or_SOZ_targets": False,
        },
    }
    postfreeze["receipt_sha256"] = _canonical_sha256(postfreeze)
    reference_path = output / "postfreeze_references.json"
    _atomic_json(reference_path, postfreeze)

    receipt: dict[str, Any] = {
        "schema_version": "clinical_eeg_tusz_support_comparison_cohort_build_receipt_v1",
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "cohort_id": manifest["cohort_id"],
        "source_bindings": bindings,
        "extraction_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "event_count": len(entries),
            "recording_count": len(recording_ids),
            "patient_group_count": len(patient_ids),
            "seizure_offsets_present": False,
        },
        "postfreeze_references": {
            "path": str(reference_path),
            "sha256": _file_sha256(reference_path),
            "opened_by_extraction": False,
        },
        "patient_partition_audit": {
            "dev_train_overlap": 0,
            "dev_eval_overlap": 0,
            "source_eval_event_opened": False,
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    _atomic_json(output / "receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--phase-manifest", type=Path, default=DEFAULT_PHASE_MANIFEST)
    value.add_argument("--roster-projection", type=Path, default=DEFAULT_ROSTER_PROJECTION)
    value.add_argument("--event-inputs", type=Path, default=DEFAULT_EVENT_INPUTS)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    result = build(parser().parse_args())
    print(json.dumps(result["extraction_manifest"], ensure_ascii=False, indent=2))

