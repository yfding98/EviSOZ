#!/usr/bin/env python3
"""Project the frozen private SZ-marker roster into an onset-only ledger.

This is a retrospective diagnostic input, not an EEG-only production input.
The producer opens the historical target-free signal roster and the current
pseudonymous 141-record inventory.  It never opens the sibling SOZ target
ledger, EDF annotations, Excel workbooks, model predictions, or doctor-label
release.  Only annotation-derived onset times associated upstream with
exact-ID SZ event matches and legacy [-12,+48] support pre-eligibility are
published; all other legacy columns are discarded.  The downstream common17
reader must still verify actual sample support and signal/channel QC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "private_common17_oracle_onset_roster_v2"
DEFAULT_INVENTORY = (
    ROOT / "outputs/private_long_recording_inventory_v1_full141_20260819.json"
)
DEFAULT_SOURCE_ROSTER = (
    ROOT
    / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/signal_roster.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs/private_common17_oracle_onset_roster_v2_20260825"
EXPECTED_RECORDS = 141
EXPECTED_PATIENTS = 45
EXPECTED_SOURCE_EVENTS = 123
EXPECTED_EXACT_SUPPORTED = 81
EXPECTED_STRICT_PRIMARY = 75
REQUIRED_SOURCE_FIELDS = frozenset(
    {
        "event_id",
        "patient_id",
        "relative_edf_path",
        "global_event_t0_sec",
        "duration_sec",
        "time_source",
        "time_support_preeligible",
        "primary_anchor_preeligible",
        "expanded_anchor_preeligible",
    }
)
FORBIDDEN_SOZ_FIELDS = frozenset(
    {
        "candidate_positive_electrodes",
        "standard19_positive_electrodes",
        "known_spread_electrodes",
        "outside_head_positive_electrodes",
        "diffuse_spread_present",
        "has_c18_positive",
        "primary_reference_preeligible",
        "expanded_reference_preeligible",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _safe_relative(value: object) -> str:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
        raise ValueError("unsafe EDF path in frozen onset source")
    return relative.as_posix()


def _read_source(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.resolve(strict=True).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not REQUIRED_SOURCE_FIELDS <= set(fields):
        raise ValueError("frozen onset source lacks required signal/time fields")
    if FORBIDDEN_SOZ_FIELDS & set(fields):
        raise RuntimeError("frozen onset source unexpectedly contains SOZ target columns")
    if len(rows) != EXPECTED_SOURCE_EVENTS:
        raise ValueError("frozen onset source no longer contains 123 events")
    if len({row["event_id"] for row in rows}) != len(rows):
        raise ValueError("frozen onset source event IDs are duplicated")
    return rows, fields


def materialize(
    inventory_path: Path,
    source_roster_path: Path,
) -> dict[str, Any]:
    inventory = _read_json(inventory_path)
    records = inventory.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORDS:
        raise ValueError("private inventory no longer contains 141 records")
    if len({str(row["patient_pseudonym"]) for row in records}) != EXPECTED_PATIENTS:
        raise ValueError("private inventory no longer contains 45 patients")
    by_relative: dict[str, Mapping[str, Any]] = {}
    for row in records:
        relative = _safe_relative(row["edf_relative_path"])
        if relative in by_relative:
            raise ValueError("private inventory contains duplicate EDF relative paths")
        by_relative[relative] = row

    source_rows, source_fields = _read_source(source_roster_path)
    old_to_current: dict[str, set[str]] = {}
    current_to_old: dict[str, set[str]] = {}
    projected: list[dict[str, Any]] = []
    for source in source_rows:
        relative = _safe_relative(source["relative_edf_path"])
        current = by_relative.get(relative)
        if current is None:
            raise ValueError("frozen onset source EDF is absent from current inventory")
        old_patient = str(source["patient_id"])
        current_patient = str(current["patient_pseudonym"])
        old_to_current.setdefault(old_patient, set()).add(current_patient)
        current_to_old.setdefault(current_patient, set()).add(old_patient)

        if source["time_source"] != "exact_sz_marker":
            continue
        if source["time_support_preeligible"] != "1":
            continue
        anchor = float(source["global_event_t0_sec"])
        duration = float(source["duration_sec"])
        if not math.isfinite(anchor) or not math.isfinite(duration):
            raise ValueError("exact SZ marker has non-finite time support")
        if anchor < 42.0 or duration < anchor + 48.0:
            raise ValueError("exact SZ marker violates frozen time-support contract")
        strict = source["primary_anchor_preeligible"] == "1"
        identity = {
            "recording_id": str(current["recording_id"]),
            "source_signal_sha256": str(current["source_signal_sha256"]),
            "source_event_id": str(source["event_id"]),
            "anchor_seconds": anchor,
        }
        projected.append(
            {
                "oracle_event_id": f"ORACLE-{_canonical_sha256(identity)[:20]}",
                "recording_id": identity["recording_id"],
                "patient_pseudonym": current_patient,
                "source_signal_sha256": identity["source_signal_sha256"],
                "anchor_seconds": anchor,
                "anchor_time_semantics": (
                    "doctor_SZ_event_exact_ID_matched_annotation_derived_onset"
                ),
                "strict_single_marker_primary": strict,
                "legacy_time_support_preeligible": True,
            }
        )

    if any(len(values) != 1 for values in old_to_current.values()):
        raise RuntimeError("historical patient maps to multiple current pseudonyms")
    if any(len(values) != 1 for values in current_to_old.values()):
        raise RuntimeError("current patient pseudonym maps to multiple historical patients")
    if len(projected) != EXPECTED_EXACT_SUPPORTED:
        raise RuntimeError("exact supported oracle event count drifted")
    if sum(row["strict_single_marker_primary"] for row in projected) != EXPECTED_STRICT_PRIMARY:
        raise RuntimeError("strict primary oracle event count drifted")
    if len({row["oracle_event_id"] for row in projected}) != len(projected):
        raise RuntimeError("projected oracle event IDs are duplicated")
    if len({row["recording_id"] for row in projected}) != len(projected):
        raise RuntimeError("exact supported source unexpectedly repeats a recording")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_private_exact_SZ_event_matched_onset_only_projection",
        "interpretation": (
            "retrospective_reference_timing_diagnostic_not_EEG_only_and_not_"
            "clinician_verified_electrographic_onset"
        ),
        "cohort": {
            "inventory_records": EXPECTED_RECORDS,
            "inventory_patients": EXPECTED_PATIENTS,
            "historical_source_events": len(source_rows),
            "exact_supported_events": len(projected),
            "exact_supported_records": len({row["recording_id"] for row in projected}),
            "exact_supported_patients": len({row["patient_pseudonym"] for row in projected}),
            "strict_primary_events": sum(
                row["strict_single_marker_primary"] for row in projected
            ),
            "strict_primary_records": len(
                {
                    row["recording_id"]
                    for row in projected
                    if row["strict_single_marker_primary"]
                }
            ),
            "strict_primary_patients": len(
                {
                    row["patient_pseudonym"]
                    for row in projected
                    if row["strict_single_marker_primary"]
                }
            ),
        },
        "events": projected,
        "access_receipt": {
            "historical_signal_roster_opened": True,
            "historical_signal_roster_fields_present": list(source_fields),
            "non_onset_legacy_columns_projected": False,
            "sibling_SOZ_target_ledger_opened": False,
            "doctor_label_release_opened": False,
            "raw_EDF_annotations_opened": False,
            "Excel_workbook_opened": False,
            "private_EEG_samples_loaded": False,
            "model_predictions_loaded": False,
            "training_calibration_or_model_selection_performed": False,
            "upstream_doctor_event_to_annotation_join_used": True,
            "upstream_EDF_annotation_timing_used": True,
            "EEG_only_eligible": False,
        },
        "lineage": {
            "inventory_sha256": _sha256(inventory_path),
            "historical_signal_roster_sha256": _sha256(source_roster_path),
        },
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


def publish(output: Path, payload: Mapping[str, Any]) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        receipt = {
            "schema_version": f"{SCHEMA}_receipt",
            "status": payload["status"],
            "content_sha256": payload["content_sha256"],
            "manifest_sha256": _sha256(manifest_path),
            "exact_supported_events": payload["cohort"]["exact_supported_events"],
            "strict_primary_events": payload["cohort"]["strict_primary_events"],
            "SOZ_targets_loaded": False,
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--source-roster", type=Path, default=DEFAULT_SOURCE_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = materialize(args.inventory, args.source_roster)
    output = publish(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "exact_supported_events": payload["cohort"]["exact_supported_events"],
                "strict_primary_events": payload["cohort"]["strict_primary_events"],
                "SOZ_targets_loaded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
