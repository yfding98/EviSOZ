#!/usr/bin/env python3
"""Build or preflight a target-free continuous-detector split roster.

The command supports CSV and JSONL manifests.  It hashes the exact raw file,
then retains only an allowlisted patient identity, recording identity and
already-frozen split field.  Extra seizure intervals, channel/SOZ targets,
annotations, spreadsheet-derived fields and clinical text are never projected
into the in-memory row roster or output receipt.

This is an inventory/isolation utility.  It does not open EDF signals, apply a
detector, inspect reference labels or authorize a complete inventory,
performance, SOTA or production claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_detection_roster import (  # noqa: E402
    CONTINUOUS_DETECTOR_ROSTER_ALLOWED_PATIENT_FIELDS,
    CONTINUOUS_DETECTOR_ROSTER_ALLOWED_RECORDING_FIELDS,
    CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_FIELDS,
    CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_VALUES,
    CONTINUOUS_DETECTOR_ROSTER_INVENTORY_SCOPES,
    build_continuous_detector_split_roster,
    validate_continuous_detector_projection_fields,
)


DEFAULT_MAX_MANIFEST_BYTES = 128 * 1024 * 1024


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _project_row(
    row: Mapping[str, object],
    *,
    patient_field: str,
    recording_field: str,
    split_field: str,
    context: str,
) -> dict[str, object]:
    fields = (patient_field, recording_field, split_field)
    missing = [name for name in fields if name not in row]
    if missing:
        raise ValueError(f"{context} lacks required projection fields {missing}")
    return {name: row[name] for name in fields}


def load_manifest_identity_split_rows(
    manifest_path: Path,
    *,
    patient_field: str,
    recording_field: str,
    split_field: str,
    maximum_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
) -> tuple[list[dict[str, object]], str, str]:
    """Read CSV/JSONL while retaining only three allowlisted fields."""

    patient_key, recording_key, split_key = (
        validate_continuous_detector_projection_fields(
            patient_field=patient_field,
            recording_field=recording_field,
            split_field=split_field,
        )
    )
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("maximum manifest bytes must be a positive integer")
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError("manifest path must name an existing regular file")
    raw = path.read_bytes()
    if not raw:
        raise ValueError("manifest file is empty")
    if len(raw) > maximum_bytes:
        raise ValueError("manifest exceeds the configured byte limit")
    raw_sha256 = _sha256_bytes(raw)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("manifest must be UTF-8 or UTF-8 with BOM") from error

    suffix = path.suffix.lower()
    fields = (patient_key, recording_key, split_key)
    rows: list[dict[str, object]] = []
    if suffix == ".csv":
        reader = csv.reader(io.StringIO(text, newline=""))
        try:
            header = next(reader)
        except StopIteration as error:  # pragma: no cover - raw non-empty guard
            raise ValueError("CSV manifest has no header") from error
        if (
            not header
            or any(not value or value != value.strip() for value in header)
            or len(set(header)) != len(header)
        ):
            raise ValueError("CSV manifest header is empty, duplicated or untrimmed")
        missing = [name for name in fields if name not in header]
        if missing:
            raise ValueError(f"CSV manifest lacks required projection fields {missing}")
        indices = {name: header.index(name) for name in fields}
        for line_number, values in enumerate(reader, start=2):
            if not values:
                raise ValueError(f"CSV manifest line {line_number} is blank")
            if len(values) != len(header):
                raise ValueError(
                    f"CSV manifest line {line_number} has a non-rectangular field count"
                )
            rows.append({name: values[indices[name]] for name in fields})
        manifest_format = "csv"
    elif suffix == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"JSONL manifest line {line_number} is blank")
            try:
                parsed = json.loads(
                    line,
                    object_pairs_hook=_duplicate_rejecting_object,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {value}")
                    ),
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"JSONL manifest line {line_number} is invalid: {error}"
                ) from error
            if not isinstance(parsed, Mapping):
                raise ValueError(f"JSONL manifest line {line_number} is not an object")
            rows.append(
                _project_row(
                    parsed,
                    patient_field=patient_key,
                    recording_field=recording_key,
                    split_field=split_key,
                    context=f"JSONL manifest line {line_number}",
                )
            )
        manifest_format = "jsonl"
    else:
        raise ValueError("manifest extension must be .csv or .jsonl")
    if not rows:
        raise ValueError("manifest contains no data rows")
    return rows, raw_sha256, manifest_format


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    if output.exists():
        raise ValueError("output already exists; roster receipts are append-only")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            raise ValueError("output appeared during materialization; refusing overwrite")
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a content-bound identity/split-only continuous-detector roster"
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--patient-field",
        default="local_patient_id",
        choices=sorted(CONTINUOUS_DETECTOR_ROSTER_ALLOWED_PATIENT_FIELDS),
    )
    parser.add_argument(
        "--recording-field",
        default="local_edf_path",
        choices=sorted(CONTINUOUS_DETECTOR_ROSTER_ALLOWED_RECORDING_FIELDS),
    )
    parser.add_argument(
        "--split-field",
        default="model_split",
        choices=sorted(CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_FIELDS),
    )
    parser.add_argument(
        "--inventory-scope",
        default="manifest_scope_not_claimed",
        choices=sorted(CONTINUOUS_DETECTOR_ROSTER_INVENTORY_SCOPES),
    )
    parser.add_argument(
        "--require-only-split",
        choices=sorted(CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_VALUES),
        help="fail unless the manifest contains exactly this one frozen split",
    )
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument(
        "--maximum-bytes", type=int, default=DEFAULT_MAX_MANIFEST_BYTES
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate and summarize without writing an artifact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        rows, manifest_sha256, manifest_format = load_manifest_identity_split_rows(
            args.manifest,
            patient_field=args.patient_field,
            recording_field=args.recording_field,
            split_field=args.split_field,
            maximum_bytes=args.maximum_bytes,
        )
        if (
            args.expected_manifest_sha256 is not None
            and manifest_sha256 != args.expected_manifest_sha256
        ):
            raise ValueError("manifest raw SHA-256 differs from the expected content")
        roster = build_continuous_detector_split_roster(
            manifest_rows=rows,
            manifest_file_sha256=manifest_sha256,
            patient_field=args.patient_field,
            recording_field=args.recording_field,
            split_field=args.split_field,
            inventory_scope=args.inventory_scope,
        )
        split_names = sorted(roster["split_rosters"])
        if args.require_only_split is not None and split_names != [
            args.require_only_split
        ]:
            raise ValueError(
                "manifest split roster does not equal --require-only-split; no rows "
                "were filtered or relabelled"
            )
        if not args.preflight_only and args.output is None:
            raise ValueError("--output is required unless --preflight-only is set")
        if args.preflight_only and args.output is not None:
            raise ValueError("--preflight-only cannot be combined with --output")
        summary: dict[str, Any] = {
            "status": "preflight_complete" if args.preflight_only else "written",
            "manifest_format": manifest_format,
            "source_manifest_file_sha256": manifest_sha256,
            "roster_id": roster["roster_id"],
            "inventory_scope": roster["inventory_scope"],
            "total_patient_count": roster["total_patient_count"],
            "total_recording_count": roster["total_recording_count"],
            "split_counts": {
                split: {
                    "patient_count": roster["split_rosters"][split]["patient_count"],
                    "recording_count": roster["split_rosters"][split][
                        "recording_count"
                    ],
                }
                for split in split_names
            },
            "target_annotation_excel_fields_projected": False,
            "complete_split_inventory_verified": False,
        }
        if not args.preflight_only:
            _write_new_json(args.output, roster)
            summary["output"] = str(args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2  # pragma: no cover - argparse.error exits


if __name__ == "__main__":
    raise SystemExit(main())

