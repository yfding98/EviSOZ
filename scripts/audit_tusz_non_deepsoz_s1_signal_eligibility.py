#!/usr/bin/env python3
"""Audit label-fresh TUSZ seizure patients for the frozen S1 signal contract.

This audit is deliberately target-free with respect to SOZ.  It reads only:

* the 124 local DeepSOZ patient identities, for exclusion;
* official global TUSZ seizure intervals, as event-navigation anchors; and
* the corresponding EDF signal under the frozen direct-standard-19 causal
  preprocessing contract.

It never opens per-channel TUSZ involvement annotations, DeepSOZ SOZ values,
model predictions, or private data.  An eligible event therefore means only
that a complete ``[-12,+48)`` second standard-19 signal can be constructed;
it does not mean that an SOZ label exists or is inferable.

The output is a resumable JSON receipt.  Progress is committed after every
EDF record so an interrupted full train/dev/eval scan can continue without
replaying completed signal reads.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.edf import (  # noqa: E402
    CausalEDFConfig,
    EDFEventEligibilityError,
    load_standard19_edf_event,
)
from src.soz.data.tusz import list_tusz_global_seizure_events  # noqa: E402
from src.soz.geometry import STANDARD_19  # noqa: E402


DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_DEEPSOZ_SPLIT = ROOT / "outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv"
DEFAULT_OUTPUT = ROOT / "outputs/tusz_non_deepsoz_s1_signal_eligibility_v1_20260813.json"

SCHEMA_VERSION = "tusz_non_deepsoz_s1_signal_eligibility_v1"
STATUS_RUNNING = "target_free_signal_eligibility_audit_in_progress"
STATUS_COMPLETE = "target_free_signal_eligibility_audit_complete"
EXPECTED_DEEPSOZ_IDENTITIES = 124
SPLITS = ("train", "dev", "eval")
_PATIENT_RE = re.compile(r"^[a-z0-9]+$")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_deepsoz_local_ids(path: Path) -> set[str]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    identities = {str(row.get("local_patient_id", "")).strip().lower() for row in rows}
    if (
        len(rows) != EXPECTED_DEEPSOZ_IDENTITIES
        or len(identities) != EXPECTED_DEEPSOZ_IDENTITIES
        or "" in identities
        or any(_PATIENT_RE.fullmatch(value) is None for value in identities)
    ):
        raise ValueError("DeepSOZ exclusion roster is not exactly 124 local identities")
    return identities


def _record_identity(root: Path, edf: Path) -> tuple[str, str, str]:
    relative = edf.relative_to(root)
    if len(relative.parts) != 5 or relative.parts[0] not in SPLITS:
        raise ValueError(f"Non-canonical TUSZ EDF path: {relative.as_posix()}")
    split, patient, _session, _montage, filename = relative.parts
    if _PATIENT_RE.fullmatch(patient) is None or not filename.endswith(".edf"):
        raise ValueError(f"Non-canonical TUSZ record identity: {relative.as_posix()}")
    return split, patient, relative.as_posix()


def _new_receipt(root: Path, deepsoz_split: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_RUNNING,
        "tusz_root": str(root),
        "deepsoz_identity_roster": str(deepsoz_split),
        "completed_relative_edf_paths": [],
        "records": [],
        "summary": {},
        "signal_contract": {
            "channels": list(STANDARD_19),
            "event_window_sec": [-12.0, 48.0],
            "sampling_frequency_hz": 200.0,
            "output_shape": [19, 12000],
            "reference": "direct_uniform_REF_then_CAR19",
            "filter": "causal_0.5-45Hz_with_30s_warmup",
        },
        "target_semantics": (
            "signal_eligibility_only_no_soz_target_no_channel_involvement_target"
        ),
        "access_receipt": {
            "deepsoz_identity_column_loaded_for_exclusion_only": True,
            "deepsoz_soz_values_loaded": False,
            "tusz_global_event_intervals_loaded_for_navigation": True,
            "tusz_per_channel_involvement_annotations_opened": False,
            "tusz_involvement_values_loaded": False,
            "model_or_pseudolabel_predictions_loaded": False,
            "private_eeg_loaded": False,
            "private_targets_loaded": False,
            "training_performed": False,
        },
    }


def _load_or_initialize(
    output: Path,
    *,
    root: Path,
    deepsoz_split: Path,
    resume: bool,
) -> dict[str, object]:
    if not output.exists():
        return _new_receipt(root, deepsoz_split)
    if not resume:
        raise FileExistsError(output)
    value = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Existing eligibility receipt has an incompatible schema")
    if value.get("status") not in {STATUS_RUNNING, STATUS_COMPLETE}:
        raise ValueError("Existing eligibility receipt has an invalid status")
    if value.get("tusz_root") != str(root) or value.get("deepsoz_identity_roster") != str(
        deepsoz_split
    ):
        raise ValueError("Resume inputs differ from the existing eligibility receipt")
    access = value.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "deepsoz_soz_values_loaded",
            "tusz_per_channel_involvement_annotations_opened",
            "tusz_involvement_values_loaded",
            "model_or_pseudolabel_predictions_loaded",
            "private_eeg_loaded",
            "private_targets_loaded",
            "training_performed",
        )
    ):
        raise ValueError("Existing eligibility receipt violates the target-free firewall")
    return value


def _summarize(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in SPLITS:
        split_records = [row for row in records if row.get("official_split") == split]
        nondeep = [row for row in split_records if row.get("excluded_deepsoz_identity") is False]
        eligible_events = [
            event
            for row in nondeep
            for event in row.get("events", [])
            if isinstance(event, Mapping) and event.get("signal_eligible") is True
        ]
        all_events = [
            event
            for row in split_records
            for event in row.get("events", [])
            if isinstance(event, Mapping)
        ]
        nondeep_events = [
            event
            for row in nondeep
            for event in row.get("events", [])
            if isinstance(event, Mapping)
        ]
        reasons = Counter(
            str(event.get("eligibility_code"))
            for event in nondeep_events
            if event.get("signal_eligible") is False
        )
        result[split] = {
            "seizure_record_count": sum(bool(row.get("events")) for row in split_records),
            "seizure_patient_count": len(
                {str(row["patient_id"]) for row in split_records if row.get("events")}
            ),
            "global_seizure_event_count": len(all_events),
            "non_deepsoz_seizure_record_count": sum(bool(row.get("events")) for row in nondeep),
            "non_deepsoz_seizure_patient_count": len(
                {str(row["patient_id"]) for row in nondeep if row.get("events")}
            ),
            "non_deepsoz_global_seizure_event_count": len(nondeep_events),
            "signal_eligible_event_count": len(eligible_events),
            "signal_eligible_patient_count": len(
                {str(event["patient_id"]) for event in eligible_events}
            ),
            "event_ineligibility_codes": dict(sorted(reasons.items())),
        }
    eligible_by_patient: dict[str, int] = defaultdict(int)
    for row in records:
        for event in row.get("events", []):
            if isinstance(event, Mapping) and event.get("signal_eligible") is True:
                eligible_by_patient[str(event["patient_id"])] += 1
    result["eligible_patient_event_counts"] = dict(sorted(eligible_by_patient.items()))
    return result


def run_audit(
    *,
    tusz_root: Path,
    deepsoz_split: Path,
    output: Path,
    resume: bool,
    limit_records: int | None = None,
) -> dict[str, object]:
    root = tusz_root.resolve(strict=True)
    split_path = deepsoz_split.resolve(strict=True)
    excluded = _load_deepsoz_local_ids(split_path)
    receipt = _load_or_initialize(
        output.absolute(), root=root, deepsoz_split=split_path, resume=resume
    )
    completed = set(str(value) for value in receipt.get("completed_relative_edf_paths", []))
    raw_records = receipt.get("records")
    if not isinstance(raw_records, list):
        raise TypeError("Eligibility receipt records must be an array")
    records: list[dict[str, object]] = [dict(row) for row in raw_records]

    edfs = tuple(
        path
        for split in SPLITS
        for path in sorted((root / split).rglob("*.edf"))
    )
    if not edfs:
        raise ValueError("TUSZ tree contains no EDF files")
    processed_now = 0
    config = CausalEDFConfig()
    for edf in edfs:
        split, patient, relative = _record_identity(root, edf)
        if relative in completed:
            continue
        if limit_records is not None and processed_now >= limit_records:
            break
        global_annotation = edf.with_suffix(".csv_bi")
        if not global_annotation.is_file():
            raise FileNotFoundError(global_annotation)
        anchors = list_tusz_global_seizure_events(global_annotation)
        is_excluded = patient in excluded
        event_rows: list[dict[str, object]] = []
        for event in anchors:
            row: dict[str, object] = {
                "event_id": f"{edf.stem}__ev{event.event_index:04d}",
                "patient_id": patient,
                "official_split": split,
                "relative_edf_path": relative,
                "global_event_index": event.event_index,
                "global_event_start_sec": float(event.start_sec),
                "global_event_stop_sec": float(event.stop_sec),
                "event_anchor_semantics": "official_global_ictal_label_start_not_channel_onset",
            }
            if is_excluded:
                row.update(
                    {
                        "signal_eligible": None,
                        "eligibility_code": "excluded_deepsoz_identity_before_signal_read",
                    }
                )
            else:
                try:
                    loaded = load_standard19_edf_event(
                        edf,
                        event.start_sec,
                        config=config,
                    )
                except EDFEventEligibilityError as error:
                    row.update(
                        {
                            "signal_eligible": False,
                            "eligibility_code": error.code,
                            "eligibility_detail": str(error),
                        }
                    )
                else:
                    if tuple(loaded.window.data.shape) != (19, 12000):
                        raise RuntimeError("Eligible S1 signal output shape changed")
                    row.update(
                        {
                            "signal_eligible": True,
                            "eligibility_code": "eligible",
                            "source_sfreq_hz": loaded.edf_receipt.source_sfreq_hz,
                            "raw_channel_names": list(loaded.edf_receipt.raw_channel_names),
                            "raw_units": list(loaded.edf_receipt.raw_units),
                            "labram_position_names": list(
                                loaded.edf_receipt.labram_position_names
                            ),
                        }
                    )
            event_rows.append(row)
        records.append(
            {
                "patient_id": patient,
                "official_split": split,
                "relative_edf_path": relative,
                "excluded_deepsoz_identity": is_excluded,
                "global_seizure_event_count": len(anchors),
                "events": event_rows,
            }
        )
        completed.add(relative)
        processed_now += 1
        receipt["records"] = records
        receipt["completed_relative_edf_paths"] = sorted(completed)
        receipt["summary"] = _summarize(records)
        _atomic_write_json(output.absolute(), receipt)

    all_done = len(completed) == len(edfs)
    receipt["status"] = STATUS_COMPLETE if all_done else STATUS_RUNNING
    receipt["discovered_edf_count"] = len(edfs)
    receipt["completed_edf_count"] = len(completed)
    receipt["summary"] = _summarize(records)
    _atomic_write_json(output.absolute(), receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--deepsoz-split", type=Path, default=DEFAULT_DEEPSOZ_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-records", type=int)
    args = parser.parse_args(argv)
    if args.limit_records is not None and args.limit_records < 1:
        parser.error("--limit-records must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run_audit(
        tusz_root=args.tusz_root,
        deepsoz_split=args.deepsoz_split,
        output=args.output,
        resume=args.resume,
        limit_records=args.limit_records,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "completed_edf_count": receipt["completed_edf_count"],
                "discovered_edf_count": receipt["discovered_edf_count"],
                "summary": receipt["summary"],
                "output": str(args.output.absolute()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
