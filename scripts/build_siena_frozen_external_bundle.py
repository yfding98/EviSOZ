#!/usr/bin/env python3
"""Split the audited Siena release into signal-only and weak-target ledgers.

The signal roster contains no lobe or laterality values and is the only file
that downstream EEG materialization may open.  Patient-level weak phenotype
labels are written once per patient to a physically separate ledger.  This
builder never reads EEG samples and never runs, trains, or selects a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCHEMA = "siena_external_weak_region_audit_v1"
BUNDLE_SCHEMA = "siena_frozen_external_bundle_v1"
DEFAULT_AUDIT = (
    ROOT / "outputs/siena_external_weak_region_audit_v1_20260814.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/siena_frozen_external_bundle_v1_20260815"
PROTOCOL = (
    ROOT
    / "research/02_method/siena_frozen_external_weak_region_protocol_20260815_zh.md"
)

EXPECTED_COUNTS = {
    "patients": 14,
    "edf_records": 41,
    "parsed_seizures": 47,
    "strict_external_ready_seizures": 44,
    "complete_unique_standard19_edfs": 41,
}
SIGNAL_FIELDS = (
    "event_id",
    "patient_id",
    "source_event_key",
    "relative_edf_path",
    "global_event_t0_sec",
    "edf_duration_sec",
    "source_sampling_rate_hz",
    "time_support_preeligible",
    "strict_external_ready",
    "exclusion_reason_codes",
    "filename_correction_count",
    "reference_policy",
)
TARGET_FIELDS = (
    "patient_id",
    "weak_localization",
    "weak_lateralization",
    "label_granularity",
    "label_role",
    "declared_seizure_count",
    "ready_event_count",
)
FORBIDDEN_SIGNAL_FIELDS = frozenset(
    {
        "weak_localization",
        "weak_lateralization",
        "soz",
        "soz_channels",
        "candidate_positive_electrodes",
        "standard19_positive_electrodes",
    }
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError("Siena audit schema mismatch")
    return value


def _write_csv(
    path: Path,
    fields: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    names = tuple(fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})


def _safe_relative_edf(value: object) -> str:
    relative = PurePosixPath(str(value).strip())
    if (
        not str(value).strip()
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix.lower() != ".edf"
    ):
        raise ValueError(f"unsafe Siena EDF path: {value!r}")
    return str(relative)


def _as_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _as_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def project_ledgers(
    audit: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Project an already validated audit without opening any EEG file."""

    if audit.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError("Siena audit schema mismatch")
    subjects_raw = _as_list(audit.get("subjects"), name="subjects")
    events_by_subject = _as_mapping(
        audit.get("events_by_subject"), name="events_by_subject"
    )
    headers_raw = _as_list(audit.get("edf_headers"), name="edf_headers")
    if not subjects_raw or not headers_raw:
        raise ValueError("Siena audit subjects/headers are empty")

    subjects = [_as_mapping(value, name="subject") for value in subjects_raw]
    headers = [_as_mapping(value, name="edf_header") for value in headers_raw]
    source_ids = sorted(str(row.get("patient_id", "")).strip() for row in subjects)
    if "" in source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("Siena patient identities are empty or duplicated")
    patient_ids = {
        source_id: f"SIENA-P{ordinal:03d}"
        for ordinal, source_id in enumerate(source_ids, start=1)
    }

    header_by_relative: dict[str, Mapping[str, object]] = {}
    for row in headers:
        relative = _safe_relative_edf(row.get("relative_edf"))
        if relative in header_by_relative:
            raise ValueError(f"duplicate Siena EDF header: {relative}")
        if row.get("standard19_complete_unique") is not True:
            raise ValueError(f"Siena EDF lacks unique standard-19: {relative}")
        sfreq = float(row.get("sampling_rate_hz", math.nan))
        duration = float(row.get("duration_sec", math.nan))
        if sfreq != 512.0 or not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"Siena EDF sampling/duration drifted: {relative}")
        header_by_relative[relative] = row

    subject_by_id = {
        str(row.get("patient_id", "")).strip(): row for row in subjects
    }
    if set(events_by_subject) != set(source_ids):
        raise ValueError("Siena subject/event patient sets do not match")

    signal_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    ready_by_patient: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    correction_count = 0
    event_ordinal = 0

    for source_id in source_ids:
        raw_events = _as_list(
            events_by_subject[source_id], name=f"events_by_subject[{source_id}]"
        )
        events = [_as_mapping(value, name="event") for value in raw_events]
        seizure_numbers = [int(value.get("seizure_number", -1)) for value in events]
        if len(seizure_numbers) != len(set(seizure_numbers)) or any(
            value < 1 for value in seizure_numbers
        ):
            raise ValueError(f"Siena seizure numbers invalid for {source_id}")
        for event in sorted(events, key=lambda value: int(value["seizure_number"])):
            event_ordinal += 1
            file_candidates = _as_list(
                event.get("file_candidates"), name="file_candidates"
            )
            if len(file_candidates) != 1:
                raise ValueError("Siena event EDF must resolve uniquely")
            relative = _safe_relative_edf(f"{source_id}/{file_candidates[0]}")
            if relative not in header_by_relative:
                raise ValueError(f"Siena event EDF lacks audited header: {relative}")
            header = header_by_relative[relative]
            strict_ready = event.get("strict_external_ready") is True
            reason_codes_raw = _as_list(
                event.get("strict_external_reason_codes"),
                name="strict_external_reason_codes",
            )
            reason_codes = tuple(sorted(set(str(value) for value in reason_codes_raw)))
            if strict_ready and reason_codes:
                raise ValueError("ready Siena event carries exclusion reasons")
            if not strict_ready and not reason_codes:
                raise ValueError("non-ready Siena event lacks exclusion reason")
            onset_raw = event.get("onset_sec_from_record_start")
            onset = float(onset_raw) if onset_raw is not None else math.nan
            duration = float(header["duration_sec"])
            crop_ready = bool(
                strict_ready
                and math.isfinite(onset)
                and onset >= 12.0
                and onset + 48.0 <= duration
            )
            if strict_ready and not crop_ready:
                raise ValueError("strict Siena event lacks exact 60-second crop")
            for reason in reason_codes:
                exclusion_counts[reason] += 1
            corrections = _as_list(
                event.get("filename_corrections"), name="filename_corrections"
            )
            correction_count += int(bool(corrections))
            patient_id = patient_ids[source_id]
            ready_by_patient[patient_id] += int(crop_ready)
            signal_rows.append(
                {
                    "event_id": f"SIENA-E{event_ordinal:03d}",
                    "patient_id": patient_id,
                    "source_event_key": (
                        f"{source_id}-SZ{int(event['seizure_number']):03d}"
                    ),
                    "relative_edf_path": relative,
                    "global_event_t0_sec": f"{onset:.9f}" if crop_ready else "",
                    "edf_duration_sec": f"{duration:.9f}",
                    "source_sampling_rate_hz": "512.000000000",
                    "time_support_preeligible": int(crop_ready),
                    "strict_external_ready": int(strict_ready),
                    "exclusion_reason_codes": ";".join(reason_codes),
                    "filename_correction_count": len(corrections),
                    "reference_policy": "unlabeled_common_car19",
                }
            )

        subject = subject_by_id[source_id]
        localization = str(subject.get("weak_localization", "")).strip()
        lateralization = str(subject.get("weak_lateralization", "")).strip()
        if localization not in {"T", "F"} or lateralization not in {
            "L",
            "R",
            "Bilateral",
        }:
            raise ValueError(f"unexpected Siena weak phenotype for {source_id}")
        target_rows.append(
            {
                "patient_id": patient_ids[source_id],
                "weak_localization": localization,
                "weak_lateralization": lateralization,
                "label_granularity": "patient",
                "label_role": "weak_external_phenotype_not_soz",
                "declared_seizure_count": int(subject["declared_seizure_count"]),
                "ready_event_count": ready_by_patient[patient_ids[source_id]],
            }
        )

    if FORBIDDEN_SIGNAL_FIELDS & set(SIGNAL_FIELDS):
        raise RuntimeError("Siena signal roster schema contains target fields")
    if len({row["event_id"] for row in signal_rows}) != len(signal_rows):
        raise RuntimeError("Siena event pseudonyms are duplicated")
    if len({row["patient_id"] for row in target_rows}) != len(target_rows):
        raise RuntimeError("Siena target ledger patients are duplicated")
    if {row["patient_id"] for row in signal_rows} != {
        row["patient_id"] for row in target_rows
    }:
        raise RuntimeError("Siena signal/target patient sets do not align")

    summary: dict[str, object] = {
        "patient_count": len(target_rows),
        "event_count": len(signal_rows),
        "time_support_preeligible": sum(
            int(row["time_support_preeligible"]) for row in signal_rows
        ),
        "time_support_not_preeligible": sum(
            not bool(row["time_support_preeligible"]) for row in signal_rows
        ),
        "patients_with_ready_event": sum(
            int(row["ready_event_count"]) > 0 for row in target_rows
        ),
        "filename_corrected_events": correction_count,
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "weak_localization_distribution": dict(
            sorted(Counter(str(row["weak_localization"]) for row in target_rows).items())
        ),
        "weak_lateralization_distribution": dict(
            sorted(
                Counter(str(row["weak_lateralization"]) for row in target_rows).items()
            )
        ),
    }
    return signal_rows, target_rows, summary


def build(audit_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    audit = _read_json(audit_path.resolve(strict=True))
    counts = _as_mapping(audit.get("counts"), name="counts")
    for key, expected in EXPECTED_COUNTS.items():
        if int(counts.get(key, -1)) != expected:
            raise ValueError(f"official Siena count drifted for {key}")
    signal_rows, target_rows, summary = project_ledgers(audit)
    if (
        summary["patient_count"] != 14
        or summary["event_count"] != 47
        or summary["time_support_preeligible"] != 44
        or summary["patients_with_ready_event"] != 14
        or summary["filename_corrected_events"] != 6
        or summary["weak_localization_distribution"] != {"F": 1, "T": 13}
        or summary["weak_lateralization_distribution"]
        != {"Bilateral": 1, "L": 9, "R": 4}
    ):
        raise ValueError("official Siena projected ledger counts drifted")

    dataset_root = Path(str(audit.get("root", ""))).resolve(strict=True)
    for row in signal_rows:
        source = dataset_root.joinpath(*PurePosixPath(str(row["relative_edf_path"])).parts)
        resolved = source.resolve(strict=True)
        resolved.relative_to(dataset_root)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)

    output.mkdir(parents=True)
    _write_csv(output / "signal_roster.csv", SIGNAL_FIELDS, signal_rows)
    _write_csv(output / "weak_patient_target_ledger.csv", TARGET_FIELDS, target_rows)
    manifest: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "source_audit": str(audit_path),
        "dataset_root": str(dataset_root),
        "summary": summary,
        "files": {
            "target_excluding_signal_roster": "signal_roster.csv",
            "weak_patient_target_ledger": "weak_patient_target_ledger.csv",
        },
        "frozen_policy": {
            "input_window_sec": [-12, 48],
            "target_sampling_rate_hz": 200,
            "reference_policy": "unlabeled_common_car19",
            "prediction_unit": "event",
            "evaluation_unit": "patient",
            "event_aggregation": "equal_event_mean",
            "weak_labels_are_soz": False,
            "not_ready_action": "retain_in_coverage_and_exclude_from_signal",
        },
        "access_receipt": {
            "weak_label_values_loaded_for_ledger_projection": True,
            "c18_soz_target_values_loaded": False,
            "eeg_samples_loaded": False,
            "model_predictions_loaded": False,
            "private_data_loaded": False,
            "training_performed": False,
            "calibration_performed": False,
            "model_or_threshold_selection_performed": False,
            "llm_annotation_performed": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.audit, args.output)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
