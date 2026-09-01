#!/usr/bin/env python3
"""Materialize and independently replay real-EDF S01 native QC Findings.

The script opens only the signal-facing canonical EDF API.  It never opens
EDF annotations, patient/recording header text, spreadsheets, doctor labels
or reports, clinical text, video/behaviour, sleep/activation data,
ECG/EMG/EOG or an LLM.  The first invocation writes append-only canonical
JSON.  A second invocation with ``--verify-existing`` independently reopens
the EDF, recomputes every sample denominator and requires exact payload
identity before writing a replay receipt.

This is a software/A0 replay smoke, not clinical validation and not evidence
that any clinical artefact classifier has been qualified.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.canonical_edf_materialization import (  # noqa: E402
    CanonicalEDFConfig,
    load_canonical_edf_views,
)
from src.clinical_eeg_long_recording.event_native_signal_quality_findings_v1 import (  # noqa: E402
    materialize_event_native_signal_quality_findings_v1,
    validate_event_native_signal_quality_findings_v1,
)


_ARTIFACT_FILES = {
    "s01_findings": "s01_event_native_signal_quality_findings.json",
    "smoke_receipt": "smoke_receipt.json",
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _self_hash(value: Mapping[str, object]) -> str:
    body = deepcopy(dict(value))
    body.pop("receipt_sha256", None)
    return _sha(body)


def _file_sha(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8", errors="strict"))
    canonical = _canonical_bytes(value)
    if type(value) is not dict or raw not in {canonical, canonical[:-1]}:
        raise ValueError(
            f"{path.name} is not canonical JSON with an optional final newline"
        )
    return value


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value))


def _materialize(
    *,
    edf: Path,
    event_id: str | None,
    requested_start: float | None,
    requested_stop: float | None,
) -> dict[str, dict[str, Any]]:
    bundle = load_canonical_edf_views(edf, config=CanonicalEDFConfig())
    canonical = bundle.canonical_record.canonical_receipt
    duration = float(canonical["recording_duration_seconds"])
    start = 0.0 if requested_start is None else float(requested_start)
    stop = duration if requested_stop is None else float(requested_stop)
    resolved_event_id = (
        event_id
        if event_id is not None
        else f"S01-REAL-{canonical['source_signal_sha256'][:24]}"
    )
    findings = materialize_event_native_signal_quality_findings_v1(
        event_id=resolved_event_id,
        bundle=bundle,
        requested_analysis_interval_seconds=(start, stop),
    )
    size, container_sha = _file_sha(edf)
    denominators = findings["event_denominators"]
    smoke: dict[str, Any] = {
        "schema_version": "clinical_eeg_real_findings_s01_smoke_v1",
        "method_id": "REAL-EDF-S01-NATIVE-QC-EXACT-REPLAY-SMOKE-V1",
        "event_id": resolved_event_id,
        "recording_id": canonical["recording_id"],
        "canonical_signal_id": canonical["canonical_signal_id"],
        "canonical_receipt_sha256": canonical["receipt_sha256"],
        "source_signal_sha256": canonical["source_signal_sha256"],
        "source_container": {
            "size_bytes": size,
            "sha256": container_sha,
            "container_bytes_hashed_without_semantic_header_or_annotation_parse": True,
        },
        "requested_analysis_interval_seconds": [start, stop],
        "s01_findings_receipt_sha256": findings["receipt_sha256"],
        "typed_unit_count": denominators["typed_unit_count"],
        "whole_bipolar_lead_unit_count": denominators["whole_bipolar_lead_unit_count"],
        "view_opportunity_count": denominators["view_opportunity_count"],
        "not_evaluable_native_typed_unit_count": denominators[
            "not_evaluable_native_typed_unit_count"
        ],
        "native_usable_fraction": denominators["native_usable_fraction"],
        "input_firewall": {
            "real_eeg_samples_used": True,
            "allowlisted_edf_signal_header_used": True,
            "edf_patient_or_recording_header_opened": False,
            "edf_annotations_opened": False,
            "spreadsheets_opened": False,
            "doctor_labels_or_reports_opened": False,
            "clinical_text_opened": False,
            "video_or_behavior_opened": False,
            "sleep_or_activation_labels_opened": False,
            "ecg_emg_eog_opened": False,
            "qwen_or_other_llm_used": False,
        },
        "authorization": {
            "software_and_exact_replay_smoke_only": True,
            "clinical_artifact_classification_authorized": False,
            "negative_clinical_assertion_authorized": False,
            "onset_or_soz_support_authorized": False,
            "clinical_or_production_use_authorized": False,
            "performance_claim_authorized": False,
            "report_eligible_term_allowlist": [],
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    smoke["receipt_sha256"] = _self_hash(smoke)
    return {
        "s01_findings": validate_event_native_signal_quality_findings_v1(findings),
        "smoke_receipt": smoke,
    }


def _verify_existing(
    output: Path,
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    observed = {
        key: _read_json(output / filename) for key, filename in _ARTIFACT_FILES.items()
    }
    if observed != expected:
        raise ValueError("real S01 artifacts do not replay exactly from reopened EDF")
    verification: dict[str, Any] = {
        "schema_version": "clinical_eeg_real_findings_s01_replay_v1",
        "method_id": "SEPARATE-PROCESS-REAL-EDF-S01-EXACT-REPLAY-V1",
        "source_smoke_receipt_sha256": expected["smoke_receipt"]["receipt_sha256"],
        "s01_findings_receipt_sha256": expected["s01_findings"]["receipt_sha256"],
        "separate_process_recomputation": True,
        "edf_reopened": True,
        "native_samples_and_signal_header_remeasured": True,
        "all_payloads_exact_match": True,
        "annotations_or_external_labels_used": False,
        "performance_or_clinical_claim_authorized": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    verification["receipt_sha256"] = _self_hash(verification)
    _write_json_no_clobber(output / "replay_verification.json", verification)
    print(json.dumps(verification, sort_keys=True, ensure_ascii=False))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-id")
    parser.add_argument("--requested-start", type=float)
    parser.add_argument("--requested-stop", type=float)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    edf = args.edf.resolve(strict=True)
    if not edf.is_file() or edf.is_symlink() or edf.suffix.lower() != ".edf":
        raise ValueError("--edf must be a non-symlink regular EDF file")
    artifacts = _materialize(
        edf=edf,
        event_id=args.event_id,
        requested_start=args.requested_start,
        requested_stop=args.requested_stop,
    )
    output = args.output.resolve()
    if args.verify_existing:
        if not output.is_dir() or output.is_symlink():
            raise ValueError("--verify-existing requires an existing regular output")
        _verify_existing(output, artifacts)
        return
    output.mkdir(parents=True, exist_ok=False)
    for key, filename in _ARTIFACT_FILES.items():
        _write_json_no_clobber(output / filename, artifacts[key])
    summary = {
        "output": str(output),
        "receipt_sha256": artifacts["smoke_receipt"]["receipt_sha256"],
        "s01_findings_receipt_sha256": artifacts["s01_findings"]["receipt_sha256"],
        "typed_unit_count": artifacts["smoke_receipt"]["typed_unit_count"],
        "separate_process_replay_pending": True,
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
