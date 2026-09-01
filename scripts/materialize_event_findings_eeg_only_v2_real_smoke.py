#!/usr/bin/env python3
"""Materialize two EEG-only v2 Findings profiles with real waveform evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.event_findings_eeg_only_v2_materializer import (  # noqa: E402
    load_verified_source_rollout_receipt,
    materialize_common17_waveform_artifacts,
    materialize_event_findings_eeg_only_v2,
)


DEFAULT_SOURCES = (
    ROOT
    / "outputs"
    / "tusz_real_edf_adaptive_findings_v1_20260825"
    / "events"
    / "REAL-EDF-MEDIUM-NON256-SAME-START-01"
    / "receipt.json",
    ROOT
    / "outputs"
    / "tusz_real_edf_adaptive_findings_v1_20260825"
    / "events"
    / "REAL-EDF-SHORT-MISSING-MIDLINE-START-03"
    / "receipt.json",
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "clinical_eeg_event_findings_eeg_only_v2_real_smoke_v1_20260825"
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source_path in arguments.source_receipt:
        source_path = source_path.resolve(strict=True)
        source = load_verified_source_rollout_receipt(source_path)
        record_id = str(source["event_findings_evidence"]["recording_id"])
        source_suffix = str(source["receipt_sha256"])[:16]
        event_root = output / "records" / record_id / source_suffix
        waveform = materialize_common17_waveform_artifacts(
            source_receipt=source,
            source_receipt_path=source_path,
            tusz_root=arguments.tusz_root,
            output_directory=event_root / "waveform",
        )
        waveform_manifest_path = event_root / "waveform" / "manifest.json"
        profile = materialize_event_findings_eeg_only_v2(
            source_receipt=source,
            waveform_manifest=waveform,
            waveform_manifest_path=waveform_manifest_path,
        )
        profile_path = event_root / "event_findings_eeg_only_v2.json"
        _atomic_json(profile_path, profile)
        rows.append(
            {
                "record_id": record_id,
                "event_id": profile["event_id"],
                "source_receipt_path": str(source_path),
                "source_receipt_file_sha256": _file_sha256(source_path),
                "source_receipt_content_sha256": source["receipt_sha256"],
                "waveform_manifest_path": str(waveform_manifest_path),
                "waveform_manifest_receipt_sha256": waveform["receipt_sha256"],
                "raw_npz_sha256": waveform["artifacts"]["raw_npz"]["sha256"],
                "display_png_sha256": waveform["artifacts"]["display_png"]["sha256"],
                "profile_path": str(profile_path),
                "profile_file_sha256": _file_sha256(profile_path),
                "positive_observation_count": sum(
                    row["evidence_level"] != "not_evaluable"
                    for row in profile["observations"]
                ),
                "not_evaluable_observation_count": sum(
                    row["evidence_level"] == "not_evaluable"
                    for row in profile["observations"]
                ),
                "source_status": source["event_findings_evidence"]["status"],
            }
        )
    body: dict[str, Any] = {
        "schema_version": "clinical_eeg_event_findings_eeg_only_v2_real_smoke_receipt_v1",
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "record_count": len(rows),
        "records": rows,
        "scope_receipt": {
            "direct_common17_EEG_samples_used": True,
            "edf_annotations_used": False,
            "term_or_szcore_sidecar_opened_at_runtime": False,
            "soz_or_channel_labels_used": False,
            "spreadsheet_used": False,
            "doctor_or_clinical_text_used": False,
            "patient_header_fields_used": False,
            "video_or_behavior_used": False,
            "sleep_or_provocation_used": False,
            "ecg_emg_eog_used": False,
            "llm_used": False,
        },
        "claim_limits": {
            "engineering_real_EDF_smoke_only": True,
            "detector_performance_measured": False,
            "SOZ_accuracy_measured": False,
            "clinical_term_qualification_claimed": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    _atomic_json(output / "receipt.json", body)
    return body


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--source-receipt",
        type=Path,
        action="append",
        default=None,
        help="Verified adaptive event receipt; repeat for multiple events.",
    )
    value.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.source_receipt is None:
        arguments.source_receipt = list(DEFAULT_SOURCES)
    result = run(arguments)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
