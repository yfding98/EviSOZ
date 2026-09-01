#!/usr/bin/env python3
"""Materialize a compact provisional DeepSOZ navigation roster.

The input is the already materialized, reference-free source-train posterior
batch.  A source-development decoder candidate is frozen before any
source-train reference is accepted, then every complete posterior timeline is
decoded and projected to compact interval rows.  Dense timelines are never
copied.

This command intentionally does *not* build or bless the strict G0a A1 roster.
The published fold checkpoints lack a complete training-run/preprocess/usage
exposure receipt and the selected diagnostic decoder failed the preregistered
high-recall gate.  Every output therefore keeps strict patient-OOF G0a,
operating-point qualification, model training, promotion, clinical use, and
SOZ/Findings fact use disabled.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_detection import (  # noqa: E402
    decode_continuous_seizure_posterior,
)
from src.clinical_eeg_long_recording.deepsoz_posterior_batch_validation import (  # noqa: E402
    DEEPSOZ_MATERIALIZER_CODE_SHA256,
    revalidate_deepsoz_posterior_batch_without_references,
    validate_deepsoz_posterior_batch_without_references,
)
from src.clinical_eeg_long_recording.deepsoz_reference_free_batch_validation_artifact import (  # noqa: E402
    load_deepsoz_identity_roster_binding,
)


_DIAGNOSTIC_POLICY_CANDIDATE_ID = "CONTCAND-f8ac73068e76398c22b2"
_DIAGNOSTIC_POLICY_SHA256 = (
    "f8ac73068e76398c22b237d7a1f717cba7a0ee9833fd582ec797b86b01d3799b"
)
_ARTIFACT_FILENAMES = {
    "decoder_freeze": "decoder_freeze.json",
    "records": "records.jsonl",
    "candidates": "candidates.jsonl",
    "bundle_receipt": "bundle_receipt.json",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) for row in rows)


def _load_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must contain one object")
    return value, payload


def _safe_posterior_path(root: Path, value: object) -> str:
    text = str(value)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "posteriors"
        or path.suffix != ".json"
        or path.as_posix() != text
    ):
        raise ValueError("posterior relative path is unsafe")
    physical = root.joinpath(*path.parts)
    if physical.is_symlink() or not physical.is_file():
        raise ValueError("posterior relative path is not a regular file")
    physical.resolve(strict=True).relative_to(root)
    return text


def _load_index(root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    path = root / "posterior_index.jsonl"
    if path.is_symlink() or not path.is_file():
        raise ValueError("posterior index must be a regular non-symlink file")
    payload = path.read_bytes()
    rows: dict[str, dict[str, Any]] = {}
    for ordinal, raw in enumerate(payload.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("posterior index contains invalid JSON") from error
        if type(row) is not dict or row.get("ordinal") != ordinal:
            raise ValueError("posterior index ordinal/schema drifted")
        recording_id = str(row.get("recording_id"))
        if recording_id in rows:
            raise ValueError("posterior index repeats a recording")
        _safe_posterior_path(root, row.get("posterior_relative_path"))
        rows[recording_id] = row
    if not rows:
        raise ValueError("posterior index is empty")
    return rows, hashlib.sha256(payload).hexdigest()


def _freeze_diagnostic_policy(
    calibration: Mapping[str, Any], *, calibration_file_sha256: str
) -> dict[str, Any]:
    if (
        calibration.get("provider_id") != "deepsoz_temporal_oof_candidate_v1"
        or calibration.get("calibration_split") != "source_dev"
        or calibration.get("constraint_status")
        != "not_met_no_operating_point_frozen"
        or calibration.get("selected_operating_point") is not None
    ):
        raise ValueError("DeepSOZ source-dev calibration status drifted")
    candidates = calibration.get("candidate_results")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("DeepSOZ calibration has no decoder candidates")
    selected = next(
        (
            row
            for row in candidates
            if row.get("candidate_id") == _DIAGNOSTIC_POLICY_CANDIDATE_ID
        ),
        None,
    )
    if selected is None:
        raise ValueError("registered diagnostic decoder candidate disappeared")
    maximum_sensitivity = max(
        float(row["pooled_metrics"]["event_sensitivity"]) for row in candidates
    )
    sensitivity = float(selected["pooled_metrics"]["event_sensitivity"])
    if (
        selected.get("decoder_policy_sha256") != _DIAGNOSTIC_POLICY_SHA256
        or _canonical_sha256(selected.get("decoder_policy"))
        != _DIAGNOSTIC_POLICY_SHA256
        or selected.get("high_recall_constraints_met") is not False
        or abs(sensitivity - maximum_sensitivity) > 1e-12
    ):
        raise ValueError("diagnostic decoder identity/rationale drifted")
    body: dict[str, Any] = {
        "schema_version": "deepsoz_provisional_navigation_decoder_freeze_v1",
        "method_id": "SOURCE_DEV_MAX_POOLED_RECALL_DIAGNOSTIC_FREEZE_V1",
        "provider_id": "deepsoz_temporal_oof_candidate_v1",
        "candidate_id": _DIAGNOSTIC_POLICY_CANDIDATE_ID,
        "decoder_policy": deepcopy(selected["decoder_policy"]),
        "decoder_policy_sha256": _DIAGNOSTIC_POLICY_SHA256,
        "source_calibration_receipt_id": calibration["calibration_receipt_id"],
        "source_calibration_content_sha256": _canonical_sha256(calibration),
        "source_calibration_file_sha256": calibration_file_sha256,
        "source_dev_observed_metrics": {
            "event_sensitivity": sensitivity,
            "patient_macro_event_sensitivity": selected["patient_macro_metrics"][
                "event_sensitivity_macro"
            ],
            "false_alarms_per_24h": selected["pooled_metrics"][
                "alarm_false_alarms_per_24h"
            ],
            "onset_hit_rate_within_5s_reference_denominator": selected[
                "pooled_metrics"
            ]["onset_absolute_hit_rate"]["5s"]["rate"],
        },
        "selection_semantics": (
            "highest_observed_pooled_event_sensitivity_within_the_"
            "preregistered_source_dev_grid_for_engineering_distribution_only"
        ),
        "source_dev_reference_metrics_used_for_diagnostic_choice": True,
        "source_train_reference_available_to_decoder": False,
        "source_train_reference_opened": False,
        "operating_point_qualified": False,
        "high_recall_constraints_met": False,
        "selected_operating_point_in_source_calibration": None,
        "engineering_navigation_only": True,
        "strict_patient_oof_g0a_verified": False,
        "checkpoint_exposure_pending": True,
        "model_training_authorized": False,
        "promotion_authorized": False,
        "clinical_or_production_use_authorized": False,
        "receipt_sha256": "",
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return body


def _candidate_row(
    *,
    recording: object,
    index: Mapping[str, Any],
    proposal: Mapping[str, Any],
    decoder_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    material = {
        "schema_version": "deepsoz_provisional_navigation_candidate_v1",
        "provider_id": "deepsoz_temporal_oof_candidate_v1",
        "patient_id": recording.patient_id,
        "recording_id": recording.recording_id,
        "source_signal_sha256": recording.canonical_source_signal_sha256,
        "posterior_artifact_id": recording.posterior_artifact_id,
        "posterior_file_sha256": recording.posterior_file_sha256,
        "decoder_freeze_receipt_sha256": decoder_freeze["receipt_sha256"],
        "start_offset_seconds": float(proposal["start_offset_seconds"]),
        "stop_offset_seconds": float(proposal["stop_offset_seconds"]),
        "anchor_offset_seconds": float(proposal["anchor_offset_seconds"]),
        "peak_probability_navigation_score": float(proposal["peak_probability"]),
        "mean_probability_navigation_score": float(proposal["mean_probability"]),
        "right_censored": bool(proposal["right_censored"]),
        "decision_available_offset_seconds": float(recording.duration_seconds),
        "decision_availability_semantics": (
            "offline_after_complete_record_capture_preprocessing_and_all_"
            "held_out_fold_inference"
        ),
        "held_out_fold_indices": list(index["held_out_fold_indices"]),
        "offline_future_dependent": True,
        "candidate_is_confirmed_seizure_or_onset": False,
        "candidate_score_may_support_findings_or_soz_fact": False,
        "strict_patient_oof_g0a_verified": False,
        "operating_point_qualified": False,
        "candidate_id": "",
        "candidate_receipt_sha256": "",
    }
    id_source = deepcopy(material)
    id_source["candidate_id"] = "DEEPSOZ-NAV-CANDIDATE-PENDING"
    id_source["candidate_receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    material["candidate_id"] = "DSZNAV-" + _canonical_sha256(id_source)[:24]
    receipt_source = deepcopy(material)
    receipt_source["candidate_receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    material["candidate_receipt_sha256"] = _canonical_sha256(receipt_source)
    return material


def _materialize(
    *,
    posterior_batch_root: Path,
    split_roster_receipt: Path,
    reference_free_validation_receipt: Path,
    calibration_receipt: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = posterior_batch_root.resolve(strict=True)
    if posterior_batch_root.is_symlink() or not root.is_dir():
        raise ValueError("posterior batch root must be a regular directory")
    binding = load_deepsoz_identity_roster_binding(
        split_roster_receipt,
        selected_split="source_train",
    )
    sealed = validate_deepsoz_posterior_batch_without_references(
        root,
        expected_split="source_train",
        expected_manifest_sha256=binding.source_manifest_file_sha256,
        expected_recording_ids=binding.recording_ids,
        expected_patient_ids=binding.patient_ids,
        expected_materializer_code_sha256=DEEPSOZ_MATERIALIZER_CODE_SHA256,
        require_complete_inventory=True,
    )
    sealed = revalidate_deepsoz_posterior_batch_without_references(sealed)
    observed_validation, validation_bytes = _load_json(
        reference_free_validation_receipt,
        "reference-free validation receipt",
    )
    if sealed.validation_receipt() != observed_validation:
        raise ValueError("posterior batch no longer replays its frozen validation")
    validation_file_sha256 = hashlib.sha256(validation_bytes).hexdigest()

    calibration, calibration_bytes = _load_json(
        calibration_receipt,
        "source-dev calibration receipt",
    )
    decoder_freeze = _freeze_diagnostic_policy(
        calibration,
        calibration_file_sha256=hashlib.sha256(calibration_bytes).hexdigest(),
    )
    decoder_freeze["source_posterior_validation_receipt_sha256"] = (
        observed_validation["receipt_sha256"]
    )
    decoder_freeze["source_posterior_validation_file_sha256"] = (
        validation_file_sha256
    )
    decoder_freeze["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in decoder_freeze.items() if key != "receipt_sha256"}
    )
    index_by_record, index_sha256 = _load_index(root)
    if index_sha256 != observed_validation["posterior_index_file_sha256"]:
        raise ValueError("posterior index differs from frozen validation")

    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for recording in sealed.recordings:
        index = index_by_record.get(recording.recording_id)
        if index is None:
            raise ValueError("sealed recording is absent from posterior index")
        decoded = decode_continuous_seizure_posterior(
            recording_id=recording.recording_id,
            source_signal_sha256=recording.canonical_source_signal_sha256,
            recording_duration_seconds=recording.duration_seconds,
            provider_receipt=recording.provider_receipt(),
            posterior_timeline=recording.posterior_timeline(),
            policy=decoder_freeze["decoder_policy"],
        )
        current_candidates = [
            _candidate_row(
                recording=recording,
                index=index,
                proposal=proposal,
                decoder_freeze=decoder_freeze,
            )
            for proposal in decoded["event_proposals"]
        ]
        candidates.extend(current_candidates)
        record: dict[str, Any] = {
            "schema_version": "deepsoz_provisional_navigation_record_v1",
            "patient_id": recording.patient_id,
            "recording_id": recording.recording_id,
            "recording_duration_seconds": float(recording.duration_seconds),
            "source_signal_sha256": recording.canonical_source_signal_sha256,
            "held_out_fold_indices": list(index["held_out_fold_indices"]),
            "posterior_artifact_id": recording.posterior_artifact_id,
            "posterior_relative_path": _safe_posterior_path(
                root, index["posterior_relative_path"]
            ),
            "posterior_file_sha256": recording.posterior_file_sha256,
            "posterior_record_binding_sha256": recording.record_binding_sha256,
            "timeline_window_count": int(index["timeline_window_count"]),
            "partial_tail_present": bool(index["partial_tail_present"]),
            "detector_imputed_channel_count": int(
                index["detector_imputed_channel_count"]
            ),
            "outcome": (
                "completed_with_candidates"
                if current_candidates
                else "completed_zero_candidate"
            ),
            "candidate_count": len(current_candidates),
            "candidate_ids": [row["candidate_id"] for row in current_candidates],
            "complete_recording_scanned": True,
            "offline_future_dependent": True,
            "source_train_reference_opened": False,
            "strict_patient_oof_g0a_verified": False,
            "operating_point_qualified": False,
            "checkpoint_exposure_pending": True,
            "geometry_matched_background_pending": True,
            "support_lineage_pending": True,
        }
        records.append(record)

    records.sort(key=lambda row: (row["patient_id"], row["recording_id"]))
    candidates.sort(
        key=lambda row: (
            row["patient_id"],
            row["recording_id"],
            row["start_offset_seconds"],
            row["candidate_id"],
        )
    )
    if len(records) != len(index_by_record) or len(
        {row["recording_id"] for row in records}
    ) != len(records):
        raise ValueError("provisional record denominator drifted")

    return decoder_freeze, records, candidates


def _artifact_descriptor(payload: bytes, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "size_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if rows is not None:
        result["row_count"] = rows
    return result


def _build_payloads(
    *,
    posterior_batch_root: Path,
    split_roster_receipt: Path,
    reference_free_validation_receipt: Path,
    calibration_receipt: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    decoder, records, candidates = _materialize(
        posterior_batch_root=posterior_batch_root,
        split_roster_receipt=split_roster_receipt,
        reference_free_validation_receipt=reference_free_validation_receipt,
        calibration_receipt=calibration_receipt,
    )
    payloads = {
        "decoder_freeze": _canonical_bytes(decoder),
        "records": _jsonl_bytes(records),
        "candidates": _jsonl_bytes(candidates),
    }
    record_seconds = sum(row["recording_duration_seconds"] for row in records)
    receipt: dict[str, Any] = {
        "schema_version": "deepsoz_provisional_navigation_roster_bundle_v1",
        "method_id": "REFERENCE_FREE_POSTERIOR_TO_COMPACT_DIAGNOSTIC_INTERVALS_V1",
        "provider_id": "deepsoz_temporal_oof_candidate_v1",
        "model_split": "source_train",
        "materializer_code_sha256": _file_sha(Path(__file__))[1],
        "decoder_freeze_receipt_sha256": decoder["receipt_sha256"],
        "counts": {
            "patients": len({row["patient_id"] for row in records}),
            "records": len(records),
            "recording_seconds": record_seconds,
            "completed_with_candidates": sum(
                row["outcome"] == "completed_with_candidates" for row in records
            ),
            "completed_zero_candidate": sum(
                row["outcome"] == "completed_zero_candidate" for row in records
            ),
            "partial_coverage": 0,
            "technical_failure": 0,
            "detector_candidates": len(candidates),
            "partial_tail_records": sum(
                row["partial_tail_present"] for row in records
            ),
            "detector_imputed_records": sum(
                row["detector_imputed_channel_count"] > 0 for row in records
            ),
            "detector_imputed_channels": sum(
                row["detector_imputed_channel_count"] for row in records
            ),
        },
        "artifacts": {
            _ARTIFACT_FILENAMES[key]: _artifact_descriptor(
                payload,
                rows=(len(records) if key == "records" else len(candidates))
                if key in {"records", "candidates"}
                else None,
            )
            for key, payload in payloads.items()
        },
        "lineage_status": {
            "published_patient_held_out_fold_predictions": True,
            "strict_checkpoint_training_exposure_verified": False,
            "strict_patient_oof_g0a_verified": False,
            "operating_point_qualified": False,
            "checkpoint_exposure_pending": True,
            "geometry_matched_background_pending": True,
            "stable_origin_registry_pending": True,
            "reference_free_support_lineage_pending": True,
            "postfreeze_reference_join_performed": False,
        },
        "scope_receipt": {
            "dense_posterior_timelines_copied": False,
            "source_train_reference_path_parameter_accepted": False,
            "source_train_reference_opened": False,
            "edf_annotations_opened": False,
            "excel_or_clinical_text_opened": False,
            "doctor_labels_or_reports_opened": False,
            "source_eval_opened": False,
            "offline_future_dependent_navigation_only": True,
            "candidate_score_available_to_findings_or_soz_positive_evidence": False,
            "model_training_authorized": False,
            "primary_promotion_authorized": False,
            "clinical_or_production_use_authorized": False,
        },
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    payloads["bundle_receipt"] = _canonical_bytes(receipt)
    return payloads, receipt


def _write_no_clobber(output: Path, payloads: Mapping[str, bytes]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("provisional navigation output exists; no-clobber refused")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    for key, filename in _ARTIFACT_FILENAMES.items():
        with (output / filename).open("xb") as handle:
            handle.write(payloads[key])


def _verify_existing(output: Path, payloads: Mapping[str, bytes]) -> None:
    if not output.is_dir() or output.is_symlink():
        raise ValueError("--verify-existing requires a regular output directory")
    if {path.name for path in output.iterdir()} != set(_ARTIFACT_FILENAMES.values()):
        raise ValueError("provisional navigation output inventory drifted")
    for key, filename in _ARTIFACT_FILENAMES.items():
        path = output / filename
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payloads[key]:
            raise ValueError(f"provisional navigation artifact {filename} does not replay")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior-batch-root", type=Path, required=True)
    parser.add_argument("--split-roster-receipt", type=Path, required=True)
    parser.add_argument("--reference-free-validation-receipt", type=Path, required=True)
    parser.add_argument("--source-dev-calibration-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    payloads, receipt = _build_payloads(
        posterior_batch_root=args.posterior_batch_root,
        split_roster_receipt=args.split_roster_receipt,
        reference_free_validation_receipt=args.reference_free_validation_receipt,
        calibration_receipt=args.source_dev_calibration_receipt,
    )
    output = args.output.resolve()
    if args.verify_existing:
        _verify_existing(output, payloads)
    else:
        _write_no_clobber(output, payloads)
    print(
        json.dumps(
            {
                "output": str(output),
                "verified_existing": bool(args.verify_existing),
                "receipt_sha256": receipt["receipt_sha256"],
                "counts": receipt["counts"],
                "lineage_status": receipt["lineage_status"],
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
