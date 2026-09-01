#!/usr/bin/env python3
"""Read-only closure audit for cached common17 EventNet source-dev outputs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


PENDING = "CONTENT-ADDRESS-PENDING"
LEGACY_PREDICTION_SCHEMA = "eventnet_common17_dev_prediction_global_posterior_runtime_v3"
REPLAYABLE_PREDICTION_SCHEMA = (
    "eventnet_common17_dev_prediction_replayable_pre_nms_runtime_v4"
)
PRE_NMS_CACHE_SCHEMA = "eventnet_common17_global_pre_nms_candidate_cache_v1"
ACCEPTED_PREDICTION_SCHEMAS = frozenset(
    {LEGACY_PREDICTION_SCHEMA, REPLAYABLE_PREDICTION_SCHEMA}
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = PENDING
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def validate_metrics_content_address(metrics: Mapping[str, Any]) -> bool:
    claimed = metrics.get("receipt_sha256")
    replay = deepcopy(dict(metrics))
    replay["receipt_sha256"] = PENDING
    return isinstance(claimed, str) and claimed == canonical_sha256(replay)


def replayable_cache_contract_is_valid(payload: Mapping[str, Any]) -> bool:
    """Validate the v4 pre-NMS cache without requiring dense posteriors."""

    cache = payload.get("pre_nms_candidate_cache")
    if not isinstance(cache, dict):
        return False
    count = cache.get("candidate_count")
    if type(count) is not int or count < 0:
        return False
    vector_fields = (
        "center_sample",
        "center_probability",
        "duration_fraction",
        "left_valley_probability",
        "right_valley_probability",
    )
    if any(
        not isinstance(cache.get(field), list) or len(cache[field]) != count
        for field in vector_fields
    ):
        return False
    samples = cache["center_sample"]
    if any(type(value) is not int or value < 0 for value in samples):
        return False
    if samples != sorted(samples) or len(samples) != len(set(samples)):
        return False
    numeric_fields = vector_fields[1:]
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for field in numeric_fields
        for value in cache[field]
    ):
        return False
    return (
        cache.get("schema_version") == PRE_NMS_CACHE_SCHEMA
        and cache.get("stage")
        == "full_record_smoothed_center_posterior_before_distance_nms"
        and cache.get("target_fs_hz") == 256
        and cache.get("minimum_peak_threshold")
        == payload.get("minimum_peak_threshold")
        and cache.get("smoothing_sigma_samples")
        == payload.get("smoothing_sigma_samples")
        and cache.get("replay_scope")
        == "threshold_and_minimum_distance_nms_and_adjacent_valley_deblending"
        and cache.get("dense_posterior_preserved") is False
        and cache.get("reference_or_annotation_used") is False
    )


def payload_schema_contract_is_valid(payload: Mapping[str, Any]) -> bool:
    schema = payload.get("schema_version")
    if schema == LEGACY_PREDICTION_SCHEMA:
        return "pre_nms_candidate_cache" not in payload
    if schema == REPLAYABLE_PREDICTION_SCHEMA:
        return replayable_cache_contract_is_valid(payload)
    return False


def run(manifest_path: Path, evaluation_dir: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    metrics_path = evaluation_dir.resolve(strict=True) / "metrics.json"
    prediction_dir = evaluation_dir.resolve(strict=True) / "predictions"
    metrics = read_json(metrics_path)
    if not prediction_dir.is_dir():
        raise NotADirectoryError(prediction_dir)

    records = [row for row in manifest["records"] if row["model_split"] == "source_dev"]
    expected = {str(row["analysis_identity_id"]): row for row in records}
    if len(expected) != len(records):
        raise RuntimeError("manifest contains duplicate source-dev identities")
    channel_order = tuple(manifest["channel_contract"]["common17_channel_order"])
    checkpoint_sha = str(metrics["checkpoint_file_sha256"])
    checkpoint_step = int(metrics["checkpoint_global_step"])

    files = sorted(prediction_dir.glob("*.json.gz"))
    payloads: dict[str, dict[str, Any]] = {}
    bad_files: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []
    for path in files:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise TypeError("prediction root is not an object")
        except Exception as error:  # audit must retain every parse failure
            bad_files.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
            continue
        identity = str(payload.get("analysis_identity_id", ""))
        if identity in payloads:
            mismatches.append({"identity": identity, "reason": "duplicate_payload_identity"})
            continue
        payloads[identity] = payload
        row = expected.get(identity)
        checks = {
            "identity_not_in_manifest": row is None,
            "schema": not payload_schema_contract_is_valid(payload),
            "patient": row is not None and payload.get("patient_id") != row.get("patient_id"),
            "checkpoint_sha": payload.get("checkpoint_file_sha256") != checkpoint_sha,
            "checkpoint_step": payload.get("checkpoint_global_step") != checkpoint_step,
            "channel_order": tuple(payload.get("common17_channel_order", ())) != channel_order,
            "fz_pz_axis": payload.get("FZ_or_PZ_model_axis_present") is not False,
            "minimum_threshold": payload.get("minimum_peak_threshold") != 0.001,
            "smoothing_sigma": payload.get("smoothing_sigma_samples") != 100,
            "minimum_peak_distance": payload.get("minimum_peak_distance_seconds") != 60,
        }
        runtime = payload.get("runtime")
        checks["runtime"] = not isinstance(runtime, dict) or any(
            not isinstance(runtime.get(key), (int, float))
            or not math.isfinite(float(runtime[key]))
            or float(runtime[key]) < 0
            for key in (
                "model_inference_seconds",
                "EEG_IO_and_resample_seconds",
                "end_to_end_pipeline_seconds",
            )
        )
        for name, failed in checks.items():
            if failed:
                mismatches.append({"identity": identity, "reason": name})

    missing = sorted(set(expected) - set(payloads))
    extra = sorted(set(payloads) - set(expected))
    temporary = sorted(str(path) for path in prediction_dir.rglob("*.tmp*"))

    observed_schemas = sorted(
        {
            str(payload.get("schema_version"))
            for payload in payloads.values()
            if payload.get("schema_version") is not None
        }
    )
    invariant_checks = {
        "manifest_expected_count": len(expected) == 1821,
        "prediction_file_count": len(files) == len(expected),
        "all_gzip_json_parse": not bad_files,
        "identity_roster_exact": not missing and not extra and len(payloads) == len(expected),
        "payload_contract_exact": not mismatches,
        "no_active_temporary_files": not temporary,
        "metrics_manifest_binding": metrics.get("manifest_receipt_sha256") == manifest.get("receipt_sha256"),
        "metrics_checkpoint_binding": metrics.get("checkpoint_file_sha256") == checkpoint_sha,
        "metrics_channel_binding": tuple(metrics.get("common17_channel_order", ())) == channel_order,
        "metrics_complete_denominator": metrics.get("complete_source_dev_denominator") is True,
        "metrics_counts": (
            metrics.get("recording_count") == 1821
            and metrics.get("reference_event_count") == 1074
        ),
        "metrics_content_address": validate_metrics_content_address(metrics),
    }
    status = "pass_complete_active_prediction_roster" if all(invariant_checks.values()) else "fail"
    receipt = content_address(
        {
            "schema_version": "eventnet_common17_source_dev_prediction_audit_v2",
            "status": status,
            "inference_performed": False,
            "active_prediction_inventory": {
                "expected_manifest_identities": len(expected),
                "prediction_gzip_files": len(files),
                "parsed_unique_identities": len(payloads),
                "missing_identity_count": len(missing),
                "extra_identity_count": len(extra),
                "bad_file_count": len(bad_files),
                "payload_mismatch_count": len(mismatches),
                "active_temporary_file_count": len(temporary),
            },
            "frozen_contract": {
                "accepted_prediction_schemas": sorted(ACCEPTED_PREDICTION_SCHEMAS),
                "observed_prediction_schemas": observed_schemas,
                "pre_nms_cache_schema": PRE_NMS_CACHE_SCHEMA,
                "checkpoint_global_step": checkpoint_step,
                "checkpoint_file_sha256": checkpoint_sha,
                "common17_channel_order": list(channel_order),
                "minimum_peak_threshold": 0.001,
                "smoothing_sigma_samples": 100,
                "minimum_peak_distance_seconds": 60,
                "FZ_or_PZ_model_axis_present": False,
            },
            "metrics": {
                "receipt_sha256": metrics["receipt_sha256"],
                "file_sha256": file_sha256(metrics_path),
                "recording_count": metrics["recording_count"],
                "recording_hours": metrics["recording_hours"],
                "reference_event_count": metrics["reference_event_count"],
                "complete_source_dev_denominator": metrics["complete_source_dev_denominator"],
                "threshold_selection_status": metrics["threshold_selection_status"],
            },
            "invariant_checks": invariant_checks,
            "failures": {
                "missing_identities": missing,
                "extra_identities": extra,
                "bad_files": bad_files,
                "payload_mismatches": mismatches,
                "active_temporary_files": temporary,
            },
            "quarantine_policy": {
                "evaluation_sibling_quarantine_exists": (evaluation_dir / "quarantine_concurrent_writer_20260824").is_dir(),
                "quarantine_is_outside_active_prediction_inventory": True,
            },
            "lineage": {
                "manifest": {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path.resolve())},
                "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
                "prediction_directory": str(prediction_dir),
                "script": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
            },
            "receipt_sha256": PENDING,
        }
    )
    if status != "pass_complete_active_prediction_roster":
        raise RuntimeError(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.manifest, args.evaluation_dir)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
