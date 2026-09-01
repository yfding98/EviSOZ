#!/usr/bin/env python3
"""Real-EDF, signal-only numeric replay of serial versus parallel ST16 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording import (  # noqa: E402
    seizuretransformer_cleanroom_registry_v1 as st,
)
from src.clinical_eeg_long_recording.st16_common17_exploratory_runner_v1 import (  # noqa: E402
    PROVIDER_ID,
    TARGET_FS_HZ,
    TILE_SAMPLES,
    _canonical_sha256,
    _content_address,
    _file_sha256,
    _tile_id,
    _validate_content_address,
    _write_json_atomic,
    materialize_epoch_tile_cache,
)


DEFAULT_MANIFEST = (
    ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json"
)
DEFAULT_STAGE_ROOT = Path("/tmp/clinical_eeg_st16_local_stage")
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/st16_epoch_cache_serial_parallel_equivalence_v1_20260825"
    / "receipt.json"
)
RECORD_PATHS = (
    "train/aaaaantp/s001_2012/01_tcp_ar/aaaaantp_s001_t000.edf",
    "train/aaaaapnl/s006_2013/01_tcp_ar/aaaaapnl_s006_t002.edf",
)
PENDING = "CONTENT-ADDRESS-PENDING"


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("ST16 audit manifest must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("ST16 audit manifest must be a JSON object")
    return _validate_content_address(value, artifact_name="common17 manifest")


def _signal_only_plan(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_path = {
        str(row["edf_relative_path"]): dict(row)
        for row in manifest["records"]
        if row["model_split"] == "source_train"
    }
    selected: list[dict[str, Any]] = []
    catalog: dict[str, dict[str, Any]] = {}
    for relative_path in RECORD_PATHS:
        row = by_path.get(relative_path)
        if row is None:
            raise FileNotFoundError(f"frozen source-train audit EDF missing: {relative_path}")
        if int(row["target_sample_count_256hz"]) < TILE_SAMPLES:
            raise ValueError("frozen ST16 audit EDF has less than one complete tile")
        identity = str(row["analysis_identity_id"])
        tile_id = _tile_id(identity, 0)
        catalog[tile_id] = {
            "tile_id": tile_id,
            "analysis_identity_id": identity,
            "patient_id": str(row["patient_id"]),
            "edf_relative_path": relative_path,
            "target_start_sample": 0,
            "target_stop_sample_exclusive": TILE_SAMPLES,
            "positive_tile": None,
        }
        selected.append(
            {
                "analysis_identity_id": identity,
                "patient_id": str(row["patient_id"]),
                "edf_relative_path": relative_path,
                "canonical_source_tensor_sha256": str(
                    row["canonical_source_tensor_sha256"]
                ),
                "target_sample_count_256hz": int(
                    row["target_sample_count_256hz"]
                ),
            }
        )
    plan = _content_address(
        {
            "schema_version": "st16_epoch_cache_equivalence_signal_only_plan_v1",
            "provider_id": PROVIDER_ID,
            "claim_status": "real_source_train_signal_only_engineering_audit",
            "variant_id": st.ST16_VARIANT_ID,
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "target_clock_hz": TARGET_FS_HZ,
            "tile_samples": TILE_SAMPLES,
            "selected_tile_catalog": catalog,
            "source_train_signal_bindings": selected,
            "source_train_TERM_targets_used": False,
            "source_dev_or_source_eval_EEG_opened": False,
            "receipt_sha256": PENDING,
        }
    )
    return plan, selected


def _raw_array_sha256(path: Path) -> str:
    value = np.load(path, allow_pickle=False)
    if value.shape != (len(st.ST16_TYPED_UNITS), TILE_SAMPLES):
        raise ValueError("ST16 equivalence tile shape drifted")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _sidecar(path: Path) -> dict[str, Any]:
    value = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return _validate_content_address(value, artifact_name="ST16 cache tile sidecar")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest.resolve(strict=True))
    plan, signal_bindings = _signal_only_plan(manifest)
    tusz_root = Path(manifest["source_bindings"]["tusz_root"]).resolve(strict=True)
    stage_root = args.stage_root.resolve(strict=True)
    if stage_root.is_symlink() or not stage_root.is_dir():
        raise PermissionError("ST16 audit stage root must be a regular directory")

    worker_environment = "CLINICAL_EEG_ST16_CACHE_WORKERS"
    stage_environment = "CLINICAL_EEG_ST16_LOCAL_STAGE_ROOT"
    previous_worker = os.environ.get(worker_environment)
    previous_stage = os.environ.get(stage_environment)
    try:
        os.environ[stage_environment] = str(stage_root)
        with tempfile.TemporaryDirectory(
            prefix="st16_cache_equivalence_", dir=stage_root
        ) as scratch_text:
            scratch = Path(scratch_text)
            serial_cache = scratch / "serial"
            parallel_cache = scratch / "parallel"
            serial_cache.mkdir()
            parallel_cache.mkdir()
            os.environ[worker_environment] = "1"
            serial_paths, serial_receipt = materialize_epoch_tile_cache(
                plan, tusz_root=tusz_root, cache_root=serial_cache
            )
            os.environ[worker_environment] = "2"
            parallel_paths, parallel_receipt = materialize_epoch_tile_cache(
                plan, tusz_root=tusz_root, cache_root=parallel_cache
            )
            rows: list[dict[str, Any]] = []
            for tile_id in sorted(plan["selected_tile_catalog"]):
                serial_path = serial_paths[tile_id]
                parallel_path = parallel_paths[tile_id]
                serial_sidecar = _sidecar(serial_path)
                parallel_sidecar = _sidecar(parallel_path)
                row = {
                    "tile_id": tile_id,
                    "analysis_identity_id": plan["selected_tile_catalog"][tile_id][
                        "analysis_identity_id"
                    ],
                    "serial_npy_file_sha256": _file_sha256(serial_path),
                    "parallel_npy_file_sha256": _file_sha256(parallel_path),
                    "serial_raw_array_sha256": _raw_array_sha256(serial_path),
                    "parallel_raw_array_sha256": _raw_array_sha256(parallel_path),
                    "serial_tile_receipt_sha256": serial_sidecar[
                        "receipt_sha256"
                    ],
                    "parallel_tile_receipt_sha256": parallel_sidecar[
                        "receipt_sha256"
                    ],
                    "bitwise_array_equal": bool(
                        np.array_equal(
                            np.load(serial_path, allow_pickle=False),
                            np.load(parallel_path, allow_pickle=False),
                        )
                    ),
                }
                if not (
                    row["serial_npy_file_sha256"]
                    == row["parallel_npy_file_sha256"]
                    and row["serial_raw_array_sha256"]
                    == row["parallel_raw_array_sha256"]
                    and row["serial_tile_receipt_sha256"]
                    == row["parallel_tile_receipt_sha256"]
                    and row["bitwise_array_equal"]
                ):
                    raise RuntimeError("serial/parallel ST16 cache numeric drift")
                rows.append(row)
            os.environ[worker_environment] = "1"
            _, serial_reuse = materialize_epoch_tile_cache(
                plan, tusz_root=tusz_root, cache_root=serial_cache
            )
            os.environ[worker_environment] = "2"
            _, parallel_reuse = materialize_epoch_tile_cache(
                plan, tusz_root=tusz_root, cache_root=parallel_cache
            )
    finally:
        if previous_worker is None:
            os.environ.pop(worker_environment, None)
        else:
            os.environ[worker_environment] = previous_worker
        if previous_stage is None:
            os.environ.pop(stage_environment, None)
        else:
            os.environ[stage_environment] = previous_stage

    if not (
        serial_receipt["unique_record_transform_count"] == 2
        and parallel_receipt["unique_record_transform_count"] == 2
        and serial_receipt["cache_worker_count"] == 1
        and parallel_receipt["cache_worker_count"] == 2
        and serial_reuse["unique_record_transform_count"] == 0
        and parallel_reuse["unique_record_transform_count"] == 0
        and serial_reuse["reused_content_verified_tile_count"] == 2
        and parallel_reuse["reused_content_verified_tile_count"] == 2
    ):
        raise RuntimeError("ST16 cache serial/parallel or reuse ledger drifted")
    source_files = []
    for binding in signal_bindings:
        source = (tusz_root / binding["edf_relative_path"]).resolve(strict=True)
        source_files.append(
            {
                **binding,
                "edf_file_sha256": _file_sha256(source),
                "edf_size_bytes": source.stat().st_size,
            }
        )
    result = _content_address(
        {
            "schema_version": "st16_epoch_cache_serial_parallel_equivalence_v1",
            "status": "pass_bitwise_identical_real_source_train_tiles",
            "claim_status": "engineering_equivalence_not_detector_performance",
            "provider_id": PROVIDER_ID,
            "variant_id": st.ST16_VARIANT_ID,
            "signal_only_audit_plan_receipt_sha256": plan["receipt_sha256"],
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "source_train_signal_bindings": source_files,
            "source_train_TERM_targets_used": False,
            "source_dev_or_source_eval_EEG_opened": False,
            "serial_cache_worker_count": 1,
            "parallel_cache_worker_count": 2,
            "serial_cache_contract_receipt_sha256": serial_receipt[
                "cache_contract_receipt_sha256"
            ],
            "parallel_cache_contract_receipt_sha256": parallel_receipt[
                "cache_contract_receipt_sha256"
            ],
            "serial_parallel_contract_equal": (
                serial_receipt["cache_contract_receipt_sha256"]
                == parallel_receipt["cache_contract_receipt_sha256"]
            ),
            "tile_results": rows,
            "all_npy_files_bitwise_equal": True,
            "all_raw_float32_arrays_bitwise_equal": True,
            "all_content_addressed_tile_receipts_equal": True,
            "serial_reuse_transform_count": serial_reuse[
                "unique_record_transform_count"
            ],
            "parallel_reuse_transform_count": parallel_reuse[
                "unique_record_transform_count"
            ],
            "serial_reused_content_verified_tile_count": serial_reuse[
                "reused_content_verified_tile_count"
            ],
            "parallel_reused_content_verified_tile_count": parallel_reuse[
                "reused_content_verified_tile_count"
            ],
            "temporary_cache_deleted_after_audit": True,
            "source_eval_opened": False,
            "receipt_sha256": PENDING,
        }
    )
    if result["serial_parallel_contract_equal"] is not True:
        raise RuntimeError("serial/parallel ST16 cache contract drifted")
    output = args.output.resolve(strict=False)
    if output.is_file() and not output.is_symlink():
        existing = _validate_content_address(
            json.loads(output.read_text(encoding="utf-8")),
            artifact_name="existing serial/parallel ST16 cache audit",
        )
        if existing != result:
            raise FileExistsError("existing ST16 cache audit differs from replay")
    else:
        _write_json_atomic(output, result, replace=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
