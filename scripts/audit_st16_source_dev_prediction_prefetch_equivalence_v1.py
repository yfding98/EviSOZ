#!/usr/bin/env python3
"""Bitwise audit of serial versus bounded-parallel ST16 source-dev prediction.

The serial arm is read from already atomically published per-record artifacts.
The parallel arm replays the same first target-free source-dev records with CPU
spawn-worker preprocessing and parent-only single-GPU inference.  TERM,
source-eval, EDF annotations, and clinical text are never opened.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording import (  # noqa: E402
    st16_common17_exploratory_runner_v1 as runner,
)


DEFAULT_PROJECTION = (
    ROOT
    / "outputs/tusz_complete_detector_roster_v2_20260823/analysis_projection.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--analysis-projection", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--serial-output-dir", type=Path, required=True)
    parser.add_argument("--parallel-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--record-count", type=int, default=2)
    parser.add_argument("--preprocess-workers", type=int, default=2)
    parser.add_argument("--preprocess-prefetch", type=int, default=2)
    parser.add_argument("--live-serial-pid", type=int, default=None)
    return parser


def _pid_is_live(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _canonical_sha(value: object) -> str:
    return runner._canonical_sha256(value)


def _address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = runner._PENDING
    result["receipt_sha256"] = _canonical_sha(result)
    return result


def main() -> int:
    args = _parser().parse_args()
    if args.record_count != 2:
        raise ValueError("this admission audit is frozen to exactly two records")
    checkpoint = args.checkpoint.resolve(strict=True)
    projection_path = args.analysis_projection.resolve(strict=True)
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    rows = runner.select_target_free_prediction_rows(projection, split="source_dev")[
        : args.record_count
    ]
    checkpoint_sha = runner._file_sha256(checkpoint)
    serial_root = args.serial_output_dir.resolve(strict=True)
    serial_rows: list[dict[str, Any]] = []
    for identity_row in rows:
        identity = str(identity_row["analysis_identity_id"])
        receipt_path = serial_root / "records" / identity / "receipt.json"
        receipt = runner._reuse_prediction_receipt(
            receipt_path,
            checkpoint_sha256=checkpoint_sha,
            projection_receipt_sha256=str(projection["receipt_sha256"]),
        )
        if receipt is None or receipt.get("status") != "dense_prediction_complete":
            raise RuntimeError(f"serial dense baseline is unavailable for {identity}")
        serial_rows.append(receipt)

    live_before = _pid_is_live(args.live_serial_pid)
    if args.live_serial_pid is not None and live_before is not True:
        raise RuntimeError("declared live serial process is not alive before audit")
    parallel_root = args.parallel_output_dir.resolve(strict=False)
    if parallel_root.exists() or parallel_root.is_symlink():
        raise FileExistsError(parallel_root)
    parallel_manifest = runner.predict_source_dev_dense(
        checkpoint_path=checkpoint,
        analysis_projection_path=projection_path,
        tusz_root=args.tusz_root,
        output_dir=parallel_root,
        device_name=args.device,
        inference_batch_size=args.inference_batch_size,
        maximum_records=args.record_count,
        preprocess_workers=args.preprocess_workers,
        preprocess_prefetch=args.preprocess_prefetch,
    )
    live_after = _pid_is_live(args.live_serial_pid)
    if args.live_serial_pid is not None and live_after is not True:
        raise RuntimeError("live serial process exited during parallel audit")

    comparisons: list[dict[str, Any]] = []
    for expected_identity, serial, parallel in zip(
        [str(row["analysis_identity_id"]) for row in rows],
        serial_rows,
        parallel_manifest["prediction_rows"],
        strict=True,
    ):
        if (
            serial["analysis_identity_id"] != expected_identity
            or parallel["analysis_identity_id"] != expected_identity
            or parallel.get("status") != "dense_prediction_complete"
        ):
            raise RuntimeError("serial/parallel prediction roster or status drifted")
        serial_path = Path(str(serial["dense_probability_path"])).resolve(strict=True)
        parallel_path = Path(str(parallel["dense_probability_path"])).resolve(
            strict=True
        )
        serial_array = np.load(serial_path, mmap_mode="r", allow_pickle=False)
        parallel_array = np.load(parallel_path, mmap_mode="r", allow_pickle=False)
        shape_equal = serial_array.shape == parallel_array.shape
        dtype_equal = serial_array.dtype == parallel_array.dtype == np.dtype("float32")
        array_bitwise_equal = bool(
            shape_equal
            and dtype_equal
            and np.array_equal(serial_array, parallel_array, equal_nan=True)
        )
        serial_npy_sha = runner._file_sha256(serial_path)
        parallel_npy_sha = runner._file_sha256(parallel_path)
        row = {
            "analysis_identity_id": expected_identity,
            "recording_id": serial["recording_id"],
            "sample_count": int(serial_array.shape[0]),
            "serial_dense_npy_sha256": serial_npy_sha,
            "parallel_dense_npy_sha256": parallel_npy_sha,
            "npy_file_bitwise_equal": serial_npy_sha == parallel_npy_sha,
            "float32_array_bitwise_equal": array_bitwise_equal,
            "shape_equal": shape_equal,
            "dtype_equal": dtype_equal,
            "transform_receipt_sha256_equal": serial.get(
                "transform_receipt_sha256"
            )
            == parallel.get("transform_receipt_sha256"),
            "OLA_result_receipt_sha256_equal": serial.get(
                "OLA_result_receipt_sha256"
            )
            == parallel.get("OLA_result_receipt_sha256"),
            "OLA_plan_receipt_sha256_equal": serial.get("OLA_plan_receipt_sha256")
            == parallel.get("OLA_plan_receipt_sha256"),
            "parallel_preprocess_worker_pid": parallel.get(
                "preprocess_worker_pid"
            ),
            "parallel_GPU_inference_parent_pid": parallel.get(
                "GPU_model_inference_process_pid"
            ),
            "parallel_preprocess_worker_CUDA_initialized": parallel.get(
                "preprocess_worker_CUDA_initialized"
            ),
        }
        serial_array._mmap.close()
        parallel_array._mmap.close()
        if not all(
            row[key]
            for key in (
                "npy_file_bitwise_equal",
                "float32_array_bitwise_equal",
                "shape_equal",
                "dtype_equal",
                "transform_receipt_sha256_equal",
                "OLA_result_receipt_sha256_equal",
                "OLA_plan_receipt_sha256_equal",
            )
        ):
            raise RuntimeError(f"bitwise equivalence failed for {expected_identity}")
        if row["parallel_preprocess_worker_CUDA_initialized"] is not False:
            raise RuntimeError("parallel CPU preprocessing worker initialized CUDA")
        comparisons.append(row)

    audit = _address(
        {
            "schema_version": (
                "st16_common17_source_dev_prediction_prefetch_equivalence_v1"
            ),
            "status": "passed",
            "claim_status": "engineering_equivalence_only_nonpromotable",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "analysis_projection_path": str(projection_path),
            "analysis_projection_receipt_sha256": projection["receipt_sha256"],
            "serial_output_dir": str(serial_root),
            "parallel_output_dir": str(parallel_root),
            "record_count": len(comparisons),
            "comparisons": comparisons,
            "all_dense_npy_and_float32_arrays_bitwise_equal": True,
            "all_transform_and_OLA_receipts_equal": True,
            "serial_live_process_pid": args.live_serial_pid,
            "serial_live_before_parallel_audit": live_before,
            "serial_live_after_parallel_audit": live_after,
            "live_serial_process_was_signalled_or_stopped": False,
            "preprocess_workers": args.preprocess_workers,
            "preprocess_prefetch": args.preprocess_prefetch,
            "maximum_inflight_preprocess_futures_observed": parallel_manifest[
                "maximum_inflight_preprocess_futures_observed"
            ],
            "prediction_roster_order_preserved": parallel_manifest[
                "prediction_roster_order_preserved"
            ],
            "intention_to_evaluate_denominator_preserved": parallel_manifest[
                "intention_to_evaluate_denominator_preserved"
            ],
            "parent_only_GPU_model_inference": parallel_manifest[
                "parent_only_GPU_model_inference"
            ],
            "source_dev_only": True,
            "source_eval_opened": False,
            "TERM_reference_annotation_or_target_opened": False,
            "EDF_annotation_doctor_text_or_clinical_context_opened": False,
            "switch_authorization": (
                "equivalence_admitted_but_speedup_must_be_measured_before_restart"
            ),
            "receipt_sha256": runner._PENDING,
        }
    )
    runner._write_json_atomic(args.output, audit, replace=False)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
