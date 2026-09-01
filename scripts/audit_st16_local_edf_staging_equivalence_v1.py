#!/usr/bin/env python3
"""Prove that local SSD EDF staging does not change the ST16 transform."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording import (  # noqa: E402
    seizuretransformer_cleanroom_registry_v1 as st,
)
from src.clinical_eeg_long_recording.st16_common17_exploratory_runner_v1 import (  # noqa: E402
    _transform_st16_record,
)


DEFAULT_SOURCE = Path(
    "/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf/train/aaaaakqu/"
    "s002_2010/03_tcp_ar_a/aaaaakqu_s002_t004.edf"
)
DEFAULT_STAGE_ROOT = Path("/tmp/clinical_eeg_st16_local_stage")
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/st16_common17_local_edf_staging_equivalence_v1_20260825/receipt.json"
)
PENDING = "CONTENT-ADDRESS-PENDING"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    stage_candidate = args.stage_root
    if stage_candidate.is_symlink():
        raise PermissionError("stage root may not be a symlink")
    stage_root = stage_candidate.resolve(strict=True)
    if not stage_root.is_dir():
        raise NotADirectoryError(stage_root)
    registry = st.load_registry(ROOT / st.CONFIG_RELATIVE_PATH)
    environment_name = "CLINICAL_EEG_ST16_LOCAL_STAGE_ROOT"
    previous = os.environ.pop(environment_name, None)
    try:
        direct = _transform_st16_record(source, registry)
        os.environ[environment_name] = str(stage_root)
        staged = _transform_st16_record(source, registry)
    finally:
        if previous is None:
            os.environ.pop(environment_name, None)
        else:
            os.environ[environment_name] = previous
    direct_signal = np.asarray(direct.signal)
    staged_signal = np.asarray(staged.signal)
    direct_sha = hashlib.sha256(direct_signal.tobytes(order="C")).hexdigest()
    staged_sha = hashlib.sha256(staged_signal.tobytes(order="C")).hexdigest()
    result = {
        "schema_version": "st16_local_edf_staging_equivalence_v1",
        "status": "pass_bitwise_identical" if direct_sha == staged_sha else "fail",
        "source": {
            "path": str(source),
            "file_sha256": _sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "stage_root": str(stage_root),
        "variant_id": st.ST16_VARIANT_ID,
        "shape": list(direct_signal.shape),
        "dtype": str(direct_signal.dtype),
        "direct_signal_sha256": direct_sha,
        "staged_signal_sha256": staged_sha,
        "bitwise_array_equal": bool(np.array_equal(direct_signal, staged_signal)),
        "transform_receipt_equal": direct.receipt == staged.receipt,
        "source_eval_opened": False,
        "receipt_sha256": PENDING,
    }
    if not (
        result["bitwise_array_equal"]
        and result["transform_receipt_equal"]
        and direct_signal.shape == staged_signal.shape
        and direct_signal.dtype == staged_signal.dtype == np.dtype("float32")
    ):
        raise RuntimeError("local EDF staging changed the frozen ST16 transform")
    result["receipt_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(result) + b"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
