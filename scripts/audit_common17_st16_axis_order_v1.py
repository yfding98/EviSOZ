#!/usr/bin/env python3
"""Materialize the repaired ST16 named-axis drift and numeric replay receipt."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.common17_continuous_detector_benchmark_v3 import (
    CANONICAL_ST16_PAIRS as BENCHMARK_CANONICAL_PAIRS,
)
from src.clinical_eeg_long_recording.seizuretransformer_cleanroom_registry_v1 import (
    ST16_TYPED_UNITS as REGISTRY_ST16_TYPED_UNITS,
)
from src.clinical_eeg_long_recording.st16_common17_axis_contract_v1 import (
    CANONICAL_ST16_PAIRS,
    CANONICAL_ST16_TYPED_UNITS,
    COMMON17_REFERENTIAL_AXIS_ORDER,
    derive_st16_lb16_by_name,
)


PENDING = "CONTENT-ADDRESS-PENDING"
OLD_BENCHMARK_ORDER = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
)
OLD_TOP_ARCHITECTURE_ORDER = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"expected object: {path}")
    return value


def _derive_in_order(raw: np.ndarray, units: tuple[str, ...]) -> np.ndarray:
    index = {name: i for i, name in enumerate(COMMON17_REFERENTIAL_AXIS_ORDER)}
    return np.stack(
        [raw[index[left]] - raw[index[right]] for left, right in (unit.split("-", 1) for unit in units)],
        axis=0,
    )


def build_receipt() -> dict:
    benchmark_path = ROOT / "configs/common17_continuous_detector_benchmark_v3.json"
    architecture_path = ROOT / "configs/clinical_eeg_top_tier_method_architecture_v3.json"
    benchmark = _load(benchmark_path)
    architecture = _load(architecture_path)
    view = next(row for row in benchmark["channel_contract"]["allowed_primary_provider_views"] if row["view_id"] == "C17-LB16")
    benchmark_units = tuple("-".join(pair) for pair in view["pairs"])
    architecture_units = tuple(architecture["common17"]["seizuretransformer_induced_bipolar16"])
    if not (
        benchmark_units
        == architecture_units
        == tuple(REGISTRY_ST16_TYPED_UNITS)
        == CANONICAL_ST16_TYPED_UNITS
        and tuple(BENCHMARK_CANONICAL_PAIRS) == CANONICAL_ST16_PAIRS
    ):
        raise ValueError("active ST16 axis declarations are not exactly identical")

    samples = 257
    time = np.arange(samples, dtype=np.float64)
    raw = np.stack(
        [(axis + 1) * 1000.0 + time * (axis + 3) for axis in range(17)], axis=0
    )
    canonical = derive_st16_lb16_by_name(
        raw, electrode_order=COMMON17_REFERENTIAL_AXIS_ORDER
    )
    permutation = tuple(reversed(range(17)))
    permuted = derive_st16_lb16_by_name(
        raw[list(permutation)],
        electrode_order=[COMMON17_REFERENTIAL_AXIS_ORDER[i] for i in permutation],
    )
    if not np.array_equal(canonical, permuted):
        raise AssertionError("named-axis replay changed under referential permutation")
    replay_rows = []
    for name, old_units in (
        ("old_benchmark_v3", OLD_BENCHMARK_ORDER),
        ("old_top_architecture_v3", OLD_TOP_ARCHITECTURE_ORDER),
    ):
        old = _derive_in_order(raw, old_units)
        by_name = {unit: old[i] for i, unit in enumerate(old_units)}
        reordered = np.stack([by_name[unit] for unit in CANONICAL_ST16_TYPED_UNITS])
        replay_rows.append(
            {
                "declaration": name,
                "old_order": list(old_units),
                "old_order_sha256": _sha(old_units),
                "positionally_equal_to_canonical": bool(np.array_equal(old, canonical)),
                "lead_name_reordered_exactly_equal_to_canonical": bool(np.array_equal(reordered, canonical)),
                "canonical_to_old_position_permutation": [old_units.index(unit) for unit in CANONICAL_ST16_TYPED_UNITS],
            }
        )
    if any(row["positionally_equal_to_canonical"] for row in replay_rows):
        raise AssertionError("historical drift unexpectedly equals canonical positions")
    if not all(row["lead_name_reordered_exactly_equal_to_canonical"] for row in replay_rows):
        raise AssertionError("historical lead-name replay is not numerically exact")
    receipt = {
        "schema_version": "common17_st16_axis_order_drift_audit_v1",
        "status": "pass_active_contracts_repaired_and_named_numeric_replay_exact",
        "canonical_common17_referential_axis_order": list(COMMON17_REFERENTIAL_AXIS_ORDER),
        "canonical_st16_typed_units": list(CANONICAL_ST16_TYPED_UNITS),
        "canonical_order_derivation": "upstream_ST18_order_delete_only_FZ-CZ_and_CZ-PZ",
        "polarity": "first_named_referential_axis_minus_second_named_referential_axis",
        "active_contract_bindings": [
            {"path": str(benchmark_path.relative_to(ROOT)), "file_sha256": _file_sha(benchmark_path)},
            {"path": str(architecture_path.relative_to(ROOT)), "file_sha256": _file_sha(architecture_path)},
        ],
        "historical_drift_replay": replay_rows,
        "synthetic_raw_shape": list(raw.shape),
        "canonical_lb16_shape": list(canonical.shape),
        "canonical_payload_sha256": hashlib.sha256(np.ascontiguousarray(canonical, dtype="<f8").tobytes()).hexdigest(),
        "referential_axis_permutation_named_replay_exact": True,
        "FZ_or_PZ_axis_present": False,
        "zero_fill_interpolation_or_imputation_used": False,
        "receipt_sha256": PENDING,
    }
    receipt["receipt_sha256"] = _sha(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_receipt()
    target = args.output.resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(target), "receipt_sha256": value["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
