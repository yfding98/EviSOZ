#!/usr/bin/env python3
"""One-time, fail-closed adoption of the authorized live ST16 epoch-0 cache.

This command is intentionally not a general migration utility.  It accepts
only the exact epoch-plan and local-staging equivalence receipts frozen for the
markerless cache created on 2026-08-25.  It must be run only after the training
producer has stopped unexpectedly.  A normally completed epoch deletes the
temporary cache and needs no adoption.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording import (  # noqa: E402
    seizuretransformer_cleanroom_registry_v1 as st,
)
from src.clinical_eeg_long_recording.st16_common17_exploratory_runner_v1 import (  # noqa: E402
    CACHE_CONTRACT_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    LEGACY_LOCAL_STAGING_EQUIVALENCE_RECEIPT_SHA256,
    LEGACY_MARKERLESS_ADOPTION_PLAN_SHA256,
    PROVIDER_ID,
    TILE_SAMPLES,
    _cache_contract_core,
    _cache_tile_sidecar,
    _canonical_sha256,
    _content_address,
    _file_sha256,
    _validate_content_address,
    _write_json_atomic,
    build_exploratory_epoch_plan,
)


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/st16_common17_exploratory_source_train_epoch1_v1_20260825"
)
DEFAULT_CACHE = (
    DEFAULT_OUTPUT_DIR
    / "temporary_epoch_cache"
    / "epoch_0000_c72jhei8"
)
DEFAULT_MANIFEST = (
    ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json"
)
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "last.pt"
DEFAULT_STAGING_RECEIPT = (
    ROOT
    / "outputs/st16_common17_local_edf_staging_equivalence_v1_20260825"
    / "receipt.json"
)
DEFAULT_DURABLE_RECEIPT = (
    ROOT
    / "outputs/st16_markerless_epoch_cache_adoption_v1_20260825"
    / "receipt.json"
)
PENDING = "CONTENT-ADDRESS-PENDING"


def _running_train_producers(output_dir: Path) -> list[int]:
    """Return Linux PIDs whose command line is this ST16 train/output pair."""

    matches: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("producer-stop audit requires Linux /proc")
    expected = output_dir.resolve(strict=False)
    for candidate in proc.iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            parts = [
                value.decode("utf-8", errors="replace")
                for value in (candidate / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        joined = " ".join(parts)
        if (
            "run_st16_common17_exploratory_v1.py" not in joined
            or " train " not in f" {joined} "
            or "--output-dir" not in parts
        ):
            continue
        output_index = parts.index("--output-dir") + 1
        if output_index >= len(parts):
            continue
        raw_output = Path(parts[output_index])
        try:
            process_cwd = (candidate / "cwd").resolve(strict=True)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        process_output = (
            raw_output.resolve(strict=False)
            if raw_output.is_absolute()
            else (process_cwd / raw_output).resolve(strict=False)
        )
        if process_output == expected:
            matches.append(int(candidate.name))
    return sorted(set(matches))


def _read_verified_receipt(path: Path, *, artifact_name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"{artifact_name} must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{artifact_name} must be a JSON object")
    return _validate_content_address(value, artifact_name=artifact_name)


def _checkpoint_audit(path: Path, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("adoption requires a regular partial resume checkpoint")
    before = path.stat()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("ST16 checkpoint changed during adoption audit")
    required = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "provider_id": PROVIDER_ID,
        "claim_status": "exploratory_nonpromotable",
        "variant_id": st.ST16_VARIANT_ID,
        "epoch_plan_receipt_sha256": plan["receipt_sha256"],
        "source_eval_opened": False,
        "architecture_promotable": False,
        "checkpoint_role": "partial_epoch_resume_only",
        "inference_eligible": False,
        "training_complete": False,
    }
    if not isinstance(checkpoint, Mapping) or any(
        checkpoint.get(key) != value for key, value in required.items()
    ):
        raise PermissionError("checkpoint is not the authorized partial epoch-0 run")
    if int(checkpoint.get("next_epoch", -1)) != 0 or int(
        checkpoint.get("next_batch", -1)
    ) < 1:
        raise PermissionError("checkpoint cursor is not a resumable epoch-0 prefix")
    return {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "next_epoch": int(checkpoint["next_epoch"]),
        "next_batch": int(checkpoint["next_batch"]),
        "global_step": int(checkpoint["global_step"]),
        "checkpoint_role": checkpoint["checkpoint_role"],
    }


def _inventory_markerless_tiles(
    cache: Path, *, catalog: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    before_names = sorted(path.name for path in cache.iterdir())
    if not before_names:
        raise ValueError("authorized markerless cache is empty")
    recoverable_metadata = [
        name
        for name in before_names
        if name == "cache_adoption_receipt.json"
        or (name.endswith(".json") and name.removesuffix(".json") in catalog)
    ]
    unexpected = [
        name
        for name in before_names
        if not name.endswith(".npy") and name not in recoverable_metadata
    ]
    if unexpected:
        raise PermissionError(
            "markerless cache contains unexpected or partial files: "
            + ", ".join(unexpected[:10])
        )
    for name in recoverable_metadata:
        candidate = cache / name
        if candidate.is_symlink() or not candidate.is_file():
            raise PermissionError("recoverable adoption metadata is not regular")
    npy_names = [name for name in before_names if name.endswith(".npy")]
    inventory: list[dict[str, Any]] = []
    expected_size = len(st.ST16_TYPED_UNITS) * TILE_SAMPLES * 4 + 128
    for name in npy_names:
        target = cache / name
        tile_id = target.stem
        if target.is_symlink() or not target.is_file() or tile_id not in catalog:
            raise PermissionError("markerless cache tile is outside the frozen plan")
        before = target.stat()
        value = np.load(target, mmap_mode="r", allow_pickle=False)
        valid = bool(
            value.shape == (len(st.ST16_TYPED_UNITS), TILE_SAMPLES)
            and value.dtype == np.dtype("float32")
            and value.nbytes == len(st.ST16_TYPED_UNITS) * TILE_SAMPLES * 4
            and bool(np.isfinite(value).all())
        )
        del value
        digest = _file_sha256(target)
        after = target.stat()
        if (
            not valid
            or before.st_size != expected_size
            or (before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError(f"markerless ST16 tile failed replay: {tile_id}")
        inventory.append(
            {
                "tile_id": tile_id,
                "npy_sha256": digest,
                "npy_size_bytes": before.st_size,
            }
        )
    after_names = sorted(path.name for path in cache.iterdir())
    if before_names != after_names:
        raise RuntimeError("markerless cache changed during adoption audit")
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--staging-equivalence-receipt",
        type=Path,
        default=DEFAULT_STAGING_RECEIPT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DURABLE_RECEIPT)
    args = parser.parse_args()

    cache = args.cache.resolve(strict=True)
    if cache.is_symlink() or not cache.is_dir():
        raise PermissionError("markerless cache must be a regular directory")
    output_dir = cache.parents[1]
    running = _running_train_producers(output_dir)
    if running:
        raise RuntimeError(
            "refusing to adopt while ST16 producer is live: "
            + ",".join(str(value) for value in running)
        )
    if (cache / "cache_contract.json").exists() or (
        cache / "cache_contract.json"
    ).is_symlink():
        raise FileExistsError("cache is already contracted; adoption is one-time")

    plan = build_exploratory_epoch_plan(
        args.manifest,
        epoch_index=0,
        batch_size=8,
        partial_batch_policy="emit_explicit",
    )
    if plan["receipt_sha256"] != LEGACY_MARKERLESS_ADOPTION_PLAN_SHA256:
        raise PermissionError("one-time authorized epoch plan is unavailable")
    staging = _read_verified_receipt(
        args.staging_equivalence_receipt,
        artifact_name="ST16 local-staging equivalence receipt",
    )
    if (
        staging["receipt_sha256"]
        != LEGACY_LOCAL_STAGING_EQUIVALENCE_RECEIPT_SHA256
        or staging.get("status") != "pass_bitwise_identical"
        or staging.get("bitwise_array_equal") is not True
        or staging.get("transform_receipt_equal") is not True
        or staging.get("source_eval_opened") is not False
    ):
        raise PermissionError("local-staging equivalence receipt is not authorized")
    checkpoint_audit = _checkpoint_audit(args.checkpoint.resolve(strict=True), plan=plan)
    catalog = {
        str(tile_id): dict(row)
        for tile_id, row in plan["selected_tile_catalog"].items()
    }
    inventory = _inventory_markerless_tiles(cache, catalog=catalog)
    roster = [row["tile_id"] for row in inventory]
    manifest_rows = [
        [row["tile_id"], row["npy_size_bytes"], row["npy_sha256"]]
        for row in inventory
    ]
    adoption_receipt = _content_address(
        {
            "schema_version": "st16_markerless_epoch_cache_adoption_v1",
            "status": "one_time_markerless_cache_adoption_passed",
            "claim_status": "temporary_nonpromotable_training_cache_recovery",
            "cache_path": str(cache),
            "epoch_plan_receipt_sha256": plan["receipt_sha256"],
            "manifest_receipt_sha256": plan["manifest_receipt_sha256"],
            "local_staging_equivalence_receipt_sha256": staging[
                "receipt_sha256"
            ],
            "checkpoint_audit": checkpoint_audit,
            "adopted_tile_count": len(inventory),
            "full_epoch_unique_tile_count": len(catalog),
            "adopted_tile_roster_sha256": _canonical_sha256(roster),
            "adopted_tile_manifest_sha256": _canonical_sha256(manifest_rows),
            "all_entries_regular_non_symlink_npy_files": True,
            "all_tiles_belong_to_frozen_plan": True,
            "all_tiles_float32_shape_exact_and_finite": True,
            "all_tiles_content_hashed": True,
            "unexpected_or_partial_file_count": 0,
            "producer_process_live_during_adoption": False,
            "numeric_content_recomputed_from_source_during_adoption": False,
            "audit_boundary": (
                "Adoption proves structural validity, finiteness, frozen-plan "
                "membership, stable bytes, and subsequent content replay. It "
                "does not retroactively recompute every tile from its EDF."
            ),
            "source_eval_opened": False,
            "receipt_sha256": PENDING,
        }
    )
    contract = _content_address(
        {
            **_cache_contract_core(plan),
            "legacy_markerless_adoption": {
                "epoch_plan_receipt_sha256": plan["receipt_sha256"],
                "local_staging_equivalence_receipt_sha256": staging[
                    "receipt_sha256"
                ],
                "adoption_status": "one_time_markerless_cache_adoption_passed",
                "adoption_receipt_file_name": "cache_adoption_receipt.json",
                "adoption_receipt_sha256": adoption_receipt["receipt_sha256"],
            },
            "receipt_sha256": PENDING,
        }
    )
    if contract["schema_version"] != CACHE_CONTRACT_SCHEMA_VERSION:
        raise AssertionError("cache-contract schema drifted")

    by_id = {row["tile_id"]: row for row in inventory}
    for tile_id in sorted(by_id):
        target = cache / f"{tile_id}.npy"
        observed = by_id[tile_id]
        sidecar = _cache_tile_sidecar(
            catalog[tile_id],
            target=target,
            cache_contract_receipt_sha256=contract["receipt_sha256"],
            npy_size_bytes=int(observed["npy_size_bytes"]),
            npy_sha256=str(observed["npy_sha256"]),
        )
        _write_json_atomic(target.with_suffix(".json"), sidecar, replace=True)
    in_cache_receipt = cache / "cache_adoption_receipt.json"
    _write_json_atomic(in_cache_receipt, adoption_receipt, replace=True)
    output = args.output.resolve(strict=False)
    if output.is_file() and not output.is_symlink():
        existing = _read_verified_receipt(
            output, artifact_name="durable ST16 cache-adoption receipt"
        )
        if existing != adoption_receipt:
            raise FileExistsError("durable cache-adoption receipt differs")
    else:
        _write_json_atomic(output, adoption_receipt, replace=False)
    # The contract is the final commit point.  Before it exists, a stopped
    # adoption can be replayed; after it exists, the regular cache validator
    # owns all subsequent resume decisions.
    _write_json_atomic(cache / "cache_contract.json", contract, replace=False)
    print(
        json.dumps(
            {
                "status": adoption_receipt["status"],
                "adopted_tile_count": len(inventory),
                "cache_contract_receipt_sha256": contract["receipt_sha256"],
                "receipt_sha256": adoption_receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
