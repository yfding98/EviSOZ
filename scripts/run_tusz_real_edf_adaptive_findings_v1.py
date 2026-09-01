#!/usr/bin/env python3
"""Run the frozen target-blind common-17 adaptive Findings EDF rollout."""

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

from src.clinical_eeg_long_recording.tusz_real_edf_adaptive_findings_v1 import (  # noqa: E402
    load_tusz_real_edf_adaptive_manifest,
    materialize_tusz_real_edf_adaptive_entry,
    summarize_tusz_real_edf_adaptive_rollouts,
)


DEFAULT_MANIFEST = ROOT / "configs/tusz_real_edf_adaptive_findings_cohort_v1.json"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = ROOT / "outputs/tusz_real_edf_adaptive_findings_v1_20260825"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    manifest_path = arguments.manifest.resolve(strict=True)
    manifest = load_tusz_real_edf_adaptive_manifest(manifest_path)
    manifest_sha256 = _file_sha256(manifest_path)
    root = arguments.tusz_root.resolve(strict=True)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested = set(arguments.rollout_id or [])
    available = {str(row["rollout_id"]) for row in manifest["entries"]}
    unknown = requested - available
    if unknown:
        raise ValueError(f"unknown rollout IDs: {sorted(unknown)}")
    entries = [
        row
        for row in manifest["entries"]
        if not requested or str(row["rollout_id"]) in requested
    ]
    rollouts: list[dict[str, Any]] = []
    for entry in entries:
        receipt = materialize_tusz_real_edf_adaptive_entry(
            entry=entry,
            tusz_root=root,
            manifest_sha256=manifest_sha256,
        )
        target = output / "events" / str(entry["rollout_id"]) / "receipt.json"
        _atomic_json(target, receipt)
        rollouts.append(receipt)
    summary = summarize_tusz_real_edf_adaptive_rollouts(
        manifest_sha256=manifest_sha256,
        rollouts=rollouts,
    )
    summary["source_bindings"] = {
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_sha256,
        "tusz_root": str(root),
        "runtime_sidecars_opened": [],
    }
    summary["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in summary.items() if key != "receipt_sha256"}
    )
    _atomic_json(output / "receipt.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument(
        "--rollout-id",
        action="append",
        help="Run only this frozen rollout ID (repeatable); default runs all.",
    )
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True, indent=2))
