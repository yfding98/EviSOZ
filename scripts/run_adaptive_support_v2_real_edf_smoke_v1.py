#!/usr/bin/env python3
"""Run the frozen two-record adaptive-support-v2 real-EDF smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.adaptive_support_v2_real_edf_smoke_v1 import (  # noqa: E402
    load_real_edf_smoke_manifest_v1,
    materialize_real_edf_smoke_entry_v1,
    summarize_real_edf_smoke_v1,
)


DEFAULT_MANIFEST = (
    ROOT / "configs/clinical_eeg_adaptive_support_v2_real_edf_smoke_v1.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = (
    ROOT / "outputs/clinical_eeg_adaptive_support_v2_real_edf_smoke_v1_20260825"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_workspace_path(value: object) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("manifest workspace binding is unsafe")
    path = ROOT.joinpath(*relative.parts).resolve(strict=True)
    path.relative_to(ROOT)
    return path


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
    manifest = load_real_edf_smoke_manifest_v1(manifest_path)
    manifest_sha256 = _file_sha256(manifest_path)
    detector = manifest["detector_cache_contract"]
    decoder_path = _safe_workspace_path(detector["decoder_source_relative_path"])
    roster_audit_path = _safe_workspace_path(
        detector["prediction_roster_audit_relative_path"]
    )
    if _file_sha256(decoder_path) != detector["decoder_source_file_sha256"]:
        raise ValueError("frozen EventNet decoder source hash drifted")
    if (
        _file_sha256(roster_audit_path)
        != detector["prediction_roster_audit_file_sha256"]
    ):
        raise ValueError("frozen peak-cache roster audit hash drifted")

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        event = materialize_real_edf_smoke_entry_v1(
            entry=entry,
            detector_contract=detector,
            manifest_sha256=manifest_sha256,
            tusz_root=arguments.tusz_root,
            workspace_root=ROOT,
        )
        target = output / "events" / str(entry["recording_id"]) / "receipt.json"
        _atomic_json(target, event)
        events.append(event)
    receipt = summarize_real_edf_smoke_v1(
        manifest_sha256=manifest_sha256,
        events=events,
    )
    _atomic_json(output / "receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
