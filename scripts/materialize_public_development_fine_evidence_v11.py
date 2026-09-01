#!/usr/bin/env python3
"""Materialize target-free fine temporal evidence for the v11 public union."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.fine_temporal_evidence import (  # noqa: E402
    FINE_TEMPORAL_EVIDENCE_SCHEMA,
    FINE_TEMPORAL_FEATURE_NAMES,
    extract_fine_temporal_evidence,
)
from src.soz.frozen_h_crosswalk import _signal_tensor_sha256  # noqa: E402
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)
from src.soz.v11_development_union import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    load_public_development_union,
)


DEFAULT_UNION = ROOT / "outputs/public_development_union_v11_20260811"
DEFAULT_SIGNAL = ROOT / "outputs/deepsoz_signal_preflight_v2_20260809_current"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = ROOT / "outputs/public_development_fine_evidence_v11_20260811"
EXPECTED_SIGNAL_ARTIFACT_SHA256 = (
    "a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66"
)
EXPECTED_SIGNAL_RECEIPT_SHA256 = (
    "10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446"
)
FULL_SCHEMA = "soz_public_development_fine_evidence_v11_full"
SMOKE_SCHEMA = "soz_public_development_fine_evidence_v11_smoke"
MANIFEST_NAME = "manifest.json"
TENSOR_NAME = "evidence.safetensors"


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    metadata = _canonical_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _safe_edf(root: Path, relative_value: str) -> Path:
    relative = PurePosixPath(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("union EDF path is not a safe relative EDF path")
    source = root.joinpath(*relative.parts)
    for component in (source, *source.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("union EDF path cannot traverse symlinks")
    resolved = source.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("union EDF path escaped the pinned TUSZ root")
    return resolved


def _event_digest(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in tensors:
        encoded = _tensor_sha(value).encode("ascii")
        digest.update(encoded)
    return digest.hexdigest()


def materialize(
    *,
    union_directory: Path,
    signal_directory: Path,
    tusz_root: Path,
    output_directory: Path,
    limit: int | None,
    progress_every: int,
) -> tuple[Path, Mapping[str, object]]:
    union = load_public_development_union(
        union_directory,
        expected_manifest_sha256=EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    )
    signal = load_bound_deepsoz_signal_preflight_artifact(
        signal_directory,
        expected_artifact_sha256=EXPECTED_SIGNAL_ARTIFACT_SHA256,
        expected_receipt_sha256=EXPECTED_SIGNAL_RECEIPT_SHA256,
    )
    if signal.receipt.get("eligible_event_count") != len(union.events):
        raise ValueError("signal preflight and public union event counts differ")
    config_payload = signal.receipt.get("preprocess_config")
    if not isinstance(config_payload, Mapping):
        raise TypeError("signal preflight lacks a preprocessing configuration")
    config = CausalEDFConfig(**dict(config_payload))

    raw_root = Path(os.path.abspath(tusz_root)).resolve(strict=True)
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise FileNotFoundError(raw_root)
    target = Path(os.path.abspath(output_directory))
    if target.exists():
        raise FileExistsError(target)
    for source in (union.path, Path(signal_directory).resolve(), raw_root):
        if target == source or target in source.parents or source in target.parents:
            raise ValueError("fine-evidence output overlaps an immutable input")
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)

    if limit is None:
        selected = union.events
        full_scope = True
    else:
        if isinstance(limit, bool) or not 1 <= int(limit) < len(union.events):
            raise ValueError("--limit must be a smoke prefix in [1,987]")
        selected = union.events[: int(limit)]
        full_scope = False

    signal_rows = signal.receipt.get("events")
    if not isinstance(signal_rows, list):
        raise TypeError("signal preflight event rows are missing")
    signal_by_id = {str(row["event_id"]): row for row in signal_rows}
    if len(signal_by_id) != len(signal_rows):
        raise ValueError("signal preflight repeats an event ID")

    features = []
    composite = []
    frequency = []
    node_detected = []
    node_latency = []
    bipolar_detected = []
    bipolar_latency = []
    output_rows = []
    expected_grid: torch.Tensor | None = None
    started = time.monotonic()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    for position, event in enumerate(selected, start=1):
        source_row = signal_by_id.get(event.event_id)
        if source_row is None:
            raise ValueError(f"union event disappeared from signal preflight: {event.event_id}")
        path = _safe_edf(raw_root, event.relative_edf_path)
        loaded = load_standard19_edf_event(
            path,
            event.global_t0_sec,
            config=config,
        )
        edf_receipt_sha = _canonical_sha(asdict(loaded.edf_receipt))
        signal_receipt_sha = _canonical_sha(asdict(loaded.signal_receipt))
        replay_sha = _signal_tensor_sha256(loaded.window.data)
        checks = {
            "EDF": loaded.edf_receipt.edf_sha256 == event.edf_sha256,
            "EDF receipt": edf_receipt_sha == event.edf_receipt_sha256,
            "signal receipt": signal_receipt_sha == event.signal_receipt_sha256,
            "processed window": replay_sha == event.processed_window_sha256,
            "shape": tuple(loaded.window.data.shape) == (19, 12000),
            "sampling": float(loaded.window.sfreq_hz) == 200.0,
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"raw replay failed {event.event_id}: {failed}")
        evidence = extract_fine_temporal_evidence(
            loaded.window.data,
            sfreq_hz=loaded.window.sfreq_hz,
        )
        if expected_grid is None:
            expected_grid = evidence.window_center_sec.cpu()
        elif not torch.equal(expected_grid, evidence.window_center_sec.cpu()):
            raise RuntimeError("fine temporal grid changed between events")

        values = (
            evidence.features.cpu(),
            evidence.composite_trace.cpu(),
            evidence.dominant_frequency_hz.cpu(),
            evidence.node_change_detected.cpu(),
            evidence.node_change_latency_sec.cpu(),
            evidence.bipolar_change_detected.cpu(),
            evidence.bipolar_change_latency_sec.cpu(),
        )
        features.append(values[0])
        composite.append(values[1])
        frequency.append(values[2])
        node_detected.append(values[3])
        node_latency.append(values[4])
        bipolar_detected.append(values[5])
        bipolar_latency.append(values[6])
        output_rows.append(
            {
                "ordinal": event.ordinal,
                "event_id": event.event_id,
                "patient_id": event.patient_id,
                "outer_fold": event.outer_fold,
                "legacy_model_split": event.legacy_model_split,
                "processed_window_sha256": replay_sha,
                "fine_evidence_sha256": _event_digest(values),
                "node_change_detected_count": int(values[3].sum().item()),
                "bipolar_change_detected_count": int(values[5].sum().item()),
            }
        )
        if position % progress_every == 0 or position == len(selected):
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "event": position,
                        "total": len(selected),
                        "elapsed_sec": round(elapsed, 2),
                        "seconds_per_event": round(elapsed / position, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    assert expected_grid is not None
    tensors = {
        "features": torch.stack(features).contiguous(),
        "composite_trace": torch.stack(composite).contiguous(),
        "dominant_frequency_hz": torch.stack(frequency).contiguous(),
        "node_change_detected": torch.stack(node_detected).contiguous(),
        "node_change_latency_sec": torch.stack(node_latency).contiguous(),
        "bipolar_change_detected": torch.stack(bipolar_detected).contiguous(),
        "bipolar_change_latency_sec": torch.stack(bipolar_latency).contiguous(),
        "window_center_sec": expected_grid.contiguous(),
    }
    if not torch.isfinite(tensors["features"]).all() or not torch.isfinite(
        tensors["composite_trace"]
    ).all():
        raise RuntimeError("formal fine evidence contains non-finite model inputs")
    tensor_specs = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "tensor_sha256": _tensor_sha(value),
        }
        for name, value in tensors.items()
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / TENSOR_NAME
        save_file(tensors, str(tensor_path))
        elapsed = time.monotonic() - started
        manifest = {
            "schema_version": FULL_SCHEMA if full_scope else SMOKE_SCHEMA,
            "evidence_schema_version": FINE_TEMPORAL_EVIDENCE_SCHEMA,
            "purpose": "target_free_subsecond_change_evidence_for_v11_development",
            "development_only": True,
            "public_confirmation_forbidden": True,
            "full_scope": full_scope,
            "smoke_only": not full_scope,
            "event_count": len(selected),
            "patient_count": len({event.patient_id for event in selected}),
            "event_ids": [event.event_id for event in selected],
            "event_order_sha256": _canonical_sha(
                [event.event_id for event in selected]
            ),
            "events": output_rows,
            "feature_names": list(FINE_TEMPORAL_FEATURE_NAMES),
            "temporal_contract": {
                "input_shape": [19, 12000],
                "sampling_frequency_hz": 200.0,
                "event_interval_sec": [-12.0, 48.0],
                "analysis_window_sec": 1.0,
                "analysis_stride_sec": 0.25,
                "effective_temporal_resolution_sec": 1.0,
                "output_grid_stride_sec": 0.25,
                "tusz_t0_is_alignment_not_soz_onset": True,
                "change_is_not_cortical_onset": True,
                "relative_delay_is_not_propagation_truth": True,
            },
            "tensor_file": TENSOR_NAME,
            "tensor_file_sha256": _file_sha(tensor_path),
            "tensor_file_size_bytes": tensor_path.stat().st_size,
            "tensor_specs": tensor_specs,
            "lineage": {
                "public_union_manifest_sha256": union.manifest_sha256,
                "signal_preflight_artifact_sha256": signal.artifact_sha256,
                "signal_preflight_receipt_sha256": signal.receipt_sha256,
                "signal_preprocess_config_sha256": signal.receipt[
                    "preprocess_config_sha256"
                ],
                "tusz_root": str(raw_root),
            },
            "elapsed_sec": elapsed,
            "seconds_per_event": elapsed / len(selected),
            "access_receipt": {
                "raw_public_eeg_loaded": True,
                "raw_public_event_count": len(selected),
                "deepsoz_target_values_loaded": False,
                "tusz_channel_annotation_values_loaded": False,
                "historical_prediction_artifacts_loaded": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "foundation_training_performed": False,
                "reasoner_training_performed": False,
            },
        }
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_bytes(_canonical_bytes(manifest, newline=True))
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--signal-directory", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    path, manifest = materialize(
        union_directory=args.union_directory,
        signal_directory=args.signal_directory,
        tusz_root=args.tusz_root,
        output_directory=args.output_directory,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": "FINE_EVIDENCE_V11_MATERIALIZED",
                "path": str(path),
                "manifest_sha256": _file_sha(path / MANIFEST_NAME),
                "event_count": manifest["event_count"],
                "full_scope": manifest["full_scope"],
                "target_values_loaded": False,
                "private_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
