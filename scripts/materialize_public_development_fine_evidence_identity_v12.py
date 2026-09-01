#!/usr/bin/env python3
"""Append only recovered events to the frozen v11 fine-evidence cache."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.data.identity_v12_cache_extension import (  # noqa: E402
    LEGACY_EVENT_COUNT,
    append_event_tensor_exact,
    canonical_bytes,
    canonical_sha256,
    event_tensor_sha256,
    file_sha256,
    load_identity_v12_extension_contract,
    load_legacy_representation_cache,
    select_appended_events,
    tensor_bitwise_equal,
    tensor_sha256,
)
from src.soz.fine_temporal_evidence import (  # noqa: E402
    FINE_TEMPORAL_EVIDENCE_SCHEMA,
    FINE_TEMPORAL_FEATURE_NAMES,
    extract_fine_temporal_evidence,
)
from src.soz.frozen_h_crosswalk import _signal_tensor_sha256  # noqa: E402


DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_SIGNAL = ROOT / "outputs/deepsoz_signal_preflight_identity_v3_20260812"
DEFAULT_LEGACY_CACHE = (
    ROOT / "outputs/public_development_fine_evidence_v11_20260811"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = (
    ROOT / "outputs/public_development_fine_evidence_identity_v12_20260812"
)

EXPECTED_UNION_MANIFEST_SHA256 = (
    "645c55541c37dfc204fdd48c21e0a3c81fe7201f76b862556d1c4dc3bfa4d429"
)
EXPECTED_SIGNAL_ARTIFACT_SHA256 = (
    "2a6bb8a7be20993949e7250b10c83d11fe027ff1afc0fa0919124f7fa371ef8e"
)
EXPECTED_LEGACY_MANIFEST_SHA256 = (
    "60ce6c5af15dcff3a0c0dcbac1451f4d5cb3bb28e7b9c22180ab7adecfb417a2"
)
EXPECTED_LEGACY_TENSOR_FILE_SHA256 = (
    "24dc5da224c79446992cde08d800877ff1ea4349d217c225da95588c9e173bbb"
)
LEGACY_SCHEMA = "soz_public_development_fine_evidence_v11_full"
FULL_SCHEMA = "soz_public_development_fine_evidence_identity_v12"
SMOKE_SCHEMA = "soz_public_development_fine_evidence_identity_v12_smoke"
MANIFEST_NAME = "manifest.json"
TENSOR_NAME = "evidence.safetensors"
EVENT_TENSOR_NAMES = (
    "features",
    "composite_trace",
    "dominant_frequency_hz",
    "node_change_detected",
    "node_change_latency_sec",
    "bipolar_change_detected",
    "bipolar_change_latency_sec",
)
SHARED_TENSOR_NAME = "window_center_sec"


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


def _validate_legacy_tensors(
    tensors: Mapping[str, torch.Tensor],
    manifest: Mapping[str, object],
) -> tuple[dict[str, torch.Tensor], str]:
    if set(tensors) != {*EVENT_TENSOR_NAMES, SHARED_TENSOR_NAME}:
        raise ValueError("legacy fine-evidence tensor names changed")
    specs = manifest.get("tensor_specs")
    rows = manifest.get("events")
    if not isinstance(specs, Mapping) or not isinstance(rows, list):
        raise TypeError("legacy fine-evidence manifest lacks specs/events")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        tensor = value.detach().cpu().contiguous()
        spec = specs.get(name)
        if not isinstance(spec, Mapping):
            raise ValueError(f"legacy fine-evidence spec missing: {name}")
        if (
            list(tensor.shape) != spec.get("shape")
            or str(tensor.dtype) != spec.get("dtype")
            or tensor_sha256(tensor) != spec.get("tensor_sha256")
        ):
            raise ValueError(f"legacy fine-evidence tensor integrity failed: {name}")
        if name in EVENT_TENSOR_NAMES and tensor.shape[0] != LEGACY_EVENT_COUNT:
            raise ValueError("legacy fine-evidence event dimension changed")
        normalized[name] = tensor
    per_event_hashes: list[str] = []
    for ordinal, row in enumerate(rows):
        digest = event_tensor_sha256(
            tuple(normalized[name][ordinal] for name in EVENT_TENSOR_NAMES)
        )
        if digest != row.get("fine_evidence_sha256"):
            raise ValueError(
                f"legacy fine-evidence event SHA failed: {row.get('event_id')}"
            )
        per_event_hashes.append(digest)
    return normalized, canonical_sha256(per_event_hashes)


def _validate_raw_replay(
    loaded: object,
    event: Mapping[str, object],
) -> str:
    edf_receipt_sha = canonical_sha256(asdict(loaded.edf_receipt))
    signal_receipt_sha = canonical_sha256(asdict(loaded.signal_receipt))
    replay_sha = _signal_tensor_sha256(loaded.window.data)
    checks = {
        "EDF": loaded.edf_receipt.edf_sha256 == event["edf_sha256"],
        "EDF receipt": edf_receipt_sha == event["edf_receipt_sha256"],
        "signal receipt": signal_receipt_sha == event["signal_receipt_sha256"],
        "processed window": replay_sha == event["processed_window_sha256"],
        "shape": tuple(loaded.window.data.shape) == (19, 12_000),
        "sampling": float(loaded.window.sfreq_hz) == 200.0,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"raw replay failed {event['event_id']}: {failed}")
    return replay_sha


def materialize(
    *,
    union_directory: Path,
    signal_directory: Path,
    legacy_cache_directory: Path,
    tusz_root: Path,
    output_directory: Path,
    append_limit: int | None,
    progress_every: int,
) -> tuple[Path, Mapping[str, object]]:
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("progress_every must be a positive integer")
    contract = load_identity_v12_extension_contract(
        union_directory,
        signal_directory,
        expected_union_manifest_sha256=EXPECTED_UNION_MANIFEST_SHA256,
        expected_signal_artifact_sha256=EXPECTED_SIGNAL_ARTIFACT_SHA256,
    )
    legacy = load_legacy_representation_cache(
        legacy_cache_directory,
        contract=contract,
        expected_schema=LEGACY_SCHEMA,
        tensor_filename=TENSOR_NAME,
        expected_manifest_sha256=EXPECTED_LEGACY_MANIFEST_SHA256,
        expected_tensor_file_sha256=EXPECTED_LEGACY_TENSOR_FILE_SHA256,
    )
    selected_append, full_scope = select_appended_events(contract, append_limit)

    config_payload = contract.signal_bundle.receipt.get("preprocess_config")
    if not isinstance(config_payload, Mapping):
        raise TypeError("identity-v3 signal receipt lacks preprocessing config")
    config = CausalEDFConfig(**dict(config_payload))
    raw_root = Path(os.path.abspath(tusz_root)).resolve(strict=True)
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise FileNotFoundError(raw_root)
    target = Path(os.path.abspath(output_directory))
    if os.path.lexists(target):
        raise FileExistsError(target)
    for source in (
        contract.union_path,
        Path(signal_directory).resolve(),
        legacy.path,
        raw_root,
    ):
        if target == source or target in source.parents or source in target.parents:
            raise ValueError("fine-evidence extension output overlaps an input")
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)

    legacy_tensors, legacy_event_sha_roster = _validate_legacy_tensors(
        load_file(str(legacy.tensor_path), device="cpu"), legacy.manifest
    )
    new_values: dict[str, list[torch.Tensor]] = {
        name: [] for name in EVENT_TENSOR_NAMES
    }
    new_rows: list[dict[str, object]] = []
    started = time.monotonic()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    for position, event in enumerate(selected_append, start=1):
        path = _safe_edf(raw_root, str(event["relative_edf_path"]))
        loaded = load_standard19_edf_event(
            path, float(event["global_t0_sec"]), config=config
        )
        replay_sha = _validate_raw_replay(loaded, event)
        evidence = extract_fine_temporal_evidence(
            loaded.window.data, sfreq_hz=loaded.window.sfreq_hz
        )
        if not tensor_bitwise_equal(
            legacy_tensors[SHARED_TENSOR_NAME],
            evidence.window_center_sec.detach().cpu().contiguous(),
        ):
            raise RuntimeError("fine temporal grid differs from frozen v11 cache")
        values = {
            "features": evidence.features.detach().cpu().contiguous(),
            "composite_trace": evidence.composite_trace.detach().cpu().contiguous(),
            "dominant_frequency_hz": (
                evidence.dominant_frequency_hz.detach().cpu().contiguous()
            ),
            "node_change_detected": (
                evidence.node_change_detected.detach().cpu().contiguous()
            ),
            "node_change_latency_sec": (
                evidence.node_change_latency_sec.detach().cpu().contiguous()
            ),
            "bipolar_change_detected": (
                evidence.bipolar_change_detected.detach().cpu().contiguous()
            ),
            "bipolar_change_latency_sec": (
                evidence.bipolar_change_latency_sec.detach().cpu().contiguous()
            ),
        }
        for name in EVENT_TENSOR_NAMES:
            if values[name].shape != legacy_tensors[name].shape[1:] or (
                values[name].dtype != legacy_tensors[name].dtype
            ):
                raise ValueError(f"new fine-evidence tensor contract changed: {name}")
            new_values[name].append(values[name])
        new_rows.append(
            {
                "ordinal": int(event["ordinal"]),
                "event_id": str(event["event_id"]),
                "patient_id": str(event["patient_id"]),
                "outer_fold": int(event["outer_fold"]),
                "legacy_model_split": str(event["legacy_model_split"]),
                "processed_window_sha256": replay_sha,
                "fine_evidence_sha256": event_tensor_sha256(
                    tuple(values[name] for name in EVENT_TENSOR_NAMES)
                ),
                "node_change_detected_count": int(
                    values["node_change_detected"].sum().item()
                ),
                "bipolar_change_detected_count": int(
                    values["bipolar_change_detected"].sum().item()
                ),
            }
        )
        if position % progress_every == 0 or position == len(selected_append):
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "new_event": position,
                        "new_total": len(selected_append),
                        "elapsed_sec": round(elapsed, 2),
                        "seconds_per_new_event": round(elapsed / position, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    tensors: dict[str, torch.Tensor] = {}
    for name in EVENT_TENSOR_NAMES:
        appended = torch.stack(new_values[name]).contiguous()
        tensors[name] = append_event_tensor_exact(
            legacy_tensors[name], appended
        )
    tensors[SHARED_TENSOR_NAME] = legacy_tensors[SHARED_TENSOR_NAME].clone().contiguous()
    if tensor_sha256(tensors[SHARED_TENSOR_NAME]) != tensor_sha256(
        legacy_tensors[SHARED_TENSOR_NAME]
    ):
        raise RuntimeError("shared fine temporal grid changed during extension")
    if not torch.isfinite(tensors["features"]).all() or not torch.isfinite(
        tensors["composite_trace"]
    ).all():
        raise RuntimeError("extended fine evidence contains non-finite model inputs")

    legacy_rows = [dict(row) for row in legacy.manifest["events"]]
    output_rows = [*legacy_rows, *new_rows]
    selected_events = (*contract.legacy_events, *selected_append)
    event_ids = [str(event["event_id"]) for event in selected_events]
    if [str(row["event_id"]) for row in output_rows] != event_ids:
        raise RuntimeError("extended fine-evidence event order changed")
    if output_rows[:LEGACY_EVENT_COUNT] != legacy_rows:
        raise RuntimeError("legacy fine-evidence manifest rows changed")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / TENSOR_NAME
        save_file(tensors, str(tensor_path))
        elapsed = time.monotonic() - started
        tensor_specs = {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_sha256": tensor_sha256(value),
            }
            for name, value in tensors.items()
        }
        manifest: dict[str, object] = {
            "schema_version": FULL_SCHEMA if full_scope else SMOKE_SCHEMA,
            "evidence_schema_version": FINE_TEMPORAL_EVIDENCE_SCHEMA,
            "purpose": (
                "target_free_append_only_fine_evidence_for_identity_v12_development"
            ),
            "development_only": True,
            "public_confirmation_forbidden": True,
            "full_scope": full_scope,
            "smoke_only": not full_scope,
            "event_count": len(selected_events),
            "patient_count": len({str(event["patient_id"]) for event in selected_events}),
            "legacy_reused_event_count": LEGACY_EVENT_COUNT,
            "newly_computed_event_count": len(selected_append),
            "event_ids": event_ids,
            "event_order_sha256": canonical_sha256(event_ids),
            "events": output_rows,
            "feature_names": list(FINE_TEMPORAL_FEATURE_NAMES),
            "temporal_contract": dict(legacy.manifest["temporal_contract"]),
            "tensor_file": TENSOR_NAME,
            "tensor_file_sha256": file_sha256(tensor_path),
            "tensor_file_size_bytes": tensor_path.stat().st_size,
            "tensor_specs": tensor_specs,
            "cache_extension_receipt": {
                "append_only": True,
                "legacy_event_rows_exact_prefix": output_rows[:LEGACY_EVENT_COUNT]
                == legacy_rows,
                "legacy_event_ids_exact_prefix": event_ids[:LEGACY_EVENT_COUNT]
                == list(legacy.manifest["event_ids"]),
                "legacy_tensor_prefix_exact": all(
                    tensor_bitwise_equal(
                        tensors[name][:LEGACY_EVENT_COUNT], legacy_tensors[name]
                    )
                    for name in EVENT_TENSOR_NAMES
                ),
                "legacy_shared_grid_exact": tensor_bitwise_equal(
                    tensors[SHARED_TENSOR_NAME], legacy_tensors[SHARED_TENSOR_NAME]
                ),
                "legacy_event_sha_roster_sha256": legacy_event_sha_roster,
                "old_raw_eeg_replayed": False,
                "new_raw_eeg_event_count": len(selected_append),
            },
            "lineage": {
                "public_union_manifest_sha256": contract.union_manifest_sha256,
                "signal_identity_recovery_artifact_sha256": (
                    contract.signal_bundle.artifact_sha256
                ),
                "signal_identity_recovery_receipt_sha256": (
                    contract.signal_bundle.receipt_sha256
                ),
                "signal_preprocess_config_sha256": (
                    contract.signal_bundle.receipt["preprocess_config_sha256"]
                ),
                "legacy_cache_manifest_sha256": legacy.manifest_sha256,
                "legacy_cache_tensor_file_sha256": legacy.tensor_file_sha256,
                "tusz_root": str(raw_root),
            },
            "elapsed_sec": elapsed,
            "seconds_per_new_event": elapsed / len(selected_append),
            "access_receipt": {
                "legacy_target_free_tensor_cache_loaded": True,
                "raw_public_eeg_loaded": True,
                "raw_public_event_count": len(selected_append),
                "legacy_raw_public_event_count": 0,
                "deepsoz_target_values_loaded": False,
                "tusz_channel_annotation_values_loaded": False,
                "historical_prediction_artifacts_loaded": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "foundation_training_performed": False,
                "reasoner_training_performed": False,
            },
        }
        if not all(
            manifest["cache_extension_receipt"][key]
            for key in (
                "append_only",
                "legacy_event_rows_exact_prefix",
                "legacy_event_ids_exact_prefix",
                "legacy_tensor_prefix_exact",
                "legacy_shared_grid_exact",
            )
        ):
            raise RuntimeError("fine-evidence append-only receipt failed")
        (staging / MANIFEST_NAME).write_bytes(canonical_bytes(manifest, newline=True))
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--signal-directory", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument(
        "--legacy-cache-directory", type=Path, default=DEFAULT_LEGACY_CACHE
    )
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path, manifest = materialize(
        union_directory=args.union_directory,
        signal_directory=args.signal_directory,
        legacy_cache_directory=args.legacy_cache_directory,
        tusz_root=args.tusz_root,
        output_directory=args.output_directory,
        append_limit=args.append_limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": "FINE_EVIDENCE_IDENTITY_V12_EXTENDED",
                "path": str(path),
                "manifest_sha256": file_sha256(path / MANIFEST_NAME),
                "event_count": manifest["event_count"],
                "legacy_reused_event_count": manifest["legacy_reused_event_count"],
                "newly_computed_event_count": manifest[
                    "newly_computed_event_count"
                ],
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
