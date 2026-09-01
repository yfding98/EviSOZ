#!/usr/bin/env python3
"""Materialize target-blind frozen LaBraM block-9 prefixes for S1.

The input is the already projected TUSZ EEG-only S1 reader pack.  This script
never opens reader/adjudication labels beyond verifying that every template is
still blank, and it never opens DeepSOZ or private artifacts.  One 60-second
standard-19 event becomes 15 four-second LaBraM calls and one
``[15, 77, 200]`` block-9 prefix.  The shared cache is suitable for a frozen
head control and for a capacity-matched top-suffix PEFT candidate after S1
development labels become available.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
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
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    OfficialLaBraMEncoder,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import (  # noqa: E402
    OfficialLaBraMFrozenPrefixEncoder,
    OfficialLaBraMMinimalPEFTSuffix,
)


DEFAULT_READER_PACK = ROOT / "outputs/tusz_eeg_only_s1_reader_pack_v1_20260813"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path("/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth")
DEFAULT_OUTPUT = ROOT / "outputs/tusz_eeg_only_s1_labram_prefix_v1_20260813"

PACK_SCHEMA = "tusz_eeg_only_patient_s1_reader_pack_v1"
LOCKED_EXTENSION_PACK_SCHEMA = (
    "tusz_non_deepsoz_s1_locked_extension_reader_pack_v1"
)
SUPPORTED_PACK_SCHEMAS = frozenset({PACK_SCHEMA, LOCKED_EXTENSION_PACK_SCHEMA})
FULL_SCHEMA = "tusz_eeg_only_s1_frozen_labram_block9_prefix_v1"
SMOKE_SCHEMA = "tusz_eeg_only_s1_frozen_labram_block9_prefix_smoke_v1"
EXPECTED_PATIENT_COUNT = 60
EXPECTED_EVENT_COUNT = 433
EXPECTED_LOCKED_EXTENSION_PATIENT_COUNT = 25
EXPECTED_LOCKED_EXTENSION_EVENT_COUNT = 330
PREFIX_EVENT_SHAPE = (15, 77, 200)
TENSOR_FILENAME = "prefix.safetensors"
TENSOR_NAME = "prefix_tokens"


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    metadata = _canonical_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _safe_input_file(directory: Path, name: str) -> Path:
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"S1 reader-pack file is missing or is a symlink: {name}")
    return path.resolve(strict=True)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _require_blank_templates(directory: Path, manifest: Mapping[str, object]) -> int:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("S1 manifest lacks file roster")
    names = files.get("independent_reader_templates")
    adjudication = files.get("adjudication_templates")
    if not isinstance(names, list) or not isinstance(adjudication, list):
        raise TypeError("S1 manifest lacks annotation template roster")
    count = 0
    for name in (*names, *adjudication):
        for line in _safe_input_file(directory, str(name)).read_text(
            encoding="utf-8"
        ).splitlines():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("S1 annotation row must be an object")
            status = row.get("review_status", row.get("adjudication_status"))
            if status != "unreviewed":
                raise ValueError(
                    "S1 labels have been opened; target-blind cache materialization "
                    "must use the pre-label pack"
                )
            if row.get("candidate_positive_electrodes") != []:
                raise ValueError("Blank S1 template unexpectedly contains positives")
            states = row.get("electrode_states")
            if not isinstance(states, Mapping) or any(
                value is not None for value in states.values()
            ):
                raise ValueError("Blank S1 template unexpectedly contains electrode states")
            count += 1
    return count


def _load_pack(
    directory: Path,
    *,
    expected_patient_count: int | None = None,
    expected_event_count: int | None = None,
) -> tuple[dict[str, object], list[dict[str, str]], int]:
    root = directory.resolve(strict=True)
    manifest = _read_json(_safe_input_file(root, "manifest.json"))
    schema = manifest.get("schema_version")
    if schema not in SUPPORTED_PACK_SCHEMAS:
        raise ValueError("Unexpected S1 reader-pack schema")
    if (expected_patient_count is None) != (expected_event_count is None):
        raise ValueError("Expected S1 patient/event counts must be provided together")
    if expected_patient_count is None:
        if schema == PACK_SCHEMA:
            expected_patient_count = EXPECTED_PATIENT_COUNT
            expected_event_count = EXPECTED_EVENT_COUNT
        else:
            expected_patient_count = EXPECTED_LOCKED_EXTENSION_PATIENT_COUNT
            expected_event_count = EXPECTED_LOCKED_EXTENSION_EVENT_COUNT
    assert expected_event_count is not None
    if manifest.get("patient_count") != expected_patient_count or (
        manifest.get("event_count") != expected_event_count
    ):
        raise ValueError("S1 reader-pack scope changed")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping):
        raise TypeError("S1 reader pack lacks access receipt")
    for flag in (
        "deepsoz_target_values_loaded",
        "model_predictions_loaded",
        "private_eeg_loaded",
        "private_target_loaded",
        "automatic_soz_annotation_performed",
        "new_reader_labels_opened",
        "tusz_channel_time_target_values_used_for_selection_or_s1_labels",
        "tusz_channel_time_target_values_exported",
    ):
        if access.get(flag) is not False:
            raise ValueError(f"S1 target-blind access contract changed: {flag}")
    blank_count = _require_blank_templates(root, manifest)
    with _safe_input_file(root, "patient_linkage.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        patient_rows = list(csv.DictReader(stream))
    with _safe_input_file(root, "event_linkage.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        events = list(csv.DictReader(stream))
    patients = {row.get("patient_pseudonym", "") for row in patient_rows}
    case_ids = {row.get("case_id", "") for row in patient_rows}
    if (
        len(patient_rows) != expected_patient_count
        or len(patients) != expected_patient_count
        or "" in patients
        or len(case_ids) != expected_patient_count
        or "" in case_ids
    ):
        raise ValueError("S1 patient linkage changed")
    if len(events) != expected_event_count:
        raise ValueError("S1 event linkage changed")
    event_ids = {row.get("event_pseudonym", "") for row in events}
    if "" in event_ids or len(event_ids) != expected_event_count:
        raise ValueError("S1 event identity is empty or duplicated")
    for row in events:
        if row.get("case_id") not in case_ids or row.get("patient_pseudonym") not in patients:
            raise ValueError("S1 event no longer resolves to one patient")
    return manifest, events, blank_count


def _safe_edf(root: Path, relative_value: object) -> Path:
    relative = PurePosixPath(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("Unsafe S1 relative EDF path")
    source = root.joinpath(*relative.parts)
    for component in (source, *source.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("S1 EDF path cannot traverse a symlink")
    resolved = source.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("S1 EDF path escaped the pinned TUSZ root")
    return resolved


def _split_calls(eeg: torch.Tensor) -> torch.Tensor:
    if tuple(eeg.shape) != (19, 12000) or eeg.dtype != torch.float32:
        raise ValueError("S1 LaBraM event must be float32 [19,12000]")
    calls = eeg.reshape(19, 15, 4, 200).permute(1, 0, 2, 3).contiguous()
    if not torch.equal(calls.permute(1, 0, 2, 3).reshape(19, 12000), eeg):
        raise RuntimeError("LaBraM four-second calls do not exactly reassemble S1 EEG")
    return calls


def materialize(
    *,
    reader_pack_directory: Path,
    tusz_root: Path,
    modeling_path: Path,
    checkpoint_path: Path,
    output_directory: Path,
    device: torch.device,
    limit: int | None,
    progress_every: int,
) -> tuple[Path, Mapping[str, object]]:
    if device.type not in {"cpu", "cuda"} or device.index is not None:
        raise ValueError("device must be cpu or cuda without an explicit index")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("progress_every must be a positive integer")
    pack, events, blank_count = _load_pack(reader_pack_directory)
    config_payload = pack.get("signal_preprocessing_contract")
    if not isinstance(config_payload, Mapping) or not isinstance(
        config_payload.get("preprocess_config"), Mapping
    ):
        raise TypeError("S1 reader pack lacks signal preprocessing contract")
    config = CausalEDFConfig(**dict(config_payload["preprocess_config"]))
    if limit is None:
        selected = events
        full_scope = True
    else:
        if isinstance(limit, bool) or not 1 <= int(limit) < len(events):
            raise ValueError("--limit must be a smoke prefix shorter than the full pack")
        selected = events[: int(limit)]
        full_scope = False

    raw_root = tusz_root.resolve(strict=True)
    model_path = modeling_path.resolve(strict=True)
    checkpoint = checkpoint_path.resolve(strict=True)
    target = output_directory.absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    for immutable in (reader_pack_directory.resolve(), raw_root, model_path, checkpoint):
        if target == immutable or target in immutable.parents or immutable in target.parents:
            raise ValueError("S1 prefix output overlaps an immutable input")

    prefix_encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=model_path,
        checkpoint_path=checkpoint,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(device).eval()
    if any(parameter.requires_grad for parameter in prefix_encoder.parameters()):
        raise RuntimeError("Frozen LaBraM blocks 0-9 expose trainable parameters")

    rows: list[dict[str, object]] = []
    prefixes: list[torch.Tensor] = []
    equivalence_error: float | None = None
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ordinal, event in enumerate(selected):
        source = _safe_edf(raw_root, event["relative_edf_path"])
        loaded = load_standard19_edf_event(
            source,
            float(event["global_event_t0_sec"]),
            config=config,
        )
        calls = _split_calls(loaded.window.data).to(device)
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        with torch.inference_mode():
            prefix = prefix_encoder.forward_with_record_binding(calls, binding)
        prefix = prefix.detach().cpu().float().contiguous()
        if tuple(prefix.shape) != PREFIX_EVENT_SHAPE or not torch.isfinite(prefix).all():
            raise RuntimeError("LaBraM block-9 prefix shape/value contract failed")

        if equivalence_error is None:
            suffix = OfficialLaBraMMinimalPEFTSuffix(
                modeling_path=model_path,
                checkpoint_path=checkpoint,
                expected_sha256=AUDITED_LABRAM_BASE_SHA256,
                expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
            ).to(device).eval()
            official = OfficialLaBraMEncoder(
                modeling_path=model_path,
                checkpoint_path=checkpoint,
                expected_sha256=AUDITED_LABRAM_BASE_SHA256,
                expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
                tile_seconds=4,
                position_names=binding.position_names,
            ).to(device).eval()
            for parameter in official.parameters():
                parameter.requires_grad_(False)
            with torch.inference_mode():
                restored = suffix(prefix.to(device))
                expected = official(calls)
            equivalence_error = float((restored - expected).abs().amax().cpu())
            if equivalence_error > 1e-6:
                raise RuntimeError(
                    "Block-9 prefix does not recover official LaBraM output: "
                    f"{equivalence_error}"
                )
            del suffix, official, restored, expected

        prefixes.append(prefix)
        rows.append(
            {
                "ordinal": ordinal,
                "case_id": event["case_id"],
                "event_case_id": event["event_case_id"],
                "event_id": event["event_pseudonym"],
                "patient_id": event["patient_pseudonym"],
                "cohort": event["cohort"],
                "relative_edf_path": event["relative_edf_path"],
                "global_event_t0_sec": float(event["global_event_t0_sec"]),
                "event_anchor_semantics": event["event_anchor_semantics"],
                "edf_sha256": loaded.edf_receipt.edf_sha256,
                "signal_receipt": asdict(loaded.signal_receipt),
                "position_names": list(binding.position_names),
                "position_ids": list(binding.position_ids),
                "prefix_tensor_sha256": _tensor_sha256(prefix),
            }
        )
        position = ordinal + 1
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

    values = torch.stack(prefixes).contiguous()
    if tuple(values.shape) != (len(selected), *PREFIX_EVENT_SHAPE):
        raise RuntimeError("Complete S1 prefix tensor has the wrong shape")
    if equivalence_error is None:
        raise RuntimeError("LaBraM equivalence check did not execute")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    published = False
    try:
        tensor_path = staging / TENSOR_FILENAME
        save_file({TENSOR_NAME: values}, str(tensor_path))
        elapsed = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        manifest: dict[str, object] = {
            "schema_version": FULL_SCHEMA if full_scope else SMOKE_SCHEMA,
            "status": "target_blind_frozen_labram_block9_s1_prefix_ready",
            "full_scope": full_scope,
            "smoke_only": not full_scope,
            "patient_count": len({row["patient_id"] for row in rows}),
            "event_count": len(rows),
            "cohort_event_counts": {
                cohort: sum(row["cohort"] == cohort for row in rows)
                for cohort in ("s1_development", "s1_calibration", "s1_locked")
            },
            "events": rows,
            "event_ids": [row["event_id"] for row in rows],
            "foundation_backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "foundation_prefix_blocks": list(range(10)),
            "foundation_prefix_stop_exclusive": 10,
            "foundation_optimizer_parameter_count": 0,
            "input_channels": list(STANDARD_19),
            "input_shape_per_event": [19, 12000],
            "sampling_frequency_hz": 200.0,
            "event_interval_sec": [-12.0, 48.0],
            "call_count_per_event": 15,
            "call_duration_sec": 4.0,
            "call_input_shape": [19, 4, 200],
            "call_output_shape": [77, 200],
            "prefix_event_shape": list(PREFIX_EVENT_SHAPE),
            "prefix_tensor_shape": list(values.shape),
            "tensor_file": TENSOR_FILENAME,
            "tensor_name": TENSOR_NAME,
            "tensor_file_sha256": _file_sha256(tensor_path),
            "prefix_tensor_sha256": _tensor_sha256(values),
            "zero_adapter_official_equivalence_max_abs_error": equivalence_error,
            "zero_adapter_official_equivalence_verified": True,
            "preprocess_config": asdict(config),
            "materialization_device": str(device),
            "elapsed_sec": elapsed,
            "seconds_per_event": elapsed / len(rows),
            "peak_cuda_memory_bytes": int(peak),
            "lineage": {
                "reader_pack_manifest_sha256": _file_sha256(
                    _safe_input_file(reader_pack_directory.resolve(), "manifest.json")
                ),
                "reader_pack_path": str(reader_pack_directory.resolve()),
                "tusz_root": str(raw_root),
            },
            "access_receipt": {
                "blank_annotation_template_records_verified": blank_count,
                "completed_s1_labels_loaded": False,
                "deepsoz_artifacts_loaded": False,
                "deepsoz_targets_loaded": False,
                "tusz_channel_time_targets_loaded": False,
                "model_predictions_loaded": False,
                "private_eeg_loaded": False,
                "private_targets_loaded": False,
                "raw_public_eeg_loaded": True,
                "raw_public_event_count": len(rows),
                "training_performed": False,
                "foundation_training_performed": False,
                "reasoner_training_performed": False,
            },
        }
        (staging / "manifest.json").write_bytes(
            _canonical_bytes(manifest, newline=True)
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return target, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--reader-pack", type=Path, default=DEFAULT_READER_PACK)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--modeling-path", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path, manifest = materialize(
        reader_pack_directory=args.reader_pack,
        tusz_root=args.tusz_root,
        modeling_path=args.modeling_path,
        checkpoint_path=args.checkpoint_path,
        output_directory=args.output_directory,
        device=torch.device(args.device),
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "path": str(path),
                "event_count": manifest["event_count"],
                "full_scope": manifest["full_scope"],
                "training_performed": False,
                "s1_labels_loaded": False,
                "private_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
