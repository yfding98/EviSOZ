#!/usr/bin/env python3
"""Target-free C-REF19/C-CAR19 robustness audit for frozen LaBraM block 9.

This command reads only a TUSZ concept-source manifest and raw EEG.  It never
loads DeepSOZ/private target values and it does not train, select, calibrate,
or alter the frozen SOZ localizer.  C-REF19 and C-CAR19 share one causal
filter/resample/crop; the latter is derived algebraically from the former.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.geometry import CHANNEL_INDEX, TCP_20_EDGES  # noqa: E402
from src.soz.models.labram import bind_labram_record_positions  # noqa: E402
from src.soz.models.labram_peft import (  # noqa: E402
    OfficialLaBraMFrozenPrefixEncoder,
)
from src.soz.preprocessing_arm_runtime import (  # noqa: E402
    CAUSAL_REFERENCE_PAIR_ROLE,
    CAUSAL_REFERENCE_PAIR_SCHEMA,
)


DEFAULT_SOURCE_RECEIPT = (
    ROOT
    / "outputs/tusz_ictal_master_manifest_v4_20260809_preflight/receipt.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/labram_reference_robustness_source_only_20260811.json"
)
OUTPUT_SCHEMA = "soz_labram_reference_robustness_source_only_v1"


def _load_json_object(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Source receipt must be a canonical regular file")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Source receipt must contain one JSON object")
    return payload


def _safe_edf(root: Path, relative_value: object) -> Path:
    relative = PurePosixPath(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("Event contains an unsafe relative EDF path")
    source = root.joinpath(*relative.parts)
    for component in (source, *source.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("EDF path cannot traverse a symbolic link")
    resolved = source.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("EDF path escaped the pinned TUSZ root")
    return resolved


def _select_one_event_per_patient(
    events: Sequence[object], *, limit: int
) -> tuple[Mapping[str, object], ...]:
    selected: list[Mapping[str, object]] = []
    seen: set[str] = set()
    rows = sorted(
        (row for row in events if isinstance(row, Mapping)),
        key=lambda row: (str(row.get("patient_id", "")), str(row.get("event_id", ""))),
    )
    for row in rows:
        patient = str(row.get("patient_id", "")).strip()
        if not patient or patient in seen:
            continue
        seen.add(patient)
        selected.append(row)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f"Requested {limit} unique patients but found {len(selected)}")
    return tuple(selected)


def _select_explicit_target_free_event_manifest(
    path: Path, *, limit: int | None
) -> tuple[tuple[Mapping[str, object], ...], dict[str, object]]:
    """Load an explicit signal-only roster without opening SOZ target values."""

    payload = _load_json_object(path)
    if payload.get("schema_version") != "soz_public_development_union_v11":
        raise ValueError("Expected the frozen public-development signal roster v11")
    access = payload.get("access_receipt")
    if not isinstance(access, Mapping):
        raise TypeError("Explicit event manifest lacks an access receipt")
    forbidden_access = (
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "source_eval_target_values_loaded",
        "prediction_artifacts_loaded",
    )
    if any(access.get(key) is not False for key in forbidden_access):
        raise ValueError("Explicit event manifest is not target-free")
    if access.get("signal_metadata_loaded") is not True:
        raise ValueError("Explicit event manifest lacks verified signal metadata")

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("Explicit event manifest lacks event rows")
    if payload.get("event_count") != len(raw_events):
        raise ValueError("Explicit event manifest event count disagrees with its rows")
    normalized: list[Mapping[str, object]] = []
    seen_events: set[str] = set()
    seen_patients: set[str] = set()
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            raise TypeError("Explicit event manifest contains a non-object row")
        event_id = str(raw.get("event_id", "")).strip()
        patient_id = str(raw.get("patient_id", "")).strip()
        relative_edf_path = str(raw.get("relative_edf_path", "")).strip()
        try:
            onset = float(raw.get("global_t0_sec"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Explicit event row has an invalid global t0") from exc
        if (
            not event_id
            or event_id in seen_events
            or not patient_id
            or not relative_edf_path
            or not math.isfinite(onset)
        ):
            raise ValueError("Explicit event row has invalid or duplicate identity")
        seen_events.add(event_id)
        seen_patients.add(patient_id)
        row = dict(raw)
        row["event_t0_sec"] = onset
        normalized.append(row)
    if payload.get("patient_count") != len(seen_patients):
        raise ValueError("Explicit event manifest patient count disagrees with its rows")
    if limit is not None:
        if isinstance(limit, bool) or int(limit) < 1:
            raise ValueError("limit must be a positive integer when provided")
        normalized = normalized[: int(limit)]
    metadata = {
        "schema_version": str(payload["schema_version"]),
        "cohort_name": str(payload.get("cohort_name", "")),
        "manifest_payload_sha256": str(payload.get("manifest_payload_sha256", "")),
        "declared_event_count": int(payload["event_count"]),
        "declared_patient_count": int(payload["patient_count"]),
    }
    return tuple(normalized), metadata


def _event_calls(window: torch.Tensor) -> torch.Tensor:
    if tuple(window.shape) != (19, 12_000):
        raise ValueError("Reference audit requires one [19,12000] event")
    return window.reshape(19, 15, 4, 200).permute(1, 0, 2, 3).contiguous()


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty or non-finite values")
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _event_metrics(
    ref_waveform: torch.Tensor,
    car_waveform: torch.Tensor,
    ref_prefix: torch.Tensor,
    car_prefix: torch.Tensor,
) -> dict[str, object]:
    if tuple(ref_prefix.shape) != (15, 77, 200) or tuple(car_prefix.shape) != (
        15,
        77,
        200,
    ):
        raise ValueError("LaBraM block-9 prefixes must have shape [15,77,200]")
    expected_car = ref_waveform - ref_waveform.mean(dim=0, keepdim=True)
    replay_error = float((car_waveform - expected_car).abs().max().item())
    car_mean_error = float(car_waveform.mean(dim=0).abs().max().item())
    edge_error = 0.0
    for left, right in TCP_20_EDGES:
        left_index = CHANNEL_INDEX[left]
        right_index = CHANNEL_INDEX[right]
        ref_edge = ref_waveform[left_index] - ref_waveform[right_index]
        car_edge = car_waveform[left_index] - car_waveform[right_index]
        edge_error = max(edge_error, float((ref_edge - car_edge).abs().max().item()))

    ref_node = ref_prefix[:, 1:].reshape(15, 19, 4, 200)
    car_node = car_prefix[:, 1:].reshape(15, 19, 4, 200)
    token_cosine = F.cosine_similarity(
        ref_node.reshape(-1, 200), car_node.reshape(-1, 200), dim=-1
    )
    channel_cosine = F.cosine_similarity(
        ref_node.permute(1, 0, 2, 3).reshape(19, -1),
        car_node.permute(1, 0, 2, 3).reshape(19, -1),
        dim=-1,
    )
    cls_cosine = F.cosine_similarity(ref_prefix[:, 0], car_prefix[:, 0], dim=-1)
    denominator = float(torch.linalg.vector_norm(car_prefix).item())
    relative_l2 = float(torch.linalg.vector_norm(ref_prefix - car_prefix).item()) / max(
        denominator, torch.finfo(car_prefix.dtype).tiny
    )
    waveform_denominator = float(torch.linalg.vector_norm(car_waveform).item())
    waveform_relative_l2 = float(
        torch.linalg.vector_norm(ref_waveform - car_waveform).item()
    ) / max(waveform_denominator, torch.finfo(car_waveform.dtype).tiny)
    return {
        "car_replay_max_abs_volts": replay_error,
        "car_channel_mean_max_abs_volts": car_mean_error,
        "bipolar_reference_invariance_max_abs_volts": edge_error,
        "waveform_ref_car_relative_l2": waveform_relative_l2,
        "block9_prefix_ref_car_relative_l2": relative_l2,
        "block9_node_token_cosine": _quantiles(token_cosine.cpu().tolist()),
        "block9_channel_cosine": _quantiles(channel_cosine.cpu().tolist()),
        "block9_cls_cosine": _quantiles(cls_cosine.cpu().tolist()),
    }


def audit(
    *,
    source_receipt: Path,
    tusz_root: Path,
    modeling_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    limit: int | None,
    event_manifest: Path | None,
    device_name: str,
    progress_every: int,
) -> dict[str, object]:
    payload = _load_json_object(source_receipt)
    if payload.get("schema_version") != "tusz_ictal_training_manifest_v4.0.0":
        raise ValueError("Expected the frozen TUSZ source-concept v4 receipt")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("TUSZ source receipt lacks event rows")
    preprocess_payload = payload.get("preprocess_config")
    if not isinstance(preprocess_payload, Mapping):
        raise TypeError("TUSZ source receipt lacks preprocessing configuration")
    primary_config = CausalEDFConfig(**dict(preprocess_payload))
    if not primary_config.apply_car19:
        raise ValueError("Frozen primary source preprocessing must be C-CAR19")
    ref_config = replace(primary_config, apply_car19=False)
    explicit_manifest_metadata: dict[str, object] | None = None
    if event_manifest is None:
        effective_limit = 8 if limit is None else limit
        if isinstance(effective_limit, bool) or int(effective_limit) < 1:
            raise ValueError("limit must be a positive integer")
        rows = _select_one_event_per_patient(raw_events, limit=int(effective_limit))
        selection_name = "lexical_first_event_per_unique_patient"
        source_dataset = "TUSZ concept-source receipt"
    else:
        rows, explicit_manifest_metadata = _select_explicit_target_free_event_manifest(
            event_manifest, limit=limit
        )
        selection_name = "explicit_target_free_event_manifest_in_declared_order"
        source_dataset = "TUSZ public-development signal roster"

    raw_root = tusz_root.resolve(strict=True)
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ValueError("TUSZ root must be a canonical directory")
    target = output_path.absolute()
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
    ).to(device)
    encoder.eval()
    dtype = next(encoder.parameters()).dtype

    event_outputs: list[dict[str, object]] = []
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        event_id = str(row.get("event_id", "")).strip()
        patient_id = str(row.get("patient_id", "")).strip()
        onset = float(row.get("event_t0_sec"))
        if not event_id or not patient_id or not math.isfinite(onset):
            raise ValueError("Selected source row has invalid event identity/timing")
        source = _safe_edf(raw_root, row.get("relative_edf_path"))
        loaded = load_standard19_edf_event(source, onset, config=ref_config)
        ref_waveform = loaded.window.data.detach().cpu().contiguous()
        car_waveform = (
            ref_waveform - ref_waveform.mean(dim=0, keepdim=True)
        ).contiguous()
        calls = torch.cat((_event_calls(ref_waveform), _event_calls(car_waveform)), dim=0)
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        with torch.no_grad():
            prefixes = encoder.forward_with_record_binding(
                calls.to(device=device, dtype=dtype), binding
            ).detach().cpu()
        metrics = _event_metrics(
            ref_waveform,
            car_waveform,
            prefixes[:15],
            prefixes[15:],
        )
        event_outputs.append(
            {
                "patient_id": patient_id,
                "event_id": event_id,
                "relative_edf_path": str(row.get("relative_edf_path")),
                "event_t0_sec": onset,
                "raw_reference_names": list(loaded.signal_receipt.source_references),
                "edf_preprocess_receipt": asdict(loaded.edf_receipt),
                "metrics": metrics,
            }
        )
        if progress_every > 0 and (index % progress_every == 0 or index == len(rows)):
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(rows),
                        "elapsed_sec": round(time.monotonic() - started, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    scalar_names = (
        "car_replay_max_abs_volts",
        "car_channel_mean_max_abs_volts",
        "bipolar_reference_invariance_max_abs_volts",
        "waveform_ref_car_relative_l2",
        "block9_prefix_ref_car_relative_l2",
    )
    aggregate: dict[str, object] = {
        name: _quantiles(
            [float(row["metrics"][name]) for row in event_outputs]  # type: ignore[index]
        )
        for name in scalar_names
    }
    for family in (
        "block9_node_token_cosine",
        "block9_channel_cosine",
        "block9_cls_cosine",
    ):
        aggregate[f"event_median_{family}"] = _quantiles(
            [
                float(row["metrics"][family]["median"])  # type: ignore[index]
                for row in event_outputs
            ]
        )

    result: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "target_free_reference_robustness_audit_only",
        "reference_pair": {
            "schema_version": CAUSAL_REFERENCE_PAIR_SCHEMA,
            "role": CAUSAL_REFERENCE_PAIR_ROLE,
            "primary": "C-CAR19",
            "sensitivity": "C-REF19",
            "shared_filter_resample_crop": True,
        },
        "access_receipt": {
            "source_dataset": source_dataset,
            "selection": selection_name,
            "selected_patient_count": len(
                {str(row["patient_id"]) for row in event_outputs}
            ),
            "selected_event_count": len(event_outputs),
            "explicit_event_manifest": explicit_manifest_metadata,
            "tusz_native_target_values_loaded": False,
            "deepsoz_target_values_loaded": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "training_performed": False,
            "model_selection_performed": False,
        },
        "device": str(device),
        "elapsed_sec": time.monotonic() - started,
        "labram_receipt": asdict(encoder.receipt),
        "preprocess_config": asdict(ref_config),
        "aggregate": aggregate,
        "events": event_outputs,
    }
    staging_fd, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(staging_fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging_name, target)
    except Exception:
        try:
            os.unlink(staging_name)
        except FileNotFoundError:
            pass
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, default=DEFAULT_SOURCE_RECEIPT)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--modeling-path", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--event-manifest",
        type=Path,
        help=(
            "Optional frozen target-free signal roster. When present, all declared "
            "events are audited unless --limit is supplied."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()
    result = audit(
        source_receipt=args.source_receipt,
        tusz_root=args.tusz_root,
        modeling_path=args.modeling_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output,
        limit=args.limit,
        event_manifest=args.event_manifest,
        device_name=args.device,
        progress_every=args.progress_every,
    )
    print(json.dumps({"output": str(args.output), "aggregate": result["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
