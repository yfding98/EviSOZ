#!/usr/bin/env python3
"""Materialize the locked, target-free LaBraM source-eval prefix cache.

Only the externally pinned signal preflight and the corresponding raw EDFs are
opened.  The script never accepts a DeepSOZ target path, a TUSZ channel-target
path, a fitted prediction artifact, or a training option.  ``--limit`` is a
smoke-only strict roster prefix that formal loading rejects.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.frozen_h_crosswalk import _signal_tensor_sha256  # noqa: E402
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)
from src.soz.labram_source_eval_prefix import (  # noqa: E402
    EXPECTED_PREPROCESS_CONFIG_SHA256,
    EXPECTED_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
    EXPECTED_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
    EXPECTED_SOURCE_EVAL_EVENT_COUNT,
    EXPECTED_SOURCE_EVAL_EVENT_ORDER_SHA256,
    EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
    EXPECTED_SOURCE_EVAL_PATIENT_ROSTER_SHA256,
    LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE,
    publish_labram_source_eval_prefix,
)
from src.soz.locked_source_eval_roster import (  # noqa: E402
    derive_locked_source_eval_roster_receipt,
)
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import (  # noqa: E402
    OfficialLaBraMFrozenPrefixEncoder,
)


SIGNAL_PATH = ROOT / "outputs/deepsoz_signal_preflight_v2_20260809_current"
FORMAL_OUTPUT_PATH = ROOT / "outputs/labram_source_eval_prefix_v1_20260811"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING_PATH = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py"
)
DEFAULT_CHECKPOINT_PATH = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def split_source_eval_event_into_labram_calls(eeg: torch.Tensor) -> torch.Tensor:
    """Return an exact channel-major ``[15,19,4,200]`` decomposition."""

    if not isinstance(eeg, torch.Tensor) or tuple(eeg.shape) != (19, 12_000):
        raise ValueError("Source-eval EEG must have shape [19,12000]")
    if eeg.dtype != torch.float32 or eeg.requires_grad:
        raise TypeError("Source-eval EEG must be detached float32")
    if not torch.isfinite(eeg).all().item():
        raise ValueError("Source-eval EEG must be finite")
    calls = (
        eeg.contiguous()
        .reshape(19, 15, 4, 200)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    reassembled = calls.permute(1, 0, 2, 3).contiguous().reshape(19, 12_000)
    if not torch.equal(reassembled, eeg):
        raise RuntimeError("Four-second source-eval calls do not exactly reassemble EEG")
    return calls


def _absolute_directory(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    if not result.is_dir():
        raise FileNotFoundError(result)
    return result


def _guard_output_topology(target: Path, inputs: Sequence[Path]) -> None:
    output = Path(os.path.abspath(target))
    for value in inputs:
        source = Path(os.path.abspath(value))
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Source-eval prefix output overlaps immutable input {source}")


def _safe_source_file(root: Path, relative_value: object) -> tuple[str, Path]:
    relative = PurePosixPath(str(relative_value))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 5
        or relative.parts[0] != "eval"
        or relative.suffix.lower() != ".edf"
    ):
        raise ValueError("Source-eval EDF path is not canonical")
    source = root.joinpath(*relative.parts)
    for component in (source, *source.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Source-eval EDF path cannot traverse symlinks")
    resolved = source.resolve(strict=True)
    try:
        resolved_relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Source-eval EDF escaped the pinned TUSZ root") from exc
    if resolved_relative.as_posix() != relative.as_posix():
        raise ValueError("Source-eval EDF escaped the pinned TUSZ root")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return relative.as_posix(), resolved


@dataclass(frozen=True)
class PreparedSourceEvalPrefixInputs:
    rows: tuple[Mapping[str, object], ...]
    source_event_ids: tuple[str, ...]
    source_patient_ids: tuple[str, ...]
    preprocess_config: Mapping[str, object]
    lineage: Mapping[str, str]


def prepare_source_eval_prefix_inputs(
    signal_preflight_bundle: str | Path = SIGNAL_PATH,
) -> PreparedSourceEvalPrefixInputs:
    """Project the pinned signal artifact into a label-free prefix roster."""

    signal = load_bound_deepsoz_signal_preflight_artifact(
        signal_preflight_bundle,
        expected_artifact_sha256=EXPECTED_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
        expected_receipt_sha256=EXPECTED_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
    )
    locked = derive_locked_source_eval_roster_receipt(
        signal.receipt,
        signal_artifact_sha256=signal.artifact_sha256,
        signal_receipt_sha256=signal.receipt_sha256,
    )
    locked_receipt_sha = _canonical_sha256(locked)
    accepted = signal.receipt.get("events")
    if not isinstance(accepted, list):
        raise TypeError("Signal preflight lacks accepted event rows")
    signal_by_id: dict[str, Mapping[str, object]] = {}
    for value in accepted:
        if not isinstance(value, Mapping):
            raise TypeError("Signal preflight event must be an object")
        event_id = str(value["event_id"])
        if event_id in signal_by_id:
            raise ValueError("Signal preflight repeats an event ID")
        signal_by_id[event_id] = value

    locked_rows = locked.get("events")
    if not isinstance(locked_rows, list):
        raise TypeError("Locked source-eval roster lacks event rows")
    rows: list[dict[str, object]] = []
    for ordinal, locked_value in enumerate(locked_rows):
        if not isinstance(locked_value, Mapping):
            raise TypeError("Locked source-eval event must be an object")
        event_id = str(locked_value["event_id"])
        signal_row = signal_by_id.get(event_id)
        if signal_row is None:
            raise ValueError("Locked source-eval event disappeared from signal preflight")
        edf_receipt = signal_row.get("edf_receipt")
        if not isinstance(edf_receipt, Mapping):
            raise TypeError("Signal preflight event lacks an EDF receipt")
        position_names = edf_receipt.get("labram_position_names")
        position_ids = edf_receipt.get("labram_position_ids")
        if not isinstance(position_names, list) or not isinstance(position_ids, list):
            raise TypeError("Signal preflight EDF receipt lacks LaBraM position binding")
        binding = bind_labram_record_positions(position_names)
        if (
            list(binding.position_names) != position_names
            or list(binding.position_ids) != position_ids
            or binding.policy != edf_receipt.get("labram_position_binding_policy")
        ):
            raise ValueError("Signal preflight LaBraM position binding changed")
        checks = {
            "ordinal": locked_value["ordinal"] == ordinal,
            "event record": locked_value["signal_event_record_sha256"]
            == signal_row["event_record_sha256"],
            "official split": signal_row["official_split"] == "eval",
            "model split": signal_row["model_split"] == "source_eval",
            "EDF receipt": _canonical_sha256(edf_receipt)
            == signal_row["edf_receipt_sha256"],
            "signal receipt": _canonical_sha256(signal_row["signal_receipt"])
            == signal_row["signal_receipt_sha256"],
            "preprocess": signal_row["preprocess_config_sha256"]
            == EXPECTED_PREPROCESS_CONFIG_SHA256,
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"Source-eval projection failed for {event_id}: {failed}")
        rows.append(
            {
                "ordinal": ordinal,
                "event_id": event_id,
                "patient_id": str(locked_value["patient_id"]),
                "local_patient_id": str(signal_row["local_patient_id"]),
                "official_split": "eval",
                "model_split": "source_eval",
                "relative_edf_path": str(locked_value["relative_edf_path"]),
                "global_event_index": int(locked_value["global_event_index"]),
                "global_t0_sec": float(locked_value["global_t0_sec"]),
                "global_stop_sec": float(locked_value["global_stop_sec"]),
                "event_record_sha256": str(
                    locked_value["signal_event_record_sha256"]
                ),
                "edf_sha256": str(locked_value["edf_sha256"]),
                "preprocess_config_sha256": str(
                    locked_value["preprocess_config_sha256"]
                ),
                "edf_receipt_sha256": str(locked_value["edf_receipt_sha256"]),
                "signal_receipt_sha256": str(
                    locked_value["signal_receipt_sha256"]
                ),
                "processed_window_sha256": str(
                    locked_value["processed_window_sha256"]
                ),
                "processed_window_shape": list(
                    locked_value["processed_window_shape"]
                ),
                "processed_window_dtype": str(
                    locked_value["processed_window_dtype"]
                ),
                "labram_position_binding_policy": binding.policy,
                "labram_position_names": list(binding.position_names),
                "labram_position_ids": list(binding.position_ids),
                "chunk_reassembly_exact": True,
            }
        )

    source_event_ids = tuple(str(row["event_id"]) for row in rows)
    source_patient_ids = tuple(sorted({str(row["patient_id"]) for row in rows}))
    if (
        len(rows) != EXPECTED_SOURCE_EVAL_EVENT_COUNT
        or len(source_patient_ids) != EXPECTED_SOURCE_EVAL_PATIENT_COUNT
        or _canonical_sha256(source_event_ids)
        != EXPECTED_SOURCE_EVAL_EVENT_ORDER_SHA256
        or _canonical_sha256(source_patient_ids)
        != EXPECTED_SOURCE_EVAL_PATIENT_ROSTER_SHA256
    ):
        raise ValueError("Source-eval prefix roster no longer matches the pinned boundary")
    preprocess_config = signal.receipt.get("preprocess_config")
    if not isinstance(preprocess_config, Mapping):
        raise TypeError("Signal preflight lacks a preprocess configuration")
    if (
        signal.receipt.get("preprocess_config_sha256")
        != EXPECTED_PREPROCESS_CONFIG_SHA256
    ):
        raise ValueError("Source-eval preprocess configuration SHA changed")
    return PreparedSourceEvalPrefixInputs(
        rows=tuple(rows),
        source_event_ids=source_event_ids,
        source_patient_ids=source_patient_ids,
        preprocess_config=dict(preprocess_config),
        lineage={
            "signal_preflight_artifact_sha256": signal.artifact_sha256,
            "signal_preflight_receipt_sha256": signal.receipt_sha256,
            "locked_source_eval_roster_receipt_sha256": locked_receipt_sha,
            "preprocess_config_sha256": EXPECTED_PREPROCESS_CONFIG_SHA256,
        },
    )


def materialize_labram_source_eval_prefix(
    *,
    tusz_root: str | Path,
    modeling_path: str | Path,
    checkpoint_path: str | Path,
    output_directory: str | Path,
    device: str | torch.device,
    limit: int | None,
    progress_every: int = 10,
):
    raw_root = _absolute_directory(tusz_root, field="TUSZ root")
    target = Path(os.path.abspath(output_directory))
    _guard_output_topology(target, (raw_root, SIGNAL_PATH))
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"} or execution_device.index is not None:
        raise ValueError("device must be cpu or cuda without an explicit index")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if isinstance(progress_every, bool) or progress_every < 1:
        raise ValueError("progress_every must be positive")

    prepared = prepare_source_eval_prefix_inputs()
    full_scope = limit is None
    if full_scope:
        selected_count = EXPECTED_SOURCE_EVAL_EVENT_COUNT
    else:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit < EXPECTED_SOURCE_EVAL_EVENT_COUNT
        ):
            raise ValueError("--limit smoke must be an integer in [1,184]")
        selected_count = limit
    rows = prepared.rows[:selected_count]
    config = CausalEDFConfig(**dict(prepared.preprocess_config))

    prefix = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(execution_device)
    prefix.eval()
    if any(parameter.requires_grad for parameter in prefix.parameters()):
        raise RuntimeError("Frozen LaBraM source-eval prefix has trainable parameters")
    foundation_feature_receipt = prefix.receipt

    tokens = torch.empty(
        (selected_count, *LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE),
        dtype=torch.float32,
    )
    if execution_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for ordinal, row in enumerate(rows):
        relative_path, source = _safe_source_file(raw_root, row["relative_edf_path"])
        if relative_path != row["relative_edf_path"]:
            raise ValueError("Source-eval EDF path representation changed")
        loaded = load_standard19_edf_event(
            source,
            float(row["global_t0_sec"]),
            config=config,
        )
        eeg = loaded.window.data.detach().cpu().contiguous()
        loaded_binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names
        )
        checks = {
            "EDF SHA": loaded.edf_receipt.edf_sha256 == row["edf_sha256"],
            "processed SHA": _signal_tensor_sha256(eeg)
            == row["processed_window_sha256"],
            "processed shape": list(eeg.shape) == row["processed_window_shape"],
            "processed dtype": str(eeg.dtype) == row["processed_window_dtype"],
            "EDF receipt": _canonical_sha256(asdict(loaded.edf_receipt))
            == row["edf_receipt_sha256"],
            "signal receipt": _canonical_sha256(asdict(loaded.signal_receipt))
            == row["signal_receipt_sha256"],
            "position names": list(loaded_binding.position_names)
            == row["labram_position_names"],
            "position IDs": list(loaded_binding.position_ids)
            == row["labram_position_ids"],
            "position policy": loaded_binding.policy
            == row["labram_position_binding_policy"],
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Raw source-eval replay failed for {row['event_id']}: {failed}"
            )
        calls = split_source_eval_event_into_labram_calls(eeg)
        with torch.inference_mode():
            encoded = (
                prefix.forward_with_record_binding(
                    calls.to(execution_device), loaded_binding
                )
                .detach()
                .cpu()
                .float()
                .contiguous()
            )
        if (
            tuple(encoded.shape) != LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE
            or not torch.isfinite(encoded).all().item()
        ):
            raise RuntimeError("Frozen LaBraM source-eval prefix returned invalid values")
        tokens[ordinal].copy_(encoded)
        del encoded, calls, eeg, loaded
        if execution_device.type == "cuda":
            torch.cuda.empty_cache()
        if (ordinal + 1) % progress_every == 0 or ordinal + 1 == selected_count:
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "stage": "labram_source_eval_prefix_materialization",
                        "completed": ordinal + 1,
                        "total": selected_count,
                        "elapsed_sec": elapsed,
                        "seconds_per_event": elapsed / (ordinal + 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    del prefix
    if execution_device.type == "cuda":
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - started
    return publish_labram_source_eval_prefix(
        target,
        tokens=tokens,
        event_rows=rows,
        source_event_ids=prepared.source_event_ids,
        source_patient_ids=prepared.source_patient_ids,
        lineage=prepared.lineage,
        foundation_feature_receipt=foundation_feature_receipt,
        full_scope=full_scope,
        materialization_device=str(execution_device),
        elapsed_sec=elapsed,
        peak_cuda_memory_bytes=(
            int(torch.cuda.max_memory_allocated())
            if execution_device.type == "cuda"
            else None
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument(
        "--labram-modeling-path", type=Path, default=DEFAULT_MODELING_PATH
    )
    parser.add_argument(
        "--labram-checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight_only:
        prepared = prepare_source_eval_prefix_inputs()
        print(
            json.dumps(
                {
                    "stage": "locked_source_eval_prefix_preflight",
                    "status": "ready_target_free_source_eval_prefix",
                    "event_count": len(prepared.rows),
                    "patient_count": len(prepared.source_patient_ids),
                    "event_order_sha256": _canonical_sha256(
                        prepared.source_event_ids
                    ),
                    "patient_roster_sha256": _canonical_sha256(
                        prepared.source_patient_ids
                    ),
                    "preprocess_config_sha256": prepared.lineage[
                        "preprocess_config_sha256"
                    ],
                    "source_eval_eeg_loaded": False,
                    "source_eval_target_values_loaded": False,
                    "deepsoz_target_values_loaded": False,
                    "tusz_channel_target_values_loaded": False,
                    "training_performed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.output_directory is None:
        output = (
            FORMAL_OUTPUT_PATH
            if args.limit is None
            else ROOT
            / f"outputs/labram_source_eval_prefix_v1_smoke_limit{args.limit}_20260811"
        )
    else:
        output = args.output_directory
    artifact = materialize_labram_source_eval_prefix(
        tusz_root=args.tusz_root,
        modeling_path=args.labram_modeling_path,
        checkpoint_path=args.labram_checkpoint_path,
        output_directory=output,
        device=args.device,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "schema_version": artifact.manifest["schema_version"],
                "full_scope": artifact.full_scope,
                "event_count": len(artifact.events),
                "patient_count": len(artifact.patient_ids),
                "tensor_shape": list(artifact.tokens.shape),
                "source_eval_target_values_loaded": False,
                "training_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
