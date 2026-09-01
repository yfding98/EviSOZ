#!/usr/bin/env python3
"""Materialize the frozen LaBraM 8 s / stride 4 source-train recovery cache.

This is a development-only interface experiment.  It replays only the exact
65-patient/582-event source-train crosswalk, never reads SOZ target values, and
does not touch source-dev, source-eval, or private EEG.  Fourteen correlated
eight-second calls are inverse-coverage averaged back to one token per
absolute second; overlap windows are never serialized or counted as samples.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    _load_target_free_signal,
    load_formal_token_corpus,
)
from scripts.run_labram_frozen_h_nested_oof_v3 import (  # noqa: E402
    CAPABILITY_PATH,
    CROSSWALK_MANIFEST_SHA256,
    CROSSWALK_PATH,
    CROSSWALK_RECEIPT_SHA256,
    FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
    FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
    FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
    MASTER_BUNDLE_SHA256,
    MASTER_MANIFEST_PATH,
    MASTER_SOURCE_SHA256,
    OOF_PROTOCOL_PATH,
    PREPROCESSING_ARTIFACT_SHA256,
    PREPROCESSING_PATH,
    PREPROCESSING_PROTOCOL_SHA256,
    SIGNAL_ARTIFACT_SHA256,
    SIGNAL_PATH,
    SIGNAL_RECEIPT_SHA256,
    TOKEN_CORPUS_PATH,
    TOKEN_INDEX_SHA256,
)
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
)
from src.soz.development_reasoner_v1_1 import (  # noqa: E402
    load_development_iv_evidence_capability_v1_1,
)
from src.soz.frozen_h_crosswalk import (  # noqa: E402
    _signal_tensor_sha256,
    load_source_train_frozen_h_crosswalk,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)
from src.soz.ictal_recovery_evidence_v1_2 import (  # noqa: E402
    load_target_free_ictal_oof_protocol,
)
from src.soz.models.foundation import (  # noqa: E402
    OverlappingContextFoundationEncoder,
)
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    OfficialLaBraMEncoder,
    bind_labram_record_positions,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)


NATIVE_CONTEXT_CACHE_SCHEMA = "soz_labram_native_8s_source_train_cache_v7"
NATIVE_CONTEXT_TENSOR_NAME = "node_tokens_8s_stride4"
NATIVE_CONTEXT_COVERAGE_NAME = "coverage_counts"
NATIVE_CONTEXT_FILENAME = "tokens.safetensors"
NATIVE_CONTEXT_MANIFEST_FILENAME = "manifest.json"
NATIVE_CONTEXT_SHAPE_TAIL = (19, 60, 200)
EXPECTED_SOURCE_EVENT_COUNT = 582
EXPECTED_SOURCE_PATIENT_COUNT = 65
EXPECTED_EVENT_ORDER_SHA256 = (
    "c45fe14fc4cdc1767710aa5bc22b3dce4cb08caa340f9e99a035bf134e59d434"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING_PATH = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT_PATH = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_receipt_payload(encoder: OfficialLaBraMEncoder) -> dict[str, object]:
    payload = asdict(encoder.receipt)
    payload["semantic_channels"] = list(payload["semantic_channels"])
    payload["position_names"] = list(payload["position_names"])
    payload["position_ids"] = list(payload["position_ids"])
    return payload


def _absolute_directory(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    if not result.is_dir():
        raise FileNotFoundError(result)
    return result


def _new_output_directory(path: str | Path) -> Path:
    result = Path(os.path.abspath(path))
    if result.name in {"", ".", ".."} or os.path.lexists(result):
        raise FileExistsError(f"native-context output exists or is invalid: {result}")
    if not result.parent.is_dir() or result.parent.is_symlink():
        raise FileNotFoundError(result.parent)
    return result


def _load_target_free_crosswalk():
    signal = load_bound_deepsoz_signal_preflight_artifact(
        SIGNAL_PATH,
        expected_artifact_sha256=SIGNAL_ARTIFACT_SHA256,
        expected_receipt_sha256=SIGNAL_RECEIPT_SHA256,
    )
    protocol = load_target_free_ictal_oof_protocol(
        OOF_PROTOCOL_PATH,
        expected_artifact_sha256=FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
        expected_protocol_receipt_sha256=FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
    )
    capability = load_development_iv_evidence_capability_v1_1(
        CAPABILITY_PATH,
        signal,
        protocol,
        expected_manifest_sha256=FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
    )
    master = load_tusz_ictal_training_manifest(
        MASTER_MANIFEST_PATH,
        expected_bundle_manifest_sha256=MASTER_BUNDLE_SHA256,
        expected_source_manifest_sha256=MASTER_SOURCE_SHA256,
    )
    preprocessing = load_preprocessing_selection_capability(
        PREPROCESSING_PATH,
        expected_artifact_sha256=PREPROCESSING_ARTIFACT_SHA256,
        expected_protocol_receipt_sha256=PREPROCESSING_PROTOCOL_SHA256,
    )
    token_corpus = load_formal_token_corpus(
        TOKEN_CORPUS_PATH,
        expected_index_sha256=TOKEN_INDEX_SHA256,
        preprocessing_selection=preprocessing,
    )
    crosswalk = load_source_train_frozen_h_crosswalk(
        CROSSWALK_PATH,
        capability=capability,
        signal=signal,
        protocol=protocol,
        master_manifest=master,
        token_corpus=token_corpus,
        expected_manifest_sha256=CROSSWALK_MANIFEST_SHA256,
        expected_receipt_sha256=CROSSWALK_RECEIPT_SHA256,
    )
    if len(crosswalk.events) != EXPECTED_SOURCE_EVENT_COUNT:
        raise RuntimeError("source-train event roster changed")
    if len(crosswalk.patient_ids) != EXPECTED_SOURCE_PATIENT_COUNT:
        raise RuntimeError("source-train patient roster changed")
    if crosswalk.receipt["event_order_sha256"] != EXPECTED_EVENT_ORDER_SHA256:
        raise RuntimeError("source-train event order changed")
    return crosswalk, signal, master


@dataclass(frozen=True)
class LoadedNativeContextCache:
    path: Path
    manifest_sha256: str
    manifest: Mapping[str, object]
    tokens: torch.Tensor
    coverage_counts: torch.Tensor

    def __post_init__(self) -> None:
        event_count = int(self.manifest["event_count"])
        if tuple(self.tokens.shape) != (event_count, *NATIVE_CONTEXT_SHAPE_TAIL):
            raise ValueError("native-context tensor shape disagrees with manifest")
        if self.tokens.dtype != torch.float32 or self.tokens.requires_grad:
            raise TypeError("native-context tokens must be detached float32")
        if not torch.isfinite(self.tokens).all():
            raise ValueError("native-context tokens contain non-finite values")
        expected_coverage = torch.tensor([1] * 4 + [2] * 52 + [1] * 4)
        if not torch.equal(self.coverage_counts.cpu(), expected_coverage):
            raise ValueError("native-context coverage counts changed")


def load_native_context_cache(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> LoadedNativeContextCache:
    source = _absolute_directory(directory, field="native-context cache")
    if {entry.name for entry in source.iterdir()} != {
        NATIVE_CONTEXT_FILENAME,
        NATIVE_CONTEXT_MANIFEST_FILENAME,
    }:
        raise ValueError("native-context cache has unknown or missing files")
    manifest_path = source / NATIVE_CONTEXT_MANIFEST_FILENAME
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if not _SHA256_RE.fullmatch(expected_manifest_sha256) or (
        manifest_sha != expected_manifest_sha256
    ):
        raise ValueError("native-context manifest SHA-256 mismatch")
    try:
        manifest = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("native-context manifest is not strict JSON") from exc
    if not isinstance(manifest, dict) or _canonical_bytes(manifest) != raw:
        raise ValueError("native-context manifest is not canonical JSON")
    checks = {
        "schema": manifest.get("schema_version") == NATIVE_CONTEXT_CACHE_SCHEMA,
        "scope": manifest.get("model_split") == "source_train",
        "backbone": manifest.get("foundation_backbone")
        == "official_pretrained_LaBraM_Base_frozen",
        "context": manifest.get("context_seconds") == 8,
        "stride": manifest.get("stride_seconds") == 4,
        "starts": manifest.get("start_seconds") == list(range(0, 53, 4)),
        "shape": manifest.get("tensor_shape")
        == [int(manifest.get("event_count", -1)), *NATIVE_CONTEXT_SHAPE_TAIL],
        "dtype": manifest.get("tensor_dtype") == "torch.float32",
        "checkpoint": manifest.get("foundation_checkpoint_sha256")
        == AUDITED_LABRAM_BASE_SHA256,
        "modeling": manifest.get("foundation_modeling_sha256")
        == AUDITED_LABRAM_MODELING_SHA256,
        "frozen": manifest.get("foundation_trainable_parameter_count") == 0,
        "no targets": manifest.get("deepsoz_target_values_loaded") is False,
        "no dev": manifest.get("source_dev_used") is False,
        "no eval": manifest.get("source_eval_used") is False,
        "no private": manifest.get("private_used") is False,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"native-context manifest contract failed: {failed}")
    tensor_path = source / NATIVE_CONTEXT_FILENAME
    if _file_sha256(tensor_path) != manifest.get("tensor_file_sha256"):
        raise ValueError("native-context tensor file SHA-256 mismatch")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != {NATIVE_CONTEXT_TENSOR_NAME, NATIVE_CONTEXT_COVERAGE_NAME}:
        raise ValueError("native-context safetensors keys changed")
    return LoadedNativeContextCache(
        path=source,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        tokens=tensors[NATIVE_CONTEXT_TENSOR_NAME].float().contiguous(),
        coverage_counts=tensors[NATIVE_CONTEXT_COVERAGE_NAME].long().contiguous(),
    )


def materialize_native_context_cache(
    *,
    tusz_root: str | Path,
    modeling_path: str | Path,
    checkpoint_path: str | Path,
    output_directory: str | Path,
    device: str | torch.device,
    limit: int | None,
    progress_every: int = 10,
) -> LoadedNativeContextCache:
    target = _new_output_directory(output_directory)
    raw_root = _absolute_directory(tusz_root, field="TUSZ root")
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"} or execution_device.index is not None:
        raise ValueError("device must be cpu or cuda without an explicit index")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    crosswalk, signal, master = _load_target_free_crosswalk()
    total_events = len(crosswalk.events)
    if limit is None:
        selected_count = total_events
    else:
        if isinstance(limit, bool) or not 1 <= int(limit) <= total_events:
            raise ValueError("limit must lie within the source-train event roster")
        selected_count = int(limit)
    rows = tuple(crosswalk.receipt["events"][:selected_count])
    master_by_id = {event.event_id: event for event in master}

    encoder = OfficialLaBraMEncoder(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
        tile_seconds=8,
    ).to(execution_device)
    encoder.eval()
    overlapping = OverlappingContextFoundationEncoder(encoder).to(execution_device)
    overlapping.eval()
    if any(parameter.requires_grad for parameter in overlapping.parameters()):
        raise RuntimeError("LaBraM was not frozen")

    # The operator control is independent of representation values: copying
    # existing 4 s second tokens into the 8 s overlap layout and averaging must
    # reproduce every element exactly.
    existing = crosswalk.load_event_tokens(0).reshape(1, *NATIVE_CONTEXT_SHAPE_TAIL)
    reconstructed = overlapping.aggregate_existing_second_tokens(
        existing.to(execution_device)
    ).cpu()
    pipeline_control_max_abs = float((reconstructed - existing).abs().max())
    if pipeline_control_max_abs != 0.0:
        raise RuntimeError("inverse-coverage pipeline control is not exact")
    del reconstructed, existing

    tokens = torch.empty(
        (selected_count, *NATIVE_CONTEXT_SHAPE_TAIL), dtype=torch.float32
    )
    replay_rows: list[dict[str, object]] = []
    if execution_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for ordinal, row_value in enumerate(rows):
        row = dict(row_value)
        token_event_id = str(row["token_event_id"])
        master_event = master_by_id.get(token_event_id)
        if master_event is None:
            raise ValueError("crosswalk event no longer exists in master manifest")
        eeg = _load_target_free_signal(
            master_event,
            manifest=master,
            edf_root=raw_root,
            reader_factory=None,
            foundation_receipt=encoder.receipt,
        )
        checks = {
            "processed tensor": _signal_tensor_sha256(eeg)
            == row["processed_window_sha256"],
            "shape": list(eeg.shape) == row["processed_window_shape"],
            "dtype": str(eeg.dtype) == row["processed_window_dtype"],
            "preprocess": master_event.signal_preflight_receipt_sha256
            == row["tusz_signal_preflight_receipt_sha256"],
            "EDF": master_event.edf_sha256 == row["edf_sha256"],
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"raw replay failed for {token_event_id}: {failed}")
        binding = bind_labram_record_positions(
            row["labram_position_names"],
        )
        if list(binding.position_names) != row["labram_position_names"] or list(
            binding.position_ids
        ) != row["labram_position_ids"]:
            raise ValueError("record position binding changed")
        with torch.inference_mode():
            encoded = overlapping.forward_with_record_binding(
                eeg.unsqueeze(0).to(execution_device), binding
            )[0].detach().cpu().float().contiguous()
        if tuple(encoded.shape) != NATIVE_CONTEXT_SHAPE_TAIL or not torch.isfinite(
            encoded
        ).all():
            raise RuntimeError("8 s LaBraM returned invalid event tokens")
        tokens[ordinal].copy_(encoded)
        replay_rows.append(
            {
                "ordinal": ordinal,
                "evidence_event_id": row["evidence_event_id"],
                "token_event_id": token_event_id,
                "target_patient_id": row["target_patient_id"],
                "public_patient_id": row["public_patient_id"],
                "oof_fold": row["oof_fold"],
                "relative_edf_path": row["relative_edf_path"],
                "global_t0_sec": row["global_t0_sec"],
                "edf_sha256": row["edf_sha256"],
                "processed_window_sha256": row["processed_window_sha256"],
                "signal_preflight_receipt_sha256": row[
                    "tusz_signal_preflight_receipt_sha256"
                ],
                "labram_position_names": row["labram_position_names"],
                "labram_position_ids": row["labram_position_ids"],
            }
        )
        del encoded, eeg
        if execution_device.type == "cuda":
            torch.cuda.empty_cache()
        if (ordinal + 1) % progress_every == 0 or ordinal + 1 == selected_count:
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "stage": "native_context_materialization",
                        "completed": ordinal + 1,
                        "total": selected_count,
                        "elapsed_sec": elapsed,
                        "seconds_per_event": elapsed / (ordinal + 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed = time.monotonic() - started
    if not torch.isfinite(tokens).all():
        raise RuntimeError("materialized native-context tensor is non-finite")
    event_ids = [str(row["evidence_event_id"]) for row in replay_rows]
    patient_ids = sorted({str(row["target_patient_id"]) for row in replay_rows})
    full_scope = selected_count == total_events
    if full_scope and _canonical_sha256(event_ids) != EXPECTED_EVENT_ORDER_SHA256:
        raise RuntimeError("full native-context event order changed")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required") from exc
        tensor_path = staging / NATIVE_CONTEXT_FILENAME
        save_file(
            {
                NATIVE_CONTEXT_TENSOR_NAME: tokens.contiguous(),
                NATIVE_CONTEXT_COVERAGE_NAME: overlapping.coverage_counts
                .detach()
                .cpu()
                .contiguous(),
            },
            str(tensor_path),
        )
        tensor_file_sha = _file_sha256(tensor_path)
        receipt_payload = _feature_receipt_payload(encoder)
        manifest = {
            "schema_version": NATIVE_CONTEXT_CACHE_SCHEMA,
            "purpose": "post_hoc_source_train_native_context_interface_test_only",
            "development_only": True,
            "formal_promotion": False,
            "model_split": "source_train",
            "full_scope": full_scope,
            "smoke_only": not full_scope,
            "source_roster_event_count": total_events,
            "source_roster_patient_count": len(crosswalk.patient_ids),
            "event_count": selected_count,
            "patient_count": len(patient_ids),
            "event_ids": event_ids,
            "patient_ids": patient_ids,
            "events": replay_rows,
            "event_order_sha256": _canonical_sha256(event_ids),
            "patient_roster_sha256": _canonical_sha256(patient_ids),
            "crosswalk_manifest_sha256": crosswalk.manifest_sha256,
            "crosswalk_receipt_sha256": crosswalk.receipt_sha256,
            "crosswalk_full_event_order_sha256": crosswalk.receipt[
                "event_order_sha256"
            ],
            "signal_preflight_artifact_sha256": signal.artifact_sha256,
            "signal_preflight_receipt_sha256": signal.receipt_sha256,
            "master_manifest_source_sha256": master.manifest_sha256,
            "raw_replay_verified": True,
            "preprocess_config": signal.receipt["preprocess_config"],
            "foundation_backbone": "official_pretrained_LaBraM_Base_frozen",
            "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "foundation_feature_receipt": receipt_payload,
            "foundation_feature_receipt_sha256": _canonical_sha256(
                receipt_payload
            ),
            "foundation_trainable_parameter_count": 0,
            "raw_event_shape": [19, 12000],
            "sampling_frequency_hz": 200,
            "event_seconds": 60,
            "context_seconds": 8,
            "stride_seconds": 4,
            "start_seconds": list(range(0, 53, 4)),
            "call_count_per_event": 14,
            "call_input_shape": [19, 8, 200],
            "call_token_shape": [19, 8, 200],
            "aggregation": "equal_inverse_absolute_second_coverage_mean",
            "coverage_counts": [1] * 4 + [2] * 52 + [1] * 4,
            "overlap_windows_counted_as_samples": False,
            "tensor_shape": [selected_count, *NATIVE_CONTEXT_SHAPE_TAIL],
            "tensor_dtype": "torch.float32",
            "tensor_file": NATIVE_CONTEXT_FILENAME,
            "tensor_file_sha256": tensor_file_sha,
            "tensor_file_size_bytes": tensor_path.stat().st_size,
            "pipeline_control_event_id": replay_rows[0]["evidence_event_id"],
            "pipeline_control_max_abs_error": pipeline_control_max_abs,
            "materialization_device": str(execution_device),
            "elapsed_sec": elapsed,
            "seconds_per_event": elapsed / selected_count,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if execution_device.type == "cuda"
                else None
            ),
            "deepsoz_target_values_loaded": False,
            "tusz_involvement_target_values_loaded": False,
            "source_train_evidence_values_used": False,
            "source_dev_used": False,
            "source_eval_used": False,
            "private_used": False,
        }
        manifest_path = staging / NATIVE_CONTEXT_MANIFEST_FILENAME
        manifest_path.write_bytes(_canonical_bytes(manifest))
        manifest_sha = _file_sha256(manifest_path)
        loaded = load_native_context_cache(
            staging, expected_manifest_sha256=manifest_sha
        )
        if os.path.lexists(target):
            raise FileExistsError(target)
        os.rename(staging, target)
        published = True
        return load_native_context_cache(
            target, expected_manifest_sha256=loaded.manifest_sha256
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--labram-modeling-path", type=Path, default=DEFAULT_MODELING_PATH)
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
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")
    if args.preflight_only:
        crosswalk, _, _ = _load_target_free_crosswalk()
        print(
            json.dumps(
                {
                    "status": "ready_native_8s_stride4_source_train_only",
                    "event_count": len(crosswalk.events),
                    "patient_count": len(crosswalk.patient_ids),
                    "event_order_sha256": crosswalk.receipt["event_order_sha256"],
                    "context_seconds": 8,
                    "stride_seconds": 4,
                    "source_dev_used": False,
                    "source_eval_used": False,
                    "private_used": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.output_directory is None:
        raise ValueError("materialization requires --output-directory")
    artifact = materialize_native_context_cache(
        tusz_root=args.tusz_root,
        modeling_path=args.labram_modeling_path,
        checkpoint_path=args.labram_checkpoint_path,
        output_directory=args.output_directory,
        device=args.device,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": "native_context_cache_complete",
                "path": str(artifact.path),
                "manifest_sha256": artifact.manifest_sha256,
                "event_count": artifact.manifest["event_count"],
                "patient_count": artifact.manifest["patient_count"],
                "full_scope": artifact.manifest["full_scope"],
                "tensor_shape": list(artifact.tokens.shape),
                "elapsed_sec": artifact.manifest["elapsed_sec"],
                "seconds_per_event": artifact.manifest["seconds_per_event"],
                "peak_cuda_memory_bytes": artifact.manifest[
                    "peak_cuda_memory_bytes"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
