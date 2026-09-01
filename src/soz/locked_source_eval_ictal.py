"""Locked target-free ictal-evidence producer for source-eval.

The producer consumes only a physically isolated source-eval signal roster,
the audited official LaBraM encoder, and the frozen formal-v4 final ictal
head.  It has no target/annotation input port.  Its availability mask is
derived from hash-replayed finite physical standard-19 signal endpoints, not
from TUSZ annotation coverage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from .concept_checkpoint import (
    LoadedIctalConceptCheckpoint,
    load_ictal_concept_checkpoint,
)
from .concept_token_io import labram_feature_receipt_sha256
from .data.edf import CausalEDFConfig, load_standard19_edf_event
from .development_reasoner import pool_ictal_seconds_to_tiles
from .development_reasoner_v1_1 import (
    FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
    FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
)
from .geometry import N_STANDARD_CHANNELS, N_TCP_EDGES, STANDARD_19
from .locked_source_eval_roster import (
    EXPECTED_SOURCE_EVAL_EVENT_COUNT,
    EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
    LockedSourceEvalEvent,
    VerifiedLockedSourceEvalRoster,
    load_locked_source_eval_roster,
)
from .models.concept_heads import IctalInvolvementHead
from .models.foundation import TiledFoundationEncoder, sha256_file
from .models.labram import (
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LaBraMFeatureReceipt,
    OfficialLaBraMEncoder,
    bind_labram_record_positions,
    require_feature_receipt_position_binding,
)
from .temporal_masks import physical_node_to_edge_mask


LOCKED_SOURCE_EVAL_ICTAL_SCHEMA = "soz_locked_source_eval_ictal_evidence_v1"
LOCKED_SOURCE_EVAL_ICTAL_EVENT_SCHEMA = (
    "soz_locked_source_eval_ictal_event_roster_v1"
)
LOCKED_SOURCE_EVAL_ICTAL_PURPOSE = (
    "locked_target_free_source_eval_formal_v4_ictal_evidence"
)
LOCKED_SOURCE_EVAL_ICTAL_MANIFEST_FILENAME = "manifest.json"
LOCKED_SOURCE_EVAL_ICTAL_EVENTS_FILENAME = "events.json"
LOCKED_SOURCE_EVAL_ICTAL_TENSORS_FILENAME = "evidence.safetensors"

FORMAL_V4_FINAL_HEAD_MANIFEST_SHA256 = (
    "ff3cce555ed21475a420f9431b213d20ac477e0a77beb22e3c65fc462c350b41"
)
FORMAL_V4_FINAL_HEAD_CHECKPOINT_SHA256 = (
    "065b9e2150cd9ea3c3e43b4bb2755da0156ac11956b30c0acf23d6b4f151f990"
)
FORMAL_V4_FOUNDATION_FEATURE_RECEIPT_SHA256 = (
    "a6c2512405a5be95ea988e96d69a8f3e27870a67a2e84e1d677720c26f22a9a5"
)

N_SECONDS = 60
N_TILES = 15
N_POOL_FEATURES = 2
_FILES = frozenset(
    {
        LOCKED_SOURCE_EVAL_ICTAL_MANIFEST_FILENAME,
        LOCKED_SOURCE_EVAL_ICTAL_EVENTS_FILENAME,
        LOCKED_SOURCE_EVAL_ICTAL_TENSORS_FILENAME,
    }
)
_TENSOR_KEYS = frozenset(
    {
        "scores",
        "availability_mask",
        "pooled_scores",
        "pooled_availability_mask",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "serialization",
        "model_split",
        "official_split",
        "locked_evaluation",
        "training_authorized",
        "model_selection_authorized",
        "threshold_tuning_authorized",
        "contains_tusz_channel_targets_or_masks",
        "contains_deepsoz_targets",
        "contains_private_data",
        "target_or_annotation_paths_accepted",
        "raw_eeg_serialized",
        "foundation_tokens_serialized",
        "source_eval_label_release_used",
        "score_semantics",
        "score_transform",
        "availability_semantics",
        "pooling",
        "event_count",
        "patient_count",
        "event_order_sha256",
        "patient_roster_sha256",
        "lineage",
        "tensor_specs",
        "files",
    }
)
_LINEAGE_FIELDS = frozenset(
    {
        "locked_source_eval_roster_artifact_sha256",
        "locked_source_eval_roster_receipt_sha256",
        "signal_preflight_artifact_sha256",
        "signal_preflight_receipt_sha256",
        "formal_v4_final_head_manifest_sha256",
        "formal_v4_final_head_checkpoint_sha256",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "foundation_modeling_sha256",
    }
)
_EVENT_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "model_split",
        "event_count",
        "patient_count",
        "event_order_sha256",
        "patient_roster_sha256",
        "events",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "ordinal",
        "event_id",
        "patient_id",
        "signal_event_record_sha256",
        "processed_window_sha256",
    }
)
_FILE_RECORD_FIELDS = frozenset({"sha256", "size_bytes"})
_TENSOR_SPEC_FIELDS = frozenset({"shape", "dtype", "tensor_sha256"})
_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_TENSOR_BYTES = 256 * 1024 * 1024


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Locked ictal artifact is not canonical JSON data") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(name: str, value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode("ascii")
    digest.update(len(metadata).to_bytes(4, "little"))
    digest.update(metadata)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


def _signal_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    metadata = _canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _strict_json(raw: bytes, *, field: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"{field} contains forbidden constant {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != raw:
        raise ValueError(f"{field} is not a canonical JSON object")
    return payload


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return result


def _stable_file(path: Path, *, field: str, maximum_bytes: int) -> bytes:
    source = _absolute_no_symlink(path, field=field)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field} must be a regular file")
    before = source.stat()
    if not 1 <= before.st_size <= maximum_bytes:
        raise ValueError(f"{field} has an invalid size")
    raw = source.read_bytes()
    after = source.stat()
    def identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    if identity(before) != identity(after):
        raise RuntimeError(f"{field} changed while it was read")
    return raw


def _safe_new_directory(path: str | Path) -> Path:
    target = _absolute_no_symlink(path, field="locked ictal output")
    if target.name in {"", ".", ".."}:
        raise ValueError("Locked ictal output requires a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    return target


def _guard_output_topology(
    output_directory: str | Path,
    inputs: "LockedSourceEvalIctalInputs",
) -> Path:
    output = _absolute_no_symlink(output_directory, field="locked ictal output")
    sources = (
        inputs.roster.path,
        inputs.head_checkpoint_path,
        inputs.tusz_root,
        inputs.labram_modeling_path,
        inputs.labram_checkpoint_path,
    )
    resolved_sources = []
    for value in sources:
        if value is None:
            continue
        source = _absolute_no_symlink(value, field="locked ictal input").resolve(
            strict=True
        )
        resolved_sources.append(source)
    for source in resolved_sources:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Locked ictal output overlaps an immutable input")
    return output


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _foundation_receipt_from_checkpoint(
    checkpoint: LoadedIctalConceptCheckpoint,
) -> LaBraMFeatureReceipt:
    raw = checkpoint.metadata.get("foundation_feature_receipt")
    if not isinstance(raw, Mapping):
        raise TypeError("Formal-v4 head lacks a foundation feature receipt")
    payload = dict(raw)
    for field in ("semantic_channels", "position_names", "position_ids"):
        if not isinstance(payload.get(field), list):
            raise TypeError(f"foundation_feature_receipt.{field} must be a list")
        payload[field] = tuple(payload[field])
    try:
        receipt = LaBraMFeatureReceipt(**payload)
    except TypeError as exc:
        raise ValueError("Formal-v4 foundation feature receipt changed") from exc
    declared = checkpoint.metadata.get("foundation_feature_receipt_sha256")
    actual = labram_feature_receipt_sha256(receipt)
    if declared != actual or actual != FORMAL_V4_FOUNDATION_FEATURE_RECEIPT_SHA256:
        raise ValueError("Formal-v4 foundation feature receipt SHA changed")
    fixed = {
        "checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
        "modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
        "semantic_channels": STANDARD_19,
        "tile_seconds": 4,
        "samples_per_token": 200,
        "token_dim": 200,
    }
    changed = tuple(
        name for name, expected in fixed.items() if getattr(receipt, name) != expected
    )
    if changed:
        raise ValueError(f"Formal-v4 foundation contract changed: {changed}")
    return receipt


@dataclass(frozen=True)
class LockedSourceEvalIctalInputs:
    roster: VerifiedLockedSourceEvalRoster
    checkpoint: LoadedIctalConceptCheckpoint
    foundation_receipt: LaBraMFeatureReceipt
    head_checkpoint_path: Path
    tusz_root: Path
    labram_modeling_path: Path
    labram_checkpoint_path: Path


def load_locked_source_eval_ictal_inputs(
    *,
    roster_directory: str | Path,
    expected_roster_artifact_sha256: str,
    head_checkpoint_directory: str | Path,
    tusz_root: str | Path,
    labram_modeling_path: str | Path,
    labram_checkpoint_path: str | Path,
) -> LockedSourceEvalIctalInputs:
    """Strict pre-forward loader with no target or annotation path input."""

    roster = load_locked_source_eval_roster(
        roster_directory,
        expected_artifact_sha256=expected_roster_artifact_sha256,
        expected_signal_artifact_sha256=(
            FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256
        ),
        expected_signal_receipt_sha256=(
            FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256
        ),
    )
    if len(roster.events) != EXPECTED_SOURCE_EVAL_EVENT_COUNT or len(
        roster.patient_ids
    ) != EXPECTED_SOURCE_EVAL_PATIENT_COUNT:
        raise ValueError("Locked source-eval roster count changed")

    head_path = _absolute_no_symlink(
        head_checkpoint_directory, field="formal-v4 final head"
    )
    checkpoint = load_ictal_concept_checkpoint(
        head_path,
        expected_manifest_sha256=FORMAL_V4_FINAL_HEAD_MANIFEST_SHA256,
    )
    metadata = checkpoint.metadata
    if (
        checkpoint.manifest_sha256 != FORMAL_V4_FINAL_HEAD_MANIFEST_SHA256
        or checkpoint.checkpoint_sha256 != FORMAL_V4_FINAL_HEAD_CHECKPOINT_SHA256
        or metadata.get("oof_fold") is not None
        or metadata.get("epoch") != 19
        or metadata.get("seed") != 20260808
        or metadata.get("head_config") != {"hidden_dim": 128, "token_dim": 200}
        or type(checkpoint.head) is not IctalInvolvementHead
    ):
        raise ValueError("Checkpoint is not the frozen formal-v4 final ictal head")
    checkpoint.head.requires_grad_(False)
    checkpoint.head.eval()
    foundation_receipt = _foundation_receipt_from_checkpoint(checkpoint)

    modeling = _absolute_no_symlink(
        labram_modeling_path, field="official LaBraM modeling source"
    ).resolve(strict=True)
    foundation = _absolute_no_symlink(
        labram_checkpoint_path, field="official LaBraM checkpoint"
    ).resolve(strict=True)
    if str(modeling) != foundation_receipt.modeling_path or str(
        foundation
    ) != foundation_receipt.checkpoint_path:
        raise ValueError("LaBraM paths differ from the formal-v4 feature receipt")
    if sha256_file(modeling) != AUDITED_LABRAM_MODELING_SHA256 or sha256_file(
        foundation
    ) != AUDITED_LABRAM_BASE_SHA256:
        raise ValueError("Official LaBraM source or checkpoint SHA changed")

    root = _absolute_no_symlink(tusz_root, field="TUSZ root").resolve(strict=True)
    if not root.is_dir():
        raise ValueError("TUSZ root must be a regular directory")
    for event in roster.events:
        _resolve_event_edf(root, event.relative_edf_path)
    return LockedSourceEvalIctalInputs(
        roster=roster,
        checkpoint=checkpoint,
        foundation_receipt=foundation_receipt,
        head_checkpoint_path=head_path,
        tusz_root=root,
        labram_modeling_path=modeling,
        labram_checkpoint_path=foundation,
    )


def preflight_locked_source_eval_ictal(
    *,
    roster_directory: str | Path,
    expected_roster_artifact_sha256: str,
    head_checkpoint_directory: str | Path,
    tusz_root: str | Path,
    labram_modeling_path: str | Path,
    labram_checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
) -> dict[str, object]:
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"} or execution_device.index is not None:
        raise ValueError("Locked ictal forward supports cpu or cuda")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    inputs = load_locked_source_eval_ictal_inputs(
        roster_directory=roster_directory,
        expected_roster_artifact_sha256=expected_roster_artifact_sha256,
        head_checkpoint_directory=head_checkpoint_directory,
        tusz_root=tusz_root,
        labram_modeling_path=labram_modeling_path,
        labram_checkpoint_path=labram_checkpoint_path,
    )
    return {
        "schema_version": LOCKED_SOURCE_EVAL_ICTAL_SCHEMA,
        "status": "ready_locked_target_free_source_eval_ictal_forward",
        "event_count": len(inputs.roster.events),
        "patient_count": len(inputs.roster.patient_ids),
        "model_split": "source_eval",
        "planned_device": str(execution_device),
        "roster_artifact_sha256": inputs.roster.artifact_sha256,
        "roster_receipt_sha256": inputs.roster.receipt_sha256,
        "head_manifest_sha256": inputs.checkpoint.manifest_sha256,
        "head_checkpoint_sha256": inputs.checkpoint.checkpoint_sha256,
        "foundation_feature_receipt_sha256": (
            labram_feature_receipt_sha256(inputs.foundation_receipt)
        ),
        "foundation_trainable_parameters": 0,
        "head_trainable_parameters": 0,
        "contains_tusz_channel_targets_or_masks": False,
        "contains_deepsoz_targets": False,
        "contains_private_data": False,
        "target_or_annotation_paths_accepted": False,
        "source_eval_label_release_used": False,
        "formal_forward_run": False,
    }


def _resolve_event_edf(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise ValueError("Source-eval EDF path must be canonical POSIX relative")
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or len(relative.parts) < 2
        or relative.parts[0] != "eval"
        or relative.suffix.lower() != ".edf"
    ):
        raise ValueError("Source-eval path must identify an official eval EDF")
    path = _absolute_no_symlink(
        root.joinpath(*relative.parts), field="source-eval EDF"
    )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Source-eval EDF path escapes the TUSZ root") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError("Source-eval EDF must be a regular non-symlinked file")
    return path


def _physical_edge_second_mask(signal: torch.Tensor) -> torch.Tensor:
    if tuple(signal.shape) != (N_STANDARD_CHANNELS, 12_000):
        raise ValueError("Source-eval signal must have shape [19,12000]")
    node_seconds = torch.isfinite(
        signal.reshape(N_STANDARD_CHANNELS, N_SECONDS, 200)
    ).all(dim=-1)
    return physical_node_to_edge_mask(node_seconds.unsqueeze(0))[0].contiguous()


def _replay_target_free_signal(
    event: LockedSourceEvalEvent,
    inputs: LockedSourceEvalIctalInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if type(event) is not LockedSourceEvalEvent or event not in inputs.roster.events:
        raise TypeError("Signal replay requires an event from the strict locked roster")
    edf = _resolve_event_edf(inputs.tusz_root, event.relative_edf_path)
    config = CausalEDFConfig(**dict(inputs.roster.receipt["preprocess_config"]))
    loaded = load_standard19_edf_event(edf, event.global_t0_sec, config=config)
    checks = {
        "edf_sha256": loaded.edf_receipt.edf_sha256 == event.edf_sha256,
        "edf_receipt_sha256": _canonical_sha256(asdict(loaded.edf_receipt))
        == event.edf_receipt_sha256,
        "signal_receipt_sha256": _canonical_sha256(asdict(loaded.signal_receipt))
        == event.signal_receipt_sha256,
        "processed_window_sha256": _signal_tensor_sha256(loaded.window.data)
        == event.processed_window_sha256,
        "processed_window_shape": tuple(loaded.window.data.shape)
        == event.processed_window_shape,
        "processed_window_dtype": str(loaded.window.data.dtype)
        == event.processed_window_dtype,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Source-eval signal replay failed bindings: {failed}")
    binding = bind_labram_record_positions(
        loaded.edf_receipt.raw_channel_names,
        semantic_channels=loaded.edf_receipt.semantic_channels,
    )
    require_feature_receipt_position_binding(inputs.foundation_receipt, binding)
    signal = loaded.window.data.detach().cpu().to(torch.float32).contiguous()
    availability = _physical_edge_second_mask(signal)
    if not availability.all():
        raise ValueError("Locked roster event lacks complete physical edge-seconds")
    return signal, availability


def _validate_evidence_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if set(tensors) != set(_TENSOR_KEYS):
        raise ValueError("Locked ictal evidence tensor keys changed")
    values = {
        name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()
    }
    expected = {
        "scores": (EXPECTED_SOURCE_EVAL_EVENT_COUNT, N_TCP_EDGES, N_SECONDS),
        "availability_mask": (
            EXPECTED_SOURCE_EVAL_EVENT_COUNT,
            N_TCP_EDGES,
            N_SECONDS,
        ),
        "pooled_scores": (
            EXPECTED_SOURCE_EVAL_EVENT_COUNT,
            N_TCP_EDGES,
            N_TILES,
            N_POOL_FEATURES,
        ),
        "pooled_availability_mask": (
            EXPECTED_SOURCE_EVAL_EVENT_COUNT,
            N_TCP_EDGES,
            N_TILES,
        ),
    }
    if any(tuple(values[name].shape) != shape for name, shape in expected.items()):
        raise ValueError("Locked ictal evidence tensor shape changed")
    if (
        values["scores"].dtype != torch.float32
        or values["pooled_scores"].dtype != torch.float32
        or values["availability_mask"].dtype != torch.bool
        or values["pooled_availability_mask"].dtype != torch.bool
    ):
        raise TypeError("Locked ictal evidence tensor dtype changed")
    if not torch.isfinite(values["scores"]).all() or not torch.isfinite(
        values["pooled_scores"]
    ).all():
        raise ValueError("Locked ictal evidence contains non-finite scores")
    if torch.any((values["scores"] < 0) | (values["scores"] > 1)):
        raise ValueError("Locked ictal probabilities must lie in [0,1]")
    if not values["availability_mask"].all() or not values[
        "pooled_availability_mask"
    ].all():
        raise ValueError("Locked complete-signal availability mask must be all true")
    pooled, pooled_mask = pool_ictal_seconds_to_tiles(
        values["scores"], values["availability_mask"]
    )
    if not torch.equal(pooled_mask, values["pooled_availability_mask"]) or not (
        torch.equal(pooled, values["pooled_scores"])
    ):
        raise ValueError("Stored four-second pooling does not replay exactly")
    return values


def _run_locked_forward(
    inputs: LockedSourceEvalIctalInputs,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if device.type not in {"cpu", "cuda"} or device.index is not None:
        raise ValueError("Locked ictal forward supports cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    encoder = OfficialLaBraMEncoder(
        modeling_path=inputs.labram_modeling_path,
        checkpoint_path=inputs.labram_checkpoint_path,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
        tile_seconds=4,
        position_names=inputs.foundation_receipt.position_names,
    )
    if _canonical_json_bytes(encoder.receipt.to_dict()) != _canonical_json_bytes(
        inputs.foundation_receipt.to_dict()
    ):
        raise ValueError("Runtime LaBraM receipt differs from formal-v4 head lineage")
    encoder.requires_grad_(False).to(device).eval()
    tiled = TiledFoundationEncoder(encoder, n_calls=15).to(device).eval()
    head = inputs.checkpoint.head.requires_grad_(False).to(device).eval()
    if any(parameter.requires_grad for parameter in tiled.parameters()) or any(
        parameter.requires_grad for parameter in head.parameters()
    ):
        raise RuntimeError("Locked inference unexpectedly has trainable parameters")

    scores: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for event in inputs.roster.events:
        signal, availability = _replay_target_free_signal(event, inputs)
        with torch.inference_mode():
            tokens = tiled(signal.unsqueeze(0).to(device=device))[0]
            logits = head(tokens.unsqueeze(0))
            probability = head.probabilities(logits)[0, :, :, 0]
        if tuple(tokens.shape) != (N_STANDARD_CHANNELS, N_SECONDS, 200) or tuple(
            probability.shape
        ) != (N_TCP_EDGES, N_SECONDS):
            raise ValueError("LaBraM/head output shape changed")
        scores.append(probability.detach().cpu().to(torch.float32).contiguous())
        masks.append(availability)
        del signal, tokens, logits, probability
    second_scores = torch.stack(scores, dim=0).contiguous()
    availability_mask = torch.stack(masks, dim=0).contiguous()
    pooled_scores, pooled_mask = pool_ictal_seconds_to_tiles(
        second_scores, availability_mask
    )
    return _validate_evidence_tensors(
        {
            "scores": second_scores,
            "availability_mask": availability_mask,
            "pooled_scores": pooled_scores,
            "pooled_availability_mask": pooled_mask,
        }
    )


def _event_document(roster: VerifiedLockedSourceEvalRoster) -> dict[str, object]:
    rows = [
        {
            "ordinal": event.ordinal,
            "event_id": event.event_id,
            "patient_id": event.patient_id,
            "signal_event_record_sha256": event.signal_event_record_sha256,
            "processed_window_sha256": event.processed_window_sha256,
        }
        for event in roster.events
    ]
    return {
        "schema_version": LOCKED_SOURCE_EVAL_ICTAL_EVENT_SCHEMA,
        "model_split": "source_eval",
        "event_count": len(rows),
        "patient_count": len(roster.patient_ids),
        "event_order_sha256": roster.receipt["event_order_sha256"],
        "patient_roster_sha256": roster.receipt["patient_roster_sha256"],
        "events": rows,
    }


def _validate_event_document(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(_EVENT_DOCUMENT_FIELDS):
        raise ValueError("Locked ictal event document violates its closed schema")
    document = dict(value)
    if (
        document["schema_version"] != LOCKED_SOURCE_EVAL_ICTAL_EVENT_SCHEMA
        or document["model_split"] != "source_eval"
        or document["event_count"] != EXPECTED_SOURCE_EVAL_EVENT_COUNT
        or document["patient_count"] != EXPECTED_SOURCE_EVAL_PATIENT_COUNT
    ):
        raise ValueError("Locked ictal event document boundary changed")
    rows = document["events"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_SOURCE_EVAL_EVENT_COUNT:
        raise ValueError("Locked ictal event rows changed count")
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(_EVENT_FIELDS):
            raise ValueError("Locked ictal event row violates its closed schema")
        if row["ordinal"] != ordinal:
            raise ValueError("Locked ictal event row order changed")
        if not isinstance(row["event_id"], str) or not isinstance(
            row["patient_id"], str
        ):
            raise TypeError("Locked ictal event identity must be text")
        _require_sha256(
            row["signal_event_record_sha256"], field="signal_event_record_sha256"
        )
        _require_sha256(
            row["processed_window_sha256"], field="processed_window_sha256"
        )
    event_ids = tuple(str(row["event_id"]) for row in rows)
    patients = tuple(sorted({str(row["patient_id"]) for row in rows}))
    if len(set(event_ids)) != len(event_ids) or len(patients) != (
        EXPECTED_SOURCE_EVAL_PATIENT_COUNT
    ):
        raise ValueError("Locked ictal event or patient roster changed")
    if document["event_order_sha256"] != _canonical_sha256(event_ids) or document[
        "patient_roster_sha256"
    ] != _canonical_sha256(patients):
        raise ValueError("Locked ictal roster hash changed")
    return document


def _tensor_specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, object]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "tensor_sha256": _tensor_sha256(name, value),
        }
        for name, value in sorted(tensors.items())
    }


def _manifest_payload(
    *,
    inputs: LockedSourceEvalIctalInputs,
    events: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    files: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": LOCKED_SOURCE_EVAL_ICTAL_SCHEMA,
        "purpose": LOCKED_SOURCE_EVAL_ICTAL_PURPOSE,
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "model_split": "source_eval",
        "official_split": "eval",
        "locked_evaluation": True,
        "training_authorized": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
        "contains_tusz_channel_targets_or_masks": False,
        "contains_deepsoz_targets": False,
        "contains_private_data": False,
        "target_or_annotation_paths_accepted": False,
        "raw_eeg_serialized": False,
        "foundation_tokens_serialized": False,
        "source_eval_label_release_used": False,
        "score_semantics": "retrospective_scalp_visible_ictal_involvement_probability_not_soz",
        "score_transform": "sigmoid_of_frozen_formal_v4_final_head_logit",
        "availability_semantics": (
            "finite_hash_replayed_complete_physical_standard19_endpoints;"
            "not_annotation_coverage"
        ),
        "pooling": "existing_pool_ictal_seconds_to_tiles_complete_4s_mean_max",
        "event_count": events["event_count"],
        "patient_count": events["patient_count"],
        "event_order_sha256": events["event_order_sha256"],
        "patient_roster_sha256": events["patient_roster_sha256"],
        "lineage": {
            "locked_source_eval_roster_artifact_sha256": (
                inputs.roster.artifact_sha256
            ),
            "locked_source_eval_roster_receipt_sha256": inputs.roster.receipt_sha256,
            "signal_preflight_artifact_sha256": (
                FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256
            ),
            "signal_preflight_receipt_sha256": (
                FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256
            ),
            "formal_v4_final_head_manifest_sha256": inputs.checkpoint.manifest_sha256,
            "formal_v4_final_head_checkpoint_sha256": (
                inputs.checkpoint.checkpoint_sha256
            ),
            "foundation_feature_receipt_sha256": (
                labram_feature_receipt_sha256(inputs.foundation_receipt)
            ),
            "foundation_checkpoint_sha256": inputs.foundation_receipt.checkpoint_sha256,
            "foundation_modeling_sha256": inputs.foundation_receipt.modeling_sha256,
        },
        "tensor_specs": _tensor_specs(tensors),
        "files": dict(files),
    }


@dataclass(frozen=True)
class VerifiedLockedSourceEvalIctalArtifact:
    path: Path
    manifest_sha256: str
    manifest: Mapping[str, object]
    events: Mapping[str, object]
    scores: torch.Tensor
    availability_mask: torch.Tensor
    pooled_scores: torch.Tensor
    pooled_availability_mask: torch.Tensor

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(str(row["event_id"]) for row in self.events["events"])

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({str(row["patient_id"]) for row in self.events["events"]}))


def _validate_manifest(
    value: object,
    *,
    expected_roster_artifact_sha256: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(_MANIFEST_FIELDS):
        raise ValueError("Locked ictal manifest violates its closed schema")
    manifest = dict(value)
    fixed = {
        "schema_version": LOCKED_SOURCE_EVAL_ICTAL_SCHEMA,
        "purpose": LOCKED_SOURCE_EVAL_ICTAL_PURPOSE,
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "model_split": "source_eval",
        "official_split": "eval",
        "locked_evaluation": True,
        "training_authorized": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
        "contains_tusz_channel_targets_or_masks": False,
        "contains_deepsoz_targets": False,
        "contains_private_data": False,
        "target_or_annotation_paths_accepted": False,
        "raw_eeg_serialized": False,
        "foundation_tokens_serialized": False,
        "source_eval_label_release_used": False,
        "score_semantics": "retrospective_scalp_visible_ictal_involvement_probability_not_soz",
        "score_transform": "sigmoid_of_frozen_formal_v4_final_head_logit",
        "availability_semantics": (
            "finite_hash_replayed_complete_physical_standard19_endpoints;"
            "not_annotation_coverage"
        ),
        "pooling": "existing_pool_ictal_seconds_to_tiles_complete_4s_mean_max",
        "event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
        "patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
    }
    changed = tuple(
        field for field, expected in fixed.items() if manifest.get(field) != expected
    )
    if changed:
        raise ValueError(f"Locked ictal scientific boundary changed: {changed}")
    lineage = manifest.get("lineage")
    if not isinstance(lineage, dict) or set(lineage) != set(_LINEAGE_FIELDS):
        raise ValueError("Locked ictal lineage violates its closed schema")
    for field, value in lineage.items():
        _require_sha256(value, field=field)
    expected_lineage = {
        "locked_source_eval_roster_artifact_sha256": _require_sha256(
            expected_roster_artifact_sha256,
            field="expected_roster_artifact_sha256",
        ),
        "signal_preflight_artifact_sha256": (
            FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256
        ),
        "signal_preflight_receipt_sha256": (
            FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256
        ),
        "formal_v4_final_head_manifest_sha256": (
            FORMAL_V4_FINAL_HEAD_MANIFEST_SHA256
        ),
        "formal_v4_final_head_checkpoint_sha256": (
            FORMAL_V4_FINAL_HEAD_CHECKPOINT_SHA256
        ),
        "foundation_feature_receipt_sha256": (
            FORMAL_V4_FOUNDATION_FEATURE_RECEIPT_SHA256
        ),
        "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
        "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
    }
    failed = tuple(
        field
        for field, expected in expected_lineage.items()
        if lineage.get(field) != expected
    )
    if failed:
        raise ValueError(f"Locked ictal external lineage changed: {failed}")
    specs = manifest.get("tensor_specs")
    if not isinstance(specs, dict) or set(specs) != set(_TENSOR_KEYS):
        raise ValueError("Locked ictal tensor-spec schema changed")
    for name, spec in specs.items():
        if not isinstance(spec, dict) or set(spec) != set(_TENSOR_SPEC_FIELDS):
            raise ValueError(f"Locked ictal tensor spec changed: {name}")
        _require_sha256(spec["tensor_sha256"], field=f"{name}.tensor_sha256")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        LOCKED_SOURCE_EVAL_ICTAL_EVENTS_FILENAME,
        LOCKED_SOURCE_EVAL_ICTAL_TENSORS_FILENAME,
    }:
        raise ValueError("Locked ictal file receipt schema changed")
    for name, record in files.items():
        if not isinstance(record, dict) or set(record) != set(_FILE_RECORD_FIELDS):
            raise ValueError(f"Locked ictal file receipt changed: {name}")
        _require_sha256(record["sha256"], field=f"{name}.sha256")
    return manifest


def load_locked_source_eval_ictal_artifact(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_roster_artifact_sha256: str,
) -> VerifiedLockedSourceEvalIctalArtifact:
    """Strictly load label-free source-eval ictal evidence."""

    try:
        from safetensors.torch import load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    source = _absolute_no_symlink(directory, field="locked ictal artifact")
    if source.is_symlink() or not source.is_dir() or {
        entry.name for entry in source.iterdir()
    } != set(_FILES):
        raise ValueError("Locked ictal artifact violates its closed file schema")
    manifest_raw = _stable_file(
        source / LOCKED_SOURCE_EVAL_ICTAL_MANIFEST_FILENAME,
        field="locked ictal manifest",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Locked ictal manifest SHA mismatch")
    manifest = _validate_manifest(
        _strict_json(manifest_raw, field="locked ictal manifest"),
        expected_roster_artifact_sha256=expected_roster_artifact_sha256,
    )
    files = manifest["files"]
    raw_files: dict[str, bytes] = {}
    for name, maximum in (
        (LOCKED_SOURCE_EVAL_ICTAL_EVENTS_FILENAME, _MAX_JSON_BYTES),
        (LOCKED_SOURCE_EVAL_ICTAL_TENSORS_FILENAME, _MAX_TENSOR_BYTES),
    ):
        raw = _stable_file(
            source / name, field=f"locked ictal {name}", maximum_bytes=maximum
        )
        record = files[name]
        if len(raw) != record["size_bytes"] or hashlib.sha256(raw).hexdigest() != (
            record["sha256"]
        ):
            raise ValueError(f"Locked ictal file changed: {name}")
        raw_files[name] = raw
    events = _validate_event_document(
        _strict_json(
            raw_files[LOCKED_SOURCE_EVAL_ICTAL_EVENTS_FILENAME],
            field="locked ictal events",
        )
    )
    for field in ("event_count", "patient_count", "event_order_sha256", "patient_roster_sha256"):
        if manifest[field] != events[field]:
            raise ValueError(f"Locked ictal manifest/events disagree: {field}")
    tensors = _validate_evidence_tensors(
        load(raw_files[LOCKED_SOURCE_EVAL_ICTAL_TENSORS_FILENAME])
    )
    if _tensor_specs(tensors) != manifest["tensor_specs"]:
        raise ValueError("Locked ictal tensor specifications changed")
    return VerifiedLockedSourceEvalIctalArtifact(
        path=source,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        events=events,
        scores=tensors["scores"],
        availability_mask=tensors["availability_mask"],
        pooled_scores=tensors["pooled_scores"],
        pooled_availability_mask=tensors["pooled_availability_mask"],
    )


def _publish_locked_source_eval_ictal_artifact(
    output_directory: str | Path,
    *,
    inputs: LockedSourceEvalIctalInputs,
    tensors: Mapping[str, torch.Tensor],
) -> VerifiedLockedSourceEvalIctalArtifact:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    output = _safe_new_directory(output_directory)
    values = _validate_evidence_tensors(tensors)
    events = _validate_event_document(_event_document(inputs.roster))
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        event_path = staging / LOCKED_SOURCE_EVAL_ICTAL_EVENTS_FILENAME
        tensor_path = staging / LOCKED_SOURCE_EVAL_ICTAL_TENSORS_FILENAME
        event_path.write_bytes(_canonical_json_bytes(events))
        save_file(values, str(tensor_path))
        files = {
            path.name: {
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (event_path, tensor_path)
        }
        manifest = _manifest_payload(
            inputs=inputs, events=events, tensors=values, files=files
        )
        manifest_raw = _canonical_json_bytes(manifest)
        manifest_path = staging / LOCKED_SOURCE_EVAL_ICTAL_MANIFEST_FILENAME
        manifest_path.write_bytes(manifest_raw)
        for path in (event_path, tensor_path, manifest_path):
            _fsync_file(path)
        _fsync_directory(staging)
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        load_locked_source_eval_ictal_artifact(
            staging,
            expected_manifest_sha256=manifest_sha,
            expected_roster_artifact_sha256=inputs.roster.artifact_sha256,
        )
        if os.path.lexists(output):
            raise FileExistsError(output)
        os.rename(staging, output)
        published = True
        _fsync_directory(output.parent)
        return load_locked_source_eval_ictal_artifact(
            output,
            expected_manifest_sha256=manifest_sha,
            expected_roster_artifact_sha256=inputs.roster.artifact_sha256,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def materialize_locked_source_eval_ictal(
    *,
    roster_directory: str | Path,
    expected_roster_artifact_sha256: str,
    head_checkpoint_directory: str | Path,
    tusz_root: str | Path,
    labram_modeling_path: str | Path,
    labram_checkpoint_path: str | Path,
    output_directory: str | Path,
    device: str | torch.device = "cuda",
) -> VerifiedLockedSourceEvalIctalArtifact:
    """Run the frozen encoder/head and atomically publish target-free evidence."""

    inputs = load_locked_source_eval_ictal_inputs(
        roster_directory=roster_directory,
        expected_roster_artifact_sha256=expected_roster_artifact_sha256,
        head_checkpoint_directory=head_checkpoint_directory,
        tusz_root=tusz_root,
        labram_modeling_path=labram_modeling_path,
        labram_checkpoint_path=labram_checkpoint_path,
    )
    _guard_output_topology(output_directory, inputs)
    execution_device = torch.device(device)
    tensors = _run_locked_forward(inputs, device=execution_device)
    return _publish_locked_source_eval_ictal_artifact(
        output_directory, inputs=inputs, tensors=tensors
    )


__all__ = [
    "FORMAL_V4_FINAL_HEAD_CHECKPOINT_SHA256",
    "FORMAL_V4_FINAL_HEAD_MANIFEST_SHA256",
    "FORMAL_V4_FOUNDATION_FEATURE_RECEIPT_SHA256",
    "LOCKED_SOURCE_EVAL_ICTAL_SCHEMA",
    "LockedSourceEvalIctalInputs",
    "VerifiedLockedSourceEvalIctalArtifact",
    "load_locked_source_eval_ictal_artifact",
    "load_locked_source_eval_ictal_inputs",
    "materialize_locked_source_eval_ictal",
    "preflight_locked_source_eval_ictal",
]
