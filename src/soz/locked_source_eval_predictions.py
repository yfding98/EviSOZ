"""Sealed, target-free predictions for the one-shot source-eval analysis.

The artifact published here is the hard boundary between frozen inference and
the later DeepSOZ target release.  It contains one row per source-eval patient
and a target-free copy of the already locked 185-event roster.  It neither
imports nor accepts a target loader/path.

The v9 score row is required to be either byte-identical to its exact-anchor
row or the result of swapping exactly one frozen-graph adjacent endpoint.
Consequently, a strict loader can detect accidental post-hoc score edits
without seeing any SOZ value.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import torch

from .anchor_endpoint_features import endpoint_adjacency_edges
from .geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS, STANDARD_19
from .locked_source_eval_roster import (
    EXPECTED_SOURCE_EVAL_EVENT_COUNT,
    EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
    LOCKED_SOURCE_EVAL_MODEL_SPLIT,
    VerifiedLockedSourceEvalRoster,
    load_locked_source_eval_roster,
)


LOCKED_SOURCE_EVAL_PREDICTION_SCHEMA = (
    "soz_locked_target_free_source_eval_predictions_v1"
)
LOCKED_SOURCE_EVAL_PREDICTION_ROSTER_SCHEMA = (
    "soz_locked_target_free_source_eval_prediction_roster_v1"
)
LOCKED_SOURCE_EVAL_PREDICTION_SERIALIZATION = (
    "canonical_json_plus_safetensors_no_pickle_atomic_directory_v1"
)
LOCKED_SOURCE_EVAL_PREDICTION_PURPOSE = (
    "frozen_exact_anchor_and_v9_predictions_before_source_eval_target_release"
)
LOCKED_SOURCE_EVAL_PREDICTION_PROTOCOL = (
    "labram_locked_source_eval_protocol_v10_20260811"
)
LOCKED_SOURCE_EVAL_PREDICTION_MANIFEST = "manifest.json"
LOCKED_SOURCE_EVAL_PREDICTION_ROSTER = "roster.json"
LOCKED_SOURCE_EVAL_PREDICTION_TENSORS = "predictions.safetensors"

PZ_INDEX = CHANNEL_INDEX["PZ"]
FROZEN_PAIR_MARGIN = math.log(3.0)
FROZEN_MAX_GAP_Z = 1.0

_TENSOR_NAMES = (
    "exact_anchor_logits",
    "v9_logits",
    "deployment_mask",
    "flip_applied",
    "anchor_index",
    "candidate_index",
    "pair_margin",
    "gap_z",
)
_LINEAGE_FIELDS = frozenset(
    {
        "protocol_document_sha256",
        "exact_anchor_checkpoint_sha256",
        "v9_reranker_checkpoint_sha256",
        "source_eval_prefix_manifest_sha256",
        "source_eval_ictal_manifest_sha256",
        "source_eval_vaq_manifest_sha256",
        "producer_source_sha256",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "serialization",
        "purpose",
        "protocol",
        "model_split",
        "official_split",
        "locked_evaluation",
        "training_authorized",
        "model_selection_authorized",
        "threshold_tuning_authorized",
        "contains_soz_labels",
        "contains_deepsoz_targets_or_masks",
        "contains_tusz_channel_targets_or_masks",
        "contains_private_data",
        "target_values_loaded",
        "target_paths_accepted",
        "source_eval_label_release_used",
        "prediction_generated_before_target_release",
        "foundation_backbone",
        "foundation_trainable_parameter_count",
        "event_count",
        "patient_count",
        "channel_count",
        "standard_19",
        "pz_index",
        "pz_fixed_masked",
        "event_order_sha256",
        "patient_roster_sha256",
        "split_manifest_sha256",
        "locked_roster_artifact_sha256",
        "locked_roster_receipt_sha256",
        "lineage",
        "tensor_specs",
        "files",
    }
)
_ROSTER_FIELDS = frozenset(
    {
        "schema_version",
        "model_split",
        "official_split",
        "event_count",
        "patient_count",
        "event_order_sha256",
        "patient_roster_sha256",
        "patient_ids",
        "patient_event_counts",
        "events",
    }
)
_EVENT_FIELDS = frozenset({"ordinal", "event_id", "patient_id"})
_TENSOR_SPEC_FIELDS = frozenset({"shape", "dtype", "tensor_sha256"})
_FILE_FIELDS = frozenset({"sha256", "size_bytes"})
_EXPECTED_FILES = frozenset(
    {
        LOCKED_SOURCE_EVAL_PREDICTION_MANIFEST,
        LOCKED_SOURCE_EVAL_PREDICTION_ROSTER,
        LOCKED_SOURCE_EVAL_PREDICTION_TENSORS,
    }
)
_ADJACENT = frozenset(endpoint_adjacency_edges())
_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_TENSOR_BYTES = 16 * 1024 * 1024


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
        raise ValueError("Locked prediction metadata is not canonical JSON data") from exc


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _require_closed_fields(
    value: Mapping[str, object], expected: frozenset[str], *, field: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise ValueError(
            f"{field} violates its closed schema; "
            f"missing={sorted(set(expected)-actual)}, "
            f"unknown={sorted(actual-set(expected))}"
        )


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
        result = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(result, dict) or _canonical_json_bytes(result) != raw:
        raise ValueError(f"{field} must be a canonical JSON object")
    return result


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
    identity = lambda stat: (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError(f"{field} changed while it was read")
    return raw


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_lineage(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TypeError("prediction lineage must be an object")
    lineage = dict(value)
    _require_closed_fields(lineage, _LINEAGE_FIELDS, field="lineage")
    return {
        field: _require_sha256(lineage[field], field=f"lineage.{field}")
        for field in sorted(_LINEAGE_FIELDS)
    }


def _validate_prediction_tensors(
    values: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if set(values) != set(_TENSOR_NAMES):
        raise ValueError(
            "Locked prediction tensor keys changed; "
            f"missing={sorted(set(_TENSOR_NAMES)-set(values))}, "
            f"unknown={sorted(set(values)-set(_TENSOR_NAMES))}"
        )
    tensors = {
        name: value.detach().cpu().contiguous() for name, value in values.items()
    }
    matrix_shape = (EXPECTED_SOURCE_EVAL_PATIENT_COUNT, N_STANDARD_CHANNELS)
    vector_shape = (EXPECTED_SOURCE_EVAL_PATIENT_COUNT,)
    for name in ("exact_anchor_logits", "v9_logits"):
        value = tensors[name]
        if tuple(value.shape) != matrix_shape or value.dtype != torch.float32:
            raise TypeError(f"{name} must be float32 [21,19]")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    deployment_mask = tensors["deployment_mask"]
    if tuple(deployment_mask.shape) != matrix_shape or deployment_mask.dtype != torch.bool:
        raise TypeError("deployment_mask must be bool [21,19]")
    expected_mask = torch.ones(matrix_shape, dtype=torch.bool)
    expected_mask[:, PZ_INDEX] = False
    if not torch.equal(deployment_mask, expected_mask):
        raise ValueError("deployment_mask must mask only canonical PZ for every patient")
    if tuple(tensors["flip_applied"].shape) != vector_shape or tensors[
        "flip_applied"
    ].dtype != torch.bool:
        raise TypeError("flip_applied must be bool [21]")
    for name in ("anchor_index", "candidate_index"):
        if tuple(tensors[name].shape) != vector_shape or tensors[name].dtype != torch.int64:
            raise TypeError(f"{name} must be int64 [21]")
    for name in ("pair_margin", "gap_z"):
        if tuple(tensors[name].shape) != vector_shape or tensors[name].dtype != torch.float32:
            raise TypeError(f"{name} must be float32 [21]")
        if not torch.isfinite(tensors[name]).all():
            raise ValueError(f"{name} must be finite")

    anchor_logits = tensors["exact_anchor_logits"]
    candidate_logits = tensors["v9_logits"]
    anchor_index = tensors["anchor_index"]
    candidate_index = tensors["candidate_index"]
    applied = tensors["flip_applied"]
    masked_anchor = anchor_logits.masked_fill(~deployment_mask, -torch.inf)
    top_values = masked_anchor.max(dim=1).values
    top_set = deployment_mask & (anchor_logits == top_values[:, None])
    if not bool((top_set.sum(dim=1) == 1).all()):
        raise ValueError("every exact-anchor row must have one unique deployable Top-1")
    expected_anchor = top_set.to(torch.int64).argmax(dim=1)
    if not torch.equal(anchor_index, expected_anchor):
        raise ValueError("anchor_index disagrees with exact_anchor_logits")

    for patient in range(EXPECTED_SOURCE_EVAL_PATIENT_COUNT):
        anchor = int(anchor_index[patient].item())
        candidate = int(candidate_index[patient].item())
        flip = bool(applied[patient].item())
        if anchor == PZ_INDEX or not 0 <= anchor < N_STANDARD_CHANNELS:
            raise ValueError("anchor_index lies outside the fixed deployment mask")
        expected_row = anchor_logits[patient].clone()
        if candidate == -1:
            if flip:
                raise ValueError("a flip cannot be applied without a candidate")
            if float(tensors["pair_margin"][patient].item()) != 0.0:
                raise ValueError("an unavailable candidate must use zero pair_margin")
        else:
            edge = (min(anchor, candidate), max(anchor, candidate))
            if (
                not 0 <= candidate < N_STANDARD_CHANNELS
                or candidate == PZ_INDEX
                or edge not in _ADJACENT
            ):
                raise ValueError("candidate_index lies outside the frozen evaluable graph")
            if flip:
                if float(tensors["pair_margin"][patient].item()) < FROZEN_PAIR_MARGIN:
                    raise ValueError("an applied flip violates the frozen log(3) margin")
                if float(tensors["gap_z"][patient].item()) > FROZEN_MAX_GAP_Z:
                    raise ValueError("an applied flip violates the frozen gap-z gate")
                expected_row[anchor], expected_row[candidate] = (
                    expected_row[candidate].clone(),
                    expected_row[anchor].clone(),
                )
        if not torch.equal(candidate_logits[patient], expected_row):
            raise ValueError(
                "v9_logits must equal the anchor row or exactly one declared endpoint swap"
            )
        expected_top = candidate if flip else anchor
        masked_v9 = candidate_logits[patient].masked_fill(
            ~deployment_mask[patient], -torch.inf
        )
        if int(masked_v9.argmax().item()) != expected_top:
            raise ValueError("v9 Top-1 disagrees with the declared flip")
    return tensors


def _prediction_roster(roster: VerifiedLockedSourceEvalRoster) -> dict[str, object]:
    if type(roster) is not VerifiedLockedSourceEvalRoster:
        raise TypeError("roster must be a strictly verified locked source-eval roster")
    event_counts = {patient_id: 0 for patient_id in roster.patient_ids}
    rows = []
    for event in roster.events:
        if event.patient_id not in event_counts:
            raise ValueError("locked roster event has an undeclared patient")
        event_counts[event.patient_id] += 1
        rows.append(
            {
                "ordinal": event.ordinal,
                "event_id": event.event_id,
                "patient_id": event.patient_id,
            }
        )
    if len(rows) != EXPECTED_SOURCE_EVAL_EVENT_COUNT or any(
        count < 1 for count in event_counts.values()
    ):
        raise ValueError("locked source-eval roster is incomplete")
    return {
        "schema_version": LOCKED_SOURCE_EVAL_PREDICTION_ROSTER_SCHEMA,
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "official_split": "eval",
        "event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
        "patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
        "event_order_sha256": roster.receipt["event_order_sha256"],
        "patient_roster_sha256": roster.receipt["patient_roster_sha256"],
        "patient_ids": list(roster.patient_ids),
        "patient_event_counts": [event_counts[value] for value in roster.patient_ids],
        "events": rows,
    }


def _validate_prediction_roster(
    value: object, roster: VerifiedLockedSourceEvalRoster
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("prediction roster must be an object")
    payload = dict(value)
    _require_closed_fields(payload, _ROSTER_FIELDS, field="prediction roster")
    fixed = {
        "schema_version": LOCKED_SOURCE_EVAL_PREDICTION_ROSTER_SCHEMA,
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "official_split": "eval",
        "event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
        "patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
        "event_order_sha256": roster.receipt["event_order_sha256"],
        "patient_roster_sha256": roster.receipt["patient_roster_sha256"],
        "patient_ids": list(roster.patient_ids),
    }
    changed = tuple(field for field, expected in fixed.items() if payload[field] != expected)
    if changed:
        raise ValueError(f"prediction roster differs from locked roster: {changed}")
    counts = payload["patient_event_counts"]
    events = payload["events"]
    if (
        not isinstance(counts, list)
        or len(counts) != EXPECTED_SOURCE_EVAL_PATIENT_COUNT
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts)
        or sum(counts) != EXPECTED_SOURCE_EVAL_EVENT_COUNT
    ):
        raise ValueError("prediction patient event counts are invalid")
    if not isinstance(events, list) or len(events) != EXPECTED_SOURCE_EVAL_EVENT_COUNT:
        raise ValueError("prediction event roster is incomplete")
    observed_counts = {patient_id: 0 for patient_id in roster.patient_ids}
    for ordinal, (row_value, expected_event) in enumerate(zip(events, roster.events)):
        if not isinstance(row_value, dict):
            raise TypeError("prediction event row must be an object")
        row = dict(row_value)
        _require_closed_fields(row, _EVENT_FIELDS, field=f"events[{ordinal}]")
        expected = {
            "ordinal": ordinal,
            "event_id": expected_event.event_id,
            "patient_id": expected_event.patient_id,
        }
        if row != expected:
            raise ValueError(f"prediction event row {ordinal} changed")
        observed_counts[expected_event.patient_id] += 1
    expected_counts = [observed_counts[value] for value in roster.patient_ids]
    if counts != expected_counts:
        raise ValueError("prediction patient event counts disagree with events")
    return payload


def _tensor_specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, object]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "tensor_sha256": _tensor_sha256(name, value),
        }
        for name, value in sorted(tensors.items())
    }


def _validate_manifest(
    value: object,
    *,
    roster: VerifiedLockedSourceEvalRoster,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("prediction manifest must be an object")
    manifest = dict(value)
    _require_closed_fields(manifest, _MANIFEST_FIELDS, field="prediction manifest")
    fixed = {
        "schema_version": LOCKED_SOURCE_EVAL_PREDICTION_SCHEMA,
        "serialization": LOCKED_SOURCE_EVAL_PREDICTION_SERIALIZATION,
        "purpose": LOCKED_SOURCE_EVAL_PREDICTION_PURPOSE,
        "protocol": LOCKED_SOURCE_EVAL_PREDICTION_PROTOCOL,
        "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
        "official_split": "eval",
        "locked_evaluation": True,
        "training_authorized": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
        "contains_soz_labels": False,
        "contains_deepsoz_targets_or_masks": False,
        "contains_tusz_channel_targets_or_masks": False,
        "contains_private_data": False,
        "target_values_loaded": False,
        "target_paths_accepted": False,
        "source_eval_label_release_used": False,
        "prediction_generated_before_target_release": True,
        "foundation_backbone": "official_pretrained_labram_base_frozen",
        "foundation_trainable_parameter_count": 0,
        "event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
        "patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
        "channel_count": N_STANDARD_CHANNELS,
        "standard_19": list(STANDARD_19),
        "pz_index": PZ_INDEX,
        "pz_fixed_masked": True,
        "event_order_sha256": roster.receipt["event_order_sha256"],
        "patient_roster_sha256": roster.receipt["patient_roster_sha256"],
        "split_manifest_sha256": roster.receipt["split_manifest_sha256"],
        "locked_roster_artifact_sha256": roster.artifact_sha256,
        "locked_roster_receipt_sha256": roster.receipt_sha256,
    }
    changed = tuple(field for field, expected in fixed.items() if manifest[field] != expected)
    if changed:
        raise ValueError(f"locked prediction boundary changed: {changed}")
    manifest["lineage"] = _validate_lineage(manifest["lineage"])
    specs = manifest["tensor_specs"]
    if not isinstance(specs, dict) or set(specs) != set(_TENSOR_NAMES):
        raise ValueError("prediction manifest tensor specs changed")
    for name, spec_value in specs.items():
        if not isinstance(spec_value, dict):
            raise TypeError(f"tensor_specs.{name} must be an object")
        _require_closed_fields(spec_value, _TENSOR_SPEC_FIELDS, field=f"tensor_specs.{name}")
        _require_sha256(spec_value["tensor_sha256"], field=f"tensor_specs.{name}.tensor_sha256")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {
        LOCKED_SOURCE_EVAL_PREDICTION_ROSTER,
        LOCKED_SOURCE_EVAL_PREDICTION_TENSORS,
    }:
        raise ValueError("prediction manifest file receipts changed")
    for filename, receipt_value in files.items():
        if not isinstance(receipt_value, dict):
            raise TypeError(f"files.{filename} must be an object")
        _require_closed_fields(receipt_value, _FILE_FIELDS, field=f"files.{filename}")
        _require_sha256(receipt_value["sha256"], field=f"files.{filename}.sha256")
        size = receipt_value["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"files.{filename}.size_bytes is invalid")
    return manifest


@dataclass(frozen=True)
class VerifiedLockedSourceEvalPredictions:
    path: Path
    manifest_sha256: str
    manifest: Mapping[str, object]
    roster: Mapping[str, object]
    exact_anchor_logits: torch.Tensor
    v9_logits: torch.Tensor
    deployment_mask: torch.Tensor
    flip_applied: torch.Tensor
    anchor_index: torch.Tensor
    candidate_index: torch.Tensor
    pair_margin: torch.Tensor
    gap_z: torch.Tensor

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.roster["patient_ids"])

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(str(row["event_id"]) for row in self.roster["events"])


def publish_locked_source_eval_predictions(
    output_directory: str | Path,
    *,
    roster: VerifiedLockedSourceEvalRoster,
    tensors: Mapping[str, torch.Tensor],
    lineage: Mapping[str, str],
) -> VerifiedLockedSourceEvalPredictions:
    """Atomically publish one non-overwriting, target-free 21x19 artifact."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for locked predictions") from exc
    prediction_tensors = _validate_prediction_tensors(tensors)
    prediction_roster = _prediction_roster(roster)
    locked_lineage = _validate_lineage(dict(lineage))
    output = _absolute_no_symlink(output_directory, field="locked prediction output")
    if output.name in {"", ".", ".."}:
        raise ValueError("locked prediction output requires a concrete directory")
    if os.path.lexists(output):
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        tensor_path = temporary / LOCKED_SOURCE_EVAL_PREDICTION_TENSORS
        save_file(prediction_tensors, str(tensor_path))
        roster_raw = _canonical_json_bytes(prediction_roster)
        roster_path = temporary / LOCKED_SOURCE_EVAL_PREDICTION_ROSTER
        with roster_path.open("xb") as stream:
            stream.write(roster_raw)
            stream.flush()
            os.fsync(stream.fileno())
        tensor_size = tensor_path.stat().st_size
        if not 1 <= tensor_size <= _MAX_TENSOR_BYTES:
            raise ValueError("prediction tensor file has an invalid size")
        manifest = {
            "schema_version": LOCKED_SOURCE_EVAL_PREDICTION_SCHEMA,
            "serialization": LOCKED_SOURCE_EVAL_PREDICTION_SERIALIZATION,
            "purpose": LOCKED_SOURCE_EVAL_PREDICTION_PURPOSE,
            "protocol": LOCKED_SOURCE_EVAL_PREDICTION_PROTOCOL,
            "model_split": LOCKED_SOURCE_EVAL_MODEL_SPLIT,
            "official_split": "eval",
            "locked_evaluation": True,
            "training_authorized": False,
            "model_selection_authorized": False,
            "threshold_tuning_authorized": False,
            "contains_soz_labels": False,
            "contains_deepsoz_targets_or_masks": False,
            "contains_tusz_channel_targets_or_masks": False,
            "contains_private_data": False,
            "target_values_loaded": False,
            "target_paths_accepted": False,
            "source_eval_label_release_used": False,
            "prediction_generated_before_target_release": True,
            "foundation_backbone": "official_pretrained_labram_base_frozen",
            "foundation_trainable_parameter_count": 0,
            "event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
            "patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
            "channel_count": N_STANDARD_CHANNELS,
            "standard_19": list(STANDARD_19),
            "pz_index": PZ_INDEX,
            "pz_fixed_masked": True,
            "event_order_sha256": roster.receipt["event_order_sha256"],
            "patient_roster_sha256": roster.receipt["patient_roster_sha256"],
            "split_manifest_sha256": roster.receipt["split_manifest_sha256"],
            "locked_roster_artifact_sha256": roster.artifact_sha256,
            "locked_roster_receipt_sha256": roster.receipt_sha256,
            "lineage": locked_lineage,
            "tensor_specs": _tensor_specs(prediction_tensors),
            "files": {
                LOCKED_SOURCE_EVAL_PREDICTION_ROSTER: {
                    "sha256": _bytes_sha256(roster_raw),
                    "size_bytes": len(roster_raw),
                },
                LOCKED_SOURCE_EVAL_PREDICTION_TENSORS: {
                    "sha256": _file_sha256(tensor_path),
                    "size_bytes": tensor_size,
                },
            },
        }
        _validate_manifest(manifest, roster=roster)
        manifest_raw = _canonical_json_bytes(manifest)
        if not 1 <= len(manifest_raw) <= _MAX_JSON_BYTES:
            raise ValueError("prediction manifest has an invalid size")
        manifest_path = temporary / LOCKED_SOURCE_EVAL_PREDICTION_MANIFEST
        with manifest_path.open("xb") as stream:
            stream.write(manifest_raw)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_file(tensor_path)
        _fsync_directory(temporary)
        if os.path.lexists(output):
            raise FileExistsError(output)
        os.rename(temporary, output)
        published = True
        _fsync_directory(output.parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return load_locked_source_eval_predictions(
        output,
        expected_manifest_sha256=_bytes_sha256(manifest_raw),
        verified_roster=roster,
    )


def load_locked_source_eval_predictions(
    bundle_directory: str | Path,
    *,
    expected_manifest_sha256: str,
    verified_roster: VerifiedLockedSourceEvalRoster | None = None,
    roster_bundle: str | Path | None = None,
    expected_roster_artifact_sha256: str | None = None,
    expected_signal_artifact_sha256: str | None = None,
    expected_signal_receipt_sha256: str | None = None,
) -> VerifiedLockedSourceEvalPredictions:
    """Strictly load predictions and replay their complete locked roster."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for locked predictions") from exc
    if verified_roster is not None:
        if any(
            value is not None
            for value in (
                roster_bundle,
                expected_roster_artifact_sha256,
                expected_signal_artifact_sha256,
                expected_signal_receipt_sha256,
            )
        ):
            raise ValueError("supply either verified_roster or all roster loader inputs")
        roster = verified_roster
    else:
        if any(
            value is None
            for value in (
                roster_bundle,
                expected_roster_artifact_sha256,
                expected_signal_artifact_sha256,
                expected_signal_receipt_sha256,
            )
        ):
            raise ValueError("strict prediction loading requires the locked roster inputs")
        roster = load_locked_source_eval_roster(
            roster_bundle,  # type: ignore[arg-type]
            expected_artifact_sha256=expected_roster_artifact_sha256,  # type: ignore[arg-type]
            expected_signal_artifact_sha256=expected_signal_artifact_sha256,  # type: ignore[arg-type]
            expected_signal_receipt_sha256=expected_signal_receipt_sha256,  # type: ignore[arg-type]
        )
    if type(roster) is not VerifiedLockedSourceEvalRoster:
        raise TypeError("prediction loader requires a strictly verified roster")
    bundle = _absolute_no_symlink(bundle_directory, field="locked prediction bundle")
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("locked prediction bundle must be a regular directory")
    entries = tuple(bundle.iterdir())
    if {path.name for path in entries} != set(_EXPECTED_FILES) or len(entries) != len(
        _EXPECTED_FILES
    ):
        raise ValueError("locked prediction bundle violates its closed file schema")
    manifest_raw = _stable_file(
        bundle / LOCKED_SOURCE_EVAL_PREDICTION_MANIFEST,
        field="locked prediction manifest",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    manifest_sha = _bytes_sha256(manifest_raw)
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("locked prediction manifest SHA mismatch")
    manifest = _validate_manifest(
        _strict_json(manifest_raw, field="locked prediction manifest"),
        roster=roster,
    )
    roster_raw = _stable_file(
        bundle / LOCKED_SOURCE_EVAL_PREDICTION_ROSTER,
        field="locked prediction roster",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    tensor_path = bundle / LOCKED_SOURCE_EVAL_PREDICTION_TENSORS
    _stable_file(
        tensor_path,
        field="locked prediction tensors",
        maximum_bytes=_MAX_TENSOR_BYTES,
    )
    for filename, raw_or_none in (
        (LOCKED_SOURCE_EVAL_PREDICTION_ROSTER, roster_raw),
        (LOCKED_SOURCE_EVAL_PREDICTION_TENSORS, None),
    ):
        receipt = manifest["files"][filename]
        path = bundle / filename
        actual_sha = _bytes_sha256(raw_or_none) if raw_or_none is not None else _file_sha256(path)
        if actual_sha != receipt["sha256"] or path.stat().st_size != receipt["size_bytes"]:
            raise ValueError(f"locked prediction payload receipt mismatch: {filename}")
    prediction_roster = _validate_prediction_roster(
        _strict_json(roster_raw, field="locked prediction roster"), roster
    )
    tensors = _validate_prediction_tensors(load_file(str(tensor_path), device="cpu"))
    expected_specs = _tensor_specs(tensors)
    if manifest["tensor_specs"] != expected_specs:
        raise ValueError("locked prediction tensor specs or hashes changed")
    return VerifiedLockedSourceEvalPredictions(
        path=bundle,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        roster=prediction_roster,
        exact_anchor_logits=tensors["exact_anchor_logits"],
        v9_logits=tensors["v9_logits"],
        deployment_mask=tensors["deployment_mask"],
        flip_applied=tensors["flip_applied"],
        anchor_index=tensors["anchor_index"],
        candidate_index=tensors["candidate_index"],
        pair_margin=tensors["pair_margin"],
        gap_z=tensors["gap_z"],
    )


__all__ = [
    "FROZEN_MAX_GAP_Z",
    "FROZEN_PAIR_MARGIN",
    "LOCKED_SOURCE_EVAL_PREDICTION_MANIFEST",
    "LOCKED_SOURCE_EVAL_PREDICTION_PROTOCOL",
    "LOCKED_SOURCE_EVAL_PREDICTION_PURPOSE",
    "LOCKED_SOURCE_EVAL_PREDICTION_ROSTER",
    "LOCKED_SOURCE_EVAL_PREDICTION_SCHEMA",
    "LOCKED_SOURCE_EVAL_PREDICTION_SERIALIZATION",
    "LOCKED_SOURCE_EVAL_PREDICTION_TENSORS",
    "PZ_INDEX",
    "VerifiedLockedSourceEvalPredictions",
    "load_locked_source_eval_predictions",
    "publish_locked_source_eval_predictions",
]
