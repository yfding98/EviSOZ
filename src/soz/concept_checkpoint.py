"""Safe, lineage-bound checkpoints for the ictal-involvement concept head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

from .data.provenance import ConceptExtractorReceipt
from .concept_run import (
    IctalDeterminismPolicyReceipt,
    IctalTrainingRunArtifact,
    ictal_head_state_sha256,
    load_ictal_training_run_receipt,
    save_ictal_training_run_receipt,
)
from .geometry import STANDARD_19, normalize_electrode_name
from .models.concept_heads import IctalInvolvementHead
from .models.labram import LaBraMFeatureReceipt

try:
    from safetensors.numpy import load_file as _load_safetensors
    from safetensors.numpy import save_file as _save_safetensors
except ImportError as exc:  # pragma: no cover - required production dependency
    raise ImportError(
        "safetensors is required for safe ictal concept checkpoints"
    ) from exc


ICTAL_CHECKPOINT_SCHEMA = "soz_ictal_concept_checkpoint_v6"
_CHECKPOINT_FILE = "model.safetensors"
_MANIFEST_FILE = "manifest.json"
_TRAINING_RUN_DIRECTORY = "training_run"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
_HEAD_CONFIG_FIELDS = frozenset({"token_dim", "hidden_dim"})
_STATE_SPEC_FIELDS = frozenset({"shape", "dtype"})
_FOUNDATION_FIELDS = frozenset(
    {
        "checkpoint_path",
        "checkpoint_sha256",
        "modeling_path",
        "modeling_sha256",
        "encoder_tensor_count",
        "semantic_channels",
        "position_names",
        "position_ids",
        "tile_seconds",
        "pretraining_window_seconds",
        "samples_per_token",
        "token_dim",
        "input_scale_from_volts",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_format",
        "checkpoint_file",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "head_config",
        "state_specs",
        "foundation_feature_receipt",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "tusz_annotation_sha256",
        "tusz_manifest_sha256",
        "split_manifest_sha256",
        "oof_plan_receipt_sha256",
        "oof_protocol_receipt_sha256",
        "training_run_artifact_directory",
        "training_run_artifact_sha256",
        "training_run_receipt_sha256",
        "determinism_policy",
        "determinism_policy_sha256",
        "formal_token_corpus_index_sha256",
        "formal_token_corpus_training_bundle_manifest_sha256",
        "formal_token_corpus_event_roster_sha256",
        "formal_token_corpus_patient_roster_sha256",
        "formal_token_corpus_tensor_roster_sha256",
        "scaler_sha256",
        "training_target_patient_ids",
        "held_out_target_patient_ids",
        "training_target_roster_sha256",
        "held_out_target_roster_sha256",
        "oof_fold",
        "epoch",
        "seed",
    }
)
_DETERMINISM_POLICY_FIELDS = frozenset(
    {
        "execution_device_type",
        "required_cublas_workspace_config",
        "observed_cublas_workspace_config",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "schema_version",
    }
)


@dataclass(frozen=True)
class IctalConceptCheckpointArtifact:
    path: Path
    checkpoint_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.checkpoint_sha256, field="checkpoint_sha256")
        _require_sha256(self.manifest_sha256, field="manifest_sha256")

    @property
    def lineage_checkpoint_sha256(self) -> str:
        """Hash used by ConceptExtractorReceipt to bind weights and manifest."""

        return self.manifest_sha256


@dataclass(frozen=True)
class LoadedIctalConceptCheckpoint:
    """Strictly loaded head plus manifest metadata and extractor conversion."""

    head: IctalInvolvementHead
    metadata: Mapping[str, object]
    manifest_sha256: str

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.metadata["checkpoint_sha256"])

    def concept_extractor_receipt(self) -> ConceptExtractorReceipt:
        """Return the exact provenance object consumed by evidence caches."""

        return ConceptExtractorReceipt(
            concept_family="ictal_involvement",
            # The manifest hash binds the weight-file hash and every lineage
            # receipt. Using the bare weight-file hash here would allow the
            # same tensors to be paired with a different OOF/training manifest.
            checkpoint_sha256=self.manifest_sha256,
            scaler_sha256=self.metadata["scaler_sha256"],
            split_manifest_sha256=self.metadata["split_manifest_sha256"],
            oof_fold=self.metadata["oof_fold"],
            training_target_patient_ids=tuple(
                self.metadata["training_target_patient_ids"]
            ),
            held_out_target_patient_ids=tuple(
                self.metadata["held_out_target_patient_ids"]
            ),
            training_target_roster_sha256=self.metadata[
                "training_target_roster_sha256"
            ],
            held_out_target_roster_sha256=self.metadata[
                "held_out_target_roster_sha256"
            ],
        )


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _parse_canonical_json(raw: bytes) -> dict[str, object]:
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Ictal checkpoint manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Ictal checkpoint manifest must be a JSON object")
    if raw != _canonical_json(payload):
        raise ValueError("Ictal checkpoint manifest must use canonical JSON")
    return payload


def _require_exact_fields(
    payload: Mapping[str, object], expected: frozenset[str], *, label: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match the closed schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: object, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _head_config(head: IctalInvolvementHead) -> dict[str, int]:
    if not isinstance(head, IctalInvolvementHead):
        raise TypeError("Only IctalInvolvementHead can use this checkpoint format")
    token_dim = int(head.edge_tokens.token_dim)
    first_layer = head.adapter[0]
    if not isinstance(first_layer, torch.nn.Linear):
        raise TypeError("Ictal head adapter does not match the audited architecture")
    hidden_dim = int(first_layer.out_features)
    return {"token_dim": token_dim, "hidden_dim": hidden_dim}


def _expected_head(config: Mapping[str, object]) -> IctalInvolvementHead:
    _require_exact_fields(config, _HEAD_CONFIG_FIELDS, label="head_config")
    token_dim = _positive_int(config["token_dim"], field="head_config.token_dim")
    hidden_dim = _positive_int(
        config["hidden_dim"], field="head_config.hidden_dim"
    )
    return IctalInvolvementHead(token_dim=token_dim, hidden_dim=hidden_dim)


def _state_specs(state: Mapping[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
        for name, tensor in sorted(state.items())
    }


def _expected_state_specs(config: Mapping[str, object]) -> dict[str, dict[str, object]]:
    return _state_specs(_expected_head(config).state_dict())


def _validate_state_arrays(
    arrays: Mapping[str, np.ndarray], config: Mapping[str, object]
) -> dict[str, np.ndarray]:
    expected_head = _expected_head(config)
    expected_state = expected_head.state_dict()
    if set(arrays) != set(expected_state):
        raise ValueError(
            "Checkpoint tensor fields do not match IctalInvolvementHead; "
            f"missing={sorted(set(expected_state)-set(arrays))}, "
            f"unknown={sorted(set(arrays)-set(expected_state))}"
        )
    validated: dict[str, np.ndarray] = {}
    for name, expected in expected_state.items():
        array = np.asarray(arrays[name])
        expected_dtype = str(expected.dtype).removeprefix("torch.")
        if tuple(array.shape) != tuple(expected.shape):
            raise ValueError(
                f"Checkpoint tensor shape mismatch for {name}: "
                f"expected {tuple(expected.shape)}, got {array.shape}"
            )
        if str(array.dtype) != expected_dtype:
            raise TypeError(
                f"Checkpoint tensor dtype mismatch for {name}: "
                f"expected {expected_dtype}, got {array.dtype}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"Checkpoint tensor contains non-finite values: {name}")
        if not np.issubdtype(array.dtype, np.floating):
            canonical = expected.detach().cpu().numpy()
            if not np.array_equal(array, canonical):
                raise ValueError(f"Checkpoint topology buffer was modified: {name}")
        validated[name] = np.ascontiguousarray(array)
    return validated


def _state_arrays(head: IctalInvolvementHead) -> dict[str, np.ndarray]:
    config = _head_config(head)
    state = head.state_dict()
    arrays = {
        name: np.ascontiguousarray(tensor.detach().cpu().numpy())
        for name, tensor in state.items()
    }
    return _validate_state_arrays(arrays, config)


def _foundation_payload(receipt: LaBraMFeatureReceipt) -> dict[str, object]:
    if not isinstance(receipt, LaBraMFeatureReceipt):
        raise TypeError("foundation_feature_receipt must be LaBraMFeatureReceipt")
    payload = receipt.to_dict()
    # JSON arrays are canonicalized as lists rather than Python tuples.
    payload["semantic_channels"] = list(payload["semantic_channels"])
    payload["position_names"] = list(payload["position_names"])
    payload["position_ids"] = list(payload["position_ids"])
    return _validate_foundation_payload(payload)


def _validate_foundation_payload(
    payload: Mapping[str, object], *, expected_token_dim: int | None = None
) -> dict[str, object]:
    _require_exact_fields(
        payload, _FOUNDATION_FIELDS, label="foundation_feature_receipt"
    )
    normalized = dict(payload)
    for path_field in ("checkpoint_path", "modeling_path"):
        if not isinstance(normalized[path_field], str) or not normalized[
            path_field
        ].strip():
            raise ValueError(f"foundation_feature_receipt.{path_field} cannot be empty")
    normalized["checkpoint_sha256"] = _require_sha256(
        normalized["checkpoint_sha256"],
        field="foundation_feature_receipt.checkpoint_sha256",
    )
    normalized["modeling_sha256"] = _require_sha256(
        normalized["modeling_sha256"],
        field="foundation_feature_receipt.modeling_sha256",
    )
    normalized["encoder_tensor_count"] = _positive_int(
        normalized["encoder_tensor_count"],
        field="foundation_feature_receipt.encoder_tensor_count",
    )
    semantic = normalized["semantic_channels"]
    positions = normalized["position_names"]
    position_ids = normalized["position_ids"]
    if semantic != list(STANDARD_19):
        raise ValueError("Foundation semantic channels must use frozen standard-19")
    if not isinstance(positions, list) or len(positions) != len(STANDARD_19):
        raise ValueError("Foundation position_names must align with standard-19")
    if any(
        not isinstance(name, str)
        or normalize_electrode_name(name) != semantic_name
        for name, semantic_name in zip(positions, STANDARD_19)
    ):
        raise ValueError("Foundation position names are semantically misaligned")
    if (
        not isinstance(position_ids, list)
        or len(position_ids) != len(STANDARD_19)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in position_ids)
        or len(set(position_ids)) != len(position_ids)
        or any(value <= 0 for value in position_ids)
    ):
        raise ValueError("Foundation position IDs must be 19 unique positive integers")
    for field in (
        "tile_seconds",
        "pretraining_window_seconds",
        "samples_per_token",
        "token_dim",
    ):
        normalized[field] = _positive_int(
            normalized[field], field=f"foundation_feature_receipt.{field}"
        )
    if normalized["pretraining_window_seconds"] < normalized["tile_seconds"]:
        raise ValueError("Foundation tile exceeds the documented pretraining window")
    if expected_token_dim is not None and normalized["token_dim"] != expected_token_dim:
        raise ValueError("Head token_dim does not match the foundation feature receipt")
    scale = normalized["input_scale_from_volts"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("Foundation input scale must be numeric")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("Foundation input scale must be finite and positive")
    normalized["input_scale_from_volts"] = float(scale)
    return normalized


def _normalized_rosters(
    *,
    training_target_patient_ids: Sequence[object],
    held_out_target_patient_ids: Sequence[object],
    training_target_roster_sha256: str,
    held_out_target_roster_sha256: str,
    oof_fold: int | None,
    scaler_sha256: str,
    split_manifest_sha256: str,
    checkpoint_sha256: str,
) -> ConceptExtractorReceipt:
    return ConceptExtractorReceipt(
        concept_family="ictal_involvement",
        checkpoint_sha256=checkpoint_sha256,
        scaler_sha256=scaler_sha256,
        split_manifest_sha256=split_manifest_sha256,
        oof_fold=oof_fold,
        training_target_patient_ids=tuple(
            str(value) for value in training_target_patient_ids
        ),
        held_out_target_patient_ids=tuple(
            str(value) for value in held_out_target_patient_ids
        ),
        training_target_roster_sha256=training_target_roster_sha256,
        held_out_target_roster_sha256=held_out_target_roster_sha256,
    )


_RUN_CORPUS_FIELDS = (
    "formal_token_corpus_index_sha256",
    "formal_token_corpus_training_bundle_manifest_sha256",
    "formal_token_corpus_event_roster_sha256",
    "formal_token_corpus_patient_roster_sha256",
    "formal_token_corpus_tensor_roster_sha256",
)


def _reload_training_run_artifact(
    artifact: IctalTrainingRunArtifact,
) -> IctalTrainingRunArtifact:
    if not isinstance(artifact, IctalTrainingRunArtifact):
        raise TypeError("training_run_artifact must be IctalTrainingRunArtifact")
    reloaded = load_ictal_training_run_receipt(
        artifact.path,
        expected_artifact_sha256=artifact.artifact_sha256,
        expected_training_run_receipt_sha256=(
            artifact.training_run_receipt_sha256
        ),
    )
    if reloaded.training_run_receipt != artifact.training_run_receipt:
        raise ValueError("Training-run artifact object disagrees with its disk artifact")
    return reloaded


def _validate_run_for_checkpoint(
    artifact: IctalTrainingRunArtifact,
    *,
    final_head_state_sha256: str,
    foundation_feature_receipt_sha256: str,
    split_manifest_sha256: str,
    tusz_manifest_sha256: str,
    oof_plan_receipt_sha256: str,
    oof_protocol_receipt_sha256: str,
    training_target_patient_ids: Sequence[object],
    held_out_target_patient_ids: Sequence[object],
    training_target_roster_sha256: str,
    held_out_target_roster_sha256: str,
    oof_fold: int | None,
    epoch: int,
    seed: int,
) -> IctalTrainingRunArtifact:
    verified = _reload_training_run_artifact(artifact)
    run = verified.training_run_receipt
    if run.formal_token_corpus_verified is not True:
        raise ValueError("Ictal checkpoint requires a verified formal token-corpus run")
    checks = {
        "final_head_state_sha256": (
            run.final_head_state_sha256 == final_head_state_sha256
        ),
        "foundation_feature_receipt_sha256": (
            run.foundation_feature_receipt_sha256
            == foundation_feature_receipt_sha256
        ),
        "tusz_manifest_sha256": run.training_manifest_sha256 == tusz_manifest_sha256,
        "split_manifest_sha256": (
            run.split_manifest_sha256 == split_manifest_sha256
        ),
        "token_source_manifest_sha256": (
            run.token_source_manifest_sha256 == tusz_manifest_sha256
        ),
        "oof_plan_receipt_sha256": (
            run.oof_plan_receipt_sha256 == oof_plan_receipt_sha256
        ),
        "oof_protocol_receipt_sha256": (
            run.oof_protocol_receipt_sha256 == oof_protocol_receipt_sha256
        ),
        "training_target_patient_ids": (
            run.training_target_patient_ids
            == tuple(sorted(str(value) for value in training_target_patient_ids))
        ),
        "held_out_target_patient_ids": (
            run.held_out_target_patient_ids
            == tuple(sorted(str(value) for value in held_out_target_patient_ids))
        ),
        "training_target_roster_sha256": (
            run.training_target_roster_sha256
            == training_target_roster_sha256
        ),
        "held_out_target_roster_sha256": (
            run.held_out_target_roster_sha256
            == held_out_target_roster_sha256
        ),
        "oof_fold": run.oof_fold == oof_fold,
        "epoch": run.selected_epoch == epoch,
        "seed": run.config.seed == seed,
    }
    failed = tuple(field for field, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Training-run artifact disagrees with checkpoint lineage: {failed}")
    for field in _RUN_CORPUS_FIELDS:
        _require_sha256(getattr(run, field), field=f"training_run.{field}")
    return verified


def _cross_validate_embedded_training_run(
    manifest: Mapping[str, object],
    artifact: IctalTrainingRunArtifact,
) -> None:
    run = artifact.training_run_receipt
    checks = {
        "training_run_artifact_sha256": (
            artifact.artifact_sha256 == manifest["training_run_artifact_sha256"]
        ),
        "training_run_receipt_sha256": (
            run.receipt_sha256 == manifest["training_run_receipt_sha256"]
        ),
        "determinism_policy": (
            asdict(run.determinism_policy) == manifest["determinism_policy"]
        ),
        "determinism_policy_sha256": (
            run.determinism_policy_sha256
            == manifest["determinism_policy_sha256"]
        ),
        "foundation_feature_receipt_sha256": (
            run.foundation_feature_receipt_sha256
            == manifest["foundation_feature_receipt_sha256"]
        ),
        "tusz_manifest_sha256": (
            run.training_manifest_sha256 == manifest["tusz_manifest_sha256"]
        ),
        "token_source_manifest_sha256": (
            run.token_source_manifest_sha256 == manifest["tusz_manifest_sha256"]
        ),
        "oof_plan_receipt_sha256": (
            run.oof_plan_receipt_sha256 == manifest["oof_plan_receipt_sha256"]
        ),
        "oof_protocol_receipt_sha256": (
            run.oof_protocol_receipt_sha256
            == manifest["oof_protocol_receipt_sha256"]
        ),
        "training_target_patient_ids": (
            run.training_target_patient_ids
            == tuple(manifest["training_target_patient_ids"])
        ),
        "held_out_target_patient_ids": (
            run.held_out_target_patient_ids
            == tuple(manifest["held_out_target_patient_ids"])
        ),
        "training_target_roster_sha256": (
            run.training_target_roster_sha256
            == manifest["training_target_roster_sha256"]
        ),
        "held_out_target_roster_sha256": (
            run.held_out_target_roster_sha256
            == manifest["held_out_target_roster_sha256"]
        ),
        "split_manifest_sha256": (
            run.split_manifest_sha256 == manifest["split_manifest_sha256"]
        ),
        "oof_fold": run.oof_fold == manifest["oof_fold"],
        "epoch": run.selected_epoch == manifest["epoch"],
        "seed": run.config.seed == manifest["seed"],
        "formal": run.formal_token_corpus_verified is True,
    }
    checks.update(
        {
            field: getattr(run, field) == manifest[field]
            for field in _RUN_CORPUS_FIELDS
        }
    )
    failed = tuple(field for field, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Embedded training-run artifact disagrees with checkpoint manifest: {failed}"
        )


def _validate_determinism_policy_payload(
    value: object,
) -> tuple[dict[str, object], IctalDeterminismPolicyReceipt]:
    if not isinstance(value, dict):
        raise TypeError("determinism_policy must be a JSON object")
    _require_exact_fields(
        value,
        _DETERMINISM_POLICY_FIELDS,
        label="determinism_policy",
    )
    try:
        receipt = IctalDeterminismPolicyReceipt(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint determinism policy is invalid") from exc
    return asdict(receipt), receipt


def _validate_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    _require_exact_fields(payload, _MANIFEST_FIELDS, label="Ictal checkpoint manifest")
    if payload["schema_version"] != ICTAL_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported ictal checkpoint schema: {payload['schema_version']!r}")
    if payload["checkpoint_format"] != "safetensors":
        raise ValueError("Ictal checkpoint format must be safetensors")
    if payload["checkpoint_file"] != _CHECKPOINT_FILE:
        raise ValueError("Ictal checkpoint filename does not match the frozen schema")
    checkpoint_sha256 = _require_sha256(
        payload["checkpoint_sha256"], field="checkpoint_sha256"
    )
    size = _positive_int(
        payload["checkpoint_size_bytes"], field="checkpoint_size_bytes"
    )
    if size > _MAX_CHECKPOINT_BYTES:
        raise ValueError("Ictal checkpoint exceeds the maximum accepted size")
    head_config = payload["head_config"]
    if not isinstance(head_config, dict):
        raise TypeError("head_config must be a JSON object")
    expected_head = _expected_head(head_config)

    state_specs = payload["state_specs"]
    if not isinstance(state_specs, dict):
        raise TypeError("state_specs must be a JSON object")
    expected_specs = _state_specs(expected_head.state_dict())
    if set(state_specs) != set(expected_specs):
        raise ValueError("state_specs do not match IctalInvolvementHead")
    for name, spec in state_specs.items():
        if not isinstance(spec, dict):
            raise TypeError(f"State spec {name} must be a JSON object")
        _require_exact_fields(spec, _STATE_SPEC_FIELDS, label=f"State spec {name}")
        if spec != expected_specs[name]:
            raise ValueError(f"State spec mismatch for {name}")

    foundation = payload["foundation_feature_receipt"]
    if not isinstance(foundation, dict):
        raise TypeError("foundation_feature_receipt must be a JSON object")
    foundation = _validate_foundation_payload(
        foundation, expected_token_dim=int(head_config["token_dim"])
    )
    foundation_receipt_sha = _require_sha256(
        payload["foundation_feature_receipt_sha256"],
        field="foundation_feature_receipt_sha256",
    )
    if _sha256_bytes(_canonical_json(foundation)) != foundation_receipt_sha:
        raise ValueError("Foundation feature receipt SHA-256 mismatch")
    foundation_checkpoint_sha = _require_sha256(
        payload["foundation_checkpoint_sha256"],
        field="foundation_checkpoint_sha256",
    )
    if foundation_checkpoint_sha != foundation["checkpoint_sha256"]:
        raise ValueError("Foundation checkpoint SHA disagrees with its feature receipt")

    for field in (
        "tusz_annotation_sha256",
        "tusz_manifest_sha256",
        "split_manifest_sha256",
        "oof_plan_receipt_sha256",
        "oof_protocol_receipt_sha256",
        "training_run_artifact_sha256",
        "training_run_receipt_sha256",
        "determinism_policy_sha256",
        *_RUN_CORPUS_FIELDS,
        "scaler_sha256",
    ):
        _require_sha256(payload[field], field=field)
    determinism_policy, determinism_receipt = (
        _validate_determinism_policy_payload(payload["determinism_policy"])
    )
    if payload["determinism_policy_sha256"] != (
        determinism_receipt.receipt_sha256
    ):
        raise ValueError("Checkpoint determinism policy SHA mismatch")
    if payload["training_run_artifact_directory"] != _TRAINING_RUN_DIRECTORY:
        raise ValueError("training_run_artifact_directory is not canonical")
    epoch = _positive_int(payload["epoch"], field="epoch", allow_zero=True)
    seed = _positive_int(payload["seed"], field="seed", allow_zero=True)
    for roster_field in (
        "training_target_patient_ids",
        "held_out_target_patient_ids",
    ):
        roster = payload[roster_field]
        if not isinstance(roster, list) or any(not isinstance(value, str) for value in roster):
            raise TypeError(f"{roster_field} must be a JSON string list")
    extractor = _normalized_rosters(
        training_target_patient_ids=payload["training_target_patient_ids"],
        held_out_target_patient_ids=payload["held_out_target_patient_ids"],
        training_target_roster_sha256=payload["training_target_roster_sha256"],
        held_out_target_roster_sha256=payload["held_out_target_roster_sha256"],
        oof_fold=payload["oof_fold"],
        scaler_sha256=payload["scaler_sha256"],
        split_manifest_sha256=payload["split_manifest_sha256"],
        checkpoint_sha256=checkpoint_sha256,
    )
    normalized = dict(payload)
    normalized["checkpoint_sha256"] = checkpoint_sha256
    normalized["foundation_feature_receipt"] = foundation
    normalized["foundation_feature_receipt_sha256"] = foundation_receipt_sha
    normalized["foundation_checkpoint_sha256"] = foundation_checkpoint_sha
    normalized["determinism_policy"] = determinism_policy
    normalized["determinism_policy_sha256"] = (
        determinism_receipt.receipt_sha256
    )
    normalized["training_target_patient_ids"] = list(
        extractor.training_target_patient_ids
    )
    normalized["held_out_target_patient_ids"] = list(
        extractor.held_out_target_patient_ids
    )
    normalized["training_target_roster_sha256"] = (
        extractor.training_target_roster_sha256
    )
    normalized["held_out_target_roster_sha256"] = (
        extractor.held_out_target_roster_sha256
    )
    normalized["oof_fold"] = extractor.oof_fold
    normalized["epoch"] = epoch
    normalized["seed"] = seed
    return normalized


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_ictal_concept_checkpoint(
    path: str | Path,
    head: IctalInvolvementHead,
    *,
    foundation_feature_receipt: LaBraMFeatureReceipt,
    tusz_annotation_sha256: str,
    tusz_manifest_sha256: str,
    split_manifest_sha256: str,
    oof_plan_receipt_sha256: str,
    oof_protocol_receipt_sha256: str,
    training_run_artifact: IctalTrainingRunArtifact,
    scaler_sha256: str,
    training_target_patient_ids: Sequence[object],
    held_out_target_patient_ids: Sequence[object],
    training_target_roster_sha256: str,
    held_out_target_roster_sha256: str,
    oof_fold: int | None,
    epoch: int,
    seed: int,
) -> IctalConceptCheckpointArtifact:
    """Atomically publish a safe ictal-head checkpoint; overwrite is forbidden."""

    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Ictal checkpoint target already exists: {target}")
    config = _head_config(head)
    arrays = _state_arrays(head)
    foundation = _foundation_payload(foundation_feature_receipt)
    _validate_foundation_payload(foundation, expected_token_dim=config["token_dim"])
    foundation_receipt_sha256 = _sha256_bytes(_canonical_json(foundation))
    for field, value in (
        ("tusz_annotation_sha256", tusz_annotation_sha256),
        ("tusz_manifest_sha256", tusz_manifest_sha256),
        ("split_manifest_sha256", split_manifest_sha256),
        ("oof_plan_receipt_sha256", oof_plan_receipt_sha256),
        ("oof_protocol_receipt_sha256", oof_protocol_receipt_sha256),
        ("scaler_sha256", scaler_sha256),
    ):
        _require_sha256(value, field=field)
    _positive_int(epoch, field="epoch", allow_zero=True)
    _positive_int(seed, field="seed", allow_zero=True)
    # Validate roster semantics before creating temporary files.
    preflight_extractor = _normalized_rosters(
        training_target_patient_ids=training_target_patient_ids,
        held_out_target_patient_ids=held_out_target_patient_ids,
        training_target_roster_sha256=training_target_roster_sha256,
        held_out_target_roster_sha256=held_out_target_roster_sha256,
        oof_fold=oof_fold,
        scaler_sha256=scaler_sha256,
        split_manifest_sha256=split_manifest_sha256,
        checkpoint_sha256="0" * 64,
    )
    verified_run_artifact = _validate_run_for_checkpoint(
        training_run_artifact,
        final_head_state_sha256=ictal_head_state_sha256(head),
        foundation_feature_receipt_sha256=foundation_receipt_sha256,
        split_manifest_sha256=split_manifest_sha256,
        tusz_manifest_sha256=tusz_manifest_sha256,
        oof_plan_receipt_sha256=oof_plan_receipt_sha256,
        oof_protocol_receipt_sha256=oof_protocol_receipt_sha256,
        training_target_patient_ids=(
            preflight_extractor.training_target_patient_ids
        ),
        held_out_target_patient_ids=(
            preflight_extractor.held_out_target_patient_ids
        ),
        training_target_roster_sha256=(
            preflight_extractor.training_target_roster_sha256
        ),
        held_out_target_roster_sha256=(
            preflight_extractor.held_out_target_roster_sha256
        ),
        oof_fold=preflight_extractor.oof_fold,
        epoch=int(epoch),
        seed=int(seed),
    )
    training_run = verified_run_artifact.training_run_receipt

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        checkpoint_path = temporary / _CHECKPOINT_FILE
        _save_safetensors(arrays, str(checkpoint_path))
        _fsync_file(checkpoint_path)
        checkpoint_size = checkpoint_path.stat().st_size
        if checkpoint_size < 1 or checkpoint_size > _MAX_CHECKPOINT_BYTES:
            raise ValueError("Serialized ictal checkpoint has an invalid size")
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        embedded_run_artifact = save_ictal_training_run_receipt(
            temporary / _TRAINING_RUN_DIRECTORY,
            training_run,
        )
        if (
            embedded_run_artifact.artifact_sha256
            != verified_run_artifact.artifact_sha256
        ):
            raise RuntimeError("Embedded training-run artifact bytes changed")
        manifest = {
            "schema_version": ICTAL_CHECKPOINT_SCHEMA,
            "checkpoint_format": "safetensors",
            "checkpoint_file": _CHECKPOINT_FILE,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": checkpoint_size,
            "head_config": config,
            "state_specs": _expected_state_specs(config),
            "foundation_feature_receipt": foundation,
            "foundation_feature_receipt_sha256": foundation_receipt_sha256,
            "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
            "tusz_annotation_sha256": _require_sha256(
                tusz_annotation_sha256, field="tusz_annotation_sha256"
            ),
            "tusz_manifest_sha256": _require_sha256(
                tusz_manifest_sha256, field="tusz_manifest_sha256"
            ),
            "split_manifest_sha256": _require_sha256(
                split_manifest_sha256, field="split_manifest_sha256"
            ),
            "oof_plan_receipt_sha256": _require_sha256(
                oof_plan_receipt_sha256, field="oof_plan_receipt_sha256"
            ),
            "oof_protocol_receipt_sha256": _require_sha256(
                oof_protocol_receipt_sha256,
                field="oof_protocol_receipt_sha256",
            ),
            "training_run_artifact_directory": _TRAINING_RUN_DIRECTORY,
            "training_run_artifact_sha256": embedded_run_artifact.artifact_sha256,
            "training_run_receipt_sha256": training_run.receipt_sha256,
            "determinism_policy": asdict(training_run.determinism_policy),
            "determinism_policy_sha256": (
                training_run.determinism_policy_sha256
            ),
            **{
                field: getattr(training_run, field)
                for field in _RUN_CORPUS_FIELDS
            },
            "scaler_sha256": _require_sha256(
                scaler_sha256, field="scaler_sha256"
            ),
            "training_target_patient_ids": list(
                preflight_extractor.training_target_patient_ids
            ),
            "held_out_target_patient_ids": list(
                preflight_extractor.held_out_target_patient_ids
            ),
            "training_target_roster_sha256": (
                preflight_extractor.training_target_roster_sha256
            ),
            "held_out_target_roster_sha256": (
                preflight_extractor.held_out_target_roster_sha256
            ),
            "oof_fold": preflight_extractor.oof_fold,
            "epoch": int(epoch),
            "seed": int(seed),
        }
        manifest = _validate_manifest(manifest)
        manifest_bytes = _canonical_json(manifest)
        manifest_path = temporary / _MANIFEST_FILE
        manifest_path.write_bytes(manifest_bytes)
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        return IctalConceptCheckpointArtifact(
            path=target,
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=manifest_sha256,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_ictal_concept_checkpoint(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedIctalConceptCheckpoint:
    """Strictly validate and reconstruct an ictal-involvement concept head."""

    source = Path(path)
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Ictal checkpoint must be a regular directory: {source}")
    actual_files = {item.name for item in source.iterdir()}
    expected_files = {
        _CHECKPOINT_FILE,
        _MANIFEST_FILE,
        _TRAINING_RUN_DIRECTORY,
    }
    if actual_files != expected_files:
        raise ValueError(
            "Ictal checkpoint contains missing or unknown files; "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    manifest_path = source / _MANIFEST_FILE
    checkpoint_path = source / _CHECKPOINT_FILE
    if (
        manifest_path.is_symlink()
        or checkpoint_path.is_symlink()
        or not manifest_path.is_file()
        or not checkpoint_path.is_file()
    ):
        raise ValueError("Ictal checkpoint members must be regular files")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    if expected_manifest_sha256 is not None:
        expected = _require_sha256(
            expected_manifest_sha256, field="expected_manifest_sha256"
        )
        if manifest_sha256 != expected:
            raise ValueError("Ictal checkpoint manifest SHA-256 mismatch")
    manifest = _validate_manifest(_parse_canonical_json(manifest_bytes))
    embedded_run_artifact = load_ictal_training_run_receipt(
        source / _TRAINING_RUN_DIRECTORY,
        expected_artifact_sha256=manifest["training_run_artifact_sha256"],
        expected_training_run_receipt_sha256=manifest[
            "training_run_receipt_sha256"
        ],
    )
    _cross_validate_embedded_training_run(manifest, embedded_run_artifact)
    if checkpoint_path.stat().st_size != manifest["checkpoint_size_bytes"]:
        raise ValueError("Ictal checkpoint file size mismatch")
    if _file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("Ictal checkpoint file SHA-256 mismatch")

    arrays = _validate_state_arrays(
        dict(_load_safetensors(str(checkpoint_path))), manifest["head_config"]
    )
    head = _expected_head(manifest["head_config"])
    state = {
        name: torch.from_numpy(np.array(array, copy=True))
        for name, array in arrays.items()
    }
    head.load_state_dict(state, strict=True)
    if ictal_head_state_sha256(head) != (
        embedded_run_artifact.training_run_receipt.final_head_state_sha256
    ):
        raise ValueError(
            "Ictal checkpoint weights do not match the training-run final head state"
        )
    head.eval()
    return LoadedIctalConceptCheckpoint(
        head=head,
        metadata=manifest,
        manifest_sha256=manifest_sha256,
    )


__all__ = [
    "ICTAL_CHECKPOINT_SCHEMA",
    "IctalConceptCheckpointArtifact",
    "LoadedIctalConceptCheckpoint",
    "load_ictal_concept_checkpoint",
    "save_ictal_concept_checkpoint",
]
