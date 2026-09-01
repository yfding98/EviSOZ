"""Minimal, target-free LaBraM k31 inference projections for v13.

The legacy v1.2 recovery bundle is intentionally *not* imported here.  A
separate one-time broker may parse those historical manifests and checkpoints,
then call :func:`publish_k31_inference_projection_v13`.  The resulting bundle
contains only the six inference heads, fit/gate rosters, fixed scientific
identity, and an honest record of historical process exposure.  It contains no
held/native-evaluation patient identities, training-run metrics, clinical
identity, or clinical outcomes.

Stage-A code can therefore import and strict-load this module without making
the legacy recovery loader or any target snapshot reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load as _load_safetensors_bytes

from .ictal_inference_primitives_v13 import (
    canonical_patient_roster as _patient_roster,
    ictal_head_state_sha256,
    patient_roster_sha256,
)
from .models.concept_heads import LongContextTemporalResidualIctalInvolvementHead


K31_INFERENCE_PROJECTION_SCHEMA_V13 = "soz_labram_k31_inference_projection_v13"
K31_INFERENCE_PROJECTION_PURPOSE_V13 = (
    "minimal_k31_heads_for_target_inaccessible_gate_inference_only"
)
K31_INFERENCE_PROJECTION_SERIALIZATION_V13 = (
    "canonical_json_plus_exact_legacy_safetensors_no_pickle"
)
K31_INFERENCE_PROJECTION_MANIFEST = "manifest.json"
K31_INFERENCE_PROJECTION_BROKER_RECEIPT_SCHEMA_V13 = (
    "soz_labram_k31_inference_projection_broker_receipt_v13"
)

EXPECTED_PRODUCER_ORDER = (
    "fold0",
    "fold1",
    "fold2",
    "fold3",
    "fold4",
    "final",
)
EXPECTED_CANDIDATE = "labram_temporal_residual_k31"
EXPECTED_CONTEXT_SECONDS = 31
EXPECTED_CONTEXT_DIRECTION = "symmetric_retrospective_not_causal_onset"
EXPECTED_TARGET_SEMANTICS = "tusz_bipolar_edge_time_involvement_not_soz"
EXPECTED_HEAD_CONFIG = {"token_dim": 200, "hidden_dim": 128}
EXPECTED_GATE_PATIENT_COUNT = 12

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
_ARTIFACT_MARKER = object()

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "serialization",
        "development_confirmation_only",
        "formal_promotion",
        "authorized_for_formal_evidence_or_reasoner",
        "v13_execution_hold",
        "producer_order",
        "producer_count",
        "v5_split_sha256",
        "gate_patient_ids",
        "gate_patient_roster_sha256",
        "gate_patient_count",
        "producers",
        "broker_receipt",
    }
)
_PRODUCER_FIELDS = frozenset(
    {
        "selection",
        "oof_fold",
        "candidate",
        "context_seconds",
        "context_direction",
        "target_semantics",
        "head_config",
        "checkpoint_filename",
        "checkpoint_sha256",
        "head_state_sha256",
        "legacy_recovery_manifest_sha256",
        "fit_patient_ids",
        "fit_patient_roster_sha256",
        "fit_patient_count",
        "gate_patient_ids",
        "gate_patient_roster_sha256",
        "gate_patient_count",
        "v5_split_sha256",
        "fit_gate_intersection_count",
        "legacy_training_process_loaded_full_tusz_target_snapshot_arrays",
        "legacy_training_process_snapshot_contained_gate_rows",
        "legacy_gate_rows_used_for_fit_loss_gradient_or_native_metric",
        "legacy_gate_confirmation_metrics_computed",
        "candidate_retrained_by_broker",
        "checkpoint_modified_by_broker",
        "projection_record_sha256",
    }
)
_BROKER_FIELDS = frozenset(
    {
        "schema_version",
        "legacy_bundle_count",
        "legacy_full_manifests_loaded",
        "legacy_native_evaluation_roster_metadata_loaded",
        "legacy_training_run_metrics_loaded",
        "legacy_checkpoint_weights_loaded",
        "broker_target_snapshot_files_opened",
        "broker_target_values_loaded",
        "broker_target_masks_loaded",
        "broker_gate_signal_or_tokens_loaded",
        "broker_forward_performed",
        "broker_evaluation_performed",
        "projection_excludes_legacy_held_out_exclusion_ids",
        "projection_excludes_legacy_native_eval_ids",
        "projection_excludes_training_run_metrics",
        "projection_excludes_clinical_identity_or_outcomes",
    }
)


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
        raise ValueError("Projection value is not canonical JSON data") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_exact_fields(
    value: object, expected: frozenset[str], *, field: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{field} violates its closed schema; missing={missing}, unknown={unknown}"
        )
    return dict(value)


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON value is forbidden: {value}")


def _parse_canonical_json(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Projection manifest is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise ValueError("Projection manifest must be one canonical JSON object")
    return value


def _reject_symlink_components(path: Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot contain symlink components")
    return absolute


def _read_stable_regular_file(
    path: Path, *, field: str, maximum_bytes: int
) -> tuple[bytes, str]:
    source = _reject_symlink_components(path, field=field)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{field} must be a regular file: {source}")
    before = source.stat()
    if before.st_size < 1 or before.st_size > maximum_bytes:
        raise ValueError(f"{field} has an invalid size")
    raw = source.read_bytes()
    after = source.stat()
    fingerprint = lambda stat: (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
    if fingerprint(before) != fingerprint(after):
        raise RuntimeError(f"{field} changed while it was read")
    return raw, hashlib.sha256(raw).hexdigest()


def _selection(value: object) -> tuple[str, int | None]:
    selection = str(value).strip().lower()
    if selection not in EXPECTED_PRODUCER_ORDER:
        raise ValueError("selection must be fold0..fold4 or final")
    return (
        selection,
        None if selection == "final" else int(selection.removeprefix("fold")),
    )


def _checkpoint_filename(selection: str) -> str:
    if selection not in EXPECTED_PRODUCER_ORDER:
        raise ValueError("Unknown projection selection")
    return f"{selection}.safetensors"


def _producer_record_without_hash(row: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key != "projection_record_sha256"}


def _load_head_from_bytes(
    raw: bytes, *, expected_head_state_sha256: str, field: str
) -> LongContextTemporalResidualIctalInvolvementHead:
    try:
        state = _load_safetensors_bytes(raw)
    except Exception as exc:
        raise ValueError(f"{field} is not valid safetensors") from exc
    head = LongContextTemporalResidualIctalInvolvementHead(
        token_dim=EXPECTED_HEAD_CONFIG["token_dim"],
        hidden_dim=EXPECTED_HEAD_CONFIG["hidden_dim"],
    )
    expected = head.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"{field} tensor names changed")
    for name, reference in expected.items():
        tensor = state[name]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(f"{field} tensor shape/dtype changed: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{field} contains non-finite weights: {name}")
    head.load_state_dict(state, strict=True)
    expected_sha = _require_sha256(
        expected_head_state_sha256, field="expected_head_state_sha256"
    )
    if ictal_head_state_sha256(head) != expected_sha:
        raise ValueError(f"{field} head-state receipt mismatch")
    head.eval()
    return head


@dataclass(frozen=True)
class LegacyK31ProjectionSourceV13:
    """Minimal values extracted only after a strict legacy-bundle load."""

    selection: str
    oof_fold: int | None
    legacy_recovery_manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    head_state_sha256: str
    fit_patient_ids: tuple[str, ...]
    fit_patient_roster_sha256: str
    gate_patient_ids: tuple[str, ...]
    gate_patient_roster_sha256: str
    v5_split_sha256: str
    legacy_training_process_loaded_full_tusz_target_snapshot_arrays: bool = True
    legacy_training_process_snapshot_contained_gate_rows: bool = True
    legacy_gate_rows_used_for_fit_loss_gradient_or_native_metric: bool = False
    legacy_gate_confirmation_metrics_computed: bool = False

    def __post_init__(self) -> None:
        selection, expected_fold = _selection(self.selection)
        object.__setattr__(self, "selection", selection)
        if self.oof_fold != expected_fold:
            raise ValueError("Source selection and OOF fold disagree")
        for field in (
            "legacy_recovery_manifest_sha256",
            "checkpoint_sha256",
            "head_state_sha256",
            "fit_patient_roster_sha256",
            "gate_patient_roster_sha256",
            "v5_split_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        checkpoint = _reject_symlink_components(
            Path(self.checkpoint_path), field="legacy checkpoint"
        )
        object.__setattr__(self, "checkpoint_path", checkpoint)
        fit = _patient_roster(self.fit_patient_ids, field="fit_patient_ids")
        gate = _patient_roster(self.gate_patient_ids, field="gate_patient_ids")
        object.__setattr__(self, "fit_patient_ids", fit)
        object.__setattr__(self, "gate_patient_ids", gate)
        if len(gate) != EXPECTED_GATE_PATIENT_COUNT:
            raise ValueError("Projection source requires the frozen 12-patient gate")
        if patient_roster_sha256(fit) != self.fit_patient_roster_sha256:
            raise ValueError("Fit roster receipt mismatch")
        if patient_roster_sha256(gate) != self.gate_patient_roster_sha256:
            raise ValueError("Gate roster receipt mismatch")
        if set(fit) & set(gate):
            raise ValueError("Fit roster intersects the I-gate")
        if (
            self.legacy_training_process_loaded_full_tusz_target_snapshot_arrays
            is not True
            or self.legacy_training_process_snapshot_contained_gate_rows is not True
            or self.legacy_gate_rows_used_for_fit_loss_gradient_or_native_metric
            is not False
            or self.legacy_gate_confirmation_metrics_computed is not False
        ):
            raise ValueError("Historical target-snapshot exposure was softened")


@dataclass(frozen=True)
class LoadedK31InferenceProjectionProducerV13:
    selection: str
    oof_fold: int | None
    projection_record_sha256: str
    legacy_recovery_manifest_sha256: str
    checkpoint_sha256: str
    head_state_sha256: str
    fit_patient_ids: tuple[str, ...]
    fit_patient_roster_sha256: str
    gate_patient_ids: tuple[str, ...]
    gate_patient_roster_sha256: str
    v5_split_sha256: str
    manifest: Mapping[str, object]
    head: torch.nn.Module


@dataclass(frozen=True, init=False)
class LoadedK31InferenceProjectionV13:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    producers: tuple[LoadedK31InferenceProjectionProducerV13, ...]

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        manifest: Mapping[str, object],
        manifest_sha256: str,
        producers: Sequence[LoadedK31InferenceProjectionProducerV13],
    ) -> None:
        if _verification_marker is not _ARTIFACT_MARKER:
            raise TypeError("Inference projections require the strict loader")
        values = tuple(producers)
        if tuple(item.selection for item in values) != EXPECTED_PRODUCER_ORDER:
            raise ValueError("Projection producer order changed")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "manifest", dict(manifest))
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "producers", values)

    @property
    def gate_patient_ids(self) -> tuple[str, ...]:
        return tuple(self.manifest["gate_patient_ids"])

    @property
    def gate_patient_roster_sha256(self) -> str:
        return str(self.manifest["gate_patient_roster_sha256"])

    @property
    def v5_split_sha256(self) -> str:
        return str(self.manifest["v5_split_sha256"])


def _broker_receipt() -> dict[str, object]:
    return {
        "schema_version": K31_INFERENCE_PROJECTION_BROKER_RECEIPT_SCHEMA_V13,
        "legacy_bundle_count": 6,
        "legacy_full_manifests_loaded": True,
        "legacy_native_evaluation_roster_metadata_loaded": True,
        "legacy_training_run_metrics_loaded": True,
        "legacy_checkpoint_weights_loaded": True,
        "broker_target_snapshot_files_opened": False,
        "broker_target_values_loaded": False,
        "broker_target_masks_loaded": False,
        "broker_gate_signal_or_tokens_loaded": False,
        "broker_forward_performed": False,
        "broker_evaluation_performed": False,
        "projection_excludes_legacy_held_out_exclusion_ids": True,
        "projection_excludes_legacy_native_eval_ids": True,
        "projection_excludes_training_run_metrics": True,
        "projection_excludes_clinical_identity_or_outcomes": True,
    }


def _producer_payload(
    source: LegacyK31ProjectionSourceV13, *, checkpoint_filename: str
) -> dict[str, object]:
    row: dict[str, object] = {
        "selection": source.selection,
        "oof_fold": source.oof_fold,
        "candidate": EXPECTED_CANDIDATE,
        "context_seconds": EXPECTED_CONTEXT_SECONDS,
        "context_direction": EXPECTED_CONTEXT_DIRECTION,
        "target_semantics": EXPECTED_TARGET_SEMANTICS,
        "head_config": dict(EXPECTED_HEAD_CONFIG),
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_sha256": source.checkpoint_sha256,
        "head_state_sha256": source.head_state_sha256,
        "legacy_recovery_manifest_sha256": (
            source.legacy_recovery_manifest_sha256
        ),
        "fit_patient_ids": list(source.fit_patient_ids),
        "fit_patient_roster_sha256": source.fit_patient_roster_sha256,
        "fit_patient_count": len(source.fit_patient_ids),
        "gate_patient_ids": list(source.gate_patient_ids),
        "gate_patient_roster_sha256": source.gate_patient_roster_sha256,
        "gate_patient_count": len(source.gate_patient_ids),
        "v5_split_sha256": source.v5_split_sha256,
        "fit_gate_intersection_count": 0,
        "legacy_training_process_loaded_full_tusz_target_snapshot_arrays": True,
        "legacy_training_process_snapshot_contained_gate_rows": True,
        "legacy_gate_rows_used_for_fit_loss_gradient_or_native_metric": False,
        "legacy_gate_confirmation_metrics_computed": False,
        "candidate_retrained_by_broker": False,
        "checkpoint_modified_by_broker": False,
    }
    row["projection_record_sha256"] = _canonical_sha256(row)
    return row


def _build_manifest(
    sources: Sequence[LegacyK31ProjectionSourceV13],
) -> dict[str, object]:
    values = tuple(sources)
    first = values[0]
    return {
        "schema_version": K31_INFERENCE_PROJECTION_SCHEMA_V13,
        "purpose": K31_INFERENCE_PROJECTION_PURPOSE_V13,
        "serialization": K31_INFERENCE_PROJECTION_SERIALIZATION_V13,
        "development_confirmation_only": True,
        "formal_promotion": False,
        "authorized_for_formal_evidence_or_reasoner": False,
        "v13_execution_hold": True,
        "producer_order": list(EXPECTED_PRODUCER_ORDER),
        "producer_count": len(values),
        "v5_split_sha256": first.v5_split_sha256,
        "gate_patient_ids": list(first.gate_patient_ids),
        "gate_patient_roster_sha256": first.gate_patient_roster_sha256,
        "gate_patient_count": len(first.gate_patient_ids),
        "producers": [
            _producer_payload(source, checkpoint_filename=_checkpoint_filename(source.selection))
            for source in values
        ],
        "broker_receipt": _broker_receipt(),
    }


def _normalize_sources(
    sources: Sequence[LegacyK31ProjectionSourceV13],
) -> tuple[LegacyK31ProjectionSourceV13, ...]:
    values = tuple(sources)
    if (
        len(values) != len(EXPECTED_PRODUCER_ORDER)
        or any(not isinstance(item, LegacyK31ProjectionSourceV13) for item in values)
        or tuple(item.selection for item in values) != EXPECTED_PRODUCER_ORDER
    ):
        raise TypeError("Projection requires strict fold0..fold4,final sources")
    first = values[0]
    for source in values:
        if (
            source.gate_patient_ids != first.gate_patient_ids
            or source.gate_patient_roster_sha256
            != first.gate_patient_roster_sha256
            or source.v5_split_sha256 != first.v5_split_sha256
        ):
            raise ValueError("Projection sources do not share the frozen gate")
    if (
        len({item.legacy_recovery_manifest_sha256 for item in values}) != 6
        or len({item.checkpoint_sha256 for item in values}) != 6
    ):
        raise ValueError("Projection requires six distinct legacy producers")
    return values


def _write_exclusive_file(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_k31_inference_projection_v13(
    output_directory: str | Path,
    *,
    sources: Sequence[LegacyK31ProjectionSourceV13],
) -> LoadedK31InferenceProjectionV13:
    """Atomically project six strict-loaded legacy runs into a minimal bundle."""

    values = _normalize_sources(sources)
    target = _reject_symlink_components(Path(output_directory), field="projection output")
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("Projection output requires a concrete path with an existing parent")
    if os.path.lexists(target):
        raise FileExistsError(f"Projection output already exists: {target}")

    checkpoint_bytes: dict[str, bytes] = {}
    for source in values:
        raw, file_sha = _read_stable_regular_file(
            source.checkpoint_path,
            field=f"legacy {source.selection} checkpoint",
            maximum_bytes=_MAX_CHECKPOINT_BYTES,
        )
        if file_sha != source.checkpoint_sha256:
            raise ValueError(f"Legacy {source.selection} checkpoint SHA mismatch")
        _load_head_from_bytes(
            raw,
            expected_head_state_sha256=source.head_state_sha256,
            field=f"legacy {source.selection} checkpoint",
        )
        checkpoint_bytes[source.selection] = raw

    manifest = _build_manifest(values)
    manifest_raw = _canonical_json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    moved = False
    try:
        for selection in EXPECTED_PRODUCER_ORDER:
            _write_exclusive_file(
                staging / _checkpoint_filename(selection), checkpoint_bytes[selection]
            )
        _write_exclusive_file(
            staging / K31_INFERENCE_PROJECTION_MANIFEST, manifest_raw
        )
        _fsync_directory(staging)
        load_k31_inference_projection_v13(
            staging, expected_manifest_sha256=manifest_sha
        )
        os.rename(staging, target)
        moved = True
        _fsync_directory(target.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    if not moved:
        raise RuntimeError("Inference projection was not atomically published")
    return load_k31_inference_projection_v13(
        target, expected_manifest_sha256=manifest_sha
    )


def _validate_broker_receipt(value: object) -> dict[str, object]:
    receipt = _require_exact_fields(value, _BROKER_FIELDS, field="broker_receipt")
    if receipt != _broker_receipt():
        raise ValueError("Projection broker disclosure was softened or changed")
    return receipt


def _validate_producer_row(
    value: object,
    *,
    index: int,
    shared_gate: tuple[str, ...],
    shared_gate_sha256: str,
    shared_v5_split_sha256: str,
) -> dict[str, object]:
    row = _require_exact_fields(value, _PRODUCER_FIELDS, field=f"producers[{index}]")
    selection = EXPECTED_PRODUCER_ORDER[index]
    expected_fold = None if selection == "final" else index
    fixed = {
        "selection": selection,
        "oof_fold": expected_fold,
        "candidate": EXPECTED_CANDIDATE,
        "context_seconds": EXPECTED_CONTEXT_SECONDS,
        "context_direction": EXPECTED_CONTEXT_DIRECTION,
        "target_semantics": EXPECTED_TARGET_SEMANTICS,
        "head_config": EXPECTED_HEAD_CONFIG,
        "checkpoint_filename": _checkpoint_filename(selection),
        "gate_patient_ids": list(shared_gate),
        "gate_patient_roster_sha256": shared_gate_sha256,
        "gate_patient_count": len(shared_gate),
        "v5_split_sha256": shared_v5_split_sha256,
        "fit_gate_intersection_count": 0,
        "legacy_training_process_loaded_full_tusz_target_snapshot_arrays": True,
        "legacy_training_process_snapshot_contained_gate_rows": True,
        "legacy_gate_rows_used_for_fit_loss_gradient_or_native_metric": False,
        "legacy_gate_confirmation_metrics_computed": False,
        "candidate_retrained_by_broker": False,
        "checkpoint_modified_by_broker": False,
    }
    if any(row.get(field) != expected for field, expected in fixed.items()):
        raise ValueError(f"Projection producer {selection} changed a fixed boundary")
    for field in (
        "checkpoint_sha256",
        "head_state_sha256",
        "legacy_recovery_manifest_sha256",
        "fit_patient_roster_sha256",
        "gate_patient_roster_sha256",
        "v5_split_sha256",
        "projection_record_sha256",
    ):
        _require_sha256(row[field], field=f"producers[{index}].{field}")
    fit_raw = row["fit_patient_ids"]
    if not isinstance(fit_raw, list):
        raise TypeError("fit_patient_ids must be a string list")
    fit = _patient_roster(fit_raw, field=f"producers[{index}].fit_patient_ids")
    if (
        row["fit_patient_count"] != len(fit)
        or patient_roster_sha256(fit) != row["fit_patient_roster_sha256"]
        or set(fit) & set(shared_gate)
    ):
        raise ValueError(f"Projection producer {selection} fit roster changed")
    if row["projection_record_sha256"] != _canonical_sha256(
        _producer_record_without_hash(row)
    ):
        raise ValueError(f"Projection producer {selection} record receipt mismatch")
    _require_sha256(shared_v5_split_sha256, field="shared_v5_split_sha256")
    return row


def load_k31_inference_projection_v13(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
) -> LoadedK31InferenceProjectionV13:
    """Strictly load only the minimal projection and its six safetensors heads."""

    source = _reject_symlink_components(Path(path), field="inference projection")
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Inference projection must be a regular directory")
    expected_files = {
        K31_INFERENCE_PROJECTION_MANIFEST,
        *(_checkpoint_filename(selection) for selection in EXPECTED_PRODUCER_ORDER),
    }
    if {entry.name for entry in source.iterdir()} != expected_files:
        raise ValueError("Inference projection has missing or unknown files")
    raw, manifest_sha = _read_stable_regular_file(
        source / K31_INFERENCE_PROJECTION_MANIFEST,
        field="projection manifest",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Projection manifest SHA-256 mismatch")
    manifest = _require_exact_fields(
        _parse_canonical_json(raw), _MANIFEST_FIELDS, field="manifest"
    )
    fixed = {
        "schema_version": K31_INFERENCE_PROJECTION_SCHEMA_V13,
        "purpose": K31_INFERENCE_PROJECTION_PURPOSE_V13,
        "serialization": K31_INFERENCE_PROJECTION_SERIALIZATION_V13,
        "development_confirmation_only": True,
        "formal_promotion": False,
        "authorized_for_formal_evidence_or_reasoner": False,
        "v13_execution_hold": True,
        "producer_order": list(EXPECTED_PRODUCER_ORDER),
        "producer_count": 6,
        "gate_patient_count": EXPECTED_GATE_PATIENT_COUNT,
    }
    if any(manifest.get(field) != expected for field, expected in fixed.items()):
        raise ValueError("Projection manifest changed a fixed scientific boundary")
    v5_split_sha = _require_sha256(
        manifest["v5_split_sha256"], field="v5_split_sha256"
    )
    gate_raw = manifest["gate_patient_ids"]
    if not isinstance(gate_raw, list):
        raise TypeError("gate_patient_ids must be a string list")
    gate = _patient_roster(gate_raw, field="gate_patient_ids")
    if len(gate) != EXPECTED_GATE_PATIENT_COUNT:
        raise ValueError("Projection gate patient count changed")
    gate_sha = _require_sha256(
        manifest["gate_patient_roster_sha256"],
        field="gate_patient_roster_sha256",
    )
    if gate_sha != patient_roster_sha256(gate):
        raise ValueError("Projection gate roster receipt mismatch")
    _validate_broker_receipt(manifest["broker_receipt"])

    raw_rows = manifest["producers"]
    if not isinstance(raw_rows, list) or len(raw_rows) != 6:
        raise ValueError("Projection requires exactly six producer rows")
    producers: list[LoadedK31InferenceProjectionProducerV13] = []
    normalized_rows: list[dict[str, object]] = []
    for index, raw_row in enumerate(raw_rows):
        row = _validate_producer_row(
            raw_row,
            index=index,
            shared_gate=gate,
            shared_gate_sha256=gate_sha,
            shared_v5_split_sha256=v5_split_sha,
        )
        checkpoint_raw, checkpoint_sha = _read_stable_regular_file(
            source / str(row["checkpoint_filename"]),
            field=f"projection {row['selection']} checkpoint",
            maximum_bytes=_MAX_CHECKPOINT_BYTES,
        )
        if checkpoint_sha != row["checkpoint_sha256"]:
            raise ValueError(f"Projection {row['selection']} checkpoint SHA mismatch")
        head = _load_head_from_bytes(
            checkpoint_raw,
            expected_head_state_sha256=str(row["head_state_sha256"]),
            field=f"projection {row['selection']} checkpoint",
        )
        fit = tuple(row["fit_patient_ids"])
        producers.append(
            LoadedK31InferenceProjectionProducerV13(
                selection=str(row["selection"]),
                oof_fold=row["oof_fold"],
                projection_record_sha256=str(row["projection_record_sha256"]),
                legacy_recovery_manifest_sha256=str(
                    row["legacy_recovery_manifest_sha256"]
                ),
                checkpoint_sha256=str(row["checkpoint_sha256"]),
                head_state_sha256=str(row["head_state_sha256"]),
                fit_patient_ids=fit,
                fit_patient_roster_sha256=str(row["fit_patient_roster_sha256"]),
                gate_patient_ids=gate,
                gate_patient_roster_sha256=gate_sha,
                v5_split_sha256=v5_split_sha,
                manifest=row,
                head=head,
            )
        )
        normalized_rows.append(row)
    if (
        len({item.projection_record_sha256 for item in producers}) != 6
        or len({item.legacy_recovery_manifest_sha256 for item in producers}) != 6
        or len({item.checkpoint_sha256 for item in producers}) != 6
    ):
        raise ValueError("Projection duplicated a producer")
    manifest["producers"] = normalized_rows
    return LoadedK31InferenceProjectionV13(
        _verification_marker=_ARTIFACT_MARKER,
        path=source,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        producers=producers,
    )


def _issue_k31_inference_projection_for_synthetic_test(
    *,
    path: Path,
    manifest: Mapping[str, object],
    manifest_sha256: str,
    producers: Sequence[LoadedK31InferenceProjectionProducerV13],
) -> LoadedK31InferenceProjectionV13:
    """Private in-memory test seam; it is not a production projection issuer."""

    return LoadedK31InferenceProjectionV13(
        _verification_marker=_ARTIFACT_MARKER,
        path=path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        producers=producers,
    )


__all__ = (
    "EXPECTED_CANDIDATE",
    "EXPECTED_CONTEXT_DIRECTION",
    "EXPECTED_CONTEXT_SECONDS",
    "EXPECTED_GATE_PATIENT_COUNT",
    "EXPECTED_HEAD_CONFIG",
    "EXPECTED_PRODUCER_ORDER",
    "EXPECTED_TARGET_SEMANTICS",
    "K31_INFERENCE_PROJECTION_MANIFEST",
    "K31_INFERENCE_PROJECTION_SCHEMA_V13",
    "LegacyK31ProjectionSourceV13",
    "LoadedK31InferenceProjectionProducerV13",
    "LoadedK31InferenceProjectionV13",
    "load_k31_inference_projection_v13",
    "patient_roster_sha256",
    "publish_k31_inference_projection_v13",
)
