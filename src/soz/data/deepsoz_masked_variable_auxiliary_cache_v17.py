"""Strict event/replay contract for independent v17 auxiliary caches.

The cache producers consume two public, immutable inputs:

* the target-independent DeepSOZ identity-overlay signal universe; and
* the admission-only projection of the masked-variable target join.

The admission projection contains no channel target, target state, loss mask,
or positive/negative channel list.  It nevertheless carries an explicitly
target-conditioned roster lineage.  This distinction lets representation
materialization remain honest about all three lineage axes without opening
the target-bearing join artifact.

This module never loads an existing 1,149-event representation cache.  Its
event contract is exactly the independently admitted auxiliary roster.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Callable, Mapping, Sequence

import torch
from safetensors.torch import save_file

from ..frozen_h_crosswalk import _signal_tensor_sha256
from . import deepsoz_masked_variable_auxiliary_join as _join
from . import deepsoz_signal_preflight as _base
from .deepsoz_target_independent_signal_universe import (
    TARGET_INDEPENDENT_SIGNAL_UNIVERSE_POLICY,
    TARGET_INDEPENDENT_SIGNAL_UNIVERSE_SCHEMA,
    VerifiedTargetIndependentSignalUniverse,
    load_target_independent_signal_universe,
)
from .edf import CausalEDFConfig, EDF_PREPROCESS_SCHEMA


FORMAL_SIGNAL_UNIVERSE_ARTIFACT_SHA256 = (
    "f80ce2f606b673871d5de359eb690707ae5ddf1b6a2bf41ed0704c133b083ea4"
)
FORMAL_AUXILIARY_ADMISSION_ARTIFACT_SHA256 = (
    "a3a69550a4b0d7445d8311ed4641c25ef1cd28551f70b022d520984983dead7e"
)
FORMAL_SIGNAL_UNIVERSE_PATIENT_COUNT = 124
FORMAL_SIGNAL_UNIVERSE_CANDIDATE_EVENT_COUNT = 1812
FORMAL_SIGNAL_UNIVERSE_ELIGIBLE_EVENT_COUNT = 1364
FORMAL_AUXILIARY_PATIENT_COUNT = 9
FORMAL_AUXILIARY_EVENT_COUNT = 182
MANIFEST_FILENAME = "manifest.json"

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_CACHE_LINEAGE_AXIS_FIELDS = frozenset(
    {
        "direct_target_values",
        "upstream_target_conditioned_roster",
        "target_supervised_model",
    }
)


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    metadata = canonical_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def event_tensor_sha256(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def tensor_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.shape != right.shape
        or left.dtype != right.dtype
    ):
        return False
    return bool(
        torch.equal(
            left.detach().cpu().contiguous().view(torch.uint8),
            right.detach().cpu().contiguous().view(torch.uint8),
        )
    )


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def cache_lineage_axes() -> dict[str, dict[str, object]]:
    """Return the closed three-axis lineage for representation extraction."""

    return {
        "direct_target_values": {
            "used": False,
            "evidence": (
                "the producer opens only the admission-only projection, whose "
                "closed schema forbids targets, loss masks, target states, and "
                "positive/negative channel fields"
            ),
        },
        "upstream_target_conditioned_roster": {
            "used": True,
            "evidence": (
                "the admitted patient/event roster is an exact target-conditioned "
                "projection of the v17 masked-variable join"
            ),
        },
        "target_supervised_model": {
            "used": False,
            "evidence": (
                "fine evidence is deterministic and the prefix uses only the "
                "official pretrained LaBraM-Base with blocks 0-9 frozen"
            ),
        },
    }


@dataclass(frozen=True)
class AuxiliaryCacheContract:
    admission_path: Path
    admission: object
    signal_path: Path
    signal: VerifiedTargetIndependentSignalUniverse
    events: tuple[Mapping[str, object], ...]
    patient_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    preprocess_config: CausalEDFConfig

    def __post_init__(self) -> None:
        if not self.events or not self.patient_ids or not self.event_ids:
            raise ValueError("auxiliary cache contract cannot be empty")
        if len(self.events) != len(self.event_ids):
            raise ValueError("auxiliary event rows and IDs differ in length")
        if tuple(str(row["event_id"]) for row in self.events) != self.event_ids:
            raise ValueError("auxiliary event rows differ from the admitted order")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("auxiliary event IDs are not unique")
        if set(str(row["patient_id"]) for row in self.events) != set(
            self.patient_ids
        ):
            raise ValueError("auxiliary event/patient rosters do not close")


def _admission_loader() -> Callable[..., object]:
    loader = getattr(_join, "load_masked_variable_auxiliary_admission", None)
    if loader is None:
        raise RuntimeError(
            "v17 admission-only loader is unavailable; refusing to open the "
            "target-bearing join artifact in a representation producer"
        )
    return loader


def load_auxiliary_cache_contract(
    admission_directory: str | Path,
    signal_universe_directory: str | Path,
    *,
    expected_admission_artifact_sha256: str,
    expected_signal_universe_artifact_sha256: str,
) -> AuxiliaryCacheContract:
    """Strict-load and cross-bind admission rows to full signal receipts."""

    admission_sha = _require_sha256(
        expected_admission_artifact_sha256,
        field="expected_admission_artifact_sha256",
    )
    signal_sha = _require_sha256(
        expected_signal_universe_artifact_sha256,
        field="expected_signal_universe_artifact_sha256",
    )
    admission = _admission_loader()(
        admission_directory,
        expected_artifact_sha256=admission_sha,
    )
    signal = load_target_independent_signal_universe(
        signal_universe_directory,
        expected_artifact_sha256=signal_sha,
    )
    admission_receipt = admission.receipt
    signal_receipt = signal.receipt

    admission_axes = admission_receipt.get("lineage_axes")
    if (
        not isinstance(admission_axes, Mapping)
        or set(admission_axes) != _CACHE_LINEAGE_AXIS_FIELDS
        or {key: bool(value.get("used")) for key, value in admission_axes.items()}
        != {
            "direct_target_values": False,
            "upstream_target_conditioned_roster": True,
            "target_supervised_model": False,
        }
    ):
        raise ValueError("admission-only lineage axes are incompatible with caching")
    if admission_receipt.get("private_data_accessed") is not False or (
        admission_receipt.get("model_or_training_executed") is not False
    ):
        raise ValueError("admission-only artifact crossed the cache access boundary")
    if (
        signal_receipt.get("schema_version")
        != TARGET_INDEPENDENT_SIGNAL_UNIVERSE_SCHEMA
        or signal_receipt.get("policy")
        != TARGET_INDEPENDENT_SIGNAL_UNIVERSE_POLICY
    ):
        raise ValueError("target-independent signal-universe contract drifted")
    if any(
        bool(state.get("used"))
        for state in signal_receipt.get("lineage_axes", {}).values()
    ):
        raise ValueError("signal universe is not independent on all lineage axes")

    hash_pairs = {
        "signal universe artifact": (
            admission_receipt.get("signal_universe_artifact_sha256"),
            signal.artifact_sha256,
        ),
        "signal universe receipt": (
            admission_receipt.get("signal_universe_receipt_sha256"),
            signal.receipt_sha256,
        ),
        "eligible event roster": (
            admission_receipt.get("signal_universe_eligible_event_roster_sha256"),
            signal_receipt.get("eligible_event_roster_sha256"),
        ),
    }
    for field, (left, right) in hash_pairs.items():
        if left != right:
            raise ValueError(f"admission/{field} binding mismatch")
    if signal.artifact_sha256 != signal_sha:
        raise ValueError("loaded signal universe differs from the caller pin")

    config_payload = signal_receipt.get("preprocess_config")
    if not isinstance(config_payload, Mapping):
        raise TypeError("signal universe lacks a preprocessing configuration")
    config = CausalEDFConfig(**dict(config_payload))
    if asdict(config) != asdict(CausalEDFConfig()):
        raise ValueError("v17 caching requires the frozen primary CausalEDFConfig")
    if config.apply_car19 is not True or config.reference_policy != "primary_ref":
        raise ValueError("v17 caching requires primary-reference C-CAR19")
    if signal_receipt.get("preprocess_schema") != EDF_PREPROCESS_SCHEMA:
        raise ValueError("signal-universe preprocessing schema drifted")
    if _base._config_sha256(config) != signal_receipt.get(
        "preprocess_config_sha256"
    ):
        raise ValueError("signal-universe preprocessing SHA drifted")

    signal_events_value = signal_receipt.get("events")
    admission_events_value = admission_receipt.get("events")
    admission_patients_value = admission_receipt.get("patients")
    if not all(
        isinstance(value, list)
        for value in (
            signal_events_value,
            admission_events_value,
            admission_patients_value,
        )
    ):
        raise TypeError("signal/admission event rosters must be JSON arrays")
    signal_by_id = {
        str(row["event_id"]): row for row in signal_events_value
    }
    if len(signal_by_id) != len(signal_events_value):
        raise ValueError("signal universe contains duplicate eligible event IDs")
    patient_folds: dict[str, int] = {}
    for row in admission_patients_value:
        patient_id = str(row["patient_id"])
        fold = row["aux_outer_fold"]
        if (
            not patient_id
            or patient_id in patient_folds
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or not 0 <= fold < 5
        ):
            raise ValueError("admission patient/fold roster is invalid")
        patient_folds[patient_id] = fold

    paired_fields = (
        "event_id",
        "patient_id",
        "official_split",
        "event_record_sha256",
        "crosswalk_record_sha256",
        "processed_window_sha256",
        "preprocess_config_sha256",
    )
    events: list[dict[str, object]] = []
    for admission_row in admission_events_value:
        event_id = str(admission_row["event_id"])
        signal_row = signal_by_id.get(event_id)
        if signal_row is None:
            raise ValueError("admitted event is absent from the signal universe")
        for field in paired_fields:
            if admission_row[field] != signal_row[field]:
                raise ValueError(
                    f"admission/signal event mismatch: {event_id}:{field}"
                )
        if admission_row["source_model_split"] != signal_row["model_split"]:
            raise ValueError(
                f"admission/signal model split mismatch: {event_id}"
            )
        patient_id = str(signal_row["patient_id"])
        if patient_id not in patient_folds:
            raise ValueError("admitted event lacks an auxiliary patient fold")
        events.append(
            {
                **dict(signal_row),
                "source_model_split": str(admission_row["source_model_split"]),
                "aux_outer_fold": patient_folds[patient_id],
                "admission_event_record_sha256": canonical_sha256(
                    dict(admission_row)
                ),
            }
        )
    event_ids = tuple(str(value) for value in admission_receipt["admitted_event_ids"])
    patient_ids = tuple(
        str(value) for value in admission_receipt["admitted_patient_ids"]
    )
    if tuple(str(row["event_id"]) for row in events) != event_ids:
        raise ValueError("materialization roster differs from admitted_event_ids")
    if tuple(patient_folds) != patient_ids:
        raise ValueError("materialization patients differ from admitted_patient_ids")
    if admission_receipt.get("admitted_event_count") != len(events) or (
        admission_receipt.get("admitted_patient_count") != len(patient_ids)
    ):
        raise ValueError("admission counts disagree with the cache roster")
    preregistered_count = getattr(
        _join, "PREREGISTERED_AUXILIARY_PATIENT_COUNT", 9
    )
    if len(patient_ids) != preregistered_count:
        raise ValueError("formal auxiliary patient startup gate failed")

    return AuxiliaryCacheContract(
        admission_path=Path(os.path.abspath(admission_directory)).resolve(strict=True),
        admission=admission,
        signal_path=Path(os.path.abspath(signal_universe_directory)).resolve(
            strict=True
        ),
        signal=signal,
        events=tuple(events),
        patient_ids=patient_ids,
        event_ids=event_ids,
        preprocess_config=config,
    )


def select_cache_events(
    contract: AuxiliaryCacheContract,
    limit: int | None,
) -> tuple[tuple[Mapping[str, object], ...], bool]:
    if limit is None:
        return contract.events, True
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit < len(contract.events)
    ):
        raise ValueError(
            f"limit must be a smoke prefix in [1,{len(contract.events) - 1}]"
        )
    return contract.events[:limit], False


def require_formal_cache_scope(
    contract: AuxiliaryCacheContract,
    selected: Sequence[Mapping[str, object]],
    *,
    full_scope: bool,
) -> None:
    """Prevent a nonformal artifact from being published under the full schema."""

    if not full_scope:
        return
    checks = {
        "admission artifact pin": contract.admission.artifact_sha256
        == FORMAL_AUXILIARY_ADMISSION_ARTIFACT_SHA256,
        "signal artifact pin": contract.signal.artifact_sha256
        == FORMAL_SIGNAL_UNIVERSE_ARTIFACT_SHA256,
        "auxiliary patient count": len(contract.patient_ids)
        == FORMAL_AUXILIARY_PATIENT_COUNT,
        "auxiliary event count": len(contract.event_ids)
        == FORMAL_AUXILIARY_EVENT_COUNT,
        "selected event count": len(selected) == FORMAL_AUXILIARY_EVENT_COUNT,
        "selected event order": tuple(str(row["event_id"]) for row in selected)
        == contract.event_ids,
        "signal identity patient count": contract.signal.receipt.get(
            "identity_patient_count"
        )
        == FORMAL_SIGNAL_UNIVERSE_PATIENT_COUNT,
        "signal candidate event count": contract.signal.receipt.get(
            "candidate_event_count"
        )
        == FORMAL_SIGNAL_UNIVERSE_CANDIDATE_EVENT_COUNT,
        "signal eligible event count": contract.signal.receipt.get(
            "eligible_event_count"
        )
        == FORMAL_SIGNAL_UNIVERSE_ELIGIBLE_EVENT_COUNT,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "formal v17 auxiliary cache startup gate failed: " + ", ".join(failed)
        )


def safe_edf_path(root: Path, relative_value: object) -> Path:
    relative = PurePosixPath(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("signal-universe EDF path is not a safe relative EDF path")
    source = root.joinpath(*relative.parts)
    for component in (source, *source.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("signal-universe EDF path cannot traverse symlinks")
    resolved = source.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("signal-universe EDF path escaped the pinned TUSZ root")
    return resolved


def validate_raw_replay(loaded: object, event: Mapping[str, object]) -> str:
    """Bind one raw replay to all event-level signal-universe receipts."""

    edf_receipt = asdict(loaded.edf_receipt)
    signal_receipt = asdict(loaded.signal_receipt)
    edf_receipt_sha = canonical_sha256(edf_receipt)
    signal_receipt_sha = canonical_sha256(signal_receipt)
    replay_sha = _signal_tensor_sha256(loaded.window.data)
    checks = {
        "EDF content": loaded.edf_receipt.edf_sha256 == event["edf_sha256"],
        # ``asdict`` preserves tuple-valued dataclass fields, while the same
        # receipt round-tripped through the signal-universe JSON necessarily
        # contains lists.  Compare their canonical JSON values so that this
        # representation-only difference does not reject an otherwise exact
        # replay.  The content hashes below still bind every scalar and item.
        "EDF receipt payload": canonical_bytes(edf_receipt)
        == canonical_bytes(event["edf_receipt"]),
        "EDF receipt SHA": edf_receipt_sha == event["edf_receipt_sha256"],
        "signal receipt payload": canonical_bytes(signal_receipt)
        == canonical_bytes(event["signal_receipt"]),
        "signal receipt SHA": signal_receipt_sha == event["signal_receipt_sha256"],
        "processed window": replay_sha == event["processed_window_sha256"],
        "shape": tuple(loaded.window.data.shape) == (19, 12_000),
        "dtype": loaded.window.data.dtype == torch.float32,
        "sampling": float(loaded.window.sfreq_hz) == 200.0,
        "onset index": int(loaded.window.onset_index) == 2_400,
        "requested t0": abs(
            float(loaded.edf_receipt.requested_onset_sec)
            - float(event["global_t0_sec"])
        )
        <= 1e-6,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"v17 auxiliary raw replay failed {event['event_id']}: {failed}"
        )
    return replay_sha


def resolve_raw_root(path: str | Path) -> Path:
    root = Path(os.path.abspath(path)).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(root)
    return root


def prepare_output_directory(
    output_directory: str | Path,
    *,
    input_paths: Sequence[str | Path],
) -> Path:
    target = Path(os.path.abspath(output_directory))
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise FileNotFoundError(target.parent)
    resolved_inputs = tuple(
        Path(os.path.abspath(path)).resolve(strict=True) for path in input_paths
    )
    for source in resolved_inputs:
        if target == source or target in source.parents or source in target.parents:
            raise ValueError("auxiliary cache output overlaps an input")
    for component in (target.parent, *target.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("auxiliary cache output cannot traverse symlinks")
    return target


def _fsync_regular_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_publish_safetensors(
    target: Path,
    *,
    tensor_filename: str,
    tensors: Mapping[str, torch.Tensor],
    build_manifest: Callable[[Path], Mapping[str, object]],
) -> tuple[Path, Mapping[str, object]]:
    """Publish one two-file cache atomically without overwriting a destination."""

    if not tensor_filename.endswith(".safetensors") or "/" in tensor_filename:
        raise ValueError("tensor filename must be a local .safetensors name")
    if not tensors or any(
        not isinstance(value, torch.Tensor) or value.device.type != "cpu"
        for value in tensors.values()
    ):
        raise ValueError("published tensors must be a non-empty CPU mapping")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / tensor_filename
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
            str(tensor_path),
        )
        _fsync_regular_file(tensor_path)
        manifest = dict(build_manifest(tensor_path))
        manifest_path = staging / MANIFEST_FILENAME
        with manifest_path.open("xb") as stream:
            stream.write(canonical_bytes(manifest, newline=True))
            stream.flush()
            os.fsync(stream.fileno())
        _base._fsync_directory(staging)
        if os.path.lexists(target):
            raise FileExistsError(target)
        os.rename(staging, target)
        published = True
        _base._fsync_directory(target.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target, manifest


__all__ = [
    "AuxiliaryCacheContract",
    "FORMAL_AUXILIARY_ADMISSION_ARTIFACT_SHA256",
    "FORMAL_AUXILIARY_EVENT_COUNT",
    "FORMAL_AUXILIARY_PATIENT_COUNT",
    "FORMAL_SIGNAL_UNIVERSE_ARTIFACT_SHA256",
    "FORMAL_SIGNAL_UNIVERSE_CANDIDATE_EVENT_COUNT",
    "FORMAL_SIGNAL_UNIVERSE_ELIGIBLE_EVENT_COUNT",
    "FORMAL_SIGNAL_UNIVERSE_PATIENT_COUNT",
    "MANIFEST_FILENAME",
    "atomic_publish_safetensors",
    "cache_lineage_axes",
    "canonical_bytes",
    "canonical_sha256",
    "event_tensor_sha256",
    "file_sha256",
    "load_auxiliary_cache_contract",
    "prepare_output_directory",
    "resolve_raw_root",
    "require_formal_cache_scope",
    "safe_edf_path",
    "select_cache_events",
    "tensor_bitwise_equal",
    "tensor_sha256",
    "validate_raw_replay",
]
