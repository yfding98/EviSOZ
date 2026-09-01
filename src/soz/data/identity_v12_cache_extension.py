"""Fail-closed contracts for append-only identity-v12 representation caches.

The public-development identity recovery keeps the legacy v11 988-event
prefix immutable and appends 161 newly signal-eligible events.  This module
contains only target-free metadata and tensor-integrity helpers.  It never
loads DeepSOZ target values, private data, channel annotations, or historical
predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .deepsoz_signal_identity_recovery import (
    VerifiedDeepSOZSignalIdentityRecoveryBundle,
    load_deepsoz_signal_identity_recovery_bundle,
)
from .public_development_union_identity_v12 import (
    PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FILENAME,
    PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_SCHEMA,
)


LEGACY_EVENT_COUNT = 988
RECOVERED_APPEND_EVENT_COUNT = 161
IDENTITY_V12_EVENT_COUNT = 1149
IDENTITY_V12_PATIENT_COUNT = 103
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_MAX_JSON_BYTES = 128 * 1024 * 1024


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


def tensor_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare exact tensor bytes, including identical NaN payloads."""

    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.shape != right.shape
        or left.dtype != right.dtype
    ):
        return False
    left_bytes = left.detach().cpu().contiguous().view(torch.uint8)
    right_bytes = right.detach().cpu().contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def event_tensor_sha256(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def append_event_tensor_exact(
    legacy: torch.Tensor,
    appended: torch.Tensor,
    *,
    expected_legacy_count: int = LEGACY_EVENT_COUNT,
) -> torch.Tensor:
    """Append on axis zero and prove the legacy tensor is bitwise unchanged."""

    if not isinstance(legacy, torch.Tensor) or not isinstance(appended, torch.Tensor):
        raise TypeError("legacy and appended values must be tensors")
    if legacy.ndim < 1 or appended.ndim != legacy.ndim:
        raise ValueError("legacy/appended tensors must share a non-scalar rank")
    if legacy.shape[0] != expected_legacy_count:
        raise ValueError("legacy tensor event dimension changed")
    if legacy.shape[1:] != appended.shape[1:]:
        raise ValueError("appended tensor trailing shape differs from legacy")
    if legacy.dtype != appended.dtype:
        raise TypeError("appended tensor dtype differs from legacy")
    if legacy.device.type != "cpu" or appended.device.type != "cpu":
        raise ValueError("cache extension tensors must be materialized on CPU")
    combined = torch.cat(
        (legacy.contiguous(), appended.contiguous()), dim=0
    ).contiguous()
    if not tensor_bitwise_equal(combined[:expected_legacy_count], legacy):
        raise RuntimeError("legacy tensor prefix changed during append")
    if tensor_sha256(combined[:expected_legacy_count]) != tensor_sha256(legacy):
        raise RuntimeError("legacy tensor prefix SHA changed during append")
    return combined


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return absolute


def _strict_json(path: Path, *, field: str) -> tuple[dict[str, object], bytes]:
    source = _absolute_no_symlink(path, field=field)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not 1 <= source.stat().st_size <= _MAX_JSON_BYTES:
        raise ValueError(f"{field} has an invalid size")
    raw = source.read_bytes()

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"Non-finite JSON constant is forbidden: {value}")

    try:
        parsed = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"{field} must contain one JSON object")
    if canonical_bytes(parsed, newline=True) != raw:
        raise ValueError(f"{field} is not canonical JSON")
    return parsed, raw


def _closed_directory(
    directory: str | Path,
    *,
    expected_names: Sequence[str],
    field: str,
) -> tuple[Path, dict[str, Path]]:
    root = _absolute_no_symlink(directory, field=field)
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    if tuple(path.name for path in entries) != tuple(sorted(expected_names)):
        raise ValueError(f"{field} violates its closed file schema")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError(f"{field} contains a non-regular file")
    return root, {path.name: path for path in entries}


@dataclass(frozen=True)
class IdentityV12ExtensionContract:
    union_path: Path
    union_manifest: Mapping[str, object]
    union_manifest_sha256: str
    signal_bundle: VerifiedDeepSOZSignalIdentityRecoveryBundle
    events: tuple[Mapping[str, object], ...]
    legacy_events: tuple[Mapping[str, object], ...]
    appended_events: tuple[Mapping[str, object], ...]
    signal_events_by_id: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        if len(self.events) != IDENTITY_V12_EVENT_COUNT:
            raise ValueError("identity-v12 contract must contain 1149 events")
        if len(self.legacy_events) != LEGACY_EVENT_COUNT:
            raise ValueError("identity-v12 legacy prefix must contain 988 events")
        if len(self.appended_events) != RECOVERED_APPEND_EVENT_COUNT:
            raise ValueError("identity-v12 append must contain 161 events")
        if self.events != (*self.legacy_events, *self.appended_events):
            raise ValueError("identity-v12 events are not legacy-prefix plus append")


@dataclass(frozen=True)
class LegacyRepresentationCache:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    tensor_path: Path
    tensor_file_sha256: str


def load_identity_v12_extension_contract(
    union_directory: str | Path,
    signal_directory: str | Path,
    *,
    expected_union_manifest_sha256: str,
    expected_signal_artifact_sha256: str,
) -> IdentityV12ExtensionContract:
    """Load and cross-bind the 1149-event union and signal receipt."""

    union_root, union_files = _closed_directory(
        union_directory,
        expected_names=(PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FILENAME,),
        field="identity-v12 public union",
    )
    manifest, raw = _strict_json(
        union_files[PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_FILENAME],
        field="identity-v12 union manifest",
    )
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha != _require_sha256(
        expected_union_manifest_sha256,
        field="expected_union_manifest_sha256",
    ):
        raise ValueError("identity-v12 union manifest SHA mismatch")
    if manifest.get("schema_version") != PUBLIC_DEVELOPMENT_UNION_IDENTITY_V12_SCHEMA:
        raise ValueError("identity-v12 union schema mismatch")
    payload_sha = _require_sha256(
        manifest.get("manifest_payload_sha256"),
        field="manifest_payload_sha256",
    )
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if canonical_sha256(payload) != payload_sha:
        raise ValueError("identity-v12 union payload SHA mismatch")
    expected_counts = {
        "patient_count": IDENTITY_V12_PATIENT_COUNT,
        "event_count": IDENTITY_V12_EVENT_COUNT,
        "legacy_v11_event_prefix_count": LEGACY_EVENT_COUNT,
        "recovered_append_event_count": RECOVERED_APPEND_EVENT_COUNT,
    }
    if any(manifest.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("identity-v12 union count contract changed")
    events_value = manifest.get("events")
    event_ids_value = manifest.get("event_ids")
    if not isinstance(events_value, list) or not isinstance(event_ids_value, list):
        raise TypeError("identity-v12 union event rosters are missing")
    events = tuple(events_value)
    event_ids = tuple(str(value) for value in event_ids_value)
    if len(events) != IDENTITY_V12_EVENT_COUNT or len(event_ids) != len(events):
        raise ValueError("identity-v12 union event roster length changed")
    if tuple(str(row["event_id"]) for row in events) != event_ids:
        raise ValueError("identity-v12 event rows differ from event_ids")
    if tuple(int(row["ordinal"]) for row in events) != tuple(range(len(events))):
        raise ValueError("identity-v12 event ordinals are not contiguous")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("identity-v12 event IDs are not unique")
    if canonical_sha256(list(event_ids)) != manifest.get("event_order_sha256"):
        raise ValueError("identity-v12 event-order SHA mismatch")
    immutability = manifest.get("immutability_receipt")
    required_immutability = (
        "legacy_102_patient_outer_folds_preserved",
        "legacy_988_event_rows_exact_prefix",
        "legacy_988_event_ids_exact_prefix",
        "recovered_events_append_only",
    )
    if not isinstance(immutability, Mapping) or not all(
        immutability.get(key) is True for key in required_immutability
    ):
        raise ValueError("identity-v12 union immutability receipt failed")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(key) is not False
        for key in (
            "raw_eeg_loaded",
            "deepsoz_target_values_loaded",
            "private_eeg_loaded",
            "private_target_values_loaded",
            "prediction_artifacts_loaded",
        )
    ):
        raise ValueError("identity-v12 union access boundary is not target-free")

    signal = load_deepsoz_signal_identity_recovery_bundle(
        signal_directory,
        expected_artifact_sha256=expected_signal_artifact_sha256,
    )
    signal_rows_value = signal.receipt.get("events")
    if not isinstance(signal_rows_value, list):
        raise TypeError("identity-v3 signal receipt lacks event rows")
    signal_by_id = {
        str(row["event_id"]): row for row in signal_rows_value
    }
    if len(signal_by_id) != IDENTITY_V12_EVENT_COUNT or set(signal_by_id) != set(
        event_ids
    ):
        raise ValueError("union and signal identity-v3 event rosters differ")
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping) or (
        lineage.get("signal_identity_recovery_artifact_sha256")
        != signal.artifact_sha256
        or lineage.get("signal_identity_recovery_receipt_sha256")
        != signal.receipt_sha256
        or lineage.get("preprocess_config_sha256")
        != signal.receipt.get("preprocess_config_sha256")
    ):
        raise ValueError("identity-v12 union and identity-v3 signal lineage differ")

    paired_fields = (
        "event_id",
        "patient_id",
        "relative_edf_path",
        "global_event_index",
        "global_t0_sec",
        "global_stop_sec",
        "event_record_sha256",
        "edf_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
        "processed_window_shape",
        "processed_window_dtype",
    )
    for event in events:
        signal_event = signal_by_id[str(event["event_id"])]
        for field in paired_fields:
            left = event[field]
            right = signal_event[field]
            if field == "processed_window_shape":
                left, right = tuple(left), tuple(right)
            if left != right:
                raise ValueError(
                    f"identity-v12 union/signal event mismatch: "
                    f"{event['event_id']}:{field}"
                )
        if event["legacy_model_split"] != signal_event["model_split"]:
            raise ValueError("identity-v12 union/signal split metadata differ")

    legacy = events[:LEGACY_EVENT_COUNT]
    appended = events[LEGACY_EVENT_COUNT:]
    return IdentityV12ExtensionContract(
        union_path=union_root,
        union_manifest=manifest,
        union_manifest_sha256=manifest_sha,
        signal_bundle=signal,
        events=events,
        legacy_events=legacy,
        appended_events=appended,
        signal_events_by_id=signal_by_id,
    )


def select_appended_events(
    contract: IdentityV12ExtensionContract,
    append_limit: int | None,
) -> tuple[tuple[Mapping[str, object], ...], bool]:
    if append_limit is None:
        return contract.appended_events, True
    if isinstance(append_limit, bool) or not (
        1 <= int(append_limit) < RECOVERED_APPEND_EVENT_COUNT
    ):
        raise ValueError("append_limit must be a smoke prefix in [1,160]")
    return contract.appended_events[: int(append_limit)], False


def load_legacy_representation_cache(
    cache_directory: str | Path,
    *,
    contract: IdentityV12ExtensionContract,
    expected_schema: str,
    tensor_filename: str,
    expected_manifest_sha256: str,
    expected_tensor_file_sha256: str,
) -> LegacyRepresentationCache:
    """Load one exact v11 target-free cache without accepting extra files."""

    root, files = _closed_directory(
        cache_directory,
        expected_names=("manifest.json", tensor_filename),
        field="legacy representation cache",
    )
    manifest, raw = _strict_json(
        files["manifest.json"], field="legacy representation manifest"
    )
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_legacy_manifest_sha256"
    ):
        raise ValueError("legacy representation manifest SHA mismatch")
    tensor_file_sha = file_sha256(files[tensor_filename])
    if tensor_file_sha != _require_sha256(
        expected_tensor_file_sha256,
        field="expected_legacy_tensor_file_sha256",
    ):
        raise ValueError("legacy representation tensor-file SHA mismatch")
    if manifest.get("schema_version") != expected_schema:
        raise ValueError("legacy representation schema mismatch")
    if (
        manifest.get("full_scope") is not True
        or manifest.get("smoke_only") is not False
        or manifest.get("event_count") != LEGACY_EVENT_COUNT
        or manifest.get("patient_count") != 102
        or manifest.get("tensor_file") != tensor_filename
        or manifest.get("tensor_file_sha256") != tensor_file_sha
        or manifest.get("tensor_file_size_bytes")
        != files[tensor_filename].stat().st_size
    ):
        raise ValueError("legacy representation manifest contract changed")
    event_ids = tuple(str(value) for value in manifest.get("event_ids", ()))
    rows = manifest.get("events")
    legacy_ids = tuple(str(row["event_id"]) for row in contract.legacy_events)
    if (
        event_ids != legacy_ids
        or not isinstance(rows, list)
        or tuple(str(row["event_id"]) for row in rows) != legacy_ids
        or tuple(int(row["ordinal"]) for row in rows)
        != tuple(range(LEGACY_EVENT_COUNT))
    ):
        raise ValueError("legacy representation event prefix changed")
    if manifest.get("event_order_sha256") != canonical_sha256(list(legacy_ids)):
        raise ValueError("legacy representation event-order SHA mismatch")
    for cache_row, union_row in zip(rows, contract.legacy_events):
        if (
            cache_row.get("patient_id") != union_row["patient_id"]
            or cache_row.get("outer_fold") != union_row["outer_fold"]
            or cache_row.get("legacy_model_split")
            != union_row["legacy_model_split"]
            or cache_row.get("processed_window_sha256")
            != union_row["processed_window_sha256"]
        ):
            raise ValueError("legacy representation row differs from union prefix")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(key) is not False
        for key in (
            "deepsoz_target_values_loaded",
            "tusz_channel_annotation_values_loaded",
            "historical_prediction_artifacts_loaded",
            "private_eeg_loaded",
            "private_target_values_loaded",
        )
    ):
        raise ValueError("legacy representation cache is not target-free")
    return LegacyRepresentationCache(
        path=root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        tensor_path=files[tensor_filename],
        tensor_file_sha256=tensor_file_sha,
    )


__all__ = [
    "IDENTITY_V12_EVENT_COUNT",
    "IDENTITY_V12_PATIENT_COUNT",
    "IdentityV12ExtensionContract",
    "LEGACY_EVENT_COUNT",
    "LegacyRepresentationCache",
    "RECOVERED_APPEND_EVENT_COUNT",
    "append_event_tensor_exact",
    "canonical_bytes",
    "canonical_sha256",
    "event_tensor_sha256",
    "file_sha256",
    "load_identity_v12_extension_contract",
    "load_legacy_representation_cache",
    "select_appended_events",
    "tensor_bitwise_equal",
    "tensor_sha256",
]
