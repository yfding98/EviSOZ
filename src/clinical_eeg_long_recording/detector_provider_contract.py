"""Fail-closed contracts for continuous long-EEG detector providers.

The report pipeline must not import and deserialize an arbitrary model merely
because a path was configured.  This module separates four operations:

* static, non-executing checkpoint-container inspection;
* immutable provider inventory and role selection;
* research execution authorization; and
* a stronger, independently issued production-promotion receipt.

No function in this module calls :func:`torch.load`, :mod:`pickle`, or model
code.  A PyTorch pickle checkpoint therefore remains non-executable here even
after it has passed the static inspection.  An adapter may use
``torch.load(..., weights_only=True)`` only after an exact hash match and only
when its provider inventory explicitly records that loader as verified.
Checkpoints which require ``weights_only=False`` must first be converted in an
isolated, audited environment to a non-executable tensor-only artifact.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import pickletools
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
import zipfile


PROVIDER_REGISTRY_SCHEMA_VERSION = "continuous_detector_provider_registry_v1"
CHECKPOINT_STATIC_AUDIT_SCHEMA_VERSION = "checkpoint_static_audit_v1"
PROVIDER_PROMOTION_SCHEMA_VERSION = "continuous_detector_promotion_receipt_v2"
PROVIDER_EXECUTION_SCHEMA_VERSION = "continuous_detector_execution_receipt_v1"
FULL_RECORD_PROVIDER_RESULT_SCHEMA_VERSION = (
    "continuous_detector_full_record_result_v1"
)
# A portable promotion object currently proves only shape and self-binding.
# No trusted registry yet binds its booleans to calibration, inventory,
# benchmark, paired-bootstrap and external-evaluation artifacts.  Production
# authorization therefore remains disconnected even if a caller supplies a
# syntactically valid object declaring every gate true.
PRODUCTION_PROVIDER_EXECUTION_CONNECTED = False

RESEARCH_ROLES = (
    "target_domain_primary_research_candidate",
    "shadow_continuous_comparator",
    "representation_only_future_candidate",
)
IMPLEMENTATION_STATUSES = (
    "unavailable",
    "adapter_pending",
    "runnable_research",
    "production_qualified",
)
QUALIFICATION_STATUSES = (
    "not_evaluated",
    "benchmark_pending",
    "research_only",
    "passed_external_promotion_gate",
)
LOADER_POLICIES = (
    "artifact_unavailable_no_load",
    "hash_allowlist_then_torch_weights_only_true",
    "hash_allowlist_then_safetensors",
    "isolated_conversion_required_no_inprocess_load",
    "no_checkpoint_detection_head_absent",
)

FULL_RECORD_PROVIDER_OUTCOMES = (
    "completed_with_alarms",
    "completed_zero_alarm",
    "partial_coverage",
    "technical_failure",
)
FULL_RECORD_POSTERIOR_ROW_STATUSES = (
    "modeled",
    "signal_unusable",
    "unmodeled_partial_tail",
)
FULL_RECORD_FAILURE_STAGES = (
    "input_validation",
    "preprocessing",
    "model_load",
    "inference",
    "postprocessing",
    "decoding",
    "unknown",
)
FULL_RECORD_TIME_SEMANTICS: dict[str, Any] = {
    "timebase": "recording_relative_seconds",
    "target_interval_semantics": "posterior_applies_to_target_interval",
    "observed_support_semantics": (
        "all_observed_eeg_samples_accessed_for_this_posterior"
    ),
    "decision_available_semantics": (
        "earliest_recording_time_all_observed_support_is_available"
    ),
    "future_lookahead_is_observed_support_stop_minus_target_stop": True,
    "right_padding_is_not_observed_eeg": True,
}

_SHA256_LENGTH = 64
_MAX_PICKLE_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_ZIP_MEMBER_COUNT = 100_000
_MAX_LEGACY_CHECKPOINT_AUDIT_BYTES = 128 * 1024 * 1024
_PICKLE_EXECUTION_RELEVANT_OPCODES = frozenset(
    {
        "GLOBAL",
        "STACK_GLOBAL",
        "REDUCE",
        "BUILD",
        "INST",
        "OBJ",
        "NEWOBJ",
        "NEWOBJ_EX",
        "EXT1",
        "EXT2",
        "EXT4",
        "PERSID",
        "BINPERSID",
    }
)


class ProviderAuthorizationError(RuntimeError):
    """Raised when a provider is not qualified for the requested role."""


@runtime_checkable
class ContinuousPosteriorProvider(Protocol):
    """Minimal runtime interface implemented by a model-specific adapter.

    ``standardized_eeg`` is deliberately opaque here.  The adapter owns a
    separately versioned preprocessing contract and must return the dense,
    full-record posterior rows consumed by
    :func:`decode_continuous_seizure_posterior`.
    """

    provider_id: str

    def predict_dense_posterior(
        self,
        *,
        recording_id: str,
        standardized_eeg: object,
        sampling_rate_hz: float,
        channel_names: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        """Return legacy dense rows without using annotations or labels.

        A bare row sequence cannot establish full-record coverage, future
        lookahead, decision availability, partial-tail handling, or a
        distinction between a valid zero-alarm completion and a technical
        failure.  Any provider considered beyond legacy research use must
        additionally materialize a result accepted by
        :func:`validate_full_record_provider_result`.
        """


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    return value


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{context} must be a non-negative integer")
    return int(value)


def _stable_sha256(path: Path) -> tuple[str, int]:
    before = path.stat()
    if not path.is_file() or path.is_symlink():
        raise ValueError("checkpoint must be a regular non-symlink file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or size != before.st_size:
        raise RuntimeError("checkpoint changed during hashing")
    return digest.hexdigest(), size


def _looks_like_safetensors(path: Path, payload: bytes) -> bool:
    size = len(payload)
    if path.suffix.lower() != ".safetensors" or size < 10:
        return False
    header_size = int.from_bytes(payload[:8], byteorder="little", signed=False)
    if header_size < 2 or header_size > min(size - 8, 64 * 1024 * 1024):
        return False
    try:
        header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(header, dict)


def _inspect_pickle_program(payload: bytes, *, member_name: str) -> dict[str, Any]:
    globals_seen: set[str] = set()
    opcodes_seen: set[str] = set()
    try:
        operations = list(pickletools.genops(payload))
    except Exception as error:
        raise ValueError(f"malformed pickle metadata {member_name!r}") from error
    if not operations or operations[-1][0].name != "STOP":
        raise ValueError(f"pickle metadata {member_name!r} has no STOP opcode")
    for opcode, argument, _ in operations:
        if opcode.name in _PICKLE_EXECUTION_RELEVANT_OPCODES:
            opcodes_seen.add(opcode.name)
        if opcode.name == "GLOBAL":
            globals_seen.add(str(argument))
        elif opcode.name == "STACK_GLOBAL":
            globals_seen.add("<dynamic STACK_GLOBAL>")
    return {
        "member_name": member_name,
        "uncompressed_bytes": int(operations[-1][2] + 1),
        "referenced_globals": sorted(globals_seen),
        "execution_relevant_opcodes": sorted(opcodes_seen),
    }


def audit_checkpoint_container(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    """Inspect a checkpoint container without deserializing its pickle.

    The returned receipt never authorizes direct Python-pickle loading.  Its
    purpose is to bind a file hash and expose pickle globals/opcodes for human
    review before a separate tensor-only loader or isolated conversion step.
    """

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    actual_sha256, size = _stable_sha256(path)
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise ValueError("expected_sha256 must be lowercase SHA-256")
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "checkpoint SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    snapshot: bytes | None = None
    if size <= _MAX_LEGACY_CHECKPOINT_AUDIT_BYTES:
        snapshot = path.read_bytes()
        if hashlib.sha256(snapshot).hexdigest() != actual_sha256:
            raise RuntimeError("checkpoint changed after stable hashing")

    container_format = "unknown_binary"
    pickle_members: list[dict[str, Any]] = []
    zip_member_count = 0
    unparsed_trailing_bytes = 0
    if snapshot is not None and zipfile.is_zipfile(io.BytesIO(snapshot)):
        container_format = "pytorch_zip_pickle"
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            infos = archive.infolist()
            zip_member_count = len(infos)
            if zip_member_count > _MAX_ZIP_MEMBER_COUNT:
                raise ValueError("checkpoint ZIP contains too many members")
            for info in infos:
                if not info.filename.endswith("data.pkl"):
                    continue
                if info.file_size > _MAX_PICKLE_MEMBER_BYTES:
                    raise ValueError("checkpoint pickle metadata member is too large")
                payload = archive.read(info)
                inspected = _inspect_pickle_program(
                    payload, member_name=info.filename
                )
                inspected["uncompressed_bytes"] = int(info.file_size)
                pickle_members.append(inspected)
    elif snapshot is not None and _looks_like_safetensors(path, snapshot):
        container_format = "safetensors"
    elif snapshot is not None:
        # Legacy PyTorch serialization is a sequence of pickle programs
        # followed by raw tensor storage bytes.  Parsing opcodes does not call
        # any referenced GLOBAL/REDUCE target.
        payload = snapshot
        offset = 0
        while offset < len(payload) and payload[offset] == 0x80:
            try:
                inspected = _inspect_pickle_program(
                    payload[offset:],
                    member_name=f"legacy_pickle_stream_{len(pickle_members):02d}",
                )
            except ValueError:
                break
            consumed = int(inspected["uncompressed_bytes"])
            pickle_members.append(inspected)
            offset += consumed
        if pickle_members:
            container_format = "pytorch_legacy_pickle"
            unparsed_trailing_bytes = len(payload) - offset

    if container_format == "safetensors":
        loader_requirement = "hash_allowlist_plus_safetensors_parser"
    elif container_format == "pytorch_zip_pickle":
        loader_requirement = (
            "hash_allowlist_plus_static_audit_plus_weights_only_or_isolated_conversion"
        )
    elif container_format == "pytorch_legacy_pickle":
        loader_requirement = "isolated_conversion_required_no_inprocess_pickle_load"
    else:
        loader_requirement = "unsupported_until_isolated_format_audit"

    body: dict[str, Any] = {
        "schema_version": CHECKPOINT_STATIC_AUDIT_SCHEMA_VERSION,
        "audit_id": "CHECKPOINT-AUDIT-PENDING",
        "artifact_id": _nonempty_string(
            artifact_id if artifact_id is not None else path.name,
            "artifact_id",
        ),
        "filename": path.name,
        "size_bytes": int(size),
        "sha256": actual_sha256,
        "container_format": container_format,
        "zip_member_count": zip_member_count,
        "unparsed_trailing_bytes": unparsed_trailing_bytes,
        "pickle_members": pickle_members,
        "direct_python_pickle_load_allowed": False,
        "static_audit_executes_pickle": False,
        "loader_requirement": loader_requirement,
    }
    body["audit_id"] = "CKPTAUD-" + _canonical_sha256(body)[:24]
    return body


def checkpoint_manifest_sha256(artifacts: Sequence[Mapping[str, Any]]) -> str:
    """Hash an ordered artifact manifest without opening any checkpoint."""

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(artifacts):
        if type(raw) is not dict:
            raise TypeError(f"checkpoint artifact {index} must be an object")
        artifact_id = _nonempty_string(raw.get("artifact_id"), "artifact_id")
        if artifact_id in seen:
            raise ValueError("checkpoint artifact IDs must be unique")
        seen.add(artifact_id)
        sha256 = raw.get("sha256")
        if not _is_sha256(sha256):
            raise ValueError("checkpoint artifact SHA-256 is invalid")
        size = raw.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("checkpoint artifact size must be a positive integer")
        normalized.append(
            {"artifact_id": artifact_id, "sha256": sha256, "size_bytes": size}
        )
    if not normalized:
        raise ValueError("checkpoint artifact manifest must not be empty")
    normalized.sort(key=lambda item: item["artifact_id"])
    return _canonical_sha256(normalized)


def _validate_full_record_posterior_rows(
    value: object,
    *,
    recording_duration_seconds: float,
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("full-record posterior timeline must be a sequence")
    required = {
        "window_id",
        "target_start_offset_seconds",
        "target_stop_offset_seconds",
        "observed_support_start_offset_seconds",
        "observed_support_stop_offset_seconds",
        "decision_available_offset_seconds",
        "future_lookahead_seconds",
        "right_padding_seconds",
        "seizure_probability",
        "signal_usable",
        "row_status",
    }
    rows: list[dict[str, Any]] = []
    window_ids: set[str] = set()
    previous_target_start = -math.inf
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != required:
            raise ValueError(
                f"full-record posterior timeline row {index} has invalid fields"
            )
        row = deepcopy(raw)
        window_id = _nonempty_string(row["window_id"], "posterior window_id")
        if window_id in window_ids:
            raise ValueError("full-record posterior window IDs must be unique")
        window_ids.add(window_id)
        start = _finite_number(
            row["target_start_offset_seconds"], "posterior target start"
        )
        stop = _finite_number(
            row["target_stop_offset_seconds"], "posterior target stop"
        )
        if (
            start < 0
            or stop <= start
            or stop > recording_duration_seconds + 1e-9
            or start < previous_target_start - 1e-9
        ):
            raise ValueError("full-record posterior target interval is invalid")
        previous_target_start = start
        if type(row["signal_usable"]) is not bool:
            raise TypeError("full-record posterior signal_usable must be boolean")
        status = row["row_status"]
        if status not in FULL_RECORD_POSTERIOR_ROW_STATUSES:
            raise ValueError("full-record posterior row status is invalid")
        probability = _finite_number(
            row["seizure_probability"], "posterior seizure probability"
        )
        if not 0 <= probability <= 1:
            raise ValueError("posterior seizure probability must be in [0,1]")
        padding = _finite_number(
            row["right_padding_seconds"], "posterior right padding"
        )
        if padding < 0:
            raise ValueError("posterior right padding must be non-negative")

        if status == "unmodeled_partial_tail":
            if (
                row["signal_usable"] is not False
                or probability != 0.0
                or padding != 0.0
                or any(
                    row[name] is not None
                    for name in (
                        "observed_support_start_offset_seconds",
                        "observed_support_stop_offset_seconds",
                        "decision_available_offset_seconds",
                        "future_lookahead_seconds",
                    )
                )
            ):
                raise ValueError(
                    "unmodeled partial tail must contain no observed/model evidence"
                )
        else:
            support_start = _finite_number(
                row["observed_support_start_offset_seconds"],
                "posterior observed support start",
            )
            support_stop = _finite_number(
                row["observed_support_stop_offset_seconds"],
                "posterior observed support stop",
            )
            decision_available = _finite_number(
                row["decision_available_offset_seconds"],
                "posterior decision availability",
            )
            future_lookahead = _finite_number(
                row["future_lookahead_seconds"], "posterior future lookahead"
            )
            if (
                support_start < 0
                or support_stop <= support_start
                or support_stop > recording_duration_seconds + 1e-9
                or support_start > start + 1e-9
                or support_stop < stop - 1e-9
            ):
                raise ValueError(
                    "posterior observed support does not contain its target interval"
                )
            if abs(decision_available - support_stop) > 1e-9:
                raise ValueError(
                    "posterior decision availability must equal the time at which "
                    "all observed support is available"
                )
            expected_lookahead = max(0.0, support_stop - stop)
            if (
                future_lookahead < 0
                or abs(future_lookahead - expected_lookahead) > 1e-9
            ):
                raise ValueError(
                    "posterior future lookahead does not bind observed support"
                )
            if padding > 0 and abs(support_stop - recording_duration_seconds) > 1e-9:
                raise ValueError(
                    "right-padded provider rows must end observed support at recording EOF"
                )
            if status == "modeled" and row["signal_usable"] is not True:
                raise ValueError("modeled posterior row must be signal-usable")
            if status == "signal_unusable" and (
                row["signal_usable"] is not False or probability != 0.0
            ):
                raise ValueError(
                    "signal-unusable posterior row must carry a neutral probability"
                )

        rows.append(
            {
                "window_id": window_id,
                "target_start_offset_seconds": start,
                "target_stop_offset_seconds": stop,
                "observed_support_start_offset_seconds": (
                    None
                    if status == "unmodeled_partial_tail"
                    else float(row["observed_support_start_offset_seconds"])
                ),
                "observed_support_stop_offset_seconds": (
                    None
                    if status == "unmodeled_partial_tail"
                    else float(row["observed_support_stop_offset_seconds"])
                ),
                "decision_available_offset_seconds": (
                    None
                    if status == "unmodeled_partial_tail"
                    else float(row["decision_available_offset_seconds"])
                ),
                "future_lookahead_seconds": (
                    None
                    if status == "unmodeled_partial_tail"
                    else float(row["future_lookahead_seconds"])
                ),
                "right_padding_seconds": padding,
                "seizure_probability": probability,
                "signal_usable": row["signal_usable"],
                "row_status": status,
            }
        )

    partial_indices = [
        index
        for index, row in enumerate(rows)
        if row["row_status"] == "unmodeled_partial_tail"
    ]
    if partial_indices:
        if len(partial_indices) != 1 or partial_indices[0] != len(rows) - 1:
            raise ValueError("unmodeled partial tail must be one final row")
        tail = rows[-1]
        if abs(tail["target_stop_offset_seconds"] - recording_duration_seconds) > 1e-9:
            raise ValueError("unmodeled partial tail must terminate at recording EOF")
    return rows


def _target_coverage_receipt(
    rows: Sequence[Mapping[str, Any]],
    *,
    recording_duration_seconds: float,
    complete_recording_scan_attempted: bool,
) -> dict[str, Any]:
    intervals = sorted(
        (
            float(row["target_start_offset_seconds"]),
            float(row["target_stop_offset_seconds"]),
        )
        for row in rows
        if row["row_status"] != "unmodeled_partial_tail"
    )
    merged: list[list[float]] = []
    for start, stop in intervals:
        if not merged or start > merged[-1][1] + 1e-9:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    modeled = sum(stop - start for start, stop in merged)
    gaps: list[float] = []
    cursor = 0.0
    for start, stop in merged:
        gaps.append(max(0.0, start - cursor))
        cursor = max(cursor, stop)
    gaps.append(max(0.0, recording_duration_seconds - cursor))
    maximum_gap = max(gaps) if gaps else recording_duration_seconds
    unmodeled = max(0.0, recording_duration_seconds - modeled)
    tail_rows = [
        row for row in rows if row["row_status"] == "unmodeled_partial_tail"
    ]
    tail_seconds = (
        float(tail_rows[0]["target_stop_offset_seconds"])
        - float(tail_rows[0]["target_start_offset_seconds"])
        if tail_rows
        else 0.0
    )
    maximum_padding = max(
        [float(row["right_padding_seconds"]) for row in rows] or [0.0]
    )
    complete = unmodeled <= 1e-9

    if tail_rows:
        tail_start = float(tail_rows[0]["target_start_offset_seconds"])
        valid_prefix = (
            abs(tail_start) <= 1e-9
            if not merged
            else (
                len(merged) == 1
                and abs(merged[0][0]) <= 1e-9
                and abs(merged[-1][1] - tail_start) <= 1e-9
            )
        )
        if not valid_prefix:
            raise ValueError(
                "declared partial tail cannot hide an earlier posterior coverage gap"
            )
        partial_tail_policy = "declared_unmodeled_tail"
    elif not complete:
        raise ValueError(
            "posterior target coverage gap is not an explicit final partial tail"
        )
    elif maximum_padding > 0:
        partial_tail_policy = "modeled_with_explicit_right_padding"
    else:
        partial_tail_policy = "none_complete"

    return {
        "complete_recording_scan_attempted": complete_recording_scan_attempted,
        "posterior_target_coverage_complete": complete,
        "modeled_target_coverage_seconds": modeled,
        "unmodeled_target_coverage_seconds": unmodeled,
        "maximum_target_coverage_gap_seconds": maximum_gap,
        "declared_partial_tail_seconds": tail_seconds,
        "maximum_right_padding_seconds": maximum_padding,
        "partial_tail_policy": partial_tail_policy,
    }


def _validate_full_record_technical_failure(value: object) -> dict[str, Any]:
    required = {
        "failure_code",
        "failure_stage",
        "retryable",
        "failure_detail_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("technical failure has missing or unknown fields")
    result = deepcopy(value)
    _nonempty_string(result["failure_code"], "technical failure code")
    if result["failure_stage"] not in FULL_RECORD_FAILURE_STAGES:
        raise ValueError("technical failure stage is invalid")
    if type(result["retryable"]) is not bool:
        raise TypeError("technical failure retryable must be boolean")
    if not _is_sha256(result["failure_detail_sha256"]):
        raise ValueError("technical failure detail SHA-256 is invalid")
    return result


def materialize_full_record_provider_result(
    *,
    provider_id: str,
    provider_execution_receipt_id: str,
    recording_id: str,
    source_signal_sha256: str,
    recording_duration_seconds: float,
    posterior_timeline: Sequence[Mapping[str, Any]],
    decoding_receipt_id: str | None = None,
    decoder_policy_sha256: str | None = None,
    event_proposal_count: int | None = None,
    complete_recording_scan_attempted: bool = True,
    technical_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one provider attempt, its temporal support, and terminal outcome.

    A valid zero-alarm completion is represented only when a non-failed run
    has complete posterior target coverage and a frozen decoder emitted zero
    proposals.  Missing/partial coverage and technical failure remain distinct
    outcomes and cannot be silently counted as zero alarms.
    """

    provider_id = _nonempty_string(provider_id, "provider_id")
    execution_id = _nonempty_string(
        provider_execution_receipt_id, "provider_execution_receipt_id"
    )
    recording_id = _nonempty_string(recording_id, "recording_id")
    if not _is_sha256(source_signal_sha256):
        raise ValueError("source signal SHA-256 is invalid")
    duration = _finite_number(
        recording_duration_seconds, "recording_duration_seconds"
    )
    if duration <= 0:
        raise ValueError("recording duration must be positive")
    if type(complete_recording_scan_attempted) is not bool:
        raise TypeError("complete_recording_scan_attempted must be boolean")

    if technical_failure is not None:
        if posterior_timeline:
            raise ValueError("technical failure cannot carry posterior rows")
        if any(
            value is not None
            for value in (
                decoding_receipt_id,
                decoder_policy_sha256,
                event_proposal_count,
            )
        ):
            raise ValueError("technical failure cannot carry a decoder outcome")
        rows: list[dict[str, Any]] = []
        failure = _validate_full_record_technical_failure(dict(technical_failure))
        decoder_outcome = None
        outcome_status = "technical_failure"
        coverage = {
            "complete_recording_scan_attempted": complete_recording_scan_attempted,
            "posterior_target_coverage_complete": False,
            "modeled_target_coverage_seconds": 0.0,
            "unmodeled_target_coverage_seconds": duration,
            "maximum_target_coverage_gap_seconds": duration,
            "declared_partial_tail_seconds": 0.0,
            "maximum_right_padding_seconds": 0.0,
            "partial_tail_policy": "not_applicable_technical_failure",
        }
    else:
        if complete_recording_scan_attempted is not True:
            raise ValueError("completed provider result must attempt the full recording")
        rows = _validate_full_record_posterior_rows(
            posterior_timeline, recording_duration_seconds=duration
        )
        if not rows:
            raise ValueError("completed provider result requires posterior rows")
        if decoding_receipt_id is None or decoder_policy_sha256 is None:
            raise ValueError("completed provider result requires decoder binding")
        if event_proposal_count is None:
            raise ValueError("completed provider result requires event proposal count")
        decoder_outcome = {
            "decoding_receipt_id": _nonempty_string(
                decoding_receipt_id, "decoding_receipt_id"
            ),
            "decoder_policy_sha256": decoder_policy_sha256,
            "event_proposal_count": _nonnegative_integer(
                event_proposal_count, "event_proposal_count"
            ),
            "zero_candidates_is_valid": True,
        }
        if not _is_sha256(decoder_outcome["decoder_policy_sha256"]):
            raise ValueError("decoder policy SHA-256 is invalid")
        coverage = _target_coverage_receipt(
            rows,
            recording_duration_seconds=duration,
            complete_recording_scan_attempted=True,
        )
        if coverage["posterior_target_coverage_complete"] is not True:
            outcome_status = "partial_coverage"
        elif decoder_outcome["event_proposal_count"] == 0:
            outcome_status = "completed_zero_alarm"
        else:
            outcome_status = "completed_with_alarms"
        failure = None

    body: dict[str, Any] = {
        "schema_version": FULL_RECORD_PROVIDER_RESULT_SCHEMA_VERSION,
        "result_id": "FULL-RECORD-PROVIDER-RESULT-PENDING",
        "provider_id": provider_id,
        "provider_execution_receipt_id": execution_id,
        "recording_id": recording_id,
        "source_signal_sha256": source_signal_sha256,
        "recording_duration_seconds": duration,
        "outcome_status": outcome_status,
        "time_semantics": deepcopy(FULL_RECORD_TIME_SEMANTICS),
        "posterior_timeline": rows,
        "decoder_outcome": decoder_outcome,
        "coverage_receipt": coverage,
        "technical_failure": failure,
    }
    body["result_id"] = "DETFULL-" + _canonical_sha256(body)[:24]
    return validate_full_record_provider_result(body)


def validate_full_record_provider_result(payload: object) -> dict[str, Any]:
    """Validate full-record time, coverage, zero-alarm, and failure semantics."""

    required = {
        "schema_version",
        "result_id",
        "provider_id",
        "provider_execution_receipt_id",
        "recording_id",
        "source_signal_sha256",
        "recording_duration_seconds",
        "outcome_status",
        "time_semantics",
        "posterior_timeline",
        "decoder_outcome",
        "coverage_receipt",
        "technical_failure",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("full-record provider result has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != FULL_RECORD_PROVIDER_RESULT_SCHEMA_VERSION:
        raise ValueError("full-record provider result schema drifted")
    _nonempty_string(data["provider_id"], "provider_id")
    _nonempty_string(
        data["provider_execution_receipt_id"], "provider_execution_receipt_id"
    )
    _nonempty_string(data["recording_id"], "recording_id")
    if not _is_sha256(data["source_signal_sha256"]):
        raise ValueError("source signal SHA-256 is invalid")
    duration = _finite_number(
        data["recording_duration_seconds"], "recording_duration_seconds"
    )
    if duration <= 0:
        raise ValueError("recording duration must be positive")
    if data["outcome_status"] not in FULL_RECORD_PROVIDER_OUTCOMES:
        raise ValueError("full-record provider outcome is invalid")
    if data["time_semantics"] != FULL_RECORD_TIME_SEMANTICS:
        raise ValueError("full-record provider time semantics drifted")

    if data["outcome_status"] == "technical_failure":
        if data["posterior_timeline"] != [] or data["decoder_outcome"] is not None:
            raise ValueError("technical failure cannot masquerade as a zero alarm")
        failure = _validate_full_record_technical_failure(data["technical_failure"])
        if failure != data["technical_failure"]:
            raise ValueError("technical failure normalization drifted")
        coverage_receipt = data["coverage_receipt"]
        if type(coverage_receipt) is not dict:
            raise TypeError("technical failure coverage receipt must be an object")
        expected_coverage = {
            "complete_recording_scan_attempted": coverage_receipt.get(
                "complete_recording_scan_attempted"
            ),
            "posterior_target_coverage_complete": False,
            "modeled_target_coverage_seconds": 0.0,
            "unmodeled_target_coverage_seconds": duration,
            "maximum_target_coverage_gap_seconds": duration,
            "declared_partial_tail_seconds": 0.0,
            "maximum_right_padding_seconds": 0.0,
            "partial_tail_policy": "not_applicable_technical_failure",
        }
        if type(expected_coverage["complete_recording_scan_attempted"]) is not bool:
            raise TypeError("technical failure scan-attempt flag must be boolean")
    else:
        if data["technical_failure"] is not None:
            raise ValueError("completed provider result cannot carry a failure")
        rows = _validate_full_record_posterior_rows(
            data["posterior_timeline"], recording_duration_seconds=duration
        )
        if not rows:
            raise ValueError("non-failed provider result requires posterior rows")
        decoder = data["decoder_outcome"]
        expected_decoder_fields = {
            "decoding_receipt_id",
            "decoder_policy_sha256",
            "event_proposal_count",
            "zero_candidates_is_valid",
        }
        if type(decoder) is not dict or set(decoder) != expected_decoder_fields:
            raise ValueError("provider decoder outcome has invalid fields")
        _nonempty_string(decoder["decoding_receipt_id"], "decoding_receipt_id")
        if not _is_sha256(decoder["decoder_policy_sha256"]):
            raise ValueError("decoder policy SHA-256 is invalid")
        count = _nonnegative_integer(
            decoder["event_proposal_count"], "event_proposal_count"
        )
        if decoder["zero_candidates_is_valid"] is not True:
            raise ValueError("provider result must retain valid zero-alarm records")
        expected_coverage = _target_coverage_receipt(
            rows,
            recording_duration_seconds=duration,
            complete_recording_scan_attempted=True,
        )
        expected_status = (
            "partial_coverage"
            if expected_coverage["posterior_target_coverage_complete"] is not True
            else "completed_zero_alarm"
            if count == 0
            else "completed_with_alarms"
        )
        if data["outcome_status"] != expected_status:
            raise ValueError("provider outcome disagrees with coverage/alarm count")
    if data["coverage_receipt"] != expected_coverage:
        raise ValueError("provider coverage receipt is not replayable")

    digest = deepcopy(data)
    digest["result_id"] = "FULL-RECORD-PROVIDER-RESULT-PENDING"
    if data["result_id"] != "DETFULL-" + _canonical_sha256(digest)[:24]:
        raise ValueError("full-record provider result ID does not bind its content")
    return data


def validate_provider_definition(payload: object) -> dict[str, Any]:
    """Validate the executable subset of one provider-registry entry."""

    if type(payload) is not dict:
        raise TypeError("provider definition must be an object")
    required = {
        "provider_id",
        "model_family",
        "research_role",
        "implementation_status",
        "qualification_status",
        "weights_manifest_sha256",
        "adapter_code_sha256",
        "checkpoint_loader_policy",
        "training_corpus",
        "posterior_calibration_status",
        "continuous_operating_point_status",
        "eeg_signal_only",
        "edf_annotations_allowed",
        "excel_or_clinical_labels_allowed",
        "claimed_sota",
    }
    if set(payload) != required:
        raise ValueError("provider definition has missing or unknown fields")
    result = deepcopy(payload)
    for name in ("provider_id", "model_family", "training_corpus"):
        _nonempty_string(result[name], name)
    if result["research_role"] not in RESEARCH_ROLES:
        raise ValueError("provider research role is invalid")
    if result["implementation_status"] not in IMPLEMENTATION_STATUSES:
        raise ValueError("provider implementation status is invalid")
    if result["qualification_status"] not in QUALIFICATION_STATUSES:
        raise ValueError("provider qualification status is invalid")
    if result["checkpoint_loader_policy"] not in LOADER_POLICIES:
        raise ValueError("provider checkpoint loader policy is invalid")
    for name in ("weights_manifest_sha256", "adapter_code_sha256"):
        if result[name] is not None and not _is_sha256(result[name]):
            raise ValueError(f"provider {name} is invalid")
    for name, expected in (
        ("eeg_signal_only", True),
        ("edf_annotations_allowed", False),
        ("excel_or_clinical_labels_allowed", False),
        ("claimed_sota", False),
    ):
        if result[name] is not expected:
            raise ValueError(f"provider {name} must be {expected!r}")
    if result["implementation_status"] in {"runnable_research", "production_qualified"}:
        if result["weights_manifest_sha256"] is None:
            raise ValueError("runnable provider must bind its weight manifest")
        if result["adapter_code_sha256"] is None:
            raise ValueError("runnable provider must bind its adapter code")
    if result["implementation_status"] == "production_qualified" and result[
        "qualification_status"
    ] != "passed_external_promotion_gate":
        raise ValueError("production provider lacks an external promotion gate")
    if result["checkpoint_loader_policy"] == (
        "isolated_conversion_required_no_inprocess_load"
    ) and result["implementation_status"] in {
        "runnable_research",
        "production_qualified",
    }:
        raise ValueError("provider requiring isolated conversion is not runnable")
    _nonempty_string(
        result["posterior_calibration_status"], "posterior_calibration_status"
    )
    _nonempty_string(
        result["continuous_operating_point_status"],
        "continuous_operating_point_status",
    )
    return result


def validate_provider_registry(payload: object) -> dict[str, Any]:
    """Validate role selection and the fail-closed production state."""

    if type(payload) is not dict:
        raise TypeError("provider registry must be an object")
    required = {
        "schema_version",
        "registry_status",
        "selection_policy",
        "promotion_requirements",
        "providers",
    }
    if set(payload) != required:
        raise ValueError("provider registry has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != PROVIDER_REGISTRY_SCHEMA_VERSION:
        raise ValueError("provider registry schema drifted")
    if data["registry_status"] not in {
        "research_only_fail_closed",
        "production_qualified",
    }:
        raise ValueError("provider registry status is invalid")
    if not isinstance(data["providers"], list) or not data["providers"]:
        raise TypeError("provider registry providers must be a non-empty array")
    providers: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(data["providers"]):
        if type(raw) is not dict or set(raw) != {
            "execution_definition",
            "scientific_scope",
            "local_checkpoint_audit",
            "license_and_redistribution",
        }:
            raise ValueError(f"provider registry entry {index} has invalid fields")
        definition = validate_provider_definition(raw["execution_definition"])
        provider_id = definition["provider_id"]
        if provider_id in providers:
            raise ValueError("provider registry IDs must be unique")
        audit = raw["local_checkpoint_audit"]
        if not isinstance(audit, dict) or audit.get("direct_unpickling_allowed") is not False:
            raise ValueError("every provider must explicitly forbid direct unpickling")
        artifacts = audit.get("artifacts")
        if artifacts is not None:
            actual_manifest = checkpoint_manifest_sha256(artifacts)
            if actual_manifest != definition["weights_manifest_sha256"]:
                raise ValueError("provider weight artifact manifest hash drifted")
        providers[provider_id] = definition

    policy = data["selection_policy"]
    expected_policy = {
        "target_domain_primary_research_provider_id",
        "shadow_comparator_provider_ids",
        "representation_only_provider_ids",
        "production_provider_id",
        "fail_closed_without_deployment_qualified_provider",
        "research_primary_rationale",
        "sota_claim_authorized",
    }
    if type(policy) is not dict or set(policy) != expected_policy:
        raise ValueError("provider selection policy has invalid fields")
    primary_id = policy["target_domain_primary_research_provider_id"]
    if primary_id not in providers or providers[primary_id]["research_role"] != (
        "target_domain_primary_research_candidate"
    ):
        raise ValueError("target-domain primary provider role is invalid")
    role_lists = (
        ("shadow_comparator_provider_ids", "shadow_continuous_comparator"),
        ("representation_only_provider_ids", "representation_only_future_candidate"),
    )
    selected_ids = {primary_id}
    for field, expected_role in role_lists:
        values = policy[field]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError(f"provider policy {field} is invalid")
        for provider_id in values:
            if provider_id not in providers or providers[provider_id][
                "research_role"
            ] != expected_role:
                raise ValueError(f"provider policy {field} contains a wrong role")
            if provider_id in selected_ids:
                raise ValueError("provider occurs in more than one selection role")
            selected_ids.add(provider_id)
    if selected_ids != set(providers):
        raise ValueError("provider selection policy does not cover every provider")
    if policy["fail_closed_without_deployment_qualified_provider"] is not True:
        raise ValueError("provider registry must fail closed without production model")
    if policy["sota_claim_authorized"] is not False:
        raise ValueError("unverified provider registry must not claim SOTA")
    _nonempty_string(policy["research_primary_rationale"], "research rationale")

    production_id = policy["production_provider_id"]
    if production_id is None:
        if data["registry_status"] != "research_only_fail_closed":
            raise ValueError("registry without production provider is not research-only")
    else:
        if (
            production_id not in providers
            or providers[production_id]["implementation_status"]
            != "production_qualified"
            or data["registry_status"] != "production_qualified"
        ):
            raise ValueError("production provider selection is not qualified")

    promotion = data["promotion_requirements"]
    required_promotion = {
        "patient_level_development_evaluation_isolation",
        "operating_point_frozen_before_evaluation",
        "source_dev_operating_point_calibration_receipt_required",
        "complete_source_dev_recording_inventory_required",
        "zero_alarm_records_retained_during_calibration",
        "patient_macro_recall_constraint_required",
        "complete_long_recording_scan",
        "seizure_free_recordings_required",
        "event_sensitivity_required",
        "event_precision_f1_required",
        "alarm_count_false_alarms_per_hour_required",
        "alarm_count_false_alarms_per_24h_required",
        "onset_latency_median_iqr_required",
        "onset_absolute_error_and_coverage_required",
        "onset_absolute_hit_rates_seconds",
        "event_iou_required",
        "typed_boundary_f1_required",
        "patient_bootstrap_confidence_intervals_required",
        "external_long_recording_validation_required",
        "threshold_values_status",
        "generic_benchmark_receipt_can_promote_production",
    }
    if type(promotion) is not dict or set(promotion) != required_promotion:
        raise ValueError("provider promotion requirements have invalid fields")
    boolean_true_fields = required_promotion.difference(
        {
            "onset_absolute_hit_rates_seconds",
            "threshold_values_status",
            "generic_benchmark_receipt_can_promote_production",
        }
    )
    if any(promotion[field] is not True for field in boolean_true_fields):
        raise ValueError("provider promotion requirements were weakened")
    if promotion["onset_absolute_hit_rates_seconds"] != [1, 3, 5, 10]:
        raise ValueError("provider onset hit-rate tolerances drifted")
    if promotion["generic_benchmark_receipt_can_promote_production"] is not False:
        raise ValueError("metrics-only benchmark must not promote production")
    if promotion["threshold_values_status"] != (
        "must_be_prospectively_frozen_before_promotion_evaluation"
    ):
        raise ValueError("provider promotion thresholds are not prospectively frozen")
    return data


def validate_promotion_receipt(payload: object) -> dict[str, Any]:
    """Validate the portable shape of a provider-promotion candidate.

    This validator does not establish independent issuance.  Until a trusted
    evidence registry cross-checks every promotion claim, the candidate cannot
    authorize production execution.
    """

    if type(payload) is not dict:
        raise TypeError("promotion receipt must be an object")
    required = {
        "schema_version",
        "receipt_id",
        "provider_id",
        "weights_manifest_sha256",
        "adapter_code_sha256",
        "patient_level_split_verified",
        "operating_point_frozen_before_evaluation",
        "source_dev_operating_point_calibration_receipt_verified",
        "complete_source_dev_recording_inventory_verified",
        "zero_alarm_records_retained_during_calibration",
        "patient_macro_recall_constraint_verified",
        "independent_external_long_recording_evaluation",
        "seizure_free_recordings_included",
        "event_sensitivity_reported",
        "event_precision_f1_reported",
        "alarm_false_alarms_per_hour_reported",
        "alarm_false_alarms_per_24h_reported",
        "onset_latency_reported",
        "onset_absolute_error_and_coverage_reported",
        "onset_tolerance_hit_rates_reported",
        "event_iou_reported",
        "boundary_f1_reported",
        "patient_bootstrap_confidence_intervals_reported",
        "approved_for_production",
    }
    if set(payload) != required:
        raise ValueError("promotion receipt has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != PROVIDER_PROMOTION_SCHEMA_VERSION:
        raise ValueError("promotion receipt schema drifted")
    _nonempty_string(data["provider_id"], "promotion provider_id")
    for name in ("weights_manifest_sha256", "adapter_code_sha256"):
        if not _is_sha256(data[name]):
            raise ValueError(f"promotion {name} is invalid")
    boolean_fields = required.difference(
        {
            "schema_version",
            "receipt_id",
            "provider_id",
            "weights_manifest_sha256",
            "adapter_code_sha256",
        }
    )
    for name in boolean_fields:
        if data[name] is not True:
            raise ValueError(f"promotion gate {name} must be true")
    digest = deepcopy(data)
    digest["receipt_id"] = "PROMOTION-PENDING"
    expected = "DETPROMO-" + _canonical_sha256(digest)[:24]
    if data["receipt_id"] != expected:
        raise ValueError("promotion receipt ID does not bind its content")
    return data


def authorize_provider_execution(
    provider_definition: Mapping[str, Any],
    *,
    requested_role: str,
    promotion_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authorize research or production execution without loading the model."""

    provider = validate_provider_definition(dict(provider_definition))
    if requested_role not in {"research", "production"}:
        raise ValueError("requested_role must be research or production")
    if provider["implementation_status"] not in {
        "runnable_research",
        "production_qualified",
    }:
        raise ProviderAuthorizationError(
            f"provider {provider['provider_id']} adapter is not runnable"
        )
    if provider["checkpoint_loader_policy"] not in {
        "hash_allowlist_then_torch_weights_only_true",
        "hash_allowlist_then_safetensors",
    }:
        raise ProviderAuthorizationError("provider has no approved in-process loader")

    promotion_sha256: str | None = None
    deployment_status = "research_candidate"
    if requested_role == "production":
        if provider["implementation_status"] != "production_qualified":
            raise ProviderAuthorizationError("provider is not production qualified")
        if not PRODUCTION_PROVIDER_EXECUTION_CONNECTED:
            raise ProviderAuthorizationError(
                "production provider execution is disconnected pending trusted "
                "promotion evidence registries"
            )
        if promotion_receipt is None:
            raise ProviderAuthorizationError("production requires a promotion receipt")
        promotion = validate_promotion_receipt(dict(promotion_receipt))
        for name in ("provider_id", "weights_manifest_sha256", "adapter_code_sha256"):
            if promotion[name] != provider[name]:
                raise ProviderAuthorizationError(
                    f"promotion receipt does not bind provider {name}"
                )
        promotion_sha256 = _canonical_sha256(promotion)
        deployment_status = "deployment_qualified"

    body: dict[str, Any] = {
        "schema_version": PROVIDER_EXECUTION_SCHEMA_VERSION,
        "execution_receipt_id": "PROVIDER-EXECUTION-PENDING",
        "provider_id": provider["provider_id"],
        "requested_role": requested_role,
        "model_family": provider["model_family"],
        "checkpoint_sha256": provider["weights_manifest_sha256"],
        "code_sha256": provider["adapter_code_sha256"],
        "training_corpus": provider["training_corpus"],
        "posterior_calibration_status": provider["posterior_calibration_status"],
        "deployment_qualification_status": deployment_status,
        "promotion_receipt_sha256": promotion_sha256,
        "annotations_used_for_current_recording": False,
        "labels_used_for_current_recording": False,
    }
    body["execution_receipt_id"] = "DETPROV-" + _canonical_sha256(body)[:24]
    return body


def to_continuous_decoder_provider_receipt(
    execution_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Project an authorization receipt to the existing decoder contract."""

    receipt = deepcopy(dict(execution_receipt))
    required = {
        "schema_version",
        "execution_receipt_id",
        "provider_id",
        "requested_role",
        "model_family",
        "checkpoint_sha256",
        "code_sha256",
        "training_corpus",
        "posterior_calibration_status",
        "deployment_qualification_status",
        "promotion_receipt_sha256",
        "annotations_used_for_current_recording",
        "labels_used_for_current_recording",
    }
    if set(receipt) != required or receipt["schema_version"] != (
        PROVIDER_EXECUTION_SCHEMA_VERSION
    ):
        raise ValueError("provider execution receipt is invalid")
    digest = deepcopy(receipt)
    digest["execution_receipt_id"] = "PROVIDER-EXECUTION-PENDING"
    if receipt["execution_receipt_id"] != (
        "DETPROV-" + _canonical_sha256(digest)[:24]
    ):
        raise ValueError("provider execution receipt content binding failed")
    if receipt["requested_role"] == "production":
        if not PRODUCTION_PROVIDER_EXECUTION_CONNECTED:
            raise ProviderAuthorizationError(
                "production decoder projection is disconnected pending trusted "
                "promotion evidence registries"
            )
        if (
            receipt["deployment_qualification_status"] != "deployment_qualified"
            or not _is_sha256(receipt["promotion_receipt_sha256"])
        ):
            raise ValueError("production provider lacks promotion evidence")
    else:
        if (
            receipt["requested_role"] != "research"
            or receipt["deployment_qualification_status"] != "research_candidate"
            or receipt["promotion_receipt_sha256"] is not None
        ):
            raise ValueError("research provider receipt drifted")
    return {
        "provider_id": receipt["provider_id"],
        "model_family": receipt["model_family"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "code_sha256": receipt["code_sha256"],
        "training_corpus": receipt["training_corpus"],
        "posterior_calibration_status": receipt["posterior_calibration_status"],
        "deployment_qualification_status": receipt[
            "deployment_qualification_status"
        ],
        "annotations_used_for_current_recording": False,
        "labels_used_for_current_recording": False,
    }


__all__ = [
    "CHECKPOINT_STATIC_AUDIT_SCHEMA_VERSION",
    "ContinuousPosteriorProvider",
    "FULL_RECORD_FAILURE_STAGES",
    "FULL_RECORD_POSTERIOR_ROW_STATUSES",
    "FULL_RECORD_PROVIDER_OUTCOMES",
    "FULL_RECORD_PROVIDER_RESULT_SCHEMA_VERSION",
    "FULL_RECORD_TIME_SEMANTICS",
    "PROVIDER_EXECUTION_SCHEMA_VERSION",
    "PROVIDER_PROMOTION_SCHEMA_VERSION",
    "PROVIDER_REGISTRY_SCHEMA_VERSION",
    "PRODUCTION_PROVIDER_EXECUTION_CONNECTED",
    "ProviderAuthorizationError",
    "audit_checkpoint_container",
    "authorize_provider_execution",
    "checkpoint_manifest_sha256",
    "materialize_full_record_provider_result",
    "to_continuous_decoder_provider_receipt",
    "validate_full_record_provider_result",
    "validate_promotion_receipt",
    "validate_provider_definition",
    "validate_provider_registry",
]
