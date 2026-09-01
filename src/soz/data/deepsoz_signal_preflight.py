"""Closed, replay-verified DeepSOZ/TUSZ causal signal-preflight bundles.

The bundle produced here is the only formal bridge between the verified
patient-level DeepSOZ target and event-level TUSZ signals.  It deliberately
replays the official global event timeline and
``load_standard19_edf_event`` for every candidate event.  Header-only flags in
legacy ``event_inputs.csv`` files are never used as eligibility decisions.

The on-disk artifact contains no target values and no EEG tensors.  It binds
the exact source/mapping/derived input tables, verified target-v2 receipt, EDF/annotation hashes,
fixed ``[-12,+48)`` event identity, preprocessing receipts, and a hash of the
processed event window.  Publication is atomic and an existing destination is
never overwritten.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, fields
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Mapping, Sequence

import pandas as pd
import torch

from .deepsoz import normalize_patient_id
from .deepsoz_target_v2 import VerifiedDeepSOZTargetV2Artifact
from .edf import (
    CAUSAL_IIR_GROUP_DELAY_ESTIMATOR,
    CAUSAL_IIR_INITIAL_STATE_POLICY,
    CAUSAL_IIR_PHASE_POLICY,
    CausalEDFConfig,
    EDFEventEligibilityError,
    EDFLoadReceipt,
    EDF_PREPROCESS_SCHEMA,
    load_standard19_edf_event,
)
from ..models.labram import (
    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
    bind_labram_record_positions,
)
from .tusz import inspect_tusz_annotation_pair
from ..signal import ChannelSignalQC, SignalProcessingReceipt


DEEPSOZ_SIGNAL_PREFLIGHT_SCHEMA = "soz_deepsoz_signal_preflight_v2"
DEEPSOZ_SIGNAL_PREFLIGHT_ARTIFACT_SCHEMA = (
    "soz_deepsoz_signal_preflight_artifact_v2"
)
DEEPSOZ_SIGNAL_PREFLIGHT_FILENAME = "deepsoz_signal_preflight.json"
DEEPSOZ_SIGNAL_SOURCE = "deepsoz_tusz_overlay"
DEEPSOZ_EVENT_ANCHOR = "tusz_csv_bi_TERM_seiz_start"
DEEPSOZ_SIGNAL_PREFLIGHT_POLICY = (
    "verified_target_v2_direct_physical19_causal_replay_only"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_INPUT_BYTES = 256 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_TIME_TOLERANCE_SEC = 1e-6
_MAPPING_TIME_TOLERANCE_SEC = 0.25
_OFFICIAL_SPLITS = ("train", "dev", "eval")
_MODEL_SPLIT_BY_OFFICIAL = {
    "train": "source_train",
    "dev": "source_dev",
    "eval": "source_eval",
}

_STANDARD19_SUFFIX_COLUMNS = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ",
)
_EVENT_INPUT_COLUMNS = (
    "source", "deepsoz_row", "deepsoz_patient_id", "patient_target_key",
    "deepsoz_record", "local_patient_id", "official_split", "event_id",
    "event_index", "local_edf_path", "local_csv_path", "local_csv_bi_path",
    "t0_sec", "t0_provenance", "seizure_end_sec", "seizure_duration_sec",
    "seizure_type", "window_start_sec", "window_stop_sec", "edf_duration_sec",
    "sfreq_hz", "header_read_ok", "full19_available",
    "missing_physical_channels", "full_minus12_plus48_in_bounds",
    "causal_warmup_30s_available", "signal_input_eligible",
    "warmup_signal_input_eligible", "fnsz_signal_input_eligible",
    "fnsz_warmup_signal_input_eligible", "signal_quarantine_reasons",
    *(f"signal_available_{channel}" for channel in _STANDARD19_SUFFIX_COLUMNS),
)
_CROSSWALK_INPUT_COLUMNS = (
    "source", "deepsoz_row", "deepsoz_patient_id", "deepsoz_record",
    "source_official_split", "source_event_count", "mapping_status",
    "candidate_count", "max_time_error_sec", "local_patient_id",
    "local_official_split", "split_agreement", "local_edf_path",
    "local_csv_path", "local_csv_bi_path", "local_edf_exists",
    "local_csv_exists", "local_csv_bi_exists", "candidate_local_csv_bi",
    "candidate_max_errors_sec", "patient_label_stability_primary",
    "patient_quarantine_reason", "header_read_ok", "header_error", "sfreq_hz",
    "edf_duration_sec", "n_raw_channels", "missing_physical_channels",
    "duplicate_physical_channels", "full19_available", "canonical_channel_map_json",
    *(f"signal_available_{channel}" for channel in _STANDARD19_SUFFIX_COLUMNS),
    *(f"raw_edf_name_{channel}" for channel in _STANDARD19_SUFFIX_COLUMNS),
)
_SPLIT_INPUT_COLUMNS = (
    "source", "deepsoz_patient_id", "local_patient_id", "official_split",
    "model_split", "cohort_status", "label_stability_primary",
    "source_record_count", "unique_mapped_record_count", "event_count",
    "signal_input_event_count", "warmup_signal_input_event_count",
    "primary_analysis_event_count", "warmup_primary_analysis_event_count",
    "strict_fnsz_event_count", "warmup_strict_fnsz_event_count",
    "concept_oof_fold", "oof_fold_scope", "ordinary_bce_ready",
    "weak_supervision_blocker", "oof_n_folds", "oof_fold_seed",
)
_CONSERVATIVE_MAPPING_INPUT_COLUMNS = (
    "deepsoz_row", "deepsoz_patient", "deepsoz_record", "local_patient",
    "local_csv_bi", "local_edf", "max_time_error_s", "candidate_count",
    "mapping_status", "candidate_local_csv_bi", "candidate_max_errors_s",
)
_DEEPSOZ_SOURCE_REQUIRED_COLUMNS = frozenset(
    {"pt_id", "fn", "loc", "nsz", "sz_starts", "sz_ends"}
)
_MAPPING_STATUSES = frozenset({"unique", "ambiguous", "unmapped"})
_SOURCE_TIMELINE_AUDIT_KEYS = tuple(
    (status, state)
    for status in ("unique", "ambiguous", "unmapped")
    for state in ("complete", "empty")
)

_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "event_record_sha256",
        "crosswalk_record_sha256",
        "deepsoz_source_record_sha256",
        "patient_id",
        "local_patient_id",
        "official_split",
        "model_split",
        "deepsoz_row",
        "deepsoz_record",
        "relative_edf_path",
        "relative_channel_annotation_path",
        "relative_global_annotation_path",
        "global_event_index",
        "global_t0_sec",
        "global_stop_sec",
        "global_seizure_type",
        "window_start_sec",
        "window_stop_sec",
        "edf_sha256",
        "channel_annotation_sha256",
        "global_annotation_sha256",
        "annotation_pair_sha256",
        "preprocess_config_sha256",
        "edf_receipt",
        "edf_receipt_sha256",
        "signal_receipt",
        "signal_receipt_sha256",
        "processed_window_sha256",
        "processed_window_shape",
        "processed_window_dtype",
    }
)
_EXCLUSION_FIELDS = frozenset(
    {
        "event_id",
        "event_record_sha256",
        "crosswalk_record_sha256",
        "deepsoz_source_record_sha256",
        "patient_id",
        "local_patient_id",
        "official_split",
        "model_split",
        "relative_edf_path",
        "deepsoz_record",
        "global_event_index",
        "global_t0_sec",
        "global_stop_sec",
        "edf_sha256",
        "annotation_pair_sha256",
        "eligibility_code",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "event_inputs_sha256",
        "record_crosswalk_sha256",
        "split_manifest_sha256",
        "deepsoz_source_sha256",
        "conservative_mapping_sha256",
        "verified_target_v2_receipt_sha256",
        "verified_target_v2_artifact_sha256",
        "verified_target_v2_policy_sha256",
        "preprocess_schema",
        "preprocess_config",
        "preprocess_config_sha256",
        "candidate_event_roster_sha256",
        "eligible_event_roster_sha256",
        "excluded_event_roster_sha256",
        "eligible_patient_roster_sha256",
        "eligible_split_patient_ids",
        "candidate_event_count",
        "source_record_count",
        "source_timeline_audit_counts",
        "negative_start_source_record_count",
        "negative_start_source_record_roster_sha256",
        "eligible_event_count",
        "excluded_event_count",
        "eligible_patient_count",
        "events",
        "exclusions",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "serialization",
        "receipt_sha256",
        "receipt",
    }
)
_EDF_RECEIPT_FIELDS = frozenset(field.name for field in fields(EDFLoadReceipt))
_SIGNAL_RECEIPT_FIELDS = frozenset(
    field.name for field in fields(SignalProcessingReceipt)
)
_CHANNEL_QC_FIELDS = frozenset(field.name for field in fields(ChannelSignalQC))


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Signal-preflight data are not canonical JSON") from exc
    return (encoded + "\n").encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)[:-1]).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _reject_symlink_components(path: Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot contain symlink components")
    return absolute


def _read_stable_regular_file(
    path: str | Path,
    *,
    field: str,
    max_bytes: int,
) -> tuple[bytes, str]:
    source = _reject_symlink_components(Path(path), field=field)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field} must be a regular non-symlinked file")
    before = source.stat()
    if before.st_size < 1 or before.st_size > max_bytes:
        raise ValueError(f"{field} has an invalid size")
    payload = source.read_bytes()
    after = source.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"{field} changed while it was read")
    return payload, _bytes_sha256(payload)


def _check_expected_sha(actual: str, expected: object, *, field: str) -> None:
    if actual != _require_sha256(expected, field=field):
        raise ValueError(f"{field} does not match the exact input bytes")


def _strict_csv(
    path: str | Path,
    *,
    expected_sha256: str,
    allowed_columns: Sequence[str],
    label: str,
) -> tuple[pd.DataFrame, str]:
    payload, actual_sha = _read_stable_regular_file(
        path, field=label, max_bytes=_MAX_INPUT_BYTES
    )
    _check_expected_sha(actual_sha, expected_sha256, field=f"expected_{label}_sha256")
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be strict UTF-8 CSV") from exc
    try:
        header = next(csv.reader(io.StringIO(text)))
    except (StopIteration, csv.Error) as exc:
        raise ValueError(f"{label} has no valid CSV header") from exc
    header = [str(value).strip() for value in header]
    if not header or any(not value for value in header):
        raise ValueError(f"{label} contains empty CSV columns")
    if len(set(header)) != len(header):
        raise ValueError(f"{label} contains duplicate CSV columns")
    expected_header = tuple(allowed_columns)
    if tuple(header) != expected_header:
        missing = sorted(set(expected_header) - set(header))
        unknown = sorted(set(header) - set(expected_header))
        raise ValueError(
            f"{label} violates its exact allowed CSV schema; "
            f"missing={missing}, unknown={unknown}, order_matches=False"
        )
    try:
        frame = pd.read_csv(
            io.StringIO(text), dtype=str, keep_default_na=False
        )
    except (csv.Error, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"{label} is not valid CSV") from exc
    if tuple(frame.columns) != tuple(header):
        raise ValueError(f"{label} CSV header changed during parsing")
    if frame.empty:
        raise ValueError(f"{label} cannot be empty")
    return frame, actual_sha


def _strict_deepsoz_source_csv(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[pd.DataFrame, str]:
    """Read the exact target-v2 source bytes while preserving legacy PZ twins."""

    payload, actual_sha = _read_stable_regular_file(
        path, field="deepsoz_source", max_bytes=_MAX_INPUT_BYTES
    )
    _check_expected_sha(
        actual_sha,
        expected_sha256,
        field="expected_deepsoz_source_sha256",
    )
    try:
        text = payload.decode("utf-8-sig", errors="strict")
        header = next(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise ValueError("deepsoz_source has no strict UTF-8 CSV header") from exc
    header = [str(value).strip() for value in header]
    if not header or any(not value for value in header):
        raise ValueError("deepsoz_source contains empty CSV columns")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates not in ([], ["pz"]) or header.count("pz") not in {1, 2}:
        raise ValueError(
            f"deepsoz_source contains unsupported duplicate columns: {duplicates}"
        )
    missing = sorted(_DEEPSOZ_SOURCE_REQUIRED_COLUMNS - set(header))
    if missing:
        raise ValueError(f"deepsoz_source is missing identity columns: {missing}")
    try:
        frame = pd.read_csv(
            io.StringIO(text, newline=""), dtype=str, keep_default_na=False
        )
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ValueError("deepsoz_source is not a valid CSV table") from exc
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("deepsoz_source must contain rows and unique parsed columns")
    if header.count("pz") == 2 and not {"pz", "pz.1"} <= set(frame.columns):
        raise ValueError("deepsoz_source did not preserve its legacy pz/pz.1 pair")
    return frame, actual_sha


def _clean(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} cannot be empty")
    return text


def _strict_int(value: object, *, field: str) -> int:
    text = _clean(value, field=field)
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ValueError(f"{field} must be an integer")
    return int(text)


def _strict_float(value: object, *, field: str) -> float:
    try:
        number = float(_clean(value, field=field))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _strict_one(value: object, *, field: str) -> None:
    if _clean(value, field=field) not in {"1", "1.0", "True", "true"}:
        raise ValueError(f"{field} must be explicit true/1")


def _relative_source_path(root: Path, value: object, *, field: str) -> tuple[str, Path]:
    text = _clean(value, field=field)
    if "\\" in text:
        raise ValueError(f"{field} must use canonical forward-slash separators")
    relative = Path(text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{field} must be a normalized relative path")
    candidate = _reject_symlink_components(root / relative, field=field)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the frozen TUSZ root") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{field} must resolve to a regular source file")
    return relative.as_posix(), candidate


def _mapping_source_path(
    root: Path, value: object, *, field: str
) -> tuple[str, Path]:
    text = _clean(value, field=field)
    declared = Path(text)
    if declared.is_absolute():
        absolute = _reject_symlink_components(declared, field=field)
        try:
            relative = absolute.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{field} escapes the frozen TUSZ root") from exc
        return _relative_source_path(root, relative.as_posix(), field=field)
    return _relative_source_path(root, text, field=field)


def _canonical_tusz_record_identity(
    root: Path,
    edf_value: object,
    channel_value: object,
    global_value: object,
) -> tuple[str, Path, str, Path, str, Path, str, str, tuple[str, str]]:
    relative_edf, edf_path = _relative_source_path(
        root, edf_value, field="local_edf_path"
    )
    relative_channel, channel_path = _relative_source_path(
        root, channel_value, field="local_csv_path"
    )
    relative_global, global_path = _relative_source_path(
        root, global_value, field="local_csv_bi_path"
    )
    expected_channel = Path(relative_edf).with_suffix(".csv").as_posix()
    expected_global = Path(relative_edf).with_suffix(".csv_bi").as_posix()
    if relative_channel != expected_channel or relative_global != expected_global:
        raise ValueError(
            "EDF/csv/csv_bi must be exact same-record standard sidecars"
        )
    parts = Path(relative_edf).parts
    if len(parts) != 5 or parts[0] not in _OFFICIAL_SPLITS:
        raise ValueError("Local EDF path is not a canonical TUSZ five-level path")
    split, patient_id, session_directory, montage_directory, filename = parts
    if not re.fullmatch(r"[a-z0-9]+", patient_id):
        raise ValueError("TUSZ patient path component is not canonical")
    session_match = re.fullmatch(r"(s\d{3})_\d{4}", session_directory)
    if session_match is None:
        raise ValueError("TUSZ session directory is not canonical")
    if re.fullmatch(r"\d{2}_tcp_(?:ar|le)(?:_a)?", montage_directory) is None:
        raise ValueError("TUSZ montage directory is not canonical")
    record_match = re.fullmatch(
        rf"{re.escape(patient_id)}_(s\d{{3}})_(t\d{{3}})\.edf", filename
    )
    if record_match is None or record_match.group(1) != session_match.group(1):
        raise ValueError("TUSZ EDF filename/session identity is not canonical")
    record_key = (record_match.group(1), record_match.group(2))
    return (
        relative_edf,
        edf_path,
        relative_channel,
        channel_path,
        relative_global,
        global_path,
        split,
        patient_id,
        record_key,
    )


def _source_official_split(value: object) -> str:
    text = _clean(value, field="deepsoz_source.loc").replace("\\", "/")
    parts = tuple(part for part in text.split("/") if part)
    matches = tuple(part for part in parts if part in _OFFICIAL_SPLITS)
    if len(matches) != 1:
        raise ValueError("DeepSOZ source loc must encode exactly one official split")
    return matches[0]


def _source_record_key(value: object) -> tuple[str, str]:
    text = _clean(value, field="deepsoz_source.fn")
    if Path(text).name != text:
        raise ValueError("DeepSOZ source fn must be a basename")
    match = re.fullmatch(r"[^/]+_(s\d{3})_(t\d{3})\.edf", text)
    if match is None:
        raise ValueError("DeepSOZ source fn lacks a canonical session/trial identity")
    return match.group(1), match.group(2)


def _strict_number_sequence(
    value: object, *, field: str, allow_empty: bool = False
) -> tuple[float, ...]:
    try:
        parsed = ast.literal_eval(_clean(value, field=field))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{field} must be a numeric list") from exc
    if not isinstance(parsed, (list, tuple)) or (not parsed and not allow_empty):
        qualifier = "a numeric list" if allow_empty else "a non-empty numeric list"
        raise ValueError(f"{field} must be {qualifier}")
    try:
        values = tuple(float(item) for item in parsed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a numeric list") from exc
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"{field} must contain finite times")
    return values


def _config_payload(config: CausalEDFConfig) -> dict[str, object]:
    return asdict(config)


def _config_sha256(config: CausalEDFConfig) -> str:
    return _canonical_sha256(
        {"preprocess_schema": EDF_PREPROCESS_SCHEMA, "config": _config_payload(config)}
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = json.dumps(
        {"dtype": str(values.dtype), "shape": list(values.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = values.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _row_payload(row: Mapping[str, object], fields_: Sequence[str]) -> dict[str, object]:
    return {field: str(row[field]).strip() for field in fields_}


def _roster_sha256(values: Sequence[str]) -> str:
    ordered = tuple(sorted(values))
    if len(set(ordered)) != len(ordered):
        raise ValueError("A preflight roster contains duplicates")
    return _canonical_sha256(ordered)


def _closed_object(value: object, *, expected: frozenset[str], field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{field} violates its closed schema; missing={missing}, unknown={unknown}"
        )
    return value


def _validate_nested_receipts(event: Mapping[str, object], *, index: int) -> None:
    edf = _closed_object(
        event["edf_receipt"], expected=_EDF_RECEIPT_FIELDS, field=f"events[{index}].edf_receipt"
    )
    signal = _closed_object(
        event["signal_receipt"],
        expected=_SIGNAL_RECEIPT_FIELDS,
        field=f"events[{index}].signal_receipt",
    )
    channel_qc = signal["channel_qc"]
    if not isinstance(channel_qc, (list, tuple)) or len(channel_qc) != 19:
        raise ValueError("Stored signal receipt must contain 19 channel-QC rows")
    for qc_index, qc in enumerate(channel_qc):
        _closed_object(
            qc,
            expected=_CHANNEL_QC_FIELDS,
            field=f"events[{index}].signal_receipt.channel_qc[{qc_index}]",
        )
    if edf["schema_version"] != EDF_PREPROCESS_SCHEMA:
        raise ValueError("Stored EDF receipt uses a legacy preprocessing schema")
    semantic_channels = edf["semantic_channels"]
    raw_channel_names = edf["raw_channel_names"]
    if (
        not isinstance(semantic_channels, (list, tuple))
        or not isinstance(raw_channel_names, (list, tuple))
    ):
        raise TypeError("Stored EDF channel bindings must be arrays")
    binding = bind_labram_record_positions(
        raw_channel_names,
        semantic_channels=semantic_channels,
    )
    if (
        edf["labram_position_binding_policy"]
        != LABRAM_RAW_HEADER_POSITION_BINDING_POLICY
        or edf["labram_position_binding_policy"] != binding.policy
        or tuple(edf["labram_position_names"]) != binding.position_names
        or tuple(edf["labram_position_ids"]) != binding.position_ids
    ):
        raise ValueError("Stored EDF LaBraM binding drifted from its raw headers")
    if (
        edf["iir_initial_state_policy"] != CAUSAL_IIR_INITIAL_STATE_POLICY
        or edf["iir_phase_policy"] != CAUSAL_IIR_PHASE_POLICY
        or edf["iir_group_delay_estimator"]
        != CAUSAL_IIR_GROUP_DELAY_ESTIMATOR
        or edf["iir_scalar_delay_correction_applied"] is not False
    ):
        raise ValueError("Stored EDF causal-IIR phase/state policy is invalid")
    frequencies = edf["iir_group_delay_frequency_hz"]
    delays = edf["iir_group_delay_seconds"]
    if (
        not isinstance(frequencies, (list, tuple))
        or not isinstance(delays, (list, tuple))
        or len(frequencies) != 17
        or len(delays) != 17
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (*frequencies, *delays)
        )
        or any(
            float(left) >= float(right)
            for left, right in zip(frequencies, frequencies[1:])
        )
    ):
        raise ValueError("Stored EDF IIR group-delay grid is invalid")
    if (
        edf["iir_state_reset_sample"] != edf["read_start_sample"]
        or isinstance(edf["iir_warmup_samples"], bool)
        or not isinstance(edf["iir_warmup_samples"], int)
        or edf["iir_warmup_samples"] < 1
    ):
        raise ValueError("Stored EDF IIR state receipt is invalid")
    if event["edf_receipt_sha256"] != _canonical_sha256(edf):
        raise ValueError("Stored EDF receipt SHA mismatch")
    if event["signal_receipt_sha256"] != _canonical_sha256(signal):
        raise ValueError("Stored signal receipt SHA mismatch")


def _validate_receipt(receipt_value: object) -> dict[str, object]:
    receipt = _closed_object(
        receipt_value, expected=_RECEIPT_FIELDS, field="signal-preflight receipt"
    )
    if receipt["schema_version"] != DEEPSOZ_SIGNAL_PREFLIGHT_SCHEMA:
        raise ValueError("Unsupported DeepSOZ signal-preflight receipt schema")
    if receipt["policy"] != DEEPSOZ_SIGNAL_PREFLIGHT_POLICY:
        raise ValueError("DeepSOZ signal-preflight policy cannot be changed")
    for field in (
        "event_inputs_sha256",
        "record_crosswalk_sha256",
        "split_manifest_sha256",
        "deepsoz_source_sha256",
        "conservative_mapping_sha256",
        "verified_target_v2_receipt_sha256",
        "verified_target_v2_artifact_sha256",
        "verified_target_v2_policy_sha256",
        "preprocess_config_sha256",
        "candidate_event_roster_sha256",
        "eligible_event_roster_sha256",
        "excluded_event_roster_sha256",
        "eligible_patient_roster_sha256",
        "negative_start_source_record_roster_sha256",
    ):
        _require_sha256(receipt[field], field=field)
    if receipt["preprocess_schema"] != EDF_PREPROCESS_SCHEMA:
        raise ValueError("Signal-preflight preprocessing schema drifted")
    config = _closed_object(
        receipt["preprocess_config"],
        expected=frozenset(field.name for field in fields(CausalEDFConfig)),
        field="preprocess_config",
    )
    if receipt["preprocess_config_sha256"] != _canonical_sha256(
        {"preprocess_schema": EDF_PREPROCESS_SCHEMA, "config": config}
    ):
        raise ValueError("Signal-preflight preprocess config SHA mismatch")
    if _canonical_json_bytes(config) != _canonical_json_bytes(
        _config_payload(CausalEDFConfig())
    ):
        raise ValueError("Signal-preflight must use the complete frozen causal config")
    events = receipt["events"]
    exclusions = receipt["exclusions"]
    if not isinstance(events, list) or not isinstance(exclusions, list):
        raise ValueError("Signal-preflight events/exclusions must be JSON arrays")
    for index, event_value in enumerate(events):
        event = _closed_object(event_value, expected=_EVENT_FIELDS, field=f"events[{index}]")
        _validate_nested_receipts(event, index=index)
    for index, exclusion in enumerate(exclusions):
        _closed_object(exclusion, expected=_EXCLUSION_FIELDS, field=f"exclusions[{index}]")
    event_ids = tuple(str(event["event_id"]) for event in events)
    excluded_ids = tuple(str(event["event_id"]) for event in exclusions)
    candidate_ids = tuple(sorted((*event_ids, *excluded_ids)))
    if event_ids != tuple(sorted(event_ids)) or excluded_ids != tuple(sorted(excluded_ids)):
        raise ValueError("Signal-preflight event arrays must be canonically ordered")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Signal-preflight candidate event IDs are not unique")
    patients = tuple(sorted({str(event["patient_id"]) for event in events}))
    if receipt["candidate_event_count"] != len(candidate_ids):
        raise ValueError("Signal-preflight candidate event count mismatch")
    source_record_count = receipt["source_record_count"]
    if (
        isinstance(source_record_count, bool)
        or not isinstance(source_record_count, int)
        or source_record_count < 1
    ):
        raise ValueError("Signal-preflight source_record_count must be positive")
    timeline_audit = receipt["source_timeline_audit_counts"]
    if not isinstance(timeline_audit, list) or len(timeline_audit) != len(
        _SOURCE_TIMELINE_AUDIT_KEYS
    ):
        raise ValueError("Signal-preflight source timeline audit has the wrong schema")
    audit_total = 0
    for row, expected_key in zip(timeline_audit, _SOURCE_TIMELINE_AUDIT_KEYS):
        if (
            not isinstance(row, list)
            or len(row) != 3
            or tuple(row[:2]) != expected_key
            or isinstance(row[2], bool)
            or not isinstance(row[2], int)
            or row[2] < 0
        ):
            raise ValueError("Signal-preflight source timeline audit row is invalid")
        audit_total += row[2]
    if audit_total != source_record_count:
        raise ValueError("Signal-preflight source timeline audit count mismatch")
    unique_empty = timeline_audit[_SOURCE_TIMELINE_AUDIT_KEYS.index(("unique", "empty"))][2]
    if unique_empty != 0:
        raise ValueError("A unique mapping cannot have an empty source timeline")
    negative_start_count = receipt["negative_start_source_record_count"]
    if (
        isinstance(negative_start_count, bool)
        or not isinstance(negative_start_count, int)
        or negative_start_count < 0
        or negative_start_count > source_record_count
    ):
        raise ValueError(
            "Signal-preflight negative-start source record count is invalid"
        )
    if receipt["eligible_event_count"] != len(events):
        raise ValueError("Signal-preflight eligible event count mismatch")
    if receipt["excluded_event_count"] != len(exclusions):
        raise ValueError("Signal-preflight excluded event count mismatch")
    if receipt["eligible_patient_count"] != len(patients):
        raise ValueError("Signal-preflight eligible patient count mismatch")
    checks = {
        "candidate_event_roster_sha256": _roster_sha256(candidate_ids),
        "eligible_event_roster_sha256": _roster_sha256(event_ids),
        "excluded_event_roster_sha256": _roster_sha256(excluded_ids),
        "eligible_patient_roster_sha256": _roster_sha256(patients),
    }
    for field, expected in checks.items():
        if receipt[field] != expected:
            raise ValueError(f"Signal-preflight {field} mismatch")
    split_rosters = receipt["eligible_split_patient_ids"]
    if not isinstance(split_rosters, list) or [row[0] for row in split_rosters] != [
        "source_train",
        "source_dev",
        "source_eval",
    ]:
        raise ValueError("Signal-preflight split rosters use the wrong schema/order")
    flattened: list[str] = []
    for row in split_rosters:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], list):
            raise ValueError("Signal-preflight split roster row is invalid")
        roster = [str(value) for value in row[1]]
        if roster != sorted(set(roster)):
            raise ValueError("Signal-preflight split patient roster is not sorted/unique")
        flattened.extend(roster)
    if sorted(flattened) != list(patients):
        raise ValueError("Signal-preflight split rosters do not partition eligible patients")
    return receipt


@dataclass(frozen=True)
class VerifiedDeepSOZSignalPreflightBundle:
    """Path-free verified receipt returned after complete signal replay."""

    receipt: Mapping[str, object]
    artifact_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        validated = _validate_receipt(dict(self.receipt))
        object.__setattr__(self, "receipt", validated)
        _require_sha256(self.artifact_sha256, field="artifact_sha256")
        _require_sha256(self.receipt_sha256, field="receipt_sha256")
        if self.receipt_sha256 != _canonical_sha256(validated):
            raise ValueError("Signal-preflight receipt SHA disagrees with receipt")

    @property
    def eligible_event_ids(self) -> tuple[str, ...]:
        return tuple(str(row["event_id"]) for row in self.receipt["events"])

    @property
    def eligible_patient_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({str(row["patient_id"]) for row in self.receipt["events"]})
        )


def _build_receipt(
    event_inputs_csv: str | Path,
    record_crosswalk_csv: str | Path,
    split_manifest_csv: str | Path,
    deepsoz_source_csv: str | Path,
    conservative_mapping_csv: str | Path,
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    tusz_root: str | Path,
    *,
    expected_event_inputs_sha256: str,
    expected_record_crosswalk_sha256: str,
    expected_split_manifest_sha256: str,
    expected_deepsoz_source_sha256: str,
    expected_conservative_mapping_sha256: str,
    config: CausalEDFConfig,
    reader_factory: Callable[[str], object] | None,
) -> dict[str, object]:
    if not isinstance(verified_target_v2, VerifiedDeepSOZTargetV2Artifact):
        raise TypeError("verified_target_v2 must be a strictly verified target-v2 artifact")
    if not isinstance(config, CausalEDFConfig):
        raise TypeError("config must be CausalEDFConfig")
    if _canonical_json_bytes(_config_payload(config)) != _canonical_json_bytes(
        _config_payload(CausalEDFConfig())
    ):
        raise ValueError("Formal DeepSOZ preflight requires the complete frozen causal config")
    root = _reject_symlink_components(Path(tusz_root), field="TUSZ root")
    if not root.is_dir():
        raise FileNotFoundError("TUSZ root directory does not exist")

    events_frame, event_inputs_sha = _strict_csv(
        event_inputs_csv,
        expected_sha256=expected_event_inputs_sha256,
        allowed_columns=_EVENT_INPUT_COLUMNS,
        label="event_inputs",
    )
    crosswalk_frame, crosswalk_sha = _strict_csv(
        record_crosswalk_csv,
        expected_sha256=expected_record_crosswalk_sha256,
        allowed_columns=_CROSSWALK_INPUT_COLUMNS,
        label="record_crosswalk",
    )
    split_frame, split_sha = _strict_csv(
        split_manifest_csv,
        expected_sha256=expected_split_manifest_sha256,
        allowed_columns=_SPLIT_INPUT_COLUMNS,
        label="split_manifest",
    )
    source_frame, source_sha = _strict_deepsoz_source_csv(
        deepsoz_source_csv,
        expected_sha256=expected_deepsoz_source_sha256,
    )
    mapping_frame, mapping_sha = _strict_csv(
        conservative_mapping_csv,
        expected_sha256=expected_conservative_mapping_sha256,
        allowed_columns=_CONSERVATIVE_MAPPING_INPUT_COLUMNS,
        label="conservative_mapping",
    )
    target_receipt = verified_target_v2.receipt
    if target_receipt.split_input_sha256 != split_sha:
        raise ValueError("Verified target-v2 and signal preflight use different split bytes")
    if target_receipt.source_input_sha256 != source_sha:
        raise ValueError("Verified target-v2 and signal preflight use different source bytes")

    for label, frame in (
        ("event_inputs", events_frame),
        ("record_crosswalk", crosswalk_frame),
        ("split_manifest", split_frame),
    ):
        if set(frame["source"].map(str).str.strip()) != {DEEPSOZ_SIGNAL_SOURCE}:
            raise ValueError(f"{label} contains an unauthorized source/schema")

    split_frame = split_frame.copy()
    split_frame["deepsoz_patient_id"] = split_frame["deepsoz_patient_id"].map(
        normalize_patient_id
    )
    if split_frame["deepsoz_patient_id"].duplicated().any():
        raise ValueError("split_manifest contains duplicate target patients")
    split_by_patient = split_frame.set_index("deepsoz_patient_id", drop=False)
    target_ids = tuple(reference.patient_id for reference in verified_target_v2.registry)
    if set(split_by_patient.index) != set(target_ids):
        raise ValueError("split_manifest roster differs from verified target-v2")
    for patient_id in target_ids:
        reference = verified_target_v2.registry.get(patient_id)
        row = split_by_patient.loc[patient_id]
        if str(row["model_split"]).strip() != reference.model_split:
            raise ValueError("split_manifest model split differs from verified target-v2")
        if str(row["official_split"]).strip() != reference.official_split:
            raise ValueError("split_manifest official split differs from verified target-v2")

    eligible_target_ids = set(target_receipt.eligible_patient_ids)
    split_local_by_patient: dict[str, str] = {}
    split_patient_by_local: dict[str, str] = {}
    for patient_id in sorted(eligible_target_ids):
        local_patient = _clean(
            split_by_patient.loc[patient_id, "local_patient_id"],
            field="split_manifest.local_patient_id",
        )
        previous = split_patient_by_local.setdefault(local_patient, patient_id)
        if previous != patient_id:
            raise ValueError(
                "split_manifest maps one local patient to multiple DeepSOZ patients"
            )
        split_local_by_patient[patient_id] = local_patient

    source_frame = source_frame.reset_index(drop=True)
    mapping_frame = mapping_frame.copy()
    crosswalk_frame = crosswalk_frame.copy()
    mapping_frame["deepsoz_row_int"] = [
        _strict_int(value, field=f"conservative_mapping.deepsoz_row[{index}]")
        for index, value in enumerate(mapping_frame["deepsoz_row"])
    ]
    crosswalk_frame["deepsoz_row_int"] = [
        _strict_int(value, field=f"record_crosswalk.deepsoz_row[{index}]")
        for index, value in enumerate(crosswalk_frame["deepsoz_row"])
    ]
    expected_source_rows = tuple(range(len(source_frame)))
    if (
        tuple(mapping_frame["deepsoz_row_int"]) != expected_source_rows
        or tuple(crosswalk_frame["deepsoz_row_int"]) != expected_source_rows
    ):
        raise ValueError(
            "Source, conservative mapping, and record crosswalk must have the same "
            "complete ordered DeepSOZ row roster"
        )

    local_patient_owner: dict[str, str] = {}
    deepsoz_patient_owner: dict[str, str] = {}
    edf_owner: dict[str, str] = {}
    crosswalk_by_row: dict[int, Mapping[str, object]] = {}
    source_by_row: dict[int, Mapping[str, object]] = {}
    source_timeline_audit_counts = {
        key: 0 for key in _SOURCE_TIMELINE_AUDIT_KEYS
    }
    negative_start_source_record_hashes: list[str] = []
    for deepsoz_row in expected_source_rows:
        source_row = source_frame.iloc[deepsoz_row].to_dict()
        mapping_row = mapping_frame.iloc[deepsoz_row].to_dict()
        row = crosswalk_frame.iloc[deepsoz_row].to_dict()
        patient_id = normalize_patient_id(source_row["pt_id"])
        if patient_id not in split_by_patient.index:
            raise ValueError("DeepSOZ source row refers to an unknown target patient")
        source_record = _clean(source_row["fn"], field="deepsoz_source.fn")
        source_split = _source_official_split(source_row["loc"])
        source_starts = _strict_number_sequence(
            source_row["sz_starts"],
            field="deepsoz_source.sz_starts",
            allow_empty=True,
        )
        source_stops = _strict_number_sequence(
            source_row["sz_ends"],
            field="deepsoz_source.sz_ends",
            allow_empty=True,
        )
        source_event_count = _strict_int(
            source_row["nsz"], field="deepsoz_source.nsz"
        )
        complete_timeline = (
            source_event_count > 0
            and len(source_starts) == source_event_count
            and len(source_stops) == source_event_count
            and all(stop > start for start, stop in zip(source_starts, source_stops))
        )
        empty_timeline = (
            source_event_count == 0 and not source_starts and not source_stops
        )
        if not complete_timeline and not empty_timeline:
            raise ValueError(
                "DeepSOZ source nsz/starts/ends timeline is internally inconsistent"
            )
        source_timeline_state = "complete" if complete_timeline else "empty"

        identity_checks = {
            "mapping patient": normalize_patient_id(mapping_row["deepsoz_patient"])
            == patient_id,
            "crosswalk patient": normalize_patient_id(row["deepsoz_patient_id"])
            == patient_id,
            "mapping record": str(mapping_row["deepsoz_record"]).strip()
            == source_record,
            "crosswalk record": str(row["deepsoz_record"]).strip()
            == source_record,
            "crosswalk source split": str(row["source_official_split"]).strip()
            == source_split,
            "crosswalk source event count": _strict_int(
                row["source_event_count"], field="record_crosswalk.source_event_count"
            )
            == source_event_count,
        }
        failed_identity = sorted(
            name for name, passed in identity_checks.items() if not passed
        )
        if failed_identity:
            raise ValueError(
                "DeepSOZ source/mapping/crosswalk identity drift: "
                f"{failed_identity}"
            )

        mapping_status = _clean(
            mapping_row["mapping_status"], field="conservative_mapping.mapping_status"
        ).lower()
        crosswalk_status = _clean(
            row["mapping_status"], field="record_crosswalk.mapping_status"
        ).lower()
        if mapping_status not in _MAPPING_STATUSES or crosswalk_status != mapping_status:
            raise ValueError("Conservative mapping status drifted in record_crosswalk")
        source_timeline_audit_counts[(mapping_status, source_timeline_state)] += 1
        if mapping_status == "unique" and source_timeline_state != "complete":
            raise ValueError(
                "A unique conservative mapping requires a complete non-empty source timeline"
            )
        candidate_count = _strict_int(
            mapping_row["candidate_count"], field="conservative_mapping.candidate_count"
        )
        if _strict_int(
            row["candidate_count"], field="record_crosswalk.candidate_count"
        ) != candidate_count:
            raise ValueError("Conservative candidate count drifted in record_crosswalk")
        candidate_paths = tuple(
            value.strip()
            for value in str(mapping_row["candidate_local_csv_bi"]).split(";")
            if value.strip()
        )
        candidate_errors_text = tuple(
            value.strip()
            for value in str(mapping_row["candidate_max_errors_s"]).split(";")
            if value.strip()
        )
        if len(candidate_paths) != candidate_count or len(candidate_errors_text) != candidate_count:
            raise ValueError("Conservative mapping candidate roster/count is inconsistent")
        candidate_relatives: list[str] = []
        candidate_errors: list[float] = []
        for candidate_index, (candidate_path, error_text) in enumerate(
            zip(candidate_paths, candidate_errors_text)
        ):
            relative_global, _ = _mapping_source_path(
                root,
                candidate_path,
                field=f"conservative_mapping.candidate_local_csv_bi[{candidate_index}]",
            )
            if not relative_global.endswith(".csv_bi"):
                raise ValueError("A conservative mapping candidate is not a csv_bi sidecar")
            candidate_relatives.append(relative_global)
            error = _strict_float(
                error_text,
                field=f"conservative_mapping.candidate_max_errors_s[{candidate_index}]",
            )
            if error < 0 or error > _MAPPING_TIME_TOLERANCE_SEC + _TIME_TOLERANCE_SEC:
                raise ValueError("A conservative mapping candidate exceeds frozen tolerance")
            candidate_errors.append(error)
        expected_status = (
            "unmapped" if candidate_count == 0 else "unique" if candidate_count == 1 else "ambiguous"
        )
        if mapping_status != expected_status:
            raise ValueError("Conservative mapping status is inconsistent with candidate count")

        source_record_sha256 = _canonical_sha256(
            {
                "deepsoz_row": deepsoz_row,
                "pt_id": patient_id,
                "fn": source_record,
                "loc": str(source_row["loc"]).strip(),
                "nsz": source_event_count,
                "sz_starts": source_starts,
                "sz_ends": source_stops,
            }
        )
        if any(start < 0 for start in source_starts):
            negative_start_source_record_hashes.append(source_record_sha256)
        source_by_row[deepsoz_row] = {
            "patient_id": patient_id,
            "record": source_record,
            "official_split": source_split,
            "event_count": source_event_count,
            "starts": source_starts,
            "stops": source_stops,
            "record_key": _source_record_key(source_record),
            "timeline_state": source_timeline_state,
            "record_sha256": source_record_sha256,
        }
        if mapping_status != "unique":
            if any(
                str(mapping_row[field]).strip()
                for field in ("local_patient", "local_csv_bi", "local_edf", "max_time_error_s")
            ) or any(
                str(row[field]).strip()
                for field in (
                    "local_patient_id", "local_official_split", "local_edf_path",
                    "local_csv_path", "local_csv_bi_path",
                )
            ):
                raise ValueError("Non-unique mappings cannot declare a selected local record")
            continue

        local_patient = _clean(row["local_patient_id"], field="local_patient_id")
        mapping_local_patient = _clean(
            mapping_row["local_patient"], field="conservative_mapping.local_patient"
        )
        if mapping_local_patient != local_patient:
            raise ValueError("Conservative mapping local patient drifted in crosswalk")
        owner = local_patient_owner.setdefault(local_patient, patient_id)
        if owner != patient_id:
            raise ValueError("One local patient maps to multiple DeepSOZ patients")
        reverse_owner = deepsoz_patient_owner.setdefault(patient_id, local_patient)
        if reverse_owner != local_patient:
            raise ValueError("One DeepSOZ patient maps to multiple local patients")
        if (
            patient_id in eligible_target_ids
            and split_local_by_patient[patient_id] != local_patient
        ):
            raise ValueError(
                "record_crosswalk local patient differs from the verified split manifest"
            )
        official_split = _clean(row["local_official_split"], field="local_official_split")
        reference = verified_target_v2.registry.get(patient_id)
        if official_split != reference.official_split or official_split != source_split:
            raise ValueError("record_crosswalk split differs from verified target-v2")
        _strict_one(row["split_agreement"], field="record_crosswalk.split_agreement")
        (
            relative_edf,
            _,
            relative_channel,
            _,
            relative_global,
            _,
            derived_split,
            derived_patient,
            local_record_key,
        ) = _canonical_tusz_record_identity(
            root,
            row["local_edf_path"],
            row["local_csv_path"],
            row["local_csv_bi_path"],
        )
        mapping_relative_edf, _ = _mapping_source_path(
            root, mapping_row["local_edf"], field="conservative_mapping.local_edf"
        )
        mapping_relative_global, _ = _mapping_source_path(
            root,
            mapping_row["local_csv_bi"],
            field="conservative_mapping.local_csv_bi",
        )
        if (
            derived_patient != local_patient
            or derived_split != official_split
            or local_record_key != source_by_row[deepsoz_row]["record_key"]
            or mapping_relative_edf != relative_edf
            or mapping_relative_global != relative_global
            or tuple(candidate_relatives) != (relative_global,)
        ):
            raise ValueError(
                "Canonical TUSZ path identity differs from source/mapping/crosswalk"
            )
        declared_mapping_error = _strict_float(
            mapping_row["max_time_error_s"],
            field="conservative_mapping.max_time_error_s",
        )
        crosswalk_mapping_error = _strict_float(
            row["max_time_error_sec"], field="record_crosswalk.max_time_error_sec"
        )
        if (
            abs(declared_mapping_error - candidate_errors[0]) > _TIME_TOLERANCE_SEC
            or abs(crosswalk_mapping_error - declared_mapping_error) > _TIME_TOLERANCE_SEC
        ):
            raise ValueError("Conservative mapping error drifted in crosswalk")
        edf_previous = edf_owner.setdefault(relative_edf, patient_id)
        if edf_previous != patient_id:
            raise ValueError("One local EDF maps to multiple DeepSOZ patients")
        crosswalk_by_row[deepsoz_row] = {
            **row,
            "local_edf_path": relative_edf,
            "local_csv_path": relative_channel,
            "local_csv_bi_path": relative_global,
        }

    if not crosswalk_by_row:
        raise ValueError("record_crosswalk contains no unique records")

    mapped_eligible_ids = eligible_target_ids & set(deepsoz_patient_owner)
    if mapped_eligible_ids != eligible_target_ids:
        raise ValueError(
            "Every target-eligible DeepSOZ patient requires a unique record crosswalk"
        )

    events_frame = events_frame.copy()
    events_frame["deepsoz_patient_id"] = events_frame["deepsoz_patient_id"].map(
        normalize_patient_id
    )
    events_frame["deepsoz_row_int"] = [
        _strict_int(value, field=f"event_inputs.deepsoz_row[{index}]")
        for index, value in enumerate(events_frame["deepsoz_row"])
    ]
    if events_frame["event_id"].map(str).str.strip().duplicated().any():
        raise ValueError("event_inputs contains duplicate event IDs")
    if events_frame.duplicated(["local_edf_path", "event_index"]).any():
        raise ValueError("event_inputs contains duplicate EDF/event-index pairs")

    rows_by_crosswalk: dict[int, list[Mapping[str, object]]] = {}
    for _, raw_row in events_frame.iterrows():
        row = raw_row.to_dict()
        deepsoz_row = int(row["deepsoz_row_int"])
        crosswalk = crosswalk_by_row.get(deepsoz_row)
        if crosswalk is None:
            raise ValueError("event_inputs row has no unique record_crosswalk foreign key")
        comparisons = {
            "deepsoz_patient_id": normalize_patient_id(row["deepsoz_patient_id"])
            == normalize_patient_id(crosswalk["deepsoz_patient_id"]),
            "patient_target_key": normalize_patient_id(row["patient_target_key"])
            == normalize_patient_id(crosswalk["deepsoz_patient_id"]),
            "deepsoz_record": str(row["deepsoz_record"]).strip()
            == str(crosswalk["deepsoz_record"]).strip(),
            "local_patient_id": str(row["local_patient_id"]).strip()
            == str(crosswalk["local_patient_id"]).strip(),
            "official_split": str(row["official_split"]).strip()
            == str(crosswalk["local_official_split"]).strip(),
            "local_edf_path": str(row["local_edf_path"]).strip()
            == str(crosswalk["local_edf_path"]).strip(),
            "local_csv_path": str(row["local_csv_path"]).strip()
            == str(crosswalk["local_csv_path"]).strip(),
            "local_csv_bi_path": str(row["local_csv_bi_path"]).strip()
            == str(crosswalk["local_csv_bi_path"]).strip(),
        }
        failed = sorted(name for name, passed in comparisons.items() if not passed)
        if failed:
            raise ValueError(
                "event_inputs patient/EDF foreign-key join differs from record_crosswalk: "
                f"{failed}"
            )
        rows_by_crosswalk.setdefault(deepsoz_row, []).append(row)

    preprocess_sha = _config_sha256(config)
    accepted: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    candidate_ids: list[str] = []
    join_fields = (
        "source",
        "deepsoz_row",
        "deepsoz_patient_id",
        "patient_target_key",
        "deepsoz_record",
        "local_patient_id",
        "official_split",
        "event_id",
        "event_index",
        "local_edf_path",
        "local_csv_path",
        "local_csv_bi_path",
        "t0_sec",
        "t0_provenance",
        "seizure_end_sec",
        "window_start_sec",
        "window_stop_sec",
    )
    crosswalk_join_fields = (
        "source",
        "deepsoz_row",
        "deepsoz_patient_id",
        "deepsoz_record",
        "source_official_split",
        "source_event_count",
        "mapping_status",
        "candidate_count",
        "max_time_error_sec",
        "local_patient_id",
        "local_official_split",
        "split_agreement",
        "local_edf_path",
        "local_csv_path",
        "local_csv_bi_path",
    )

    for deepsoz_row, crosswalk in sorted(crosswalk_by_row.items()):
        patient_id = normalize_patient_id(crosswalk["deepsoz_patient_id"])
        if patient_id not in eligible_target_ids:
            continue
        reference = verified_target_v2.registry.get(patient_id)
        relative_edf, edf_path = _relative_source_path(
            root, crosswalk["local_edf_path"], field="local_edf_path"
        )
        relative_csv, csv_path = _relative_source_path(
            root, crosswalk["local_csv_path"], field="local_csv_path"
        )
        relative_csv_bi, csv_bi_path = _relative_source_path(
            root, crosswalk["local_csv_bi_path"], field="local_csv_bi_path"
        )
        pair = inspect_tusz_annotation_pair(
            csv_path, csv_bi_path, source_path=edf_path
        )
        source_identity = source_by_row[deepsoz_row]
        if not pair.global_seizure_events:
            raise ValueError(
                "Target-eligible unique crosswalk has zero official global seizure events"
            )
        if len(pair.global_seizure_events) != source_identity["event_count"]:
            raise ValueError(
                "Selected local record event count differs from original DeepSOZ source"
            )
        mapping_errors = [
            abs(float(source_start) - float(event.start_sec))
            for source_start, event in zip(
                source_identity["starts"], pair.global_seizure_events
            )
        ]
        mapping_errors.extend(
            abs(float(source_stop) - float(event.stop_sec))
            for source_stop, event in zip(
                source_identity["stops"], pair.global_seizure_events
            )
        )
        replayed_mapping_error = max(mapping_errors)
        declared_mapping_error = _strict_float(
            crosswalk["max_time_error_sec"],
            field="record_crosswalk.max_time_error_sec",
        )
        if (
            replayed_mapping_error
            > _MAPPING_TIME_TOLERANCE_SEC + _TIME_TOLERANCE_SEC
            or abs(replayed_mapping_error - declared_mapping_error)
            > _TIME_TOLERANCE_SEC
        ):
            raise ValueError(
                "Conservative unique complete-timeline mapping did not replay"
            )
        event_rows = rows_by_crosswalk.get(deepsoz_row, [])
        by_index: dict[int, Mapping[str, object]] = {}
        for row in event_rows:
            event_index = _strict_int(row["event_index"], field="event_index")
            if event_index in by_index:
                raise ValueError("event_inputs repeats a global event index")
            by_index[event_index] = row
        expected_indices = set(range(len(pair.global_seizure_events)))
        if set(by_index) != expected_indices:
            raise ValueError(
                "event_inputs does not exactly enumerate the official global TUSZ timeline"
            )
        crosswalk_record_sha = _canonical_sha256(
            _row_payload(crosswalk, crosswalk_join_fields)
        )
        for global_event in pair.global_seizure_events:
            row = by_index[global_event.event_index]
            event_id = _clean(row["event_id"], field="event_id")
            expected_event_id = f"{edf_path.stem}__ev{global_event.event_index:04d}"
            if event_id != expected_event_id:
                raise ValueError("event_id does not encode the official EDF/global-event identity")
            if _clean(row["t0_provenance"], field="t0_provenance") != DEEPSOZ_EVENT_ANCHOR:
                raise ValueError("event_inputs uses an unauthorized onset anchor")
            t0 = _strict_float(row["t0_sec"], field="t0_sec")
            stop = _strict_float(row["seizure_end_sec"], field="seizure_end_sec")
            window_start = _strict_float(row["window_start_sec"], field="window_start_sec")
            window_stop = _strict_float(row["window_stop_sec"], field="window_stop_sec")
            if t0 < 0:
                raise ValueError("TUSZ event t0 must be non-negative")
            comparisons = (
                (t0, global_event.start_sec, "global t0"),
                (stop, global_event.stop_sec, "global stop"),
                (window_start, t0 - config.pre_onset_sec, "window start"),
                (window_stop, t0 + config.post_onset_sec, "window stop"),
            )
            for actual, expected, label in comparisons:
                if abs(actual - expected) > _TIME_TOLERANCE_SEC:
                    raise ValueError(f"event_inputs {label} drifted from the frozen timeline")
            if abs((window_stop - window_start) - 60.0) > _TIME_TOLERANCE_SEC:
                raise ValueError("Formal DeepSOZ event window must be exactly 60 seconds")
            event_record_sha = _canonical_sha256(_row_payload(row, join_fields))
            candidate_ids.append(event_id)
            common = {
                "event_id": event_id,
                "event_record_sha256": event_record_sha,
                "crosswalk_record_sha256": crosswalk_record_sha,
                "deepsoz_source_record_sha256": source_identity["record_sha256"],
                "patient_id": patient_id,
                "local_patient_id": _clean(
                    crosswalk["local_patient_id"], field="local_patient_id"
                ),
                "official_split": reference.official_split,
                "model_split": reference.model_split,
                "relative_edf_path": relative_edf,
                "deepsoz_record": source_identity["record"],
                "global_event_index": global_event.event_index,
                "global_t0_sec": float(global_event.start_sec),
                "global_stop_sec": float(global_event.stop_sec),
                "edf_sha256": pair.source_sha256,
                "annotation_pair_sha256": pair.annotation_pair_sha256,
            }
            try:
                loaded = load_standard19_edf_event(
                    edf_path,
                    global_event.start_sec,
                    config=config,
                    reader_factory=reader_factory,
                )
            except EDFEventEligibilityError as exc:
                excluded.append({**common, "eligibility_code": exc.code})
                continue
            if loaded.edf_receipt.edf_sha256 != pair.source_sha256:
                raise RuntimeError("EDF loader and annotation-pair EDF hashes disagree")
            if abs(loaded.edf_receipt.requested_onset_sec - global_event.start_sec) > _TIME_TOLERANCE_SEC:
                raise RuntimeError("EDF preprocessing receipt used the wrong global t0")
            if (
                tuple(loaded.window.data.shape) != (19, 12_000)
                or loaded.window.onset_index != 2_400
                or abs(loaded.window.sfreq_hz - 200.0) > _TIME_TOLERANCE_SEC
            ):
                raise RuntimeError(
                    "EDF preprocessing did not produce [19,12000] with onset index 2400"
                )
            edf_receipt = asdict(loaded.edf_receipt)
            signal_receipt = asdict(loaded.signal_receipt)
            accepted.append(
                {
                    **common,
                    "deepsoz_row": deepsoz_row,
                    "relative_channel_annotation_path": relative_csv,
                    "relative_global_annotation_path": relative_csv_bi,
                    "global_seizure_type": global_event.seizure_type,
                    "window_start_sec": float(window_start),
                    "window_stop_sec": float(window_stop),
                    "channel_annotation_sha256": pair.channel_annotation_sha256,
                    "global_annotation_sha256": pair.global_annotation_sha256,
                    "preprocess_config_sha256": preprocess_sha,
                    "edf_receipt": edf_receipt,
                    "edf_receipt_sha256": _canonical_sha256(edf_receipt),
                    "signal_receipt": signal_receipt,
                    "signal_receipt_sha256": _canonical_sha256(signal_receipt),
                    "processed_window_sha256": _tensor_sha256(loaded.window.data),
                    "processed_window_shape": list(loaded.window.data.shape),
                    "processed_window_dtype": str(loaded.window.data.dtype),
                }
            )

    accepted.sort(key=lambda row: str(row["event_id"]))
    excluded.sort(key=lambda row: str(row["event_id"]))
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Formal signal-preflight candidate event IDs are not unique")
    eligible_patients = tuple(sorted({str(row["patient_id"]) for row in accepted}))
    split_rosters = [
        [
            model_split,
            sorted(
                {
                    str(row["patient_id"])
                    for row in accepted
                    if row["model_split"] == model_split
                }
            ),
        ]
        for model_split in ("source_train", "source_dev", "source_eval")
    ]
    receipt: dict[str, object] = {
        "schema_version": DEEPSOZ_SIGNAL_PREFLIGHT_SCHEMA,
        "policy": DEEPSOZ_SIGNAL_PREFLIGHT_POLICY,
        "event_inputs_sha256": event_inputs_sha,
        "record_crosswalk_sha256": crosswalk_sha,
        "split_manifest_sha256": split_sha,
        "deepsoz_source_sha256": source_sha,
        "conservative_mapping_sha256": mapping_sha,
        "verified_target_v2_receipt_sha256": target_receipt.receipt_sha256,
        "verified_target_v2_artifact_sha256": target_receipt.target_artifact_sha256,
        "verified_target_v2_policy_sha256": target_receipt.policy_sha256,
        "preprocess_schema": EDF_PREPROCESS_SCHEMA,
        "preprocess_config": _config_payload(config),
        "preprocess_config_sha256": preprocess_sha,
        "candidate_event_roster_sha256": _roster_sha256(candidate_ids),
        "eligible_event_roster_sha256": _roster_sha256(
            [str(row["event_id"]) for row in accepted]
        ),
        "excluded_event_roster_sha256": _roster_sha256(
            [str(row["event_id"]) for row in excluded]
        ),
        "eligible_patient_roster_sha256": _roster_sha256(eligible_patients),
        "eligible_split_patient_ids": split_rosters,
        "candidate_event_count": len(candidate_ids),
        "source_record_count": len(source_frame),
        "source_timeline_audit_counts": [
            [status, state, source_timeline_audit_counts[(status, state)]]
            for status, state in _SOURCE_TIMELINE_AUDIT_KEYS
        ],
        "negative_start_source_record_count": len(
            negative_start_source_record_hashes
        ),
        "negative_start_source_record_roster_sha256": _roster_sha256(
            negative_start_source_record_hashes
        ),
        "eligible_event_count": len(accepted),
        "excluded_event_count": len(excluded),
        "eligible_patient_count": len(eligible_patients),
        "events": accepted,
        "exclusions": excluded,
    }
    _validate_receipt(receipt)
    return receipt


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_deepsoz_signal_preflight_bundle(
    event_inputs_csv: str | Path,
    record_crosswalk_csv: str | Path,
    split_manifest_csv: str | Path,
    deepsoz_source_csv: str | Path,
    conservative_mapping_csv: str | Path,
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    tusz_root: str | Path,
    output_directory: str | Path,
    *,
    expected_event_inputs_sha256: str,
    expected_record_crosswalk_sha256: str,
    expected_split_manifest_sha256: str,
    expected_deepsoz_source_sha256: str,
    expected_conservative_mapping_sha256: str,
    config: CausalEDFConfig = CausalEDFConfig(),
    reader_factory: Callable[[str], object] | None = None,
) -> VerifiedDeepSOZSignalPreflightBundle:
    """Replay every target-eligible event and atomically publish one bundle."""

    receipt = _build_receipt(
        event_inputs_csv,
        record_crosswalk_csv,
        split_manifest_csv,
        deepsoz_source_csv,
        conservative_mapping_csv,
        verified_target_v2,
        tusz_root,
        expected_event_inputs_sha256=expected_event_inputs_sha256,
        expected_record_crosswalk_sha256=expected_record_crosswalk_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_deepsoz_source_sha256=expected_deepsoz_source_sha256,
        expected_conservative_mapping_sha256=expected_conservative_mapping_sha256,
        config=config,
        reader_factory=reader_factory,
    )
    receipt_sha = _canonical_sha256(receipt)
    payload = {
        "schema_version": DEEPSOZ_SIGNAL_PREFLIGHT_ARTIFACT_SCHEMA,
        "serialization": "canonical_json_utf8_newline_no_pickle",
        "receipt_sha256": receipt_sha,
        "receipt": receipt,
    }
    encoded = _canonical_json_bytes(payload)
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("DeepSOZ signal-preflight artifact exceeds the size limit")
    output = _reject_symlink_components(Path(output_directory), field="output bundle")
    if output.name in {"", ".", ".."}:
        raise ValueError("Output bundle requires a concrete directory name")
    if os.path.lexists(output):
        raise FileExistsError("Signal-preflight bundle destination already exists")
    parent = _reject_symlink_components(output.parent, field="output parent")
    if not parent.is_dir():
        raise FileNotFoundError("Signal-preflight output parent does not exist")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    published = False
    try:
        artifact_path = temporary / DEEPSOZ_SIGNAL_PREFLIGHT_FILENAME
        with artifact_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        if os.path.lexists(output):
            raise FileExistsError("Signal-preflight bundle destination already exists")
        os.rename(temporary, output)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    return VerifiedDeepSOZSignalPreflightBundle(
        receipt=receipt,
        artifact_sha256=_bytes_sha256(encoded),
        receipt_sha256=receipt_sha,
    )


def _parse_artifact(encoded: bytes) -> tuple[dict[str, object], dict[str, object]]:
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
        payload = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Signal-preflight artifact is not strict UTF-8 JSON") from exc
    payload = _closed_object(payload, expected=_ARTIFACT_FIELDS, field="artifact")
    if _canonical_json_bytes(payload) != encoded:
        raise ValueError("Signal-preflight artifact bytes are not canonical JSON")
    if payload["schema_version"] != DEEPSOZ_SIGNAL_PREFLIGHT_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported signal-preflight artifact schema")
    if payload["serialization"] != "canonical_json_utf8_newline_no_pickle":
        raise ValueError("Signal-preflight artifact uses unsafe serialization")
    receipt = _validate_receipt(payload["receipt"])
    declared = _require_sha256(payload["receipt_sha256"], field="receipt_sha256")
    if declared != _canonical_sha256(receipt):
        raise ValueError("Signal-preflight artifact receipt SHA mismatch")
    return payload, receipt


def load_deepsoz_signal_preflight_bundle(
    bundle_directory: str | Path,
    event_inputs_csv: str | Path,
    record_crosswalk_csv: str | Path,
    split_manifest_csv: str | Path,
    deepsoz_source_csv: str | Path,
    conservative_mapping_csv: str | Path,
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    tusz_root: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_event_inputs_sha256: str,
    expected_record_crosswalk_sha256: str,
    expected_split_manifest_sha256: str,
    expected_deepsoz_source_sha256: str,
    expected_conservative_mapping_sha256: str,
    config: CausalEDFConfig = CausalEDFConfig(),
    reader_factory: Callable[[str], object] | None = None,
) -> VerifiedDeepSOZSignalPreflightBundle:
    """Strictly load and fully replay a formal DeepSOZ signal-preflight bundle."""

    bundle = _reject_symlink_components(Path(bundle_directory), field="preflight bundle")
    if not bundle.is_dir():
        raise FileNotFoundError("Signal-preflight bundle directory does not exist")
    entries = tuple(sorted(bundle.iterdir(), key=lambda path: path.name))
    if len(entries) != 1 or entries[0].name != DEEPSOZ_SIGNAL_PREFLIGHT_FILENAME:
        raise ValueError("Signal-preflight bundle violates its closed file schema")
    if entries[0].is_symlink() or not entries[0].is_file():
        raise ValueError("Signal-preflight artifact must be regular and non-symlinked")
    encoded, artifact_sha = _read_stable_regular_file(
        entries[0], field="signal-preflight artifact", max_bytes=_MAX_ARTIFACT_BYTES
    )
    _check_expected_sha(
        artifact_sha, expected_artifact_sha256, field="expected_artifact_sha256"
    )
    _, receipt = _parse_artifact(encoded)
    rebuilt = _build_receipt(
        event_inputs_csv,
        record_crosswalk_csv,
        split_manifest_csv,
        deepsoz_source_csv,
        conservative_mapping_csv,
        verified_target_v2,
        tusz_root,
        expected_event_inputs_sha256=expected_event_inputs_sha256,
        expected_record_crosswalk_sha256=expected_record_crosswalk_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_deepsoz_source_sha256=expected_deepsoz_source_sha256,
        expected_conservative_mapping_sha256=expected_conservative_mapping_sha256,
        config=config,
        reader_factory=reader_factory,
    )
    if _canonical_json_bytes(receipt) != _canonical_json_bytes(rebuilt):
        raise ValueError(
            "Signal-preflight artifact does not exactly match full input/signal replay"
        )
    return VerifiedDeepSOZSignalPreflightBundle(
        receipt=receipt,
        artifact_sha256=artifact_sha,
        receipt_sha256=_canonical_sha256(receipt),
    )


__all__ = [
    "DEEPSOZ_SIGNAL_PREFLIGHT_ARTIFACT_SCHEMA",
    "DEEPSOZ_SIGNAL_PREFLIGHT_FILENAME",
    "DEEPSOZ_SIGNAL_PREFLIGHT_POLICY",
    "DEEPSOZ_SIGNAL_PREFLIGHT_SCHEMA",
    "VerifiedDeepSOZSignalPreflightBundle",
    "build_deepsoz_signal_preflight_bundle",
    "load_deepsoz_signal_preflight_bundle",
]
