"""EEG-only technical clock extraction for discontinuous EDF+D containers.

EDF+D stores a mandatory data-record onset in the first TAL of every data
record.  The same signal can also contain descriptive annotations.  This
module reads only the signed numeric prefix before the first TAL separator
byte (0x14), one byte at a time, and then seeks directly to the next record.
It never reads or decodes the bytes after that separator.

The resulting receipt is a physical, piecewise clock plan.  It is not an
annotation event source and does not make current continuous canonical views
eligible: when gaps exist, materialization remains fail-closed until a
segmented canonical-signal contract is implemented.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

from src.soz.geometry import STANDARD_19, normalize_electrode_name


EDFD_TECHNICAL_CLOCK_SCHEMA_VERSION = "clinical_eeg_edfd_technical_clock_v1"
EDFD_TECHNICAL_CLOCK_METHOD_ID = "edf_d_first_tal_numeric_prefix_only_v1"
EDFD_SEGMENTED_CANONICAL_GATE = "blocked_piecewise_clock_required_v1"
EDFD_CONTIGUOUS_CANONICAL_GATE = "eligible_contiguous_clock_v1"

EDFD_TECHNICAL_CLOCK_SCOPE: dict[str, object] = {
    "eeg_signal_header_used": True,
    "edf_identity_header_bytes_read": False,
    "edf_annotation_api_called": False,
    "technical_timekeeping_tal_prefix_used": True,
    "descriptive_tal_bytes_read": False,
    "descriptive_tal_decoded": False,
    "descriptive_edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "identity_fields_used": False,
}

_TIMEKEEPING_PREFIX = re.compile(rb"^[+-][0-9]+(?:\.[0-9]+)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_TIMEKEEPING_PREFIX_BYTES = 32
_TAL_SEPARATOR = b"\x14"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_edf(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ValueError("EDF+D source must not be a symbolic link")
    resolved = source.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix.lower() != ".edf":
        raise ValueError("EDF+D source must be a regular EDF file")
    return resolved


def _ascii_integer(value: bytes, *, context: str) -> int:
    try:
        return int(value.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError, OverflowError) as exc:
        raise ValueError(f"EDF+D {context} is invalid") from exc


def _ascii_decimal(value: bytes, *, context: str) -> Decimal:
    try:
        result = Decimal(value.decode("ascii").strip())
    except (UnicodeDecodeError, InvalidOperation) as exc:
        raise ValueError(f"EDF+D {context} is invalid") from exc
    if not result.is_finite():
        raise ValueError(f"EDF+D {context} must be finite")
    return result


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _split_fields(
    header: bytes,
    *,
    count: int,
    width: int,
    offset: int,
) -> tuple[list[str], int]:
    stop = offset + count * width
    if stop > len(header):
        raise ValueError("EDF+D signal header is truncated")
    values = [
        header[offset + index * width : offset + (index + 1) * width]
        .decode("latin-1")
        .strip()
        for index in range(count)
    ]
    return values, stop


def _rational_rate(samples_per_record: int, duration: Decimal) -> tuple[int, int]:
    ratio = Fraction(samples_per_record, 1) / Fraction(duration)
    return ratio.numerator, ratio.denominator


def _read_timekeeping_prefix(
    stream: object,
    *,
    byte_offset: int,
) -> Decimal:
    stream.seek(byte_offset)
    prefix = bytearray()
    for _ in range(_MAX_TIMEKEEPING_PREFIX_BYTES + 1):
        value = stream.read(1)
        if not value:
            raise ValueError("EDF+D timekeeping TAL is truncated")
        if value == _TAL_SEPARATOR:
            break
        if value in {b"\x00", b"\x15"}:
            raise ValueError("EDF+D first TAL is not a pure timekeeping TAL")
        prefix.extend(value)
        if len(prefix) > _MAX_TIMEKEEPING_PREFIX_BYTES:
            raise ValueError("EDF+D timekeeping prefix is unexpectedly long")
    else:  # pragma: no cover - loop always exits by break/error
        raise ValueError("EDF+D timekeeping separator is absent")
    encoded = bytes(prefix)
    if not _TIMEKEEPING_PREFIX.fullmatch(encoded):
        raise ValueError("EDF+D timekeeping prefix is not a signed decimal")
    try:
        result = Decimal(encoded.decode("ascii"))
    except (UnicodeDecodeError, InvalidOperation) as exc:  # defensive
        raise ValueError("EDF+D timekeeping prefix is invalid") from exc
    if not result.is_finite():
        raise ValueError("EDF+D timekeeping prefix must be finite")
    return result


def _build_runs(
    onsets: Sequence[Decimal],
    *,
    duration: Decimal,
    samples_per_record: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not onsets:
        raise ValueError("EDF+D technical clock has no data records")
    origin = onsets[0]
    run_starts = [0]
    gaps: list[dict[str, object]] = []
    for index in range(1, len(onsets)):
        expected = onsets[index - 1] + duration
        if onsets[index] < expected:
            raise ValueError("EDF+D data-record technical times overlap or go backward")
        if onsets[index] > expected:
            gap_start = expected - origin
            gap_stop = onsets[index] - origin
            gaps.append(
                {
                    "gap_id": f"GAP-{len(gaps) + 1:04d}",
                    "previous_data_record_index": index - 1,
                    "next_data_record_index": index,
                    "start_recording_seconds": float(gap_start),
                    "stop_recording_seconds": float(gap_stop),
                    "duration_seconds": float(gap_stop - gap_start),
                }
            )
            run_starts.append(index)
    run_stops = run_starts[1:] + [len(onsets)]
    runs: list[dict[str, object]] = []
    for run_index, (start, stop) in enumerate(zip(run_starts, run_stops), start=1):
        recording_start = onsets[start] - origin
        recording_stop = onsets[stop - 1] - origin + duration
        runs.append(
            {
                "run_id": f"RUN-{run_index:04d}",
                "data_record_interval": [start, stop],
                "source_sample_interval": [
                    start * samples_per_record,
                    stop * samples_per_record,
                ],
                "recording_seconds_interval": [
                    float(recording_start),
                    float(recording_stop),
                ],
            }
        )
    return runs, gaps


def inspect_edfd_technical_clock(edf_path: str | Path) -> dict[str, Any]:
    """Extract only mandatory EDF+D data-record onset prefixes.

    Patient/recording header bytes 8:184 are skipped.  For each annotation
    signal block, this reader consumes bytes only until the first 0x14 and
    seeks over the remaining block, so descriptive TAL content is not read.
    """

    path = _source_edf(edf_path)
    file_size = path.stat().st_size
    # Unbuffered I/O is intentional: a buffered ``read(1)`` could prefetch
    # descriptive TAL bytes beyond the first separator into process memory.
    with path.open("rb", buffering=0) as stream:
        stream.seek(184)
        fixed_tail = stream.read(72)
        if len(fixed_tail) != 72:
            raise ValueError("EDF+D fixed header is truncated")
        header_bytes = _ascii_integer(
            fixed_tail[0:8], context="header byte count"
        )
        reserved = fixed_tail[8:52].decode("latin-1").strip()
        declared_records = _ascii_integer(
            fixed_tail[52:60], context="data-record count"
        )
        record_duration = _ascii_decimal(
            fixed_tail[60:68], context="data-record duration"
        )
        signal_count = _ascii_integer(fixed_tail[68:72], context="signal count")
        if reserved != "EDF+D":
            raise ValueError("technical clock parser accepts only EDF+D")
        if signal_count < 1 or signal_count > 1024 or record_duration <= 0:
            raise ValueError("EDF+D signal count or data-record duration is invalid")
        if header_bytes != 256 + signal_count * 256:
            raise ValueError("EDF+D header byte count is inconsistent")

        stream.seek(256)
        signal_header = stream.read(signal_count * 256)
        if len(signal_header) != signal_count * 256:
            raise ValueError("EDF+D signal header is truncated")
        offset = 0
        labels, offset = _split_fields(
            signal_header, count=signal_count, width=16, offset=offset
        )
        _, offset = _split_fields(
            signal_header, count=signal_count, width=80, offset=offset
        )
        dimensions, offset = _split_fields(
            signal_header, count=signal_count, width=8, offset=offset
        )
        # Skip calibration fields without decoding clinical or identity data.
        for width in (8, 8, 8, 8):
            _, offset = _split_fields(
                signal_header, count=signal_count, width=width, offset=offset
            )
        prefilters, offset = _split_fields(
            signal_header, count=signal_count, width=80, offset=offset
        )
        sample_text, offset = _split_fields(
            signal_header, count=signal_count, width=8, offset=offset
        )
        _, offset = _split_fields(
            signal_header, count=signal_count, width=32, offset=offset
        )
        if offset != len(signal_header):
            raise ValueError("EDF+D signal header framing drifted")
        try:
            samples_per_record = [int(value) for value in sample_text]
        except ValueError as exc:
            raise ValueError("EDF+D samples-per-record field is invalid") from exc
        if any(value <= 0 for value in samples_per_record):
            raise ValueError("EDF+D samples per data record must be positive")

        annotation_indices = [
            index for index, label in enumerate(labels) if label == "EDF Annotations"
        ]
        if len(annotation_indices) != 1:
            raise ValueError("EDF+D requires exactly one technical annotation signal")
        annotation_index = annotation_indices[0]
        record_bytes = 2 * sum(samples_per_record)
        payload_bytes = file_size - header_bytes
        if payload_bytes <= 0 or payload_bytes % record_bytes:
            raise ValueError("EDF+D data-record framing is invalid")
        actual_records = payload_bytes // record_bytes
        if declared_records not in {-1, actual_records}:
            raise ValueError("EDF+D declared data-record count disagrees with file size")
        annotation_offset = 2 * sum(samples_per_record[:annotation_index])

        onsets = [
            _read_timekeeping_prefix(
                stream,
                byte_offset=header_bytes
                + record_index * record_bytes
                + annotation_offset,
            )
            for record_index in range(actual_records)
        ]

    candidates: dict[str, list[int]] = {channel: [] for channel in STANDARD_19}
    for index, label in enumerate(labels):
        canonical = normalize_electrode_name(label)
        if canonical in candidates:
            candidates[canonical].append(index)
    duplicates = [channel for channel, indices in candidates.items() if len(indices) > 1]
    if duplicates:
        raise ValueError("EDF+D has ambiguous direct standard-19 signal labels")
    observed = [channel for channel in STANDARD_19 if candidates[channel]]
    if not observed:
        raise ValueError("EDF+D has no directly observed standard-19 EEG channels")
    eeg_indices = [candidates[channel][0] for channel in observed]
    eeg_samples = {samples_per_record[index] for index in eeg_indices}
    if len(eeg_samples) != 1:
        raise ValueError("EDF+D selected EEG channels do not share one clock")
    eeg_samples_per_record = next(iter(eeg_samples))
    sampling_rate = _rational_rate(eeg_samples_per_record, record_duration)
    runs, gaps = _build_runs(
        onsets,
        duration=record_duration,
        samples_per_record=eeg_samples_per_record,
    )
    onset_text = [_decimal_text(value) for value in onsets]
    layout_core = {
        "standard19_observed": observed,
        "standard19_unobserved": [
            channel for channel in STANDARD_19 if channel not in set(observed)
        ],
        "selected_signal_layout": [
            {
                "channel_id": channel,
                "physical_dimension": dimensions[index],
                "prefilter": prefilters[index],
                "samples_per_data_record": samples_per_record[index],
            }
            for channel, index in zip(observed, eeg_indices)
        ],
        "annotation_signal_index": annotation_index,
        "annotation_samples_per_data_record": samples_per_record[annotation_index],
        "data_record_bytes": record_bytes,
    }
    gate_status = (
        EDFD_SEGMENTED_CANONICAL_GATE
        if gaps
        else EDFD_CONTIGUOUS_CANONICAL_GATE
    )
    body: dict[str, Any] = {
        "schema_version": EDFD_TECHNICAL_CLOCK_SCHEMA_VERSION,
        "method_id": EDFD_TECHNICAL_CLOCK_METHOD_ID,
        "edf_plus_mode": "EDF+D",
        "signal_layout_sha256": _canonical_sha256(layout_core),
        "data_record_count": actual_records,
        "data_record_duration_seconds": float(record_duration),
        "technical_data_record_onsets_seconds": onset_text,
        "technical_onset_vector_sha256": _canonical_sha256(onset_text),
        "clock_origin_policy": "first_data_record_timekeeping_onset_is_recording_zero_v1",
        "first_data_record_technical_onset_seconds": float(onsets[0]),
        "common_eeg_clock": {
            "sampling_rate_numerator": sampling_rate[0],
            "sampling_rate_denominator": sampling_rate[1],
            "samples_per_data_record": eeg_samples_per_record,
        },
        "standard19_observed": observed,
        "standard19_unobserved": [
            channel for channel in STANDARD_19 if channel not in set(observed)
        ],
        "continuous_runs": runs,
        "gaps": gaps,
        "canonical_materialization_gate": {
            "status": gate_status,
            "reason_codes": (
                ["edf_d_discontinuities_require_piecewise_physical_clock"]
                if gaps
                else []
            ),
        },
        "scope_receipt": deepcopy(EDFD_TECHNICAL_CLOCK_SCOPE),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_edfd_technical_clock_receipt(body)


def _strict_keys(value: object, required: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"{context} has missing or unknown fields")
    return deepcopy(value)


def validate_edfd_technical_clock_receipt(payload: object) -> dict[str, Any]:
    """Validate exact onset, run, gap, gate and privacy invariants."""

    required = {
        "schema_version",
        "method_id",
        "edf_plus_mode",
        "signal_layout_sha256",
        "data_record_count",
        "data_record_duration_seconds",
        "technical_data_record_onsets_seconds",
        "technical_onset_vector_sha256",
        "clock_origin_policy",
        "first_data_record_technical_onset_seconds",
        "common_eeg_clock",
        "standard19_observed",
        "standard19_unobserved",
        "continuous_runs",
        "gaps",
        "canonical_materialization_gate",
        "scope_receipt",
        "receipt_sha256",
    }
    data = _strict_keys(payload, required, "EDF+D technical clock receipt")
    if data["schema_version"] != EDFD_TECHNICAL_CLOCK_SCHEMA_VERSION:
        raise ValueError("unsupported EDF+D technical-clock schema")
    if data["method_id"] != EDFD_TECHNICAL_CLOCK_METHOD_ID or data["edf_plus_mode"] != "EDF+D":
        raise ValueError("EDF+D technical-clock method or mode drifted")
    for field in ("signal_layout_sha256", "technical_onset_vector_sha256", "receipt_sha256"):
        if not isinstance(data[field], str) or not _SHA256.fullmatch(data[field]):
            raise ValueError(f"EDF+D {field} must be SHA-256")
    count = data["data_record_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("EDF+D data_record_count is invalid")
    try:
        duration = Decimal(str(data["data_record_duration_seconds"]))
    except InvalidOperation as exc:
        raise ValueError("EDF+D data-record duration is invalid") from exc
    if not duration.is_finite() or duration <= 0:
        raise ValueError("EDF+D data-record duration is invalid")
    raw_onsets = data["technical_data_record_onsets_seconds"]
    if not isinstance(raw_onsets, list) or len(raw_onsets) != count:
        raise ValueError("EDF+D technical onset vector length is invalid")
    onsets: list[Decimal] = []
    for value in raw_onsets:
        if not isinstance(value, str):
            raise ValueError("EDF+D technical onsets must use exact decimal strings")
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("EDF+D technical onset string is invalid") from exc
        if not parsed.is_finite() or _decimal_text(parsed) != value:
            raise ValueError("EDF+D technical onset string is not canonical")
        onsets.append(parsed)
    if data["technical_onset_vector_sha256"] != _canonical_sha256(raw_onsets):
        raise ValueError("EDF+D onset-vector hash drifted")
    if data["clock_origin_policy"] != "first_data_record_timekeeping_onset_is_recording_zero_v1":
        raise ValueError("EDF+D clock origin policy drifted")
    if not math.isclose(
        float(data["first_data_record_technical_onset_seconds"]),
        float(onsets[0]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("EDF+D first technical onset drifted")
    clock = _strict_keys(
        data["common_eeg_clock"],
        {
            "sampling_rate_numerator",
            "sampling_rate_denominator",
            "samples_per_data_record",
        },
        "EDF+D common EEG clock",
    )
    numerator = clock["sampling_rate_numerator"]
    denominator = clock["sampling_rate_denominator"]
    samples_per_record = clock["samples_per_data_record"]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in (numerator, denominator, samples_per_record)
    ):
        raise ValueError("EDF+D common EEG clock values are invalid")
    if math.gcd(numerator, denominator) != 1:
        raise ValueError("EDF+D sampling rate must be reduced")
    if Fraction(numerator, denominator) != Fraction(samples_per_record, 1) / Fraction(duration):
        raise ValueError("EDF+D common EEG clock disagrees with data records")
    observed = data["standard19_observed"]
    unobserved = data["standard19_unobserved"]
    if (
        not isinstance(observed, list)
        or not isinstance(unobserved, list)
        or observed != [item for item in STANDARD_19 if item in set(observed)]
        or unobserved != [item for item in STANDARD_19 if item in set(unobserved)]
        or set(observed).intersection(unobserved)
        or set(observed).union(unobserved) != set(STANDARD_19)
    ):
        raise ValueError("EDF+D standard-19 coverage partition is invalid")
    expected_runs, expected_gaps = _build_runs(
        onsets,
        duration=duration,
        samples_per_record=samples_per_record,
    )
    if data["continuous_runs"] != expected_runs or data["gaps"] != expected_gaps:
        raise ValueError("EDF+D continuous-run or gap plan drifted")
    gate = _strict_keys(
        data["canonical_materialization_gate"],
        {"status", "reason_codes"},
        "EDF+D canonical materialization gate",
    )
    expected_gate = {
        "status": (
            EDFD_SEGMENTED_CANONICAL_GATE
            if expected_gaps
            else EDFD_CONTIGUOUS_CANONICAL_GATE
        ),
        "reason_codes": (
            ["edf_d_discontinuities_require_piecewise_physical_clock"]
            if expected_gaps
            else []
        ),
    }
    if gate != expected_gate:
        raise ValueError("EDF+D canonical materialization gate drifted")
    if data["scope_receipt"] != EDFD_TECHNICAL_CLOCK_SCOPE:
        raise ValueError("EDF+D technical clock violates the EEG-only scope")
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("EDF+D technical-clock receipt hash drifted")
    return data


def edfd_source_sample_index_to_recording_seconds(
    technical_clock_receipt: object,
    *,
    sample_index: int,
) -> float:
    """Map a flattened EEG sample edge through the piecewise EDF+D clock."""

    receipt = validate_edfd_technical_clock_receipt(technical_clock_receipt)
    if not isinstance(sample_index, int) or isinstance(sample_index, bool) or sample_index < 0:
        raise ValueError("EDF+D sample_index must be a non-negative integer")
    samples_per_record = int(receipt["common_eeg_clock"]["samples_per_data_record"])
    total_samples = int(receipt["data_record_count"]) * samples_per_record
    if sample_index > total_samples:
        raise ValueError("EDF+D sample_index lies outside the recording")
    if sample_index == total_samples:
        record_index = int(receipt["data_record_count"]) - 1
        within = samples_per_record
    else:
        record_index, within = divmod(sample_index, samples_per_record)
    numerator = int(receipt["common_eeg_clock"]["sampling_rate_numerator"])
    denominator = int(receipt["common_eeg_clock"]["sampling_rate_denominator"])
    onsets = [
        Decimal(value) for value in receipt["technical_data_record_onsets_seconds"]
    ]
    relative = (
        onsets[record_index]
        - onsets[0]
        + Decimal(within * denominator) / Decimal(numerator)
    )
    return float(relative)


__all__ = [
    "EDFD_CONTIGUOUS_CANONICAL_GATE",
    "EDFD_SEGMENTED_CANONICAL_GATE",
    "EDFD_TECHNICAL_CLOCK_METHOD_ID",
    "EDFD_TECHNICAL_CLOCK_SCHEMA_VERSION",
    "EDFD_TECHNICAL_CLOCK_SCOPE",
    "edfd_source_sample_index_to_recording_seconds",
    "inspect_edfd_technical_clock",
    "validate_edfd_technical_clock_receipt",
]
