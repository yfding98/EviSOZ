"""Strict TUSZ ``csv_v1.0.0`` edge-time involvement adapter.

The adapter anchors a fixed ``[-12, +48)`` second window to the start of an
official global ``TERM`` row whose label belongs to the frozen TUSZ seizure-
type vocabulary. Spatial supervision remains on the native bipolar edge:
per-channel onset order is never converted into an SOZ label, and the two
auricular TCP22 derivations are explicitly discarded.

A zero is observed only when one explicit ``bckg`` row covers an entire
one-second bin.  A one is observed only when one legal seizure-label row
covers the entire bin.  Gaps, partial coverage, within-bin transitions, and
conflicts stay masked.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Final, Literal, Sequence

import torch

from ..geometry import N_TCP_EDGES, TCP_20_EDGES
from ..temporal_masks import (
    OffsetAwarePhaseMasks,
    build_offset_aware_phase_masks,
)


TUSZ_ANNOTATION_VERSION: Final[str] = "csv_v1.0.0"
TUSZ_EDGE_TARGET_SCHEMA: Final[str] = "tusz_edge_ictal_involvement_v2"
TUSZ_EVENT_ANCHOR_SEMANTICS: Final[str] = (
    "official_global_ictal_label_start_not_earliest_channel"
)
TUSZ_WINDOW_START_SEC: Final[float] = -12.0
TUSZ_WINDOW_STOP_SEC: Final[float] = 48.0
TUSZ_BIN_SECONDS: Final[float] = 1.0
TUSZ_N_BINS: Final[int] = 60
TIME_EPS_SEC: Final[float] = 1e-6

TUSZ_SEIZURE_TYPE_LABELS: Final[tuple[str, ...]] = (
    "seiz",
    "fnsz",
    "gnsz",
    "spsz",
    "cpsz",
    "absz",
    "tnsz",
    "cnsz",
    "tcsz",
    "atsz",
    "mysz",
    "nesz",
)
TUSZ_CHANNEL_LABEL_VOCABULARY: Final[tuple[str, ...]] = (
    "bckg",
    *TUSZ_SEIZURE_TYPE_LABELS,
)
TUSZ_GLOBAL_LABEL_VOCABULARY: Final[tuple[str, ...]] = (
    "bckg",
    *TUSZ_SEIZURE_TYPE_LABELS,
)

MODERN_TCP20_NAMES: Final[tuple[str, ...]] = tuple(
    f"{left}-{right}" for left, right in TCP_20_EDGES
)
LEGACY_TCP22_NAMES: Final[tuple[str, ...]] = (
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
)
DROPPED_LEGACY_TCP_EDGES: Final[tuple[str, ...]] = ("A1-T3", "T4-A2")
LEGACY_TCP22_TO_MODERN_TCP20: Final[dict[str, str | None]] = {
    "FP1-F7": "FP1-F7",
    "F7-T3": "F7-T7",
    "T3-T5": "T7-P7",
    "T5-O1": "P7-O1",
    "FP2-F8": "FP2-F8",
    "F8-T4": "F8-T8",
    "T4-T6": "T8-P8",
    "T6-O2": "P8-O2",
    "A1-T3": None,
    "T3-C3": "T7-C3",
    "C3-CZ": "C3-CZ",
    "CZ-C4": "CZ-C4",
    "C4-T4": "C4-T8",
    "T4-A2": None,
    "FP1-F3": "FP1-F3",
    "F3-C3": "F3-C3",
    "C3-P3": "C3-P3",
    "P3-O1": "P3-O1",
    "FP2-F4": "FP2-F4",
    "F4-C4": "F4-C4",
    "C4-P4": "C4-P4",
    "P4-O2": "P4-O2",
}

BIN_STATE_EXPLICIT_BACKGROUND: Final[str] = "explicit_bckg"
BIN_STATE_EXPLICIT_ICTAL: Final[str] = "explicit_ictal"
BIN_STATE_GAP: Final[str] = "gap"
BIN_STATE_PARTIAL: Final[str] = "partial_coverage"
BIN_STATE_TRANSITION: Final[str] = "transition"
BIN_STATE_CONFLICT: Final[str] = "conflict"
BIN_STATE_LOW_CONFIDENCE: Final[str] = "low_confidence"
_BIN_STATES: Final[frozenset[str]] = frozenset(
    {
        BIN_STATE_EXPLICIT_BACKGROUND,
        BIN_STATE_EXPLICIT_ICTAL,
        BIN_STATE_GAP,
        BIN_STATE_PARTIAL,
        BIN_STATE_TRANSITION,
        BIN_STATE_CONFLICT,
        BIN_STATE_LOW_CONFIDENCE,
    }
)

_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "channel",
    "start_time",
    "stop_time",
    "label",
    "confidence",
)
_REQUIRED_METADATA: Final[frozenset[str]] = frozenset(
    {"version", "bname", "duration", "montage_file"}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_DURATION_PATTERN = re.compile(
    r"(?P<seconds>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s+secs",
    flags=re.IGNORECASE,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, field: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")


@dataclass(frozen=True)
class _AnnotationRow:
    channel: str
    start_sec: float
    stop_sec: float
    label: str
    confidence: float
    source_row: int


@dataclass(frozen=True)
class _ParsedAnnotation:
    path: Path
    sha256: str
    bname: str
    duration_sec: float
    montage_file: str
    rows: tuple[_AnnotationRow, ...]


@dataclass(frozen=True)
class TUSZGlobalSeizureEvent:
    """One chronological official global ictal-label interval.

    This object deliberately contains no channel-derived onset field.  Its
    index is the same index accepted by
    :func:`load_tusz_ictal_involvement_target`.
    """

    event_index: int
    start_sec: float
    stop_sec: float
    seizure_type: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.event_index, bool)
            or not isinstance(self.event_index, int)
            or self.event_index < 0
        ):
            raise ValueError("TUSZ global event index must be a non-negative integer")
        if not (
            math.isfinite(float(self.start_sec))
            and math.isfinite(float(self.stop_sec))
            and float(self.stop_sec) > float(self.start_sec)
        ):
            raise ValueError("TUSZ global seizure interval is invalid")
        if self.seizure_type not in TUSZ_SEIZURE_TYPE_LABELS:
            raise ValueError("TUSZ global seizure type is outside the frozen vocabulary")


@dataclass(frozen=True)
class TUSZAnnotationPairSummary:
    """Strict record-level validation summary without derived spatial labels."""

    bname: str
    duration_sec: float
    source_sha256: str
    channel_annotation_sha256: str
    global_annotation_sha256: str
    annotation_pair_sha256: str
    global_seizure_events: tuple[TUSZGlobalSeizureEvent, ...]

    def __post_init__(self) -> None:
        if not self.bname or not math.isfinite(self.duration_sec) or self.duration_sec <= 0:
            raise ValueError("TUSZ annotation-pair summary requires valid metadata")
        for field in (
            "source_sha256",
            "channel_annotation_sha256",
            "global_annotation_sha256",
            "annotation_pair_sha256",
        ):
            _require_sha256(str(getattr(self, field)), field=field)
        if tuple(
            event.event_index for event in self.global_seizure_events
        ) != tuple(range(len(self.global_seizure_events))):
            raise ValueError("TUSZ global seizure indices must be contiguous and ordered")


@dataclass(frozen=True)
class TUSZAnnotationReceipt:
    """Cryptographic and semantic receipt for one event target tensor."""

    source_path: str
    source_sha256: str
    channel_annotation_path: str
    channel_annotation_sha256: str
    global_annotation_path: str
    global_annotation_sha256: str
    annotation_pair_sha256: str
    bname: str
    duration_sec: float
    selected_global_event_index: int
    global_seizure_event_count: int
    selected_global_t0_sec: float
    selected_global_stop_sec: float
    selected_global_seizure_type: str
    label_vocabulary: tuple[str, ...]
    observed_channel_labels: tuple[str, ...]
    observed_global_labels: tuple[str, ...]
    label_vocabulary_sha256: str
    canonical_edge_names: tuple[str, ...]
    legacy_mapping_sha256: str
    dropped_legacy_edges: tuple[str, ...]
    dropped_row_counts: tuple[tuple[str, int], ...]
    annotation_version: str = TUSZ_ANNOTATION_VERSION
    schema_version: str = TUSZ_EDGE_TARGET_SCHEMA
    target_semantics: str = "bipolar_edge_ictal_involvement_not_soz"
    event_anchor_semantics: str = TUSZ_EVENT_ANCHOR_SEMANTICS
    produces_soz_labels: bool = False

    def __post_init__(self) -> None:
        for field in (
            "source_sha256",
            "channel_annotation_sha256",
            "global_annotation_sha256",
            "annotation_pair_sha256",
            "label_vocabulary_sha256",
            "legacy_mapping_sha256",
        ):
            _require_sha256(str(getattr(self, field)), field=field)
        if self.annotation_version != TUSZ_ANNOTATION_VERSION:
            raise ValueError("Unsupported TUSZ annotation version in receipt")
        if self.schema_version != TUSZ_EDGE_TARGET_SCHEMA:
            raise ValueError("Unsupported TUSZ target schema in receipt")
        if self.label_vocabulary != TUSZ_CHANNEL_LABEL_VOCABULARY:
            raise ValueError("Receipt label vocabulary is not the frozen vocabulary")
        if self.label_vocabulary_sha256 != _sha256_json(self.label_vocabulary):
            raise ValueError("Receipt label vocabulary hash does not match")
        if self.canonical_edge_names != MODERN_TCP20_NAMES:
            raise ValueError("Receipt edge order is not the frozen modern TCP20")
        if self.dropped_legacy_edges != DROPPED_LEGACY_TCP_EDGES:
            raise ValueError("Receipt must explicitly retain both dropped auricular edges")
        if tuple(name for name, _ in self.dropped_row_counts) != self.dropped_legacy_edges:
            raise ValueError("Dropped-row counts must follow the frozen edge order")
        if any(count < 0 for _, count in self.dropped_row_counts):
            raise ValueError("Dropped-row counts cannot be negative")
        if self.produces_soz_labels:
            raise ValueError("TUSZ involvement adapter cannot produce SOZ labels")
        if self.target_semantics != "bipolar_edge_ictal_involvement_not_soz":
            raise ValueError("TUSZ target semantics cannot be changed")
        if self.event_anchor_semantics != TUSZ_EVENT_ANCHOR_SEMANTICS:
            raise ValueError("TUSZ event anchor semantics cannot be changed")
        if not self.bname or not math.isfinite(self.duration_sec) or self.duration_sec <= 0:
            raise ValueError("Receipt requires a valid source name and duration")
        if (
            isinstance(self.selected_global_event_index, bool)
            or self.selected_global_event_index < 0
            or self.selected_global_event_index >= self.global_seizure_event_count
        ):
            raise ValueError("Receipt global event index is out of range")
        if not (
            math.isfinite(self.selected_global_t0_sec)
            and math.isfinite(self.selected_global_stop_sec)
            and self.selected_global_stop_sec > self.selected_global_t0_sec
        ):
            raise ValueError("Receipt global event interval is invalid")
        if self.selected_global_seizure_type not in TUSZ_SEIZURE_TYPE_LABELS:
            raise ValueError(
                "Receipt selected global seizure type is outside the frozen vocabulary"
            )
        if any(
            label not in TUSZ_CHANNEL_LABEL_VOCABULARY
            for label in self.observed_channel_labels
        ):
            raise ValueError("Receipt contains an unknown observed channel label")
        if any(
            label not in TUSZ_GLOBAL_LABEL_VOCABULARY
            for label in self.observed_global_labels
        ):
            raise ValueError("Receipt contains an unknown observed global label")
        if self.selected_global_seizure_type not in self.observed_global_labels:
            raise ValueError("Selected seizure type is absent from observed global labels")


@dataclass(frozen=True)
class TUSZIctalInvolvementTarget:
    """One global-event-anchored edge-time target with shape ``[20, 60]``."""

    targets: torch.Tensor
    source_target_mask: torch.Tensor
    bin_states: tuple[tuple[str, ...], ...]
    event_t0_sec: float
    event_stop_sec: float
    previous_global_event_stop_sec: float | None
    relative_bin_edges_sec: tuple[float, ...]
    receipt: TUSZAnnotationReceipt

    def __post_init__(self) -> None:
        expected_shape = (N_TCP_EDGES, TUSZ_N_BINS)
        if tuple(self.targets.shape) != expected_shape:
            raise ValueError("TUSZ targets must have shape [20,60]")
        if tuple(self.source_target_mask.shape) != expected_shape:
            raise ValueError("TUSZ source target mask must have shape [20,60]")
        if (
            not self.targets.is_floating_point()
            or self.source_target_mask.dtype != torch.bool
        ):
            raise TypeError(
                "TUSZ targets must be float and source_target_mask must be bool"
            )
        if not torch.isfinite(self.targets).all():
            raise ValueError("TUSZ targets must be finite")
        if not torch.all((self.targets == 0) | (self.targets == 1)):
            raise ValueError("TUSZ targets must be binary even at masked positions")
        if torch.any(self.targets[~self.source_target_mask] != 0):
            raise ValueError("Masked TUSZ target fill values must be zero")
        if len(self.bin_states) != N_TCP_EDGES or any(
            len(row) != TUSZ_N_BINS for row in self.bin_states
        ):
            raise ValueError("bin_states must have shape [20,60]")
        if any(state not in _BIN_STATES for row in self.bin_states for state in row):
            raise ValueError("bin_states contains an unsupported state")
        for edge_index, row in enumerate(self.bin_states):
            for bin_index, state in enumerate(row):
                is_observed = bool(
                    self.source_target_mask[edge_index, bin_index]
                )
                if state == BIN_STATE_EXPLICIT_BACKGROUND:
                    if not is_observed or self.targets[edge_index, bin_index] != 0:
                        raise ValueError("Explicit background state must be observed zero")
                elif state == BIN_STATE_EXPLICIT_ICTAL:
                    if not is_observed or self.targets[edge_index, bin_index] != 1:
                        raise ValueError("Explicit ictal state must be observed one")
                elif is_observed:
                    raise ValueError("Ambiguous TUSZ bin states must stay masked")
        expected_edges = tuple(
            TUSZ_WINDOW_START_SEC + index * TUSZ_BIN_SECONDS
            for index in range(TUSZ_N_BINS + 1)
        )
        if self.relative_bin_edges_sec != expected_edges:
            raise ValueError("TUSZ relative bin grid must be exactly [-12,+48]")
        if self.event_t0_sec != self.receipt.selected_global_t0_sec:
            raise ValueError("Target t0 disagrees with its global-event receipt")
        if self.event_stop_sec != self.receipt.selected_global_stop_sec:
            raise ValueError("Target event stop disagrees with its receipt")
        if self.receipt.selected_global_event_index == 0:
            if self.previous_global_event_stop_sec is not None:
                raise ValueError("First global seizure cannot have a previous stop")
        else:
            if self.previous_global_event_stop_sec is None or not math.isfinite(
                float(self.previous_global_event_stop_sec)
            ):
                raise ValueError("Non-first global seizure requires a previous stop")
            if (
                float(self.previous_global_event_stop_sec)
                > self.event_t0_sec + TIME_EPS_SEC
            ):
                raise ValueError("Previous global seizure overlaps current t0")

    @property
    def mask(self) -> torch.Tensor:
        """Compatibility alias for the source-only supervision mask.

        New code should use :attr:`source_target_mask`.  This alias does not
        make annotation coverage a deployment or physical-signal mask.
        """

        return self.source_target_mask

    @property
    def source_positive_count(self) -> int:
        return int(self.targets[self.source_target_mask].sum().item())

    @property
    def source_explicit_negative_count(self) -> int:
        return int(self.source_target_mask.sum().item()) - self.source_positive_count

    def offset_aware_phase_masks(
        self,
        *,
        offset_trustworthy: bool,
        previous_timeline_trustworthy: bool,
    ) -> OffsetAwarePhaseMasks:
        """Build one-event phase semantics from the official global interval.

        Trust is an explicit caller decision because the adapter does not
        equate mere presence of a global stop with clinical reliability.  The
        fixed crop stop is never used as a seizure offset.
        """

        if not isinstance(offset_trustworthy, bool) or not isinstance(
            previous_timeline_trustworthy, bool
        ):
            raise TypeError("Current offset and previous timeline trust must be boolean")
        previous_gap = (
            None
            if self.previous_global_event_stop_sec is None
            else self.event_t0_sec - self.previous_global_event_stop_sec
        )
        return build_offset_aware_phase_masks(
            [self.event_stop_sec - self.event_t0_sec],
            offset_trustworthy=[offset_trustworthy],
            previous_seizure_gap_sec=[previous_gap],
            previous_timeline_trustworthy=[previous_timeline_trustworthy],
        )


def _parse_duration(value: str, *, path: Path) -> float:
    match = _DURATION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"{path}: invalid csv_v1.0.0 duration metadata {value!r}")
    duration = float(match.group("seconds"))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{path}: duration must be finite and positive")
    return duration


def _parse_float(value: str, *, field: str, path: Path, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: row {row_number} has non-numeric {field}={value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path}: row {row_number} has non-finite {field}")
    return parsed


def _parse_annotation(
    path: str | Path,
    *,
    kind: Literal["channel", "global"],
) -> _ParsedAnnotation:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        text = source.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: annotation is not valid UTF-8") from exc

    metadata: dict[str, str] = {}
    csv_lines: list[str] = []
    csv_started = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if csv_started:
                raise ValueError(f"{source}: comment after CSV header at line {line_number}")
            comment = stripped[1:].strip()
            if not comment:
                continue
            if "=" not in comment:
                raise ValueError(f"{source}: malformed metadata at line {line_number}")
            key, value = (part.strip() for part in comment.split("=", maxsplit=1))
            if key not in _REQUIRED_METADATA:
                raise ValueError(f"{source}: unsupported metadata key {key!r}")
            if key in metadata:
                raise ValueError(f"{source}: duplicate metadata key {key!r}")
            if not value:
                raise ValueError(f"{source}: empty metadata value for {key!r}")
            metadata[key] = value
            continue
        csv_started = True
        csv_lines.append(line)

    missing_metadata = sorted(_REQUIRED_METADATA - set(metadata))
    if missing_metadata:
        raise ValueError(f"{source}: missing metadata keys {missing_metadata}")
    if metadata["version"] != TUSZ_ANNOTATION_VERSION:
        raise ValueError(
            f"{source}: expected version {TUSZ_ANNOTATION_VERSION}, "
            f"got {metadata['version']!r}"
        )
    if not csv_lines:
        raise ValueError(f"{source}: missing CSV header")

    try:
        reader = csv.reader(csv_lines, strict=True)
        header = tuple(next(reader))
        if header != _CSV_COLUMNS:
            raise ValueError(
                f"{source}: invalid CSV schema {header}; expected {_CSV_COLUMNS}"
            )
        duration = _parse_duration(metadata["duration"], path=source)
        rows: list[_AnnotationRow] = []
        for data_index, values in enumerate(reader, start=2):
            if len(values) != len(_CSV_COLUMNS):
                raise ValueError(
                    f"{source}: row {data_index} has {len(values)} fields; expected 5"
                )
            if any(not str(value).strip() for value in values):
                raise ValueError(f"{source}: row {data_index} contains an empty value")
            channel, start_text, stop_text, label, confidence_text = (
                str(value).strip() for value in values
            )
            channel = channel.upper()
            label = label.lower()
            start = _parse_float(
                start_text, field="start_time", path=source, row_number=data_index
            )
            stop = _parse_float(
                stop_text, field="stop_time", path=source, row_number=data_index
            )
            confidence = _parse_float(
                confidence_text,
                field="confidence",
                path=source,
                row_number=data_index,
            )
            if start < 0 or stop <= start or stop > duration + TIME_EPS_SEC:
                raise ValueError(
                    f"{source}: row {data_index} has invalid interval [{start},{stop}) "
                    f"for duration {duration}"
                )
            if confidence < 0 or confidence > 1:
                raise ValueError(
                    f"{source}: row {data_index} confidence must be in [0,1]"
                )
            if kind == "global":
                if channel != "TERM":
                    raise ValueError(
                        f"{source}: global row {data_index} channel must be TERM"
                    )
                if label not in TUSZ_GLOBAL_LABEL_VOCABULARY:
                    raise ValueError(
                        f"{source}: global row {data_index} has illegal label {label!r}"
                    )
            else:
                if channel not in LEGACY_TCP22_TO_MODERN_TCP20 and channel not in MODERN_TCP20_NAMES:
                    raise ValueError(
                        f"{source}: channel row {data_index} has illegal edge {channel!r}"
                    )
                if label not in TUSZ_CHANNEL_LABEL_VOCABULARY:
                    raise ValueError(
                        f"{source}: channel row {data_index} has illegal label {label!r}"
                    )
            rows.append(
                _AnnotationRow(
                    channel=channel,
                    start_sec=start,
                    stop_sec=stop,
                    label=label,
                    confidence=confidence,
                    source_row=data_index,
                )
            )
    except csv.Error as exc:
        raise ValueError(f"{source}: malformed CSV: {exc}") from exc

    return _ParsedAnnotation(
        path=source,
        sha256=_sha256_file(source),
        bname=metadata["bname"],
        duration_sec=duration,
        montage_file=metadata["montage_file"],
        rows=tuple(rows),
    )


def _validate_annotation_pair(
    channel: _ParsedAnnotation,
    global_annotation: _ParsedAnnotation,
    source_path: Path,
) -> None:
    if channel.path.suffix != ".csv":
        raise ValueError("Per-channel annotation path must end in .csv")
    expected_global_name = f"{channel.path.stem}.csv_bi"
    if global_annotation.path.name != expected_global_name:
        raise ValueError(
            "Global annotation must be the paired .csv_bi sidecar; "
            f"expected {expected_global_name!r}"
        )
    if channel.path.parent.resolve() != global_annotation.path.parent.resolve():
        raise ValueError("Per-channel and global annotations must be sibling sidecars")
    if channel.bname != global_annotation.bname:
        raise ValueError("Per-channel and global bname metadata disagree")
    if channel.bname != channel.path.stem:
        raise ValueError("Annotation bname metadata does not match the sidecar basename")
    if abs(channel.duration_sec - global_annotation.duration_sec) > TIME_EPS_SEC:
        raise ValueError("Per-channel and global duration metadata disagree")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.stem != channel.bname:
        raise ValueError("Source basename does not match annotation bname metadata")


def _validate_global_timeline(rows: Sequence[_AnnotationRow], *, path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row.start_sec, row.stop_sec, row.source_row))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start_sec < previous.stop_sec - TIME_EPS_SEC:
            raise ValueError(
                f"{path}: overlapping global TERM rows "
                f"{previous.source_row} and {current.source_row}"
            )


def list_tusz_global_seizure_events(
    global_annotation_path: str | Path,
) -> tuple[TUSZGlobalSeizureEvent, ...]:
    """Return chronological official global ictal-label event anchors.

    The strict ``csv_v1.0.0`` parser and global-timeline checks are shared
    with target construction.  Per-channel rows are intentionally not an
    input, so this API cannot reinterpret the earliest involved edge as the
    event anchor or as an SOZ label.
    """

    parsed = _parse_annotation(global_annotation_path, kind="global")
    _validate_global_timeline(parsed.rows, path=parsed.path)
    seizures = tuple(
        sorted(
            (
                row
                for row in parsed.rows
                if row.label in TUSZ_SEIZURE_TYPE_LABELS
            ),
            key=lambda row: (row.start_sec, row.stop_sec, row.source_row),
        )
    )
    return tuple(
        TUSZGlobalSeizureEvent(
            event_index=index,
            start_sec=row.start_sec,
            stop_sec=row.stop_sec,
            seizure_type=row.label,
        )
        for index, row in enumerate(seizures)
    )


def inspect_tusz_annotation_pair(
    channel_annotation_path: str | Path,
    global_annotation_path: str | Path,
    *,
    source_path: str | Path | None = None,
) -> TUSZAnnotationPairSummary:
    """Validate one record's two sidecars and enumerate global event anchors.

    This is the record-discovery counterpart to target materialization.  It
    validates both strict annotation schemas, their metadata pairing, the EDF
    basename, and the non-overlapping global timeline without producing an
    SOZ-like endpoint label.
    """

    channel = _parse_annotation(channel_annotation_path, kind="channel")
    global_annotation = _parse_annotation(global_annotation_path, kind="global")
    source = (
        Path(source_path)
        if source_path is not None
        else Path(channel_annotation_path).with_suffix(".edf")
    )
    _validate_annotation_pair(channel, global_annotation, source)
    _validate_global_timeline(global_annotation.rows, path=global_annotation.path)
    seizures = tuple(
        sorted(
            (
                row
                for row in global_annotation.rows
                if row.label in TUSZ_SEIZURE_TYPE_LABELS
            ),
            key=lambda row: (row.start_sec, row.stop_sec, row.source_row),
        )
    )
    events = tuple(
        TUSZGlobalSeizureEvent(
            event_index=index,
            start_sec=row.start_sec,
            stop_sec=row.stop_sec,
            seizure_type=row.label,
        )
        for index, row in enumerate(seizures)
    )
    annotation_pair_sha256 = _sha256_json(
        {
            "channel_annotation_sha256": channel.sha256,
            "global_annotation_sha256": global_annotation.sha256,
        }
    )
    return TUSZAnnotationPairSummary(
        bname=channel.bname,
        duration_sec=channel.duration_sec,
        source_sha256=_sha256_file(source),
        channel_annotation_sha256=channel.sha256,
        global_annotation_sha256=global_annotation.sha256,
        annotation_pair_sha256=annotation_pair_sha256,
        global_seizure_events=events,
    )


def _canonical_edge(raw_edge: str) -> str | None:
    if raw_edge in LEGACY_TCP22_TO_MODERN_TCP20:
        return LEGACY_TCP22_TO_MODERN_TCP20[raw_edge]
    if raw_edge in MODERN_TCP20_NAMES:
        return raw_edge
    raise RuntimeError("Unvalidated TUSZ edge reached canonical mapping")


def _overlaps(row: _AnnotationRow, start: float, stop: float) -> bool:
    return row.start_sec < stop - TIME_EPS_SEC and row.stop_sec > start + TIME_EPS_SEC


def _fully_covers(row: _AnnotationRow, start: float, stop: float) -> bool:
    return row.start_sec <= start + TIME_EPS_SEC and row.stop_sec >= stop - TIME_EPS_SEC


def _multiple_row_state(
    rows: Sequence[_AnnotationRow], start: float, stop: float
) -> str:
    clipped = sorted(
        (
            max(start, row.start_sec),
            min(stop, row.stop_sec),
            row.source_row,
        )
        for row in rows
    )
    for previous, current in zip(clipped, clipped[1:]):
        if current[0] < previous[1] - TIME_EPS_SEC:
            return BIN_STATE_CONFLICT
    return BIN_STATE_TRANSITION


def load_tusz_ictal_involvement_target(
    channel_annotation_path: str | Path,
    global_annotation_path: str | Path,
    *,
    event_index: int,
    source_path: str | Path | None = None,
) -> TUSZIctalInvolvementTarget:
    """Load one strict global-event-anchored TUSZ edge-time target.

    ``event_index`` indexes chronologically sorted official global ictal-label
    rows, never per-channel annotations. The function has no option to derive
    an event anchor, an endpoint label, or an SOZ target from the earliest edge.
    """

    if isinstance(event_index, bool) or not isinstance(event_index, int) or event_index < 0:
        raise ValueError("event_index must be a non-negative integer")
    channel = _parse_annotation(channel_annotation_path, kind="channel")
    global_annotation = _parse_annotation(global_annotation_path, kind="global")
    source = (
        Path(source_path)
        if source_path is not None
        else Path(channel_annotation_path).with_suffix(".edf")
    )
    _validate_annotation_pair(channel, global_annotation, source)
    _validate_global_timeline(global_annotation.rows, path=global_annotation.path)

    global_seizures = tuple(
        sorted(
            (
                row
                for row in global_annotation.rows
                if row.label in TUSZ_SEIZURE_TYPE_LABELS
            ),
            key=lambda row: (row.start_sec, row.stop_sec, row.source_row),
        )
    )
    if not global_seizures:
        raise ValueError("Global .csv_bi contains no official ictal-label event")
    if event_index >= len(global_seizures):
        raise IndexError(
            f"event_index={event_index} but global .csv_bi has "
            f"{len(global_seizures)} seizure events"
        )
    event = global_seizures[event_index]

    by_edge: dict[str, list[_AnnotationRow]] = {
        edge: [] for edge in MODERN_TCP20_NAMES
    }
    dropped_counts = {edge: 0 for edge in DROPPED_LEGACY_TCP_EDGES}
    for row in channel.rows:
        canonical = _canonical_edge(row.channel)
        if canonical is None:
            dropped_counts[row.channel] += 1
            continue
        by_edge[canonical].append(row)

    targets = torch.zeros((N_TCP_EDGES, TUSZ_N_BINS), dtype=torch.float32)
    mask = torch.zeros((N_TCP_EDGES, TUSZ_N_BINS), dtype=torch.bool)
    states: list[tuple[str, ...]] = []
    for edge_index, edge in enumerate(MODERN_TCP20_NAMES):
        edge_states: list[str] = []
        rows = by_edge[edge]
        for bin_index in range(TUSZ_N_BINS):
            relative_start = TUSZ_WINDOW_START_SEC + bin_index * TUSZ_BIN_SECONDS
            absolute_start = event.start_sec + relative_start
            absolute_stop = absolute_start + TUSZ_BIN_SECONDS
            overlapping = tuple(
                row for row in rows if _overlaps(row, absolute_start, absolute_stop)
            )
            if not overlapping:
                state = BIN_STATE_GAP
            elif len(overlapping) > 1:
                state = _multiple_row_state(
                    overlapping, absolute_start, absolute_stop
                )
            else:
                row = overlapping[0]
                if not _fully_covers(row, absolute_start, absolute_stop):
                    state = BIN_STATE_PARTIAL
                elif row.confidence < 1.0 - TIME_EPS_SEC:
                    # A label with non-unit source confidence is not silently
                    # promoted to binary truth. It remains available in the
                    # annotation receipt but is masked for BCE supervision.
                    state = BIN_STATE_LOW_CONFIDENCE
                elif row.label == "bckg":
                    state = BIN_STATE_EXPLICIT_BACKGROUND
                    mask[edge_index, bin_index] = True
                elif row.label in TUSZ_SEIZURE_TYPE_LABELS:
                    state = BIN_STATE_EXPLICIT_ICTAL
                    targets[edge_index, bin_index] = 1.0
                    mask[edge_index, bin_index] = True
                else:  # pragma: no cover - parser guarantees the vocabulary
                    raise RuntimeError("Unvalidated label reached target construction")
            edge_states.append(state)
        states.append(tuple(edge_states))

    mapping_payload = tuple(
        (edge, LEGACY_TCP22_TO_MODERN_TCP20[edge]) for edge in LEGACY_TCP22_NAMES
    )
    annotation_pair_sha256 = _sha256_json(
        {
            "channel_annotation_sha256": channel.sha256,
            "global_annotation_sha256": global_annotation.sha256,
        }
    )
    receipt = TUSZAnnotationReceipt(
        source_path=str(source),
        source_sha256=_sha256_file(source),
        channel_annotation_path=str(channel.path),
        channel_annotation_sha256=channel.sha256,
        global_annotation_path=str(global_annotation.path),
        global_annotation_sha256=global_annotation.sha256,
        annotation_pair_sha256=annotation_pair_sha256,
        bname=channel.bname,
        duration_sec=channel.duration_sec,
        selected_global_event_index=event_index,
        global_seizure_event_count=len(global_seizures),
        selected_global_t0_sec=event.start_sec,
        selected_global_stop_sec=event.stop_sec,
        selected_global_seizure_type=event.label,
        label_vocabulary=TUSZ_CHANNEL_LABEL_VOCABULARY,
        observed_channel_labels=tuple(sorted({row.label for row in channel.rows})),
        observed_global_labels=tuple(
            sorted({row.label for row in global_annotation.rows})
        ),
        label_vocabulary_sha256=_sha256_json(TUSZ_CHANNEL_LABEL_VOCABULARY),
        canonical_edge_names=MODERN_TCP20_NAMES,
        legacy_mapping_sha256=_sha256_json(mapping_payload),
        dropped_legacy_edges=DROPPED_LEGACY_TCP_EDGES,
        dropped_row_counts=tuple(
            (edge, dropped_counts[edge]) for edge in DROPPED_LEGACY_TCP_EDGES
        ),
    )
    relative_bin_edges = tuple(
        TUSZ_WINDOW_START_SEC + index * TUSZ_BIN_SECONDS
        for index in range(TUSZ_N_BINS + 1)
    )
    return TUSZIctalInvolvementTarget(
        targets=targets,
        source_target_mask=mask,
        bin_states=tuple(states),
        event_t0_sec=event.start_sec,
        event_stop_sec=event.stop_sec,
        previous_global_event_stop_sec=(
            None if event_index == 0 else global_seizures[event_index - 1].stop_sec
        ),
        relative_bin_edges_sec=relative_bin_edges,
        receipt=receipt,
    )


_mapped_legacy_edges = tuple(
    LEGACY_TCP22_TO_MODERN_TCP20[edge]
    for edge in LEGACY_TCP22_NAMES
    if LEGACY_TCP22_TO_MODERN_TCP20[edge] is not None
)
if _mapped_legacy_edges != MODERN_TCP20_NAMES:
    raise RuntimeError("Legacy TCP22 mapping does not reproduce the frozen modern TCP20")
if len(MODERN_TCP20_NAMES) != N_TCP_EDGES:
    raise RuntimeError("Modern TCP20 names disagree with canonical geometry")


__all__ = [
    "BIN_STATE_CONFLICT",
    "BIN_STATE_EXPLICIT_BACKGROUND",
    "BIN_STATE_EXPLICIT_ICTAL",
    "BIN_STATE_GAP",
    "BIN_STATE_LOW_CONFIDENCE",
    "BIN_STATE_PARTIAL",
    "BIN_STATE_TRANSITION",
    "DROPPED_LEGACY_TCP_EDGES",
    "LEGACY_TCP22_NAMES",
    "LEGACY_TCP22_TO_MODERN_TCP20",
    "MODERN_TCP20_NAMES",
    "TUSZAnnotationPairSummary",
    "TUSZAnnotationReceipt",
    "TUSZGlobalSeizureEvent",
    "TUSZIctalInvolvementTarget",
    "TUSZ_ANNOTATION_VERSION",
    "TUSZ_CHANNEL_LABEL_VOCABULARY",
    "TUSZ_EVENT_ANCHOR_SEMANTICS",
    "TUSZ_GLOBAL_LABEL_VOCABULARY",
    "TUSZ_SEIZURE_TYPE_LABELS",
    "inspect_tusz_annotation_pair",
    "list_tusz_global_seizure_events",
    "load_tusz_ictal_involvement_target",
]
