"""Strict adapter for native TUEV ``.rec`` CE6 interval annotations.

The official annotation coordinate is a 22-derivation ACNS TCP montage.  This
adapter keeps labels on bipolar edges, drops the two A1/A2 derivations, and
maps the remaining derivations bijectively to the frozen modern TCP20 geometry.
It never derives a class from a filename or expands an edge label to endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Sequence

import torch

from ..geometry import MORPHOLOGY_CLASSES, STANDARD_19, TCP_20_EDGES


TUEV_REC_SCHEMA_VERSION = "tuev-rec-ce6-v1.0.0"
TUEV_LABEL_MAP: tuple[tuple[int, str], ...] = tuple(
    (index + 1, name) for index, name in enumerate(MORPHOLOGY_CLASSES)
)
TUEV_LABEL_BY_CODE = dict(TUEV_LABEL_MAP)

# Exact geometry documented in the TUEV release README.  Legacy temporal names
# are intentionally retained in the receipt; only identity aliases are applied
# when constructing the modern common geometry.
TUEV_OFFICIAL_TCP22: tuple[tuple[str, str], ...] = (
    ("FP1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("FP2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("A1", "T3"),
    ("T3", "C3"),
    ("C3", "CZ"),
    ("CZ", "C4"),
    ("C4", "T4"),
    ("T4", "A2"),
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
)
TUEV_DROPPED_OFFICIAL_INDICES = (8, 13)
_LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _modern_name(name: str) -> str:
    return _LEGACY_TO_MODERN.get(name, name)


def _build_official_to_modern() -> tuple[int | None, ...]:
    mapping: list[int | None] = []
    for official_index, (left, right) in enumerate(TUEV_OFFICIAL_TCP22):
        modern_edge = (_modern_name(left), _modern_name(right))
        if official_index in TUEV_DROPPED_OFFICIAL_INDICES:
            if all(endpoint in STANDARD_19 for endpoint in modern_edge):
                raise RuntimeError("A dropped TUEV edge unexpectedly lies in standard19")
            mapping.append(None)
            continue
        try:
            mapping.append(TCP_20_EDGES.index(modern_edge))
        except ValueError as exc:
            raise RuntimeError(
                f"Official TUEV edge {official_index}:{modern_edge} is absent from TCP20"
            ) from exc
    return tuple(mapping)


TUEV_OFFICIAL_TO_MODERN_TCP20 = _build_official_to_modern()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TUEVInterval:
    official_channel_index: int
    modern_edge_index: int | None
    start_sec: float
    stop_sec: float
    label_code: int
    label_index: int
    label_name: str
    source_line: int

    def __post_init__(self) -> None:
        if not 0 <= self.official_channel_index < len(TUEV_OFFICIAL_TCP22):
            raise ValueError("official_channel_index must be in [0,21]")
        expected_modern = TUEV_OFFICIAL_TO_MODERN_TCP20[self.official_channel_index]
        if self.modern_edge_index != expected_modern:
            raise ValueError("modern_edge_index disagrees with frozen TUEV geometry")
        if not math.isfinite(self.start_sec) or not math.isfinite(self.stop_sec):
            raise ValueError("TUEV interval times must be finite")
        if self.start_sec < 0 or self.stop_sec <= self.start_sec:
            raise ValueError("TUEV interval requires 0 <= start < stop")
        if self.label_code not in TUEV_LABEL_BY_CODE:
            raise ValueError("TUEV label code must be in [1,6]")
        if self.label_index != self.label_code - 1:
            raise ValueError("label_index must be the zero-based official label code")
        if self.label_name != TUEV_LABEL_BY_CODE[self.label_code]:
            raise ValueError("label_name disagrees with the official TUEV map")
        if self.source_line < 1:
            raise ValueError("source_line must be positive")


@dataclass(frozen=True)
class TUEVAnnotationReceipt:
    schema_version: str
    rec_path: str
    rec_sha256: str
    official_label_map: tuple[tuple[int, str], ...]
    official_tcp22_geometry: tuple[tuple[str, str], ...]
    official_to_modern_tcp20: tuple[int | None, ...]
    dropped_official_indices: tuple[int, ...]
    row_count: int
    retained_row_count: int
    dropped_row_count: int

    def __post_init__(self) -> None:
        if self.schema_version != TUEV_REC_SCHEMA_VERSION:
            raise ValueError("Unexpected TUEV receipt schema version")
        if not re.fullmatch(r"[0-9a-f]{64}", self.rec_sha256):
            raise ValueError("rec_sha256 must be a lowercase SHA256 digest")
        if self.official_label_map != TUEV_LABEL_MAP:
            raise ValueError("Receipt label map is not the frozen official CE6 map")
        if self.official_tcp22_geometry != TUEV_OFFICIAL_TCP22:
            raise ValueError("Receipt geometry is not the frozen official TCP22")
        if self.official_to_modern_tcp20 != TUEV_OFFICIAL_TO_MODERN_TCP20:
            raise ValueError("Receipt mapping is not the frozen TCP22-to-TCP20 map")
        if self.dropped_official_indices != TUEV_DROPPED_OFFICIAL_INDICES:
            raise ValueError("Receipt dropped-edge policy is not frozen")
        if min(self.row_count, self.retained_row_count, self.dropped_row_count) < 0:
            raise ValueError("Receipt row counts cannot be negative")
        if self.retained_row_count + self.dropped_row_count != self.row_count:
            raise ValueError("Receipt retained/dropped counts do not sum to row_count")


@dataclass(frozen=True)
class TUEVAnnotation:
    intervals: tuple[TUEVInterval, ...]
    receipt: TUEVAnnotationReceipt

    def __post_init__(self) -> None:
        if len(self.intervals) != self.receipt.row_count:
            raise ValueError("Annotation interval count disagrees with receipt")


@dataclass(frozen=True)
class TUEVCE6Targets:
    """Dense labels on caller-specified absolute one-second bins."""

    labels: torch.Tensor
    mask: torch.Tensor
    bin_starts_sec: tuple[float, ...]
    receipt: TUEVAnnotationReceipt

    def __post_init__(self) -> None:
        expected = (len(TCP_20_EDGES), len(self.bin_starts_sec))
        if tuple(self.labels.shape) != expected or tuple(self.mask.shape) != expected:
            raise ValueError(f"TUEV labels and mask must have shape {expected}")
        if self.labels.dtype != torch.long or self.mask.dtype != torch.bool:
            raise TypeError("TUEV labels must be long and mask must be bool")
        observed = self.labels[self.mask]
        if observed.numel() and not torch.all((observed >= 0) & (observed < 6)):
            raise ValueError("Observed TUEV CE6 labels must be in [0,5]")


def _parse_integer_field(
    text: str,
    *,
    field: str,
    path: Path,
    line_number: int,
) -> int:
    token = text.strip()
    if not re.fullmatch(r"[+-]?\d+", token):
        raise ValueError(
            f"Invalid TUEV {field} at {path}:{line_number}: {text!r}"
        )
    return int(token)


def _parse_time_field(
    text: str,
    *,
    field: str,
    path: Path,
    line_number: int,
) -> float:
    try:
        value = float(text.strip())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid TUEV {field} at {path}:{line_number}: {text!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Invalid TUEV {field} at {path}:{line_number}: {text!r}"
        )
    return value


def parse_tuev_rec(path: str | Path) -> TUEVAnnotation:
    """Parse an official four-column TUEV ``.rec`` file fail-closed."""

    rec_path = Path(path)
    if rec_path.suffix.lower() != ".rec":
        raise ValueError(f"TUEV annotation must use the .rec suffix: {rec_path}")
    if not rec_path.is_file():
        raise FileNotFoundError(rec_path)
    payload = rec_path.read_bytes()
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"TUEV .rec is not valid UTF-8 text: {rec_path}") from exc

    intervals: list[TUEVInterval] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = raw_line.split(",")
        if len(fields) != 4:
            raise ValueError(
                f"TUEV .rec requires exactly four columns at "
                f"{rec_path}:{line_number}: {raw_line!r}"
            )
        official_index = _parse_integer_field(
            fields[0],
            field="channel index",
            path=rec_path,
            line_number=line_number,
        )
        if not 0 <= official_index < len(TUEV_OFFICIAL_TCP22):
            raise ValueError(
                f"Invalid TUEV channel index at {rec_path}:{line_number}: "
                f"{official_index}; expected 0..21"
            )
        start = _parse_time_field(
            fields[1], field="start time", path=rec_path, line_number=line_number
        )
        stop = _parse_time_field(
            fields[2], field="stop time", path=rec_path, line_number=line_number
        )
        if start < 0 or stop <= start:
            raise ValueError(
                f"Invalid TUEV interval at {rec_path}:{line_number}: "
                f"require 0 <= start < stop, got [{start},{stop})"
            )
        label_code = _parse_integer_field(
            fields[3], field="label", path=rec_path, line_number=line_number
        )
        if label_code not in TUEV_LABEL_BY_CODE:
            raise ValueError(
                f"Invalid TUEV label at {rec_path}:{line_number}: "
                f"{label_code}; expected 1..6"
            )
        intervals.append(
            TUEVInterval(
                official_channel_index=official_index,
                modern_edge_index=TUEV_OFFICIAL_TO_MODERN_TCP20[official_index],
                start_sec=start,
                stop_sec=stop,
                label_code=label_code,
                label_index=label_code - 1,
                label_name=TUEV_LABEL_BY_CODE[label_code],
                source_line=line_number,
            )
        )

    dropped_count = sum(interval.modern_edge_index is None for interval in intervals)
    receipt = TUEVAnnotationReceipt(
        schema_version=TUEV_REC_SCHEMA_VERSION,
        rec_path=str(rec_path.expanduser().resolve()),
        rec_sha256=_sha256_bytes(payload),
        official_label_map=TUEV_LABEL_MAP,
        official_tcp22_geometry=TUEV_OFFICIAL_TCP22,
        official_to_modern_tcp20=TUEV_OFFICIAL_TO_MODERN_TCP20,
        dropped_official_indices=TUEV_DROPPED_OFFICIAL_INDICES,
        row_count=len(intervals),
        retained_row_count=len(intervals) - dropped_count,
        dropped_row_count=dropped_count,
    )
    return TUEVAnnotation(intervals=tuple(intervals), receipt=receipt)


def _validate_bin_starts(bin_starts_sec: Sequence[float]) -> tuple[float, ...]:
    starts: list[float] = []
    for index, raw_value in enumerate(bin_starts_sec):
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid absolute bin start at index {index}: {raw_value!r}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid absolute bin start at index {index}: {raw_value!r}")
        starts.append(value)
    if not starts:
        raise ValueError("At least one absolute one-second bin is required")
    tolerance = 1e-9
    for previous, current in zip(starts, starts[1:]):
        if current < previous + 1.0 - tolerance:
            raise ValueError(
                "Absolute one-second bins must be strictly ordered and non-overlapping"
            )
    return tuple(starts)


def _complete_single_label(
    intervals: Sequence[TUEVInterval],
    *,
    bin_start: float,
    tolerance: float,
) -> int | None:
    bin_stop = bin_start + 1.0
    overlaps: list[tuple[float, float, int]] = []
    for interval in intervals:
        left = max(bin_start, interval.start_sec)
        right = min(bin_stop, interval.stop_sec)
        if right > left + tolerance:
            overlaps.append((left, right, interval.label_index))
    if not overlaps:
        return None
    label_indices = {label_index for _, _, label_index in overlaps}
    if len(label_indices) != 1:
        # This covers both a clean within-bin transition and an overlapping
        # conflicting annotation.  Neither is a single-label training target.
        return None
    segments = sorted((left, right) for left, right, _ in overlaps)
    if segments[0][0] > bin_start + tolerance:
        return None
    covered_until = segments[0][1]
    for left, right in segments[1:]:
        if left > covered_until + tolerance:
            return None
        covered_until = max(covered_until, right)
    if covered_until < bin_stop - tolerance:
        return None
    return next(iter(label_indices))


def materialize_tuev_ce6(
    annotation: TUEVAnnotation,
    bin_starts_sec: Sequence[float],
    *,
    boundary_tolerance_sec: float = 1e-6,
) -> TUEVCE6Targets:
    """Materialize labels only for fully covered, single-label absolute bins.

    A bin remains masked when it has no interval, partial coverage, an internal
    class transition, or overlapping conflicting labels.  Adjacent/overlapping
    intervals with the same class may jointly provide complete coverage.
    """

    if not math.isfinite(boundary_tolerance_sec) or not 0 <= boundary_tolerance_sec < 0.5:
        raise ValueError("boundary_tolerance_sec must be finite and in [0,0.5)")
    starts = _validate_bin_starts(bin_starts_sec)
    labels = torch.zeros((len(TCP_20_EDGES), len(starts)), dtype=torch.long)
    mask = torch.zeros((len(TCP_20_EDGES), len(starts)), dtype=torch.bool)
    by_edge: list[list[TUEVInterval]] = [[] for _ in TCP_20_EDGES]
    for interval in annotation.intervals:
        if interval.modern_edge_index is not None:
            by_edge[interval.modern_edge_index].append(interval)
    for edge_index, edge_intervals in enumerate(by_edge):
        for bin_index, bin_start in enumerate(starts):
            label_index = _complete_single_label(
                edge_intervals,
                bin_start=bin_start,
                tolerance=float(boundary_tolerance_sec),
            )
            if label_index is not None:
                labels[edge_index, bin_index] = label_index
                mask[edge_index, bin_index] = True
    return TUEVCE6Targets(
        labels=labels,
        mask=mask,
        bin_starts_sec=starts,
        receipt=annotation.receipt,
    )


def load_tuev_ce6(
    rec_path: str | Path,
    bin_starts_sec: Sequence[float],
    *,
    boundary_tolerance_sec: float = 1e-6,
) -> TUEVCE6Targets:
    """Parse a native ``.rec`` and materialize strict TCP20 CE6 targets."""

    return materialize_tuev_ce6(
        parse_tuev_rec(rec_path),
        bin_starts_sec,
        boundary_tolerance_sec=boundary_tolerance_sec,
    )


if len(TUEV_OFFICIAL_TCP22) != 22:
    raise RuntimeError("Frozen TUEV official geometry must contain 22 derivations")
if TUEV_OFFICIAL_TO_MODERN_TCP20.count(None) != 2:
    raise RuntimeError("Exactly A1-T3 and T4-A2 must be dropped")
if sorted(index for index in TUEV_OFFICIAL_TO_MODERN_TCP20 if index is not None) != list(
    range(20)
):
    raise RuntimeError("Retained TUEV derivations must map bijectively to TCP20")


__all__ = [
    "TUEVAnnotation",
    "TUEVAnnotationReceipt",
    "TUEVCE6Targets",
    "TUEVInterval",
    "TUEV_DROPPED_OFFICIAL_INDICES",
    "TUEV_LABEL_MAP",
    "TUEV_OFFICIAL_TCP22",
    "TUEV_OFFICIAL_TO_MODERN_TCP20",
    "TUEV_REC_SCHEMA_VERSION",
    "load_tuev_ce6",
    "materialize_tuev_ce6",
    "parse_tuev_rec",
]
