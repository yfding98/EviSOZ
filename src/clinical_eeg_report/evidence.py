"""Fail-closed EEG waveform attachments for clinical EEG reports.

The attachment manifest is a presentation-evidence boundary, not another
source of clinical facts.  Every figure must therefore be selected by an EEG
``evidence_id`` that is already present in a validated
``clinical_eeg_report_v1`` ledger.  The language model is deliberately absent
from this module and cannot choose a figure, event, or crop.

Only a validated :class:`~src.clinical_eeg_report.schema.ClinicalEEGReport`
may be supplied.  Paths are resolved below a caller-selected manifest root,
symbolic links are rejected, PNG structure and checksums are verified, and
the returned objects are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
from typing import Any, Mapping, Sequence
import zlib

from .schema import ClinicalEEGReport, FactSection


EVIDENCE_MANIFEST_SCHEMA_VERSION = "clinical_eeg_waveform_manifest_v1"
WAVEFORM_SELECTION_POLICY = "report_fact_evidence_ids_only_no_llm"
WAVEFORM_REFERENCE = "common_average_standard19"

# Kept local so importing the clinical reporting package does not import the
# research model stack (``src.soz.geometry`` imports torch).  This tuple is the
# exact STANDARD_19 order used by the SOZ signal processor.
STANDARD_19_CHANNEL_ORDER: tuple[str, ...] = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "report_id",
        "patient_pseudonym",
        "selection_policy",
        "attachments",
    }
)
_ATTACHMENT_KEYS = frozenset(
    {
        "evidence_id",
        "fact_ids",
        "eeg_event_id",
        "figure_file",
        "figure_sha256",
        "source_signal_sha256",
        "preprocessing_receipt_sha256",
        "processed_window_sha256",
        "channel_order",
        "sampling_rate_hz",
        "filter_hz",
        "reference",
        "event_window_seconds",
        "event_anchor_offset_seconds",
        "representative_event",
        "caption_zh",
    }
)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_PNG_BYTES = 64 * 1024 * 1024
_MAX_PNG_DIMENSION = 32_768
_MAX_PNG_PIXELS = 100_000_000
_MAX_WINDOW_SECONDS = 24 * 60 * 60
_ANCHOR_TOLERANCE_SECONDS = 1e-6


@dataclass(frozen=True)
class ValidatedWaveformAttachment:
    """One immutable, report-bound and file-verified waveform attachment."""

    evidence_id: str
    fact_ids: tuple[str, ...]
    eeg_event_id: str
    figure_file: str
    figure_sha256: str
    source_signal_sha256: str
    preprocessing_receipt_sha256: str
    processed_window_sha256: str
    channel_order: tuple[str, ...]
    sampling_rate_hz: float
    filter_hz: tuple[float, float]
    reference: str
    event_window_seconds: tuple[float, float]
    event_anchor_offset_seconds: float
    representative_event: bool
    caption_zh: str
    source_path: Path
    image_width_px: int
    image_height_px: int

    def to_dict(self) -> dict[str, object]:
        """Return the portable manifest fields (never the host source path)."""

        return {
            "evidence_id": self.evidence_id,
            "fact_ids": list(self.fact_ids),
            "eeg_event_id": self.eeg_event_id,
            "figure_file": self.figure_file,
            "figure_sha256": self.figure_sha256,
            "source_signal_sha256": self.source_signal_sha256,
            "preprocessing_receipt_sha256": self.preprocessing_receipt_sha256,
            "processed_window_sha256": self.processed_window_sha256,
            "channel_order": list(self.channel_order),
            "sampling_rate_hz": self.sampling_rate_hz,
            "filter_hz": list(self.filter_hz),
            "reference": self.reference,
            "event_window_seconds": list(self.event_window_seconds),
            "event_anchor_offset_seconds": self.event_anchor_offset_seconds,
            "representative_event": self.representative_event,
            "caption_zh": self.caption_zh,
        }


@dataclass(frozen=True)
class ValidatedWaveformManifest:
    """An immutable waveform manifest bound to one validated report."""

    schema_version: str
    report_id: str
    patient_pseudonym: str
    selection_policy: str
    attachments: tuple[ValidatedWaveformAttachment, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "patient_pseudonym": self.patient_pseudonym,
            "selection_policy": self.selection_policy,
            "attachments": [attachment.to_dict() for attachment in self.attachments],
        }


def _strict_dict(value: object, required: frozenset[str], context: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    result = value
    keys = set(result)
    missing = required.difference(keys)
    extra = keys.difference(required)
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return result


def _string(value: object, context: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty, trimmed string")
    if len(value) > maximum:
        raise ValueError(f"{context} must be at most {maximum} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{context} contains control characters")
    return value


def _identifier(value: object, context: str, *, source: bool = False) -> str:
    result = _string(value, context, maximum=128 if source else 64)
    pattern = _SOURCE_ID_RE if source else _ID_RE
    if pattern.fullmatch(result) is None:
        raise ValueError(f"{context} has an invalid identifier: {result!r}")
    return result


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty, trimmed string")
    result = value
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return result


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _number_pair(value: object, context: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{context} must be a two-number array")
    return (
        _finite_number(value[0], f"{context}[0]"),
        _finite_number(value[1], f"{context}[1]"),
    )


def _identifier_list(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be a non-empty identifier array")
    result = tuple(
        _identifier(item, f"{context}[{index}]") for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicates")
    return result


def _safe_relative_png(value: object) -> tuple[str, tuple[str, ...]]:
    text = _string(value, "attachment.figure_file", maximum=512)
    if "\\" in text or ":" in text or text.startswith("/"):
        raise ValueError("attachment.figure_file must be a safe POSIX relative path")
    raw_parts = text.split("/")
    if any(
        part in {"", ".", ".."} or _PATH_SEGMENT_RE.fullmatch(part) is None
        for part in raw_parts
    ):
        raise ValueError("attachment.figure_file contains an unsafe path segment")
    relative = PurePosixPath(text)
    if relative.is_absolute() or relative.as_posix() != text:
        raise ValueError("attachment.figure_file must be a canonical relative path")
    if relative.suffix != ".png":
        raise ValueError("attachment.figure_file must use the lowercase .png suffix")
    return text, tuple(raw_parts)


def _read_regular_file_no_follow(path: Path, *, maximum_bytes: int, context: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{context} must be a regular file")
        if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
            raise ValueError(
                f"{context} size must be between 1 and {maximum_bytes} bytes"
            )
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{context} changed while it was being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{context} changed while it was being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _resolve_png(root: Path, parts: Sequence[str]) -> tuple[Path, bytes]:
    if root.is_symlink():
        raise ValueError("manifest root must not be a symbolic link")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"manifest root does not exist: {root}") from exc
    if not resolved_root.is_dir():
        raise NotADirectoryError(resolved_root)

    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("attachment.figure_file must not traverse a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"waveform figure does not exist: {candidate}") from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("attachment.figure_file resolves outside the manifest root") from exc
    content = _read_regular_file_no_follow(
        resolved,
        maximum_bytes=_MAX_PNG_BYTES,
        context="waveform PNG",
    )
    return resolved, content


def _validate_png(content: bytes, context: str) -> tuple[int, int]:
    if not content.startswith(_PNG_SIGNATURE):
        raise ValueError(f"{context} has an invalid PNG signature")
    offset = len(_PNG_SIGNATURE)
    chunk_index = 0
    seen_ihdr = False
    seen_idat = False
    seen_iend = False
    image_dimensions: tuple[int, int] | None = None
    while offset < len(content):
        if len(content) - offset < 12:
            raise ValueError(f"{context} contains a truncated PNG chunk")
        length = struct.unpack(">I", content[offset : offset + 4])[0]
        chunk_type = content[offset + 4 : offset + 8]
        if not all(
            ord("A") <= character <= ord("Z")
            or ord("a") <= character <= ord("z")
            for character in chunk_type
        ):
            raise ValueError(f"{context} contains an invalid PNG chunk type")
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(content):
            raise ValueError(f"{context} contains a truncated PNG chunk")
        expected_crc = struct.unpack(">I", content[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(content[data_start:data_end], actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"{context} contains a PNG chunk with an invalid CRC")

        if chunk_index == 0 and chunk_type != b"IHDR":
            raise ValueError(f"{context} must begin with an IHDR chunk")
        if chunk_type == b"IHDR":
            if seen_ihdr or chunk_index != 0 or length != 13:
                raise ValueError(f"{context} has an invalid IHDR chunk")
            seen_ihdr = True
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", content[data_start:data_end])
            )
            if (
                width == 0
                or height == 0
                or width > _MAX_PNG_DIMENSION
                or height > _MAX_PNG_DIMENSION
                or width * height > _MAX_PNG_PIXELS
            ):
                raise ValueError(f"{context} has unreasonable IHDR dimensions")
            image_dimensions = (width, height)
            valid_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if color_type not in valid_depths or bit_depth not in valid_depths[color_type]:
                raise ValueError(f"{context} has an invalid IHDR color/bit-depth pair")
            if compression != 0 or filtering != 0 or interlace not in {0, 1}:
                raise ValueError(f"{context} has unsupported IHDR encoding fields")
        elif chunk_type == b"IDAT":
            if not seen_ihdr or seen_iend:
                raise ValueError(f"{context} has an out-of-order IDAT chunk")
            seen_idat = True
        elif chunk_type == b"IEND":
            if length != 0 or not seen_idat or seen_iend:
                raise ValueError(f"{context} has an invalid IEND chunk")
            seen_iend = True
            if crc_end != len(content):
                raise ValueError(f"{context} contains trailing bytes after IEND")
        elif seen_iend:
            raise ValueError(f"{context} contains a chunk after IEND")
        offset = crc_end
        chunk_index += 1

    if not seen_ihdr or not seen_idat or not seen_iend:
        raise ValueError(f"{context} is not a complete PNG image")
    if image_dimensions is None:  # defensive; ``seen_ihdr`` already proves it
        raise ValueError(f"{context} has no PNG dimensions")
    return image_dimensions


def _caption(value: object, context: str) -> str:
    result = _string(value, context, maximum=1000)
    if _HAN_RE.search(result) is None:
        raise ValueError(f"{context} must contain Chinese text")
    return result


def _validate_attachment(
    raw: object,
    *,
    index: int,
    report: ClinicalEEGReport,
    root: Path,
    report_facts_by_id: Mapping[str, object],
) -> ValidatedWaveformAttachment:
    context = f"attachments[{index}]"
    data = _strict_dict(raw, _ATTACHMENT_KEYS, context)
    evidence_id = _identifier(data["evidence_id"], f"{context}.evidence_id", source=True)
    fact_ids = _identifier_list(data["fact_ids"], f"{context}.fact_ids")
    event_id = _identifier(data["eeg_event_id"], f"{context}.eeg_event_id")
    if event_id not in report.eeg_event_ids:
        raise ValueError(f"{context}.eeg_event_id is not present in the report")

    declared_facts = []
    for fact_id in fact_ids:
        fact = report_facts_by_id.get(fact_id)
        if fact is None:
            raise ValueError(f"{context}.fact_ids references unknown fact {fact_id!r}")
        declared_facts.append(fact)
    for fact in declared_facts:
        if fact.section is not FactSection.ICTAL or fact.eeg_event_id != event_id:
            raise ValueError(
                f"{context}.fact_ids crosses an EEG event or references a non-ictal fact"
            )
        if evidence_id not in fact.evidence_ids:
            raise ValueError(
                f"{context}.evidence_id is not referenced by every declared fact"
            )

    actual_fact_ids = tuple(
        fact.fact_id for fact in report.facts if evidence_id in fact.evidence_ids
    )
    if not actual_fact_ids:
        raise ValueError(f"{context}.evidence_id is not referenced by the report")
    if set(actual_fact_ids) != set(fact_ids):
        raise ValueError(
            f"{context}.fact_ids must exactly identify every report fact that "
            "references its evidence_id"
        )

    figure_file, path_parts = _safe_relative_png(data["figure_file"])
    source_path, png = _resolve_png(root, path_parts)
    image_width_px, image_height_px = _validate_png(
        png, f"{context}.figure_file"
    )
    figure_sha256 = _sha256(data["figure_sha256"], f"{context}.figure_sha256")
    actual_figure_sha256 = hashlib.sha256(png).hexdigest()
    if figure_sha256 != actual_figure_sha256:
        raise ValueError(f"{context}.figure_sha256 does not match the PNG file")

    channel_order_raw = data["channel_order"]
    if not isinstance(channel_order_raw, list) or tuple(channel_order_raw) != STANDARD_19_CHANNEL_ORDER:
        raise ValueError(
            f"{context}.channel_order must exactly match the canonical STANDARD_19 order"
        )
    sampling_rate = _finite_number(data["sampling_rate_hz"], f"{context}.sampling_rate_hz")
    if sampling_rate <= 0 or sampling_rate > 100_000:
        raise ValueError(f"{context}.sampling_rate_hz is outside the supported range")
    filter_hz = _number_pair(data["filter_hz"], f"{context}.filter_hz")
    if filter_hz[0] < 0 or filter_hz[1] <= filter_hz[0]:
        raise ValueError(f"{context}.filter_hz must be an increasing non-negative band")
    if filter_hz[1] > sampling_rate / 2:
        raise ValueError(f"{context}.filter_hz exceeds the Nyquist frequency")
    reference = _string(data["reference"], f"{context}.reference")
    if reference != WAVEFORM_REFERENCE:
        raise ValueError(f"{context}.reference must be {WAVEFORM_REFERENCE!r}")
    event_window = _number_pair(
        data["event_window_seconds"], f"{context}.event_window_seconds"
    )
    if event_window[1] <= event_window[0]:
        raise ValueError(f"{context}.event_window_seconds must be increasing")
    if event_window[0] > 0 or event_window[1] < 0:
        raise ValueError(
            f"{context}.event_window_seconds must be relative to and include "
            "the zero-second event anchor"
        )
    if event_window[1] - event_window[0] > _MAX_WINDOW_SECONDS:
        raise ValueError(f"{context}.event_window_seconds is unreasonably long")
    anchor = _finite_number(
        data["event_anchor_offset_seconds"],
        f"{context}.event_anchor_offset_seconds",
    )
    if anchor < 0 or anchor > event_window[1] - event_window[0]:
        raise ValueError(
            f"{context}.event_anchor_offset_seconds must lie inside the waveform segment"
        )
    if abs(anchor + event_window[0]) > _ANCHOR_TOLERANCE_SECONDS:
        raise ValueError(
            f"{context}.event_anchor_offset_seconds must equal the time from "
            "the segment start to the zero-second event anchor"
        )
    representative = data["representative_event"]
    if type(representative) is not bool:
        raise TypeError(f"{context}.representative_event must be a boolean")

    return ValidatedWaveformAttachment(
        evidence_id=evidence_id,
        fact_ids=fact_ids,
        eeg_event_id=event_id,
        figure_file=figure_file,
        figure_sha256=figure_sha256,
        source_signal_sha256=_sha256(
            data["source_signal_sha256"], f"{context}.source_signal_sha256"
        ),
        preprocessing_receipt_sha256=_sha256(
            data["preprocessing_receipt_sha256"],
            f"{context}.preprocessing_receipt_sha256",
        ),
        processed_window_sha256=_sha256(
            data["processed_window_sha256"],
            f"{context}.processed_window_sha256",
        ),
        channel_order=STANDARD_19_CHANNEL_ORDER,
        sampling_rate_hz=sampling_rate,
        filter_hz=filter_hz,
        reference=reference,
        event_window_seconds=event_window,
        event_anchor_offset_seconds=anchor,
        representative_event=representative,
        caption_zh=_caption(data["caption_zh"], f"{context}.caption_zh"),
        source_path=source_path,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
    )


def validate_waveform_manifest_payload(
    payload: object,
    report: ClinicalEEGReport,
    *,
    manifest_root: str | Path,
) -> ValidatedWaveformManifest:
    """Validate an untrusted waveform manifest against one validated report.

    ``manifest_root`` is the only directory from which figure files may be
    resolved.  The returned ``source_path`` values are absolute paths whose
    content and declared SHA-256 were verified during this call.
    """

    if not isinstance(report, ClinicalEEGReport):
        raise TypeError("report must be a validated ClinicalEEGReport")
    data = _strict_dict(payload, _TOP_LEVEL_KEYS, "waveform manifest")
    schema_version = _string(data["schema_version"], "manifest.schema_version")
    if schema_version != EVIDENCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest.schema_version must be {EVIDENCE_MANIFEST_SCHEMA_VERSION!r}"
        )
    report_id = _identifier(data["report_id"], "manifest.report_id", source=True)
    patient = _identifier(
        data["patient_pseudonym"], "manifest.patient_pseudonym", source=True
    )
    if report_id != report.report_id:
        raise ValueError("waveform manifest report_id does not match the report")
    if patient != report.patient_pseudonym:
        raise ValueError("waveform manifest patient_pseudonym does not match the report")
    policy = _string(data["selection_policy"], "manifest.selection_policy")
    if policy != WAVEFORM_SELECTION_POLICY:
        raise ValueError(
            f"manifest.selection_policy must be {WAVEFORM_SELECTION_POLICY!r}"
        )
    raw_attachments = data["attachments"]
    if not isinstance(raw_attachments, list) or not raw_attachments:
        raise TypeError("manifest.attachments must be a non-empty array")
    root = Path(manifest_root)
    facts_by_id = {fact.fact_id: fact for fact in report.facts}
    attachments = tuple(
        _validate_attachment(
            raw,
            index=index,
            report=report,
            root=root,
            report_facts_by_id=facts_by_id,
        )
        for index, raw in enumerate(raw_attachments)
    )

    evidence_ids = [attachment.evidence_id for attachment in attachments]
    figure_files = [attachment.figure_file for attachment in attachments]
    processed_hashes = [attachment.processed_window_sha256 for attachment in attachments]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("manifest.attachments contains duplicate evidence_id values")
    if len(figure_files) != len(set(figure_files)):
        raise ValueError("manifest.attachments contains duplicate figure_file values")
    if len(processed_hashes) != len(set(processed_hashes)):
        raise ValueError(
            "manifest.attachments contains duplicate processed_window_sha256 values"
        )

    return ValidatedWaveformManifest(
        schema_version=schema_version,
        report_id=report_id,
        patient_pseudonym=patient,
        selection_policy=policy,
        attachments=attachments,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"waveform manifest JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"waveform manifest JSON contains invalid constant {value!r}")


def load_waveform_manifest(
    manifest_path: str | Path,
    report: ClinicalEEGReport,
) -> ValidatedWaveformManifest:
    """Strictly read and validate a JSON manifest and its sibling figures."""

    path = Path(manifest_path)
    if path.suffix != ".json":
        raise ValueError("waveform manifest path must use the lowercase .json suffix")
    if path.is_symlink():
        raise ValueError("waveform manifest path must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(path) from exc
    content = _read_regular_file_no_follow(
        resolved,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        context="waveform manifest JSON",
    )
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("waveform manifest JSON must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("waveform manifest is not valid JSON") from exc
    return validate_waveform_manifest_payload(
        payload,
        report,
        manifest_root=resolved.parent,
    )


__all__ = [
    "EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "WAVEFORM_SELECTION_POLICY",
    "WAVEFORM_REFERENCE",
    "STANDARD_19_CHANNEL_ORDER",
    "ValidatedWaveformAttachment",
    "ValidatedWaveformManifest",
    "validate_waveform_manifest_payload",
    "load_waveform_manifest",
]
