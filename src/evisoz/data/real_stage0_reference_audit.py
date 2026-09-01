"""Privacy-safe reference/inventory audit for real EviSOZ Stage-0 EDFs.

The audit opens only the target-free event manifest and EDF signal metadata.
It never serializes a patient identifier, source path, raw signal label or
clinical field.  Per-file identities are represented only by content hashes;
all other results are aggregate counts.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from src.clinical_eeg_long_recording.canonical_edf_materialization import (
    _reader_factory as _signal_only_reader_factory,
)
from src.clinical_eeg_long_recording.montage_reference_observability import (
    classify_signal_labels,
)
from src.evisoz.data.artifact_ref import canonical_json_sha256
from src.evisoz.data.channel_registry import build_default_channel_registry
from src.soz.geometry import STANDARD_19, normalize_electrode_name


REAL_STAGE0_REFERENCE_AUDIT_SCHEMA_VERSION = (
    "evisoz_real_stage0_reference_inventory_audit_v1"
)
PARENT_ELECTRODES = (*STANDARD_19, "A1", "A2")
_PLACEHOLDER = "0" * 64
_DISCONTINUOUS_HEADER_READER_POLICY = (
    "evisoz_edf_discontinuous_standard_acquisition_header_only_v1"
)
_REFERENCE_SUFFIXES = (
    "LINKED-EARS",
    "LINKED-EAR",
    "A1A2",
    "M1M2",
    "REF",
    "LE",
    "AR",
    "AVG",
    "AV",
    "CAR",
    "A1",
    "A2",
    "M1",
    "M2",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counter(counter: Counter[object]) -> dict[str, int]:
    return {
        str(key): int(counter[key])
        for key in sorted(counter, key=lambda item: str(item))
    }


def _safe_edf(root: Path, value: object) -> Path:
    relative = PurePosixPath(str(value).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("private EDF manifest contains an unsafe relative path")
    if relative.suffix.lower() != ".edf":
        raise ValueError("private EDF manifest contains a non-EDF source")
    source = root.joinpath(*relative.parts)
    if source.is_symlink():
        raise ValueError("private Stage-0 EDF source must not be a symbolic link")
    resolved = source.resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise ValueError("private Stage-0 EDF source must be a regular file")
    return resolved


def _reference_suffix(label: object) -> str | None:
    text = str(label).strip().upper().replace("_", "-")
    if text.startswith("EEG ") or text.startswith("EEG-"):
        text = text[4:]
    for suffix in _REFERENCE_SUFFIXES:
        if text.endswith(f"-{suffix}"):
            return suffix
    return None


def _parent_inventory(labels: Sequence[object]) -> tuple[dict[str, list[int]], dict[str, str | None]]:
    indices: dict[str, list[int]] = {name: [] for name in PARENT_ELECTRODES}
    references: dict[str, str | None] = {}
    for index, label in enumerate(labels):
        normalized = normalize_electrode_name(label)
        if normalized not in indices:
            continue
        indices[normalized].append(index)
        if len(indices[normalized]) == 1:
            references[normalized] = _reference_suffix(label)
    return indices, references


def _tcp22_edges() -> tuple[tuple[str, str], ...]:
    registry = build_default_channel_registry()
    return tuple(
        (
            str(row["positive_electrode"]["normalized"]),
            str(row["negative_electrode"]["normalized"]),
        )
        for row in registry["tcp22_derivations"]
    )


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Stage-0 source manifest must be a regular non-symlinked file")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "edf_path" not in rows[0]:
        raise ValueError("Stage-0 source manifest is empty or lacks edf_path")
    return rows


def _ascii_number(raw: bytes, *, integer: bool = False) -> float | int:
    try:
        value = raw.decode("ascii", errors="strict").strip()
        return int(value) if integer else float(value)
    except (UnicodeDecodeError, ValueError, OverflowError) as exc:
        raise ValueError("EDF acquisition-header numeric field is invalid") from exc


class _DiscontinuousStandardHeaderReader:
    """Read only signal acquisition fields from a standard EDF+D header.

    The reader deliberately seeks around patient, recording, date/time and
    reserved bytes.  It does not open a signal payload or annotation/TAL row.
    Unknown channel labels remain internal to the aggregate inventory audit.
    """

    canonical_reader_policy = _DISCONTINUOUS_HEADER_READER_POLICY

    def __init__(self, source: Path) -> None:
        with source.open("rb") as stream:
            def read_at(offset: int, width: int) -> bytes:
                stream.seek(offset)
                raw = stream.read(width)
                if len(raw) != width:
                    raise ValueError("EDF acquisition signal header is truncated")
                return raw

            header_bytes = int(_ascii_number(read_at(184, 8), integer=True))
            duration = float(_ascii_number(read_at(244, 8)))
            signal_count = int(_ascii_number(read_at(252, 4), integer=True))
            if (
                not 1 <= signal_count <= 1024
                or duration <= 0
                or header_bytes != 256 * (signal_count + 1)
                or header_bytes > source.stat().st_size
            ):
                raise ValueError("EDF acquisition signal-header framing is invalid")
            self._labels = tuple(
                read_at(256 + index * 16, 16).decode("latin-1").strip()
                for index in range(signal_count)
            )
            self._dimensions = tuple(
                read_at(256 + signal_count * 96 + index * 8, 8)
                .decode("latin-1")
                .strip()
                for index in range(signal_count)
            )
            samples = tuple(
                int(
                    _ascii_number(
                        read_at(256 + signal_count * 216 + index * 8, 8),
                        integer=True,
                    )
                )
                for index in range(signal_count)
            )
        if any(not label for label in self._labels) or any(value <= 0 for value in samples):
            raise ValueError("EDF acquisition signal inventory is invalid")
        self._rates = tuple(value / duration for value in samples)

    def getSignalLabels(self) -> list[str]:
        return list(self._labels)

    def getSampleFrequency(self, index: int) -> float:
        return self._rates[index]

    def getPhysicalDimension(self, index: int) -> str:
        return self._dimensions[index]

    def close(self) -> None:
        return None


def _default_audit_reader(source: Path, source_sha256: str) -> object:
    try:
        return _signal_only_reader_factory(str(source))
    except ValueError as exc:
        if "discontinuous EDF+D" not in str(exc):
            raise
        if len(source_sha256) != 64:
            raise ValueError("EDF source binding hash is invalid")
        return _DiscontinuousStandardHeaderReader(source)


def audit_private_stage0_reference_inventory(
    manifest_path: str | Path,
    eeg_root: str | Path,
    *,
    reader_factory: Callable[[str], object] | None = None,
) -> dict[str, Any]:
    """Audit unique EDF signal headers without emitting direct identifiers."""

    manifest = Path(manifest_path).resolve(strict=True)
    root = Path(eeg_root).resolve(strict=True)
    rows = _read_manifest(manifest)
    sources = {_safe_edf(root, row["edf_path"]) for row in rows}
    factory = _signal_only_reader_factory if reader_factory is None else reader_factory
    edges = _tcp22_edges()

    montage_classes: Counter[object] = Counter()
    reason_codes: Counter[object] = Counter()
    signal_counts: Counter[object] = Counter()
    standard19_counts: Counter[object] = Counter()
    parent_counts: Counter[object] = Counter()
    aux_profiles: Counter[object] = Counter()
    tcp22_edge_counts: Counter[object] = Counter()
    sampling_rates: Counter[object] = Counter()
    selected_units: Counter[object] = Counter()
    reader_policies: Counter[object] = Counter()
    inventory_hash_counts: Counter[object] = Counter()
    opaque_candidates = 0
    explicit_candidates = 0
    complete_standard19 = 0
    mixed_selected_clock = 0
    source_hashes: list[str] = []

    for source in sorted(sources, key=lambda item: str(item)):
        source_hash = _sha256_file(source)
        source_hashes.append(source_hash)
        reader = (
            _default_audit_reader(source, source_hash)
            if reader_factory is None
            else factory(str(source))
        )
        try:
            labels = tuple(str(value).strip() for value in reader.getSignalLabels())
            classified = classify_signal_labels(labels)
            inventory_hash = str(classified["signal_labels_sha256"])
            inventory_hash_counts[inventory_hash] += 1
            montage_class = str(classified["montage_class"])
            montage_classes[montage_class] += 1
            for reason in classified["classification_reason_codes"]:
                reason_codes[str(reason)] += 1
            signal_counts[len(labels)] += 1
            reader_policies[
                str(getattr(reader, "canonical_reader_policy", type(reader).__name__))
            ] += 1

            indices, references = _parent_inventory(labels)
            duplicate = {name for name, values in indices.items() if len(values) > 1}
            observed = {name for name, values in indices.items() if len(values) == 1}
            standard_observed = observed.intersection(STANDARD_19)
            standard19_counts[len(standard_observed)] += 1
            parent_counts[len(observed)] += 1
            if len(standard_observed) == len(STANDARD_19):
                complete_standard19 += 1
            has_a1 = "A1" in observed
            has_a2 = "A2" in observed
            aux_profiles[
                "both" if has_a1 and has_a2 else "a1_only" if has_a1 else "a2_only" if has_a2 else "none"
            ] += 1
            tcp22_edge_counts[
                sum(left in observed and right in observed for left, right in edges)
            ] += 1

            selected_indices = [indices[name][0] for name in PARENT_ELECTRODES if name in observed]
            rates = [float(reader.getSampleFrequency(index)) for index in selected_indices]
            if rates and max(rates) - min(rates) <= 1e-9:
                sampling_rates[format(rates[0], ".12g")] += 1
            else:
                mixed_selected_clock += 1
            for index in selected_indices:
                selected_units[str(reader.getPhysicalDimension(index)).strip()] += 1

            exact_reference_tokens = {
                references[name] for name in observed if references.get(name) is not None
            }
            explicit_parent = (
                classified["common_reference_compatible"] is True
                and not duplicate
                and bool(observed)
                and len(exact_reference_tokens) == 1
                and all(references.get(name) in exact_reference_tokens for name in observed)
            )
            opaque_parent = (
                montage_class == "unknown"
                and set(classified["classification_reason_codes"])
                == {"direct_electrode_reference_token_unobservable"}
                and not duplicate
                and len(standard_observed) == len(STANDARD_19)
                and all(references.get(name) is None for name in observed)
            )
            explicit_candidates += int(explicit_parent)
            opaque_candidates += int(opaque_parent)
        finally:
            if hasattr(reader, "close"):
                reader.close()

    source_hashes = sorted(source_hashes)
    body: dict[str, Any] = {
        "schema_version": REAL_STAGE0_REFERENCE_AUDIT_SCHEMA_VERSION,
        "status": "completed_signal_header_inventory_audit",
        "source_manifest_sha256": _sha256_file(manifest),
        "manifest_event_row_count": len(rows),
        "unique_edf_count": len(sources),
        "source_edf_sha256_roster": source_hashes,
        "source_edf_sha256_roster_sha256": canonical_json_sha256(
            {"domain": "evisoz-private-stage0-edf-roster-v1", "sha256": source_hashes}
        ),
        "aggregate": {
            "montage_class_counts": _counter(montage_classes),
            "classification_reason_code_counts": _counter(reason_codes),
            "signal_count_distribution": _counter(signal_counts),
            "standard19_observed_count_distribution": _counter(standard19_counts),
            "parent_electrode_observed_count_distribution": _counter(parent_counts),
            "auxiliary_coverage_counts": _counter(aux_profiles),
            "derivable_tcp22_edge_count_distribution": _counter(tcp22_edge_counts),
            "selected_parent_sampling_rate_hz_counts": _counter(sampling_rates),
            "selected_parent_physical_unit_counts": _counter(selected_units),
            "reader_policy_counts": _counter(reader_policies),
            "ordered_signal_inventory_sha256_counts": _counter(inventory_hash_counts),
            "complete_standard19_edf_count": complete_standard19,
            "explicit_common_reference_candidate_edf_count": explicit_candidates,
            "opaque_common_reference_candidate_edf_count": opaque_candidates,
            "mixed_selected_sampling_clock_edf_count": mixed_selected_clock,
            "discontinuous_header_only_fallback_edf_count": int(
                reader_policies.get(_DISCONTINUOUS_HEADER_READER_POLICY, 0)
            ),
        },
        "interpretation": {
            "explicit_candidate_semantics": (
                "one shared reference token is observable in every selected direct endpoint"
            ),
            "opaque_candidate_semantics": (
                "complete unique suffix-free Standard19 field; shared physical reference is not proven by labels"
            ),
            "tcp22_count_semantics": (
                "arithmetic endpoint support only; evidence release still requires reference authority"
            ),
            "header_audit_alone_authorizes_opaque_common_reference": False,
            "discontinuous_header_fallback_authorizes_event_clock": False,
        },
        "access_receipt": {
            "source_manifest_opened": True,
            "edf_signal_labels_used": True,
            "edf_signal_sampling_rates_used": True,
            "edf_signal_units_used": True,
            "eeg_samples_used": False,
            "edf_patient_or_recording_header_api_called": False,
            "edf_annotation_api_called": False,
            "clinical_label_fields_serialized": False,
            "patient_or_event_identifiers_serialized": False,
            "source_paths_or_raw_signal_labels_serialized": False,
        },
        "receipt_sha256": _PLACEHOLDER,
    }
    body["receipt_sha256"] = canonical_json_sha256(body)
    return body


__all__ = [
    "PARENT_ELECTRODES",
    "REAL_STAGE0_REFERENCE_AUDIT_SCHEMA_VERSION",
    "audit_private_stage0_reference_inventory",
]
