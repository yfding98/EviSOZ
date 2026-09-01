"""Strict, identity-free EDF acquisition-header projection.

The parser in this module is deliberately *not* a general EDF reader.  It
seeks to an explicit allowlist of fixed-width acquisition fields and returns
only EEG signal acquisition metadata.  In particular it never reads the EDF
patient/recording identity fields, start date/time, reserved/free-text fields,
transducer text, raw prefilter text, sample payload, or EDF+ annotation
payload.  ``EDF Annotations`` labels are read only to identify and exclude the
corresponding signal-header row; no annotation-channel metadata is emitted.

Raw EDF prefilter fields are free text.  V1 therefore reports acquisition
high/low-pass cutoffs as ``not_evaluable`` instead of attempting a permissive
numeric extraction that could copy arbitrary text into a receipt.

This is an acquisition metadata/software receipt.  It is not a seizure,
Finding, SOZ, or clinical claim.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Final, Mapping


EEG_ACQUISITION_HEADER_ALLOWLIST_V1_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_identity_free_acquisition_header_allowlist_v1"
EEG_ACQUISITION_HEADER_RECEIPT_V1_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_identity_free_acquisition_header_receipt_v1"
EEG_ACQUISITION_HEADER_PARSER_ID_V1: Final[
    str
] = "edf_exact_byte_range_identity_free_eeg_acquisition_parser_v1"

_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")

# Offsets in the 256-byte EDF fixed header.  No other fixed-header byte range
# is opened by this parser.
_FIXED_ALLOWED: Final[dict[str, tuple[int, int]]] = {
    "header_bytes": (184, 8),
    "data_record_count": (236, 8),
    "data_record_duration_seconds": (244, 8),
    "declared_signal_count": (252, 4),
}
_FIXED_FORBIDDEN: Final[dict[str, tuple[int, int]]] = {
    "version": (0, 8),
    "patient_identity": (8, 80),
    "recording_identity_or_free_text": (88, 80),
    "start_date": (168, 8),
    "start_time": (176, 8),
    "reserved_or_edf_subtype_free_text": (192, 44),
}

# EDF signal-header fields are stored field-major.  Offsets are multiples of
# the declared signal count.  Only labels are read for every declared signal;
# all remaining allowed fields are read only for channels classified as EEG.
_SIGNAL_FIELD_LAYOUT: Final[dict[str, tuple[int, int]]] = {
    "channel_label": (0, 16),
    "transducer_free_text": (16, 80),
    "physical_dimension": (96, 8),
    "physical_minimum": (104, 8),
    "physical_maximum": (112, 8),
    "digital_minimum": (120, 8),
    "digital_maximum": (128, 8),
    "prefilter_free_text": (136, 80),
    "samples_per_data_record": (216, 8),
    "reserved_free_text": (224, 32),
}
_SIGNAL_ALLOWED: Final[tuple[str, ...]] = (
    "channel_label",
    "physical_dimension",
    "physical_minimum",
    "physical_maximum",
    "digital_minimum",
    "digital_maximum",
    "samples_per_data_record",
)
_SIGNAL_FORBIDDEN: Final[tuple[str, ...]] = (
    "transducer_free_text",
    "prefilter_free_text",
    "reserved_free_text",
)

_ANNOTATION_LABELS: Final[frozenset[str]] = frozenset(
    {"edf annotations", "bdf annotations"}
)
_AUXILIARY_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[\s_\-/])(?:(?:ECG|EKG|EOG|EMG|RESP)[0-9]*|"
    r"ROC|LOC|LUC|RLC|AIRFLOW|PLETH|PULSE|SAO2|SPO2|PHOTIC|"
    r"TRIGGER|EVENT|DC|IBI|BURSTS?|SUPPR)(?:$|[\s_\-/])",
    flags=re.IGNORECASE,
)
_SCALP_ELECTRODE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "FP1", "FPZ", "FP2",
        "AF7", "AF3", "AFZ", "AF4", "AF8",
        "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
        "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
        "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
        "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
        "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
        "PO9", "PO7", "PO3", "POZ", "PO4", "PO8", "PO10",
        "O1", "OZ", "O2", "IZ",
        # Legacy 10--20 temporal and auricular/mastoid aliases retained as
        # acquisition identities; later canonicalization remains explicit.
        "T1", "T2", "T3", "T4", "T5", "T6", "A1", "A2", "M1", "M2",
    }
)
_REFERENCE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {"REF", "LE", "AR", "AVG", "AV", "A1", "A2", "M1", "M2"}
)
_REGISTERED_NON_TARGET_EEG_TOKENS: Final[frozenset[str]] = frozenset(
    {"SP1", "SP2", "PG1", "PG2", "C3P", "C4P"}
)
_NUMBERED_NON_TARGET_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]{1,3}$")
_PHYSICAL_DIMENSION_CANONICAL: Final[dict[str, str]] = {
    "v": "V",
    "mv": "mV",
    "uv": "uV",
}

_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "parser_id",
        "policy_id",
        "policy_receipt_sha256",
        "parser_source_sha256",
        "external_source_binding_sha256",
        "container_size_bytes",
        "header_bytes",
        "data_record_count",
        "data_record_duration_seconds_fraction",
        "recording_duration_seconds_fraction",
        "declared_signal_count",
        "eeg_signal_count",
        "excluded_annotation_channel_count",
        "excluded_auxiliary_channel_count",
        "excluded_non_target_signal_count",
        "channels",
        "acquisition_filter_cutoffs",
        "allowlisted_header_bytes_sha256",
        "byte_access_ledger",
        "byte_access_ledger_sha256",
        "scope_receipt",
        "receipt_sha256",
    }
)
_CHANNEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "channel_index",
        "channel_label",
        "physical_dimension",
        "physical_minimum_fraction",
        "physical_maximum_fraction",
        "digital_minimum",
        "digital_maximum",
        "samples_per_data_record",
        "sampling_rate_hz_fraction",
        "sample_count",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquisition_header_parser_source_sha256_v1() -> str:
    """Return the byte hash of this parser implementation."""

    return _file_sha256(Path(__file__).resolve(strict=True))


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _strict_ascii(raw: bytes, context: str) -> str:
    try:
        decoded = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not strict ASCII") from error
    if "\x00" in decoded or any(ord(character) < 32 for character in decoded):
        raise ValueError(f"{context} contains a control character")
    return decoded.strip()


def _integer(raw: bytes, context: str, *, positive: bool) -> int:
    text = _strict_ascii(raw, context)
    if not re.fullmatch(r"[+-]?[0-9]+", text):
        raise ValueError(f"{context} is not an EDF integer")
    result = int(text)
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context} must be {qualifier}")
    return result


def _fraction(raw: bytes, context: str, *, positive: bool) -> Fraction:
    text = _strict_ascii(raw, context)
    try:
        decimal = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"{context} is not an EDF decimal") from error
    if not decimal.is_finite():
        raise ValueError(f"{context} must be finite")
    result = Fraction(decimal)
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context} must be {qualifier}")
    return result


def _fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _normalize_label(value: str) -> str:
    return " ".join(value.split())


def _looks_like_eeg_label(label: str) -> bool:
    """Accept only a one-node or two-node EEG voltage label.

    ``EEG`` is an optional modality prefix, never an authorization shortcut.
    After removing it, a label must be either one scalp electrode, one scalp
    electrode plus one registered reference suffix, or two scalp electrodes.
    This deliberately rejects free text such as ``EEG CLINICAL NOTE`` and
    malformed multi-reference strings such as ``F7-REF-LE``.
    """

    upper = _normalize_label(label).upper()
    if upper == "EEG":
        return False
    if upper.startswith("EEG "):
        upper = upper[4:].strip()
    compact = upper.replace(" ", "")
    if not compact:
        return False
    parts = compact.split("-")
    if any(not part for part in parts) or len(parts) not in {1, 2}:
        return False
    if parts[0] not in _SCALP_ELECTRODE_TOKENS:
        return False
    if len(parts) == 1:
        return True
    return bool(
        parts[1] in _REFERENCE_SUFFIXES
        or parts[1] in _SCALP_ELECTRODE_TOKENS
    )


def _looks_like_registered_non_target_signal(label: str) -> bool:
    upper = _normalize_label(label).upper()
    if upper.startswith("EEG "):
        upper = upper[4:].strip()
    parts = upper.replace(" ", "").split("-")
    if len(parts) not in {1, 2} or any(not part for part in parts):
        return False
    if parts[0] not in _REGISTERED_NON_TARGET_EEG_TOKENS and (
        _NUMBERED_NON_TARGET_TOKEN_RE.fullmatch(parts[0]) is None
    ):
        return False
    return len(parts) == 1 or parts[1] in _REFERENCE_SUFFIXES


def _physical_dimension(raw: bytes, context: str) -> str:
    text = _strict_ascii(raw, context)
    compact = text.replace(" ", "").casefold()
    canonical = _PHYSICAL_DIMENSION_CANONICAL.get(compact)
    if canonical is None:
        raise ValueError(f"{context} is outside the V/mV/uV unit allowlist")
    return canonical


def _fraction_from_json(value: object, context: str) -> Fraction:
    if (
        type(value) is not list
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
    ):
        raise ValueError(f"{context} must be an exact [numerator, denominator]")
    return Fraction(value[0], value[1])


def _channel_kind(label: str) -> str:
    normalized = _normalize_label(label)
    if normalized.casefold() in _ANNOTATION_LABELS:
        return "annotation"
    if _AUXILIARY_TOKEN_RE.search(normalized):
        return "auxiliary"
    if _looks_like_registered_non_target_signal(normalized):
        return "non_target"
    if _looks_like_eeg_label(normalized):
        return "eeg"
    return "unknown"


def build_eeg_acquisition_header_allowlist_policy_v1() -> dict[str, Any]:
    """Build the source-independent allowlist policy body."""

    policy: dict[str, Any] = {
        "schema_version": EEG_ACQUISITION_HEADER_ALLOWLIST_V1_SCHEMA_VERSION,
        "policy_id": "EEG-ACQUISITION-HEADER-ALLOWLIST-V1-20260824",
        "status": "additive_parser_policy_frozen_no_dataset_headers_opened",
        "parser_id": EEG_ACQUISITION_HEADER_PARSER_ID_V1,
        "allowed_fixed_header_fields": {
            key: {"offset_bytes": value[0], "width_bytes": value[1]}
            for key, value in _FIXED_ALLOWED.items()
        },
        "allowed_signal_header_fields": list(_SIGNAL_ALLOWED),
        "forbidden_fixed_header_fields": list(_FIXED_FORBIDDEN),
        "forbidden_signal_header_fields": list(_SIGNAL_FORBIDDEN),
        "channel_scope": {
            "output_modality": "scalp_EEG_only",
            "annotation_labels_excluded": sorted(_ANNOTATION_LABELS),
            "auxiliary_channels_excluded": True,
            "registered_non_target_signals_excluded": True,
            "registered_non_target_signal_tokens": sorted(
                _REGISTERED_NON_TARGET_EEG_TOKENS
            ),
            "numbered_non_target_signal_range": "one_to_three_ASCII_digits",
            "optional_EEG_prefix_is_not_sufficient_for_admission": True,
            "admitted_label_grammar": (
                "optional_EEG_prefix_then_one_scalp_electrode_optionally_followed_"
                "by_one_registered_reference_or_second_scalp_electrode"
            ),
            "scalp_electrode_ontology_id": "frozen_international_10_10_plus_legacy_temporal_v1",
            "scalp_electrode_token_count": len(_SCALP_ELECTRODE_TOKENS),
            "scalp_electrode_roster_sha256": _canonical_sha256(
                sorted(_SCALP_ELECTRODE_TOKENS)
            ),
            "unknown_channel_label_behavior": "fail_closed",
            "annotation_channel_header_fields_emitted": False,
            "annotation_channel_payload_read": False,
        },
        "physical_dimension_policy": {
            "accepted_strict_ASCII_case_insensitive_spellings": ["V", "mV", "uV"],
            "canonical_output_units": ["V", "mV", "uV"],
            "unknown_or_free_text_behavior": "fail_closed_without_echo",
            "non_ASCII_microvolt_glyph_behavior": (
                "fail_closed_EDF_header_is_strict_ASCII_use_uV"
            ),
        },
        "raw_prefilter_policy": {
            "raw_prefilter_free_text_read": False,
            "raw_prefilter_free_text_emitted": False,
            "derived_acquisition_highpass_or_lowpass": "not_evaluable",
            "reason": "raw_prefilter_free_text_is_outside_the_allowlist",
        },
        "source_identity_policy": {
            "patient_identity_read_or_emitted": False,
            "recording_identity_or_free_text_read_or_emitted": False,
            "start_date_or_time_read_or_emitted": False,
            "path_or_filename_emitted": False,
            "caller_may_bind_only_an_opaque_sha256": True,
        },
        "payload_policy": {
            "eeg_sample_payload_read": False,
            "annotation_payload_read": False,
            "container_size_stat_allowed": True,
            "closed_recording_required": True,
        },
        "scientific_scope": {
            "EEG_acquisition_metadata_only": True,
            "seizure_or_SOZ_target_used": False,
            "finding_or_clinical_claim_authorized": False,
            "research_only": True,
        },
        "policy_receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    policy["policy_receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in policy.items() if key != "policy_receipt_sha256"}
    )
    return policy


def validate_eeg_acquisition_header_allowlist_policy_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    policy = deepcopy(dict(value))
    expected = build_eeg_acquisition_header_allowlist_policy_v1()
    if policy != expected:
        raise ValueError("EEG acquisition-header allowlist policy drifted")
    return policy


def _safe_regular_edf(path: Path) -> tuple[Path, int]:
    source = Path(path)
    if source.is_symlink():
        raise ValueError("EDF path must not be a symlink")
    resolved = source.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("EDF path must be a regular file")
    size = resolved.stat().st_size
    if size < 512:
        raise ValueError("EDF is too short for a standard signal header")
    return resolved, size


def parse_eeg_acquisition_header_v1(
    edf_path: Path,
    *,
    external_source_binding_sha256: str,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read and seal an identity-free allowlisted acquisition projection.

    ``external_source_binding_sha256`` is intentionally opaque.  A caller may
    use a previously validated canonical-signal receipt; a path, patient ID,
    filename, or free-text identifier has no API slot and is never emitted.
    """

    source_binding = _sha256(external_source_binding_sha256, "external source binding")
    active_policy = validate_eeg_acquisition_header_allowlist_policy_v1(
        policy
        if policy is not None
        else build_eeg_acquisition_header_allowlist_policy_v1()
    )
    resolved, container_size = _safe_regular_edf(Path(edf_path))

    accesses: list[dict[str, Any]] = []
    content_hasher = hashlib.sha256()

    with resolved.open("rb") as handle:

        def read_at(
            offset: int, width: int, field: str, channel: int | None = None
        ) -> bytes:
            handle.seek(offset)
            raw = handle.read(width)
            if len(raw) != width:
                raise ValueError(f"EDF {field} field is truncated")
            access = {
                "field": field,
                "offset_bytes": offset,
                "width_bytes": width,
                "channel_index": channel,
            }
            accesses.append(access)
            content_hasher.update(_canonical_json_bytes(access))
            content_hasher.update(raw)
            return raw

        fixed_raw = {
            field: read_at(offset, width, field)
            for field, (offset, width) in _FIXED_ALLOWED.items()
        }
        header_bytes = _integer(
            fixed_raw["header_bytes"], "EDF header_bytes", positive=True
        )
        record_count = _integer(
            fixed_raw["data_record_count"],
            "EDF data_record_count",
            positive=True,
        )
        record_duration = _fraction(
            fixed_raw["data_record_duration_seconds"],
            "EDF data_record_duration_seconds",
            positive=True,
        )
        signal_count = _integer(
            fixed_raw["declared_signal_count"],
            "EDF declared_signal_count",
            positive=True,
        )
        if header_bytes != 256 * (signal_count + 1):
            raise ValueError("EDF standard signal-header framing drifted")
        if header_bytes > container_size:
            raise ValueError("EDF declared header exceeds container size")

        labels: list[str] = []
        for index in range(signal_count):
            offset = 256 + index * _SIGNAL_FIELD_LAYOUT["channel_label"][1]
            label = _normalize_label(
                _strict_ascii(
                    read_at(offset, 16, "channel_label", index),
                    f"EDF channel {index} label",
                )
            )
            if not label:
                raise ValueError("EDF channel label must be non-empty")
            labels.append(label)

        kinds = [_channel_kind(label) for label in labels]
        unknown = [
            labels[index] for index, kind in enumerate(kinds) if kind == "unknown"
        ]
        if unknown:
            # Do not echo the potentially identifying/malformed label.
            raise ValueError(
                f"{len(unknown)} EDF channel label(s) are outside the EEG/auxiliary/annotation registry"
            )

        channels: list[dict[str, Any]] = []
        for index, (label, kind) in enumerate(zip(labels, kinds)):
            if kind != "eeg":
                continue
            raw_fields: dict[str, bytes] = {}
            for field in _SIGNAL_ALLOWED[1:]:
                field_start, width = _SIGNAL_FIELD_LAYOUT[field]
                offset = 256 + signal_count * field_start + index * width
                raw_fields[field] = read_at(offset, width, field, index)

            dimension = _physical_dimension(
                raw_fields["physical_dimension"],
                f"EDF channel {index} physical dimension",
            )
            # A physical minimum is normally negative, so parse the maximum
            # independently rather than applying a non-negative constraint.
            try:
                physical_minimum = Fraction(
                    Decimal(
                        _strict_ascii(
                            raw_fields["physical_minimum"], "physical minimum"
                        )
                    )
                )
                physical_maximum = Fraction(
                    Decimal(
                        _strict_ascii(
                            raw_fields["physical_maximum"], "physical maximum"
                        )
                    )
                )
            except (InvalidOperation, ValueError) as error:
                raise ValueError("EDF physical range is invalid") from error
            if physical_minimum >= physical_maximum:
                raise ValueError("EDF physical range must be strictly increasing")
            digital_minimum = int(
                _strict_ascii(raw_fields["digital_minimum"], "digital minimum")
            )
            digital_maximum = int(
                _strict_ascii(raw_fields["digital_maximum"], "digital maximum")
            )
            if digital_minimum >= digital_maximum:
                raise ValueError("EDF digital range must be strictly increasing")
            samples_per_record = _integer(
                raw_fields["samples_per_data_record"],
                f"EDF channel {index} samples_per_data_record",
                positive=True,
            )
            sampling_rate = Fraction(samples_per_record, 1) / record_duration
            channels.append(
                {
                    "channel_index": index,
                    "channel_label": label,
                    "physical_dimension": dimension,
                    "physical_minimum_fraction": _fraction_json(physical_minimum),
                    "physical_maximum_fraction": _fraction_json(physical_maximum),
                    "digital_minimum": digital_minimum,
                    "digital_maximum": digital_maximum,
                    "samples_per_data_record": samples_per_record,
                    "sampling_rate_hz_fraction": _fraction_json(sampling_rate),
                    "sample_count": samples_per_record * record_count,
                }
            )

    if not channels:
        raise ValueError("EDF contains no allowlisted scalp EEG signal channel")
    recording_duration = record_duration * record_count
    parser_source_sha256 = acquisition_header_parser_source_sha256_v1()
    scope = {
        "signal_acquisition_fields_only": True,
        "patient_identity_bytes_read": 0,
        "recording_identity_or_free_text_bytes_read": 0,
        "start_date_or_time_bytes_read": 0,
        "transducer_free_text_bytes_read": 0,
        "raw_prefilter_free_text_bytes_read": 0,
        "reserved_free_text_bytes_read": 0,
        "eeg_sample_payload_bytes_read": 0,
        "annotation_payload_bytes_read": 0,
        "annotation_channel_header_rows_emitted": 0,
        "auxiliary_channel_header_rows_emitted": 0,
        "non_target_signal_header_rows_emitted": 0,
        "path_filename_or_patient_identifier_emitted": False,
        "external_source_binding_recomputed_from_sample_payload_by_parser": False,
        "external_binding_requires_upstream_validated_canonical_signal_receipt": True,
        "seizure_or_SOZ_target_used": False,
        "clinical_claim_authorized": False,
    }
    receipt: dict[str, Any] = {
        "schema_version": EEG_ACQUISITION_HEADER_RECEIPT_V1_SCHEMA_VERSION,
        "parser_id": EEG_ACQUISITION_HEADER_PARSER_ID_V1,
        "policy_id": active_policy["policy_id"],
        "policy_receipt_sha256": active_policy["policy_receipt_sha256"],
        "parser_source_sha256": parser_source_sha256,
        "external_source_binding_sha256": source_binding,
        "container_size_bytes": container_size,
        "header_bytes": header_bytes,
        "data_record_count": record_count,
        "data_record_duration_seconds_fraction": _fraction_json(record_duration),
        "recording_duration_seconds_fraction": _fraction_json(recording_duration),
        "declared_signal_count": signal_count,
        "eeg_signal_count": len(channels),
        "excluded_annotation_channel_count": sum(
            kind == "annotation" for kind in kinds
        ),
        "excluded_auxiliary_channel_count": sum(kind == "auxiliary" for kind in kinds),
        "excluded_non_target_signal_count": sum(
            kind == "non_target" for kind in kinds
        ),
        "channels": channels,
        "acquisition_filter_cutoffs": {
            "state": "not_evaluable",
            "highpass_hz": None,
            "lowpass_hz": None,
            "reason": "raw_prefilter_free_text_not_read_by_allowlist_v1",
        },
        "allowlisted_header_bytes_sha256": content_hasher.hexdigest(),
        "byte_access_ledger": accesses,
        "byte_access_ledger_sha256": _canonical_sha256(accesses),
        "scope_receipt": scope,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    receipt["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    return validate_eeg_acquisition_header_receipt_v1(
        receipt, policy=active_policy, verify_current_parser_source=True
    )


def validate_eeg_acquisition_header_receipt_v1(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
    verify_current_parser_source: bool = True,
) -> dict[str, Any]:
    """Fail closed on field, content-address, or permission drift."""

    receipt = deepcopy(dict(value))
    if set(receipt) != set(_RECEIPT_FIELDS):
        raise ValueError("EEG acquisition-header receipt fields drifted")
    if receipt["schema_version"] != EEG_ACQUISITION_HEADER_RECEIPT_V1_SCHEMA_VERSION:
        raise ValueError("EEG acquisition-header receipt schema drifted")
    if receipt["parser_id"] != EEG_ACQUISITION_HEADER_PARSER_ID_V1:
        raise ValueError("EEG acquisition-header parser ID drifted")
    active_policy = validate_eeg_acquisition_header_allowlist_policy_v1(
        policy
        if policy is not None
        else build_eeg_acquisition_header_allowlist_policy_v1()
    )
    if (
        receipt["policy_id"] != active_policy["policy_id"]
        or receipt["policy_receipt_sha256"] != active_policy["policy_receipt_sha256"]
    ):
        raise ValueError("EEG acquisition-header policy binding drifted")
    _sha256(receipt["parser_source_sha256"], "parser source hash")
    _sha256(receipt["external_source_binding_sha256"], "source binding")
    _sha256(receipt["allowlisted_header_bytes_sha256"], "header byte hash")
    _sha256(receipt["byte_access_ledger_sha256"], "byte ledger hash")
    if verify_current_parser_source and receipt["parser_source_sha256"] != (
        acquisition_header_parser_source_sha256_v1()
    ):
        raise ValueError("EEG acquisition-header parser source drifted")
    observed_receipt = receipt["receipt_sha256"]
    _sha256(observed_receipt, "receipt hash")
    if observed_receipt != _canonical_sha256(
        {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    ):
        raise ValueError("EEG acquisition-header receipt does not replay")

    for field, positive in (
        ("container_size_bytes", True),
        ("header_bytes", True),
        ("data_record_count", True),
        ("declared_signal_count", True),
        ("eeg_signal_count", True),
        ("excluded_annotation_channel_count", False),
        ("excluded_auxiliary_channel_count", False),
        ("excluded_non_target_signal_count", False),
    ):
        item = receipt[field]
        if type(item) is not int or (positive and item <= 0) or (not positive and item < 0):
            raise ValueError(f"EEG acquisition {field} is invalid")
    record_duration = _fraction_from_json(
        receipt["data_record_duration_seconds_fraction"],
        "data record duration",
    )
    recording_duration = _fraction_from_json(
        receipt["recording_duration_seconds_fraction"],
        "recording duration",
    )
    if record_duration <= 0 or recording_duration != record_duration * receipt[
        "data_record_count"
    ]:
        raise ValueError("EEG acquisition recording clock does not close")
    if receipt["data_record_duration_seconds_fraction"] != _fraction_json(
        record_duration
    ) or receipt["recording_duration_seconds_fraction"] != _fraction_json(
        recording_duration
    ):
        raise ValueError("EEG acquisition clock fractions are not canonical")

    if type(receipt["channels"]) is not list or not receipt["channels"]:
        raise ValueError("EEG acquisition-header channel roster is empty")
    indices: list[int] = []
    for row in receipt["channels"]:
        if type(row) is not dict or set(row) != set(_CHANNEL_FIELDS):
            raise ValueError("EEG acquisition channel fields drifted")
        index = row["channel_index"]
        if type(index) is not int or index < 0:
            raise ValueError("EEG acquisition channel index is invalid")
        if (
            not isinstance(row["channel_label"], str)
            or row["channel_label"] != _normalize_label(row["channel_label"])
            or _channel_kind(row["channel_label"]) != "eeg"
        ):
            raise ValueError("non-EEG channel entered the acquisition receipt")
        if row["physical_dimension"] not in set(
            _PHYSICAL_DIMENSION_CANONICAL.values()
        ):
            raise ValueError("EEG acquisition physical dimension is not canonical")
        physical_minimum = _fraction_from_json(
            row["physical_minimum_fraction"], "physical minimum"
        )
        physical_maximum = _fraction_from_json(
            row["physical_maximum_fraction"], "physical maximum"
        )
        if (
            row["physical_minimum_fraction"] != _fraction_json(physical_minimum)
            or row["physical_maximum_fraction"] != _fraction_json(physical_maximum)
            or physical_minimum >= physical_maximum
        ):
            raise ValueError("EEG acquisition physical range is invalid")
        if (
            type(row["digital_minimum"]) is not int
            or type(row["digital_maximum"]) is not int
            or row["digital_minimum"] >= row["digital_maximum"]
        ):
            raise ValueError("EEG acquisition digital range is invalid")
        if type(row["samples_per_data_record"]) is not int or row[
            "samples_per_data_record"
        ] <= 0:
            raise ValueError("EEG acquisition sample count per record is invalid")
        sampling_rate = _fraction_from_json(
            row["sampling_rate_hz_fraction"], "sampling rate"
        )
        expected_sampling_rate = Fraction(
            row["samples_per_data_record"], 1
        ) / record_duration
        if (
            sampling_rate != expected_sampling_rate
            or row["sampling_rate_hz_fraction"] != _fraction_json(sampling_rate)
        ):
            raise ValueError("EEG acquisition sampling clock is inconsistent")
        if type(row["sample_count"]) is not int or row["sample_count"] != (
            row["samples_per_data_record"] * receipt["data_record_count"]
        ):
            raise ValueError("EEG acquisition channel sample count is inconsistent")
        indices.append(index)
    if indices != sorted(set(indices)):
        raise ValueError("EEG acquisition channel indices are not unique and sorted")
    if any(index >= receipt["declared_signal_count"] for index in indices):
        raise ValueError("EEG acquisition channel index exceeds declared roster")
    if receipt["eeg_signal_count"] != len(receipt["channels"]):
        raise ValueError("EEG acquisition channel count drifted")
    if receipt["declared_signal_count"] != (
        receipt["eeg_signal_count"]
        + receipt["excluded_annotation_channel_count"]
        + receipt["excluded_auxiliary_channel_count"]
        + receipt["excluded_non_target_signal_count"]
    ):
        raise ValueError("EDF channel accounting does not close")
    if receipt["header_bytes"] != 256 * (receipt["declared_signal_count"] + 1):
        raise ValueError("EDF signal-header framing drifted")
    if receipt["container_size_bytes"] < receipt["header_bytes"]:
        raise ValueError("EDF container/header size drifted")

    ledger = receipt["byte_access_ledger"]
    if type(ledger) is not list or receipt["byte_access_ledger_sha256"] != (
        _canonical_sha256(ledger)
    ):
        raise ValueError("EEG acquisition byte-access ledger drifted")
    expected_ledger = [
        {
            "field": field,
            "offset_bytes": offset,
            "width_bytes": width,
            "channel_index": None,
        }
        for field, (offset, width) in _FIXED_ALLOWED.items()
    ]
    expected_ledger.extend(
        {
            "field": "channel_label",
            "offset_bytes": 256 + index * _SIGNAL_FIELD_LAYOUT["channel_label"][1],
            "width_bytes": _SIGNAL_FIELD_LAYOUT["channel_label"][1],
            "channel_index": index,
        }
        for index in range(receipt["declared_signal_count"])
    )
    for index in indices:
        for field in _SIGNAL_ALLOWED[1:]:
            field_start, width = _SIGNAL_FIELD_LAYOUT[field]
            expected_ledger.append(
                {
                    "field": field,
                    "offset_bytes": (
                        256 + receipt["declared_signal_count"] * field_start + index * width
                    ),
                    "width_bytes": width,
                    "channel_index": index,
                }
            )
    if ledger != expected_ledger:
        raise PermissionError(
            "EEG acquisition byte ledger is not the exact allowlisted offset schedule"
        )
    scope = receipt["scope_receipt"]
    expected_scope = {
        "signal_acquisition_fields_only": True,
        "patient_identity_bytes_read": 0,
        "recording_identity_or_free_text_bytes_read": 0,
        "start_date_or_time_bytes_read": 0,
        "transducer_free_text_bytes_read": 0,
        "raw_prefilter_free_text_bytes_read": 0,
        "reserved_free_text_bytes_read": 0,
        "eeg_sample_payload_bytes_read": 0,
        "annotation_payload_bytes_read": 0,
        "annotation_channel_header_rows_emitted": 0,
        "auxiliary_channel_header_rows_emitted": 0,
        "non_target_signal_header_rows_emitted": 0,
        "path_filename_or_patient_identifier_emitted": False,
        "external_source_binding_recomputed_from_sample_payload_by_parser": False,
        "external_binding_requires_upstream_validated_canonical_signal_receipt": True,
        "seizure_or_SOZ_target_used": False,
        "clinical_claim_authorized": False,
    }
    if scope != expected_scope:
        raise PermissionError("EEG acquisition-only scope receipt drifted")
    if receipt["acquisition_filter_cutoffs"] != {
        "state": "not_evaluable",
        "highpass_hz": None,
        "lowpass_hz": None,
        "reason": "raw_prefilter_free_text_not_read_by_allowlist_v1",
    }:
        raise ValueError("raw prefilter text was promoted to an acquisition fact")
    return receipt


__all__ = [
    "EEG_ACQUISITION_HEADER_ALLOWLIST_V1_SCHEMA_VERSION",
    "EEG_ACQUISITION_HEADER_PARSER_ID_V1",
    "EEG_ACQUISITION_HEADER_RECEIPT_V1_SCHEMA_VERSION",
    "acquisition_header_parser_source_sha256_v1",
    "build_eeg_acquisition_header_allowlist_policy_v1",
    "parse_eeg_acquisition_header_v1",
    "validate_eeg_acquisition_header_allowlist_policy_v1",
    "validate_eeg_acquisition_header_receipt_v1",
]
