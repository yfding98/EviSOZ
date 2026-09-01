"""Complete, reference-free TUSZ inventory for continuous detection.

This module freezes the *opportunity denominator* before detector calibration
or evaluation.  It reads EDF container bytes and acquisition headers, and it
checks the identity of the expected ``.csv_bi`` sidecar without opening that
sidecar.  Seizure intervals, annotations, channel targets, reports, Excel and
clinical text are therefore absent from the artifact.

The receipt proves a complete local tree relative to a predeclared release
inventory, exact EDF-container deduplication and patient split isolation.  It
does not prove canonical physical-signal deduplication, detector performance,
clinical validity, or permission to open official-evaluation references.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping, Sequence


TUSZ_COMPLETE_DETECTOR_ROSTER_SCHEMA_VERSION = "tusz_complete_detector_roster_v1"
TUSZ_COMPLETE_DETECTOR_ROSTER_METHOD_ID = (
    "full_edf_container_header_and_sidecar_identity_inventory_v1"
)

TUSZ_V203_EXPECTED_INVENTORY: Final[dict[str, Any]] = {
    "release_id": "TUSZ-v2.0.3-local-complete-inventory-v1",
    "split_expectations": {
        "train": {"patient_count": 579, "recording_count": 4667},
        "dev": {"patient_count": 53, "recording_count": 1832},
        "eval": {"patient_count": 43, "recording_count": 865},
    },
    "total_patient_count": 675,
    "total_recording_count": 7364,
}

_OFFICIAL_TO_BENCHMARK_SPLIT: Final[dict[str, str]] = {
    "train": "source_train",
    "dev": "source_dev",
    "eval": "source_eval",
}
_HEX = frozenset("0123456789abcdef")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a non-empty trimmed identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, context: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _fraction_pair(
    value: object, context: str, *, allow_zero: bool = False
) -> list[int]:
    if (
        type(value) is not list
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
        or value[0] < (0 if allow_zero else 1)
    ):
        raise ValueError(f"{context} must be a canonical positive fraction pair")
    fraction = Fraction(value[0], value[1])
    if [fraction.numerator, fraction.denominator] != value:
        raise ValueError(f"{context} fraction is not reduced")
    return [fraction.numerator, fraction.denominator]


def _ascii_number(raw: bytes, context: str, *, integer: bool) -> int | Decimal:
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not ASCII") from error
    if not text:
        raise ValueError(f"{context} is empty")
    try:
        return int(text) if integer else Decimal(text)
    except (ValueError, InvalidOperation) as error:
        raise ValueError(f"{context} is not numeric") from error


def _fraction_from_decimal(value: Decimal, context: str) -> Fraction:
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{context} must be finite and positive")
    return Fraction(value)


def _safe_regular_file(path: Path, root: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escaped the pinned TUSZ root") from error
    return resolved


def inspect_edf_container_header_v1(path: Path) -> dict[str, Any]:
    """Read acquisition-only EDF metadata and close it against file size."""

    file_size = path.stat().st_size
    if file_size < 256:
        raise ValueError("EDF container is shorter than the fixed header")
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError("EDF fixed header could not be read completely")
        # EDF is a 16-bit format whose version field is ASCII ``0`` padded to
        # eight bytes.  In particular, BDF uses a different version marker and
        # three-byte samples, so accepting it here would make the payload-size
        # closure below scientifically false.
        if fixed[:8] != b"0       ":
            raise ValueError("unsupported EDF version (BDF is not accepted)")
        header_bytes = int(
            _ascii_number(fixed[184:192], "EDF header bytes", integer=True)
        )
        declared_records = int(
            _ascii_number(fixed[236:244], "EDF data record count", integer=True)
        )
        record_duration_decimal = _ascii_number(
            fixed[244:252], "EDF data record duration", integer=False
        )
        assert isinstance(record_duration_decimal, Decimal)
        record_duration = _fraction_from_decimal(
            record_duration_decimal, "EDF data record duration"
        )
        signal_count = int(
            _ascii_number(fixed[252:256], "EDF signal count", integer=True)
        )
        if signal_count < 1:
            raise ValueError("EDF signal count must be positive")
        expected_header_bytes = 256 * (signal_count + 1)
        if header_bytes != expected_header_bytes or header_bytes > file_size:
            raise ValueError("EDF declared header length is inconsistent")
        remainder = handle.read(header_bytes - 256)
        if len(remainder) != header_bytes - 256:
            raise ValueError("EDF signal headers could not be read completely")

    labels_end = signal_count * 16
    labels_raw = remainder[:labels_end]
    labels = [
        labels_raw[index * 16 : (index + 1) * 16].decode("latin-1").strip()
        for index in range(signal_count)
    ]
    samples_offset = signal_count * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    samples_raw = remainder[samples_offset : samples_offset + signal_count * 8]
    if len(samples_raw) != signal_count * 8:
        raise ValueError("EDF samples-per-record header is truncated")
    samples_per_record = [
        int(
            _ascii_number(
                samples_raw[index * 8 : (index + 1) * 8],
                f"EDF signal {index} samples per record",
                integer=True,
            )
        )
        for index in range(signal_count)
    ]
    if any(value < 1 for value in samples_per_record):
        raise ValueError("EDF samples per record must be positive")

    bytes_per_record = 2 * sum(samples_per_record)
    payload_bytes = file_size - header_bytes
    if payload_bytes < 0 or payload_bytes % bytes_per_record:
        raise ValueError("EDF payload size does not close to complete data records")
    derived_records = payload_bytes // bytes_per_record
    if declared_records == -1:
        record_count = derived_records
        record_count_source = "derived_from_closed_file_size"
    elif declared_records < 1:
        raise ValueError("EDF data record count must be positive or -1")
    else:
        record_count = declared_records
        record_count_source = "edf_header"
        if declared_records != derived_records:
            raise ValueError("EDF declared record count disagrees with file size")
    if record_count < 1:
        raise ValueError("EDF contains no complete data record")

    recording_duration = record_duration * record_count
    sampling_rates = sorted(
        {Fraction(samples, 1) / record_duration for samples in samples_per_record}
    )
    return {
        "container_bytes": file_size,
        "header_bytes": header_bytes,
        "header_sha256": _sha256_bytes(fixed + remainder),
        "signal_count": signal_count,
        "signal_label_roster_sha256": _canonical_sha256(labels),
        "data_record_count": record_count,
        "data_record_count_source": record_count_source,
        "data_record_bytes": bytes_per_record,
        "data_record_duration_fraction": [
            record_duration.numerator,
            record_duration.denominator,
        ],
        "recording_duration_fraction": [
            recording_duration.numerator,
            recording_duration.denominator,
        ],
        "native_sampling_rate_fractions": [
            [value.numerator, value.denominator] for value in sampling_rates
        ],
        "edf_file_size_closed": True,
    }


def validate_tusz_complete_expected_inventory_v1(value: object) -> dict[str, Any]:
    required = {
        "release_id",
        "split_expectations",
        "total_patient_count",
        "total_recording_count",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("TUSZ expected inventory fields drifted")
    data = deepcopy(value)
    _identifier(data["release_id"], "TUSZ release ID")
    splits = data["split_expectations"]
    if type(splits) is not dict or set(splits) != set(_OFFICIAL_TO_BENCHMARK_SPLIT):
        raise ValueError("TUSZ expected inventory must contain train/dev/eval")
    patient_total = 0
    recording_total = 0
    for split in sorted(splits):
        row = splits[split]
        if type(row) is not dict or set(row) != {"patient_count", "recording_count"}:
            raise ValueError("TUSZ expected split fields drifted")
        patient_total += _positive_integer(row["patient_count"], "expected patients")
        recording_total += _positive_integer(
            row["recording_count"], "expected recordings"
        )
    if (
        _positive_integer(data["total_patient_count"], "expected total patients")
        != patient_total
    ):
        raise ValueError("TUSZ expected patient total does not close")
    if (
        _positive_integer(data["total_recording_count"], "expected total recordings")
        != recording_total
    ):
        raise ValueError("TUSZ expected recording total does not close")
    return data


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    all_patients: set[str] = set()
    all_recordings: list[str] = []
    total_duration = Fraction(0, 1)
    for official_split in sorted(_OFFICIAL_TO_BENCHMARK_SPLIT):
        selected = [row for row in records if row["official_split"] == official_split]
        patients = sorted({str(row["patient_id"]) for row in selected})
        recording_ids = sorted(str(row["recording_id"]) for row in selected)
        duration = sum(
            (
                Fraction(
                    row["recording_duration_fraction"][0],
                    row["recording_duration_fraction"][1],
                )
                for row in selected
            ),
            Fraction(0, 1),
        )
        montage_counts: dict[str, int] = {}
        sampling_rate_record_counts: dict[str, int] = {}
        for row in selected:
            montage = str(row["montage"])
            montage_counts[montage] = montage_counts.get(montage, 0) + 1
            for pair in row["native_sampling_rate_fractions"]:
                key = f"{pair[0]}/{pair[1]}"
                sampling_rate_record_counts[key] = (
                    sampling_rate_record_counts.get(key, 0) + 1
                )
        summaries[official_split] = {
            "benchmark_split": _OFFICIAL_TO_BENCHMARK_SPLIT[official_split],
            "patient_count": len(patients),
            "recording_count": len(selected),
            "duration_seconds_fraction": [duration.numerator, duration.denominator],
            "patient_roster_sha256": _canonical_sha256(patients),
            "recording_roster_sha256": _canonical_sha256(recording_ids),
            "container_binding_roster_sha256": _canonical_sha256(
                [[row["recording_id"], row["container_sha256"]] for row in selected]
            ),
            "montage_record_counts": dict(sorted(montage_counts.items())),
            "sampling_rate_record_counts": dict(
                sorted(sampling_rate_record_counts.items())
            ),
        }
        if all_patients.intersection(patients):
            raise ValueError("one TUSZ patient occurs in multiple official splits")
        all_patients.update(patients)
        all_recordings.extend(recording_ids)
        total_duration += duration
    return {
        "split_summaries": summaries,
        "total_patient_count": len(all_patients),
        "total_recording_count": len(records),
        "total_duration_seconds_fraction": [
            total_duration.numerator,
            total_duration.denominator,
        ],
        "all_patient_roster_sha256": _canonical_sha256(sorted(all_patients)),
        "all_recording_roster_sha256": _canonical_sha256(sorted(all_recordings)),
        "records_payload_sha256": _canonical_sha256(records),
    }


def _validate_record_row(value: object, index: int) -> dict[str, Any]:
    required = {
        "recording_id",
        "patient_id",
        "official_split",
        "benchmark_split",
        "montage",
        "container_bytes",
        "container_sha256",
        "header_bytes",
        "header_sha256",
        "signal_count",
        "signal_label_roster_sha256",
        "data_record_count",
        "data_record_count_source",
        "data_record_bytes",
        "data_record_duration_fraction",
        "recording_duration_fraction",
        "native_sampling_rate_fractions",
        "edf_file_size_closed",
        "reference_sidecar_id",
        "reference_sidecar_exists",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"TUSZ roster record {index} fields drifted")
    row = deepcopy(value)
    recording_id = _identifier(row["recording_id"], "recording ID")
    patient_id = _identifier(row["patient_id"], "patient ID")
    split = row["official_split"]
    if split not in _OFFICIAL_TO_BENCHMARK_SPLIT:
        raise ValueError("TUSZ official split is invalid")
    if row["benchmark_split"] != _OFFICIAL_TO_BENCHMARK_SPLIT[split]:
        raise ValueError("TUSZ benchmark split mapping drifted")
    parts = Path(recording_id).parts
    if (
        Path(recording_id).is_absolute()
        or ".." in parts
        or len(parts) < 3
        or parts[0] != split
        or parts[1] != patient_id
        or not recording_id.endswith(".edf")
    ):
        raise ValueError("TUSZ recording path identity is invalid")
    _identifier(row["montage"], "montage")
    if Path(recording_id).parent.name != row["montage"]:
        raise ValueError("TUSZ montage does not match recording path")
    container_bytes = _positive_integer(row["container_bytes"], "container bytes")
    _sha256(row["container_sha256"], "container SHA-256")
    header_bytes = _positive_integer(row["header_bytes"], "header bytes")
    _sha256(row["header_sha256"], "header SHA-256")
    signal_count = _positive_integer(row["signal_count"], "signal count")
    if header_bytes != 256 * (signal_count + 1):
        raise ValueError("EDF header bytes do not close to the signal count")
    _sha256(row["signal_label_roster_sha256"], "signal label roster SHA-256")
    record_count = _positive_integer(row["data_record_count"], "data record count")
    if row["data_record_count_source"] not in {
        "edf_header",
        "derived_from_closed_file_size",
    }:
        raise ValueError("EDF data record count source drifted")
    data_record_bytes = _positive_integer(row["data_record_bytes"], "data record bytes")
    record_duration_pair = _fraction_pair(
        row["data_record_duration_fraction"], "data record duration"
    )
    recording_duration_pair = _fraction_pair(
        row["recording_duration_fraction"], "recording duration"
    )
    record_duration = Fraction(*record_duration_pair)
    recording_duration = Fraction(*recording_duration_pair)
    if recording_duration != record_duration * record_count:
        raise ValueError("EDF recording duration does not close to its records")
    if container_bytes != header_bytes + record_count * data_record_bytes:
        raise ValueError("EDF container bytes do not close to complete records")
    rates = row["native_sampling_rate_fractions"]
    if type(rates) is not list or not rates:
        raise ValueError("native sampling-rate roster is empty")
    normalized_rates = [_fraction_pair(pair, "native sampling rate") for pair in rates]
    if normalized_rates != sorted(normalized_rates, key=lambda pair: Fraction(*pair)):
        raise ValueError("native sampling-rate roster is not sorted")
    if len({tuple(pair) for pair in normalized_rates}) != len(normalized_rates):
        raise ValueError("native sampling-rate roster contains duplicates")
    if row["edf_file_size_closed"] is not True:
        raise ValueError("EDF file-size closure is missing")
    sidecar_id = _identifier(row["reference_sidecar_id"], "reference sidecar ID")
    if sidecar_id != str(Path(recording_id).with_suffix(".csv_bi")):
        raise ValueError("reference sidecar identity drifted")
    if row["reference_sidecar_exists"] is not True:
        raise ValueError("reference sidecar pairing is incomplete")
    return row


def validate_tusz_complete_detector_roster_v1(payload: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "roster_id",
        "expected_inventory",
        "records",
        "observed_inventory",
        "reference_sidecar_inventory",
        "exact_container_duplicate_audit",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("TUSZ complete detector roster fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != TUSZ_COMPLETE_DETECTOR_ROSTER_SCHEMA_VERSION
        or data["method_id"] != TUSZ_COMPLETE_DETECTOR_ROSTER_METHOD_ID
    ):
        raise ValueError("TUSZ complete detector roster schema/method drifted")
    expected = validate_tusz_complete_expected_inventory_v1(data["expected_inventory"])
    records = data["records"]
    if type(records) is not list or not records:
        raise ValueError("TUSZ complete detector roster has no records")
    validated_records = [
        _validate_record_row(row, index) for index, row in enumerate(records)
    ]
    canonical_order = sorted(
        validated_records,
        key=lambda row: (row["official_split"], row["patient_id"], row["recording_id"]),
    )
    if validated_records != canonical_order:
        raise ValueError("TUSZ complete detector records are not canonically sorted")
    recording_ids = [row["recording_id"] for row in validated_records]
    if len(set(recording_ids)) != len(recording_ids):
        raise ValueError("TUSZ recording IDs are not unique")

    observed = _summarize_records(validated_records)
    if data["observed_inventory"] != observed:
        raise ValueError("TUSZ observed inventory is not replayable")
    if (
        observed["total_patient_count"] != expected["total_patient_count"]
        or observed["total_recording_count"] != expected["total_recording_count"]
    ):
        raise ValueError("TUSZ total inventory differs from the expected release")
    for split, expected_row in expected["split_expectations"].items():
        observed_row = observed["split_summaries"][split]
        if (
            observed_row["patient_count"] != expected_row["patient_count"]
            or observed_row["recording_count"] != expected_row["recording_count"]
        ):
            raise ValueError("TUSZ split inventory differs from the expected release")

    sidecars = data["reference_sidecar_inventory"]
    expected_sidecar_ids = sorted(row["reference_sidecar_id"] for row in records)
    required_sidecars = {
        "sidecar_count",
        "sidecar_identity_roster_sha256",
        "one_to_one_with_edf_verified",
        "sidecar_contents_opened",
    }
    if type(sidecars) is not dict or set(sidecars) != required_sidecars:
        raise ValueError("TUSZ sidecar inventory fields drifted")
    if sidecars != {
        "sidecar_count": len(expected_sidecar_ids),
        "sidecar_identity_roster_sha256": _canonical_sha256(expected_sidecar_ids),
        "one_to_one_with_edf_verified": True,
        "sidecar_contents_opened": False,
    }:
        raise ValueError("TUSZ sidecar identity inventory drifted")

    hash_to_recordings: dict[str, list[str]] = {}
    for row in records:
        hash_to_recordings.setdefault(row["container_sha256"], []).append(
            row["recording_id"]
        )
    duplicate_groups = sorted(
        [sorted(values) for values in hash_to_recordings.values() if len(values) > 1]
    )
    cross_split_groups = [
        group
        for group in duplicate_groups
        if len({Path(recording_id).parts[0] for recording_id in group}) > 1
    ]
    expected_duplicate_audit = {
        "audit_scope": "exact_edf_container_bytes_only",
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_recording_count": sum(len(group) for group in duplicate_groups),
        "cross_split_duplicate_group_count": len(cross_split_groups),
        "duplicate_groups": duplicate_groups,
        "canonical_physical_signal_duplicate_audit_complete": False,
        "robust_tile_fingerprint_audit_complete": False,
    }
    if data["exact_container_duplicate_audit"] != expected_duplicate_audit:
        raise ValueError("TUSZ exact-container duplicate audit drifted")
    if duplicate_groups:
        raise ValueError("TUSZ exact EDF-container duplicates must be resolved")

    expected_scope = {
        "complete_local_inventory_against_pinned_counts_verified": True,
        "all_edf_container_bytes_hashed": True,
        "all_edf_headers_closed_to_file_size": True,
        "patient_official_split_isolation_verified": True,
        "csv_bi_one_to_one_identity_verified": True,
        "csv_bi_contents_read": False,
        "reference_labels_retained": False,
        "edf_annotations_used_as_model_input": False,
        "excel_doctor_or_clinical_text_used": False,
        "canonical_physical_signal_hashes_materialized": False,
        "cross_corpus_patient_alias_audit_complete": False,
        "official_eval_reference_access_authorized": False,
        "detector_performance_or_sota_claim_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("TUSZ complete detector roster scope drifted")
    digest = deepcopy(data)
    digest["roster_id"] = "TUSZ-COMPLETE-ROSTER-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "TUSZCROSTER-" + _canonical_sha256(digest)[:24]
    if data["roster_id"] != expected_id:
        raise ValueError("TUSZ complete detector roster is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("TUSZ complete detector roster receipt hash drifted")
    return data


def build_tusz_complete_detector_roster_v1(
    *,
    tusz_root: str | Path,
    expected_inventory: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Scan and hash one complete local TUSZ tree without reading references."""

    root_path = Path(tusz_root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError("TUSZ root must be a regular non-symlink directory")
    root = root_path.resolve(strict=True)
    expectation = validate_tusz_complete_expected_inventory_v1(
        deepcopy(
            dict(expected_inventory)
            if expected_inventory is not None
            else TUSZ_V203_EXPECTED_INVENTORY
        )
    )
    edf_paths = sorted(
        root.rglob("*.edf"), key=lambda path: path.relative_to(root).as_posix()
    )
    if not edf_paths:
        raise ValueError("TUSZ root contains no EDF files")
    discovered_sidecars = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.csv_bi")
        if _safe_regular_file(path, root, "TUSZ csv_bi sidecar")
    }
    records: list[dict[str, Any]] = []
    expected_sidecars: set[str] = set()
    for edf_path in edf_paths:
        edf = _safe_regular_file(edf_path, root, "TUSZ EDF")
        relative = edf.relative_to(root)
        if (
            len(relative.parts) < 3
            or relative.parts[0] not in _OFFICIAL_TO_BENCHMARK_SPLIT
        ):
            raise ValueError("EDF path is outside train/dev/eval official layout")
        split = relative.parts[0]
        patient_id = relative.parts[1]
        recording_id = relative.as_posix()
        sidecar = edf.with_suffix(".csv_bi")
        _safe_regular_file(sidecar, root, "TUSZ csv_bi sidecar")
        sidecar_id = sidecar.relative_to(root).as_posix()
        expected_sidecars.add(sidecar_id)
        before = edf.stat()
        metadata = inspect_edf_container_header_v1(edf)
        container_sha256 = _sha256_file(edf)
        after = edf.stat()
        stable_fields_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_fields_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if stable_fields_before != stable_fields_after:
            raise ValueError("TUSZ EDF changed while it was being inventoried")
        records.append(
            {
                "recording_id": recording_id,
                "patient_id": patient_id,
                "official_split": split,
                "benchmark_split": _OFFICIAL_TO_BENCHMARK_SPLIT[split],
                "montage": relative.parent.name,
                "container_sha256": container_sha256,
                **metadata,
                "reference_sidecar_id": sidecar_id,
                "reference_sidecar_exists": True,
            }
        )
    if expected_sidecars != discovered_sidecars:
        missing = sorted(expected_sidecars - discovered_sidecars)[:5]
        orphan = sorted(discovered_sidecars - expected_sidecars)[:5]
        raise ValueError(
            f"EDF/csv_bi inventory is not one-to-one; missing={missing}, orphan={orphan}"
        )
    records.sort(
        key=lambda row: (row["official_split"], row["patient_id"], row["recording_id"])
    )
    observed = _summarize_records(records)
    hash_to_recordings: dict[str, list[str]] = {}
    for row in records:
        hash_to_recordings.setdefault(row["container_sha256"], []).append(
            row["recording_id"]
        )
    duplicate_groups = sorted(
        [sorted(values) for values in hash_to_recordings.values() if len(values) > 1]
    )
    cross_split_groups = [
        group
        for group in duplicate_groups
        if len({Path(recording_id).parts[0] for recording_id in group}) > 1
    ]
    body: dict[str, Any] = {
        "schema_version": TUSZ_COMPLETE_DETECTOR_ROSTER_SCHEMA_VERSION,
        "method_id": TUSZ_COMPLETE_DETECTOR_ROSTER_METHOD_ID,
        "roster_id": "TUSZ-COMPLETE-ROSTER-PENDING",
        "expected_inventory": expectation,
        "records": records,
        "observed_inventory": observed,
        "reference_sidecar_inventory": {
            "sidecar_count": len(expected_sidecars),
            "sidecar_identity_roster_sha256": _canonical_sha256(
                sorted(expected_sidecars)
            ),
            "one_to_one_with_edf_verified": True,
            "sidecar_contents_opened": False,
        },
        "exact_container_duplicate_audit": {
            "audit_scope": "exact_edf_container_bytes_only",
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_recording_count": sum(len(group) for group in duplicate_groups),
            "cross_split_duplicate_group_count": len(cross_split_groups),
            "duplicate_groups": duplicate_groups,
            "canonical_physical_signal_duplicate_audit_complete": False,
            "robust_tile_fingerprint_audit_complete": False,
        },
        "scope_receipt": {
            "complete_local_inventory_against_pinned_counts_verified": True,
            "all_edf_container_bytes_hashed": True,
            "all_edf_headers_closed_to_file_size": True,
            "patient_official_split_isolation_verified": True,
            "csv_bi_one_to_one_identity_verified": True,
            "csv_bi_contents_read": False,
            "reference_labels_retained": False,
            "edf_annotations_used_as_model_input": False,
            "excel_doctor_or_clinical_text_used": False,
            "canonical_physical_signal_hashes_materialized": False,
            "cross_corpus_patient_alias_audit_complete": False,
            "official_eval_reference_access_authorized": False,
            "detector_performance_or_sota_claim_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["roster_id"] = "TUSZCROSTER-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_tusz_complete_detector_roster_v1(body)


__all__ = [
    "TUSZ_COMPLETE_DETECTOR_ROSTER_METHOD_ID",
    "TUSZ_COMPLETE_DETECTOR_ROSTER_SCHEMA_VERSION",
    "TUSZ_V203_EXPECTED_INVENTORY",
    "build_tusz_complete_detector_roster_v1",
    "inspect_edf_container_header_v1",
    "validate_tusz_complete_detector_roster_v1",
    "validate_tusz_complete_expected_inventory_v1",
]
