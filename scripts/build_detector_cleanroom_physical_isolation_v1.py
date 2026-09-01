#!/usr/bin/env python3
"""Build and audit physically isolated detector train/dev manifests.

This program deliberately has two asymmetric data paths:

* source-train rows may resolve and open their public TUSZ ``.csv_bi``
  sidecar and retain only exact global ``TERM,seiz`` intervals; and
* source-dev rows are projected only from the already frozen physical EEG
  identity audit and fold plan.  No sidecar path is ever resolved for dev.

The output is a preparation/audit artifact.  It does not train a detector,
run inference, open source-eval, or estimate performance.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.st16_common17_axis_contract_v1 import (
    CANONICAL_ST16_TYPED_UNITS,
    COMMON17_REFERENTIAL_AXIS_ORDER,
)
from src.clinical_eeg_long_recording.tusz_canonical_physical_signal_audit_v1 import (
    validate_tusz_canonical_physical_analysis_projection_v1,
    validate_tusz_canonical_physical_duplicate_audit_v1,
)
from src.clinical_eeg_long_recording.tusz_detector_cleanroom_fold_plan_v1 import (
    validate_tusz_detector_cleanroom_fold_plan_v1,
)


DEFAULT_CONFIG = (
    ROOT / "configs/clinical_eeg_detector_cleanroom_physical_isolation_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/clinical_eeg_detector_cleanroom_physical_isolation_v1_20260825"
)

CONFIG_SCHEMA_VERSION = "clinical_eeg_detector_cleanroom_physical_isolation_v1"
TRAIN_SCHEMA_VERSION = "clinical_eeg_detector_source_train_labeled_manifest_v1"
DEV_SCHEMA_VERSION = "clinical_eeg_detector_source_dev_eeg_only_prediction_roster_v1"
RECEIPT_SCHEMA_VERSION = "clinical_eeg_detector_cleanroom_physical_isolation_receipt_v1"
METHOD_ID = "canonical_physical_source_train_targets_source_dev_eeg_only_v1"
PENDING = "CONTENT-ADDRESS-PENDING"

COMMON17_SENSOR_ONTOLOGY_ORDER = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
)
LEGACY_NAME_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
PREDICTION_TERMINAL_STATES = (
    "completed_with_candidates",
    "completed_zero_candidate",
    "partial_coverage",
    "technical_failure",
)

IDENTITY_ROW_FIELDS = frozenset(
    {
        "analysis_identity_id",
        "model_split",
        "official_split",
        "local_patient_id",
        "local_edf_path",
        "source_edf_container_sha256",
        "canonical_physical_equivalence_id",
        "canonical_physical_source_tensor_sha256",
        "canonical_source_signal_sha256",
        "canonical_source_header_receipt_sha256",
        "source_physical_identity_multiplicity",
        "recording_duration_seconds_fraction",
        "sampling_rate_hz_fraction",
        "sample_count",
        "common17_available_names",
        "common17_source_names",
        "channel_resolution_sha256",
        "channel_contract_sha256",
        "row_sha256",
    }
)
TRAIN_ROW_FIELDS = IDENTITY_ROW_FIELDS | frozenset(
    {
        "held_out_fold_id",
        "target_sidecar_relative_path",
        "target_sidecar_sha256",
        "global_TERM_seiz_intervals_seconds",
        "target_parse_sha256",
    }
)
DEV_ROW_FIELDS = IDENTITY_ROW_FIELDS

DEV_FORBIDDEN_KEY_FRAGMENTS = (
    "label",
    "reference",
    "csv_bi",
    "seizure",
    "ictal",
    "onset",
    "offset",
    "annotation",
)
DEV_FORBIDDEN_VALUE_FRAGMENTS = (
    "csv_bi",
    "seizure",
    "ictal",
    "onset",
    "offset",
    "annotation",
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not set(value).difference("0123456789abcdef")
    )


def _strict_object(value: object, fields: Iterable[str], context: str) -> dict[str, Any]:
    expected = set(fields)
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{context} fields drifted")
    return deepcopy(value)


def _fraction(value: object, context: str, *, positive: bool = False) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
        or (positive and value[0] <= 0)
        or (not positive and value[0] < 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context} must be a reduced {qualifier} fraction")
    result = Fraction(value[0], value[1])
    if [result.numerator, result.denominator] != value:
        raise ValueError(f"{context} fraction is not reduced")
    return result


def _fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a normalized non-empty string")
    return value


def _content_address(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(payload))
    if value.get("content_sha256") != PENDING:
        raise ValueError("content-address input must carry the pending marker")
    value["content_sha256"] = canonical_sha256(value)
    return value


def _validate_content_address(payload: Mapping[str, Any], context: str) -> None:
    observed = payload.get("content_sha256")
    if not _is_sha256(observed):
        raise ValueError(f"{context} content hash is invalid")
    replay = deepcopy(dict(payload))
    replay["content_sha256"] = PENDING
    if observed != canonical_sha256(replay):
        raise ValueError(f"{context} content hash drifted")


def _attach_row_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(payload))
    if "row_sha256" in row:
        raise ValueError("row hash field already exists")
    row["row_sha256"] = canonical_sha256(row)
    return row


def _validate_row_hash(row: Mapping[str, Any], context: str) -> None:
    observed = row.get("row_sha256")
    if not _is_sha256(observed):
        raise ValueError(f"{context} row hash is invalid")
    replay = deepcopy(dict(row))
    del replay["row_sha256"]
    if observed != canonical_sha256(replay):
        raise ValueError(f"{context} row hash drifted")


def normalize_electrode_name(value: object) -> str:
    """Normalize only spelling/case and the four frozen legacy aliases."""

    name = _identifier(value, "electrode name").upper()
    return LEGACY_NAME_ALIASES.get(name, name)


def resolve_common17_names(observed_channel_ids: Sequence[object]) -> dict[str, Any]:
    """Resolve common17 by exact names, rejecting absent or duplicate endpoints."""

    if not isinstance(observed_channel_ids, Sequence) or isinstance(
        observed_channel_ids, (str, bytes)
    ):
        raise TypeError("observed channel IDs must be a sequence")
    normalized_to_source: dict[str, str] = {}
    normalized_observed: list[str] = []
    for raw in observed_channel_ids:
        source = _identifier(raw, "observed channel ID").upper()
        canonical = normalize_electrode_name(source)
        normalized_observed.append(source)
        if canonical in COMMON17_SENSOR_ONTOLOGY_ORDER:
            if canonical in normalized_to_source:
                raise ValueError(f"common17 channel {canonical} is duplicated")
            normalized_to_source[canonical] = source
    missing = sorted(set(COMMON17_SENSOR_ONTOLOGY_ORDER).difference(normalized_to_source))
    if missing:
        raise ValueError(f"common17 channels are missing: {missing}")
    source_names = [normalized_to_source[name] for name in COMMON17_SENSOR_ONTOLOGY_ORDER]
    receipt_payload = {
        "observed_channel_ids": normalized_observed,
        "common17_available_names": list(COMMON17_SENSOR_ONTOLOGY_ORDER),
        "common17_source_names": source_names,
        "legacy_name_aliases": LEGACY_NAME_ALIASES,
        "resolution": "exact_normalized_electrode_name_only",
    }
    return {
        "common17_available_names": list(COMMON17_SENSOR_ONTOLOGY_ORDER),
        "common17_source_names": source_names,
        "channel_resolution_sha256": canonical_sha256(receipt_payload),
    }


def lb16_pairs_by_name() -> tuple[tuple[str, str], ...]:
    return tuple(tuple(unit.split("-", 1)) for unit in CANONICAL_ST16_TYPED_UNITS)  # type: ignore[return-value]


def derive_lb16_from_named_values(values_by_name: Mapping[str, float]) -> tuple[float, ...]:
    """Small dependency-free name transform used by contract tests/audits."""

    normalized: dict[str, float] = {}
    for raw_name, raw_value in values_by_name.items():
        name = normalize_electrode_name(raw_name)
        if name in normalized:
            raise ValueError(f"named value repeats {name}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError("named values must be finite numbers")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("named values must be finite")
        normalized[name] = value
    if set(normalized) != set(COMMON17_SENSOR_ONTOLOGY_ORDER):
        raise ValueError("named values must contain exact common17")
    return tuple(normalized[left] - normalized[right] for left, right in lb16_pairs_by_name())


def resolve_source_train_target_path(
    source_train_root: str | Path,
    local_edf_path: str,
) -> tuple[Path, str]:
    """Resolve a train sidecar; dev/eval paths fail before filesystem access."""

    local = PurePosixPath(_identifier(local_edf_path, "local EDF path"))
    if (
        local.is_absolute()
        or ".." in local.parts
        or "\\" in local_edf_path
        or local.suffix.lower() != ".edf"
        or not local.parts
        or local.parts[0] != "train"
    ):
        raise PermissionError("target resolver accepts source-train EDF paths only")
    root_path = Path(source_train_root)
    if not root_path.is_dir():
        raise ValueError("source-train target root is not a directory")
    root = root_path.resolve(strict=True)
    relative_under_train = PurePosixPath(*local.parts[1:]).with_suffix(".csv_bi")
    candidate = root.joinpath(*relative_under_train.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"source-train target sidecar is absent or unsafe: {candidate}")
    resolved = candidate.resolve(strict=True)
    if root != resolved.parent and root not in resolved.parents:
        raise ValueError("source-train target sidecar escapes its root")
    return resolved, str(local.with_suffix(".csv_bi"))


def _decimal(value: str, context: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{context} is not decimal") from error
    if not parsed.is_finite():
        raise ValueError(f"{context} is not finite")
    return parsed


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    result = format(value.normalize(), "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def parse_source_train_global_term_seiz(
    payload: bytes,
    *,
    recording_duration: Fraction,
) -> tuple[list[dict[str, str]], str]:
    """Parse only exact global TERM/seiz rows from a UTF-8 TUSZ sidecar."""

    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("source-train target sidecar is not UTF-8") from error
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("source-train target sidecar has no table")
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    expected_fields = ["channel", "start_time", "stop_time", "label", "confidence"]
    if reader.fieldnames != expected_fields:
        raise ValueError("source-train target sidecar columns drifted")
    duration_decimal = Decimal(recording_duration.numerator) / Decimal(
        recording_duration.denominator
    )
    intervals: list[dict[str, str]] = []
    for index, row in enumerate(reader):
        if set(row) != set(expected_fields) or any(value is None for value in row.values()):
            raise ValueError(f"malformed target row {index}")
        if row["channel"] != "TERM" or row["label"] != "seiz":
            continue
        start = _decimal(row["start_time"], f"target row {index} start")
        stop = _decimal(row["stop_time"], f"target row {index} stop")
        confidence = _decimal(row["confidence"], f"target row {index} confidence")
        if start < 0 or stop <= start or stop > duration_decimal:
            raise ValueError(f"target row {index} interval is outside recording")
        if confidence < 0 or confidence > 1:
            raise ValueError(f"target row {index} confidence is outside [0,1]")
        interval = {
            "start_seconds_decimal": _decimal_text(start),
            "stop_seconds_decimal": _decimal_text(stop),
            "confidence_decimal": _decimal_text(confidence),
        }
        if intervals and _decimal(intervals[-1]["stop_seconds_decimal"], "prior stop") > start:
            raise ValueError("global TERM,seiz intervals overlap or are unsorted")
        intervals.append(interval)
    parse_receipt = canonical_sha256(
        {
            "parser": "exact_channel_TERM_and_exact_label_seiz_decimal_v1",
            "sidecar_sha256": sha256_bytes(payload),
            "recording_duration_seconds_fraction": _fraction_json(recording_duration),
            "intervals": intervals,
        }
    )
    return intervals, parse_receipt


def _validate_channel_contract(value: object) -> dict[str, Any]:
    required = {
        "common17_sensor_ontology_order",
        "st16_common17_referential_axis_order",
        "legacy_name_aliases",
        "st16_lb16_typed_units",
        "resolution",
        "positional_axis_slicing_allowed",
        "missing_or_duplicate_common17_name_policy",
        "FZ_or_PZ_enter_model_tensor",
        "FZ_or_PZ_signal_imputation_interpolation_or_zero_fill",
    }
    contract = _strict_object(value, required, "channel contract")
    if tuple(contract["common17_sensor_ontology_order"]) != COMMON17_SENSOR_ONTOLOGY_ORDER:
        raise ValueError("common17 sensor ontology drifted")
    if tuple(contract["st16_common17_referential_axis_order"]) != tuple(
        COMMON17_REFERENTIAL_AXIS_ORDER
    ):
        raise ValueError("ST16 common17 referential axis order drifted")
    if contract["legacy_name_aliases"] != LEGACY_NAME_ALIASES:
        raise ValueError("legacy electrode aliases drifted")
    if tuple(contract["st16_lb16_typed_units"]) != tuple(CANONICAL_ST16_TYPED_UNITS):
        raise ValueError("LB16 endpoint roster drifted")
    if any("FZ" in unit or "PZ" in unit for unit in contract["st16_lb16_typed_units"]):
        raise ValueError("FZ/PZ entered LB16")
    if (
        contract["resolution"]
        != "exact_normalized_electrode_name_and_explicit_lb16_endpoints_only"
        or contract["positional_axis_slicing_allowed"] is not False
        or contract["missing_or_duplicate_common17_name_policy"] != "technical_failure"
        or contract["FZ_or_PZ_enter_model_tensor"] is not False
        or contract["FZ_or_PZ_signal_imputation_interpolation_or_zero_fill"] is not False
    ):
        raise ValueError("channel safety policy drifted")
    return contract


def _validate_prediction_output_contract(value: object) -> dict[str, Any]:
    required = {
        "allowed_terminal_states",
        "exactly_one_terminal_state_per_roster_record",
        "completed_zero_candidate_requires_complete_eeg_coverage",
        "partial_or_technical_failure_may_be_reclassified_zero_candidate",
        "partial_or_technical_failure_rows_may_be_dropped",
        "primary_admission",
    }
    contract = _strict_object(value, required, "prediction output contract")
    if tuple(contract["allowed_terminal_states"]) != PREDICTION_TERMINAL_STATES:
        raise ValueError("prediction terminal-state domain drifted")
    if (
        contract["exactly_one_terminal_state_per_roster_record"] is not True
        or contract["completed_zero_candidate_requires_complete_eeg_coverage"] is not True
        or contract["partial_or_technical_failure_may_be_reclassified_zero_candidate"] is not False
        or contract["partial_or_technical_failure_rows_may_be_dropped"] is not False
        or contract["primary_admission"]
        != {
            "physical_record_coverage_fraction": 1.0,
            "partial_coverage_count": 0,
            "technical_failure_count": 0,
        }
    ):
        raise ValueError("prediction completion/failure policy drifted")
    return contract


def validate_config(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "status",
        "dataset",
        "authorities",
        "expected_denominators",
        "channel_contract",
        "target_access_contract",
        "source_dev_prediction_output_contract",
        "claim_limits",
    }
    config = _strict_object(value, required, "physical isolation config")
    if config["schema_version"] != CONFIG_SCHEMA_VERSION or config["method_id"] != METHOD_ID:
        raise ValueError("physical isolation config schema/method drifted")
    _validate_channel_contract(config["channel_contract"])
    _validate_prediction_output_contract(config["source_dev_prediction_output_contract"])
    expected = config["expected_denominators"]
    if set(expected) != {"source_train", "source_dev", "source_eval"}:
        raise ValueError("expected split denominator roster drifted")
    fixed = {
        "source_train": (4664, 579),
        "source_dev": (1821, 53),
        "source_eval": (864, 43),
    }
    for split, (records, patients) in fixed.items():
        row = expected[split]
        if row["recording_count"] != records or row["patient_count"] != patients:
            raise ValueError(f"{split} denominator drifted")
        _fraction(row["duration_seconds_fraction"], f"{split} duration", positive=True)
        if not _is_sha256(row["analysis_identity_roster_sha256"]):
            raise ValueError(f"{split} roster hash is invalid")
    access = config["target_access_contract"]
    if (
        access["fit_target_scope"] != "source_train_global_TERM_exact_seiz_rows_only"
        or access["source_train_sidecar_suffix"] != ".csv_bi"
        or access["source_dev_sidecar_path_resolution_allowed"] is not False
        or access["source_dev_sidecar_open_allowed"] is not False
        or access["source_eval_in_any_output_allowed"] is not False
        or access["opened_path_roster_and_each_sidecar_byte_hash_required"] is not True
    ):
        raise ValueError("target access contract drifted")
    claims = config["claim_limits"]
    if (
        claims["this_contract_trains_a_detector"] is not False
        or claims["this_contract_runs_source_dev_inference"] is not False
        or claims["this_contract_estimates_performance"] is not False
        or claims["source_eval_opened"] is not False
        or claims["clinical_use_authorized"] is not False
        or claims["exact_duplicate_scope_only"] is not True
        or claims["near_duplicate_or_partial_overlap_audit_complete"] is not False
    ):
        raise ValueError("claim boundary drifted")
    return config


def _workspace_file(relative: object) -> Path:
    path = Path(_identifier(relative, "workspace artifact path"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("workspace artifact path is unsafe")
    resolved = (ROOT / path).resolve(strict=True)
    if ROOT != resolved.parent and ROOT not in resolved.parents:
        raise ValueError("workspace artifact escapes repository")
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("workspace artifact is not a regular file")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def load_bound_authorities(config: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    expected_names = {
        "canonical_physical_audit",
        "canonical_physical_projection",
        "target_blind_fold_plan",
    }
    if set(config["authorities"]) != expected_names:
        raise ValueError("authority roster drifted")
    values: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, str]] = {}
    for name in sorted(expected_names):
        binding = config["authorities"][name]
        if set(binding) != {"path", "file_sha256", "embedded_receipt_sha256"}:
            raise ValueError(f"authority binding fields drifted: {name}")
        path = _workspace_file(binding["path"])
        observed = sha256_file(path)
        if observed != binding["file_sha256"]:
            raise ValueError(f"authority file hash drifted: {name}")
        value = _load_json(path)
        if value.get("receipt_sha256") != binding["embedded_receipt_sha256"]:
            raise ValueError(f"authority embedded receipt drifted: {name}")
        values[name] = value
        receipts[name] = {
            "path": str(Path(binding["path"])),
            "file_sha256": observed,
            "embedded_receipt_sha256": str(value["receipt_sha256"]),
        }
    validate_tusz_canonical_physical_duplicate_audit_v1(
        values["canonical_physical_audit"]
    )
    validate_tusz_canonical_physical_analysis_projection_v1(
        values["canonical_physical_projection"]
    )
    validate_tusz_detector_cleanroom_fold_plan_v1(values["target_blind_fold_plan"])
    projection_binding = values["canonical_physical_projection"]["source_binding"]
    audit = values["canonical_physical_audit"]
    if (
        projection_binding["source_canonical_physical_audit_id"] != audit["audit_id"]
        or projection_binding["source_canonical_physical_audit_receipt_sha256"]
        != audit["receipt_sha256"]
    ):
        raise ValueError("physical projection does not bind the audited physical source")
    plan_binding = values["target_blind_fold_plan"]["source_binding"]
    projection = values["canonical_physical_projection"]
    if (
        plan_binding["source_canonical_physical_projection_id"] != projection["projection_id"]
        or plan_binding["source_canonical_physical_projection_receipt_sha256"]
        != projection["receipt_sha256"]
        or plan_binding["source_canonical_physical_audit_id"] != audit["audit_id"]
        or plan_binding["source_canonical_physical_audit_receipt_sha256"]
        != audit["receipt_sha256"]
    ):
        raise ValueError("fold plan does not bind the supplied physical authorities")
    return values, receipts


def _source_binding(
    authorities: Mapping[str, Mapping[str, Any]],
    authority_receipts: Mapping[str, Mapping[str, str]],
    channel_contract: Mapping[str, Any],
) -> dict[str, str]:
    audit = authorities["canonical_physical_audit"]
    projection = authorities["canonical_physical_projection"]
    plan = authorities["target_blind_fold_plan"]
    return {
        "physical_audit_id": str(audit["audit_id"]),
        "physical_audit_receipt_sha256": str(audit["receipt_sha256"]),
        "physical_audit_file_sha256": authority_receipts["canonical_physical_audit"][
            "file_sha256"
        ],
        "physical_projection_id": str(projection["projection_id"]),
        "physical_projection_receipt_sha256": str(projection["receipt_sha256"]),
        "physical_projection_file_sha256": authority_receipts[
            "canonical_physical_projection"
        ]["file_sha256"],
        "fold_plan_id": str(plan["plan_id"]),
        "fold_plan_receipt_sha256": str(plan["receipt_sha256"]),
        "fold_plan_file_sha256": authority_receipts["target_blind_fold_plan"][
            "file_sha256"
        ],
        "channel_contract_sha256": canonical_sha256(channel_contract),
    }


def _base_identity_row(
    projection_row: Mapping[str, Any],
    physical_outcome: Mapping[str, Any],
    duration_row: Mapping[str, Any],
    *,
    channel_contract_sha256: str,
) -> dict[str, Any]:
    identity = projection_row["analysis_identity_id"]
    if (
        physical_outcome["analysis_identity_id"] != identity
        or physical_outcome["terminal_status"] != "success"
        or physical_outcome["failure"] is not None
    ):
        raise ValueError("canonical physical outcome is absent or failed")
    signal = physical_outcome["physical_signal"]
    if (
        physical_outcome["container_sha256_recomputed"]
        != projection_row["source_edf_container_sha256"]
        or signal["canonical_source_tensor_sha256"]
        != projection_row["canonical_physical_source_tensor_sha256"]
        or "TUSZPHYS-" + signal["canonical_physical_equivalence_sha256"]
        != projection_row["canonical_physical_equivalence_id"]
        or signal["duration_seconds_fraction"]
        != duration_row["recording_duration_seconds_fraction"]
    ):
        raise ValueError("canonical physical row lineage drifted")
    resolved = resolve_common17_names(signal["observed_channel_ids"])
    base = {
        "analysis_identity_id": identity,
        "model_split": projection_row["model_split"],
        "official_split": projection_row["official_split"],
        "local_patient_id": projection_row["local_patient_id"],
        "local_edf_path": projection_row["local_edf_path"],
        "source_edf_container_sha256": projection_row["source_edf_container_sha256"],
        "canonical_physical_equivalence_id": projection_row[
            "canonical_physical_equivalence_id"
        ],
        "canonical_physical_source_tensor_sha256": projection_row[
            "canonical_physical_source_tensor_sha256"
        ],
        "canonical_source_signal_sha256": signal["canonical_source_signal_sha256"],
        "canonical_source_header_receipt_sha256": signal[
            "canonical_source_header_receipt_sha256"
        ],
        "source_physical_identity_multiplicity": projection_row[
            "source_physical_identity_multiplicity"
        ],
        "recording_duration_seconds_fraction": duration_row[
            "recording_duration_seconds_fraction"
        ],
        "sampling_rate_hz_fraction": signal["sampling_rate_fraction"],
        "sample_count": signal["sample_count"],
        **resolved,
        "channel_contract_sha256": channel_contract_sha256,
    }
    return base


def _validate_identity_row(row: object, *, split: str, fields: frozenset[str]) -> dict[str, Any]:
    value = _strict_object(row, fields, f"{split} record")
    _validate_row_hash(value, f"{split} record")
    official = {"source_train": "train", "source_dev": "dev"}[split]
    if value["model_split"] != split or value["official_split"] != official:
        raise ValueError(f"{split} record split mapping drifted")
    patient = _identifier(value["local_patient_id"], "patient ID")
    local_path = PurePosixPath(_identifier(value["local_edf_path"], "EDF path"))
    if (
        local_path.is_absolute()
        or ".." in local_path.parts
        or not local_path.parts
        or local_path.parts[0] != official
        or len(local_path.parts) < 3
        or local_path.parts[1] != patient
        or local_path.suffix.lower() != ".edf"
    ):
        raise ValueError(f"{split} EDF path binding drifted")
    for field in (
        "source_edf_container_sha256",
        "canonical_physical_source_tensor_sha256",
        "canonical_source_signal_sha256",
        "canonical_source_header_receipt_sha256",
        "channel_resolution_sha256",
        "channel_contract_sha256",
    ):
        if not _is_sha256(value[field]):
            raise ValueError(f"{split} {field} is invalid")
    if value["analysis_identity_id"] != "TUSZANALYSIS-" + value["source_edf_container_sha256"]:
        raise ValueError(f"{split} analysis/container identity drifted")
    physical_id = value["canonical_physical_equivalence_id"]
    if (
        not isinstance(physical_id, str)
        or not physical_id.startswith("TUSZPHYS-")
        or not _is_sha256(physical_id[len("TUSZPHYS-") :])
    ):
        raise ValueError(f"{split} physical identity is invalid")
    if type(value["source_physical_identity_multiplicity"]) is not int or value[
        "source_physical_identity_multiplicity"
    ] < 1:
        raise ValueError(f"{split} physical multiplicity is invalid")
    _fraction(value["recording_duration_seconds_fraction"], "recording duration", positive=True)
    _fraction(value["sampling_rate_hz_fraction"], "sampling rate", positive=True)
    if type(value["sample_count"]) is not int or value["sample_count"] < 1:
        raise ValueError(f"{split} sample count is invalid")
    if tuple(value["common17_available_names"]) != COMMON17_SENSOR_ONTOLOGY_ORDER:
        raise ValueError(f"{split} common17 availability drifted")
    if len(value["common17_source_names"]) != len(COMMON17_SENSOR_ONTOLOGY_ORDER):
        raise ValueError(f"{split} common17 source-name roster length drifted")
    resolved = [normalize_electrode_name(item) for item in value["common17_source_names"]]
    if tuple(resolved) != COMMON17_SENSOR_ONTOLOGY_ORDER or len(set(resolved)) != 17:
        raise ValueError(f"{split} common17 source-name mapping drifted")
    return value


def build_source_train_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    source_binding: Mapping[str, str],
    channel_contract: Mapping[str, Any],
    target_root: str,
) -> dict[str, Any]:
    rows = sorted((deepcopy(dict(row)) for row in records), key=lambda row: row["local_edf_path"])
    paths = [row["target_sidecar_relative_path"] for row in rows]
    hashes = [
        {"path": row["target_sidecar_relative_path"], "sha256": row["target_sidecar_sha256"]}
        for row in rows
    ]
    interval_count = sum(len(row["global_TERM_seiz_intervals_seconds"]) for row in rows)
    positive_duration = Fraction(0, 1)
    positive_records = 0
    for row in rows:
        intervals = row["global_TERM_seiz_intervals_seconds"]
        positive_records += bool(intervals)
        for interval in intervals:
            positive_duration += Fraction(Decimal(interval["stop_seconds_decimal"])) - Fraction(
                Decimal(interval["start_seconds_decimal"])
            )
    identities = [row["analysis_identity_id"] for row in rows]
    patients = sorted({row["local_patient_id"] for row in rows})
    total_duration = sum(
        (_fraction(row["recording_duration_seconds_fraction"], "train duration", positive=True) for row in rows),
        Fraction(0, 1),
    )
    body = {
        "schema_version": TRAIN_SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "split": "source_train",
        "source_binding": deepcopy(dict(source_binding)),
        "channel_contract": deepcopy(dict(channel_contract)),
        "target_access_receipt": {
            "source_train_target_root": target_root,
            "opened_file_count": len(rows),
            "source_train_files_opened": len(rows),
            "source_dev_files_opened": 0,
            "source_eval_files_opened": 0,
            "opened_relative_path_roster_sha256": canonical_sha256(paths),
            "opened_file_hash_binding_sha256": canonical_sha256(hashes),
        },
        "inventory": {
            "recording_count": len(rows),
            "patient_count": len(patients),
            "recording_with_positive_interval_count": positive_records,
            "global_interval_count": interval_count,
            "total_positive_duration_seconds_fraction": _fraction_json(positive_duration),
            "total_recording_duration_seconds_fraction": _fraction_json(total_duration),
            "analysis_identity_roster_sha256": canonical_sha256(sorted(identities)),
            "patient_roster_sha256": canonical_sha256(patients),
        },
        "records": rows,
        "content_sha256": PENDING,
    }
    return _content_address(body)


def build_source_dev_roster(
    records: Sequence[Mapping[str, Any]],
    *,
    source_binding: Mapping[str, str],
    channel_contract: Mapping[str, Any],
    prediction_output_contract: Mapping[str, Any],
) -> dict[str, Any]:
    rows = sorted((deepcopy(dict(row)) for row in records), key=lambda row: row["local_edf_path"])
    identities = [row["analysis_identity_id"] for row in rows]
    patients = sorted({row["local_patient_id"] for row in rows})
    total_duration = sum(
        (_fraction(row["recording_duration_seconds_fraction"], "dev duration", positive=True) for row in rows),
        Fraction(0, 1),
    )
    body = {
        "schema_version": DEV_SCHEMA_VERSION,
        "method_id": "canonical_physical_source_dev_eeg_only_prediction_v1",
        "split": "source_dev",
        "source_binding": deepcopy(dict(source_binding)),
        "channel_contract": deepcopy(dict(channel_contract)),
        "prediction_output_contract": deepcopy(dict(prediction_output_contract)),
        "inventory": {
            "recording_count": len(rows),
            "patient_count": len(patients),
            "total_recording_duration_seconds_fraction": _fraction_json(total_duration),
            "analysis_identity_roster_sha256": canonical_sha256(sorted(identities)),
            "patient_roster_sha256": canonical_sha256(patients),
        },
        "records": rows,
        "content_sha256": PENDING,
    }
    result = _content_address(body)
    assert_source_dev_eeg_only(result)
    return result


def validate_source_train_manifest(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "method_id",
        "split",
        "source_binding",
        "channel_contract",
        "target_access_receipt",
        "inventory",
        "records",
        "content_sha256",
    }
    value = _strict_object(payload, fields, "source-train manifest")
    if value["schema_version"] != TRAIN_SCHEMA_VERSION or value["method_id"] != METHOD_ID:
        raise ValueError("source-train manifest schema/method drifted")
    _validate_content_address(value, "source-train manifest")
    _validate_channel_contract(value["channel_contract"])
    if value["split"] != "source_train" or not isinstance(value["records"], list):
        raise ValueError("source-train manifest split/records drifted")
    rows = [
        _validate_identity_row(row, split="source_train", fields=TRAIN_ROW_FIELDS)
        for row in value["records"]
    ]
    if rows != sorted(rows, key=lambda row: row["local_edf_path"]):
        raise ValueError("source-train rows are not canonically sorted")
    for row in rows:
        if type(row["held_out_fold_id"]) is not int or row["held_out_fold_id"] not in range(5):
            raise ValueError("source-train held-out fold is invalid")
        target_path = PurePosixPath(row["target_sidecar_relative_path"])
        if (
            target_path.is_absolute()
            or not target_path.parts
            or target_path.parts[0] != "train"
            or target_path.suffix != ".csv_bi"
        ):
            raise ValueError("source-train target path is outside train")
        if not _is_sha256(row["target_sidecar_sha256"]) or not _is_sha256(
            row["target_parse_sha256"]
        ):
            raise ValueError("source-train target hash is invalid")
        intervals = row["global_TERM_seiz_intervals_seconds"]
        if not isinstance(intervals, list):
            raise ValueError("source-train global intervals are not a list")
        prior = Decimal("-1")
        duration = _fraction(row["recording_duration_seconds_fraction"], "duration", positive=True)
        duration_decimal = Decimal(duration.numerator) / Decimal(duration.denominator)
        for interval in intervals:
            if set(interval) != {
                "start_seconds_decimal",
                "stop_seconds_decimal",
                "confidence_decimal",
            }:
                raise ValueError("source-train interval fields drifted")
            start = _decimal(interval["start_seconds_decimal"], "interval start")
            stop = _decimal(interval["stop_seconds_decimal"], "interval stop")
            confidence = _decimal(interval["confidence_decimal"], "confidence")
            if start < 0 or start < prior or stop <= start or stop > duration_decimal:
                raise ValueError("source-train interval geometry drifted")
            if confidence < 0 or confidence > 1:
                raise ValueError("source-train interval confidence drifted")
            prior = stop
    rebuilt = build_source_train_manifest(
        rows,
        source_binding=value["source_binding"],
        channel_contract=value["channel_contract"],
        target_root=value["target_access_receipt"]["source_train_target_root"],
    )
    if rebuilt != value:
        raise ValueError("source-train manifest is not internally replayable")
    return value


def _walk_json(value: object, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], object]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_json(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, (*path, str(index)))


def assert_source_dev_eeg_only(payload: object) -> None:
    for path, value in _walk_json(payload):
        if path:
            key = path[-1]
            lower_key = key.lower()
            if key.upper() == "TERM" or any(
                fragment in lower_key for fragment in DEV_FORBIDDEN_KEY_FRAGMENTS
            ):
                raise PermissionError(f"source-dev contains target-bearing key: {'.'.join(path)}")
        if isinstance(value, str):
            lower_value = value.lower()
            if value.upper() == "TERM" or any(
                fragment in lower_value for fragment in DEV_FORBIDDEN_VALUE_FRAGMENTS
            ):
                raise PermissionError(f"source-dev contains target-bearing value at {'.'.join(path)}")


def validate_source_dev_roster(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "method_id",
        "split",
        "source_binding",
        "channel_contract",
        "prediction_output_contract",
        "inventory",
        "records",
        "content_sha256",
    }
    value = _strict_object(payload, fields, "source-dev roster")
    assert_source_dev_eeg_only(value)
    if (
        value["schema_version"] != DEV_SCHEMA_VERSION
        or value["method_id"] != "canonical_physical_source_dev_eeg_only_prediction_v1"
        or value["split"] != "source_dev"
    ):
        raise ValueError("source-dev roster schema/method/split drifted")
    _validate_content_address(value, "source-dev roster")
    _validate_channel_contract(value["channel_contract"])
    _validate_prediction_output_contract(value["prediction_output_contract"])
    if not isinstance(value["records"], list):
        raise ValueError("source-dev records are not a list")
    rows = [
        _validate_identity_row(row, split="source_dev", fields=DEV_ROW_FIELDS)
        for row in value["records"]
    ]
    if rows != sorted(rows, key=lambda row: row["local_edf_path"]):
        raise ValueError("source-dev rows are not canonically sorted")
    rebuilt = build_source_dev_roster(
        rows,
        source_binding=value["source_binding"],
        channel_contract=value["channel_contract"],
        prediction_output_contract=value["prediction_output_contract"],
    )
    if rebuilt != value:
        raise ValueError("source-dev roster is not internally replayable")
    return value


def validate_split_isolation(
    train_manifest: Mapping[str, Any], dev_roster: Mapping[str, Any]
) -> dict[str, Any]:
    train_rows = train_manifest["records"]
    dev_rows = dev_roster["records"]
    dimensions = {
        "patient": "local_patient_id",
        "analysis_identity": "analysis_identity_id",
        "physical_equivalence": "canonical_physical_equivalence_id",
        "physical_tensor": "canonical_physical_source_tensor_sha256",
        "edf_path": "local_edf_path",
        "edf_container": "source_edf_container_sha256",
    }
    intersections: dict[str, int] = {}
    for name, field in dimensions.items():
        left = {row[field] for row in train_rows}
        right = {row[field] for row in dev_rows}
        intersections[name] = len(left.intersection(right))
        if intersections[name]:
            raise ValueError(f"train/dev {name} overlap is nonzero")
    unique_within_split_fields = (
        "analysis_identity_id",
        "canonical_physical_equivalence_id",
        "canonical_physical_source_tensor_sha256",
        "local_edf_path",
        "source_edf_container_sha256",
    )
    for rows, split in ((train_rows, "source_train"), (dev_rows, "source_dev")):
        for field in unique_within_split_fields:
            values = [row[field] for row in rows]
            if len(values) != len(set(values)):
                raise ValueError(f"{split} repeats {field}")
    return {
        "train_dev_intersection_counts": intersections,
        "all_required_intersections_zero": all(value == 0 for value in intersections.values()),
    }


def validate_prediction_terminal_inventory(
    dev_roster: Mapping[str, Any], outcomes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate a later provider's one-row-per-record completion inventory."""

    roster = validate_source_dev_roster(dev_roster)
    required = {
        "analysis_identity_id",
        "terminal_state",
        "observed_sample_count",
        "expected_sample_count",
        "candidate_count",
        "failure_code",
    }
    expected_by_id = {
        row["analysis_identity_id"]: row["sample_count"] for row in roster["records"]
    }
    seen: set[str] = set()
    state_counts = {state: 0 for state in PREDICTION_TERMINAL_STATES}
    for raw in outcomes:
        row = _strict_object(raw, required, "prediction completion row")
        identity = row["analysis_identity_id"]
        if identity not in expected_by_id or identity in seen:
            raise ValueError("prediction completion identity is absent or duplicated")
        seen.add(identity)
        state = row["terminal_state"]
        if state not in PREDICTION_TERMINAL_STATES:
            raise ValueError("prediction completion terminal state is invalid")
        state_counts[state] += 1
        observed = row["observed_sample_count"]
        expected = row["expected_sample_count"]
        candidates = row["candidate_count"]
        if (
            type(observed) is not int
            or type(expected) is not int
            or type(candidates) is not int
            or observed < 0
            or expected != expected_by_id[identity]
            or candidates < 0
        ):
            raise ValueError("prediction completion counters are invalid")
        if state in {"completed_with_candidates", "completed_zero_candidate"}:
            if observed != expected or row["failure_code"] is not None:
                raise ValueError("completed prediction does not cover the full EEG")
        if state == "completed_with_candidates" and candidates < 1:
            raise ValueError("completed-with-candidates row has no candidate")
        if state == "completed_zero_candidate" and candidates != 0:
            raise ValueError("zero-candidate row carries a candidate")
        if state == "partial_coverage":
            if observed >= expected or row["failure_code"] is None:
                raise ValueError("partial row is not explicitly partial")
        if state == "technical_failure" and row["failure_code"] is None:
            raise ValueError("technical failure has no failure code")
    if seen != set(expected_by_id):
        raise ValueError("prediction completion inventory silently dropped roster rows")
    admitted = (
        state_counts["partial_coverage"] == 0
        and state_counts["technical_failure"] == 0
        and len(seen) == len(expected_by_id)
    )
    return {
        "roster_record_count": len(expected_by_id),
        "completion_record_count": len(seen),
        "terminal_state_counts": state_counts,
        "physical_record_coverage_fraction": len(seen) / len(expected_by_id),
        "primary_completeness_admitted": admitted,
    }


def materialize_contract(
    config: Mapping[str, Any],
    authorities: Mapping[str, Mapping[str, Any]],
    authority_receipts: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    projection = authorities["canonical_physical_projection"]
    audit = authorities["canonical_physical_audit"]
    plan = authorities["target_blind_fold_plan"]
    projection_by_id = {
        row["analysis_identity_id"]: row for row in projection["records"]
    }
    outcome_by_id = {row["analysis_identity_id"]: row for row in audit["outcomes"]}
    duration_by_id = {
        row["analysis_identity_id"]: row for row in plan["source_record_duration_rows"]
    }
    fold_by_patient = {
        row["local_patient_id"]: row["held_out_fold_id"]
        for row in plan["patient_fold_assignments"]
    }
    channel_contract = _validate_channel_contract(config["channel_contract"])
    channel_hash = canonical_sha256(channel_contract)
    binding = _source_binding(authorities, authority_receipts, channel_contract)
    target_root = config["dataset"]["source_train_target_root"]

    train_records: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []
    for split in ("source_train", "source_dev"):
        split_roster = plan["source_split_rosters"][split]
        for identity in split_roster["analysis_identity_ids"]:
            projection_row = projection_by_id.get(identity)
            outcome = outcome_by_id.get(identity)
            duration_row = duration_by_id.get(identity)
            if projection_row is None or outcome is None or duration_row is None:
                raise ValueError(f"{split} canonical identity is absent from an authority")
            if projection_row["model_split"] != split or duration_row["model_split"] != split:
                raise ValueError(f"{split} canonical identity split drifted")
            base = _base_identity_row(
                projection_row,
                outcome,
                duration_row,
                channel_contract_sha256=channel_hash,
            )
            if split == "source_train":
                path, relative = resolve_source_train_target_path(
                    target_root, projection_row["local_edf_path"]
                )
                payload = path.read_bytes()
                intervals, parse_hash = parse_source_train_global_term_seiz(
                    payload,
                    recording_duration=_fraction(
                        base["recording_duration_seconds_fraction"],
                        "train recording duration",
                        positive=True,
                    ),
                )
                train_records.append(
                    _attach_row_hash(
                        {
                            **base,
                            "held_out_fold_id": fold_by_patient[
                                projection_row["local_patient_id"]
                            ],
                            "target_sidecar_relative_path": relative,
                            "target_sidecar_sha256": sha256_bytes(payload),
                            "global_TERM_seiz_intervals_seconds": intervals,
                            "target_parse_sha256": parse_hash,
                        }
                    )
                )
            else:
                dev_records.append(_attach_row_hash(base))

    train = build_source_train_manifest(
        train_records,
        source_binding=binding,
        channel_contract=channel_contract,
        target_root=str(Path(target_root).resolve(strict=True)),
    )
    dev = build_source_dev_roster(
        dev_records,
        source_binding=binding,
        channel_contract=channel_contract,
        prediction_output_contract=config["source_dev_prediction_output_contract"],
    )
    validate_source_train_manifest(train)
    validate_source_dev_roster(dev)
    isolation = validate_split_isolation(train, dev)
    audit_summary = audit_against_authorities(
        train,
        dev,
        config=config,
        authorities=authorities,
        isolation=isolation,
    )
    return train, dev, audit_summary


def audit_against_authorities(
    train_manifest: Mapping[str, Any],
    dev_roster: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    authorities: Mapping[str, Mapping[str, Any]],
    isolation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    train = validate_source_train_manifest(train_manifest)
    dev = validate_source_dev_roster(dev_roster)
    isolation_receipt = (
        deepcopy(dict(isolation)) if isolation is not None else validate_split_isolation(train, dev)
    )
    projection = authorities["canonical_physical_projection"]
    plan = authorities["target_blind_fold_plan"]
    expected_by_split = {
        split: {
            row["analysis_identity_id"]
            for row in projection["records"]
            if row["model_split"] == split
        }
        for split in ("source_train", "source_dev", "source_eval")
    }
    train_ids = {row["analysis_identity_id"] for row in train["records"]}
    dev_ids = {row["analysis_identity_id"] for row in dev["records"]}
    if train_ids != expected_by_split["source_train"] or dev_ids != expected_by_split[
        "source_dev"
    ]:
        raise ValueError("materialized train/dev rosters differ from physical projection")
    if (train_ids | dev_ids).intersection(expected_by_split["source_eval"]):
        raise ValueError("source-eval entered a materialized output")
    assignment = {
        row["local_patient_id"]: row["held_out_fold_id"]
        for row in plan["patient_fold_assignments"]
    }
    if any(
        row["held_out_fold_id"] != assignment.get(row["local_patient_id"])
        for row in train["records"]
    ):
        raise ValueError("source-train held-out fold mapping drifted")
    channel_hash = canonical_sha256(config["channel_contract"])
    if any(row["channel_contract_sha256"] != channel_hash for row in train["records"]):
        raise ValueError("source-train channel contract binding drifted")
    if any(row["channel_contract_sha256"] != channel_hash for row in dev["records"]):
        raise ValueError("source-dev channel contract binding drifted")
    for split, artifact in (("source_train", train), ("source_dev", dev)):
        expected = config["expected_denominators"][split]
        if (
            artifact["inventory"]["recording_count"] != expected["recording_count"]
            or artifact["inventory"]["patient_count"] != expected["patient_count"]
            or artifact["inventory"]["total_recording_duration_seconds_fraction"]
            != expected["duration_seconds_fraction"]
            or artifact["inventory"]["analysis_identity_roster_sha256"]
            != expected["analysis_identity_roster_sha256"]
        ):
            raise ValueError(f"{split} production denominator drifted")
    access = train["target_access_receipt"]
    if (
        access["opened_file_count"] != len(train["records"])
        or access["source_train_files_opened"] != len(train["records"])
        or access["source_dev_files_opened"] != 0
        or access["source_eval_files_opened"] != 0
    ):
        raise ValueError("target access accounting drifted")
    assert_source_dev_eeg_only(dev)
    return {
        "source_train_recording_count": len(train["records"]),
        "source_train_patient_count": len(
            {row["local_patient_id"] for row in train["records"]}
        ),
        "source_dev_recording_count": len(dev["records"]),
        "source_dev_patient_count": len(
            {row["local_patient_id"] for row in dev["records"]}
        ),
        "source_eval_recording_count_in_outputs": 0,
        "source_train_target_files_opened": access["source_train_files_opened"],
        "source_dev_target_files_opened": access["source_dev_files_opened"],
        "source_eval_target_files_opened": access["source_eval_files_opened"],
        "common17_missing_or_duplicate_record_count": 0,
        "FZ_or_PZ_in_model_tensor": False,
        "LB16_constructed_by_explicit_names": True,
        "dev_target_bearing_field_or_value_count": 0,
        "source_eval_excluded": True,
        "split_isolation": isolation_receipt,
        "physical_duplicate_scope": {
            "exact_full_observed_tensor_duplicate_audit_bound": True,
            "same_patient_same_split_exact_aliases_deduplicated": True,
            "cross_patient_or_split_exact_duplicates_present": False,
            "common17_projected_near_duplicate_audit_performed": False,
        },
    }


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _receipt_body(
    *,
    config_path: Path,
    authority_receipts: Mapping[str, Mapping[str, str]],
    train_path: Path,
    train: Mapping[str, Any],
    dev_path: Path,
    dev: Mapping[str, Any],
    audit_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "status": "pass_physical_isolation_manifests_materialized_training_not_started",
        "config_binding": {
            "path": str(config_path.relative_to(ROOT)),
            "file_sha256": sha256_file(config_path),
        },
        "authority_bindings": deepcopy(dict(authority_receipts)),
        "artifact_bindings": {
            "source_train_labeled_manifest": {
                "path": str(train_path.relative_to(ROOT)),
                "file_sha256": sha256_file(train_path),
                "content_sha256": train["content_sha256"],
            },
            "source_dev_eeg_only_prediction_roster": {
                "path": str(dev_path.relative_to(ROOT)),
                "file_sha256": sha256_file(dev_path),
                "content_sha256": dev["content_sha256"],
            },
        },
        "audit": deepcopy(dict(audit_summary)),
        "prediction_execution": {
            "detector_trained": False,
            "source_dev_inference_run": False,
            "performance_estimated": False,
            "required_terminal_state_domain": list(PREDICTION_TERMINAL_STATES),
            "primary_completeness_admission_not_yet_assessed": True,
        },
        "claim_limits": {
            "manifest_preparation_only": True,
            "source_eval_opened": False,
            "performance_or_SOTA_claim_authorized": False,
            "clinical_use_authorized": False,
            "near_duplicate_or_partial_overlap_exclusion_claim_authorized": False,
        },
        "content_sha256": PENDING,
    }


def build(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = validate_config(_load_json(config_path))
    authorities, authority_receipts = load_bound_authorities(config)
    train, dev, audit_summary = materialize_contract(
        config, authorities, authority_receipts
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "source_train_labeled_manifest.json"
    dev_path = output_dir / "source_dev_eeg_only_prediction_roster.json"
    receipt_path = output_dir / "receipt.json"
    _atomic_write_json(train_path, train)
    _atomic_write_json(dev_path, dev)
    receipt = _content_address(
        _receipt_body(
            config_path=config_path,
            authority_receipts=authority_receipts,
            train_path=train_path,
            train=train,
            dev_path=dev_path,
            dev=dev,
            audit_summary=audit_summary,
        )
    )
    _atomic_write_json(receipt_path, receipt)
    return receipt


def audit_existing(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = validate_config(_load_json(config_path))
    authorities, authority_receipts = load_bound_authorities(config)
    expected_train, expected_dev, expected_summary = materialize_contract(
        config, authorities, authority_receipts
    )
    train_path = output_dir / "source_train_labeled_manifest.json"
    dev_path = output_dir / "source_dev_eeg_only_prediction_roster.json"
    receipt_path = output_dir / "receipt.json"
    train = validate_source_train_manifest(_load_json(train_path))
    dev = validate_source_dev_roster(_load_json(dev_path))
    if train != expected_train or dev != expected_dev:
        raise ValueError("materialized artifact differs from independent source replay")
    receipt = _load_json(receipt_path)
    _validate_content_address(receipt, "physical isolation receipt")
    expected_receipt = _content_address(
        _receipt_body(
            config_path=config_path,
            authority_receipts=authority_receipts,
            train_path=train_path,
            train=train,
            dev_path=dev_path,
            dev=dev,
            audit_summary=expected_summary,
        )
    )
    if receipt != expected_receipt:
        raise ValueError("physical isolation receipt is not replayable")
    return {
        "status": "pass_independent_replay_exact",
        "receipt_content_sha256": receipt["content_sha256"],
        "source_train_manifest_file_sha256": sha256_file(train_path),
        "source_dev_roster_file_sha256": sha256_file(dev_path),
        **expected_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        child.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve(strict=True)
    output_dir = args.output_dir.resolve() if args.output_dir.exists() else args.output_dir.absolute()
    if ROOT != output_dir and ROOT not in output_dir.parents:
        raise ValueError("output directory must remain inside the workspace")
    result = build(config_path, output_dir) if args.command == "build" else audit_existing(
        config_path, output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
