"""Closed DeepSOZ signal extension for identity-recovered TUSZ records.

This module does not reinterpret DeepSOZ seizure times.  It pins the existing
607-record/988-event causal preflight as an immutable core, selects only rows
whose *original* conservative mapping was ambiguous or unmapped, and anchors
their events to the official local TUSZ ``TERM,seiz`` timeline.  Every new
event is replayed with the same frozen :class:`CausalEDFConfig` used by the
core artifact.

The recovery gate is deliberately identity-only: SOZ values, private data,
and model outputs never choose records or events.  The target-v2 registry is
used only for the already-frozen patient quarantine/split contract and for a
separate fixed-18 target-completeness view.  In particular, signal-eligible
events from a partial-reference patient remain in the target-free receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import csv
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

from . import deepsoz_signal_preflight as _base
from .deepsoz import normalize_patient_id
from .deepsoz_identity_recovery import (
    AUDIT_FILENAME as IDENTITY_AUDIT_FILENAME,
    IDENTITY_RECOVERY_POLICY,
    IDENTITY_RECOVERY_SCHEMA,
    MAPPING_FILENAME as IDENTITY_MAPPING_FILENAME,
)
from .deepsoz_target_v2 import VerifiedDeepSOZTargetV2Artifact
from .edf import (
    CausalEDFConfig,
    EDFEventEligibilityError,
    EDF_PREPROCESS_SCHEMA,
    load_standard19_edf_event,
)
from .tusz import inspect_tusz_annotation_pair
from ..geometry import CHANNEL_INDEX, STANDARD_19


DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_SCHEMA = (
    "soz_deepsoz_signal_identity_recovery_v3"
)
DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_ARTIFACT_SCHEMA = (
    "soz_deepsoz_signal_identity_recovery_artifact_v3"
)
DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_FILENAME = (
    "deepsoz_signal_preflight_identity_v3.json"
)
DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_POLICY = (
    "pinned_v2_core_plus_identity_recovered_local_tusz_timeline_"
    "frozen_causal_replay_no_private_no_model_selection"
)

_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_TIME_TOLERANCE_SEC = 1e-6
_MODEL_SPLITS = ("source_train", "source_dev", "source_eval")
_RECOVERED_ORIGINAL_STATUSES = frozenset({"ambiguous", "unmapped"})
_IDENTITY_AUDIT_COLUMNS = (
    "schema_version",
    "policy",
    "deepsoz_row",
    "deepsoz_patient",
    "deepsoz_record",
    "original_mapping_status",
    "recovery_status",
    "local_patient",
    "relative_edf_path",
    "path_candidate_count",
    "split_match",
    "session_year_match",
    "montage_match",
    "record_key_match",
    "patient_binding_match",
    "source_nsamples",
    "local_sample_count_values",
    "nsamples_match",
    "source_event_count",
    "local_event_count",
    "timeline_class",
    "direct_timeline_max_error_sec",
)
_RECOVERY_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "base_signal_preflight_artifact_sha256",
        "base_signal_preflight_receipt_sha256",
        "identity_audit_sha256",
        "identity_mapping_sha256",
        "event_inputs_sha256",
        "record_crosswalk_sha256",
        "split_manifest_sha256",
        "deepsoz_source_sha256",
        "verified_target_v2_receipt_sha256",
        "verified_target_v2_artifact_sha256",
        "verified_target_v2_policy_sha256",
        "preprocess_schema",
        "preprocess_config",
        "preprocess_config_sha256",
        "source_record_count",
        "identity_recovered_row_ids",
        "identity_recovered_patient_ids",
        "variable_label_patient_ids",
        "base_candidate_event_count",
        "base_eligible_event_count",
        "base_excluded_event_count",
        "base_eligible_patient_count",
        "recovered_candidate_event_ids",
        "recovered_eligible_event_ids",
        "recovered_excluded_event_ids",
        "recovered_candidate_event_count",
        "recovered_eligible_event_count",
        "recovered_excluded_event_count",
        "recovered_eligible_patient_ids",
        "recovered_eligible_split_patient_ids",
        "recovered_exclusion_code_counts",
        "combined_candidate_event_count",
        "combined_eligible_event_count",
        "combined_excluded_event_count",
        "combined_eligible_patient_count",
        "combined_candidate_event_roster_sha256",
        "combined_eligible_event_roster_sha256",
        "combined_excluded_event_roster_sha256",
        "combined_eligible_patient_roster_sha256",
        "combined_eligible_split_patient_ids",
        "partial_reference_signal_patient_ids",
        "fixed18_primary_event_count",
        "fixed18_primary_patient_count",
        "fixed18_primary_event_roster_sha256",
        "fixed18_primary_patient_roster_sha256",
        "fixed18_primary_split_patient_ids",
        "events",
        "exclusions",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {"schema_version", "serialization", "receipt_sha256", "receipt"}
)
_HASH_FIELDS = (
    "base_signal_preflight_artifact_sha256",
    "base_signal_preflight_receipt_sha256",
    "identity_audit_sha256",
    "identity_mapping_sha256",
    "event_inputs_sha256",
    "record_crosswalk_sha256",
    "split_manifest_sha256",
    "deepsoz_source_sha256",
    "verified_target_v2_receipt_sha256",
    "verified_target_v2_artifact_sha256",
    "verified_target_v2_policy_sha256",
    "preprocess_config_sha256",
    "combined_candidate_event_roster_sha256",
    "combined_eligible_event_roster_sha256",
    "combined_excluded_event_roster_sha256",
    "combined_eligible_patient_roster_sha256",
    "fixed18_primary_event_roster_sha256",
    "fixed18_primary_patient_roster_sha256",
)


def _sorted_unique_strings(values: Sequence[object], *, field: str) -> list[str]:
    normalized = [str(value) for value in values]
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{field} must be sorted and unique")
    return normalized


def _sorted_unique_ints(values: object, *, field: str) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a JSON array")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{field} must contain integers")
    if values != sorted(set(values)):
        raise ValueError(f"{field} must be sorted and unique")
    return list(values)


def _validate_split_rosters(
    value: object,
    *,
    expected_patients: Sequence[str],
    field: str,
) -> list[list[object]]:
    if not isinstance(value, list) or len(value) != len(_MODEL_SPLITS):
        raise ValueError(f"{field} has the wrong split roster schema")
    flattened: list[str] = []
    result: list[list[object]] = []
    for row, expected_split in zip(value, _MODEL_SPLITS):
        if (
            not isinstance(row, list)
            or len(row) != 2
            or row[0] != expected_split
            or not isinstance(row[1], list)
        ):
            raise ValueError(f"{field} has an invalid split row")
        roster = _sorted_unique_strings(row[1], field=f"{field}.{expected_split}")
        flattened.extend(roster)
        result.append([expected_split, roster])
    if sorted(flattened) != sorted(expected_patients):
        raise ValueError(f"{field} does not partition its patient roster")
    return result


def _split_rosters(events: Sequence[Mapping[str, object]]) -> list[list[object]]:
    return [
        [
            split,
            sorted(
                {
                    str(event["patient_id"])
                    for event in events
                    if str(event["model_split"]) == split
                }
            ),
        ]
        for split in _MODEL_SPLITS
    ]


def _validate_receipt(value: object) -> dict[str, object]:
    receipt = _base._closed_object(
        value,
        expected=_RECOVERY_RECEIPT_FIELDS,
        field="identity-recovery signal receipt",
    )
    if receipt["schema_version"] != DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_SCHEMA:
        raise ValueError("Unsupported signal identity-recovery receipt schema")
    if receipt["policy"] != DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_POLICY:
        raise ValueError("Signal identity-recovery policy cannot be changed")
    for field in _HASH_FIELDS:
        _base._require_sha256(receipt[field], field=field)
    if receipt["preprocess_schema"] != EDF_PREPROCESS_SCHEMA:
        raise ValueError("Recovery preprocessing schema drifted")
    config = _base._closed_object(
        receipt["preprocess_config"],
        expected=frozenset(field.name for field in fields(CausalEDFConfig)),
        field="preprocess_config",
    )
    if _base._canonical_json_bytes(config) != _base._canonical_json_bytes(
        _base._config_payload(CausalEDFConfig())
    ):
        raise ValueError("Recovery requires the complete frozen causal config")
    if receipt["preprocess_config_sha256"] != _base._canonical_sha256(
        {"preprocess_schema": EDF_PREPROCESS_SCHEMA, "config": config}
    ):
        raise ValueError("Recovery preprocess config SHA mismatch")

    source_record_count = receipt["source_record_count"]
    if (
        isinstance(source_record_count, bool)
        or not isinstance(source_record_count, int)
        or source_record_count < 1
    ):
        raise ValueError("source_record_count must be a positive integer")
    recovered_rows = _sorted_unique_ints(
        receipt["identity_recovered_row_ids"], field="identity_recovered_row_ids"
    )
    if any(row < 0 or row >= source_record_count for row in recovered_rows):
        raise ValueError("Recovered row index is outside the source manifest")
    identity_patients = _sorted_unique_strings(
        receipt["identity_recovered_patient_ids"],
        field="identity_recovered_patient_ids",
    )
    variable_patients = _sorted_unique_strings(
        receipt["variable_label_patient_ids"], field="variable_label_patient_ids"
    )
    if set(variable_patients) - set(identity_patients):
        raise ValueError("Variable-label recovery patients are not identity-recovered")

    events_value = receipt["events"]
    exclusions_value = receipt["exclusions"]
    if not isinstance(events_value, list) or not isinstance(exclusions_value, list):
        raise ValueError("Recovery events and exclusions must be JSON arrays")
    events: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for index, event_value in enumerate(events_value):
        event = _base._closed_object(
            event_value, expected=_base._EVENT_FIELDS, field=f"events[{index}]"
        )
        _base._validate_nested_receipts(event, index=index)
        events.append(event)
    for index, exclusion_value in enumerate(exclusions_value):
        exclusions.append(
            _base._closed_object(
                exclusion_value,
                expected=_base._EXCLUSION_FIELDS,
                field=f"exclusions[{index}]",
            )
        )
    event_ids = [str(event["event_id"]) for event in events]
    excluded_ids = [str(event["event_id"]) for event in exclusions]
    if event_ids != sorted(event_ids) or excluded_ids != sorted(excluded_ids):
        raise ValueError("Recovery event arrays must be canonically ordered")
    combined_candidate_ids = sorted((*event_ids, *excluded_ids))
    if len(set(combined_candidate_ids)) != len(combined_candidate_ids):
        raise ValueError("Combined signal receipt contains duplicate event IDs")

    recovered_candidate_ids = _sorted_unique_strings(
        receipt["recovered_candidate_event_ids"],
        field="recovered_candidate_event_ids",
    )
    recovered_event_ids = _sorted_unique_strings(
        receipt["recovered_eligible_event_ids"],
        field="recovered_eligible_event_ids",
    )
    recovered_excluded_ids = _sorted_unique_strings(
        receipt["recovered_excluded_event_ids"],
        field="recovered_excluded_event_ids",
    )
    if sorted((*recovered_event_ids, *recovered_excluded_ids)) != recovered_candidate_ids:
        raise ValueError("Recovered eligible/excluded IDs do not partition candidates")
    if not set(recovered_event_ids) <= set(event_ids):
        raise ValueError("Recovered eligible IDs are absent from combined events")
    if not set(recovered_excluded_ids) <= set(excluded_ids):
        raise ValueError("Recovered excluded IDs are absent from combined exclusions")

    integer_counts = (
        "base_candidate_event_count",
        "base_eligible_event_count",
        "base_excluded_event_count",
        "base_eligible_patient_count",
        "recovered_candidate_event_count",
        "recovered_eligible_event_count",
        "recovered_excluded_event_count",
        "combined_candidate_event_count",
        "combined_eligible_event_count",
        "combined_excluded_event_count",
        "combined_eligible_patient_count",
        "fixed18_primary_event_count",
        "fixed18_primary_patient_count",
    )
    for field in integer_counts:
        count = receipt[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    count_checks = {
        "recovered_candidate_event_count": len(recovered_candidate_ids),
        "recovered_eligible_event_count": len(recovered_event_ids),
        "recovered_excluded_event_count": len(recovered_excluded_ids),
        "combined_candidate_event_count": len(combined_candidate_ids),
        "combined_eligible_event_count": len(event_ids),
        "combined_excluded_event_count": len(excluded_ids),
    }
    for field, expected in count_checks.items():
        if receipt[field] != expected:
            raise ValueError(f"{field} disagrees with its roster")
    if (
        receipt["base_candidate_event_count"] + len(recovered_candidate_ids)
        != len(combined_candidate_ids)
        or receipt["base_eligible_event_count"] + len(recovered_event_ids)
        != len(event_ids)
        or receipt["base_excluded_event_count"] + len(recovered_excluded_ids)
        != len(excluded_ids)
    ):
        raise ValueError("Base plus recovered counts do not close to combined counts")

    combined_patients = sorted({str(event["patient_id"]) for event in events})
    recovered_patients = sorted(
        {
            str(event["patient_id"])
            for event in events
            if str(event["event_id"]) in set(recovered_event_ids)
        }
    )
    declared_recovered_patients = _sorted_unique_strings(
        receipt["recovered_eligible_patient_ids"],
        field="recovered_eligible_patient_ids",
    )
    if declared_recovered_patients != recovered_patients:
        raise ValueError("Recovered eligible patient roster mismatch")
    if receipt["combined_eligible_patient_count"] != len(combined_patients):
        raise ValueError("Combined eligible patient count mismatch")
    _validate_split_rosters(
        receipt["recovered_eligible_split_patient_ids"],
        expected_patients=recovered_patients,
        field="recovered_eligible_split_patient_ids",
    )
    _validate_split_rosters(
        receipt["combined_eligible_split_patient_ids"],
        expected_patients=combined_patients,
        field="combined_eligible_split_patient_ids",
    )

    recovered_exclusion_counts: dict[str, int] = {}
    recovered_excluded_set = set(recovered_excluded_ids)
    for row in exclusions:
        if str(row["event_id"]) in recovered_excluded_set:
            code = str(row["eligibility_code"])
            recovered_exclusion_counts[code] = recovered_exclusion_counts.get(code, 0) + 1
    expected_code_rows = [
        [code, recovered_exclusion_counts[code]]
        for code in sorted(recovered_exclusion_counts)
    ]
    if receipt["recovered_exclusion_code_counts"] != expected_code_rows:
        raise ValueError("Recovered exclusion-code counts mismatch")

    roster_hash_checks = {
        "combined_candidate_event_roster_sha256": _base._roster_sha256(
            combined_candidate_ids
        ),
        "combined_eligible_event_roster_sha256": _base._roster_sha256(event_ids),
        "combined_excluded_event_roster_sha256": _base._roster_sha256(excluded_ids),
        "combined_eligible_patient_roster_sha256": _base._roster_sha256(
            combined_patients
        ),
    }
    for field, expected in roster_hash_checks.items():
        if receipt[field] != expected:
            raise ValueError(f"{field} mismatch")

    partial_patients = _sorted_unique_strings(
        receipt["partial_reference_signal_patient_ids"],
        field="partial_reference_signal_patient_ids",
    )
    if not set(partial_patients) <= set(combined_patients):
        raise ValueError("Partial-reference patients must be signal-eligible")
    fixed_events = [
        event for event in events if str(event["patient_id"]) not in set(partial_patients)
    ]
    fixed_event_ids = [str(event["event_id"]) for event in fixed_events]
    fixed_patients = sorted({str(event["patient_id"]) for event in fixed_events})
    if receipt["fixed18_primary_event_count"] != len(fixed_event_ids):
        raise ValueError("Fixed-18 event count mismatch")
    if receipt["fixed18_primary_patient_count"] != len(fixed_patients):
        raise ValueError("Fixed-18 patient count mismatch")
    if receipt["fixed18_primary_event_roster_sha256"] != _base._roster_sha256(
        fixed_event_ids
    ):
        raise ValueError("Fixed-18 event roster SHA mismatch")
    if receipt["fixed18_primary_patient_roster_sha256"] != _base._roster_sha256(
        fixed_patients
    ):
        raise ValueError("Fixed-18 patient roster SHA mismatch")
    _validate_split_rosters(
        receipt["fixed18_primary_split_patient_ids"],
        expected_patients=fixed_patients,
        field="fixed18_primary_split_patient_ids",
    )
    return receipt


@dataclass(frozen=True)
class VerifiedDeepSOZSignalIdentityRecoveryBundle:
    """Path-free verified receipt for the core plus recovered signal cohort."""

    receipt: Mapping[str, object]
    artifact_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        validated = _validate_receipt(dict(self.receipt))
        object.__setattr__(self, "receipt", validated)
        _base._require_sha256(self.artifact_sha256, field="artifact_sha256")
        _base._require_sha256(self.receipt_sha256, field="receipt_sha256")
        if self.receipt_sha256 != _base._canonical_sha256(validated):
            raise ValueError("Recovery receipt SHA disagrees with receipt")

    @property
    def eligible_event_ids(self) -> tuple[str, ...]:
        return tuple(str(row["event_id"]) for row in self.receipt["events"])

    @property
    def eligible_patient_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({str(row["patient_id"]) for row in self.receipt["events"]})
        )


def _read_base_bundle(
    bundle_directory: str | Path,
    *,
    expected_artifact_sha256: str,
) -> tuple[dict[str, object], str, str]:
    bundle = _base._reject_symlink_components(
        Path(bundle_directory), field="base signal-preflight bundle"
    )
    if not bundle.is_dir():
        raise FileNotFoundError("Base signal-preflight bundle does not exist")
    entries = tuple(sorted(bundle.iterdir(), key=lambda path: path.name))
    if (
        len(entries) != 1
        or entries[0].name != _base.DEEPSOZ_SIGNAL_PREFLIGHT_FILENAME
        or entries[0].is_symlink()
        or not entries[0].is_file()
    ):
        raise ValueError("Base signal-preflight bundle violates its closed schema")
    encoded, artifact_sha = _base._read_stable_regular_file(
        entries[0], field="base signal-preflight artifact", max_bytes=_MAX_ARTIFACT_BYTES
    )
    _base._check_expected_sha(
        artifact_sha,
        expected_artifact_sha256,
        field="expected_base_signal_preflight_artifact_sha256",
    )
    _, receipt = _base._parse_artifact(encoded)
    return receipt, artifact_sha, _base._canonical_sha256(receipt)


def _source_record_payload(
    source_row: Mapping[str, object], *, deepsoz_row: int
) -> tuple[str, dict[str, object]]:
    patient_id = normalize_patient_id(source_row["pt_id"])
    record = _base._clean(source_row["fn"], field="deepsoz_source.fn")
    starts = _base._strict_number_sequence(
        source_row["sz_starts"], field="deepsoz_source.sz_starts", allow_empty=True
    )
    stops = _base._strict_number_sequence(
        source_row["sz_ends"], field="deepsoz_source.sz_ends", allow_empty=True
    )
    count = _base._strict_int(source_row["nsz"], field="deepsoz_source.nsz")
    complete = (
        count > 0
        and len(starts) == count
        and len(stops) == count
        and all(stop > start for start, stop in zip(starts, stops))
    )
    empty = count == 0 and not starts and not stops
    if not complete and not empty:
        raise ValueError("DeepSOZ source timeline is internally inconsistent")
    payload = {
        "deepsoz_row": deepsoz_row,
        "pt_id": patient_id,
        "fn": record,
        "loc": str(source_row["loc"]).strip(),
        "nsz": count,
        "sz_starts": starts,
        "sz_ends": stops,
    }
    return _base._canonical_sha256(payload), payload


def _build_receipt(
    base_bundle_directory: str | Path,
    identity_audit_csv: str | Path,
    identity_mapping_csv: str | Path,
    event_inputs_csv: str | Path,
    record_crosswalk_csv: str | Path,
    split_manifest_csv: str | Path,
    deepsoz_source_csv: str | Path,
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    tusz_root: str | Path,
    *,
    expected_base_artifact_sha256: str,
    expected_identity_audit_sha256: str,
    expected_identity_mapping_sha256: str,
    expected_event_inputs_sha256: str,
    expected_record_crosswalk_sha256: str,
    expected_split_manifest_sha256: str,
    expected_deepsoz_source_sha256: str,
    config: CausalEDFConfig,
    reader_factory: Callable[[str], object] | None,
    expected_recovered_candidate_count: int | None,
    expected_recovered_eligible_count: int | None,
    expected_recovered_excluded_count: int | None,
    expected_combined_patient_count: int | None,
    expected_combined_event_count: int | None,
    expected_fixed18_patient_count: int | None,
    expected_fixed18_event_count: int | None,
) -> dict[str, object]:
    if not isinstance(verified_target_v2, VerifiedDeepSOZTargetV2Artifact):
        raise TypeError("verified_target_v2 must be a strictly verified artifact")
    if not isinstance(config, CausalEDFConfig):
        raise TypeError("config must be CausalEDFConfig")
    if _base._canonical_json_bytes(_base._config_payload(config)) != (
        _base._canonical_json_bytes(_base._config_payload(CausalEDFConfig()))
    ):
        raise ValueError("Formal recovery requires the complete frozen causal config")
    root = _base._reject_symlink_components(Path(tusz_root), field="TUSZ root")
    if not root.is_dir():
        raise FileNotFoundError("TUSZ root directory does not exist")

    base_receipt, base_artifact_sha, base_receipt_sha = _read_base_bundle(
        base_bundle_directory,
        expected_artifact_sha256=expected_base_artifact_sha256,
    )
    audit, audit_sha = _base._strict_csv(
        identity_audit_csv,
        expected_sha256=expected_identity_audit_sha256,
        allowed_columns=_IDENTITY_AUDIT_COLUMNS,
        label="identity_audit",
    )
    mapping, mapping_sha = _base._strict_csv(
        identity_mapping_csv,
        expected_sha256=expected_identity_mapping_sha256,
        allowed_columns=_base._CONSERVATIVE_MAPPING_INPUT_COLUMNS,
        label="identity_mapping",
    )
    event_inputs, event_inputs_sha = _base._strict_csv(
        event_inputs_csv,
        expected_sha256=expected_event_inputs_sha256,
        allowed_columns=_base._EVENT_INPUT_COLUMNS,
        label="event_inputs",
    )
    crosswalk, crosswalk_sha = _base._strict_csv(
        record_crosswalk_csv,
        expected_sha256=expected_record_crosswalk_sha256,
        allowed_columns=_base._CROSSWALK_INPUT_COLUMNS,
        label="record_crosswalk",
    )
    split, split_sha = _base._strict_csv(
        split_manifest_csv,
        expected_sha256=expected_split_manifest_sha256,
        allowed_columns=_base._SPLIT_INPUT_COLUMNS,
        label="split_manifest",
    )
    source, source_sha = _base._strict_deepsoz_source_csv(
        deepsoz_source_csv, expected_sha256=expected_deepsoz_source_sha256
    )
    target_receipt = verified_target_v2.receipt
    if target_receipt.source_input_sha256 != source_sha:
        raise ValueError("Recovered target-v2 and source manifest bytes differ")
    if target_receipt.split_input_sha256 != split_sha:
        raise ValueError("Recovered target-v2 and split manifest bytes differ")

    for label, frame in (
        ("event_inputs", event_inputs),
        ("record_crosswalk", crosswalk),
        ("split_manifest", split),
    ):
        if set(frame["source"].map(str).str.strip()) != {_base.DEEPSOZ_SIGNAL_SOURCE}:
            raise ValueError(f"{label} contains an unauthorized source")

    source = source.reset_index(drop=True)
    source_count = len(source)
    for label, frame in (("audit", audit), ("mapping", mapping), ("crosswalk", crosswalk)):
        rows = [
            _base._strict_int(value, field=f"{label}.deepsoz_row")
            for value in frame["deepsoz_row"]
        ]
        if rows != list(range(source_count)):
            raise ValueError(f"{label} must exactly preserve the ordered source-row roster")

    if set(audit["schema_version"]) != {IDENTITY_RECOVERY_SCHEMA}:
        raise ValueError("Identity audit schema drifted")
    if set(audit["policy"]) != {IDENTITY_RECOVERY_POLICY}:
        raise ValueError("Identity audit policy drifted")
    recovered_audit = audit.loc[
        audit["original_mapping_status"].isin(_RECOVERED_ORIGINAL_STATUSES)
    ].copy()
    if recovered_audit.empty:
        raise ValueError("Identity audit contains no recovered rows")
    if set(recovered_audit["recovery_status"]) != {"identity_recovered"}:
        raise ValueError("An originally non-unique row was not identity-recovered")
    for field in (
        "path_candidate_count",
        "split_match",
        "session_year_match",
        "montage_match",
        "record_key_match",
        "patient_binding_match",
        "nsamples_match",
    ):
        if any(str(value).strip() != "1" for value in recovered_audit[field]):
            raise ValueError(f"Recovered identity evidence failed: {field}")
    recovered_rows = sorted(int(value) for value in recovered_audit["deepsoz_row"])
    recovered_row_set = set(recovered_rows)
    recovered_paths = [str(value).strip() for value in recovered_audit["relative_edf_path"]]
    if not all(recovered_paths) or len(set(recovered_paths)) != len(recovered_paths):
        raise ValueError("Recovered EDF paths must be non-empty and unique")

    audit_by_row = {
        int(row["deepsoz_row"]): row for row in recovered_audit.to_dict("records")
    }
    mapping_by_row = {
        int(row["deepsoz_row"]): row for row in mapping.to_dict("records")
    }
    crosswalk_by_row = {
        int(row["deepsoz_row"]): row for row in crosswalk.to_dict("records")
    }
    identity_patients: set[str] = set()
    for deepsoz_row in recovered_rows:
        audit_row = audit_by_row[deepsoz_row]
        mapping_row = mapping_by_row[deepsoz_row]
        crosswalk_row = crosswalk_by_row[deepsoz_row]
        source_row = source.iloc[deepsoz_row].to_dict()
        patient_id = normalize_patient_id(source_row["pt_id"])
        identity_patients.add(patient_id)
        source_record = str(source_row["fn"]).strip()
        checks = {
            "audit patient": normalize_patient_id(audit_row["deepsoz_patient"]) == patient_id,
            "mapping patient": normalize_patient_id(mapping_row["deepsoz_patient"]) == patient_id,
            "crosswalk patient": normalize_patient_id(crosswalk_row["deepsoz_patient_id"]) == patient_id,
            "audit record": str(audit_row["deepsoz_record"]).strip() == source_record,
            "mapping record": str(mapping_row["deepsoz_record"]).strip() == source_record,
            "crosswalk record": str(crosswalk_row["deepsoz_record"]).strip() == source_record,
            "mapping local patient": str(mapping_row["local_patient"]).strip() == str(audit_row["local_patient"]).strip(),
            "crosswalk local patient": str(crosswalk_row["local_patient_id"]).strip() == str(audit_row["local_patient"]).strip(),
            "crosswalk EDF": str(crosswalk_row["local_edf_path"]).strip() == str(audit_row["relative_edf_path"]).strip(),
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"Recovered identity/crosswalk drift: {failed}")
        if str(mapping_row["mapping_status"]).strip() != "unique":
            raise ValueError("Recovered identity mapping must be selected exactly once")
        (
            relative_edf,
            _,
            relative_csv,
            _,
            relative_csv_bi,
            _,
            local_split,
            local_patient,
            local_record_key,
        ) = _base._canonical_tusz_record_identity(
            root,
            crosswalk_row["local_edf_path"],
            crosswalk_row["local_csv_path"],
            crosswalk_row["local_csv_bi_path"],
        )
        mapping_edf, _ = _base._mapping_source_path(
            root, mapping_row["local_edf"], field="identity_mapping.local_edf"
        )
        mapping_csv_bi, _ = _base._mapping_source_path(
            root,
            mapping_row["local_csv_bi"],
            field="identity_mapping.local_csv_bi",
        )
        if (
            relative_edf != mapping_edf
            or relative_csv_bi != mapping_csv_bi
            or relative_edf != str(audit_row["relative_edf_path"]).strip()
            or local_split != _base._source_official_split(source_row["loc"])
            or local_patient != str(audit_row["local_patient"]).strip()
            or local_record_key != _base._source_record_key(source_record)
            or relative_csv != Path(relative_edf).with_suffix(".csv").as_posix()
        ):
            raise ValueError("Recovered canonical path identity did not replay")

    split = split.copy()
    split["deepsoz_patient_id"] = split["deepsoz_patient_id"].map(normalize_patient_id)
    if split["deepsoz_patient_id"].duplicated().any():
        raise ValueError("split_manifest contains duplicate patients")
    split_by_patient = split.set_index("deepsoz_patient_id", drop=False)
    registry_ids = {reference.patient_id for reference in verified_target_v2.registry}
    if set(split_by_patient.index) != registry_ids:
        raise ValueError("Recovered split and target-v2 patient rosters differ")
    eligible_ids = set(target_receipt.eligible_patient_ids)
    nonquarantine_ids = set(
        split.loc[split["model_split"].isin(_MODEL_SPLITS), "deepsoz_patient_id"]
    )
    if nonquarantine_ids != eligible_ids:
        raise ValueError("Split quarantine and verified target-v2 eligibility differ")
    variable_patients = sorted(
        set(
            split.loc[
                split["label_stability_primary"].str.strip().eq("variable"),
                "deepsoz_patient_id",
            ]
        )
        & identity_patients
    )
    if set(variable_patients) & eligible_ids:
        raise ValueError("A variable-label patient escaped target quarantine")
    for patient_id in sorted(eligible_ids):
        reference = verified_target_v2.registry.get(patient_id)
        split_row = split_by_patient.loc[patient_id]
        if (
            str(split_row["model_split"]).strip() != reference.model_split
            or str(split_row["official_split"]).strip() != reference.official_split
        ):
            raise ValueError("Split manifest differs from verified target-v2")

    event_inputs = event_inputs.copy()
    event_inputs["deepsoz_patient_id"] = event_inputs["deepsoz_patient_id"].map(
        normalize_patient_id
    )
    event_inputs["deepsoz_row_int"] = [
        _base._strict_int(value, field="event_inputs.deepsoz_row")
        for value in event_inputs["deepsoz_row"]
    ]
    if event_inputs["event_id"].map(str).str.strip().duplicated().any():
        raise ValueError("event_inputs contains duplicate event IDs")
    if event_inputs.duplicated(["local_edf_path", "event_index"]).any():
        raise ValueError("event_inputs contains duplicate EDF/event-index pairs")
    recovered_event_inputs = event_inputs.loc[
        event_inputs["deepsoz_row_int"].isin(recovered_row_set)
    ]
    audit_event_counts = {
        row: _base._strict_int(
            audit_by_row[row]["local_event_count"], field="audit.local_event_count"
        )
        for row in recovered_rows
    }
    actual_event_counts = recovered_event_inputs.groupby("deepsoz_row_int").size().to_dict()
    for row in recovered_rows:
        if int(actual_event_counts.get(row, 0)) != audit_event_counts[row]:
            raise ValueError("Recovered event_inputs count differs from identity audit")

    candidate_frame = recovered_event_inputs.loc[
        recovered_event_inputs["deepsoz_patient_id"].isin(eligible_ids)
    ].copy()
    if set(candidate_frame["deepsoz_patient_id"]) & set(variable_patients):
        raise ValueError("Variable-label patient entered recovered candidates")
    candidate_rows_by_record: dict[int, list[Mapping[str, object]]] = {}
    for row in candidate_frame.to_dict("records"):
        candidate_rows_by_record.setdefault(int(row["deepsoz_row_int"]), []).append(row)

    accepted: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    candidate_ids: list[str] = []
    preprocess_sha = _base._config_sha256(config)
    event_join_fields = (
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

    for deepsoz_row in recovered_rows:
        source_row = source.iloc[deepsoz_row].to_dict()
        patient_id = normalize_patient_id(source_row["pt_id"])
        if patient_id not in eligible_ids:
            continue
        crosswalk_row = crosswalk_by_row[deepsoz_row]
        reference = verified_target_v2.registry.get(patient_id)
        (
            relative_edf,
            edf_path,
            relative_csv,
            csv_path,
            relative_csv_bi,
            csv_bi_path,
            local_split,
            local_patient,
            _,
        ) = _base._canonical_tusz_record_identity(
            root,
            crosswalk_row["local_edf_path"],
            crosswalk_row["local_csv_path"],
            crosswalk_row["local_csv_bi_path"],
        )
        if local_split != reference.official_split:
            raise ValueError("Recovered record split differs from target-v2")
        if local_patient != str(crosswalk_row["local_patient_id"]).strip():
            raise ValueError("Recovered record local patient drifted")
        pair = inspect_tusz_annotation_pair(csv_path, csv_bi_path, source_path=edf_path)
        rows = candidate_rows_by_record.get(deepsoz_row, [])
        by_index: dict[int, Mapping[str, object]] = {}
        for row in rows:
            event_index = _base._strict_int(row["event_index"], field="event_index")
            if event_index in by_index:
                raise ValueError("Recovered event_inputs repeats a global event index")
            by_index[event_index] = row
        expected_indices = set(range(len(pair.global_seizure_events)))
        if set(by_index) != expected_indices:
            raise ValueError(
                "Recovered event_inputs does not enumerate the local TUSZ timeline"
            )
        source_record_sha, source_payload = _source_record_payload(
            source_row, deepsoz_row=deepsoz_row
        )
        if source_payload["pt_id"] != patient_id:
            raise RuntimeError("Source patient normalization drifted")
        crosswalk_record_sha = _base._canonical_sha256(
            _base._row_payload(crosswalk_row, crosswalk_join_fields)
        )
        for global_event in pair.global_seizure_events:
            row = by_index[global_event.event_index]
            comparisons = {
                "patient": normalize_patient_id(row["deepsoz_patient_id"]) == patient_id,
                "target key": normalize_patient_id(row["patient_target_key"]) == patient_id,
                "record": str(row["deepsoz_record"]).strip() == str(source_row["fn"]).strip(),
                "local patient": str(row["local_patient_id"]).strip() == local_patient,
                "official split": str(row["official_split"]).strip() == local_split,
                "EDF": str(row["local_edf_path"]).strip() == relative_edf,
                "channel annotation": str(row["local_csv_path"]).strip() == relative_csv,
                "global annotation": str(row["local_csv_bi_path"]).strip() == relative_csv_bi,
            }
            failed = sorted(name for name, passed in comparisons.items() if not passed)
            if failed:
                raise ValueError(f"Recovered event foreign-key drift: {failed}")
            event_id = _base._clean(row["event_id"], field="event_id")
            expected_event_id = f"{edf_path.stem}__ev{global_event.event_index:04d}"
            if event_id != expected_event_id:
                raise ValueError("Recovered event_id does not encode EDF/event identity")
            if _base._clean(row["t0_provenance"], field="t0_provenance") != (
                _base.DEEPSOZ_EVENT_ANCHOR
            ):
                raise ValueError("Recovered event uses an unauthorized onset anchor")
            t0 = _base._strict_float(row["t0_sec"], field="t0_sec")
            stop = _base._strict_float(row["seizure_end_sec"], field="seizure_end_sec")
            window_start = _base._strict_float(
                row["window_start_sec"], field="window_start_sec"
            )
            window_stop = _base._strict_float(
                row["window_stop_sec"], field="window_stop_sec"
            )
            timing_checks = (
                (t0, global_event.start_sec, "local global t0"),
                (stop, global_event.stop_sec, "local global stop"),
                (window_start, t0 - config.pre_onset_sec, "window start"),
                (window_stop, t0 + config.post_onset_sec, "window stop"),
            )
            for actual, expected, label in timing_checks:
                if abs(actual - expected) > _TIME_TOLERANCE_SEC:
                    raise ValueError(f"Recovered {label} drifted")
            if t0 < 0 or abs((window_stop - window_start) - 60.0) > _TIME_TOLERANCE_SEC:
                raise ValueError("Recovered event window is not valid [-12,+48)")
            event_record_sha = _base._canonical_sha256(
                _base._row_payload(row, event_join_fields)
            )
            candidate_ids.append(event_id)
            common = {
                "event_id": event_id,
                "event_record_sha256": event_record_sha,
                "crosswalk_record_sha256": crosswalk_record_sha,
                "deepsoz_source_record_sha256": source_record_sha,
                "patient_id": patient_id,
                "local_patient_id": local_patient,
                "official_split": reference.official_split,
                "model_split": reference.model_split,
                "relative_edf_path": relative_edf,
                "deepsoz_record": str(source_row["fn"]).strip(),
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
                raise RuntimeError("EDF loader and annotation-pair hashes disagree")
            if (
                abs(loaded.edf_receipt.requested_onset_sec - global_event.start_sec)
                > _TIME_TOLERANCE_SEC
            ):
                raise RuntimeError("Recovered EDF replay used the wrong local t0")
            if (
                tuple(loaded.window.data.shape) != (19, 12_000)
                or loaded.window.onset_index != 2_400
                or abs(loaded.window.sfreq_hz - 200.0) > _TIME_TOLERANCE_SEC
            ):
                raise RuntimeError("Recovered preprocessing output shape/alignment drifted")
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
                    "edf_receipt_sha256": _base._canonical_sha256(edf_receipt),
                    "signal_receipt": signal_receipt,
                    "signal_receipt_sha256": _base._canonical_sha256(signal_receipt),
                    "processed_window_sha256": _base._tensor_sha256(
                        loaded.window.data
                    ),
                    "processed_window_shape": list(loaded.window.data.shape),
                    "processed_window_dtype": str(loaded.window.data.dtype),
                }
            )

    accepted.sort(key=lambda row: str(row["event_id"]))
    excluded.sort(key=lambda row: str(row["event_id"]))
    candidate_ids = sorted(candidate_ids)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Recovered candidate event IDs are not unique")
    accepted_ids = [str(row["event_id"]) for row in accepted]
    excluded_ids = [str(row["event_id"]) for row in excluded]
    if sorted((*accepted_ids, *excluded_ids)) != candidate_ids:
        raise RuntimeError("Recovered replay did not close its candidate roster")

    base_events = [dict(row) for row in base_receipt["events"]]
    base_exclusions = [dict(row) for row in base_receipt["exclusions"]]
    base_candidate_ids = {
        str(row["event_id"]) for row in (*base_events, *base_exclusions)
    }
    if base_candidate_ids & set(candidate_ids):
        raise ValueError("Recovered events overlap the immutable core candidate roster")
    combined_events = sorted((*base_events, *accepted), key=lambda row: str(row["event_id"]))
    combined_exclusions = sorted(
        (*base_exclusions, *excluded), key=lambda row: str(row["event_id"])
    )
    combined_event_ids = [str(row["event_id"]) for row in combined_events]
    combined_excluded_ids = [str(row["event_id"]) for row in combined_exclusions]
    combined_candidate_ids = sorted((*combined_event_ids, *combined_excluded_ids))
    combined_patients = sorted({str(row["patient_id"]) for row in combined_events})
    recovered_eligible_patients = sorted({str(row["patient_id"]) for row in accepted})

    for row in combined_events:
        patient_id = str(row["patient_id"])
        if patient_id not in eligible_ids:
            raise ValueError("Combined signal event belongs to a quarantined target patient")
        reference = verified_target_v2.registry.get(patient_id)
        if (
            str(row["model_split"]) != reference.model_split
            or str(row["official_split"]) != reference.official_split
        ):
            raise ValueError("Core/recovered event split differs from recovered target-v2")

    non_pz_indices = tuple(
        index for index, channel in enumerate(STANDARD_19) if index != CHANNEL_INDEX["PZ"]
    )
    partial_reference_patients = sorted(
        patient_id
        for patient_id in combined_patients
        if not bool(
            verified_target_v2.registry.get(patient_id).mask[
                list(non_pz_indices)
            ].all()
        )
    )
    partial_set = set(partial_reference_patients)
    fixed_events = [
        row for row in combined_events if str(row["patient_id"]) not in partial_set
    ]
    fixed_event_ids = [str(row["event_id"]) for row in fixed_events]
    fixed_patients = sorted({str(row["patient_id"]) for row in fixed_events})

    exclusion_counts: dict[str, int] = {}
    for row in excluded:
        code = str(row["eligibility_code"])
        exclusion_counts[code] = exclusion_counts.get(code, 0) + 1
    recovered_exclusion_code_counts = [
        [code, exclusion_counts[code]] for code in sorted(exclusion_counts)
    ]
    receipt: dict[str, object] = {
        "schema_version": DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_SCHEMA,
        "policy": DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_POLICY,
        "base_signal_preflight_artifact_sha256": base_artifact_sha,
        "base_signal_preflight_receipt_sha256": base_receipt_sha,
        "identity_audit_sha256": audit_sha,
        "identity_mapping_sha256": mapping_sha,
        "event_inputs_sha256": event_inputs_sha,
        "record_crosswalk_sha256": crosswalk_sha,
        "split_manifest_sha256": split_sha,
        "deepsoz_source_sha256": source_sha,
        "verified_target_v2_receipt_sha256": target_receipt.receipt_sha256,
        "verified_target_v2_artifact_sha256": target_receipt.target_artifact_sha256,
        "verified_target_v2_policy_sha256": target_receipt.policy_sha256,
        "preprocess_schema": EDF_PREPROCESS_SCHEMA,
        "preprocess_config": _base._config_payload(config),
        "preprocess_config_sha256": preprocess_sha,
        "source_record_count": source_count,
        "identity_recovered_row_ids": recovered_rows,
        "identity_recovered_patient_ids": sorted(identity_patients),
        "variable_label_patient_ids": variable_patients,
        "base_candidate_event_count": int(base_receipt["candidate_event_count"]),
        "base_eligible_event_count": len(base_events),
        "base_excluded_event_count": len(base_exclusions),
        "base_eligible_patient_count": int(base_receipt["eligible_patient_count"]),
        "recovered_candidate_event_ids": candidate_ids,
        "recovered_eligible_event_ids": accepted_ids,
        "recovered_excluded_event_ids": excluded_ids,
        "recovered_candidate_event_count": len(candidate_ids),
        "recovered_eligible_event_count": len(accepted),
        "recovered_excluded_event_count": len(excluded),
        "recovered_eligible_patient_ids": recovered_eligible_patients,
        "recovered_eligible_split_patient_ids": _split_rosters(accepted),
        "recovered_exclusion_code_counts": recovered_exclusion_code_counts,
        "combined_candidate_event_count": len(combined_candidate_ids),
        "combined_eligible_event_count": len(combined_events),
        "combined_excluded_event_count": len(combined_exclusions),
        "combined_eligible_patient_count": len(combined_patients),
        "combined_candidate_event_roster_sha256": _base._roster_sha256(
            combined_candidate_ids
        ),
        "combined_eligible_event_roster_sha256": _base._roster_sha256(
            combined_event_ids
        ),
        "combined_excluded_event_roster_sha256": _base._roster_sha256(
            combined_excluded_ids
        ),
        "combined_eligible_patient_roster_sha256": _base._roster_sha256(
            combined_patients
        ),
        "combined_eligible_split_patient_ids": _split_rosters(combined_events),
        "partial_reference_signal_patient_ids": partial_reference_patients,
        "fixed18_primary_event_count": len(fixed_events),
        "fixed18_primary_patient_count": len(fixed_patients),
        "fixed18_primary_event_roster_sha256": _base._roster_sha256(fixed_event_ids),
        "fixed18_primary_patient_roster_sha256": _base._roster_sha256(fixed_patients),
        "fixed18_primary_split_patient_ids": _split_rosters(fixed_events),
        "events": combined_events,
        "exclusions": combined_exclusions,
    }
    _validate_receipt(receipt)

    expectations = {
        "recovered_candidate_event_count": expected_recovered_candidate_count,
        "recovered_eligible_event_count": expected_recovered_eligible_count,
        "recovered_excluded_event_count": expected_recovered_excluded_count,
        "combined_eligible_patient_count": expected_combined_patient_count,
        "combined_eligible_event_count": expected_combined_event_count,
        "fixed18_primary_patient_count": expected_fixed18_patient_count,
        "fixed18_primary_event_count": expected_fixed18_event_count,
    }
    for field, expected in expectations.items():
        if expected is not None and receipt[field] != expected:
            raise ValueError(
                f"Formal recovery expected {field}={expected}, got {receipt[field]}"
            )
    return receipt


def _publish_receipt(
    receipt: Mapping[str, object], output_directory: str | Path
) -> VerifiedDeepSOZSignalIdentityRecoveryBundle:
    receipt_sha = _base._canonical_sha256(receipt)
    payload = {
        "schema_version": DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_ARTIFACT_SCHEMA,
        "serialization": "canonical_json_utf8_newline_no_pickle",
        "receipt_sha256": receipt_sha,
        "receipt": receipt,
    }
    encoded = _base._canonical_json_bytes(payload)
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("Signal identity-recovery artifact exceeds its size limit")
    output = _base._reject_symlink_components(
        Path(output_directory), field="signal identity-recovery output"
    )
    if output.name in {"", ".", ".."}:
        raise ValueError("Output requires a concrete directory name")
    if os.path.lexists(output):
        raise FileExistsError("Signal identity-recovery destination already exists")
    parent = _base._reject_symlink_components(output.parent, field="output parent")
    if not parent.is_dir():
        raise FileNotFoundError("Signal identity-recovery output parent does not exist")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    published = False
    try:
        artifact_path = temporary / DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_FILENAME
        with artifact_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _base._fsync_directory(temporary)
        if os.path.lexists(output):
            raise FileExistsError("Signal identity-recovery destination already exists")
        os.rename(temporary, output)
        published = True
        _base._fsync_directory(parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    return VerifiedDeepSOZSignalIdentityRecoveryBundle(
        receipt=receipt,
        artifact_sha256=_base._bytes_sha256(encoded),
        receipt_sha256=receipt_sha,
    )


def build_deepsoz_signal_identity_recovery_bundle(
    base_bundle_directory: str | Path,
    identity_audit_csv: str | Path,
    identity_mapping_csv: str | Path,
    event_inputs_csv: str | Path,
    record_crosswalk_csv: str | Path,
    split_manifest_csv: str | Path,
    deepsoz_source_csv: str | Path,
    verified_target_v2: VerifiedDeepSOZTargetV2Artifact,
    tusz_root: str | Path,
    output_directory: str | Path,
    *,
    expected_base_artifact_sha256: str,
    expected_identity_audit_sha256: str,
    expected_identity_mapping_sha256: str,
    expected_event_inputs_sha256: str,
    expected_record_crosswalk_sha256: str,
    expected_split_manifest_sha256: str,
    expected_deepsoz_source_sha256: str,
    config: CausalEDFConfig = CausalEDFConfig(),
    reader_factory: Callable[[str], object] | None = None,
    expected_recovered_candidate_count: int | None = None,
    expected_recovered_eligible_count: int | None = None,
    expected_recovered_excluded_count: int | None = None,
    expected_combined_patient_count: int | None = None,
    expected_combined_event_count: int | None = None,
    expected_fixed18_patient_count: int | None = None,
    expected_fixed18_event_count: int | None = None,
) -> VerifiedDeepSOZSignalIdentityRecoveryBundle:
    """Replay recovered events and atomically publish the combined receipt."""

    if os.path.lexists(output_directory):
        raise FileExistsError("Signal identity-recovery destination already exists")
    receipt = _build_receipt(
        base_bundle_directory,
        identity_audit_csv,
        identity_mapping_csv,
        event_inputs_csv,
        record_crosswalk_csv,
        split_manifest_csv,
        deepsoz_source_csv,
        verified_target_v2,
        tusz_root,
        expected_base_artifact_sha256=expected_base_artifact_sha256,
        expected_identity_audit_sha256=expected_identity_audit_sha256,
        expected_identity_mapping_sha256=expected_identity_mapping_sha256,
        expected_event_inputs_sha256=expected_event_inputs_sha256,
        expected_record_crosswalk_sha256=expected_record_crosswalk_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_deepsoz_source_sha256=expected_deepsoz_source_sha256,
        config=config,
        reader_factory=reader_factory,
        expected_recovered_candidate_count=expected_recovered_candidate_count,
        expected_recovered_eligible_count=expected_recovered_eligible_count,
        expected_recovered_excluded_count=expected_recovered_excluded_count,
        expected_combined_patient_count=expected_combined_patient_count,
        expected_combined_event_count=expected_combined_event_count,
        expected_fixed18_patient_count=expected_fixed18_patient_count,
        expected_fixed18_event_count=expected_fixed18_event_count,
    )
    return _publish_receipt(receipt, output_directory)


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
        raise ValueError("Recovery artifact is not strict UTF-8 JSON") from exc
    payload = _base._closed_object(
        payload, expected=_ARTIFACT_FIELDS, field="recovery artifact"
    )
    if _base._canonical_json_bytes(payload) != encoded:
        raise ValueError("Recovery artifact bytes are not canonical JSON")
    if payload["schema_version"] != DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported recovery artifact schema")
    if payload["serialization"] != "canonical_json_utf8_newline_no_pickle":
        raise ValueError("Recovery artifact uses an unsafe serialization")
    receipt = _validate_receipt(payload["receipt"])
    declared = _base._require_sha256(payload["receipt_sha256"], field="receipt_sha256")
    if declared != _base._canonical_sha256(receipt):
        raise ValueError("Recovery artifact receipt SHA mismatch")
    return payload, receipt


def load_deepsoz_signal_identity_recovery_bundle(
    bundle_directory: str | Path,
    *,
    expected_artifact_sha256: str,
) -> VerifiedDeepSOZSignalIdentityRecoveryBundle:
    """Strictly parse and validate a published recovery receipt.

    The artifact pins every source and every processed-window digest.  Call
    :func:`build_deepsoz_signal_identity_recovery_bundle` again in a new
    destination when a full source/EDF replay is required.
    """

    bundle = _base._reject_symlink_components(
        Path(bundle_directory), field="signal identity-recovery bundle"
    )
    if not bundle.is_dir():
        raise FileNotFoundError("Signal identity-recovery bundle does not exist")
    entries = tuple(sorted(bundle.iterdir(), key=lambda path: path.name))
    if (
        len(entries) != 1
        or entries[0].name != DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_FILENAME
        or entries[0].is_symlink()
        or not entries[0].is_file()
    ):
        raise ValueError("Signal identity-recovery bundle violates its closed schema")
    encoded, artifact_sha = _base._read_stable_regular_file(
        entries[0], field="signal identity-recovery artifact", max_bytes=_MAX_ARTIFACT_BYTES
    )
    _base._check_expected_sha(
        artifact_sha,
        expected_artifact_sha256,
        field="expected_signal_identity_recovery_artifact_sha256",
    )
    _, receipt = _parse_artifact(encoded)
    return VerifiedDeepSOZSignalIdentityRecoveryBundle(
        receipt=receipt,
        artifact_sha256=artifact_sha,
        receipt_sha256=_base._canonical_sha256(receipt),
    )


__all__ = [
    "DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_ARTIFACT_SCHEMA",
    "DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_FILENAME",
    "DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_POLICY",
    "DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_SCHEMA",
    "VerifiedDeepSOZSignalIdentityRecoveryBundle",
    "build_deepsoz_signal_identity_recovery_bundle",
    "load_deepsoz_signal_identity_recovery_bundle",
]
