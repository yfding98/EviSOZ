"""Target-independent TUSZ signal universe for the DeepSOZ identity overlay.

This module deliberately stops before any SOZ endpoint join.  Its only roster
input is the complete identity-recovery audit: one row for every DeepSOZ
source record, projected onto identity/provenance fields only.  Candidate
events are rediscovered from the local TUSZ ``TERM,seiz`` annotation timeline
and signal eligibility is decided exclusively by the frozen causal EDF
loader.

The builder therefore does *not* accept a target artifact, a target-derived
split/quarantine manifest, a prefiltered event table, the historical signal
preflight core, or a model checkpoint.  Official TUSZ split membership is
read from the canonical EDF path and mapped mechanically to ``source_*``.
SOZ target completeness and any task-supervised model lineage belong to a
separate downstream endpoint join and must not alter this universe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Mapping, Sequence

from . import deepsoz_signal_preflight as _base
from .deepsoz import normalize_patient_id
from .deepsoz_identity_recovery import (
    IDENTITY_RECOVERY_POLICY,
    IDENTITY_RECOVERY_SCHEMA,
)
from .edf import (
    CausalEDFConfig,
    EDFEventEligibilityError,
    EDF_PREPROCESS_SCHEMA,
    load_standard19_edf_event,
)
from .tusz import inspect_tusz_annotation_pair


TARGET_INDEPENDENT_SIGNAL_UNIVERSE_SCHEMA = (
    "soz_deepsoz_target_independent_signal_universe_v1"
)
TARGET_INDEPENDENT_SIGNAL_UNIVERSE_ARTIFACT_SCHEMA = (
    "soz_deepsoz_target_independent_signal_universe_artifact_v1"
)
TARGET_INDEPENDENT_SIGNAL_UNIVERSE_FILENAME = (
    "deepsoz_target_independent_signal_universe.json"
)
TARGET_INDEPENDENT_SIGNAL_UNIVERSE_POLICY = (
    "complete_identity_audit_local_tusz_TERM_seiz_frozen_causal_replay_"
    "no_target_no_target_roster_no_model"
)

_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_TIME_TOLERANCE_SEC = 1e-6
_MODEL_SPLIT_BY_OFFICIAL = {
    "train": "source_train",
    "dev": "source_dev",
    "eval": "source_eval",
}
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
_LINEAGE_AXIS_FIELDS = frozenset(
    {
        "direct_target_values",
        "upstream_target_conditioned_roster",
        "target_supervised_model",
    }
)
_LINEAGE_STATE_FIELDS = frozenset({"used", "evidence"})
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "lineage_axes",
        "benchmark_identity_overlay_conditioned",
        "roster_scope",
        "identity_audit_sha256",
        "preprocess_schema",
        "preprocess_config",
        "preprocess_config_sha256",
        "identity_record_count",
        "identity_patient_count",
        "identity_record_sha256s",
        "identity_patient_ids",
        "identity_record_roster_sha256",
        "identity_patient_roster_sha256",
        "candidate_event_count",
        "eligible_event_count",
        "excluded_event_count",
        "eligible_patient_count",
        "candidate_event_roster_sha256",
        "eligible_event_roster_sha256",
        "excluded_event_roster_sha256",
        "eligible_patient_roster_sha256",
        "candidate_official_split_event_counts",
        "eligible_official_split_event_counts",
        "exclusion_code_counts",
        "events",
        "exclusions",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {"schema_version", "serialization", "receipt_sha256", "receipt"}
)
_HASH_FIELDS = (
    "identity_audit_sha256",
    "preprocess_config_sha256",
    "identity_record_roster_sha256",
    "identity_patient_roster_sha256",
    "candidate_event_roster_sha256",
    "eligible_event_roster_sha256",
    "excluded_event_roster_sha256",
    "eligible_patient_roster_sha256",
)


def _lineage_axes() -> dict[str, dict[str, object]]:
    """Return the closed three-axis lineage declaration for this producer."""

    return {
        "direct_target_values": {
            "used": False,
            "evidence": (
                "builder API accepts no target artifact or target-bearing table"
            ),
        },
        "upstream_target_conditioned_roster": {
            "used": False,
            "evidence": (
                "within the benchmark identity overlay, all ordered identity-audit "
                "rows are required; no patient, record, or event is selected by "
                "target value, label stability, quarantine, or C18 mask"
            ),
        },
        "target_supervised_model": {
            "used": False,
            "evidence": (
                "signal eligibility is produced by deterministic preprocessing and "
                "the builder accepts no model artifact"
            ),
        },
    }


def _validate_lineage_axes(value: object) -> dict[str, dict[str, object]]:
    axes = _base._closed_object(
        value, expected=_LINEAGE_AXIS_FIELDS, field="lineage_axes"
    )
    expected = _lineage_axes()
    normalized: dict[str, dict[str, object]] = {}
    for axis in sorted(_LINEAGE_AXIS_FIELDS):
        state = _base._closed_object(
            axes[axis], expected=_LINEAGE_STATE_FIELDS, field=f"lineage_axes.{axis}"
        )
        if state != expected[axis]:
            raise ValueError(f"Target-independent lineage axis drifted: {axis}")
        normalized[axis] = dict(state)
    return normalized


def _strict_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _split_counts(
    rows: Sequence[Mapping[str, object]],
) -> list[list[object]]:
    return [
        [
            official_split,
            sum(str(row["official_split"]) == official_split for row in rows),
        ]
        for official_split in _MODEL_SPLIT_BY_OFFICIAL
    ]


def _validate_split_counts(value: object, *, field: str, total: int) -> None:
    expected_splits = list(_MODEL_SPLIT_BY_OFFICIAL)
    if not isinstance(value, list) or len(value) != len(expected_splits):
        raise ValueError(f"{field} has an invalid split-count schema")
    observed_total = 0
    for row, expected_split in zip(value, expected_splits):
        if not isinstance(row, list) or len(row) != 2 or row[0] != expected_split:
            raise ValueError(f"{field} split order drifted")
        observed_total += _strict_nonnegative_int(
            row[1], field=f"{field}.{expected_split}"
        )
    if observed_total != total:
        raise ValueError(f"{field} does not sum to its event roster")


def _validate_exclusion_counts(
    value: object, exclusions: Sequence[Mapping[str, object]]
) -> None:
    if not isinstance(value, list):
        raise ValueError("exclusion_code_counts must be a JSON array")
    expected: dict[str, int] = {}
    for row in exclusions:
        code = str(row["eligibility_code"])
        expected[code] = expected.get(code, 0) + 1
    canonical = [[code, expected[code]] for code in sorted(expected)]
    if value != canonical:
        raise ValueError("exclusion_code_counts disagrees with exclusions")


def _validate_receipt(value: object) -> dict[str, object]:
    receipt = _base._closed_object(
        value, expected=_RECEIPT_FIELDS, field="target-independent signal receipt"
    )
    if receipt["schema_version"] != TARGET_INDEPENDENT_SIGNAL_UNIVERSE_SCHEMA:
        raise ValueError("Unsupported target-independent signal-universe schema")
    if receipt["policy"] != TARGET_INDEPENDENT_SIGNAL_UNIVERSE_POLICY:
        raise ValueError("Target-independent signal-universe policy drifted")
    _validate_lineage_axes(receipt["lineage_axes"])
    if receipt["benchmark_identity_overlay_conditioned"] is not True:
        raise ValueError("Benchmark identity-overlay conditioning must be explicit")
    if receipt["roster_scope"] != (
        "complete_deepsoz_identity_overlay_not_complete_tusz"
    ):
        raise ValueError("Signal-universe roster scope drifted")
    for field in _HASH_FIELDS:
        _base._require_sha256(receipt[field], field=field)
    if receipt["preprocess_schema"] != EDF_PREPROCESS_SCHEMA:
        raise ValueError("Signal-universe preprocessing schema drifted")
    config = _base._closed_object(
        receipt["preprocess_config"],
        expected=frozenset(field.name for field in fields(CausalEDFConfig)),
        field="preprocess_config",
    )
    if _base._canonical_json_bytes(config) != _base._canonical_json_bytes(
        _base._config_payload(CausalEDFConfig())
    ):
        raise ValueError("Signal universe requires the frozen causal config")
    if receipt["preprocess_config_sha256"] != _base._config_sha256(
        CausalEDFConfig()
    ):
        raise ValueError("Signal-universe preprocessing hash drifted")

    for field in (
        "identity_record_count",
        "identity_patient_count",
        "candidate_event_count",
        "eligible_event_count",
        "excluded_event_count",
        "eligible_patient_count",
    ):
        _strict_nonnegative_int(receipt[field], field=field)

    record_hashes = receipt["identity_record_sha256s"]
    patient_ids = receipt["identity_patient_ids"]
    if (
        not isinstance(record_hashes, list)
        or record_hashes != sorted(set(record_hashes))
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in record_hashes
        )
    ):
        raise ValueError("identity_record_sha256s must be sorted unique SHA256s")
    if (
        not isinstance(patient_ids, list)
        or patient_ids != sorted(set(patient_ids))
        or any(not isinstance(value, str) or not value for value in patient_ids)
    ):
        raise ValueError("identity_patient_ids must be sorted unique strings")
    if receipt["identity_record_count"] != len(record_hashes):
        raise ValueError("identity_record_count disagrees with record receipts")
    if receipt["identity_patient_count"] != len(patient_ids):
        raise ValueError("identity_patient_count disagrees with patient roster")
    if receipt["identity_record_roster_sha256"] != _base._roster_sha256(
        record_hashes
    ):
        raise ValueError("identity_record_roster_sha256 disagrees with receipts")
    if receipt["identity_patient_roster_sha256"] != _base._roster_sha256(
        patient_ids
    ):
        raise ValueError("identity_patient_roster_sha256 disagrees with patient IDs")
    record_hash_set = set(record_hashes)
    patient_id_set = set(patient_ids)

    events_value = receipt["events"]
    exclusions_value = receipt["exclusions"]
    if not isinstance(events_value, list) or not isinstance(exclusions_value, list):
        raise ValueError("Signal-universe events/exclusions must be JSON arrays")
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
    event_ids = [str(row["event_id"]) for row in events]
    excluded_ids = [str(row["event_id"]) for row in exclusions]
    if event_ids != sorted(event_ids) or excluded_ids != sorted(excluded_ids):
        raise ValueError("Signal-universe event arrays are not canonically ordered")
    candidate_ids = sorted((*event_ids, *excluded_ids))
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Signal universe contains duplicate candidate events")
    eligible_patients = sorted({str(row["patient_id"]) for row in events})
    for index, row in enumerate((*events, *exclusions)):
        patient_id = str(row["patient_id"])
        official_split = str(row["official_split"])
        if patient_id not in patient_id_set:
            raise ValueError("Signal event is outside the identity-overlay roster")
        if official_split not in _MODEL_SPLIT_BY_OFFICIAL:
            raise ValueError("Signal event has an invalid official TUSZ split")
        if str(row["model_split"]) != _MODEL_SPLIT_BY_OFFICIAL[official_split]:
            raise ValueError("Model split is not mechanically derived from TUSZ split")
        for field in (
            "event_record_sha256",
            "crosswalk_record_sha256",
            "deepsoz_source_record_sha256",
            "edf_sha256",
            "annotation_pair_sha256",
        ):
            _base._require_sha256(row[field], field=f"candidate[{index}].{field}")
        if str(row["crosswalk_record_sha256"]) not in record_hash_set:
            raise ValueError("Signal event is not bound to an identity-audit record")
        if row["deepsoz_source_record_sha256"] != row["crosswalk_record_sha256"]:
            raise ValueError("Identity-overlay record hash aliases drifted")
    if not set(eligible_patients) <= patient_id_set:
        raise ValueError("Eligible patients are outside the identity overlay")
    count_checks = {
        "candidate_event_count": len(candidate_ids),
        "eligible_event_count": len(events),
        "excluded_event_count": len(exclusions),
        "eligible_patient_count": len(eligible_patients),
    }
    for field, expected in count_checks.items():
        if receipt[field] != expected:
            raise ValueError(f"{field} disagrees with its stored roster")
    roster_checks = {
        "candidate_event_roster_sha256": candidate_ids,
        "eligible_event_roster_sha256": event_ids,
        "excluded_event_roster_sha256": excluded_ids,
        "eligible_patient_roster_sha256": eligible_patients,
    }
    for field, roster in roster_checks.items():
        if receipt[field] != _base._roster_sha256(roster):
            raise ValueError(f"{field} disagrees with its stored roster")
    _validate_split_counts(
        receipt["candidate_official_split_event_counts"],
        field="candidate_official_split_event_counts",
        total=len(candidate_ids),
    )
    _validate_split_counts(
        receipt["eligible_official_split_event_counts"],
        field="eligible_official_split_event_counts",
        total=len(events),
    )
    if receipt["candidate_official_split_event_counts"] != _split_counts(
        (*events, *exclusions)
    ):
        raise ValueError("Candidate official-split counts disagree with events")
    if receipt["eligible_official_split_event_counts"] != _split_counts(events):
        raise ValueError("Eligible official-split counts disagree with events")
    _validate_exclusion_counts(receipt["exclusion_code_counts"], exclusions)
    return receipt


@dataclass(frozen=True)
class VerifiedTargetIndependentSignalUniverse:
    receipt: Mapping[str, object]
    artifact_sha256: str
    receipt_sha256: str

    @property
    def eligible_event_ids(self) -> tuple[str, ...]:
        return tuple(str(row["event_id"]) for row in self.receipt["events"])

    @property
    def eligible_patient_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({str(row["patient_id"]) for row in self.receipt["events"]})
        )


def _identity_record_payload(row: Mapping[str, object]) -> dict[str, object]:
    return {
        field: str(row[field]).strip()
        for field in _IDENTITY_AUDIT_COLUMNS
    }


def _build_receipt(
    identity_audit_csv: str | Path,
    tusz_root: str | Path,
    *,
    expected_identity_audit_sha256: str,
    config: CausalEDFConfig,
    reader_factory: Callable[[str], object] | None,
    expected_identity_record_count: int | None,
    expected_identity_patient_count: int | None,
    expected_candidate_event_count: int | None,
    expected_eligible_event_count: int | None,
) -> dict[str, object]:
    if not isinstance(config, CausalEDFConfig):
        raise TypeError("config must be CausalEDFConfig")
    if _base._canonical_json_bytes(_base._config_payload(config)) != (
        _base._canonical_json_bytes(_base._config_payload(CausalEDFConfig()))
    ):
        raise ValueError("Formal target-independent replay requires the frozen config")
    root = _base._reject_symlink_components(Path(tusz_root), field="TUSZ root")
    if not root.is_dir():
        raise FileNotFoundError("TUSZ root directory does not exist")
    audit, audit_sha = _base._strict_csv(
        identity_audit_csv,
        expected_sha256=expected_identity_audit_sha256,
        allowed_columns=_IDENTITY_AUDIT_COLUMNS,
        label="identity_audit",
    )
    if audit.empty:
        raise ValueError("Identity audit cannot be empty")
    if set(audit["schema_version"]) != {IDENTITY_RECOVERY_SCHEMA}:
        raise ValueError("Identity-audit schema drifted")
    if set(audit["policy"]) != {IDENTITY_RECOVERY_POLICY}:
        raise ValueError("Identity-audit policy drifted")
    deepsoz_rows = [
        _base._strict_int(value, field="identity_audit.deepsoz_row")
        for value in audit["deepsoz_row"]
    ]
    if deepsoz_rows != list(range(len(audit))):
        raise ValueError(
            "Target-independent universe requires every ordered identity-audit row"
        )

    patient_to_local: dict[str, str] = {}
    local_to_patient: dict[str, str] = {}
    patient_to_split: dict[str, str] = {}
    relative_paths: set[str] = set()
    identity_row_hashes: list[str] = []
    identity_patients: set[str] = set()
    accepted: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    candidate_ids: list[str] = []
    preprocess_sha = _base._config_sha256(config)

    for deepsoz_row, raw in enumerate(audit.to_dict("records")):
        original_status = str(raw["original_mapping_status"]).strip()
        recovery_status = str(raw["recovery_status"]).strip()
        expected_recovery = (
            "conservative_unique_preserved"
            if original_status == "unique"
            else "identity_recovered"
            if original_status in {"ambiguous", "unmapped"}
            else None
        )
        if recovery_status != expected_recovery:
            raise ValueError("Identity audit does not close every source record")
        for field in (
            "path_candidate_count",
            "split_match",
            "session_year_match",
            "montage_match",
            "record_key_match",
            "patient_binding_match",
            "nsamples_match",
        ):
            if _base._strict_int(raw[field], field=f"identity_audit.{field}") != 1:
                raise ValueError(f"Identity evidence failed: {field}")
        patient_id = normalize_patient_id(raw["deepsoz_patient"])
        local_patient = _base._clean(
            raw["local_patient"], field="identity_audit.local_patient"
        )
        relative_edf_declared = _base._clean(
            raw["relative_edf_path"], field="identity_audit.relative_edf_path"
        )
        relative_channel_declared = Path(relative_edf_declared).with_suffix(
            ".csv"
        ).as_posix()
        relative_global_declared = Path(relative_edf_declared).with_suffix(
            ".csv_bi"
        ).as_posix()
        (
            relative_edf,
            edf_path,
            relative_channel,
            channel_path,
            relative_global,
            global_path,
            official_split,
            derived_local_patient,
            local_record_key,
        ) = _base._canonical_tusz_record_identity(
            root,
            relative_edf_declared,
            relative_channel_declared,
            relative_global_declared,
        )
        if relative_edf in relative_paths:
            raise ValueError("Identity audit repeats a local EDF path")
        relative_paths.add(relative_edf)
        if local_patient != derived_local_patient:
            raise ValueError("Identity audit local-patient binding drifted")
        if local_record_key != _base._source_record_key(raw["deepsoz_record"]):
            raise ValueError("Identity audit local/source record key drifted")
        if patient_to_local.setdefault(patient_id, local_patient) != local_patient:
            raise ValueError("One DeepSOZ identity maps to multiple local patients")
        if local_to_patient.setdefault(local_patient, patient_id) != patient_id:
            raise ValueError("One local patient maps to multiple DeepSOZ identities")
        if patient_to_split.setdefault(patient_id, official_split) != official_split:
            raise ValueError("One patient spans multiple official TUSZ splits")
        identity_patients.add(patient_id)

        identity_payload = _identity_record_payload(raw)
        identity_row_sha = _base._canonical_sha256(identity_payload)
        identity_row_hashes.append(identity_row_sha)
        pair = inspect_tusz_annotation_pair(
            channel_path, global_path, source_path=edf_path
        )
        declared_local_events = _base._strict_int(
            raw["local_event_count"], field="identity_audit.local_event_count"
        )
        if declared_local_events != len(pair.global_seizure_events):
            raise ValueError(
                "Identity audit event count differs from local TERM,seiz timeline"
            )
        for global_event in pair.global_seizure_events:
            event_id = f"{edf_path.stem}__ev{global_event.event_index:04d}"
            if event_id in candidate_ids:
                raise ValueError("Target-independent candidate event IDs collide")
            candidate_ids.append(event_id)
            if global_event.start_sec < 0 or global_event.stop_sec <= global_event.start_sec:
                raise ValueError("Local TUSZ global seizure interval is invalid")
            event_identity = {
                "identity_audit_record_sha256": identity_row_sha,
                "deepsoz_row": deepsoz_row,
                "patient_id": patient_id,
                "relative_edf_path": relative_edf,
                "global_event_index": global_event.event_index,
                "global_t0_sec": float(global_event.start_sec),
                "global_stop_sec": float(global_event.stop_sec),
            }
            event_record_sha = _base._canonical_sha256(event_identity)
            common = {
                "event_id": event_id,
                "event_record_sha256": event_record_sha,
                "crosswalk_record_sha256": identity_row_sha,
                "deepsoz_source_record_sha256": identity_row_sha,
                "patient_id": patient_id,
                "local_patient_id": local_patient,
                "official_split": official_split,
                "model_split": _MODEL_SPLIT_BY_OFFICIAL[official_split],
                "relative_edf_path": relative_edf,
                "deepsoz_record": _base._clean(
                    raw["deepsoz_record"], field="identity_audit.deepsoz_record"
                ),
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
                raise RuntimeError("EDF loader and annotation hashes disagree")
            if (
                abs(loaded.edf_receipt.requested_onset_sec - global_event.start_sec)
                > _TIME_TOLERANCE_SEC
            ):
                raise RuntimeError("EDF replay used the wrong local TUSZ t0")
            if (
                tuple(loaded.window.data.shape) != (19, 12_000)
                or loaded.window.onset_index != 2_400
                or abs(loaded.window.sfreq_hz - 200.0) > _TIME_TOLERANCE_SEC
            ):
                raise RuntimeError(
                    "Frozen preprocessing must produce [19,12000] at 200 Hz"
                )
            edf_receipt = asdict(loaded.edf_receipt)
            signal_receipt = asdict(loaded.signal_receipt)
            accepted.append(
                {
                    **common,
                    "deepsoz_row": deepsoz_row,
                    "relative_channel_annotation_path": relative_channel,
                    "relative_global_annotation_path": relative_global,
                    "global_seizure_type": global_event.seizure_type,
                    "window_start_sec": float(
                        global_event.start_sec - config.pre_onset_sec
                    ),
                    "window_stop_sec": float(
                        global_event.start_sec + config.post_onset_sec
                    ),
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
    eligible_ids = [str(row["event_id"]) for row in accepted]
    excluded_ids = [str(row["event_id"]) for row in excluded]
    if sorted((*eligible_ids, *excluded_ids)) != candidate_ids:
        raise RuntimeError("Signal replay did not close the complete candidate roster")
    patients = sorted(identity_patients)
    eligible_patients = sorted({str(row["patient_id"]) for row in accepted})
    exclusion_counts: dict[str, int] = {}
    for row in excluded:
        code = str(row["eligibility_code"])
        exclusion_counts[code] = exclusion_counts.get(code, 0) + 1
    receipt: dict[str, object] = {
        "schema_version": TARGET_INDEPENDENT_SIGNAL_UNIVERSE_SCHEMA,
        "policy": TARGET_INDEPENDENT_SIGNAL_UNIVERSE_POLICY,
        "lineage_axes": _lineage_axes(),
        "benchmark_identity_overlay_conditioned": True,
        "roster_scope": "complete_deepsoz_identity_overlay_not_complete_tusz",
        "identity_audit_sha256": audit_sha,
        "preprocess_schema": EDF_PREPROCESS_SCHEMA,
        "preprocess_config": _base._config_payload(config),
        "preprocess_config_sha256": preprocess_sha,
        "identity_record_count": len(audit),
        "identity_patient_count": len(patients),
        "identity_record_sha256s": sorted(identity_row_hashes),
        "identity_patient_ids": patients,
        "identity_record_roster_sha256": _base._roster_sha256(
            identity_row_hashes
        ),
        "identity_patient_roster_sha256": _base._roster_sha256(patients),
        "candidate_event_count": len(candidate_ids),
        "eligible_event_count": len(accepted),
        "excluded_event_count": len(excluded),
        "eligible_patient_count": len(eligible_patients),
        "candidate_event_roster_sha256": _base._roster_sha256(candidate_ids),
        "eligible_event_roster_sha256": _base._roster_sha256(eligible_ids),
        "excluded_event_roster_sha256": _base._roster_sha256(excluded_ids),
        "eligible_patient_roster_sha256": _base._roster_sha256(
            eligible_patients
        ),
        "candidate_official_split_event_counts": _split_counts(
            (*accepted, *excluded)
        ),
        "eligible_official_split_event_counts": _split_counts(accepted),
        "exclusion_code_counts": [
            [code, exclusion_counts[code]] for code in sorted(exclusion_counts)
        ],
        "events": accepted,
        "exclusions": excluded,
    }
    _validate_receipt(receipt)
    expectations = {
        "identity_record_count": expected_identity_record_count,
        "identity_patient_count": expected_identity_patient_count,
        "candidate_event_count": expected_candidate_event_count,
        "eligible_event_count": expected_eligible_event_count,
    }
    for field, expected in expectations.items():
        if expected is not None and receipt[field] != expected:
            raise ValueError(
                f"Formal target-independent universe expected {field}={expected}, "
                f"got {receipt[field]}"
            )
    return receipt


def _publish_receipt(
    receipt: Mapping[str, object], output_directory: str | Path
) -> VerifiedTargetIndependentSignalUniverse:
    receipt_sha = _base._canonical_sha256(receipt)
    artifact = {
        "schema_version": TARGET_INDEPENDENT_SIGNAL_UNIVERSE_ARTIFACT_SCHEMA,
        "serialization": "canonical_json_utf8_newline_no_pickle",
        "receipt_sha256": receipt_sha,
        "receipt": receipt,
    }
    encoded = _base._canonical_json_bytes(artifact)
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("Target-independent signal artifact exceeds size limit")
    output = _base._reject_symlink_components(
        Path(output_directory), field="target-independent signal output"
    )
    if output.name in {"", ".", ".."}:
        raise ValueError("Output requires a concrete directory name")
    if os.path.lexists(output):
        raise FileExistsError("Target-independent signal destination exists")
    parent = _base._reject_symlink_components(output.parent, field="output parent")
    if not parent.is_dir():
        raise FileNotFoundError("Target-independent signal output parent is missing")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    published = False
    try:
        artifact_path = temporary / TARGET_INDEPENDENT_SIGNAL_UNIVERSE_FILENAME
        with artifact_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _base._fsync_directory(temporary)
        if os.path.lexists(output):
            raise FileExistsError("Target-independent signal destination exists")
        os.rename(temporary, output)
        published = True
        _base._fsync_directory(parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    return VerifiedTargetIndependentSignalUniverse(
        receipt=receipt,
        artifact_sha256=_base._bytes_sha256(encoded),
        receipt_sha256=receipt_sha,
    )


def build_target_independent_signal_universe(
    identity_audit_csv: str | Path,
    tusz_root: str | Path,
    output_directory: str | Path,
    *,
    expected_identity_audit_sha256: str,
    config: CausalEDFConfig = CausalEDFConfig(),
    reader_factory: Callable[[str], object] | None = None,
) -> VerifiedTargetIndependentSignalUniverse:
    """Replay the formally fixed 652-record identity-overlay universe.

    Roster expectations are intentionally not caller-overridable.  Synthetic
    tests may exercise the private receipt producer, but every public artifact
    built through this API must close 652 records, 124 patients, and 1812
    local ``TERM,seiz`` candidates before publication.
    """

    if os.path.lexists(output_directory):
        raise FileExistsError("Target-independent signal destination exists")
    receipt = _build_receipt(
        identity_audit_csv,
        tusz_root,
        expected_identity_audit_sha256=expected_identity_audit_sha256,
        config=config,
        reader_factory=reader_factory,
        expected_identity_record_count=652,
        expected_identity_patient_count=124,
        expected_candidate_event_count=1812,
        expected_eligible_event_count=None,
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
        artifact = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Target-independent artifact is not strict JSON") from exc
    artifact = _base._closed_object(
        artifact, expected=_ARTIFACT_FIELDS, field="signal-universe artifact"
    )
    if _base._canonical_json_bytes(artifact) != encoded:
        raise ValueError("Target-independent artifact bytes are not canonical")
    if artifact["schema_version"] != TARGET_INDEPENDENT_SIGNAL_UNIVERSE_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported target-independent artifact schema")
    if artifact["serialization"] != "canonical_json_utf8_newline_no_pickle":
        raise ValueError("Target-independent artifact serialization is unsafe")
    receipt = _validate_receipt(artifact["receipt"])
    declared = _base._require_sha256(
        artifact["receipt_sha256"], field="receipt_sha256"
    )
    if declared != _base._canonical_sha256(receipt):
        raise ValueError("Target-independent artifact receipt SHA mismatch")
    return artifact, receipt


def load_target_independent_signal_universe(
    bundle_directory: str | Path,
    *,
    expected_artifact_sha256: str,
) -> VerifiedTargetIndependentSignalUniverse:
    """Strictly load one previously published signal-universe artifact."""

    bundle = _base._reject_symlink_components(
        Path(bundle_directory), field="target-independent signal bundle"
    )
    if not bundle.is_dir():
        raise FileNotFoundError("Target-independent signal bundle is missing")
    entries = tuple(sorted(bundle.iterdir(), key=lambda path: path.name))
    if (
        len(entries) != 1
        or entries[0].name != TARGET_INDEPENDENT_SIGNAL_UNIVERSE_FILENAME
        or entries[0].is_symlink()
        or not entries[0].is_file()
    ):
        raise ValueError("Target-independent signal bundle violates closed schema")
    encoded, artifact_sha = _base._read_stable_regular_file(
        entries[0],
        field="target-independent signal artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    _base._check_expected_sha(
        artifact_sha,
        expected_artifact_sha256,
        field="expected_target_independent_signal_artifact_sha256",
    )
    _, receipt = _parse_artifact(encoded)
    return VerifiedTargetIndependentSignalUniverse(
        receipt=receipt,
        artifact_sha256=artifact_sha,
        receipt_sha256=_base._canonical_sha256(receipt),
    )


__all__ = [
    "TARGET_INDEPENDENT_SIGNAL_UNIVERSE_ARTIFACT_SCHEMA",
    "TARGET_INDEPENDENT_SIGNAL_UNIVERSE_FILENAME",
    "TARGET_INDEPENDENT_SIGNAL_UNIVERSE_POLICY",
    "TARGET_INDEPENDENT_SIGNAL_UNIVERSE_SCHEMA",
    "VerifiedTargetIndependentSignalUniverse",
    "build_target_independent_signal_universe",
    "load_target_independent_signal_universe",
]
