#!/usr/bin/env python3
"""Append recovered identity-v3 events to the frozen 988-event phenotype cache."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import materialize_deepsoz_event_phenotypes_target_free as legacy_core  # noqa: E402
from src.soz.clinical_reporting import (  # noqa: E402
    EventScalpPhenotypeAbstention,
    EventScalpPhenotypeEvidence,
)
from src.soz.data.deepsoz_signal_identity_recovery import (  # noqa: E402
    DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_ARTIFACT_SCHEMA,
    DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_FILENAME,
    DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_SCHEMA,
    load_deepsoz_signal_identity_recovery_bundle,
)
from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.event_phenotype_producer import (  # noqa: E402
    EVENT_PHENOTYPE_PRODUCER_SCHEMA,
    EventPhenotypeProducerIdentity,
    produce_event_scalp_phenotype,
)
from src.soz.event_reference_consistency import (  # noqa: E402
    EVENT_REFERENCE_TEMPORAL_TOLERANCE_SEC,
    assess_event_reference_consistency,
)
from src.soz.fine_temporal_evidence import extract_fine_temporal_evidence  # noqa: E402
from src.soz.later_visible_region_producer import (  # noqa: E402
    LATER_VISIBLE_REGION_PRODUCER_SCHEMA,
    LATER_VISIBLE_REGION_RECEIPT_SCHEMA,
    build_later_visible_region_receipt,
    produce_later_visible_region,
)
from src.soz.preprocessing_arm_runtime import CAUSAL_REFERENCE_PAIR_SCHEMA  # noqa: E402


DEFAULT_SIGNAL_RECOVERY_ARTIFACT = (
    ROOT
    / "outputs/deepsoz_signal_preflight_identity_v3_20260812"
    / DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_FILENAME
)
DEFAULT_LEGACY_PHENOTYPE = (
    ROOT / "outputs/deepsoz_event_phenotype_target_free_oof_v1_20260812.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = (
    ROOT / "outputs/deepsoz_event_phenotype_identity_v12_target_free_20260812.json"
)

EXPECTED_SIGNAL_RECOVERY_ARTIFACT_SHA256 = (
    "2a6bb8a7be20993949e7250b10c83d11fe027ff1afc0fa0919124f7fa371ef8e"
)
EXPECTED_SIGNAL_RECOVERY_RECEIPT_SHA256 = (
    "be79e8dc70f553976864a3b8ac6d85a24ff7d7a9769e9e11739ecff43ad701e3"
)
EXPECTED_LEGACY_PHENOTYPE_SHA256 = (
    "a30b94ad623f29c2b9dbb9fc562f700706c980d44c3016eb1a55716bf8d2c90c"
)
LEGACY_PHENOTYPE_SCHEMA = "soz_deepsoz_event_phenotype_target_free_oof_v1"
LEGACY_PHENOTYPE_STATUS = (
    "completed_target_free_development_signal_application_not_evaluation"
)
OUTPUT_SCHEMA = "soz_deepsoz_event_phenotype_identity_v12_target_free_v1"
OUTPUT_STATUS = (
    "completed_target_free_identity_v12_signal_application_not_evaluation"
)
LEGACY_EVENT_COUNT = 988
RECOVERED_EVENT_COUNT = 161
COMBINED_EVENT_COUNT = 1149


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json(path: Path, *, expected_sha256: str) -> dict[str, object]:
    source = legacy_core._absolute_no_symlink(path, field="legacy phenotype artifact")
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Legacy phenotype artifact SHA mismatch")

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
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Legacy phenotype artifact is not strict JSON") from exc
    if not isinstance(payload, dict) or _canonical_bytes(payload, newline=True) != raw:
        raise ValueError("Legacy phenotype artifact is not canonical JSON")
    return payload


def _receipt_sha_matches(row: Mapping[str, object]) -> bool:
    receipt = row.get("event_reference_consistency_receipt")
    declared = row.get("event_reference_consistency_receipt_sha256")
    return isinstance(receipt, Mapping) and declared == _canonical_sha256(receipt)


def _validate_legacy_event(
    row: Mapping[str, object],
    signal_row: Mapping[str, object],
    *,
    ordinal: int,
) -> None:
    identity_fields = (
        "patient_id",
        "local_patient_id",
        "event_id",
        "relative_edf_path",
        "global_t0_sec",
        "global_stop_sec",
        "global_event_index",
        "official_split",
        "model_split",
        "edf_sha256",
        "event_record_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
    )
    if row.get("ordinal") != ordinal or any(
        row.get(field) != signal_row.get(field) for field in identity_fields
    ):
        raise ValueError(f"Legacy phenotype identity drifted: {row.get('event_id')}")
    if not _receipt_sha_matches(row):
        raise ValueError("Legacy event-reference receipt SHA mismatch")
    status = row.get("status")
    if status not in {"reportable", "abstained"}:
        raise ValueError("Legacy phenotype has an invalid status")
    if (row.get("phenotype") is not None) != (status == "reportable") or (
        row.get("abstention") is not None
    ) != (status == "abstained"):
        raise ValueError("Legacy phenotype typed status is inconsistent")
    primary = row.get("primary_arm")
    sensitivity = row.get("sensitivity_arm")
    if not isinstance(primary, Mapping) or not isinstance(sensitivity, Mapping):
        raise TypeError("Legacy phenotype lacks paired arm outputs")
    if (
        primary.get("arm_id") != "C-CAR19"
        or sensitivity.get("arm_id") != "C-REF19"
        or primary.get("processed_window_sha256")
        != signal_row.get("processed_window_sha256")
    ):
        raise ValueError("Legacy phenotype arm binding changed")
    bound = row.get("phenotype") or row.get("abstention")
    if not isinstance(bound, Mapping) or not isinstance(bound.get("receipt"), Mapping):
        raise TypeError("Legacy phenotype lacks its bound event receipt")
    receipt = bound["receipt"]
    if (
        receipt.get("patient_pseudonym") != row.get("patient_id")
        or receipt.get("event_pseudonym") != row.get("event_id")
        or receipt.get("signal_artifact_sha256") != row.get("edf_sha256")
        or receipt.get("montages") != ["C-CAR19", "C-REF19"]
        or receipt.get("soz_labels_used_for_event_evidence") is not False
        or receipt.get("private_labels_used_for_event_evidence") is not False
    ):
        raise ValueError("Legacy bound event receipt identity changed")
    reference = row["event_reference_consistency_receipt"]
    if (
        reference.get("patient_pseudonym") != row.get("patient_id")
        or reference.get("event_pseudonym") != row.get("event_id")
        or reference.get("target_labels_used") is not False
        or reference.get("private_data_used") is not False
        or reference.get("localization_scores_used") is not False
        or reference.get("training_performed") is not False
    ):
        raise ValueError("Legacy reference receipt is not target-free")
    later = row.get("later_visible_region")
    if not isinstance(later, Mapping):
        raise TypeError("Legacy phenotype lacks later-region availability")
    later_receipt = later.get("receipt")
    if later_receipt is None:
        if later.get("receipt_sha256") is not None or later.get("status") != "unavailable":
            raise ValueError("Legacy unavailable later-region receipt is malformed")
    elif (
        not isinstance(later_receipt, Mapping)
        or later.get("status") != "mapped"
        or later.get("receipt_sha256") != _canonical_sha256(later_receipt)
        or later_receipt.get("patient_pseudonym") != row.get("patient_id")
        or later_receipt.get("event_pseudonym") != row.get("event_id")
        or later_receipt.get("target_labels_used") is not False
        or later_receipt.get("private_data_used") is not False
        or later_receipt.get("localization_scores_used") is not False
        or later_receipt.get("training_performed") is not False
    ):
        raise ValueError("Legacy later-region receipt is not exactly bound")
    availability = row.get("slot_availability")
    if not isinstance(availability, Mapping) or (
        availability.get("artifact_assessment") is not False
    ):
        raise ValueError("Legacy artifact slot must remain unavailable")


def _load_and_validate_legacy(
    path: Path,
    *,
    recovery_receipt: Mapping[str, object],
) -> tuple[dict[str, object], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    payload = _strict_json(path, expected_sha256=EXPECTED_LEGACY_PHENOTYPE_SHA256)
    if (
        payload.get("schema_version") != LEGACY_PHENOTYPE_SCHEMA
        or payload.get("status") != LEGACY_PHENOTYPE_STATUS
        or payload.get("producer_schema") != EVENT_PHENOTYPE_PRODUCER_SCHEMA
        or payload.get("reference_pair_schema") != CAUSAL_REFERENCE_PAIR_SCHEMA
        or payload.get("later_visible_region_producer_schema")
        != LATER_VISIBLE_REGION_PRODUCER_SCHEMA
        or payload.get("later_visible_region_receipt_schema")
        != LATER_VISIBLE_REGION_RECEIPT_SCHEMA
    ):
        raise ValueError("Legacy phenotype producer/schema contract changed")
    source = payload.get("source_preflight")
    if not isinstance(source, Mapping) or (
        source.get("artifact_sha256")
        != recovery_receipt.get("base_signal_preflight_artifact_sha256")
        or source.get("receipt_sha256")
        != recovery_receipt.get("base_signal_preflight_receipt_sha256")
        or source.get("preprocess_schema")
        != recovery_receipt.get("preprocess_schema")
        or source.get("preprocess_config_sha256")
        != recovery_receipt.get("preprocess_config_sha256")
        or source.get("eligible_event_count") != LEGACY_EVENT_COUNT
    ):
        raise ValueError("Legacy phenotype and recovery-v3 base lineage differ")
    access = payload.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(key) is not False
        for key in (
            "deepsoz_target_values_loaded",
            "deepsoz_target_fields_accessed",
            "tusz_native_target_values_loaded",
            "private_eeg_loaded",
            "private_target_values_loaded",
            "localization_scores_loaded",
            "training_performed",
            "model_selection_performed",
            "calibration_performed",
            "threshold_selection_performed",
        )
    ):
        raise ValueError("Legacy phenotype cache is not target-free")
    legacy_rows_value = payload.get("events")
    recovery_rows_value = recovery_receipt.get("events")
    recovered_ids_value = recovery_receipt.get("recovered_eligible_event_ids")
    if (
        not isinstance(legacy_rows_value, list)
        or not isinstance(recovery_rows_value, list)
        or not isinstance(recovered_ids_value, list)
    ):
        raise TypeError("Phenotype/recovery event rosters are missing")
    recovered_ids = tuple(str(value) for value in recovered_ids_value)
    if (
        len(recovered_ids) != RECOVERED_EVENT_COUNT
        or recovered_ids != tuple(sorted(set(recovered_ids)))
    ):
        raise ValueError("Recovery-v3 append roster is not 161 sorted unique events")
    recovered_set = set(recovered_ids)
    signal_by_id = {str(row["event_id"]): row for row in recovery_rows_value}
    if len(signal_by_id) != COMBINED_EVENT_COUNT:
        raise ValueError("Recovery-v3 signal event roster changed")
    base_signal_rows = tuple(
        signal_by_id[event_id]
        for event_id in sorted(set(signal_by_id) - recovered_set)
    )
    if len(base_signal_rows) != LEGACY_EVENT_COUNT:
        raise ValueError("Recovery-v3 base roster is not 988 events")
    legacy_rows = tuple(legacy_rows_value)
    if tuple(str(row["event_id"]) for row in legacy_rows) != tuple(
        str(row["event_id"]) for row in base_signal_rows
    ):
        raise ValueError("Legacy phenotype roster is not recovery-v3 base roster")
    for ordinal, (row, signal_row) in enumerate(zip(legacy_rows, base_signal_rows)):
        _validate_legacy_event(row, signal_row, ordinal=ordinal)
    recovered_rows = tuple(signal_by_id[event_id] for event_id in recovered_ids)
    return payload, legacy_rows, recovered_rows


def _select_recovered(
    rows: tuple[Mapping[str, object], ...], append_limit: int | None
) -> tuple[tuple[Mapping[str, object], ...], bool]:
    if append_limit is None:
        return rows, True
    if isinstance(append_limit, bool) or not 1 <= int(append_limit) < len(rows):
        raise ValueError("append_limit must be a smoke prefix in [1,160]")
    return rows[: int(append_limit)], False


def _materialize_one(
    source_row: Mapping[str, object],
    *,
    ordinal: int,
    raw_root: Path,
    config_car: CausalEDFConfig,
    config_ref: CausalEDFConfig,
) -> dict[str, object]:
    patient_id = str(source_row["patient_id"])
    event_id = str(source_row["event_id"])
    onset = float(source_row["global_t0_sec"])
    if not patient_id or not event_id or not math.isfinite(onset):
        raise ValueError("Recovered event has invalid identity/t0")
    edf_sha = str(source_row["edf_sha256"])
    path = legacy_core._safe_edf(raw_root, source_row["relative_edf_path"])
    loaded_car = load_standard19_edf_event(path, onset, config=config_car)
    loaded_ref = load_standard19_edf_event(path, onset, config=config_ref)
    if (
        loaded_car.edf_receipt.edf_sha256 != edf_sha
        or loaded_ref.edf_receipt.edf_sha256 != edf_sha
    ):
        raise ValueError("Recovered preflight and paired EDF receipts disagree")
    car = loaded_car.window.data.detach().cpu().float().contiguous()
    ref = loaded_ref.window.data.detach().cpu().float().contiguous()
    if tuple(car.shape) != (19, 12_000) or tuple(ref.shape) != (19, 12_000):
        raise ValueError("Recovered paired event windows must be [19,12000]")
    car_sha = legacy_core._tensor_sha256(car)
    ref_sha = legacy_core._tensor_sha256(ref)
    if car_sha != source_row["processed_window_sha256"]:
        raise ValueError("Recovered C-CAR19 replay differs from recovery-v3")
    car_evidence = extract_fine_temporal_evidence(car, sfreq_hz=200.0)
    ref_evidence = extract_fine_temporal_evidence(ref, sfreq_hz=200.0)
    primary = produce_event_scalp_phenotype(
        car,
        car_evidence,
        identity=EventPhenotypeProducerIdentity(
            patient_pseudonym=patient_id,
            event_pseudonym=event_id,
            signal_artifact_sha256=edf_sha,
            evidence_artifact_sha256=legacy_core._evidence_sha256(car_evidence),
        ),
        event_anchor_coordinate_sec=onset,
        time_coordinate_semantics="recording_start_seconds",
        sfreq_hz=200.0,
        reference_arm_id="C-CAR19",
    )
    sensitivity = produce_event_scalp_phenotype(
        ref,
        ref_evidence,
        identity=EventPhenotypeProducerIdentity(
            patient_pseudonym=patient_id,
            event_pseudonym=event_id,
            signal_artifact_sha256=edf_sha,
            evidence_artifact_sha256=legacy_core._evidence_sha256(ref_evidence),
        ),
        event_anchor_coordinate_sec=onset,
        time_coordinate_semantics="recording_start_seconds",
        sfreq_hz=200.0,
        reference_arm_id="C-REF19",
    )
    consistency = assess_event_reference_consistency(primary, sensitivity)
    bound = consistency.event
    if isinstance(bound, EventScalpPhenotypeEvidence) and bound.later_visible_derivations:
        production = produce_later_visible_region(bound.later_visible_derivations)
        receipt = build_later_visible_region_receipt(production, bound.receipt)
        later = {
            "status": "mapped",
            "reason_codes": [],
            "receipt": asdict(receipt),
            "receipt_sha256": receipt.receipt_sha256,
        }
    else:
        later = {
            "status": "unavailable",
            "reason_codes": [
                "primary_event_phenotype_abstained"
                if isinstance(bound, EventScalpPhenotypeAbstention)
                else "no_observed_later_visible_derivations"
            ],
            "receipt": None,
            "receipt_sha256": None,
        }
    availability = legacy_core._slot_availability(
        bound, later_region_available=later["receipt"] is not None
    )
    return {
        "ordinal": ordinal,
        "patient_id": patient_id,
        "local_patient_id": str(source_row["local_patient_id"]),
        "event_id": event_id,
        "relative_edf_path": str(source_row["relative_edf_path"]),
        "global_t0_sec": onset,
        "global_stop_sec": float(source_row["global_stop_sec"]),
        "global_event_index": int(source_row["global_event_index"]),
        "official_split": str(source_row["official_split"]),
        "model_split": str(source_row["model_split"]),
        "edf_sha256": edf_sha,
        "event_record_sha256": str(source_row["event_record_sha256"]),
        "edf_receipt_sha256": str(source_row["edf_receipt_sha256"]),
        "signal_receipt_sha256": str(source_row["signal_receipt_sha256"]),
        "processed_window_sha256": car_sha,
        "status": primary.status,
        "reason_codes": list(primary.reason_codes),
        "phenotype": asdict(bound) if isinstance(bound, EventScalpPhenotypeEvidence) else None,
        "abstention": asdict(bound) if isinstance(bound, EventScalpPhenotypeAbstention) else None,
        "primary_arm": legacy_core._serialize_production(
            primary, arm_id="C-CAR19", processed_window_sha256=car_sha
        ),
        "sensitivity_arm": legacy_core._serialize_production(
            sensitivity, arm_id="C-REF19", processed_window_sha256=ref_sha
        ),
        "event_reference_consistency_receipt": asdict(consistency.receipt),
        "event_reference_consistency_receipt_sha256": consistency.receipt.receipt_sha256,
        "later_visible_region": later,
        "slot_availability": availability,
    }


def _counts(events: Sequence[Mapping[str, object]]) -> dict[str, object]:
    primary: dict[str, int] = {}
    sensitivity: dict[str, int] = {}
    reference: dict[str, int] = {}
    later: dict[str, int] = {}
    slots = {name: 0 for name in legacy_core.SLOT_NAMES}
    for row in events:
        for counter, key in (
            (primary, str(row["status"])),
            (sensitivity, str(row["sensitivity_arm"]["status"])),
            (
                reference,
                str(
                    row["event_reference_consistency_receipt"].get(
                        "montage_stability"
                    )
                    or "unassessed"
                ),
            ),
            (later, str(row["later_visible_region"]["status"])),
        ):
            counter[key] = counter.get(key, 0) + 1
        for name, available in row["slot_availability"].items():
            if available:
                slots[name] += 1
    return {
        "primary_status": dict(sorted(primary.items())),
        "sensitivity_status": dict(sorted(sensitivity.items())),
        "reference_state": dict(sorted(reference.items())),
        "later_visible_region_state": dict(sorted(later.items())),
        "slot_available": slots,
    }


def materialize(
    *,
    signal_recovery_artifact: Path,
    legacy_phenotype_artifact: Path,
    tusz_root: Path,
    output: Path,
    append_limit: int | None,
    progress_every: int,
) -> tuple[dict[str, object], Mapping[str, object]]:
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("progress_every must be a positive integer")
    signal_path = legacy_core._absolute_no_symlink(
        signal_recovery_artifact, field="signal recovery artifact"
    )
    if signal_path.name != DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_FILENAME:
        raise ValueError("Signal recovery artifact filename is not frozen")
    signal = load_deepsoz_signal_identity_recovery_bundle(
        signal_path.parent,
        expected_artifact_sha256=EXPECTED_SIGNAL_RECOVERY_ARTIFACT_SHA256,
    )
    if (
        signal.receipt_sha256 != EXPECTED_SIGNAL_RECOVERY_RECEIPT_SHA256
        or signal.receipt.get("schema_version")
        != DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_SCHEMA
        or signal.receipt.get("combined_eligible_event_count")
        != COMBINED_EVENT_COUNT
        or signal.receipt.get("recovered_eligible_event_count")
        != RECOVERED_EVENT_COUNT
        or signal.receipt.get("base_eligible_event_count") != LEGACY_EVENT_COUNT
    ):
        raise ValueError("Signal recovery schema/count/receipt contract changed")
    legacy_payload, legacy_rows, recovered_rows = _load_and_validate_legacy(
        legacy_phenotype_artifact, recovery_receipt=signal.receipt
    )
    selected, full_scope = _select_recovered(recovered_rows, append_limit)
    config_payload = signal.receipt.get("preprocess_config")
    if not isinstance(config_payload, Mapping):
        raise TypeError("Signal recovery lacks preprocessing config")
    config_car = CausalEDFConfig(**dict(config_payload))
    if not config_car.apply_car19:
        raise ValueError("Signal recovery primary preprocessing is not C-CAR19")
    config_ref = replace(config_car, apply_car19=False)
    raw_root = legacy_core._absolute_no_symlink(tusz_root, field="TUSZ root")
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    target = legacy_core._absolute_no_symlink(output, field="output artifact")
    if os.path.lexists(target):
        raise FileExistsError(target)
    if target.parent != output.absolute().parent or not target.parent.is_dir():
        raise ValueError("Output parent must be an existing canonical directory")

    started = time.monotonic()
    new_rows: list[dict[str, object]] = []
    for position, row in enumerate(selected, start=1):
        new_rows.append(
            _materialize_one(
                row,
                ordinal=LEGACY_EVENT_COUNT + position - 1,
                raw_root=raw_root,
                config_car=config_car,
                config_ref=config_ref,
            )
        )
        if position % progress_every == 0 or position == len(selected):
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "new_event": position,
                        "new_total": len(selected),
                        "elapsed_sec": round(elapsed, 2),
                        "seconds_per_new_event": round(elapsed / position, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    events = [*(dict(row) for row in legacy_rows), *new_rows]
    if events[:LEGACY_EVENT_COUNT] != list(legacy_rows):
        raise RuntimeError("Legacy 988 phenotype rows changed during append")
    event_ids = [str(row["event_id"]) for row in events]
    expected_ids = [str(row["event_id"]) for row in (*legacy_rows, *selected)]
    if event_ids != expected_ids or len(set(event_ids)) != len(event_ids):
        raise RuntimeError("Extended phenotype event ordering contract failed")

    payload: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "producer_schema": EVENT_PHENOTYPE_PRODUCER_SCHEMA,
        "reference_pair_schema": CAUSAL_REFERENCE_PAIR_SCHEMA,
        "event_reference_temporal_tolerance_sec": EVENT_REFERENCE_TEMPORAL_TOLERANCE_SEC,
        "later_visible_region_producer_schema": LATER_VISIBLE_REGION_PRODUCER_SCHEMA,
        "later_visible_region_receipt_schema": LATER_VISIBLE_REGION_RECEIPT_SCHEMA,
        "scientific_boundary": dict(legacy_payload["scientific_boundary"]),
        "source_signal_recovery": {
            "artifact_schema_version": DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_ARTIFACT_SCHEMA,
            "receipt_schema_version": DEEPSOZ_SIGNAL_IDENTITY_RECOVERY_SCHEMA,
            "artifact_sha256": signal.artifact_sha256,
            "receipt_sha256": signal.receipt_sha256,
            "combined_eligible_event_count": signal.receipt[
                "combined_eligible_event_count"
            ],
            "combined_eligible_patient_count": signal.receipt[
                "combined_eligible_patient_count"
            ],
            "combined_eligible_event_roster_sha256": signal.receipt[
                "combined_eligible_event_roster_sha256"
            ],
            "recovered_eligible_event_count": signal.receipt[
                "recovered_eligible_event_count"
            ],
            "preprocess_schema": signal.receipt["preprocess_schema"],
            "preprocess_config_sha256": signal.receipt[
                "preprocess_config_sha256"
            ],
        },
        "legacy_phenotype_cache": {
            "schema_version": LEGACY_PHENOTYPE_SCHEMA,
            "artifact_sha256": EXPECTED_LEGACY_PHENOTYPE_SHA256,
            "event_count": LEGACY_EVENT_COUNT,
            "event_rows_sha256": _canonical_sha256(list(legacy_rows)),
        },
        "cache_extension_receipt": {
            "append_only": True,
            "legacy_988_event_rows_exact_prefix": events[:LEGACY_EVENT_COUNT]
            == list(legacy_rows),
            "legacy_raw_eeg_replayed": False,
            "new_raw_eeg_event_count": len(selected),
            "full_recovered_append": full_scope,
        },
        "access_receipt": {
            "input_event_selection": (
                "legacy_988_exact_plus_all_161_recovered_events"
                if full_scope
                else "legacy_988_exact_plus_first_n_recovered_events_smoke"
            ),
            "append_limit": append_limit,
            "input_signal_eligible_event_count": COMBINED_EVENT_COUNT,
            "materialized_event_count": len(events),
            "legacy_target_free_phenotype_cache_loaded": True,
            "raw_public_tusz_eeg_loaded": True,
            "raw_public_tusz_event_count": len(selected),
            "deepsoz_identity_roster_loaded": True,
            "deepsoz_preprocess_config_loaded": True,
            "deepsoz_target_values_loaded": False,
            "deepsoz_target_fields_accessed": False,
            "tusz_native_target_values_loaded": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "localization_scores_loaded": False,
            "training_performed": False,
            "model_selection_performed": False,
            "calibration_performed": False,
            "threshold_selection_performed": False,
        },
        "counts": {
            "input_signal_eligible_patients": signal.receipt[
                "combined_eligible_patient_count"
            ],
            "input_signal_eligible_events": COMBINED_EVENT_COUNT,
            "materialized_patients": len({str(row["patient_id"]) for row in events}),
            "materialized_events": len(events),
            "legacy_reused_events": LEGACY_EVENT_COUNT,
            "newly_computed_events": len(selected),
            **_counts(events),
        },
        "events": events,
        "elapsed_sec": time.monotonic() - started,
    }
    legacy_core._atomic_write_json(output, payload)
    return payload, legacy_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--signal-recovery-artifact",
        type=Path,
        default=DEFAULT_SIGNAL_RECOVERY_ARTIFACT,
    )
    parser.add_argument(
        "--legacy-phenotype-artifact",
        type=Path,
        default=DEFAULT_LEGACY_PHENOTYPE,
    )
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-limit", type=int)
    parser.add_argument("--progress-every", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload, _ = materialize(
        signal_recovery_artifact=args.signal_recovery_artifact,
        legacy_phenotype_artifact=args.legacy_phenotype_artifact,
        tusz_root=args.tusz_root,
        output=args.output,
        append_limit=args.append_limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "schema_version": payload["schema_version"],
                "output": str(args.output),
                "counts": payload["counts"],
                "target_values_loaded": False,
                "private_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
