#!/usr/bin/env python3
"""Audit v1 versus temporal-only v2 event phenotypes without target access.

The audit is intentionally narrow. It proves exact event-roster parity,
requires all non-rhythm observable facts to remain unchanged, and summarizes
only rhythm-state/frequency differences. It does not load labels, compute SOZ
performance, select a producer, or authorize clinical evolution wording.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping


V1_OUTPUT_SCHEMA = "soz_deepsoz_event_phenotype_target_free_oof_v1"
V2_OUTPUT_SCHEMA = "soz_deepsoz_event_phenotype_target_free_oof_v2"
SOURCE_STATUS = "completed_target_free_development_signal_application_not_evaluation"
V1_PRODUCER_SCHEMA = "soz_target_free_event_scalp_phenotype_producer_v1"
V2_PRODUCER_SCHEMA = "soz_target_free_event_scalp_phenotype_producer_v2"
OUTPUT_SCHEMA = "soz_event_phenotype_v1_v2_temporal_only_audit_v1"


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _events(payload: Mapping[str, object], *, name: str) -> list[Mapping[str, object]]:
    value = payload.get("events")
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{name}.events must be a list of objects")
    return value


def _require_target_free(payload: Mapping[str, object], *, name: str) -> None:
    access = _object(payload.get("access_receipt"), name=f"{name}.access_receipt")
    for field in (
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
    ):
        if access.get(field) is not False:
            raise ValueError(f"{name} is not target/private/selection free: {field}")


def _phenotype_core(value: object) -> object:
    if value is None:
        return None
    phenotype = dict(_object(value, name="phenotype"))
    phenotype.pop("receipt", None)
    phenotype.pop("rhythm_state", None)
    phenotype.pop("frequency_range_hz", None)
    return phenotype


def _abstention_core(value: object) -> object:
    if value is None:
        return None
    abstention = dict(_object(value, name="abstention"))
    abstention.pop("receipt", None)
    return abstention


def _arm_core(value: object) -> dict[str, object]:
    arm = _object(value, name="reference arm")
    return {
        "arm_id": arm.get("arm_id"),
        "status": arm.get("status"),
        "reason_codes": arm.get("reason_codes"),
        "detected_bipolar_edge_count": arm.get("detected_bipolar_edge_count"),
        "phenotype": _phenotype_core(arm.get("phenotype")),
        "abstention": _abstention_core(arm.get("abstention")),
    }


def _reference_core(value: object) -> dict[str, object]:
    receipt = _object(value, name="event reference receipt")
    fields = (
        "primary_arm_id",
        "sensitivity_arm_id",
        "primary_result_status",
        "sensitivity_result_status",
        "temporal_alignment_tolerance_sec",
        "onset_start_delta_sec",
        "primary_first_visible_derivations",
        "sensitivity_first_visible_derivations",
        "montage_stability",
        "reason_codes",
        "target_labels_used",
        "private_data_used",
        "localization_scores_used",
        "training_performed",
    )
    return {field: receipt.get(field) for field in fields}


def _later_core(value: object) -> dict[str, object]:
    later = _object(value, name="later-visible result")
    receipt_value = later.get("receipt")
    if receipt_value is None:
        receipt_core = None
    else:
        receipt = _object(receipt_value, name="later-visible receipt")
        receipt_core = {
            "observed_derivations": receipt.get("observed_derivations"),
            "later_visible_region_zh": receipt.get("later_visible_region_zh"),
            "mapping_status": receipt.get("mapping_status"),
            "mapping_reason_codes": receipt.get("mapping_reason_codes"),
            "target_labels_used": receipt.get("target_labels_used"),
            "private_data_used": receipt.get("private_data_used"),
            "localization_scores_used": receipt.get("localization_scores_used"),
            "training_performed": receipt.get("training_performed"),
        }
    return {
        "status": later.get("status"),
        "reason_codes": later.get("reason_codes"),
        "receipt": receipt_core,
    }


def _non_rhythm_core(row: Mapping[str, object]) -> dict[str, object]:
    slots = dict(_object(row.get("slot_availability"), name="slot_availability"))
    slots.pop("rhythm_state", None)
    slots.pop("frequency_range_hz", None)
    return {
        "patient_id": row.get("patient_id"),
        "local_patient_id": row.get("local_patient_id"),
        "event_id": row.get("event_id"),
        "relative_edf_path": row.get("relative_edf_path"),
        "global_t0_sec": row.get("global_t0_sec"),
        "global_stop_sec": row.get("global_stop_sec"),
        "global_event_index": row.get("global_event_index"),
        "official_split": row.get("official_split"),
        "model_split": row.get("model_split"),
        "status": row.get("status"),
        "reason_codes": row.get("reason_codes"),
        "phenotype": _phenotype_core(row.get("phenotype")),
        "abstention": _abstention_core(row.get("abstention")),
        "primary_arm": _arm_core(row.get("primary_arm")),
        "sensitivity_arm": _arm_core(row.get("sensitivity_arm")),
        "reference": _reference_core(
            row.get("event_reference_consistency_receipt")
        ),
        "later_visible": _later_core(row.get("later_visible_region")),
        "slot_availability": slots,
    }


def _rhythm_state(row: Mapping[str, object]) -> str:
    phenotype = row.get("phenotype")
    if phenotype is None:
        return "abstained"
    state = _object(phenotype, name="phenotype").get("rhythm_state")
    return "none" if state is None else str(state)


def _frequency(row: Mapping[str, object]) -> object:
    phenotype = row.get("phenotype")
    if phenotype is None:
        return None
    return _object(phenotype, name="phenotype").get("frequency_range_hz")


def audit(v1: dict[str, object], v2: dict[str, object]) -> dict[str, object]:
    expected = (
        (v1, "v1", V1_OUTPUT_SCHEMA, V1_PRODUCER_SCHEMA),
        (v2, "v2", V2_OUTPUT_SCHEMA, V2_PRODUCER_SCHEMA),
    )
    for payload, name, output_schema, producer_schema in expected:
        if payload.get("schema_version") != output_schema:
            raise ValueError(f"{name} output schema drifted")
        if payload.get("producer_schema") != producer_schema:
            raise ValueError(f"{name} producer schema drifted")
        if payload.get("status") != SOURCE_STATUS:
            raise ValueError(f"{name} status drifted")
        _require_target_free(payload, name=name)

    rows_v1 = _events(v1, name="v1")
    rows_v2 = _events(v2, name="v2")
    by_id_v1 = {str(row.get("event_id")): row for row in rows_v1}
    by_id_v2 = {str(row.get("event_id")): row for row in rows_v2}
    if len(by_id_v1) != len(rows_v1) or len(by_id_v2) != len(rows_v2):
        raise ValueError("duplicate event ID")
    if set(by_id_v1) != set(by_id_v2):
        raise ValueError("v1/v2 event rosters differ")

    transitions: Counter[tuple[str, str]] = Counter()
    frequency_changed: list[str] = []
    non_rhythm_mismatch: list[str] = []
    downgraded_patients: set[str] = set()
    for event_id in sorted(by_id_v1):
        old = by_id_v1[event_id]
        new = by_id_v2[event_id]
        if _non_rhythm_core(old) != _non_rhythm_core(new):
            non_rhythm_mismatch.append(event_id)
        old_state = _rhythm_state(old)
        new_state = _rhythm_state(new)
        transitions[(old_state, new_state)] += 1
        if _frequency(old) != _frequency(new):
            frequency_changed.append(event_id)
        if old_state == "evolving_rhythmic" and new_state == "rhythmic":
            downgraded_patients.add(str(new.get("patient_id")))
    if non_rhythm_mismatch:
        raise ValueError("v2 changed a non-rhythm observable fact")

    transition_rows = [
        {"v1": old, "v2": new, "events": count}
        for (old, new), count in sorted(transitions.items())
    ]
    v1_evolving = sum(
        count for (old, _), count in transitions.items() if old == "evolving_rhythmic"
    )
    downgraded = transitions[("evolving_rhythmic", "rhythmic")]
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": "completed_target_free_paired_implementation_audit",
        "counts": {
            "events": len(by_id_v1),
            "patients": len({str(row.get("patient_id")) for row in rows_v2}),
            "event_id_mismatch": 0,
            "non_rhythm_observable_fact_mismatch": 0,
            "frequency_range_changed": len(frequency_changed),
            "v1_evolving_events": v1_evolving,
            "v1_evolving_to_v2_rhythmic": downgraded,
            "affected_patients": len(downgraded_patients),
        },
        "fractions": {
            "v1_evolving_downgraded_fraction": (
                None if v1_evolving == 0 else downgraded / v1_evolving
            )
        },
        "rhythm_state_transitions": transition_rows,
        "examples": {
            "frequency_changed_event_ids": frequency_changed[:12],
        },
        "access_receipt": {
            "raw_eeg_loaded_by_audit": False,
            "deepsoz_target_values_loaded": False,
            "tusz_native_target_values_loaded": False,
            "private_data_loaded": False,
            "localization_scores_loaded": False,
            "training_performed": False,
            "model_selection_performed": False,
            "clinical_threshold_selected": False,
        },
        "scientific_boundary": {
            "audit_scope": "producer_implementation_semantics_only",
            "clinical_temporal_evolution_claim_allowed": False,
            "soz_performance_claim_allowed": False,
            "v2_promotion_allowed_from_this_audit": False,
            "reader_validation_still_required": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v1 = json.loads(args.v1.read_text(encoding="utf-8"))
    v2 = json.loads(args.v2.read_text(encoding="utf-8"))
    result = audit(v1, v2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
