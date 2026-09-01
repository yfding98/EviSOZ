#!/usr/bin/env python3
"""Evaluate frozen private oracle-onset common17 predictions.

This evaluator is deliberately prediction-first.  It validates and hashes the
oracle-onset and EEG-only heuristic prediction artifacts, freezes their common
record/patient masks, and only then opens the post-freeze physician-label
release.  Oracle timing is a capability/sensitivity condition; it is not an
EEG-only detector result and must never be relabelled as end-to-end accuracy.

One invocation evaluates one frozen oracle scope (``strict_primary`` or
``all_exact_supported``).  Multiple events are pooled inside their recording
and patient by the frozen predictor.  Event-level accuracy is prohibited
because the physician reference is not independently bound to each detected or
oracle-timed event.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sota_soz.train_common17_oracle_event_oof_v1 import (  # noqa: E402
    induced_common17_neighbors,
)
from scripts.evaluate_private_common17_literal_soz_v1 import (  # noqa: E402
    METRICS,
    SOFT_METRICS,
    _map_target,
    _metric_atoms,
    _soft_atoms,
    _summary,
)
from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402


SCHEMA = "private_common17_literal_oracle_onset_postfreeze_evaluation_v1"
ORACLE_PREDICTION_SCHEMA = (
    "private_common17_literal_oracle_onset_public_fold_ensemble_v1"
)
HEURISTIC_PREDICTION_SCHEMA = "private_common17_literal_public_fold_ensemble_v1"
EXPECTED_RECORDS = 141
EXPECTED_PATIENTS = 45
EXPECTED_FOLDS = 5
COMMON17 = tuple(channel for channel in STANDARD_19 if channel not in {"FZ", "PZ"})
ORACLE_SCOPES = {"strict_primary", "all_exact_supported"}

DEFAULT_HEURISTIC = ROOT / "outputs/private_common17_literal_zero_shot_v1_20260825"
DEFAULT_LABELS = (
    ROOT / "outputs/private_clinical_eeg_doctor_labels_postfreeze_v2_3_20260820.json"
)

_TENSOR_KEYS = {
    "phase_features",
    "event_position_ids",
    "event_record_index",
    "event_patient_index",
    "event_fold_probability",
    "event_probability",
    "record_fold_probability",
    "record_probability",
    "record_prediction_mask",
    "record_event_count",
    "patient_fold_probability",
    "patient_probability",
    "patient_prediction_mask",
    "patient_event_count",
}


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input must be a regular non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _as_nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be a non-empty trimmed string")
    return value


def _shape(source: Any, key: str) -> tuple[int, ...]:
    return tuple(int(value) for value in source.get_slice(key).get_shape())


def _ranking(probability: torch.Tensor) -> list[str]:
    return [
        COMMON17[index]
        for index in torch.argsort(probability, descending=True, stable=True).tolist()
    ]


def _validate_serialized_ranking(
    raw: object,
    probability: torch.Tensor,
    *,
    available: bool,
    context: str,
) -> None:
    if not isinstance(raw, list):
        raise TypeError(f"{context} ranking must be a list")
    if not available:
        if raw:
            raise ValueError(f"{context} unavailable prediction has a ranking")
        return
    expected = _ranking(probability)
    if len(raw) != len(COMMON17):
        raise ValueError(f"{context} ranking must contain all common17 electrodes")
    # Frozen predictors serialize rank/electrode/score objects.  Accepting a
    # plain electrode list as well keeps this evaluator compatible with a
    # target-blind projection, while both representations are checked against
    # the tensor rather than trusted.
    if all(isinstance(item, str) for item in raw):
        if raw != expected:
            raise ValueError(f"{context} ranking disagrees with tensor")
        return
    for rank, (item, electrode) in enumerate(zip(raw, expected), start=1):
        if not isinstance(item, Mapping):
            raise TypeError(f"{context} structured ranking row must be an object")
        if int(item.get("rank", -1)) != rank or item.get("electrode") != electrode:
            raise ValueError(f"{context} structured ranking order disagrees with tensor")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"{context} structured ranking score must be numeric")
        index = COMMON17.index(electrode)
        if not math.isclose(
            float(score), float(probability[index]), rel_tol=1e-7, abs_tol=1e-8
        ):
            raise ValueError(f"{context} structured ranking score disagrees with tensor")


def _validate_probability_rows(
    probability: torch.Tensor,
    mask: torch.Tensor,
    *,
    context: str,
) -> None:
    if not probability.is_floating_point() or mask.dtype != torch.bool:
        raise TypeError(f"{context} probability/mask dtypes changed")
    if not torch.isfinite(probability).all():
        raise ValueError(f"{context} probability contains non-finite values")
    if bool(((probability < -1e-7) | (probability > 1.0 + 1e-7)).any()):
        raise ValueError(f"{context} probability escaped [0,1]")
    if bool(mask.any()):
        sums = probability[mask].sum(dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=2e-5, rtol=2e-5):
            raise ValueError(f"{context} predicted rows are not probability simplices")


def _validate_prediction_directory(
    root: Path,
    *,
    expected_schema: str,
    oracle: bool,
) -> dict[str, Any]:
    """Validate one prediction artifact without opening any reference labels."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"prediction root must be a regular directory: {root}")
    manifest_path = root / "manifest.json"
    receipt_path = root / "receipt.json"
    manifest = _read_json(manifest_path)
    receipt = _read_json(receipt_path)
    if manifest.get("schema_version") != expected_schema:
        raise ValueError(f"wrong prediction schema in {root}")
    status = _as_nonempty_string(manifest.get("status"), "prediction status")
    if not status.startswith("completed_"):
        raise ValueError("prediction artifact is not complete")
    if tuple(manifest.get("common17_channels", ())) != COMMON17:
        raise ValueError("prediction common17 axis drifted")

    if oracle:
        scope = manifest.get("oracle_scope")
        if scope not in ORACLE_SCOPES:
            raise ValueError("oracle_scope must be strict_primary or all_exact_supported")
        anchor_contract = manifest.get("anchor_contract")
        if (
            not isinstance(anchor_contract, Mapping)
            or anchor_contract.get("reference_timing_oracle") is not True
        ):
            raise ValueError("oracle prediction lacks an explicit oracle-onset contract")
    else:
        scope = "heuristic_EEG_only"
        anchor_contract = manifest.get("anchor_contract")
        if not isinstance(anchor_contract, Mapping) or anchor_contract.get("oracle_onset") is not False:
            raise ValueError("heuristic comparator is not explicitly non-oracle")

    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping):
        raise TypeError("prediction access_receipt must be an object")
    doctor_firewall_keys = (
        ("doctor_SOZ_targets_loaded",) if oracle else ("doctor_label_release_loaded",)
    )
    for key in (
        *doctor_firewall_keys,
        "private_target_values_loaded",
        "private_training_or_parameter_fitting_performed",
    ):
        if access.get(key) is not False:
            raise RuntimeError(f"prediction firewall failed: {key}")

    records = manifest.get("records")
    patients = manifest.get("patients")
    events = manifest.get("events")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORDS:
        raise ValueError("prediction record roster must contain 141 rows")
    if not isinstance(patients, list) or len(patients) != EXPECTED_PATIENTS:
        raise ValueError("prediction patient roster must contain 45 rows")
    if not isinstance(events, list):
        raise TypeError("prediction event roster must be a list")

    tensor_name = _as_nonempty_string(manifest.get("tensor_file"), "tensor_file")
    if Path(tensor_name).name != tensor_name or tensor_name != "predictions_and_phase.safetensors":
        raise ValueError("unexpected prediction tensor filename")
    tensor_path = root / tensor_name
    if tensor_path.is_symlink() or not tensor_path.is_file():
        raise ValueError("prediction tensor must be a regular non-symlink file")
    hashes = {
        "manifest_sha256": _sha256(manifest_path),
        "receipt_sha256": _sha256(receipt_path),
        "tensor_sha256": _sha256(tensor_path),
    }
    if hashes["manifest_sha256"] != receipt.get("manifest_sha256"):
        raise RuntimeError("prediction receipt manifest hash mismatch")
    if hashes["tensor_sha256"] != manifest.get("tensor_sha256"):
        raise RuntimeError("prediction manifest tensor hash mismatch")
    if hashes["tensor_sha256"] != receipt.get("tensor_sha256"):
        raise RuntimeError("prediction receipt tensor hash mismatch")
    if manifest.get("prediction_roster_sha256") != receipt.get("prediction_roster_sha256"):
        raise RuntimeError("prediction roster hash differs between manifest and receipt")

    record_ids: list[str] = []
    record_patients: list[str] = []
    record_hashes: list[str] = []
    record_eligible: list[bool] = []
    record_anchor_count: list[int] = []
    record_success_count: list[int] = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise TypeError("prediction record row must be an object")
        if int(row.get("record_index", -1)) != index:
            raise ValueError("prediction record_index order drifted")
        record_ids.append(_as_nonempty_string(row.get("recording_id"), "recording_id"))
        record_patients.append(
            _as_nonempty_string(row.get("patient_pseudonym"), "patient_pseudonym")
        )
        record_hashes.append(
            _as_nonempty_string(row.get("source_signal_sha256"), "source_signal_sha256")
        )
        if oracle:
            if not isinstance(row.get("oracle_anchor_eligible"), bool):
                raise TypeError("oracle record must expose oracle_anchor_eligible")
            eligible = bool(row["oracle_anchor_eligible"])
            anchor_count = int(row.get("oracle_anchor_event_count", -1))
            success_count = int(row.get("successful_event_count", -1))
            if anchor_count < 0 or success_count < 0 or success_count > anchor_count:
                raise ValueError("oracle record event counts are invalid")
            if eligible != (anchor_count > 0):
                raise ValueError("oracle record eligibility/count disagree")
        else:
            anchor_count = int(row.get("event_count", -1))
            success_count = anchor_count
            if anchor_count < 0:
                raise ValueError("heuristic record event_count is invalid")
            eligible = anchor_count > 0
        record_eligible.append(eligible)
        record_anchor_count.append(anchor_count)
        record_success_count.append(success_count)
    if len(set(record_ids)) != EXPECTED_RECORDS:
        raise ValueError("prediction record IDs must be unique")

    patient_ids: list[str] = []
    patient_eligible: list[bool] = []
    patient_anchor_count: list[int] = []
    patient_success_count: list[int] = []
    for index, row in enumerate(patients):
        if not isinstance(row, Mapping):
            raise TypeError("prediction patient row must be an object")
        if int(row.get("patient_index", -1)) != index:
            raise ValueError("prediction patient_index order drifted")
        patient_ids.append(
            _as_nonempty_string(row.get("patient_pseudonym"), "patient_pseudonym")
        )
        if oracle:
            if not isinstance(row.get("oracle_anchor_eligible"), bool):
                raise TypeError("oracle patient must expose oracle_anchor_eligible")
            eligible = bool(row["oracle_anchor_eligible"])
            anchor_count = int(row.get("oracle_anchor_event_count", -1))
            success_count = int(row.get("successful_event_count", -1))
            if anchor_count < 0 or success_count < 0 or success_count > anchor_count:
                raise ValueError("oracle patient event counts are invalid")
            if eligible != (anchor_count > 0):
                raise ValueError("oracle patient eligibility/count disagree")
        else:
            anchor_count = int(row.get("event_count", -1))
            success_count = anchor_count
            if anchor_count < 0:
                raise ValueError("heuristic patient event_count is invalid")
            eligible = anchor_count > 0
        patient_eligible.append(eligible)
        patient_anchor_count.append(anchor_count)
        patient_success_count.append(success_count)
    if len(set(patient_ids)) != EXPECTED_PATIENTS:
        raise ValueError("prediction patient IDs must be unique")
    if set(record_patients) != set(patient_ids):
        raise ValueError("prediction record/patient rosters disagree")

    if oracle:
        successful_events = [
            event
            for event in events
            if isinstance(event, Mapping)
            and event.get("technical_status") == "completed_actual_reader_and_encoder"
        ]
        if len(successful_events) != int(sum(record_success_count)):
            raise ValueError("oracle successful event roster/count disagree")
    else:
        successful_events = events
    event_count = len(successful_events)
    selected: dict[str, torch.Tensor] = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as source:
        available = set(source.keys())
        missing = sorted(_TENSOR_KEYS - available)
        if missing:
            raise KeyError(f"prediction tensor lacks required keys: {missing}")
        expected_shapes = {
            "phase_features": (event_count, 17, 5, 200),
            "event_position_ids": (event_count, 17),
            "event_record_index": (event_count,),
            "event_patient_index": (event_count,),
            "event_fold_probability": (event_count, EXPECTED_FOLDS, 17),
            "event_probability": (event_count, 17),
            "record_fold_probability": (EXPECTED_RECORDS, EXPECTED_FOLDS, 17),
            "record_probability": (EXPECTED_RECORDS, 17),
            "record_prediction_mask": (EXPECTED_RECORDS,),
            "record_event_count": (EXPECTED_RECORDS,),
            "patient_fold_probability": (EXPECTED_PATIENTS, EXPECTED_FOLDS, 17),
            "patient_probability": (EXPECTED_PATIENTS, 17),
            "patient_prediction_mask": (EXPECTED_PATIENTS,),
            "patient_event_count": (EXPECTED_PATIENTS,),
        }
        for key, shape in expected_shapes.items():
            if _shape(source, key) != shape:
                raise ValueError(f"prediction tensor shape changed for {key}")
        for key in (
            "event_record_index",
            "event_patient_index",
            "event_fold_probability",
            "event_probability",
            "record_fold_probability",
            "record_probability",
            "record_prediction_mask",
            "record_event_count",
            "patient_fold_probability",
            "patient_probability",
            "patient_prediction_mask",
            "patient_event_count",
        ):
            selected[key] = source.get_tensor(key)

    event_record = selected["event_record_index"].long()
    event_patient = selected["event_patient_index"].long()
    record_mask = selected["record_prediction_mask"]
    patient_mask = selected["patient_prediction_mask"]
    if record_mask.dtype != torch.bool or patient_mask.dtype != torch.bool:
        raise TypeError("prediction masks must be boolean")
    if event_count:
        if int(event_record.min()) < 0 or int(event_record.max()) >= EXPECTED_RECORDS:
            raise ValueError("event_record_index escaped roster")
        if int(event_patient.min()) < 0 or int(event_patient.max()) >= EXPECTED_PATIENTS:
            raise ValueError("event_patient_index escaped roster")
    expected_record_counts = torch.bincount(event_record, minlength=EXPECTED_RECORDS)
    expected_patient_counts = torch.bincount(event_patient, minlength=EXPECTED_PATIENTS)
    if not torch.equal(selected["record_event_count"].long(), expected_record_counts):
        raise ValueError("record_event_count disagrees with event indices")
    if not torch.equal(selected["patient_event_count"].long(), expected_patient_counts):
        raise ValueError("patient_event_count disagrees with event indices")
    if record_success_count != expected_record_counts.tolist():
        raise ValueError("manifest record successful-event counts disagree with tensor")
    if patient_success_count != expected_patient_counts.tolist():
        raise ValueError("manifest patient successful-event counts disagree with tensor")
    if not torch.equal(record_mask, expected_record_counts > 0):
        raise ValueError("record prediction mask disagrees with successful events")
    if not torch.equal(patient_mask, expected_patient_counts > 0):
        raise ValueError("patient prediction mask disagrees with successful events")

    event_probability = selected["event_probability"].float()
    record_probability = selected["record_probability"].float()
    patient_probability = selected["patient_probability"].float()
    _validate_probability_rows(
        event_probability,
        torch.ones(event_count, dtype=torch.bool),
        context="event",
    )
    _validate_probability_rows(record_probability, record_mask, context="record")
    _validate_probability_rows(patient_probability, patient_mask, context="patient")
    for level, aggregate, folds in (
        ("event", event_probability, selected["event_fold_probability"].float()),
        ("record", record_probability, selected["record_fold_probability"].float()),
        ("patient", patient_probability, selected["patient_fold_probability"].float()),
    ):
        if not torch.isfinite(folds).all():
            raise ValueError(f"{level} fold probabilities contain non-finite values")
        if not torch.allclose(aggregate, folds.mean(dim=1), atol=2e-6, rtol=2e-6):
            raise ValueError(f"{level} probability is not the five-fold mean")

    patient_axis = {patient: index for index, patient in enumerate(patient_ids)}
    successful_index = 0
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("prediction event row must be an object")
        record_index = int(event.get("record_index", -1))
        if record_index < 0 or record_index >= EXPECTED_RECORDS:
            raise ValueError("manifest event record index escaped roster")
        if str(event.get("recording_id")) != record_ids[record_index]:
            raise ValueError("manifest event recording identity mismatch")
        patient = record_patients[record_index]
        if str(event.get("patient_pseudonym")) != patient:
            raise ValueError("manifest event patient identity mismatch")
        successful = (
            not oracle
            or event.get("technical_status") == "completed_actual_reader_and_encoder"
        )
        if successful:
            if int(event.get("event_index", -1)) != successful_index:
                raise ValueError("successful event_index order drifted")
            if record_index != int(event_record[successful_index]):
                raise ValueError("manifest/tensor event record index mismatch")
            if int(event_patient[successful_index]) != patient_axis[patient]:
                raise ValueError("manifest/tensor event patient index mismatch")
            successful_index += 1
        elif event.get("event_index") is not None:
            raise ValueError("technically excluded oracle event must not have tensor index")
    if successful_index != event_count:
        raise RuntimeError("successful event validation did not exhaust tensor rows")

    for index, row in enumerate(records):
        _validate_serialized_ranking(
            row.get("ranking"),
            record_probability[index],
            available=bool(record_mask[index]),
            context="record",
        )
    for index, row in enumerate(patients):
        _validate_serialized_ranking(
            row.get("ranking"),
            patient_probability[index],
            available=bool(patient_mask[index]),
            context="patient",
        )

    return {
        "scope": scope,
        "manifest": manifest,
        "hashes": hashes,
        "record_ids": record_ids,
        "record_patients": record_patients,
        "record_hashes": record_hashes,
        "patient_ids": patient_ids,
        "record_eligible": torch.tensor(record_eligible, dtype=torch.bool),
        "patient_eligible": torch.tensor(patient_eligible, dtype=torch.bool),
        "record_mask": record_mask.clone(),
        "patient_mask": patient_mask.clone(),
        "record_probability": record_probability.clone(),
        "patient_probability": patient_probability.clone(),
        "event_count": event_count,
        "anchor_event_count": int(sum(record_anchor_count)),
        "successful_event_count": int(sum(record_success_count)),
    }


def _assert_prediction_alignment(
    oracle: Mapping[str, Any], heuristic: Mapping[str, Any]
) -> None:
    for key in ("record_ids", "record_patients", "record_hashes", "patient_ids"):
        if oracle[key] != heuristic[key]:
            raise ValueError(f"oracle/heuristic prediction rosters differ: {key}")
    oracle_model = oracle["manifest"].get("lineage", {}).get("public_model_tensor", {})
    heuristic_model = heuristic["manifest"].get("lineage", {}).get(
        "public_model_tensor", {}
    )
    if oracle_model.get("sha256") != heuristic_model.get("sha256"):
        raise ValueError("oracle/heuristic predictions do not use the same public model")
    if oracle["manifest"].get("execution_arm") != heuristic["manifest"].get(
        "execution_arm"
    ):
        raise ValueError("oracle/heuristic execution arms differ")


def _reference_rows(
    labels: Mapping[str, Any], oracle: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_records = labels.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != EXPECTED_RECORDS:
        raise ValueError("doctor-label release must contain 141 records")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in raw_records:
        if not isinstance(row, Mapping):
            raise TypeError("doctor-label record must be an object")
        recording_id = _as_nonempty_string(row.get("recording_id"), "recording_id")
        if recording_id in by_id:
            raise ValueError("duplicate doctor-label recording_id")
        by_id[recording_id] = row
    if set(by_id) != set(oracle["record_ids"]):
        raise ValueError("doctor-label and prediction record rosters differ")

    records: list[dict[str, Any]] = []
    patient_hard_standard: dict[str, set[str]] = defaultdict(set)
    patient_hard: dict[str, set[str]] = defaultdict(set)
    patient_spread_standard: dict[str, set[str]] = defaultdict(set)
    patient_spread: dict[str, set[str]] = defaultdict(set)
    for index, recording_id in enumerate(oracle["record_ids"]):
        source = by_id[recording_id]
        if str(source.get("patient_pseudonym")) != oracle["record_patients"][index]:
            raise ValueError("doctor-label patient identity differs from prediction")
        doctor_labels = source.get("doctor_labels")
        if not isinstance(doctor_labels, list):
            raise TypeError("doctor_labels must be a list")
        eligible = [
            item
            for item in doctor_labels
            if isinstance(item, Mapping) and item.get("evaluation_eligible") is True
        ]
        hard: set[str] = set()
        hard_standard: set[str] = set()
        hard_outside: set[str] = set()
        spread: set[str] = set()
        spread_standard: set[str] = set()
        spread_outside: set[str] = set()
        for item in eligible:
            channel = item.get("physician_channel_reference")
            if not isinstance(channel, Mapping) or channel.get("status") != "available":
                continue
            mapped, standard, outside = _map_target(
                channel.get("significant_electrodes", [])
            )
            hard |= mapped
            hard_standard |= standard
            hard_outside |= outside
            mapped, standard, outside = _map_target(channel.get("spread_electrodes", []))
            spread |= mapped
            spread_standard |= standard
            spread_outside |= outside
        soft = spread - hard
        patient = oracle["record_patients"][index]
        patient_hard_standard[patient] |= hard_standard
        patient_hard[patient] |= hard
        patient_spread_standard[patient] |= spread_standard
        patient_spread[patient] |= spread
        records.append(
            {
                "record_index": index,
                "recording_id": recording_id,
                "patient_pseudonym": patient,
                "eligible_doctor_label_count": len(eligible),
                "hard": sorted(hard),
                "soft": sorted(soft),
                "pre_mapping_hard_count": len(hard_standard),
                "hard_outside_standard19": sorted(hard_outside),
                "spread_outside_standard19": sorted(spread_outside),
            }
        )

    patient_axis = {value: index for index, value in enumerate(oracle["patient_ids"])}
    patients: list[dict[str, Any]] = []
    for patient in oracle["patient_ids"]:
        hard = patient_hard.get(patient, set())
        spread = patient_spread.get(patient, set())
        patients.append(
            {
                "patient_index": patient_axis[patient],
                "patient_pseudonym": patient,
                "hard": sorted(hard),
                "soft": sorted(spread - hard),
                "pre_mapping_hard_count": len(patient_hard_standard.get(patient, set())),
            }
        )
    return records, patients


def _hard_atoms(
    probability: torch.Tensor,
    available: bool,
    reference: Mapping[str, Any],
    graph: Sequence[Sequence[int]],
) -> dict[str, float]:
    if not available:
        return {metric: 0.0 for metric in METRICS}
    atoms, _ = _metric_atoms(
        probability,
        set(reference["hard"]),
        set(reference["soft"]),
        int(reference["pre_mapping_hard_count"]),
        graph,
    )
    return atoms


def _paired_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if not rows:
        return {"denominator": 0, "patient_denominator": 0}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["patient_pseudonym"])].append(row)
    patients = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(patients), size=(replicates, len(patients)))
    output: dict[str, Any] = {
        "denominator": len(rows),
        "patient_denominator": len(patients),
        "metrics": {},
        "bootstrap": {
            "method": "paired_patient_cluster_nonparametric_bootstrap_percentile_95",
            "seed": seed,
            "replicates": replicates,
            "patient_clusters": len(patients),
        },
    }
    patient_counts = np.asarray([len(grouped[patient]) for patient in patients])
    for metric in METRICS:
        oracle_values = np.asarray(
            [float(row["oracle_atoms"][metric]) for row in rows], dtype=np.float64
        )
        heuristic_values = np.asarray(
            [float(row["heuristic_atoms"][metric]) for row in rows], dtype=np.float64
        )
        delta_values = oracle_values - heuristic_values
        patient_delta_sum = np.asarray(
            [
                sum(
                    float(row["oracle_atoms"][metric])
                    - float(row["heuristic_atoms"][metric])
                    for row in grouped[patient]
                )
                for patient in patients
            ],
            dtype=np.float64,
        )
        boot_micro = patient_delta_sum[samples].sum(axis=1) / patient_counts[samples].sum(
            axis=1
        )
        patient_delta_mean = patient_delta_sum / patient_counts
        boot_macro = patient_delta_mean[samples].mean(axis=1)
        payload: dict[str, Any] = {
            "oracle": float(oracle_values.mean()),
            "heuristic_same_subset": float(heuristic_values.mean()),
            "paired_delta_oracle_minus_heuristic": float(delta_values.mean()),
            "paired_delta_patient_macro": float(patient_delta_mean.mean()),
            "paired_delta_record_micro_ci95": [
                float(np.quantile(boot_micro, 0.025)),
                float(np.quantile(boot_micro, 0.975)),
            ],
            "paired_delta_patient_macro_ci95": [
                float(np.quantile(boot_macro, 0.025)),
                float(np.quantile(boot_macro, 0.975)),
            ],
        }
        if metric == "exact_top1" and all(
            value in (0.0, 1.0) for value in (*oracle_values.tolist(), *heuristic_values.tolist())
        ):
            oracle_wins = int(((oracle_values == 1.0) & (heuristic_values == 0.0)).sum())
            heuristic_wins = int(((oracle_values == 0.0) & (heuristic_values == 1.0)).sum())
            payload["discordance"] = {
                "oracle_only_correct": oracle_wins,
                "heuristic_only_correct": heuristic_wins,
                "both_correct": int(((oracle_values == 1.0) & (heuristic_values == 1.0)).sum()),
                "both_incorrect": int(((oracle_values == 0.0) & (heuristic_values == 0.0)).sum()),
                "record_level_mcnemar_p_not_reported_due_patient_clustering": True,
            }
        output["metrics"][metric] = payload
    return output


def _build_metric_rows(
    oracle: Mapping[str, Any],
    heuristic: Mapping[str, Any],
    record_references: Sequence[Mapping[str, Any]],
    patient_references: Sequence[Mapping[str, Any]],
    graph: Sequence[Sequence[int]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    record_rows: list[dict[str, Any]] = []
    paired_records: list[dict[str, Any]] = []
    for reference in record_references:
        if not reference["hard"]:
            continue
        index = int(reference["record_index"])
        oracle_available = bool(oracle["record_mask"][index])
        heuristic_available = bool(heuristic["record_mask"][index])
        oracle_atoms = _hard_atoms(
            oracle["record_probability"][index], oracle_available, reference, graph
        )
        row = {
            **dict(reference),
            "oracle_eligible": bool(oracle["record_eligible"][index]),
            "oracle_prediction_available": oracle_available,
            "heuristic_prediction_available": heuristic_available,
            "oracle_atoms": oracle_atoms,
        }
        record_rows.append(row)
        if oracle_available and heuristic_available:
            paired_records.append(
                {
                    "recording_id": reference["recording_id"],
                    "patient_pseudonym": reference["patient_pseudonym"],
                    "oracle_atoms": oracle_atoms,
                    "heuristic_atoms": _hard_atoms(
                        heuristic["record_probability"][index], True, reference, graph
                    ),
                }
            )

    patient_rows: list[dict[str, Any]] = []
    paired_patients: list[dict[str, Any]] = []
    for reference in patient_references:
        if not reference["hard"]:
            continue
        index = int(reference["patient_index"])
        oracle_available = bool(oracle["patient_mask"][index])
        heuristic_available = bool(heuristic["patient_mask"][index])
        oracle_atoms = _hard_atoms(
            oracle["patient_probability"][index], oracle_available, reference, graph
        )
        row = {
            **dict(reference),
            "oracle_eligible": bool(oracle["patient_eligible"][index]),
            "oracle_prediction_available": oracle_available,
            "heuristic_prediction_available": heuristic_available,
            "oracle_atoms": oracle_atoms,
        }
        patient_rows.append(row)
        if oracle_available and heuristic_available:
            paired_patients.append(
                {
                    "patient_pseudonym": reference["patient_pseudonym"],
                    "oracle_atoms": oracle_atoms,
                    "heuristic_atoms": _hard_atoms(
                        heuristic["patient_probability"][index], True, reference, graph
                    ),
                }
            )
    return record_rows, patient_rows, paired_records, paired_patients


def _endpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    eligible = [row for row in rows if row["oracle_eligible"]]
    conditional = [row for row in eligible if row["oracle_prediction_available"]]
    forced_rows = [
        {"patient_pseudonym": row["patient_pseudonym"], "atoms": row["oracle_atoms"]}
        for row in eligible
    ]
    conditional_rows = [
        {"patient_pseudonym": row["patient_pseudonym"], "atoms": row["oracle_atoms"]}
        for row in conditional
    ]
    return {
        "full_reference_coverage_only": {
            "denominator": len(rows),
            "oracle_eligible_count": len(eligible),
            "oracle_prediction_count": sum(
                bool(row["oracle_prediction_available"]) for row in rows
            ),
            "accuracy_metrics_reported": False,
            "reason": "absence_of_oracle_timing_is_coverage_not_localization_error",
        },
        "oracle_eligible_forced": _summary(
            forced_rows, METRICS, seed=seed, replicates=replicates
        ),
        "conditional_on_oracle_prediction": _summary(
            conditional_rows, METRICS, seed=seed + 1, replicates=replicates
        ),
        "oracle_eligible_missing_prediction_count": len(eligible) - len(conditional),
    }


def _soft_endpoint(
    oracle: Mapping[str, Any],
    record_references: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for reference in record_references:
        if not reference["soft"]:
            continue
        index = int(reference["record_index"])
        eligible = bool(oracle["record_eligible"][index])
        available = bool(oracle["record_mask"][index])
        atoms = (
            _soft_atoms(oracle["record_probability"][index], set(reference["soft"]))
            if available
            else {metric: 0.0 for metric in SOFT_METRICS}
        )
        rows.append(
            {
                "recording_id": reference["recording_id"],
                "patient_pseudonym": reference["patient_pseudonym"],
                "oracle_eligible": eligible,
                "oracle_prediction_available": available,
                "atoms": atoms,
            }
        )
    eligible = [row for row in rows if row["oracle_eligible"]]
    conditional = [row for row in eligible if row["oracle_prediction_available"]]
    return {
        "full_reference_coverage_only": {
            "denominator": len(rows),
            "oracle_eligible_count": len(eligible),
            "oracle_prediction_count": sum(
                bool(row["oracle_prediction_available"]) for row in rows
            ),
            "accuracy_metrics_reported": False,
        },
        "oracle_eligible_forced": _summary(
            eligible, SOFT_METRICS, seed=seed, replicates=replicates
        ),
        "conditional_on_oracle_prediction": _summary(
            conditional, SOFT_METRICS, seed=seed + 1, replicates=replicates
        ),
        "oracle_eligible_missing_prediction_count": len(eligible) - len(conditional),
    }, rows


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Sequence[Mapping[str, Any]]]]:
    # Prediction validation and hashing are intentionally complete before the
    # first stat/resolve/open of args.doctor_label_release.
    oracle_root = args.oracle_predictions.resolve(strict=True)
    heuristic_root = args.heuristic_predictions.resolve(strict=True)
    oracle = _validate_prediction_directory(
        oracle_root,
        expected_schema=ORACLE_PREDICTION_SCHEMA,
        oracle=True,
    )
    heuristic = _validate_prediction_directory(
        heuristic_root,
        expected_schema=HEURISTIC_PREDICTION_SCHEMA,
        oracle=False,
    )
    _assert_prediction_alignment(oracle, heuristic)
    prelabel_snapshot = {
        "oracle_scope": oracle["scope"],
        "record_ids": oracle["record_ids"],
        "patient_ids": oracle["patient_ids"],
        "oracle_record_eligible": oracle["record_eligible"].tolist(),
        "oracle_patient_eligible": oracle["patient_eligible"].tolist(),
        "oracle_record_mask": oracle["record_mask"].tolist(),
        "oracle_patient_mask": oracle["patient_mask"].tolist(),
        "heuristic_record_mask": heuristic["record_mask"].tolist(),
        "heuristic_patient_mask": heuristic["patient_mask"].tolist(),
        "oracle_hashes": oracle["hashes"],
        "heuristic_hashes": heuristic["hashes"],
    }
    prelabel_snapshot_sha256 = _canonical_sha256(prelabel_snapshot)

    # Reference access begins here, after both prediction artifacts and their
    # paired masks have been frozen in prelabel_snapshot.
    label_path = args.doctor_label_release.resolve(strict=True)
    labels = _read_json(label_path)
    record_references, patient_references = _reference_rows(labels, oracle)
    common_indices = torch.tensor(
        [CHANNEL_INDEX[channel] for channel in COMMON17], dtype=torch.long
    )
    graph = induced_common17_neighbors(common_indices)
    record_rows, patient_rows, paired_records, paired_patients = _build_metric_rows(
        oracle,
        heuristic,
        record_references,
        patient_references,
        graph,
    )
    if len(record_rows) != 92 or len(patient_rows) != 34:
        raise RuntimeError("private hard-GT denominator drifted from 92 records/34 patients")
    soft_endpoint, soft_rows = _soft_endpoint(
        oracle,
        record_references,
        seed=args.bootstrap_seed + 20,
        replicates=args.bootstrap_replicates,
    )
    if len(soft_rows) != 97:
        raise RuntimeError("private soft-spread denominator drifted from 97 records")

    record_endpoint = _endpoint(
        record_rows,
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    patient_endpoint = _endpoint(
        patient_rows,
        seed=args.bootstrap_seed + 10,
        replicates=args.bootstrap_replicates,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "completed_prediction_first_oracle_onset_postfreeze_evaluation",
        "oracle_scope": oracle["scope"],
        "evaluation_unit": {
            "primary": "one_unique_long_EEG_recording_after_multi_oracle_event_pooling",
            "secondary": "one_patient_after_pooling_oracle_events_across_records",
            "event_accuracy_reported": False,
            "event_reason": (
                "physician record/patient references are not independent one-to-one "
                "labels for oracle-timed or heuristic events"
            ),
        },
        "prediction_first_gate": {
            "logical_order": [
                "validate_hash_and_freeze_oracle_prediction",
                "validate_hash_and_freeze_heuristic_prediction",
                "verify_exact_roster_and_model_identity",
                "freeze_oracle_eligibility_and_paired_prediction_masks",
                "open_postfreeze_doctor_label_release",
                "score_without_tuning",
            ],
            "both_prediction_artifacts_validated_before_first_reference_path_access": True,
            "prelabel_prediction_snapshot_sha256": prelabel_snapshot_sha256,
            "threshold_window_weight_pooling_or_ranking_changed_after_label_open": False,
            "oracle_prediction_hashes": oracle["hashes"],
            "heuristic_prediction_hashes": heuristic["hashes"],
            "doctor_label_release_sha256": _sha256(label_path),
        },
        "coverage": {
            "full_records": EXPECTED_RECORDS,
            "full_patients": EXPECTED_PATIENTS,
            "oracle_anchor_eligible_records": int(oracle["record_eligible"].sum()),
            "oracle_prediction_records": int(oracle["record_mask"].sum()),
            "oracle_anchor_eligible_patients": int(oracle["patient_eligible"].sum()),
            "oracle_prediction_patients": int(oracle["patient_mask"].sum()),
            "oracle_anchor_events": oracle["anchor_event_count"],
            "oracle_successful_events": oracle["successful_event_count"],
            "hard_GT_records": len(record_rows),
            "hard_GT_patients": len(patient_rows),
            "soft_spread_records": len(soft_rows),
            "paired_hard_records": len(paired_records),
            "paired_hard_record_patient_clusters": len(
                {row["patient_pseudonym"] for row in paired_records}
            ),
            "paired_hard_patients": len(paired_patients),
        },
        "metrics": {
            "record_level_hard_significant": record_endpoint,
            "patient_level_hard_significant_union": patient_endpoint,
            "record_level_soft_spread_separate_endpoint": soft_endpoint,
        },
        "paired_oracle_vs_heuristic_same_oracle_prediction_subset": {
            "record_level_hard": _paired_summary(
                paired_records,
                seed=args.bootstrap_seed + 30,
                replicates=args.bootstrap_replicates,
            ),
            "patient_level_hard": _paired_summary(
                paired_patients,
                seed=args.bootstrap_seed + 40,
                replicates=args.bootstrap_replicates,
            ),
            "interpretation": (
                "paired arm-level effect of frozen oracle timing/event support versus "
                "frozen EEG-only heuristic candidates; not pure onset-timestamp error"
            ),
        },
        "metric_contract": {
            "candidate_space": list(COMMON17),
            "GT_only_FZ_or_PZ_to_CZ": True,
            "prediction_side_score_mapping": False,
            "hard_endpoint": "record union of evaluation-eligible significant electrodes",
            "soft_endpoint": "mapped spread minus mapped hard, scored separately",
            "unlisted_electrodes_are_negative": False,
            "exact_top1_equals_accuracy": True,
            "N2_N4_are_not_top2_or_top4": True,
            "N2_N4_graph": (
                "DeepSOZ_STANDARD19_one_hop_induced_by_deleting_FZ_PZ_nodes_and_edges"
            ),
            "N2_N4_gate_uses_pre_mapping_distinct_hard_count": True,
            "known_soft_spread_removed_from_neighbor_acceptable_set": True,
            "full_92_hard_GT_forced_accuracy_reported": False,
            "full_92_role": "coverage_only",
        },
        "claim_boundary": {
            "oracle_timing_is_EEG_only": False,
            "end_to_end_detector_to_SOZ_claim_permitted": False,
            "oracle_result_is_capability_or_sensitivity_only": True,
            "event_accuracy_permitted": False,
            "doctor_labels_used_for_prediction_or_pair_selection": False,
            "fresh_confirmatory_external_test": False,
        },
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload, {
        "record_rows": record_rows,
        "patient_rows": patient_rows,
        "soft_rows": soft_rows,
        "paired_record_rows": paired_records,
        "paired_patient_rows": paired_patients,
    }


def publish(
    output: Path,
    payload: Mapping[str, Any],
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        result_path = staging / "result.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        row_hashes: dict[str, str] = {}
        for name, values in rows.items():
            path = staging / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in values:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            row_hashes[f"{name}_sha256"] = _sha256(path)
        receipt = {
            "schema_version": f"{SCHEMA}_receipt",
            "status": payload["status"],
            "oracle_scope": payload["oracle_scope"],
            "content_sha256": payload["content_sha256"],
            "result_sha256": _sha256(result_path),
            **row_hashes,
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--oracle-predictions", type=Path, required=True)
    parser.add_argument("--heuristic-predictions", type=Path, default=DEFAULT_HEURISTIC)
    parser.add_argument("--doctor-label-release", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap-replicates must be at least 1000")
    payload, rows = run(args)
    output = publish(args.output, payload, rows)
    record = payload["metrics"]["record_level_hard_significant"]
    paired = payload["paired_oracle_vs_heuristic_same_oracle_prediction_subset"]
    print(
        json.dumps(
            {
                "output": str(output),
                "oracle_scope": payload["oracle_scope"],
                "record_oracle_eligible_forced_exact": record[
                    "oracle_eligible_forced"
                ]["metrics"]["exact_top1"]["value"],
                "record_oracle_conditional_exact": record[
                    "conditional_on_oracle_prediction"
                ]["metrics"]["exact_top1"]["value"],
                "paired_record_exact_delta": paired["record_level_hard"]["metrics"][
                    "exact_top1"
                ]["paired_delta_oracle_minus_heuristic"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
