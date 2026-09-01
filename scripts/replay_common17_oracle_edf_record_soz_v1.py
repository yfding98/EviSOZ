#!/usr/bin/env python3
"""Replay frozen common17 oracle-event SOZ heads at one-EDF granularity.

This is a read-only developmental diagnostic.  Each EDF prediction uses only
the oracle-onset event representations belonging to that EDF and the fold head
that held the entire patient out.  The DeepSOZ SOZ target is patient-level; it
is deliberately reused for each EDF only after predictions are frozen.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sota_soz.train_common17_oracle_event_oof_v1 import (  # noqa: E402
    Common17EventSetReasoner,
    induced_common17_neighbors,
)
from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402


SCHEMA = "clinical_eeg_common17_oracle_edf_record_soz_replay_v1"
PREDICTION_SCHEMA = f"{SCHEMA}_prediction"
PENDING = "CONTENT-ADDRESS-PENDING"
DEFAULT_CONFIG = ROOT / "configs/clinical_eeg_common17_oracle_edf_record_replay_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/clinical_eeg_common17_oracle_edf_record_replay_v1r2_20260825"
MODEL_NAMES = ("literal", "verified")
GT_NAMES = ("literal", "verified")
METRIC_NAMES = ("exact_top1", "deepsoz_N2", "deepsoz_N4", "hit_at_3", "hit_at_5", "mrr")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _load_selected(path: Path, keys: Sequence[str]) -> dict[str, torch.Tensor]:
    requested = tuple(str(key) for key in keys)
    if len(set(requested)) != len(requested):
        raise ValueError("safetensor key request contains duplicates")
    with safe_open(str(path.resolve(strict=True)), framework="pt", device="cpu") as source:
        available = set(source.keys())
        missing = sorted(set(requested) - available)
        if missing:
            raise KeyError(f"missing tensors in {path}: {missing}")
        return {key: source.get_tensor(key) for key in requested}


def _state_keys(path: Path, prefix: str, fold_count: int) -> list[str]:
    stem = tuple(f"{prefix}.fold{fold}." for fold in range(fold_count))
    with safe_open(str(path.resolve(strict=True)), framework="pt", device="cpu") as source:
        keys = [key for key in source.keys() if key.startswith(stem)]
    expected_per_fold = {
        "input_norm.weight",
        "input_norm.bias",
        "projection.weight",
        "projection.bias",
        "phase_logits",
        "fusion_norm.weight",
        "fusion_norm.bias",
        "event_attention.weight",
        "event_attention.bias",
        "channel_scorer.weight",
        "channel_scorer.bias",
        "prior_logits",
    }
    for fold in range(fold_count):
        fold_prefix = f"{prefix}.fold{fold}."
        suffixes = {key[len(fold_prefix) :] for key in keys if key.startswith(fold_prefix)}
        if suffixes != expected_per_fold:
            raise RuntimeError(
                f"checkpoint state schema changed for {prefix} fold {fold}: "
                f"missing={sorted(expected_per_fold - suffixes)}, "
                f"extra={sorted(suffixes - expected_per_fold)}"
            )
    return sorted(keys)


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = PENDING
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def verify_content_address(value: Mapping[str, Any]) -> bool:
    observed = value.get("receipt_sha256")
    if not isinstance(observed, str) or observed == PENDING:
        return False
    candidate = deepcopy(dict(value))
    candidate["receipt_sha256"] = PENDING
    return observed == _canonical_sha256(candidate)


def _metric_atoms(
    probability: torch.Tensor,
    targets: torch.Tensor,
    pre_mapping_positive_count: torch.Tensor,
    graph: Sequence[Sequence[int]],
) -> dict[str, torch.Tensor]:
    """Return one scalar metric atom per row under the frozen tie policy."""

    if probability.ndim != 2 or tuple(probability.shape) != tuple(targets.shape):
        raise ValueError("probability and targets must be aligned [R,C]")
    if probability.shape[1] != len(graph):
        raise ValueError("neighbor graph is not aligned to channel axis")
    if tuple(pre_mapping_positive_count.shape) != (len(probability),):
        raise ValueError("pre-mapping positive count must be [R]")
    if not torch.isfinite(probability).all() or not torch.isfinite(targets).all():
        raise ValueError("metric input contains non-finite values")
    if bool((targets.sum(dim=1) < 1).any()):
        raise ValueError("every metric row requires at least one hard-positive target")

    order = torch.argsort(probability, dim=1, descending=True, stable=True)
    ranked = targets.gather(1, order)
    first = ranked.argmax(dim=1)
    top_value = probability.max(dim=1).values
    tied = probability == top_value.unsqueeze(1)
    positive = targets == 1
    exact = (tied & positive).sum(dim=1).float() / tied.sum(dim=1).float()

    relaxed: dict[int, torch.Tensor] = {}
    for gate in (2, 4):
        acceptable = positive.clone()
        for row in range(len(probability)):
            if int(pre_mapping_positive_count[row]) <= gate:
                for index in torch.nonzero(positive[row], as_tuple=False).flatten().tolist():
                    acceptable[row, list(graph[index])] = True
        relaxed[gate] = (tied & acceptable).sum(dim=1).float() / tied.sum(dim=1).float()
    return {
        "exact_top1": exact,
        "deepsoz_N2": relaxed[2],
        "deepsoz_N4": relaxed[4],
        "hit_at_3": (ranked[:, :3].sum(dim=1) > 0).float(),
        "hit_at_5": (ranked[:, :5].sum(dim=1) > 0).float(),
        "mrr": 1.0 / (first.float() + 1.0),
    }


def _summarize_atoms(
    atoms: Mapping[str, torch.Tensor],
    patient_indices: torch.Tensor,
    pre_mapping_positive_count: torch.Tensor,
) -> dict[str, Any]:
    if not atoms or any(len(values) != len(patient_indices) for values in atoms.values()):
        raise ValueError("metric atoms and patient indices are not aligned")
    patients = sorted(set(int(value) for value in patient_indices.tolist()))
    if not patients:
        raise ValueError("cannot summarize an empty record set")
    record_weighted = {}
    patient_macro = {}
    for name in METRIC_NAMES:
        values = atoms[name].float()
        record_weighted[name] = float(values.mean())
        per_patient = []
        for patient in patients:
            rows = patient_indices == patient
            per_patient.append(values[rows].mean())
        patient_macro[name] = float(torch.stack(per_patient).mean())
    record_weighted["accuracy"] = record_weighted["exact_top1"]
    patient_macro["accuracy"] = patient_macro["exact_top1"]
    return {
        "n_records": len(patient_indices),
        "n_patients": len(patients),
        "record_weighted": record_weighted,
        "patient_macro": patient_macro,
        "deepsoz_gate_eligibility": {
            "N2_records": int((pre_mapping_positive_count <= 2).sum()),
            "N4_records": int((pre_mapping_positive_count <= 4).sum()),
        },
    }


def _event_count_stratum(value: int, strata: Sequence[Mapping[str, Any]]) -> str:
    matches = []
    for row in strata:
        minimum = int(row["minimum"])
        maximum = row.get("maximum")
        if value >= minimum and (maximum is None or value <= int(maximum)):
            matches.append(str(row["id"]))
    if len(matches) != 1:
        raise ValueError(f"event count {value} matched {len(matches)} strata")
    return matches[0]


def _load_model(
    *,
    tensors: Mapping[str, torch.Tensor],
    prefix: str,
    fold: int,
    latent_dimension: int,
    feature_dimension: int,
    phase_count: int,
) -> Common17EventSetReasoner:
    state_prefix = f"{prefix}.fold{fold}."
    state = {
        key[len(state_prefix) :]: value.float().contiguous()
        for key, value in tensors.items()
        if key.startswith(state_prefix)
    }
    model = Common17EventSetReasoner(
        input_dim=feature_dimension,
        phase_count=phase_count,
        latent_dim=latent_dimension,
        prior_logits=state["prior_logits"],
    )
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model


@torch.inference_mode()
def _predict_record(model: Common17EventSetReasoner, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4 or len(features) < 1:
        raise ValueError("one EDF must provide at least one [17,phase,feature] event")
    event_owner = torch.zeros(len(features), dtype=torch.long)
    output = model(features.float(), event_owner, patient_count=1)
    return torch.softmax(output.patient_logits[0], dim=0).cpu().contiguous()


@torch.inference_mode()
def _predict_all_patient_events(
    model: Common17EventSetReasoner,
    features: torch.Tensor,
    event_patient: torch.Tensor,
    patient: int,
) -> torch.Tensor:
    rows = torch.nonzero(event_patient == patient, as_tuple=False).flatten()
    return _predict_record(model, features.index_select(0, rows))


def _metrics_for_pair(
    *,
    probability: torch.Tensor,
    targets: torch.Tensor,
    pre_count: torch.Tensor,
    patient_indices: torch.Tensor,
    event_counts: torch.Tensor,
    graph: Sequence[Sequence[int]],
    strata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repeated_targets = targets.index_select(0, patient_indices)
    repeated_pre_count = pre_count.index_select(0, patient_indices)
    atoms = _metric_atoms(probability, repeated_targets, repeated_pre_count, graph)
    result = _summarize_atoms(atoms, patient_indices, repeated_pre_count)
    result["n_events"] = int(event_counts.sum())
    result["event_count_strata"] = []
    for spec in strata:
        stratum_id = str(spec["id"])
        mask = torch.tensor(
            [_event_count_stratum(int(value), strata) == stratum_id for value in event_counts],
            dtype=torch.bool,
        )
        if not bool(mask.any()):
            continue
        subset_atoms = {name: values[mask] for name, values in atoms.items()}
        subset_patients = patient_indices[mask]
        subset_pre_count = repeated_pre_count[mask]
        summary = _summarize_atoms(subset_atoms, subset_patients, subset_pre_count)
        summary.update(
            {
                "id": stratum_id,
                "minimum_events": int(spec["minimum"]),
                "maximum_events": None if spec.get("maximum") is None else int(spec["maximum"]),
                "n_events": int(event_counts[mask].sum()),
            }
        )
        result["event_count_strata"].append(summary)
    return result


def _lineage(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": _file_sha256(path)}


def run(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _read_json(config_path)
    expected = config["expected"]
    channels = tuple(str(value) for value in config["common17_channels"])
    expected_channels = tuple(channel for channel in STANDARD_19 if channel not in {"FZ", "PZ"})
    if channels != expected_channels:
        raise ValueError("config common17 axis is not STANDARD_19 minus FZ/PZ")
    common_indices = torch.tensor([CHANNEL_INDEX[channel] for channel in channels], dtype=torch.long)
    graph = induced_common17_neighbors(common_indices)

    feature_dir = ROOT / str(config["feature_cache"])
    feature_manifest_path = feature_dir / "manifest.json"
    feature_manifest = _read_json(feature_manifest_path)
    if feature_manifest.get("status") != "completed_full_1145_target_blind_phase_cache":
        raise ValueError("feature cache is not the frozen complete target-blind CAR17 cache")
    access = feature_manifest.get("access_receipt", {})
    if access.get("SOZ_targets_loaded") is not False or access.get("FZ_or_PZ_samples_loaded") is not False:
        raise RuntimeError("feature-cache source firewall changed")
    event_roster = feature_manifest.get("scope", {}).get("event_roster")
    if not isinstance(event_roster, list) or len(event_roster) != int(expected["events"]):
        raise ValueError("event roster cardinality changed")
    feature_tensor_path = feature_dir / str(feature_manifest["tensor_file"])
    feature_payload = _load_selected(feature_tensor_path, ("phase_features", "event_patient_index"))
    features = feature_payload["phase_features"].float().contiguous()
    event_patient = feature_payload["event_patient_index"].long().contiguous()
    expected_shape = (
        int(expected["events"]),
        len(channels),
        int(expected["phase_count"]),
        int(expected["feature_dimension"]),
    )
    if tuple(features.shape) != expected_shape or tuple(event_patient.shape) != (expected_shape[0],):
        raise ValueError("feature tensor shape changed")

    patient_id_by_index: dict[int, str] = {}
    record_rows: dict[str, list[int]] = defaultdict(list)
    record_patient: dict[str, int] = {}
    for ordinal, row in enumerate(event_roster):
        if int(row["ordinal"]) != ordinal:
            raise RuntimeError("event roster ordinal drifted")
        patient = int(event_patient[ordinal])
        patient_id = str(row["patient_id"])
        if patient_id_by_index.setdefault(patient, patient_id) != patient_id:
            raise RuntimeError("one event-patient index maps to multiple public patient IDs")
        path = str(row["relative_edf_path"])
        if record_patient.setdefault(path, patient) != patient:
            raise RuntimeError("one EDF maps to multiple patient indices")
        record_rows[path].append(ordinal)
    if sorted(patient_id_by_index) != list(range(int(expected["patients"]))):
        raise ValueError("patient index carrier is not contiguous 0..101")
    if len(record_rows) != int(expected["unique_edfs"]):
        raise ValueError("unique EDF cardinality changed")
    if sum(len(rows) for rows in record_rows.values()) != int(expected["events"]):
        raise RuntimeError("record grouping lost an event")

    fold_count = int(config["fold_count"])
    arm_payloads: dict[str, dict[str, torch.Tensor]] = {}
    arm_tensor_paths: dict[str, Path] = {}
    for name in MODEL_NAMES:
        arm = config["model_arms"][name]
        directory = ROOT / str(arm["source"])
        manifest_path = directory / "manifest.json"
        manifest = _read_json(manifest_path)
        tensor_path = directory / str(manifest.get("tensor_file", "oof_predictions_and_states.safetensors"))
        prefix = str(arm["state_prefix"])
        state_keys = _state_keys(tensor_path, prefix, fold_count)
        fixed_keys = [
            "patient_folds",
            "event_patient_index",
            str(arm["oof_probability_key"]),
            "common17_standard19_indices",
        ]
        payload = _load_selected(tensor_path, fixed_keys + state_keys)
        if not torch.equal(payload["event_patient_index"].long(), event_patient):
            raise RuntimeError(f"{name} checkpoint event order differs from phase cache")
        if not torch.equal(payload["common17_standard19_indices"].long(), common_indices):
            raise RuntimeError(f"{name} checkpoint common17 axis differs")
        arm_payloads[name] = payload
        arm_tensor_paths[name] = tensor_path
    folds = arm_payloads["literal"]["patient_folds"].long()
    if not torch.equal(folds, arm_payloads["verified"]["patient_folds"].long()):
        raise RuntimeError("literal and verified checkpoints use different patient folds")
    if tuple(folds.shape) != (int(expected["patients"]),) or sorted(folds.unique().tolist()) != list(range(fold_count)):
        raise ValueError("frozen five-fold patient assignment changed")

    models: dict[str, dict[int, Common17EventSetReasoner]] = {}
    for name in MODEL_NAMES:
        arm = config["model_arms"][name]
        models[name] = {
            fold: _load_model(
                tensors=arm_payloads[name],
                prefix=str(arm["state_prefix"]),
                fold=fold,
                latent_dimension=int(config["latent_dimension"]),
                feature_dimension=int(expected["feature_dimension"]),
                phase_count=int(expected["phase_count"]),
            )
            for fold in range(fold_count)
        }

    record_paths = sorted(record_rows)
    record_probabilities = {
        name: torch.empty((len(record_paths), len(channels)), dtype=torch.float32)
        for name in MODEL_NAMES
    }
    record_patient_indices = torch.empty(len(record_paths), dtype=torch.long)
    event_counts = torch.empty(len(record_paths), dtype=torch.long)
    prediction_rows = []
    for record_index, path in enumerate(record_paths):
        rows = torch.tensor(record_rows[path], dtype=torch.long)
        patient = record_patient[path]
        fold = int(folds[patient])
        record_patient_indices[record_index] = patient
        event_counts[record_index] = len(rows)
        rankings = {}
        for name in MODEL_NAMES:
            probability = _predict_record(models[name][fold], features.index_select(0, rows))
            record_probabilities[name][record_index] = probability
            order = torch.argsort(probability, descending=True, stable=True)
            rankings[name] = [
                {"rank": rank + 1, "channel": channels[index], "normalized_support": float(probability[index])}
                for rank, index in enumerate(order.tolist())
            ]
        event_metadata = [event_roster[index] for index in rows.tolist()]
        row = {
            "schema_version": PREDICTION_SCHEMA,
            "record_ordinal": record_index,
            "relative_edf_path": path,
            "patient_id": patient_id_by_index[patient],
            "patient_index": patient,
            "held_patient_fold": fold,
            "event_count": len(rows),
            "oracle_event_ids": [str(value["event_id"]) for value in event_metadata],
            "oracle_event_anchor_seconds": [float(value["global_t0_sec"]) for value in event_metadata],
            "models": rankings,
            "target_values_present": False,
            "prediction_side_FZ_PZ_to_CZ_mapping": False,
            "raw_edf_loaded": False,
        }
        row["prediction_sha256"] = _canonical_sha256(row)
        prediction_rows.append(row)

    # State reloading is accepted only if it reconstructs the originally
    # published all-event patient OOF probabilities for both model arms.
    replay_integrity = {}
    for name in MODEL_NAMES:
        replay = torch.empty((int(expected["patients"]), len(channels)), dtype=torch.float32)
        for patient in range(int(expected["patients"])):
            fold = int(folds[patient])
            replay[patient] = _predict_all_patient_events(
                models[name][fold], features, event_patient, patient
            )
        frozen = arm_payloads[name][str(config["model_arms"][name]["oof_probability_key"])].float()
        maximum_error = float((replay - frozen).abs().max())
        if maximum_error > 1e-6:
            raise RuntimeError(f"{name} checkpoint replay did not reproduce frozen OOF: {maximum_error}")
        replay_integrity[name] = {
            "all_patient_events_reproduced_frozen_oof": True,
            "maximum_absolute_probability_error": maximum_error,
        }

    # Evaluation join: target values are requested only after every record
    # prediction and the target-free checkpoint replay audit are complete.
    gt = {}
    for name in GT_NAMES:
        payload = _load_selected(
            arm_tensor_paths[name],
            ("targets", "target_mask", "pre_mapping_positive_count"),
        )
        targets = payload["targets"].float().contiguous()
        mask = payload["target_mask"].bool().contiguous()
        pre_count = payload["pre_mapping_positive_count"].long().contiguous()
        if tuple(targets.shape) != (int(expected["patients"]), len(channels)) or not bool(mask.all()):
            raise ValueError(f"{name} target carrier is not fully observed common17")
        if not torch.isfinite(targets).all() or bool((targets.sum(dim=1) < 1).any()):
            raise ValueError(f"{name} target carrier is invalid")
        gt[name] = {"targets": targets, "pre_count": pre_count}

    metric_matrix = {}
    for model_name in MODEL_NAMES:
        for gt_name in GT_NAMES:
            pair = f"model_{model_name}__gt_{gt_name}"
            metric_matrix[pair] = _metrics_for_pair(
                probability=record_probabilities[model_name],
                targets=gt[gt_name]["targets"],
                pre_count=gt[gt_name]["pre_count"],
                patient_indices=record_patient_indices,
                event_counts=event_counts,
                graph=graph,
                strata=config["event_count_strata"],
            )

    split_records: dict[str, int] = defaultdict(int)
    split_events: dict[str, int] = defaultdict(int)
    for path, rows in record_rows.items():
        split = path.split("/", 1)[0]
        split_records[split] += 1
        split_events[split] += len(rows)
    lineage = {
        "config": _lineage(config_path),
        "feature_manifest": _lineage(feature_manifest_path),
        "feature_tensor": _lineage(feature_tensor_path),
        "script": _lineage(Path(__file__)),
    }
    for name in MODEL_NAMES:
        directory = ROOT / str(config["model_arms"][name]["source"])
        lineage[f"{name}_manifest"] = _lineage(directory / "manifest.json")
        lineage[f"{name}_checkpoint_tensor"] = _lineage(arm_tensor_paths[name])

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "completed_read_only_oracle_onset_EDF_level_developmental_diagnostic",
        "analysis_role": "oracle_onset_record_subset_replay_not_detector_to_SOZ_end_to_end",
        "cohort": {
            "patients": len(patient_id_by_index),
            "unique_edfs": len(record_paths),
            "oracle_events": int(event_counts.sum()),
            "patient_held_out_folds": fold_count,
            "records_by_historical_official_directory": dict(sorted(split_records.items())),
            "events_by_historical_official_directory": dict(sorted(split_events.items())),
        },
        "prediction_contract": {
            "unit": "one EDF",
            "event_input": "only frozen strict-CAR17 oracle-onset representations belonging to that EDF",
            "model": "the fold head that held the EDF's patient out",
            "all_events_in_each_EDF_used": True,
            "silent_event_selection": False,
            "channel_output_semantics": "normalized ranking support, not calibrated clinical probability",
            "common17_channels": list(channels),
            "FZ_or_PZ_prediction_axis_present": False,
            "prediction_side_midline_remapping": False,
        },
        "evaluation_contract": {
            "GT_is_patient_level_and_reused_for_each_patient_EDF": True,
            "record_level_rows_are_not_independent_SOZ_labels": True,
            "primary_scientific_SOZ_denominator_remains_102_patients": True,
            "record_weighted_definition": "unweighted mean over 455 EDF rows",
            "patient_macro_definition": "mean EDF atoms within patient, then unweighted mean over 102 patients",
            "deepsoz_neighbor_graph": "STANDARD19 one-hop induced on common17 with deleted FZ/PZ nodes and incident edges",
            "N2_N4_are_neighbor_tolerant_not_Top2_or_Top4": True,
            "N2_N4_gate_uses_GT_specific_mapping_before_positive_count": True,
            "literal_GT": "raw duplicate PZ literal OR plus FZ mapped to CZ, then FZ/PZ deleted",
            "verified_GT": "duplicate-PZ conflict masked; verified FZ mapped to CZ, then FZ/PZ deleted",
        },
        "metric_matrix": metric_matrix,
        "checkpoint_replay_integrity": replay_integrity,
        "source_firewall": {
            "training_or_tuning_performed": False,
            "raw_EDF_loaded": False,
            "EDF_annotation_loaded": False,
            "Excel_or_doctor_text_loaded": False,
            "clinical_metadata_loaded": False,
            "continuous_detector_prediction_loaded": False,
            "continuous_detector_TERM_or_source_eval_reference_loaded": False,
            "SOZ_targets_loaded_only_after_record_predictions_materialized_in_memory": True,
            "existing_frozen_oracle_cache_contains_historical_train_dev_eval_paths": True,
            "new_raw_or_continuous_source_eval_endpoint_opened": False,
        },
        "claim_boundary": {
            "developmental_diagnostic_only": True,
            "oracle_onset_conditional": True,
            "not_end_to_end_long_recording_performance": True,
            "not_independent_455_record_GT": True,
            "not_cortical_SOZ_EZ_or_surgical_target": True,
            "endpoint": "scalp-visible ictal-onset channel ranking",
            "clinical_deployment_allowed": False,
        },
        "lineage": lineage,
        "prediction_file": "record_predictions.jsonl",
        "prediction_count": len(prediction_rows),
    }
    return receipt, prediction_rows


def publish(output: Path, receipt: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Path:
    destination = output.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    completed = False
    try:
        prediction_path = staging / str(receipt["prediction_file"])
        with prediction_path.open("wb") as handle:
            for row in rows:
                handle.write(_canonical_bytes(row) + b"\n")
        final = deepcopy(dict(receipt))
        final["prediction_file_sha256"] = _file_sha256(prediction_path)
        final = _content_address(final)
        if not verify_content_address(final):
            raise RuntimeError("receipt content-address verification failed before publish")
        (staging / "receipt.json").write_text(
            json.dumps(final, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
        completed = True
    finally:
        if not completed and staging.exists():
            shutil.rmtree(staging)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    receipt, rows = run(args.config.resolve(strict=True))
    output = publish(args.output, receipt, rows)
    metrics = receipt["metric_matrix"]["model_literal__gt_literal"]
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "records": metrics["n_records"],
                "record_exact_top1": metrics["record_weighted"]["exact_top1"],
                "record_N4": metrics["record_weighted"]["deepsoz_N4"],
                "patient_macro_exact_top1": metrics["patient_macro"]["exact_top1"],
                "patient_macro_N4": metrics["patient_macro"]["deepsoz_N4"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
