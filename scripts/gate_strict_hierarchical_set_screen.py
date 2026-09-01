#!/usr/bin/env python3
"""Apply the locked promotion gate to strict pooled-validation predictions only.

This program never discovers or opens historical experiment outputs.  It reads
only the run directories enumerated by ``SCREENING_PLAN.json``.  Before opening
metrics or predictions it rejects any run directory containing a test-named
artifact, then recomputes all gate endpoints from raw ``val_predictions.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from pathlib import PurePosixPath
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_strict_hierarchical_set_screen import (  # noqa: E402
    CONFIG_SPECS,
    ARM_CLASSIFIER_EPOCHS,
    BASE_CLASSIFIER_EPOCHS,
    BASE_TOKENIZER_EPOCHS,
    DEFAULT_PROTOCOL,
    DEFAULT_PREPROCESSED,
    DEFAULT_SCREENING_ROOT,
    EXPECTED_PANEL_EVENT_COUNTS,
    EXPECTED_PANELS,
    EXPECTED_SEEDS,
    EXPECTED_SET_ADDED_CLASSIFIER_PARAMETERS,
    EXPECTED_TRAIN_PATIENTS,
    REPO_ROOT,
    SET_BOTTLENECK,
    SET_RESIDUAL_INIT,
    TKPR_K,
    TKPR_WEIGHT,
    ContractError,
    build_plan,
    gate_policy,
    load_json,
    pooled_validation_patients,
    relative_to_repo,
    require_equal,
    require_no_symlink_chain,
    resolve_under_repo,
    sha256_file,
    sha256_jsonable,
    validate_locked_inputs,
    verify_completed_run,
)


CHANNELS = (
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "F7-F3",
    "F3-FZ",
    "T5-P3",
    "FZ-CZ",
    "P3-PZ",
    "CZ-PZ",
    "PZ-P4",
    "F4-F8",
    "FZ-F4",
    "P4-T6",
)
REGIONS = ("left_frontal", "right_frontal", "left_temporal", "right_temporal", "central_parietal")
REGION_CHANNEL_INDICES = (
    (0, 8, 22, 23),
    (4, 12, 29, 30),
    (1, 2, 3, 16, 17, 24),
    (5, 6, 7, 20, 21, 31),
    (9, 10, 11, 13, 14, 15, 18, 19, 25, 26, 27, 28),
)
CHANNEL_TO_REGION = {
    channel_idx: region_idx
    for region_idx, channel_indices in enumerate(REGION_CHANNEL_INDICES)
    for channel_idx in channel_indices
}
PRIMARY = ("channel_top1", "region_top1")
SECONDARY = ("channel_mrr", "region_mrr", "channel_auprc", "region_auprc")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def locked_index_channel_vector(row: Mapping[str, str], *, sample_id: str) -> tuple[int, ...]:
    raw = str(row.get("soz_bipolar", "")).strip()
    if not raw:
        raise ContractError(f"Locked index has empty soz_bipolar for {sample_id}")
    tokens = [token.strip() for token in raw.split(",")]
    if any(not token for token in tokens):
        raise ContractError(f"Locked index has an empty soz_bipolar token for {sample_id}")
    if len(tokens) != len(set(tokens)):
        raise ContractError(f"Locked index has duplicate soz_bipolar tokens for {sample_id}")
    unknown = sorted(set(tokens) - set(CHANNELS))
    if unknown:
        raise ContractError(f"Locked index has unknown soz_bipolar tokens for {sample_id}: {unknown}")
    selected = set(tokens)
    return tuple(int(channel in selected) for channel in CHANNELS)


def locked_index_region_vector(row: Mapping[str, str], *, sample_id: str) -> tuple[int, ...]:
    raw = str(row.get("soz_region", "")).strip()
    if not raw:
        raise ContractError(f"Locked index has empty soz_region for {sample_id}")
    if any(separator in raw for separator in (",", ";", "|")):
        raise ContractError(f"Locked index must have exactly one soz_region for {sample_id}: {raw!r}")
    if raw not in REGIONS:
        raise ContractError(f"Locked index has unknown soz_region for {sample_id}: {raw!r}")
    return tuple(int(region == raw) for region in REGIONS)


def build_expected_validation_truth(preprocessed_dir: Path) -> dict[str, dict[str, Any]]:
    """Reconstruct the only admissible validation labels from locked NPZs."""

    index_path = preprocessed_dir / "index.csv"
    rows = read_csv_rows(index_path)
    validation_patients = set(pooled_validation_patients())
    root = preprocessed_dir.resolve()
    truth: dict[str, dict[str, Any]] = {}
    for row in rows:
        patient = str(row.get("base_patient_id") or row.get("patient_id") or "").strip()
        if patient not in validation_patients:
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ContractError("Locked validation index contains an empty sample_id")
        if sample_id in truth:
            raise ContractError(f"Locked validation index contains duplicate sample_id: {sample_id}")
        index_channel_labels = locked_index_channel_vector(row, sample_id=sample_id)
        index_region_labels = locked_index_region_vector(row, sample_id=sample_id)
        relative_text = str(row.get("npz_path", "")).strip().replace("\\", "/")
        relative = PurePosixPath(relative_text)
        if not relative_text or relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"Unsafe locked validation NPZ path: {relative_text!r}")
        lexical_path = preprocessed_dir / Path(*relative.parts)
        require_no_symlink_chain(lexical_path, anchor=preprocessed_dir)
        sample_path = lexical_path.resolve()
        try:
            sample_path.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"Locked validation NPZ escapes preprocessed root: {relative_text}") from exc
        if not sample_path.is_file():
            raise ContractError(f"Locked validation NPZ is missing: {sample_path}")
        with np.load(sample_path, allow_pickle=False) as npz:
            try:
                channel_segments = np.asarray(npz["y_segments"], dtype=float)
                region_segments = np.asarray(npz["region_y_segments"], dtype=float)
                channel_alias = np.asarray(npz["y"], dtype=float)
                region_alias = np.asarray(npz["region_y"], dtype=float)
                channel_mask = np.asarray(npz["channel_mask"], dtype=float)
            except KeyError as exc:
                raise ContractError(f"Locked validation NPZ is missing label evidence: {sample_path}") from exc
        if channel_segments.shape != (3, len(CHANNELS)):
            raise ContractError(
                f"Locked validation y_segments shape mismatch for {sample_id}: {channel_segments.shape}"
            )
        if region_segments.shape != (3, len(REGIONS)):
            raise ContractError(
                f"Locked validation region_y_segments shape mismatch for {sample_id}: {region_segments.shape}"
            )
        if channel_mask.shape != (len(CHANNELS),):
            raise ContractError(
                f"Locked validation channel_mask shape mismatch for {sample_id}: {channel_mask.shape}"
            )
        if channel_alias.shape != (len(CHANNELS),):
            raise ContractError(
                f"Locked validation y shape mismatch for {sample_id}: {channel_alias.shape}"
            )
        if region_alias.shape != (len(REGIONS),):
            raise ContractError(
                f"Locked validation region_y shape mismatch for {sample_id}: {region_alias.shape}"
            )
        channel_labels = channel_segments[1]
        region_labels = region_segments[1]
        if not bool(np.all(np.isin(channel_labels, (0.0, 1.0)))):
            raise ContractError(f"Locked validation channel labels are not binary for {sample_id}")
        if not bool(np.all(np.isin(region_labels, (0.0, 1.0)))):
            raise ContractError(f"Locked validation region labels are not binary for {sample_id}")
        if not bool(np.all(np.isin(channel_mask, (0.0, 1.0)))):
            raise ContractError(f"Locked validation availability mask is not binary for {sample_id}")
        if float(channel_labels.sum()) < 1.0:
            raise ContractError(f"Locked validation sample has no onset channel label: {sample_id}")
        if float(region_labels.sum()) != 1.0:
            raise ContractError(f"Locked validation sample lacks one explicit onset region label: {sample_id}")
        if not bool(channel_mask.any()):
            raise ContractError(f"Locked validation sample has no available channel: {sample_id}")
        if bool(np.any((channel_labels > 0.5) & ~(channel_mask > 0.5))):
            raise ContractError(f"Locked validation positive channel is unavailable: {sample_id}")
        if not bool(np.array_equal(channel_alias, channel_labels)):
            raise ContractError(f"Locked validation NPZ y != y_segments[1] for {sample_id}")
        if not bool(np.array_equal(region_alias, region_labels)):
            raise ContractError(f"Locked validation NPZ region_y != region_y_segments[1] for {sample_id}")
        require_equal(
            f"locked index soz_bipolar vs NPZ onset labels for {sample_id}",
            tuple(int(value) for value in channel_labels),
            index_channel_labels,
        )
        require_equal(
            f"locked index soz_region vs NPZ onset labels for {sample_id}",
            tuple(int(value) for value in region_labels),
            index_region_labels,
        )
        truth[sample_id] = {
            "patient_id": patient,
            "channel_labels": index_channel_labels,
            "region_labels": index_region_labels,
            "channel_available": tuple(int(value) for value in channel_mask),
        }

    require_equal(
        "locked validation sample count",
        len(truth),
        sum(EXPECTED_PANEL_EVENT_COUNTS.values()),
    )
    require_equal(
        "locked validation patient set",
        {item["patient_id"] for item in truth.values()},
        validation_patients,
    )
    patient_counts = Counter(item["patient_id"] for item in truth.values())
    for panel_name, patients in EXPECTED_PANELS.items():
        panel_events = sum(patient_counts[patient] for patient in patients)
        require_equal(
            f"locked {panel_name} event count",
            panel_events,
            EXPECTED_PANEL_EVENT_COUNTS[panel_name],
        )
    return truth


def expected_validation_truth_sha256(expected_truth: Mapping[str, Mapping[str, Any]]) -> str:
    return sha256_jsonable(
        {
            sample_id: {
                "patient_id": row["patient_id"],
                "channel_labels": list(row["channel_labels"]),
                "region_labels": list(row["region_labels"]),
                "channel_available": list(row["channel_available"]),
            }
            for sample_id, row in sorted(expected_truth.items())
        }
    )


def validate_prediction_truth(
    rows: Sequence[Mapping[str, str]],
    expected_truth: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject CSV-carried GT unless it exactly matches locked index/NPZ truth."""

    sample_ids = [str(row.get("sample_id", "")).strip() for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ContractError("Validation sample_id values must be nonempty and unique")
    require_equal("validation sample_id set", set(sample_ids), set(expected_truth))
    channel_columns = [channel.replace("-", "_") for channel in CHANNELS]
    for row, sample_id in zip(rows, sample_ids):
        expected = expected_truth[sample_id]
        require_equal(
            f"validation patient_id for {sample_id}",
            str(row.get("patient_id", "")).strip(),
            expected["patient_id"],
        )
        observed_labels = tuple(
            parse_float(row.get(f"onset_label_{column}"), label=f"onset_label_{column}")
            for column in channel_columns
        )
        observed_regions = tuple(
            parse_float(
                row.get(f"onset_region_label_{region}"),
                label=f"onset_region_label_{region}",
            )
            for region in REGIONS
        )
        observed_available = tuple(
            parse_float(row.get(f"available_{column}"), label=f"available_{column}")
            for column in channel_columns
        )
        require_equal(
            f"locked onset channel labels for {sample_id}",
            observed_labels,
            tuple(float(value) for value in expected["channel_labels"]),
        )
        require_equal(
            f"locked explicit onset region labels for {sample_id}",
            observed_regions,
            tuple(float(value) for value in expected["region_labels"]),
        )
        require_equal(
            f"locked channel availability for {sample_id}",
            observed_available,
            tuple(float(value) for value in expected["channel_available"]),
        )


def parse_float(value: object, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Non-numeric {label}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ContractError(f"Non-finite {label}: {value!r}")
    return parsed


def matrix(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[parse_float(row.get(column), label=column) for column in columns] for row in rows],
        dtype=float,
    )


def rank_first_positive(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = np.take_along_axis(labels > 0.5, order, axis=1)
    if not bool(np.all(ranked.any(axis=1))):
        raise ContractError("Every validation event must contain at least one positive label")
    return np.argmax(ranked, axis=1) + 1


def prediction_metrics(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    if not rows:
        raise ContractError("Cannot score an empty validation subset")
    channel_columns = [channel.replace("-", "_") for channel in CHANNELS]
    channel_prob_columns = [f"onset_prob_{column}" for column in channel_columns]
    channel_label_columns = [f"onset_label_{column}" for column in channel_columns]
    channel_available_columns = [f"available_{column}" for column in channel_columns]
    region_prob_columns = [f"onset_region_prob_{region}" for region in REGIONS]
    region_label_columns = [f"onset_region_label_{region}" for region in REGIONS]
    required = {
        "sample_id",
        "patient_id",
        "top1_channel",
        "top1_channel_probability",
        "top1_region",
        "top1_region_probability",
        "top1_score_source",
        *channel_prob_columns,
        *channel_label_columns,
        *channel_available_columns,
        *region_prob_columns,
        *region_label_columns,
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ContractError(f"val_predictions.csv is missing required raw columns: {missing}")

    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ContractError("Validation sample_id values must be nonempty and unique")
    channel_probs = matrix(rows, channel_prob_columns)
    channel_labels = matrix(rows, channel_label_columns)
    channel_available = matrix(rows, channel_available_columns) > 0.5
    region_probs = matrix(rows, region_prob_columns)
    region_labels = matrix(rows, region_label_columns)
    if bool(np.any((channel_probs < 0.0) | (channel_probs > 1.0))) or bool(
        np.any((region_probs < 0.0) | (region_probs > 1.0))
    ):
        raise ContractError("Raw validation probabilities must lie in [0, 1]")
    if bool(np.any(~np.isin(channel_labels, (0.0, 1.0)))) or bool(
        np.any(~np.isin(region_labels, (0.0, 1.0)))
    ):
        raise ContractError("Validation labels must be binary")
    if not bool(np.all(channel_labels.sum(axis=1) >= 1.0)):
        raise ContractError("Every validation event must have at least one soz_bipolar channel")
    if not bool(np.allclose(region_labels.sum(axis=1), 1.0)):
        raise ContractError("Every validation event must have exactly one explicit soz_region label")
    if not bool(np.allclose(region_probs.sum(axis=1), 1.0, atol=1e-5)):
        raise ContractError("Raw region probabilities must be the locked five-way softmax")
    if not bool(np.all(channel_available.any(axis=1))):
        raise ContractError("Every validation event must have at least one available channel")
    if bool(np.any((channel_labels > 0.5) & ~channel_available)):
        raise ContractError("A positive channel label is marked unavailable")

    masked_channel_probs = np.where(channel_available, channel_probs, -np.inf)
    channel_top = masked_channel_probs.argmax(axis=1)
    region_top = region_probs.argmax(axis=1)
    channel_rank = rank_first_positive(masked_channel_probs, channel_labels)
    region_rank = rank_first_positive(region_probs, region_labels)
    row_indices = np.arange(len(rows))
    for row_idx, row in enumerate(rows):
        if str(row.get("top1_score_source")) != "raw_onset_probability":
            raise ContractError("Top-1 source must be the uncorrected raw onset probability")
        expected_channel = CHANNELS[int(channel_top[row_idx])]
        expected_region = REGIONS[int(region_top[row_idx])]
        if str(row.get("top1_channel")) != expected_channel:
            raise ContractError(f"Stored Top-1 channel is not the raw argmax for {sample_ids[row_idx]}")
        if str(row.get("top1_region")) != expected_region:
            raise ContractError(f"Stored Top-1 region is not the independent raw argmax for {sample_ids[row_idx]}")
        stored_channel_prob = parse_float(row.get("top1_channel_probability"), label="top1_channel_probability")
        stored_region_prob = parse_float(row.get("top1_region_probability"), label="top1_region_probability")
        if not math.isclose(stored_channel_prob, channel_probs[row_idx, channel_top[row_idx]], abs_tol=1e-8):
            raise ContractError(f"Stored Top-1 channel probability mismatch for {sample_ids[row_idx]}")
        if not math.isclose(stored_region_prob, region_probs[row_idx, region_top[row_idx]], abs_tol=1e-8):
            raise ContractError(f"Stored Top-1 region probability mismatch for {sample_ids[row_idx]}")

    channel_flat_labels = channel_labels[channel_available]
    channel_flat_probs = channel_probs[channel_available]
    region_flat_labels = region_labels.reshape(-1)
    region_flat_probs = region_probs.reshape(-1)
    predicted_channel_regions = np.asarray([CHANNEL_TO_REGION[int(index)] for index in channel_top])
    return {
        "n_events": float(len(rows)),
        "n_patients": float(len({str(row.get("patient_id", "")) for row in rows})),
        "channel_top1": float(np.mean(channel_rank == 1)),
        "channel_top2": float(np.mean(channel_rank <= 2)),
        "channel_top3": float(np.mean(channel_rank <= 3)),
        "channel_mrr": float(np.mean(1.0 / channel_rank)),
        "channel_auprc": float(average_precision_score(channel_flat_labels, channel_flat_probs)),
        "region_top1": float(np.mean(region_rank == 1)),
        "region_top2": float(np.mean(region_rank <= 2)),
        "region_top3": float(np.mean(region_rank <= 3)),
        "region_mrr": float(np.mean(1.0 / region_rank)),
        "region_auprc": float(average_precision_score(region_flat_labels, region_flat_probs)),
        "hierarchy_consistency": float(np.mean(predicted_channel_regions == region_top)),
        "channel_positive_cardinality": float(channel_labels.sum(axis=1).mean()),
        "region_single_label_rate": float(np.mean(region_labels.sum(axis=1) == 1.0)),
        "raw_channel_top1_probability": float(channel_probs[row_indices, channel_top].mean()),
        "raw_region_top1_probability": float(region_probs[row_indices, region_top].mean()),
    }


def summarize_prediction_file(
    path: Path,
    *,
    expected_truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = read_csv_rows(path)
    validate_prediction_truth(rows, expected_truth)
    expected_validation = set(pooled_validation_patients())
    observed_validation = {str(row.get("patient_id", "")).strip() for row in rows}
    if observed_validation != expected_validation:
        raise ContractError(
            f"Pooled val_predictions patient mismatch: observed={sorted(observed_validation)}, "
            f"expected={sorted(expected_validation)}"
        )
    if len(rows) != sum(EXPECTED_PANEL_EVENT_COUNTS.values()):
        raise ContractError(f"Pooled validation must contain exactly 42 events, observed={len(rows)}")
    pooled = prediction_metrics(rows)
    per_patient = {
        patient: prediction_metrics(
            [row for row in rows if str(row.get("patient_id", "")).strip() == patient]
        )
        for patient in sorted(expected_validation)
    }
    patient_macro_metric_names = (
        "channel_top1",
        "region_top1",
        "channel_mrr",
        "region_mrr",
        "channel_auprc",
        "region_auprc",
        "hierarchy_consistency",
    )
    patient_macro = {
        metric: float(mean(patient_metrics[metric] for patient_metrics in per_patient.values()))
        for metric in patient_macro_metric_names
    }
    panels: dict[str, dict[str, float]] = {}
    panel_assignment: dict[str, str] = {}
    for panel_name, patients in EXPECTED_PANELS.items():
        for patient in patients:
            if patient in panel_assignment:
                raise ContractError(f"Validation patient assigned to multiple panels: {patient}")
            panel_assignment[patient] = panel_name
        panel_rows = [row for row in rows if str(row.get("patient_id", "")).strip() in set(patients)]
        expected_events = EXPECTED_PANEL_EVENT_COUNTS[panel_name]
        if len(panel_rows) != expected_events:
            raise ContractError(f"{panel_name} must contain {expected_events} events, observed={len(panel_rows)}")
        if {str(row.get("patient_id", "")).strip() for row in panel_rows} != set(patients):
            raise ContractError(f"{panel_name} is missing one or more locked patients")
        panels[panel_name] = prediction_metrics(panel_rows)
    return {
        "pooled": pooled,
        "patient_macro": patient_macro,
        "per_patient": per_patient,
        "panels": panels,
        "patients": sorted(observed_validation),
        "event_count": len(rows),
        "panel_assignment": panel_assignment,
    }


def expected_entries(
    plan: Mapping[str, Any],
    expected_plan: Mapping[str, Any],
    *,
    screening_root: Path,
    audit: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    commands = plan.get("commands", [])
    expected_commands = expected_plan.get("commands", [])
    if not isinstance(commands, list) or not isinstance(expected_commands, list):
        raise ContractError("SCREENING_PLAN commands must be a list")
    entries: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in commands:
        if not isinstance(entry, Mapping):
            raise ContractError("Invalid command entry in SCREENING_PLAN")
        key = (str(entry.get("config")), int(entry.get("seed", -1)))
        if key in entries:
            raise ContractError(f"Duplicate screening command entry: {key}")
        entries[key] = entry
    expected = {(config, seed) for config in CONFIG_SPECS for seed in EXPECTED_SEEDS}
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise ContractError(f"Screening matrix mismatch: missing={missing}, extra={extra}")
    expected_order = [(config, seed) for seed in EXPECTED_SEEDS for config in CONFIG_SPECS]
    observed_order = [(str(entry.get("config")), int(entry.get("seed", -1))) for entry in commands]
    if observed_order != expected_order:
        raise ContractError("Screening command order must finish each same-seed clean base before its matched arms")
    expected_by_key = {
        (str(entry["config"]), int(entry["seed"])): entry
        for entry in expected_commands
    }
    runs_root = screening_root / "runs"
    for seed in EXPECTED_SEEDS:
        base_entry = entries[("tfm_soz", seed)]
        if base_entry.get("depends_on") is not None:
            raise ContractError("A clean TFM base cannot depend on an initialization run")
        base_run_dir = base_entry.get("run_dir")
        for config in tuple(CONFIG_SPECS)[1:]:
            arm = entries[(config, seed)]
            if arm.get("depends_on") != base_run_dir:
                raise ContractError(f"{config} seed {seed} does not depend on its same-seed TFM base")
            command = list(arm.get("command", []))
            if "--init-run-dir" not in command or "--freeze-base-channel-head-only" not in command:
                raise ContractError(f"{config} seed {seed} is not a matched frozen arm")
            if command[command.index("--init-run-dir") + 1] != base_run_dir:
                raise ContractError(f"{config} seed {seed} initialization path mismatch")
    for key, entry in entries.items():
        config, seed = key
        run_text = str(entry.get("run_dir", ""))
        run_path = PurePosixPath(run_text)
        if not run_text or run_path.is_absolute() or ".." in run_path.parts:
            raise ContractError(f"Unsafe screening run_dir in plan: {run_text!r}")
        expected_run = runs_root / config / f"seed_{seed}"
        expected_run_text = relative_to_repo(expected_run, REPO_ROOT)
        if run_text != expected_run_text:
            raise ContractError(
                f"run_dir must be exactly <screening_root>/runs/<config>/seed_<seed>: {run_text}"
            )
        resolved_run = resolve_under_repo(run_text, REPO_ROOT).resolve()
        try:
            resolved_run.relative_to(runs_root.resolve())
        except ValueError as exc:
            raise ContractError(f"run_dir escapes the current screening root: {run_text}") from exc
        command = entry.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ContractError(f"Invalid command vector for {key}")
        require_equal(
            f"recomputed command SHA256 for {key}",
            entry.get("command_sha256"),
            sha256_jsonable(command),
        )
        require_equal(
            f"entry code lock for {key}",
            entry.get("code_lock_sha256"),
            audit.get("code_lock_sha256"),
        )
        require_equal(
            f"entry execution lock for {key}",
            entry.get("execution_lock_sha256"),
            audit.get("execution_lock_sha256"),
        )
        require_equal(f"rebuilt command entry for {key}", dict(entry), dict(expected_by_key[key]))
    return entries


def validate_plan(
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    expected_plan: Mapping[str, Any],
    screening_root: Path,
) -> None:
    if plan.get("schema_version") != "strict_hierarchical_set_screening_plan_v1":
        raise ContractError("Unsupported or missing SCREENING_PLAN schema")
    require = plan.get("protocol", {})
    if require.get("sha256") != audit.get("protocol_sha256"):
        raise ContractError("SCREENING_PLAN protocol hash differs from the current locked protocol")
    if require.get("test_fraction") != 0.0 or require.get("skip_test_eval") is not True:
        raise ContractError("SCREENING_PLAN is not validation-only")
    split = plan.get("split", {})
    if split.get("train_only_patients") != list(EXPECTED_TRAIN_PATIENTS):
        raise ContractError("SCREENING_PLAN fixed train-only pool mismatch")
    if split.get("pooled_validation_patients") != list(pooled_validation_patients()):
        raise ContractError("SCREENING_PLAN pooled validation union mismatch")
    if split.get("panel_specific_checkpoint_selection") is not False:
        raise ContractError("Panel-specific checkpoint selection is forbidden")
    if plan.get("promotion_gate") != gate_policy(protocol):
        raise ContractError("SCREENING_PLAN promotion gate differs from the locked predeclaration")
    if plan.get("configurations") != {name: dict(spec) for name, spec in CONFIG_SPECS.items()}:
        raise ContractError("SCREENING_PLAN configuration matrix differs from the locked two-stage design")
    training = plan.get("training", {})
    required_training = {
        "base_tokenizer_epochs": 16,
        "base_classifier_epochs": 90,
        "arm_tokenizer_epochs": 0,
        "arm_classifier_epochs": 12,
        "freeze_base_channel_head_only_for_all_arms": True,
        "region_attention_pooling_for_all_configs": True,
        "region_embedding_head_for_all_configs": True,
        "historical_or_external_checkpoint_initialization": False,
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ContractError(f"SCREENING_PLAN training contract mismatch for {key}")
    hypothesis = plan.get("hypothesis", {})
    if (
        hypothesis.get("tkpr_k") != TKPR_K
        or hypothesis.get("tkpr_weight") != TKPR_WEIGHT
        or hypothesis.get("set_bottleneck") != SET_BOTTLENECK
        or hypothesis.get("expected_set_added_classifier_parameters")
        != EXPECTED_SET_ADDED_CLASSIFIER_PARAMETERS
    ):
        raise ContractError("SCREENING_PLAN hypothesis hyperparameters changed after lock")
    if float(hypothesis.get("set_residual_init", float("nan"))) != SET_RESIDUAL_INIT:
        raise ContractError("SCREENING_PLAN set residual no longer has exact-identity initialization")
    input_audit = plan.get("input_audit", {})
    if input_audit.get("source_hashes") != audit.get("source_hashes"):
        raise ContractError("One or more locked trainer/model/constants/dataset/runner/gate source hashes changed")
    if input_audit.get("code_lock_sha256") != audit.get("code_lock_sha256"):
        raise ContractError("SCREENING_PLAN code lock changed")
    if input_audit.get("execution_lock_sha256") != audit.get("execution_lock_sha256"):
        raise ContractError("SCREENING_PLAN aggregate execution lock changed")
    require_equal("SCREENING_PLAN complete input audit", input_audit, expected_plan.get("input_audit"))
    expected_entries(plan, expected_plan, screening_root=screening_root, audit=audit)
    plan_without_commands = {key: value for key, value in plan.items() if key != "commands"}
    expected_without_commands = {key: value for key, value in expected_plan.items() if key != "commands"}
    require_equal("rebuilt SCREENING_PLAN top-level contract", plan_without_commands, expected_without_commands)


def load_run_summary(
    *,
    run_dir: Path,
    entry: Mapping[str, Any],
    protocol_sha256: str,
    expected_truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = verify_completed_run(
        run_dir,
        entry=entry,
        protocol_sha256=protocol_sha256,
        require_receipt=True,
    )
    prediction = summarize_prediction_file(
        run_dir / "val_predictions.csv",
        expected_truth=expected_truth,
    )
    metrics = load_json(run_dir / "metrics.json")
    classifier = metrics.get("parameters", {}).get("classifier", {})
    trainable = parse_float(classifier.get("trainable"), label="classifier.trainable")
    total = parse_float(classifier.get("total"), label="classifier.total")
    if (
        trainable <= 0
        or total <= 0
        or trainable > total
        or not trainable.is_integer()
        or not total.is_integer()
    ):
        raise ContractError(f"Invalid classifier parameter counts in {run_dir}")
    run_config = load_json(run_dir / "run_config.json")
    config_classifier = run_config.get("parameters", {}).get("classifier", {})
    require_equal(
        f"metrics/run_config classifier trainable count in {run_dir}",
        int(trainable),
        int(config_classifier.get("trainable", -1)),
    )
    require_equal(
        f"metrics/run_config classifier total count in {run_dir}",
        int(total),
        int(config_classifier.get("total", -1)),
    )
    return {
        **prediction,
        "classifier_trainable": int(trainable),
        "classifier_total": int(total),
        "receipt": receipt,
        "run_dir": relative_to_repo(run_dir, REPO_ROOT),
    }


def metric_mean(
    runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    config: str,
    metric: str,
    *,
    panel: str | None = None,
) -> float:
    values = []
    for seed in EXPECTED_SEEDS:
        scope = runs[config][seed]["pooled"] if panel is None else runs[config][seed]["panels"][panel]
        values.append(float(scope[metric]))
    return float(mean(values))


def comparison(
    runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    candidate: str,
    control: str,
) -> dict[str, Any]:
    metrics = (*PRIMARY, *SECONDARY, "hierarchy_consistency")
    by_seed: dict[str, dict[str, float]] = {}
    for seed in EXPECTED_SEEDS:
        by_seed[str(seed)] = {
            metric: float(runs[candidate][seed]["pooled"][metric])
            - float(runs[control][seed]["pooled"][metric])
            for metric in metrics
        }
    by_panel: dict[str, dict[str, float]] = {}
    for panel in EXPECTED_PANELS:
        by_panel[panel] = {
            metric: float(
                mean(
                    float(runs[candidate][seed]["panels"][panel][metric])
                    - float(runs[control][seed]["panels"][panel][metric])
                    for seed in EXPECTED_SEEDS
                )
            )
            for metric in metrics
        }
    pooled_mean_delta = {
        metric: metric_mean(runs, candidate, metric) - metric_mean(runs, control, metric)
        for metric in metrics
    }
    patient_win_tie_loss: dict[str, dict[str, Any]] = {}
    for endpoint in PRIMARY:
        patient_deltas: dict[str, float] = {}
        for patient in pooled_validation_patients():
            patient_deltas[patient] = float(
                mean(
                    float(runs[candidate][seed]["per_patient"][patient][endpoint])
                    - float(runs[control][seed]["per_patient"][patient][endpoint])
                    for seed in EXPECTED_SEEDS
                )
            )
        tolerance = 1e-12
        patient_win_tie_loss[endpoint] = {
            "wins": sum(delta > tolerance for delta in patient_deltas.values()),
            "ties": sum(abs(delta) <= tolerance for delta in patient_deltas.values()),
            "losses": sum(delta < -tolerance for delta in patient_deltas.values()),
            "seed_mean_delta_by_patient": patient_deltas,
        }
    return {
        "candidate": candidate,
        "control": control,
        "pooled_mean_delta": pooled_mean_delta,
        "by_seed_pooled_delta": by_seed,
        "by_panel_seed_mean_delta": by_panel,
        "patient_win_tie_loss": patient_win_tie_loss,
    }


def gate_check(name: str, passed: bool, **evidence: object) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def evaluate_promotion(
    runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons = {
        control: comparison(runs, "full", control)
        for control in ("tfm_soz", "head_only", "tkpr_only", "set_only", "cyclic_permuted")
    }
    checks: list[dict[str, Any]] = []
    noninferiority_floor = -float(policy["noninferiority_max_absolute_drop"])
    meaningful_gain = float(policy["meaningful_min_absolute_gain"])
    for control in policy["primary_controls"]:
        deltas = comparisons[control]["pooled_mean_delta"]
        for endpoint in PRIMARY:
            checks.append(
                gate_check(
                    f"noninferiority_vs_{control}_{endpoint}",
                    float(deltas[endpoint]) >= noninferiority_floor,
                    delta=deltas[endpoint],
                    floor=noninferiority_floor,
                )
            )
        for metric in SECONDARY:
            metric_kind = "mrr" if metric.endswith("mrr") else "auprc"
            floor = -float(policy["secondary_max_mean_drop"][metric_kind])
            checks.append(
                gate_check(
                    f"secondary_stability_vs_{control}_{metric}",
                    float(deltas[metric]) >= floor,
                    delta=deltas[metric],
                    floor=floor,
                )
            )

        for endpoint in PRIMARY:
            seed_deltas = [
                float(comparisons[control]["by_seed_pooled_delta"][str(seed)][endpoint])
                for seed in EXPECTED_SEEDS
            ]
            panel_deltas = [
                float(comparisons[control]["by_panel_seed_mean_delta"][panel][endpoint])
                for panel in EXPECTED_PANELS
            ]
            min_seeds = int(policy["minimum_nonnegative_seeds_per_primary_control_and_endpoint"])
            min_panels = int(policy["minimum_nonnegative_panels_per_primary_control_and_endpoint"])
            worst_floor = float(policy["worst_panel_primary_delta_floor"])
            checks.extend(
                (
                    gate_check(
                        f"seed_direction_vs_{control}_{endpoint}",
                        sum(delta >= 0.0 for delta in seed_deltas) >= min_seeds,
                        deltas=seed_deltas,
                        required_nonnegative=min_seeds,
                    ),
                    gate_check(
                        f"panel_direction_vs_{control}_{endpoint}",
                        sum(delta >= 0.0 for delta in panel_deltas) >= min_panels,
                        deltas=dict(zip(EXPECTED_PANELS, panel_deltas)),
                        required_nonnegative=min_panels,
                    ),
                    gate_check(
                        f"worst_panel_floor_vs_{control}_{endpoint}",
                        min(panel_deltas) >= worst_floor,
                        worst_delta=min(panel_deltas),
                        floor=worst_floor,
                    ),
                )
            )

    meaningful_by_endpoint = {
        endpoint: {
            control: float(comparisons[control]["pooled_mean_delta"][endpoint])
            for control in policy["primary_controls"]
        }
        for endpoint in PRIMARY
    }
    same_endpoint_passes = [
        endpoint
        for endpoint, control_deltas in meaningful_by_endpoint.items()
        if all(delta >= meaningful_gain for delta in control_deltas.values())
    ]
    checks.append(
        gate_check(
            "same_primary_endpoint_meaningful_gain_against_each_control",
            bool(same_endpoint_passes)
            and policy.get("same_primary_endpoint_must_meet_gain_against_each_control") is True,
            deltas_by_endpoint_and_control=meaningful_by_endpoint,
            passing_endpoints=same_endpoint_passes,
            minimum_gain=meaningful_gain,
        )
    )

    ablation_min_gain = float(policy["full_vs_each_ablation_min_gain_on_one_primary"])
    ablation_other_floor = -float(policy["full_vs_each_ablation_max_drop_on_other_primary"])
    for control in policy["ablation_controls"]:
        deltas = comparisons[control]["pooled_mean_delta"]
        primary_deltas = [float(deltas[endpoint]) for endpoint in PRIMARY]
        checks.append(
            gate_check(
                f"full_beats_ablation_{control}",
                max(primary_deltas) >= ablation_min_gain and min(primary_deltas) >= ablation_other_floor,
                deltas=dict(zip(PRIMARY, primary_deltas)),
                minimum_gain_on_one=ablation_min_gain,
                other_endpoint_floor=ablation_other_floor,
            )
        )

    consistency_policy = policy["hierarchy_consistency"]
    for control, threshold_key in (
        ("head_only", "min_gain_vs_head_only"),
        ("tfm_soz", "min_gain_vs_tfm_soz"),
    ):
        delta = float(comparisons[control]["pooled_mean_delta"]["hierarchy_consistency"])
        threshold = float(consistency_policy[threshold_key])
        checks.append(
            gate_check(
                f"hierarchy_consistency_vs_{control}",
                delta >= threshold,
                delta=delta,
                minimum_gain=threshold,
            )
        )

    parameter_fractions: dict[str, float] = {}
    added_total_fractions: dict[str, float] = {}
    added_over_head_trainable: dict[str, float] = {}
    parameter_structure: dict[str, dict[str, Any]] = {}
    max_fraction = float(policy["maximum_added_trainable_fraction"])
    for seed in EXPECTED_SEEDS:
        base_total = float(runs["tfm_soz"][seed]["classifier_total"])
        head_total = float(runs["head_only"][seed]["classifier_total"])
        tkpr_total = float(runs["tkpr_only"][seed]["classifier_total"])
        set_total = float(runs["set_only"][seed]["classifier_total"])
        full_total = float(runs["full"][seed]["classifier_total"])
        permuted_total = float(runs["cyclic_permuted"][seed]["classifier_total"])
        head_trainable = float(runs["head_only"][seed]["classifier_trainable"])
        tkpr_trainable = float(runs["tkpr_only"][seed]["classifier_trainable"])
        set_trainable = float(runs["set_only"][seed]["classifier_trainable"])
        full_trainable = float(runs["full"][seed]["classifier_trainable"])
        permuted_trainable = float(runs["cyclic_permuted"][seed]["classifier_trainable"])
        added_total = full_total - base_total
        added_trainable = full_trainable - head_trainable
        parameter_fractions[str(seed)] = added_trainable / base_total
        added_total_fractions[str(seed)] = added_total / base_total
        added_over_head_trainable[str(seed)] = added_trainable / head_trainable
        parameter_structure[str(seed)] = {
            "tfm_head_tkpr_total_match": base_total == head_total == tkpr_total,
            "set_full_permuted_total_match": set_total == full_total == permuted_total,
            "head_tkpr_trainable_match": head_trainable == tkpr_trainable,
            "set_full_permuted_trainable_match": set_trainable == full_trainable == permuted_trainable,
            "added_total": added_total,
            "added_total_is_locked_2707": added_total == EXPECTED_SET_ADDED_CLASSIFIER_PARAMETERS,
            "added_trainable": added_trainable,
            "all_added_parameters_are_trainable": added_trainable == added_total,
        }
    cross_seed_structure = {
        config: {
            "classifier_total_constant": len(
                {float(runs[config][seed]["classifier_total"]) for seed in EXPECTED_SEEDS}
            )
            == 1,
            "classifier_trainable_constant": len(
                {float(runs[config][seed]["classifier_trainable"]) for seed in EXPECTED_SEEDS}
            )
            == 1,
        }
        for config in CONFIG_SPECS
    }
    checks.append(
        gate_check(
            "parameter_budget",
            all(0.0 <= value <= max_fraction for value in parameter_fractions.values()),
            added_trainable_over_locked_base_classifier_total=parameter_fractions,
            added_total_over_locked_base_classifier_total=added_total_fractions,
            added_trainable_over_head_only_trainable_diagnostic=added_over_head_trainable,
            maximum=max_fraction,
            numerator="full classifier trainable - head_only classifier trainable",
            denominator="same-seed tfm_soz locked base classifier total",
        )
    )
    checks.append(
        gate_check(
            "parameter_structure_contract",
            all(
                all(
                    bool(value)
                    for key, value in fields.items()
                    if key.endswith("_match") or key.startswith("added_total_is_") or key == "all_added_parameters_are_trainable"
                )
                for fields in parameter_structure.values()
            )
            and all(all(fields.values()) for fields in cross_seed_structure.values()),
            by_seed=parameter_structure,
            cross_seed=cross_seed_structure,
        )
    )

    promoted = all(bool(check["passed"]) for check in checks)
    return {
        "comparisons": comparisons,
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "promoted": promoted,
    }


def run_summaries_for_report(runs: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for config in CONFIG_SPECS:
        report[config] = {
            "seed_mean_pooled": {
                metric: metric_mean(runs, config, metric)
                for metric in (*PRIMARY, *SECONDARY, "channel_top2", "region_top2", "hierarchy_consistency")
            },
            "seed_mean_patient_macro": {
                metric: float(
                    mean(runs[config][seed]["patient_macro"][metric] for seed in EXPECTED_SEEDS)
                )
                for metric in (
                    "channel_top1",
                    "region_top1",
                    "channel_mrr",
                    "region_mrr",
                    "channel_auprc",
                    "region_auprc",
                    "hierarchy_consistency",
                )
            },
            "seeds": {
                str(seed): {
                    "pooled": runs[config][seed]["pooled"],
                    "panels": runs[config][seed]["panels"],
                    "patient_macro": runs[config][seed]["patient_macro"],
                    "per_patient": runs[config][seed]["per_patient"],
                    "classifier_trainable": runs[config][seed]["classifier_trainable"],
                    "classifier_total": runs[config][seed]["classifier_total"],
                    "run_dir": runs[config][seed]["run_dir"],
                }
                for seed in EXPECTED_SEEDS
            },
        }
    return report


def markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Strict validation-only screening gate",
        "",
        "本报告只读取锁定 screening plan 中列出的 `val_predictions.csv`。训练固定为 28 人；15 人 pooled validation 只产生一个 checkpoint，A/B/C 仅用于同一预测文件的分层稳定性检查；未读取或生成 test 指标/预测。",
        "",
        "| Config | Ch Top-1 | Reg Top-1 | Ch MRR | Reg MRR | Ch AUPRC | Reg AUPRC | Hier. consistency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summaries = payload.get("run_summaries", {})
    for config in CONFIG_SPECS:
        metrics = summaries.get(config, {}).get("seed_mean_pooled", {})
        lines.append(
            f"| {config} | {float(metrics.get('channel_top1', float('nan'))):.4f} | "
            f"{float(metrics.get('region_top1', float('nan'))):.4f} | "
            f"{float(metrics.get('channel_mrr', float('nan'))):.4f} | "
            f"{float(metrics.get('region_mrr', float('nan'))):.4f} | "
            f"{float(metrics.get('channel_auprc', float('nan'))):.4f} | "
            f"{float(metrics.get('region_auprc', float('nan'))):.4f} | "
            f"{float(metrics.get('hierarchy_consistency', float('nan'))):.4f} |"
        )
    lines.extend(
        (
            "",
            "## Patient-macro secondary report",
            "",
            "| Config | Ch Top-1 | Reg Top-1 | Ch MRR | Reg MRR | Ch AUPRC | Reg AUPRC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for config in CONFIG_SPECS:
        metrics = summaries.get(config, {}).get("seed_mean_patient_macro", {})
        lines.append(
            f"| {config} | {float(metrics.get('channel_top1', float('nan'))):.4f} | "
            f"{float(metrics.get('region_top1', float('nan'))):.4f} | "
            f"{float(metrics.get('channel_mrr', float('nan'))):.4f} | "
            f"{float(metrics.get('region_mrr', float('nan'))):.4f} | "
            f"{float(metrics.get('channel_auprc', float('nan'))):.4f} | "
            f"{float(metrics.get('region_auprc', float('nan'))):.4f} |"
        )
    promotion = payload.get("promotion", {})
    lines.extend(
        (
            "",
            f"Promotion: **{'PASS' if promotion.get('promoted') else 'FAIL'}**",
            "",
            "Failed checks:",
            "",
        )
    )
    failed = promotion.get("failed_checks", [])
    if failed:
        lines.extend(f"- `{name}`" for name in failed)
    else:
        lines.append("- None")
    lines.extend(("", "Patient win/tie/loss (Full vs primary controls; patient score first averaged over seeds):", ""))
    comparisons = promotion.get("comparisons", {})
    for control in ("tfm_soz", "head_only"):
        patient_rows = comparisons.get(control, {}).get("patient_win_tie_loss", {})
        for endpoint in PRIMARY:
            row = patient_rows.get(endpoint, {})
            lines.append(
                f"- Full vs {control}, {endpoint}: "
                f"{row.get('wins', 0)}/{row.get('ties', 0)}/{row.get('losses', 0)}"
            )
    lines.extend(
        (
            "",
            "任何 gate 失败都禁止进入完整私有 LOPO/test。即使全部通过，本结果仍是内部 validation evidence，不是独立 confirmatory evidence。",
            "Gate 进程退出码 3 表示锁定的科学负结果：停止该方向，不得自动重试或转入 test。",
            "",
        )
    )
    return "\n".join(lines)


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise ContractError(f"Refusing to overwrite a different existing gate report: {path}")
        return
    path.write_text(content, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-root", default=str(DEFAULT_SCREENING_ROOT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--plan", default=str(DEFAULT_SCREENING_ROOT / "SCREENING_PLAN.json"))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    screening_root = resolve_under_repo(args.screening_root, REPO_ROOT)
    protocol_path = resolve_under_repo(args.protocol, REPO_ROOT)
    plan_path = resolve_under_repo(args.plan, REPO_ROOT)
    output_json = resolve_under_repo(args.output_json, REPO_ROOT) if args.output_json else screening_root / "GATE_REPORT.json"
    output_md = resolve_under_repo(args.output_md, REPO_ROOT) if args.output_md else screening_root / "GATE_REPORT.md"
    require_equal(
        "screening root",
        screening_root.resolve(),
        resolve_under_repo(DEFAULT_SCREENING_ROOT, REPO_ROOT).resolve(),
    )
    require_equal(
        "protocol path",
        protocol_path.resolve(),
        resolve_under_repo(DEFAULT_PROTOCOL, REPO_ROOT).resolve(),
    )
    require_equal("plan path", plan_path.resolve(), (screening_root / "SCREENING_PLAN.json").resolve())
    require_no_symlink_chain(screening_root / "runs", anchor=REPO_ROOT)
    require_no_symlink_chain(plan_path, anchor=REPO_ROOT)
    require_no_symlink_chain(protocol_path, anchor=REPO_ROOT)
    plan = load_json(plan_path)
    protocol = load_json(protocol_path)
    preprocessed_dir = resolve_under_repo(DEFAULT_PREPROCESSED, REPO_ROOT)
    audit = validate_locked_inputs(
        repo_root=REPO_ROOT,
        protocol_path=protocol_path,
        preprocessed_dir=preprocessed_dir,
        trainer_path=resolve_under_repo("code/tfm_soz/train_private_soz_segments.py", REPO_ROOT),
    )
    expected_truth = build_expected_validation_truth(preprocessed_dir)
    expected_truth_sha256 = expected_validation_truth_sha256(expected_truth)
    expected_plan = build_plan(
        mode="screen",
        repo_root=REPO_ROOT,
        protocol_path=protocol_path,
        preprocessed_dir=preprocessed_dir,
        screening_root=screening_root,
        trainer_path=resolve_under_repo("code/tfm_soz/train_private_soz_segments.py", REPO_ROOT),
        audit=audit,
        tokenizer_epochs=BASE_TOKENIZER_EPOCHS,
        classifier_epochs=BASE_CLASSIFIER_EPOCHS,
        arm_classifier_epochs=ARM_CLASSIFIER_EPOCHS,
        batch_size=8,
        num_workers=0,
        device="cuda",
        conservative_total_gpu_hours_upper=2.0,
    )
    validate_plan(
        plan,
        protocol,
        audit,
        expected_plan=expected_plan,
        screening_root=screening_root,
    )
    entries = expected_entries(
        plan,
        expected_plan,
        screening_root=screening_root,
        audit=audit,
    )

    runs: dict[str, dict[int, dict[str, Any]]] = {config: {} for config in CONFIG_SPECS}
    for (config, seed), entry in entries.items():
        run_dir = resolve_under_repo(entry["run_dir"], REPO_ROOT)
        runs[config][seed] = load_run_summary(
            run_dir=run_dir,
            entry=entry,
            protocol_sha256=str(audit["protocol_sha256"]),
            expected_truth=expected_truth,
        )

    promotion = evaluate_promotion(runs, plan["promotion_gate"])
    payload = {
        "schema_version": "strict_hierarchical_set_validation_gate_v1",
        "protocol_path": relative_to_repo(protocol_path, REPO_ROOT),
        "protocol_sha256": audit["protocol_sha256"],
        "plan_path": relative_to_repo(plan_path, REPO_ROOT),
        "plan_sha256": sha256_file(plan_path),
        "split_read": "pooled validation only, then locked A/B/C stratification",
        "expected_validation_truth": {
            "source": "locked index.csv plus hashed NPZ y_segments[1], region_y_segments[1], and channel_mask",
            "events": len(expected_truth),
            "sha256": expected_truth_sha256,
        },
        "test_predictions_or_metrics_read": False,
        "test_dataset_constructed": False,
        "all_expected_runs_present": True,
        "expected_run_count": len(CONFIG_SPECS) * len(EXPECTED_SEEDS),
        "run_summaries": run_summaries_for_report(runs),
        "promotion": promotion,
        "gate_exit_code": 0 if promotion["promoted"] else 3,
        "gate_exit_code_3_meaning": "locked scientific negative; stop direction, no automatic retry, no test evaluation",
        "automatic_retry_permitted": False,
        "evidence_status": "internal locked validation evidence only",
        "confirmatory_claim_allowed": False,
    }
    write_output(output_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_output(output_md, markdown(payload))
    print(
        json.dumps(
            {
                "promoted": promotion["promoted"],
                "failed_checks": promotion["failed_checks"],
                "output_json": str(output_json),
                "output_md": str(output_md),
                "test_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if promotion["promoted"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, FileNotFoundError, KeyError, ValueError, IndexError, TypeError) as exc:
        print(f"STRICT GATE REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
