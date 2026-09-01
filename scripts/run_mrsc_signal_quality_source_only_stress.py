#!/usr/bin/env python3
"""Run the frozen MRSC quality-port stress test on non-DeepSOZ TUSZ patients.

This runner reads one deterministic, annotation-free continuous C-CAR19
window from every patient in the frozen ``pretext_dev`` split.  It never opens
annotation sidecars, SOZ targets, historical correctness, or private data.
Five fixed synthetic corruptions are applied only to detached copies.

The qualification endpoint is deliberately quality-local: a severe
corruption must increase the selected physical channel's observable quality
uncertainty and make that candidate invalid.  It is not allowed to change or
evaluate an SOZ score.  Consequently this run qualifies only a fail-closed
quality port, not an SOZ predictor or a clinical artifact classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.labram_source_dapt import (  # noqa: E402
    SourceDAPTWindowDataset,
    load_source_dapt_manifest,
)
from src.soz.mrsc_signal_quality import (  # noqa: E402
    MRSC_QUALITY_CANDIDATE_CHANNELS,
    MRSC_QUALITY_STRESS_KINDS,
    MRSC_SIGNAL_QUALITY_SCHEMA,
    assess_mrsc_signal_quality,
    inject_mrsc_quality_stress,
)
from src.soz.v11_reasoner import V11_CANDIDATE_INDICES  # noqa: E402


DEFAULT_MANIFEST = (
    ROOT / "outputs/labram_source_only_dapt_manifest_v1_20260811/manifest.json"
)
DEFAULT_DEEPSOZ_SPLIT = (
    ROOT / "outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv"
)
DEFAULT_CONFIG = ROOT / "configs/preprocess_qc.yaml"
DEFAULT_OUTPUT = (
    ROOT / "outputs/labram_mrsc_quality_stress_source_only_20260812.json"
)
QUALIFICATION_SCHEMA = "soz_mrsc_quality_source_only_stress_v1"
QUALIFICATION_SPLIT = "pretext_dev"
QUALIFICATION_SEED = 20260812
BOOTSTRAP_REPLICATES = 10_000
MIN_INCREASE_RATE = 0.90
MIN_PATIENT_BOOTSTRAP_LOWER = 0.80


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patient_token(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()


def _load_config(path: Path) -> Mapping[str, object]:
    resolved = path.resolve(strict=True)
    payload = yaml.safe_load(resolved.read_text())
    if not isinstance(payload, Mapping):
        raise TypeError("quality configuration must be a mapping")
    if payload.get("processing_version") != "preprocess_qc_v1":
        raise ValueError("quality configuration version changed")
    return payload


def _select_lowest_uncertainty_valid_candidate(
    channel_uncertainty: tuple[float, ...],
    candidate_valid: tuple[bool, ...],
) -> tuple[int, int]:
    if len(channel_uncertainty) != 19 or len(candidate_valid) != 18:
        raise ValueError("quality selection requires standard-19/18-candidate inputs")
    eligible = tuple(
        (candidate_position, physical_index)
        for candidate_position, physical_index in enumerate(V11_CANDIDATE_INDICES)
        if candidate_valid[candidate_position]
    )
    if not eligible:
        raise ValueError("unperturbed window has no quality-valid candidate")
    candidate_position, physical_index = min(
        eligible,
        key=lambda item: (channel_uncertainty[item[1]], item[0]),
    )
    return candidate_position, physical_index


def _patient_cluster_bootstrap_lower(
    patient_rates: tuple[float, ...],
    *,
    seed: int = QUALIFICATION_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> float:
    if not patient_rates or any(not 0.0 <= value <= 1.0 for value in patient_rates):
        raise ValueError("patient rates must be a non-empty [0,1] tuple")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    values = np.asarray(patient_rates, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.05, method="linear"))


def _atomic_new_json(path: Path, payload: Mapping[str, object]) -> str:
    output = Path(os.path.abspath(path))
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    if output.is_symlink() or any(
        component.is_symlink() for component in (parent, *parent.parents)
    ):
        raise ValueError("output path cannot traverse a symlink")
    content = _canonical_bytes(payload)
    temporary = parent / f".{output.name}.tmp-{os.getpid()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o644)
    try:
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written < 1:
                raise OSError("short write while publishing quality receipt")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, output, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def run(
    *,
    manifest_path: Path,
    deepsoz_split_path: Path,
    config_path: Path,
) -> dict[str, object]:
    manifest = load_source_dapt_manifest(
        manifest_path,
        deepsoz_split_roster=deepsoz_split_path,
        verify_file_inventory=True,
    )
    if (
        manifest.payload["target_values_loaded"] is not False
        or manifest.payload["private_data_loaded"] is not False
        or manifest.payload["annotation_sidecars_opened"] is not False
    ):
        raise ValueError("source-only safety flags changed")
    dataset = SourceDAPTWindowDataset(manifest, split=QUALIFICATION_SPLIT)
    config = _load_config(config_path)
    patient_rows: list[dict[str, object]] = []
    per_kind_success = {kind: 0 for kind in MRSC_QUALITY_STRESS_KINDS}
    duplicate_replay_count = 0
    baseline_hard_invalid_count = 0
    for patient_id in sorted(dataset.patient_to_indices):
        dataset_index = min(dataset.patient_to_indices[patient_id])
        item = dataset[dataset_index]
        eeg = item["eeg"].detach().cpu().numpy().reshape(19, -1).astype(np.float64)
        if eeg.shape != (19, 1600):
            raise RuntimeError("source-only DAPT window changed from [19,8,200]")
        baseline = assess_mrsc_signal_quality(eeg, 200.0, config)
        baseline_hard_invalid_count += int(baseline.hard_invalid)
        candidate_position, physical_index = _select_lowest_uncertainty_valid_candidate(
            baseline.channel_uncertainty,
            baseline.candidate_valid,
        )
        corruption_rows: list[dict[str, object]] = []
        success_values: list[bool] = []
        for corruption in MRSC_QUALITY_STRESS_KINDS:
            first = inject_mrsc_quality_stress(
                eeg,
                200.0,
                channel_index=physical_index,
                corruption=corruption,
            )
            second = inject_mrsc_quality_stress(
                eeg,
                200.0,
                channel_index=physical_index,
                corruption=corruption,
            )
            duplicate_equal = bool(np.array_equal(first, second))
            duplicate_replay_count += int(duplicate_equal)
            stressed = assess_mrsc_signal_quality(first, 200.0, config)
            baseline_value = float(baseline.channel_uncertainty[physical_index])
            stressed_value = float(stressed.channel_uncertainty[physical_index])
            increased = stressed_value > baseline_value + 1e-12
            invalidated = not stressed.candidate_valid[candidate_position]
            success = bool(increased and invalidated and duplicate_equal)
            success_values.append(success)
            per_kind_success[corruption] += int(success)
            corruption_rows.append(
                {
                    "corruption": corruption,
                    "baseline_channel_uncertainty": baseline_value,
                    "stressed_channel_uncertainty": stressed_value,
                    "increased": increased,
                    "selected_candidate_invalidated": invalidated,
                    "duplicate_replay_exact": duplicate_equal,
                    "passed": success,
                }
            )
        patient_rows.append(
            {
                "patient_token_sha256": _patient_token(patient_id),
                "record_uid": item["record_uid"],
                "grid_index": int(item["grid_index"]),
                "selected_candidate": MRSC_QUALITY_CANDIDATE_CHANNELS[
                    candidate_position
                ],
                "baseline_hard_invalid_elsewhere": baseline.hard_invalid,
                "corruptions": corruption_rows,
                "patient_increase_rate": sum(success_values) / len(success_values),
            }
        )
    patient_rates = tuple(float(row["patient_increase_rate"]) for row in patient_rows)
    pair_count = len(patient_rows) * len(MRSC_QUALITY_STRESS_KINDS)
    passed_pairs = sum(per_kind_success.values())
    increase_rate = passed_pairs / pair_count
    bootstrap_lower = _patient_cluster_bootstrap_lower(patient_rates)
    duplicate_rate = duplicate_replay_count / pair_count
    passed = (
        increase_rate >= MIN_INCREASE_RATE
        and bootstrap_lower >= MIN_PATIENT_BOOTSTRAP_LOWER
        and duplicate_rate == 1.0
    )
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "status": (
            "mrsc_source_quality_port_qualified"
            if passed
            else "mrsc_source_quality_qualification_stop"
        ),
        "quality_port_schema": MRSC_SIGNAL_QUALITY_SCHEMA,
        "qualification_role": (
            "source_only_synthetic_monotonicity_not_clinical_artifact_validation"
        ),
        "split": QUALIFICATION_SPLIT,
        "manifest_sha256": manifest.sha256,
        "deepsoz_exclusion_roster_sha256": _file_sha256(
            deepsoz_split_path.resolve(strict=True)
        ),
        "quality_config_sha256": _file_sha256(config_path.resolve(strict=True)),
        "source_patient_count": len(patient_rows),
        "stress_pair_count": pair_count,
        "passed_pair_count": passed_pairs,
        "increase_and_invalidation_rate": increase_rate,
        "patient_cluster_bootstrap_one_sided_95_lower": bootstrap_lower,
        "duplicate_replay_exact_rate": duplicate_rate,
        "per_corruption_pass_count": per_kind_success,
        "baseline_hard_invalid_window_count": baseline_hard_invalid_count,
        "frozen_gates": {
            "minimum_increase_and_invalidation_rate": MIN_INCREASE_RATE,
            "minimum_patient_bootstrap_lower": MIN_PATIENT_BOOTSTRAP_LOWER,
            "required_duplicate_replay_rate": 1.0,
            "bootstrap_seed": QUALIFICATION_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "safety": {
            "target_values_loaded": False,
            "private_data_loaded": False,
            "annotation_sidecars_opened": False,
            "localization_scores_loaded": False,
            "training_performed": False,
            "model_selection_performed": False,
            "threshold_tuning_performed": False,
            "soz_ranking_changed": False,
        },
        "patients": patient_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--deepsoz-split", type=Path, default=DEFAULT_DEEPSOZ_SPLIT)
    parser.add_argument("--quality-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run(
        manifest_path=args.manifest,
        deepsoz_split_path=args.deepsoz_split,
        config_path=args.quality_config,
    )
    digest = _atomic_new_json(args.output, payload)
    summary = {
        key: payload[key]
        for key in (
            "status",
            "source_patient_count",
            "stress_pair_count",
            "increase_and_invalidation_rate",
            "patient_cluster_bootstrap_one_sided_95_lower",
            "duplicate_replay_exact_rate",
            "per_corruption_pass_count",
            "baseline_hard_invalid_window_count",
        )
    }
    summary["output"] = str(args.output.resolve())
    summary["output_sha256"] = digest
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if payload["status"] == "mrsc_source_quality_port_qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
