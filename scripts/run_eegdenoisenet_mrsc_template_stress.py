#!/usr/bin/env python3
"""Stress the frozen MRSC quality port with EEGdenoiseNet EMG/EOG templates.

This is a target-free engineering stress test, not clinical artifact
classification.  It mixes standardized, artifact-only EEGdenoiseNet epochs
into deterministic non-DeepSOZ TUSZ carriers at a fixed SNR grid.  The test
never reads SOZ labels, private data, annotation sidecars, or localization
scores and cannot change an SOZ ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
from scipy.signal import resample_poly
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mrsc_signal_quality_source_only_stress import (  # noqa: E402
    DEFAULT_DEEPSOZ_SPLIT,
    DEFAULT_MANIFEST,
    _load_config,
    _select_lowest_uncertainty_valid_candidate,
)
from src.soz.data.labram_source_dapt import (  # noqa: E402
    SourceDAPTWindowDataset,
    load_source_dapt_manifest,
)
from src.soz.mrsc_signal_quality import (  # noqa: E402
    MRSC_QUALITY_CANDIDATE_CHANNELS,
    assess_mrsc_signal_quality,
)


DEFAULT_DATA_DIRECTORY = Path(
    "/mnt/hd1/dyf/dataset/EEGdenoiseNet/raw_npy"
)
DEFAULT_CONFIG = ROOT / "configs/preprocess_qc.yaml"
DEFAULT_OUTPUT = (
    ROOT / "outputs/eegdenoisenet_mrsc_template_stress_v2_20260813.json"
)
SCHEMA = "soz_eegdenoisenet_mrsc_template_stress_v2"
SOURCE_SFREQ_HZ = 256.0
TARGET_SFREQ_HZ = 200.0
SNR_DB_GRID = (4.0, 2.0, 0.0, -3.0, -7.0)
TEMPLATE_FAMILIES = ("EMG", "EOG")
TEMPLATE_SEED = 20260813
MIN_PAIRWISE_MONOTONIC_RATE = 0.90
MIN_FAMILY_ENDPOINT_INCREASE_RATE = 0.90


def _load_template_pool(directory: Path, family: str) -> np.ndarray:
    if family not in TEMPLATE_FAMILIES:
        raise ValueError("unsupported EEGdenoiseNet template family")
    path = (directory / f"{family}_all_epochs.npy").resolve(strict=True)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_rows = {"EMG": 5598, "EOG": 3400}[family]
    if values.shape != (expected_rows, 512) or values.dtype != np.float64:
        raise ValueError(f"unexpected {family} EEGdenoiseNet array contract")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {family} EEGdenoiseNet templates")
    return values


def _template_schedule(n_templates: int, patient_count: int, family: str) -> tuple[int, ...]:
    family_offset = TEMPLATE_FAMILIES.index(family) * 100_003
    generator = np.random.default_rng(TEMPLATE_SEED + family_offset)
    indices = generator.choice(n_templates, size=patient_count, replace=False)
    return tuple(int(index) for index in indices)


def _prepare_template(template: np.ndarray, target_samples: int) -> np.ndarray:
    values = np.asarray(template, dtype=np.float64)
    if values.shape != (512,) or not np.isfinite(values).all():
        raise ValueError("template must be one finite 512-sample epoch")
    standardized = values - float(np.mean(values))
    scale = float(np.std(standardized))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("template has zero or non-finite variance")
    standardized /= scale
    resampled = resample_poly(standardized, 25, 32)
    if resampled.shape != (400,):
        raise RuntimeError("256-to-200 Hz template resampling drifted")
    repeats = int(np.ceil(target_samples / resampled.size))
    tiled = np.tile(resampled, repeats)[:target_samples]
    tiled = tiled - float(np.mean(tiled))
    tiled_scale = float(np.sqrt(np.mean(np.square(tiled))))
    if tiled_scale <= 1e-12 or not np.isfinite(tiled_scale):
        raise ValueError("prepared template has invalid RMS")
    return np.ascontiguousarray(tiled / tiled_scale)


def _mix_at_snr_db(carrier: np.ndarray, template_unit_rms: np.ndarray, snr_db: float) -> np.ndarray:
    clean = np.asarray(carrier, dtype=np.float64)
    noise = np.asarray(template_unit_rms, dtype=np.float64)
    if clean.shape != noise.shape or clean.ndim != 1:
        raise ValueError("carrier and template must be same-length vectors")
    centered = clean - float(np.mean(clean))
    clean_rms = float(np.sqrt(np.mean(np.square(centered))))
    if clean_rms <= 1e-12 or not np.isfinite(clean_rms):
        raise ValueError("carrier has invalid RMS")
    # EEGdenoiseNet defines SNR as 10*log10(RMS(clean)/RMS(noise)).
    noise_rms = clean_rms / (10.0 ** (float(snr_db) / 10.0))
    mixed = clean + noise_rms * noise
    if not np.isfinite(mixed).all():
        raise RuntimeError("template mixture became non-finite")
    return np.ascontiguousarray(mixed)


def _is_nondecreasing(values: tuple[float, ...], tolerance: float = 1e-12) -> bool:
    return all(right + tolerance >= left for left, right in zip(values, values[1:]))


def run(
    *,
    manifest_path: Path,
    deepsoz_split_path: Path,
    config_path: Path,
    data_directory: Path,
) -> dict[str, object]:
    manifest = load_source_dapt_manifest(
        manifest_path,
        deepsoz_split_roster=deepsoz_split_path,
        verify_file_inventory=True,
    )
    if any(
        manifest.payload[name] is not False
        for name in (
            "target_values_loaded",
            "private_data_loaded",
            "annotation_sidecars_opened",
        )
    ):
        raise ValueError("source-only safety flags changed")
    dataset = SourceDAPTWindowDataset(manifest, split="pretext_dev")
    patient_ids = tuple(sorted(dataset.patient_to_indices))
    config = _load_config(config_path)
    pools = {
        family: _load_template_pool(data_directory, family)
        for family in TEMPLATE_FAMILIES
    }
    schedules = {
        family: _template_schedule(len(pools[family]), len(patient_ids), family)
        for family in TEMPLATE_FAMILIES
    }
    rows: list[dict[str, object]] = []
    monotonic_count = 0
    duplicate_count = 0
    invalidation_monotonic_count = 0
    endpoint_increase_by_family = {family: 0 for family in TEMPLATE_FAMILIES}
    total_series = len(patient_ids) * len(TEMPLATE_FAMILIES)
    for patient_position, patient_id in enumerate(patient_ids):
        item = dataset[min(dataset.patient_to_indices[patient_id])]
        eeg = item["eeg"].detach().cpu().numpy().reshape(19, -1).astype(np.float64)
        if eeg.shape != (19, 1600):
            raise RuntimeError("source carrier changed from [19,1600]")
        baseline = assess_mrsc_signal_quality(eeg, TARGET_SFREQ_HZ, config)
        candidate_position, physical_index = _select_lowest_uncertainty_valid_candidate(
            baseline.channel_uncertainty, baseline.candidate_valid
        )
        baseline_value = float(baseline.channel_uncertainty[physical_index])
        for family in TEMPLATE_FAMILIES:
            template_index = schedules[family][patient_position]
            prepared = _prepare_template(pools[family][template_index], eeg.shape[1])
            uncertainty_values: list[float] = []
            invalid_values: list[bool] = []
            replay_equal = True
            for snr_db in SNR_DB_GRID:
                first = eeg.copy()
                second = eeg.copy()
                first[physical_index] = _mix_at_snr_db(
                    eeg[physical_index], prepared, snr_db
                )
                second[physical_index] = _mix_at_snr_db(
                    eeg[physical_index], prepared, snr_db
                )
                replay_equal = replay_equal and bool(np.array_equal(first, second))
                assessment = assess_mrsc_signal_quality(
                    first, TARGET_SFREQ_HZ, config
                )
                uncertainty_values.append(
                    float(assessment.channel_uncertainty[physical_index])
                )
                invalid_values.append(not assessment.candidate_valid[candidate_position])
            uncertainty_path = (baseline_value, *uncertainty_values)
            monotonic = _is_nondecreasing(uncertainty_path)
            invalidation_monotonic = all(
                (not invalid_values[index]) or all(invalid_values[index:])
                for index in range(len(invalid_values))
            )
            monotonic_count += int(monotonic)
            duplicate_count += int(replay_equal)
            invalidation_monotonic_count += int(invalidation_monotonic)
            endpoint_increase = uncertainty_values[-1] > baseline_value + 1e-12
            endpoint_increase_by_family[family] += int(endpoint_increase)
            rows.append(
                {
                    "patient_position": patient_position,
                    "record_uid": item["record_uid"],
                    "grid_index": int(item["grid_index"]),
                    "selected_candidate": MRSC_QUALITY_CANDIDATE_CHANNELS[
                        candidate_position
                    ],
                    "template_family": family,
                    "template_index": template_index,
                    "snr_db": list(SNR_DB_GRID),
                    "baseline_uncertainty": baseline_value,
                    "uncertainty_by_snr": uncertainty_values,
                    "invalid_by_snr": invalid_values,
                    "uncertainty_nondecreasing": monotonic,
                    "strongest_burden_above_baseline": endpoint_increase,
                    "invalidation_absorbing": invalidation_monotonic,
                    "duplicate_replay_exact": replay_equal,
                }
            )
    monotonic_rate = monotonic_count / total_series
    replay_rate = duplicate_count / total_series
    invalidation_rate = invalidation_monotonic_count / total_series
    endpoint_increase_rate_by_family = {
        family: endpoint_increase_by_family[family] / len(patient_ids)
        for family in TEMPLATE_FAMILIES
    }
    passed = (
        monotonic_rate >= MIN_PAIRWISE_MONOTONIC_RATE
        and replay_rate == 1.0
        and invalidation_rate == 1.0
        and all(
            value >= MIN_FAMILY_ENDPOINT_INCREASE_RATE
            for value in endpoint_increase_rate_by_family.values()
        )
    )
    return {
        "schema_version": SCHEMA,
        "status": (
            "eegdenoisenet_template_monotonicity_qualified"
            if passed
            else "eegdenoisenet_template_monotonicity_stop"
        ),
        "role": "synthetic_template_stress_not_clinical_artifact_validation",
        "source_patient_count": len(patient_ids),
        "template_series_count": total_series,
        "snr_db_grid_descending": list(SNR_DB_GRID),
        "uncertainty_nondecreasing_count": monotonic_count,
        "uncertainty_nondecreasing_rate": monotonic_rate,
        "invalidation_absorbing_count": invalidation_monotonic_count,
        "invalidation_absorbing_rate": invalidation_rate,
        "duplicate_replay_exact_count": duplicate_count,
        "duplicate_replay_exact_rate": replay_rate,
        "strongest_burden_above_baseline_count_by_family": endpoint_increase_by_family,
        "strongest_burden_above_baseline_rate_by_family": endpoint_increase_rate_by_family,
        "gate": {
            "minimum_uncertainty_nondecreasing_rate": MIN_PAIRWISE_MONOTONIC_RATE,
            "required_invalidation_absorbing_rate": 1.0,
            "required_duplicate_replay_rate": 1.0,
            "minimum_strongest_burden_above_baseline_rate_per_family": (
                MIN_FAMILY_ENDPOINT_INCREASE_RATE
            ),
        },
        "data_contract": {
            "dataset": "EEGdenoiseNet",
            "source_sampling_hz": SOURCE_SFREQ_HZ,
            "target_sampling_hz": TARGET_SFREQ_HZ,
            "emg_shape": list(pools["EMG"].shape),
            "eog_shape": list(pools["EOG"].shape),
            "template_seed": TEMPLATE_SEED,
            "template_reuse_across_patients": False,
            "patient_identity_available_for_templates": False,
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
            "clinical_artifact_label_qualified": False,
        },
        "series": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--deepsoz-split", type=Path, default=DEFAULT_DEEPSOZ_SPLIT)
    parser.add_argument("--quality-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run(
        manifest_path=args.manifest,
        deepsoz_split_path=args.deepsoz_split,
        config_path=args.quality_config,
        data_directory=args.data_directory,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "source_patient_count",
                    "template_series_count",
                    "uncertainty_nondecreasing_rate",
                    "invalidation_absorbing_rate",
                    "duplicate_replay_exact_rate",
                    "strongest_burden_above_baseline_rate_by_family",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"].endswith("qualified") else 2


if __name__ == "__main__":
    raise SystemExit(main())
