#!/usr/bin/env python3
"""Audit target-blind public-to-private input/evidence shift for frozen v29.

This audit reads the already materialized, SOZ-target-free fine temporal
evidence for the 1,145 public development events and all 88 private inference
events.  Event descriptors are reduced across the fixed standard-19 channels
and then averaged within patient, so every patient contributes one unit.

No SOZ/significant/spread reference is loaded, no model is trained, and no
threshold, filter, abstention rule, or v29 component may be selected from the
result.  The purpose is to describe the acquisition/evidence shift traversed
by the frozen post-open private transport analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
from safetensors.torch import load_file, save_file
from scipy.stats import ks_2samp, wasserstein_distance
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES  # noqa: E402


SCHEMA = "trustworthy_soz_v29_target_blind_input_shift_v56"
DEFAULT_PUBLIC_FINE = (
    ROOT / "outputs/public_development_fine_evidence_identity_v12_20260812"
)
DEFAULT_PUBLIC_PREFLIGHT = (
    ROOT
    / "outputs/deepsoz_signal_preflight_identity_v3_20260812/"
    "deepsoz_signal_preflight_identity_v3.json"
)
DEFAULT_PUBLIC_V16 = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
)
DEFAULT_PRIVATE_EVIDENCE = (
    ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_v29_target_blind_input_shift_v56_20260816"
)
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260856


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _patient_descriptors(
    event_features: torch.Tensor,
    patient_ids: Sequence[str],
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    """Return patient-equal channel-mean/SD descriptors and event counts."""

    feature_count = len(FINE_TEMPORAL_FEATURE_NAMES)
    if tuple(event_features.shape[1:]) != (19, feature_count):
        raise ValueError("event features must have shape [E,19,20]")
    if len(patient_ids) != len(event_features):
        raise ValueError("event patient IDs are not aligned with features")
    if not event_features.is_floating_point() or not torch.isfinite(event_features).all():
        raise ValueError("event features must be finite floating point")
    normalized = tuple(str(value) for value in patient_ids)
    if any(not value for value in normalized):
        raise ValueError("patient IDs must be non-empty")

    # Each event contributes a channel mean and population SD for every frozen
    # target-blind feature.  Events are then equally averaged within patient.
    event_summary = torch.cat(
        (
            event_features.mean(dim=1),
            event_features.std(dim=1, unbiased=False),
        ),
        dim=1,
    ).double()
    ordered = tuple(sorted(set(normalized)))
    index = {patient_id: position for position, patient_id in enumerate(ordered)}
    patient_index = torch.tensor([index[value] for value in normalized], dtype=torch.long)
    sums = torch.zeros((len(ordered), event_summary.shape[1]), dtype=torch.float64)
    sums.index_add_(0, patient_index, event_summary)
    counts = torch.bincount(patient_index, minlength=len(ordered)).long()
    if bool((counts < 1).any()):
        raise RuntimeError("patient aggregation lost an event bag")
    result = sums / counts.double().unsqueeze(1)
    if not torch.isfinite(result).all():
        raise RuntimeError("patient descriptors are non-finite")
    return ordered, result.float().contiguous(), counts.contiguous()


def _descriptor_names() -> tuple[str, ...]:
    return tuple(
        [f"channel_mean::{name}" for name in FINE_TEMPORAL_FEATURE_NAMES]
        + [f"channel_sd::{name}" for name in FINE_TEMPORAL_FEATURE_NAMES]
    )


def _bootstrap_smd(
    public: np.ndarray,
    private: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if replicates < 100:
        raise ValueError("bootstrap requires at least 100 replicates")
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(public), size=(replicates, len(public)))
    right = rng.integers(0, len(private), size=(replicates, len(private)))
    lhs = public[left]
    rhs = private[right]
    pooled = np.sqrt(0.5 * (lhs.var(axis=1) + rhs.var(axis=1)))
    difference = rhs.mean(axis=1) - lhs.mean(axis=1)
    smd = np.divide(
        difference,
        pooled,
        out=np.zeros_like(difference),
        where=pooled > 1e-12,
    )
    lower, upper = np.quantile(smd, (0.025, 0.975), axis=0)
    return lower, upper


def _feature_shift_rows(
    public: torch.Tensor,
    private: torch.Tensor,
    *,
    descriptor_names: Sequence[str],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, object]]:
    if public.ndim != 2 or private.ndim != 2 or public.shape[1] != private.shape[1]:
        raise ValueError("public/private descriptors must be aligned matrices")
    if len(descriptor_names) != public.shape[1]:
        raise ValueError("descriptor vocabulary is not aligned")
    lhs = public.double().cpu().numpy()
    rhs = private.double().cpu().numpy()
    pooled = np.sqrt(0.5 * (lhs.var(axis=0) + rhs.var(axis=0)))
    difference = rhs.mean(axis=0) - lhs.mean(axis=0)
    smd = np.divide(
        difference,
        pooled,
        out=np.zeros_like(difference),
        where=pooled > 1e-12,
    )
    lower, upper = _bootstrap_smd(
        lhs, rhs, replicates=replicates, seed=seed
    )
    rows: list[dict[str, object]] = []
    for column, name in enumerate(descriptor_names):
        public_values = lhs[:, column]
        private_values = rhs[:, column]
        scale = float(pooled[column])
        rows.append(
            {
                "descriptor": str(name),
                "public_mean": float(public_values.mean()),
                "private_mean": float(private_values.mean()),
                "public_median": float(np.median(public_values)),
                "private_median": float(np.median(private_values)),
                "pooled_sd": scale,
                "standardized_mean_difference_private_minus_public": float(smd[column]),
                "smd_patient_bootstrap_ci95_low": float(lower[column]),
                "smd_patient_bootstrap_ci95_high": float(upper[column]),
                "standardized_wasserstein": (
                    float(wasserstein_distance(public_values, private_values) / scale)
                    if scale > 1e-12
                    else 0.0
                ),
                "ks_statistic": float(
                    ks_2samp(public_values, private_values, method="auto").statistic
                ),
                "public_constant_and_private_equal": bool(
                    scale <= 1e-12 and np.allclose(public_values, private_values)
                ),
            }
        )
    return rows


def _multivariate_energy(public: torch.Tensor, private: torch.Tensor) -> float:
    if public.ndim != 2 or private.ndim != 2 or public.shape[1] != private.shape[1]:
        raise ValueError("energy inputs must be aligned matrices")
    lhs = public.double()
    rhs = private.double()
    center = lhs.mean(dim=0)
    scale = lhs.std(dim=0, unbiased=False)
    active = scale > 1e-8
    if not bool(active.any()):
        return 0.0
    lhs = (lhs[:, active] - center[active]) / scale[active]
    rhs = (rhs[:, active] - center[active]) / scale[active]
    value = (
        2.0 * torch.cdist(lhs, rhs).mean()
        - torch.cdist(lhs, lhs).mean()
        - torch.cdist(rhs, rhs).mean()
    )
    return float(value.clamp_min(0).item())


def _patient_equal_categorical(
    values: Sequence[object], patient_ids: Sequence[str]
) -> dict[str, float]:
    if len(values) != len(patient_ids) or not values:
        raise ValueError("categorical event values must align with patient IDs")
    bags: dict[str, list[str]] = defaultdict(list)
    for value, patient_id in zip(values, patient_ids):
        bags[str(patient_id)].append(str(value))
    result: Counter[str] = Counter()
    for bag in bags.values():
        counts = Counter(bag)
        for category, count in counts.items():
            result[category] += count / len(bag)
    patients = len(bags)
    return {
        key: float(result[key] / patients)
        for key in sorted(result)
    }


def _acquisition_summary(
    events: Sequence[Mapping[str, object]],
    *,
    patient_ids: Sequence[str],
    private: bool,
) -> dict[str, object]:
    if len(events) != len(patient_ids):
        raise ValueError("acquisition events and patients are not aligned")
    if private:
        sfreq = [float(row["source_sfreq_hz"]) for row in events]
        reference_policy = [str(row["source_reference_policy"]) for row in events]
        output_reference = [str(row["output_reference"]) for row in events]
    else:
        sfreq = [float(row["edf_receipt"]["source_sfreq_hz"]) for row in events]
        reference_policy = [str(row["signal_receipt"]["reference_policy"]) for row in events]
        output_reference = [str(row["signal_receipt"]["output_reference"]) for row in events]
    return {
        "patient_equal_source_sfreq_mass": _patient_equal_categorical(
            [f"{value:g}" for value in sfreq], patient_ids
        ),
        "event_source_sfreq_count": {
            key: int(value)
            for key, value in sorted(Counter(f"{item:g}" for item in sfreq).items())
        },
        "patient_equal_reference_policy_mass": _patient_equal_categorical(
            reference_policy, patient_ids
        ),
        "patient_equal_output_reference_mass": _patient_equal_categorical(
            output_reference, patient_ids
        ),
    }


def run(
    *,
    public_fine_directory: Path,
    public_preflight_path: Path,
    public_v16_directory: Path,
    private_evidence_directory: Path,
) -> tuple[dict[str, object], dict[str, torch.Tensor], list[dict[str, object]]]:
    public_manifest = _load_json(public_fine_directory / "manifest.json")
    public_events_all = public_manifest.get("events")
    if not isinstance(public_events_all, list) or len(public_events_all) != 1_149:
        raise ValueError("public target-free fine-event roster changed")
    if tuple(public_manifest.get("feature_names", ())) != FINE_TEMPORAL_FEATURE_NAMES:
        raise ValueError("public target-free feature vocabulary changed")
    public_tensor_path = (
        public_fine_directory / str(public_manifest["tensor_file"])
    ).resolve(strict=True)
    public_all = load_file(str(public_tensor_path), device="cpu")["features"].float()
    if tuple(public_all.shape) != (1_149, 19, len(FINE_TEMPORAL_FEATURE_NAMES)):
        raise ValueError("public target-free feature tensor changed")

    v16_manifest = _load_json(public_v16_directory / "manifest.json")
    frozen_public_patients = tuple(str(value) for value in v16_manifest.get("patient_ids", ()))
    if len(frozen_public_patients) != 102:
        raise ValueError("frozen public v29 patient roster changed")
    frozen_set = set(frozen_public_patients)
    public_rows = [
        index
        for index, event in enumerate(public_events_all)
        if str(event["patient_id"]) in frozen_set
    ]
    if len(public_rows) != 1_145:
        raise ValueError("public frozen v29 event roster must contain 1,145 events")
    public_index = torch.tensor(public_rows, dtype=torch.long)
    public_features = public_all.index_select(0, public_index)
    public_events = [public_events_all[index] for index in public_rows]
    public_event_patients = [str(row["patient_id"]) for row in public_events]

    private_manifest = _load_json(private_evidence_directory / "manifest.json")
    private_events = private_manifest.get("events")
    if not isinstance(private_events, list) or len(private_events) != 88:
        raise ValueError("private target-blind event roster changed")
    if tuple(private_manifest.get("feature_names", ())) != FINE_TEMPORAL_FEATURE_NAMES:
        raise ValueError("private target-blind feature vocabulary changed")
    private_tensor_path = (
        private_evidence_directory / str(private_manifest["tensor_file"])
    ).resolve(strict=True)
    private_features = load_file(str(private_tensor_path), device="cpu")["fine_event"].float()
    if tuple(private_features.shape) != (88, 19, len(FINE_TEMPORAL_FEATURE_NAMES)):
        raise ValueError("private target-blind feature tensor changed")
    private_event_patients = [str(row["patient_id"]) for row in private_events]

    public_patient_ids, public_patient, public_counts = _patient_descriptors(
        public_features, public_event_patients
    )
    private_patient_ids, private_patient, private_counts = _patient_descriptors(
        private_features, private_event_patients
    )
    if len(public_patient_ids) != 102 or int(public_counts.sum()) != 1_145:
        raise RuntimeError("public patient-equal aggregation changed")
    if len(private_patient_ids) != 31 or int(private_counts.sum()) != 88:
        raise RuntimeError("private patient-equal aggregation changed")

    names = _descriptor_names()
    rows = _feature_shift_rows(
        public_patient,
        private_patient,
        descriptor_names=names,
    )
    absolute_smd = np.asarray(
        [abs(float(row["standardized_mean_difference_private_minus_public"])) for row in rows]
    )
    ranked = sorted(
        rows,
        key=lambda row: abs(float(row["standardized_mean_difference_private_minus_public"])),
        reverse=True,
    )

    preflight = _load_json(public_preflight_path)
    receipt = preflight.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("public preflight receipt is missing")
    public_preflight_events_all = receipt.get("events")
    if not isinstance(public_preflight_events_all, list):
        raise ValueError("public preflight events are missing")
    preflight_by_id = {
        str(row["event_id"]): row for row in public_preflight_events_all
    }
    public_acquisition_events = []
    for event in public_events:
        matched = preflight_by_id.get(str(event["event_id"]))
        if matched is None or str(matched["patient_id"]) != str(event["patient_id"]):
            raise ValueError("public fine event lacks aligned preflight receipt")
        public_acquisition_events.append(matched)

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_target_blind_patient_equal_public_private_shift_audit",
        "cohorts": {
            "public": {
                "role": "consumed_adaptive_development",
                "patients": len(public_patient_ids),
                "events": int(public_counts.sum()),
                "event_count_median": float(public_counts.float().median()),
                "event_count_range": [int(public_counts.min()), int(public_counts.max())],
            },
            "private": {
                "role": "post_open_target_blind_transport_roster",
                "patients": len(private_patient_ids),
                "events": int(private_counts.sum()),
                "event_count_median": float(private_counts.float().median()),
                "event_count_range": [int(private_counts.min()), int(private_counts.max())],
            },
        },
        "descriptor_contract": {
            "event_input": [19, len(FINE_TEMPORAL_FEATURE_NAMES)],
            "feature_names": list(FINE_TEMPORAL_FEATURE_NAMES),
            "channel_reductions": ["mean", "population_sd"],
            "patient_reduction": "equal_mean_over_all_successful_events",
            "descriptor_count": len(names),
        },
        "acquisition": {
            "public": _acquisition_summary(
                public_acquisition_events,
                patient_ids=public_event_patients,
                private=False,
            ),
            "private": _acquisition_summary(
                private_events,
                patient_ids=private_event_patients,
                private=True,
            ),
            "all_outputs_are_common_average_standard19_200Hz": True,
        },
        "shift_summary": {
            "multivariate_energy_after_public_standardization": _multivariate_energy(
                public_patient, private_patient
            ),
            "absolute_smd_median": float(np.median(absolute_smd)),
            "absolute_smd_q90": float(np.quantile(absolute_smd, 0.9)),
            "absolute_smd_maximum": float(absolute_smd.max()),
            "descriptor_count_abs_smd_ge_0_5": int((absolute_smd >= 0.5).sum()),
            "descriptor_count_abs_smd_ge_1_0": int((absolute_smd >= 1.0).sum()),
            "largest_absolute_smd_descriptors": ranked[:10],
        },
        "files": {
            "tensor_file": "patient_descriptors.safetensors",
            "feature_table": "feature_shift.csv",
        },
        "access_receipt": {
            "public_target_free_signal_evidence_loaded": True,
            "private_target_blind_signal_evidence_loaded": True,
            "public_SOZ_target_values_loaded": False,
            "private_significant_or_spread_reference_loaded": False,
            "model_or_foundation_training_performed": False,
            "model_loss_window_anchor_fusion_or_threshold_selected": False,
        },
        "interpretation_boundary": {
            "fine_descriptors_are_complete_raw_acquisition_characterization": False,
            "fine_descriptors_are_event_anchor_dependent": True,
            "shift_score_is_clinical_error_uncertainty": False,
            "shift_threshold_or_clinical_abstention_qualified": False,
            "private_is_fresh_external_validation": False,
            "allowed_claim": (
                "the frozen v29 transport traversed measurable target-blind "
                "acquisition and signal-evidence distribution differences"
            ),
        },
        "bootstrap": {
            "unit": "patient",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confirmatory_p_values": False,
        },
    }
    tensors = {
        "public.patient_descriptor": public_patient,
        "public.patient_event_count": public_counts,
        "private.patient_descriptor": private_patient,
        "private.patient_event_count": private_counts,
    }
    return result, tensors, rows


def publish(
    *,
    output: Path,
    result: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    rows: Sequence[Mapping[str, object]],
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        save_file(dict(tensors), str(staging / "patient_descriptors.safetensors"))
        with (staging / "feature_shift.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--public-fine", type=Path, default=DEFAULT_PUBLIC_FINE)
    parser.add_argument("--public-preflight", type=Path, default=DEFAULT_PUBLIC_PREFLIGHT)
    parser.add_argument("--public-v16", type=Path, default=DEFAULT_PUBLIC_V16)
    parser.add_argument("--private-evidence", type=Path, default=DEFAULT_PRIVATE_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, tensors, rows = run(
        public_fine_directory=args.public_fine,
        public_preflight_path=args.public_preflight,
        public_v16_directory=args.public_v16,
        private_evidence_directory=args.private_evidence,
    )
    output = publish(output=args.output, result=result, tensors=tensors, rows=rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "public_patients": result["cohorts"]["public"]["patients"],
                "private_patients": result["cohorts"]["private"]["patients"],
                "private_reference_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
