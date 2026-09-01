#!/usr/bin/env python3
"""Audit frozen v29 target strata and public-to-private construct shift.

The audit uses the frozen public patient-OOF ranking and the already published
private post-open event audit.  It does not pool the two reference constructs,
fit a domain adapter, select a subgroup rule, or alter the model.  Public units
are patients.  Private headline distributions are patient-equal over events;
event-micro distributions are retained separately.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_labram_rank1_direct_token_oof_v28 as v28  # noqa: E402
from scripts.audit_private_frozen_publication_v36 import (  # noqa: E402
    _reference_laterality,
)
from scripts.audit_trustworthy_soz_ranking_distance_v22_6 import (  # noqa: E402
    rank_row,
)
from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402


SCHEMA = "trustworthy_soz_v29_public_private_construct_shift_v47"
DEFAULT_PUBLIC = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE = ROOT / "outputs/trustworthy_soz_private_frozen_publication_v36_20260816"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_public_private_construct_shift_v47_20260816"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260817
CANDIDATES = tuple(channel for channel in STANDARD_19 if channel != "PZ")
SCALP_CHAINS = {
    "left_temporal_chain": frozenset(("F7", "T7", "P7")),
    "right_temporal_chain": frozenset(("F8", "T8", "P8")),
    "left_parasagittal": frozenset(("F3", "C3", "P3")),
    "right_parasagittal": frozenset(("F4", "C4", "P4")),
    "midline": frozenset(("FZ", "CZ")),
    "frontopolar": frozenset(("FP1", "FP2")),
    "occipital": frozenset(("O1", "O2")),
}
METRICS = (
    "strict",
    "relaxed",
    "far",
    "contralateral_far",
    "hit_at_3",
    "hit_at_5",
    "laterality_agreement",
)


def _side(channel: str) -> str:
    if channel.endswith("1") or channel.endswith("3") or channel.endswith("7"):
        return "left"
    if channel.endswith("2") or channel.endswith("4") or channel.endswith("8"):
        return "right"
    return "midline"


def _chain(channel: str) -> str:
    matches = [name for name, members in SCALP_CHAINS.items() if channel in members]
    if len(matches) != 1:
        raise ValueError(f"channel {channel} does not have one scalp-chain assignment")
    return matches[0]


def _public_rows(public_directory: Path) -> tuple[list[dict[str, object]], object, Path]:
    loader_args = v28.build_parser().parse_args(["--device", "cpu"])
    stable = v28.v17._load_stable_development(loader_args)
    tensor_path = (public_directory / "oof_predictions.safetensors").resolve(
        strict=True
    )
    payload = load_file(str(tensor_path), device="cpu")
    scores = payload["oof.portable_equal_ensemble_probability"].float()
    if not torch.equal(payload["targets"].float(), stable.targets) or not torch.equal(
        payload["target_mask"].bool(), stable.target_mask
    ):
        raise ValueError("public v29 target identity changed")
    rows: list[dict[str, object]] = []
    for patient, patient_id in enumerate(stable.patient_ids):
        positives = set(
            torch.nonzero(
                stable.targets[patient].bool() & stable.target_mask[patient],
                as_tuple=False,
            ).flatten().tolist()
        )
        evaluable = set(
            torch.nonzero(stable.target_mask[patient], as_tuple=False).flatten().tolist()
        )
        row = rank_row(
            unit_id=f"PUBLIC-{patient:03d}",
            patient_id=str(patient_id),
            scores=scores[patient],
            positive_indices=positives,
            evaluable_indices=evaluable,
        )
        positive_channels = tuple(str(value) for value in row["positive_channels"])
        row.update(
            {
                "positive_set_size": len(positive_channels),
                "reference_laterality_stratum": _reference_laterality(
                    positive_channels
                ),
                "event_count": int(stable.event_counts[patient]),
                "dataset": "public_consumed_development",
            }
        )
        rows.append(row)
    return rows, stable, tensor_path


def _private_rows(private_directory: Path) -> tuple[list[dict[str, object]], Path]:
    path = (private_directory / "private_event_error_audit.csv").resolve(strict=True)
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for value in csv.DictReader(handle):
            positive_channels = tuple(ast.literal_eval(value["positive_channels"]))
            top1 = str(value["top1"])
            strict = float(value["strict"])
            neighbor = float(value["neighbor_only"])
            far = float(value["far"])
            top_side = _side(top1)
            positive_sides = {_side(channel) for channel in positive_channels}
            rows.append(
                {
                    "unit_id": str(value["unit_id"]),
                    "patient_id": str(value["patient_id"]),
                    "top1": top1,
                    "top1_side": top_side,
                    "positive_channels": list(positive_channels),
                    "positive_set_size": len(positive_channels),
                    "reference_laterality_stratum": str(
                        value["reference_laterality_stratum"]
                    ),
                    "strict": strict,
                    "neighbor_only": neighbor,
                    "relaxed": strict + neighbor,
                    "far": far,
                    "far_subtype": str(value["far_subtype"]),
                    "contralateral_far": float(
                        str(value["far_subtype"]) == "contralateral_far"
                    ),
                    "known_spread_top1": float(value["known_spread_top1"]),
                    "first_positive_rank": int(value["first_positive_rank"]),
                    "hit_at_3": float(int(value["first_positive_rank"]) <= 3),
                    "hit_at_5": float(int(value["first_positive_rank"]) <= 5),
                    "laterality_agreement": float(top_side in positive_sides),
                    "dataset": "private_post_open_transport",
                }
            )
    if len(rows) != 51 or len({row["patient_id"] for row in rows}) != 23:
        raise ValueError("private primary roster changed")
    return rows, path


def _patient_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    seed: int,
) -> dict[str, object]:
    by_patient: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_patient[str(row["patient_id"])].append(float(row[metric]))
    patient_means = np.asarray(
        [np.mean(by_patient[key]) for key in sorted(by_patient)], dtype=np.float64
    )
    if len(patient_means) < 2:
        return {
            "patient_count": len(patient_means),
            "patient_equal": float(patient_means.mean()),
            "patient_cluster_ci95": None,
        }
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(patient_means),
        size=(BOOTSTRAP_REPLICATES, len(patient_means)),
    )
    values = patient_means[sampled].mean(axis=1)
    return {
        "patient_count": len(patient_means),
        "patient_equal": float(patient_means.mean()),
        "patient_cluster_ci95": [
            float(value) for value in np.quantile(values, (0.025, 0.975))
        ],
    }


def _summary(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
) -> dict[str, object]:
    output: dict[str, object] = {
        "unit_count": len(rows),
        "patient_count": len({str(row["patient_id"]) for row in rows}),
    }
    for offset, metric in enumerate(METRICS):
        values = [float(row[metric]) for row in rows]
        output[metric] = {
            "unit_micro": float(np.mean(values)),
            **_patient_cluster_bootstrap(rows, metric=metric, seed=seed + offset),
        }
    ranks = np.asarray([int(row["first_positive_rank"]) for row in rows])
    output["first_positive_rank"] = {
        "median": float(np.median(ranks)),
        "mean": float(ranks.mean()),
        "above_5_count": int((ranks > 5).sum()),
    }
    return output


def _grouped(
    rows: Sequence[Mapping[str, object]],
    *,
    key: Callable[[Mapping[str, object]], str],
    seed: int,
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return {
        name: _summary(values, seed=seed + 100 * index)
        for index, (name, values) in enumerate(sorted(grouped.items()))
    }


def _overlapping_chain_strata(
    rows: Sequence[Mapping[str, object]], *, seed: int
) -> dict[str, object]:
    result: dict[str, object] = {}
    for index, (name, channels) in enumerate(SCALP_CHAINS.items()):
        selected = [
            row for row in rows if set(row["positive_channels"]) & channels
        ]
        result[name] = _summary(selected, seed=seed + 100 * index)
    return result


def _unit_weights(
    rows: Sequence[Mapping[str, object]], *, patient_equal: bool
) -> np.ndarray:
    if not patient_equal:
        return np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
    counts = Counter(str(row["patient_id"]) for row in rows)
    patients = len(counts)
    return np.asarray(
        [1.0 / (patients * counts[str(row["patient_id"])]) for row in rows],
        dtype=np.float64,
    )


def _channel_distributions(
    rows: Sequence[Mapping[str, object]], *, patient_equal: bool
) -> dict[str, list[float]]:
    weights = _unit_weights(rows, patient_equal=patient_equal)
    reference = np.zeros(len(CANDIDATES), dtype=np.float64)
    prediction = np.zeros(len(CANDIDATES), dtype=np.float64)
    index = {channel: position for position, channel in enumerate(CANDIDATES)}
    for weight, row in zip(weights, rows):
        positive = [str(value) for value in row["positive_channels"]]
        for channel in positive:
            reference[index[channel]] += weight / len(positive)
        prediction[index[str(row["top1"])]] += weight
    if not np.isclose(reference.sum(), 1.0) or not np.isclose(prediction.sum(), 1.0):
        raise RuntimeError("channel distribution did not normalize")
    return {
        "reference_positive_mass": reference.tolist(),
        "predicted_top1_mass": prediction.tolist(),
    }


def _categorical_distribution(
    rows: Sequence[Mapping[str, object]],
    *,
    key: Callable[[Mapping[str, object]], str],
    categories: Sequence[str],
    patient_equal: bool,
) -> list[float]:
    weights = _unit_weights(rows, patient_equal=patient_equal)
    index = {value: position for position, value in enumerate(categories)}
    result = np.zeros(len(categories), dtype=np.float64)
    for weight, row in zip(weights, rows):
        result[index[key(row)]] += weight
    return result.tolist()


def _js_and_tv(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if not np.isclose(lhs.sum(), 1.0) or not np.isclose(rhs.sum(), 1.0):
        raise ValueError("divergence inputs must be distributions")
    midpoint = 0.5 * (lhs + rhs)

    def kl(value: np.ndarray) -> float:
        mask = value > 0
        return float(np.sum(value[mask] * np.log(value[mask] / midpoint[mask])))

    return {
        "jensen_shannon_nats": 0.5 * kl(lhs) + 0.5 * kl(rhs),
        "total_variation": float(0.5 * np.abs(lhs - rhs).sum()),
    }


def run(
    *, public_directory: Path, private_directory: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    public_rows, stable, public_path = _public_rows(public_directory)
    private_rows, private_path = _private_rows(private_directory)

    laterality_categories = (
        "left_only",
        "right_only",
        "midline_only",
        "bilateral_or_mixed",
    )
    size_categories = ("1-2", "3-4", ">=5")
    size_key = lambda row: (
        "1-2"
        if int(row["positive_set_size"]) <= 2
        else "3-4"
        if int(row["positive_set_size"]) <= 4
        else ">=5"
    )
    event_key = lambda row: (
        "1"
        if int(row["event_count"]) == 1
        else "2"
        if int(row["event_count"]) == 2
        else "3-5"
        if int(row["event_count"]) <= 5
        else ">=6"
    )

    public_channel = _channel_distributions(public_rows, patient_equal=True)
    private_channel_patient = _channel_distributions(private_rows, patient_equal=True)
    private_channel_event = _channel_distributions(private_rows, patient_equal=False)
    public_laterality = _categorical_distribution(
        public_rows,
        key=lambda row: str(row["reference_laterality_stratum"]),
        categories=laterality_categories,
        patient_equal=True,
    )
    private_laterality = _categorical_distribution(
        private_rows,
        key=lambda row: str(row["reference_laterality_stratum"]),
        categories=laterality_categories,
        patient_equal=True,
    )
    public_size = _categorical_distribution(
        public_rows,
        key=size_key,
        categories=size_categories,
        patient_equal=True,
    )
    private_size = _categorical_distribution(
        private_rows,
        key=size_key,
        categories=size_categories,
        patient_equal=True,
    )

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_public_private_construct_shift_audit",
        "analysis_role": {
            "public": "posthoc_consumed_development_strata",
            "private": "post_open_cross_construct_transport_strata",
        },
        "definitions": {
            "public_unit": "patient",
            "private_unit": "event_with_patient_clustered_and_patient_equal_summaries",
            "scalp_chains": {key: sorted(value) for key, value in SCALP_CHAINS.items()},
            "scalp_chain_is_cortical_lobe": False,
            "private_unknown_complement_is_negative": False,
        },
        "public": {
            "overall": _summary(public_rows, seed=BOOTSTRAP_SEED),
            "reference_laterality": _grouped(
                public_rows,
                key=lambda row: str(row["reference_laterality_stratum"]),
                seed=BOOTSTRAP_SEED + 10_000,
            ),
            "positive_set_size": _grouped(
                public_rows, key=size_key, seed=BOOTSTRAP_SEED + 20_000
            ),
            "event_count": _grouped(
                public_rows, key=event_key, seed=BOOTSTRAP_SEED + 30_000
            ),
            "reference_chain_presence_overlapping": _overlapping_chain_strata(
                public_rows, seed=BOOTSTRAP_SEED + 40_000
            ),
        },
        "private": {
            "overall": _summary(private_rows, seed=BOOTSTRAP_SEED + 50_000),
            "reference_laterality": _grouped(
                private_rows,
                key=lambda row: str(row["reference_laterality_stratum"]),
                seed=BOOTSTRAP_SEED + 60_000,
            ),
            "positive_set_size": _grouped(
                private_rows, key=size_key, seed=BOOTSTRAP_SEED + 70_000
            ),
            "reference_chain_presence_overlapping": _overlapping_chain_strata(
                private_rows, seed=BOOTSTRAP_SEED + 80_000
            ),
            "event_micro_channel_distributions": private_channel_event,
        },
        "cross_construct_shift": {
            "comparison_weighting": "public_patient_equal_vs_private_patient_equal_events",
            "channel_order": list(CANDIDATES),
            "public_channel_distributions": public_channel,
            "private_channel_distributions": private_channel_patient,
            "reference_channel_mass_divergence": _js_and_tv(
                public_channel["reference_positive_mass"],
                private_channel_patient["reference_positive_mass"],
            ),
            "predicted_top1_mass_divergence": _js_and_tv(
                public_channel["predicted_top1_mass"],
                private_channel_patient["predicted_top1_mass"],
            ),
            "laterality_categories": list(laterality_categories),
            "public_reference_laterality": public_laterality,
            "private_reference_laterality": private_laterality,
            "reference_laterality_divergence": _js_and_tv(
                public_laterality, private_laterality
            ),
            "positive_set_size_categories": list(size_categories),
            "public_positive_set_size": public_size,
            "private_positive_set_size": private_size,
            "positive_set_size_divergence": _js_and_tv(public_size, private_size),
        },
        "source_files": {
            "public_prediction": str(public_path.relative_to(ROOT)),
            "private_event_audit": str(private_path.relative_to(ROOT)),
        },
        "access_receipt": {
            "raw_EEG_loaded": False,
            "model_training_or_adaptation_performed": False,
            "subgroup_rule_or_threshold_selected": False,
            "public_targets_loaded_for_frozen_strata": True,
            "opened_private_targets_loaded_for_descriptive_strata": True,
            "public_private_rows_pooled_for_training_or_single_accuracy": False,
        },
        "interpretation_boundary": {
            "subgroup_results_confirmatory": False,
            "private_is_fresh_external_validation": False,
            "distribution_divergence_is_performance_degradation_cause": False,
            "scalp_chain_is_cortical_SOZ_region": False,
            "allowed_claim": (
                "v29 performance and error types are reported across prespecified "
                "reference strata, while public/private construct shifts remain explicit"
            ),
        },
    }

    flat_rows: list[dict[str, object]] = []
    for dataset_name, rows in (("public", public_rows), ("private", private_rows)):
        for stratum_name, key_fn in (
            ("reference_laterality", lambda row: str(row["reference_laterality_stratum"])),
            ("positive_set_size", size_key),
        ):
            grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
            for row in rows:
                grouped[key_fn(row)].append(row)
            for stratum_value, values in sorted(grouped.items()):
                summary = _summary(values, seed=BOOTSTRAP_SEED)
                flat_rows.append(
                    {
                        "dataset": dataset_name,
                        "stratum": stratum_name,
                        "value": stratum_value,
                        "units": summary["unit_count"],
                        "patients": summary["patient_count"],
                        "strict_unit_micro": summary["strict"]["unit_micro"],
                        "strict_patient_equal": summary["strict"]["patient_equal"],
                        "neighborhood4_unit_micro": summary["relaxed"]["unit_micro"],
                        "neighborhood4_patient_equal": summary["relaxed"]["patient_equal"],
                        "contralateral_far_unit_micro": summary["contralateral_far"]["unit_micro"],
                        "laterality_unit_micro": summary["laterality_agreement"]["unit_micro"],
                    }
                )
    return result, flat_rows


def publish(
    *, output: Path, result: Mapping[str, object], rows: Sequence[Mapping[str, object]]
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
        with (staging / "primary_strata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
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
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, rows = run(
        public_directory=args.public, private_directory=args.private
    )
    output = publish(output=args.output, result=result, rows=rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "public_patients": result["public"]["overall"]["patient_count"],
                "private_events": result["private"]["overall"]["unit_count"],
                "training_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
