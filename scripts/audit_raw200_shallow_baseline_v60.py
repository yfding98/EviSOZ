#!/usr/bin/env python3
"""Post-open audit of the fixed Raw200-Shallow comparator versus frozen v29."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_private_frozen_publication_v36 import (  # noqa: E402
    BOOTSTRAP_SEED,
    _event_rows,
    _read_csv,
    _summary,
)
from scripts.audit_v29_spatial_endpoint_sensitivity_v59 import _endpoint_row  # noqa: E402
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _paired_bootstrap,
)
from src.soz.metrics import deepsoz_style_top1_metrics  # noqa: E402


SCHEMA = "trustworthy_soz_raw200_shallow_baseline_audit_v60"
DEFAULT_BASELINE = ROOT / "outputs/trustworthy_soz_raw200_shallow_baseline_v60_20260816"
DEFAULT_PUBLIC_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE_V29 = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_TARGET = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_raw200_shallow_baseline_audit_v60_20260816"
BOOTSTRAP_REPLICATES = 10_000


def _attach_n2(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        endpoint = _endpoint_row(
            unit_id=str(row["unit_id"]),
            patient_id=str(row["patient_id"]),
            top1=str(row["top1"]),
            positive_channels=tuple(str(value) for value in row["positive_channels"]),
            spread_channels=tuple(str(value) for value in row["known_spread_channels"]),
            first_positive_rank=int(row["first_positive_rank"]),
        )
        value = dict(row)
        value["official_N2"] = float(endpoint["official_N2"])
        result.append(value)
    return result


def _patient_means(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, float]:
    bags: dict[str, list[float]] = {}
    for row in rows:
        bags.setdefault(str(row["patient_id"]), []).append(float(row[key]))
    return {patient: float(np.mean(values)) for patient, values in bags.items()}


def _n2_summary(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, object]:
    event = float(np.mean([float(row["official_N2"]) for row in rows]))
    patient = _patient_means(rows, "official_N2")
    values = np.asarray([patient[key] for key in sorted(patient)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    return {
        "event_micro": event,
        "patient_equal": float(values.mean()),
        "patient_cluster_bootstrap_ci95": [
            float(value)
            for value in np.quantile(values[sampled].mean(axis=1), (0.025, 0.975))
        ],
    }


def _paired_private_metric(
    proposed: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
    *,
    proposed_key: str,
    comparator_key: str,
    seed: int,
) -> dict[str, object]:
    left = _patient_means(proposed, proposed_key)
    right = _patient_means(comparator, comparator_key)
    if set(left) != set(right):
        raise ValueError("private paired patient rosters differ")
    patients = sorted(left)
    difference = np.asarray([left[key] - right[key] for key in patients], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(difference), size=(BOOTSTRAP_REPLICATES, len(difference))
    )
    return {
        "patient_equal_delta": float(difference.mean()),
        "patient_cluster_bootstrap_ci95": [
            float(value)
            for value in np.quantile(difference[sampled].mean(axis=1), (0.025, 0.975))
        ],
    }


def audit(
    *,
    baseline_directory: Path,
    public_v29_directory: Path,
    private_v29_directory: Path,
    target_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    baseline_manifest = json.loads(
        (baseline_directory / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    if baseline_manifest.get("status") != (
        "completed_post_open_fixed_raw200_public_OOF_private_comparator"
    ):
        raise ValueError("formal raw200 shallow comparator is missing")
    access = baseline_manifest.get("access_receipt", {})
    if access.get("private_significant_or_spread_reference_loaded") is not False:
        raise ValueError("raw200 comparator training path opened private reference")
    baseline = load_file(
        str(
            (baseline_directory / "raw200_shallow_predictions.safetensors").resolve(
                strict=True
            )
        )
    )
    public_v29 = load_file(
        str((public_v29_directory / "oof_predictions.safetensors").resolve(strict=True))
    )
    private_manifest = json.loads(
        (private_v29_directory / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    private_v29 = load_file(
        str((private_v29_directory / "predictions.safetensors").resolve(strict=True))
    )

    targets = baseline["public.targets"].float()
    mask = baseline["public.target_mask"].bool()
    raw_probability = baseline["public.oof_probability"].float()
    v29_probability = public_v29["oof.portable_equal_ensemble_probability"].float()
    raw_logits = torch.log(raw_probability.clamp_min(1e-12))
    v29_logits = torch.log(v29_probability.clamp_min(1e-12))
    raw_public = _evaluate(raw_logits, targets, mask)
    v29_public = _evaluate(v29_logits, targets, mask)
    raw_public_n2 = asdict(
        deepsoz_style_top1_metrics(
            raw_logits, targets, mask, max_positive_for_neighbor=2
        )
    )
    v29_public_n2 = asdict(
        deepsoz_style_top1_metrics(
            v29_logits, targets, mask, max_positive_for_neighbor=2
        )
    )

    events = private_manifest["events"]
    raw_private_probability = baseline["private.probability"].float()
    v29_private_probability = private_v29["private_portable_equal_probability"].float()
    target_rows = _read_csv(target_path)
    raw_rows_base, raw_flow = _event_rows(
        scores=raw_private_probability, events=events, target_rows=target_rows
    )
    v29_rows_base, v29_flow = _event_rows(
        scores=v29_private_probability, events=events, target_rows=target_rows
    )
    if raw_flow != v29_flow:
        raise RuntimeError("raw200/v29 private evaluation rosters differ")
    raw_rows = _attach_n2(raw_rows_base)
    v29_rows = _attach_n2(v29_rows_base)
    raw_private = _summary(raw_rows, seed=BOOTSTRAP_SEED + 60_000)
    v29_private = _summary(v29_rows, seed=BOOTSTRAP_SEED)
    raw_n2 = _n2_summary(raw_rows, seed=BOOTSTRAP_SEED + 60_100)
    v29_n2 = _n2_summary(v29_rows, seed=BOOTSTRAP_SEED + 100)

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_post_open_raw200_shallow_vs_frozen_v29_audit",
        "analysis_role": "full_bandwidth_event_bag_raw_waveform_comparator",
        "cohort_flow": raw_flow,
        "public": {
            "v29": {**v29_public, "official_N2": v29_public_n2},
            "raw200_shallow": {**raw_public, "official_N2": raw_public_n2},
            "paired_v29_minus_raw200": _paired_bootstrap(
                v29_logits, raw_logits, targets, mask
            ),
        },
        "private": {
            "v29": {**v29_private, "official_N2": v29_n2},
            "raw200_shallow": {**raw_private, "official_N2": raw_n2},
            "paired_v29_minus_raw200_patient_equal": {
                "strict": _paired_private_metric(
                    v29_rows,
                    raw_rows,
                    proposed_key="strict",
                    comparator_key="strict",
                    seed=BOOTSTRAP_SEED + 60_200,
                ),
                "official_N2": _paired_private_metric(
                    v29_rows,
                    raw_rows,
                    proposed_key="official_N2",
                    comparator_key="official_N2",
                    seed=BOOTSTRAP_SEED + 60_300,
                ),
                "official_N4": _paired_private_metric(
                    v29_rows,
                    raw_rows,
                    proposed_key="relaxed",
                    comparator_key="relaxed",
                    seed=BOOTSTRAP_SEED + 60_400,
                ),
            },
        },
        "access_receipt": {
            "private_reference_opened_only_in_this_read_only_audit": True,
            "comparator_materialization_and_training_reference_isolated": True,
            "experiment_began_after_private_reference_was_historically_open": True,
            "private_used_for_training_model_threshold_seed_or_report_selection": False,
        },
        "interpretation_boundary": {
            "fresh_or_target_blind_private_validation": False,
            "raw200_is_exact_canonical_EEGNet_or_ShallowConvNet": False,
            "one_raw_comparator_covers_all_nonfoundation_models": False,
            "N2_or_N4_is_strict_accuracy": False,
            "v29_may_be_reselected_from_this_audit": False,
        },
    }
    table = [
        {
            "model": "frozen_v29_LaBraM_H_D",
            "trainable_parameters": "H/D low-capacity heads",
            "public_strict": v29_public["top1"]["strict_accuracy"],
            "public_N2": v29_public_n2["relaxed_accuracy"],
            "public_N4": v29_public["top1"]["relaxed_accuracy"],
            "public_macro_ap": v29_public["ranking"]["macro_average_precision"],
            "private_strict_event_micro": v29_private["event_micro"]["strict"],
            "private_N2_event_micro": v29_n2["event_micro"],
            "private_N4_event_micro": v29_private["event_micro"]["relaxed"],
            "private_strict_patient_equal": v29_private["patient_equal_event_macro"]["strict"],
            "private_N2_patient_equal": v29_n2["patient_equal"],
            "private_N4_patient_equal": v29_private["patient_equal_event_macro"]["relaxed"],
        },
        {
            "model": "raw200_channel_shallow_3425_params",
            "trainable_parameters": 3_425,
            "public_strict": raw_public["top1"]["strict_accuracy"],
            "public_N2": raw_public_n2["relaxed_accuracy"],
            "public_N4": raw_public["top1"]["relaxed_accuracy"],
            "public_macro_ap": raw_public["ranking"]["macro_average_precision"],
            "private_strict_event_micro": raw_private["event_micro"]["strict"],
            "private_N2_event_micro": raw_n2["event_micro"],
            "private_N4_event_micro": raw_private["event_micro"]["relaxed"],
            "private_strict_patient_equal": raw_private["patient_equal_event_macro"]["strict"],
            "private_N2_patient_equal": raw_n2["patient_equal"],
            "private_N4_patient_equal": raw_private["patient_equal_event_macro"]["relaxed"],
        },
    ]
    return result, table


def publish(
    *, output: Path, result: Mapping[str, object], table: Sequence[Mapping[str, object]]
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
        with (staging / "model_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--public-v29", type=Path, default=DEFAULT_PUBLIC_V29)
    parser.add_argument("--private-v29", type=Path, default=DEFAULT_PRIVATE_V29)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, table = audit(
        baseline_directory=args.baseline,
        public_v29_directory=args.public_v29,
        private_v29_directory=args.private_v29,
        target_path=args.target,
    )
    output = publish(output=args.output, result=result, table=table)
    print(json.dumps({"output": str(output), "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
