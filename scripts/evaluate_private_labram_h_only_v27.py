#!/usr/bin/env python3
"""Open private references once after v27 predictions are frozen and evaluate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_trustworthy_soz_candidate_v21 import (  # noqa: E402
    _private_metrics,
    _read_csv,
)
from scripts.run_labram_fine_temporal_nested_oof_v11 import _file_sha  # noqa: E402


PREDICTION_SCHEMA = "soz_private_target_blind_labram_h_only_prediction_v27"
OUTPUT_SCHEMA = "soz_private_external_evaluation_v27"
PAPER_POINT_ESTIMATE = 0.744
PRIVATE_TARGET = 0.75
DEFAULT_PREDICTION = ROOT / "outputs/labram_h_only_private_target_blind_v27_20260815"
DEFAULT_BUNDLE = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814"
DEFAULT_PUBLIC = ROOT / "outputs/labram_masked_variable_auxiliary_oof_v17_replay_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/labram_h_only_private_evaluation_v27_20260815"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def run(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    prediction_manifest = _json(args.prediction_directory / "manifest.json")
    if prediction_manifest.get("schema_version") != PREDICTION_SCHEMA or (
        prediction_manifest.get("status")
        != "completed_frozen_target_blind_private_inference"
    ):
        raise ValueError("v27 private predictions are not frozen")
    access = prediction_manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(name) is not False
        for name in (
            "private_target_values_loaded",
            "training_performed",
            "calibration_or_threshold_selection_performed",
            "llm_used_for_prediction_or_ranking",
        )
    ):
        raise ValueError("v27 prediction artifact is not target blind")
    tensor_path = args.prediction_directory / str(prediction_manifest["tensor_file"])
    if prediction_manifest.get("prediction_file_sha256") != _file_sha(tensor_path):
        raise ValueError("v27 prediction file changed before evaluation")
    tensors = load_file(str(tensor_path.resolve(strict=True)), device="cpu")
    probability = tensors.get("private_h_only_probability")
    events = prediction_manifest.get("events")
    if probability is None or not isinstance(events, list) or len(events) != len(probability):
        raise ValueError("v27 prediction/event roster is malformed")

    # This is the first operation in this execution chain that opens private
    # target values.  Nothing is fitted or selected after the ledger is read.
    target_rows = _read_csv(args.private_bundle / "target_ledger.csv")
    metrics, evaluation_rows = _private_metrics(probability, events, target_rows)
    relaxed = float(metrics["event_micro"]["relaxed_neighbor4"])
    strict = float(metrics["event_micro"]["exact"])
    relaxed_hits = sum(bool(row["relaxed_neighbor4"]) for row in evaluation_rows)

    public_manifest = _json(args.public_oof_directory / "manifest.json")
    public_metrics = public_manifest["primary_comparison"]["candidate_metrics"]
    payload: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "completed_single_frozen_private_external_evaluation",
        "endpoint_definition": {
            "strict_top1": "predicted physical C18 electrode is in clinician significant/SOZ-reference candidate set",
            "neighborhood4_top1": "DeepSOZ-compatible one-hop neighborhood acceptance for eligible positive sets",
            "neighborhood4_is_strict_accuracy": False,
            "unit": "private seizure event with patient-cluster uncertainty",
        },
        "public_patient_oof": {
            "patient_count": 102,
            "metrics": public_metrics,
            "relaxed_top1_exceeds_paper_point_estimate": (
                float(public_metrics["top1"]["relaxed_accuracy"]) > PAPER_POINT_ESTIMATE
            ),
            "development_reuse_warning": (
                "102 patients are a repeatedly used development benchmark; this is not fresh confirmatory superiority"
            ),
        },
        "private_external": {
            "metrics": metrics,
            "strict_hits": int(sum(bool(row["exact"]) for row in evaluation_rows)),
            "relaxed_neighbor4_hits": relaxed_hits,
            "denominator": len(evaluation_rows),
            "strict_top1": strict,
            "relaxed_neighbor4_top1": relaxed,
        },
        "goal_audit": {
            "paper_point_estimate": PAPER_POINT_ESTIMATE,
            "private_requested_floor": PRIVATE_TARGET,
            "private_relaxed_exceeds_paper_point_estimate": relaxed > PAPER_POINT_ESTIMATE,
            "private_relaxed_at_least_75_percent": relaxed >= PRIVATE_TARGET,
            "private_goal_met": relaxed > PAPER_POINT_ESTIMATE and relaxed >= PRIVATE_TARGET,
            "strict_top1_at_least_75_percent": strict >= PRIVATE_TARGET,
        },
        "access_receipt": {
            "predictions_frozen_before_target_open": True,
            "private_target_values_opened_for_evaluation": True,
            "private_used_for_training_transform_prior_calibration_threshold_or_model_selection": False,
            "post_result_parameter_scan_authorized": False,
            "llm_used_as_predictor_gold_or_fact_filler": False,
        },
        "claim_boundary": {
            "private_endpoint_is_post_open_exploratory_external_validation": True,
            "patient_count_is_small": 23,
            "event_micro_rows_are_independent": False,
            "patient_cluster_bootstrap_is_primary_uncertainty": True,
            "neighborhood4_is_a_relaxed_clinical_sensitivity": True,
            "cortical_soz_or_surgical_target_validated": False,
        },
    }
    return payload, evaluation_rows


def publish(
    output_directory: Path,
    payload: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> Path:
    target = output_directory.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (staging / "evaluation_rows.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--prediction-directory", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--private-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--public-oof-directory", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload, rows = run(args)
    output = publish(args.output_directory, payload, rows)
    private = payload["private_external"]
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "strict_top1": private["strict_top1"],
                "relaxed_neighbor4_top1": private["relaxed_neighbor4_top1"],
                "relaxed_hits": private["relaxed_neighbor4_hits"],
                "denominator": private["denominator"],
                "goal_met": payload["goal_audit"]["private_goal_met"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
