#!/usr/bin/env python3
"""Evaluate frozen v29 predictions after their target-blind publication."""

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

from scripts.audit_trustworthy_soz_candidate_v21 import _private_metrics, _read_csv  # noqa: E402


SCHEMA = "soz_private_post_open_evaluation_labram_portable_equal_v29"
DEFAULT_PREDICTION = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_BUNDLE = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814"
DEFAULT_PUBLIC = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/labram_portable_equal_private_evaluation_v29_20260815"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def run(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    prediction = _json(args.prediction / "manifest.json")
    access = prediction.get("access_receipt")
    if prediction.get("status") != "completed_frozen_target_blind_private_inference" or not isinstance(access, Mapping):
        raise ValueError("v29 predictions are not frozen")
    if access.get("private_target_values_loaded") is not False or access.get(
        "training_calibration_or_model_selection_performed"
    ) is not False:
        raise ValueError("v29 prediction artifact is not target blind")
    tensors = load_file(str((args.prediction / prediction["tensor_file"]).resolve(strict=True)))
    probability = tensors["private_portable_equal_probability"].float()
    events = prediction.get("events")
    if not isinstance(events, list) or len(events) != len(probability):
        raise ValueError("v29 prediction/event roster differs")
    target_rows = _read_csv(args.bundle / "target_ledger.csv")
    metrics, rows = _private_metrics(probability, events, target_rows)
    strict = float(metrics["event_micro"]["exact"])
    relaxed = float(metrics["event_micro"]["relaxed_neighbor4"])
    public = _json(args.public / "manifest.json")
    payload: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_post_open_exploratory_private_evaluation",
        "public_patient_oof": public["metrics"]["portable_equal_ensemble"],
        "private": {
            "metrics": metrics,
            "strict_hits": int(sum(bool(row["exact"]) for row in rows)),
            "relaxed_neighbor4_hits": int(sum(bool(row["relaxed_neighbor4"]) for row in rows)),
            "denominator": len(rows),
            "strict_top1": strict,
            "relaxed_neighbor4_top1": relaxed,
        },
        "goal_audit": {
            "relaxed_strictly_exceeds_deepsoz_paper_point_0_744": relaxed > 0.744,
            "relaxed_at_least_75_percent": relaxed >= 0.75,
            "full_coverage_private_target_met": relaxed > 0.744 and relaxed >= 0.75,
        },
        "access_receipt": {
            "prediction_frozen_before_this_target_read": True,
            "private_target_values_opened_for_evaluation": True,
            "private_used_for_training_weight_threshold_fold_or_model_selection": False,
            "post_result_parameter_change_authorized": False,
        },
        "claim_boundary": {
            "private_has_been_opened_in_prior_project_iterations": True,
            "fresh_external_confirmation": False,
            "event_rows_are_independent": False,
            "neighborhood4_is_strict_accuracy": False,
            "output_is_cortical_soz_or_surgical_target": False,
        },
    }
    return payload, rows


def publish(output: Path, payload: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> Path:
    target = output.resolve()
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
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload, rows = run(args)
    output = publish(args.output, payload, rows)
    private = payload["private"]
    print(json.dumps({
        "output": str(output),
        "strict": private["strict_top1"],
        "relaxed": private["relaxed_neighbor4_top1"],
        "relaxed_hits": private["relaxed_neighbor4_hits"],
        "denominator": private["denominator"],
        "goal_met": payload["goal_audit"]["full_coverage_private_target_met"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
