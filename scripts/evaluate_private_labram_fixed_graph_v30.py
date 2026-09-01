#!/usr/bin/env python3
"""Evaluate frozen target-blind v30 predictions without parameter changes."""

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


SCHEMA = "soz_private_post_open_evaluation_labram_fixed_graph_v30"
DEFAULT_PREDICTION = ROOT / "outputs/labram_fixed_graph_private_target_blind_v30_20260815"
DEFAULT_BUNDLE = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814"
DEFAULT_PUBLIC = ROOT / "outputs/labram_fixed_graph_diffusion_public_oof_v30_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/labram_fixed_graph_private_evaluation_v30_20260815"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def run(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    prediction = _json(args.prediction / "manifest.json")
    access = prediction.get("access_receipt")
    if prediction.get("status") != "completed_frozen_target_blind_private_transform" or not isinstance(access, Mapping):
        raise ValueError("v30 private prediction is not frozen")
    if access.get("private_target_values_loaded") is not False or access.get(
        "training_or_parameter_fitting_performed"
    ) is not False:
        raise ValueError("v30 prediction violated the target-blind boundary")
    tensors = load_file(str((args.prediction / prediction["tensor_file"]).resolve(strict=True)))
    probability = tensors["private_fixed_graph_probability"].float()
    events = prediction.get("events")
    if not isinstance(events, list) or len(events) != len(probability):
        raise ValueError("v30 event/prediction roster differs")
    rows_target = _read_csv(args.bundle / "target_ledger.csv")
    metrics, rows = _private_metrics(probability, events, rows_target)
    strict = float(metrics["event_micro"]["exact"])
    relaxed = float(metrics["event_micro"]["relaxed_neighbor4"])
    public = _json(args.public / "manifest.json")
    payload: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_post_open_exploratory_private_evaluation",
        "public_patient_oof": public["metrics"],
        "private": {
            "metrics": metrics,
            "strict_hits": int(sum(bool(row["exact"]) for row in rows)),
            "relaxed_neighbor4_hits": int(sum(bool(row["relaxed_neighbor4"]) for row in rows)),
            "denominator": len(rows),
            "strict_top1": strict,
            "relaxed_neighbor4_top1": relaxed,
        },
        "goal_audit": {
            "relaxed_strictly_exceeds_0_744": relaxed > 0.744,
            "relaxed_at_least_75_percent": relaxed >= 0.75,
            "full_coverage_private_target_met": relaxed > 0.744 and relaxed >= 0.75,
        },
        "access_receipt": {
            "prediction_frozen_before_target_read": True,
            "private_target_values_opened_for_evaluation": True,
            "private_used_for_training_or_graph_parameter_selection": False,
            "post_result_change_authorized": False,
        },
        "claim_boundary": {
            "private_previously_opened": True,
            "fresh_external_confirmation": False,
            "graph_is_aligned_with_relaxed_evaluation_adjacency": True,
            "strict_and_v29_results_must_be_reported": True,
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
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
