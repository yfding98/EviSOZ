#!/usr/bin/env python3
"""Evaluate the frozen raw25 TCN baseline after opening private reference."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_labram_v29_token_stress_v38 import _stability  # noqa: E402
from scripts.audit_private_frozen_publication_v36 import (  # noqa: E402
    BOOTSTRAP_SEED,
    _event_rows,
    _paired,
    _read_csv,
    _summary,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _paired_bootstrap,
)
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_raw25_tcn_baseline_audit_v54"
DEFAULT_BASELINE = ROOT / "outputs/trustworthy_soz_raw25_tcn_baseline_v54_20260816"
DEFAULT_PUBLIC_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE_V29 = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_TARGET = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_raw25_tcn_baseline_audit_v54_20260816"


def audit(
    *,
    baseline_directory: Path,
    public_v29_directory: Path,
    private_v29_directory: Path,
    target_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    baseline_manifest_path = (baseline_directory / "manifest.json").resolve(strict=True)
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    if baseline_manifest.get("status") != (
        "completed_fixed_public_oof_and_target_blind_private_raw25_tcn"
    ):
        raise ValueError("formal raw25 TCN baseline is missing")
    if baseline_manifest.get("access_receipt", {}).get(
        "private_significant_or_spread_reference_loaded"
    ) is not False:
        raise ValueError("raw25 TCN baseline opened private reference")
    baseline = load_file(
        str((baseline_directory / "raw25_tcn_predictions.safetensors").resolve(strict=True))
    )
    public_v29 = load_file(
        str((public_v29_directory / "oof_predictions.safetensors").resolve(strict=True))
    )
    private_v29_manifest = json.loads(
        (private_v29_directory / "manifest.json").resolve(strict=True).read_text()
    )
    private_v29 = load_file(
        str((private_v29_directory / "predictions.safetensors").resolve(strict=True))
    )
    targets = baseline["public.targets"].float()
    mask = baseline["public.target_mask"].bool()
    raw_logits = baseline["public.oof_logits"].float()
    v29_public_probability = public_v29[
        "oof.portable_equal_ensemble_probability"
    ].float()
    raw_public_metrics = _evaluate(raw_logits, targets, mask)
    v29_public_metrics = _evaluate(
        torch.log(v29_public_probability.clamp_min(1e-12)), targets, mask
    )

    events = private_v29_manifest["events"]
    raw_private_probability = baseline["private.probability"].float()
    v29_private_probability = private_v29[
        "private_portable_equal_probability"
    ].float()
    target_rows = _read_csv(target_path)
    raw_rows, raw_flow = _event_rows(
        scores=raw_private_probability, events=events, target_rows=target_rows
    )
    v29_rows, v29_flow = _event_rows(
        scores=v29_private_probability, events=events, target_rows=target_rows
    )
    if raw_flow != v29_flow:
        raise RuntimeError("raw25/v29 private evaluation rosters differ")
    raw_summary = _summary(raw_rows, seed=BOOTSTRAP_SEED + 5400)
    v29_summary = _summary(v29_rows, seed=BOOTSTRAP_SEED)
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_post_open_raw25_tcn_vs_v29_audit",
        "analysis_role": "fixed_low_capacity_raw_waveform_neural_baseline",
        "cohort_flow": raw_flow,
        "public": {
            "v29": v29_public_metrics,
            "raw25_tcn": raw_public_metrics,
            "paired_v29_minus_raw25_tcn": _paired_bootstrap(
                torch.log(v29_public_probability.clamp_min(1e-12)),
                raw_logits,
                targets,
                mask,
            ),
        },
        "private": {
            "v29": v29_summary,
            "raw25_tcn": raw_summary,
            "paired_v29_minus_raw25_tcn": _paired(
                v29_rows, raw_rows, seed=BOOTSTRAP_SEED + 54_000
            ),
            "ranking_stability_raw25_vs_v29_all_88": _stability(
                v29_private_probability,
                raw_private_probability,
                V11_CANDIDATE_MASK.unsqueeze(0).expand(88, -1),
            ),
        },
        "source_files": {
            "target_blind_baseline": str(baseline_manifest_path.relative_to(ROOT)),
            "opened_private_target_ledger": str(target_path.resolve().relative_to(ROOT)),
        },
        "access_receipt": {
            "private_reference_opened_only_after_88_baseline_predictions": True,
            "private_used_for_training_model_threshold_or_report_selection": False,
        },
        "interpretation_boundary": {
            "raw25_tcn_is_exact_canonical_EEGNet": False,
            "raw25_tcn_is_low_capacity_raw_waveform_neural_baseline": True,
            "25Hz_excludes_higher_frequency_information": True,
            "v29_improvement_proves_pretraining_is_clean": False,
            "private_is_fresh_validation": False,
        },
    }
    table = [
        {
            "model": "v29_frozen_LaBraM_H_D",
            "public_strict": v29_public_metrics["top1"]["strict_accuracy"],
            "public_neighborhood4": v29_public_metrics["top1"]["relaxed_accuracy"],
            "public_macro_ap": v29_public_metrics["ranking"]["macro_average_precision"],
            "private_strict_event_micro": v29_summary["event_micro"]["strict"],
            "private_neighborhood4_event_micro": v29_summary["event_micro"]["relaxed"],
            "private_strict_patient_equal": v29_summary["patient_equal_event_macro"]["strict"],
            "private_neighborhood4_patient_equal": v29_summary[
                "patient_equal_event_macro"
            ]["relaxed"],
        },
        {
            "model": "raw25_channel_TCN_1425_params",
            "public_strict": raw_public_metrics["top1"]["strict_accuracy"],
            "public_neighborhood4": raw_public_metrics["top1"]["relaxed_accuracy"],
            "public_macro_ap": raw_public_metrics["ranking"]["macro_average_precision"],
            "private_strict_event_micro": raw_summary["event_micro"]["strict"],
            "private_neighborhood4_event_micro": raw_summary["event_micro"]["relaxed"],
            "private_strict_patient_equal": raw_summary["patient_equal_event_macro"]["strict"],
            "private_neighborhood4_patient_equal": raw_summary[
                "patient_equal_event_macro"
            ]["relaxed"],
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
