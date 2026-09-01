#!/usr/bin/env python3
"""Audit unified-C18 CPBF results and the historical private CPBF lineage."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

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
from scripts.audit_raw200_shallow_baseline_v60 import (  # noqa: E402
    _attach_n2,
    _n2_summary,
    _paired_private_metric,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _paired_bootstrap,
)
from src.soz.metrics import deepsoz_style_top1_metrics  # noqa: E402


SCHEMA = "raw200_cpbf_unified_c18_benchmark_audit_v80"
DEFAULT_RUN = ROOT / "outputs/raw200_cpbf_unified_c18_benchmark_v80r1_20260817"
DEFAULT_DEEPSOZ = ROOT / "outputs/deepsoz_official_local_oof_full.json"
DEFAULT_LOCAL = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_20260812/manifest.json"
DEFAULT_PUBLIC_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE_V29 = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_TARGET = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
DEFAULT_OUTPUT = ROOT / "outputs/raw200_cpbf_unified_c18_benchmark_audit_v80r2_20260817"
HISTORICAL_CPBF = {
    seed: ROOT
    / f"outputs/tfm_soz/private_0622_fix_rows119_cpbf_v2_full_compact_pre_seed{seed}/lopo_test_summary.json"
    for seed in (2028, 2029, 2030)
}


def _public_result(
    probability: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> dict[str, object]:
    logits = torch.log(probability.clamp_min(1e-12))
    return {
        **_evaluate(logits, targets, mask),
        "official_N2": asdict(
            deepsoz_style_top1_metrics(
                logits, targets, mask, max_positive_for_neighbor=2
            )
        ),
    }


def _official_deepsoz_probability(
    *, deepsoz_path: Path, local_path: Path
) -> torch.Tensor:
    artifact = json.loads(deepsoz_path.resolve(strict=True).read_text(encoding="utf-8"))
    if artifact.get("status") != "full" or int(artifact.get("patient_count", 0)) != 102:
        raise ValueError("Full official DeepSOZ local replay is missing")
    local = json.loads(local_path.resolve(strict=True).read_text(encoding="utf-8"))
    patients = [str(value).lstrip("0") or "0" for value in local["patient_ids"]]
    rows = {
        str(row["patient_id"]).lstrip("0") or "0": row
        for row in artifact["held_out_ensemble_predictions"]
    }
    if set(rows) != set(patients):
        raise ValueError("DeepSOZ and unified benchmark patient rosters differ")
    score = torch.tensor([rows[patient]["score"] for patient in patients]).float()
    if tuple(score.shape) != (102, 19) or not torch.isfinite(score).all():
        raise ValueError("DeepSOZ score tensor is invalid")
    return score / score.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _historical_cpbf() -> dict[str, object]:
    result: dict[str, object] = {}
    for seed, path in HISTORICAL_CPBF.items():
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
        weighted = payload["event_weighted_metrics"]
        result[str(seed)] = {
            "events": 119,
            "patients": 43,
            "private_supervised_LOPO": True,
            "target": "soz_bipolar",
            "channel_bipolar_any_hit_top1": float(weighted["channel_onset_top1_hit"]),
            "five_region_any_hit_top1": float(weighted["fused_region_onset_top1_hit"]),
            "patient_five_region_any_hit_top1": float(
                weighted["patient_fused_region_top1_hit"]
            ),
        }
    return result


def audit(
    *,
    run_directory: Path,
    deepsoz_path: Path,
    local_path: Path,
    public_v29_directory: Path,
    private_v29_directory: Path,
    target_path: Path,
) -> dict[str, object]:
    manifest = json.loads(
        (run_directory / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    if manifest.get("status") != "completed_posthoc_public_oof_private_reference_isolated_inference":
        raise ValueError("Completed Raw200 CPBF unified run is missing")
    access = manifest.get("access_receipt", {})
    if access.get("private_significant_or_spread_reference_loaded") is not False:
        raise ValueError("Raw200 CPBF training opened private reference")
    tensors = load_file(
        str((run_directory / "predictions.safetensors").resolve(strict=True)),
        device="cpu",
    )
    targets = tensors["public.targets"].float()
    mask = tensors["public.target_mask"].bool()
    public_probability = {
        "raw200_base": tensors["baseline.public.oof_probability"].float(),
        "head_only": tensors["head_only.public.oof_probability"].float(),
        "cpbf_graph": tensors["cpbf.public.oof_probability"].float(),
        "deepsoz_official": _official_deepsoz_probability(
            deepsoz_path=deepsoz_path, local_path=local_path
        ),
    }
    v29 = load_file(
        str((public_v29_directory / "oof_predictions.safetensors").resolve(strict=True)),
        device="cpu",
    )
    public_probability["labram_v29"] = v29[
        "oof.portable_equal_ensemble_probability"
    ].float()
    for name, value in public_probability.items():
        if tuple(value.shape) != (102, 19) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid public probability for {name}")
    public = {
        name: _public_result(value, targets, mask)
        for name, value in public_probability.items()
    }
    cpbf_logits = torch.log(public_probability["cpbf_graph"].clamp_min(1e-12))
    public_paired = {
        f"cpbf_minus_{name}": _paired_bootstrap(
            cpbf_logits,
            torch.log(value.clamp_min(1e-12)),
            targets,
            mask,
        )
        for name, value in public_probability.items()
        if name != "cpbf_graph"
    }

    private_manifest = json.loads(
        (private_v29_directory / "manifest.json").resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    private_v29 = load_file(
        str((private_v29_directory / "predictions.safetensors").resolve(strict=True)),
        device="cpu",
    )
    private_probability = {
        "raw200_base": tensors["baseline.private.probability"].float(),
        "head_only": tensors["head_only.private.probability"].float(),
        "cpbf_graph": tensors["cpbf.private.probability"].float(),
        "labram_v29": private_v29["private_portable_equal_probability"].float(),
    }
    events = manifest["private"]["events"]
    event_keys = [
        (str(row.get("event_id", "")), str(row.get("patient_id", "")))
        for row in events
    ]
    v29_event_keys = [
        (str(row.get("event_id", "")), str(row.get("patient_id", "")))
        for row in private_manifest["events"]
    ]
    if (
        event_keys != v29_event_keys
        or len(event_keys) != 88
        or len(set(event_keys)) != len(event_keys)
    ):
        raise ValueError("CPBF and v29 private event rosters differ")
    target_rows = _read_csv(target_path)
    private_rows: dict[str, list[dict[str, Any]]] = {}
    flow = None
    for name, probability in private_probability.items():
        rows_base, observed_flow = _event_rows(
            scores=probability, events=events, target_rows=target_rows
        )
        if flow is None:
            flow = observed_flow
        elif observed_flow != flow:
            raise ValueError("Private evaluation flow differs between models")
        private_rows[name] = _attach_n2(rows_base)
    private = {
        name: {
            **_summary(rows, seed=BOOTSTRAP_SEED + 80_000 + index * 100),
            "official_N2": _n2_summary(
                rows, seed=BOOTSTRAP_SEED + 80_050 + index * 100
            ),
        }
        for index, (name, rows) in enumerate(private_rows.items())
    }
    private_paired = {}
    for index, name in enumerate(("raw200_base", "head_only", "labram_v29")):
        comparator = private_rows[name]
        proposed = private_rows["cpbf_graph"]
        private_paired[f"cpbf_minus_{name}"] = {
            "strict": _paired_private_metric(
                proposed,
                comparator,
                proposed_key="strict",
                comparator_key="strict",
                seed=BOOTSTRAP_SEED + 81_000 + index * 100,
            ),
            "official_N2": _paired_private_metric(
                proposed,
                comparator,
                proposed_key="official_N2",
                comparator_key="official_N2",
                seed=BOOTSTRAP_SEED + 81_010 + index * 100,
            ),
            "official_N4": _paired_private_metric(
                proposed,
                comparator,
                proposed_key="relaxed",
                comparator_key="relaxed",
                seed=BOOTSTRAP_SEED + 81_020 + index * 100,
            ),
        }

    candidate_mask = tensors["candidate_mask"].bool().view(1, -1)
    public_head = public_probability["head_only"]
    public_cpbf = public_probability["cpbf_graph"]
    private_head = private_probability["head_only"]
    private_cpbf = private_probability["cpbf_graph"]

    def prediction_change(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
        left_masked = left.masked_fill(~candidate_mask, -1.0)
        right_masked = right.masked_fill(~candidate_mask, -1.0)
        return {
            "n": len(left),
            "max_abs_probability_difference": float((left - right).abs().max()),
            "top1_agreement": float(
                (left_masked.argmax(dim=1) == right_masked.argmax(dim=1))
                .float()
                .mean()
            ),
            "complete_rank_agreement": float(
                (
                    torch.argsort(left_masked, dim=1, descending=True)
                    == torch.argsort(right_masked, dim=1, descending=True)
                )
                .all(dim=1)
                .float()
                .mean()
            ),
        }

    fold_rows = manifest["public"]["folds"]
    model_layer_diagnostics = {
        "head_only_trainable_parameters": sorted(
            {int(row["variants"]["head_only"]["trainable_parameters"]) for row in fold_rows}
        ),
        "cpbf_trainable_parameters": sorted(
            {int(row["variants"]["cpbf"]["trainable_parameters"]) for row in fold_rows}
        ),
        "cpbf_residual_scales": [
            float(row["variants"]["cpbf"]["cpbf_residual_scale"])
            for row in fold_rows
        ],
        "initial_cpbf_vs_head_max_abs_logits": [
            float(row["initial_cpbf_vs_head_max_abs_logit"]) for row in fold_rows
        ],
        "post_refinement_cpbf_vs_head": {
            "public": prediction_change(public_head, public_cpbf),
            "private": prediction_change(private_head, private_cpbf),
        },
    }

    return {
        "schema_version": SCHEMA,
        "status": "pass",
        "analysis_role": "posthoc_model_layer_audit_not_confirmatory_selection",
        "public": {
            "cohort": "102 patient C18 complete-case development benchmark",
            "models": public,
            "paired_cpbf_differences": public_paired,
        },
        "private": {
            "cohort_flow": flow,
            "models": private,
            "paired_cpbf_differences_patient_equal": private_paired,
        },
        "historical_private_cpbf": {
            "results": _historical_cpbf(),
            "comparable_to_current_C18_significant_endpoint": False,
            "reason": (
                "private-supervised 43-patient LOPO with soz_bipolar and five-region/"
                "bipolar-channel any-hit endpoints"
            ),
        },
        "model_layer_diagnostics": model_layer_diagnostics,
        "access_receipt": {
            "private_reference_opened_only_after_all_CPBF_predictions_were_saved": True,
            "private_used_for_training_model_graph_seed_epoch_or_threshold_selection": False,
            "public_and_private_historically_consumed": True,
        },
        "interpretation_boundary": {
            "cpbf_graph_is_complete_historical_TFM_CPBF": False,
            "adapter_result_is_confirmatory_or_SOTA": False,
            "DeepSOZ_replay_is_exact_original_signal_snapshot": False,
            "N2_or_N4_is_strict_accuracy": False,
        },
    }


def publish(*, output: Path, result: Mapping[str, object]) -> Path:
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
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--deepsoz", type=Path, default=DEFAULT_DEEPSOZ)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--public-v29", type=Path, default=DEFAULT_PUBLIC_V29)
    parser.add_argument("--private-v29", type=Path, default=DEFAULT_PRIVATE_V29)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit(
        run_directory=args.run,
        deepsoz_path=args.deepsoz,
        local_path=args.local,
        public_v29_directory=args.public_v29,
        private_v29_directory=args.private_v29,
        target_path=args.target,
    )
    output = publish(output=args.output, result=result)
    print(json.dumps({"output": str(output), "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
