#!/usr/bin/env python3
"""Compare frozen v29 with the existing non-foundation fine-feature baseline.

The baseline uses the same patient folds, C18 mask, fold-local spatial prior,
patient-level positive-set objective and low-capacity shared channel reasoner as
the identity-v16 study, but its only signal input is the frozen 20-dimensional
target-blind temporal/spectral/quality descriptor vector.  It receives no
LaBraM or other foundation representation.

No model is trained or selected here.  Public evidence is consumed development
and private evidence is post-open frozen transfer.
"""

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
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_labram_rank1_direct_token_oof_v28 as v28  # noqa: E402
from scripts.audit_labram_v29_token_stress_v38 import _probability_logits  # noqa: E402
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
from src.soz.fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_v29_nonfoundation_fine_baseline_v50"
DEFAULT_V16 = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
)
DEFAULT_V29_PUBLIC = (
    ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
)
DEFAULT_PRIVATE_EVIDENCE = (
    ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
)
DEFAULT_PRIVATE_V29 = (
    ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
)
DEFAULT_PRIVATE_TARGET = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_v29_nonfoundation_fine_baseline_v50_20260816"
)


def _fine_fold_probability(
    fine: torch.Tensor,
    states: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for fold in range(5):
        prefix = f"outer{fold}."
        arm = prefix + "fine_change_only."
        if not torch.equal(states[arm + "candidate_mask"], V11_CANDIDATE_MASK):
            raise ValueError("fine-only fold candidate mask drifted")
        transformed = (
            fine - states[prefix + "transform.fine_center"]
        ) / states[prefix + "transform.fine_scale"]
        logits = states[arm + "prior_logits"].expand(len(fine), -1).clone()
        logits += torch.einsum(
            "ecd,d->ec", transformed, states[arm + "fine_weight"]
        )
        probability = torch.softmax(
            logits.masked_fill(~V11_CANDIDATE_MASK, -torch.inf), dim=1
        )
        if not torch.isfinite(probability).all():
            raise RuntimeError("fine-only private probability is non-finite")
        rows.append(probability)
    return torch.stack(rows, dim=1).contiguous()


def _public_row(name: str, metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        "dataset": "public_consumed_development",
        "arm": name,
        "units": 102,
        "clusters": 102,
        "strict": metrics["top1"]["strict_accuracy"],
        "neighborhood4": metrics["top1"]["relaxed_accuracy"],
        "macro_average_precision": metrics["ranking"]["macro_average_precision"],
        "hit_at_3": metrics["ranking"]["hit_at_k"][3],
        "hit_at_5": metrics["ranking"]["hit_at_k"][5],
        "far_count": metrics["far_error_count"],
        "contralateral_far_count": "",
    }


def _private_row(name: str, summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "dataset": "private_post_open_transport",
        "arm": name,
        "units": summary["event_count"],
        "clusters": summary["patient_count"],
        "strict": summary["event_micro"]["strict"],
        "neighborhood4": summary["event_micro"]["relaxed"],
        "macro_average_precision": summary["event_micro"]["average_precision"],
        "hit_at_3": summary["event_micro"]["hit_at_3"],
        "hit_at_5": summary["event_micro"]["hit_at_5"],
        "far_count": summary["endpoint_counts"]["far"],
        "contralateral_far_count": summary["endpoint_counts"][
            "contralateral_far"
        ],
    }


def run(
    *,
    v16_directory: Path,
    public_v29_directory: Path,
    private_evidence_directory: Path,
    private_v29_directory: Path,
    private_target_path: Path,
) -> tuple[dict[str, object], dict[str, torch.Tensor], list[dict[str, object]]]:
    loader_args = v28.build_parser().parse_args(["--device", "cpu"])
    stable = v28.v17._load_stable_development(loader_args)
    v16_prediction_path = (
        v16_directory / "oof_predictions.safetensors"
    ).resolve(strict=True)
    v16_state_path = (
        v16_directory / "outer_fold_states.safetensors"
    ).resolve(strict=True)
    public_v29_path = (
        public_v29_directory / "oof_predictions.safetensors"
    ).resolve(strict=True)
    v16 = load_file(str(v16_prediction_path), device="cpu")
    public_v29 = load_file(str(public_v29_path), device="cpu")
    if not torch.equal(v16["targets"].float(), stable.targets) or not torch.equal(
        v16["target_mask"].bool(), stable.target_mask
    ):
        raise ValueError("v16 fine-only target identity differs")
    if not torch.equal(public_v29["targets"].float(), stable.targets):
        raise ValueError("v29 target identity differs")
    fine_logits = v16["oof.fine_change_only"].float()
    v29_probability = public_v29[
        "oof.portable_equal_ensemble_probability"
    ].float()
    v29_logits = _probability_logits(v29_probability, stable.target_mask)
    public_fine_metrics = _evaluate(
        fine_logits, stable.targets, stable.target_mask
    )
    public_v29_metrics = _evaluate(
        v29_logits, stable.targets, stable.target_mask
    )
    public_paired = _paired_bootstrap(
        v29_logits, fine_logits, stable.targets, stable.target_mask
    )

    evidence_manifest_path = (
        private_evidence_directory / "manifest.json"
    ).resolve(strict=True)
    evidence_manifest = json.loads(
        evidence_manifest_path.read_text(encoding="utf-8")
    )
    evidence_tensor_path = (
        private_evidence_directory / str(evidence_manifest["tensor_file"])
    ).resolve(strict=True)
    private_events = evidence_manifest.get("events")
    if not isinstance(private_events, list) or len(private_events) != 88:
        raise ValueError("private evidence roster changed")
    evidence = load_file(str(evidence_tensor_path), device="cpu")
    fine_event = evidence["fine_event"].float()
    if tuple(fine_event.shape) != (88, 19, len(FINE_TEMPORAL_FEATURE_NAMES)):
        raise ValueError("private fine-feature shape changed")
    states = load_file(str(v16_state_path), device="cpu")
    private_fine_fold = _fine_fold_probability(fine_event, states)
    private_fine_probability = private_fine_fold.mean(dim=1)

    private_v29_manifest_path = (
        private_v29_directory / "manifest.json"
    ).resolve(strict=True)
    private_v29_tensor_path = (
        private_v29_directory / "predictions.safetensors"
    ).resolve(strict=True)
    private_v29_manifest = json.loads(
        private_v29_manifest_path.read_text(encoding="utf-8")
    )
    if [str(row["event_id"]) for row in private_events] != [
        str(row["event_id"]) for row in private_v29_manifest.get("events", ())
    ]:
        raise ValueError("private fine/v29 event identity differs")
    private_v29 = load_file(str(private_v29_tensor_path), device="cpu")[
        "private_portable_equal_probability"
    ].float()
    target_rows = _read_csv(private_target_path)
    fine_event_rows, fine_flow = _event_rows(
        scores=private_fine_probability,
        events=private_events,
        target_rows=target_rows,
    )
    v29_event_rows, v29_flow = _event_rows(
        scores=private_v29,
        events=private_events,
        target_rows=target_rows,
    )
    if fine_flow != v29_flow:
        raise RuntimeError("private baseline changed the evaluation cohort")
    private_fine_summary = _summary(
        fine_event_rows, seed=BOOTSTRAP_SEED + 90_000
    )
    private_v29_summary = _summary(
        v29_event_rows, seed=BOOTSTRAP_SEED + 91_000
    )
    private_paired = _paired(
        v29_event_rows,
        fine_event_rows,
        seed=BOOTSTRAP_SEED + 92_000,
    )

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_v29_vs_nonfoundation_fine_baseline",
        "baseline": {
            "name": "fine_change_only",
            "foundation_representation_access": False,
            "input_shape": [19, len(FINE_TEMPORAL_FEATURE_NAMES)],
            "feature_names": list(FINE_TEMPORAL_FEATURE_NAMES),
            "signal_role": "target_blind_temporal_spectral_quality_descriptors",
            "candidate_mask": "C18_PZ_excluded",
            "patient_split": "same_frozen_five_outer_folds_as_v29_H",
            "loss": "same_patient_equal_positive_set_probability_mass_loss",
            "reasoner": "fold_local_shared_linear_channel_head_plus_spatial_prior",
        },
        "public": {
            "patient_count": 102,
            "v29_metrics": public_v29_metrics,
            "fine_only_metrics": public_fine_metrics,
            "paired_v29_minus_fine_only": public_paired,
        },
        "private": {
            "target_blind_event_count": 88,
            "evaluable_event_count": fine_flow["primary_intersection_events"],
            "patient_cluster_count": fine_flow["primary_intersection_patients"],
            "v29_summary": private_v29_summary,
            "fine_only_summary": private_fine_summary,
            "paired_v29_minus_fine_only": private_paired,
        },
        "source_files": {
            "public_fine_only_oof": str(v16_prediction_path.relative_to(ROOT)),
            "public_v29_oof": str(public_v29_path.relative_to(ROOT)),
            "private_fine_evidence": str(evidence_tensor_path.relative_to(ROOT)),
            "private_v29_prediction": str(private_v29_tensor_path.relative_to(ROOT)),
        },
        "access_receipt": {
            "existing_frozen_fine_only_head_loaded": True,
            "raw_EEG_or_foundation_forward_performed": False,
            "model_training_or_baseline_selection_performed": False,
            "public_targets_loaded_for_frozen_evaluation": True,
            "opened_private_targets_loaded_for_post_open_evaluation": True,
            "private_used_for_baseline_training_or_calibration": False,
        },
        "interpretation_boundary": {
            "public_comparison_confirmatory": False,
            "private_comparison_fresh_external_validation": False,
            "fine_only_is_capacity_matched_to_v29_H_D": False,
            "same_split_mask_loss_and_low_capacity_head": True,
            "allowed_claim": (
                "v29 is compared with an existing non-foundation target-blind "
                "signal baseline under the same patient split, mask and set loss"
            ),
        },
    }
    tensors = {
        "public.v29_probability": v29_probability.contiguous(),
        "public.fine_only_logits": fine_logits.contiguous(),
        "private.v29_probability": private_v29.contiguous(),
        "private.fine_only_fold_probability": private_fine_fold.contiguous(),
        "private.fine_only_probability": private_fine_probability.contiguous(),
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
    }
    rows = [
        _public_row("v29_H_D", public_v29_metrics),
        _public_row("nonfoundation_fine_only", public_fine_metrics),
        _private_row("v29_H_D", private_v29_summary),
        _private_row("nonfoundation_fine_only", private_fine_summary),
    ]
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
        save_file(dict(tensors), str(staging / "baseline_predictions.safetensors"))
        with (staging / "baseline_summary.csv").open(
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
    parser.add_argument("--v16", type=Path, default=DEFAULT_V16)
    parser.add_argument("--public-v29", type=Path, default=DEFAULT_V29_PUBLIC)
    parser.add_argument(
        "--private-evidence", type=Path, default=DEFAULT_PRIVATE_EVIDENCE
    )
    parser.add_argument("--private-v29", type=Path, default=DEFAULT_PRIVATE_V29)
    parser.add_argument("--private-target", type=Path, default=DEFAULT_PRIVATE_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, tensors, rows = run(
        v16_directory=args.v16,
        public_v29_directory=args.public_v29,
        private_evidence_directory=args.private_evidence,
        private_v29_directory=args.private_v29,
        private_target_path=args.private_target,
    )
    output = publish(output=args.output, result=result, tensors=tensors, rows=rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "public_patients": result["public"]["patient_count"],
                "private_events": result["private"]["evaluable_event_count"],
                "training_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
