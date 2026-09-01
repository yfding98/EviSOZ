#!/usr/bin/env python3
"""Frozen H-carrier reliance stress for public v29 and private transport.

The perturbation family is declared in this source before evaluation.  The
script does not fit a model, select a perturbation, change the H/D weight, or
derive a threshold.  Public stresses act on the reliability-pooled 600-D H
carrier used by each held-out fold.  Private stresses act on the target-blind
event H carrier and are combined with the already frozen D-fold predictions.

This is a representation-level reliance audit.  It is not a raw-EEG causal
intervention and does not qualify any phase block as a clinical concept.
Private targets were opened in earlier project iterations; private metrics are
therefore post-open descriptive transport evidence only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Callable, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_labram_rank1_direct_token_oof_v28 as v28  # noqa: E402
from scripts.audit_labram_v29_token_stress_v38 import (  # noqa: E402
    _probability_logits,
    _stability,
)
from scripts.audit_private_frozen_publication_v36 import (  # noqa: E402
    BOOTSTRAP_SEED,
    _event_rows,
    _paired as _private_paired,
    _read_csv,
    _summary as _private_summary,
)
from scripts.audit_trustworthy_soz_candidate_v21 import (  # noqa: E402
    _fold_h_only_probability,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _paired_bootstrap,
)
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_labram_v29_h_carrier_stress_v43"
DEFAULT_V16 = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
)
DEFAULT_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE_PREDICTION = (
    ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
)
DEFAULT_PRIVATE_H = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
DEFAULT_PRIVATE_TARGET = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_labram_v29_h_carrier_stress_v43_20260816"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(features: torch.Tensor) -> torch.Tensor:
    return features


def _zero(features: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(features)


def _channel_mean(features: torch.Tensor) -> torch.Tensor:
    return features.mean(dim=1, keepdim=True).expand_as(features).contiguous()


def _left_right_swap(features: torch.Tensor) -> torch.Tensor:
    pairs = {
        "FP1": "FP2",
        "FP2": "FP1",
        "F7": "F8",
        "F8": "F7",
        "F3": "F4",
        "F4": "F3",
        "T7": "T8",
        "T8": "T7",
        "C3": "C4",
        "C4": "C3",
        "P7": "P8",
        "P8": "P7",
        "P3": "P4",
        "P4": "P3",
        "O1": "O2",
        "O2": "O1",
        "FZ": "FZ",
        "CZ": "CZ",
        "PZ": "PZ",
    }
    index = torch.tensor(
        [STANDARD_19.index(pairs[channel]) for channel in STANDARD_19],
        dtype=torch.long,
    )
    return features.index_select(1, index)


def _zero_block(features: torch.Tensor, block: int) -> torch.Tensor:
    if block not in (0, 1, 2):
        raise ValueError("H block must be 0, 1, or 2")
    result = features.clone()
    result[:, :, 200 * block : 200 * (block + 1)] = 0.0
    return result


def _remove_onset_minus_baseline(features: torch.Tensor) -> torch.Tensor:
    return _zero_block(features, 0)


def _remove_early_minus_baseline(features: torch.Tensor) -> torch.Tensor:
    return _zero_block(features, 1)


def _remove_late_minus_early(features: torch.Tensor) -> torch.Tensor:
    return _zero_block(features, 2)


def _swap_first_third_contrast(features: torch.Tensor) -> torch.Tensor:
    blocks = features.reshape(*features.shape[:-1], 3, 200)
    return blocks.index_select(
        -2, torch.tensor((2, 1, 0), dtype=torch.long)
    ).reshape_as(features).contiguous()


PERTURBATIONS: tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor], str], ...] = (
    ("identity_replay", _identity, "saved H-carrier semantics unchanged"),
    (
        "zero_H_content",
        _zero,
        "remove all H content while retaining the fold-local H-head prior",
    ),
    (
        "channel_mean_locality_removed",
        _channel_mean,
        "replace each channel H carrier by the within-unit across-channel mean",
    ),
    (
        "left_right_channel_swap",
        _left_right_swap,
        "swap homologous left/right H-channel identities while retaining the saved prior",
    ),
    (
        "remove_onset_minus_baseline",
        _remove_onset_minus_baseline,
        "zero the first 200-D onset-minus-baseline contrast block",
    ),
    (
        "remove_early_minus_baseline",
        _remove_early_minus_baseline,
        "zero the second 200-D early-minus-baseline contrast block",
    ),
    (
        "remove_late_minus_early",
        _remove_late_minus_early,
        "zero the third 200-D late-minus-early contrast block",
    ),
    (
        "swap_first_third_contrast",
        _swap_first_third_contrast,
        "exchange the first and third 200-D contrast blocks",
    ),
)


def _public_h_probability(
    h: torch.Tensor,
    states: Mapping[str, torch.Tensor],
    folds: torch.Tensor,
) -> torch.Tensor:
    output = torch.full((len(h), 19), torch.nan, dtype=torch.float32)
    for fold in range(5):
        held = torch.nonzero(folds == fold, as_tuple=False).flatten()
        output[held] = _fold_h_only_probability(h.index_select(0, held), states, fold)
    if not torch.isfinite(output).all():
        raise RuntimeError("public H stress replay is incomplete")
    return output.contiguous()


def _private_h_fold_probability(
    h: torch.Tensor, states: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    result = torch.stack(
        [_fold_h_only_probability(h, states, fold) for fold in range(5)], dim=1
    ).contiguous()
    if tuple(result.shape) != (len(h), 5, 19) or not torch.isfinite(result).all():
        raise RuntimeError("private H stress fold replay is malformed")
    return result


def _public_table_row(
    perturbation: str,
    scope: str,
    metrics: Mapping[str, object],
    stability: Mapping[str, float],
) -> dict[str, object]:
    top1 = metrics["top1"]
    ranking = metrics["ranking"]
    assert isinstance(top1, Mapping) and isinstance(ranking, Mapping)
    return {
        "perturbation": perturbation,
        "scope": scope,
        "strict_top1": float(top1["strict_accuracy"]),
        "neighborhood4_top1": float(top1["relaxed_accuracy"]),
        "macro_average_precision": float(ranking["macro_average_precision"]),
        "hit_at_5": float(ranking["hit_at_k"][5]),
        "far_error_count": float(metrics["far_error_count"]),
        **stability,
    }


def _private_table_row(
    perturbation: str,
    scope: str,
    summary: Mapping[str, object],
    stability: Mapping[str, float],
) -> dict[str, object]:
    event_micro = summary["event_micro"]
    patient_equal = summary["patient_equal_event_macro"]
    counts = summary["endpoint_counts"]
    assert isinstance(event_micro, Mapping)
    assert isinstance(patient_equal, Mapping)
    assert isinstance(counts, Mapping)
    return {
        "perturbation": perturbation,
        "scope": scope,
        "evaluable_events": int(summary["event_count"]),
        "patient_clusters": int(summary["patient_count"]),
        "strict_event_micro": float(event_micro["strict"]),
        "strict_patient_equal": float(patient_equal["strict"]),
        "neighborhood4_event_micro": float(event_micro["relaxed"]),
        "neighborhood4_patient_equal": float(patient_equal["relaxed"]),
        "laterality_patient_equal": float(patient_equal["laterality_agreement"]),
        "far_count": int(counts["far"]),
        "contralateral_far_count": int(counts["contralateral_far"]),
        "known_spread_top1_count": int(counts["known_spread_top1_all_enrolled"]),
        **stability,
    }


def run(
    *,
    v16_directory: Path,
    v29_directory: Path,
    private_prediction_directory: Path,
    private_h_directory: Path,
    private_target_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, torch.Tensor],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    loader_args = v28.build_parser().parse_args(["--device", "cpu"])
    stable = v28.v17._load_stable_development(loader_args)

    v16_manifest_path = (v16_directory / "manifest.json").resolve(strict=True)
    h_state_path = (v16_directory / "outer_fold_states.safetensors").resolve(
        strict=True
    )
    v29_manifest_path = (v29_directory / "manifest.json").resolve(strict=True)
    v29_tensor_path = (v29_directory / "oof_predictions.safetensors").resolve(
        strict=True
    )
    h_states = load_file(str(h_state_path), device="cpu")
    v29 = load_file(str(v29_tensor_path), device="cpu")
    for name, expected in (
        ("targets", stable.targets),
        ("target_mask", stable.target_mask),
        ("patient_folds", stable.patient_folds),
    ):
        actual = v29[name]
        if name == "target_mask":
            actual = actual.bool()
        elif name == "patient_folds":
            actual = actual.long()
        if not torch.equal(actual, expected):
            raise ValueError(f"public stable/v29 {name} carrier differs")

    public_original_h = v29["oof.h_only_probability"].float()
    public_original_d = v29["oof.rank1_direct_probability"].float()
    public_original_full = v29["oof.portable_equal_ensemble_probability"].float()
    public_original_full_logits = _probability_logits(
        public_original_full, stable.target_mask
    )

    private_manifest_path = (
        private_prediction_directory / "manifest.json"
    ).resolve(strict=True)
    private_prediction_path = (
        private_prediction_directory / "predictions.safetensors"
    ).resolve(strict=True)
    private_manifest = json.loads(private_manifest_path.read_text(encoding="utf-8"))
    events = private_manifest.get("events")
    if not isinstance(events, list) or len(events) != 88:
        raise ValueError("private v29 event roster changed")
    private_prediction = load_file(str(private_prediction_path), device="cpu")
    if not torch.equal(private_prediction["candidate_mask"].bool(), V11_CANDIDATE_MASK):
        raise ValueError("private v29 candidate mask changed")
    private_original_h_fold = private_prediction[
        "private_h_only_fold_probability"
    ].float()
    private_original_d_fold = private_prediction[
        "private_rank1_direct_fold_probability"
    ].float()
    private_original_full = private_prediction[
        "private_portable_equal_probability"
    ].float()

    private_h_manifest_path = (private_h_directory / "manifest.json").resolve(
        strict=True
    )
    private_h_manifest = json.loads(private_h_manifest_path.read_text(encoding="utf-8"))
    private_h_path = (
        private_h_directory / str(private_h_manifest["tensor_file"])
    ).resolve(strict=True)
    private_h = load_file(str(private_h_path), device="cpu")["h_event"].float()
    if tuple(private_h.shape) != (88, 19, 600):
        raise ValueError("private H carrier shape changed")
    if [str(row["event_id"]) for row in events] != [
        str(row["event_id"]) for row in private_h_manifest.get("events", ())
    ]:
        raise ValueError("private prediction/H event identity differs")
    private_target_rows = _read_csv(private_target_path)

    public_metrics: dict[str, object] = {}
    public_stability: dict[str, object] = {}
    public_paired: dict[str, object] = {}
    private_metrics: dict[str, object] = {}
    private_stability: dict[str, object] = {}
    private_paired: dict[str, object] = {}
    public_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    tensors: dict[str, torch.Tensor] = {}
    private_original_rows, private_flow = _event_rows(
        scores=private_original_full,
        events=events,
        target_rows=private_target_rows,
    )

    for index, (name, transform, semantics) in enumerate(PERTURBATIONS):
        public_h_probability = _public_h_probability(
            transform(stable.h_patient), h_states, stable.patient_folds
        )
        public_full_probability = (
            0.5 * public_h_probability + 0.5 * public_original_d
        ).contiguous()
        public_h_metrics = _evaluate(
            _probability_logits(public_h_probability, stable.target_mask),
            stable.targets,
            stable.target_mask,
        )
        public_full_metrics = _evaluate(
            _probability_logits(public_full_probability, stable.target_mask),
            stable.targets,
            stable.target_mask,
        )
        public_h_stability = _stability(
            public_original_h, public_h_probability, stable.target_mask
        )
        public_full_stability = _stability(
            public_original_full, public_full_probability, stable.target_mask
        )
        public_metrics[name] = {
            "semantics": semantics,
            "stressed_H_only": public_h_metrics,
            "stressed_H_plus_frozen_D_equal": public_full_metrics,
        }
        public_stability[name] = {
            "stressed_H_only": public_h_stability,
            "stressed_H_plus_frozen_D_equal": public_full_stability,
        }
        public_paired[name] = _paired_bootstrap(
            _probability_logits(public_full_probability, stable.target_mask),
            public_original_full_logits,
            stable.targets,
            stable.target_mask,
        )
        public_rows.append(
            _public_table_row(
                name, "stressed_H_only", public_h_metrics, public_h_stability
            )
        )
        public_rows.append(
            _public_table_row(
                name,
                "stressed_H_plus_frozen_D_equal",
                public_full_metrics,
                public_full_stability,
            )
        )

        private_h_fold = _private_h_fold_probability(transform(private_h), h_states)
        private_h_probability = private_h_fold.mean(dim=1).contiguous()
        private_full_probability = (
            0.5 * private_h_fold + 0.5 * private_original_d_fold
        ).mean(dim=1).contiguous()
        private_h_event_rows, h_flow = _event_rows(
            scores=private_h_probability,
            events=events,
            target_rows=private_target_rows,
        )
        private_full_event_rows, full_flow = _event_rows(
            scores=private_full_probability,
            events=events,
            target_rows=private_target_rows,
        )
        if h_flow != private_flow or full_flow != private_flow:
            raise RuntimeError("private H stress changed the evaluation cohort")
        private_h_summary = _private_summary(
            private_h_event_rows, seed=BOOTSTRAP_SEED + 1000 + 100 * index
        )
        private_full_summary = _private_summary(
            private_full_event_rows, seed=BOOTSTRAP_SEED + 2000 + 100 * index
        )
        private_h_stability = _stability(
            private_original_h_fold.mean(dim=1),
            private_h_probability,
            V11_CANDIDATE_MASK.unsqueeze(0).expand(len(private_h), -1),
        )
        private_full_stability = _stability(
            private_original_full,
            private_full_probability,
            V11_CANDIDATE_MASK.unsqueeze(0).expand(len(private_h), -1),
        )
        private_metrics[name] = {
            "semantics": semantics,
            "stressed_H_only": private_h_summary,
            "stressed_H_plus_frozen_D_equal": private_full_summary,
        }
        private_stability[name] = {
            "stressed_H_only_all_88_target_blind_events": private_h_stability,
            "stressed_H_plus_frozen_D_equal_all_88_target_blind_events": (
                private_full_stability
            ),
        }
        private_paired[name] = _private_paired(
            private_full_event_rows,
            private_original_rows,
            seed=BOOTSTRAP_SEED + 30_000 + 100 * index,
        )
        private_rows.append(
            _private_table_row(
                name, "stressed_H_only", private_h_summary, private_h_stability
            )
        )
        private_rows.append(
            _private_table_row(
                name,
                "stressed_H_plus_frozen_D_equal",
                private_full_summary,
                private_full_stability,
            )
        )

        tensors[f"public.H.{name}"] = public_h_probability
        tensors[f"public.H_plus_D.{name}"] = public_full_probability
        tensors[f"private.H.{name}"] = private_h_probability
        tensors[f"private.H_plus_D.{name}"] = private_full_probability

    public_identity_h_difference = float(
        (tensors["public.H.identity_replay"] - public_original_h).abs().max()
    )
    public_identity_full_difference = float(
        (tensors["public.H_plus_D.identity_replay"] - public_original_full).abs().max()
    )
    private_identity_h_difference = float(
        (
            tensors["private.H.identity_replay"]
            - private_original_h_fold.mean(dim=1)
        )
        .abs()
        .max()
    )
    private_identity_full_difference = float(
        (tensors["private.H_plus_D.identity_replay"] - private_original_full)
        .abs()
        .max()
    )
    identity_differences = {
        "public_H_probability": public_identity_h_difference,
        "public_full_v29_probability": public_identity_full_difference,
        "private_H_probability": private_identity_h_difference,
        "private_full_v29_probability": private_identity_full_difference,
    }
    if max(identity_differences.values()) > 1e-6:
        raise ValueError(f"H identity replay drifted: {identity_differences}")

    declared_public = json.loads(v29_manifest_path.read_text(encoding="utf-8"))[
        "metrics"
    ]["portable_equal_ensemble"]
    identity_public = public_metrics["identity_replay"][
        "stressed_H_plus_frozen_D_equal"
    ]
    if identity_public["top1"]["strict_accuracy"] != declared_public["top1"][
        "strict_accuracy"
    ] or identity_public["top1"]["relaxed_accuracy"] != declared_public["top1"][
        "relaxed_accuracy"
    ]:
        raise ValueError("public H identity metrics do not recover v29")

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_H_carrier_public_private_reliance_stress",
        "analysis_role": {
            "public": "posthoc_consumed_public_development_audit",
            "private": "post_open_descriptive_transport_audit",
        },
        "cohort": {
            "public_patients": len(stable.patient_ids),
            "public_events_before_patient_pooling": int(stable.event_counts.sum()),
            "private_target_blind_events": len(events),
            "private_evaluable_events": private_flow["primary_intersection_events"],
            "private_evaluable_patient_clusters": private_flow[
                "primary_intersection_patients"
            ],
        },
        "perturbations": [
            {"name": name, "semantics": semantics}
            for name, _, semantics in PERTURBATIONS
        ],
        "public": {
            "metrics": public_metrics,
            "stability": public_stability,
            "paired_stressed_minus_original_v29": public_paired,
        },
        "private": {
            "cohort_flow": private_flow,
            "metrics": private_metrics,
            "stability": private_stability,
            "paired_stressed_minus_original_v29": private_paired,
        },
        "identity_replay_max_absolute_probability_difference": identity_differences,
        "source_files": {
            "v16_manifest": str(v16_manifest_path.relative_to(ROOT)),
            "v16_manifest_sha256": _sha256(v16_manifest_path),
            "h_fold_states": str(h_state_path.relative_to(ROOT)),
            "h_fold_states_sha256": _sha256(h_state_path),
            "v29_manifest": str(v29_manifest_path.relative_to(ROOT)),
            "v29_manifest_sha256": _sha256(v29_manifest_path),
            "v29_tensor": str(v29_tensor_path.relative_to(ROOT)),
            "v29_tensor_sha256": _sha256(v29_tensor_path),
            "private_prediction_manifest": str(
                private_manifest_path.relative_to(ROOT)
            ),
            "private_prediction_manifest_sha256": _sha256(private_manifest_path),
            "private_prediction_tensor": str(
                private_prediction_path.relative_to(ROOT)
            ),
            "private_prediction_tensor_sha256": _sha256(private_prediction_path),
            "private_H_manifest": str(private_h_manifest_path.relative_to(ROOT)),
            "private_H_manifest_sha256": _sha256(private_h_manifest_path),
            "private_H_tensor": str(private_h_path.relative_to(ROOT)),
            "private_H_tensor_sha256": _sha256(private_h_path),
            "private_target_ledger": str(private_target_path.resolve().relative_to(ROOT)),
            "private_target_ledger_sha256": _sha256(private_target_path.resolve()),
        },
        "access_receipt": {
            "raw_EEG_loaded": False,
            "cached_foundation_H_carriers_loaded": True,
            "foundation_forward_performed": False,
            "model_training_performed": False,
            "model_fusion_or_threshold_selected": False,
            "report_text_selected_or_changed": False,
            "existing_public_targets_loaded_for_frozen_evaluation": True,
            "previously_opened_private_targets_loaded_for_descriptive_evaluation": True,
            "private_used_to_select_any_perturbation": False,
        },
        "interpretation_boundary": {
            "stress_level": "cached_LaBraM_H_carrier_before_saved_fold_transform",
            "public_stress_unit": "reliability_pooled_patient_H",
            "private_stress_unit": "target_blind_event_H",
            "D_carrier_perturbed": False,
            "raw_EEG_causal_intervention": False,
            "clinical_phase_concept_qualified": False,
            "public_confirmatory_inference": False,
            "private_fresh_external_validation": False,
            "allowed_claim": (
                "the frozen H head and full v29 ranking show the reported reliance "
                "on prespecified H-carrier channel and contrast-block content"
            ),
        },
    }
    tensors["public.targets"] = stable.targets.contiguous()
    tensors["public.target_mask"] = stable.target_mask.contiguous()
    tensors["public.patient_folds"] = stable.patient_folds.contiguous()
    tensors["candidate_mask"] = V11_CANDIDATE_MASK.contiguous()
    return result, tensors, public_rows, private_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish(
    *,
    output: Path,
    result: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    public_rows: Sequence[Mapping[str, object]],
    private_rows: Sequence[Mapping[str, object]],
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
        save_file(dict(tensors), str(staging / "stress_predictions.safetensors"))
        _write_csv(staging / "public_stress_table.csv", public_rows)
        _write_csv(staging / "private_stress_table.csv", private_rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--v16", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v29", type=Path, default=DEFAULT_V29)
    parser.add_argument(
        "--private-prediction", type=Path, default=DEFAULT_PRIVATE_PREDICTION
    )
    parser.add_argument("--private-h", type=Path, default=DEFAULT_PRIVATE_H)
    parser.add_argument("--private-target", type=Path, default=DEFAULT_PRIVATE_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, tensors, public_rows, private_rows = run(
        v16_directory=args.v16,
        v29_directory=args.v29,
        private_prediction_directory=args.private_prediction,
        private_h_directory=args.private_h,
        private_target_path=args.private_target,
    )
    output = publish(
        output=args.output,
        result=result,
        tensors=tensors,
        public_rows=public_rows,
        private_rows=private_rows,
    )
    identity_public = result["public"]["metrics"]["identity_replay"][
        "stressed_H_plus_frozen_D_equal"
    ]["top1"]
    identity_private = result["private"]["metrics"]["identity_replay"][
        "stressed_H_plus_frozen_D_equal"
    ]["event_micro"]
    print(
        json.dumps(
            {
                "output": str(output),
                "public_strict": identity_public["strict_accuracy"],
                "private_strict": identity_private["strict"],
                "training_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
