#!/usr/bin/env python3
"""Run preregistered source-train-only shortcut controls for frozen LaBraM H."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_frozen_h_nested_oof_v3 import (  # noqa: E402
    EPOCHS,
    LEARNING_RATE,
    _common_initialization_receipt,
    _load_frozen_h_source_train,
    _predict,
    _seeded_model,
    _subset,
)
from scripts.run_labram_temporal_mil_nested_oof_v1 import (  # noqa: E402
    BASE_SEED,
    BOOTSTRAP_REPLICATES,
    MAX_GRAD_NORM,
    OUTER_FOLDS,
    WEIGHT_DECAY,
    _canonical_bytes,
    _file_sha256,
    _indices_for_folds,
    _metrics,
    _paired_bootstrap,
    _tensor_state_sha256,
)
from src.soz.frozen_h_recovery import (  # noqa: E402
    FrozenHPatientBatch,
    FrozenHStandardization,
    fit_frozen_h_standardization,
    frozen_h_objective,
)
from src.soz.frozen_h_shortcut_controls import (  # noqa: E402
    FROZEN_H_SHORTCUT_CONTROL_SCHEMA,
    POSITION_PROTOTYPE_SEMANTICS,
    Q_ONLY_SEMANTICS,
    ZERO_H_V_SEMANTICS,
    apply_event_time_shuffle,
    cross_patient_event_bijection_feasibility,
    fit_fold_local_position_prototype,
    make_cross_patient_event_time_shuffle_plan,
    replace_h_with_position_prototype,
    replace_h_with_standardization_mean,
    zero_h_q_only_patient_outputs,
)
from src.soz.temporal_mil_recovery import jeffreys_channel_prior_logits  # noqa: E402


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_frozen_h_shortcut_control_protocol_v3_1_20260811_zh.md"
)
REFERENCE_PATH = ROOT / "outputs/labram_frozen_h_nested_oof_v3_20260810"
REFERENCE_MANIFEST_SHA256 = (
    "285609d9bba2d17fc728541fadabb5a272ee36e0c0b53d2be5ac865648aa04b6"
)
REFERENCE_PREDICTIONS_SHA256 = (
    "4bacebb5dd3f616ebd6a039d51364864791b954c27e8058dc536a3fc7b83c9d1"
)
FIXED_REFERENCE = "frozen_h_v_uniform"
CONTROL_NAMES = (
    "position_only_h_v_uniform",
    "event_time_shuffled_h_v_uniform",
    "zero_h_v_uniform",
    "zero_h_q_only",
)
CONTENT_NULL_CONTROLS = CONTROL_NAMES[:2]
ZERO_H_V_CONTROL = "zero_h_v_uniform"
Q_ONLY_CONTROL = "zero_h_q_only"
EXISTING_V2_CONTROL = "v2_v_only_equal"
STRICT_ONE_PATIENT = 1.0 / 65.0
MATERIAL_AP = 0.01


def _preflight_shuffle_feasibility(
    full: FrozenHPatientBatch,
    patient_folds: tuple[int, ...],
) -> list[dict[str, object]]:
    """Fail before training if any fold-contained cross-patient shuffle is impossible."""

    owner = full.base.event_patient_index.detach().cpu().long()
    fold_by_patient = torch.tensor(patient_folds, dtype=torch.long)
    rows = []
    for outer_fold in OUTER_FOLDS:
        for scope, keep_fold in (
            ("outer_train", fold_by_patient != outer_fold),
            ("outer_held", fold_by_patient == outer_fold),
        ):
            selected_events = keep_fold.index_select(0, owner)
            scope_owner = owner[selected_events]
            receipt = cross_patient_event_bijection_feasibility(scope_owner)
            row = {"outer_fold": outer_fold, "scope": scope, **receipt}
            rows.append(row)
            if not receipt["feasible"]:
                raise RuntimeError(
                    "shuffle feasibility preflight failed before training: "
                    f"outer_fold={outer_fold}, scope={scope}, "
                    f"reason={receipt['reason']}"
                )
    return rows


def _load_reference(
    full: FrozenHPatientBatch,
    patient_folds: tuple[int, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
    dict[int, dict[str, object]],
]:
    manifest_path = REFERENCE_PATH / "manifest.json"
    prediction_path = REFERENCE_PATH / "oof_predictions.safetensors"
    if _file_sha256(manifest_path) != REFERENCE_MANIFEST_SHA256:
        raise RuntimeError("frozen-H v3 reference manifest changed")
    if _file_sha256(prediction_path) != REFERENCE_PREDICTIONS_SHA256:
        raise RuntimeError("frozen-H v3 reference prediction tensor changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_development_only":
        raise RuntimeError("frozen-H v3 reference is not complete")
    if tuple(manifest.get("patient_ids", ())) != full.base.patient_ids:
        raise RuntimeError("reference patient roster differs from source-train")
    if tuple(int(value) for value in manifest.get("patient_folds", ())) != patient_folds:
        raise RuntimeError("reference patient folds differ from source-train")
    boundary = manifest.get("scientific_boundary", {})
    if (
        manifest.get("source_dev_forward_count") != 0
        or boundary.get("source_dev_used") is not False
        or boundary.get("source_eval_used") is not False
        or boundary.get("private_used") is not False
    ):
        raise RuntimeError("reference violates the shortcut-audit data boundary")

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for reference loading") from exc
    tensors = load_file(str(prediction_path), device="cpu")
    probability_name = f"probability__{FIXED_REFERENCE}"
    required = (
        FIXED_REFERENCE,
        probability_name,
        "targets",
        "target_mask",
        "patient_folds",
        EXISTING_V2_CONTROL,
    )
    if any(name not in tensors for name in required):
        raise RuntimeError("frozen-H v3 reference tensor is incomplete")
    expected_shape = (len(full.base.patient_ids), 19)
    if tuple(tensors[FIXED_REFERENCE].shape) != expected_shape or tuple(
        tensors[probability_name].shape
    ) != expected_shape or tuple(tensors[EXISTING_V2_CONTROL].shape) != expected_shape:
        raise RuntimeError("reference prediction shape differs from source-train")
    if not torch.equal(tensors["targets"], full.base.targets.cpu()) or not torch.equal(
        tensors["target_mask"], full.base.target_mask.cpu()
    ):
        raise RuntimeError("reference target tensor differs from source-train")
    if not torch.equal(
        tensors["patient_folds"], torch.tensor(patient_folds, dtype=torch.int64)
    ):
        raise RuntimeError("reference fold tensor differs from source-train")

    outer_fit_by_fold = {}
    for row in manifest["result"]["outer_folds"]:
        fold = int(row["outer_fold"])
        outer_fit_by_fold[fold] = row["outer_candidates"][FIXED_REFERENCE]["fit"]
    if set(outer_fit_by_fold) != set(OUTER_FOLDS):
        raise RuntimeError("reference lacks a fixed-candidate fit for every fold")
    receipt = {
        "path": str(REFERENCE_PATH.relative_to(ROOT)),
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "prediction_sha256": REFERENCE_PREDICTIONS_SHA256,
        "candidate": FIXED_REFERENCE,
        "prediction_reused_without_retraining": True,
    }
    return (
        tensors[FIXED_REFERENCE].float().contiguous(),
        tensors[probability_name].float().contiguous(),
        tensors[EXISTING_V2_CONTROL].float().contiguous(),
        receipt,
        outer_fit_by_fold,
    )


def _check_reference_initialization(
    fit: Mapping[str, object],
    reference_fit: Mapping[str, object],
) -> None:
    for field in ("h_initialization_sha256", "v_initialization_sha256"):
        if fit.get(field) != reference_fit.get(field):
            raise RuntimeError(
                f"shortcut head does not share reference initialization: {field}"
            )
    if fit.get("seed") != reference_fit.get("seed"):
        raise RuntimeError("shortcut head does not share the reference model seed")


def _fit_with_frozen_fold_preprocessing(
    transformed_train: FrozenHPatientBatch,
    *,
    standardization: FrozenHStandardization,
    prior: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Fit the fixed v3 candidate while reusing original-H fold preprocessing."""

    initialization = _common_initialization_receipt(
        prior, standardization, seed=seed
    )
    model = _seeded_model(
        prior,
        standardization,
        candidate=FIXED_REFERENCE,
        seed=seed,
        device=device,
    )
    batch = transformed_train.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    first = None
    last = None
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.node_tokens, batch.base.evidence)
        objective = frozen_h_objective(output.event_probabilities, batch)
        objective.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        row = {
            "total": float(objective.total.detach().cpu()),
            "exact_set_mass": float(objective.exact_set_mass.detach().cpu()),
            "pairwise": float(objective.pairwise.detach().cpu()),
        }
        if first is None:
            first = row
        last = row
    optimizer.zero_grad(set_to_none=True)
    del optimizer, batch
    model.eval()
    model.requires_grad_(False)
    preprocessing_state = {
        "h_mean": standardization.mean.detach().cpu(),
        "h_scale": standardization.scale.detach().cpu(),
        "channel_prior": prior.detach().cpu(),
    }
    assert first is not None and last is not None
    return model, {
        "candidate": FIXED_REFERENCE,
        "seed": seed,
        "epochs": EPOCHS,
        "trainable_parameter_count": model.n_trainable_parameters,
        "fit_scope_patient_count": len(transformed_train.base.patient_ids),
        "fold_local_preprocessing_state_sha256": _tensor_state_sha256(
            preprocessing_state
        ),
        "preprocessing_reused_from_original_untransformed_h": True,
        "first_epoch": first,
        "final_epoch": last,
        **initialization,
    }


def _check_reference_preprocessing(
    fit: Mapping[str, object],
    reference_fit: Mapping[str, object],
) -> None:
    _check_reference_initialization(fit, reference_fit)
    if fit.get("fold_local_preprocessing_state_sha256") != reference_fit.get(
        "fold_local_preprocessing_state_sha256"
    ):
        raise RuntimeError("shortcut does not share the reference fold preprocessing")


def _endpoint_gate(
    reference_metrics: Mapping[str, object],
    control_metrics: Mapping[str, Mapping[str, object]],
    bootstrap: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    reference_strict = float(reference_metrics["top1"]["strict_accuracy"])
    reference_ap = float(reference_metrics["ranking"]["macro_average_precision"])
    content_strict = {
        name: float(control_metrics[name]["top1"]["strict_accuracy"])
        for name in CONTENT_NULL_CONTROLS
    }
    content_ap = {
        name: float(control_metrics[name]["ranking"]["macro_average_precision"])
        for name in CONTENT_NULL_CONTROLS
    }

    def hit3(metrics: Mapping[str, object]) -> float:
        values = metrics["ranking"]["hit_at_k"]
        return float(values[3] if 3 in values else values["3"])

    reference_hit3 = hit3(reference_metrics)
    content_hit3 = {
        name: hit3(control_metrics[name]) for name in CONTENT_NULL_CONTROLS
    }
    maximum_shortcut_strict = max(content_strict.values())
    maximum_shortcut_ap = max(content_ap.values())
    maximum_shortcut_hit3 = max(content_hit3.values())
    content_strict_pass = (
        reference_strict >= maximum_shortcut_strict - STRICT_ONE_PATIENT - 1e-12
    )
    content_ap_pass = reference_ap >= maximum_shortcut_ap + MATERIAL_AP - 1e-12
    content_hit3_pass = reference_hit3 >= maximum_shortcut_hit3 - 1e-12

    zero_metrics = control_metrics[ZERO_H_V_CONTROL]
    zero_strict_delta = reference_strict - float(
        zero_metrics["top1"]["strict_accuracy"]
    )
    zero_ap_delta = reference_ap - float(
        zero_metrics["ranking"]["macro_average_precision"]
    )
    zero_hit3_delta = reference_hit3 - hit3(zero_metrics)
    zero_h_v_pass = (
        zero_strict_delta >= -STRICT_ONE_PATIENT - 1e-12
        and zero_ap_delta >= MATERIAL_AP - 1e-12
        and zero_hit3_delta >= -1e-12
    )

    q_metrics = control_metrics[Q_ONLY_CONTROL]
    q_strict_delta = reference_strict - float(q_metrics["top1"]["strict_accuracy"])
    q_ap_delta = reference_ap - float(
        q_metrics["ranking"]["macro_average_precision"]
    )
    q_direction = q_strict_delta >= -1e-12 and q_ap_delta >= -1e-12
    q_material = (
        q_strict_delta >= STRICT_ONE_PATIENT - 1e-12
        or q_ap_delta >= MATERIAL_AP - 1e-12
    )
    q_only_pass = q_direction and q_material

    passed = (
        content_strict_pass
        and content_ap_pass
        and content_hit3_pass
        and zero_h_v_pass
        and q_only_pass
    )
    rows = {}
    for name, boot in bootstrap.items():
        metrics = control_metrics[name]
        rows[name] = {
            "reference_minus_control_strict_top1": reference_strict
            - float(metrics["top1"]["strict_accuracy"]),
            "reference_minus_control_macro_ap": reference_ap
            - float(metrics["ranking"]["macro_average_precision"]),
            "reference_minus_control_hit_at_3": reference_hit3 - hit3(metrics),
            "strict_ci95": boot["strict_top1"]["ci95"],
            "macro_ap_ci95": boot["macro_ap"]["ci95"],
        }
    status = "pass_for_peft_screening" if passed else "blocked_by_shortcut_gate"
    return {
        "status": status,
        "peft_screening_permitted": passed,
        "strict_one_patient_tolerance_or_material_delta": STRICT_ONE_PATIENT,
        "macro_ap_material_threshold": MATERIAL_AP,
        "content_shortcut_endpoint_envelope": {
            "controls": list(CONTENT_NULL_CONTROLS),
            "maximum_strict_top1": maximum_shortcut_strict,
            "maximum_macro_ap": maximum_shortcut_ap,
            "maximum_hit_at_3": maximum_shortcut_hit3,
            "reference_strict_within_one_patient": content_strict_pass,
            "reference_macro_ap_at_least_0_01_higher": content_ap_pass,
            "reference_hit_at_3_not_lower": content_hit3_pass,
            "strict_maximizing_control": max(
                content_strict, key=content_strict.__getitem__
            ),
            "ap_maximizing_control": max(content_ap, key=content_ap.__getitem__),
            "hit_at_3_maximizing_control": max(
                content_hit3, key=content_hit3.__getitem__
            ),
        },
        "matched_zero_h_v_gate": {
            "reference_minus_control_strict_top1": zero_strict_delta,
            "reference_minus_control_macro_ap": zero_ap_delta,
            "reference_minus_control_hit_at_3": zero_hit3_delta,
            "strict_within_one_patient": zero_strict_delta
            >= -STRICT_ONE_PATIENT - 1e-12,
            "macro_ap_at_least_0_01_higher": zero_ap_delta
            >= MATERIAL_AP - 1e-12,
            "hit_at_3_not_lower": zero_hit3_delta >= -1e-12,
            "pass": zero_h_v_pass,
        },
        "prevalence_only_gate": {
            "reference_minus_control_strict_top1": q_strict_delta,
            "reference_minus_control_macro_ap": q_ap_delta,
            "strict_and_ap_not_lower": q_direction,
            "strict_plus_one_patient_or_ap_plus_0_01": q_material,
            "pass": q_only_pass,
        },
        "existing_v2_v_only_is_diagnostic_not_gate": EXISTING_V2_CONTROL,
        "all_pairwise_diagnostics": rows,
        "interpretation_boundary": (
            "pass permits one preregistered PEFT trial only; it does not establish "
            "waveform, onset, causal SOZ, external, or clinical validity"
        ),
    }


def _run(
    full: FrozenHPatientBatch,
    patient_folds: tuple[int, ...],
    reference_scores: torch.Tensor,
    reference_probabilities: torch.Tensor,
    existing_v2_scores: torch.Tensor,
    reference_fits: Mapping[int, Mapping[str, object]],
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    patients = len(full.base.patient_ids)
    predictions = {
        FIXED_REFERENCE: reference_scores.clone(),
        EXISTING_V2_CONTROL: existing_v2_scores.clone(),
        **{name: torch.full((patients, 19), torch.nan) for name in CONTROL_NAMES},
    }
    probabilities = {
        FIXED_REFERENCE: reference_probabilities.clone(),
        **{name: torch.full((patients, 19), torch.nan) for name in CONTROL_NAMES},
    }
    outer_rows = []

    for outer_fold in OUTER_FOLDS:
        train_folds = tuple(fold for fold in OUTER_FOLDS if fold != outer_fold)
        train_indices = _indices_for_folds(patient_folds, train_folds)
        held_indices = _indices_for_folds(patient_folds, (outer_fold,))
        train = _subset(full, train_indices)
        held = _subset(full, held_indices)
        model_seed = BASE_SEED + 50000 + outer_fold * 1000
        reference_fit = reference_fits[outer_fold]
        if int(reference_fit["seed"]) != model_seed:
            raise RuntimeError("reference outer-fold seed differs from v3 protocol")
        standardization = fit_frozen_h_standardization(train)
        prior = jeffreys_channel_prior_logits(train.base).detach().cpu()
        preprocessing_sha256 = _tensor_state_sha256(
            {
                "h_mean": standardization.mean.detach().cpu(),
                "h_scale": standardization.scale.detach().cpu(),
                "channel_prior": prior,
            }
        )
        if preprocessing_sha256 != reference_fit.get(
            "fold_local_preprocessing_state_sha256"
        ):
            raise RuntimeError("recomputed original-H fold preprocessing differs from v3")

        q_scores, q_probabilities, q_prior = zero_h_q_only_patient_outputs(
            train, held_patient_count=len(held_indices)
        )
        if not torch.equal(q_prior, prior):
            raise RuntimeError("Q-only prior differs from shared outer-fold prior")
        predictions["zero_h_q_only"][list(held_indices)] = q_scores
        probabilities["zero_h_q_only"][list(held_indices)] = q_probabilities

        prototype = fit_fold_local_position_prototype(train)
        position_train = replace_h_with_position_prototype(train, prototype)
        position_held = replace_h_with_position_prototype(held, prototype)
        position_model, position_fit = _fit_with_frozen_fold_preprocessing(
            position_train,
            standardization=standardization,
            prior=prior,
            seed=model_seed,
            device=device,
        )
        _check_reference_preprocessing(position_fit, reference_fit)
        position_scores, position_probabilities, _ = _predict(
            position_model, position_held, device=device
        )
        predictions["position_only_h_v_uniform"][list(held_indices)] = position_scores
        probabilities["position_only_h_v_uniform"][list(held_indices)] = position_probabilities
        del position_model, position_train, position_held

        zero_h_train = replace_h_with_standardization_mean(train, standardization)
        zero_h_held = replace_h_with_standardization_mean(held, standardization)
        zero_h_model, zero_h_fit = _fit_with_frozen_fold_preprocessing(
            zero_h_train,
            standardization=standardization,
            prior=prior,
            seed=model_seed,
            device=device,
        )
        _check_reference_preprocessing(zero_h_fit, reference_fit)
        zero_h_scores, zero_h_probabilities, _ = _predict(
            zero_h_model, zero_h_held, device=device
        )
        predictions[ZERO_H_V_CONTROL][list(held_indices)] = zero_h_scores
        probabilities[ZERO_H_V_CONTROL][list(held_indices)] = zero_h_probabilities
        del zero_h_model, zero_h_train, zero_h_held

        train_shuffle_seed = BASE_SEED + 71000 + outer_fold * 1000 + 1
        held_shuffle_seed = BASE_SEED + 71000 + outer_fold * 1000 + 2
        train_plan = make_cross_patient_event_time_shuffle_plan(
            train, seed=train_shuffle_seed
        )
        held_plan = make_cross_patient_event_time_shuffle_plan(
            held, seed=held_shuffle_seed
        )
        shuffled_train = apply_event_time_shuffle(train, train_plan)
        shuffled_held = apply_event_time_shuffle(held, held_plan)
        shuffled_model, shuffled_fit = _fit_with_frozen_fold_preprocessing(
            shuffled_train,
            standardization=standardization,
            prior=prior,
            seed=model_seed,
            device=device,
        )
        _check_reference_preprocessing(shuffled_fit, reference_fit)
        shuffled_scores, shuffled_probabilities, _ = _predict(
            shuffled_model, shuffled_held, device=device
        )
        predictions["event_time_shuffled_h_v_uniform"][list(held_indices)] = shuffled_scores
        probabilities["event_time_shuffled_h_v_uniform"][list(held_indices)] = shuffled_probabilities
        del shuffled_model, shuffled_train, shuffled_held

        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "train_patient_count": len(train_indices),
                "held_patient_count": len(held_indices),
                "model_seed_shared_with_v3_reference": model_seed,
                "reference_fit_initialization": {
                    "h_initialization_sha256": reference_fit[
                        "h_initialization_sha256"
                    ],
                    "v_initialization_sha256": reference_fit[
                        "v_initialization_sha256"
                    ],
                },
                "position_prototype": prototype.receipt(),
                "position_fit": position_fit,
                "zero_h_v": {
                    "semantics": ZERO_H_V_SEMANTICS,
                    "fit": zero_h_fit,
                },
                "shuffle_train": train_plan.receipt(
                    train.base.event_patient_index
                ),
                "shuffle_held": held_plan.receipt(held.base.event_patient_index),
                "shuffle_fit": shuffled_fit,
                "q_only": {
                    "semantics": Q_ONLY_SEMANTICS,
                    "prior_logits": [float(value) for value in q_prior.tolist()],
                },
            }
        )
        del train, held, prototype, train_plan, held_plan, standardization
        print(
            json.dumps(
                {"stage": "shortcut_outer_complete", "outer_fold": outer_fold},
                sort_keys=True,
            ),
            flush=True,
        )

    if any(not torch.isfinite(value).all() for value in predictions.values()) or any(
        not torch.isfinite(value).all() for value in probabilities.values()
    ):
        raise RuntimeError("shortcut OOF left prediction cells unfilled")
    metrics = {
        name: _metrics(value, full.base.targets, full.base.target_mask)
        for name, value in predictions.items()
    }
    bootstrap = {
        name: _paired_bootstrap(
            predictions[FIXED_REFERENCE],
            predictions[name],
            full.base.targets,
            full.base.target_mask,
        )
        for name in (*CONTROL_NAMES, EXISTING_V2_CONTROL)
    }
    decision = _endpoint_gate(
        metrics[FIXED_REFERENCE],
        {
            name: metrics[name]
            for name in (*CONTROL_NAMES, EXISTING_V2_CONTROL)
        },
        bootstrap,
    )
    result = {
        "fixed_reference": FIXED_REFERENCE,
        "outer_folds": outer_rows,
        "oof_metrics": metrics,
        "reference_minus_control_paired_patient_bootstrap": bootstrap,
        "shortcut_decision": decision,
    }
    return result, predictions, probabilities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight_only and args.output_directory is None:
        raise ValueError("full shortcut run requires --output-directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    full, patient_folds, lineage = _load_frozen_h_source_train()
    shuffle_feasibility = _preflight_shuffle_feasibility(full, patient_folds)
    (
        reference_scores,
        reference_probabilities,
        existing_v2_scores,
        reference_receipt,
        reference_fits,
    ) = _load_reference(full, patient_folds)
    preflight = {
        "status": "ready_frozen_h_shortcut_controls_source_train_only",
        "schema_version": FROZEN_H_SHORTCUT_CONTROL_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "device": str(device),
        "patient_count": len(full.base.patient_ids),
        "event_count": full.base.evidence.batch_size,
        "node_token_shape": list(full.node_tokens.shape),
        "fold_counts": {
            str(fold): sum(value == fold for value in patient_folds)
            for fold in OUTER_FOLDS
        },
        "fixed_reference": FIXED_REFERENCE,
        "controls": list(CONTROL_NAMES),
        "supplemental_existing_v2_control": EXISTING_V2_CONTROL,
        "shuffle_feasibility_preflight": shuffle_feasibility,
        "position_control_semantics": POSITION_PROTOTYPE_SEMANTICS,
        "reference_receipt": reference_receipt,
        "lineage": lineage,
        "foundation_trainable_parameter_count": 0,
        "source_dev_forward_count": 0,
        "source_dev_target_values_reachable": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_promotion": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0

    output = Path(os.path.abspath(args.output_directory))
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise FileExistsError(f"output already exists or is invalid: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("output parent must be a regular directory")
    for source in (PROTOCOL_PATH, REFERENCE_PATH):
        resolved = source.resolve(strict=True)
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError("output path overlaps immutable input")

    result, predictions, probabilities = _run(
        full,
        patient_folds,
        reference_scores,
        reference_probabilities,
        existing_v2_scores,
        reference_fits,
        device=device,
    )
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for publication") from exc
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        tensors = {
            **{name: value.contiguous() for name, value in predictions.items()},
            **{
                f"probability__{name}": value.contiguous()
                for name, value in probabilities.items()
            },
            "targets": full.base.targets.detach().cpu().contiguous(),
            "target_mask": full.base.target_mask.detach().cpu().contiguous(),
            "patient_folds": torch.tensor(patient_folds, dtype=torch.int64),
        }
        prediction_path = temporary / "oof_predictions.safetensors"
        save_file(tensors, str(prediction_path))
        manifest = {
            **preflight,
            "status": "completed_shortcut_audit_source_train_only",
            "config": {
                "outer_folds": list(OUTER_FOLDS),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "fixed_candidate_no_inner_selection": FIXED_REFERENCE,
                "same_model_seed_and_initialization_as_reference": True,
                "same_original_h_fold_standardization_and_prior_as_reference": True,
                "fold_local_position_prototype": True,
                "fold_contained_cross_patient_event_shuffle": True,
                "nonzero_circular_time_shift": True,
                "matched_zero_h_v_control": True,
                "fold_local_q_only_prior": True,
                "strict_one_patient_tolerance": STRICT_ONE_PATIENT,
                "content_gate": (
                    "reference_AP_plus_0.01_HitAt3_nonlower_and_strict_within_1_of_65"
                ),
                "macro_ap_material_threshold": MATERIAL_AP,
                "matched_zero_h_v_gate": (
                    "reference_AP_plus_0.01_HitAt3_nonlower_and_strict_within_1_of_65"
                ),
                "prevalence_only_gate": (
                    "strict_and_AP_nonlower_with_strict_plus_1_of_65_or_AP_plus_0.01"
                ),
            },
            "result": result,
            "patient_ids": list(full.base.patient_ids),
            "patient_folds": list(patient_folds),
            "files": {
                "oof_predictions.safetensors": {
                    "sha256": _file_sha256(prediction_path),
                    "size_bytes": prediction_path.stat().st_size,
                }
            },
            "scientific_boundary": {
                "audit_only_not_model_selection": True,
                "foundation_backbone": "official_LaBraM_frozen",
                "position_prototype_is_not_pure_position_embedding": True,
                "uniform_pooling_makes_isolated_time_permutation_partly_invariant": True,
                "pass_does_not_establish_waveform_or_causal_soz_semantics": True,
                "source_dev_used": False,
                "source_eval_used": False,
                "private_used": False,
                "formal_promotion": False,
            },
        }
        raw = _canonical_bytes(manifest)
        (temporary / "manifest.json").write_bytes(raw)
        os.rename(temporary, output)
        published = True
        print(
            json.dumps(
                {
                    "status": "completed_shortcut_audit_source_train_only",
                    "output_directory": str(output),
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "shortcut_decision": result["shortcut_decision"],
                    "oof_metrics": result["oof_metrics"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
