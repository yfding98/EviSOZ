#!/usr/bin/env python3
"""Run the frozen-LaBraM scalp-visible onset-contrast source-train OOF screen."""

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
    _load_frozen_h_source_train,
)
from scripts.run_labram_temporal_mil_nested_oof_v1 import (  # noqa: E402
    _canonical_bytes,
    _file_sha256,
    _indices_for_folds,
    _metrics,
    _tensor_state_sha256,
)
from scripts.run_labram_v_directed_endpoint_oof_v5 import (  # noqa: E402
    TEMPORAL_ANCHOR_MANIFEST_SHA256,
    TEMPORAL_ANCHOR_PATH,
    TEMPORAL_ANCHOR_PREDICTION_SHA256,
    _direction_payload,
    _load_temporal_anchor,
    _paired_patient_bootstrap,
    _transition_diagnostic,
)
from src.soz.aggregation import aggregate_patient_logits  # noqa: E402
from src.soz.frozen_h_recovery import (  # noqa: E402
    FrozenHPatientBatch,
    FrozenHStandardization,
)
from src.soz.onset_contrast_recovery import (  # noqa: E402
    ONSET_CONTRAST_CANDIDATES,
    ONSET_CONTRAST_RECOVERY_SCHEMA,
    OnsetContrastCandidate,
    ScalpOnsetContrastNodeLocalizer,
    fit_onset_contrast_standardization,
    onset_contrast_objective,
    subset_onset_contrast_patient_batch,
)
from src.soz.safe_anchor_h_recovery import (  # noqa: E402
    within_tcp_edge_direction_metrics,
)
from src.soz.temporal_mil_recovery import (  # noqa: E402
    jeffreys_channel_prior_logits,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_scalp_onset_contrast_recovery_protocol_v6_20260811_zh.md"
)
MODULE_PATH = ROOT / "src/soz/onset_contrast_recovery.py"
OUTER_FOLDS = tuple(range(5))
PRIMARY_CANDIDATE: OnsetContrastCandidate = "onset_contrast_h_v"
V_ONLY_CONTROL: OnsetContrastCandidate = "onset_contrast_v_only"
FULL_PHASE_CONTROL: OnsetContrastCandidate = "full_phase_h_v_matched"
EPOCHS = 100
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0
BASE_SEED = 20260811


def _scope_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_bytes(list(values))).hexdigest()


def _seeded_model(
    prior: torch.Tensor,
    standardization: FrozenHStandardization,
    *,
    candidate: OnsetContrastCandidate,
    seed: int,
    device: torch.device,
) -> ScalpOnsetContrastNodeLocalizer:
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = ScalpOnsetContrastNodeLocalizer(
            prior,
            standardization,
            candidate=candidate,
        )
    return model.to(device)


def _common_initialization_receipt(
    prior: torch.Tensor,
    standardization: FrozenHStandardization,
    *,
    seed: int,
) -> dict[str, str]:
    models = {
        candidate: _seeded_model(
            prior,
            standardization,
            candidate=candidate,
            seed=seed,
            device=torch.device("cpu"),
        )
        for candidate in ONSET_CONTRAST_CANDIDATES
    }
    v_states = {
        name: {
            key: value.detach().clone()
            for key, value in model.v_scorer.state_dict().items()
        }
        for name, model in models.items()
    }
    first_v = v_states[ONSET_CONTRAST_CANDIDATES[0]]
    if any(
        set(state) != set(first_v)
        or any(not torch.equal(state[key], first_v[key]) for key in first_v)
        for state in v_states.values()
    ):
        raise RuntimeError("onset-contrast candidates do not share V initialization")

    primary_h = models[PRIMARY_CANDIDATE].h_scorer
    full_h = models[FULL_PHASE_CONTROL].h_scorer
    if primary_h is None or full_h is None or not torch.equal(
        primary_h.weight, full_h.weight
    ):
        raise RuntimeError("H+V candidates do not share H initialization")
    return {
        "v_initialization_sha256": _tensor_state_sha256(first_v),
        "h_initialization_sha256": _tensor_state_sha256(
            {"weight": primary_h.weight.detach()}
        ),
    }


def _fit_candidate(
    train: FrozenHPatientBatch,
    standardization: FrozenHStandardization,
    prior: torch.Tensor,
    *,
    candidate: OnsetContrastCandidate,
    seed: int,
    device: torch.device,
) -> tuple[ScalpOnsetContrastNodeLocalizer, dict[str, object]]:
    model = _seeded_model(
        prior,
        standardization,
        candidate=candidate,
        seed=seed,
        device=device,
    )
    parameter_count = model.n_trainable_parameters
    if parameter_count >= 500:
        raise RuntimeError("onset-contrast candidate violated the capacity gate")
    batch = train.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    first: dict[str, float] | None = None
    last: dict[str, float] | None = None
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.node_tokens, batch.base.evidence)
        objective = onset_contrast_objective(output.event_logits, batch)
        objective.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        row = {
            "total": float(objective.total.detach().cpu()),
            "exact_set_mass": float(objective.exact_set_mass.detach().cpu()),
            "pairwise": float(objective.pairwise.detach().cpu()),
            "bce": float(objective.bce.detach().cpu()),
            "consistency": float(objective.consistency.detach().cpu()),
        }
        if first is None:
            first = row
        last = row
    optimizer.zero_grad(set_to_none=True)
    del optimizer, batch
    model.eval()
    model.requires_grad_(False)
    assert first is not None and last is not None
    preprocessing_state = {
        "h_mean": standardization.mean.detach().cpu(),
        "h_scale": standardization.scale.detach().cpu(),
        "channel_prior": prior.detach().cpu(),
    }
    return model, {
        "candidate": candidate,
        "seed": seed,
        "epochs": EPOCHS,
        "trainable_parameter_count": parameter_count,
        "fit_scope_patient_count": len(train.base.patient_ids),
        "fit_scope_patient_roster_sha256": _scope_sha256(train.base.patient_ids),
        "fold_local_preprocessing_state_sha256": _tensor_state_sha256(
            preprocessing_state
        ),
        "first_epoch": first,
        "final_epoch": last,
    }


def _predict_candidate(
    model: ScalpOnsetContrastNodeLocalizer,
    batch: FrozenHPatientBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    moved = batch.to(device)
    model.eval()
    with torch.no_grad():
        output = model(moved.node_tokens, moved.base.evidence)
        patient_logits = aggregate_patient_logits(
            output.event_logits,
            moved.base.event_patient_index,
        ).logits

    event_index = moved.base.event_patient_index
    patients_without_matched = 0
    matched_event_count_by_patient: list[int] = []
    for patient in range(len(moved.base.patient_ids)):
        selected = event_index == patient
        matched_count = int(output.matched_event[selected].sum().item())
        matched_event_count_by_patient.append(matched_count)
        patients_without_matched += int(matched_count == 0)
    diagnostics = {
        "event_count": moved.base.evidence.batch_size,
        "matched_event_count": int(output.matched_event.sum().item()),
        "prior_only_event_count": int(output.prior_only_event.sum().item()),
        "late_available_event_count": int(output.late_phase_available.sum().item()),
        "patients_without_matched_event": patients_without_matched,
        "matched_event_count_by_patient": matched_event_count_by_patient,
        "v_main_valid_event_channel_count": int(output.v_main_valid.sum().item()),
        "v_late_valid_event_channel_count": int(output.v_late_valid.sum().item()),
        "mean_absolute_main_contribution": float(
            output.main_contribution.abs().mean().detach().cpu()
        ),
        "mean_late_corroboration": float(
            output.late_corroboration.mean().detach().cpu()
        ),
        "max_late_corroboration": float(
            output.late_corroboration.max().detach().cpu()
        ),
        "late_independent_positive_violation_count": int(
            (
                (output.main_contribution <= 0)
                & (output.late_corroboration > 0)
            ).sum().item()
        ),
    }
    if diagnostics["late_independent_positive_violation_count"] != 0:
        raise RuntimeError("late evidence independently increased a channel")
    return patient_logits.detach().cpu(), diagnostics


def _phase_availability(full: FrozenHPatientBatch) -> dict[str, object]:
    phase = full.base.evidence.phase_mask.detach().cpu()
    event_patient = full.base.event_patient_index.detach().cpu()
    pre_complete = phase[:, :3].all(dim=1)
    early_complete = phase[:, 3:6].all(dim=1)
    matched = pre_complete & early_complete
    late_any = phase[:, 6:].any(dim=1)
    matched_by_patient = torch.zeros(
        len(full.base.patient_ids), dtype=torch.int64
    )
    matched_by_patient.scatter_add_(0, event_patient, matched.to(torch.int64))
    early_counts = phase[:, 3:6].sum(dim=1)
    v_mask = full.base.evidence.evolution_mask.detach().cpu()
    return {
        "event_count": phase.shape[0],
        "patient_count": len(full.base.patient_ids),
        "pre_complete_event_count": int(pre_complete.sum().item()),
        "early_complete_event_count": int(early_complete.sum().item()),
        "matched_pre_early_complete_event_count": int(matched.sum().item()),
        "prior_only_event_count": int((~matched).sum().item()),
        "late_any_event_count": int(late_any.sum().item()),
        "patient_with_matched_event_count": int((matched_by_patient > 0).sum().item()),
        "patient_without_matched_event_count": int((matched_by_patient == 0).sum().item()),
        "patient_ids_without_matched_event": [
            full.base.patient_ids[index]
            for index in torch.nonzero(
                matched_by_patient == 0, as_tuple=False
            ).flatten().tolist()
        ],
        "early_valid_tile_count_histogram": {
            str(count): int((early_counts == count).sum().item())
            for count in range(4)
        },
        "v_unavailable_event_channel_tile_count": int((~v_mask).sum().item()),
        "eligibility_semantics": (
            "all three pre tiles and all three early tiles; prior-only events and "
            "patients are retained"
        ),
    }


def _run(
    full: FrozenHPatientBatch,
    patient_folds: tuple[int, ...],
    anchor: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    patients = len(full.base.patient_ids)
    oof = {
        candidate: torch.full((patients, 19), torch.nan, dtype=torch.float32)
        for candidate in ONSET_CONTRAST_CANDIDATES
    }
    outer_rows: list[dict[str, object]] = []
    primary_direction_nonlower_count = 0

    for outer_fold in OUTER_FOLDS:
        train_indices = _indices_for_folds(
            patient_folds,
            tuple(fold for fold in OUTER_FOLDS if fold != outer_fold),
        )
        held_indices = _indices_for_folds(patient_folds, (outer_fold,))
        train = subset_onset_contrast_patient_batch(full, train_indices)
        held = subset_onset_contrast_patient_batch(full, held_indices)
        standardization = fit_onset_contrast_standardization(train)
        prior = jeffreys_channel_prior_logits(train.base).detach().cpu()
        seed = BASE_SEED + outer_fold * 1000
        initialization = _common_initialization_receipt(
            prior, standardization, seed=seed
        )
        held_index = torch.tensor(held_indices, dtype=torch.long)
        anchor_held = anchor.index_select(0, held_index)
        anchor_direction = within_tcp_edge_direction_metrics(
            anchor_held,
            held.base.targets,
            held.base.target_mask,
        )
        candidate_rows: dict[str, object] = {}

        for candidate in ONSET_CONTRAST_CANDIDATES:
            model, fit = _fit_candidate(
                train,
                standardization,
                prior,
                candidate=candidate,
                seed=seed,
                device=device,
            )
            logits, diagnostics = _predict_candidate(model, held, device=device)
            oof[candidate][list(held_indices)] = logits
            direction = within_tcp_edge_direction_metrics(
                logits,
                held.base.targets,
                held.base.target_mask,
            )
            candidate_rows[candidate] = {
                "fit": {**fit, **initialization},
                "diagnostics": diagnostics,
                "metrics": _metrics(
                    logits,
                    held.base.targets,
                    held.base.target_mask,
                ),
                "within_tcp_direction": _direction_payload(direction),
                "direction_nonlower_than_anchor": (
                    direction.patient_macro_accuracy
                    >= anchor_direction.patient_macro_accuracy
                ),
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        primary_direction_nonlower_count += int(
            candidate_rows[PRIMARY_CANDIDATE][
                "direction_nonlower_than_anchor"
            ]
        )
        row = {
            "outer_fold": outer_fold,
            "train_patient_count": len(train_indices),
            "held_patient_count": len(held_indices),
            "anchor_metrics": _metrics(
                anchor_held,
                held.base.targets,
                held.base.target_mask,
            ),
            "anchor_within_tcp_direction": _direction_payload(anchor_direction),
            "candidates": candidate_rows,
        }
        outer_rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "outer_complete",
                    "outer_fold": outer_fold,
                    "primary_strict": candidate_rows[PRIMARY_CANDIDATE]["metrics"][
                        "top1"
                    ]["strict_accuracy"],
                    "primary_ap": candidate_rows[PRIMARY_CANDIDATE]["metrics"][
                        "ranking"
                    ]["macro_average_precision"],
                    "primary_direction": candidate_rows[PRIMARY_CANDIDATE][
                        "within_tcp_direction"
                    ]["patient_macro_accuracy"],
                    "anchor_direction": anchor_direction.patient_macro_accuracy,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if any(not torch.isfinite(value).all() for value in oof.values()):
        raise RuntimeError("onset-contrast OOF left a patient prediction unfilled")

    anchor_metrics = _metrics(anchor, full.base.targets, full.base.target_mask)
    anchor_direction = within_tcp_edge_direction_metrics(
        anchor,
        full.base.targets,
        full.base.target_mask,
    )
    metrics = {"temporal_mil_exact_anchor": anchor_metrics}
    directions = {"temporal_mil_exact_anchor": _direction_payload(anchor_direction)}
    bootstraps: dict[str, object] = {}
    transitions: dict[str, object] = {}
    for candidate in ONSET_CONTRAST_CANDIDATES:
        metrics[candidate] = _metrics(
            oof[candidate], full.base.targets, full.base.target_mask
        )
        direction = within_tcp_edge_direction_metrics(
            oof[candidate], full.base.targets, full.base.target_mask
        )
        directions[candidate] = _direction_payload(direction)
        bootstraps[candidate] = _paired_patient_bootstrap(
            oof[candidate],
            anchor,
            full.base.targets,
            full.base.target_mask,
        )
        transitions[candidate] = _transition_diagnostic(
            oof[candidate],
            anchor,
            full.base.targets,
            full.base.target_mask,
        )

    primary_metrics = metrics[PRIMARY_CANDIDATE]
    v_only_metrics = metrics[V_ONLY_CONTROL]
    full_metrics = metrics[FULL_PHASE_CONTROL]
    primary_direction = directions[PRIMARY_CANDIDATE]["patient_macro_accuracy"]
    v_only_direction = directions[V_ONLY_CONTROL]["patient_macro_accuracy"]
    full_direction = directions[FULL_PHASE_CONTROL]["patient_macro_accuracy"]
    anchor_direction_value = directions["temporal_mil_exact_anchor"][
        "patient_macro_accuracy"
    ]
    primary_transition = transitions[PRIMARY_CANDIDATE]
    gate_checks = {
        "strict_top1_nonlower_than_anchor": (
            primary_metrics["top1"]["strict_accuracy"]
            >= anchor_metrics["top1"]["strict_accuracy"]
        ),
        "macro_ap_nonlower_than_anchor": (
            primary_metrics["ranking"]["macro_average_precision"]
            >= anchor_metrics["ranking"]["macro_average_precision"]
        ),
        "strict_top1_nonlower_than_v_only": (
            primary_metrics["top1"]["strict_accuracy"]
            >= v_only_metrics["top1"]["strict_accuracy"]
        ),
        "macro_ap_nonlower_than_v_only": (
            primary_metrics["ranking"]["macro_average_precision"]
            >= v_only_metrics["ranking"]["macro_average_precision"]
        ),
        "strict_top1_nonlower_than_full_phase": (
            primary_metrics["top1"]["strict_accuracy"]
            >= full_metrics["top1"]["strict_accuracy"]
        ),
        "macro_ap_nonlower_than_full_phase": (
            primary_metrics["ranking"]["macro_average_precision"]
            >= full_metrics["ranking"]["macro_average_precision"]
        ),
        "within_tcp_direction_strictly_higher_than_anchor": (
            primary_direction > anchor_direction_value
        ),
        "within_tcp_direction_nonlower_than_v_only": (
            primary_direction >= v_only_direction
        ),
        "within_tcp_direction_nonlower_than_full_phase": (
            primary_direction >= full_direction
        ),
        "tcp_to_exact_rescues_exceed_exact_to_tcp_losses": primary_transition[
            "rescue_exceeds_loss"
        ],
        "far_errors_nonincreasing": primary_transition[
            "far_errors_nonincreasing"
        ],
        "direction_nonlower_in_at_least_3_of_5_folds": (
            primary_direction_nonlower_count >= 3
        ),
    }
    gate_pass = all(gate_checks.values())

    full_standardization = fit_onset_contrast_standardization(full)
    full_prior = jeffreys_channel_prior_logits(full.base).detach().cpu()
    final_states: dict[str, torch.Tensor] = {}
    final_fits: dict[str, object] = {}
    for candidate in ONSET_CONTRAST_CANDIDATES:
        model, fit = _fit_candidate(
            full,
            full_standardization,
            full_prior,
            candidate=candidate,
            seed=BASE_SEED + 99999,
            device=device,
        )
        for name, value in model.state_dict().items():
            final_states[f"{candidate}.{name}"] = value.detach().cpu().contiguous()
        final_fits[candidate] = fit
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "screen_kind": (
            "post_hoc_source_train_patient_oof_mechanism_recovery;"
            "fixed_before_run_but_informed_by_prior_source_train_oof_results"
        ),
        "primary_candidate": PRIMARY_CANDIDATE,
        "metrics": metrics,
        "within_tcp_direction": directions,
        "candidate_minus_anchor_paired_patient_bootstrap": bootstraps,
        "top1_transition_diagnostic": transitions,
        "outer_folds": outer_rows,
        "primary_directionally_nonlower_fold_count": (
            primary_direction_nonlower_count
        ),
        "frozen_go_no_go_gate": {
            "checks": gate_checks,
            "pass": gate_pass,
            "status": (
                "go_candidate_for_later_locked_validation"
                if gate_pass
                else "no_go_keep_temporal_mil_exact"
            ),
            "interpretation": (
                "post-hoc source-train point-estimate gate; not statistical "
                "noninferiority or external generalization"
            ),
        },
        "final_full_source_train_fits": final_fits,
    }
    tensors = {
        **{name: value.contiguous() for name, value in oof.items()},
        "temporal_mil_exact_anchor": anchor.contiguous(),
        "targets": full.base.targets.detach().cpu().contiguous(),
        "target_mask": full.base.target_mask.detach().cpu().contiguous(),
        "patient_folds": torch.tensor(patient_folds, dtype=torch.int64),
    }
    return result, tensors, final_states


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight_only and args.output_directory is None:
        raise ValueError("full OOF run requires --output-directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    full, patient_folds, lineage = _load_frozen_h_source_train()
    anchor, anchor_manifest = _load_temporal_anchor(full.base, patient_folds)
    availability = _phase_availability(full)
    fold_counts = {
        str(fold): sum(value == fold for value in patient_folds)
        for fold in OUTER_FOLDS
    }
    preflight = {
        "status": "ready_fixed_source_train_patient_oof",
        "schema_version": ONSET_CONTRAST_RECOVERY_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "device": str(device),
        "patient_count": len(full.base.patient_ids),
        "event_count": full.base.evidence.batch_size,
        "patient_ids": list(full.base.patient_ids),
        "patient_folds": list(patient_folds),
        "fold_counts": fold_counts,
        "phase_availability": availability,
        "lineage": {
            **lineage,
            "temporal_anchor_manifest_sha256": TEMPORAL_ANCHOR_MANIFEST_SHA256,
            "temporal_anchor_prediction_sha256": (
                TEMPORAL_ANCHOR_PREDICTION_SHA256
            ),
            "runner_sha256": _file_sha256(Path(__file__).resolve()),
            "onset_contrast_module_sha256": _file_sha256(MODULE_PATH),
        },
        "config": {
            "outer_folds": list(OUTER_FOLDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "base_seed": BASE_SEED,
            "candidate_names": list(ONSET_CONTRAST_CANDIDATES),
            "primary_candidate": PRIMARY_CANDIDATE,
            "inner_candidate_selection": False,
            "neighbor_training_auxiliary": False,
            "patient_event_pooling": "equal_event_logit_mean",
            "pre_tiles": [0, 1, 2],
            "early_tiles": [3, 4, 5],
            "late_tiles": list(range(6, 15)),
            "late_corroboration_scale": 0.25,
        },
        "foundation_backbone": "official_pretrained_LaBraM_frozen_not_replaced",
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
    for source in (PROTOCOL_PATH, MODULE_PATH, TEMPORAL_ANCHOR_PATH):
        resolved = source.resolve(strict=True)
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError("output path overlaps an input")

    result, tensors, final_states = _run(
        full,
        patient_folds,
        anchor,
        device=device,
    )
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for OOF publication") from exc
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        prediction_path = temporary / "oof_predictions.safetensors"
        save_file(tensors, str(prediction_path))
        checkpoint_path = temporary / "final_checkpoints.safetensors"
        save_file(final_states, str(checkpoint_path))
        manifest = {
            **preflight,
            "status": "completed_post_hoc_source_train_patient_oof",
            "result": result,
            "files": {
                "oof_predictions.safetensors": {
                    "sha256": _file_sha256(prediction_path),
                    "size_bytes": prediction_path.stat().st_size,
                },
                "final_checkpoints.safetensors": {
                    "sha256": _file_sha256(checkpoint_path),
                    "size_bytes": checkpoint_path.stat().st_size,
                    "state_sha256": _tensor_state_sha256(final_states),
                },
            },
            "scientific_boundary": {
                "foundation_backbone": "official_pretrained_LaBraM_frozen",
                "foundation_replaced": False,
                "foundation_trainable_parameter_count": 0,
                "h_semantics": (
                    "physical_node_indexed_cross_channel_contextualized_latent;"
                    "not_named_clinical_concept"
                ),
                "early_contrast_semantics": (
                    "retrospective_scalp_visible_change_not_cortical_onset_or_soz"
                ),
                "late_semantics": (
                    "bounded_corroboration_not_propagation_probability"
                ),
                "ictal_i_in_learned_spatial_path": False,
                "post_hoc_after_prior_source_train_oof_review": True,
                "source_dev_used": False,
                "source_eval_used": False,
                "private_used": False,
                "formal_promotion": False,
            },
            "anchor_reported_final_candidate": anchor_manifest.get("result", {}).get(
                "final_candidate"
            ),
        }
        raw = _canonical_bytes(manifest)
        (temporary / "manifest.json").write_bytes(raw)
        os.rename(temporary, output)
        published = True
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "output_directory": str(output),
                    "metrics": result["metrics"],
                    "within_tcp_direction": result["within_tcp_direction"],
                    "frozen_go_no_go_gate": result["frozen_go_no_go_gate"],
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
